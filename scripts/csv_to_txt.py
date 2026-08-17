"""
将 GMR 重定向输出 (PKL 或 CSV) 转换为 AMP 兼容的 TXT 格式，
包含与 whole_body_tracking/scripts/csv_to_npz.py 相同的信息
(每个链接的世界位姿 + 速度)，但写入 AMP .txt 容器中。

流程 (复刻 csv_to_npz):
  1. 重采样 input_fps -> output_fps   (pos/dof: 线性插值, quat: 球面插值)
  2. 对所有身体做前向运动学  (GMR KinematicsModel, 无需 Isaac Sim)
  3. 速度: 中心差分 (np.gradient) + SO3 导数用于角速度

每帧布局 (总计 = 19 + 18 + 2N + 13B, N=#关节数, B=#身体数):
  root_pos(3) + root_quat(4, xyzw) + joint_pos(N) +
  foot_pos(6) + hand_pos(6) + elbow_pos(6) + knee_pos(6) + root_lin_vel(3) + root_ang_vel(3) + joint_vel(N) +
  body_pos_w(3B) + body_quat_w(4B, wxyz) + body_lin_vel_w(3B) + body_ang_vel_w(3B)

Header 记录 "BodyNames" 和 "ColumnLayout" 使得追加块是
自描述的。npz 字段映射如下:
  fps            -> header "Fps"
  joint_pos      -> joint_pos 列
  joint_vel      -> joint_vel 列
  body_pos_w     -> body_pos_w 列
  body_quat_w    -> body_quat_w 列 (wxyz)
  body_lin_vel_w -> body_lin_vel_w 列
  body_ang_vel_w -> body_ang_vel_w 列

用法:
  # CSV -> TXT (重采样 120fps -> 50fps，--input_fps 指定输入帧率)
  python scripts/CSV/csv_to_txt.py --csv output/unitree_g1/shuangren/motion1.csv --robot unitree_g1 --input_fps 120 --output_fps 50 --output output/unitree_g1/shuangren/motion1.txt

  # CSV -> TXT (帧率从 BVH Frame Time 自动读取)
  python scripts/CSV/csv_to_txt.py --csv output/unitree_g1/shuangren/motion1.csv --robot unitree_g1 --bvh BVH/linyun/1/human1.bvh --output_fps 50 --output output/unitree_g1/shuangren/motion1.txt

  # PKL -> TXT (帧率从 pickle 自动读取)
  python scripts/CSV/csv_to_txt.py --pkl output/x3_zq_28dof/shuangren/motion1.pkl --robot x3_zq_28dof --output_fps 50 --output output/x3_zq_28dof/shuangren/motion1.txt

  # 批量转换目录下所有 CSV/ PKL
  python scripts/CSV/csv_to_txt.py --dir output/unitree_g1/shuangren --robot unitree_g1 --input_fps 120 --output_fps 50 --output output/unitree_g1/shuangren/txt
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT

# ---------------------------------------------------------------------------
# 四元数 / 插值工具 (numpy, 向量化)
# ---------------------------------------------------------------------------

def _xyzw_to_wxyz(q):
    """将 (..., 4) xyzw 格式转换为 (..., 4) wxyz 格式。"""
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)

def _quat_mul_wxyz(q, r):
    """wxyz 四元数的哈密顿乘积，在 (..., 4) 上广播。"""
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], axis=-1)

def _quat_conj_wxyz(q):
    """wxyz 四元数的共轭。"""
    return q * np.array([1.0, -1.0, -1.0, -1.0])

def _axis_angle_from_quat_wxyz(q, eps=1.0e-6):
    """从 wxyz 四元数计算旋转向量 (轴*角度) (与 IsaacLab 匹配)。"""
    # 强制 w >= 0 以取最短旋转
    q = q * (1.0 - 2.0 * (q[..., 0:1] < 0.0))
    mag = np.linalg.norm(q[..., 1:], axis=-1)
    half_angle = np.arctan2(mag, q[..., 0])
    angle = 2.0 * half_angle
    safe_angle = np.where(np.abs(angle) > eps, angle, 1.0)
    sin_half_over_angle = np.where(
        np.abs(angle) > eps,
        np.sin(half_angle) / safe_angle,
        0.5 - angle * angle / 48.0,
    )
    return q[..., 1:] / sin_half_over_angle[..., None]

def _lerp(a, b, blend):
    """线性插值。"""
    return a * (1.0 - blend) + b * blend

def _quat_slerp(a, b, t):
    """对两个 xyzw (或 wxyz) 四元数进行球面线性插值。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:  # 几乎平行 -> 线性插值
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return s0 * a + s1 * b

# ---------------------------------------------------------------------------
# BVH 辅助函数
# ---------------------------------------------------------------------------

def read_bvh_frame_duration(bvh_path):
    """从 BVH 文件读取帧时长（单位：秒）。"""
    with open(bvh_path) as f:
        for line in f:
            m = re.match(r"^Frame Time:\s*([\d.]+)", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
    raise ValueError(f"Frame Time not found in BVH: {bvh_path}")

# ---------------------------------------------------------------------------
# 重采样 (input_fps -> output_fps), 复刻 csv_to_npz
# ---------------------------------------------------------------------------

def resample_motion(root_pos, root_rot, dof_pos, input_fps, output_fps):
    """重采样到 output_fps。root_rot 为 xyzw 格式。返回重采样后的数组。

    使用 linspace(0, duration, N) 以始终包含最后一帧。pos/dof 通过
    线性插值，quat 通过球面插值。
    """
    input_frames = root_pos.shape[0]
    if input_frames < 2:
        return root_pos, root_rot, dof_pos

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (input_frames - 1) * input_dt

    output_frames = int(duration / output_dt) + 1
    times = np.linspace(0.0, duration, output_frames)
    phase = times / duration
    index_0 = np.floor(phase * (input_frames - 1)).astype(int)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - index_0

    rp = _lerp(root_pos[index_0], root_pos[index_1], blend[:, None])
    dp = _lerp(dof_pos[index_0], dof_pos[index_1], blend[:, None])
    rr = np.stack([
        _quat_slerp(root_rot[i0], root_rot[i1], b)
        for i0, i1, b in zip(index_0, index_1, blend)
    ])
    return rp, rr, dp

# ---------------------------------------------------------------------------
# 身体部位名称映射
# ---------------------------------------------------------------------------

FOOT_BODY_MAP = {
    "x3_zq":        ("left_ankle_roll_link", "right_ankle_roll_link"),
    "x3_zq_28dof":  ("left_ankle_roll_link", "right_ankle_roll_link"),
    "unitree_g1":   ("left_ankle_roll_link", "right_ankle_roll_link"),
    "x3_f1_14dof":  ("left_ankle_roll_link", "right_ankle_roll_link"),
    "x3_f2":        ("left_ankle_roll_link", "right_ankle_roll_link"),
}

HAND_BODY_MAP = {
    "x3_zq":        ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "x3_zq_28dof":  ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "unitree_g1":   ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "x3_f1_14dof":  (None, None),   # 无手臂自由度
    "x3_f2":        (None, None),
}

ELBOW_BODY_MAP = {
    "x3_zq":        ("left_elbow_link", "right_elbow_link"),
    "x3_zq_28dof":  ("left_elbow_link", "right_elbow_link"),
    "unitree_g1":   ("left_elbow_link", "right_elbow_link"),
    "x3_f1_14dof":  (None, None),   # 无手臂自由度
    "x3_f2":        (None, None),
}

KNEE_BODY_MAP = {
    "x3_zq":        ("left_knee_link", "right_knee_link"),
    "x3_zq_28dof":  ("left_knee_link", "right_knee_link"),
    "unitree_g1":   ("left_knee_link", "right_knee_link"),
    "x3_f1_14dof":  ("left_knee_link", "right_knee_link"),
    "x3_f2":        ("left_knee_link", "right_knee_link"),
}

# ---------------------------------------------------------------------------
# 核心：全身运动学 + 速度计算，然后组装 TXT 帧
# ---------------------------------------------------------------------------

def compute_body_kinematics(root_pos, root_rot, dof_pos, kin, output_dt):
    """对所有身体做前向运动学，然后计算中心差分/SO3速度。

    返回包含 body_pos_w (T,B,3), body_quat_w (T,B,4 wxyz),
    body_lin_vel_w (T,B,3), body_ang_vel_w (T,B,3), joint_vel (T,N) 的字典。
    """
    rp_t = torch.from_numpy(root_pos).float()
    rr_t = torch.from_numpy(root_rot).float()       # xyzw
    dp_t = torch.from_numpy(dof_pos).float()

    body_pos, body_rot = kin.forward_kinematics(rp_t, rr_t, dp_t)  # (T,B,3),(T,B,4 xyzw)
    body_pos = body_pos.numpy().astype(np.float64)
    body_quat_wxyz = _xyzw_to_wxyz(body_rot.numpy().astype(np.float64))

    T = body_pos.shape[0]

    # 线速度 / 关节速度: 中心差分 (匹配 torch.gradient)
    if T >= 2:
        body_lin_vel = np.gradient(body_pos, output_dt, axis=0)
        joint_vel = np.gradient(dof_pos, output_dt, axis=0)
    else:
        body_lin_vel = np.zeros_like(body_pos)
        joint_vel = np.zeros_like(dof_pos)

    # 角速度: SO3 导数 (匹配 csv_to_npz._so3_derivative)
    if T >= 3:
        q_prev = body_quat_wxyz[:-2]
        q_next = body_quat_wxyz[2:]
        q_rel = _quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev))      # (T-2,B,4)
        omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * output_dt)  # (T-2,B,3)
        body_ang_vel = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    else:
        body_ang_vel = np.zeros((T, body_pos.shape[1], 3))

    return {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat_wxyz,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "joint_vel": joint_vel,
    }


def build_frames(root_pos, root_rot, dof_pos, output_dt, kin,
                 foot_left_idx, foot_right_idx, hand_left_idx, hand_right_idx,
                 elbow_left_idx, elbow_right_idx, knee_left_idx, knee_right_idx):
    """组装扩展的 AMP 帧矩阵及其列布局。

    root_pos (T,3), root_rot (T,4 xyzw), dof_pos (T,N) 已经重采样。
    只有有效的身体 (dof>0 + root) 被包含在每体列中。
    返回 (frames, layout, active_body_names)。
    """
    T, N = dof_pos.shape

    k = compute_body_kinematics(root_pos, root_rot, dof_pos, kin, output_dt)
    body_pos_w = k["body_pos_w"]
    body_quat_w = k["body_quat_w"]
    body_lin_vel_w = k["body_lin_vel_w"]
    body_ang_vel_w = k["body_ang_vel_w"]
    joint_vel = k["joint_vel"]

    # 筛选有效身体: 保留根部 (索引 0) + 所有 dof>0 的身体
    active = [0] + [i for i in range(1, kin.num_joint) if kin._joints[i].dof_dim > 0]
    active_body_names = [kin.body_names[i] for i in active]
    # 将原始索引映射到有效索引以便脚/手/肘查找
    orig_to_active = {orig: i for i, orig in enumerate(active)}

    # AMP 前缀量 (root == 身体索引 0, 脚/手/肘相对于根部)
    root_lin_vel = body_lin_vel_w[:, 0, :]      # 中心差分, == npz root
    root_ang_vel = body_ang_vel_w[:, 0, :]      # SO3, == npz root
    root_pos_w = body_pos_w[:, 0, :]            # (T,3) 世界坐标
    def _rel_pos(idx):
        """身体相对于根部的位置; 如果 idx 为 None (身体不存在) 则为零。"""
        if idx is None:
            return np.zeros((T, 3), dtype=np.float64)
        return body_pos_w[:, idx, :] - root_pos_w

    foot_pos  = np.concatenate([_rel_pos(foot_left_idx),  _rel_pos(foot_right_idx)],  axis=1)  # (T,6)
    hand_pos  = np.concatenate([_rel_pos(hand_left_idx),  _rel_pos(hand_right_idx)],  axis=1)  # (T,6)
    elbow_pos = np.concatenate([_rel_pos(elbow_left_idx), _rel_pos(elbow_right_idx)], axis=1)  # (T,6)
    knee_pos  = np.concatenate([_rel_pos(knee_left_idx),  _rel_pos(knee_right_idx)],  axis=1)  # (T,6)

    # 仅保留有效身体的每体列
    body_pos_act = body_pos_w[:, active, :]
    body_quat_act = body_quat_w[:, active, :]
    body_lin_act = body_lin_vel_w[:, active, :]
    body_ang_act = body_ang_vel_w[:, active, :]

    Ba = len(active)

    body_pos_flat = body_pos_act.reshape(T, 3 * Ba)
    body_quat_flat = body_quat_act.reshape(T, 4 * Ba)
    body_lin_flat = body_lin_act.reshape(T, 3 * Ba)
    body_ang_flat = body_ang_act.reshape(T, 3 * Ba)

    frames = np.concatenate([
        root_pos.astype(np.float64),    # [0:3]    root_pos
        root_rot.astype(np.float64),    # [3:7]    root_quat (xyzw)
        dof_pos.astype(np.float64),     # [7:7+N]  joint_pos
        # knee_pos,                       # [7+N:7+N+6] knee_pos
        foot_pos,                       # [7+N+6:7+N+12] foot_pos
        root_lin_vel,                   # [7+N+12:7+N+15] root_lin_vel
        root_ang_vel,                   # [7+N+15:7+N+18] root_ang_vel
        joint_vel,                      # [7+N+18:7+2N+18] joint_vel
        # body_pos_flat,
        # body_quat_flat,
        # body_lin_flat,
        # body_ang_flat,
    ], axis=1)

    # 列布局 (开始, 结束) 偏移
    o = 0
    layout = {}
    def seg(name, width):
        nonlocal o
        layout[name] = [o, o + width]
        o += width
    seg("root_pos", 3)
    seg("root_quat", 4)           # xyzw  — 与参考 txt 格式对齐
    seg("joint_pos", N)
    # seg("knee_pos", 6)
    seg("foot_pos", 6)
    seg("root_lin_vel", 3)
    seg("root_ang_vel", 3)
    seg("joint_vel", N)           # 移至根部速度之后
    # seg("body_pos_w", 3 * Ba)
    # seg("body_quat_w", 4 * Ba)    # wxyz
    # seg("body_lin_vel_w", 3 * Ba)
    # seg("body_ang_vel_w", 3 * Ba)

    return frames.astype(np.float64), layout, active_body_names


_FIELD_DESC = {
    "root_pos":       "根部位置 (世界坐标, xyz)",
    "root_quat":      "根部四元数 (xyzw)",
    "joint_pos":      "关节位置",
    "foot_pos":       "脚相对于根部的位置 (左+右)",
    "hand_pos":       "手相对于根部的位置 (左+右)",
    "elbow_pos":      "肘相对于根部的位置 (左+右)",
    "knee_pos":       "膝相对于根部的位置 (左+右)",
    "root_lin_vel":   "根部线速度",
    "root_ang_vel":   "根部角速度",
    "joint_vel":      "关节速度  ← 与参考格式对齐",
    "body_pos_w":     "身体世界位置 (B×3)",
    "body_quat_w":    "身体世界四元数 wxyz (B×4)",
    "body_lin_vel_w": "身体世界线速度 (B×3)",
    "body_ang_vel_w": "身体世界角速度 (B×3)",
}

def _print_layout_summary(layout, body_names, total_cols):
    """打印帧布局摘要。"""
    B = len(body_names)
    W = 16
    print(f"\n── 帧布局  ({total_cols} 维总计, {B} 个身体) " + "─" * 30)
    for field, (start, end) in layout.items():
        dims = end - start
        desc = _FIELD_DESC.get(field, "")
        desc = desc.replace("B", str(B))
        print(f"  {field:<{W}} [{start:>4}:{end:>4}]  {dims:>4} 维   {desc}")
    print("─" * 65 + "\n")


def write_txt(frames, output_dt, output_path, layout=None, body_names=None, output_fps=None,
              motion_weight=1.0):
    """将帧写入 AMP TXT 格式文件。"""
    import os
    total_time = (len(frames) - 1) * output_dt
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("{\n")
        f.write('  "LoopMode": "Wrap",\n')
        f.write(f'  "FrameDuration": {output_dt},\n')
        f.write('  "EnableCycleOffsetPosition": true,\n')
        f.write('  "EnableCycleOffsetRotation": true,\n')
        f.write(f'  "TotalTime": {round(total_time, 5)},\n')
        f.write(f'  "MotionWeight": {motion_weight},\n')
        f.write('  "Frames": [\n')
        for i, frame in enumerate(frames):
            nums = ", ".join(f"{v:12.6f}" for v in frame)
            comma = "," if i < len(frames) - 1 else ""
            f.write(f"    [{nums}]{comma}\n")
        f.write("  ]\n")
        f.write("}\n")
    print(f"已保存 → {output_path}  ({frames.shape[0]} 帧 × {frames.shape[1]} 列)")



# ---------------------------------------------------------------------------
# 共享核心: 重采样 -> 构建 -> 写入
# ---------------------------------------------------------------------------

def process_motion(root_pos, root_rot, dof_pos, input_fps, output_fps,
                   robot_name, output_path):
    """处理运动数据: 重采样、构建帧、写入文件。"""
    kin, foot_left_idx, foot_right_idx, hand_left_idx, hand_right_idx, \
        elbow_left_idx, elbow_right_idx, knee_left_idx, knee_right_idx = _init_kin(robot_name)

    rp, rr, dp = resample_motion(root_pos, root_rot, dof_pos, input_fps, output_fps)
    output_dt = 1.0 / output_fps
    print(f"重采样: {root_pos.shape[0]} @ {input_fps:.3f}Hz -> "
          f"{rp.shape[0]} @ {output_fps}Hz")

    frames, layout, body_names = build_frames(rp, rr, dp, output_dt, kin,
                                              foot_left_idx, foot_right_idx,
                                              hand_left_idx, hand_right_idx,
                                              elbow_left_idx, elbow_right_idx,
                                              knee_left_idx, knee_right_idx)
    N = dp.shape[1]
    B = len(body_names)
    print(f"帧大小: {frames.shape[1]} "
          f"(19 + 12 + 2*{N} + 13*{B})  [+{B}个身体的body_pos/quat/lin_vel/ang_vel]")
    print(f"总时长: {(frames.shape[0]-1)*output_dt:.4f}s, "
          f"帧时长: {output_dt}s")

    write_txt(frames, output_dt, output_path, layout, body_names, output_fps)
    _print_layout_summary(layout, body_names, frames.shape[1])


# ---------------------------------------------------------------------------
# 输入: PKL
# ---------------------------------------------------------------------------

def pkl_to_txt(pkl_path, robot_name, output_path, output_fps, input_fps=None):
    """从 PKL 文件读取运动数据并转换为 TXT 格式。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot = np.asarray(data["root_rot"], dtype=np.float64)   # xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    in_fps = input_fps if input_fps is not None else data.get("fps", 100)

    print(f"PKL: {len(root_pos)} 帧, {in_fps} Hz, {dof_pos.shape[1]} 关节")
    process_motion(root_pos, root_rot, dof_pos, in_fps, output_fps,
                   robot_name, output_path)


# ---------------------------------------------------------------------------
# 输入: CSV
# ---------------------------------------------------------------------------

def csv_to_txt(csv_path, robot_name, output_path, input_fps, output_fps):
    """从 CSV 文件读取运动数据并转换为 TXT 格式。"""
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    num_frames, num_cols = data.shape
    num_joints = num_cols - 7
    assert num_joints > 0, f"CSV 有 {num_cols} 列; 期望 ≥8"

    root_pos = data[:, 0:3]
    root_rot = data[:, 3:7]   # xyzw
    dof_pos = data[:, 7:]

    print(f"CSV: {num_frames} 帧, {input_fps} Hz, {num_joints} 关节")
    process_motion(root_pos, root_rot, dof_pos, input_fps, output_fps,
                   robot_name, output_path)


# ---------------------------------------------------------------------------
# 共享: 初始化运动学 + 脚部和手部索引
# ---------------------------------------------------------------------------

def _init_kin(robot_name):
    """初始化运动学模型并查找身体部位索引（脚、手、肘、膝）。"""
    robot_xml = ROBOT_XML_DICT.get(robot_name)
    if robot_xml is None:
        raise KeyError(f"未知机器人 '{robot_name}'。可用选项: {sorted(ROBOT_XML_DICT.keys())}")

    kin = KinematicsModel(str(robot_xml), device="cpu")

    # 查找脚部身体
    if robot_name in FOOT_BODY_MAP:
        foot_left_name, foot_right_name = FOOT_BODY_MAP[robot_name]
    else:
        for keyword in ("ankle", "foot"):
            lefts  = [n for n in kin.body_names if keyword in n.lower() and "left"  in n.lower()]
            rights = [n for n in kin.body_names if keyword in n.lower() and "right" in n.lower()]
            if lefts and rights:
                foot_left_name, foot_right_name = lefts[0], rights[0]
                break
        else:
            raise RuntimeError(
                f"无法自动检测脚部身体。请传递 --foot_left / --foot_right。\n"
                f"身体列表: {kin.body_names}"
            )

    foot_left_idx  = kin.get_body_idx(foot_left_name)
    foot_right_idx = kin.get_body_idx(foot_right_name)

    # 查找手部身体 (None 表示机器人无手臂)
    if robot_name in HAND_BODY_MAP:
        hand_left_name, hand_right_name = HAND_BODY_MAP[robot_name]
    else:
        for keyword in ("wrist_yaw", "hand"):
            lefts  = [n for n in kin.body_names if keyword in n.lower() and "left"  in n.lower()]
            rights = [n for n in kin.body_names if keyword in n.lower() and "right" in n.lower()]
            if lefts and rights:
                hand_left_name, hand_right_name = lefts[0], rights[0]
                break
        else:
            raise RuntimeError(
                f"无法自动检测手部身体。请传递 --hand_left / --hand_right。\n"
                f"身体列表: {kin.body_names}"
            )

    hand_left_idx  = kin.get_body_idx(hand_left_name)  if hand_left_name  is not None else None
    hand_right_idx = kin.get_body_idx(hand_right_name) if hand_right_name is not None else None

    # 查找肘部身体 (None 表示机器人无手臂)
    if robot_name in ELBOW_BODY_MAP:
        elbow_left_name, elbow_right_name = ELBOW_BODY_MAP[robot_name]
    else:
        for keyword in ("elbow",):
            lefts  = [n for n in kin.body_names if keyword in n.lower() and "left"  in n.lower()]
            rights = [n for n in kin.body_names if keyword in n.lower() and "right" in n.lower()]
            if lefts and rights:
                elbow_left_name, elbow_right_name = lefts[0], rights[0]
                break
        else:
            raise RuntimeError(
                f"无法自动检测肘部身体。请传递 --elbow_left / --elbow_right。\n"
                f"身体列表: {kin.body_names}"
            )

    elbow_left_idx  = kin.get_body_idx(elbow_left_name)  if elbow_left_name  is not None else None
    elbow_right_idx = kin.get_body_idx(elbow_right_name) if elbow_right_name is not None else None

    # 查找膝部身体
    if robot_name in KNEE_BODY_MAP:
        knee_left_name, knee_right_name = KNEE_BODY_MAP[robot_name]
    else:
        for keyword in ("knee",):
            lefts  = [n for n in kin.body_names if keyword in n.lower() and "left"  in n.lower()]
            rights = [n for n in kin.body_names if keyword in n.lower() and "right" in n.lower()]
            if lefts and rights:
                knee_left_name, knee_right_name = lefts[0], rights[0]
                break
        else:
            raise RuntimeError(
                f"无法自动检测机器人 '{robot_name}' 的膝部身体。\n"
                f"身体列表: {kin.body_names}"
            )

    knee_left_idx  = kin.get_body_idx(knee_left_name)
    knee_right_idx = kin.get_body_idx(knee_right_name)

    print(f"脚部身体:  {foot_left_name} (idx={foot_left_idx}), {foot_right_name} (idx={foot_right_idx})")
    print(f"手部身体:  {hand_left_name} (idx={hand_left_idx}), {hand_right_name} (idx={hand_right_idx})  {'[零值]' if hand_left_name is None else ''}")
    print(f"肘部身体: {elbow_left_name} (idx={elbow_left_idx}), {elbow_right_name} (idx={elbow_right_idx})  {'[零值]' if elbow_left_name is None else ''}")
    print(f"膝部身体:  {knee_left_name} (idx={knee_left_idx}), {knee_right_name} (idx={knee_right_idx})")
    print(f"机器人 '{robot_name}': {kin.num_dof} 自由度, {kin.num_joint} 个身体")

    return kin, foot_left_idx, foot_right_idx, hand_left_idx, hand_right_idx, elbow_left_idx, elbow_right_idx, knee_left_idx, knee_right_idx


# ---------------------------------------------------------------------------
# 命令行界面 (CLI)
# ---------------------------------------------------------------------------

def _resolve_input_fps(args):
    """解析 CSV 模式的输入帧率 (pkl 模式从 pickle 读取帧率)。"""
    if args.input_fps is not None:
        return args.input_fps
    if args.frame_duration is not None:
        return 1.0 / args.frame_duration
    if args.bvh is not None:
        fd = read_bvh_frame_duration(args.bvh)
        print(f"从 BVH 读取帧时长: {fd}s ({1.0/fd:.0f} Hz)")
        return 1.0 / fd
    print("输入帧率默认值: 100 Hz")
    return 100.0


if __name__ == "__main__":
    import glob
    import os as _os

    parser = argparse.ArgumentParser(description="将 GMR PKL/CSV 转换为扩展 AMP TXT (包含与 csv_to_npz 相同的信息)")
    parser.add_argument("--pkl",    default=None, help="输入 PKL 文件的路径")
    parser.add_argument("--csv",    default=None, help="输入 CSV 文件的路径")
    parser.add_argument("--dir",    default=None, help="处理目录中的所有 PKL/CSV 文件")
    parser.add_argument("--robot",  required=True, help="机器人名称 (例如 x3_zq)")
    parser.add_argument("--output", required=True, help="输出 TXT 路径 (或使用 --dir 时的输出目录)")
    parser.add_argument("--bvh",    default=None, help="(CSV 模式) 从 BVH Frame Time 读取输入帧率")
    parser.add_argument("--input_fps", type=float, default=100,
                        help="(CSV 模式) 输入帧率。覆盖 --frame_duration/--bvh。PKL 使用 pickle 帧率除非设置。")
    parser.add_argument("--frame_duration", type=float, default=None,
                        help="(CSV 模式) 手动输入帧时长（秒）")
    parser.add_argument("--output_fps", type=float, default=50.0,
                        help="重采样目标帧率 (与 csv_to_npz 匹配，默认 50)")
    parser.add_argument("--foot_left",  default=None, help="覆盖左脚身体名称")
    parser.add_argument("--foot_right", default=None, help="覆盖右脚身体名称")
    args = parser.parse_args()

    # --- 目录模式 ---
    if args.dir is not None:
        pkl_files = sorted(glob.glob(_os.path.join(args.dir, "*.pkl")))
        csv_files = sorted(glob.glob(_os.path.join(args.dir, "*.csv")))

        if pkl_files:
            print(f"找到 {len(pkl_files)} 个 PKL 文件")
            for pkl_path in pkl_files:
                name = _os.path.splitext(_os.path.basename(pkl_path))[0]
                pkl_to_txt(pkl_path, args.robot, _os.path.join(args.output, f"{name}.txt"),
                           args.output_fps, args.input_fps)
        elif csv_files:
            print(f"找到 {len(csv_files)} 个 CSV 文件")
            in_fps = _resolve_input_fps(args)
            for csv_path in csv_files:
                name = _os.path.splitext(_os.path.basename(csv_path))[0]
                csv_to_txt(csv_path, args.robot, _os.path.join(args.output, f"{name}.txt"),
                           in_fps, args.output_fps)
        else:
            parser.error(f"在 {args.dir} 中没有找到 .pkl 或 .csv 文件")

    # --- 单个 PKL 文件 ---
    elif args.pkl is not None:
        pkl_to_txt(args.pkl, args.robot, args.output, args.output_fps, args.input_fps)

    # --- 单个 CSV 文件 ---
    elif args.csv is not None:
        in_fps = _resolve_input_fps(args)
        csv_to_txt(args.csv, args.robot, args.output, in_fps, args.output_fps)

    else:
        parser.error("必须指定 --pkl, --csv, 或 --dir 中的一个")


"""
使用示例:
 python scripts/CSV/csv_to_txt.py --csv output/linyun/csv/walk_1.33/walk_turn_slow_1.33.csv --robot x3_zq --bvh BVH/linyun/walk_1.33/walk_turn_slow_1.33.bvh --output output/linyun/txt/walk_1.33/walk_turn_slow_1.33.txt --output_fps 50
"""

