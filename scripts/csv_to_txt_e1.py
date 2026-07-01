"""
csv_to_txt_e1.py

E1_13dof 专用：把 GMR 重定向输出的 CSV/PKL 转成 AMP 训练用 TXT。

输入 CSV 格式：
    root_pos(3) + root_quat(4, xyzw) + joint_pos(13)

输出 TXT 每帧格式，严格对齐 E1 AMP loader：
    root_pos(3)
    root_quat(4, xyzw)
    joint_pos(13)
    foot_pos(6)
    root_lin_vel(3)
    root_ang_vel(3)
    joint_vel(13)

所以 E1 每帧固定 45 维：
    3 + 4 + 13 + 6 + 3 + 3 + 13 = 45
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting.params import ROBOT_XML_DICT


# =============================================================================
# E1 固定配置
# =============================================================================

E1_ROBOT_NAME = "e1"
E1_DOF = 13
E1_LEFT_FOOT_BODY = "left_ankle_roll_link"
E1_RIGHT_FOOT_BODY = "right_ankle_roll_link"

POS_SIZE = 3
ROT_SIZE = 4
FOOT_POS_SIZE = 6
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3

E1_TXT_FRAME_SIZE = (
    POS_SIZE + ROT_SIZE + E1_DOF + FOOT_POS_SIZE
    + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE + E1_DOF
)

E1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
]


# =============================================================================
# Quaternion / interpolation 工具函数
# =============================================================================

def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """四元数格式转换：xyzw -> wxyz。"""
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def _quat_mul_wxyz(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """wxyz 四元数乘法，支持 broadcast。"""
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ], axis=-1)


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """wxyz 四元数共轭。"""
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)


def _quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把世界系向量 v 通过 q 的逆旋转转到根/机体系。"""
    q_conj = _quat_conj_wxyz(q)
    zeros = np.zeros_like(v[..., :1])
    v_quat = np.concatenate([zeros, v], axis=-1)
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(q_conj, v_quat), q)
    return rotated[..., 1:]


def _axis_angle_from_quat_wxyz(q: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """从 wxyz 四元数计算旋转向量 axis * angle。"""
    # 强制 w >= 0，取最短旋转。
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


def _lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """线性插值。"""
    return a * (1.0 - blend) + b * blend


def _quat_slerp_xyzw(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """xyzw 四元数球面线性插值。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))

    if dot < 0.0:
        b = -b
        dot = -dot

    if dot > 0.9995:
        out = a + t * (b - a)
        return out / np.linalg.norm(out)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / sin_0
    s1 = np.sin(theta) / sin_0
    return s0 * a + s1 * b


# =============================================================================
# 输入帧率与重采样
# =============================================================================

def read_bvh_frame_duration(bvh_path: str) -> float:
    """从 BVH 文件里读取 Frame Time。"""
    with open(bvh_path) as f:
        for line in f:
            m = re.match(r"^Frame Time:\s*([\d.]+)", line, re.IGNORECASE)
            if m:
                return float(m.group(1))
    raise ValueError(f"Frame Time not found in BVH: {bvh_path}")


def resample_motion(root_pos: np.ndarray,
                    root_rot_xyzw: np.ndarray,
                    dof_pos: np.ndarray,
                    input_fps: float,
                    output_fps: float):
    """把 root_pos/root_quat/dof_pos 从 input_fps 重采样到 output_fps。"""
    input_frames = root_pos.shape[0]
    if input_frames < 2:
        return root_pos, root_rot_xyzw, dof_pos

    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (input_frames - 1) * input_dt
    output_frames = int(duration / output_dt) + 1

    times = np.linspace(0.0, duration, output_frames)
    phase = times / duration
    idx0 = np.floor(phase * (input_frames - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - idx0

    root_pos_out = _lerp(root_pos[idx0], root_pos[idx1], blend[:, None])
    dof_pos_out = _lerp(dof_pos[idx0], dof_pos[idx1], blend[:, None])
    root_rot_out = np.stack([
        _quat_slerp_xyzw(root_rot_xyzw[i0], root_rot_xyzw[i1], b)
        for i0, i1, b in zip(idx0, idx1, blend)
    ])
    return root_pos_out, root_rot_out, dof_pos_out


# =============================================================================
# E1 正运动学、速度计算、帧组装
# =============================================================================

def init_e1_kinematics():
    """初始化 E1 的 GMR KinematicsModel，并拿到左右脚 body index。"""
    robot_xml = ROBOT_XML_DICT.get(E1_ROBOT_NAME)
    if robot_xml is None:
        raise KeyError(
            f"ROBOT_XML_DICT 里找不到 '{E1_ROBOT_NAME}'。\n"
            f"请在 general_motion_retargeting/params.py 里注册 E1 XML。\n"
            f"当前可用机器人：{sorted(ROBOT_XML_DICT.keys())}"
        )

    kin = KinematicsModel(str(robot_xml), device="cpu")
    left_foot_idx = kin.get_body_idx(E1_LEFT_FOOT_BODY)
    right_foot_idx = kin.get_body_idx(E1_RIGHT_FOOT_BODY)

    if kin.num_dof != E1_DOF:
        raise ValueError(
            f"KinematicsModel 加载的 E1 DOF = {kin.num_dof}，但脚本期望 {E1_DOF}。\n"
            f"请检查 ROBOT_XML_DICT['{E1_ROBOT_NAME}'] 是否指向 13DOF XML。"
        )

    print(f"E1 XML       : {robot_xml}")
    print(f"E1 DOF       : {kin.num_dof}")
    print(f"Left foot    : {E1_LEFT_FOOT_BODY}  idx={left_foot_idx}")
    print(f"Right foot   : {E1_RIGHT_FOOT_BODY} idx={right_foot_idx}")
    return kin, left_foot_idx, right_foot_idx


def compute_body_kinematics(root_pos: np.ndarray,
                            root_rot_xyzw: np.ndarray,
                            dof_pos: np.ndarray,
                            kin: KinematicsModel,
                            output_dt: float):
    """用 E1 正运动学计算 body 位姿，并用差分计算速度。"""
    root_pos_t = torch.from_numpy(root_pos).float()
    root_rot_t = torch.from_numpy(root_rot_xyzw).float()  # GMR 使用 xyzw
    dof_pos_t = torch.from_numpy(dof_pos).float()

    body_pos, body_rot_xyzw = kin.forward_kinematics(root_pos_t, root_rot_t, dof_pos_t)
    body_pos_w = body_pos.numpy().astype(np.float64)
    body_quat_wxyz = _xyzw_to_wxyz(body_rot_xyzw.numpy().astype(np.float64))

    T = body_pos_w.shape[0]
    if T >= 2:
        body_lin_vel_w = np.gradient(body_pos_w, output_dt, axis=0)
        joint_vel = np.gradient(dof_pos, output_dt, axis=0)
    else:
        body_lin_vel_w = np.zeros_like(body_pos_w)
        joint_vel = np.zeros_like(dof_pos)

    if T >= 3:
        q_prev = body_quat_wxyz[:-2]
        q_next = body_quat_wxyz[2:]
        q_rel = _quat_mul_wxyz(q_next, _quat_conj_wxyz(q_prev))
        omega = _axis_angle_from_quat_wxyz(q_rel) / (2.0 * output_dt)
        body_ang_vel_w = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
    else:
        body_ang_vel_w = np.zeros((T, body_pos_w.shape[1], 3), dtype=np.float64)

    return body_pos_w, body_quat_wxyz, body_lin_vel_w, body_ang_vel_w, joint_vel


def build_e1_frames(root_pos: np.ndarray,
                    root_rot_xyzw: np.ndarray,
                    dof_pos: np.ndarray,
                    output_dt: float,
                    kin: KinematicsModel,
                    left_foot_idx: int,
                    right_foot_idx: int) -> np.ndarray:
    """组装 E1 45 维 AMP TXT 帧。"""
    if dof_pos.shape[1] != E1_DOF:
        raise ValueError(f"E1 期望 {E1_DOF} 个关节角，但输入是 {dof_pos.shape[1]} 维。")

    body_pos_w, body_quat_wxyz, body_lin_vel_w, body_ang_vel_w, joint_vel = compute_body_kinematics(
        root_pos, root_rot_xyzw, dof_pos, kin, output_dt
    )

    root_body_pos_w = body_pos_w[:, 0, :]
    root_body_quat_wxyz = body_quat_wxyz[:, 0, :]

    def rel_body_pos_in_root(body_idx: int) -> np.ndarray:
        rel_w = body_pos_w[:, body_idx, :] - root_body_pos_w
        return _quat_rotate_inverse_wxyz(root_body_quat_wxyz, rel_w)

    foot_pos = np.concatenate([
        rel_body_pos_in_root(left_foot_idx),
        rel_body_pos_in_root(right_foot_idx),
    ], axis=1)

    # 和训练环境里的 base_lin_vel/base_ang_vel 对齐：都使用根/机体系速度。
    root_lin_vel = _quat_rotate_inverse_wxyz(root_body_quat_wxyz, body_lin_vel_w[:, 0, :])
    root_ang_vel = _quat_rotate_inverse_wxyz(root_body_quat_wxyz, body_ang_vel_w[:, 0, :])

    frames = np.concatenate([
        root_pos.astype(np.float64),       # [0:3]
        root_rot_xyzw.astype(np.float64),  # [3:7]
        dof_pos.astype(np.float64),        # [7:20]
        foot_pos.astype(np.float64),       # [20:26]
        root_lin_vel.astype(np.float64),   # [26:29]
        root_ang_vel.astype(np.float64),   # [29:32]
        joint_vel.astype(np.float64),      # [32:45]
    ], axis=1)

    if frames.shape[1] != E1_TXT_FRAME_SIZE:
        raise RuntimeError(f"E1 TXT 帧维度应该是 {E1_TXT_FRAME_SIZE}，实际是 {frames.shape[1]}。")
    return frames


def write_amp_txt(frames: np.ndarray, output_dt: float, output_path: str, motion_weight: float = 1.0):
    """写出 JSON 风格 AMP TXT。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_time = (len(frames) - 1) * output_dt

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

    print(f"已保存: {output_path}")
    print(f"帧数: {frames.shape[0]}，每帧维度: {frames.shape[1]}，总时长: {total_time:.4f}s")


def print_layout_summary():
    """打印 E1 TXT 布局。"""
    layout = [
        ("root_pos", 0, 3),
        ("root_quat_xyzw", 3, 7),
        ("joint_pos", 7, 20),
        ("foot_pos", 20, 26),
        ("root_lin_vel", 26, 29),
        ("root_ang_vel", 29, 32),
        ("joint_vel", 32, 45),
    ]
    print("\nE1 AMP TXT layout:")
    for name, start, end in layout:
        print(f"  {name:<15} [{start:>2}:{end:>2}]  {end - start:>2} dims")
    print("")


# =============================================================================
# CSV / PKL 输入
# =============================================================================

def process_motion(root_pos: np.ndarray,
                   root_rot_xyzw: np.ndarray,
                   dof_pos: np.ndarray,
                   input_fps: float,
                   output_fps: float,
                   output_path: str):
    """统一处理入口：重采样 -> FK -> 组装 45 维 -> 写 txt。"""
    if dof_pos.shape[1] != E1_DOF:
        raise ValueError(
            f"E1 CSV/PKL 应该包含 {E1_DOF} 个关节角，但当前是 {dof_pos.shape[1]}。\n"
            f"输入应该是 root_pos(3)+root_quat(4,xyzw)+joint_pos(13)，CSV 总列数应为 20。"
        )

    kin, left_foot_idx, right_foot_idx = init_e1_kinematics()
    root_pos_rs, root_rot_rs, dof_pos_rs = resample_motion(
        root_pos, root_rot_xyzw, dof_pos, input_fps, output_fps
    )
    output_dt = 1.0 / output_fps

    print(f"重采样: {root_pos.shape[0]} frames @ {input_fps:.3f}Hz -> "
          f"{root_pos_rs.shape[0]} frames @ {output_fps:.3f}Hz")

    frames = build_e1_frames(
        root_pos_rs, root_rot_rs, dof_pos_rs, output_dt,
        kin, left_foot_idx, right_foot_idx
    )
    write_amp_txt(frames, output_dt, output_path)
    print_layout_summary()


def csv_to_txt(csv_path: str, output_path: str, input_fps: float, output_fps: float):
    """E1 CSV -> AMP TXT。CSV: root_pos(3)+root_quat(4,xyzw)+joint_pos(13)。"""
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 7 + E1_DOF:
        raise ValueError(
            f"E1 CSV 应该有 {7 + E1_DOF} 列，实际有 {data.shape[1]} 列。\n"
            f"期望布局: root_pos(3)+root_quat(4,xyzw)+joint_pos(13)。"
        )

    root_pos = data[:, 0:3]
    root_rot_xyzw = data[:, 3:7]
    dof_pos = data[:, 7:]

    print(f"读取 CSV: {csv_path}")
    print(f"CSV shape: {data.shape}")
    process_motion(root_pos, root_rot_xyzw, dof_pos, input_fps, output_fps, output_path)


def pkl_to_txt(pkl_path: str, output_path: str, output_fps: float, input_fps: float | None = None):
    """E1 PKL -> AMP TXT。PKL 需要 root_pos/root_rot/dof_pos。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
    fps = input_fps if input_fps is not None else float(data.get("fps", 100.0))

    print(f"读取 PKL: {pkl_path}")
    print(f"root_pos: {root_pos.shape}, root_rot: {root_rot_xyzw.shape}, dof_pos: {dof_pos.shape}, fps={fps}")
    process_motion(root_pos, root_rot_xyzw, dof_pos, fps, output_fps, output_path)


def resolve_input_fps(args) -> float:
    """解析 CSV 输入帧率。优先级：--input_fps > --frame_duration > --bvh > 默认 100Hz。"""
    if args.input_fps is not None:
        return float(args.input_fps)
    if args.frame_duration is not None:
        return 1.0 / float(args.frame_duration)
    if args.bvh is not None:
        fd = read_bvh_frame_duration(args.bvh)
        print(f"从 BVH 读取 Frame Time: {fd}s -> {1.0 / fd:.3f}Hz")
        return 1.0 / fd
    print("没有指定输入帧率，默认使用 100Hz。")
    return 100.0


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E1_13dof 专用 CSV/PKL -> AMP TXT 转换脚本，输出固定 45 维。"
    )
    parser.add_argument("--csv", default=None, help="输入 CSV 文件")
    parser.add_argument("--pkl", default=None, help="输入 PKL 文件")
    parser.add_argument("--dir", default=None, help="批量处理目录下所有 CSV 或 PKL")
    parser.add_argument("--output", required=True, help="输出 TXT 文件；使用 --dir 时为输出目录")
    parser.add_argument("--input_fps", type=float, default=None, help="CSV 输入帧率")
    parser.add_argument("--frame_duration", type=float, default=None, help="CSV 输入帧时长，单位秒")
    parser.add_argument("--bvh", default=None, help="从 BVH 的 Frame Time 读取 CSV 输入帧率")
    parser.add_argument("--output_fps", type=float, default=50.0, help="输出 TXT 帧率，默认 50Hz")
    args = parser.parse_args()

    if sum(x is not None for x in [args.csv, args.pkl, args.dir]) != 1:
        parser.error("--csv、--pkl、--dir 必须且只能指定一个。")

    if args.dir is not None:
        csv_files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
        pkl_files = sorted(glob.glob(os.path.join(args.dir, "*.pkl")))
        os.makedirs(args.output, exist_ok=True)

        if csv_files:
            input_fps = resolve_input_fps(args)
            print(f"批量处理 CSV: {len(csv_files)} 个文件")
            for csv_file in csv_files:
                stem = Path(csv_file).stem
                out_file = os.path.join(args.output, f"{stem}.txt")
                csv_to_txt(csv_file, out_file, input_fps, args.output_fps)
        elif pkl_files:
            print(f"批量处理 PKL: {len(pkl_files)} 个文件")
            for pkl_file in pkl_files:
                stem = Path(pkl_file).stem
                out_file = os.path.join(args.output, f"{stem}.txt")
                pkl_to_txt(pkl_file, out_file, args.output_fps, args.input_fps)
        else:
            parser.error(f"目录 {args.dir} 下没有找到 .csv 或 .pkl 文件。")

    elif args.csv is not None:
        csv_to_txt(args.csv, args.output, resolve_input_fps(args), args.output_fps)

    elif args.pkl is not None:
        pkl_to_txt(args.pkl, args.output, args.output_fps, args.input_fps)