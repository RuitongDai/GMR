"""
可视化 AMP 格式的机器人运动 TXT 文件到 MuJoCo 查看器

使用方式:
  python scripts/CSV/vis_txt_motion.py \
      --robot x3_zq \
      --txt /home/hwz/GIT-USST/Gym/AMP_for_DroidUp/humanoid/envs/datasets/x3_zq/walk_fb_12s_100hz.txt

  python scripts/CSV/vis_txt_motion.py \
      --robot x3_zq \
      --txt /tmp/test_x3_zq.txt \
      --record_video --video_path /tmp/output.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from general_motion_retargeting import RobotMotionViewer

# ---------------------------------------------------------------------------
# 加载 TXT 运动数据
# ---------------------------------------------------------------------------

# 位置向量大小（x, y, z）
POS_SIZE = 3
# 旋转四元数大小（x, y, z, w）
ROT_SIZE = 4


def load_txt_motion(txt_path):
    """从 AMP TXT 文件加载机器人运动数据。

    Args:
        txt_path: AMP 格式 TXT 文件路径

    Returns:
        root_pos: 每帧的基座位置 (N, 3)
        root_rot: 每帧的基座旋转四元数 (N, 4) 格式为 wxyz
        dof_pos: 每帧的关节位置 (N, num_joints)
        fps: 帧率
    """
    # 读取 JSON 格式的运动数据文件
    with open(txt_path) as f:
        data = json.load(f)

    # 从帧间隔计算帧率
    frame_duration = float(data["FrameDuration"])
    fps = 1.0 / frame_duration
    # 将所有帧转换为 numpy 数组
    frames = np.array(data["Frames"], dtype=np.float64)

    # 提取基座位置 (x, y, z)
    root_pos = frames[:, :POS_SIZE].copy()
    # 提取基座旋转四元数 (x, y, z, w) 格式
    root_rot = frames[:, POS_SIZE:POS_SIZE + ROT_SIZE].copy()  # xyzw
    # 转换为 (w, x, y, z) 格式（MuJoCo 和 RobotMotionViewer 所需的格式）
    root_rot = root_rot[:, [3, 0, 1, 2]]  # xyzw -> wxyz

    # 计算关节自由度数量
    # 总列数 - 位置(3) - 旋转(4) - 线性速度(3) - 角速度(3) - 关节速度
    num_joints = (frames.shape[1] - POS_SIZE - ROT_SIZE - 6 - 3 - 3) // 2
    # 关节位置数据在数组中的起始索引
    joint_pos_start = POS_SIZE + ROT_SIZE
    # 关节位置数据的结束索引
    joint_pos_end = joint_pos_start + num_joints
    # 提取关节位置
    dof_pos = frames[:, joint_pos_start:joint_pos_end].copy()

    print(f"加载完成：{len(frames)} 帧，{fps:.0f} Hz，{num_joints} 个自由度")

    return root_pos, root_rot, dof_pos, fps


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="在 MuJoCo 中可视化 AMP TXT 运动文件")
    parser.add_argument("--robot", required=True, help="机器人类型名称（如 x3_zq）")
    parser.add_argument("--txt", required=True, help="TXT 运动文件的路径")
    parser.add_argument("--record_video", action="store_true", help="是否录制视频")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4", help="视频输出路径")
    args = parser.parse_args()

    # 检查 TXT 文件是否存在
    if not os.path.exists(args.txt):
        raise FileNotFoundError(f"TXT 文件未找到: {args.txt}")

    # 加载运动数据
    root_pos, root_rot, dof_pos, fps = load_txt_motion(args.txt)

    # 创建机器人运动查看器
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    # 循环播放运动
    frame_idx = 0
    while True:
        # 逐帧更新查看器中的机器人位置和姿态
        viewer.step(
            root_pos[frame_idx],
            root_rot[frame_idx],
            dof_pos[frame_idx],
            rate_limit=True,  # 限制帧率以匹配原始 FPS
        )
        frame_idx += 1
        # 循环播放：到最后一帧后回到第一帧
        if frame_idx >= len(root_pos):
            frame_idx = 0
    viewer.close()
