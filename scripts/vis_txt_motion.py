"""
vis_txt_motion_e1.py

E1_13dof 专用 AMP TXT 可视化脚本。

只支持下面这个 E1 45 维 TXT 格式：
    root_pos(3)
    root_quat(4, xyzw)
    joint_pos(13)
    foot_pos(6)
    root_lin_vel(3)
    root_ang_vel(3)
    joint_vel(13)

使用示例：
    python scripts/CSV/vis_txt_motion_e1.py \
        --txt output/E1_13dof/txt/walk.txt

    python scripts/CSV/vis_txt_motion_e1.py \
        --txt output/E1_13dof/txt/walk.txt \
        --record_video \
        --video_path output/E1_13dof/videos/walk.mp4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from general_motion_retargeting import RobotMotionViewer


# =============================================================================
# E1 固定配置
# =============================================================================

E1_ROBOT_NAME = "e1"
E1_DOF = 13

POS_SIZE = 3
ROT_SIZE = 4
FOOT_POS_SIZE = 6
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
E1_TXT_FRAME_SIZE = 45


# =============================================================================
# TXT 读取
# =============================================================================

def print_layout():
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


def load_e1_txt_motion(txt_path: str, show_layout: bool = True):
    """读取 E1 45维 AMP TXT，并返回 MuJoCo viewer 需要的数据。"""
    with open(txt_path) as f:
        data = json.load(f)

    frame_duration = float(data["FrameDuration"])
    if frame_duration <= 0:
        raise ValueError(f"FrameDuration 必须大于 0，当前是 {frame_duration}")

    fps = 1.0 / frame_duration
    frames = np.asarray(data["Frames"], dtype=np.float64)

    if frames.ndim != 2:
        raise ValueError(f"Frames 必须是二维数组，当前 shape={frames.shape}")
    if frames.shape[1] != E1_TXT_FRAME_SIZE:
        raise ValueError(
            f"E1 TXT 每帧必须是 {E1_TXT_FRAME_SIZE} 维，当前是 {frames.shape[1]} 维。\n"
            f"期望格式: root_pos(3)+root_quat(4)+joint_pos(13)+foot_pos(6)+"
            f"root_lin_vel(3)+root_ang_vel(3)+joint_vel(13)。"
        )

    root_pos = frames[:, 0:3].copy()

    # TXT 里 root_quat 是 xyzw；RobotMotionViewer 需要 wxyz。
    root_quat_xyzw = frames[:, 3:7].copy()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    dof_pos = frames[:, 7:20].copy()
    foot_pos = frames[:, 20:26].copy()
    root_lin_vel = frames[:, 26:29].copy()
    root_ang_vel = frames[:, 29:32].copy()
    joint_vel = frames[:, 32:45].copy()

    if show_layout:
        print_layout()

    total_time = (len(frames) - 1) * frame_duration
    print(f"Loaded E1 TXT: {txt_path}")
    print(f"  frames       : {len(frames)}")
    print(f"  fps          : {fps:.3f} Hz")
    print(f"  total_time   : {total_time:.4f} s")
    print(f"  dof          : {dof_pos.shape[1]}")
    print(f"  root_z range : [{root_pos[:, 2].min():.4f}, {root_pos[:, 2].max():.4f}]")
    print(f"  L foot rel z : [{foot_pos[:, 2].min():.4f}, {foot_pos[:, 2].max():.4f}]")
    print(f"  R foot rel z : [{foot_pos[:, 5].min():.4f}, {foot_pos[:, 5].max():.4f}]")
    print(f"  max |root lin vel| : {np.linalg.norm(root_lin_vel, axis=1).max():.4f}")
    print(f"  max |root ang vel| : {np.linalg.norm(root_ang_vel, axis=1).max():.4f}")
    print(f"  max |joint vel|    : {np.abs(joint_vel).max():.4f}")

    return root_pos, root_quat_wxyz, dof_pos, fps


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1_13dof 专用 AMP TXT 可视化脚本。")
    parser.add_argument("--txt", required=True, help="E1 45维 AMP TXT 文件路径")
    parser.add_argument("--record_video", action="store_true", help="是否录制视频")
    parser.add_argument("--video_path", default="videos/e1_motion.mp4", help="视频输出路径")
    parser.add_argument("--start_frame", type=int, default=0, help="从第几帧开始播放")
    parser.add_argument("--once", action="store_true", help="只播放一遍后退出")
    parser.add_argument("--no_rate_limit", action="store_true", help="不按真实帧率限速播放")
    parser.add_argument("--no_layout", action="store_true", help="不打印 45 维字段布局")
    args = parser.parse_args()

    if not os.path.exists(args.txt):
        raise FileNotFoundError(f"TXT 文件不存在: {args.txt}")

    root_pos, root_rot_wxyz, dof_pos, fps = load_e1_txt_motion(
        args.txt,
        show_layout=not args.no_layout,
    )

    if args.start_frame < 0 or args.start_frame >= len(root_pos):
        raise ValueError(f"--start_frame 必须在 [0, {len(root_pos) - 1}]，当前是 {args.start_frame}")

    viewer = RobotMotionViewer(
        robot_type=E1_ROBOT_NAME,
        motion_fps=fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    frame_idx = args.start_frame
    try:
        while True:
            viewer.step(
                root_pos[frame_idx],
                root_rot_wxyz[frame_idx],
                dof_pos[frame_idx],
                rate_limit=not args.no_rate_limit,
            )

            frame_idx += 1
            if frame_idx >= len(root_pos):
                if args.once:
                    break
                frame_idx = 0
    finally:
        viewer.close()