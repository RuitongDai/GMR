from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
import pickle
import numpy as np
import time
import mujoco


# 1. 定义交互状态机
class ViewerState:
    def __init__(self):
        self.z_offset = 0.0  # 高度补偿值 (米)
        self.need_save = False  # 是否触发保存
        self.is_paused = False  # 是否暂停播放


state = ViewerState()
env = None  # 设为全局变量，以便键盘回调函数访问相机


# 2. 定义键盘回调函数
def keyboard_callback(keycode):
    global env

    if keycode == ord(' '):  # 空格键：暂停 / 播放
        state.is_paused = not state.is_paused
        status = "PAUSED" if state.is_paused else "PLAYING"
        print(f"\r[{status}] 当前高度补偿: {state.z_offset:+.3f}m    ", end="")

    elif keycode == 265:  # 向上方向键 (GLFW_KEY_UP)：抬高 0.5 厘米
        state.z_offset += 0.005
        print(f"\r[Height] 当前高度补偿: {state.z_offset:+.3f}m    ", end="")

    elif keycode == 264:  # 向下方向键 (GLFW_KEY_DOWN)：降低 0.5 厘米
        state.z_offset -= 0.005
        print(f"\r[Height] 当前高度补偿: {state.z_offset:+.3f}m    ", end="")

    elif keycode == ord('S') or keycode == ord('s'):  # S 键：触发保存
        state.need_save = True

    # === 增加键盘强制控制视角功能 (突破鼠标限制) ===
    if env and hasattr(env, 'viewer') and env.viewer is not None:
        try:
            cam = env.viewer.cam
            if keycode == ord('Q') or keycode == ord('q'):
                cam.distance = max(0.1, cam.distance - 0.2)  # Q键：拉近视角 (Zoom in)
            elif keycode == ord('E') or keycode == ord('e'):
                cam.distance += 0.2  # E键：拉远视角 (Zoom out)
            elif keycode == ord('W') or keycode == ord('w'):
                cam.lookat[2] += 0.1  # W键：视点上移 (Pan up)
            elif keycode == ord('Z') or keycode == ord('z'):
                cam.lookat[2] -= 0.1  # Z键：视点下移 (Pan down)
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="e1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    args = parser.parse_args()

    robot_type = args.robot
    robot_motion_path = args.robot_motion_path

    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")

    # 加载原始动作数据
    (motion_data, motion_fps, motion_root_pos, motion_root_rot,
     motion_dof_pos, motion_local_body_pos, motion_link_body_list) = load_robot_motion(robot_motion_path)

    print("=" * 60)
    print("🎬 动作可视化与高度补偿器启动！")
    print("------------------------------------------------------------")
    print("MuJoCo 自由视角鼠标操作指南 (已解锁硬编码限制)：")
    print("  [左键拖动] : 旋转视角")
    print("  [右键拖动] : 平移视角 (Pan)  <-- 如果失效，请用键盘 W/Z 键上移/下移")
    print("  [滚轮滑动] : 缩放视角 (Zoom) <-- 如果失效，请用键盘 Q/E 键拉近/拉远")
    print("------------------------------------------------------------")
    print("快捷键说明：")
    print("  [Space]   : 播放 / 暂停 (建议暂停后微调高度)")
    print("  [↑] / [↓] : 整体抬高 / 降低 0.005m (0.5厘米)")
    print("  [ Q / E ] : 键盘强制 拉近 / 拉远 视角")
    print("  [ W / Z ] : 键盘强制 上移 / 下移 视点")
    print("  [ S ]     : 保存补偿后的动作到新 pkl 文件")
    print("=" * 60)

    # 初始化 Viewer
    env = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        camera_follow=False,
        record_video=args.record_video,
        video_path=args.video_path,
        keyboard_callback=keyboard_callback
    )

    frame_idx = 0
    num_frames = len(motion_root_pos)

    while True:
        step_start = time.time()

        # ================================
        # 处理保存逻辑
        # ================================
        if state.need_save:
            new_root_pos = motion_root_pos.copy()
            new_root_pos[:, 2] += state.z_offset

            save_data = {
                "fps": motion_fps,
                "root_pos": new_root_pos,
                "root_rot": motion_root_rot,
                "dof_pos": motion_dof_pos,
                "local_body_pos": motion_local_body_pos,
                "link_body_list": motion_link_body_list,
            }

            base, ext = os.path.splitext(robot_motion_path)
            new_path = f"{base}_zoffset_{state.z_offset:+.3f}{ext}"

            with open(new_path, "wb") as f:
                pickle.dump(save_data, f)
            print(f"\n✅ [保存成功] 补偿后的文件已保存至: {new_path}")
            state.need_save = False

        # ================================
        # 实时渲染逻辑 (强行接管 MuJoCo，打破视角锁定)
        # ================================
        current_pos = motion_root_pos[frame_idx].copy()
        current_pos[2] += state.z_offset

        # 尝试绕过 RobotMotionViewer，直接把数据塞给底层的原生 MuJoCo 引擎
        # 这将彻底摧毁外层包装器的相机追踪逻辑，实现真·自由视角
        bypassed = False
        if hasattr(env, 'data') and hasattr(env, 'model') and hasattr(env, 'viewer'):
            try:
                env.data.qpos[:3] = current_pos
                env.data.qpos[3:7] = motion_root_rot[frame_idx]
                env.data.qpos[7:7 + len(motion_dof_pos[frame_idx])] = motion_dof_pos[frame_idx]

                mujoco.mj_forward(env.model, env.data)

                # 强制切断相机跟随绑定，设为完全自由模式
                env.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                env.viewer.cam.trackbodyid = -1

                env.viewer.sync()
                bypassed = True
            except Exception:
                pass

        # 如果破解失败（极少情况），退回原版渲染
        if not bypassed:
            env.step(current_pos,
                     motion_root_rot[frame_idx],
                     motion_dof_pos[frame_idx],
                     rate_limit=False)

        # 帧率控制 (手动接管时间，防止画面撕裂或快进)
        time_until_next_step = (1.0 / motion_fps) - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # 步进控制
        if not state.is_paused:
            frame_idx += 1
            if frame_idx >= num_frames:
                frame_idx = 0

    env.close()