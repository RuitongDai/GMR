import argparse
import pickle
import time
import os
import sys
import numpy as np
import mujoco
import mujoco.viewer


class PlayerState:
    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.current_frame = 0
        self.is_paused = False
        self.start_clip = 0
        self.end_clip = num_frames - 1
        self.fps = 30.0
        self.need_save = False


def load_motion_data(pkl_path):
    """加载 .pkl 动作数据"""
    with open(pkl_path, "rb") as f:
        motion_data = pickle.load(f)
    return motion_data


def save_clipped_data(pkl_path, original_data, start_idx, end_idx):
    """裁剪并保存数据"""
    if start_idx >= end_idx:
        print("\n[Error] 终点必须大于起点！保存失败。")
        return

    clipped_data = original_data.copy()
    clipped_data["root_pos"] = original_data["root_pos"][start_idx:end_idx + 1]
    clipped_data["root_rot"] = original_data["root_rot"][start_idx:end_idx + 1]
    clipped_data["dof_pos"] = original_data["dof_pos"][start_idx:end_idx + 1]

    if original_data.get("local_body_pos") is not None:
        clipped_data["local_body_pos"] = original_data["local_body_pos"][start_idx:end_idx + 1]

    # 生成新文件名
    dir_name = os.path.dirname(pkl_path)
    base_name = os.path.basename(pkl_path).replace(".pkl", "")
    new_name = f"{base_name}_clip_{start_idx}_to_{end_idx}.pkl"
    save_path = os.path.join(dir_name, new_name)

    with open(save_path, "wb") as f:
        pickle.dump(clipped_data, f)

    print(f"\n[Success] 裁剪成功！已保存至: {save_path}")


def print_status(state):
    """在终端实时覆盖打印状态"""
    status = "PAUSED " if state.is_paused else "PLAYING"
    sys.stdout.write(
        f"\r[{status}] Frame: {state.current_frame:04d} / {state.num_frames:04d} | "
        f"Clip Range: [{state.start_clip} : {state.end_clip}] | "
        f"FPS: {state.fps}  "
    )
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E1 动作数据播放与剪辑工具")
    parser.add_argument("--pkl_file", type=str, required=True, help="要播放的 .pkl 文件路径")
    parser.add_argument("--xml_file", type=str, default="assets/e1/E1_25dof.xml", help="机器人的 XML 模型路径")
    args = parser.parse_args()

    # 1. 加载模型
    if not os.path.exists(args.xml_file):
        print(f"找不到模型文件: {args.xml_file}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(args.xml_file)
    data = mujoco.MjData(model)

    # 2. 加载动作数据
    motion_data = load_motion_data(args.pkl_file)
    root_pos = motion_data["root_pos"]
    root_rot_xyzw = motion_data["root_rot"]
    dof_pos = motion_data["dof_pos"]

    num_frames = len(root_pos)
    state = PlayerState(num_frames)
    state.fps = motion_data.get("fps", 30.0)

    print("=" * 60)
    print("🎬 动作剪辑器已启动！")
    print("------------------------------------------------------------")
    print("快捷键说明：")
    print("  [Space]  : 播放 / 暂停")
    print("  [Left]   : 上一帧 (暂停状态下)")
    print("  [Right]  : 下一帧 (暂停状态下)")
    print("  [        : 将当前帧设为剪切起点")
    print("  ]        : 将当前帧设为剪切终点")
    print("  S        : 保存剪切后的数据")
    print("============================================================")


    # 3. 键盘回调函数
    def key_callback(keycode):
        if keycode == ord(' '):  # 空格
            state.is_paused = not state.is_paused
        elif keycode == 262:  # 右方向键 (GLFW_KEY_RIGHT)
            if state.is_paused:
                state.current_frame = min(state.current_frame + 10, state.end_clip)
        elif keycode == 263:  # 左方向键 (GLFW_KEY_LEFT)
            if state.is_paused:
                state.current_frame = max(state.current_frame - 10, state.start_clip)
        elif keycode == ord('['):  # 设为起点
            if state.current_frame < state.end_clip:
                state.start_clip = state.current_frame
        elif keycode == ord(']'):  # 设为终点
            if state.current_frame > state.start_clip:
                state.end_clip = state.current_frame
        elif keycode == ord('S') or keycode == ord('s'):  # 保存
            state.need_save = True

        print_status(state)


    # 4. 启动可视化窗口
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 调整一下视角
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 90
        viewer.cam.lookat[:] = [0, 0, 0.6]

        while viewer.is_running():
            step_start = time.time()

            # 检查是否需要保存
            if state.need_save:
                save_clipped_data(args.pkl_file, motion_data, state.start_clip, state.end_clip)
                state.need_save = False
                print_status(state)  # 恢复状态栏

            # 获取当前帧数据
            idx = state.current_frame
            pos = root_pos[idx]
            rot_xyzw = root_rot_xyzw[idx]
            # MuJoCo 需要 wxyz 格式，所以转换回来
            rot_wxyz = np.array([rot_xyzw[3], rot_xyzw[0], rot_xyzw[1], rot_xyzw[2]])
            joints = dof_pos[idx]

            # 拼装完整的 qpos 并注入物理引擎
            qpos = np.concatenate([pos, rot_wxyz, joints])
            data.qpos[:] = qpos

            # 使用 forward 而不是 step，因为我们只是查看运动学姿态，不需要跑动力学仿真
            mujoco.mj_forward(model, data)
            viewer.sync()

            # 终端状态更新
            print_status(state)

            # 帧数步进逻辑
            if not state.is_paused:
                state.current_frame += 1
                # 如果超出了剪辑终点，或者播到了末尾，则回到剪辑起点循环播放
                if state.current_frame > state.end_clip or state.current_frame >= state.num_frames:
                    state.current_frame = state.start_clip

            # 维持设定的 FPS 帧率
            time_until_next_step = (1.0 / state.fps) - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)