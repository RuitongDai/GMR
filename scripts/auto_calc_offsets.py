import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sys
import os
from general_motion_retargeting.utils.noitom_bvh import load_noitom_bvh_file


def main():
    bvh_path = "/home/dai/data/法修散打_chr01.bvh"
    xml_path = "/home/dai/GMR/assets/x3/Moya01_V2.xml"

    if not os.path.exists(bvh_path) or not os.path.exists(xml_path):
        print(f"找不到文件，请检查路径是否正确：\\nBVH: {bvh_path}\\nXML: {xml_path}")
        sys.exit(1)

    print("[1/3] 正在加载 BVH 数据，提取第 0 帧 (A-Pose)...")
    frames, height = load_noitom_bvh_file(bvh_path)
    frame0 = frames[0]

    print("[2/3] 正在加载 MuJoCo 机器人模型，计算默认姿态 (qpos=0)...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # 这一步极其关键，调用正向运动学计算出每个连杆在三维空间中的绝对坐标轴
    mujoco.mj_kinematics(model, data)

    # 之前确定的对应关系表
    mapping = {
        "pelvis": "Hips",
        "left_hip_yaw_link": "LeftUpLeg",
        "left_knee_link": "LeftLeg",
        "left_ankle_roll_link": "LeftFootMod",
        "right_hip_yaw_link": "RightUpLeg",
        "right_knee_link": "RightLeg",
        "right_ankle_roll_link": "RightFootMod",
        "torso_link": "Spine2",
        "left_shoulder_pitch_link": "LeftArm",
        "left_elbow_link": "LeftForeArm",
        "left_wrist_yaw_link": "LeftHand",
        "right_shoulder_pitch_link": "RightArm",
        "right_elbow_link": "RightForeArm",
        "right_wrist_yaw_link": "RightHand"
    }

    print("[3/3] 正在计算跨物种坐标系偏差 (R_offset = R_human_inv * R_robot)...\\n")
    print("=========================================================")
    print(" 计算完成！请将下面的内容直接复制替换到你的 JSON 文件中：")
    print("=========================================================\\n")

    print('    "ik_match_table1": {')

    items_count = len(mapping)
    for idx, (robot_link, human_bone) in enumerate(mapping.items()):
        # 1. 获取机器人连杆的绝对旋转矩阵
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_link)
        if body_id == -1:
            print(f"        // 警告: 在 XML 里找不到连杆 {robot_link}")
            continue

        R_robot_mat = data.xmat[body_id].reshape(3, 3)
        R_robot = R.from_matrix(R_robot_mat)

        # 2. 获取人类骨骼在第0帧的绝对旋转
        if human_bone not in frame0:
            print(f"        // 警告: 在 BVH 里找不到骨骼 {human_bone}")
            continue

        # noitom_bvh.py 返回的是 [w, x, y, z]，scipy 计算需要 [x, y, z, w]
        quat_wxyz = frame0[human_bone][1]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        R_human = R.from_quat(quat_xyzw)

        # 3. 核心计算：求解差值
        R_offset = R_human.inv() * R_robot
        offset_xyzw = R_offset.as_quat()

        # 转换回 [w, x, y, z] 以适配 GMR 的标准
        offset_wxyz = [offset_xyzw[3], offset_xyzw[0], offset_xyzw[1], offset_xyzw[2]]

        # 过滤微小的浮点数误差，比如 0.00000001 会被修整为 0，让配置文件更清晰干净
        offset_wxyz = np.round(offset_wxyz, 4).tolist()

        # 动态赋予权重：脚底板为100，其余为10
        weight = 100 if "ankle" in robot_link else 10

        # 处理 JSON 结尾的逗号
        comma = "," if idx < items_count - 1 else ""

        print(f'        "{robot_link}": [')
        print(f'            "{human_bone}",')
        print(f'            0,')
        print(f'            {weight},')
        print(f'            [0.0, 0.0, 0.0],')
        print(f'            {offset_wxyz}')
        print(f'        ]{comma}')

    print('    }')


if __name__ == "__main__":
    main()