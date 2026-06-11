import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sys
import os
import json

# 动态加载器映射：根据输入的 format 自动选择不同的数据加载函数
LOADER_MAP = {
    "noitom": "general_motion_retargeting.utils.noitom_bvh.load_noitom_bvh_file",
}

# 核心配置字典：统一管理不同数据源到不同机器人的映射关系
MAPPING_CONFIGS = {
    "noitom_to_x3": {
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
    },
}


def dynamic_import(func_path):
    module_path, func_name = func_path.rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def calculate_offsets(bvh_path, xml_path, mapping, data_format):
    # 1. 加载数据
    loader_func = dynamic_import(LOADER_MAP[data_format])
    frames, _ = loader_func(bvh_path)
    frame0 = frames[0]

    # 2. 加载机器人
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    # 调用正向运动学计算出每个连杆在三维空间中的绝对坐标轴
    mujoco.mj_kinematics(model, data)

    # 3. 计算结果
    result_dict = {}
    for robot_link, human_bone in mapping.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_link)
        if body_id == -1 or human_bone not in frame0:
            continue

        R_robot = R.from_matrix(data.xmat[body_id].reshape(3, 3))

        quat_wxyz = frame0[human_bone][1]
        R_human = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])

        R_offset = R_human.inv() * R_robot
        offset_xyzw = R_offset.as_quat()
        offset_wxyz = np.round([offset_xyzw[3], offset_xyzw[0], offset_xyzw[1], offset_xyzw[2]], 4).tolist()

        weight = 100 if "ankle" in robot_link else 10
        result_dict[robot_link] = [human_bone, offset_wxyz]

    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动计算静止姿态标系旋转偏移")
    parser.add_argument("--bvh", type=str, required=True, help="BVH 第0帧应为静止姿态 (A/T-Pose)")
    parser.add_argument("--xml", type=str, required=True, help="机器人 URDF/XML 模型路径")
    parser.add_argument("--config", type=str, choices=MAPPING_CONFIGS.keys(), default="noitom_to_x3",
                        help="映射配置方案")
    parser.add_argument("--format", type=str, choices=LOADER_MAP.keys(), default="noitom", help="数据源解析格式")
    args = parser.parse_args()

    print(f"正在计算 {args.config} 的旋转偏移矩阵...")
    mapping = MAPPING_CONFIGS[args.config]
    ik_table = calculate_offsets(args.bvh, args.xml, mapping, args.format)

    # 直接以完美格式化的 JSON 字符串输出，方便直接复制
    print("\n" + "=" * 50)
    print('"ik_match_table1": ' + json.dumps(ik_table, indent=4))
    print("=" * 50 + "\n")
