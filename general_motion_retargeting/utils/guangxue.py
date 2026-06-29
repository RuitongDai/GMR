import numpy as np
from scipy.spatial.transform import Rotation as R
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

def load_guangxue_bvh_file(bvh_file):
    """
    光学动捕设备 BVH 读取适配器。
    核心功能：
    1. 提取所有骨骼的全局位置与全局四元数
    2. 将 BVH 的全局坐标系转换为 Z-up 且面朝X的 MuJoCo 世界坐标系。
    3. 提取专属的足端特征 (LeftFootMod / RightFootMod)
    """
    # 1. 读取 bvh 文件中的骨骼层级、局部旋转和局部位置数据
    data = read_bvh(bvh_file)

    # 2. 使用四元数正向运动学 (FK)，计算出所有骨骼的全局位置和全局旋转
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # 3. 核心坐标系转换
    rotation_matrix = np.array([
        [0, 0, 1],  # 新的 X 轴 = 原来的 Z 轴
        [1, 0, 0],  # 新的 Y 轴 = 原来的 X 轴
        [0, 1, 0]  # 新的 Z 轴 = 原来的 Y 轴
    ])
    # 将旋转矩阵转换为四元数
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []

    # 遍历所有动作帧
    for frame in range(data.pos.shape[0]):
        result = {}
        # 遍历当前帧中的所有骨骼
        for i, bone in enumerate(data.bones):
            # 将全局旋转矩阵左乘到当前骨骼的朝向上
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            # 将旋转矩阵应用到全局位置上
            position = global_data[1][frame, i] @ rotation_matrix.T / 100
            # 存入字典
            result[bone] = [position, orientation]

        # 4. 足端接触特征构造 (Foot Modification)
        # 对于双足机器人，提取脚踝位置配合脚尖朝向，用于地面接触约束
        result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
        result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        # 将当前帧加入总序列
        frames.append(result)

    human_height = 1.75

    return frames, human_height