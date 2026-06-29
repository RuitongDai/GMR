import numpy as np
from scipy.spatial.transform import Rotation as R
import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

def load_noitom_bvh_file(bvh_file):
    """
    诺亦腾 (Noitom) 惯性动捕设备 BVH 读取适配器。
    核心功能：
    1. 提取所有骨骼的全局位置与全局四元数
    2. 解决坐标系对齐：将诺亦腾的全局坐标系转换为 Z-up 且面朝 +X 的 MuJoCo 坐标系。
    3. 特殊脚部处理：适配诺亦腾无 Toe(脚尖) 关节的骨架拓扑。
    """
    # 1. 读取 bvh 数据
    data = read_bvh(bvh_file)

    # 2. 正向运动学 (FK) 解算全局数据
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # 3. 核心坐标系转换
    rotation_matrix = np.array([
        [0, 0, 1],  # 新的 X 轴 = 原来的 Z 轴 (面朝前)
        [1, 0, 0],  # 新的 Y 轴 = 原来的 X 轴 (朝左)
        [0, 1, 0]  # 新的 Z 轴 = 原来的 Y 轴 (朝上)
    ])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []

    # 遍历所有动作帧
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            # 全局旋转修正
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            # 全局位置修正 (单位：厘米转米)
            position = global_data[1][frame, i] @ rotation_matrix.T / 100
            result[bone] = [position, orientation]

        # 4. 诺亦腾足端特征提取 (Foot Modification)
        # 因为诺亦腾 Foot 往下直接是 End Site，没有旋转通道。
        # 因此 FootMod 的【位置】和【朝向】，全部使用 Foot 节点自身的数据。
        if "LeftFoot" in result and "RightFoot" in result:
            result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftFoot"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightFoot"][1]]

        frames.append(result)

    human_height = 1.75

    return frames, human_height