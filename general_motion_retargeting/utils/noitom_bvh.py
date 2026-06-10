import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_noitom_bvh_file(bvh_file):
    """
    专门解析诺亦腾 (Noitom) 导出的带 End Site 的 BVH 数据。
    返回的字典结构将直接用于 GMR 的逆运动学 (IK) 求解。
    """
    # 1. 调用底层解析器读取文件
    data = read_bvh(bvh_file)

    # 2. 计算出所有关节的全局位置 (Global Position) 和全局四元数 (Global Quat)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    # 3. 坐标系转换矩阵：将 Y-up 转换为 Z-up，以适配绝大多数双足机器人 (如 G1, X3)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)  # 格式为 [w, x, y, z]

    frames = []

    # BVH文件中脚尖 (End Site) 相对于脚踝 (LeftFoot/RightFoot) 的局部偏移量
    foot_end_offset_m = np.array([0.0, -0.10, 0.1512])

    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            # 乘以转换矩阵，将姿态转换到 Z-up 坐标系
            # orientation 格式为 [w, x, y, z]
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100

            result[bone] = [position, orientation]

        # ==========================================
        # 构造脚底板 IK 目标 (FootMod)
        # ==========================================

        # 获取左右脚踝在当前帧的全局位置和旋转 (Z-up 坐标系下)
        l_foot_pos, l_foot_quat_wxyz = result["LeftFoot"]
        r_foot_pos, r_foot_quat_wxyz = result["RightFoot"]

        # scipy 的 R.from_quat 默认接收 [x, y, z, w] 格式，所以需要将 [w, x, y, z] 重新排列一下
        l_foot_quat_xyzw = [l_foot_quat_wxyz[1], l_foot_quat_wxyz[2], l_foot_quat_wxyz[3], l_foot_quat_wxyz[0]]
        r_foot_quat_xyzw = [r_foot_quat_wxyz[1], r_foot_quat_wxyz[2], r_foot_quat_wxyz[3], r_foot_quat_wxyz[0]]

        # 计算绝对世界坐标下的脚尖/脚底位置：
        # 公式：全局脚踝坐标 + (脚踝全局旋转矩阵 * 局部偏移量)
        l_foot_end_pos = l_foot_pos + R.from_quat(l_foot_quat_xyzw).apply(foot_end_offset_m)
        r_foot_end_pos = r_foot_pos + R.from_quat(r_foot_quat_xyzw).apply(foot_end_offset_m)

        # 生成 FootMod：使用计算出的脚尖位置，旋转依然沿用脚踝的旋转
        result["LeftFootMod"] = [l_foot_end_pos, l_foot_quat_wxyz]
        result["RightFootMod"] = [r_foot_end_pos, r_foot_quat_wxyz]

        frames.append(result)

    # 估算身高：通常用头顶 Z 轴最高点减去脚底 Z 轴最低点
    # 为了简化计算并保证稳定，这里设定一个默认身高 1.75 米。你可以根据你的数据进行微调。
    human_height = 1.75

    return frames, human_height