#!/usr/bin/env python3
"""
动作数据后处理：基座高度动态平滑补偿 (Dynamic Foot Contact Smoother)

严格适配 GMR 格式 (root_rot 为 xyzw)。
利用 MuJoCo 前向推演获取每帧脚底物理边缘的绝对高度，采用高斯平滑滤波计算动态补偿量。
"""

import argparse
import os
import pickle
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d


def print_header(title):
    print("=" * 70)
    print(f"🌟 {title}")
    print("=" * 70)


def smooth_motion_contact(
        pkl_path: str,
        xml_path: str,
        foot_geom_names: list,
        sigma: float,
        preserve_jumps: bool,
        plot: bool
):
    # 1. 加载GMR处理后的原始pkl数据 (root_rot 为 xyzw 顺序)
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"找不到动作文件: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps", 30)
    root_pos = data["root_pos"]
    root_rot_xyzw = data["root_rot"]
    dof_pos = data["dof_pos"]
    num_frames = root_pos.shape[0]

    # MuJoCo 物理推演强依赖 wxyz，推演时做临时转换，不污染原始保存数据
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    # 2. 载入 MuJoCo 物理引擎
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"找不到 XML 模型: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(model)

    # 自动探测脚底几何体的 ID 和半厚度
    foot_geom_info = []
    for name in foot_geom_names:
        g_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if g_id == -1:
            raise ValueError(f"❌ XML 中找不到几何体: {name}")

        # 智能读取：如果是 Box (长方体)，其 Z 轴半厚度存在 size[2] 中
        geom_type = model.geom_type[g_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_thickness = model.geom_size[g_id][2]
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            half_thickness = model.geom_size[g_id][0]  # 球体取半径
        else:
            half_thickness = 0.0  # 兜底逻辑

        foot_geom_info.append({"id": g_id, "name": name, "thickness": half_thickness})

    print_header(f"开始处理: {os.path.basename(pkl_path)}")
    print(f"🔹 帧数: {num_frames} | FPS: {fps}")
    for info in foot_geom_info:
        print(f"🔹 追踪碰撞体: [{info['name']}] -> 自动检测半厚度: {info['thickness']} m")

    # 3. 第一遍推演：计算每帧的真实物理最低点
    lowest_foot_z = np.zeros(num_frames)

    print("⏳ 正在进行物理引擎前向推演...")
    for i in range(num_frames):
        mj_data.qpos[:3] = root_pos[i]
        mj_data.qpos[3:7] = root_rot_wxyz[i]
        mj_data.qpos[7:7 + dof_pos.shape[1]] = dof_pos[i]
        mujoco.mj_kinematics(model, mj_data)

        # 高度 = 几何体中心高度 - 几何体半厚度 (得到真正的底部边缘)
        z_heights = [mj_data.geom_xpos[g["id"]][2] - g["thickness"] for g in foot_geom_info]
        lowest_foot_z[i] = min(z_heights)

    # 4. 计算误差与平滑补偿量
    error_z = lowest_foot_z.copy()

    # 跳跃保护：如果脚离地超过 5cm，视为合法腾空，不对该帧计算向下的引力
    if preserve_jumps:
        error_z[error_z > 0.05] = 0.0

    print(f"🌊 应用高斯平滑滤波 (Sigma={sigma})...")
    smoothed_offset = gaussian_filter1d(error_z, sigma=sigma)

    # 5. 应用补偿 (只修改基座 Z 轴，其他完全保持 GMR 原样)
    new_root_pos = root_pos.copy()
    new_root_pos[:, 2] -= smoothed_offset

    # 6. 保存新数据
    base, ext = os.path.splitext(pkl_path)
    new_pkl_path = f"{base}_smoothed{ext}"

    new_data = data.copy()
    new_data["root_pos"] = new_root_pos
    with open(new_pkl_path, "wb") as f:
        pickle.dump(new_data, f)

    print(f"✅ 处理完成！已保存至: {new_pkl_path}")

    # 7. 可视化分析
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.style.use('ggplot')
            plt.figure(figsize=(12, 6))

            time_axis = np.arange(num_frames) / fps
            plt.plot(time_axis, lowest_foot_z, label='Raw Lowest Foot Z (Error)', color='red', alpha=0.4)
            plt.plot(time_axis, smoothed_offset, label=f'Smoothed Compensation (sigma={sigma})', color='blue',
                     linewidth=2)
            plt.plot(time_axis, lowest_foot_z - smoothed_offset, label='Corrected Lowest Foot Z', color='green',
                     linewidth=2)

            plt.axhline(0, color='black', linestyle='--', label='Ground Level (Z=0)')
            plt.title("Foot Contact Dynamic Compensation Analysis")
            plt.xlabel("Time (s)")
            plt.ylabel("Height (m)")
            plt.legend()
            plt.tight_layout()

            plot_path = f"{base}_analysis.png"
            plt.savefig(plot_path)
            print(f"📊 分析图表已保存至: {plot_path}")
        except ImportError:
            print("⚠️ 未安装 matplotlib，跳过绘制图表。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动作数据脚部动态贴地平滑后处理工具 (GMR原生格式)")
    parser.add_argument("--pkl", type=str, required=True, help="输入的 .pkl 动作文件路径")
    parser.add_argument("--xml", type=str, required=True, help="机器人的 XML 模型路径")
    parser.add_argument("--foot_geoms", type=str, nargs='+', default=["l_foot_collision", "r_foot_collision"])
    parser.add_argument("--sigma", type=float, default=3.0, help="高斯滤波系数 (推荐 2.0 ~ 5.0)")
    parser.add_argument("--preserve_jumps", action="store_true", help="保留明显的腾空跳跃期不被强制拉回地面")
    parser.add_argument("--no_plot", action="store_true", help="禁用图表生成")

    args = parser.parse_args()

    smooth_motion_contact(args.pkl, args.xml, args.foot_geoms, args.sigma, args.preserve_jumps, not args.no_plot)