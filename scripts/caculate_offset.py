import json
import numpy as np
import argparse
import mujoco
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting.utils.guangxue import load_guangxue_bvh_file
from general_motion_retargeting.utils.noitom import load_noitom_bvh_file


class AutoOffsetCalibrator:
    def __init__(self, bvh_path: str, xml_path: str, json_path: str, bvh_format: str, custom_zero_pose: dict = None):
        self.bvh_path = bvh_path
        self.xml_path = xml_path
        self.json_path = json_path
        self.bvh_format = bvh_format
        self.custom_zero_pose = custom_zero_pose if custom_zero_pose else {}

        self.config_data = None
        self.bvh_frame_0 = None

        # 策略路由表
        self.bvh_loaders = {
            "guangxue": load_guangxue_bvh_file,
            "noitom": load_noitom_bvh_file
        }

    def _load_json(self):
        with open(self.json_path, 'r') as f:
            self.config_data = json.load(f)
        print(f"[1/4] 成功加载配置表: {self.json_path}")

    def _extract_bvh_pose(self):
        if self.bvh_format not in self.bvh_loaders:
            raise ValueError(
                f"❌ 错误: 不支持的 BVH 格式 '{self.bvh_format}'。支持的格式有: {list(self.bvh_loaders.keys())}")

        # 动态调用：根据 format 自动选择函数执行
        loader_function = self.bvh_loaders[self.bvh_format]
        frames, _ = loader_function(self.bvh_path)

        self.bvh_frame_0 = frames[0]
        print(f"[2/4] 成功提取 BVH 参考帧 (当前加载器: {self.bvh_format})")

    def _extract_robot_pose(self, body_names: list) -> dict:
        model = mujoco.MjModel.from_xml_path(self.xml_path)
        data = mujoco.MjData(model)

        # 注入自定义的机器人 A-pose
        if self.custom_zero_pose:
            print(f"  >>> 正在为机器人注入自定义初始姿态...")
            for joint_name, angle_rad in self.custom_zero_pose.items():
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    qpos_adr = model.jnt_qposadr[joint_id]
                    data.qpos[qpos_adr] = angle_rad
                    print(f"      [修改] 关节 '{joint_name}' 角度已设为 {angle_rad:.2f} 弧度")
                else:
                    print(f"      [警告] XML 中找不到关节: {joint_name}")

        mujoco.mj_kinematics(model, data)

        robot_quats = {}
        for body in body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id != -1:
                robot_quats[body] = data.xquat[body_id].tolist()
            else:
                print(f"  [警告] XML 中找不到 Link: {body}")

        print(f"[3/4] 成功提取机器人 XML 零位姿态")
        return robot_quats

    @staticmethod
    def _compute_offset(q_bvh_wxyz, q_robot_wxyz) -> list:
        bvh_xyzw = [q_bvh_wxyz[1], q_bvh_wxyz[2], q_bvh_wxyz[3], q_bvh_wxyz[0]]
        robot_xyzw = [q_robot_wxyz[1], q_robot_wxyz[2], q_robot_wxyz[3], q_robot_wxyz[0]]

        R_offset = R.from_quat(bvh_xyzw).inv() * R.from_quat(robot_xyzw)
        offset_wxyz = np.round(R_offset.as_quat(scalar_first=True), decimals=4)
        return offset_wxyz.tolist()

    def run_calibration(self, output_path=None):
        print("=" * 60)
        print(f" 🚀 启动端到端标定流水线 (BVH 格式: {self.bvh_format})")
        print("=" * 60)

        self._load_json()
        self._extract_bvh_pose()

        ik_table = self.config_data.get('ik_match_table1', self.config_data.get('ik_match_table', {}))
        if not ik_table:
            print("❌ 严重错误: 未找到匹配表！")
            return

        target_bodies = list(ik_table.keys())
        robot_quats = self._extract_robot_pose(target_bodies)

        print(f"\n[4/4] 开始计算并覆写 rot_offset...")
        success_count = 0

        for robot_bone, config_list in ik_table.items():
            if not isinstance(config_list, list) or len(config_list) < 5:
                continue
            bvh_bone = config_list[0]
            if not bvh_bone or bvh_bone == "None" or bvh_bone not in self.bvh_frame_0 or robot_bone not in robot_quats:
                continue

            q_bvh = self.bvh_frame_0[bvh_bone][1]
            q_robot = robot_quats[robot_bone]
            new_offset = self._compute_offset(q_bvh, q_robot)

            config_list[4] = new_offset
            print(f"  [成功] {bvh_bone:<15} -> {robot_bone:<20} | Offset: {new_offset}")
            success_count += 1

        save_path = output_path if output_path else self.json_path
        with open(save_path, 'w') as f:
            json.dump(self.config_data, f, indent=4)

        print("=" * 60)
        print(f" 🎉 标定完成！共完美更新 {success_count} 个关节。")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多源动捕数据自适应标定工具")
    parser.add_argument("--bvh", type=str, required=True, help="输入的 BVH 文件路径")
    parser.add_argument("--xml", type=str, required=True, help="机器人的 MuJoCo XML 路径")
    parser.add_argument("--config", type=str, required=True, help="对应的 JSON 配置表")
    # 把 format 变成必需的或者带 choices 校验的参数，增强鲁棒性
    parser.add_argument("--format", type=str, required=True, choices=["guangxue", "noitom"], help="BVH的数据来源格式")
    parser.add_argument("--output", type=str, default=None, help="另存为路径(可选)")

    args = parser.parse_args()

    # 填入你需要下放变成 A-pose 的机器人的关节名和角度
    my_custom_pose = {
        "left_elbow_joint": 0,
        "right_elbow_joint": 0
    }

    calibrator = AutoOffsetCalibrator(
        bvh_path=args.bvh,
        xml_path=args.xml,
        json_path=args.config,
        bvh_format=args.format,
        custom_zero_pose=my_custom_pose
    )

    calibrator.run_calibration(output_path=args.output)