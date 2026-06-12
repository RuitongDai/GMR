import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print


class GeneralMotionRetargeting:
    """
    通用动作重定向引擎 (General Motion Retargeting, GMR)。
    核心功能：接收人类的骨骼姿态数据 (Human_Frame)，结合 IK 配置文件中的权重和偏移量，
    通过求解带约束的逆运动学 (Inverse Kinematics)，计算出机器人各关节的目标角度 (qpos)。
    """

    def __init__(
            self,
            src_human: str,  # 动捕数据源名称 (如 "bvh_noitom")
            tgt_robot: str,  # 目标机器人名称 (如 "unitree_x3")
            actual_human_height: float = None,  # 实际人类身高 (用于自动缩放计算)
            solver: str = "daqp",  # 优化的底层求解器 (从 quadprog 改为了更高效的 daqp)
            damping: float = 5e-1,  # 阻尼系数：防止 IK 求解时关节出现奇异点导致的剧烈突变
            verbose: bool = True,  # 是否打印日志
            use_velocity_limit: bool = False,  # 是否启用关节速度限制
    ) -> None:

        # 1. 加载目标机器人的 MuJoCo 物理模型
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)

        # 2. 建立机器人模型的信息字典 (DOF, Body, Motor)
        # 这些字典在调试和匹配模型时非常有用
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' 是自由度的数量
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")

        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' 是连杆(刚体)的数量
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")

        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' 是电机的数量
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # 3. 加载对应的 IK 匹配配置文件
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])

        # 4. 身高比例缩放计算
        # 如果读取到了人类身高，计算其与 JSON 中假设身高(1.75m)的比例
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0

        # 根据实际身高比例，等比例缩放 JSON 中的人体各部位长度系数
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio

        # 5. 从 JSON 中提取重定向所需的核心参数
        self.ik_match_table1 = ik_config["ik_match_table1"]  # 基础姿态追踪表
        self.ik_match_table2 = ik_config.get("ik_match_table2", {})  # 第二阶段微调表(如果存在的话)
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config.get("use_ik_match_table2", False)
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])  # 地面高度的 Z 轴向量

        self.max_iter = 10  # 每一帧 IK 求解的最大迭代次数 (数值越大越精准，但计算越慢)

        self.solver = solver
        self.damping = damping

        # 初始化存储任务对象和偏移量的字典
        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        # 设置机器人的物理约束限制 (如关节不能超过最大旋转角度)
        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            # 限制关节的最大角速度
            VELOCITY_LIMITS = {k: 3 * np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

            # 6. 配置逆运动学追踪任务
        self.setup_retarget_configuration()

        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        """
        初始化 mink 的配置和追踪任务 (FrameTask)。
        遍历 IK 配置表，为每一个需要追踪的连杆创建一个任务目标。
        """
        self.configuration = mink.Configuration(self.model)

        self.tasks1 = []
        self.tasks2 = []

        # 遍历第一张追踪表
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            # 只有位置或旋转权重不为 0 时，才加入求解器任务
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,  # 设定追踪人类位置的优先级权重
                    orientation_cost=rot_weight,  # 设定追踪人类朝向的优先级权重
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                # 解析配置表中的四元数，指定标量(w)在首位格式
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []

        # 遍历第二张追踪表 (同上，通常用于高精度的末端微调)
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

    def update_targets(self, human_data, offset_to_ground=False):
        """
        在每一帧被调用，将人类动作数据进行几何变换，并更新为 IK 求解器的追踪目标 (SE3空间)。
        """
        # 1. 格式转换
        human_data = self.to_numpy(human_data)

        # 2. 比例缩放：把人类的大长腿缩短到机器人的腿长
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)

        # 3. 应用局部偏移：坐标系误差对齐
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)

        # 4. 地面平移约束
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        self.scaled_human_data = human_data  # 保存处理后的数据供外部可视化调用

        # 5. 更新 mink 任务目标：设置每个身体部位期望的绝对位置和绝对旋转
        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                # SE3 表示三维空间中的刚体变换 (位移 + 旋转)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def retarget(self, human_data, offset_to_ground=False):
        """
        【主求解循环】：处理单帧数据，返回机器人的目标关节角度数组 (qpos)。
        """
        # 1. 更新本帧的目标姿态
        self.update_targets(human_data, offset_to_ground)

        # 2. 第一阶段优化求解 (利用梯度下降思路迭代求解目标位姿)
        if self.use_ik_match_table1:
            curr_error = self.error1()  # 计算当前姿态与目标姿态的空间误差
            dt = self.configuration.model.opt.timestep

            # 使用 QP 求解器算出机器人的关节速度 (vel)
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            # 将算出的速度积分为当前帧的目标角度
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0

            # 如果误差依然大于阈值(0.001) 且未达到最大迭代次数(10)，则继续迭代逼近
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        # 3. 第二阶段微调求解 (逻辑同上)
        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)

                next_error = self.error2()
                num_iter += 1

        # 返回积分完成后的机器人全部关节位置(角度)
        return self.configuration.data.qpos.copy()

    def error1(self):
        """计算第一组任务的总体欧氏距离误差 (用于判断 IK 是否收敛)"""
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )

    def error2(self):
        """计算第二组任务的总体误差"""
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )

    def to_numpy(self, human_data):
        """将从底层解析器传来的列表统一转换为 Numpy 数组加速计算"""
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data

    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        """
        基于 Root (骨盆) 节点为原点，对人类的动作数据进行骨架等比例缩放。
        这能保证人类举起手的高度，按照比例缩放后恰好是机器人举起手的物理极限高度。
        """
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]

        # 缩放根节点的高度位移
        scaled_root_pos = human_scale_table[human_root_name] * root_pos

        # 将子节点的绝对坐标转换为相对于 Root 的局部坐标，然后缩放
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # 局部坐标 (仅处理位置) = (绝对坐标 - 根节点坐标) * 缩放比例
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]

        # 将缩放后的局部坐标再加回缩放后的根节点坐标，还原为新的绝对坐标
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global

    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """
        【跨物种对齐核心逻辑：先应用旋转偏移，再应用位置偏移】
        由于机器人的电机零位和人类自然下垂的参考系不同，必须在此处将人类关节的坐标系掰成和机器人一致。
        """
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]

            # 1. 旋转偏移：四元数乘法，先将局部的朝向纠正过来
            # 注意：scipy.Rotation.from_quat 默认是 [x,y,z,w]，但因为使用了 scalar_first=True
            # 所以这里严格处理了 GMR 体系要求的 [w,x,y,z] 格式
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat

            local_offset = pos_offsets[body_name]
            # 2. 位置偏移：在已经“掰正”的局部坐标系里，运用相对位移向量，推算出需要补偿的全局绝对位置
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)

            # 3. 更新最终的绝对空间位置
            offset_human_data[body_name][0] = pos + global_pos_offset

        return offset_human_data

    def offset_human_data_to_ground(self, human_data):
        """
        动态接地函数：找到人类数据中当前帧最低的点（通常是脚底），
        并把整个人体往下或往上平移，强迫最低点刚好贴住地面 (Z=ground_offset)。
        这用于防止动捕数据起跳或下蹲时出现穿模或悬空。
        """
        offset_human_data = {}
        ground_offset = 0.1  # 默认的地面厚度容差
        lowest_pos = np.inf

        # 寻找本帧中的最低点
        for body_name in human_data.keys():
            # 过滤逻辑：只把 Foot/foot 作为参考最低点
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name

        # 整体平移补偿
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        """设置全局静态的地面高度偏移"""
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        """统一对所有关节点应用全局静态的地面高度平移"""
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data