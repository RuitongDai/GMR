import re, os, ntpath
import numpy as np
from . import utils

# 欧拉角通道名称映射字典
# 用于将 BVH 文件中 CHANNELS 定义的长字符串缩写为计算用的短字符
channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    动画数据容器对象，用来打包从 BVH 文件中解析出来的各种矩阵和拓扑信息。
    """

    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: 局部四元数张量 (每一帧每个关节相对于父关节的旋转)
        :param pos: 局部位置张量 (通常只有根节点有动态位置变化，子节点的位置一般由 offset 决定)
        :param offsets: 局部关节偏移量 (即静止姿态下，子关节相对父关节的位置，也是骨骼长度)
        :param parents: 骨骼层级拓扑树 (用索引数组表示，如 parents[i] = j 表示第 i 个关节的父关节是第 j 个)
        :param bones: 骨骼/关节名称列表
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    读取 BVH 文件并提取动画信息。
    【核心逻辑】：先解析 HIERARCHY 建立骨骼树状结构，再解析 MOTION 逐帧填充旋转和位移数据。

    :param filename: BVH 文件路径
    :param start: 起始帧 (截取用)
    :param end: 结束帧 (截取用)
    :param order: 强制指定的欧拉角旋转顺序 (若为 None，则代码会自动从文件 CHANNELS 属性中解析)
    :return: 包含提取信息的 Anim 对象
    """

    f = open(filename, "r")

    i = 0
    active = -1  # 当前正在处理的父节点索引
    end_site = False  # 标记当前是否正在解析End Site块

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # 逐行解析 BVH 文件
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        # 匹配根节点 (ROOT)
        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)  # 默认单位四元数
            parents = np.append(parents, active)  # 根节点的 active 是 -1 (无父节点)
            active = (len(parents) - 1)  # 将游标移动到当前根节点
            continue

        if "{" in line: continue

        # 匹配大括号结束，意味着当前层级遍历完毕，游标退回到父节点
        if "}" in line:
            if end_site:
                end_site = False  # 如果刚才是End Site，只取消标记，不退级 (因为 End Site 不算骨骼节点)
            else:
                active = parents[active]  # 向上回退一层
            continue

        # 匹配相对偏移量 (OFFSET)
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                # 记录当前骨骼的偏移量 (End Site 的偏移量被忽略)
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        # 匹配通道信息 (CHANNELS) -> 这是自动识别 YXZ 还是 XYZ 的关键！
        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:  # 如果没有硬编码指定顺序
                # 如果是 3 通道(只有旋转)，取第 2~5 个单词；如果是 6 通道(位移+旋转)，取第 5~8 个单词
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                # 动态拼接出欧拉角顺序字符串，例如 "yxz"
                order = "".join([channelmap[p] for p in parts])
            continue

        # 匹配普通子关节 (JOINT)
        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)  # 记录它的父节点是当前 active 游标
            active = (len(parents) - 1)  # 游标下移到新关节
            continue

        # 匹配末端节点
        if "End Site" in line:
            end_site = True
            continue

        # 匹配总帧数
        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            # 初始化存储位移和旋转的大数组 (fnum帧 * N个关节 * 3维)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        # 匹配帧率/采样周期
        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))  # 例如 0.01 表示 100Hz
            continue

        # 过滤不需要的帧 (如果指定了 start 和 end)
        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        # ==========================================
        # 开始解析真正的逐帧运动数据块 (MOTION block)
        # ==========================================
        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i

            # 3 通道：代表没有根节点绝对位移的数据 (较少见)
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)

            # 6 通道：绝大多数正常 BVH 文件的格式
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                # 每行前 3 个数是位移 (只有根节点有效，其余子节点全为0)
                positions[fi, :] = data_block[:, 0:3]
                # 后 3 个数是对应顺序的欧拉角
                rotations[fi, :] = data_block[:, 3:6]

            # 9 通道：某些特殊软件导出的带缩放(Scale)的数据格式
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    # 将欧拉角按照刚才动态解析出的 order (如 'yxz') 统一转换为四元数
    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    # 消除四元数的正负号翻转不连续性 (防止插值或网络学习时出现突变)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    提取与 LAFAN1 论文中相同的测试/训练集数据。
    这是一个针对机器学习预处理的函数，它将长动画切割成重叠的短窗口，并提取足端接触标签。

    :param bvh_path: BVH 数据集文件夹路径
    :param actors: 需要提取的动捕演员列表 (通过文件名前缀区分)
    :param window: 滑动窗口的宽度 (例如每次切 50 帧，约等于 1.6 秒的动作)
    :param offset: 滑动窗口的步长 (偏移 20 帧，意味着相邻窗口有重叠，用于增加数据量)
    :return: 包含局部位置、四元数、拓扑树以及左右脚接触标签的元组
    """
    npast = 10  # 用于朝向统一的历史参考帧数
    subjects = []
    seq_names = []
    X = []  # 存储切片后的位置
    Q = []  # 存储切片后的旋转
    contacts_l = []  # 左脚是否触地标签
    contacts_r = []  # 右脚是否触地标签

    # 遍历文件夹提取数据
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # 使用滑动窗口切分连续的长动画
                i = 0
                while i + window < anim.pos.shape[0]:
                    # 【核心物理计算】：利用正向运动学 (FK)，通过局部旋转和局部偏移量，
                    # 计算出当前窗口内所有关节在世界坐标系下的绝对坐标 (x)
                    q, x = utils.quat_fk(anim.quats[i: i + window], anim.pos[i: i + window], anim.parents)

                    # 提取脚底接触地面的标签 (基于脚底的运动速度阈值 velfactor=0.02)
                    # 索引 [3,4] 和 [7,8] 分别代表 LAFAN1 骨骼中左右脚的末端关节
                    c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)

                    X.append(anim.pos[i: i + window])
                    Q.append(anim.quats[i: i + window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset  # 窗口向前滑动

    # 将列表转换为 Numpy 矩阵，形状通常为 (BatchSize, WindowSize, NumJoints, Dim)
    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # -----------------------------------------------------
    # 数据归一化对齐 (Data Normalization & Root Centering)
    # -----------------------------------------------------
    # 1. 位置归零：将每一段动作的 X 轴和 Z 轴(地面平面)的绝对位移统一平移到原点附近
    # 使得神经网络不需要学习由于演员起步位置不同带来的绝对坐标差异
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # 2. 朝向归一化：将每一段动作的初始面朝方向旋转对齐到统一直线上 (如 Z 轴正方向)
    # 保证策略网络学到的动作独立于机器人的初始朝向
    X, Q = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r


def get_train_stats(bvh_folder, train_set):
    """
    提取训练集数据，以计算用于神经网络输入的统计学归一化参数 (Mean 和 Std)。
    强化学习或动作生成模型在输入数据前，通常需要进行 Z-Score 标准化。

    :return: (局部位置均值向量, 局部位置标准差向量, 局部关节偏移量张量)
    """
    print('Building the train set...')
    xtrain, qtrain, parents, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # 关节偏移量 (Offsets) 决定了骨架比例，是固定常量，所以只取第一帧的即可
    offsets = xtrain[0:1, 0:1, 1:, :]  # 形状 : (1, 1, J, 3)

    # 计算全局表征 (将局部相对坐标转换为全局绝对坐标)
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # 计算全局位置的均值 (Mean) 和 标准差 (Std)
    # 在序列 (Batch) 和时间步 (Time) 维度上进行平均，保留关节维度
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                     keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                   keepdims=True)

    return x_mean, x_std, offsets