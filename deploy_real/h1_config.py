"""H1 机器人硬性参数定义。

所有与机器人硬件、关节布局、电机映射、观测维度相关的固定常量，
集中在此文件中维护。部署脚本和验证脚本均从此处引用，避免分散硬编码。

数据来源:
    - joint_xml / dof_idx: H1 MuJoCo XML 模型 + Unitree 电机 ID 布局
    - joint_seq: ONNX 策略元数据 joint_names（与训练时完全一致）
    - NUM_ACTIONS / NUM_OBS: 策略网络输入输出维度

命名约定: 使用 MuJoCo/ONNX 原生命名（left_/right_ 前缀），
         与 Isaac Lab 训练环境保持一致。
"""

# ============================================================================
# 策略维度
# ============================================================================
NUM_ACTIONS = 19       # 受控关节数（策略输出维度，H1 全身 19 DOF）
NUM_OBS = 110           # 观测向量维度

# ============================================================================
# 运动锚点
# ============================================================================
REFERENCE_BODY = "pelvis"  # 运动锚点参考刚体名称

# ============================================================================
# 关节定义
# ============================================================================

# joint_seq: ONNX 策略元数据中的关节顺序（与训练时 Isaac Lab H1 资产一致）
#            索引 0-18，对应 default_angles_seq / action_scale_seq 的顺序
#            用途：观测构造中 qpos_seq - default_angles_seq，策略输出的 action 排列
#            注意: H1 没有 wrist 关节，有 shoulder_yaw_joint
joint_seq = [
    "left_hip_yaw_joint",           # 0
    "right_hip_yaw_joint",          # 1
    "torso_joint",                  # 2   (腰部偏航，即 waist_yaw)
    "left_hip_roll_joint",          # 3
    "right_hip_roll_joint",         # 4
    "left_shoulder_pitch_joint",    # 5
    "right_shoulder_pitch_joint",   # 6
    "left_hip_pitch_joint",         # 7
    "right_hip_pitch_joint",        # 8
    "left_shoulder_roll_joint",     # 9
    "right_shoulder_roll_joint",    # 10
    "left_knee_joint",              # 11
    "right_knee_joint",             # 12
    "left_shoulder_yaw_joint",      # 13  (H1 特有肩部偏航，G1 此处为 wrist)
    "right_shoulder_yaw_joint",     # 14
    "left_ankle_joint",             # 15
    "right_ankle_joint",            # 16
    "left_elbow_joint",             # 17
    "right_elbow_joint",            # 18
]

# joint_xml: MuJoCo XML / URDF 模型中的关节顺序（按电机 ID 映射排列）
#           与 DDS motor_state 的读取顺序一致
#           包含 19 个活跃关节，不包括 not_use_joint
joint_xml = [
    # 左腿 (电机 7, 3, 4, 5, 10)
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_joint",
    # 右腿 (电机 8, 0, 1, 2, 11)
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_joint",
    # 腰部 (电机 6)
    "torso_joint",
    # 左臂 (电机 16, 17, 18, 19)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    # 右臂 (电机 12, 13, 14, 15)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

# dof_idx: 电机硬件 ID 序列，与 joint_xml 一一对应
# 用于从 low_state.motor_state[dof_idx[i]] 读取对应关节状态
# 注意：H1 电机索引非顺序排列，不同关节类型的电机 ID 交错分布
dof_idx = [
    # 左腿: left_hip_yaw(7), left_hip_roll(3), left_hip_pitch(4), left_knee(5), left_ankle(10)
    7, 3, 4, 5, 10,
    # 右腿: right_hip_yaw(8), right_hip_roll(0), right_hip_pitch(1), right_knee(2), right_ankle(11)
    8, 0, 1, 2, 11,
    # 腰部: torso_joint(6)
    6,
    # 左臂: left_shoulder_pitch(16), left_shoulder_roll(17), left_shoulder_yaw(18), left_elbow(19)
    16, 17, 18, 19,
    # 右臂: right_shoulder_pitch(12), right_shoulder_roll(13), right_shoulder_yaw(14), right_elbow(15)
    12, 13, 14, 15,
]

# ============================================================================
# 电机总数（含预留未用通道）
# ============================================================================
NUM_MOTORS = 20  # H1 电机索引 0~19，其中电机 9 预留未用 (not_use_joint)

# ============================================================================
# 电机布局说明（注释性常量，供参考）
# ============================================================================
#   DOF pos | 关节名                   | 电机ID | 说明
#   ───────────────────────────────────────────────────
#    0      | left_hip_yaw_joint       |  7     | 左髋偏航
#    1      | left_hip_roll_joint      |  3     | 左髋侧摆
#    2      | left_hip_pitch_joint     |  4     | 左髋俯仰
#    3      | left_knee_joint          |  5     | 左膝
#    4      | left_ankle_joint         | 10     | 左踝
#    5      | right_hip_yaw_joint      |  8     | 右髋偏航
#    6      | right_hip_roll_joint     |  0     | 右髋侧摆
#    7      | right_hip_pitch_joint    |  1     | 右髋俯仰
#    8      | right_knee_joint         |  2     | 右膝
#    9      | right_ankle_joint        | 11     | 右踝
#   10      | torso_joint              |  6     | 腰偏航
#   11      | left_shoulder_pitch_joint| 16     | 左肩俯仰
#   12      | left_shoulder_roll_joint | 17     | 左肩侧摆
#   13      | left_shoulder_yaw_joint  | 18     | 左肩偏航 (H1特有)
#   14      | left_elbow_joint         | 19     | 左肘
#   15      | right_shoulder_pitch_joint| 12    | 右肩俯仰
#   16      | right_shoulder_roll_joint | 13    | 右肩侧摆
#   17      | right_shoulder_yaw_joint | 14     | 右肩偏航 (H1特有)
#   18      | right_elbow_joint        | 15     | 右肘
#
# 注意：电机 9（not_use_joint）在 H1 硬件上未连接任何关节
