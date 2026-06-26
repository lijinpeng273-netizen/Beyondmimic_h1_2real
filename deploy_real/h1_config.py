"""H1 机器人硬性参数定义。

所有与机器人硬件、关节布局、电机映射、观测维度相关的固定常量，
集中在此文件中维护。部署脚本和验证脚本均从此处引用，避免分散硬编码。

数据来源:
    - joint_xml / dof_idx: H1 URDF/MJCF 模型 + Unitree 电机 ID 布局
    - joint_seq: ONNX 策略训练时的关节分组顺序（按类型对称排列）
    - NUM_ACTIONS / NUM_OBS: 策略网络输入输出维度
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

# joint_seq: ONNX 策略元数据中的关节顺序（按类型分组、左右对称排列）
#            索引 0-18，对应 default_angles_seq / action_scale_seq 的顺序
#            用途：观测构造中 qpos_seq - default_angles_seq，策略输出的 action 排列
joint_seq = [
    "L_hip_yaw_joint",           # 0
    "R_hip_yaw_joint",           # 1
    "L_hip_roll_joint",          # 2
    "R_hip_roll_joint",          # 3
    "L_hip_pitch_joint",         # 4
    "R_hip_pitch_joint",         # 5
    "L_knee_joint",              # 6
    "R_knee_joint",              # 7
    "L_ankle_pitch_joint",       # 8
    "R_ankle_pitch_joint",       # 9
    "waist_yaw_joint",           # 10
    "L_shoulder_pitch_joint",    # 11
    "R_shoulder_pitch_joint",    # 12
    "L_shoulder_roll_joint",     # 13
    "R_shoulder_roll_joint",     # 14
    "L_elbow_joint",             # 15
    "R_elbow_joint",             # 16
    "L_wrist_joint",             # 17 (H1 腕部单自由度，与 G1 三自由度不同)
    "R_wrist_joint",             # 18
]

# joint_xml: URDF/MJCF 模型中的关节顺序（按电机 ID 映射排列）
#           与 d.qpos[7:] / d.qvel[6:] 的数据布局一致
#   L_hip_yaw(m7)  L_hip_roll(m3) L_hip_pitch(m4) L_knee(m5)  L_ankle(m10)
#   R_hip_yaw(m8)  R_hip_roll(m0) R_hip_pitch(m1) R_knee(m2)  R_ankle(m11)
#   waist_yaw(m6)  L_shoulder_pitch(m16) L_shoulder_roll(m17) L_elbow(m18) L_wrist(m19)
#                   R_shoulder_pitch(m12) R_shoulder_roll(m13) R_elbow(m14) R_wrist(m15)
joint_xml = [
    # 左腿 (电机 7, 3, 4, 5, 10)
    "L_hip_yaw_joint", "L_hip_roll_joint", "L_hip_pitch_joint",
    "L_knee_joint", "L_ankle_pitch_joint",
    # 右腿 (电机 8, 0, 1, 2, 11)
    "R_hip_yaw_joint", "R_hip_roll_joint", "R_hip_pitch_joint",
    "R_knee_joint", "R_ankle_pitch_joint",
    # 腰部 (电机 6)
    "waist_yaw_joint",
    # 左臂 (电机 16, 17, 18, 19)
    "L_shoulder_pitch_joint", "L_shoulder_roll_joint",
    "L_elbow_joint", "L_wrist_joint",
    # 右臂 (电机 12, 13, 14, 15)
    "R_shoulder_pitch_joint", "R_shoulder_roll_joint",
    "R_elbow_joint", "R_wrist_joint",
]

# dof_idx: 电机硬件 ID 序列，与 joint_xml 一一对应
# 用于从 low_state.motor_state[dof_idx[i]] 读取对应关节状态
# 注意：H1 电机索引非顺序排列，不同关节类型的电机 ID 交错分布
dof_idx = [
    # 左腿: L_hip_yaw(7), L_hip_roll(3), L_hip_pitch(4), L_knee(5), L_ankle(10)
    7, 3, 4, 5, 10,
    # 右腿: R_hip_yaw(8), R_hip_roll(0), R_hip_pitch(1), R_knee(2), R_ankle(11)
    8, 0, 1, 2, 11,
    # 腰部: waist_yaw(6)
    6,
    # 左臂: L_shoulder_pitch(16), L_shoulder_roll(17), L_elbow(18), L_wrist(19)
    16, 17, 18, 19,
    # 右臂: R_shoulder_pitch(12), R_shoulder_roll(13), R_elbow(14), R_wrist(15)
    12, 13, 14, 15,
]

# ============================================================================
# 电机总数（含预留未用通道）
# ============================================================================
NUM_MOTORS = 20  # H1 电机索引 0~19，其中电机 9 预留未用

# ============================================================================
# 电机布局说明（注释性常量，供参考）
# ============================================================================
#   DOF pos | 关节名              | 电机ID | 说明
#   ─────────────────────────────────────────────
#    0      | L_hip_yaw_joint     |  7     | 左髋偏航
#    1      | L_hip_roll_joint    |  3     | 左髋侧摆
#    2      | L_hip_pitch_joint   |  4     | 左髋俯仰
#    3      | L_knee_joint        |  5     | 左膝
#    4      | L_ankle_pitch_joint | 10     | 左踝
#    5      | R_hip_yaw_joint     |  8     | 右髋偏航
#    6      | R_hip_roll_joint    |  0     | 右髋侧摆
#    7      | R_hip_pitch_joint   |  1     | 右髋俯仰
#    8      | R_knee_joint        |  2     | 右膝
#    9      | R_ankle_pitch_joint | 11     | 右踝
#   10      | waist_yaw_joint     |  6     | 腰偏航
#   11      | L_shoulder_pitch_joint | 16  | 左肩俯仰
#   12      | L_shoulder_roll_joint  | 17  | 左肩侧摆
#   13      | L_elbow_joint        | 18     | 左肘
#   14      | L_wrist_joint        | 19     | 左腕（单自由度）
#   15      | R_shoulder_pitch_joint | 12  | 右肩俯仰
#   16      | R_shoulder_roll_joint  | 13  | 右肩侧摆
#   17      | R_elbow_joint        | 14     | 右肘
#   18      | R_wrist_joint        | 15     | 右腕（单自由度）
#
# 注意：电机 9（not_use）在 H1 硬件上未连接任何关节
