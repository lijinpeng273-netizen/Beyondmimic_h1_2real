# -*- coding: utf-8 -*-
"""
hg_config.py — 10-DOF Humanoid-Gym 部署配置（H1 机器人）
============================================================
集中管理 humanoid-gym (Isaac Gym) 10-DOF 策略的部署参数，
包括策略路径、PD 增益、归一化缩放等。

===== 部署切换指南 =====
┌───────────────────┬─────────────┬────────────────┐
│ 参数              │ 仿真值      │ 实机值         │
├───────────────────┼─────────────┼────────────────┤
│ DOMAIN_ID         │ 1           │ 0              │
│ INTERFACE         │ "lo"        │ "enp2s0"       │
│ ENABLE_ELASTIC_BAND│ True       │ False          │
│ RL_ACTION_SMOOTHING│ 0.0        │ 0.1~0.2        │
│ POLICY_PATH       │ 策略.pt     │ 策略.pt        │
└───────────────────┴─────────────┴────────────────┘
===== 部署工作流 =====
1. sim2sim（纯仿真验证）：
   python hg_sim2sim.py --vx=0.4

2. sim2real 联调（DDS + MuJoCo，与 unitree_mujoco.py 配合）：
   先修改 config.py: BRIDGE_MODE="sim2real"
   终端1: python unitree_mujoco.py --task=walk
   终端2: python hg_policy_controller.py

3. 实机部署（直接在机器人上运行）：
   python hg_policy_controller.py

   实机时需通过 --domain/--interface 或直接修改 hg_config.py 中的
   DOMAIN_ID 和 INTERFACE。从 2.0 秒后自动释放弹性绳。
   （hg_policy_controller 会自动读取 hg_config.py 中的参数）

注意：
- BRIDGE_MODE 和 IMU_ON_PELVIS 是 unitree_mujoco.py（仿真桥接端）使用的参数，
  在 config.py 中设置，不属于本文件管理范围。
- RL_ACTION_SMOOTHING=0.0 适用于仿真；实机建议 0.1~0.2 减少电机高频抖动。
- 实机部署前必须设置 ENABLE_ELASTIC_BAND=False（弹性绳仅用于仿真辅助站立）
"""

import numpy as np


# =====================================================================
# 部署模式
# =====================================================================
BRIDGE_MODE = "sim2real"          # sim2sim(纯仿真) / sim2real(DDS联调+实机)
                                 # 注意：此参数供 unitree_mujoco.py 使用，
                                 # hg_policy_controller 不读取此值
ENABLE_ELASTIC_BAND = True        # 虚拟弹簧绳辅助站立（仿真True，实机必须False）

# =====================================================================
# 机器人型号与模型路径
# =====================================================================
ROBOT = "h1"
# 10 电机版本的 H1 MJCF 路径（相对 simulate_python/ 目录）
HG_MJCF_PATH = "/home/ljp/human_learn/humanoid-gym-main-finally/resources/robots/h1/mjcf/h1.xml"

# =====================================================================
# 通信配置（仿真 / 实机切换时修改）
# =====================================================================
DOMAIN_ID = 1        # DDS 域 ID：仿真用 1，实机用 0
INTERFACE = "lo"     # DDS 网卡：仿真用 lo，实机用 enp6s0/enp2s0

# =====================================================================
# 仿真控制参数
# =====================================================================
SIMULATE_DT = 0.001       # 物理仿真步长 (1000 Hz)
VIEWER_DT = 0.02          # 渲染帧率 (50 fps)
DECIMATION = 10           # 每个策略步的物理子步数
POLICY_DT = SIMULATE_DT * DECIMATION  # 策略周期 0.01s (100 Hz)

# =====================================================================
# 桥接层结构常量（必须与 H1Cfg / H1FreeEnv 训练配置完全一致）
# =====================================================================
HG_NUM_ACTION = 10                       # 策略输出维度
HG_NUM_OBS_PER_STEP = 41                 # 单帧观测维度
HG_ACTOR_OBS_HISTORY_LENGTH = 15         # frame_stack (15 帧历史)
HG_POLICY_INPUT_DIM = HG_NUM_OBS_PER_STEP * HG_ACTOR_OBS_HISTORY_LENGTH  # 615

# ----- 关节名称（左右对称，与 Isaac Gym 训练顺序一致） -----
HG_JOINT_NAMES = [
    "left_hip_yaw_joint",       # 0: L_hip_yaw
    "left_hip_roll_joint",      # 1: L_hip_roll
    "left_hip_pitch_joint",     # 2: L_hip_pitch
    "left_knee_joint",          # 3: L_knee
    "left_ankle_joint",         # 4: L_ankle
    "right_hip_yaw_joint",      # 5: R_hip_yaw
    "right_hip_roll_joint",     # 6: R_hip_roll
    "right_hip_pitch_joint",    # 7: R_hip_pitch
    "right_knee_joint",         # 8: R_knee
    "right_ankle_joint",        # 9: R_ankle
]

# ----- 默认关节角（来自 H1Cfg.init_state.default_joint_angles） -----
# 策略输出 action 被解释为 default_dof_pos 附近的偏移：
#   target_q = action * ACTION_SCALE + HG_DEFAULT_DOF_POS
HG_DEFAULT_DOF_POS = np.array(
    [0.0, 0.0, -0.4, 0.8, -0.4,    # 左腿
     0.0, 0.0, -0.4, 0.8, -0.4],   # 右腿
    dtype=np.float64,
)

# ----- PD 增益（来自 H1Cfg.control） -----
# 髋关节 (yaw/roll/pitch)：Kp=200, Kd=5
# 膝关节：Kp=300, Kd=6 
# 踝关节：Kp=40, Kd=2
HG_KPS = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40], dtype=np.float64)
HG_KDS = np.array([5, 5, 5, 6, 5, 5, 5, 5, 6, 5], dtype=np.float64)
HG_TAU_LIMIT = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40], dtype=np.float64)

# ----- PD 增益缩放（实机调参，降低脚部冲击） -----
# sim2sim 保持 1.0（与训练一致），实机部署从 0.6~0.8 开始尝试
# 减小 Kp 降低位置刚度 → 落地更柔顺，减小 Kd 降低阻尼 → 防止硬碰撞反弹
HG_KP_SCALE = 0.8
HG_KD_SCALE = 1.0

# ----- 观测归一化缩放（来自 H1Cfg.normalization.obs_scales） -----
# 【重要】sim2sim/sim2real 必须用与训练完全相同的系数
LIN_VEL_SCALE = 2.0      # 线速度缩放
ANG_VEL_SCALE = 1.0      # 角速度缩放
DOF_POS_SCALE = 1.0      # 关节位置（相对默认值）缩放
DOF_VEL_SCALE = 0.05     # 关节速度缩放
QUAT_SCALE = 1.0         # 欧拉角缩放

CLIP_OBSERVATIONS = 18.0  # 观测裁剪阈值
CLIP_ACTIONS = 18.0       # 动作裁剪阈值

# ----- 物理与策略常数 -----
ACTION_SCALE = 0.25       # 动作缩放（策略输出 = [-1,1]，目标位置 = action * 0.25 + default）
CYCLE_TIME = 0.64         # 步态周期（秒）

# ----- 速度指令限幅（键盘/手柄共用） -----
VEL_X_MAX = 1.0           # 前向最大速度 (m/s)
VEL_X_MIN = -0.5          # 后向最大速度 (m/s) ← 比前向小，防止后退过快摔倒
VEL_Y_MAX = 0.5
VEL_Y_MIN = -0.5
VEL_YAW_MAX = 1.57        # rad/s
VEL_YAW_MIN = -1.57

# ----- 预热 -----
WARMUP_STEPS = 50         # 前 N 步逐步释放策略控制

# =====================================================================
# IMU 补偿（实机 IMU 在 Torso，策略训练在 Pelvis 坐标系）
# 当 waist_yaw 关节不处于零位时，Torso 的 IMU 读数与 Pelvis 实际姿态
# 存在 yaw 偏差，需要补偿。10-DOF 策略中 waist_yaw 由 PD Hold 保持在 0 位，
# 偏差很小，但开启此选项可更精确。
# 仿真（h1.xml 无躯干关节，IMU 在 Pelvis）设为 False；
# 实机部署建议设为 True。
ENABLE_IMU_COMPENSATION = False

# 安全保护参数
# =====================================================================
# 动作平滑（一阶低通滤波）
# sim2sim 用 0（不过滤），实机推荐 0.1~0.2 减少电机高频抖动
RL_ACTION_SMOOTHING = 0.0

# 动作异常检测
SAFETY_MAX_ACTION_NORM = 50.0     # 动作向量最大范数
SAFETY_Q_DES_DELTA_MAX = 1.0      # 每帧目标位置最大变化 (rad)

# 跌落检测（实机建议开启）
# 通过 IMU 姿态判断机器人是否摔倒，超出限制后自动断开 RL
# enable: 总开关；orientation_limit: roll/pitch 角度阈值 (rad)
SAFETY_ENABLE_FALL_DETECTION = True
SAFETY_FALL_ORIENTATION_LIMIT = 1.2   # ≈ 69°，超过此角度视为摔倒

# 通信断连检测（实机建议开启）
# 如果 lowstate 超过此时间未更新，自动断开 RL
SAFETY_COMMS_TIMEOUT = 0.3        # 秒

# PD Hold 平滑插值时间（断开 RL 后从当前位姿缓变到默认姿态）
PD_HOLD_INTERP_S = 2.0

# 控制循环周期（= POLICY_DT = 0.01s，100Hz）
CONTROL_DT = POLICY_DT

# =====================================================================
# 默认策略路径
# =====================================================================
# hg_sim2sim.py 和 hg_policy_controller.py 的 --load_model 默认值。
# 如果通过命令行传了 --load_model，则覆盖此路径。
POLICY_PATH = "../logs/H1_ppo/exported/policies/h1_policy_jit.pt"


