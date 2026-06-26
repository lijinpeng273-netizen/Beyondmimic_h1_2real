# -*- coding: utf-8 -*-
"""
h1sim2sim_v2_config.py — H1 Sim2Sim 仿真配置文件
==================================================
集中管理所有仿真参数。修改此文件即可切换模型 / 调整仿真行为，
无需改动主脚本。

用法:
    python h1sim2sim_v2.py          # 直接运行，自动读取本配置文件
    python h1sim2sim_v2.py --config my_config.py  # 使用自定义配置文件
"""

import numpy as np

# ============================================================================
# 模型与场景文件路径
# ============================================================================

# MuJoCo XML 机器人模型（H1 19 关节 + 1 个 not_use_joint）
XML_PATH = "unitree_description/mjcf/h1.xml"

# ONNX 策略模型
POLICY_PATH = "deploy_real/bydmimic/2026-06-22_19-48-01_last_run1.onnx"

# ============================================================================
# 仿真时序
# ============================================================================

SIMULATION_DT = 0.005          # 物理仿真步长 (s)，与 Isaac Lab 训练保持一致
CONTROL_DECIMATION = 4         # 控制分频：每 N 个物理步执行一次策略推理
                               # 策略频率 = 1 / (SIMULATION_DT * DECIMATION) = 50 Hz
SIMULATION_DURATION = 60.0     # 仿真总时长 (s)

# ============================================================================
# H1 机器人结构参数
# ============================================================================

NUM_ACTIONS = 19               # 受控关节数
NUM_OBS = 110                  # 观测向量维度（与训练时一致）
REFERENCE_BODY = "pelvis"      # 锚点刚体名称
INIT_PELVIS_Z = 0.95          # 初始骨盆离地高度 (m)

# ============================================================================
# 弹性绳（辅助站立，仅仿真使用）
# ============================================================================

ELASTIC_BAND_ENABLED = True    # 仿真启动时是否启用
ELASTIC_BAND_STIFFNESS = 300.0 # 刚度 Kp (N/m)
ELASTIC_BAND_DAMPING = 150.0   # 阻尼 Kd (N·s/m)
ELASTIC_BAND_ROPE_Z = 3.0      # 绳子固定点 Z 坐标

# ============================================================================
# PD 增益 —— 仅在 ONNX 元数据缺失时作为后备
# ============================================================================

# 站立姿态使用策略增益的倍数（更高增益 → 更稳地站住）
STAND_STIFFNESS_SCALE = 2.0
STAND_DAMPING_SCALE = 2.0

# H1 各关节默认 PD 增益 (Nm/rad)，后备值
# 关节名必须与 h1.xml 完全一致（共 20 个，含 not_use_joint）
DEFAULT_KP = {
    # 左腿
    "left_hip_yaw_joint": 80,    "left_hip_roll_joint": 150,  "left_hip_pitch_joint": 150,
    "left_knee_joint": 200,      "left_ankle_joint": 80,
    # 右腿
    "right_hip_yaw_joint": 80,   "right_hip_roll_joint": 150, "right_hip_pitch_joint": 150,
    "right_knee_joint": 200,     "right_ankle_joint": 80,
    # 腰部
    "torso_joint": 150,
    # 左臂（含 shoulder_yaw，H1 无 wrist 关节）
    "left_shoulder_pitch_joint": 40,  "left_shoulder_roll_joint": 40,
    "left_shoulder_yaw_joint": 20,    "left_elbow_joint": 20,
    # 右臂
    "right_shoulder_pitch_joint": 40, "right_shoulder_roll_joint": 40,
    "right_shoulder_yaw_joint": 20,   "right_elbow_joint": 20,
}
DEFAULT_KD_RATIO = 0.05         # 阻尼比 = Kd / Kp

# ============================================================================
# 键盘控制键位
# ============================================================================

KEY_STAND = "6"       # 重置到站立姿态 (STATE_IDLE)
KEY_PLAY = "7"        # 开始策略播放 (STATE_PLAYING)
KEY_ELASTIC = "8"      # 切换弹性绳开关
# MuJoCo 窗口内: BACKSPACE 重置仿真, 右键拖拽旋转视角, 滚轮缩放

# ============================================================================
# 渲染
# ============================================================================

VIEWER_FPS = 50        # 渲染帧率（不影响物理精度）
PRINT_INTERVAL = 100   # 每 N 个策略步打印一次状态
