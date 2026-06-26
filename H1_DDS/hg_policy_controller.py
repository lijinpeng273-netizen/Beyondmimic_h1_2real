#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hg_policy_controller.py — 10-DOF H1 策略控制器（Sim2Real）
======================================================================
基于 DDS 的策略控制器，用于 humanoid-gym (Isaac Gym) 训练的 10-DOF
H1 策略。通过 rt/lowstate 读取机器人状态，运行策略推理，通过
rt/lowcmd 发布 PD 目标。

可以连接：
  - 仿真环境：与 unitree_mujoco.py (BRIDGE_MODE="sim2real") 配合使用
  - 实机 H1 机器人：配置 DOMAIN_ID=0, INTERFACE=<网卡名>

数据流：
  rt/lowstate → 从 20 路 DDS 信道提取 10 个下肢自由度 → 构建 41 维观测
  → 15 帧历史 (615 维) → 策略推理 → 10 维 action
  → 映射到 20 路 DDS 格式 → rt/lowcmd

控制（键盘）:
  1: 接合 RL         0: 断开 RL
  4/5: vel_x +/-0.2  6/7: vel_y +/-0.2
  8/9: yaw +/-0.2    d: 打印 PD 调试信息
  r: 重置 episode    q: 退出

控制（游戏手柄，DDS wireless_remote 或 /dev/input/js0）:
  A: 接合 RL         B: 断开 RL
  左摇杆: vel_x/vel_y    右摇杆X: yaw_rate
  Start+Select: 退出

===== sim2real 核心难点 =====
1. 20→10 电机映射：
   DDS 通信总是 20 路电机通道（匹配实机 H1 硬件），但 humanoid-gym 策略
   只控制 10 个腿部关节。必须正确映射 DDS 信道 ↔ 训练关节顺序。
   映射关系见 MJC_TO_HG 和 HG_TO_MJC 的注释。

2. 上半身始终 PD Hold：
   10-DOF 策略不能控制躯干/肩/肘关节（DDS 索引 6, 12-19）。
   这些关节始终用 PD Hold 保持在默认姿态（自然垂放），增益要温和防抖动。

3. IMU 位置补偿：
   unitree_mujoco.py 仿真中 IMU 在 pelvis 上，实机 H1 的 IMU 物理安装在
   Torso（腰部）——在 waist_yaw 关节下游。如果实机偏航关节不在零位，
   IMU 读到的姿态和 pelvis 实际姿态之间有偏航偏移。
   通过 hg_config.ENABLE_IMU_COMPENSATION 控制：
     - 仿真（h1.xml 无躯干关节）：设为 False（IMU 天然在 Pelvis）
     - 实机部署：建议设为 True
   补偿原理：读取 waist_yaw 角度，将 Torso 系 IMU 读数逆变换到 Pelvis 系。

4. CRC 校验：
   Unitree SDK 的 LowCmd_ 消息需要 CRC32 校验码，控制器发送前必须计算。
"""

import argparse
import math
import os
import signal
import struct
import threading
import time

import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

from hg_config import (
    POLICY_PATH,
    DOMAIN_ID,
    INTERFACE,
    HG_NUM_ACTION,
    HG_NUM_OBS_PER_STEP,
    HG_ACTOR_OBS_HISTORY_LENGTH,
    HG_JOINT_NAMES,
    HG_DEFAULT_DOF_POS,
    HG_KPS,
    HG_KDS,
    HG_KP_SCALE,
    HG_KD_SCALE,
    LIN_VEL_SCALE,
    ANG_VEL_SCALE,
    DOF_POS_SCALE,
    DOF_VEL_SCALE,
    QUAT_SCALE,
    CLIP_OBSERVATIONS,
    CLIP_ACTIONS,
    ACTION_SCALE,
    CYCLE_TIME,
    POLICY_DT,
    WARMUP_STEPS,
    RL_ACTION_SMOOTHING,
    SAFETY_MAX_ACTION_NORM,
    SAFETY_Q_DES_DELTA_MAX,
    PD_HOLD_INTERP_S,
    CONTROL_DT,
    ENABLE_IMU_COMPENSATION,
    SAFETY_ENABLE_FALL_DETECTION,
    SAFETY_FALL_ORIENTATION_LIMIT,
    SAFETY_COMMS_TIMEOUT,
    VEL_X_MAX, VEL_X_MIN,
    VEL_Y_MAX, VEL_Y_MIN,
    VEL_YAW_MAX, VEL_YAW_MIN,
)


# =====================================================================
# DDS 电机映射：20 路 DDS 信道 ↔ 10-DOF humanoid-gym 顺序
# =====================================================================
#
# 【背景】
# DDS 通信总是 20 路电机，对应实机 H1 的全部 20 个关节。
# humanoid-gym 训练的策略只输出 10 维（腿部）。
# 读取状态时需要从 20 个 DDS 信道挑出 10 个腿部关节，
# 发送命令时需要把 10 维 action 放回 20 路 DDS 信道的对应位置。
#
# MJC actuator 顺序（20 路 DDS 信道，legged_lab 的 h1.xml 深度优先排列）:
#   0:R_hip_roll   1:R_hip_pitch  2:R_knee
#   3:L_hip_roll   4:L_hip_pitch  5:L_knee
#   6:torso        7:L_hip_yaw    8:R_hip_yaw    9:not_use
#   10:L_ankle     11:R_ankle
#   12:R_shoulder_pitch  13:R_shoulder_roll  14:R_shoulder_yaw  15:R_elbow
#   16:L_shoulder_pitch  17:L_shoulder_roll  18:L_shoulder_yaw  19:L_elbow
#
# HG 10-DOF 训练顺序:
#   0:L_hip_yaw  1:L_hip_roll  2:L_hip_pitch  3:L_knee  4:L_ankle
#   5:R_hip_yaw  6:R_hip_roll  7:R_hip_pitch  8:R_knee  9:R_ankle
#
# 【映射推导】
# 对 HG 第 i 个关节，找到它在 MJC 中的索引：
#   HG[0]=L_hip_yaw   → MJC[7]  （因为 MJC[7] 是 L_hip_yaw）
#   HG[1]=L_hip_roll  → MJC[3]
#   HG[2]=L_hip_pitch → MJC[4]
#   HG[3]=L_knee      → MJC[5]
#   HG[4]=L_ankle     → MJC[10]
#   HG[5]=R_hip_yaw   → MJC[8]
#   HG[6]=R_hip_roll  → MJC[0]
#   HG[7]=R_hip_pitch → MJC[1]
#   HG[8]=R_knee      → MJC[2]
#   HG[9]=R_ankle     → MJC[11]
# 因此 MJC_TO_HG = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11]
# HG_TO_MJC 和 MJC_TO_HG 数值相同（对称映射），但语义不同：
#   HG_TO_MJC[hg_i] = mjc_i 表示 "HG 第 hg_i 个关节对应 MJC 第 mjc_i 个"
# =====================================================================

NUM_DDS_MOTORS = 20
DDS_NOT_USE_IDX = 9  # DDS 第 9 号未使用，始终置零

# 读取映射：DDS 信道 → HG 索引
# dds_pos[MJC_TO_HG[i]] 就是 HG 第 i 个关节的位置
MJC_TO_HG = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11]

# 写入映射：HG 索引 → DDS 信道
# dds_cmd[HG_TO_MJC[i]] = HG 第 i 个关节的值
HG_TO_MJC = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11]

# 上半身在 DDS 中的索引（10-DOF 策略不控制这些关节）
UPPER_BODY_DDS_IDXS = [6, 12, 13, 14, 15, 16, 17, 18, 19]


# =====================================================================
# PD Hold 配置（MJC 顺序，20 维）
# =====================================================================
# 【设计原则】
# 10-DOF 策略只能控制下肢 10 个关节。
# 躯干和手臂（共 9 个关节）必须始终用 PD Hold 保持在默认姿态，
# 否则手臂会因重力自然下垂、躯干会不受控地晃动。
#
# 对于 10-DOF 策略不控制的关节，PD 增益要足够保持姿态但又不能太硬：
#   - 躯干（torso, MJC[6]）：Kp=300, Kd=3（核心稳定最重要）
#   - 肩关节：Kp=100, Kd=2（防抖动）
#   - 肘关节/肩偏航：Kp=50, Kd=2（较轻，小增益足矣）
# =====================================================================

# 默认姿态（20 维，MJC 顺序）
PD_HOLD_TARGET_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
for hg_i in range(HG_NUM_ACTION):
    mjc_i = HG_TO_MJC[hg_i]
    PD_HOLD_TARGET_20[mjc_i] = HG_DEFAULT_DOF_POS[hg_i]
# 上半身保持 0 位（自然垂放）

# 启动阶段 PD Hold 增益（MJC 顺序，20 维）
# 【启动 vs RL 增益的区别】
# 启动时机器人可能不在默认姿态，用较温和的增益避免突然的力矩冲击；
# RL 模式接合后下肢切换为训练增益（更高刚度）
STARTUP_KP_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
STARTUP_KD_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)

# 下肢：比训练增益温和
for mjc_i in [0, 1, 3, 4, 7, 8]:  # 髋关节
    STARTUP_KP_20[mjc_i] = 150.0
    STARTUP_KD_20[mjc_i] = 2.0
for mjc_i in [2, 5]:  # 膝关节
    STARTUP_KP_20[mjc_i] = 200.0
    STARTUP_KD_20[mjc_i] = 4.0
for mjc_i in [10, 11]:  # 踝关节
    STARTUP_KP_20[mjc_i] = 40.0
    STARTUP_KD_20[mjc_i] = 2.0

# 躯干
STARTUP_KP_20[6] = 300.0
STARTUP_KD_20[6] = 3.0

# 肩关节
for mjc_i in [12, 13, 16, 17]:
    STARTUP_KP_20[mjc_i] = 100.0
    STARTUP_KD_20[mjc_i] = 2.0

# 肩偏航 / 肘关节
for mjc_i in [14, 15, 18, 19]:
    STARTUP_KP_20[mjc_i] = 50.0
    STARTUP_KD_20[mjc_i] = 2.0

# RL 模式增益（MJC 顺序）：下肢使用训练增益，上半身保持启动增益
RL_KP_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
RL_KD_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
for hg_i in range(HG_NUM_ACTION):
    mjc_i = HG_TO_MJC[hg_i]
    RL_KP_20[mjc_i] = HG_KPS[hg_i] * HG_KP_SCALE
    RL_KD_20[mjc_i] = HG_KDS[hg_i] * HG_KD_SCALE
# 上半身在 RL 模式中也用启动增益（策略不控制它们）
for mjc_i in UPPER_BODY_DDS_IDXS:
    RL_KP_20[mjc_i] = STARTUP_KP_20[mjc_i]
    RL_KD_20[mjc_i] = STARTUP_KD_20[mjc_i]
# 未使用关节始终零增益
RL_KP_20[DDS_NOT_USE_IDX] = 0.0
RL_KD_20[DDS_NOT_USE_IDX] = 0.0

# 软关节限位（防止策略输出超出物理范围）
Q_DES_MIN_20 = np.full(NUM_DDS_MOTORS, -3.0, dtype=np.float64)
Q_DES_MAX_20 = np.full(NUM_DDS_MOTORS, 3.0, dtype=np.float64)
for hg_i in range(HG_NUM_ACTION):
    mjc_i = HG_TO_MJC[hg_i]
    default = HG_DEFAULT_DOF_POS[hg_i]
    Q_DES_MIN_20[mjc_i] = default - 0.8   # 默认位置 ±0.8 rad
    Q_DES_MAX_20[mjc_i] = default + 0.8
Q_DES_MIN_20[DDS_NOT_USE_IDX] = 0.0
Q_DES_MAX_20[DDS_NOT_USE_IDX] = 0.0

# 安全阈值（已集中到 hg_config.py，此处不再重复定义）
# 见 hg_config.py: SAFETY_MAX_ACTION_NORM, SAFETY_Q_DES_DELTA_MAX,
#               PD_HOLD_INTERP_S, CONTROL_DT


# =====================================================================
# 原始游戏手柄读取器（可选，直接读取 /dev/input/js0）
# =====================================================================
# 当 DDS wireless_remote 信号不可用时，可作为备选手柄输入方案。

class GamepadReader:
    """通过 Linux joystick API 读取 /dev/input/js0。"""

    BUTTON_A = 0; BUTTON_B = 1; BUTTON_X = 2; BUTTON_Y = 3
    BUTTON_LB = 4; BUTTON_RB = 5; BUTTON_BACK = 6; BUTTON_START = 7
    AXIS_LX = 0; AXIS_LY = 1; AXIS_LT = 2; AXIS_RX = 3
    AXIS_RY = 4; AXIS_RT = 5; AXIS_DPAD_X = 6; AXIS_DPAD_Y = 7
    _EVENT_FORMAT = "IhBB"
    _EVENT_SIZE = 8

    def __init__(self, device="/dev/input/js0"):
        self.device = device
        self._lock = threading.Lock()
        self._buttons = {}
        self._axes = {}
        self._running = False
        self._thread = None
        self._fd = None
        try:
            self._fd = open(device, "rb", buffering=0)
            self._connected = True
        except (FileNotFoundError, PermissionError, OSError):
            self._connected = False

    def is_connected(self): return self._connected

    def start(self):
        if not self._connected or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._fd:
            try:
                self._fd.close()
            except OSError:
                pass

    def _read_loop(self):
        while self._running:
            try:
                data = self._fd.read(self._EVENT_SIZE)
                if not data:
                    break
                _, value, ev_type, number = struct.unpack(self._EVENT_FORMAT, data)
                with self._lock:
                    if ev_type == 0x01:
                        self._buttons[number] = value
                    elif ev_type == 0x02:
                        self._axes[number] = value
            except (OSError, BlockingIOError):
                time.sleep(0.001)
            except struct.error:
                break

    def get_button(self, btn_id):
        with self._lock:
            return self._buttons.get(btn_id, 0) != 0

    def get_axis(self, axis_id):
        with self._lock:
            return self._axes.get(axis_id, 0)

    def get_axis_normalized(self, axis_id, deadzone=0.15):
        raw = self.get_axis(axis_id)
        val = raw / 32767.0
        if abs(val) < deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return (abs(val) - deadzone) / (1.0 - deadzone) * sign


# =====================================================================
# DDS wireless_remote 解析器（SDK 手柄信号路径）
# =====================================================================
# Unitree SDK 的 LowState_ 消息中包含 wireless_remote 字节数组，
# 编码了手柄按键和摇杆状态。需要按协议解析出具体值。
# 当 DDS 信号可用时优先使用此路径，因为零延迟。
# =====================================================================

def parse_wireless_remote(wr):
    """解析 wireless_remote 字节数组为手柄状态字典。"""
    try:
        raw = bytes(int(wr[i]) for i in range(min(24, len(wr))))
    except (IndexError, TypeError, ValueError):
        return {'a': False, 'b': False, 'start': False, 'back': False,
                'lx': 0.0, 'ly': 0.0, 'rx': 0.0, 'ry': 0.0, 'has_data': False}

    btn2 = raw[2] if len(raw) > 2 else 0
    btn3 = raw[3] if len(raw) > 3 else 0
    a = bool(btn3 & 0x01)
    b = bool(btn3 & 0x02)
    start = bool(btn2 & 0x04)
    back = bool(btn2 & 0x08)

    try:
        lx = struct.unpack('f', raw[4:8])[0] if len(raw) >= 8 else 0.0
        rx = struct.unpack('f', raw[8:12])[0] if len(raw) >= 12 else 0.0
        ry = -struct.unpack('f', raw[12:16])[0] if len(raw) >= 16 else 0.0
        # ly 取反：Unitree 桥接端/实机固件在打包时将 ly 取反存入 wireless_remote，
        # 因此读取时需恢复原始符号：ly = -unpack(data)。
        # 最终在速度指令中 vx = -ly，前推(原始 ly 为负) → 正向速度。
        ly = -struct.unpack('f', raw[20:24])[0] if len(raw) >= 24 else 0.0
    except struct.error:
        lx = rx = ry = ly = 0.0

    has_data = any([a, b, start, back,
                    abs(lx) > 0.01, abs(ly) > 0.01,
                    abs(rx) > 0.01, abs(ry) > 0.01])
    return {'a': a, 'b': b, 'start': start, 'back': back,
            'lx': lx, 'ly': ly, 'rx': rx, 'ry': ry, 'has_data': has_data}


# =====================================================================
# HG10PolicyController
# =====================================================================
# DDS 策略控制器的主类。处理从 lowstate 解析、观测构建、策略推理、
# 到 lowcmd 打包的完整链路。
#
# 【启动三阶段】
# Phase 0: 等待第一条 lowstate 消息
# Phase 1: 填充 15 帧观测历史，同时发送 PD Hold（保持机器人稳定）
# Phase 2: 主循环，接收手柄/键盘指令，控制 RL 接合/断开
#
# 【安全机制】
# 1. NaN/Inf 检测：任一传感器数值异常时沿用上一帧有效值
# 2. 动作范数限制：动作向量超过阈值时丢弃本次推理结果
# 3. q_des 速率限制：每帧目标位置变化不超过 SAFETY_Q_DES_DELTA_MAX
# 4. 软关节限位：策略输出的目标位置被钳位到合理范围 ±0.8 rad
# 5. 跌落检测：RL 接合时监视 IMU 姿态，|roll/pitch| > 1.2 rad 自动断开
# 6. 通信断连检测：lowstate 超过 0.3s 未更新时自动断开 RL
#    跌落和通信断连的开关/阈值在 hg_config.py 的 SAFETY_* 参数
# =====================================================================

class HG10PolicyController:
    """基于 DDS 的 10-DOF humanoid-gym 策略控制器。"""

    def __init__(self, policy_path):
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"策略文件未找到: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()
        print(f"[HGController] 已加载策略: {policy_path}")

        # 运行时状态
        self.action = np.zeros(HG_NUM_ACTION, dtype=np.float64)          # 当前动作
        self.last_action = np.zeros(HG_NUM_ACTION, dtype=np.float64)     # 上一帧动作（用于 EMA）
        self.command_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # 速度指令
        self.rl_engaged = False      # RL 是否已接合
        self._episode_length = 0     # 策略步数（用于步态相位计算）

        # 15 帧观测历史 = 615 维
        self.obs_history = np.zeros(
            HG_NUM_OBS_PER_STEP * HG_ACTOR_OBS_HISTORY_LENGTH, dtype=np.float32
        )

        # 传感器数据（MJC 顺序，20 维）
        self._dof_pos_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
        self._dof_vel_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
        self._dof_tau_20 = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z（Unitree 格式）
        self._imu_gyro = np.zeros(3, dtype=np.float64)

        # PD Hold 插值状态
        self._pd_start_pos_20 = None   # 记录断开 RL 时的关节位置
        self._pd_start_time = None     # 记录断开 RL 的时间戳

        # 当前帧目标位置（MJC 格式 20 维）
        self._q_des_20 = PD_HOLD_TARGET_20.copy()
        self._q_des_prev_20 = PD_HOLD_TARGET_20.copy()

        # CRC
        self.crc = CRC()

        # 调试计数器
        self._nan_count = 0
        self._clip_count = 0
        self._debug_count = 0

        print(f"[HGController] 10-DOF 策略, obs={HG_NUM_OBS_PER_STEP}x"
              f"{HG_ACTOR_OBS_HISTORY_LENGTH}="
              f"{HG_NUM_OBS_PER_STEP * HG_ACTOR_OBS_HISTORY_LENGTH}")
        print(f"[HGController] PD Hold 插值时间: {PD_HOLD_INTERP_S}s")

    # ── 传感器数据 ───────────────────────────────────────────────

    def set_sensor_data(self, pos_20, vel_20, tau_20, imu_quat, imu_gyro):
        """存储并校验 20 路 DDS 传感器数据。

        【安全】
        - NaN/Inf 数据会被上一帧有效值替代
        - 四元数范数低于 0.1 时视为无效
        - 统计异常次数用于诊断

        返回: valid flag（False 表示该帧传感器数据存在问题）
        """
        valid = True

        # NaN/Inf 过滤
        for i in range(NUM_DDS_MOTORS):
            if np.isnan(pos_20[i]) or np.isinf(pos_20[i]):
                pos_20[i] = self._dof_pos_20[i]
                valid = False
            if np.isnan(vel_20[i]) or np.isinf(vel_20[i]):
                vel_20[i] = self._dof_vel_20[i]
                valid = False
            if np.isnan(tau_20[i]) or np.isinf(tau_20[i]):
                tau_20[i] = self._dof_tau_20[i]
                valid = False

        self._dof_pos_20 = pos_20.copy()
        self._dof_vel_20 = vel_20.copy()
        self._dof_tau_20 = tau_20.copy()

        # IMU 校验
        if not (np.isnan(imu_quat).any() or np.isinf(imu_quat).any()
                or np.linalg.norm(imu_quat) < 0.1):
            self._imu_quat = imu_quat.copy()
        else:
            valid = False
        if not (np.isnan(imu_gyro).any() or np.isinf(imu_gyro).any()):
            self._imu_gyro = imu_gyro.copy()
        else:
            valid = False

        if not valid:
            self._nan_count += 1
            if self._nan_count % 100 == 0:
                print(f"[HGController] 传感器数据异常 ({self._nan_count} 次)")

        return valid

    # ── 观测构建 ─────────────────────────────────────────────────

    @staticmethod
    def _quat_to_euler(w, x, y, z):
        """将 [w,x,y,z] 四元数转为 [roll, pitch, yaw]。

        注意：输入是 [w,x,y,z] 顺序（Unitree DDS 格式），
        与 sim2sim_bridge 中的 quaternion_to_euler_array 不同，
        那个版本输入是 [x,y,z,w]。
        """
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        euler = np.array([roll, pitch, yaw], dtype=np.float64)
        euler[euler > math.pi] -= 2.0 * math.pi
        return euler

    # ── IMU 补偿（Torso → Pelvis 坐标系） ──────────────────────────

    @staticmethod
    def _quat_wxyz_to_rotmat(quat_wxyz):
        """四元数 (w,x,y,z) → 旋转矩阵 (3x3)。"""
        w, x, y, z = quat_wxyz.astype(np.float64)
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-8:
            return np.eye(3, dtype=np.float64)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        return np.asarray([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _rotmat_to_quat_wxyz(rot):
        """旋转矩阵 (3x3) → 四元数 (w,x,y,z)。"""
        r00, r01, r02 = rot[0, 0], rot[0, 1], rot[0, 2]
        r10, r11, r12 = rot[1, 0], rot[1, 1], rot[1, 2]
        r20, r21, r22 = rot[2, 0], rot[2, 1], rot[2, 2]
        trace = r00 + r11 + r22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w, x, y, z = 0.25 * s, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s
        elif (r00 > r11) and (r00 > r22):
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            w, x, y, z = (r21 - r12) / s, 0.25 * s, (r01 + r10) / s, (r02 + r20) / s
        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            w, x, y, z = (r02 - r20) / s, (r01 + r10) / s, 0.25 * s, (r12 + r21) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
            w, x, y, z = (r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, 0.25 * s
        return np.array([w, x, y, z], dtype=np.float64)

    def _compensate_imu_with_waist(self, quat_wxyz, gyro_xyz, waist_yaw, waist_yaw_omega):
        """将 Torso 坐标系下的 IMU 补偿到 Pelvis 坐标系。

        实机 H1 的 IMU 物理安装在 Torso（腰部），在 waist_yaw 关节下游。
        如果 waist_yaw 不在零位，IMU 读到的姿态和 Pelvis 实际姿态有 yaw 偏差。

        参数:
            quat_wxyz: IMU 四元数 (w,x,y,z)，Torso 系
            gyro_xyz:  (3,) IMU 角速度，Torso 系
            waist_yaw: 腰部关节角度 (rad) — DDS 索引 6
            waist_yaw_omega: 腰部关节角速度 (rad/s)
        返回:
            comp_quat: Pelvis 系四元数 (w,x,y,z)
            comp_gyro: Pelvis 系角速度 (3,)
        """
        cos_yaw, sin_yaw = math.cos(waist_yaw), math.sin(waist_yaw)
        rz_waist = np.asarray([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # IMU 读数左乘 rz_waist^T 去除腰部偏航旋转
        r_pelvis = np.dot(self._quat_wxyz_to_rotmat(quat_wxyz), rz_waist.T)
        comp_quat = self._rotmat_to_quat_wxyz(r_pelvis)
        # 角速度：旋转到 Pelvis 系，扣除腰部角速度
        comp_gyro = np.dot(rz_waist, gyro_xyz.astype(np.float64)) \
                    - np.asarray([0.0, 0.0, waist_yaw_omega])
        return comp_quat, comp_gyro.astype(np.float64)

    def build_obs(self):
        """构建 41 维观测 + 15 帧历史 = 615 维策略输入。

        观测布局与 HG10PolicyBridge.get_obs() 完全一致。
        区别在于数据来源：
          - Bridge 从 MuJoCo 读
          - Controller 从 DDS lowstate 读
        """
        # 从 20 路 DDS 数据中提取 10 个下肢关节
        dof_pos_hg = self._dof_pos_20[MJC_TO_HG]
        dof_vel_hg = self._dof_vel_20[MJC_TO_HG]

        # ── IMU 补偿（可选） ─────────────────────────────────────
        # 实机 IMU 在 Torso（腰部关节下游），策略在 Pelvis 系训练。
        # 启用补偿后，通过 waist_yaw 角度将 IMU 读数逆变换到 Pelvis 系。
        if ENABLE_IMU_COMPENSATION:
            waist_yaw = self._dof_pos_20[6]          # DDS 索引 6 = waist
            waist_yaw_omega = self._dof_vel_20[6]
            comp_quat, comp_gyro = self._compensate_imu_with_waist(
                self._imu_quat, self._imu_gyro, waist_yaw, waist_yaw_omega
            )
            imu_quat_for_euler = comp_quat     # [w,x,y,z]
            gyro_for_obs = comp_gyro
        else:
            imu_quat_for_euler = self._imu_quat
            gyro_for_obs = self._imu_gyro

        # IMU → 欧拉角
        euler = self._quat_to_euler(
            imu_quat_for_euler[0], imu_quat_for_euler[1],
            imu_quat_for_euler[2], imu_quat_for_euler[3]
        )

        # 步态相位
        phase = self._episode_length * POLICY_DT / CYCLE_TIME

        # 构建 41 维观测
        obs = np.zeros(HG_NUM_OBS_PER_STEP, dtype=np.float32)
        obs[0] = math.sin(2.0 * math.pi * phase)
        obs[1] = math.cos(2.0 * math.pi * phase)
        obs[2] = self.command_vel[0] * LIN_VEL_SCALE
        obs[3] = self.command_vel[1] * LIN_VEL_SCALE
        obs[4] = self.command_vel[2] * ANG_VEL_SCALE
        obs[5:15] = (dof_pos_hg - HG_DEFAULT_DOF_POS) * DOF_POS_SCALE
        obs[15:25] = dof_vel_hg * DOF_VEL_SCALE
        obs[25:35] = np.clip(self.action, -CLIP_ACTIONS, CLIP_ACTIONS)
        obs[35:38] = gyro_for_obs * ANG_VEL_SCALE
        obs[38:41] = euler * QUAT_SCALE
        obs = np.clip(obs, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)

        # 更新历史
        self.obs_history = np.roll(self.obs_history, shift=-HG_NUM_OBS_PER_STEP)
        self.obs_history[-HG_NUM_OBS_PER_STEP:] = obs

        return np.clip(self.obs_history, -CLIP_OBSERVATIONS, CLIP_OBSERVATIONS)

    # ── 策略推理 ─────────────────────────────────────────────────

    def policy_step(self):
        """执行一次策略推理，更新 _q_des_20（目标位置，20 维 MJC 顺序）。

        流程：
        1. 构建观测 → 策略推理 → 10 维 action
        2. 安全检测（动作范数、NaN）
        3. 预热缩放
        4. 动作平滑（EMA，可选）
        5. action → q_des 转换（10-DOF → 20-DDS 映射）
        6. 速率限制 + 软关节限位

        返回: action_norm（用于监控）
        """
        obs = self.build_obs()

        with torch.no_grad():
            raw_action = self.policy(
                torch.tensor(obs, dtype=torch.float32)
            ).detach().numpy()[:HG_NUM_ACTION]

        # 安全检测：动作异常时沿用上一帧
        action_norm = np.linalg.norm(raw_action)
        if action_norm > SAFETY_MAX_ACTION_NORM or np.isnan(raw_action).any():
            print(f"[HGController] 异常动作! norm={action_norm:.1f}, 使用上一帧")
            raw_action = self.action.copy()

        raw_action = np.clip(raw_action, -CLIP_ACTIONS, CLIP_ACTIONS)

        # 预热
        if WARMUP_STEPS > 0 and self._episode_length < WARMUP_STEPS:
            alpha = float(self._episode_length) / float(WARMUP_STEPS)
            raw_action *= alpha

        self.action = raw_action.copy()

        # 动作平滑（一阶低通滤波/EMA）
        # 使用 hg_config.RL_ACTION_SMOOTHING：
        #   sim2sim 设为 0.0（不过滤）
        #   实机推荐 0.1~0.2 减少电机高频抖动
        _alpha = RL_ACTION_SMOOTHING
        if _alpha > 0.0:
            smooth_action = _alpha * raw_action + (1.0 - _alpha) * self.last_action
        else:
            smooth_action = raw_action
        self.last_action = smooth_action.copy()

        # 计算目标位置：10-DOF → 20-DDS 映射
        q_des_hg = smooth_action * ACTION_SCALE + HG_DEFAULT_DOF_POS

        q_des_20 = PD_HOLD_TARGET_20.copy()  # 上半身保持默认姿态
        for hg_i in range(HG_NUM_ACTION):
            mjc_i = HG_TO_MJC[hg_i]
            q_des_20[mjc_i] = q_des_hg[hg_i]
        q_des_20[DDS_NOT_USE_IDX] = 0.0

        # 速率限制：每帧变化不超过 SAFETY_Q_DES_DELTA_MAX
        # 【为什么需要】策略输出的目标位置可能大幅跳变，
        # 导致实机电机突然大角度转动，产生危险力矩
        delta = q_des_20 - self._q_des_prev_20
        for i in range(NUM_DDS_MOTORS):
            if i == DDS_NOT_USE_IDX:
                continue
            if abs(delta[i]) > SAFETY_Q_DES_DELTA_MAX:
                q_des_20[i] = self._q_des_prev_20[i] + np.sign(delta[i]) * SAFETY_Q_DES_DELTA_MAX
                self._clip_count += 1

        # 软限位
        q_des_20 = np.clip(q_des_20, Q_DES_MIN_20, Q_DES_MAX_20)
        q_des_20[DDS_NOT_USE_IDX] = 0.0

        self._q_des_20 = q_des_20.copy()
        self._q_des_prev_20 = q_des_20.copy()
        self._episode_length += 1

        return action_norm

    # ── LowCmd 消息构建 ─────────────────────────────────────────

    def get_lowcmd_msg(self):
        """构建 LowCmd_ 消息，包含正确的头部、模式和 CRC。

        【关键参数】
        - 电机模式：0x0A = HG 系列（下肢），0x01 = GO 系列（上肢）
          H1 实机下肢电机是 Unitree HG 系列，上肢是 GO 系列
        - mode=0x0A 表示位置+速度+力矩控制（接收 q/kp/kd/dq/tau）
        - mode=0x01 表示纯位置控制

        【PD Hold 插值】
        RL 断开后 2 秒内从当前位姿平滑过渡到默认姿态。
        过渡期间使用启动增益（STARUP_KP/KD），到默认姿态后保持不变。
        """
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(NUM_DDS_MOTORS):
            m = cmd.motor_cmd[i]
            # 电机模式：下肢 HG 系列 (0x0A)，上肢 GO 系列 (0x01)
            m.mode = 0x0A if i < 10 else 0x01

            if self.rl_engaged:
                # RL 模式：使用策略计算的目标位置
                m.q = float(self._q_des_20[i])
                m.kp = float(RL_KP_20[i])
                m.kd = float(RL_KD_20[i])
            elif self._pd_start_pos_20 is not None and self._pd_start_time is not None:
                # PD Hold 插值过渡
                elapsed = time.perf_counter() - self._pd_start_time
                alpha = min(1.0, elapsed / PD_HOLD_INTERP_S)
                m.q = float(
                    (1.0 - alpha) * self._pd_start_pos_20[i]
                    + alpha * PD_HOLD_TARGET_20[i]
                )
                m.kp = float(STARTUP_KP_20[i])
                m.kd = float(STARTUP_KD_20[i])
            else:
                # 纯 PD Hold
                m.q = float(PD_HOLD_TARGET_20[i])
                m.kp = float(STARTUP_KP_20[i])
                m.kd = float(STARTUP_KD_20[i])

            if hasattr(m, "qd"):
                m.qd = 0.0
            else:
                m.dq = 0.0
            m.tau = 0.0

        # Unitree SDK CRC32 校验
        cmd.crc = self.crc.Crc(cmd)
        return cmd

    # ── 模式切换 ────────────────────────────────────────────────

    def engage(self):
        """接合 RL 策略。"""
        if self.rl_engaged:
            return
        self.reset()
        self.rl_engaged = True
        print("=" * 60)
        print("[HGController] RL 策略已接合！")
        print("=" * 60)

    def disengage(self):
        """断开 RL，回到 PD Hold（带平滑过渡）。"""
        if not self.rl_engaged:
            self.command_vel[:] = 0.0
            return
        self.rl_engaged = False
        self.command_vel[:] = 0.0
        self._episode_length = 0
        self.obs_history[:] = 0.0
        self.action[:] = 0.0
        self.last_action[:] = 0.0
        # 记录断开时的关节位置 → 用于平滑过渡
        self._pd_start_pos_20 = self._dof_pos_20.copy()
        self._pd_start_time = time.perf_counter()
        self._debug_count = 0
        # 重建观测历史
        for _ in range(HG_ACTOR_OBS_HISTORY_LENGTH):
            self.build_obs()
        print("=" * 60)
        print("[HGController] RL 已断开 → PD Hold")
        print("=" * 60)

    def reset(self):
        """重置 episode 状态。"""
        self.action[:] = 0.0
        self.last_action[:] = 0.0
        self._episode_length = 0
        self.command_vel[:] = 0.0
        self.obs_history[:] = 0.0
        self._debug_count = 0
        # 用当前观测填满历史
        for _ in range(HG_ACTOR_OBS_HISTORY_LENGTH):
            self.build_obs()
        print("[HGController] Episode 已重置。")


# =====================================================================
# 键盘监听
# =====================================================================

def start_keyboard_listener(controller):
    from pynput import keyboard

    def on_press(key):
        try:
            if key.char == "1":
                if not controller.rl_engaged:
                    controller.engage()
            elif key.char == "0":
                if controller.rl_engaged:
                    controller.disengage()
                else:
                    controller.command_vel[:] = 0.0
                    print("[CMD] 速度已重置")
            elif key.char == "4":
                controller.command_vel[0] = min(controller.command_vel[0] + 0.2, VEL_X_MAX)
                print(f"[CMD] vel_x = {controller.command_vel[0]:.1f}")
            elif key.char == "5":
                controller.command_vel[0] = max(controller.command_vel[0] - 0.2, VEL_X_MIN)
                print(f"[CMD] vel_x = {controller.command_vel[0]:.1f}")
            elif key.char == "6":
                controller.command_vel[1] = max(controller.command_vel[1] - 0.2, VEL_Y_MIN)
                print(f"[CMD] vel_y = {controller.command_vel[1]:.1f}")
            elif key.char == "7":
                controller.command_vel[1] = min(controller.command_vel[1] + 0.2, VEL_Y_MAX)
                print(f"[CMD] vel_y = {controller.command_vel[1]:.1f}")
            elif key.char == "8":
                controller.command_vel[2] = max(controller.command_vel[2] - 0.2, VEL_YAW_MIN)
                print(f"[CMD] yaw = {controller.command_vel[2]:.1f}")
            elif key.char == "9":
                controller.command_vel[2] = min(controller.command_vel[2] + 0.2, VEL_YAW_MAX)
                print(f"[CMD] yaw = {controller.command_vel[2]:.1f}")
            elif key.char == "d":
                mode_str = "RL" if controller.rl_engaged else "PD Hold"
                print(f"\n[PD Debug] 模式: {mode_str}")
                print(f"{'i':>3s} {'mode':>6s} {'q_des':>7s} {'q_act':>7s} {'Kp':>7s} {'Kd':>8s}")
                print("-" * 50)
                for i in range(NUM_DDS_MOTORS):
                    if i == DDS_NOT_USE_IDX:
                        continue
                    q_des = controller._q_des_20[i]
                    q_act = controller._dof_pos_20[i]
                    if controller.rl_engaged:
                        _kp, _kd = RL_KP_20[i], RL_KD_20[i]
                    else:
                        _kp, _kd = STARTUP_KP_20[i], STARTUP_KD_20[i]
                    tag = "lower" if i < 10 else "upper"
                    print(f"{i:3d} {tag:>6s} {q_des:7.3f} {q_act:7.3f} {_kp:7.1f} {_kd:8.2f}")
                print()
            elif key.char == "r":
                controller.reset()
            elif key.char == "q":
                print("[CMD] 退出请求。")
                global running
                running = False
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


# =====================================================================
# 主程序
# =====================================================================

running = True

def signal_handler(signum, frame):
    global running
    print("\n[HGController] Ctrl+C, 正在关闭...")
    running = False

signal.signal(signal.SIGINT, signal_handler)


def main():
    global running

    parser = argparse.ArgumentParser(description="H1 10-DOF 策略控制器（DDS sim2real）")
    parser.add_argument("--load_model", type=str, default=POLICY_PATH,
                        help="导出的 JIT 策略路径（默认取自 hg_config.py）")
    parser.add_argument("--domain", type=int, default=DOMAIN_ID,
                        help=f"DDS 域 ID (hg_config: {DOMAIN_ID}, 实机=0)")
    parser.add_argument("--interface", type=str, default=INTERFACE,
                        help=f"DDS 网卡 (hg_config: {INTERFACE}, 实机=enp2s0)")
    parser.add_argument("--no-keyboard", action="store_true", help="禁用键盘控制")
    parser.add_argument("--no-gamepad", action="store_true", help="禁用手柄控制")
    args = parser.parse_args()

    # DDS 初始化
    print(f"[HGController] DDS: domain={args.domain}, interface={args.interface}")
    ChannelFactoryInitialize(args.domain, args.interface)

    # 创建控制器
    controller = HG10PolicyController(args.load_model)

    # DDS 数据共享
    latest_data = None
    latest_data_id = 0
    latest_gp_data = None
    latest_gp_id = 0
    data_lock = threading.Lock()
    lowstate_received = threading.Event()

    def on_lowstate(msg):
        nonlocal latest_data, latest_data_id, latest_gp_data, latest_gp_id
        pos = np.array([msg.motor_state[i].q for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
        vel = np.array([msg.motor_state[i].dq for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
        tau = np.array([msg.motor_state[i].tau_est for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
        imu_q = np.array([msg.imu_state.quaternion[i] for i in range(4)], dtype=np.float64)
        imu_g = np.array([msg.imu_state.gyroscope[i] for i in range(3)], dtype=np.float64)
        gp = parse_wireless_remote(msg.wireless_remote)
        with data_lock:
            latest_data = (pos, vel, tau, imu_q, imu_g)
            latest_data_id += 1
            latest_gp_data = gp
            latest_gp_id += 1
        lowstate_received.set()

    lowstate_suber = ChannelSubscriber("rt/lowstate", LowState_)
    lowstate_suber.Init(on_lowstate, 10)

    lowcmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
    lowcmd_puber.Init()

    # 键盘
    kb_listener = None
    if not args.no_keyboard:
        kb_listener = start_keyboard_listener(controller)
        print("[HGController] 键盘: 1=接合, 0=断开, "
              "4/5=vx, 6/7=vy, 8/9=yaw, r=重置, q=退出")
    else:
        print("[HGController] 键盘已禁用。")

    # 游戏手柄
    gamepad = None
    if not args.no_gamepad:
        gamepad = GamepadReader()
        if gamepad.is_connected():
            gamepad.start()
            print("[HGController] 直接手柄已连接。")

    # ── Phase 0: 等待第一条 lowstate ────────────────────────────
    # 在收到 lowstate 之前，低 cmd 未发布（电机保持之前的状态）
    print("[HGController] 等待第一条 lowstate...")
    print(f"[HGController]    domain={args.domain}, interface={args.interface}")
    print(f"[HGController]    需确保 unitree_mujoco.py 或实机 H1 正在发布 rt/lowstate")
    _phase0_start = time.perf_counter()
    _phase0_timeout = 10.0  # 10 秒超时，避免无限等待
    while not lowstate_received.is_set() and running:
        time.sleep(0.1)
        if time.perf_counter() - _phase0_start > _phase0_timeout:
            print("[HGController] ⚠ 未收到 lowstate（超时）")
            print(f"    domain={args.domain} interface={args.interface}")
            print("    建议: 仿真联调用 --interface=lo，实机确认网络和 --domain=0")
            running = False
            break
    if not running:
        return
    print("[HGController] 已收到第一条 lowstate。")

    # 记录初始姿态 → PD Hold 插值起点
    with data_lock:
        init_data = latest_data
    if init_data is not None:
        controller._pd_start_pos_20 = init_data[0].copy()
        controller._pd_start_time = time.perf_counter()
        print(f"[HGController] PD Hold 插值: "
              f"当前位置 → 默认姿态（{PD_HOLD_INTERP_S}秒渐变）")

    # 通信断连追踪
    _last_lowstate_time = time.perf_counter()
    _fall_counter = 0

    # ── Phase 1: 填充观测历史 + 发送 PD Hold ────────────────────
    # 发送 15 帧 PD Hold 指令让机器人稳定，同时积累 15 帧观测历史。
    # 这样首次接合 RL 时策略能立刻得到完整的历史观测。
    print("[HGController] 填充观测历史 + 发送 PD Hold...")
    for _ in range(HG_ACTOR_OBS_HISTORY_LENGTH):
        with data_lock:
            data = latest_data
        if data is not None:
            controller.set_sensor_data(*data)
            controller.build_obs()
        lowcmd_puber.Write(controller.get_lowcmd_msg())
        time.sleep(CONTROL_DT)

    # ── Phase 2: 主控制循环 ─────────────────────────────────────
    print("=" * 60)
    print("[HGController] PD HOLD 模式")
    print("[HGController] 按 '1' 或手柄 A 接合 RL")
    print("=" * 60)

    # 手柄速度范围取自 hg_config.py 的 VEL_X/Y/YAW_MIN/MAX
    # 注意：手柄摇杆是对称的，前向用 VEL_X_MAX，后向用 VEL_X_MIN 做非对称钳位

    next_loop = time.perf_counter()
    step_count = 0
    last_processed_id = -1
    policy_calls = 0

    # 手柄上升沿检测
    _gp_a_prev_dds = False
    _gp_b_prev_dds = False
    _gp_a_prev_dir = False
    _gp_b_prev_dir = False
    # DDS 手柄激活追踪：当摇杆回中时 has_data→False，但 command_vel
    # 需要通过这个标志清零残留速度，否则机器人持续以最后一帧速度行走。
    _dds_gp_was_active = False

    stat_loops = 0
    stat_lowstate = 0
    stat_last_print = time.perf_counter()

    while running:
        # 读取最新 DDS 数据
        with data_lock:
            data = latest_data
            data_id = latest_data_id
            gp_data = latest_gp_data

        new_data = data is not None and data_id != last_processed_id
        if new_data:
            last_processed_id = data_id
            pos, vel, tau, imu_q, imu_g = data
            controller.set_sensor_data(pos, vel, tau, imu_q, imu_g)

            if controller.rl_engaged:
                action_norm = controller.policy_step()
                policy_calls += 1
                if controller._episode_length % 100 == 0:
                    print(f"[HGController] step={controller._episode_length}, "
                          f"cmd={controller.command_vel}, action_norm={action_norm:.2f}")

            step_count += 1
            stat_lowstate += 1
            _last_lowstate_time = time.perf_counter()

        # ── 安全保护（RL 接合时） ─────────────────────────────────
        if controller.rl_engaged:
            # 通信断连检测
            if SAFETY_COMMS_TIMEOUT > 0 and time.perf_counter() - _last_lowstate_time > SAFETY_COMMS_TIMEOUT:
                print(f"[SAFETY] 通信丢失 {time.perf_counter() - _last_lowstate_time:.1f}s → 断开RL")
                controller.disengage()

            # 跌落检测（通过 IMU 姿态判断）
            if SAFETY_ENABLE_FALL_DETECTION:
                euler_fd = controller._quat_to_euler(
                    controller._imu_quat[0], controller._imu_quat[1],
                    controller._imu_quat[2], controller._imu_quat[3]
                )
                if abs(euler_fd[0]) > SAFETY_FALL_ORIENTATION_LIMIT or abs(euler_fd[1]) > SAFETY_FALL_ORIENTATION_LIMIT:
                    _fall_counter += 1
                    if _fall_counter >= 3:
                        print(f"[SAFETY] 摔倒! roll={euler_fd[0]:.2f} pitch={euler_fd[1]:.2f} → 断开RL")
                        controller.disengage()
                else:
                    _fall_counter = 0

        # ── 手柄处理 ─────────────────────────────────────────────
        # 优先使用 DDS wireless_remote（零延迟），其次直接 /dev/input/js0
        use_dds_gp = gp_data is not None and gp_data.get('has_data', False)
        direct_ok = gamepad is not None and gamepad.is_connected()

        # DDS 手柄路径
        if use_dds_gp:
            dds_a, dds_b = gp_data['a'], gp_data['b']
            dds_start, dds_back = gp_data['start'], gp_data['back']

            if dds_start and dds_back:
                print("[HGController] 手柄 Start+Select → 退出")
                running = False
            if dds_a and not _gp_a_prev_dds and not controller.rl_engaged:
                controller.engage()
            if dds_b and not _gp_b_prev_dds:
                if controller.rl_engaged:
                    controller.disengage()
                else:
                    controller.command_vel[:] = 0.0
            if controller.rl_engaged:
                controller.command_vel[0] = np.clip(-gp_data['ly'] * VEL_X_MAX, VEL_X_MIN, VEL_X_MAX)
                # vy/yaw 取反：Isaac Gym +y 向左（正 vy = 左走），游戏手柄推右期望右走
                controller.command_vel[1] = np.clip(-gp_data['lx'] * VEL_Y_MAX, VEL_Y_MIN, VEL_Y_MAX)
                controller.command_vel[2] = np.clip(-gp_data['rx'] * VEL_YAW_MAX, VEL_YAW_MIN, VEL_YAW_MAX)
            _gp_a_prev_dds, _gp_b_prev_dds = dds_a, dds_b
            _dds_gp_was_active = True
        else:
            # DDS 手柄信号消失（摇杆回中或遥控器断开）
            # 无直连手柄接替时 → 命令速度归零，防止残留速度
            if _dds_gp_was_active and controller.rl_engaged and not direct_ok:
                controller.command_vel[:] = 0.0
            _gp_a_prev_dds = _gp_b_prev_dds = False
            _dds_gp_was_active = False

        # 直接手柄路径
        if direct_ok:
            dir_a = gamepad.get_button(GamepadReader.BUTTON_A)
            dir_b = gamepad.get_button(GamepadReader.BUTTON_B)
            if dir_a and not _gp_a_prev_dir and not controller.rl_engaged:
                controller.engage()
            if dir_b and not _gp_b_prev_dir:
                if controller.rl_engaged:
                    controller.disengage()
                else:
                    controller.command_vel[:] = 0.0
            if controller.rl_engaged and not use_dds_gp:
                ly = gamepad.get_axis_normalized(GamepadReader.AXIS_LY)
                lx = gamepad.get_axis_normalized(GamepadReader.AXIS_LX)
                rx = gamepad.get_axis_normalized(GamepadReader.AXIS_RX)
                controller.command_vel[0] = np.clip(-ly * VEL_X_MAX, VEL_X_MIN, VEL_X_MAX)
                # vy/yaw 取反：Isaac Gym +y 向左，游戏手柄推右期望右走
                controller.command_vel[1] = np.clip(-lx * VEL_Y_MAX, VEL_Y_MIN, VEL_Y_MAX)
                controller.command_vel[2] = np.clip(-rx * VEL_YAW_MAX, VEL_YAW_MIN, VEL_YAW_MAX)
            _gp_a_prev_dir, _gp_b_prev_dir = dir_a, dir_b
        else:
            _gp_a_prev_dir = _gp_b_prev_dir = False

        # 发布 lowcmd
        lowcmd_puber.Write(controller.get_lowcmd_msg())

        # PD Hold 定期提醒
        if not controller.rl_engaged and step_count % 400 == 0:
            print("[HGController] PD Hold: 等待接合... (1/A)")

        stat_loops += 1

        # 周期统计
        now = time.perf_counter()
        if now - stat_last_print >= 3.0:
            rate = stat_lowstate / (now - stat_last_print)
            status = "RL" if controller.rl_engaged else "PD Hold"
            if stat_lowstate > 0:
                print(f"[HGController] [{status}] lowstate={rate:.0f}Hz "
                      f"steps={step_count} policy_calls={policy_calls}")
            stat_lowstate = 0
            stat_loops = 0
            stat_last_print = now

        # 速率控制（100Hz）
        next_loop += CONTROL_DT
        sleep_time = next_loop - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_loop = time.perf_counter() + CONTROL_DT

    # 清理
    print("[HGController] 发送最终默认姿态...")
    if controller.rl_engaged:
        controller.disengage()
    for _ in range(10):
        lowcmd_puber.Write(controller.get_lowcmd_msg())
        time.sleep(0.005)

    if kb_listener is not None:
        kb_listener.stop()
    if gamepad is not None:
        gamepad.stop()

    print(f"[HGController] 已关闭。steps={step_count}, policy_calls={policy_calls}, "
          f"NaN={controller._nan_count}, clips={controller._clip_count}")


if __name__ == "__main__":
    main()
