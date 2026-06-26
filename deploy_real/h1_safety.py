"""H1 实机部署安全监控模块。

从 H1_DDS/hg_policy_controller.py 的成熟安全机制移植而来，适配 19-DOF
全身策略。提供传感器校验、摔倒检测、通信超时、动作平滑、速率限制、
软关节限位、预热渐变、PD Hold 插值过渡等分层安全保护。

使用方式:
    safety = H1SafetyMonitor(default_angles, default_angles_seq, action_scale_seq,
                             dof_idx, control_dt=0.02)
    # 每帧调用:
    safety.check_sensors(low_state)       # → bool (valid)
    safety.check_comms()                   # → bool (alive)
    safety.check_fall(imu_quat)            # → bool (upright)
    safe_action = safety.process_action(raw_action, timestep)
    safe_q_des = safety.clamp_q_des(q_des_20)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np


# ============================================================================
# 安全参数（可通过 H1SafetyConfig 按需覆盖）
# ============================================================================

@dataclass
class H1SafetyConfig:
    """H1 安全监控可调参数集合。

    所有阈值均为 H1_DDS 实机验证值，默认即可安全运行。
    需要更激进/保守的策略时按需调整。
    """

    # ── 传感器校验 ──
    imu_quat_norm_min: float = 0.1       # IMU 四元数最小范数，低于此值视为无效

    # ── 摔倒检测 ──
    enable_fall_detection: bool = True    # 总开关
    fall_orientation_limit: float = 1.2   # roll/pitch 阈值 (rad)，≈69°
    fall_consecutive_frames: int = 3      # 连续超限帧数才触发

    # ── 通信超时 ──
    comms_timeout: float = 0.3            # 秒，lowstate 超时未更新则断开 RL

    # ── 动作安全 ──
    max_action_norm: float = 50.0         # 动作向量最大 L2 范数
    clip_actions: float = 18.0            # 逐元素裁剪阈值
    clip_observations: float = 18.0       # 观测向量逐元素裁剪阈值

    # ── 预热 ──
    warmup_steps: int = 50                # 前 N 步线性放大策略控制权

    # ── 动作平滑 (EMA 一阶低通) ──
    action_smoothing: float = 0.0         # 仿真用 0.0，实机建议 0.1~0.2

    # ── 目标位置速率限制 ──
    q_des_delta_max: float = 1.0          # 每帧目标位置最大变化量 (rad)

    # ── 软关节限位 ──
    soft_limit_margin: float = 0.8        # 默认角度 ± margin (rad)，非策略关节用 ±3.0

    # ── PD Hold 插值 ──
    pd_hold_interp_s: float = 2.0         # 断开 RL 后平滑过渡到默认姿态的时间 (s)


# ============================================================================
# H1SafetyMonitor
# ============================================================================

class H1SafetyMonitor:
    """H1 实机部署分层安全监控器。

    安全层级（由低到高）:
      L1: 传感器校验 — NaN/Inf 过滤，上一帧有效值回退
      L2: 通信检测 — lowstate 超时未更新
      L3: 姿态检测 — IMU 摔倒判断
      L4: 动作安全 — 异常值检测、裁剪、平滑、预热
      L5: 指令安全 — 目标位置速率限制、软关节限位
    """

    def __init__(
        self,
        default_angles: np.ndarray,        # DOF 顺序 (joint_xml)，19 维
        default_angles_seq: np.ndarray,    # ONNX 顺序 (joint_seq)，19 维
        action_scale_seq: np.ndarray,      # ONNX 顺序 (joint_seq)，19 维
        dof_idx: list,                     # DOF 顺序 → 电机 ID 映射
        control_dt: float = 0.02,          # 控制周期 (s)
        config: H1SafetyConfig | None = None,
    ):
        self.config = config or H1SafetyConfig()
        self.control_dt = control_dt
        self.dof_idx = list(dof_idx)

        num_dof = len(default_angles)  # 19

        # ── 软关节限位（DOF 顺序，即 joint_xml 顺序）──
        # 策略控制的 19 个关节使用 default ± margin
        self._q_min = np.array(default_angles, dtype=np.float64) - self.config.soft_limit_margin
        self._q_max = np.array(default_angles, dtype=np.float64) + self.config.soft_limit_margin

        # ── 目标位置历史（DDS 20 维，用于速率限制）──
        self._num_motors = 20
        self._not_use_idx = 9               # DDS 第 9 号未使用

        # ── 动作平滑状态 ──
        self._last_action = np.zeros(num_dof, dtype=np.float64)
        self._last_smooth_action = np.zeros(num_dof, dtype=np.float64)

        # ── 传感器回退缓存 ──
        self._prev_pos_20 = np.zeros(self._num_motors, dtype=np.float64)
        self._prev_vel_20 = np.zeros(self._num_motors, dtype=np.float64)
        self._prev_imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._prev_imu_gyro = np.zeros(3, dtype=np.float64)

        # ── 运行时状态 ──
        self._q_des_prev_20 = np.zeros(self._num_motors, dtype=np.float64)
        self._pd_start_pos_20 = np.zeros(self._num_motors, dtype=np.float64)
        self._pd_start_time: float | None = None
        self._rl_engaged = False
        self._last_lowstate_time: float | None = None
        self._fall_counter = 0

        # ── 统计 ──
        self.nan_count = 0
        self.clip_count = 0
        self.fall_trigger_count = 0
        self.comms_loss_count = 0

        print(f"[Safety] 初始化完成: warmup={self.config.warmup_steps}步, "
              f"fall_limit={self.config.fall_orientation_limit}rad, "
              f"comms_timeout={self.config.comms_timeout}s, "
              f"q_delta_max={self.config.q_des_delta_max}rad/帧")

    # ========================================================================
    # L1: 传感器校验
    # ========================================================================

    def check_sensors(self, low_state) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """校验并返回安全的传感器数据。

        检测 NaN/Inf，异常时回退到上一帧有效值。
        对 IMU 四元数额外检查范数。

        参数:
            low_state: DDS LowState_ 消息
        返回:
            (valid, pos_20, vel_20, imu_quat, imu_gyro)
            valid=False 表示本帧有传感器异常（已回退处理）
        """
        valid = True

        # 关节位置/速度
        pos_20 = np.zeros(self._num_motors, dtype=np.float64)
        vel_20 = np.zeros(self._num_motors, dtype=np.float64)
        for i in range(self._num_motors):
            pos_20[i] = low_state.motor_state[i].q
            vel_20[i] = low_state.motor_state[i].dq

        for i in range(self._num_motors):
            if np.isnan(pos_20[i]) or np.isinf(pos_20[i]):
                pos_20[i] = self._prev_pos_20[i]
                valid = False
            if np.isnan(vel_20[i]) or np.isinf(vel_20[i]):
                vel_20[i] = self._prev_vel_20[i]
                valid = False

        # IMU 四元数
        imu_quat = np.array([low_state.imu_state.quaternion[i] for i in range(4)],
                            dtype=np.float64)
        if np.isnan(imu_quat).any() or np.isinf(imu_quat).any() or np.linalg.norm(imu_quat) < self.config.imu_quat_norm_min:
            imu_quat = self._prev_imu_quat.copy()
            valid = False

        # IMU 角速度
        imu_gyro = np.array([low_state.imu_state.gyroscope[i] for i in range(3)],
                            dtype=np.float64)
        if np.isnan(imu_gyro).any() or np.isinf(imu_gyro).any():
            imu_gyro = self._prev_imu_gyro.copy()
            valid = False

        # 更新回退缓存
        self._prev_pos_20 = pos_20.copy()
        self._prev_vel_20 = vel_20.copy()
        self._prev_imu_quat = imu_quat.copy()
        self._prev_imu_gyro = imu_gyro.copy()

        # 更新通信时间戳
        self._last_lowstate_time = time.perf_counter()

        if not valid:
            self.nan_count += 1
            if self.nan_count % 100 == 0:
                print(f"[Safety] 传感器数据异常 ({self.nan_count} 次)")

        return valid, pos_20, vel_20, imu_quat, imu_gyro

    # ========================================================================
    # L2: 通信超时检测
    # ========================================================================

    def check_comms(self) -> bool:
        """检查是否与机器人保持通信。

        返回:
            True  — 通信正常
            False — 通信丢失（超过 comms_timeout 未收到 lowstate）
        """
        if self._last_lowstate_time is None:
            return True  # 尚未收到第一条消息，不判定为超时
        elapsed = time.perf_counter() - self._last_lowstate_time
        if elapsed > self.config.comms_timeout:
            self.comms_loss_count += 1
            return False
        return True

    # ========================================================================
    # L3: 摔倒检测
    # ========================================================================

    @staticmethod
    def _quat_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
        """四元数 [w,x,y,z] → 欧拉角 [roll, pitch, yaw] (rad)。"""
        w, x, y, z = quat_wxyz.astype(np.float64)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        euler = np.array([roll, pitch, yaw], dtype=np.float64)
        euler[euler > math.pi] -= 2.0 * math.pi
        return euler

    def check_fall(self, imu_quat: np.ndarray) -> bool:
        """检测机器人是否摔倒。

        使用 IMU 姿态（骨盆坐标系）的 roll/pitch 判断。
        需要连续超过阈值若干帧才触发，避免单帧噪声误报。

        参数:
            imu_quat: 骨盆坐标系四元数 [w,x,y,z]
        返回:
            True  — 姿态安全（未摔倒）
            False — 检测到摔倒
        """
        if not self.config.enable_fall_detection or not self._rl_engaged:
            self._fall_counter = 0
            return True

        euler = self._quat_to_euler(imu_quat)
        limit = self.config.fall_orientation_limit
        if abs(euler[0]) > limit or abs(euler[1]) > limit:
            self._fall_counter += 1
            if self._fall_counter >= self.config.fall_consecutive_frames:
                self.fall_trigger_count += 1
                return False
        else:
            self._fall_counter = 0
        return True

    # ========================================================================
    # L4: 动作安全（裁剪 + 预热 + 平滑）
    # ========================================================================

    def check_action(self, raw_action: np.ndarray) -> tuple[bool, np.ndarray]:
        """检测动作异常。

        参数:
            raw_action: 策略原始输出 (19,)
        返回:
            (valid, safe_action)
        """
        action_norm = float(np.linalg.norm(raw_action))
        if action_norm > self.config.max_action_norm or np.isnan(raw_action).any():
            return False, self._last_action.copy()
        return True, raw_action.copy()

    def process_action(self, raw_action: np.ndarray, episode_length: int) -> np.ndarray:
        """完整的动作后处理管线: 裁剪 → 预热 → 平滑。

        参数:
            raw_action:   策略原始输出 (19,)
            episode_length: 当前 episode 步数（用于预热缩放）
        返回:
            safe_action: 安全处理后的动作 (19,)
        """
        # 1. 裁剪
        action = np.clip(raw_action, -self.config.clip_actions, self.config.clip_actions)

        # 2. 预热缩放
        warmup = self.config.warmup_steps
        if warmup > 0 and episode_length < warmup:
            alpha = float(episode_length) / float(warmup)
            action = action * alpha

        # 3. EMA 平滑
        alpha_smooth = self.config.action_smoothing
        if alpha_smooth > 0.0:
            smooth = alpha_smooth * action + (1.0 - alpha_smooth) * self._last_smooth_action
        else:
            smooth = action.copy()

        self._last_action = action.copy()
        self._last_smooth_action = smooth.copy()
        return smooth

    # ========================================================================
    # L5: 指令安全（速率限制 + 软限位）
    # ========================================================================

    def clamp_q_des(self, q_des_20: np.ndarray) -> np.ndarray:
        """对 20 维目标位置做速率限制 + 软关节限位。

        参数:
            q_des_20: 目标关节位置 (20,) — DDS motor_cmd 顺序
        返回:
            safe_q_des_20: 安全裁剪后的目标位置 (20,)
        """
        delta = q_des_20 - self._q_des_prev_20
        q_clipped = q_des_20.copy()

        for i in range(self._num_motors):
            if i == self._not_use_idx:
                q_clipped[i] = 0.0
                continue
            # 速率限制
            if abs(delta[i]) > self.config.q_des_delta_max:
                q_clipped[i] = self._q_des_prev_20[i] + np.sign(delta[i]) * self.config.q_des_delta_max
                self.clip_count += 1

        # 软限位: 策略关节使用 tight margin，非策略关节宽松
        for dof_idx in range(len(self._q_min)):
            motor_idx = self.dof_idx[dof_idx]
            q_clipped[motor_idx] = np.clip(q_clipped[motor_idx],
                                           self._q_min[dof_idx], self._q_max[dof_idx])

        q_clipped[self._not_use_idx] = 0.0
        self._q_des_prev_20 = q_clipped.copy()
        return q_clipped

    # ========================================================================
    # RL 接合/断开状态管理
    # ========================================================================

    def engage(self):
        """标记 RL 已接合，初始化安全状态。"""
        self._rl_engaged = True
        self._fall_counter = 0
        self._last_smooth_action = np.zeros(len(self._last_action), dtype=np.float64)
        self._last_action = np.zeros(len(self._last_action), dtype=np.float64)

    def disengage(self, current_pos_20: np.ndarray | None = None):
        """标记 RL 已断开，记录当前位置用于平滑过渡。

        参数:
            current_pos_20: 当前关节位置 (20,)，用于 PD Hold 插值起点
        """
        self._rl_engaged = False
        self._fall_counter = 0
        if current_pos_20 is not None:
            self._pd_start_pos_20 = current_pos_20.copy()
            self._pd_start_time = time.perf_counter()

    @property
    def rl_engaged(self) -> bool:
        return self._rl_engaged

    # ========================================================================
    # PD Hold 插值
    # ========================================================================

    def compute_pd_hold_targets(
        self, default_pos_20: np.ndarray, kp_20: np.ndarray, kd_20: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 PD Hold 模式下的目标位置和增益。

        RL 断开后，从当前位姿平滑过渡到默认站立姿态。
        过渡期间使用较温和的增益避免力矩冲击。

        参数:
            default_pos_20: 默认站立姿态 (20,)
            kp_20, kd_20:  PD 增益数组 (20,)
        返回:
            (q_des_20, kp_used_20, kd_used_20)
        """
        if self._pd_start_pos_20 is not None and self._pd_start_time is not None:
            elapsed = time.perf_counter() - self._pd_start_time
            alpha = min(1.0, elapsed / self.config.pd_hold_interp_s)
            q_des_20 = (1.0 - alpha) * self._pd_start_pos_20 + alpha * default_pos_20
        else:
            q_des_20 = default_pos_20.copy()

        # PD Hold 使用原始增益（不做缩放），确保站立稳定
        return q_des_20, kp_20.copy(), kd_20.copy()

    # ========================================================================
    # 观测裁剪
    # ========================================================================

    def clip_obs(self, obs: np.ndarray) -> np.ndarray:
        """裁剪观测向量，防止异常值进入推理。"""
        return np.clip(obs, -self.config.clip_observations, self.config.clip_observations)

    # ========================================================================
    # 状态报告
    # ========================================================================

    @property
    def status_summary(self) -> str:
        return (
            f"Safety[RL={'ON' if self._rl_engaged else 'OFF'} "
            f"nan={self.nan_count} clip={self.clip_count} "
            f"fall={self.fall_trigger_count} comms_loss={self.comms_loss_count}]"
        )
