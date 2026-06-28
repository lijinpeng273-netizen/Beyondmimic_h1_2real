"""H1 机器人实机部署脚本（Beyondmimic 全身策略，19 DOF）。

从 ONNX 策略模型加载推理器，通过 DDS 通信协议与宇树 H1 机器人交互，
回放预录制的参考动作（.npz），驱动策略网络实时推理并下发关节指令。

使用示例:
    # 基本用法（网卡/域 ID 在 h1.yaml 中配置，policy_path/motion_file 也在此配置）:
    python h1_deploy_real.py h1.yaml

    # 所有参数均从配置文件读取，无需额外命令行参数
    python h1_deploy_real.py

    # 命令行覆盖（用于临时切换网卡或域 ID，无需修改配置文件）:
    python h1_deploy_real.py h1.yaml --interface enp4s0
    python h1_deploy_real.py --interface lo --domain 1

与 G1 部署的核心差异:
    - 19 关节（无腕部三自由度关节），msg_type="go"
    - IMU 位于躯干，需要 transform_imu_data 转换到骨盆坐标系
    - 观测 110 维，包含运动锚点相对位置和基座线速度（线速度实机近似为零）
    - 电机索引非顺序排列，dof_idx 需要显式指定
"""

from __future__ import annotations

import os
import sys

# 确保 deploy_real/ 在 sys.path 中（当前脚本所在目录，common/ 等依赖在此）
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Unitree SDK 路径（本地安装位置）
_sdk_dir = "/home/ljp/下载/unitree_sdk2_python"
if os.path.isdir(_sdk_dir) and _sdk_dir not in sys.path:
    sys.path.insert(0, _sdk_dir)

import argparse
import time

import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from common.command_helper import create_damping_cmd, init_cmd_go
from common.rotation_helper import transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from h1_config import (
    NUM_ACTIONS,
    joint_seq, joint_xml, dof_idx,
)
from h1_safety import H1SafetyMonitor, H1SafetyConfig
from observation_builder import (
    build_observation,
    remap_xml_to_seq,
    action_to_target,
    compute_init_to_world,
)

# ============================================================================
# 控制器类
# ============================================================================

class Controller:
    """H1 全身策略控制器（19 DOF）。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # ------ 策略网络 ------
        self.policy = ort.InferenceSession(config.policy_path)

        # ------ 过程变量 ------
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.counter = 0
        self.timestep = 0

        # ------ 运动参考数据 ------
        self.motion = np.load(config.motion_file)
        self.motionpos = self.motion["body_pos_w"]
        self.motionquat = self.motion["body_quat_w"]
        self.motioninputpos = self.motion["joint_pos"]
        self.motioninputvel = self.motion["joint_vel"]
        self.action_buffer = np.zeros((self.config.num_actions,), dtype=np.float32)
        # 从轨迹最后 20 帧提取站立锁定目标 (ONNX joint_seq 顺序)
        n_hold = min(20, self.motioninputpos.shape[0])
        self._hold_target_seq = np.mean(self.motioninputpos[-n_hold:, :NUM_ACTIONS], axis=0)

        # ------ Yaw 对齐矩阵 ------
        self.init_to_world = np.eye(3, dtype=np.float64)  # yaw 对齐矩阵，初始为单位阵

        # ------ 运动锚点索引（默认 pelvis=0，可在 YAML 中配置 motion_anchor_index）------
        self.motion_anchor_idx = getattr(config, "motion_anchor_index", 0)

        # ------ DDS 通信初始化（H1 使用 go 协议）------
        #订阅机器人的状态
        if config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()#发布控制指令
            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()
            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError(f"H1 部署仅支持 msg_type='go'，当前配置为: {config.msg_type}")

        # 等待连接
        self.wait_for_low_state()

        # 初始化指令消息（弱力矩电机标记来自 config.weak_motor）
        init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        # ------ 安全监控器（舞蹈优化参数） ------
        safety_config = H1SafetyConfig(
            soft_limit_margin=1.5,    # 舞蹈动作范围大
            warmup_steps=0,           # 不需要预热
            action_smoothing=0.15,    # 手臂防抖（不修改刚度）
            q_des_delta_max=2.0,      # 允许快速切换姿势
        )
        self.safety = H1SafetyMonitor(
            default_angles=config.default_angles,
            default_angles_seq=config.default_angles_seq,
            action_scale_seq=config.action_scale_seq,
            dof_idx=dof_idx,
            control_dt=config.control_dt,
            config=safety_config,
        )

        # ------ PD Hold 默认目标（20 维 DDS 顺序）------
        self._default_pos_20 = np.zeros(20, dtype=np.float64)
        for dof_i in range(len(dof_idx)):
            motor_id = dof_idx[dof_i]
            self._default_pos_20[motor_id] = config.default_angles[dof_i]
        # 舞蹈 + 站立共用实机验证增益 (H1 DDS 经验值)
        # DOF 顺序: L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
        #            R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle,
        #            torso, L_shld_pitch, L_shld_roll, L_shld_yaw, L_elbow,
        #            R_shld_pitch, R_shld_roll, R_shld_yaw, R_elbow
        # H1 实机验证增益（匹配参考值，不再修改）
        _kp_dof = np.array([
            200, 200, 200, 300, 40,        # 左腿
            200, 200, 200, 300, 40,        # 右腿
            300,                            # 腰
            100, 100, 100, 100,            # 左臂
            100, 100, 100, 100,            # 右臂
        ], dtype=np.float64)
        _kd_dof = np.array([
            5, 5, 5, 6, 5,                # 左腿
            5, 5, 5, 6, 5,                # 右腿
            6,                             # 腰
            2, 2, 2, 2,                   # 左臂
            2, 2, 2, 2,                   # 右臂
        ], dtype=np.float64)
        self._kp_20 = np.zeros(20, dtype=np.float64)
        self._kd_20 = np.zeros(20, dtype=np.float64)
        for dof_i in range(len(dof_idx)):
            motor_id = dof_idx[dof_i]
            self._kp_20[motor_id] = _kp_dof[dof_i]
            self._kd_20[motor_id] = _kd_dof[dof_i]

        # 站立专用增益 (与舞蹈相同，H1 实机验证值)
        self._stand_kp_20 = self._kp_20.copy()
        self._stand_kd_20 = self._kd_20.copy()

        # 将站立锁定目标从 ONNX 顺序转为 DDS 20 维顺序
        hold_xml = np.array([self._hold_target_seq[joint_seq.index(j)] for j in joint_xml])
        self._hold_target_20 = self._default_pos_20.copy()
        for dof_i in range(len(dof_idx)):
            self._hold_target_20[dof_idx[dof_i]] = hold_xml[dof_i]

        # 运动总帧数（用于舞蹈结束前提前介入）
        self.num_frames = self.motioninputpos.shape[0]
        # 锁定站立目标（从 NPZ 最后 20 帧提取，预计算，供 pre_blend 和站立使用）
        # 不需要预计算增益——舞蹈期间的策略增益已经验证有效，退出时保持不变

        # RL 是否已接合（供 main loop 安全检测使用）
        self._rl_engaged = False

    # ----- DDS 回调 -----
    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdGo):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("[INFO]: Successfully connected to the robot.")

    # ----- 状态机 -----
    def hold_current_position(self):
        """PD Hold 当前位置：锁定关节，等待遥控器 Start 键。

        与 zero_torque_state 不同，此方法以当前位置为目标发送 PD 指令，
        机器人始终保持主动控制，不会脱力摔倒。
        """
        print("[INFO]: PD Hold at current position. Waiting for A signal...")

        # 读取当前关节位置，锁住不动
        hold_pos = np.zeros(20, dtype=np.float64)
        for i in range(20):
            hold_pos[i] = self.low_state.motor_state[i].q
        hold_pos[9] = 0.0  # not_use 通道置零

        # 站立专用增益（踝 5x / 腿 3x / 臂 1.5x）

        while self.remote_controller.button[KeyMap.A] != 1:
            for motor_id in range(20):
                self.low_cmd.motor_cmd[motor_id].q = float(hold_pos[motor_id])
                self.low_cmd.motor_cmd[motor_id].qd = 0.0
                self.low_cmd.motor_cmd[motor_id].kp = float(self._stand_kp_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].kd = float(self._stand_kd_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        """从当前姿态平滑过渡到默认站立姿态（2 秒）。"""
        print("[INFO]: Moving to default pos...")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        dof_size = len(dof_idx)  # 19 for H1
        default_pos = self.config.default_angles.copy()

        # 记录当前关节位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # 插值过渡（使用站立专用增益）
        for step in range(num_step):
            alpha = step / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + default_pos[j] * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = float(self._stand_kp_20[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].kd = float(self._stand_kd_20[motor_idx])
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("[INFO]: Reached default pos.")

    def default_pos_state(self):
        """默认站立保持状态：等待遥控器 Start 键。"""
        print("[INFO]: Enter default pos state. Waiting for Start signal...")

        while self.remote_controller.button[KeyMap.start] != 1:
            for motor_id in range(20):
                self.low_cmd.motor_cmd[motor_id].q = float(self._default_pos_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].qd = 0.0
                self.low_cmd.motor_cmd[motor_id].kp = float(self._stand_kp_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].kd = float(self._stand_kd_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def blend_to_first_action(self):
        """1 秒平滑过渡：从站立姿态渐进到策略第一个动作。

        逐帧调用策略推理，将动作输出与当前站立姿态做 alpha 插值，
        增益同步从站立增益过渡到策略增益，避免突变力矩冲击。
        """
        print("[INFO]: Blending to first action (1s)...")
        steps = int(1.0 / self.config.control_dt)
        start_pos_20 = np.zeros(20, dtype=np.float64)
        for i in range(20):
            start_pos_20[i] = self.low_state.motor_state[i].q

        for step in range(steps):
            alpha = float(step) / float(steps)
            self._run_policy_step()
            target_dof_xml = action_to_target(
                self.action, self.config.default_angles_seq,
                self.config.action_scale_seq, joint_seq, joint_xml,
            )
            target_20 = self._default_pos_20.copy()
            for dof_i in range(len(dof_idx)):
                target_20[dof_idx[dof_i]] = target_dof_xml[dof_i]
            for motor_id in range(20):
                self.low_cmd.motor_cmd[motor_id].q = float(
                    (1.0 - alpha) * start_pos_20[motor_id] + alpha * target_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].qd = 0.0
                self.low_cmd.motor_cmd[motor_id].kp = float(
                    (1.0 - alpha) * self._stand_kp_20[motor_id] + alpha * self._kp_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].kd = float(
                    (1.0 - alpha) * self._stand_kd_20[motor_id] + alpha * self._kd_20[motor_id])
                self.low_cmd.motor_cmd[motor_id].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("[INFO]: Blend complete.")

    def _run_policy_step(self):
        """执行单帧策略推理，更新 self.action。"""
        # 传感器校验
        valid, pos_20, vel_20, imu_quat, imu_gyro = self.safety.check_sensors(self.low_state)
        # 关节状态
        for i in range(len(dof_idx)):
            self.qj[i] = pos_20[dof_idx[i]]
            self.dqj[i] = vel_20[dof_idx[i]]
        # IMU（IMU 在躯干上，需变换到骨盆坐标系）
        imu_ang_vel = imu_gyro.reshape(1, 3).astype(np.float32)
        waist_yaw_idx = self.config.arm_waist_joint2motor_idx[0]
        pelvis_quat, pelvis_ang_vel = transform_imu_data(
            waist_yaw=pos_20[waist_yaw_idx], waist_yaw_omega=vel_20[waist_yaw_idx],
            imu_quat=imu_quat, imu_omega=imu_ang_vel,
        )
        # cmd
        minp = self.motioninputpos[self.timestep, :NUM_ACTIONS]
        minv = self.motioninputvel[self.timestep, :NUM_ACTIONS]
        cmd = np.concatenate((minp, minv), axis=0)
        if self.timestep >= self.num_frames - 20:
            alpha = float(self.timestep - (self.num_frames - 20)) / 20.0
            alpha = min(alpha, 1.0)
            cmd[:NUM_ACTIONS] = (1.0 - alpha) * cmd[:NUM_ACTIONS] + alpha * self._hold_target_seq
            cmd[NUM_ACTIONS:] *= (1.0 - alpha)
        # 观测
        qj_seq = remap_xml_to_seq(self.qj, joint_xml, joint_seq)
        dqj_seq = remap_xml_to_seq(self.dqj, joint_xml, joint_seq)
        obs = build_observation(
            cmd=cmd, pelvis_quat=pelvis_quat,
            motion_ref_quat=self.motionquat[self.timestep, self.motion_anchor_idx, :],
            pelvis_ang_vel=pelvis_ang_vel, qpos_seq=qj_seq, qvel_seq=dqj_seq,
            action_buffer=self.action_buffer,
            default_angles_seq=self.config.default_angles_seq,
            init_to_world=self.init_to_world, num_obs=self.config.num_obs,
        )
        obs = self.safety.clip_obs(obs)
        # 推理
        raw_action = self.policy.run(
            ["actions"],
            {"obs": np.expand_dims(obs, axis=0),
             "time_step": np.array([self.timestep], dtype=np.float32).reshape(1, 1)},
        )[0]
        raw_action = np.asarray(raw_action).reshape(-1)
        valid_action, safe_action = self.safety.check_action(raw_action)
        self.action = self.safety.process_action(safe_action, self.timestep)
        self.action_buffer = self.action.copy()

    # ----- 主控制循环 -----
    def run(self):
        """单步控制循环：读取状态 → 构造观测 → 策略推理 → 下发指令。"""
        self.counter += 1

        # ================================================================
        # L1: 传感器校验（NaN/Inf 过滤 + 回退）
        # ================================================================
        valid, pos_20, vel_20, imu_quat, imu_gyro = self.safety.check_sensors(self.low_state)
        if not valid and self.counter % 100 == 0:
            print(f"[WARN] 传感器数据异常 (累计 {self.safety.nan_count} 次)")

        # ================================================================
        # 1. 读取关节状态（已校验安全的数据）
        # ================================================================
        for i in range(len(dof_idx)):
            self.qj[i] = pos_20[dof_idx[i]]
            self.dqj[i] = vel_20[dof_idx[i]]

        # ================================================================
        # 2. IMU 数据处理（躯干 → 骨盆坐标系变换）
        # ================================================================
        imu_ang_vel = imu_gyro.reshape(1, 3).astype(np.float32)

        waist_yaw_idx = self.config.arm_waist_joint2motor_idx[0]
        waist_yaw = pos_20[waist_yaw_idx]
        waist_yaw_omega = vel_20[waist_yaw_idx]
        pelvis_quat, pelvis_ang_vel = transform_imu_data(
            waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega,
            imu_quat=imu_quat, imu_omega=imu_ang_vel,
        )

        # ================================================================
        # 3. Yaw 对齐初始化（前 2 步）
        #    将 mocap 世界帧的 yaw 与机器人启动时的骨盆 yaw 对齐。
        # ================================================================
        if self.timestep < 2:
            ref_quat = self.motionquat[self.timestep, self.motion_anchor_idx, :]
            self.init_to_world = compute_init_to_world(pelvis_quat, ref_quat)

        # ================================================================
        # 4. 构造观测向量 (110 维) — 使用共享模块，与 h1sim2sim_v2.py 完全一致
        # ================================================================
        # cmd: 目标关节位置 + 目标关节速度 (19+19=38)，来自 .npz
        minp = self.motioninputpos[self.timestep, :NUM_ACTIONS]
        minv = self.motioninputvel[self.timestep, :NUM_ACTIONS]
        cmd = np.concatenate((minp, minv), axis=0)

        # ★ 舞蹈结束前 20 帧：命令位置 → 锁定姿态，速度 → 0
        pre_blend = 20
        if self.timestep >= self.num_frames - pre_blend:
            alpha = float(self.timestep - (self.num_frames - pre_blend)) / float(pre_blend)
            alpha = min(alpha, 1.0)
            cmd[:NUM_ACTIONS] = (1.0 - alpha) * cmd[:NUM_ACTIONS] + alpha * self._hold_target_seq
            cmd[NUM_ACTIONS:] *= (1.0 - alpha)

        # 参考锚点四元数（来自 .npz body_quat_w）
        motion_ref_quat = self.motionquat[self.timestep, self.motion_anchor_idx, :]

        # 关节状态 XML→ONNX 重排
        qj_obs_seq = remap_xml_to_seq(self.qj, joint_xml, joint_seq)
        dqj_obs_seq = remap_xml_to_seq(self.dqj, joint_xml, joint_seq)

        # 构造观测（共享模块，sim2sim 和实机同一份代码）
        self.obs = build_observation(
            cmd=cmd,
            pelvis_quat=pelvis_quat,
            motion_ref_quat=motion_ref_quat,
            pelvis_ang_vel=pelvis_ang_vel,
            qpos_seq=qj_obs_seq,
            qvel_seq=dqj_obs_seq,
            action_buffer=self.action_buffer,
            default_angles_seq=self.config.default_angles_seq,
            init_to_world=self.init_to_world,
        )
        # 观测裁剪（安全模块 L4）
        self.obs = self.safety.clip_obs(self.obs)

        # ================================================================
        # 5. 策略推理
        # ================================================================
        obs_tensor = np.expand_dims(self.obs, axis=0)
        raw_action = self.policy.run(
            ["actions"],
            {
                "obs": obs_tensor,
                "time_step": np.array([self.timestep], dtype=np.float32).reshape(1, 1),
            },
        )[0]
        raw_action = np.asarray(raw_action).reshape(-1)

        # L4: 动作安全检查 + 处理（裁剪、预热、平滑）
        valid_action, safe_action = self.safety.check_action(raw_action)
        if not valid_action:
            print(f"[WARN] 异常动作! norm={np.linalg.norm(raw_action):.1f}, 使用上一步动作")
        processed_action = self.safety.process_action(safe_action, self.timestep)
        self.action = processed_action.copy()
        self.action_buffer = processed_action.copy()

        # 动作 → 目标关节位置（ONNX 顺序 → XML 顺序，共享模块）
        target_dof_xml = action_to_target(
            self.action, self.config.default_angles_seq,
            self.config.action_scale_seq, joint_seq, joint_xml,
        )

        # DOF 顺序 (19,) → DDS 电机顺序 (20,)
        q_des_20 = self._default_pos_20.copy()
        for dof_i in range(len(dof_idx)):
            motor_id = dof_idx[dof_i]
            q_des_20[motor_id] = target_dof_xml[dof_i]

        # L5: 速率限制 + 软关节限位
        q_des_20 = self.safety.clamp_q_des(q_des_20)

        for motor_id in range(20):
            self.low_cmd.motor_cmd[motor_id].q = float(q_des_20[motor_id])
            self.low_cmd.motor_cmd[motor_id].qd = 0.0
            self.low_cmd.motor_cmd[motor_id].kp = float(self._kp_20[motor_id])
            self.low_cmd.motor_cmd[motor_id].kd = float(self._kd_20[motor_id])
            self.low_cmd.motor_cmd[motor_id].tau = 0.0

        self.timestep += 1

        # ================================================================
        # 6. 发送 low_cmd（指令已在上面填充）
        # ================================================================

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H1 机器人全身策略实机部署脚本",
    )
    parser.add_argument("config", type=str, nargs="?", default="h1.yaml",
                        help="配置文件名称（位于 configs 文件夹）")
    parser.add_argument("--interface", type=str, default=None,
                        help="DDS 网卡名（可选，覆盖配置文件中的 dds_interface）")
    parser.add_argument("--domain", type=int, default=None,
                        help="DDS 域 ID（可选，覆盖配置文件中的 dds_domain）")
    args = parser.parse_args()

    # 加载配置（相对脚本所在目录）
    config_path = os.path.join(_script_dir, "configs", args.config)
    config = Config(config_path)

    # 初始化 DDS 通信（优先用命令行参数，回退到配置文件）
    domain = args.domain if args.domain is not None else config.dds_domain
    interface = args.interface if args.interface is not None else config.dds_interface
    print(f"[INFO] DDS: domain={domain}, interface={interface}")
    ChannelFactoryInitialize(domain, interface)

    controller = Controller(config)

    # === 部署流程 ===
    # 1. PD Hold 锁定当前位置（不脱力）→ 按 Start 键继续
    controller.hold_current_position()
    # 2. 平滑过渡到默认站立姿态
    controller.move_to_default_pos()
    # 3. 默认站立保持 → 按 A 键接合 RL 策略
    controller.default_pos_state()
    # 4. 1 秒平滑过渡到第一个舞蹈动作（防突变），然后直接开始
    controller.blend_to_first_action()

    # 标记 RL 接合
    controller.safety.engage()
    controller._rl_engaged = True
    print("=" * 60)
    print("[INFO] RL 策略已接合！按 B 退出")
    print(f"[INFO] {controller.safety.status_summary}")
    print("=" * 60)

    # 4. 主控制循环
    last_status_print = time.perf_counter()
    while True:
        try:
            controller.run()

            # ── 安全检测 ──
            if not controller.safety.check_comms():
                print(f"[SAFETY] 通信丢失 → 断开 RL")
                break

            # 摔倒检测：使用骨盆 IMU 四元数（已由 run() 中 transform_imu_data 计算）
            pelvis_quat_for_fall = np.array([
                controller.safety._prev_imu_quat[0],
                controller.safety._prev_imu_quat[1],
                controller.safety._prev_imu_quat[2],
                controller.safety._prev_imu_quat[3],
            ], dtype=np.float64)
            # 注意：此处使用躯干原始 IMU 做摔倒检测是合理的 —
            # pelvis 和 torso 的 Roll/Pitch 差异仅来自腰部关节柔性，
            # 对摔倒判断影响可忽略。若需精确检测，可传入 pelvis_quat。
            if not controller.safety.check_fall(pelvis_quat_for_fall):
                print(f"[SAFETY] 检测到摔倒 → 断开 RL")
                break

            if controller.remote_controller.button[KeyMap.B] == 1:
                print("[INFO] B 键 → 中断舞蹈退出")
                break

            # 定期状态报告
            now = time.perf_counter()
            if now - last_status_print >= 3.0:
                print(f"[INFO] {controller.safety.status_summary}")
                last_status_print = now

        except KeyboardInterrupt:
            break

    # 5. 退出 → 断开 RL，切回 PD Hold 平滑过渡到默认站立
    controller.safety.disengage(
        np.array([controller.low_state.motor_state[i].q for i in range(20)], dtype=np.float64)
    )
    controller._rl_engaged = False

    # 2 秒平滑过渡到默认站立姿态
    print("[INFO] RL 已断开，平滑过渡到默认站立姿态...")
    for _ in range(int(2.0 / config.control_dt)):
        q_des, kp, kd = controller.safety.compute_pd_hold_targets(
            controller._default_pos_20, controller._kp_20, controller._kd_20,
        )
        for motor_id in range(20):
            controller.low_cmd.motor_cmd[motor_id].q = float(q_des[motor_id])
            controller.low_cmd.motor_cmd[motor_id].qd = 0.0
            controller.low_cmd.motor_cmd[motor_id].kp = float(kp[motor_id])
            controller.low_cmd.motor_cmd[motor_id].kd = float(kd[motor_id])
            controller.low_cmd.motor_cmd[motor_id].tau = 0.0
        controller.send_cmd(controller.low_cmd)
        time.sleep(config.control_dt)

    # 6. 舞蹈结束 → 锁定轨迹末尾姿态（增益不变，策略增益已验证有效）
    hold_pos = controller._hold_target_20.copy()
    print("[INFO] 锁定站立（轨迹末尾姿态，策略增益，按 X 退出）")
    while True:
        for motor_id in range(20):
            controller.low_cmd.motor_cmd[motor_id].q = float(hold_pos[motor_id])
            controller.low_cmd.motor_cmd[motor_id].qd = 0.0
            controller.low_cmd.motor_cmd[motor_id].kp = float(controller._kp_20[motor_id])
            controller.low_cmd.motor_cmd[motor_id].kd = float(controller._kd_20[motor_id])
            controller.low_cmd.motor_cmd[motor_id].tau = 0.0
        controller.send_cmd(controller.low_cmd)
        time.sleep(config.control_dt)
        if controller.remote_controller.button[KeyMap.X] == 1:
            print("[INFO] X 键 → 进入阻尼关机")
            break

    # 7. 最终阻尼模式
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print(f"[INFO] 安全退出。{controller.safety.status_summary}")
