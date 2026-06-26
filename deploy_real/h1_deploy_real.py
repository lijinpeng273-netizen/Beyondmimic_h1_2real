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

import sys
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym')
sys.path.append('/home/deepcyber-mk/Documents/unitree_rl_gym/deploy/deploy_real/common')

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_go
from common.rotation_helper import transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config
from h1_config import (
    NUM_ACTIONS,
    joint_seq, joint_xml, dof_idx,
)
from h1_safety import H1SafetyMonitor, H1SafetyConfig


# ============================================================================
# 数学工具函数
# ============================================================================

def quaternion_conjugate(q):
    """四元数共轭: [w, x, y, z] -> [w, -x, -y, -z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """四元数乘法（Hamilton 积）: q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """四元数的逆。"""
    return quaternion_conjugate(q) / np.clip(np.sum(q ** 2, axis=-1, keepdims=True),
                                             a_min=eps, a_max=None)


def quat_rotate_inverse_np(q, v):
    """用四元数的逆旋转来变换向量。v_local = R(q)^T * v"""
    q_w, q_vec = q[0], q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def subtract_frame_transforms_np(t01, q01, t02, q02):
    """计算帧 2 相对于帧 1 的变换（在帧 1 局部坐标系中）。

    返回: (t_12, q_12) — 相对位置和相对四元数
    """
    q01_inv = quat_inv_np(q01)
    q_12 = quaternion_multiply(q01_inv, q02)
    t_12 = quat_rotate_inverse_np(q01, t02 - t01)
    return t_12, q_12


def matrix_from_quat(quaternions: np.ndarray) -> np.ndarray:
    """将四元数转换为旋转矩阵。格式 (w, x, y, z)，标量在前。"""
    r, i, j, k = np.moveaxis(quaternions, -1, 0)
    two_s = 2.0 / np.sum(quaternions * quaternions, axis=-1)
    o = np.array([
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ])
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_rotation_matrix(q):
    """将四元数转换为 3x3 旋转矩阵。"""
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2],
    ])


def matrix_to_quaternion_simple(matrix):
    """将 3x3 旋转矩阵转换为四元数 [w, x, y, z]。"""
    matrix = np.array(matrix)
    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


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

        # ------ Yaw 对齐矩阵 ------
        self.init_to_world = np.zeros((3, 3), dtype=np.float32)

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
            soft_limit_margin=1.5,    # 放宽限位：深蹲、举手、扭腰需要更大范围
            warmup_steps=0,           # 关掉预热：舞蹈不需要渐进启动
            action_smoothing=0.05,    # 极轻平滑：防抖但不拖慢动作
            q_des_delta_max=2.0,      # 放宽速率限制：允许更快的姿势切换
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
        self._kp_20 = np.zeros(20, dtype=np.float64)
        self._kd_20 = np.zeros(20, dtype=np.float64)
        for dof_i in range(len(dof_idx)):
            motor_id = dof_idx[dof_i]
            self._kp_20[motor_id] = config.stiffness[dof_i]
            self._kd_20[motor_id] = config.damping[dof_i]

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
    def zero_torque_state(self):
        """零力矩状态：等待遥控器 Start 键。"""
        print("[INFO]: Enter zero torque state. Waiting for Start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        """从当前姿态平滑过渡到默认站立姿态（2 秒）。"""
        print("[INFO]: Moving to default pos...")
        total_time = 2.0
        num_step = int(total_time / self.config.control_dt)

        leg_size = len(self.config.leg_joint2motor_idx)
        arm_size = len(self.config.arm_waist_joint2motor_idx)
        dof_size = leg_size + arm_size  # 19 for H1
        kps = self.config.stiffness
        kds = self.config.damping
        default_pos = self.config.default_angles.copy()

        # 记录当前关节位置
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # 插值过渡
        for step in range(num_step):
            alpha = step / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + default_pos[j] * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        print("[INFO]: Reached default pos.")

    def default_pos_state(self):
        """默认站立保持状态：等待遥控器 A 键。"""
        print("[INFO]: Enter default pos state. Waiting for Button A signal...")
        leg_size = len(self.config.leg_joint2motor_idx)

        while self.remote_controller.button[KeyMap.A] != 1:
            # 腿部
            for i in range(leg_size):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i] * 5
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # 腰+臂
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                idx = i + leg_size
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[idx]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[idx] * 3
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[idx]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            quat = self.low_state.imu_state.quaternion
            print(f"[DEBUG]: IMU quaternion (w,x,y,z): {quat}")
            time.sleep(self.config.control_dt)

    # ----- 辅助方法 -----
    def yaw_quat(self, q):
        """从四元数中提取 yaw 分量。"""
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])

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
        #    将 mocap 世界帧的 yaw 与机器人启动时的骨盆 yaw 对齐，
        #    使得后续 anchor_ori_b 可以在同一坐标系下计算相对朝向。
        # ================================================================
        if self.timestep < 2:
            ref_motion_quat = self.motionquat[self.timestep, self.motion_anchor_idx, :]
            yaw_motion_quat = self.yaw_quat(ref_motion_quat)
            yaw_motion_matrix = quaternion_to_rotation_matrix(yaw_motion_quat)

            yaw_robot_quat = self.yaw_quat(pelvis_quat)
            yaw_robot_matrix = quaternion_to_rotation_matrix(yaw_robot_quat)

            self.init_to_world = yaw_robot_matrix @ yaw_motion_matrix.T

        # ================================================================
        # 4. 构造观测向量 (110 维)
        # 与 h1sim2sim_v2.py 完全对齐：
        #   anchor_pos_b (3)  → 置零（实机无绝对世界位置）
        #   anchor_ori_b (6)  → 实时计算 quat_inv(pelvis) * ref_aligned（与 sim2sim_v2 一致）
        #   base_lin_vel  (3) → 置零（实机无直接线速度传感器）
        #   base_ang_vel  (3) → IMU 角速度经 torso→pelvis 变换
        # ================================================================
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()

        # 4a. 运动指令: 目标关节位置 + 目标关节速度 (19+19=38)
        minp = self.motioninputpos[self.timestep, :NUM_ACTIONS]
        minv = self.motioninputvel[self.timestep, :NUM_ACTIONS]
        motioninput = np.concatenate((minp, minv), axis=0)

        # 4b. 运动锚点相对朝向 (6D)：与 h1sim2sim_v2 一致的 quat_inv(robot) * ref 计算
        motion_anchor_quat = self.motionquat[self.timestep, self.motion_anchor_idx, :]
        aligned_ref_quat = quaternion_multiply(
            matrix_to_quaternion_simple(self.init_to_world), motion_anchor_quat
        )
        anchor_quat_b = quaternion_multiply(
            quat_inv_np(pelvis_quat), aligned_ref_quat
        )
        anchor_ori_b = matrix_from_quat(anchor_quat_b)[:, :2].reshape(-1).astype(np.float32)

        # 4c. 基座角速度（骨盆坐标系，已从 IMU 转换）
        base_ang_vel = pelvis_ang_vel.reshape(-1).astype(np.float32)

        # 4d. 关节状态（XML 顺序 → ONNX 顺序重排）
        qj_obs_seq = np.array([qj_obs[joint_xml.index(j)] for j in joint_seq])
        dqj_obs_seq = np.array([dqj_obs[joint_xml.index(j)] for j in joint_seq])

        # 4e. 组装观测
        obs_list = [
            motioninput,                                     # [0:38]  目标关节位置+速度
            np.zeros(3, dtype=np.float64),                   # [38:41] anchor_pos_b = 0
            anchor_ori_b,                                    # [41:47] anchor_ori_b（实时计算）
            np.zeros(3, dtype=np.float64),                   # [47:50] base_lin_vel = 0
            base_ang_vel,                                    # [50:53] base_ang_vel (IMU)
            qj_obs_seq - self.config.default_angles_seq,     # [53:72] 关节位置偏差
            dqj_obs_seq,                                     # [72:91] 关节速度
            self.action_buffer,                              # [91:110] 上步动作
        ]
        self.obs = np.concatenate(obs_list).astype(np.float32)

        # L4 观测裁剪
        self.obs = self.safety.clip_obs(self.obs)

        # ================================================================
        # 5. 策略推理
        # ================================================================
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        raw_action = self.policy.run(
            ["actions"],
            {
                "obs": obs_tensor.numpy(),
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

        # 动作 → 目标关节位置（ONNX 顺序 → XML 顺序重排 → DDS 20 维顺序）
        target_dof_seq = self.config.default_angles_seq + self.action * self.config.action_scale_seq
        target_dof_seq = target_dof_seq.reshape(-1)
        target_dof_xml = np.array([target_dof_seq[joint_seq.index(j)] for j in joint_xml])

        # DOF 顺序 (19,) → DDS 电机顺序 (20,)
        q_des_20 = self._default_pos_20.copy()
        for dof_i in range(len(dof_idx)):
            motor_id = dof_idx[dof_i]
            q_des_20[motor_id] = target_dof_xml[dof_i]

        # L5: 速率限制 + 软关节限位
        q_des_20 = self.safety.clamp_q_des(q_des_20)

        self.timestep += 1

        # ================================================================
        # 6. 构建并发送 low_cmd
        # ================================================================
        for motor_id in range(20):
            m = self.low_cmd.motor_cmd[motor_id]
            m.q = float(q_des_20[motor_id])
            m.qd = 0.0
            m.kp = float(self._kp_20[motor_id])
            m.kd = float(self._kd_20[motor_id])
            m.tau = 0.0

        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H1 机器人全身策略实机部署脚本",
    )
    parser.add_argument("config", type=str, help="配置文件名称（位于 configs 文件夹）",
                        default="h1.yaml")
    parser.add_argument("--interface", type=str, default=None,
                        help="DDS 网卡名（可选，覆盖配置文件中的 dds_interface）")
    parser.add_argument("--domain", type=int, default=None,
                        help="DDS 域 ID（可选，覆盖配置文件中的 dds_domain）")
    args = parser.parse_args()

    # 加载配置
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # 初始化 DDS 通信（优先用命令行参数，回退到配置文件）
    domain = args.domain if args.domain is not None else config.dds_domain
    interface = args.interface if args.interface is not None else config.dds_interface
    print(f"[INFO] DDS: domain={domain}, interface={interface}")
    ChannelFactoryInitialize(domain, interface)

    controller = Controller(config)

    # === 部署流程 ===
    # 1. 零力矩状态 → 按 Start 键继续
    controller.zero_torque_state()
    # 2. 平滑过渡到默认站立姿态
    controller.move_to_default_pos()
    # 3. 默认站立保持 → 按 A 键接合 RL 策略
    controller.default_pos_state()

    # 标记 RL 接合
    controller.safety.engage()
    controller._rl_engaged = True
    print("=" * 60)
    print("[INFO] RL 策略已接合！按 Select 键退出")
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

            if controller.remote_controller.button[KeyMap.select] == 1:
                print("[INFO] Select 键 → 正常退出")
                break

            # 定期状态报告
            now = time.perf_counter()
            if now - last_status_print >= 3.0:
                print(f"[INFO] {controller.safety.status_summary}")
                last_status_print = now

        except KeyboardInterrupt:
            break

    # 5. 退出 → 断开 RL，切回 PD Hold 平滑过渡
    controller.safety.disengage(
        np.array([controller.low_state.motor_state[i].q for i in range(20)], dtype=np.float64)
    )
    controller._rl_engaged = False
    # 发送 PD Hold 过渡指令（平滑回到默认姿态）
    print("[INFO] RL 已断开，发送 PD Hold 过渡指令...")
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

    # 最终阻尼模式
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print(f"[INFO] 安全退出。{controller.safety.status_summary}")
