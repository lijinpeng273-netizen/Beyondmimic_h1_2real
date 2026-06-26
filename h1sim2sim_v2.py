# -*- coding: utf-8 -*-
"""
h1sim2sim_v2.py — H1 Sim2Sim 仿真（ONNX 自举模式，仅需 ONNX 文件）
====================================================================
与实机部署方式完全对齐：
  - 参考轨迹从 ONNX 模型内嵌查表获取（无需外置 .npz）
  - 观测仅包含实机可获取的通道（proprioception + IMU + ONNX 参考）
  - 锚点相对朝向从 ONNX ref_quat + 机器人 IMU quat 实时计算
  - base_lin_vel / anchor_pos_b 置零（实机无直接传感器）

用法:
    python h1sim2sim_v2.py                  # 使用默认配置
    python h1sim2sim_v2.py --config my.py   # 使用自定义配置

键盘控制:
    6 = 重置站立   7 = 播放策略   8 = 切换弹性绳
    MuJoCo 窗口中 BACKSPACE = 重置仿真
"""

from __future__ import annotations

import json
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime

# ============================================================================
# 加载配置
# ============================================================================

# 允许通过命令行 --config 指定自定义配置文件
_config_path = "h1sim2sim_v2_config"
for _i, _arg in enumerate(sys.argv):
    if _arg == "--config" and _i + 1 < len(sys.argv):
        _config_path = sys.argv[_i + 1].replace(".py", "")
        break

try:
    _cfg = __import__(_config_path.replace("/", ".").replace("\\", "."))
except ImportError:
    print(f"[ERROR] 无法加载配置文件: {_config_path}.py")
    sys.exit(1)

# 展开配置变量到当前命名空间
XML_PATH            = _cfg.XML_PATH
POLICY_PATH         = _cfg.POLICY_PATH
SIMULATION_DT       = _cfg.SIMULATION_DT
CONTROL_DECIMATION  = _cfg.CONTROL_DECIMATION
SIMULATION_DURATION = _cfg.SIMULATION_DURATION
NUM_ACTIONS         = _cfg.NUM_ACTIONS
NUM_OBS             = _cfg.NUM_OBS
REFERENCE_BODY      = _cfg.REFERENCE_BODY
INIT_PELVIS_Z       = _cfg.INIT_PELVIS_Z
STAND_STIFFNESS_SCALE = _cfg.STAND_STIFFNESS_SCALE
STAND_DAMPING_SCALE   = _cfg.STAND_DAMPING_SCALE
DEFAULT_KP          = _cfg.DEFAULT_KP
DEFAULT_KD_RATIO    = _cfg.DEFAULT_KD_RATIO
ELASTIC_BAND_ENABLED = _cfg.ELASTIC_BAND_ENABLED
PRINT_INTERVAL      = _cfg.PRINT_INTERVAL

# ============================================================================
# 状态机常量
# ============================================================================

STATE_IDLE = 0       # 待机：保持初始站立姿态
STATE_PLAYING = 1    # 播放：策略推理中
STATE_DONE = 2       # 完成：保持最后姿态

# ============================================================================
# 弹性绳（辅助站立，仅仿真使用）
# ============================================================================

class ElasticBand:
    """虚拟弹性拉力绳，连接头顶固定点与 torso_link。

    物理: f = Kp·(distance - length) - Kd·v（沿绳方向）
    """

    def __init__(self, point, stiffness=300.0, damping=150.0):
        self.stiffness = stiffness
        self.damping = damping
        self.point = np.array(point, dtype=np.float64)
        self.length = 0.0       # 松弛长度，0 = 始终有拉力
        self.enable = True

    def advance(self, x, dx):
        """计算拉力向量。x: 连接点位置 (3,), dx: 连接点速度 (3,)"""
        delta = self.point - x
        distance = np.linalg.norm(delta)
        if distance < 1e-8:
            return np.zeros(3)
        direction = delta / distance
        v = float(np.dot(dx, direction))
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    @staticmethod
    def keyboard_callback(key):
        """MuJoCo 键盘回调: 键 8 切换弹性绳"""
        if key == mujoco.glfw.glfw.KEY_8:
            # 回调中通过闭包引用外部的 elastic_band 实例
            pass  # 在主循环中处理


# ============================================================================
# 数学工具（纯 NumPy，与训练环境对齐）
# ============================================================================

def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵 (3,3)"""
    r, i, j, k = np.moveaxis(q, -1, 0)
    two_s = 2.0 / np.sum(q * q, axis=-1)
    o = np.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), axis=-1)
    return o.reshape(q.shape[:-1] + (3, 3))


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (Hamilton 积)，输入输出均为 wxyz"""
    if q1.shape != q2.shape:
        raise ValueError(f"quat shape mismatch: {q1.shape} != {q2.shape}")
    shape = q1.shape
    q1, q2 = q1.reshape(-1, 4), q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.stack([w, x, y, z], axis=-1).reshape(shape)


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    """四元数共轭"""
    shape = q.shape
    q = q.reshape(-1, 4)
    return np.concatenate((q[..., 0:1], -q[..., 1:]), axis=-1).reshape(shape)


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """四元数逆"""
    return quat_conjugate_np(q) / np.clip(
        np.sum(q ** 2, axis=-1, keepdims=True), a_min=eps, a_max=None)


def get_obs(data):
    """从 MuJoCo 提取机器人状态观测。

    返回: (qpos, dq, quat, v, omega, gvec, state_tau)
        quat, omega  来自 IMU sensor（若 XML 未定义 sensor 则回退 free joint）
        v            基座线速度（局部坐标系，实机不可用）
    """
    qpos = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    try:
        quat = data.sensor("orientation").data[[0, 1, 2, 3]].astype(np.double)
        omega = data.sensor("angular-velocity").data.astype(np.double)
    except (KeyError, IndexError):
        quat = data.qpos[3:7].astype(np.double)
        omega = data.qvel[3:6].astype(np.double)
    rotm = np.zeros(9)
    mujoco.mju_quat2Mat(rotm, quat)
    rotm = rotm.reshape((3, 3))
    v = (rotm.T @ data.qvel[:3]).astype(np.double)
    gvec = (rotm.T @ np.array([0.0, 0.0, -1.0])).astype(np.double)
    state_tau = data.qfrc_actuator.astype(np.double) - data.qfrc_bias.astype(np.double)
    return qpos, dq, quat, v, omega, gvec, state_tau


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 位置控制器: τ = Kp·(q_tgt - q) + Kd·(dq_tgt - dq)"""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ============================================================================
# ONNX 元数据解析
# ============================================================================

def _parse_str_list(val: str) -> list[str]:
    if not val:
        return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [x.strip() for x in val.split(",")]


def _parse_csv_list(val: str, dtype=float) -> list:
    if not val:
        return []
    try:
        parsed = json.loads(val.replace("'", '"'))
        if isinstance(parsed, list):
            return [dtype(x) if isinstance(x, (int, float, str)) else x for x in parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    return [dtype(x.strip()) for x in val.split(",")]


# ============================================================================
# ONNX 模型加载 & 元数据提取
# ============================================================================

def load_onnx_policy(policy_path: str):
    """加载 ONNX 策略，提取元数据（关节名、PD 增益、action_scale 等）。

    返回: (session, metadata_dict, joint_seq, input_shape)
    """
    if not os.path.isfile(policy_path):
        raise FileNotFoundError(f"ONNX 模型未找到: {policy_path}")

    # 元数据
    model = onnx.load(policy_path)
    joint_seq = None
    joint_pos_array_seq = None
    stiffness_array_seq = None
    damping_array_seq = None
    action_scale_seq = None
    anchor_body_name = None

    for prop in model.metadata_props:
        if prop.key == "joint_names":
            joint_seq = _parse_str_list(prop.value)
        elif prop.key == "default_joint_pos":
            joint_pos_array_seq = np.array(_parse_csv_list(prop.value))
        elif prop.key == "joint_stiffness":
            stiffness_array_seq = np.array(_parse_csv_list(prop.value))
        elif prop.key == "joint_damping":
            damping_array_seq = np.array(_parse_csv_list(prop.value))
        elif prop.key == "action_scale":
            action_scale_seq = np.array(_parse_csv_list(prop.value))
        elif prop.key == "anchor_body_name":
            anchor_body_name = prop.value

    if joint_seq is None:
        raise ValueError("ONNX 模型中未找到 joint_names 元数据")

    # 推理会话
    session = onnxruntime.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    input_shape = session.get_inputs()[0].shape

    print(f"[INFO] ONNX joints: {len(joint_seq)}  输入维度: {input_shape}")

    metadata = {
        "joint_seq": joint_seq,
        "joint_pos_array_seq": joint_pos_array_seq,
        "stiffness_array_seq": stiffness_array_seq,
        "damping_array_seq": damping_array_seq,
        "action_scale_seq": action_scale_seq,
        "anchor_body_name": anchor_body_name,
    }
    return session, metadata, input_shape


# ============================================================================
# 主仿真
# ============================================================================

def run():
    """运行 ONNX 自举模式的 H1 Sim2Sim 仿真。"""

    # ------------------------------------------------------------------
    # 1. 加载 ONNX 策略
    # ------------------------------------------------------------------
    session, meta, input_shape = load_onnx_policy(POLICY_PATH)
    joint_seq = meta["joint_seq"]
    joint_pos_array_seq = meta["joint_pos_array_seq"]
    stiffness_array_seq = meta["stiffness_array_seq"]
    damping_array_seq = meta["damping_array_seq"]
    action_scale_seq = meta["action_scale_seq"]
    anchor_body_name = meta["anchor_body_name"]

    num_actions = len(joint_seq)

    # ------------------------------------------------------------------
    # 2. 加载 MuJoCo 模型
    # ------------------------------------------------------------------
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    # ------------------------------------------------------------------
    # 3. 建立关节名称映射 (ONNX 顺序 ↔ XML 顺序)
    # ------------------------------------------------------------------
    xml_joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(1, m.njnt)]
    joint_xml = [j for j in xml_joint_names if j in joint_seq]
    skipped = set(xml_joint_names) - set(joint_xml)
    if skipped:
        print(f"[INFO] XML 中存在但 ONNX 中无的关节（将被跳过）: {skipped}")
    print(f"[INFO] 关节映射: ONNX {len(joint_seq)} ↔ XML {len(joint_xml)}")

    # 将 ONNX 顺序的参数重映射到 XML 顺序
    def _remap_to_xml(arr_seq, fallback=None):
        """将 ONNX 顺序的数组 arr_seq 重映射到 XML 顺序"""
        if arr_seq is not None and len(arr_seq) == len(joint_seq):
            return np.array([arr_seq[joint_seq.index(j)] for j in joint_xml])
        return fallback

    # default_joint_pos
    if joint_pos_array_seq is None or len(joint_pos_array_seq) == 0:
        joint_pos_array_xml = m.qpos0[7:7 + len(joint_xml)]
        joint_pos_array_seq = np.array([joint_pos_array_xml[joint_xml.index(j)] for j in joint_seq])
        print("[INFO] default_joint_pos 取自 XML qpos0")
    joint_pos_array = _remap_to_xml(joint_pos_array_seq)

    # PD 增益
    if stiffness_array_seq is None or len(stiffness_array_seq) == 0:
        print("[WARNING] ONNX 元数据缺失 joint_stiffness，使用默认 PD 增益")
        stiffness_array = np.array([DEFAULT_KP.get(j, 50.0) for j in joint_xml])
        damping_array = np.array([DEFAULT_KP.get(j, 50.0) * DEFAULT_KD_RATIO for j in joint_xml])
    else:
        stiffness_array = _remap_to_xml(stiffness_array_seq)
        damping_array = _remap_to_xml(damping_array_seq)

    # 站立 / 策略两套增益
    stand_kp = stiffness_array * STAND_STIFFNESS_SCALE
    stand_kd = damping_array * STAND_DAMPING_SCALE
    policy_kp = stiffness_array
    policy_kd = damping_array
    cur_kp = stand_kp.copy()
    cur_kd = stand_kd.copy()

    # action_scale —— 保持 ONNX 顺序，不做 XML 重映射
    # 原因: target_dof_seq = action * scale + default_pos 是 ONNX 顺序的运算，
    #       最后才通过 joint_seq.index 重映射到 XML 顺序
    if action_scale_seq is None or len(action_scale_seq) == 0:
        action_scale = np.ones(num_actions)
    else:
        action_scale = action_scale_seq.copy()
    print(f"[INFO] action_scale (前3, ONNX顺序): {[f'{s:.3f}' for s in action_scale[:3]]}")

    # ------------------------------------------------------------------
    # 4. 锚点刚体
    # ------------------------------------------------------------------
    ref_body_name = anchor_body_name or REFERENCE_BODY
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ref_body_name)
    if body_id == -1:
        raise ValueError(f"锚点刚体 '{ref_body_name}' 未在 XML 中找到")
    # ONNX body_quat_w 输出中 pelvis 固定为 index 0
    motion_body_idx = 0
    print(f"[INFO] 锚点: '{ref_body_name}' (XML id={body_id})")

    # ------------------------------------------------------------------
    # 5. ONNX 自举引导
    # ------------------------------------------------------------------
    policy_input_names = [inp.name for inp in session.get_inputs()]
    obs_name = policy_input_names[0]      # "obs"
    ts_name = policy_input_names[1]       # "time_step"

    bootstrap_obs = np.zeros((1, input_shape[1]), dtype=np.float32)
    bootstrap_ts = np.array([[0.0]], dtype=np.float32)
    bootstrap_out = session.run(
        ["joint_pos", "joint_vel", "body_quat_w"],
        {obs_name: bootstrap_obs, ts_name: bootstrap_ts},
    )
    # 参考轨迹缓存（供下一次观测使用）
    # 对部分观测数据置零
    motion_command_t = np.concatenate([
        np.asarray(bootstrap_out[0], dtype=np.float32).reshape(1, -1),
        np.asarray(bootstrap_out[1], dtype=np.float32).reshape(1, -1),
    ], axis=1)   # (1, 38)
    motion_ref_quat_t = np.asarray(
        bootstrap_out[2], dtype=np.float64
    )[0, motion_body_idx, :].copy()  # pelvis 四元数 (4,)
    print(f"[INFO] ONNX 自举完成 — motion_command_t: {motion_command_t.shape}")

    # ------------------------------------------------------------------
    # 6. 初始化仿真状态
    # ------------------------------------------------------------------
    num_frames = int(SIMULATION_DURATION / (CONTROL_DECIMATION * SIMULATION_DT))
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    action_buffer = np.zeros(num_actions, dtype=np.float32)
    counter = 0
    timestep = 0
    v = np.zeros(3, dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # 7. 弹性绳 & 键盘状态
    # ------------------------------------------------------------------
    sim_state = {"state": STATE_IDLE, "key_pressed": None}
    initial_target = joint_pos_array.copy()

    rope_anchor = np.array([0.0, 0.0, float(INIT_PELVIS_Z) + 1.0])
    elastic_band = ElasticBand(
        point=rope_anchor,
        stiffness=_cfg.ELASTIC_BAND_STIFFNESS,
        damping=_cfg.ELASTIC_BAND_DAMPING,
    )
    elastic_band.enable = ELASTIC_BAND_ENABLED
    band_attached_body = m.body("torso_link").id

    def key_callback(keycode):
        sim_state["key_pressed"] = keycode

    def reset_to_stand():
        """重置到站立姿态，启用站立增益 + 弹性绳"""
        nonlocal timestep, counter, action_buffer, cur_kp, cur_kd
        _target = initial_target.copy()
        timestep = 0
        counter = 0
        action_buffer = np.zeros(num_actions, dtype=np.float32)
        cur_kp = stand_kp.copy()
        cur_kd = stand_kd.copy()
        if elastic_band is not None:
            elastic_band.enable = True
        d.qpos[2] = INIT_PELVIS_Z
        d.qpos[7:7 + len(joint_xml)] = joint_pos_array
        d.qvel[:] = 0.0
        if m.nu > 0:
            d.ctrl[:] = 0.0
        d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        obs[:] = 0.0
        print("[INFO] → IDLE (站立姿态)")
        return _target

    def start_playback():
        """开始单次策略播放，重新自举获取初始参考"""
        nonlocal timestep, counter, action_buffer, cur_kp, cur_kd
        nonlocal motion_command_t, motion_ref_quat_t
        _target = initial_target.copy()
        timestep = 0
        counter = 0
        cur_kp = policy_kp
        cur_kd = policy_kd
        action_buffer = np.zeros(num_actions, dtype=np.float32)
        obs[:] = 0.0
        # 重新自举
        bootstrap_out_local = session.run(
            ["joint_pos", "joint_vel", "body_quat_w"],
            {obs_name: np.zeros((1, input_shape[1]), dtype=np.float32),
             ts_name: np.array([[0.0]], dtype=np.float32)},
        )
        motion_command_t = np.concatenate([
            np.asarray(bootstrap_out_local[0], dtype=np.float32).reshape(1, -1),
            np.asarray(bootstrap_out_local[1], dtype=np.float32).reshape(1, -1),
        ], axis=1)
        motion_ref_quat_t = np.asarray(
            bootstrap_out_local[2], dtype=np.float64
        )[0, motion_body_idx, :].copy()
        print(f"[INFO] → PLAYING ({num_frames} 步, ONNX 自举模式)")
        return _target

    # ------------------------------------------------------------------
    # 8. 设置初始姿态
    # ------------------------------------------------------------------
    target_dof_pos = initial_target.copy()
    d.qpos[2] = INIT_PELVIS_Z
    if len(d.qpos) > 7:
        d.qpos[7:7 + len(joint_xml)] = joint_pos_array
    d.qvel[:] = 0.0
    if m.nu > 0:
        d.ctrl[:] = 0.0
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)

    print("[INFO] 键盘: 6=站立  7=播放策略  8=弹性绳")
    print("[INFO] 观测模式: Realistic（仅 proprioception + IMU + ONNX 参考）")
    print(f"[INFO] 仿真时长: {SIMULATION_DURATION}s  策略步数: {num_frames}")

    # ------------------------------------------------------------------
    # 9. 主仿真循环
    # ------------------------------------------------------------------
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # ---- 读取机器人当前状态 ----
            if sim_state["state"] == STATE_PLAYING:
                _, _, _, v, omega, _, _ = get_obs(d)

            # ---- 键盘输入 ----
            key = sim_state["key_pressed"]
            if key is not None:
                sim_state["key_pressed"] = None
                glfw = mujoco.glfw.glfw
                if key == glfw.KEY_6:
                    target_dof_pos = reset_to_stand()
                    sim_state["state"] = STATE_IDLE
                elif key == glfw.KEY_7:
                    target_dof_pos = start_playback()
                    sim_state["state"] = STATE_PLAYING
                elif key == glfw.KEY_8:
                    elastic_band.enable = not elastic_band.enable
                    print(f"[弹力绳] {'拉紧' if elastic_band.enable else '解开'}")

            # ---- 策略推理（每 CONTROL_DECIMATION 物理步执行一次）----
            if sim_state["state"] == STATE_PLAYING:
                if counter % CONTROL_DECIMATION == 0:
                    if timestep < num_frames:
                        idx = timestep

                        # 关节状态: XML 顺序 → ONNX 顺序
                        qpos_xml = d.qpos[7:7 + len(joint_xml)]
                        qpos_seq = np.array([qpos_xml[joint_xml.index(j)] for j in joint_seq])
                        qvel_xml = d.qvel[6:6 + len(joint_xml)]
                        qvel_seq = np.array([qvel_xml[joint_xml.index(j)] for j in joint_seq])

                        # 机器人锚点朝向（用于计算 relative orientation）
                        robot_anchor_quat = d.xquat[body_id].copy()

                        # 从缓存构建 command（来自上步 ONNX 输出）
                        motioninput = motion_command_t.reshape(-1)
                        cmd = motioninput[:num_actions * 2]

                        # anchor_ori_b: 从 ONNX ref_quat + 机器人当前朝向实时计算
                        anchor_quat_b = quat_mul_np(
                            quat_inv_np(robot_anchor_quat), motion_ref_quat_t
                        )
                        mat = matrix_from_quat(anchor_quat_b)
                        anchor_ori_b = mat[..., :2].reshape(-1).astype(np.float32)

                        # 构建 110 维观测
                        # [0:38]  command（参考关节位置+速度）
                        # [38:41] anchor_pos_b = 0（实机不可获取）
                        # [41:47] anchor_ori_b（ONNX ref_quat + IMU quat 计算）
                        # [47:50] base_lin_vel = 0（实机不可获取）
                        # [50:53] base_ang_vel（IMU gyro）
                        # [53:72] joint_pos（当前 - 默认）
                        # [72:91] joint_vel
                        # [91:110] last_action
                        obs_list = [
                            cmd,
                            np.zeros(3, dtype=np.float64),
                            anchor_ori_b,
                            np.zeros(3, dtype=np.float64),
                            omega,
                            qpos_seq - joint_pos_array_seq,
                            qvel_seq,
                            action_buffer,
                        ]
                        obs_array = np.concatenate(obs_list).astype(np.float32)
                        obs[:len(obs_array)] = obs_array

                        # 维度对齐
                        target_dim = input_shape[1]
                        if obs.shape[0] < target_dim:
                            obs = np.pad(obs, (0, target_dim - obs.shape[0]))
                        elif obs.shape[0] > target_dim:
                            obs = obs[:target_dim]

                        # ONNX 推理: actions + 下步参考轨迹
                        obs_tensor = np.expand_dims(obs, axis=0)
                        model_out = session.run(
                            ["actions", "joint_pos", "joint_vel", "body_quat_w"],
                            {obs_name: obs_tensor,
                             ts_name: np.array([[float(idx)]], dtype=np.float32)},
                        )

                        # 策略输出 → 目标关节位置
                        action_array = np.asarray(model_out[0]).reshape(-1)
                        action_buffer = action_array.copy()
                        target_dof_seq = action_array * action_scale + joint_pos_array_seq
                        target_dof_pos = np.array(
                            [target_dof_seq[joint_seq.index(j)] for j in joint_xml]
                        )

                        # 缓存参考轨迹供下一步观测使用
                        motion_command_t = np.concatenate([
                            np.asarray(model_out[1], dtype=np.float32).reshape(1, -1),
                            np.asarray(model_out[2], dtype=np.float32).reshape(1, -1),
                        ], axis=1)
                        motion_ref_quat_t = np.asarray(
                            model_out[3], dtype=np.float64
                        )[0, motion_body_idx, :].copy()

                        # 周期性状态打印，打印的是 ONNX 输入的 command 范围和动作幅度
                        if timestep % PRINT_INTERVAL == 0:
                            print(f"[t={timestep}] cmd range: [{cmd.min():.2f}, {cmd.max():.2f}]  "
                                  f"|action| max: {np.max(np.abs(action_array)):.3f}")

                        timestep += 1
                    else:
                        sim_state["state"] = STATE_DONE
                        cur_kp = stand_kp.copy()
                        cur_kd = stand_kd.copy()
                        print("[INFO] → DONE (播放完毕)")

            elif sim_state["state"] == STATE_IDLE:
                # 保持初始站立姿态
                if counter % CONTROL_DECIMATION == 0:
                    target_dof_pos = initial_target.copy()

            # ---- PD 控制 + 物理步进 ----
            if np.any(cur_kp > 0):
                tau = pd_control(
                    target_dof_pos, d.qpos[7:7 + len(joint_xml)],
                    cur_kp, np.zeros(len(joint_xml)),
                    d.qvel[6:6 + len(joint_xml)], cur_kd,
                )
                if m.nu == len(joint_xml):
                    d.ctrl[:] = tau
                    d.qfrc_applied[6:] = 0.0
                else:
                    d.qfrc_applied[:] = 0.0
                    d.qfrc_applied[6:6 + len(joint_xml)] = tau
            else:
                if m.nu == len(joint_xml):
                    d.ctrl[:] = target_dof_pos

            # ---- 弹性绳（笛卡尔力，比 qfrc_applied 更稳定） ----
            if elastic_band is not None and elastic_band.enable:
                rope_force = elastic_band.advance(
                    d.xpos[band_attached_body], d.qvel[0:3]
                )
                d.xfrc_applied[band_attached_body, :3] = rope_force

            mujoco.mj_step(m, d)
            counter += 1
            viewer.sync()

            # 实时同步
            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    run()
