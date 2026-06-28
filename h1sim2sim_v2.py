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
    python h1sim2sim_v2.py                        # ONNX 自举模式（默认）
    python h1sim2sim_v2.py --config my.py          # 使用自定义配置
    python h1sim2sim_v2.py --motion_file xxx.npz  # .npz 实机模式（与部署完全一致）

键盘控制:
    6 = 重置站立   7 = 播放策略   8 = 切换弹性绳
    MuJoCo 窗口中 BACKSPACE = 重置仿真
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime

# ============================================================================
# 加载配置 (--config / --motion_file)
# ============================================================================

# --motion_file: 可选，传入 .npz 时切换到实机模式（观测与 h1_deploy_real.py 完全一致）
NPZ_MODE = False
MOTION_FILE = None
_motion_idx = None
for _i, _arg in enumerate(sys.argv):
    if _arg == "--motion_file" and _i + 1 < len(sys.argv):
        MOTION_FILE = sys.argv[_i + 1]
        NPZ_MODE = True
        _motion_idx = _i
        break

# --config: 可选，自定义仿真配置文件
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

# .npz 模式下导入观测构造共享模块
if NPZ_MODE:
    sys.path.insert(0, "deploy_real")
    from observation_builder import (
        build_observation,
        remap_xml_to_seq,
        action_to_target,
        compute_init_to_world,
    )

# 手柄模式: python h1sim2sim_v2.py --gamepad
USE_GAMEPAD = "--gamepad" in sys.argv

# ============================================================================
# 手柄读取器（Linux /dev/input/js0）
# ============================================================================

class GamepadReader:
    """Linux 游戏手柄读取器。非阻塞后台线程读取 /dev/input/js0。

    按键映射（Xbox 布局）:
      A=0  B=1  X=2  Y=3  LB=4  RB=5  BACK=6  START=7
    """

    BUTTON_A = 0; BUTTON_B = 1; BUTTON_X = 2; BUTTON_Y = 3
    _EVENT_FORMAT = "IhBB"; _EVENT_SIZE = 8

    def __init__(self, device="/dev/input/js0"):
        self._lock = threading.Lock()
        self._buttons = {}
        self._running = False
        self._thread = None
        self._fd = None
        self._connected = False
        try:
            self._fd = open(device, "rb", buffering=0)
            self._connected = True
        except (FileNotFoundError, PermissionError, OSError):
            print(f"[Gamepad] ⚠ 未检测到手柄 ({device})，回退键盘模式")

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        if not self._connected or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[Gamepad] ✅ 手柄已连接 (A=播放 X=锁定 Y=弹性绳)")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._fd:
            try: self._fd.close()
            except OSError: pass

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
            except (OSError, BlockingIOError):
                time.sleep(0.001)
            except struct.error:
                break

    def rising(self, btn_id):
        """上升沿（按下瞬间为 True，持续按住为 False）"""
        with self._lock:
            cur = self._buttons.get(btn_id, 0) != 0
            prev = getattr(self, f"_btn_{btn_id}", False)
            setattr(self, f"_btn_{btn_id}", cur)
            return cur and not prev

    def held(self, btn_id):
        with self._lock:
            return self._buttons.get(btn_id, 0) != 0


# ============================================================================
# 状态机常量
# ============================================================================

STATE_IDLE = 0       # 待机：保持默认站立姿态
STATE_PLAYING = 1    # 播放：策略推理中
STATE_DONE = 2       # 完成：从最后姿态平滑过渡到默认站立
STATE_HOLD = 3       # 锁定：PD Hold 当前位置（对齐实机 hold_current_position）
STATE_BLEND = 4      # 过渡：1s 站立→舞蹈首帧平滑渐变（对齐实机 blend_to_first_action）
STATE_WAIT = 5       # 等待（已废弃，BLEND 结束后直接进入 PLAYING）

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

    # ------------------------------------------------------------------
    # 3b. [.npz 模式] 加载运动参考数据
    # ------------------------------------------------------------------
    npz_motion = None
    npz_num_frames = 0
    npz_cmd_cache = None       # (N, 38) 预拼接
    npz_quat_cache = None      # (N, 4) 骨盆四元数
    if NPZ_MODE:
        if not os.path.isfile(MOTION_FILE):
            raise FileNotFoundError(f".npz 文件未找到: {MOTION_FILE}")
        npz_motion = np.load(MOTION_FILE)
        npz_num_frames = min(
            npz_motion["joint_pos"].shape[0],
            npz_motion["joint_vel"].shape[0],
            npz_motion["body_pos_w"].shape[0],
            npz_motion["body_quat_w"].shape[0],
        )
        npz_cmd_cache = np.concatenate([
            npz_motion["joint_pos"][:, :num_actions],
            npz_motion["joint_vel"][:, :num_actions],
        ], axis=1).astype(np.float32)
        npz_quat_cache = npz_motion["body_quat_w"][:, 0, :].astype(np.float64)
        # 从轨迹末尾提取站立姿态
        n_hold = min(20, npz_num_frames)
        hold_target_seq = np.mean(npz_motion["joint_pos"][-n_hold:, :num_actions], axis=0)  # joint_seq 顺序
        hold_target = np.array([hold_target_seq[joint_seq.index(j)] for j in joint_xml])     # XML 顺序
        print(f"[INFO] .npz 模式: {npz_num_frames} 帧, motion_file={MOTION_FILE}")
        print(f"[INFO] 实机模式: 观测构造与 h1_deploy_real.py 完全一致")
        print(f"[INFO] 锁定目标: 轨迹最后 {n_hold} 帧关节位置均值 (XML顺序)")

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
    # 5. ONNX 自举引导（仅非 .npz 模式）
    # ------------------------------------------------------------------
    policy_input_names = [inp.name for inp in session.get_inputs()]
    obs_name = policy_input_names[0]      # "obs"
    ts_name = policy_input_names[1]       # "time_step"

    if not NPZ_MODE:
        bootstrap_obs = np.zeros((1, input_shape[1]), dtype=np.float32)
        bootstrap_ts = np.array([[0.0]], dtype=np.float32)
        bootstrap_out = session.run(
            ["joint_pos", "joint_vel", "body_quat_w"],
            {obs_name: bootstrap_obs, ts_name: bootstrap_ts},
        )
        motion_command_t = np.concatenate([
            np.asarray(bootstrap_out[0], dtype=np.float32).reshape(1, -1),
            np.asarray(bootstrap_out[1], dtype=np.float32).reshape(1, -1),
        ], axis=1)
        motion_ref_quat_t = np.asarray(
            bootstrap_out[2], dtype=np.float64
        )[0, motion_body_idx, :].copy()
        print(f"[INFO] ONNX 自举完成 — motion_command_t: {motion_command_t.shape}")
    else:
        motion_command_t = np.zeros((1, num_actions * 2), dtype=np.float32)
        motion_ref_quat_t = np.zeros(4, dtype=np.float64)

    # ------------------------------------------------------------------
    # 6. 初始化仿真状态
    # ------------------------------------------------------------------
    num_frames = npz_num_frames if NPZ_MODE else int(
        SIMULATION_DURATION / (CONTROL_DECIMATION * SIMULATION_DT))
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    action_buffer = np.zeros(num_actions, dtype=np.float32)
    counter = 0
    timestep = 0
    v = np.zeros(3, dtype=np.float64)
    omega = np.zeros(3, dtype=np.float64)
    init_to_world = np.eye(3, dtype=np.float64)  # .npz 模式 yaw 对齐矩阵

    # ------------------------------------------------------------------
    # 7. 弹性绳 & 键盘状态
    # ------------------------------------------------------------------
    sim_state = {"state": STATE_HOLD, "key_pressed": None}
    initial_target = joint_pos_array.copy()
    # 锁定目标：从轨迹末尾提取（NPZ 模式取最后20帧均值，ONNX 模式运行时更新）
    hold_frames = min(20, num_frames)
    hold_target = initial_target.copy()  # 默认值，NPZ 模式会被覆盖
    hold_buf = np.zeros((hold_frames, num_actions), dtype=np.float64)
    hold_buf_idx = 0
    # BLEND 过渡（1s 站立→首帧，对齐实机 blend_to_first_action）
    blend_steps = int(1.0 / (CONTROL_DECIMATION * SIMULATION_DT))
    blend_step = 0
    blend_start_pos = None
    blend_first_target = None  # 首帧目标，供 STATE_WAIT 保持姿态
    # DONE 过渡（3s 缓慢挪到锁定姿态，不改变增益）
    done_transition_steps = int(3.0 / (CONTROL_DECIMATION * SIMULATION_DT))
    done_step = 0
    done_start_target = None

    rope_anchor = np.array([0.0, 0.0, float(INIT_PELVIS_Z) + 1.0])
    elastic_band = ElasticBand(
        point=rope_anchor,
        stiffness=_cfg.ELASTIC_BAND_STIFFNESS,
        damping=_cfg.ELASTIC_BAND_DAMPING,
    )
    elastic_band.enable = ELASTIC_BAND_ENABLED
    band_attached_body = m.body("torso_link").id

    # 手柄
    gamepad = GamepadReader() if USE_GAMEPAD else None
    if gamepad is not None and gamepad.connected:
        gamepad.start()

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
        print("[INFO] → IDLE (默认站立姿态)")
        return _target

    def hold_current_position():
        """PD Hold 锁定当前关节位置（对齐实机 hold_current_position）。

        与 reset_to_stand 不同：
          - 不移动机器人，不修改 qpos
          - 以当前位姿为目标，用 1.5x 策略增益锁住
        """
        nonlocal cur_kp, cur_kd
        cur_kp = stand_kp.copy()
        cur_kd = stand_kd.copy()
        if elastic_band is not None:
            elastic_band.enable = True
        _target = d.qpos[7:7 + len(joint_xml)].copy()
        print("[INFO] → HOLD (PD 锁定当前位置，等待指令)")
        return _target

    def start_playback():
        """开始单次策略播放。.npz 模式重置 yaw 对齐；ONNX 模式重新自举。"""
        nonlocal timestep, counter, action_buffer, cur_kp, cur_kd
        nonlocal motion_command_t, motion_ref_quat_t, init_to_world
        nonlocal blend_start_pos, blend_step, blend_first_target
        _target = initial_target.copy()
        timestep = 0
        counter = 0
        cur_kp = policy_kp
        cur_kd = policy_kd
        action_buffer = np.zeros(num_actions, dtype=np.float32)
        obs[:] = 0.0
        init_to_world = np.eye(3, dtype=np.float64)
        if NPZ_MODE:
            motion_command_t = npz_cmd_cache[0:1, :].copy()
            motion_ref_quat_t = npz_quat_cache[0, :].copy()
            print(f"[INFO] → BLEND ({num_frames} 步, .npz 实机模式)")
        else:
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
            print(f"[INFO] → BLEND ({num_frames} 步, ONNX 自举模式)")
        blend_start_pos = d.qpos[7:7 + len(joint_xml)].copy()
        blend_step = 0
        return _target

    # ------------------------------------------------------------------
    # 8. 设置初始姿态（PD Hold 当前位置，对齐实机 hold_current_position）
    # ------------------------------------------------------------------
    target_dof_pos = hold_current_position()
    d.qpos[2] = INIT_PELVIS_Z
    d.qvel[:] = 0.0
    if m.nu > 0:
        d.ctrl[:] = 0.0
    d.qfrc_applied[:] = 0.0
    mujoco.mj_forward(m, d)

    if USE_GAMEPAD and gamepad is not None and gamepad.connected:
        print("[INFO] 🎮 手柄模式: A=站立  B=播放  X=锁定  Y=弹性绳")
    else:
        print("[INFO] ⌨ 键盘模式: 5=锁定  6=站立  7=播放  8=弹性绳")
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

            # ---- 键盘 / 手柄输入 ----
            if USE_GAMEPAD and gamepad is not None and gamepad.connected:
                # 手柄模式: A=站立 B=播放 X=锁定 Y=弹性绳
                if gamepad.rising(GamepadReader.BUTTON_Y):
                    elastic_band.enable = not elastic_band.enable
                    print(f"[弹力绳] {'拉紧' if elastic_band.enable else '解开'}")
                elif gamepad.rising(GamepadReader.BUTTON_X):
                    target_dof_pos = hold_current_position()
                    sim_state["state"] = STATE_HOLD
                elif gamepad.rising(GamepadReader.BUTTON_A):
                    target_dof_pos = reset_to_stand()
                    sim_state["state"] = STATE_IDLE
                elif gamepad.rising(GamepadReader.BUTTON_B):
                    target_dof_pos = start_playback()
                    sim_state["state"] = STATE_BLEND
            else:
                # 键盘模式
                key = sim_state["key_pressed"]
                if key is not None:
                    sim_state["key_pressed"] = None
                    glfw = mujoco.glfw.glfw
                    if key == glfw.KEY_5:
                        target_dof_pos = hold_current_position()
                        sim_state["state"] = STATE_HOLD
                    elif key == glfw.KEY_6:
                        target_dof_pos = reset_to_stand()
                        sim_state["state"] = STATE_IDLE
                    elif key == glfw.KEY_7:
                        target_dof_pos = start_playback()
                        sim_state["state"] = STATE_BLEND
                    elif key == glfw.KEY_8:
                        elastic_band.enable = not elastic_band.enable
                        print(f"[弹力绳] {'拉紧' if elastic_band.enable else '解开'}")

            # ---- BLEND 过渡（1s 站立→首帧） ----
            if sim_state["state"] == STATE_BLEND:
                if counter % CONTROL_DECIMATION == 0 and blend_start_pos is not None:
                    if blend_step < blend_steps:
                        alpha = float(blend_step) / float(blend_steps)
                        # 运行首帧推理获取目标
                        _, _, _, _, omega, _, _ = get_obs(d)
                        qpos_xml = d.qpos[7:7 + len(joint_xml)]
                        qpos_seq = np.array([qpos_xml[joint_xml.index(j)] for j in joint_seq])
                        qvel_xml = d.qvel[6:6 + len(joint_xml)]
                        qvel_seq = np.array([qvel_xml[joint_xml.index(j)] for j in joint_seq])
                        pelvis_quat = d.xquat[body_id].copy()
                        if NPZ_MODE:
                            cmd = npz_cmd_cache[0, :].copy()
                            obs = build_observation(cmd=cmd, pelvis_quat=pelvis_quat,
                                motion_ref_quat=npz_quat_cache[0, :], pelvis_ang_vel=omega,
                                qpos_seq=qpos_seq, qvel_seq=qvel_seq, action_buffer=action_buffer,
                                default_angles_seq=joint_pos_array_seq,
                                init_to_world=np.eye(3), num_obs=NUM_OBS)
                        else:
                            cmd = motion_command_t.reshape(-1)[:num_actions * 2].copy()
                            anchor_quat_b = quat_mul_np(quat_inv_np(pelvis_quat), motion_ref_quat_t)
                            anchor_ori_b = matrix_from_quat(anchor_quat_b)[..., :2].reshape(-1).astype(np.float32)
                            obs_list = [cmd, np.zeros(3), anchor_ori_b, np.zeros(3), omega,
                                        qpos_seq - joint_pos_array_seq, qvel_seq, action_buffer]
                            obs = np.concatenate(obs_list).astype(np.float32)[:NUM_OBS]
                        obs_tensor = np.expand_dims(obs, axis=0)
                        model_out = session.run(
                            ["actions", "joint_pos", "joint_vel", "body_quat_w"],
                            {obs_name: obs_tensor, ts_name: np.array([[0.0]], dtype=np.float32)})
                        action_array = np.asarray(model_out[0]).reshape(-1)
                        if NPZ_MODE:
                            first_target = action_to_target(action_array, joint_pos_array_seq, action_scale, joint_seq, joint_xml)
                        else:
                            first_target = np.array([(action_array * action_scale + joint_pos_array_seq)[joint_seq.index(j)] for j in joint_xml])
                        # 混合目标 + 混合增益
                        target_dof_pos = (1.0 - alpha) * blend_start_pos + alpha * first_target
                        for i in range(len(cur_kp)):
                            cur_kp[i] = stand_kp[i] + (policy_kp[i] - stand_kp[i]) * alpha
                            cur_kd[i] = stand_kd[i] + (policy_kd[i] - stand_kd[i]) * alpha
                        if blend_step == blend_steps - 1:
                            blend_first_target = first_target.copy()
                        blend_step += 1
                    else:
                        sim_state["state"] = STATE_PLAYING
                        print("[INFO] → PLAYING (首帧就位，直接开始舞蹈)")

            # ---- WAIT 状态（已废弃，直接进入 PLAYING） ----
            elif sim_state["state"] == STATE_WAIT:
                sim_state["state"] = STATE_PLAYING
                if counter % CONTROL_DECIMATION == 0 and blend_first_target is not None:
                    target_dof_pos = blend_first_target.copy()

            # ---- 策略推理（每 CONTROL_DECIMATION 物理步执行一次）----
            elif sim_state["state"] == STATE_PLAYING:
                if counter % CONTROL_DECIMATION == 0:
                    if timestep < num_frames:
                        idx = timestep

                        # 关节状态: XML 顺序 → ONNX 顺序
                        qpos_xml = d.qpos[7:7 + len(joint_xml)]
                        qpos_seq = np.array([qpos_xml[joint_xml.index(j)] for j in joint_seq])
                        qvel_xml = d.qvel[6:6 + len(joint_xml)]
                        qvel_seq = np.array([qvel_xml[joint_xml.index(j)] for j in joint_seq])

                        # 机器人骨盆朝向 (MuJoCo world frame)
                        robot_pelvis_quat = d.xquat[body_id].copy()

                        if NPZ_MODE:
                            # ---- .npz 实机模式：观测构造与 h1_deploy_real.py 完全一致 ----
                            # Yaw 对齐初始化
                            if timestep < 2:
                                ref_q = npz_quat_cache[timestep, :]
                                init_to_world = compute_init_to_world(robot_pelvis_quat, ref_q)
                            # cmd 来自 .npz，pre_blend 时混合到锁定姿态
                            cmd = npz_cmd_cache[timestep, :].copy()
                            if timestep >= num_frames - 20:
                                alpha = float(timestep - (num_frames - 20)) / 20.0
                                alpha = min(alpha, 1.0)
                                cmd[:num_actions] = (1.0 - alpha) * cmd[:num_actions] + alpha * hold_target_seq
                                cmd[num_actions:] *= (1.0 - alpha)  # 速度 fade to 0
                            ref_quat = npz_quat_cache[timestep, :]
                            # 使用共享模块构造观测
                            obs = build_observation(
                                cmd=cmd,
                                pelvis_quat=robot_pelvis_quat,
                                motion_ref_quat=ref_quat,
                                pelvis_ang_vel=omega,
                                qpos_seq=qpos_seq,
                                qvel_seq=qvel_seq,
                                action_buffer=action_buffer,
                                default_angles_seq=joint_pos_array_seq,
                                init_to_world=init_to_world,
                                num_obs=NUM_OBS,
                            )
                        else:
                            # ---- ONNX 自举模式 ----
                            motioninput = motion_command_t.reshape(-1)
                            cmd = motioninput[:num_actions * 2].copy()
                            if timestep >= num_frames - 20:
                                alpha = float(timestep - (num_frames - 20)) / 20.0
                                alpha = min(alpha, 1.0)
                                cmd[:num_actions] = (1.0 - alpha) * cmd[:num_actions] + alpha * hold_target_seq
                                cmd[num_actions:] *= (1.0 - alpha)
                            anchor_quat_b = quat_mul_np(
                                quat_inv_np(robot_pelvis_quat), motion_ref_quat_t
                            )
                            mat = matrix_from_quat(anchor_quat_b)
                            anchor_ori_b = mat[..., :2].reshape(-1).astype(np.float32)
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
                        if NPZ_MODE:
                            target_dof_pos = action_to_target(
                                action_array, joint_pos_array_seq,
                                action_scale, joint_seq, joint_xml,
                            )
                        else:
                            target_dof_seq = action_array * action_scale + joint_pos_array_seq
                            target_dof_pos = np.array(
                                [target_dof_seq[joint_seq.index(j)] for j in joint_xml]
                            )

                        # 缓存参考轨迹供下一步观测使用（仅 ONNX 自举模式需要）
                        if not NPZ_MODE:
                            motion_command_t = np.concatenate([
                                np.asarray(model_out[1], dtype=np.float32).reshape(1, -1),
                                np.asarray(model_out[2], dtype=np.float32).reshape(1, -1),
                            ], axis=1)
                            motion_ref_quat_t = np.asarray(
                                model_out[3], dtype=np.float64
                            )[0, motion_body_idx, :].copy()
                            # 更新锁定目标缓冲区
                            hold_buf[hold_buf_idx % hold_frames, :] = np.asarray(model_out[1]).reshape(-1)[:num_actions]
                            hold_buf_idx += 1
                            # 最后 10 帧时更新 hold_target (ONNX输出是joint_seq顺序 → 转XML)
                            if timestep >= num_frames - 10:
                                seq_mean = np.mean(hold_buf, axis=0)
                                hold_target = np.array([seq_mean[joint_seq.index(j)] for j in joint_xml])

                        # 周期性状态打印
                        if timestep % PRINT_INTERVAL == 0:
                            tag = " [PRE-BLEND]" if timestep >= num_frames - 20 else ""
                            print(f"[t={timestep}{tag}] cmd range: [{cmd.min():.2f}, {cmd.max():.2f}]  "
                                  f"|action| max: {np.max(np.abs(action_array)):.3f}")

                        timestep += 1
                    else:
                        # 目标已渐进到锁定姿态，增益保持不变（策略增益在舞蹈中已验证有效）
                        sim_state["state"] = STATE_DONE
                        done_start_target = target_dof_pos.copy()
                        done_step = 0
                        print(f"[INFO] → DONE (锁定姿态: {np.mean(hold_target):.2f})")

            elif sim_state["state"] in (STATE_IDLE, STATE_HOLD):
                # 保持当前 target_dof_pos 不变
                # STATE_IDLE: 由 reset_to_stand() 设为默认站立姿态
                # STATE_HOLD: 由 hold_current_position() 设为锁定时的关节位置
                pass

            elif sim_state["state"] == STATE_DONE:
                if counter % CONTROL_DECIMATION == 0 and done_start_target is not None:
                    if done_step < done_transition_steps:
                        alpha = float(done_step) / float(done_transition_steps)
                        target_dof_pos = (1.0 - alpha) * done_start_target + alpha * hold_target
                        done_step += 1
                    else:
                        target_dof_pos = hold_target.copy()

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
