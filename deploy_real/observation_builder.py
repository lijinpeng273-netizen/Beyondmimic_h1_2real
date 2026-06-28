"""H1 观测构造与动作映射共享模块。

sim2sim (h1sim2sim_v2.py) 和实机部署 (h1_deploy_real.py) 均导入此模块，
确保两边的观测构造、关节重排、动作还原逻辑完全一致。

观测布局 (110 维):
    [0:38]   cmd — 参考关节位置+速度 (19+19)，来自 .npz 或 ONNX 自举
    [38:41]  anchor_pos_b — 锚点相对位置 (实机=0)
    [41:47]  anchor_ori_b — 锚点相对朝向 6D (quat_inv(robot)*ref_quat)
    [47:50]  base_lin_vel — 基座线速度 (实机=0)
    [50:53]  base_ang_vel — 基座角速度 (IMU gyro, 骨盆坐标系)
    [53:72]  joint_pos — 当前关节位置 - default_angles_seq
    [72:91]  joint_vel — 当前关节速度
    [91:110] last_action — 上一步策略输出
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# 四元数工具（纯 NumPy，无外部依赖）
# ============================================================================

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数 Hamilton 积 (w,x,y,z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inv(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return quat_conjugate(q) / np.clip(np.sum(q ** 2), a_min=eps, a_max=None)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵"""
    w, x, y, z = q
    two_s = 2.0 / (w*w + x*x + y*y + z*z)
    return np.array([
        [1 - two_s*(y*y + z*z), two_s*(x*y - z*w), two_s*(x*z + y*w)],
        [two_s*(x*y + z*w), 1 - two_s*(x*x + z*z), two_s*(y*z - x*w)],
        [two_s*(x*z - y*w), two_s*(y*z + x*w), 1 - two_s*(x*x + y*y)],
    ])


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """提取四元数的 yaw 分量 → 纯 yaw 四元数"""
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)])


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 → 四元数 [w,x,y,z]"""
    m = np.array(matrix, dtype=np.float64)
    m00, m01, m02 = m[0]; m10, m11, m12 = m[1]; m20, m21, m22 = m[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([0.25/s, (m21-m12)*s, (m02-m20)*s, (m10-m01)*s])
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        return np.array([(m21-m12)/s, 0.25*s, (m01+m10)/s, (m02+m20)/s])
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        return np.array([(m02-m20)/s, (m01+m10)/s, 0.25*s, (m12+m21)/s])
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        return np.array([(m10-m01)/s, (m02+m20)/s, (m12+m21)/s, 0.25*s])


# ============================================================================
# 观测构造
# ============================================================================

def compute_anchor_ori_b(
    pelvis_quat: np.ndarray,
    ref_quat: np.ndarray,
    init_to_world: np.ndarray | None = None,
) -> np.ndarray:
    """计算锚点相对朝向 6D 表示。

    与 h1sim2sim_v2.py 的 anchor_ori_b 计算完全一致:
        anchor_quat_b = quat_inv(pelvis) * (init_to_world @ ref_quat)

    参数:
        pelvis_quat:   机器人骨盆四元数 (w,x,y,z)，IMU 经 torso→pelvis 变换后
        ref_quat:      参考运动锚点四元数 (w,x,y,z)，来自 .npz 或 ONNX 输出
        init_to_world: yaw 对齐矩阵 (3,3)，用于对齐 mocap/ONNX 世界帧与机器人世界帧
                       仿真中为 identity，实机中为前2帧计算的 yaw 对齐

    返回:
        anchor_ori_b: (6,) float32，旋转矩阵前两列展平
    """
    if init_to_world is None:
        init_to_world = np.eye(3, dtype=np.float64)

    aligned_quat = quat_mul(matrix_to_quat(init_to_world), ref_quat)
    rel_quat = quat_mul(quat_inv(pelvis_quat), aligned_quat)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化防止漂移
    rot_mat = matrix_from_quat(rel_quat)
    return rot_mat[:, :2].reshape(-1).astype(np.float32)


def build_observation(
    cmd: np.ndarray,
    pelvis_quat: np.ndarray,
    motion_ref_quat: np.ndarray,
    pelvis_ang_vel: np.ndarray,
    qpos_seq: np.ndarray,
    qvel_seq: np.ndarray,
    action_buffer: np.ndarray,
    default_angles_seq: np.ndarray,
    init_to_world: np.ndarray | None = None,
    *,
    num_obs: int = 110,
) -> np.ndarray:
    """构造 110 维观测向量。

    此函数被 sim2sim 和实机部署共用，确保观测构造完全一致。

    参数:
        cmd:              参考关节位置+速度 (38,) — .npz 或 ONNX bootstrap
        pelvis_quat:      机器人骨盆四元数 (4,) (w,x,y,z)
        motion_ref_quat:  运动参考锚点四元数 (4,) (w,x,y,z)
        pelvis_ang_vel:   骨盆角速度 (3,) — IMU gyro 经 torso→pelvis 变换
        qpos_seq:         当前关节位置 (19,) — ONNX joint_seq 顺序
        qvel_seq:         当前关节速度 (19,) — ONNX joint_seq 顺序
        action_buffer:    上一步策略输出 (19,)
        default_angles_seq: 策略默认关节角度 (19,) — ONNX joint_seq 顺序
        init_to_world:    yaw 对齐矩阵 (3,3)，默认 identity
        num_obs:          观测总维度，默认 110

    返回:
        obs: (num_obs,) float32
    """
    anchor_ori_b = compute_anchor_ori_b(pelvis_quat, motion_ref_quat, init_to_world)

    num_actions = len(action_buffer)
    obs_list = [
        cmd.astype(np.float32),                                  # [0:38]
        np.zeros(3, dtype=np.float32),                           # [38:41] anchor_pos=0
        anchor_ori_b,                                            # [41:47] anchor_ori
        np.zeros(3, dtype=np.float32),                           # [47:50] lin_vel=0
        pelvis_ang_vel.reshape(-1).astype(np.float32)[:3],       # [50:53] ang_vel
        (qpos_seq - default_angles_seq).astype(np.float32),      # [53:72] joint_pos
        qvel_seq.astype(np.float32),                             # [72:91] joint_vel
        action_buffer.astype(np.float32),                        # [91:110] last_action
    ]
    obs = np.concatenate(obs_list).astype(np.float32)

    # 维度对齐（兼容不同 num_obs 配置）
    if obs.shape[0] < num_obs:
        obs = np.pad(obs, (0, num_obs - obs.shape[0]))
    elif obs.shape[0] > num_obs:
        obs = obs[:num_obs]
    return obs


# ============================================================================
# 关节映射
# ============================================================================

def remap_xml_to_seq(values_xml: np.ndarray, joint_xml: list[str],
                     joint_seq: list[str]) -> np.ndarray:
    """XML/DOF 顺序 → ONNX joint_seq 顺序重排。"""
    return np.array([values_xml[joint_xml.index(j)] for j in joint_seq])


def remap_seq_to_xml(values_seq: np.ndarray, joint_seq: list[str],
                     joint_xml: list[str]) -> np.ndarray:
    """ONNX joint_seq 顺序 → XML/DOF 顺序重排。"""
    return np.array([values_seq[joint_seq.index(j)] for j in joint_xml])


# ============================================================================
# 动作 → 目标位置
# ============================================================================

def action_to_target(
    action: np.ndarray,
    default_angles_seq: np.ndarray,
    action_scale_seq: np.ndarray,
    joint_seq: list[str],
    joint_xml: list[str],
) -> np.ndarray:
    """策略输出动作 → XML/DOF 顺序的目标关节位置。

    参数:
        action:             策略原始输出 (19,) — ONNX joint_seq 顺序
        default_angles_seq: 默认关节角度 (19,) — ONNX joint_seq 顺序
        action_scale_seq:   动作缩放因子 (19,) — ONNX joint_seq 顺序
        joint_seq:          ONNX 关节名列表
        joint_xml:          XML/DOF 关节名列表

    返回:
        target_dof_xml: 目标关节位置 (19,) — XML/DOF 顺序
    """
    # ONNX 顺序: target = default + action * scale
    target_seq = default_angles_seq + action.reshape(-1) * action_scale_seq
    # ONNX → XML 重排
    return remap_seq_to_xml(target_seq, joint_seq, joint_xml)


# ============================================================================
# Yaw 对齐初始化
# ============================================================================

def compute_init_to_world(
    pelvis_quat: np.ndarray,
    motion_ref_quat: np.ndarray,
) -> np.ndarray:
    """计算 yaw 对齐矩阵。

    将对齐 mocap 世界帧的 yaw 到机器人启动时骨盆的 yaw。

    参数:
        pelvis_quat:     机器人骨盆四元数 (w,x,y,z)
        motion_ref_quat: 第一帧运动参考四元数 (w,x,y,z)

    返回:
        init_to_world: (3,3) 旋转矩阵
    """
    yaw_mot = yaw_quat(motion_ref_quat)
    yaw_mot_mat = matrix_from_quat(yaw_mot)
    yaw_rob = yaw_quat(pelvis_quat)
    yaw_rob_mat = matrix_from_quat(yaw_rob)
    return yaw_rob_mat @ yaw_mot_mat.T
