#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hg_mujoco_dds_bridge.py — 10-DOF MuJoCo + DDS 桥接端（Sim2Real 联调用）
=======================================================================
配合 hg_policy_controller.py 使用：

  终端 1: python hg_mujoco_dds_bridge.py   # 物理仿真 + DDS 发布
  终端 2: python hg_policy_controller.py    # 策略推理 + DDS 订阅

数据流:
  hg_mujoco_dds_bridge.py  ←rt/lowcmd←  hg_policy_controller.py
  hg_mujoco_dds_bridge.py  →rt/lowstate→ hg_policy_controller.py

相当于在仿真中模拟实机 H1 的 DDS 通信行为，使 hg_policy_controller.py
可以在不上实机的情况下完成完整联调。

===== 关节映射说明 =====
10-DOF h1.xml 只有 10 个腿部执行器，而 DDS 协议固定 20 路电机。
桥接层负责在两种格式之间正确映射：

  DDS 索引 (0-19)         h1.xml 执行器 (0-9)
  ─────────────────────────────────────────
  0: R_hip_roll         ←  h1_act[6] (right_hip_roll)
  1: R_hip_pitch        ←  h1_act[7] (right_hip_pitch)
  2: R_knee             ←  h1_act[8] (right_knee)
  3: L_hip_roll         ←  h1_act[1] (left_hip_roll)
  4: L_hip_pitch        ←  h1_act[2] (left_hip_pitch)
  5: L_knee             ←  h1_act[3] (left_knee)
  6: torso              ←  无对应
  7: L_hip_yaw          ←  h1_act[0] (left_hip_yaw)
  8: R_hip_yaw          ←  h1_act[5] (right_hip_yaw)
  9: not_use            ←  无对应
  10: L_ankle           ←  h1_act[4] (left_ankle)
  11: R_ankle           ←  h1_act[9] (right_ankle)
  12-19: 手臂           ←  无对应
"""

import threading
import time
import os
import numpy as np
import mujoco
import mujoco.viewer

from hg_config import (
    HG_MJCF_PATH,
    DOMAIN_ID, INTERFACE,
    SIMULATE_DT, VIEWER_DT, DECIMATION, POLICY_DT,
    HG_NUM_ACTION, HG_JOINT_NAMES, HG_DEFAULT_DOF_POS,
    HG_KPS, HG_KDS, HG_KP_SCALE, HG_KD_SCALE,
    ENABLE_ELASTIC_BAND, ROBOT,
)

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_


# =====================================================================
# DDS 电机映射（与 hg_policy_controller.py 完全一致）
# =====================================================================
NUM_DDS_MOTORS = 20

# HG 10-DOF 训练顺序 → DDS 20 路电机索引
HG_TO_MJC = [7, 3, 4, 5, 10, 8, 0, 1, 2, 11]

# DDS 索引 → h1.xml 执行器索引
# 10-DOF h1.xml 的执行器顺序与 HG_JOINT_NAMES 一致
DDS_TO_H1_ACT = [-1] * NUM_DDS_MOTORS
for _h1_i, _dds_i in enumerate(HG_TO_MJC):
    DDS_TO_H1_ACT[_dds_i] = _h1_i

# h1.xml 执行器索引 → DDS 索引
H1_ACT_TO_DDS = list(HG_TO_MJC)


# =====================================================================
# 工具函数
# =====================================================================

def get_joint_addresses(mj_model):
    """按名称查询 10 个关节在 qpos/qvel 中的索引。"""
    qpos_adrs, qvel_adrs = [], []
    for name in HG_JOINT_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"关节在 MuJoCo 模型中未找到: {name}")
        qpos_adrs.append(mj_model.jnt_qposadr[jid])
        qvel_adrs.append(mj_model.jnt_dofadr[jid])
    return np.array(qpos_adrs), np.array(qvel_adrs)


# =====================================================================
# 加载 MuJoCo 模型
# =====================================================================
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MJCF_PATH = os.path.normpath(os.path.join(_SRC_DIR, HG_MJCF_PATH))

if not os.path.isfile(_MJCF_PATH):
    raise FileNotFoundError(f"MJCF 模型未找到: {_MJCF_PATH}")

mj_model = mujoco.MjModel.from_xml_path(_MJCF_PATH)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = SIMULATE_DT

# 解析关节地址
QPOS_ADRS, QVEL_ADRS = get_joint_addresses(mj_model)
print(f"[HGBridge] 关节地址: qpos={QPOS_ADRS.tolist()}, qvel={QVEL_ADRS.tolist()}")

# 设置初始站立姿态（与 sim2sim 的 HG10PolicyBridge 使用相同的初始状态）
mj_data.qpos[:] = 0.0
mj_data.qpos[2] = 1.0          # 骨盆高度 (m)
mj_data.qpos[3] = 1.0          # 四元数 w (单位四元数)
mj_data.qpos[QPOS_ADRS] = HG_DEFAULT_DOF_POS  # 关节角
mj_data.qvel[:] = 0.0
mujoco.mj_forward(mj_model, mj_data)
_base_z_init = mj_data.qpos[2]
print(f"[HGBridge] 初始基底高度: z={_base_z_init:.3f}")

# 线程锁（保护 mj_data 的并发访问）
locker = threading.Lock()


# =====================================================================
# 弹性绳（可选，辅助站立）
# =====================================================================
elastic_band = None
band_attached_link = None
if ENABLE_ELASTIC_BAND:
    from hg_sim2sim_bridge import ElasticBand
    elastic_band = ElasticBand()
    band_attached_link = mj_model.body("pelvis").id


# =====================================================================
# DDS 相关
# =====================================================================

# 最近收到的 lowcmd（DDS 回调线程写入，主线程读取）
_last_lowcmd_time = 0.0
_last_q_des = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
_last_kp = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
_last_kd = np.zeros(NUM_DDS_MOTORS, dtype=np.float64)
_cmd_lock = threading.Lock()

# 是否已收到第一条 lowcmd
_lowcmd_received = False



def on_lowcmd(msg: LowCmd_):
    """DDS lowcmd 回调：保存接收到的 PD 目标。"""
    global _last_lowcmd_time, _lowcmd_received
    q_des = np.array([msg.motor_cmd[i].q for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
    kp = np.array([msg.motor_cmd[i].kp for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
    kd = np.array([msg.motor_cmd[i].kd for i in range(NUM_DDS_MOTORS)], dtype=np.float64)
    with _cmd_lock:
        _last_q_des[:] = q_des
        _last_kp[:] = kp
        _last_kd[:] = kd
        _last_lowcmd_time = time.perf_counter()
        if not _lowcmd_received:
            _lowcmd_received = True
            print("[HGBridge] 已收到第一条 lowcmd。")


def build_lowstate():
    """从 MuJoCo 数据构建 LowState_ 消息。"""
    state = unitree_go_msg_dds__LowState_()

    # 读取 IMU 数据
    raw_quat = mj_data.sensor("orientation").data.copy()
    gyro = mj_data.sensor("angular-velocity").data.copy()

    # IMU 状态（Unitree 格式 = [w, x, y, z]，MuJoCo framequat 也是 [w, x, y, z]）
    for i in range(4):
        state.imu_state.quaternion[i] = float(raw_quat[i])
    for i in range(3):
        state.imu_state.gyroscope[i] = float(gyro[i])

    # 20 路电机状态
    q_h1 = mj_data.qpos[QPOS_ADRS].copy()
    dq_h1 = mj_data.qvel[QVEL_ADRS].copy()

    for dds_i in range(NUM_DDS_MOTORS):
        h1_i = DDS_TO_H1_ACT[dds_i]
        if h1_i >= 0:
            state.motor_state[dds_i].q = float(q_h1[h1_i])
            state.motor_state[dds_i].dq = float(dq_h1[h1_i])
        else:
            state.motor_state[dds_i].q = 0.0
            state.motor_state[dds_i].dq = 0.0
        state.motor_state[dds_i].tau_est = 0.0

    state.tick = int(time.perf_counter() * 1000)
    return state


# =====================================================================
# 主循环
# =====================================================================

def main():
    global _last_lowcmd_time, _lowcmd_received

    # ── DDS 初始化 ──────────────────────────────────────────────────
    print(f"[HGBridge] DDS: domain={DOMAIN_ID}, interface={INTERFACE}")
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)

    lowcmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    lowcmd_suber.Init(on_lowcmd, 10)

    lowstate_puber = ChannelPublisher("rt/lowstate", LowState_)
    lowstate_puber.Init()
    print("[HGBridge] DDS 初始化完成。")

    # ── MuJoCo 可视化 ──────────────────────────────────────────────
    # 修复数字键 0-5 冲突：MuJoCo viewer 默认会切换 geom group 可见性
    def _viewer_key_callback(key):
        if viewer is not None and ord('0') <= key <= ord('5'):
            idx = key - ord('0')
            viewer.opt.geomgroup[idx] = 1
        if elastic_band is not None:
            elastic_band.MujuocoKeyCallback(key)

    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=_viewer_key_callback
    )

    # ── 状态 ────────────────────────────────────────────────────────
    running = True
    _last_publish_time = time.perf_counter()
    _publish_interval = POLICY_DT  # 100Hz
    step_count = 0

    # 首次发布延迟：先跑几步物理再发布 lowstate，确保机器人稳定站立
    _initial_steps = 50  # 0.5s 仿真时间的初始 PD hold
    _initialized = False

    print("=" * 60)
    print("[HGBridge] 仿真已启动，等待控制器连接...")
    print(f"[HGBridge] DDS domain={DOMAIN_ID}, interface={INTERFACE}")
    print(f"[HGBridge] 初始 PD hold {_initial_steps} 步稳定姿态")
    if ENABLE_ELASTIC_BAND:
        print("[HGBridge] 弹性绳已启用")
    print("=" * 60)

    # ── 仿真循环 ────────────────────────────────────────────────────
    while viewer.is_running() and running:
        step_start = time.perf_counter()

        # ── 发布 lowstate（100Hz，在物理步进之前） ─────────────────
        # 前 _initial_steps 步不发布，让机器人充分稳定
        if step_count >= _initial_steps:
            now = time.perf_counter()
            if now - _last_publish_time >= _publish_interval:
                locker.acquire()
                state = build_lowstate()
                locker.release()
                lowstate_puber.Write(state)
                _last_publish_time = now
                if not _initialized:
                    _initialized = True
                    print(f"[HGBridge] 开始发布 lowstate (z={mj_data.qpos[2]:.3f})")

        locker.acquire()

        # 弹性绳
        if ENABLE_ELASTIC_BAND and elastic_band is not None and elastic_band.enable:
            mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                mj_data.qpos[:3], mj_data.qvel[:3]
            )

        # ── 物理步进 ───────────────────────────────────────────────
        # 每个 decimation block 只读一次 lowcmd，10 个子步共用同一组
        # q_des/kp/kd，与 sim2sim 的 _pd_control() 行为一致。
        if _lowcmd_received:
            with _cmd_lock:
                q_des_20 = _last_q_des.copy()
                kp_20 = _last_kp.copy()
                kd_20 = _last_kd.copy()
            use_external_cmd = True
        else:
            use_external_cmd = False

        for _ in range(DECIMATION):
            if use_external_cmd:
                for h1_i in range(HG_NUM_ACTION):
                    dds_i = H1_ACT_TO_DDS[h1_i]
                    err = q_des_20[dds_i] - mj_data.qpos[QPOS_ADRS[h1_i]]
                    dq = mj_data.qvel[QVEL_ADRS[h1_i]]
                    mj_data.ctrl[h1_i] = kp_20[dds_i] * err - kd_20[dds_i] * dq
            else:
                for h1_i in range(HG_NUM_ACTION):
                    err = HG_DEFAULT_DOF_POS[h1_i] - mj_data.qpos[QPOS_ADRS[h1_i]]
                    dq = mj_data.qvel[QVEL_ADRS[h1_i]]
                    mj_data.ctrl[h1_i] = HG_KPS[h1_i] * err - HG_KDS[h1_i] * dq

            mujoco.mj_step(mj_model, mj_data)

        locker.release()

        # 同步可视化（通知渲染线程有新帧可渲染）
        viewer.sync()

        # 状态打印
        if step_count % 500 == 0:
            z = mj_data.qpos[2]
            q_h1 = mj_data.qpos[QPOS_ADRS]
            recv = "lowcmd" if _lowcmd_received else "no-cmd"
            print(f"[HGBridge] t={mj_data.time:.2f}  z={z:.3f}  "
                  f"hip={q_h1[2]:.2f}/{q_h1[7]:.2f}  "
                  f"knee={q_h1[3]:.3f}/{q_h1[8]:.3f}  [{recv}]")

        # 仿真频率控制
        elapsed = time.perf_counter() - step_start
        sleep = SIMULATE_DT * DECIMATION - elapsed
        if sleep > 0:
            time.sleep(sleep)

        step_count += 1

    print("[HGBridge] 退出。")


if __name__ == "__main__":
    main()
