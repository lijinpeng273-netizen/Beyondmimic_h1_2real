# Beyondmimic Deploy G1/H1

基于 [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking) 训练策略的宇树（Unitree）人形机器人 **Sim2Sim** 与 **Sim2Real** 全身运动部署框架。

支持 **H1**（19 DOF）和 **G1**（29 DOF）两种机器人平台，通过 ONNX Runtime 进行策略推理，MuJoCo 完成物理仿真，Unitree DDS 协议实现实机通信。

---

## 目录结构

```
Beyondmimic_Deploy_G1/
├── h1sim2sim_v2.py                  # H1 Sim2Sim 仿真（ONNX 自举模式，仅需 ONNX）
├── h1sim2sim_v2_config.py           # H1 Sim2Sim 仿真配置文件
├── deploy_mujoco_1.py               # G1 Sim2Sim 仿真（29 DOF，外置 NPZ 运动）
├── csv_to_npz_with_Interpolation.py # 运动数据转换：CSV → NPZ（Isaac Lab）
├── deploy_real/                     # 实机部署代码
│   ├── h1_deploy_real.py            # H1 实机部署主脚本（19 DOF，生产级）
│   ├── h1_config.py                 # H1 硬性参数（关节布局、电机映射、维度）
│   ├── h1_safety.py                 # H1 五层安全监控器
│   ├── deploy_real4bydmimic.py      # G1 实机部署脚本（29 DOF）
│   ├── config.py                    # 通用 YAML 配置加载器
│   ├── configs/
│   │   ├── h1.yaml                  # H1 部署配置文件
│   │   └── g1_for_bydmimic.yaml     # G1 部署配置文件
│   ├── common/
│   │   ├── command_helper.py        # DDS 指令辅助函数
│   │   ├── rotation_helper.py       # IMU/旋转变换工具
│   │   └── remote_controller.py     # 无线遥控器解析
│   └── bydmimic/                    # 策略模型与运动数据
├── H1_DDS/                          # H1 10-DOF 行走策略子系统（独立）
│   ├── hg_policy_controller.py      # DDS 策略控制器
│   ├── hg_mujoco_dds_bridge.py      # MuJoCo-DDS 联合调试桥
│   └── hg_config.py                 # 10-DOF 行走策略配置
├── unitree_description/             # 机器人模型文件
│   ├── mjcf/                        # MuJoCo XML 模型（h1.xml, g1.xml, g1_liao.xml）
│   ├── urdf/                        # URDF 模型（h1/, g1/）
│   └── meshes/                      # STL 网格文件（h1/, g1/）
├── other/                           # 参考配置文件（旧版）
└── CLAUDE.md                        # 项目编码规范
```

---

## 环境配置

本项目依赖 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)，推荐目录结构如下：

```bash
# 工作目录布局
~/Documents/
├── unitree_rl_gym/          # Unitree RL Gym 仓库
│   └── deploy/
│       └── deploy_real/
│           ├── common/      # 公共模块（command_helper, rotation_helper 等）
│           └── configs/     # 配置文件
└── Beyondmimic_Deploy_G1/   # 本仓库（软链接或放置于 deploy/ 下）
```

### 依赖项

- Python 3.8+
- `onnxruntime` — ONNX 策略推理
- `onnx` — ONNX 模型元数据解析
- `numpy` — 数值计算
- `torch` — 观测张量构造
- `mujoco` — 物理仿真（Sim2Sim）
- `unitree_sdk2py` — 宇树 DDS 通信（Sim2Real）
- `isaaclab` / `isaac-sim` — 运动数据重定向（仅 `csv_to_npz_with_Interpolation.py`）

安装核心依赖：

```bash
pip install onnxruntime onnx numpy torch mujoco
```

宇树 SDK 安装请参考 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 官方文档。

---

## Sim2Sim 仿真部署

Sim2Sim 在 MuJoCo 物理引擎中验证策略，无需实机即可测试。

### H1 仿真（19 DOF，ONNX 自举模式）

**特点：** 仅需 ONNX 策略文件，无需外置 NPZ 运动数据。参考轨迹由 ONNX 模型自举（bootstrap）生成。

```bash
# 使用默认配置
python h1sim2sim_v2.py

# 使用自定义配置文件
python h1sim2sim_v2.py --config my_config.py
```

**配置文件** `h1sim2sim_v2_config.py` 关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `XML_PATH` | MuJoCo 模型路径 | `unitree_description/mjcf/h1.xml` |
| `POLICY_PATH` | ONNX 策略路径 | `deploy_real/bydmimic/xxx.onnx` |
| `SIMULATION_DT` | 物理步长 | 0.005 s |
| `CONTROL_DECIMATION` | 控制分频 | 4（策略频率 50 Hz） |
| `SIMULATION_DURATION` | 仿真时长 | 60 s |

**键盘控制：**

| 按键 | 功能 |
|------|------|
| `6` | 重置站立姿态 |
| `7` | 开始策略播放 |
| `8` | 切换弹性绳（辅助站立） |
| `Backspace` | 重置仿真 |

### G1 仿真（29 DOF，NPZ 运动模式）

```bash
python deploy_mujoco_1.py
```

需要配套的 ONNX 模型和 NPZ 运动文件（在脚本内硬编码路径，使用前需修改）。

---

## Sim2Real 实机部署

### H1 实机部署（推荐使用）

H1 部署脚本是当前最完善的实机部署方案，包含多层级安全监控和完整的关节映射。

```bash
# 进入部署目录，初始化 DDS 通信并运行
cd deploy_real

# 基本用法（使用配置文件中的默认网卡和域 ID）
python h1_deploy_real.py h1.yaml

# 指定网卡（实机通常为 enp2s0 或 enp6s0）
python h1_deploy_real.py h1.yaml --interface enp2s0

# 仿真调试模式（使用 lo 回环网卡 + 域 ID 1）
python h1_deploy_real.py h1.yaml --interface lo --domain 1
```

**部署前修改配置文件** [deploy_real/configs/h1.yaml](deploy_real/configs/h1.yaml)：

1. **策略模型路径：** `policy_path`
2. **运动数据路径：** `motion_file`
3. **PD 增益：** `stiffness` / `damping`（19 个关节，DOF 顺序）
4. **默认站立角度：** `default_angles`
5. **动作缩放因子：** `action_scale_seq`

**部署流程（状态机）：**

```
零力矩等待 ──[Start键]──▶ 平滑过渡到默认站立（2秒） ──▶ 站立保持 ──[A键]──▶ RL策略运行 ──[Select键]──▶ PD Hold退出
```

**安全监控**（`h1_safety.py`）五层保护：

| 层级 | 功能 | 说明 |
|------|------|------|
| L1 | 传感器校验 | NaN/Inf 过滤，IMU 四元数范数检查 |
| L2 | 通信超时 | lowstate 更新检测（默认 0.3s） |
| L3 | 摔倒检测 | IMU Roll/Pitch 阈值（1.2 rad） |
| L4 | 动作安全 | 裁剪、预热斜坡、EMA 平滑 |
| L5 | 指令安全 | 速率限制、软关节限位、退出时 PD Hold 插值 |

### G1 实机部署

```bash
cd deploy_real
python deploy_real4bydmimic.py enp4s0 g1_for_bydmimic.yaml
```

**G1 需要修改的路径（在脚本内）：**

1. `config_path` — 配置文件路径（主函数中）
2. `self.motion = np.load(...)` — NPZ 运动文件路径
3. `policy_path` — ONNX 模型路径（在 YAML 配置中）

---

## 运动数据准备

将 CSV 格式的动捕数据转换为 NPZ 格式（需要 Isaac Lab 环境）：

```bash
python csv_to_npz_with_Interpolation.py \
    --input_file LAFAN1/dance2_subject5.csv \
    --input_fps 30 \
    --output_name dance2_subject5 \
    --output_fps 50 \
    --frame_range 122 722
```

输出 NPZ 包含四个数组：

| 数组 | 维度 | 说明 |
|------|------|------|
| `body_pos_w` | (N帧, N刚体, 3) | 各刚体世界位置 |
| `body_quat_w` | (N帧, N刚体, 4) | 各刚体世界四元数 |
| `joint_pos` | (N帧, 19/29) | 关节目标位置 |
| `joint_vel` | (N帧, 19/29) | 关节目标速度 |

---

## 观测空间说明

### H1 观测（110 维）

```
[0:38]   command（ONNX 输出的参考关节位置 + 速度，19+19=38）
[38:41]  anchor_pos_b（实机置零）
[41:47]  anchor_ori_b（从 ONNX ref_quat + 机器人骨盆 quat 实时计算，6D）
[47:50]  base_lin_vel（实机置零）
[50:53]  base_ang_vel（IMU 角速度，经 torso→pelvis 变换）
[53:72]  joint_pos（当前关节位置 - 默认位置，19 维）
[72:91]  joint_vel（当前关节速度，19 维）
[91:110] last_action（上一步策略输出，19 维）
```

### G1 观测（154 维）

与 H1 类似，但关节维度增至 29，command 维度增至 58。

---

## 关节映射说明

代码中存在三种关节排列顺序：

| 顺序 | 来源 | 用途 |
|------|------|------|
| **XML 顺序** (`joint_xml`) | MuJoCo/URDF 模型 | 读取 `d.qpos[7:]` / `d.qvel[6:]` |
| **ONNX 顺序** (`joint_seq`) | ONNX 策略元数据 | 策略输入输出（按类型分组、左右对称） |
| **DDS 顺序** (0-19/0-29) | Unitree SDK 电机 ID | 实机通信 `motor_state[motor_id]` |

部署代码中通过索引映射在三种顺序之间自动转换。

---

## 安全注意事项

> ⚠️ **使用前请务必阅读**

1. **首次部署建议使用仿真模式**（`--interface lo --domain 1`）验证策略输出
2. **默认站立时检查终端打印的四元数**是否接近 `[1, 0, 0, 0]`，偏差过大请勿启动运动
3. **高难度动作存在翻倒风险**，建议先使用简单动作测试，确认可行后再替换策略
4. 测试时确保机器人周围有足够空间，操作人员保持安全距离
5. 随时可用 **Select 键** 紧急退出（机器人自动 PD Hold 回到默认姿态）
6. 因使用本代码造成的机器人损坏，使用者自行负责
7. **本项目禁止盈利行为**，引用请标明出处

---

## 策略训练说明

本代码基于 Beyondmimic 作者开源算法中 **不带状态估计** 的训练配置：

```
Tracking-Flat-G1-Wo-State-Estimation-v0（154 维观测）
```

如需使用含状态估计的配置，需在 `anchor_ori` 之前增加 3 维相对位置，以及在 `angvel` 之前增加 3 维根坐标系速度。

---

## 致谢

- [Beyondmimic 训练源码](https://github.com/HybridRobotics/whole_body_tracking)
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)
- 特别感谢 [Owen-SuQ](https://github.com/Owen-SuQ) [642X](https://github.com/642X) 对本项目的支持
