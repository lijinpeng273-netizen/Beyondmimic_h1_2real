# H1 Beyondmimic 部署

基于 Beyondmimic 训练策略的宇树 H1 机器人（19 DOF）Sim2Sim + Sim2Real 全身运动部署。

---

## 快速开始

```bash
# Sim2Sim 仿真
python h1sim2sim_v2.py --motion_file deploy_real/bydmimic/dance1_subject3_last.npz

# Sim2Real 实机（连机器人后）
cd deploy_real
python h1_deploy_real.py h1.yaml --interface enp12s0 --domain 0
```

---

## 仿真控制

```bash
# ONNX 自举模式（无需 NPZ）
python h1sim2sim_v2.py

# NPZ 实机模式（与部署一致）
python h1sim2sim_v2.py --motion_file xxx.npz

# 手柄模式
python h1sim2sim_v2.py --motion_file xxx.npz --gamepad
```

| 键盘 | 手柄 | 功能 |
|---|---|---|
| `6` | A | 站立 |
| `7` | B | 播放 |
| `5` | X | 锁定 |
| `8` | Y | 弹力绳 |

---

## 实机部署

### 按键流程

| 按键 | 功能 |
|---|---|
| **A** | 锁定当前位置，启动 |
| (自动) | 2s 过渡到默认站立 |
| **Start** | 1s 缓冲渐变到舞蹈首帧 → 开始 |
| **B** | 中断舞蹈 |
| **X** | 退出站立，阻尼关机 |

### 配置

编辑 `deploy_real/configs/h1.yaml`，修改网卡和模型路径：

```yaml
dds_interface: "enp12s0"   # 机器人网卡
policy_path: "xxx.onnx"    # 策略模型
motion_file: "xxx.npz"     # 动作数据
```

### 安全机制

运行中自动生效，无需手动操作：

- NaN/Inf 传感器数据自动过滤
- 0.3s 无通信自动断开 RL
- IMU 检测摔倒自动断开
- 动作异常值丢弃
- 关节目标速率限制 + 软限位

---

## 目录结构

```
├── h1sim2sim_v2.py              # MuJoCo 仿真
├── h1sim2sim_v2_config.py       # 仿真配置
├── deploy_real/
│   ├── h1_deploy_real.py        # 实机主脚本
│   ├── h1_config.py             # 关节定义 / 电机映射
│   ├── h1_safety.py             # 安全监控
│   ├── config.py                # YAML 加载
│   ├── observation_builder.py   # 观测构造
│   ├── configs/h1.yaml          # 配置文件
│   ├── common/                  # DDS / IMU / 遥控器
│   └── bydmimic/                # ONNX + NPZ 数据
├── unitree_description/mjcf/    # H1 MuJoCo 模型
└── H1_DDS/                      # 10-DOF 行走参考
```

---

## 依赖

```bash
pip install onnxruntime onnx numpy mujoco cyclonedds
```

`unitree_sdk2_python` 需单独下载并加入 `sys.path`。

---

## 致谢

- [Beyondmimic](https://github.com/HybridRobotics/whole_body_tracking)
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)
