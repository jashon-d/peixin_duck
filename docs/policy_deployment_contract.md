# 策略部署契约 (Policy Deployment Contract)

> **阶段 0 产出 · 冻结文档**
> 本文档冻结 MicroDuck 策略从 ONNX 导出到板端推理的接口契约,是阶段 3 `robotd` 开发的**接口规格书**。
> 违反此契约会导致 sim2real 失败(策略在仿真里能跑、真机上崩)。
> 来源:`microduck_rl` 源码 + AGENTS.md 交叉验证 + 2026-11-20 冒烟实测。

---

## 1. 验证证据(可复现)

### 1.1 导出

- 命令(在 `microduck_rl` 目录):
  ```bash
  export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897 UV_HTTP_TIMEOUT=600
  uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file logs/rsl_rl/velocity/2026-09-27_18-54-25_velocity/model_4.pt \
    --onnx-file /tmp/dummy_velocity.onnx
  ```
- checkpoint:smoke test 产物(5 iters,策略未收敛,仅用于验证链路)
- 导出日志关键证据:
  ```
  Actor Model: MLPModel(
    (obs_normalizer): EmpiricalNormalization()        ← normalizer 是子模块 → bake 进 ONNX
    (mlp):
      (0): Linear(in_features=61, out_features=512)    ← 输入 61 维
      (6): Linear(in_features=128, out_features=14)    ← 输出 14 维
  )
  ```

### 1.2 推理(onnxruntime 实测)

| 项 | 值 |
|----|-----|
| 输入 name | `obs` |
| 输入 shape | `[1, 61]`, dtype `float32` |
| 输出 name | `actions` |
| 输出 shape | `[1, 14]`, dtype `float32` |
| normalizer 确定性 | 同一输入两次推理输出完全一致 → 已 bake 进图,推理时不更新统计 |

复现命令:
```bash
cd /home/wang/MicroDuck/microduck_rl
uv run python -c "import onnxruntime as ort; s=ort.InferenceSession('/tmp/dummy_velocity.onnx'); print(s.get_inputs()[0].shape, s.get_outputs()[0].shape)"
```

---

## 2. ONNX 接口契约

### 2.1 输入:`obs` — shape `[1, 61]` float32

全策略家族共享此布局(walk / stand / kick / sitstand 可热插拔)。段位**严格如下,顺序不可变**:

| 段位 | 维度 | 内容 | 物理来源 / 计算方式 |
|------|------|------|---------------------|
| `[0:3]` | 3 | `base_ang_vel` | IMU 陀螺仪,机体坐标系下的角速度 (rad/s) |
| `[3:6]` | 3 | `projected_gravity` | 由 IMU 加速度计算的重力方向在机体系的投影(单位向量) |
| `[6:20]` | 14 | `joint_pos_relative` | 14 舵机当前关节角 − `DEFAULT_POSE`(弧度) |
| `[20:34]` | 14 | `joint_vel` | 14 舵机当前关节角速度 (rad/s) |
| `[34:48]` | 14 | `last_action` | 上一帧策略输出的 action(给策略"惯性"感知) |
| `[48:61]` | 13 | `command` block | `[twist(3), head_pose(4), body_pose(6)]`,见下 |

**command block 13 维细分**(`new_cmd_obs` 模式,所有新策略统一):

| 子段位 | 维度 | 内容 |
|--------|------|------|
| `[48:51]` | 3 | `twist`: `[vx, vy, vtheta]`(线速度 x/y + 角速度 yaw) |
| `[51:55]` | 4 | `head_pose`: `[neck_pitch, head_pitch, head_yaw, head_roll]` 增量 |
| `[55:61]` | 6 | `body_pose`: `[x, y, z, roll, pitch, yaw]` |

> ⚠️ **铁律**:不用的 command slot 必须 **zero-pad**,绝对不能删(删了维度对不上,策略热插拔就废了)。
> 参考:[infer_policy.py `_update_command()`](file:///home/wang/MicroDuck/microduck_rl/scripts/infer_policy.py#L461-L513)

### 2.2 输出:`actions` — shape `[1, 14]` float32

14 维对应 14 个舵机,关节顺序**锁死**(ctrl idx = joint idx):

| idx | 关节 | 部位 |
|-----|------|------|
| 0 | `hip_yaw` | 左腿 |
| 1 | `hip_roll` | 左腿 |
| 2 | `hip_pitch` | 左腿 |
| 3 | `knee` | 左腿 |
| 4 | `ankle` | 左腿 |
| 5 | `neck_pitch` | 头颈 |
| 6 | `head_pitch` | 头颈 |
| 7 | `head_yaw` | 头颈 |
| 8 | `head_roll` | 头颈 |
| 9 | `hip_yaw` | 右腿 |
| 10 | `hip_roll` | 右腿 |
| 11 | `hip_pitch` | 右腿 |
| 12 | `knee` | 右腿 |
| 13 | `ankle` | 右腿 |

> 参考:[AGENTS.md 关节布局约定](file:///home/wang/MicroDuck/microduck_rl/AGENTS.md) + [infer_policy.py `DEFAULT_POSE`](file:///home/wang/MicroDuck/microduck_rl/scripts/infer_policy.py#L126-L141)

---

## 3. Action → 舵机的变换链路

策略输出的 `action` 不是直接的舵机角度,需经过两步变换:

```
action (14D, 范围约 [-1, 1])
   │
   ▼  ① × action_scale
   │
   ▼  ② + DEFAULT_POSE (14D 参考姿态, 弧度)
   │
target_positions (14D, 目标关节角)
   │
   ▼  ③ 交给 BAM 控制器,固件 PD 环 → 电机力矩 → 舵机转动
```

### 3.1 `DEFAULT_POSE`(精确数值,弧度)

来源:[infer_policy.py](file:///home/wang/MicroDuck/microduck_rl/scripts/infer_policy.py#L126-L141),与 `microduck_constants.py` 的 `HOME_FRAME` 一致:

```python
DEFAULT_POSE = [
    0.0,      # 0  left_hip_yaw
   -0.0873,   # 1  left_hip_roll
   -0.4579,   # 2  left_hip_pitch
   -0.0049,   # 3  left_knee
    0.4530,   # 4  left_ankle
    0.3491,   # 5  neck_pitch
    0.3491,   # 6  head_pitch
    0.0,      # 7  head_yaw
    0.0,      # 8  head_roll
    0.0,      # 9  right_hip_yaw
    0.0873,   # 10 right_hip_roll
    0.4579,   # 11 right_hip_pitch
    0.0049,   # 12 right_knee
   -0.4530,   # 13 right_ankle
]
```

### 3.2 `action_scale`

默认 `1.0`(推理脚本默认值)。最终目标角:
```
target_positions[i] = DEFAULT_POSE[i] + action[i] * action_scale
```

### 3.3 关键机理

> `joint_pos_relative`(obs `[6:20]`)是 `当前角 − DEFAULT_POSE`,而 action 是相对 DEFAULT_POSE 的增量 —— **观测和动作都锚定在 DEFAULT_POSE 上**,这是策略能泛化的关键。写 robotd 时两处必须用同一个 DEFAULT_POSE,否则观测和动作不在同一参考系。

---

## 4. BAM 电压控制律参数

训练时执行器建模为 BAM(电压控制的 XL330 模型),**真机部署必须复刻**,否则力矩上限对不上 → sim2real gap。

来源:[infer_policy.py BAM 常量](file:///home/wang/MicroDuck/microduck_rl/scripts/infer_policy.py#L34-L46) + [test_infer_policy_bam.py](file:///home/wang/MicroDuck/microduck_rl/tests/test_infer_policy_bam.py):

| 参数 | 值 | 含义 |
|------|-----|------|
| `motor_name` | `"xl330"` | 舵机型号 |
| `model` | `"m6"` | BAM 模型版本 |
| `kp_fw` | `200.0` | 固件 PD 刚度 |
| `vin_range` | `(6.5, 8.2)` | 电池电压域随机化范围(V) |
| `vin_drop_gain_range` | `(0.0, 0.2)` | 负载相关压降增益 |
| `vin_min` | `6.0` | 压降后有效电压下限(V) |
| `max_current` | `None` | 训练时无固件电流限制 |

### 4.1 力矩上限公式

```
force_limit = vin × kt / R
```
(`kt` 转矩常数、`R` 线圈电阻,由 BAM 模型给出)

### 4.2 为什么电池电压重要

电池电压跌落 → `vin` 降低 → `force_limit` 降低 → 舵机输出力矩不足 → 策略在仿真里(电压充足)能做的动作,真机(电压跌)做不出来。**写 robotd 时必须把真实电池电压喂给 BAM 模型**,不能写死 7.4V。

---

## 5. Normalizer 机理(部署最大的坑)

- 训练时观测经过 `EmpiricalNormalization`(在线统计均值/方差 → 归一化)
- 导出时 `runner.export_policy_to_onnx` 把 normalizer 作为子模块 **bake 进计算图**,产出 `actor(normalizer(obs))`
- **部署端喂给 ONNX 的必须是原始未归一化的物理量**(关节角=弧度、角速度=rad/s),ONNX 内部自己归一化

### 5.1 绝对不能做的事

| 错误做法 | 后果 |
|----------|------|
| 手动把 checkpoint 转 ONNX(不用 `export.py`) | 丢掉 normalizer → 策略必崩 |
| 部署端自己先归一化再喂 ONNX | 双重归一化 → 策略必崩 |
| 改 obs 段位顺序 | 输入语义错乱 → 策略必崩 |

> 惨痛教训(AGENTS.md):in-sim `play` 会自己应用 normalizer,所以手动导出的 bug 在仿真里看不见,**一上真机就炸**。唯一合法导出路径是 `scripts/export.py`。

---

## 6. 部署伪代码(C++ robotd 视角)

```cpp
// 每 20ms (50Hz) 执行一次
void control_loop() {
    // 1. 采集传感器 → 组装 61D obs(原始物理量,不归一化)
    float obs[61];
    obs[0:3]   = imu.get_ang_vel();            // 机体角速度
    obs[3:6]   = imu.get_projected_gravity();  // 重力方向投影
    obs[6:20]  = servo.get_positions() - DEFAULT_POSE;  // 相对默认姿态
    obs[20:34] = servo.get_velocities();
    obs[34:48] = last_action;                  // 上一帧输出
    obs[48:61] = command;                      // [twist(3), head(4), body(6)]

    // 2. ONNX 推理(obs 原始值喂入,normalizer 在图内)
    float action[14];
    onnx_session.run(obs, action);  // obs[1,61] → action[1,14]

    // 3. 变换 → 目标关节角
    float target[14];
    for (int i = 0; i < 14; i++)
        target[i] = DEFAULT_POSE[i] + action[i] * action_scale;

    // 4. 交给 BAM 控制器(固件 PD → 电压 → 力矩 → 舵机)
    bam.set_q_target(target, battery_voltage);  // 真实电池电压!
    bam.update();
    last_action = action;
}
```

---

## 7. 验证依赖 vs 上线依赖边界

| | 验证(PC,已完成) | 上线(板端 robotd,阶段3) |
|---|---|---|
| 导出 | `torch` + `warp` + `mjlab` + `mujoco`(训练栈全套) | ❌ 不需要(ONNX 已导出) |
| 推理验证 | `onnxruntime`(Python) + `mujoco` 仿真 | `onnxruntime` C++ API |
| 传感器/舵机 | 仿真数据 | 真实驱动(阶段1/2 自建) |
| 依赖集合 | 大(.venv 7.9G) | 小(只留 onnxruntime + 自研驱动) |

> 经验教训:上线链路的依赖集合必须与验证阶段**显式拆分**,板端只保留 `onnxruntime` + 自研 C++ 驱动,绝不引入 torch(会让板端内存/启动时间爆炸)。

---

## 8. 不变量(Invariants,不可破坏)

1. obs layout 必须 61D 且全策略共享 → 可热插拔
2. 不用的 command slot 必须 zero-pad,不能删
3. 关节顺序 0–13 锁死,不能硬编码错位
4. obs normalization 必须由 `export.py` bake 进 ONNX,不能手动处理
5. 策略不滤波(训练时无 action low-pass),部署端也不能加 EMA(除非同步改训练)
6. `joint_pos_relative` 和 action 必须锚定同一个 `DEFAULT_POSE`
7. BAM 电压控制律必须复刻,真实电池电压必须喂入

---

## 9. 变更记录

| 日期 | 版本 | 变更 | 证据 |
|------|------|------|------|
| 2026-11-20 | v1.0 | 初始冻结:61D/14D 契约 + DEFAULT_POSE + BAM 参数 + 冒烟验证通过 | 本文档 + `/tmp/dummy_velocity.onnx` 推理实测 |
