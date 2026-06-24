# Arcdog 可伸缩腿模型笔记

## 当前机器人资源

配置文件：

```text
source/robot_lab/robot_lab/assets/arclab.py
```

当前 USD：

```text
source/robot_lab/data/Robots/Arclab/Arcdog_adjustable_leg_short_limit/arcdog_adjustable_leg_short_limit.usd
```

## 训练使用的关节顺序

当前 highstep/bodyflat action 顺序：

```text
FL_hip_joint
FR_hip_joint
RL_hip_joint
RR_hip_joint
FL_thigh_joint
FR_thigh_joint
RL_thigh_joint
RR_thigh_joint
FL_calf_joint
FR_calf_joint
RL_calf_joint
RR_calf_joint
FL_box_joint
FR_box_joint
RL_box_joint
RR_box_joint
```

## 初始关节位置

当前 `arclab.py` 默认初始姿态：

```text
hip:   0.0
thigh: 0.7
calf: -1.3
box:   0.03
```

box 初始值理由：

```text
当前 box_joint 范围是 0.00 到 0.06，所以 0.03 是中位。
这有利于 action 输出围绕默认值对称。
```

## Box Joint 方向

永久约定：

```text
box_joint 数值越小 -> 腿越长
box_joint 数值越大 -> 腿越短
```

这个方向之前多次出错。以后新增 reward 或 prior 前，必须用 play 的 action mapping 验证。

当前 highstep action prior 范围：

```text
min_box_target = 0.000
max_box_target = 0.060
```

## Action Scaling

当前 highstep action scale：

```text
box joints: 0.02
hip/thigh/calf: 0.1
raw action clip: -60 到 60
```

注意 raw action clip 不等于物理安全范围，真实安全范围仍由关节限位和 target clamp 决定。

## 执行器参数

当前 `arclab.py`：

```text
hip:
  effort_limit = 35.0
  velocity_limit = 45.0
  stiffness = 45.0
  damping = 1.5

thigh:
  effort_limit = 35.0
  velocity_limit = 45.0
  stiffness = 50.0
  damping = 1.5

calf:
  effort_limit = 80.0
  velocity_limit = 45.0
  stiffness = 60.0
  damping = 2.0

box:
  effort_limit = 1000.0
  velocity_limit = 0.13
  stiffness = 8000.0
  damping = 100.0
  armature = 0.4
```

## 部署相关

仿真中应该能暴露 box joint 超限问题，不能完全靠部署端 clamp 隐藏。部署端 clamp 可作为安全保护，但训练策略本身也应尽量不撞限位。

