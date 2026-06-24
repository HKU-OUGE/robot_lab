# WandB 曲线判断指南

## Terrain Levels

`Curriculum/terrain_levels` 有参考价值，但不是最终判断。

Highstep 中它会受这些因素影响：

- 多环境平均；
- 混合地形类型；
- reset 分布；
- 高台/低处出生比例；
- curriculum 判断方式。

因此它可能在 play 已经能上高台时仍然不高，也可能曲线好看但 play 很烂。

已知例子：

```text
2026-06-15_13-03-10 terrain levels 较高，但后面 play 姿态差。
2026-06-16_04-22-56 terrain levels 不高，但 play 已经能上高台。
```

2026-06-19 关键澄清：

```text
terrain_levels 是所有训练 env 在 curriculum 升降级后的 terrain row 平均值。
它不是机器人能爬的最大高台高度。
```

不要脱离任务直接比较 bodyflat 和 highstep 的 terrain 数值。截图中的 2026-05/06 bodyflat 样本大约在 `5.2-5.7`，但它们是 bodyflat 任务、标准 `terrain_levels_vel`、完整 `[-1, 1]` command 范围和不同地形混合。当前 highstep 在 `2.7-2.8` 左右，play 里仍可能能爬指定的 0.35 m 高台。

当前 highstep 例子：

```text
2026-06-17_05-07-05: terrain ~= 3.27, bad_orientation avg100 ~= 0.0083
2026-06-19_01-42-04: terrain ~= 2.79, bad_orientation avg100 ~= 0.0014
```

解释：新 run 更稳定，但 terrain 曲线更低。这是取舍，不是自动失败。

## Command Levels

`Curriculum/command_levels` 返回当前 x 方向速度命令上限，不是归一化 0-1 分数。

当前 highstep：

```text
初始 x max = 0.2975
中档 x max = 0.4675
最终 x max = 0.6375
最终 x min = -0.225
```

当前最终正向值比 bodyflat 小，因为 highstep 使用：

```text
lin_vel_x = (-0.30, 0.85)
range_multiplier final = 0.75
```

bodyflat 使用：

```text
lin_vel_x upper = 1.0
range_multiplier final = 1.0
```

## Highstep 关键曲线

一起看这些：

```text
Curriculum/terrain_levels
Curriculum/command_levels
Train/mean_reward
Metrics/base_velocity/error_vel_xy
Metrics/base_velocity/error_vel_yaw
Episode_Reward/highstep_forward_progress
Episode_Reward/highstep_base_advance_lift
Episode_Reward/highstep_rear_push_posture
Episode_Reward/highstep_rear_box_push
Episode_Termination/bad_orientation
Episode_Termination/terrain_out_of_bounds
Episode_Termination/time_out
Episode_Reward/highstep_box_phase_prior
Episode_Reward/prismatic_joint_pos_penalty
Loss/entropy
Policy/mean_noise_std
```

健康信号：

- terrain 缓慢上升或至少不崩；
- command 到达目标阶段；
- highstep 奖励保持高位；
- bad orientation 低；
- episode length 接近最大；
- command 放开后速度误差只适度上升。
- 平地场景 box-joint 默认位惩罚没有爆炸，play 中平地静止/行走时 box joint 接近默认位。

危险信号：

- terrain 下降，同时 highstep 奖励也下降；
- `error_vel_xy` 长时间高于约 `0.35`；
- `bad_orientation` 长时间高于约 `0.015-0.02`；
- `highstep_forward_progress` 掉回约 `0.08` 以下；
- action-rate 惩罚和 entropy 上升，同时 mean reward 下降。
- 平地静止/普通行走时 box joint 持续明显偏移，即使 terrain 或 reward 曲线看起来还可以。

## 当前 `model_63700` 快照

`2026-06-17_03-05-16/model_63700` 附近：

```text
terrain_levels ~ 3.0
command_levels = 0.6375
highstep_forward_progress ~ 0.18
highstep_base_advance_lift ~ 0.15
highstep_rear_box_push ~ 0.06
bad_orientation ~ 0.008-0.010
```

解释：

```text
没有崩。是目前较好的 teacher 候选，但切 student 前还要看后续 checkpoint 是否稳定。
```

## 当前 2026-06-19 Teacher 精修快照

Run：

```text
2026-06-19_01-42-04
```

本次记忆更新时最新观测：

```text
step ~= 86422
model_86400.pt
terrain_levels avg100 ~= 2.789
command_levels = 0.6375
bad_orientation avg100 ~= 0.00144
time_out avg100 ~= 0.99856
error_vel_xy avg100 ~= 0.19183
mean_reward avg100 ~= 167.0
```

解释：

```text
相比旧 teacher/student run 非常稳定。继续训练时要重点 play 检查：terrain_levels 不高的情况下，上高台能力是否保住，以及平地 box joint 是否保持自然。
```
