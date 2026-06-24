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

## Command Levels

`Curriculum/command_levels` 返回当前 x 方向速度命令上限，不是归一化 0-1 分数。

当前 highstep：

```text
初始 x max = 0.2975
中档 x max = 0.4675
最终 x max = 0.6375
```

当前最终值比 bodyflat 小，因为 highstep 使用：

```text
lin_vel_x upper = 0.85
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

危险信号：

- terrain 下降，同时 highstep 奖励也下降；
- `error_vel_xy` 长时间高于约 `0.35`；
- `bad_orientation` 长时间高于约 `0.015-0.02`；
- `highstep_forward_progress` 掉回约 `0.08` 以下；
- action-rate 惩罚和 entropy 上升，同时 mean reward 下降。

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

