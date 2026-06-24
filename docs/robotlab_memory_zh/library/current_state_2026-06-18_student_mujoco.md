# 当前状态

更新时间：2026-06-17。

## 当前任务

任务名：

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

常规训练命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless
```

从当前最好 teacher 锚点继续：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --load_run 2026-06-17_03-05-16 \
  --checkpoint model_63700.pt
```

## 当前有用 Teacher

Run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_03-05-16
```

当前锚点 checkpoint：

```text
model_63700.pt
```

相关 play 视频：

```text
/home/lxq/Videos/Kazam_screencast_00083_2026-06-17_03-05-16_model_63700.mp4
```

`model_63700` 附近的状态：

- 目前是 highstep 方向里视觉效果最好的 teacher 候选。
- 平地步态比前面自然，前腿能主动搭台，后腿和 box joint 的推进更像有效动作。
- `Curriculum/terrain_levels` 约突破 `3.0`。
- `Curriculum/command_levels` 已达到当前配置最高 `0.6375`。
- highstep 奖励仍在高位，没有崩。
- `bad_orientation` 没有爆炸。
- 速度误差升高，但在 command 放开后的压力区内还可以接受。

当前决策：

```text
继续 teacher。不要在 teacher 还没跨多个 checkpoint 稳定前切 student。
```

如果后续 teacher checkpoint 变差，退回：

```text
2026-06-17_03-05-16/model_63700.pt
```

## 当前 Highstep 配置重点

配置文件：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
```

当前 command 范围：

```text
lin_vel_x = (0.20, 0.85)
lin_vel_y = (-0.35, 0.35)
ang_vel_z = (-0.60, 0.60)
range_multiplier = (0.35, 0.75)
gated_multiplier = 0.55
terrain_gate_level = 2.6
```

有效 command levels：

```text
初始 x max = 0.85 * 0.35 = 0.2975
中档 x max = 0.85 * 0.55 = 0.4675
最终 x max = 0.85 * 0.75 = 0.6375
```

当前 terrain curriculum：

```text
stage_update_thresholds = (300, 900, 1800)
stage_max_levels = (2, 3, 5, 8)
```

当前 highstep 地形物理限制：

```text
楼梯单阶高度 = (0.04, 0.22)
楼梯宽度 = 0.30
高台高度 = (0.04, 0.35)
坑深度 = (0.04, 0.35)
```

当前 box action prior 目标范围：

```text
min_box_target = 0.000
max_box_target = 0.060
```

