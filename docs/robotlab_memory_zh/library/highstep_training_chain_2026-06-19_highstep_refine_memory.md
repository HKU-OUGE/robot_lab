# Highstep 训练链路

这个文件记录 highstep 训练主链路，以及 WandB 中哪些 run 值得保留对比。

## 目标

训练 `RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`，让 Arcdog 可伸缩腿机器人能上高台和陡峭台阶。这个任务不优先追求 body flat，允许机器人利用腿部/机体和地形接触辅助上台。

参考 3 月视频中的目标动作：

- 前腿先伸到高台/台阶上；
- 后腿强力推进；
- 后腿 box joint 在推进阶段伸长；
- 机器人身体真正越过高台，而不是只有前腿搭上去卡住；
- 平地行走必须自然，不能一直趴着、乱抬腿或姿态扭曲。

参考视频：

```text
/home/lxq/Videos/2026.03.18/Kazam_screencast_00053.mp4
/home/lxq/Videos/2026.03.18/Kazam_screencast_00059.mp4
```

## 主要有效链路

WandB 中建议保留这些 run：

```text
2026-06-15_23-16-43
2026-06-16_04-22-56
2026-06-16_22-07-48
2026-06-17_00-56-56
2026-06-17_03-05-16
```

概要：

| Run | Checkpoint 范围 | 意义 |
| --- | --- | --- |
| `2026-06-15_23-16-43` | `model_49600` 到 `model_52200` | 重新往视觉正确的 highstep 动作靠近。`model_52200` 虽然 terrain levels 不高，但 play 开始像样。 |
| `2026-06-16_04-22-56` | `model_53200` 到 `model_57699` | 重要视觉基准。`model_57699` 在 play 中已经能上 highstep 地形。 |
| `2026-06-16_22-07-48` | `model_57700` 到 `model_60400` | 加入高低侧 reset/逆向地形/接近高台分布，让训练更接近从低处爬上高台。 |
| `2026-06-17_00-56-56` | `model_60400` 到 `model_61900` | highstep 奖励更明显，terrain 到约 2.82，但 command 还卡在 0.4675。之后训练中视频/桌面卡死。 |
| `2026-06-17_03-05-16` | `model_61900` 到 `model_63700` | 放宽 command gate，command 到 0.6375，terrain 过 3.0，`model_63700` 是当前视觉锚点。 |

## 默认隐藏的 run

这些只作为失败参考，不适合常规对比：

```text
2026-06-15_01-59-10
2026-06-15_06-40-11
2026-06-15_13-03-10
2026-06-16_02-44-57
2026-06-16_21-33-09
```

原因：

- `2026-06-15_01-59-10`：早期失败，terrain 没到 1。
- `2026-06-15_13-03-10`：terrain 曲线不错，但后面 play 姿态很烂，是“曲线骗人”的反例。
- `2026-06-16_21-33-09`：太短，只有一个 checkpoint。

## Student 分支前的历史决策点

截至 `model_63700`，如果时间允许，不要立刻切 student。继续 teacher，并把 `model_63700` 作为保底。

建议后续优先检查：

```text
model_63700
model_64000
model_64500
model_65000
model_65500
```

满足以下条件再切 student：

- 平地行走自然；
- 上高台动作仍然强；
- command 保持 `0.6375`；
- terrain levels 不崩；
- bad orientation 保持低；
- 速度误差没有持续恶化。

## 2026-06-18 Student 蒸馏分支

视觉检查后，`2026-06-17_05-07-05/model_68199.pt` 成为当前实际使用的 highstep student 蒸馏 teacher 来源。第一条 highstep no-prior student 分支：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10
```

重要 checkpoint：

```text
model_69998.pt
```

状态：

- 上高台动作勉强够用；
- `terrain_levels ~= 3.0`，command 到 `0.6375`；
- `bad_orientation` 高于 teacher：student last100 约 `0.0144`，teacher last100 约 `0.0083`；
- 键盘后退是最明显弱点，容易后仰翻车。

判断：

- teacher 训练 command 是纯前进分布：`lin_vel_x=(0.20, 0.85)`；
- teacher 对后退键盘命令的容忍主要来自 PPO/privileged latent 的泛化；
- student 是盲走压缩策略，最先丢失 OOD 后退鲁棒性；
- stage 2 student 当前不跑 PPO，所以 reward 不会直接修复后退后仰。

2026-06-18 已采用 student-first 修复：

```text
ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg:
    self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.85)
```

先从 `model_69998.pt` 继续 student 蒸馏。如果轻微后退蒸馏仍不能降低后仰/`bad_orientation`，再回头修改 teacher 的 command 分布。
