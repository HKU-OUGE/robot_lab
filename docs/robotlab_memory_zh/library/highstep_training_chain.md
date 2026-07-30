# Highstep 训练链路

> **2026-07-30 更新：**当前有效链路已推进到
> `B300 Teacher model_173499 -> E5700 model_179198 -> E7700 model_181198`，
> 并完成有人保护的真机整机上台。完整 SHA、W&B、视频、bag 和结论见
> [highstep_stage_archive_20260730.md](highstep_stage_archive_20260730.md)。
> 下文早期 run 排序保留为历史，不再代表当前候选。

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

当时原计划：先从 `model_69998.pt` 继续 student 蒸馏。如果轻微后退蒸馏仍不能降低后仰/`bad_orientation`，再回头修改 teacher 的 command 分布。

后续判断：这条 student-first 分支不够。继续 student 后 play 更差，后退稳定仍然弱，所以路线切回 teacher 侧先精修，再重新蒸馏最终 student。

## 2026-06-18/19 Teacher 后退稳定与 Box Joint 精修

第一条 student 暴露后退鲁棒性差以后，后续改法转向 teacher，而不是继续指望 student 蒸馏自动补齐。

重要 run：

| Run | 来源 | 意义 |
| --- | --- | --- |
| `2026-06-18_05-10-44_teacher_backward_stability_long_from_68199` | `2026-06-17_05-07-05/model_68199.pt` | 加入 teacher 后退 command 覆盖和 pitch 稳定。后退 play 安全很多，但平地/正常运动时 box joint 位置异常。 |
| `2026-06-18_21-19-11_teacher_box_default_gate_from_80198` | 后续 teacher checkpoint | 加强 box joint 默认位门控。box joint 有改善，但 terrain progress 和 command 释放仍受门控影响。 |
| `2026-06-18_23-20-22_teacher_box_default_resume_gate_free_from_82100` | `model_82100.pt` | resume/refine 时放宽 command terrain gate。terrain 到约 2.78，`bad_orientation` 更低，但上高台能力比旧高风险 run 弱。 |
| `2026-06-19_01-42-04` | `model_84400.pt` | 当前精修分支，降低高台阶段 box 默认位惩罚。记忆更新时最新 checkpoint 是 `model_86400.pt`；稳定，但 terrain levels 仍在约 2.8。 |

当前 teacher 侧改动：

- teacher command 现在覆盖轻微后退：`lin_vel_x=(-0.30, 0.85)`；
- `backward_pitch_stability` 用于惩罚后退时后仰/抬身；
- `non_forward_highstep_pitch` 继续约束非前进场景的 pitch；
- `prismatic_joint_pos_penalty` 改为 `highstep_box_default_position_penalty`；
- box joint 在非主动上高台时强力靠近默认位，前进高台时放松惩罚（`hold_scale=16.0`，`highstep_scale=0.08`）；
- `scripts/rsl_rl/base/train.py` 对 highstep teacher 的 resume/refine 自动把 `terrain_gate_level` 放宽到 `0.0`，fresh 训练仍保留门控。

当前判断：

- 第一条 student 分支不能作为最终部署 policy；
- 不要为了追旧的高 `terrain_levels` 牺牲 box joint 平地自然性；
- 继续 teacher 精修，直到 play 同时确认平地、后退和上高台都可接受；
- 之后再重新蒸馏新的 student。

## 2026-06-22/23 真机高台后腿分支精修

2026-06-22 的真机高台问题推动了仿真中 35 cm 高台能力增强，但仍存在后腿卡台、sim-to-real 成功率不够的问题。随后 `2026-06-23_00-52-26` 训练暴露了更具体的失败模式：

```text
右后腿先登台分支：通常更顺，更容易成功
左后腿先登台分支：容易卡住；到 model_105397 时可能变成主导失败动作
```

相关视频：

```text
Kazam_screencast_00095_2026-06-23_00-52-26_model_97300.mp4
Kazam_screencast_00096_2026-06-23_00-52-26_model_105397.mp4
Kazam_screencast_00096_2026-06-23_00-52-26_model_105397_new.mp4
```

解释：

- `model_97300` 仍能上高台，可作为对照，但重新核对 run 继承关系后，它不再是优先父 checkpoint。
- `2026-06-23_00-52-26` 是从 `2026-06-22_12-41-28/model_95398.pt` resume 的；因此 `model_97300` 已经包含约 1900 update 的旧奖励 shaping，早于 second-rear-foot / worst-rear-foot 修复。
- 当前代码更干净的父 checkpoint 是 `2026-06-22_12-41-28/model_95398.pt`。
- `model_105397` 仍有右后腿先登台的成功案例，但不适合作为继续训练来源，因为高概率行为太容易变成更弱的左后腿先登台序列。
- 标量奖励会骗人：后腿 clearance、rear drive、box push、phase prior 都可能上升，但左右后腿分支可靠性反而变差。

奖励设计经验：

```text
不要把“一条后腿登上高台”当作完整成功来给高分。
一条后腿登台后，奖励必须继续推动第二条后腿也越过台阶边缘。
后腿卡台惩罚应该按最差后腿计算，而不是两条后腿取平均。
```

当前建议：

```text
从这里继续 teacher 精修：
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt

这个只作为对照/备选：
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-23_00-52-26/model_97300.pt
```
