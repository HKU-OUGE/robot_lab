# 当前状态

## 2026-07-30 Highstep 阶段归档（覆盖下文旧状态）

```text
workflow=highstep_b300_rl_preedge_continuation_e7700_20260719
phase=archived_after_successful_real_robot_test
status=archived_after_successful_real_robot_test
effective_updates=7700
active train/eval/play=none
automatic training/evaluation/restart/deployment=false
```

E7700 Student 已在 2026-07-23 有人保护的真机实验中完成整机上高台，并在用户切换
`Fixed Down` 前保持站立。切换后的下压动作使一条后腿被台边挤出，不属于上台失败。

当前唯一完整说明：
[Highstep E7700 真机成功上台阶段归档](highstep_stage_archive_20260730.md)。

下文 2026-06-24 状态仅作历史记录。

更新时间：2026-06-24。

## 2026-06-23 后腿登台分支不对称更新

最新有用 teacher run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-23_00-52-26
```

关键 checkpoint：

```text
2026-06-22_12-41-28/model_95398.pt   当前新后腿分支平衡代码的优先干净重启锚点
2026-06-23_00-52-26/model_97300.pt   只作为对照/备选；它是在旧奖励组合下从 95398 继续约 1900 update 得到的
2026-06-23_00-52-26/model_105397.pt  失败参考 checkpoint，不建议作为继续训练来源
```

视频复查：

```text
/home/lxq/Videos/Kazam_screencast_00095_2026-06-23_00-52-26_model_97300.mp4
/home/lxq/Videos/Kazam_screencast_00096_2026-06-23_00-52-26_model_105397.mp4
/home/lxq/Videos/Kazam_screencast_00096_2026-06-23_00-52-26_model_105397_new.mp4
```

观察结果：

- `model_97300` 仍能上高台。右后腿先登台更顺，左后腿先登台容易小卡一下，但多数还能恢复。
- `model_105397` 不是完全不会上；补充 `_new` 视频确认它还保留右后腿先登台的成功分支。
- 真正失败点是分支概率和分支质量：后期训练让更弱的左后腿先登台分支变成高概率行为。左后腿先登台时，机身 yaw/roll/骨盆几何更容易把另一条后腿卡在台阶立面下。
- 成功上台常常依赖“已经登上高台的那条后腿作为支撑/驱动腿，把身体顶起来或带起骨盆，让另一条后腿越过台阶边缘”。

`model_97300` 到 `model_105397` 的曲线警告：

```text
rear_feet_highstep_clearance 明显上升
rear_legs_drive_bonus / highstep_rear_box_push / highstep_box_phase_prior 上升
但 roll/yaw 惩罚、速度误差、后腿卡台症状也同时变差
```

经验：平均奖励曲线可以显示“后腿更积极”，但真实行为可能已经塌到单侧不可靠策略。

checkpoint 选择：

```text
优先用当前代码从 2026-06-22_12-41-28/model_95398.pt 重新接着训。
不要把 2026-06-23_00-52-26/model_97300.pt 当主起点，除非 95398 新代码重启后明确丢失上高台能力。
不要从 model_105397.pt 继续。
```

原因：

```text
2026-06-23_00-52-26 的 load_checkpoint 就是 2026-06-22_12-41-28/model_95398.pt，所以 model_97300 不是独立更优父节点。
95398 -> 97300 这段是在最新 second-rear-foot / worst-rear-foot 奖励修复之前训出来的。
这段训练提高了 forward progress、base lift、rear box push、phase prior 等奖励，但也来自旧 shaping，可能已经强化“一条后腿上台就算有效”的错误倾向。
```

本次修正决策用到的标量快照：

```text
model_95398: terrain_levels=2.8294, bad_orientation=0.001709, time_out=0.998291, roll_yaw_orientation_penalty=-0.004110
model_97300: terrain_levels=2.3444, bad_orientation=0.000844, time_out=0.999156, roll_yaw_orientation_penalty=-0.006171
model_97300 的短期终止指标更好，但它作为继续训练父节点不够干净，因为它已经包含 95398 之后旧奖励引导下的训练。
```

本次诊断后的代码修改：

```text
rewards.py:
  rear_feet_highstep_clearance_bonus 改为混合 max/mean/min 后足 clearance，不再主要奖励单条后腿上台。
  rear_feet_under_step_after_commit_penalty 改为混合最差后腿和平均后腿，避免一条后腿卡台被平均值掩盖。
  新增 rear_second_foot_highstep_clearance_bonus：任意一条后腿先到高台后，继续奖励第二条后腿也越过边缘。

highstep_env_cfg.py:
  rear_feet_highstep_clearance 权重降到 1.15，single_rear_weight=0.20，min_rear_weight=0.50。
  rear_feet_under_step_after_commit 权重加强到 -0.50，worst_rear_weight=0.82。
  新增 rear_second_foot_highstep_clearance，权重 0.95。
  highstep_rear_box_push 降到 1.10，highstep_box_phase_prior 降到 0.85，避免 box/phase 模板奖励压过真实完成质量。

train.py:
  把 rear_second_foot_highstep_clearance 加入 highstep --resume 阶段奖励立即放开名单。
```

## 当前任务

Teacher 任务名：

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

Student 任务名：

```bash
RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
```

## 2026-06-19 Highstep Teacher 精修状态

`2026-06-17_19-54-10/model_69998.pt` 之后的 student-first 修复不够。student 虽然勉强能上高台，但 `bad_orientation` 更高，键盘后退容易后仰翻倒。当前路线已经切换为：

```text
先强化 teacher 的后退稳定和 box joint 平地行为 -> play 检查 -> 再重新蒸馏 student
```

当前要重点观察的 teacher refinement run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-19_01-42-04
```

本次记忆更新时看到的最新 checkpoint：

```text
model_86400.pt
```

step `86422` 附近的最新 scalar：

```text
terrain_levels last ~= 2.798, avg100 ~= 2.789
command_levels = 0.6375
bad_orientation avg100 ~= 0.00144
time_out avg100 ~= 0.99856
error_vel_xy avg100 ~= 0.19183
mean_reward avg100 ~= 167.02
```

解释：

- 这条 run 比 `2026-06-17_05-07-05` 和第一条 student 稳定很多；
- `terrain_levels` 比旧曲线/bodyflat 曲线低，不能单独判定失败；
- 最新 play 观察到 box joint 暂时没有之前那种离谱偏移；
- 继续判断时要同时看 play/video、`bad_orientation`、`time_out`、速度误差和 box joint 行为。

继续当前 teacher refinement 的命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-19_01-42-04/model_86400.pt
```

## 2026-06-18 Student 与部署更新

当前 highstep student 蒸馏的 teacher 来源：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_05-07-05/model_68199.pt
```

当前临时 student checkpoint：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

`model_69998.pt` 状态：

- 上高台能力勉强够用；
- 平地/接近高台步态还不是最终质量；
- `bad_orientation` 明显高于 teacher；
- 键盘后退是明显弱点，容易后仰并触发终止；
- 只能先作为 Mujoco 临时探测 policy，不能当最终部署版本。

最新 scalar 对比：

```text
teacher 2026-06-17_05-07-05 last100 bad_orientation ~= 0.0083
student 2026-06-17_19-54-10 last100 bad_orientation ~= 0.0144
student latest bad_orientation = 0.014394
student latest time_out = 0.985606
student latest terrain_levels = 2.998851
student latest command_levels = 0.637500
student latest Teacher_Action_MSE = 0.148666
student latest Distill_Latent_MSE = 0.103713
student latest VAE_Vel_MSE = 0.017040
student latest Mu_Out_Of_Bounds_Ratio = 0.124424
```

诊断：

- 原 highstep command 是纯前进分布：`lin_vel_x=(0.20, 0.85)`；
- teacher 偶尔能泛化到键盘后退，是因为 privileged latent/PPO 训练出的鲁棒性更强；
- student 是压缩后的盲走策略，stage 2 不跑 PPO（`Student_PPO_Adapt_Enabled=0`），所以 OOD 后退命令更容易失稳；
- 最初路线是先修 student：在 student 蒸馏中加入轻微后退样本；但 play 变差后确认 teacher 侧也必须覆盖后退稳定。

student 分支结论：

```text
`2026-06-17_19-54-10/model_69998.pt` 只能作为 Mujoco 临时信号测试，不能作为最终部署候选。
```

该改动备份：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg_2026-06-18_00-02-07.py
```

除非明确要复现实验旧分支，否则不要把这条 student 作为主路线继续。优先继续 teacher 精修，再重新蒸馏。

旧 student 分支对比命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

当前代码里的 highstep student runner 是 `max_iterations=6000`、`student_actor_warmup_updates=1200`。这是因为旧 highstep student 的 `1800` iteration 更像 quick deploy 短训，不适合最终稳定蒸馏。

当前 student run 的临时 Mujoco 导出：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/exported/policy_student.pt
```

这个文件接受部署端 570 维盲走观测并输出 16 维 action。不要用 `exported/policy.pt` 跑 Mujoco；它需要 634 维 actor+latent 输入。

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

## 历史有用 Teacher 锚点

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

当时的历史决策：

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
lin_vel_x = (-0.30, 0.85)
lin_vel_y = (-0.35, 0.35)
ang_vel_z = (-0.60, 0.60)
range_multiplier = (0.35, 0.75)
gated_multiplier = 0.55
terrain_gate_level = 2.6；fresh 训练保留门控，resume/refine 时 train.py 自动放宽到 0.0
```

有效 command levels：

```text
初始 x max = 0.85 * 0.35 = 0.2975
中档 x max = 0.85 * 0.55 = 0.4675
最终 x max = 0.85 * 0.75 = 0.6375
最终 x min = -0.30 * 0.75 = -0.225
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

当前 box joint 默认位约束：

```text
prismatic_joint_pos_penalty = highstep_box_default_position_penalty
weight = -2.5
hold_scale = 16.0
highstep_scale = 0.08
```

含义：平地/非前进场景强力拉回默认 box joint；只有前进命令和高台地形门控同时成立时，才释放 box joint 去辅助上高台。
