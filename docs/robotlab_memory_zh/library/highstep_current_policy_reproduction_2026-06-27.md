# Highstep 当前 Teacher Policy 完整复现链路

更新时间：2026-06-27，Asia/Hong_Kong。

本文档记录“当前正在训练的 highstep teacher policy”的完整父链、任务、地形、关键配置和复现流程。目标是让另一个使用者把本文件交给 Codex 后，可以理解这条 policy 不是从 highstep 随机初始化直接训出来的，而是从 bodyflat 0 轮起步，经过多段 checkpoint 继承、highstep 迁移和后续精修得到的。

## 当前基准

当前宿主机进程确认结果：

```text
PID: 2744554
command:
python scripts/rsl_rl/base/train.py
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
  --logger wandb
  --headless
  --resume
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_19-07-54/model_101200.pt
  --max_iterations 4000

current run:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-27_00-27-17

latest saved checkpoint when this file was written:
model_101400.pt
```

注意：`--max_iterations 4000` 表示从 `model_101200.pt` 额外训练 4000 个 update，不是训练到 `model_4000.pt`。

## 复现原则

最可靠复现方式不是只看当前源码，而是每一段都使用该 run 自己保存的：

```text
params/agent.yaml
params/env.yaml
git/robot_lab.diff
model_*.pt
events.out.tfevents.*
```

每个 run 的 `params/env.yaml` 是当时实际进入训练的环境快照；`params/agent.yaml` 是当时实际的 RSL-RL/PPO 配置；`git/robot_lab.diff` 是当时记录的代码差异。历史链路经历过多次代码修改，不能简单用 2026-06-27 当前源码从 0 轮直接重训并期望得到相同策略。

若没有历史 checkpoint，只能近似复现：按下面阶段顺序重新训练，并在相同 checkpoint 编号附近做 play/video 筛选。由于随机性、Isaac Sim 版本、GPU、驱动、手动中止时机和代码快照差异，不能保证 bitwise 相同。

## 新电脑无 checkpoint 时的从 0 压缩复现方案

适用场景：

```text
另一台电脑没有本机当前 checkpoint，也没有历史父链 checkpoint。
目标不是 bitwise 复刻当前 model_101200 / model_101400，而是让对方用当前代码和相同任务配置，从 0 训练出一条行为接近当前 highstep teacher 的新链路。
```

硬性说明：

```text
1. 没有 checkpoint 时，不能直接复现当前 policy，只能重新训练一条新分支。
2. 不建议从 0 直接训练 Highstep；当前主线依赖 Bodyflat 预训练。
3. 不建议机械复刻下面“完整父链总览”的 21 个历史 run；那些 run 含有多次人工筛选、中止、代码变化和失败分支排除。
4. 新电脑应采用压缩链路：Bodyflat 从 0 预训练 -> Bodyflat 稳定化 -> Highstep 迁移 -> Highstep refine。
5. 所有训练命令使用 robot_lab/scripts/rsl_rl/base/train.py，不使用 IsaacLab 原始 train.py。
6. 不加 --run_name，除非使用者明确需要自定义 WandB 名称。
7. 长训不要加 --video；之前已有录视频导致电脑卡死的经验。
8. --max_iterations N 表示本次 invocation 从当前起点额外训练 N 个 update，不是训练到 model_N。
```

### A. Bodyflat fresh，从 0 建立基础步态

命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --max_iterations 17000
```

这一段是从 0 随机初始化开始训练。历史链路第一段也是 Bodyflat fresh，曾在 `model_12400.pt` 附近被选为后续父模型，但新电脑不必强行在 12400 停止。建议先跑到 `model_17000.pt` 附近，检查：

```text
1. 平地和普通地形行走是否稳定。
2. bad_orientation 是否处于低水平。
3. time_out 是否接近 1。
4. command tracking 是否已经基本稳定。
5. box joint 是否没有在普通运动中出现离谱偏移。
```

如果 `model_17000.pt` 表现明显不稳，不要切 Highstep，继续 Bodyflat。

### B. Bodyflat 稳定化，靠近历史 highstep 父节点质量

命令模板：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae/<新电脑bodyflat时间戳>/model_17000.pt \
  --max_iterations 17000
```

如果第二段结束后仍不够稳，可以继续用同样命令从最新 bodyflat checkpoint 接着跑。历史主线用于 Highstep 迁移的 bodyflat 父模型大约是 `model_49600.pt`，所以新电脑比较稳妥的目标是：

```text
最低可尝试切 highstep：model_17000 到 model_34000 之间，前提是 play 稳定。
更接近历史链路：model_45000 到 model_50000 附近。
```

推荐优先级：

```text
如果时间足够：Bodyflat 训到 45000-50000 附近再切 Highstep。
如果只想先快速验证链路：Bodyflat 17000-34000 稳定后可以先切 Highstep，但要接受成功率和动作质量不如当前主线的风险。
```

### C. Highstep 迁移，使用当前温和 terrain 和当前 reward 结构

从选中的最佳 bodyflat checkpoint 切换到 Highstep teacher：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae/<新电脑bodyflat时间戳>/<best_bodyflat_model>.pt \
  --max_iterations 4500
```

这一段对应历史链路里从 `bodyflat model_49600` 切到 highstep 的迁移阶段。新电脑必须使用当前 Highstep 任务：

```text
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

当前 Highstep terrain 应保持温和配置：

```text
pyramid_stairs:      0.30, step_height_range (0.04, 0.22)
pyramid_stairs_inv:  0.15, step_height_range (0.04, 0.22)
box:                 0.25, box_height_range (0.04, 0.35)
pit:                 0.15, pit_depth_range (0.04, 0.35)
hf_pyramid_slope:    0.03, slope_range (0.0, 0.3)
hf_pyramid_slope_inv:0.03, slope_range (0.0, 0.3)
random_rough:        0.09, noise_range (0.01, 0.04)
box_hard:            不启用
```

当前 Highstep command 应保持：

```text
lin_vel_x: (-0.30, 0.85)
lin_vel_y: (-0.35, 0.35)
ang_vel_z: (-0.60, 0.60)
rel_standing_envs: 0.05
```

当前 action 中 box bias 应保持：

```text
front_reach_box_bias: -0.020
rear_approach_box_bias: 0.002
front_support_box_bias: 0.002
rear_push_box_bias: -0.022
min_box_target: 0.0
max_box_target: 0.06
commit_height_delta_min: 0.04
commit_height_delta_target: 0.18
commit_gate_floor: 0.0
```

迁移段的判断标准：

```text
1. terrain_levels 不需要立刻达到历史最高，但应能持续上升，不应早早完全平台。
2. command_levels 应能逐步进入 0.4675 到 0.6375 区间。
3. bad_orientation 不应明显恶化。
4. 后腿上台相关 reward 应逐步出现非零。
5. play 中应开始出现前腿搭台、后腿尝试抬高和上台动作。
```

### D. Highstep refine，继续形成当前主线类似行为

第一段 highstep 迁移后，从该段最好的 highstep checkpoint 继续：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/<新电脑highstep时间戳>/<best_highstep_model>.pt \
  --max_iterations 8000
```

如果这段之后 play 中已经能上 0.32-0.35 m 普通 box，但后腿仍有拖边、卡边、第一条后腿抬不高，可以继续一段 refine：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/<上一段highstep时间戳>/<best_highstep_model>.pt \
  --max_iterations 4000
```

这段用于观察当前后腿阶段任务是否起作用，重点看：

```text
Episode_Reward/rear_first_foot_highstep_preclearance
Episode_Reward/rear_second_foot_highstep_clearance
Episode_Reward/second_rear_clear_deadline
Episode_Reward/one_sided_rear_stall_time
Episode_Reward/lead_rear_support_drive
Curriculum/highstep_rear_branch_metrics/valid_rate
Curriculum/highstep_rear_branch_metrics/second_clear_rate
Curriculum/highstep_rear_branch_metrics/one_sided_stall_ratio
```

如果这些 tag 长时间不存在或全 0，不能盲目长训，应先检查 reward 是否被注册、stage gate 是否打开、branch buffer 是否正常更新。

### E. 从 0 新链路的推荐执行顺序

推荐顺序：

```text
1. Bodyflat fresh: 0 -> 17000
2. Bodyflat continue: 17000 -> 34000
3. 若 bodyflat play 仍不稳，继续到 45000-50000
4. Highstep migration: best_bodyflat -> +4500
5. Highstep refine: best_highstep -> +8000
6. Play 普通 box level10，观察 0.322-0.350 m 高台
7. 若后腿第一条腿仍抬高不足，再从 best_highstep 继续 +4000 或 +8000
```

最小可运行版本：

```text
Bodyflat 17000 -> Highstep 4500 -> Highstep refine 8000
```

更接近当前主线的版本：

```text
Bodyflat 45000-50000 -> Highstep 4500 -> Highstep 8000 -> Highstep 4000/8000
```

### F. 新电脑复现时的停止和筛选原则

不要只按固定轮次机械推进。每一阶段都应保存 checkpoint 并筛选：

```text
Bodyflat 阶段：
  以稳定行走、低 bad_orientation、box joint 不乱偏为主要标准。

Highstep 初期：
  以能否形成前腿搭台、后腿抬高、后腿上台尝试为主要标准。

Highstep 中后期：
  以 box level10 play 视频为主；terrain_levels 只能辅助判断。

后腿专项：
  看第一条后腿是否更容易搭上台；
  看已登台后腿是否有支撑身体往上蹬的动作；
  看第二条后腿是否能在较短时间内清台；
  看 one_sided_stall_ratio 是否下降。
```

如果出现以下情况，应暂停继续长训：

```text
1. 平地或普通运动中 box joint 明显异常偏移。
2. bad_orientation 明显升高。
3. terrain_levels 很早平台，且 play 中也没有上高台能力。
4. branch/rear reward 指标长期不存在或全 0。
5. play 中高台动作比上一可用 checkpoint 明显退化。
```

## 完整父链总览

当前 policy 的祖先链路从 bodyflat 0 轮开始：

| 阶段 | 任务 | run | 输入 checkpoint | 本链路选用输出 | agent max_iterations | 说明 |
| --- | --- | --- | --- | --- | ---: | --- |
| 0 | Bodyflat | `arclab_arcdog_adjustable_leg_bodyflat_vae/2026-05-08_08-59-28` | fresh / `resume=false` | `model_12400.pt` | 17000 | 从 0 轮开始的最早父 run。 |
| 1 | Bodyflat Teacher | `arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-05-11_09-20-46` | `bodyflat_vae/2026-05-08_08-59-28/model_12400.pt` | `model_25300.pt` | 17000 | 迁移到 Teacher 实验名。 |
| 2 | Bodyflat Teacher | `2026-05-11_21-33-05` | `2026-05-11_09-20-46/model_25300.pt` | `model_27000.pt` | 17000 | 短接续。 |
| 3 | Bodyflat Teacher | `2026-05-31_16-58-47` | `2026-05-11_21-33-05/model_27000.pt` | `model_29600.pt` | 17000 | bodyflat/sidestep 后续基线。 |
| 4 | Bodyflat Teacher | `2026-06-03_21-47-54` | `2026-05-31_16-58-47/model_29600.pt` | `model_45500.pt` | 17000 | 该 run 最后到 `46599`，本链路选 `45500`。 |
| 5 | Bodyflat Teacher | `2026-06-04_13-18-36` | `2026-06-03_21-47-54/model_45500.pt` | `model_49600.pt` | 17000 | highstep 主链的 bodyflat 父模型。 |
| 6 | Highstep Teacher | `2026-06-15_23-16-43` | `bodyflat_Teacher/2026-06-04_13-18-36/model_49600.pt` | `model_52200.pt` | 4500 | highstep 迁移开始。 |
| 7 | Highstep Teacher | `2026-06-16_02-44-57` | `2026-06-15_23-16-43/model_52200.pt` | `model_53200.pt` | 4500 | 短接续。 |
| 8 | Highstep Teacher | `2026-06-16_04-22-56` | `2026-06-16_02-44-57/model_53200.pt` | `model_57699.pt` | 4500 | 视觉上开始能上 highstep。 |
| 9 | Highstep Teacher | `2026-06-16_22-07-48` | `2026-06-16_04-22-56/model_57699.pt` | `model_60400.pt` | 4500 | 加强接近高台/逆向地形分布。 |
| 10 | Highstep Teacher | `2026-06-17_00-56-56` | `2026-06-16_22-07-48/model_60400.pt` | `model_61900.pt` | 4500 | terrain levels 上升，command 仍低。 |
| 11 | Highstep Teacher | `2026-06-17_03-05-16` | `2026-06-17_00-56-56/model_61900.pt` | `model_63700.pt` | 4500 | command 放宽到约 `0.6375`。 |
| 12 | Highstep Teacher | `2026-06-17_05-07-05` | `2026-06-17_03-05-16/model_63700.pt` | `model_68199.pt` | 4500 | 第一版较强 highstep teacher。 |
| 13 | Highstep Teacher | `2026-06-18_05-10-44_teacher_backward_stability_long_from_68199` | `2026-06-17_05-07-05/model_68199.pt` | `model_80198.pt` | 12000 | 加后退 command 覆盖和 pitch 稳定。 |
| 14 | Highstep Teacher | `2026-06-18_21-19-11_teacher_box_default_gate_from_80198` | `2026-06-18_05-10-44.../model_80198.pt` | `model_82100.pt` | 5000 | 修平地/普通运动 box joint 异常。 |
| 15 | Highstep Teacher | `2026-06-18_23-20-22_teacher_box_default_resume_gate_free_from_82100` | `2026-06-18_21-19-11.../model_82100.pt` | `model_84400.pt` | 3000 | highstep resume 放宽 command terrain gate。 |
| 16 | Highstep Teacher | `2026-06-19_01-42-04` | `2026-06-18_23-20-22.../model_84400.pt` | `model_87399.pt` | 3000 | 真机前的稳定 teacher 来源之一。 |
| 17 | Highstep Teacher | `2026-06-22_12-41-28` | `2026-06-19_01-42-04/model_87399.pt` | `model_95398.pt` | 8000 | 当前主线的干净父节点。 |
| 18 | Highstep Teacher | `2026-06-26_16-43-13` | `2026-06-22_12-41-28/model_95398.pt` | `model_97200.pt` | 16000 | 修复 branch buffer 后的短训主线。 |
| 19 | Highstep Teacher | `2026-06-26_19-07-54` | `2026-06-26_16-43-13/model_97200.pt` | `model_101200.pt` | 16000 | 101200 视频对应策略来源。 |
| 20 | Highstep Teacher | `2026-06-27_00-27-17` | `2026-06-26_19-07-54/model_101200.pt` | 当前运行中，已见 `model_101400.pt` | 4000 | 加入第一后腿预清台 reward 后的当前 run。 |

以下 run 不是当前 policy 的祖先，只作为失败或对照参考，不要混进复现主链：

```text
2026-06-23_00-52-26/model_97300.pt
2026-06-23_00-52-26/model_105397.pt
2026-06-24_00-36-54/*
2026-06-24_22-59-55/*
2026-06-25_05-59-47/*
2026-06-26_03-40-49/model_107400.pt
2026-06-26_15-42-45/*
2026-06-26_16-24-18/*
```

这些分支对分析很重要，但不是当前 `2026-06-27_00-27-17` 的父链。

## 当前 Highstep 任务和环境

当前 teacher 任务名：

```text
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

训练脚本：

```text
cd /home/lxq/Softwares/robot_lab
python scripts/rsl_rl/base/train.py
```

不要使用原始 IsaacLab 的 train.py 路径。当前项目的 highstep 训练使用 RobotLab 中的脚本。

当前 run 的核心环境参数来自：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-27_00-27-17/params/env.yaml
```

核心环境：

```text
seed: 42
num_envs: 4096
decimation: 4
episode_length_s: 20
terrain_type: generator
use_terrain_origins: true
max_init_terrain_level: 1
terrain_generator.num_rows: 10
terrain_generator.num_cols: 20
terrain_generator.curriculum: false
difficulty_range: (0.0, 1.0)
```

当前 highstep terrain 分布：

```text
pyramid_stairs:      proportion 0.30, step_height_range (0.04, 0.22), step_width 0.30, platform_width 3.0
pyramid_stairs_inv:  proportion 0.15, step_height_range (0.04, 0.22), step_width 0.30, platform_width 3.0
box:                 proportion 0.25, box_height_range (0.04, 0.35), double_box false, platform_width 3.0
pit:                 proportion 0.15, pit_depth_range (0.04, 0.35), platform_width 3.0
hf_pyramid_slope:    proportion 0.03, slope_range (0.0, 0.3), platform_width 2.0
hf_pyramid_slope_inv:proportion 0.03, slope_range (0.0, 0.3), platform_width 2.0
random_rough:        proportion 0.09, noise_range (0.01, 0.04), noise_step 0.005
```

当前主线已去掉 `box_hard`。普通 `box` 最高高度为 0.35 m。

当前 command：

```text
base_velocity.resampling_time_range: (5.0, 10.0)
rel_standing_envs: 0.05
heading_command: false
lin_vel_x: (-0.30, 0.85)
lin_vel_y: (-0.35, 0.35)
ang_vel_z: (-0.60, 0.60)
```

当前 action：

```text
class_type: robot_lab.tasks.locomotion.velocity.mdp.actions:PhasedHighstepBoxBiasJointPositionAction
joint_names:
  FL/FR/RL/RR hip
  FL/FR/RL/RR thigh
  FL/FR/RL/RR calf
  FL/FR/RL/RR box

front_reach_box_bias: -0.020
rear_approach_box_bias: 0.002
front_support_box_bias: 0.002
rear_push_box_bias: -0.022
min_box_target: 0.0
max_box_target: 0.06
commit_height_delta_min: 0.04
commit_height_delta_target: 0.18
commit_gate_floor: 0.0
```

永久方向约定：

```text
box_joint 数值越小，腿越长。
box_joint 数值越大，腿越短。
当前可用范围约 0.00 到 0.06。
```

## 当前 Agent / PPO 配置

当前 run 的 agent 参数来自：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-27_00-27-17/params/agent.yaml
```

核心参数：

```text
runner: OnPolicyRunner
policy: VAEActorCritic
algorithm: VAEPPO
device: cuda:0
num_steps_per_env: 24
save_interval: 100
empirical_normalization: false
actor_hidden_dims: [512, 256, 128]
critic_hidden_dims: [512, 256, 128]
activation: elu
init_noise_std: 1.0
vae_latent_dim: 64
vae_hidden_dims: [256, 128]
distill_stage: 1

num_learning_epochs: 5
num_mini_batches: 4
learning_rate: 3e-4
schedule: adaptive
gamma: 0.99
lam: 0.95
entropy_coef: 0.006
desired_kl: 0.01
clip_param: 0.2
max_grad_norm: 1.0
```

当前是 teacher/VAE-PPO 训练，不是 student 蒸馏训练。虽然 `agent.yaml` 中保留了 student 相关字段，这些字段属于同一个 `VAEActorCritic` 配置对象，并不表示当前 run 是 student policy 阶段。

## 当前 Highstep 关键 Reward

当前高台相关 reward 重点如下。完整列表以当前 run 的 `params/env.yaml` 为准。

```text
front_legs_reach: weight 0.75
front_feet_highstep_clearance: weight 0.40
rear_feet_highstep_clearance: weight 1.15
  clearance_margin 0.045
  clearance_window 0.18
  single_rear_weight 0.20
  min_rear_weight 0.50
  rear_x_weight 0.05
  commit_gate_scale 0.90

rear_first_foot_highstep_preclearance: weight 0.45
  clearance_margin 0.06
  clearance_window 0.16
  first_clear_min 0.55
  second_clear_suppress_min 0.72
  commit_gate_min 0.25
  commit_gate_scale 0.90
  stage_start_update 450
  stage_ramp_updates 500

rear_feet_under_step_after_commit: weight -0.50
  worst_rear_weight 0.82
  clearance_margin 0.02

rear_second_foot_highstep_clearance: weight 0.95
  branch_lead_threshold 0.45
  branch_second_clear_threshold 0.60
  branch_one_sided_gap 0.12

second_rear_clear_deadline: weight 0.65
  second_clear_min 0.58
  deadline_steps 36
  deadline_grace_steps 8

one_sided_rear_stall_time: weight -0.55
  lead_threshold 0.45
  one_sided_gap 0.14
  grace_steps 14
  ramp_steps 28

lead_rear_support_drive: weight 0.70
  rear_push_target 0.003
  target_std 0.016
  target_height_gain 0.09
  target_forward_vel 0.28

highstep_forward_progress: weight 1.60
highstep_body_lift: weight 0.85
highstep_base_advance_lift: weight 1.75
rear_legs_drive_bonus: weight 0.90
highstep_rear_push_posture: weight 1.30
highstep_rear_box_push: weight 1.10
highstep_bridge_stall_penalty: weight -1.10
highstep_box_phase_prior: weight 0.85
```

重要细节：

- `rear_first_foot_highstep_preclearance` 是 2026-06-27 当前 run 新增的 reward，用于解决 101200 视频中“第一条后腿抬高不够、要反复试探”的问题。
- 当前 `train.py` 的 highstep resume 放宽名单尚未包含 `rear_first_foot_highstep_preclearance`，所以当前 run 的 `env.yaml` 中它仍是 `stage_start_update=450`，不是第 0 个 update 立即打开。
- 其余已在 `train.py` staged list 中的 highstep reward 在 `--resume` 时会被放宽为 `stage_start_update=0, stage_ramp_updates=1`。

## 当前 Curriculum

当前 curriculum：

```text
terrain_levels:
  func: terrain_levels_vel_highstep
  move_up_distance: 3.2
  move_down_command_factor: 0.28
  move_down_min_distance: 0.35
  move_down_max_distance: 1.2
  min_height_gain: 0.015
  height_gain_required_level: 2
  climb_up_distance: 0.5
  climb_height_gain: 0.025
  climb_height_required_level: 2
  climb_hold_distance: 0.35
  climb_hold_height_gain: 0.015
  stage_update_thresholds: (300, 900, 1800)
  stage_max_levels: (2, 3, 5, 8)
  num_steps_per_update: 24

command_levels:
  func: command_levels_vel_highstep
  range_multiplier: (0.35, 0.75)
  gated_multiplier: 0.55
  terrain_gate_level: 0.0
  delta: 0.03
  reward_threshold: 0.68

highstep_rear_branch_metrics:
  min_commit_steps: 1
```

`terrain_levels` 不是“能爬高台高度”的直接数值，只是 curriculum 难度 row 的平均占用指标。最终高台能力必须用 play/video 判断。

## 关键指标参考

以下为每段 run 尾部 100 个 event 标量的均值，方便复现实验时做 sanity check：

| run | checkpoint 范围 | terrain | command | bad_orientation | time_out | mean_reward | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| bodyflat `2026-05-08_08-59-28` | 0 -> 12400 | 5.682 | 1.000 | 0.00497 | 0.995 | 137.84 | fresh bodyflat |
| bodyflat `2026-06-04_13-18-36` | 45500 -> 49600 | 5.391 | 1.000 | 0.00709 | 0.993 | 195.91 | highstep 父模型 |
| highstep `2026-06-15_23-16-43` | 49600 -> 52200 | 2.516 | 0.4675 | 0.00383 | 0.996 | 179.20 | highstep 起步 |
| highstep `2026-06-17_05-07-05` | 63700 -> 68199 | 3.274 | 0.6375 | 0.00831 | 0.992 | 171.36 | 早期强 highstep |
| highstep `2026-06-19_01-42-04` | 84400 -> 87399 | 2.812 | 0.6375 | 0.00073 | 0.999 | 167.34 | 真机前稳定父节点 |
| highstep `2026-06-22_12-41-28` | 87400 -> 95398 | 2.834 | 0.6375 | 0.00132 | 0.999 | 163.74 | 当前主线干净父节点 |
| highstep `2026-06-26_16-43-13` | 95400 -> 97200 | 2.710 | 0.6375 | 0.00127 | 0.999 | 163.60 | branch buffer 后 |
| highstep `2026-06-26_19-07-54` | 97200 -> 101200 | 2.773 | 0.6375 | 0.00130 | 0.999 | 165.00 | 101200 视频来源 |
| highstep `2026-06-27_00-27-17` | 101200 -> 101400 | 1.251 | 0.4438 | 0.00118 | 0.999 | 167.38 | 刚重启，早期 curriculum 尚未恢复 |

当前 run 刚从 `101200` 重新开短训，tail100 的 `terrain_levels/command_levels` 低于父 run 是正常现象，不能直接拿来判定退化。

当前 run 新增项早期指标：

```text
Episode_Reward/rear_first_foot_highstep_preclearance tail100 = 0
原因：当前 env.yaml 中该项 stage_start_update=450，写本文档时 run 只到约 model_101400，仍可能未打开。
```

## 逐段训练命令模板

以下命令是复现实验结构用的模板。真实路径按自己的机器调整。每段的 `--max_iterations` 是该次 invocation 的额外训练上限；历史 run 很多是手动中止后选用中间 checkpoint，所以需要在目标 checkpoint 保存后停止。

### 0. Bodyflat fresh，从 0 轮开始

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --max_iterations 17000
```

目标输出：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae/2026-05-08_08-59-28/model_12400.pt
```

### 1. Bodyflat Teacher 接续到 49600

按下面 checkpoint 顺序接续。任务仍是 Bodyflat：

```text
model_12400 -> 25300
model_25300 -> 27000
model_27000 -> 29600
model_29600 -> 45500
model_45500 -> 49600
```

命令形式：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint <上一段选中的 model_xxx.pt> \
  --max_iterations 17000
```

最终用于 highstep 迁移的 checkpoint：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
```

### 2. Highstep 迁移，从 49600 到 68199

切换任务：

```bash
python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint <上一段选中的 model_xxx.pt> \
  --max_iterations 4500
```

选用 checkpoint 顺序：

```text
bodyflat model_49600 -> highstep 2026-06-15_23-16-43/model_52200
model_52200 -> 2026-06-16_02-44-57/model_53200
model_53200 -> 2026-06-16_04-22-56/model_57699
model_57699 -> 2026-06-16_22-07-48/model_60400
model_60400 -> 2026-06-17_00-56-56/model_61900
model_61900 -> 2026-06-17_03-05-16/model_63700
model_63700 -> 2026-06-17_05-07-05/model_68199
```

### 3. 后退稳定与 box joint 平地约束，从 68199 到 87399

使用 Highstep task，继续接续：

```text
model_68199 -> 2026-06-18_05-10-44_teacher_backward_stability_long_from_68199/model_80198
model_80198 -> 2026-06-18_21-19-11_teacher_box_default_gate_from_80198/model_82100
model_82100 -> 2026-06-18_23-20-22_teacher_box_default_resume_gate_free_from_82100/model_84400
model_84400 -> 2026-06-19_01-42-04/model_87399
```

对应 `--max_iterations`：

```text
68199 -> 80198: 12000
80198 -> 82100: 5000
82100 -> 84400: 3000
84400 -> 87399: 3000
```

### 4. 当前干净主线，从 87399 到 101200

```text
model_87399 -> 2026-06-22_12-41-28/model_95398   max_iterations 8000
model_95398 -> 2026-06-26_16-43-13/model_97200   max_iterations 16000
model_97200 -> 2026-06-26_19-07-54/model_101200  max_iterations 16000
```

这些阶段使用当前温和 highstep terrain：

```text
box 0.04-0.35
无 box_hard
```

### 5. 当前正在跑的 101200 后短训

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_19-07-54/model_101200.pt \
  --max_iterations 4000
```

这段会生成当前 run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-27_00-27-17
```

## 当前代码修改脉络

当前主线相比 2026-06-19 之前，关键修改包括：

```text
1. teacher command 覆盖轻微后退：
   lin_vel_x = (-0.30, 0.85)

2. 加入 backward_pitch_stability / non_forward_highstep_pitch，降低键盘后退后仰风险。

3. box joint 非高台阶段默认位约束：
   平地和普通运动时 box joint 不应离谱偏移；
   highstep 主动上台阶段放松 box 默认位惩罚。

4. highstep resume/refine:
   train.py 在 --resume 且 task 为 highstep teacher/student 时，
   command_levels.terrain_gate_level 自动改为 0.0，
   已列入 staged_reward_names 的 highstep reward 立即打开。

5. terrain 回到温和配置：
   box 0.04-0.35, proportion 0.25；
   无 box_hard。

6. 后腿阶段任务：
   rear_second_foot_highstep_clearance
   second_rear_clear_deadline
   one_sided_rear_stall_time
   lead_rear_support_drive

7. branch buffer device 修复：
   使用 torch.device(env.device)，避免每次调用误清空 branch buffer。

8. 2026-06-27 当前新增：
   rear_first_foot_highstep_preclearance，用于第一条后腿预抬高；
   rear_feet_highstep_clearance.rear_x_weight 从 0.10 降到 0.05。
```

## 复现时必须避免的错误

```text
1. 不要把 2026-06-23、2026-06-24、2026-06-25 的失败分支混进当前父链。
2. 不要从 2026-06-26_03-40-49/model_107400.pt 接当前链路；那段被判定为行为改进无效。
3. 不要用 IsaacLab 原始 train.py；使用 robot_lab/scripts/rsl_rl/base/train.py。
4. 不要在长训时加 --video；之前录视频导致电脑卡死过。
5. 不要把 --max_iterations 理解成训练到某个 model 编号。
6. 如果网络差，优先看本地 events/params/checkpoint，不要只看 WandB 网页。
7. 当前效果最终必须用 play/video 判断，terrain_levels 不能替代视频。
```

## Play 验证命令

使用最高等级普通 box 高台验证当前 teacher：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --debug \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-27_00-27-17/model_101400.pt \
  --play_terrain_type box \
  --play_terrain_level 10
```

如果当前 run 已继续训练，应把 `model_101400.pt` 换成最新保存的 checkpoint。

当前普通 `box level10` 约对应：

```text
0.322 m 到 0.350 m 高台
```

## 需要另一个 Codex 继续检查的点

如果另一个使用者拿到本文件继续训练，应先检查：

```text
1. 当前是否已经跑过 rear_first_foot_highstep_preclearance 的 stage_start_update=450。
2. Event 中是否出现 Episode_Reward/rear_first_foot_highstep_preclearance 非零。
3. one_sided_stall_ratio 是否下降。
4. second_clear_rate 是否保持或上升。
5. 视频中第一条后腿是否少刮台、少反复试探。
6. 已上台后腿蹬起身体的动作是否比 101200 更明显。
7. 平地/后退是否仍稳定，box joint 是否没有平地异常偏移。
```

如果 `rear_first_foot_highstep_preclearance` 长时间仍为 0，不能继续盲训，应先检查 `train.py` 的 highstep resume staged_reward_names 是否需要加入该 reward，或检查它的门控条件是否太严。
