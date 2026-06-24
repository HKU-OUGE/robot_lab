# 当前状态

更新时间：2026-06-18。

## 当前任务

Teacher 任务名：

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

Student 任务名：

```bash
RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
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
- 当前路线是先修 student：在 student 蒸馏中加入轻微后退样本；如果效果不行，再回头改 teacher。

当前 student-only 代码改动：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg:
    self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.85)
```

该改动备份：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg_2026-06-18_00-02-07.py
```

继续 student 训练命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

当前代码里的 highstep student runner 是 `max_iterations=6000`、`student_actor_warmup_updates=1200`。建议继续训练后先看 1500-2500 update 附近的 checkpoint。

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
