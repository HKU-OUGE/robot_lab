# Current State

Last updated: 2026-06-18.

## Active Task

Teacher task:

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

Student task:

```bash
RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
```

## 2026-06-18 Student And Deployment Update

The current teacher source for highstep student distillation is:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_05-07-05/model_68199.pt
```

The current temporary student checkpoint is:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

Student `model_69998.pt` status:

- high platform ability is barely usable;
- flat/approach gait is not final quality;
- `bad_orientation` is worse than teacher;
- keyboard backward walking is a clear weak point and can rear up/fall;
- use this student only for temporary Mujoco probing, not as a final deployable checkpoint.

Latest scalar comparison:

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

Diagnosis:

- the original highstep command range was forward-only: `lin_vel_x=(0.20, 0.85)`;
- teacher can sometimes generalize to keyboard backward commands because it has privileged latent/PPO-trained robustness;
- student is a compressed blind policy and stage 2 does not run PPO (`Student_PPO_Adapt_Enabled=0`), so OOD backward commands degrade more severely;
- fix student first by adding mild backward samples to student distillation; retrain teacher only if that does not work.

Current student-only code change:

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg:
    self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.85)
```

Backup for that edit:

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg_2026-06-18_00-02-07.py
```

Continue student with:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

Current code uses `max_iterations=6000` and `student_actor_warmup_updates=1200` for the highstep student runner. Check 1500-2500 updates into the continuation before judging.

Temporary Mujoco export from the current student run:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/exported/policy_student.pt
```

This file accepts the deployment stack's 570-D blind observation and returns 16 actions. Do not use `exported/policy.pt` for Mujoco; it expects 634-D actor input with latent.

Normal training command:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless
```

Resume from current best teacher anchor:

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

## Historical Useful Teacher Anchor

Run:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_03-05-16
```

Current anchor checkpoint:

```text
model_63700.pt
```

Relevant play video:

```text
/home/lxq/Videos/Kazam_screencast_00083_2026-06-17_03-05-16_model_63700.mp4
```

Observed status around `model_63700`:

- Visual action is currently the best highstep direction: more natural flat walking, front legs can step onto high platform, rear legs and box joints contribute more plausibly.
- `Curriculum/terrain_levels` crossed about `3.0`.
- `Curriculum/command_levels` reached the current configured max `0.6375`.
- Highstep rewards were in high range, not collapsed.
- `bad_orientation` was not exploding.
- Velocity errors were elevated but tolerable for the new full-command phase.

Historical decision at that time:

```text
Continue teacher first. Do not switch to student yet unless teacher stays visually stable across several later checkpoints.
```

If later teacher checkpoints degrade, return to:

```text
2026-06-17_03-05-16/model_63700.pt
```

## Current Highstep Config Highlights

Config file:

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
```

Current command range:

```text
lin_vel_x = (0.20, 0.85)
lin_vel_y = (-0.35, 0.35)
ang_vel_z = (-0.60, 0.60)
range_multiplier = (0.35, 0.75)
gated_multiplier = 0.55
terrain_gate_level = 2.6
```

Effective command levels:

```text
initial x max = 0.85 * 0.35 = 0.2975
gated  x max = 0.85 * 0.55 = 0.4675
final  x max = 0.85 * 0.75 = 0.6375
```

Current terrain curriculum:

```text
stage_update_thresholds = (300, 900, 1800)
stage_max_levels = (2, 3, 5, 8)
```

Current highstep terrain physical limits:

```text
stairs step_height_range = (0.04, 0.22)
stairs step_width = 0.30
box_height_range = (0.04, 0.35)
pit_depth_range = (0.04, 0.35)
```

Current action-prior box target range:

```text
min_box_target = 0.000
max_box_target = 0.060
```
