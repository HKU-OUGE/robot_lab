# Current State

Last updated: 2026-06-17.

## Active Task

Task:

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

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

## Latest Useful Teacher

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

Decision:

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

