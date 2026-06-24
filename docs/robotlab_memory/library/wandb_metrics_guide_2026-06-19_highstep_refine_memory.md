# WandB Metrics Guide

## Terrain Levels

`Curriculum/terrain_levels` is useful but not decisive.

For highstep:

- it is averaged across many environments and terrain types;
- it is affected by reset distribution;
- it can stay modest even when play shows a usable highstep skill;
- it can also look good while play becomes visually bad.

Keep `terrain_levels`, but always compare with video/play.

Known example:

```text
2026-06-15_13-03-10 had relatively high terrain levels but poor visual behavior later.
2026-06-16_04-22-56 had modest terrain levels but produced usable highstep behavior.
```

## Command Levels

`Curriculum/command_levels` returns current x-velocity upper bound, not a normalized 0-to-1 score.

Current highstep:

```text
initial x max = 0.2975
gated x max = 0.4675
final x max = 0.6375
```

Current final value is lower than bodyflat because highstep uses:

```text
lin_vel_x upper = 0.85
range_multiplier final = 0.75
```

Bodyflat used:

```text
lin_vel_x upper = 1.0
range_multiplier final = 1.0
```

## Highstep Key Metrics

Watch these together:

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

Healthy signs:

- terrain slowly rises or at least does not collapse;
- command reaches intended stage;
- highstep rewards remain high;
- bad orientation stays low;
- episode length stays near max;
- velocity error rises only moderately after command is released.

Danger signs:

- terrain falls while highstep rewards also fall;
- `error_vel_xy` stays above about `0.35`;
- `bad_orientation` stays above about `0.015-0.02`;
- `highstep_forward_progress` falls below about `0.08`;
- action-rate penalties and entropy rise while mean reward falls.

## Current `model_63700` Snapshot

Around `2026-06-17_03-05-16/model_63700`:

```text
terrain_levels ~ 3.0
command_levels = 0.6375
highstep_forward_progress ~ 0.18
highstep_base_advance_lift ~ 0.15
highstep_rear_box_push ~ 0.06
bad_orientation ~ 0.008-0.010
```

Interpretation:

```text
Not collapsed. It is a good teacher candidate, but it should be validated across later checkpoints before student distillation.
```

