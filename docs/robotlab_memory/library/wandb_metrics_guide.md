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

2026-06-19 clarification:

```text
terrain_levels is the mean terrain-row index across all training envs after curriculum updates.
It is not the maximum platform height that the robot can climb.
```

Do not directly compare bodyflat and highstep terrain-level values without checking the task. The screenshot bodyflat samples from 2026-05/06 were around `5.2-5.7`, but they used the bodyflat task, standard `terrain_levels_vel`, full `[-1, 1]` command ranges, and a different terrain mix. Current highstep around `2.7-2.8` can still climb selected 0.35 m platform scenes in play.

Current highstep examples:

```text
2026-06-17_05-07-05: terrain ~= 3.27, bad_orientation avg100 ~= 0.0083
2026-06-19_01-42-04: terrain ~= 2.79, bad_orientation avg100 ~= 0.0014
```

Interpretation: the newer run is more stable but its terrain-level curve is lower. That is a tradeoff, not an automatic failure.

## Command Levels

`Curriculum/command_levels` returns current x-velocity upper bound, not a normalized 0-to-1 score.

Current highstep:

```text
initial x max = 0.2975
gated x max = 0.4675
final x max = 0.6375
final x min = -0.225
```

Current final positive value is lower than bodyflat because highstep uses:

```text
lin_vel_x = (-0.30, 0.85)
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
Episode_Termination/time_out
Episode_Reward/highstep_box_phase_prior
Episode_Reward/prismatic_joint_pos_penalty
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
- box-joint default penalty does not explode on flat scenes, and play shows box joints near default during flat standing/walking.

Danger signs:

- terrain falls while highstep rewards also fall;
- `error_vel_xy` stays above about `0.35`;
- `bad_orientation` stays above about `0.015-0.02`;
- `highstep_forward_progress` falls below about `0.08`;
- action-rate penalties and entropy rise while mean reward falls.
- box joints remain visibly offset on flat standing/walking, even if terrain or reward curves look acceptable.

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

## Current 2026-06-19 Teacher Refinement Snapshot

Run:

```text
2026-06-19_01-42-04
```

Latest observed:

```text
step ~= 86422
model_86400.pt
terrain_levels avg100 ~= 2.789
command_levels = 0.6375
bad_orientation avg100 ~= 0.00144
time_out avg100 ~= 0.99856
error_vel_xy avg100 ~= 0.19183
mean_reward avg100 ~= 167.0
```

Interpretation:

```text
Very stable compared with old teacher/student runs. Continue to check play: the main remaining risk is whether highstep ability and box-joint flat behavior stay acceptable while terrain_levels remains modest.
```
