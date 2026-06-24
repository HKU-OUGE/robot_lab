# Current State

Last updated: 2026-06-24.

## 2026-06-23 Rear-Leg Branch Asymmetry Update

Latest useful teacher run:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-23_00-52-26
```

Important checkpoints:

```text
2026-06-22_12-41-28/model_95398.pt   preferred clean restart anchor for the new rear-branch-balance code
2026-06-23_00-52-26/model_97300.pt   comparison/fallback only; trained ~1900 updates from 95398 under the older reward mix
2026-06-23_00-52-26/model_105397.pt  failure-reference checkpoint, not a refinement source
```

Video review:

```text
/home/lxq/Videos/Kazam_screencast_00095_2026-06-23_00-52-26_model_97300.mp4
/home/lxq/Videos/Kazam_screencast_00096_2026-06-23_00-52-26_model_105397.mp4
/home/lxq/Videos/Kazam_screencast_00096_2026-06-23_00-52-26_model_105397_new.mp4
```

Observed behavior:

- `model_97300` can still climb. Right-rear-first attempts are smoother; left-rear-first attempts tend to pause or jam briefly but can often recover.
- `model_105397` is not simply unable to climb; it still has a right-rear-first success branch, confirmed by the `_new` video.
- The real failure is branch probability and quality: later training makes the weaker left-rear-first branch too dominant. When left rear climbs first, the body/yaw/roll geometry often traps the other rear foot under the platform edge.
- Successful climbs often work because the rear foot already on the platform becomes the support/drive leg and lifts/rotates the body enough for the second rear foot to clear the edge.

Scalar warning from `model_97300` to `model_105397`:

```text
rear_feet_highstep_clearance rose strongly
rear_legs_drive_bonus / highstep_rear_box_push / highstep_box_phase_prior rose
but roll/yaw penalty, velocity error, and rear-foot-under-step symptoms also worsened
```

Lesson: averaged reward curves can show more rear-leg activity while the actual behavior collapses into a one-sided unreliable strategy.

Checkpoint decision:

```text
Prefer restarting from 2026-06-22_12-41-28/model_95398.pt with the current code.
Do not make 2026-06-23_00-52-26/model_97300.pt the primary source unless the 95398 restart clearly loses highstep ability.
Do not continue from model_105397.pt.
```

Reason:

```text
2026-06-23_00-52-26 loaded 2026-06-22_12-41-28/model_95398.pt, so model_97300 is not an independent better parent.
The 95398 -> 97300 interval was trained before the latest second-rear-foot / worst-rear-foot reward fixes.
That interval raised forward progress, base lift, rear box push, and phase prior rewards, but it also came from the older shaping that may reinforce one-rear-foot completion.
```

Scalar snapshot used for this revised decision:

```text
model_95398: terrain_levels=2.8294, bad_orientation=0.001709, time_out=0.998291, roll_yaw_orientation_penalty=-0.004110
model_97300: terrain_levels=2.3444, bad_orientation=0.000844, time_out=0.999156, roll_yaw_orientation_penalty=-0.006171
model_97300 has better short-term termination numbers, but is less clean as a parent because it already includes old reward-guided training after 95398.
```

Code update applied after this diagnosis:

```text
rewards.py:
  rear_feet_highstep_clearance_bonus now blends max/mean/min rear-foot clearance instead of paying mostly for one rear foot.
  rear_feet_under_step_after_commit_penalty now blends worst rear foot with mean rear foot so one stuck foot is not hidden.
  rear_second_foot_highstep_clearance_bonus was added to reward the second rear foot clearing after either rear foot reaches the platform.

highstep_env_cfg.py:
  rear_feet_highstep_clearance weight reduced to 1.15, single_rear_weight set to 0.20, min_rear_weight set to 0.50.
  rear_feet_under_step_after_commit weight strengthened to -0.50 with worst_rear_weight 0.82.
  rear_second_foot_highstep_clearance added at weight 0.95.
  highstep_rear_box_push reduced to 1.10 and highstep_box_phase_prior reduced to 0.85 to avoid box/phase rewards overpowering real completion.

train.py:
  rear_second_foot_highstep_clearance added to the highstep --resume staged-reward release list.
```

## Active Task

Teacher task:

```bash
RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0
```

Student task:

```bash
RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
```

## 2026-06-19 Highstep Teacher Refinement

The student-first repair attempt after `2026-06-17_19-54-10/model_69998.pt` was not enough. The student could climb high platforms only barely, had worse `bad_orientation`, and keyboard backward walking could rear up/fall. The current route is:

```text
refine teacher for backward stability and sane box-joint behavior -> inspect play -> distill a new student later
```

Current teacher refinement run to monitor:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-19_01-42-04
```

Latest checkpoint observed during this memory update:

```text
model_86400.pt
```

Latest scalar snapshot at step `86422`:

```text
terrain_levels last ~= 2.798, avg100 ~= 2.789
command_levels = 0.6375
bad_orientation avg100 ~= 0.00144
time_out avg100 ~= 0.99856
error_vel_xy avg100 ~= 0.19183
mean_reward avg100 ~= 167.02
```

Interpretation:

- this run is much more stable than `2026-06-17_05-07-05` and the first student;
- `terrain_levels` remains lower than some older/bodyflat curves, but this is not by itself a failure;
- play showed the box-joint offset was no longer obviously extreme after the latest box-default changes;
- continue to judge with play/video plus `bad_orientation`, `time_out`, velocity error, and box-joint behavior.

Continue current teacher refinement with:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-19_01-42-04/model_86400.pt
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
- the initial fix was to add mild backward samples to student distillation, but play became worse; this confirmed that teacher-side backward stability needed to be improved before final distillation.

Student branch result:

```text
2026-06-17_19-54-10/model_69998.pt is useful as a temporary Mujoco signal test only. It should not be used as the final deployment candidate.
```

Backup for that edit:

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg_2026-06-18_00-02-07.py
```

Do not continue this student as the main route unless explicitly testing the old branch. Prefer teacher refinement first, then re-distill.

Old student continuation command, only for controlled comparison:

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt
```

Current code uses `max_iterations=6000` and `student_actor_warmup_updates=1200` for the highstep student runner. This was changed because the old highstep student used a short `1800`-iteration setup similar to quick deploy and was too short for a stable final student.

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
lin_vel_x = (-0.30, 0.85)
lin_vel_y = (-0.35, 0.35)
ang_vel_z = (-0.60, 0.60)
range_multiplier = (0.35, 0.75)
gated_multiplier = 0.55
terrain_gate_level = 2.6 for fresh training; resume/refine runs relax it to 0.0 in train.py
```

Effective command levels:

```text
initial x max = 0.85 * 0.35 = 0.2975
gated  x max = 0.85 * 0.55 = 0.4675
final  x max = 0.85 * 0.75 = 0.6375
final  x min = -0.30 * 0.75 = -0.225
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

Current box-joint default constraint:

```text
prismatic_joint_pos_penalty = highstep_box_default_position_penalty
weight = -2.5
hold_scale = 16.0
highstep_scale = 0.08
```

Meaning: box joints are strongly pulled to default on flat/non-forward scenes. The penalty is relaxed only when both forward command and highstep terrain gate are active.
