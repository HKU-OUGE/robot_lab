# RobotLab Memory Index

Last updated: 2026-06-24, Asia/Hong_Kong.

This is the always-on entry point for future conversations. Read this file first, then open only the linked detail file that matches the current question.

## Current Focus

- Main teacher task: `RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`
- Main student task: `RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0`
- Student branch that exposed the problem: `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt`
- Current student judgement: platform ability is barely usable, but `bad_orientation` is worse than teacher and keyboard backward walking can rear up/fall. It is useful only for temporary Mujoco sim-to-sim probing, not as a final deployable policy.
- Current route after the failed student-first attempt: refine the teacher first, then distill again.
- Current highstep teacher refinement anchor: prefer restarting the new rear-branch-balance code from `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt`, not from `2026-06-23_00-52-26/model_97300.pt`. `model_97300` was produced by continuing from `model_95398.pt` for about 1900 updates under the older reward mix, so it may already contain wrong guidance toward one-rear-foot/box-push behavior. Keep `model_97300.pt` as a comparison/fallback only.
- Do not continue highstep teacher refinement from `2026-06-23_00-52-26/model_105397.pt` unless explicitly studying failure. Videos show it still has a right-rear-first success branch, but the dominant behavior collapses too often into the weaker left-rear-first branch.
- Current highstep teacher command range is now `lin_vel_x=(-0.30, 0.85)`, so backward stability is part of teacher training, not only student distillation.
- Current box-joint safety rule: keep box joints near default on flat/non-forward scenes, and release them only during active forward highstep climbing via `highstep_box_default_position_penalty`.
- Current Mujoco test export: use `exported/policy_student.pt` from the student run. It accepts the deployment stack's 570-D blind observation. Do not use `exported/policy.pt` for Mujoco; it expects 634-D actor+latent input.

## Read This Next

- [Current state](library/current_state.md): latest checkpoint, current recommendation, active config values.
- [Highstep training chain](library/highstep_training_chain.md): which WandB runs to keep or hide, and why.
- [Bodyflat and sidestep memory](library/bodyflat_sidestep_memory.md): bodyflat teacher/student history, sidestep action-prior lessons.
- [Repository usage rules](library/repo_usage_rules.md): commands, backups, logs, videos, monitoring, and safe operating rules.
- [Arcdog adjustable leg model](library/arcdog_adjustable_leg_model.md): joint order, action scaling, box joint direction, actuator settings.
- [WandB metrics guide](library/wandb_metrics_guide.md): how to judge training from curves without being fooled.
- [Failure modes and lessons](library/failure_modes_and_lessons.md): repeated mistakes and practical fixes.
- [Deployment notes](library/deployment_notes.md): ROS2/Mujoco/sim-to-real paths and student-policy constraints.
- [Sim-to-sim deployment chain](library/sim_to_sim_deployment.md): full data flow from the two launch commands through Mujoco, ros2_control, StateRL, and policy config.

## Hard Rules To Preserve

- Before code changes, back up every modified file. Backup suffix should be the previous or current relevant training log folder name, e.g. `_2026-06-17_03-05-16`.
- Do not judge highstep only by `Curriculum/terrain_levels`; play/video is decisive.
- `terrain_levels` is an average terrain-row occupancy metric, not a direct "max platform height climbed" score. Bodyflat curves around 5.x and highstep curves around 2.7-3.3 are not directly comparable without checking task, curriculum function, command range, and terrain mix.
- For highstep, explicitly watch left-rear-first versus right-rear-first attempts. Averaged rear-foot rewards can hide one-sided branch collapse. A successful climb often works because the rear foot already on the platform becomes the support/drive leg and lifts the body so the second rear foot can clear the edge.
- Avoid `--video` during long training. It has repeatedly increased crash risk.
- For highstep, do not reintroduce hip abduction/feet-width constraints from sidestep unless explicitly needed.
- For box joints, preserve the confirmed convention: smaller `box_joint` value means longer leg extension; larger value means shorter leg.
- Temporary highstep student testing is allowed in Isaac/Mujoco, but do not treat the current student as final until backward stability and `bad_orientation` improve.
