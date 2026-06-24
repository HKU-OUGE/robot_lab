# RobotLab Memory Index

Last updated: 2026-06-18, Asia/Hong_Kong.

This is the always-on entry point for future conversations. Read this file first, then open only the linked detail file that matches the current question.

## Current Focus

- Main task: `RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`
- Current useful teacher run: `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_03-05-16`
- Current visual anchor: `model_63700.pt`
- Current judgement: continue teacher from `model_63700.pt` before switching to student. Use `model_63700.pt` as fallback if later checkpoints get visually worse.
- Current highstep command gate: command max has reached `0.6375` after relaxing `terrain_gate_level` from `3.0` to `2.6`.

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
- Avoid `--video` during long training. It has repeatedly increased crash risk.
- For highstep, do not reintroduce hip abduction/feet-width constraints from sidestep unless explicitly needed.
- For box joints, preserve the confirmed convention: smaller `box_joint` value means longer leg extension; larger value means shorter leg.
- Do not switch highstep to student until teacher behavior is visually stable across several checkpoints at full current command range.
