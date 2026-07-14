# RobotLab Memory Index

Last updated: 2026-07-15, Asia/Hong_Kong.

This is the always-on entry point for future conversations. Read this file first, then open only the linked detail file that matches the current question.

## Current 2026-07-15 Entry (Overrides Older "Current" Text Below)

- [Codex start-here entry for a new machine/session](../robotlab_memory_zh/CODEX_START_HERE.md): distinguishes live workstation authority from the frozen GitHub handoff.
- [Portable GitHub handoff: code, history, and v1.11 progress](../robotlab_memory_zh/library/highstep_repository_transfer_handoff_20260715.md): branch scope, historical reading order, external artifact boundary, and the current read-only Teacher A/B workflow.
- [Highstep recovery and automation specification v1.11](../robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md): the top v1.11 section is the current formal authority; conflicting v1.10-and-older sections are historical evidence only.
- [Old-vs-new Teacher zero-training robustness A/B decision](../robotlab_memory_zh/library/highstep_teacher_robustness_ab_decision_20260715.md): current single variable, read-only contract, and terminal behavior.

All older items below remain useful history, but their old "current" labels, PIDs, and resume commands must not be treated as executable authority.

## Highest-Priority Workflow Constraints

- [Highstep 0707 real-video/rosbag audit and paused checkpoint](../robotlab_memory_zh/library/highstep_0707_real_log_audit_handoff_20260712.md): current highest-priority recovery point. Training and automation are safely stopped at `2026-07-11_23-45-31/model_600.pt`; do not resume before fixing cross-process schedules, target-limit safety, deployment handoff, provenance, and strict-eval gates.
- [Highstep 0707 real-data analysis and remediation report](../robotlab_memory_zh/library/highstep_0707_real_data_analysis_20260712.md): detailed alignment of both video/bag trials, cross-trial evidence, full change-chain reassessment, and implementation status for schedule, target, deployment, and strict-eval fixes.
- [Codex accountability protocol](library/accountability_protocol.md): read before every answer, judgement, command recommendation, code edit, training analysis, or video analysis. It records the user's highest-standard constraints: do not hide behind model labels; every judgement needs evidence; every command needs risk explanation; no code edit without backup; no conclusion before premise checks; no mutually conflicting recommendations; every phenomenon must be mapped to data, video, and code mechanisms.
- [Highstep short-term memory after 2026-06-25 20:00](../robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md): read before every future answer. The answer must start with the exact Chinese sentence `已按照要求提前检索记忆和约束`. This file records the late-2026-06-25 student failure, fixes, teacher stage-task edits, WandB judgements, and current training state.

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
- [Highstep short-term memory after 2026-06-25 20:00](../robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md): highest-priority recent context; read first.
- [Codex accountability protocol](library/accountability_protocol.md): highest-priority workflow constraints; read first.
- [Highstep training chain](library/highstep_training_chain.md): which WandB runs to keep or hide, and why.
- [Bodyflat and sidestep memory](library/bodyflat_sidestep_memory.md): bodyflat teacher/student history, sidestep action-prior lessons.
- [Repository usage rules](library/repo_usage_rules.md): commands, backups, logs, videos, monitoring, and safe operating rules.
- [Arcdog adjustable leg model](library/arcdog_adjustable_leg_model.md): joint order, action scaling, box joint direction, actuator settings.
- [WandB metrics guide](library/wandb_metrics_guide.md): how to judge training from curves without being fooled.
- [Failure modes and lessons](library/failure_modes_and_lessons.md): repeated mistakes and practical fixes.
- [Deployment notes](library/deployment_notes.md): ROS2/Mujoco/sim-to-real paths and student-policy constraints.
- [Sim-to-sim deployment chain](library/sim_to_sim_deployment.md): full data flow from the two launch commands through Mujoco, ros2_control, StateRL, and policy config.

## Hard Rules To Preserve

- Highest priority: read [Codex accountability protocol](library/accountability_protocol.md). Every judgement needs evidence; every command needs risk explanation; every code edit needs a backup; no conclusion before premise checks; no mutually conflicting recommendations.
- Every answer must start with the exact Chinese sentence `已按照要求提前检索记忆和约束`. Before answering, actively read the [Highstep short-term memory](../robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md) and [Codex accountability protocol](library/accountability_protocol.md).
- Before code changes, back up every modified file. Backup suffix should be the previous or current relevant training log folder name, e.g. `_2026-06-17_03-05-16`.
- Do not judge highstep only by `Curriculum/terrain_levels`; play/video is decisive.
- `terrain_levels` is an average terrain-row occupancy metric, not a direct "max platform height climbed" score. Bodyflat curves around 5.x and highstep curves around 2.7-3.3 are not directly comparable without checking task, curriculum function, command range, and terrain mix.
- For highstep, explicitly watch left-rear-first versus right-rear-first attempts. Averaged rear-foot rewards can hide one-sided branch collapse. A successful climb often works because the rear foot already on the platform becomes the support/drive leg and lifts the body so the second rear foot can clear the edge.
- Avoid `--video` during long training. It has repeatedly increased crash risk.
- For highstep, do not reintroduce hip abduction/feet-width constraints from sidestep unless explicitly needed.
- For box joints, preserve the confirmed convention: smaller `box_joint` value means longer leg extension; larger value means shorter leg.
- Temporary highstep student testing is allowed in Isaac/Mujoco, but do not treat the current student as final until backward stability and `bad_orientation` improve.
