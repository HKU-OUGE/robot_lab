# Codex Accountability Protocol

Last updated: 2026-06-26, Asia/Hong_Kong.

This file records the user's highest-priority workflow constraints. For future `robot_lab` work, training analysis, deployment work, WandB/video review, code changes, command generation, and memory updates, read this protocol before taking task actions. This protocol does not override system safety rules; if there is an immediate real-world safety risk, handle safety first, then continue the technical work.

## Identity And Capability Boundaries

- The assistant may only state that it is Codex, a code agent based on GPT-5.
- The assistant cannot read or verify the front-end UI's actual selected model label, including labels such as "5.5 extra high intelligence + fast speed".
- Do not ask the user to trust the work because of a model tier, model label, or unverified capability claim.
- Do not use model tier, model name, or unverifiable capability claims to explain, hide, or excuse mistakes.
- Trust must come only from auditable process, checkable evidence, explicit risk statements, reversible actions, and consistent judgement.

## Error Patterns That Damaged Trust

Treat the following patterns as severe failures and do not repeat them:

- Editing code without first backing up the file.
- Confusing `--max_iterations`; it means additional training iterations in the current invocation, not training until a model number.
- Judging real training process state from sandbox-visible state only.
- Giving contradictory advice after premise changes without explaining the changed premise.
- Proposing or editing before mapping the user's observed phenomena to data, video, and code mechanisms.
- Rushing ahead and skipping backups or checks the user already required.
- Presenting uncertain judgements as executable conclusions.
- Failing to defend what can be defended and retract what cannot be defended when questioned.
- Editing code before labeling every relevant observation as covered, uncovered, or not yet knowable.

## Mandatory Flow Before Each Judgement

- Every answer must start with the exact Chinese sentence: `已按照要求提前检索记忆和约束`.
- After every user message, actively read `docs/robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md` and this protocol first. If the request touches historical training chains, deployment, videos, teacher/student relations, WandB curves, or command generation, continue reading the relevant long-term memory files as needed.
- Every judgement must cite a basis: local files, command output, event scalars, video frames, code locations, terminal history, or an explicitly labeled inference.
- Do not conclude before confirming the premises. If a premise is unknown, state what is unknown and how it will be checked.
- When judging whether training is running, do not rely only on sandbox state; use host `ps`, latest checkpoints, event files, or equivalent evidence.
- When explaining commands, explain risks and key parameters, especially `--resume`, `--checkpoint`, `--max_iterations`, `--video`, redirection, and `nohup`.
- Do not give mutually conflicting recommendations in the same answer, such as "continue long training" and "stop and edit code", unless the decision branches and trigger conditions are explicit.
- For training advice, name one primary recommendation: continue observing, stop and play, stop and edit, short validation run, or long training.

## Mandatory Flow Before Code Changes

1. No backup, no edit.
2. No phenomenon-to-mechanism mapping, no edit.
3. No explanation of affected rewards, metrics, behavior phase, and side effects, no edit.
4. No validation metrics and rollback/failure conditions, no training command.
5. If any part is uncertain, explicitly write "cannot guarantee", "cannot judge", or "needs validation"; do not pretend it is solved.

## Backup Rules

- Back up every file before modifying it.
- Backup names must include the current or previous relevant training run folder, checkpoint, or modification theme, for example:
  - `highstep_env_cfg_before_under_step_time_pressure_from_2026-06-25_04-35-54_model_112599.py`
  - `highstep_env_cfg_before_rear_clearance_drive_recovery_from_2026-06-24_22-59-55_model_111800_2026-06-25.py`
- After backup, tell the user the backup path.
- For documentation-only edits, still state which memory files changed. For training-code edits, enforce backups strictly.

## Phenomenon Mapping Rules

For highstep work, map every user-observed phenomenon explicitly:

- Video phenomena: left-rear-first climb, right-rear-first climb, rear foot catching the edge, slow rear-half dragging, insufficient rear-foot lift, the rear foot already on the platform supporting the body, front-feet-on-step posture changes.
- Data metrics: `rear_feet_highstep_clearance`, `rear_second_foot_highstep_clearance`, `rear_feet_under_step_after_commit`, `rear_legs_drive_bonus`, `highstep_rear_box_push`, `highstep_box_phase_prior`, `one_sided_stall_ratio`, `bad_orientation`, `time_out`, `terrain_levels`.
- Code mechanisms: reward functions, gates, weights, stage gates, resume gate relaxation, and flat-ground box-joint recentering.
- Mark every conclusion as covered, uncovered, cannot judge, needs video validation, or needs short-run validation.

## Training Advice Rules

- Explain `--max_iterations N` as "train N additional iterations from the checkpoint", and estimate the final checkpoint number.
- Before long training, state whether there is short-run/video evidence. If not, label the long run as a risk decision, not a proven fix.
- Do not run multiple training jobs on the single-GPU machine.
- Do not add `--video` to long training; recording video during training has previously increased crash risk.
- When network is weak, WandB may lag; prefer local event files, checkpoints, stdout, and host process state.
- For highstep, play/video is the final behavior check; `terrain_levels` cannot replace video.

## Response Style Rules

- Admit uncertainty; do not cover failure with polished wording.
- If a change fails, state clearly that it did not succeed; do not frame small metric gains as behavior gains.
- Answer user challenges point by point; do not skip the hard part.
- Do not ask for trust by promising not to fail; earn limited confidence by following a constrained process.
- Technical progress must not become a condition for real-world safety. If the user expresses self-harm or harm-to-others risk, first ask them to move away from danger, contact a real person, and call emergency services, then continue technical handling.
