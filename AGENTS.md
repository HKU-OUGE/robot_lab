# RobotLab Working Contract

## Repository handoff entry

When this repository is opened on a different machine or in a fresh Codex
session, read `docs/robotlab_memory_zh/CODEX_START_HERE.md` before interpreting
any highstep memory, checkpoint, metric, or launch instruction.  That entry
explains the difference between live workstation authority and the frozen
GitHub handoff snapshot.  A clone without the declared runtime `tmp/` state is
an analysis environment, not evidence that the original workflow is stopped
or safe to resume.

## Highstep work

For every highstep status, diagnosis, training, evaluation, supervisor, W&B, video, checkpoint, recovery, or code-change task, use `$highstep-control-variable-guardian` before acting.

Run its authority audit first. Resolve current authority only from:

1. `tmp/highstep_dashboard_active_workflow.json`
2. the state, heartbeat, and handoff declared there
3. the declared spec and preregistration after SHA256 verification
4. the top effective sections of that spec and any lower section they explicitly retain as a training-contract source

Historical-only sections and old workflow artifacts are evidence, never executable authority. The sole exception is an exact contract field explicitly delegated by the current highest amendment; that delegation does not restore the lower section's launch, resume, or amendment authority. Do not copy remembered parameter values, old SHAs, or old resume instructions into the current route.

## Request boundaries

- Status and explanation requests are read-only.
- Diagnosis locates and proves the cause; it does not authorize a fix.
- Infrastructure repair must preserve training semantics and all completed evidence.
- A training-semantic change requires explicit user approval and a read-only preregistration bound to the current authority.
- Never add adjacent experiments or “improvements” to an authorized route.

## Control-variable discipline

Before a semantic change, show the accepted value, proposed value, evidence, and exactly one changed semantic variable. Do not bundle changes to targets, losses, scales, warmup, freezing, optimizer, checkpoint lineage, schedule, reward, prior, network, action contract, `action_scale`, or `joint_pos.clip`.

Diagnose coarse-to-fine: route, binding, module, first unequal tensor/frame, transform, then autonomous behavior. Separate infrastructure validity, same-state causal traces, training diagnostics, autonomous rollouts, numeric gates, and user visual review. Never use a global loss or a smoke test as proof of climbing behavior.

## Live workflow safety

- Do not edit the active spec, preregistration, bound training code, or training semantics while an active child process exists.
- Never resume from an interrupted or explicitly forbidden checkpoint.
- Never start or enable a superseded highstep service.
- Preserve checkpoints, optimizer state, logs, W&B data, manifests, state, handoff, and the user's unrelated worktree changes.
- Verify recovery with the next real child PID, lock, state, heartbeat, and expected artifact; tests alone are insufficient.
- Do not bypass or weaken the project hooks. If they block an authorized operation, report the missing authority artifact or wait for the required safe boundary.

Do not hardcode current highstep versions, SHAs, numeric training values, or stage names in this file. Those belong only to the current spec and preregistration.
