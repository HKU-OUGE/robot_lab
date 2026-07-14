---
name: highstep-control-variable-guardian
description: Enforce the RobotLab highstep authority chain, control-variable discipline, safe checkpoint handling, and coarse-to-fine causal diagnosis. Use for any highstep training, evaluation, supervisor, spec, checkpoint, W&B, tlog/mon, video, deployment-candidate, failure-recovery, or status task in robot_lab. Also use before changing highstep code or interpreting a highstep metric, gate, or failure.
---

# Highstep Control-Variable Guardian

Protect the accepted highstep route from stale authority, bundled variable changes, speculative scope expansion, and infrastructure failures misclassified as behavior failures.

## Start with the authority audit

Run:

```bash
python3 .agents/skills/highstep-control-variable-guardian/scripts/highstep_guard.py audit
```

Stop if the audit reports an authority mismatch. Do not repair it by substituting a remembered SHA or an older spec section.

Read authority in this order:

1. `tmp/highstep_dashboard_active_workflow.json`
2. The declared `state_path` and current heartbeat/handoff
3. The declared spec and preregistration, verified by SHA256
4. The current top-level effective spec sections and any lower section they explicitly retain as a training-contract source

Treat every section explicitly marked `historical-only` as evidence only, except for the exact fields that the current highest amendment explicitly delegates to it. A delegated lower section supplies only those contract values; it does not regain launch, resume, amendment, or authority precedence. Never use an old section's “current”, “resume”, or parameter wording as executable authority.

## Classify the request before acting

- **Status or explanation:** inspect only; do not write, restart, retrain, or recover.
- **Diagnosis:** locate the first differing module/tensor/transform; do not implement a fix unless requested.
- **Infrastructure repair:** preserve training semantics and completed artifacts; repair only the broken infrastructure layer.
- **Training-semantic change:** require explicit user approval plus a read-only, single-variable preregistration before editing or launching.
- **Authorized route execution:** follow the current spec and preregistration exactly; do not add adjacent “improvements”.

## Preserve the control variable

Derive all numeric values from the current preregistration. Do not copy values from this skill, memory, chat summaries, old checkpoints, or historical spec sections.

Before any semantic edit, write a comparison table:

| Field | Accepted baseline | Proposed value | Changed? | Evidence |
|---|---|---|---|---|

Require exactly one changed semantic variable. A rename, workflow migration, checkpoint recovery, dashboard repair, or W&B repair must show zero training-semantic changes.

Never switch loss/target/warmup/freeze/optimizer semantics inside an existing optimizer lineage. Fresh-start when the current authority requires fresh initialization.

## Diagnose from coarse to fine

Use this funnel and close each level with evidence before moving deeper:

1. **Route:** Teacher, Student, or infrastructure?
2. **Binding:** checkpoint SHA, lineage, independent storage, action contract.
3. **Module:** actor, estimator, prior, optimizer, evaluator, or deployment adapter.
4. **Tensor boundary:** first unequal input/output tensor and first unequal frame.
5. **Transform:** order, scale, normalization, clipping, target selection, gradient, or schedule.
6. **Behavior:** candidate-driven rollout, fixed matrix, clear video, and user review.

Do not end with “probably”, “training may be insufficient”, or a global loss. If evidence is insufficient, add only the smallest read-only instrumentation required to find the first difference.

## Separate evidence classes

- A smoke test proves implementation and recovery mechanics, not behavior.
- Same-state traces prove causal action differences, not autonomous climbing.
- Loss, latent, terrain, and online colors are diagnostics unless the current spec explicitly promotes them to a gate.
- Infrastructure invalidity is not behavior failure.
- One checkpoint or one direction failing does not invalidate the Teacher, route, or directional mechanism.
- Numeric gates do not override a required user visual review.

## Protect live work

- Do not change the spec, preregistration, bound code, or training semantics while an active child exists.
- Do not use interrupted or explicitly forbidden checkpoints as recovery sources.
- Do not start, enable, or repair superseded services.
- Preserve existing checkpoint, optimizer, logs, W&B data, manifests, state, and handoff.
- Verify a recovery through the next real child PID, lock, state, heartbeat, and expected artifact. Unit tests alone do not prove recovery.

The project hook blocks several of these operations. Do not bypass or disable it merely to complete a task. If an authorized change is blocked, stop and report the required safe boundary or missing single-variable authority artifact.

## Report compactly

For status work, report:

```text
authority / workflow / phase / live PIDs / checkpoint / evidence progress / fault class / user action
```

For diagnosis, report:

```text
layer | input A | input B | first difference | action effect | excluded? | evidence
```

For changes, report the single-variable diff, safe boundary, tests, rebinding SHA, and next real PID witness.
