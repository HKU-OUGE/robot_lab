# Highstep Training Chain

This file records the highstep training lineage and which WandB runs are useful for comparison.

## Goal

Train `RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0` so the Arcdog adjustable-leg robot can climb high platforms and steep steps. Body flat is not the primary objective. It is acceptable to use leg/body contact with terrain if it helps climbing.

Desired motion from the March reference videos:

- front legs reach and land on the high step/platform;
- rear legs push strongly;
- rear box joints extend during push;
- robot advances body over the platform rather than stalling with only front legs on top;
- flat-ground walking must remain natural, not crouched or exaggerated.

Reference videos:

```text
/home/lxq/Videos/2026.03.18/Kazam_screencast_00053.mp4
/home/lxq/Videos/2026.03.18/Kazam_screencast_00059.mp4
```

## Main Useful Chain

Keep these runs visible in WandB when comparing current training:

```text
2026-06-15_23-16-43
2026-06-16_04-22-56
2026-06-16_22-07-48
2026-06-17_00-56-56
2026-06-17_03-05-16
```

Summary:

| Run | Checkpoints | Why it matters |
| --- | --- | --- |
| `2026-06-15_23-16-43` | `model_49600` to `model_52200` | Rebooted toward visually correct highstep. `model_52200` looked surprisingly good despite modest terrain levels. |
| `2026-06-16_04-22-56` | `model_53200` to `model_57699` | Important visual baseline. `model_57699` could climb highstep terrain in play even though terrain curve stayed around 2.5. |
| `2026-06-16_22-07-48` | `model_57700` to `model_60400` | Added/reset distribution and inverse/approach logic so training better represents climbing from low side to high platform. |
| `2026-06-17_00-56-56` | `model_60400` to `model_61900` | Highstep rewards strengthened; terrain rose to about 2.82; command was still gated at 0.4675. Crashed after video/desktop stall. |
| `2026-06-17_03-05-16` | `model_61900` to `model_63700` | Command gate relaxed; command reached 0.6375; terrain crossed about 3.0; `model_63700` is current visual anchor. |

## Runs To Hide By Default

These are useful only as failure references, not normal comparison curves:

```text
2026-06-15_01-59-10
2026-06-15_06-40-11
2026-06-15_13-03-10
2026-06-16_02-44-57
2026-06-16_21-33-09
```

Notes:

- `2026-06-15_01-59-10`: early failed run; terrain stayed below 1.
- `2026-06-15_13-03-10`: terrain level looked high, but play at later checkpoints became ugly. Keep as a warning that terrain levels can mislead.
- `2026-06-16_21-33-09`: too short; one checkpoint only.

## Historical Decision Point Before Student Branch

As of `model_63700`, do not switch to student yet if time allows. Continue teacher and use `model_63700` as fallback. Student should start only after teacher remains visually good at full current command range for several checkpoints.

Suggested next teacher checkpoints to inspect:

```text
model_63700
model_64000
model_64500
model_65000
model_65500
```

Switch to student only if:

- flat walking remains natural;
- highstep movement remains strong;
- command stays at `0.6375`;
- `terrain_levels` does not collapse;
- `bad_orientation` stays low;
- velocity error does not keep worsening.

## 2026-06-18 Student Distillation Branch

Teacher `2026-06-17_05-07-05/model_68199.pt` became the practical distillation source after visual review. The first highstep no-prior student branch is:

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10
```

Important checkpoint:

```text
model_69998.pt
```

Status:

- high platform motion is barely usable;
- `terrain_levels ~= 3.0` and command is at `0.6375`;
- `bad_orientation` is higher than teacher: student last100 about `0.0144`, teacher last100 about `0.0083`;
- keyboard backward walking is the clearest weakness and can rear up/fall.

Reasoning:

- the teacher training command range was forward-only (`lin_vel_x=(0.20, 0.85)`);
- teacher can tolerate some backward keyboard commands through PPO/privileged-latent robustness;
- student is a blind compressed policy and loses OOD robustness first;
- stage 2 student currently does not run PPO, so reward terms do not directly fix backward rearing.

Student-first fix applied on 2026-06-18:

```text
ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg:
    self.commands.base_velocity.ranges.lin_vel_x = (-0.30, 0.85)
```

Original plan at that time: continue from `model_69998.pt` before changing teacher. If mild backward distillation did not reduce rearing/bad orientation, then revisit teacher command distribution.

Later judgement: this student-first branch was not enough. Play became worse and backward stability was still poor, so the route moved back to teacher refinement before any final student distillation.

## 2026-06-18/19 Teacher Backward And Box-Joint Refinement

After the first student branch exposed poor backward robustness, teacher training was changed instead of trying to hide the issue in distillation.

Important runs:

| Run | Source | Meaning |
| --- | --- | --- |
| `2026-06-18_05-10-44_teacher_backward_stability_long_from_68199` | `2026-06-17_05-07-05/model_68199.pt` | Added teacher backward command coverage and pitch stability. Backward play became much safer, but flat/normal box-joint positions became abnormal. |
| `2026-06-18_21-19-11_teacher_box_default_gate_from_80198` | later teacher checkpoint | Added stronger box-joint default-position gating. Box-joint behavior improved, but terrain progress and command release were still affected by gates. |
| `2026-06-18_23-20-22_teacher_box_default_resume_gate_free_from_82100` | `model_82100.pt` | Resume/refine run with command terrain gate relaxed. Terrain rose to about 2.78 with lower `bad_orientation`, but highstep ability looked weaker than the old high-risk run. |
| `2026-06-19_01-42-04` | `model_84400.pt` | Current refinement after lowering highstep-time box default penalty. Latest observed checkpoint `model_86400.pt`; stable but terrain levels remain around 2.8. |

Current teacher-side changes:

- teacher command range now includes mild backward motion: `lin_vel_x=(-0.30, 0.85)`;
- `backward_pitch_stability` penalizes backward rearing/body lift;
- `non_forward_highstep_pitch` remains active to reduce non-forward pitching;
- `prismatic_joint_pos_penalty` now uses `highstep_box_default_position_penalty`;
- box joints are strongly held near default outside active forward highstep, but the penalty is softened during forward highstep (`hold_scale=16.0`, `highstep_scale=0.08`);
- `scripts/rsl_rl/base/train.py` relaxes `terrain_gate_level` to `0.0` for highstep teacher resume/refine runs, while fresh training still keeps the terrain gate.

Current judgement:

- do not use the first student branch as final deployable policy;
- do not chase old high `terrain_levels` at the cost of box-joint sanity;
- continue teacher refinement until play confirms flat behavior, backward behavior, and highstep behavior are all acceptable;
- then distill a new student from the refined teacher.

## 2026-06-22/23 Real-Highstep Rear-Leg Branch Refinement

The 2026-06-22 real-highstep work improved simulated 35 cm platform climbs but still left rear-foot jamming risk for sim-to-real. Later training in `2026-06-23_00-52-26` exposed a more specific failure:

```text
right-rear-first branch: usually smoother and can succeed
left-rear-first branch: often jams; by model_105397 it can dominate behavior and fail
```

Videos:

```text
Kazam_screencast_00095_2026-06-23_00-52-26_model_97300.mp4
Kazam_screencast_00096_2026-06-23_00-52-26_model_105397.mp4
Kazam_screencast_00096_2026-06-23_00-52-26_model_105397_new.mp4
```

Interpretation:

- `model_97300` still climbs and is useful for comparison, but it is no longer the preferred parent checkpoint after re-checking the run lineage.
- `2026-06-23_00-52-26` was resumed from `2026-06-22_12-41-28/model_95398.pt`; therefore `model_97300` already contains about 1900 updates of older reward shaping before the second-rear-foot / worst-rear-foot fixes.
- The cleaner parent for the current code is `2026-06-22_12-41-28/model_95398.pt`.
- `model_105397` still has successful right-rear-first attempts, but it is not a good source checkpoint because the high-probability behavior is too often the weaker left-rear-first sequence.
- Scalar rewards can be deceptive here: rear clearance, rear drive, box push, and phase-prior rewards can rise while left/right branch reliability gets worse.

Reward-design lesson:

```text
Do not pay the policy as if "one rear foot up" is the climb success.
After one rear foot reaches the platform, the reward must make the second rear foot clear the edge too.
The stuck-foot penalty should use the worst rear foot rather than the mean of both rear feet.
```

Current recommendation:

```text
Resume refined teacher training from:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt

Keep this only as a comparison/fallback:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-23_00-52-26/model_97300.pt
```
