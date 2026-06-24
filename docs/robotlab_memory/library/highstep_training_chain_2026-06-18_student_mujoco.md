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

## Current Decision Point

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

