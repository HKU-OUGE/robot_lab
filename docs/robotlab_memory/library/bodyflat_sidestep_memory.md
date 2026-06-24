# Bodyflat And Sidestep Memory

## Bodyflat Objective

Task:

```text
RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0
```

Original goal:

- robot adjusts `box_joint` plus revolute joints to keep body level on slopes/stairs;
- body should remain flat during both static and dynamic states;
- avoid low-frequency body rocking at standstill.

Important earlier teacher reference:

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
```

Important sidestep/bodyflat later teacher anchor:

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-11_00-12-35/model_45699.pt
```

Later bodyflat/sidestep candidate played by user:

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-12_21-52-29/model_47400.pt
```

User confirmed that play with and without `--disable_action_prior` looked acceptable for this checkpoint.

## Sidestep Objective

Target scenario:

- one side of the robot stands on a step/platform;
- the other side stays on lower ground;
- prevent side roll toward the lower side;
- lower-side legs should abduct more for stability;
- box joints should show visible length difference.

Important lesson:

```text
Do not apply sidestep hip-abduction or feet-width objectives to highstep.
```

For highstep, these lateral-stability constraints make climbing harder and can corrupt the forward climbing gait.

## Action Prior Direction Lesson

There was a serious repeated error with box-joint direction in sidestep. The user caught that the policy/reward/prior direction was reversed in some versions.

Permanent convention to preserve:

```text
Smaller box_joint value means longer leg extension.
Larger box_joint value means shorter leg.
Current usable range is 0.00 to 0.06.
```

Every reward, action prior, student distillation target, and deployment clamp must be checked against this convention.

## Student Distillation Lessons From Sidestep

The no-prior student policy is delicate.

Key lesson:

- If teacher behavior depends on an environment-side action prior, a student exported without that prior cannot simply learn by standard latent distillation unless the final post-prior behavior is represented in the student's raw action.
- However, forcing post-prior action too aggressively can preserve box behavior while destroying raw locomotion.
- Opening PPO inside student distillation was debated and produced confusion. It should not be added casually. If used, the reason and trainable parameter scope must be explicit.

Safe principle:

```text
First preserve locomotion. Then internalize action-prior behavior in a limited way.
```

For highstep, student should wait until teacher is visually stable.

