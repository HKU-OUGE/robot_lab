# Highstep Teacher Virtual Terrain Adapter Spec v1.0

## 1. Authority and objective

- Workflow: `highstep_teacher_virtual_terrain_adapter_20260718`.
- This is an independent, zero-training deployment-diagnostic route. It does not overwrite the formal v1.13.1 route or the archived B300 Student D1/D2/D3 route.
- The only objective before user review is one manually driven MuJoCo nominal run with a physical 0.30 m platform and the frozen B300 virtual scan semantics (about 0.378 m).
- No Student/Teacher training, optimizer step, W&B stage, perturbation matrix, or real-robot deployment is authorized.

## 2. Frozen Teacher and references

- Teacher checkpoint:
  `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt`
- Teacher SHA256: `d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`.
- Canonical tensor trajectory:
  `/home/lxq/Softwares/robot_lab/tmp/highstep_b300_canonical_hybrid_prior_latent_20260718/canonical_tensor_trajectory_attempt2/canonical_tensor_trajectory.pt`
- Canonical tensor SHA256: `5b53cd6e14e7760199c18a6036e3a9fd41d549890b76e80b6607a4eee9d2f9cd`.
- The visually approved B300 Teacher batch under
  `/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_15_rollouts_visual_review_20260717/approved_batch15_20260717/`
  is frozen as the visual/action reference.

## 3. Exact inference contract

The checkpoint itself is authoritative about tensor dimensions:

1. The existing production observation history remains 570-dimensional.
2. The privileged encoder receives the current 57-dimensional uncorrupted Teacher proprioceptive vector plus the 102-dimensional virtual height scan: `159 -> 64`.
3. The Teacher actor receives `570 + 64 = 634` values and emits 16 raw actions. The original wording `57 + 64` is incompatible with the frozen checkpoint and must not be implemented.
4. The privileged latent is clamped to `[-1, 1]` before the actor, matching native deterministic Teacher inference.
5. The production phased highstep action prior is applied exactly once after action scaling/default offset and before the existing physical target clamp.
6. `action_scale`, `joint_pos.clip`, `default_dof_pos`, `kp/kd`, joint order, and the 16-output action contract remain unchanged.

## 4. User controls and initial calibration

- Policy2 consumes the live joystick continuously. There is no one-shot, command latch, fixed playback, or hidden phase clock.
- Full forward joystick is continuously calibrated to the approved `vx=0.7200000286`; partial input remains proportional, release returns command to zero, and `vy/yaw` remain live.
- The virtual adapter resets every time policy2 is entered.
- Calibrated placement domain: edge gap `0.55 +/- 0.02 m`, lateral `+/-0.02 m`, yaw `+/-2 deg`, body-height offset `+/-0.02 m`.

## 5. Virtual adapter and prohibited truth inputs

The adapter state is exactly:

- `longitudinal_progress`
- `lateral_offset`
- `relative_yaw`
- `body_height_offset`
- `confidence`

Its causal update is calibrated initial pose plus joystick prediction, corrected by onboard IMU and joint/leg-motion evidence. Commands are predictions, never displacement truth. Phase is generated from virtual relative pose and current body/joint state; `vx` never directly selects phase.

The runtime implementation is shared by MuJoCo and the real controller. It may use only IMU, joint position/velocity, last raw action, live command, and its frozen internal reference. It must not import, subscribe to, or call MuJoCo free-joint translation, platform geom data, raycast, world contact coordinates, or any current-scene terrain truth.

The 102 scan is produced by continuous state-conditioned interpolation of the frozen canonical reference. Body-height, lateral, and yaw corrections are continuous. Low confidence fails closed instead of guessing phase.

## 6. Safety

- NaN/Inf, invalid scan/phase jumps, output-limit violation, excessive single-step target jump, excessive roll/pitch/angular speed, or sustained low confidence causes a fail-closed exit.
- Before platform contact with stable attitude, the requested exit is Fixed Stand.
- During front support/rear push or rapid tipping, the requested exit is Passive.
- Automatic real-robot deployment is forbidden.

## 7. Required evidence and terminal state

Only these checks are authorized:

1. Static tensor dimensions, joint order, scale, and prior-once checks.
2. Canonical offline replay reporting scan, latent, pre-prior, and post-prior max-absolute error, MAE, and first differing frame.
3. One user-triggered MuJoCo nominal run with physical platform 0.30 m and virtual reference about 0.378 m, with adapter log and video.

Before the user accepts that nominal video, no perturbation matrix, Student/Teacher training, true-robot experiment, or other route may start. The ready state is `waiting_user_mujoco_manual_play`; after a recorded run it is `pending_user_visual_review`.
