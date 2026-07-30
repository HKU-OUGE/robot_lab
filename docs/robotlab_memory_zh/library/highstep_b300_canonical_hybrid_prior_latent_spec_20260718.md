# B300 canonical hybrid-prior latent distillation specification

Version: 1.0  
Workflow: `highstep_b300_canonical_hybrid_prior_latent_20260718`

## 1. Objective and isolation

Train one deployable 570-observation/64-latent/16-action Student for the user-approved
B300 fixed high-step motion, then perform at most three narrow DAgger rounds.  This
workflow is independent of v1.14.  v1.14 remains `stopped_by_gate`; none of its
checkpoints or optimizer state may be resumed or reinterpreted.

## 2. Frozen Teacher and motion reference

- Teacher checkpoint:
  `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt`
- Teacher SHA256:
  `d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`
- Approved reference root:
  `/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_15_rollouts_visual_review_20260717/approved_batch15_20260717`
- The fifteen approved trajectories contain one unique numeric episode.  They are
  one canonical information unit and must never be multiplied into fifteen training
  episodes.
- Canonical command sequence, at 0.02 s per step: steps 0--20 use vx=0; steps
  21--79 use vx=0.7200000286; steps 80--137 use vx=0.

## 3. Deployment contract

The deployed interaction stays identical to the historical policy2 interface:
manual/FSM switch to policy2, live joystick vx while held, zero command when released.
There is no trigger, latch, playback clock, new phase input, new FSM state, new sensor,
or deployment-side automation.  The 570-D observation history, 16-D output,
action_scale, joint_pos.clip, default_dof_pos, kp/kd and action contract are immutable.

## 4. Student and optimizer

- Architecture is unchanged: 570-D history -> estimator 64-D latent -> existing actor
  -> 16-D raw policy action.
- Student actor is freshly initialized from the frozen Teacher; the Teacher actor and
  privileged encoder are independent, frozen copies.
- PPO, critic and reward training are permanently disabled.
- Actor rows/bias 0--11 are permanently frozen.
- Estimator and actor rows/bias 12--15 train from effective update zero.
- The optimizer is fresh.  No v1.14 or direct-action optimizer may be loaded.
- Forward latent clamp is hard [-1, 1].  Its backward graph uses the already-audited
  straight-through implementation solely to preserve action-loss gradients.

## 5. Mixed supervision

For every pre-step state:

- dimensions 0--11 target the Teacher pre-prior raw action;
- dimensions 12--15 target the Teacher post-prior box action expressed in the same
  normalized policy-action units as the Student output;
- the post-prior policy-unit target is the exact inverse of the frozen production
  affine action map and must reproduce the final mapped box target within 1e-6;
- the estimator targets the Teacher privileged 64-D latent;
- the existing velocity, latent, reconstruction, KL and action loss coefficients are
  unchanged; no additional loss family is introduced.

## 6. Canonical tensor collection

Exactly one 138-step trajectory is recollected under the approved reset, task,
box_hard level 9, seed 11, gap 0.55 m, zero lateral/yaw offset, delay=0 and command
sequence.  It records pre-step 570-D Student history, Teacher latent raw/clamped,
Teacher pre-prior action, policy-unit post-prior action, mapped target, pre/post joint
position and velocity, reset identity, command, episode step and alignment metadata.
The recollected joint/action centerline must match the approved CSV within the frozen
numeric tolerances before training.

## 7. Centerline training and gate

Save/evaluate only E100, E300, E500 and E700.  Each checkpoint is evaluated with
fifteen Student-driven runs; same-state evidence is diagnostic only.  A pass requires
valid=15/15, full_climb>=12/15, rear_hold>=12/15, and 15/15 free from NaN, illegal
output, out-of-contract target or violent joint action.  Stop at the first passing
checkpoint.  E700 is the absolute centerline limit.

## 8. Narrow DAgger

After a centerline pass only, permit gap +/-0.03 m, lateral +/-0.02 m, yaw +/-2 deg,
small initial joint perturbation and delay=0.  The Student drives physics and the
Teacher labels the same Student pre-step state with latent and the mixed action target.
Each DAgger round has a fresh dataset/checkpoint manifest and independent W&B run.
At most three rounds are allowed.  No stairs, pits, slopes, flat-walking training,
multi-command generalization, broad DR, reward change, or other terrain is permitted.

## 9. W&B and terminal states

Centerline and every DAgger round use independent online W&B runs with group equal to
the workflow id.  Remote config/history/summary and checkpoint SHA must be verified
before advancing.  Infrastructure failures do not count as behavior failures.
At >=12/15 full and rear-hold, produce fifteen Student videos and a numbered montage
and stop as `student_fixed_motion_candidate_pending_user_visual_review`.  No real-robot
candidate claim or automatic deployment is authorized before user visual approval.

## 10. Permitted validation

Only static checks, Teacher/Student independent storage, frozen rows 0--11 SHA and
zero-gradient proof, nonzero estimator/rows 12--15 gradients from update zero,
post-prior policy-unit inversion, a 1--5 update smoke, and complete optimizer/effective
update save/restore are permitted.  The supervisor may repair infrastructure within
this contract but may not add experiments, A/B branches, tests, or training variables.
