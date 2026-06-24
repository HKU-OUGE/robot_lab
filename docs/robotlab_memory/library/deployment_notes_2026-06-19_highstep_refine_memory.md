# Deployment Notes

Deployment repository:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

Known simulation commands:

```bash
ros2 launch mujoco_simulator mujoco.launch.py
ros2 launch rl_quadruped_adjustable_leg_controller mujoco_adjustable_leg.launch.py
```

Relevant deployment components:

```text
robot_description: arcdog_adjustable_leg_description
controller: rl_quadruped_adjustable_leg_controller
inference / RL switch code: StateRL.cpp
```

For the full sim-to-sim launch chain, topic data flow, StateRL inference logic, joint order, and policy config, see:

```text
library/sim_to_sim_deployment.md
```

## Deployment Policy Constraint

Final deployable policy should be a blind student policy.

Important:

- If teacher uses environment-side action prior, deployed student either needs equivalent deployment-side logic or must internalize that behavior.
- For sidestep, this caused major student-distillation problems.
- For highstep, temporary student sim-to-sim probing is allowed, but final deployment requires a stable blind student with acceptable `bad_orientation` and backward/flat-ground behavior.

## Current Temporary Highstep Student For Mujoco

As of 2026-06-18, the current highstep student is not final but can be used for temporary sim-to-sim probing:

```text
Isaac checkpoint:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt

TorchScript file for Mujoco:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/exported/policy_student.pt
```

Use `policy_student.pt`, not `policy.pt`.

Verified locally:

```text
policy_student.pt: input 570 -> output 16
policy.pt: input 634 -> output 16
```

The ROS2/Mujoco `StateRL.cpp` stack expects the 570-D blind student input. The deployment config currently still points to the older:

```text
policy_student_2026-06-12_23-33-05_for_quick_deploy.pt
```

To temporarily test current highstep student, copy `policy_student.pt` into:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/robot_description/arcdog_adjustable_leg_description/config/rl_policy/
```

and update:

```text
config/rl_policy/config.yaml
model_name: "policy_student_highstep_2026-06-17_19-54-10_model_69998.pt"
```

Expected weakness: backward/negative-x commands can still rear up. This test is only for quick Mujoco signal, not final validation.

## Safety Clamp

Deployment-side clamp for `box_joint` can be useful as a final safety guard, but training should still expose limit problems. Do not hide unsafe policy behavior only by clamping in deployment.

Recommended clamp logic if used:

```text
box_joint target clamp: 0.00 to 0.06
```

But the policy should naturally stay within this range in Isaac play.

## Sim-To-Sim Gap

A policy that works in Isaac play may be weaker in Mujoco or real hardware, especially on one-side-high terrain. For highstep:

- verify flat walking first;
- verify approach to platform;
- verify front-leg placement;
- verify rear-leg push and box extension;
- verify no joint limit hits.
