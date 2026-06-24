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
- For highstep, do not switch to student until teacher is stable.

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
