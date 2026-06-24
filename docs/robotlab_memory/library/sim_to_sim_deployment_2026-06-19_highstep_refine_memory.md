# Sim-To-Sim Deployment Chain Memory

Last updated: 2026-06-18, Asia/Hong_Kong.

Deployment repository:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

Common sim-to-sim launch commands:

```bash
ros2 launch mujoco_simulator mujoco.launch.py
ros2 launch rl_quadruped_adjustable_leg_controller mujoco_adjustable_leg.launch.py
```

## One-Line Summary

This sim-to-sim deployment path does not run an Isaac policy directly. The actual chain is:

```text
Mujoco simulation node
  -> publishes /imu_data, /joint_states, /mujoco_msg
ros2_control HardwareMujoco
  -> maps those topics into controller state interfaces
rl_quadruped_adjustable_leg_controller
  -> StateRL.cpp builds observations, runs the TorchScript policy, outputs joint targets
HardwareMujoco
  -> publishes /actuators_cmds
Mujoco handler
  -> writes Mujoco actuator ctrl using kp * pos_error + kd * vel_error + torque
```

The final deployable target is still a blind student policy. A teacher policy that works in Isaac does not automatically work in this deployment stack.

## Launch Chain

### 1. Mujoco Simulation Side

Entry point:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/mujoco_simulator/launch/mujoco.launch.py
```

Default arguments:

```text
robot_pkg = arcdog_adjustable_leg_description
xml_file_path = arcdog_adjustable_leg_description/xml/scene.xml
```

`scene.xml` includes:

```text
robot_description/arcdog_adjustable_leg_description/xml/arcdog_adjustable_leg.xml
```

Mujoco executable source:

```text
mujoco_simulator/src/mujoco_node.cpp
```

Current code sets:

```text
robot_type = 4
```

So the active message handler is:

```text
ArcLab::AdjustableLegMujocoMsgHandler
```

Relevant files:

```text
mujoco_simulator/src/adjustable_leg_mujoco_msg_handler.cpp
mujoco_simulator/include/mujoco_node/adjustable_leg_mujoco_msg_handler.h
```

This handler:

- publishes `imu_data` every 1 ms;
- publishes `joint_states`;
- publishes `mujoco_msg`;
- subscribes to `actuators_cmds`;
- converts controller position/KP/KD commands into Mujoco actuator commands.

Actual Mujoco control formula:

```text
ctrl = kp * (target_pos - joint_pos) + kd * (target_vel - joint_vel) + torque
```

### 2. Controller Side

Entry point:

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/rl_quadruped_adjustable_leg_controller/launch/mujoco_adjustable_leg.launch.py
```

Default description package:

```text
pkg_description = arcdog_adjustable_leg_description
```

This launch reads:

```text
arcdog_adjustable_leg_description/xacro/robot.xacro
arcdog_adjustable_leg_description/config/robot_control.yaml
```

Main nodes:

- `robot_state_publisher`
- `controller_manager/ros2_control_node`
- `joint_state_broadcaster`
- `imu_sensor_broadcaster`
- `rl_quadruped_adjustable_leg_controller`
- `joy/game_controller_node`
- `rviz2`

Note: `cmd_mapping` is defined in this launch file, but the current `nodes_to_start` list does not include it. In the current sim-to-sim path, the controller mainly reads `/joy` directly instead of relying on `cmd_mapping`.

Real robot launch:

```text
rl_quadruped_adjustable_leg_controller/launch/real_robot_adjustable_leg.launch.py
```

The real robot launch reads:

```text
arcdog_adjustable_leg_description/xacro/real_robot.xacro
```

The key difference is the hardware plugin:

```text
sim-to-sim: hardware_mujoco/HardwareMujoco
real robot: hardware_arcdog_adjustable_leg/HardwareArcdog_adjustable_leg
```

## ROS2 Control Data Flow

Sim-to-sim hardware interface:

```text
hardware_mujoco/src/HardwareMujoco.cpp
```

It subscribes to:

```text
/joint_states
/imu_data
```

It publishes:

```text
/actuators_cmds
```

It exposes these interfaces to the controller:

```text
state_interfaces: position, velocity, effort
command_interfaces: position, velocity, effort, kp, kd
```

Real robot hardware interface:

```text
hardware_arcdog_adjustable_leg/src/HardwareArcdog_adjustable_leg.cpp
```

The real robot path reads and writes physical motors through USB/CAN and has a separate motor activation flow:

```text
motor_mode = 1: disabled
motor_mode = 8: activation preparation
motor_mode = 0: command control enabled
```

Real robot safety limits include:

```text
limit_prismatic_.min = -0.015
limit_prismatic_.max =  0.065
```

Training and policy deployment should still target the true safe range `0.00` to `0.06`; do not rely on real robot emergency stop as the main safety mechanism.

## RL Controller FSM

Main controller:

```text
rl_quadruped_adjustable_leg_controller/src/RlQuadrupedControllerAdjustableLeg.cpp
```

Main RL state:

```text
rl_quadruped_adjustable_leg_controller/src/FSM/StateRL.cpp
rl_quadruped_adjustable_leg_controller/include/rl_quadruped_adjustable_leg_controller/FSM/StateRL.h
```

State switching is handled by `StateRL::checkChange()` and the controller `/joy` callback.

Approximate gamepad mapping:

```text
RB + B -> command = 0 -> PASSIVEADJUSTABLELEG
RB + A -> command = 1 -> FIXEDDOWNADJUSTABLELEG
RB + Y -> command = 2 -> FIXEDSTANDADJUSTABLELEG
RB + X -> command = 3 -> RL
```

Gamepad command scaling:

```text
lx = 0.5 * axes[1]
ly = 0.34 * axes[0]
rx = -1.0 * axes[2]
StateRL uses control.yaw = -rx
```

So deployment-side maximum commands are currently around `0.5` forward and `0.34` lateral. This may not exactly match the training command range and must be checked when sim-to-sim gait looks weaker than Isaac play.

## Policy Configuration

Current deployment policy config:

```text
robot_description/arcdog_adjustable_leg_description/config/rl_policy/config.yaml
```

Current `model_name`:

```text
policy_student_2026-06-12_23-33-05_for_quick_deploy.pt
```

This means the current deployment stack is not automatically using the latest highstep policy. To deploy a newer highstep teacher/student, export a TorchScript policy, copy it into:

```text
robot_description/arcdog_adjustable_leg_description/config/rl_policy/
```

Then update:

```text
model_name: "new_policy.pt"
```

2026-06-18 temporary highstep student test:

```text
Source run:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10

Checkpoint:
model_69998.pt

Use this TorchScript in Mujoco:
exported/policy_student.pt
```

Verified input shapes:

```text
exported/policy_student.pt accepts 570-D blind deployment observation and outputs 16 actions.
exported/policy.pt accepts 634-D actor+latent input and should not be used for current Mujoco blind-student testing.
```

Suggested temporary deployment filename:

```text
policy_student_highstep_2026-06-17_19-54-10_model_69998.pt
```

Known caveat: this student is not final; it can rear up on backward/negative-x commands and has higher `bad_orientation` than the teacher.

Key config values:

```text
framework: isaacsim
rows: 4
cols: 4
decimation: 4
num_of_dofs: 16
observations: ["ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"]
observations_history: [0,1,2,3,4,5,6,7,8,9]
clip_obs: 60.0
lin_vel_scale: 2.0
ang_vel_scale: 0.25
dof_pos_scale: 1.0
dof_vel_scale: 0.05
commands_scale: [1.0, 1.0, 1.0]
```

`StateRL.cpp` assembles observations in the Isaac Lab-compatible term-major layout:

```text
ang_vel history, 10 frames
gravity_vec history, 10 frames
commands history, 10 frames
dof_pos history, 10 frames
dof_vel history, 10 frames
actions history, 10 frames
```

Total policy input size:

```text
57 * 10 = 570
```

This was an important previous fix. Do not interleave full per-frame observations; the deployment layout must match the Isaac Lab exported student policy.

## StateRL Inference Logic

Core `StateRL.cpp` flow:

```text
getState()
  -> read IMU, joint q/dq, and gamepad command
runModel()
  -> build observation
  -> forward policy
  -> action clip
  -> actions_scaled = action * action_scale
  -> output_dof_pos = actions_scaled + default_dof_pos
  -> output_dof_pos clamp
  -> write position/velocity/kp/kd/torque command interfaces
```

Current code first tries:

```text
model(final_obs)
```

If that fails, it appends a zero 64-D latent and tries:

```text
model(concat(final_obs, zero_latent))
```

This fallback can prevent a crash, but it can also hide a wrong exported policy input shape. For deployment, the preferred policy should be a blind student that directly accepts the 570-D observation.

## Joint Order

`robot_control.yaml` lists RL controller joints grouped by joint type:

```text
FL_hip_joint
FR_hip_joint
RL_hip_joint
RR_hip_joint
FL_thigh_joint
FR_thigh_joint
RL_thigh_joint
RR_thigh_joint
FL_calf_joint
FR_calf_joint
RL_calf_joint
RR_calf_joint
FL_box_joint
FR_box_joint
RL_box_joint
RR_box_joint
```

Many arrays in `config.yaml` are written leg-major:

```text
[hip, thigh, calf, box] * 4
```

When `framework == "isaacsim"`, `StateRL.cpp` uses `rows=4, cols=4` to transpose these arrays into the grouped controller order.

Therefore:

```text
action_scale: [0.1, 0.1, 0.1, 0.02] * 4
```

becomes:

```text
4 hip scales = 0.1
4 thigh scales = 0.1
4 calf scales = 0.1
4 box scales = 0.02
```

Current `hip_scale_reduction_indices`:

```text
[0, 1, 2, 3]
```

This matches the grouped order. Older configs used `[0,4,8,12]`, which corresponds to leg-major order and is not correct for the current transposed controller order.

## Box Joint Direction And Safety

Permanent convention:

```text
smaller box_joint value = longer leg extension
larger box_joint value = shorter leg
```

Training and deployment target range:

```text
safe target range: 0.00 to 0.06
default value: 0.03
```

Current `config.yaml` has final target clamp:

```text
output_dof_pos_lower: box joint = 0.0
output_dof_pos_upper: box joint = 0.06
```

The clamp is applied after:

```text
output_dof_pos = action * action_scale + default_dof_pos
```

Note: Mujoco XML currently has a wider slide-joint range:

```text
arcdog_adjustable_leg.xml: range="-0.05 0.10"
```

But xacro/URDF prismatic joint limits are:

```text
lower="0"
upper="0.06"
```

The previous reasoning was: Mujoco XML may remain wider to reveal unsafe policy tendencies, but the final deployment target should still be clamped to the real range. Do not treat "Mujoco did not hit a limit" as proof of real robot safety.

## Important Historical Fixes

1. `StateRL.cpp` must load TorchScript on CPU:

```text
torch::jit::load(model_path, torch::kCPU)
```

This avoids CUDA backend crashes in the ROS2 deployment process.

2. `default_dof_pos` must be present in `config.yaml`.

If missing, the code falls back to `stand_pos_adjustable_leg_`, which can make `dof_pos - default_dof_pos` wrong and damage VAE/student latent behavior.

3. Observation history order must match Isaac Lab.

Current code keeps an independent deque per observation term, pushes newest frames at the end, and concatenates by term.

4. `dof_vel_scale` is `0.05`.

There was a previous suspected estimator/deployment scale mismatch. Current `StateRL.cpp` has `/rl_scale_mismatch_debug` for checking this.

5. Deployment-side clamp is a safety guard, not a training objective.

If the policy often triggers:

```text
RL output_dof_pos was clamped
```

then the policy itself is still unsafe or mismatched.

## Debug Topics

Current `StateRL.cpp` publishes:

```text
/rl_obs_dimension_debug
/rl_scale_mismatch_debug
```

Meaning:

- `/rl_obs_dimension_debug`: checks current joint position versus wrong/default/correct baseline.
- `/rl_scale_mismatch_debug`: checks raw joint velocity and the scaled value fed to the network.

Useful commands:

```bash
ros2 topic echo /rl_obs_dimension_debug
ros2 topic echo /rl_scale_mismatch_debug
ros2 topic echo /joint_states
ros2 topic echo /actuators_cmds
ros2 topic echo /control_input
```

## Pre-Deployment Checklist

Before testing a new policy in sim-to-sim, check:

- `config.yaml` `model_name` points to the intended policy.
- The policy is a blind student if it is meant for deployment.
- If the model is teacher-only or expects privileged latent, the zero-latent fallback is not a valid proof of deployability.
- `num_observations`, `observations`, and `observations_history` match the exported training policy.
- `default_dof_pos` matches the training default pose.
- `action_scale` matches training, especially box joint `0.02`.
- `output_dof_pos_lower/upper` still protects box joints in `0.00` to `0.06`.
- Flat-ground x/y/yaw tracking works before highstep testing.
- On highstep: front legs place on the step, rear legs push, box joints move sensibly.
- Box joint targets do not repeatedly hit clamp.
- Mujoco success is not enough for real hardware; real tests should increase speed and terrain difficulty gradually.

## Relation To Training Memory

Deployment currently observes only blind proprioceptive history, commands, and previous action. It does not receive a terrain height map. Therefore:

- Highstep teacher abilities that rely on privileged height scan cannot be deployed directly.
- Final deployment must use student distillation, and the student must preserve flat gait and highstep motion under blind observations.
- If teacher behavior depends on environment-side action prior, deployed student either needs equivalent deployment-side logic or must internalize that prior into raw actions.
- Sidestep previously failed when action prior, box direction, and no-prior student targets were inconsistent. Highstep distillation should avoid repeating that failure mode.

## Future Modification Targets

If deployment code needs changes, likely files are:

```text
StateRL.cpp
  - policy input shape, observation assembly, output clamp, debug topics

config/rl_policy/config.yaml
  - model_name, default_dof_pos, action_scale, output_dof_pos clamp

robot_control.yaml
  - controller joint order, stand_pos_adjustable_leg, update_rate

HardwareMujoco.cpp
  - sim-to-sim topic read/write

adjustable_leg_mujoco_msg_handler.cpp
  - Mujoco PD control formula, contact/IMU/joint state publishers

HardwareArcdog_adjustable_leg.cpp
  - real robot USB/CAN, motor activation, joint-limit emergency stop
```

Back up deployment files before changing them. Use the relevant training or deployment test run name as suffix, for example `_2026-06-17_03-05-16`.
