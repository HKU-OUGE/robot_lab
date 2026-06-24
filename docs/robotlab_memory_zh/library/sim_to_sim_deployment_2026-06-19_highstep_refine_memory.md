# Sim-To-Sim 部署链路记忆

更新时间：2026-06-18，Asia/Hong_Kong。

部署端仓库：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

常用 sim-to-sim 启动命令：

```bash
ros2 launch mujoco_simulator mujoco.launch.py
ros2 launch rl_quadruped_adjustable_leg_controller mujoco_adjustable_leg.launch.py
```

## 一句话结论

这套 sim-to-sim 部署不是直接运行 Isaac policy，而是：

```text
Mujoco 仿真节点
  -> 发布 /imu_data, /joint_states, /mujoco_msg
ros2_control HardwareMujoco
  -> 把这些 topic 转成 controller state interfaces
rl_quadruped_adjustable_leg_controller
  -> StateRL.cpp 拼观测、跑 TorchScript policy、输出关节目标
HardwareMujoco
  -> 发布 /actuators_cmds
Mujoco handler
  -> 用 kp * pos_error + kd * vel_error + torque 写入 Mujoco actuator ctrl
```

最终部署目标仍然是盲走 student policy；teacher 在 Isaac 里能用，不等于部署端能直接用。

## 启动链路

### 1. Mujoco 仿真端

入口：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/mujoco_simulator/launch/mujoco.launch.py
```

默认参数：

```text
robot_pkg = arcdog_adjustable_leg_description
xml_file_path = arcdog_adjustable_leg_description/xml/scene.xml
```

`scene.xml` 会 include：

```text
robot_description/arcdog_adjustable_leg_description/xml/arcdog_adjustable_leg.xml
```

Mujoco 主程序：

```text
mujoco_simulator/src/mujoco_node.cpp
```

当前代码里 `robot_type = 4`，因此实际使用：

```text
ArcLab::AdjustableLegMujocoMsgHandler
```

相关文件：

```text
mujoco_simulator/src/adjustable_leg_mujoco_msg_handler.cpp
mujoco_simulator/include/mujoco_node/adjustable_leg_mujoco_msg_handler.h
```

这个 handler 做三件事：

- 每 1 ms 发布 `imu_data`
- 发布 `joint_states`
- 发布 `mujoco_msg`
- 订阅 `actuators_cmds`，将控制器输出转成 Mujoco actuator 控制量

Mujoco 里实际控制公式：

```text
ctrl = kp * (target_pos - joint_pos) + kd * (target_vel - joint_vel) + torque
```

### 2. 控制器端

入口：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/rl_quadruped_adjustable_leg_controller/launch/mujoco_adjustable_leg.launch.py
```

默认描述包：

```text
pkg_description = arcdog_adjustable_leg_description
```

这个 launch 会读取：

```text
arcdog_adjustable_leg_description/xacro/robot.xacro
arcdog_adjustable_leg_description/config/robot_control.yaml
```

主要启动节点：

- `robot_state_publisher`
- `controller_manager/ros2_control_node`
- `joint_state_broadcaster`
- `imu_sensor_broadcaster`
- `rl_quadruped_adjustable_leg_controller`
- `joy/game_controller_node`
- `rviz2`

注意：这个 sim launch 里定义了 `cmd_mapping` 节点，但当前 `nodes_to_start` 没有把它加入返回列表。也就是说 sim-to-sim 里主要靠控制器内部直接订阅 `/joy`，而不是靠 `cmd_mapping`。

真机 launch：

```text
rl_quadruped_adjustable_leg_controller/launch/real_robot_adjustable_leg.launch.py
```

真机会读取：

```text
arcdog_adjustable_leg_description/xacro/real_robot.xacro
```

区别是底层硬件插件不同：

```text
sim-to-sim: hardware_mujoco/HardwareMujoco
real robot: hardware_arcdog_adjustable_leg/HardwareArcdog_adjustable_leg
```

## ROS2 Control 与数据流

Sim-to-sim 硬件接口：

```text
hardware_mujoco/src/HardwareMujoco.cpp
```

它订阅：

```text
/joint_states
/imu_data
```

它发布：

```text
/actuators_cmds
```

它暴露给 controller 的接口：

```text
state_interfaces: position, velocity, effort
command_interfaces: position, velocity, effort, kp, kd
```

真机硬件接口：

```text
hardware_arcdog_adjustable_leg/src/HardwareArcdog_adjustable_leg.cpp
```

真机通过 USB/CAN 读写真实电机，并有独立电机使能逻辑：

```text
motor_mode = 1: 失能
motor_mode = 8: 使能准备
motor_mode = 0: 允许控制命令驱动
```

真机侧有硬件安全检查：

```text
limit_prismatic_.min = -0.015
limit_prismatic_.max =  0.065
```

但训练和部署 policy 的目标仍应按真实安全范围 `0.00` 到 `0.06` 处理，不要依赖真机急停兜底。

## RL 控制器状态机

核心控制器：

```text
rl_quadruped_adjustable_leg_controller/src/RlQuadrupedControllerAdjustableLeg.cpp
```

核心 RL 状态：

```text
rl_quadruped_adjustable_leg_controller/src/FSM/StateRL.cpp
rl_quadruped_adjustable_leg_controller/include/rl_quadruped_adjustable_leg_controller/FSM/StateRL.h
```

状态切换逻辑在 `StateRL::checkChange()` 和 controller `/joy` 回调里。

手柄命令大致是：

```text
RB + B -> command = 0 -> PASSIVEADJUSTABLELEG
RB + A -> command = 1 -> FIXEDDOWNADJUSTABLELEG
RB + Y -> command = 2 -> FIXEDSTANDADJUSTABLELEG
RB + X -> command = 3 -> RL
```

手柄速度缩放：

```text
lx = 0.5 * axes[1]
ly = 0.34 * axes[0]
rx = -1.0 * axes[2]
StateRL 中 control.yaw = -rx
```

所以部署端最大前向速度命令当前只有约 `0.5`，侧向约 `0.34`。这和训练端 command range 不一定完全相同，排查 sim-to-sim 行走弱时要一起看。

## Policy 配置

当前部署 policy 配置：

```text
robot_description/arcdog_adjustable_leg_description/config/rl_policy/config.yaml
```

当前 `model_name`：

```text
policy_student_2026-06-12_23-33-05_for_quick_deploy.pt
```

这意味着：如果要部署最新 highstep teacher/student，需要先导出 TorchScript policy，复制到：

```text
robot_description/arcdog_adjustable_leg_description/config/rl_policy/
```

然后修改 `config.yaml` 的：

```text
model_name: "新的_policy.pt"
```

2026-06-18 临时 highstep student 测试：

```text
来源 run:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10

checkpoint:
model_69998.pt

Mujoco 应使用的 TorchScript:
exported/policy_student.pt
```

已验证输入维度：

```text
exported/policy_student.pt 接受 570 维盲走部署观测并输出 16 维 action。
exported/policy.pt 接受 634 维 actor+latent 输入，不适合当前 Mujoco 盲走 student 测试。
```

建议临时部署文件名：

```text
policy_student_highstep_2026-06-17_19-54-10_model_69998.pt
```

已知问题：这个 student 不是最终版；后退/负 x command 仍可能后仰，`bad_orientation` 高于 teacher。

关键配置：

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

`StateRL.cpp` 会把观测按 Isaac Lab 的顺序拼成：

```text
ang_vel 历史 10 帧
gravity_vec 历史 10 帧
commands 历史 10 帧
dof_pos 历史 10 帧
dof_vel 历史 10 帧
actions 历史 10 帧
```

总输入维度：

```text
57 * 10 = 570
```

这是之前专门修过的点：不能把每一帧完整 obs 交错拼接，必须按“每个观测项连续历史”拼接，才能和 Isaac Lab 导出的 student policy 对齐。

## StateRL 推理逻辑

`StateRL.cpp` 的核心流程：

```text
getState()
  -> 读 IMU、关节 q/dq、手柄命令
runModel()
  -> 构造 obs
  -> forward policy
  -> action clip
  -> actions_scaled = action * action_scale
  -> output_dof_pos = actions_scaled + default_dof_pos
  -> output_dof_pos clamp
  -> 写 position/velocity/kp/kd/torque command interface
```

当前代码会先尝试：

```text
model(final_obs)
```

如果失败，会自动补一个 64 维 zero latent：

```text
model(concat(final_obs, zero_latent))
```

这个 fallback 可以避免节点直接崩，但也可能掩盖“导出的 policy 输入维度不对”的问题。部署前最好确认导出的 student policy 本身就是盲走输入维度 `570`。

## 关节顺序

`robot_control.yaml` 里 RL controller 的关节顺序是按关节类型分组：

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

但 `config.yaml` 里很多数组以每条腿为一组写：

```text
[hip, thigh, calf, box] * 4
```

`StateRL.cpp` 读取 YAML 时，如果 `framework == "isaacsim"`，会用 `rows=4, cols=4` 做转置，把每腿分组转成 controller 需要的分组顺序。

因此：

```text
action_scale: [0.1, 0.1, 0.1, 0.02] * 4
```

最终会变成：

```text
4 个 hip scale = 0.1
4 个 thigh scale = 0.1
4 个 calf scale = 0.1
4 个 box scale = 0.02
```

当前 `hip_scale_reduction_indices` 是：

```text
[0, 1, 2, 3]
```

这对应分组顺序下的 4 个 hip。早期配置里出现过 `[0,4,8,12]`，那是按每腿分组理解的索引，不适合当前转置后的 controller 顺序。

## Box Joint 方向和安全

必须一直记住：

```text
box_joint 数值越小，腿越长
box_joint 数值越大，腿越短
```

训练和部署约束：

```text
真实目标范围: 0.00 到 0.06
默认值: 0.03
```

当前 `config.yaml` 已经有最终 target clamp：

```text
output_dof_pos_lower: box joint = 0.0
output_dof_pos_upper: box joint = 0.06
```

这个 clamp 发生在：

```text
output_dof_pos = action * action_scale + default_dof_pos
```

之后。

注意：`arcdog_adjustable_leg_description/xml/arcdog_adjustable_leg.xml` 里 Mujoco slide joint 当前范围更宽：

```text
range="-0.05 0.10"
```

而 xacro/URDF 的 prismatic joint 是：

```text
lower="0"
upper="0.06"
```

之前的原则是：Mujoco XML 可以偏宽，用来暴露策略是否有超限趋势；但最终部署端 `StateRL.cpp + config.yaml` 必须把目标 clamp 到真实范围内。不能把 “Mujoco 看起来没撞限位” 当成真机安全。

## 重要历史修复

这些是之前 sim-to-sim / sim-to-real 对话里已经踩过的坑：

1. `StateRL.cpp` 必须在 CPU 上加载 TorchScript：

```text
torch::jit::load(model_path, torch::kCPU)
```

否则可能出现 CUDA backend 相关崩溃。

2. `default_dof_pos` 必须写在 `config.yaml` 中。

如果缺失，代码会回退到 `stand_pos_adjustable_leg_`，这会让 `dof_pos - default_dof_pos` 的相对位置错误，尤其会影响 VAE/student latent。

3. 观测历史顺序必须和 Isaac Lab 一致。

当前修复逻辑是每个 observation term 独立维护 deque，最新帧放末尾，最后按 term 拼接。

4. `dof_vel_scale` 是 `0.05`。

之前曾怀疑 estimator/部署端尺度错位；当前 `StateRL.cpp` 有 `/rl_scale_mismatch_debug` 用于排查速度缩放。

5. 部署端 clamp 是安全保护，不是训练目标。

如果 policy 经常触发 `RL output_dof_pos was clamped`，说明策略本身仍有问题，不应该靠 clamp 掩盖。

## Debug Topic

`StateRL.cpp` 当前发布两个调试 topic：

```text
/rl_obs_dimension_debug
/rl_scale_mismatch_debug
```

含义：

- `/rl_obs_dimension_debug`：帮助检查当前关节位置、错误 default、正确 default 之间的相对位置差异。
- `/rl_scale_mismatch_debug`：帮助检查真实关节速度和喂给网络的缩放速度。

常用检查命令：

```bash
ros2 topic echo /rl_obs_dimension_debug
ros2 topic echo /rl_scale_mismatch_debug
ros2 topic echo /joint_states
ros2 topic echo /actuators_cmds
ros2 topic echo /control_input
```

## 部署前检查清单

换一个新 policy 到 sim-to-sim 前，至少检查：

- `config.yaml` 的 `model_name` 是否指向真正要测试的 policy。
- policy 是否是盲走 student；如果是 teacher 或带 privileged latent 的模型，部署端可能只能靠 fallback dummy latent，效果不可当真。
- `num_observations`、`observations`、`observations_history` 是否和训练导出一致。
- `default_dof_pos` 是否和训练默认关节位一致。
- `action_scale` 是否和训练环境一致，特别是 box joint `0.02`。
- `output_dof_pos_lower/upper` 是否仍保护 box joint 在 `0.00` 到 `0.06`。
- 平地 x/y/yaw 是否能正常跟踪。
- 上台阶时前腿是否能搭上去，后腿是否能推进。
- box joint 是否有合理差异，且没有持续撞限位。
- Mujoco 里能走，不等于真机能走；真机测试前还要低速、低高度、逐步放开。

## 和训练端记忆的关系

部署端当前只看本体状态历史、命令、上一次 action，没有地形高度图。因此：

- Highstep teacher 在 Isaac 中依赖 privileged height scan 的能力不能直接部署。
- 最终部署必须走 student 蒸馏，且 student 要在 blind obs 下保住平地步态和上地形动作。
- 如果 teacher 好效果依赖环境侧 action prior，那么部署端要么保留等价 action prior，要么让 student 把 prior 内化进 raw action。
- Sidestep 任务曾经因为 action prior / box 方向 / student no-prior 目标不一致，导致“姿态像但走路坏掉”的问题；Highstep 后续蒸馏必须避免重复。

## 未来修改建议

如果后续要改部署端，优先考虑这些位置：

```text
StateRL.cpp
  - policy 输入维度、obs 拼接、输出 clamp、debug topic

config/rl_policy/config.yaml
  - model_name、default_dof_pos、action_scale、output_dof_pos clamp

robot_control.yaml
  - controller joint order、stand_pos_adjustable_leg、update_rate

HardwareMujoco.cpp
  - sim-to-sim topic read/write

adjustable_leg_mujoco_msg_handler.cpp
  - Mujoco PD 控制公式、接触/IMU/关节状态发布

HardwareArcdog_adjustable_leg.cpp
  - 真机 USB/CAN、motor activation、关节限位急停
```

改部署代码前也要备份；部署端备份后缀建议沿用当次测试或训练 log 名，例如 `_2026-06-17_03-05-16`。
