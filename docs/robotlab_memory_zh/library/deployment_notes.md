# 部署端笔记

> **2026-07-30 更新：**最终 E7700 Student 已完成有人保护的真机整机上台。
> 当前准确部署证据、TorchScript SHA、真机 bag 指纹和参考 gains 见
> [highstep_stage_archive_20260730.md](highstep_stage_archive_20260730.md) 与
> [config_e7700_real_test_20260723_reference.yaml](config_e7700_real_test_20260723_reference.yaml)。
> 下文 model69998/E1600 等配置均为历史，不得作为当前 policy2 绑定。

部署仓库：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

已知仿真启动命令：

```bash
ros2 launch mujoco_simulator mujoco.launch.py
ros2 launch rl_quadruped_adjustable_leg_controller mujoco_adjustable_leg.launch.py
```

相关部署组件：

```text
robot_description: arcdog_adjustable_leg_description
controller: rl_quadruped_adjustable_leg_controller
推理/RL切换代码: StateRL.cpp
```

完整 sim-to-sim 启动链路、topic 数据流、StateRL 推理逻辑、关节顺序和 policy 配置见：

```text
library/sim_to_sim_deployment.md
```

## 部署 Policy 约束

最终部署目标是盲走 student policy。

注意：

- 如果 teacher 依赖环境侧 action prior，部署时 student 要么保留等价部署逻辑，要么必须把这种行为内化进 raw action。
- Sidestep 曾因此出现严重 student 蒸馏问题。
- Highstep 可以临时用 student 做 sim-to-sim 探测，但最终部署必须等盲走 student 的 `bad_orientation`、后退和平地稳定性达标。

## 当前临时 Highstep Student Mujoco 测试

截至 2026-06-19，当前 highstep student 仍然不是最终版本，但可以先临时放进 Mujoco 做 sim-to-sim 探测：

```text
Isaac checkpoint:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt

Mujoco 应使用的 TorchScript：
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/exported/policy_student.pt
```

用 `policy_student.pt`，不要用 `policy.pt`。

本地已验证：

```text
policy_student.pt: input 570 -> output 16
policy.pt: input 634 -> output 16
```

ROS2/Mujoco 的 `StateRL.cpp` 部署栈期望 570 维盲走 student 输入。当前部署配置仍然指向旧 quick deploy policy：

```text
policy_student_2026-06-12_23-33-05_for_quick_deploy.pt
```

临时测试当前 highstep student 时，把 `policy_student.pt` 复制到：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/robot_description/arcdog_adjustable_leg_description/config/rl_policy/
```

并修改：

```text
config/rl_policy/config.yaml
model_name: "policy_student_highstep_2026-06-17_19-54-10_model_69998.pt"
```

已知弱点：后退/负 x command 仍可能后仰。这个测试只用于快速看 Mujoco 信号，不代表最终部署验证通过。

当前训练路线：

```text
不要把这个 student 当作最终部署分支。
先精修 teacher 的后退稳定和平地 box joint 行为，再重新蒸馏新的盲走 student。
```

本次记忆更新时的 teacher refinement 候选：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-19_01-42-04/model_86400.pt
```

这仍然是 teacher checkpoint，不是可直接部署的盲走 policy。

## 安全 Clamp

部署端对 `box_joint` 加 clamp 可以作为安全保护，但训练不能依赖 clamp 掩盖策略错误。

如果使用 clamp，建议：

```text
box_joint target clamp: 0.00 到 0.06
```

但 Isaac play 中策略本身也应该自然待在这个范围。

## Sim-To-Sim 差距

Isaac play 可用，不代表 Mujoco 或真机一定可用。Highstep 部署前重点检查：

- 平地走路是否自然；
- 接近高台是否正常；
- 前腿是否能正确搭台；
- 后腿是否能推进；
- box joint 是否配合伸长；
- 是否撞关节限位。

## Mujoco 地形备注

Highstep 快速 sim-to-sim 检查使用的可伸缩腿 Mujoco 地形文件是：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2/robot_description/arcdog_adjustable_leg_description/xml/scene_terrain.xml
```

2026-06-18 highstep 测试时，旧地形块已注释，并加入了一组多高度高台，用来对应检查几个 Isaac 风格难度等级。这只是有针对性的部署端探测，不等价于 Isaac 完整训练分布。
