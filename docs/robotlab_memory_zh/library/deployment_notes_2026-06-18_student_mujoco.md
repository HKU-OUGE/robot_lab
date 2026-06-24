# 部署端笔记

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
- Highstep 当前不要在 teacher 稳定前切 student。

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
