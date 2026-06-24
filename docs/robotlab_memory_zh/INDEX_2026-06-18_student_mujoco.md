# RobotLab 记忆索引

更新时间：2026-06-18，Asia/Hong_Kong。

这是未来新对话的常驻入口。先读这个文件，再根据问题按需打开下层细节文件。

## 当前重点

- 主任务：`RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`
- 当前有用 teacher run：`logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-17_03-05-16`
- 当前视觉锚点：`model_63700.pt`
- 当前判断：先从 `model_63700.pt` 继续 teacher，不要急着切 student。如果后续 checkpoint 视觉效果变差，退回 `model_63700.pt`。
- 当前 highstep command gate：`terrain_gate_level` 已从 `3.0` 放宽到 `2.6`，command 已能到当前最高 `0.6375`。

## 按需读取

- [当前状态](library/current_state.md)：最新 checkpoint、当前建议、关键配置值。
- [Highstep 训练链路](library/highstep_training_chain.md)：哪些 WandB run 要保留/隐藏，以及原因。
- [Bodyflat 与 Sidestep 记忆](library/bodyflat_sidestep_memory.md)：bodyflat teacher/student 历史、sidestep 先验动作经验。
- [仓库使用规范](library/repo_usage_rules.md)：命令、备份、日志、视频、监控、安全规则。
- [Arcdog 可伸缩腿模型](library/arcdog_adjustable_leg_model.md)：关节顺序、action scale、box joint 方向、执行器参数。
- [WandB 曲线判断指南](library/wandb_metrics_guide.md)：如何看曲线，避免被 terrain levels 误导。
- [失败模式与经验教训](library/failure_modes_and_lessons.md)：反复踩过的坑和处理原则。
- [部署端笔记](library/deployment_notes.md)：ROS2/Mujoco/sim-to-real 路径和 student policy 约束。
- [Sim-To-Sim 部署链路](library/sim_to_sim_deployment.md)：从两个 launch 指令反推 Mujoco、ros2_control、StateRL、policy config 的完整数据流。

## 必须保留的硬规则

- 改代码前必须备份所有要改的文件。备份后缀用相关训练 log 文件夹名，例如 `_2026-06-17_03-05-16`。
- Highstep 不能只看 `Curriculum/terrain_levels`；play/video 才是最终判断。
- 长训尽量不要开 `--video`，训练中录视频多次增加卡死风险。
- Highstep 不要重新引入 sidestep 的 hip 外展/足端宽度约束，除非明确要做混合任务。
- box joint 方向必须记住：`box_joint` 数值越小，腿越长；数值越大，腿越短。
- teacher 在完整当前 command 范围下多个 checkpoint 视觉稳定之前，不要急着切 student。
