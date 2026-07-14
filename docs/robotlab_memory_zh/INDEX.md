# RobotLab 记忆索引

更新时间：2026-07-15，Asia/Hong_Kong。

这是未来新对话的常驻入口。先读这个文件，再根据问题按需打开下层细节文件。

## 2026-07-15 当前入口（覆盖下方旧“当前”描述）

- [Codex 新电脑/新会话起点](../CODEX_START_HERE.md)：区分原工作站实时 authority 与 GitHub 冻结快照；克隆后必须先读。
- [GitHub 可移植交接：代码、历史与 v1.11 当前进度](library/highstep_repository_transfer_handoff_20260715.md)：本分支的代码范围、历史阅读顺序、外部 checkpoint/日志边界、当前只读 Teacher A/B 流程及恢复限制。
- [Highstep Student 恢复与自动化规范 v1.11](library/highstep_student_recovery_spec_20260712.md)：当前正式最高规范。顶部 v1.11 覆盖所有冲突旧章节；v1.10 及以下仅为历史证据。
- [新旧 Teacher 零训练鲁棒性 A/B 决策](library/highstep_teacher_robustness_ab_decision_20260715.md)：当前唯一控制变量、只读约束与终态。

下面 2026-07-12 及更早的“当前最高优先级”“当前重点”和 PID/恢复命令全部保留为历史链路，不得覆盖上述入口或直接作为启动依据。

## 最高优先级流程约束

- [Highstep 0707 真机视频/rosbag 审计与暂停断点：2026-07-12](library/highstep_0707_real_log_audit_handoff_20260712.md)：**当前最高优先级入口**。两组独立真机数据已确认 158797 的后支撑内收/卡边，同时发现 revolute target 越物理限位、policy 切换 60--70 ms 全零掉线、部署配置指纹缺失和跨进程 schedule 重置。训练/监督器已安全停止，可靠断点为 `2026-07-11_23-45-31/model_600.pt`；修完 schedule、严格门禁和部署安全前禁止直接续训。
- [Highstep 0707 真机数据复盘、训练链审计与整改报告：2026-07-12](library/highstep_0707_real_data_analysis_20260712.md)：两组视频/bag 的精细配准、量化互证、0707→当前完整修改链、有效/失效/副作用分类，以及 schedule、target contract、StateRL 和 schema-5 严格门禁的整改状态。
- [Highstep 显示链卡死紧急暂停与恢复交接：2026-07-11](library/highstep_display_hang_emergency_pause_handoff_2026-07-11.md)：**当前最高优先级入口**。机器已重启，clean B 已从 iter401 派生锚点恢复；脚本已执行一次，禁止重跑。实时事实以新 schema-v4 handoff 为准。
- [Highstep 对话迁移交接：Robust clean B 决策与恢复期经验](library/highstep_dialogue_migration_clean_b_2026-07-11.md)：记录 clean B lineage、恢复期解释、45 场门禁、当前 run，以及自动化已经改绑本对话的事实。
- [Highstep 新对话交接：2026-07-03](library/highstep_new_dialogue_handoff_2026-07-03.md)：ActionScore/refine 失败后的新对话入口。下一轮必须先做零训练评分校准，再考虑修改 support 目标；不要直接开训。
- [Highstep Live Context Handoff](library/highstep_live_context_handoff.md)：压缩/恢复用的当前交接文件。每次回答前必须先读；如果读取成功，回答开头必须带 `HLC-OK` 标识。该文件只保存当前有效决策态，采用覆盖更新，不做无限追加。
- [Highstep Automation V4 Short-Term Handoff](library/highstep_automation_v4_handoff_20260711.md)：2026-07-11 当前训练、schema-v4 闭环升级和安全切换 monitor/orchestrator 的精确恢复点；上下文压缩后优先读取。
- [Codex 工作责任协议](library/accountability_protocol.md)：每次回答、判断、命令建议、代码修改、训练分析、视频分析之前必须先参考。它记录了用户要求的最高规格约束：不能用模型档位推脱；判断必须有依据；命令必须解释风险；改代码前必须备份；不确认前提不能给结论；不得同时给互相冲突的建议；每个现象必须映射到数据、视频和代码机制。
- [Highstep 短期记忆：2026-06-25 20:00 之后](library/highstep_short_term_2026-06-25_20h.md)：后续每次回答前必须先读；回答开头必须写“已按照要求提前检索记忆和约束｜HLC-OK”或失败时写 `HLC-MISSING`。记录 2026-06-25 晚上之后的 student 烂训、修复、teacher 阶段任务改动、WandB 判断和当前训练状态。

## 当前重点

- 主 teacher 任务：`RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`
- 主 student 任务：`RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0`
- 暴露问题的 student 分支：`logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-17_19-54-10/model_69998.pt`
- 当前 student 判断：上高台能力勉强够用，但 `bad_orientation` 比 teacher 高，键盘后退时容易后仰触发终止。只能作为 Mujoco sim-to-sim 临时探测 policy，不能当最终部署 policy。
- student-first 尝试效果不够后，当前路线已切回先强化 teacher，再重新蒸馏。
- 当前 highstep teacher 精修锚点：优先从 `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt` 接当前新的后腿分支平衡代码重新训练，不优先从 `2026-06-23_00-52-26/model_97300.pt` 继续。`model_97300.pt` 是从 `model_95398.pt` 在旧奖励组合下继续约 1900 update 得到的，可能已经吸收了“单后腿上台/box push 过强”的错误引导；只作为对照/备选。
- 除非明确研究失败，不要从 `2026-06-23_00-52-26/model_105397.pt` 继续 highstep teacher 精修。视频显示它仍保留右后腿先登台的成功分支，但主导行为太容易塌到更弱的左后腿先登台分支。
- 当前 highstep teacher command 已覆盖轻微后退：`lin_vel_x=(-0.30, 0.85)`，后退稳定不再只靠 student 蒸馏补。
- 当前 box joint 安全原则：平地/非前进场景强力保持默认位置，只在主动前进上高台时释放，用 `highstep_box_default_position_penalty` 实现。
- 当前 Mujoco 临时测试导出：用 student run 下的 `exported/policy_student.pt`。它接受部署端 570 维盲走观测；不要用 `exported/policy.pt`，后者需要 634 维 actor+latent 输入。

## 按需读取

- [当前状态](library/current_state.md)：最新 checkpoint、当前建议、关键配置值。
- [Highstep Live Context Handoff](library/highstep_live_context_handoff.md)：压缩/恢复用当前交接文件，必须先读。
- [Highstep 短期记忆：2026-06-25 20:00 之后](library/highstep_short_term_2026-06-25_20h.md)：近期最高优先级上下文，必须先读。
- [Codex 工作责任协议](library/accountability_protocol.md)：最高优先级流程约束，必须先读。
- [Highstep 训练链路](library/highstep_training_chain.md)：哪些 WandB run 要保留/隐藏，以及原因。
- [Highstep 历史分析与回答摘录：2026-06-25 至 2026-07-03](library/highstep_historical_reasoning_notes_20260625_20260703.md)：从源码目录整理出的早期 Teacher、terrain、reward 和蒸馏判断；仅作历史证据。
- [Bodyflat 与 Sidestep 记忆](library/bodyflat_sidestep_memory.md)：bodyflat teacher/student 历史、sidestep 先验动作经验。
- [仓库使用规范](library/repo_usage_rules.md)：命令、备份、日志、视频、监控、安全规则。
- [Arcdog 可伸缩腿模型](library/arcdog_adjustable_leg_model.md)：关节顺序、action scale、box joint 方向、执行器参数。
- [WandB 曲线判断指南](library/wandb_metrics_guide.md)：如何看曲线，避免被 terrain levels 误导。
- [失败模式与经验教训](library/failure_modes_and_lessons.md)：反复踩过的坑和处理原则。
- [部署端笔记](library/deployment_notes.md)：ROS2/Mujoco/sim-to-real 路径和 student policy 约束。
- [Sim-To-Sim 部署链路](library/sim_to_sim_deployment.md)：从两个 launch 指令反推 Mujoco、ros2_control、StateRL、policy config 的完整数据流。

## 必须保留的硬规则

- 最高优先读取 [Highstep Live Context Handoff](library/highstep_live_context_handoff.md)、[Highstep 短期记忆](library/highstep_short_term_2026-06-25_20h.md) 和 [Codex 工作责任协议](library/accountability_protocol.md)。每次判断必须有依据；每次命令必须解释风险；每次改代码前必须备份；不确认前提不能给结论；不能同时给互相冲突的建议。
- 每次回答开头必须写：`已按照要求提前检索记忆和约束｜HLC-OK`。如果 live 交接文件缺失或读取失败，必须写：`已按照要求提前检索记忆和约束｜HLC-MISSING` 并说明原因。
- 改代码前必须备份所有要改的文件。备份后缀用相关训练 log 文件夹名，例如 `_2026-06-17_03-05-16`。
- Highstep 不能只看 `Curriculum/terrain_levels`；play/video 才是最终判断。
- `terrain_levels` 是训练环境 terrain row 平均占用，不是“最大能爬多高”的直接分数。bodyflat 的 5.x 和 highstep 的 2.7-3.3 不能脱离任务、curriculum 函数、command 范围和地形混合比例直接对比。
- Highstep 必须显式观察左后腿先登台和右后腿先登台的差别。平均化的后足奖励会掩盖单侧分支崩坏。成功上台常常依赖“已经登上高台的那条后腿作为支撑/驱动腿把身体顶起来，再带动另一条后腿清过台阶边缘”。
- 长训尽量不要开 `--video`，训练中录视频多次增加卡死风险。
- Highstep 不要重新引入 sidestep 的 hip 外展/足端宽度约束，除非明确要做混合任务。
- box joint 方向必须记住：`box_joint` 数值越小，腿越长；数值越大，腿越短。
- 可以临时测试 highstep student，但当前 student 不是最终版本；必须等后退稳定性和 `bad_orientation` 改善后再作为部署候选。
