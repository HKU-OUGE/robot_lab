# RobotLab Codex 起点

> **2026-07-30 Highstep 阶段归档：**
> E7700 Student 已在有人保护的真机实验中完成整机上高台，并在用户切换
> `Fixed Down` 前保持站立。当前阶段已归档，无自动训练、评估或部署任务。
> 继续工作前先读
> [`library/highstep_stage_archive_20260730.md`](library/highstep_stage_archive_20260730.md)。
> 下文 v1.11 和更早“当前”描述均为历史快照。

这是把 `dev_lxq_new` 分支交给另一台电脑或一个全新 Codex 会话时的稳定入口。

可直接转发给协作者的中文 PDF 手册：`deliverables/highstep_collaborator_handoff_20260715.pdf`。

## 先判断你在哪种环境

### 原工作站

如果仓库同时存在 `tmp/highstep_dashboard_active_workflow.json` 及它声明的 state、heartbeat、handoff 和 preregistration：

1. 先读仓库根目录 `AGENTS.md`；
2. 运行 `python3 .agents/skills/highstep-control-variable-guardian/scripts/highstep_guard.py audit --json`；
3. 只从 dashboard 声明的链路解析实时 authority；
4. audit 不通过时 fail-closed，不能用 GitHub 快照或聊天记忆替代实时 SHA。

### 新电脑或普通 clone

Git 不包含原工作站的 `tmp/`、`logs/`、`wandb/`、checkpoint、rosbag 和视频。缺少这些文件不表示原流程已停止，也不构成恢复授权。

新 clone 默认只能：

- 阅读和审查代码；
- 重放纯 Python 测试；
- 根据冻结的 spec、preregistration 和历史报告继续分析；
- 列出继续实验还缺哪些外部 artifact。

在 checkpoint、运行配置和数据被独立传输并重新校验前，不得声称已完整恢复训练或评估。

## 固定阅读顺序

1. `AGENTS.md`
2. 本文件
3. `library/highstep_repository_transfer_handoff_20260715.md`
4. `library/highstep_student_recovery_spec_20260712.md` 顶部当前有效版本
5. `library/highstep_teacher_robustness_ab_decision_20260715.md`
6. `INDEX.md`
7. 按问题读取 `failure_modes_and_lessons.md`、0707 真机数据报告和其它历史记忆

## 当前分支的真实性边界（2026-07-30）

- 分支：`dev_lxq_new`
- GitHub：`HKU-OUGE/robot_lab`
- 当前归档入口：`library/highstep_stage_archive_20260730.md`
- 当前结果：E7700 Student 已完成有人保护的真机整机上台
- 当前工作：阶段归档；没有活动的 Student/Teacher 训练、评估或自动部署
- 2026-07-30 以前的训练路线和旧“当前”描述：历史证据，不得自动恢复
- GitHub 内的 runtime snapshot：只代表写入时刻，不会伪装成实时 heartbeat

如果未来正式 authority 更新，应新增带日期的 handoff，并更新本文件和两个 memory index；不要删除旧证据或悄悄改写旧结果。
