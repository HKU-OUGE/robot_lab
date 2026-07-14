# 新旧 Teacher 零训练鲁棒性 A/B 决策记录

## 目的

v1.10 Student E1400 continuation 已暂停且不启动训练。当前先回答一个更上游的问题：新 Teacher `model_172300` 相比 0707 使用的旧 Teacher `model_151399`，是否真的在后腿内收恢复、分阶段横向冲量和组合压力下获得了可重复的明显鲁棒性提升。

## 唯一变量

唯一变量是 Teacher checkpoint。两者共享同一个 Robust Teacher 环境、action prior、real gains、delay、固定命令、平台、seed、逐场物理 snapshot 和干预时序。普通随机事件关闭；只允许预注册的初始后腿内收或单次机身横向速度冲量。

本轮为纯评估：不调用训练、backward 或 optimizer.step，不修改模型、checkpoint、Student、reward、DR、history、网络或部署合同。

## 决策边界

所有矩阵、恢复定义、等级通过标准和“明显改善”阈值均在第一场结果出现前冻结。完成 Teacher A 后只能让 Teacher B 复用 A 冻结的逐场物理 snapshot；不能按 B 的结果调整初始状态或干预方向。

完成后停止，只报告 A/B 对照。无论结论如何，都不得自动启动任何 Student 或 Teacher 训练，也不得自动恢复 v1.10 continuation。
