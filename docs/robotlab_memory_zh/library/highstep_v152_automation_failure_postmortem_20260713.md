# Highstep v1.5.2 自动化故障复盘（2026-07-13）

## 结论

训练 checkpoint 本身没有损坏。自动化停滞来自 v1.5.2 authority 没有贯穿“训练恢复→checkpoint scope→core9 聚合→directional 转换”整条消费链。

## 已发生的三个接口故障

1. `highstep_v15_imitation_gate.checkpoint_scope()` 曾只接受旧 v1.5 preregistration，拒绝正确的 v1.5.2 rebound checkpoint，导致 E500 被重复训练。
2. schema-4 monitor 曾要求训练 task 与评估 task 字符串完全相同，未登记 v1.5.2 的精确别名对：
   - train: `RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV15-ArcdogAdjustableLeg-v0`
   - eval: `RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0`
   因而把真实完整的 9 场行为失败误判成 lineage/runtime evidence 无效。
3. v1.5.2 directional 转换错误地从 schema-7 eval payload 读取不存在的 `lateral_offset_m`；该字段实际属于经过验证的 ledger row，payload 只提供 `yaw_offset_deg` 见证。

## 修复原则

- task alias 只对 `highstep_student_recovery_v152_20260713` 和上述精确 task 对生效。
- 不放宽 checkpoint SHA、Teacher lineage、schedule、runtime snapshot、action contract、9/9 完整性或行为阈值。
- 明确区分：
  - evidence invalid：不能用于决策，必须修复或重跑；
  - behavior failed：证据有效但策略未成功，必须按固定保存点继续训练。
- 已完成且 SHA/运行合同一致的训练和评估证据必须复用，禁止因后处理代码错误重复消耗 GPU。

## 防复发门禁

以后任何 workflow authority 变更必须建立 consumer matrix，并逐项验证：

1. 训练配置读取 authority；
2. full checkpoint 恢复读取 authority；
3. checkpoint scope 接受并验证 authority；
4. imitation/reference gate 接受并验证 authority；
5. core9 monitor 验证训练 task→评估 task 映射；
6. aggregate manifest 能把行为失败记录为有效结果；
7. directional 能消费真实 ledger schema；
8. W&B 完成同步；
9. supervisor 实际启动下一个保存点，而不是仅通过静态测试。

保存点恢复 smoke 的完成定义从“训练 1–5 iteration 成功”升级为：至少贯穿一次合成/冻结 fixture 的 `checkpoint_scope → aggregate → directional row conversion → continue decision`。真实流程修复完成的验收标准是观察到下一训练阶段 PID。

## 本次恢复证据

- E500 checkpoint SHA256：`9811759e0b0ae44613101cd81fbd8cb4d4c41a01840d26d7e11df43d66ba8450`
- core9：valid 9/9，full 0/9，rear_hold 0/9，front_top_support 0/9，no_severe_inward 9/9。
- 左右 directional corridor：均 valid 9/9，full/rear_hold 0/9；被正确分类为 `behavior_not_yet_successful`。
- E500 W&B run `3ryi16p9` 已同步。
- E900 已从上述精确 E500 full checkpoint 启动，additional effective updates=400。
