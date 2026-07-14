# E1400 continuation A/B 决策记录

## 为什么原 E1400 停止判断不充分

原 v1.8 的 E1400 上限是预先固定的流程边界，但后来被错误地解释成了模型已经收敛失败。这个判断遗漏了四个同时发生的边界：terrain 第三级到 E1300 才开放，support bottleneck 到约 E1350 才完全生效，score curriculum 在 E1400 切换，Student actor warmup 也在 E1400 结束。E1400 因而只在完整 support 条件下学习了很短时间，并没有真正测试 post-warmup box-row adaptation。

同时，已有行为结果明显非单调：E300 为 4/9，E500、E900 回落，E1400 又恢复到 5/9；E1000--E1300 中间点评估也呈振荡而非稳定平台。只看一个终点、没有先把训练曲线与课程边界叠加，是本次判断错误的直接原因。

## 本轮防错设计

两支均从同一个 E1400 完整 checkpoint 和原 optimizer 独立恢复，最多增加 300 updates，并在 E1500/E1600/E1700 保存和评估。A 只延长 estimator；B 按历史 0707 机制开启四个 box rows/bias 适配。除此之外所有训练语义相同。任一保存点的结论必须基于两支同点 core9，禁止再次用单个课程边界、loss 或单场表现作终止判断。
