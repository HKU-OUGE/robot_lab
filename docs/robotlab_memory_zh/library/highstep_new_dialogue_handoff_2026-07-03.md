# Highstep 新对话交接：ActionScore 失败后的固化约束

更新时间：2026-07-03，Asia/Hong_Kong。

用途：这是给下一段全新对话使用的最小高优先级交接文件。新对话不要沿着旧长对话继续猜；必须先读本文件，再读 `highstep_live_context_handoff.md`、`highstep_phase_task_logic_priority_2026-06-30_2026-07-01.md`、`highstep_short_term_2026-06-25_20h.md` 和 `accountability_protocol.md`。

## 当前结论

2026-07-03 的 ActionScore/refine 补丁链不能算成功，不能继续作为主线长训。

失败 run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-03_05-40-48
checkpoint source:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_141000.pt
last saved:
model_141500.pt
last local event observed:
step 141552
```

最后数据：

```text
Train/mean_reward tail20 ~= 83.43
Curriculum/terrain_levels tail20 ~= 0.185
Curriculum/highstep_action_score/total tail20 ~= 0.469
Curriculum/highstep_action_score/hard_total tail20 ~= 0.469
Curriculum/highstep_action_score/entry_score tail20 ~= 0.975
Curriculum/highstep_action_score/support_score tail20 ~= 0.344 < support_floor 0.45
Curriculum/highstep_action_score/support_floor_violation_rate tail20 ~= 0.992
Curriculum/highstep_action_score/post_lead_drive_signal_mean tail20 ~= 0.00186
Episode_Reward/post_lead_body_drive tail20 ~= 0.00627
```

解释：

```text
entry_score 很高，post_lead_drive_progress_mean 很高，但 support_score 低且 violation 接近 1。
这说明策略学到了进入/前进/局部抬升的代理行为，没有学到第一后腿搭台后真正支撑身体。
terrain_levels 退到很低后 total 变高，不是高台动作变好。
```

## 必须 defend 的观点

阶段瓶颈大方向仍然正确。

理由：

```text
2026-07-01_18-19-06 student 链路及其上游 teacher model_141000.pt 已证明：
流畅高台动作可以训出来；
但旧目标会出现 terrain/reward/MSE 继续变好、真实 highstep/support 动作退化的反例。
```

因此，不能回到只看 `terrain_levels`、总 reward、蒸馏 MSE 或普通 locomotion 的路线。必须继续以阶段目标为核心。

## 不能 defend 的实现

不能 defend 当前 ActionScore/refine 实现。主要错误：

1. 没有先做正负样本评分校准。
   - 新指标必须先证明能把 `model_141000.pt` 判成好样本，把后期退化 checkpoint 和当前 `model_141500.pt` 判成坏样本。
   - 没通过这个校准就开训，是严重流程错误。

2. 把“诚实暴露失败的指标”误当成“能训练出支撑动作的机制”。
   - support_floor_violation 能发现失败，但不会自动制造有效梯度。
   - 失败停止条件不能替代训练目标本身。

3. support reward 仍可被代理行为钻空子。
   - 当前 `support_score` / `post_lead_body_drive` 仍让 forward progress、second score 或 floor 项参与加权。
   - 结果可能是往前蹭、局部抬升、低 terrain 刷分，而不是真正支撑身体。

4. 从珍贵好 checkpoint 继续 PPO 时没有保护动作记忆。
   - `model_141000.pt` 是正样本锚点，不应直接用大改 reward surface 长训。
   - 下一步应该先保护和校准，再短训验证。

## 下一轮新对话的第一原则

新对话不要立刻改 reward 开训。先做零训练校准。

必须先回答和执行以下流程：

```text
1. 读取本文件和 highstep 必读记忆。
2. 检查当前 git diff，确认已有代码修改边界。
3. 不修改训练代码，先设计/实现或复用离线评分脚本。
4. 对正样本与负样本做评分校准。
5. 只有指标能区分正负样本，才允许修改 support 训练目标。
6. 修改前备份；修改后 py_compile 和 git diff --check。
7. 只允许 100-200 轮短训验证；失败条件触发立即停，不赌长训。
```

## 推荐的正负样本

正样本锚点：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_141000.pt
```

可参考的 student 正样本窗口：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-07-01_18-19-06
141700-142200 附近
```

负样本：

```text
teacher 2026-07-01_06-52-29 后期退化 checkpoint
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-03_05-40-48/model_141500.pt
```

注意：蒸馏不能验证 teacher 正确性。teacher 还没稳定前，不要把 student 蒸馏作为主线验证。

## 下一版修改边界

下一版只允许围绕 support 判定和支撑奖励做最小修改，避免再次引入大面积副作用。

推荐边界：

```text
只改 support 判定/支撑奖励/ActionScore 校准；
暂时不要同时改 yaw、lin_vel、terrain 分布、普通 locomotion 权重、student 蒸馏；
forward progress 不能再补偿 support 主分；
support 主分必须主要来自：
  第一后腿已明确上台；
  base 明显上升；
  姿态不过分坏；
  第二后腿没有长期拖边；
  必要时加入真实接触/足端高度/elapsed 时序约束。
```

## 新对话应让用户回答的问题

如果用户还没有回答，下一轮先问或确认以下问题。推荐答案都为“同意”：

```text
1. 是否同意下一步先做零训练评分校准，不立刻改 reward 开训？
2. 正样本是否固定为 2026-07-01_06-52-29/model_141000.pt，并参考 student 2026-07-01_18-19-06 的 141700-142200 窗口？
3. 负样本是否使用 teacher 后期退化 checkpoint 和当前 ActionScore 2026-07-03_05-40-48/model_141500.pt？
4. 下一版是否只改 support 判定/支撑奖励，不同时改 yaw、lin_vel、terrain、普通 locomotion 权重？
5. 是否允许把 forward progress 从 support 主分中移除，使 base 抬升和真实支撑成为硬核心？
6. 下一次训练是否只允许 100-200 轮短训，失败条件触发就停？
7. 如果校准发现当前 ActionScore 指标连 model_141000.pt 都判不高，是否接受先推翻指标实现，而不是继续调权重？
```

建议：这些问题最好在新对话回答。这样新对话的第一条用户确认就是干净决策源，不会被本长对话的上下文压缩污染。

## 新对话建议开场白

用户可以在新对话直接发送：

```text
工作仓库是 /home/lxq/Softwares/robot_lab。继续 highstep 训练修复。
先读取 docs/robotlab_memory_zh/INDEX.md、
docs/robotlab_memory_zh/library/highstep_new_dialogue_handoff_2026-07-03.md、
docs/robotlab_memory_zh/library/highstep_live_context_handoff.md、
docs/robotlab_memory_zh/library/highstep_phase_task_logic_priority_2026-06-30_2026-07-01.md、
docs/robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md、
docs/robotlab_memory_zh/library/accountability_protocol.md。
读取成功后回答开头写“已按照要求提前检索记忆和约束｜HLC-OK”。

我同意先做零训练评分校准：
正样本用 2026-07-01_06-52-29/model_141000.pt，
参考 student 2026-07-01_18-19-06 的 141700-142200 窗口，
负样本用 teacher 后期退化 checkpoint 和 2026-07-03_05-40-48/model_141500.pt。
下一步先不要训练，不要改无关参数，只检查当前 diff，并设计/运行评分校准。
```
