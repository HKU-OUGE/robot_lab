# Highstep Compact Handoff 2026-07-02

用途：这是给新对话启动用的精简交接文件。旧 highstep 对话已经过重，容易 reconnect/thinking 卡住；新对话应优先读本文件，再按需读取完整记忆。

## 启动要求

新 highstep 对话开头先读：

```text
docs/robotlab_memory_zh/library/highstep_compact_handoff_2026-07-02.md
docs/robotlab_memory_zh/library/highstep_phase_task_logic_priority_2026-06-30_2026-07-01.md
docs/robotlab_memory_zh/library/accountability_protocol.md
```

只有需要历史细节时，再读：

```text
docs/robotlab_memory_zh/library/highstep_live_context_handoff.md
docs/robotlab_memory_zh/library/highstep_short_term_2026-06-25_20h.md
```

## 当前最高原则

上高台能力永远是最高优先级，尤其是：

```text
第一后腿搭台
第一后腿搭台后支撑并把身体撑上去
第二后腿清过边缘
base 和至少三足稳定在台面上
不触发 bad_orientation
```

不能再接受“早期 checkpoint 肉眼好、后期 terrain_levels/总 reward 高但真实上台动作退化”的结果。后续训练必须让 late window 的 highstep action/support score 稳定接近历史最优：

```text
late_window_score >= 0.90 * best_score
late_window_support >= 0.90 * best_support
```

保存多个 checkpoint 是审计流程，不是把早期偶然最优当作成功。

## 目标场景

当前目标已经收敛为工程可执行版本：

```text
机器人正向面对高台，允许小 yaw 微调。
真机目标 30cm，仿真主目标 35cm。
40cm 只作为挑战评估，不作为当前主训练目标。
highstep 可以是专用策略，由部署端 FSM 切换平地/上台/恢复策略。
切入和切出姿态不能离谱，但不能压过爬台主能力。
```

## 关键历史经验

### 2026-07-01 teacher/student 反例链路

必须参考这条链路：

```text
teacher:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29
peak checkpoint:
model_141000.pt

student:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-07-01_18-19-06
```

经验：

```text
teacher model_141000 是该段高台动作评分峰值，highstep_score 约 90.6，support_score 约 99.8。
student 在 141700-142200 附近也能到 highstep_score 约 90，support_score 约 97。
说明 teacher/student 结构不是完全错误，动作先验可以学到。
```

失败：

```text
teacher 后期 terrain_levels 升到约 4.38，但 highstep_score 退到约 27.9，support_score 退到约 4.9。
student 后期 terrain 继续升，但 highstep_score/support_score 从高分平台明显回落。
因此 terrain_levels、总 reward、蒸馏 MSE 下降都不能单独证明上台动作变好。
```

## 2026-07-02 ActionScore 新任务

旧 highstep 不再继续无限补丁，已经新增 ActionScore 任务：

```text
RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0
RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0
```

核心思想：

```text
entry 只能提供准备阶段学习信号。
support 不达标时，总分必须被 cap。
support 不达标时，terrain curriculum 不能晋级。
entry/safety 不能补偿 support 失败。
```

关键指标：

```text
Curriculum/highstep_action_score/total
Curriculum/highstep_action_score/raw_total
Curriculum/highstep_action_score/entry_score
Curriculum/highstep_action_score/support_score
Curriculum/highstep_action_score/stage_cap
Curriculum/highstep_action_score/bottleneck_gap
Curriculum/highstep_action_score/support_floor_violation_rate
Curriculum/highstep_action_score/score_drop_from_best
Episode_Reward/highstep_action_score
Episode_Reward/lead_rear_support_drive
Episode_Reward/post_lead_body_drive
Curriculum/terrain_levels
Episode_Termination/bad_orientation
```

## 2026-07-02 晚最新状态

最近旧线程分析到：

```text
训练 run:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-02_21-18-28

最后保存:
model_142100.pt

event 到:
142188 左右

状态:
训练已停止；当时未发现残留 train.py 进程。
```

失败复盘：

```text
代码不是完全没生效。
support_bottleneck_gate / bottleneck_gap 生效。
lead_rear_support_drive 从上一组 tail100 约 0.098 提到约 0.125。
post_lead_body_drive 从约 0.0036 提到约 0.0044。

但核心仍没过线：
support_score tail100 约 0.421，低于 support_floor 0.45。
support_floor_violation_rate tail100 约 0.802，仍很高。
terrain_levels tail100 约 0.190，curriculum 仍被压住。
```

旧线程给出的结论：

```text
从 2026-07-01_06-52-29/model_141000.pt 继续 refine 有一点改善，但不够。
旧 highstep 行为惯性太强。
继续加权重容易混淆代码问题和 checkpoint 污染问题。
下一步更应该用现有瓶颈代码，从更干净的 bodyflat 基础策略做 migration 验证。
```

建议起点：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
```

如果从 bodyflat checkpoint 迁移，必须走 migration 模式，避免 highstep refine 逻辑误开 gate：

```text
--highstep_resume_mode migration
```

## 新对话下一步问题

用户最新要解决的是：

```text
检查最新训练的 wandb/event log，感觉还是不太行，分析可能原因。
```

新对话不要只读网页 wandb。优先读取本地 event，确认宿主训练进程，再按以下维度判断：

```text
1. 最新 run 是否真的是 2026-07-02_21-18-28 或后续新 run。
2. 是否还在训练，不能只看沙箱内 ps。
3. support_score 是否持续低于 0.45。
4. support_floor_violation_rate 是否仍高。
5. post_lead_drive_gate_mean / post_lead_body_drive 是否进入有效量级。
6. terrain_levels 被压住是 curriculum 正常阻止失败晋级，还是目标函数过严导致学不到。
7. 如果从 model_141000 refine 仍失败，优先考虑 bodyflat model_49600 migration，而不是继续在旧 highstep checkpoint 上补丁。
```

## 对话卡顿处理

旧 highstep 对话已超过 3.9 亿累计 tokens，最近单轮约 21 万 tokens。不要再让新对话读取整条旧 rollout。若需要旧经验，优先读本文件和上述记忆文件。
