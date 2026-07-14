# Highstep 最高优先级记忆：阶段任务逻辑重构

更新时间：2026-07-01，Asia/Hong_Kong。

本文件是 highstep 项目当前最高优先级记忆。后续每次回答、判断、修改代码、给训练命令前，都必须先读本文件，再读 live handoff、短期记忆和责任协议。

## 至高原则

机器人的上高台能力永远是最高优先级。

任何修改都必须默认保护以下能力，不能让它们退化：

1. 第一条后腿能够顺利搭上高台。
2. 第一条后腿搭上高台后，能够支撑并把身体推上去。
3. 第二条后腿能够继续通过边缘，完成登台。
4. 已经肉眼证明有效的爬台能力不能被为了平地步态、branch balance、姿态美观或普通稳定性而牺牲。

如果某个修改可能削弱上高台能力，必须先明确说明风险、给出可观测指标和回滚条件；不能把“看起来更规整”的指标优先级放在爬台成功率之上。

## 后期收敛验收硬约束

新建或重构 highstep 任务的目的不是继续靠人工挑早期偶然好 checkpoint，而是让目标函数更接近用户真正要的上高台行为。因此后续 teacher 和 student 的成功标准必须包含“后期稳定接近最优”：

```text
训练过程中，真正的上高台动作评分应该随训练推进升高并进入平台期；
不接受早期 checkpoint 动作评分高、后期 terrain_levels/总 reward 更高但动作评分明显退化；
不接受最终只能靠挑早期 checkpoint 才有好动作的训练结果。
```

允许保存和比较多个 checkpoint，但这是审计目标函数是否对齐的安全流程，不是把早期偶然最优当作成功的理由。若出现以下情况，必须判定为目标函数仍未充分对齐或本轮修改未成功：

```text
早期 checkpoint 的 highstep action score / support score 明显高；
后期 checkpoint 的 terrain_levels 或普通总 reward 继续上升；
但后期 highstep action score、entry/support score 或 play 上台动作明显退化。
```

后续判断必须同时报告：

```text
best_score checkpoint
late_window_score
late_window_score / best_score
best_support checkpoint
late_window_support
late_window_support / best_support
score_drop_from_best
support_floor_violation_rate
```

默认验收线：

```text
late_window_score >= 0.90 * best_score
late_window_support >= 0.90 * best_support
score_drop_from_best 不应在后期持续扩大
support_floor_violation_rate 不应在后期持续升高
```

如果后期只能达到早期最优的 50%-70%，不能解释为“选 checkpoint 策略”，必须正面承认训练目标仍失败，并继续修正目标函数、curriculum 或训练流程。

## 必须参考的反例链路：2026-07-01_18-19-06

后续每次分析 highstep teacher/student、蒸馏、checkpoint、目标函数是否对齐时，必须主动参考 `2026-07-01_18-19-06` 这条 student 链路及其上游 teacher：

```text
student:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-07-01_18-19-06

teacher source:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_141000.pt
```

这条链路证明两件事：

1. 可取之处：蒸馏结构本身不是完全错误。teacher `model_141000` 是该 teacher 段评分峰值，`highstep_score ~= 90.6`、`support_score ~= 99.8`；student 训练到 `141700-142200` 附近也能达到 `highstep_score ~= 90`、`support_score ~= 97` 的平台，说明 student 能学到强上台动作先验。
2. 失败经验：teacher 和 student 都没有满足“越训越接近真实目标”。teacher 后期 `terrain_levels` 升到约 `4.38`，但动作评分退到约 `27.9`、`support_score` 退到约 `4.9`；student 后期 terrain 继续升，但 `highstep_score/support_score` 从高分平台回落。这是目标函数与真实上台动作不对齐的明确证据。

因此，后续不能把 `terrain_levels`、总 reward、蒸馏 MSE 下降或最后 checkpoint 作为主要成功证据。必须把“后期窗口 action/support score 稳定接近峰值”作为 teacher 和 student 的共同硬约束。

## 2026-07-02 确认后的阶段瓶颈目标

用户已确认后续 highstep 主目标不是任意地形泛化，而是：

```text
正向面对高台，允许小 yaw 微调；
真机目标 30cm，仿真主目标 35cm；
高台策略可作为专用策略，由部署端 FSM 切换；
但切入/切出姿态不能离谱。
```

成功动作定义：

```text
前腿上台
第一条后腿越过台面
第一条后腿支撑后 base 明显上升
第二后腿越过边缘
base 和至少三足稳定在台面上
不触发 bad_orientation
```

时间上不设置硬秒数，但必须尽量快，不能长期卡住。

2026-07-02 起，目标函数必须使用阶段瓶颈思想：

```text
entry 只能提供准备阶段学习信号；
entry 和 safety 不能补偿 support 失败；
support 不达标时 highstep_action_score 必须被 cap；
support 不达标时 terrain curriculum 不能继续晋级；
teacher 和 student 都必须满足 late_window_score/support >= 90% best。
```

新增必须观察的瓶颈诊断：

```text
Curriculum/highstep_action_score/raw_total
Curriculum/highstep_action_score/total
Curriculum/highstep_action_score/stage_cap
Curriculum/highstep_action_score/entry_gate
Curriculum/highstep_action_score/support_gate
Curriculum/highstep_action_score/second_gate
Curriculum/highstep_action_score/bottleneck_gap
Curriculum/highstep_action_score/support_floor_violation_rate
Curriculum/highstep_action_score/score_drop_from_best
```

## 2026-06-30 到 2026-07-01 的主线变化

这两天不再把 highstep 视为普通 dense reward 调参，而是改成“阶段任务”：

```text
approach
-> front_commit
-> rear_first_clear
-> lead_rear_support
-> second_rear_clear
-> post_clear_recovery
-> done / on_top
```

用户真正要的行为不是平均越过各种地形，而是：

```text
正向接近高台
前腿搭台
第一条后腿抬高并搭上
已搭台后腿向下/向后发力，把身体撑上去
第二后腿清过边缘
恢复正常状态
```

此前很多失败来自目标函数和真实目标不等价：reward 里有后腿 clearance、box push、body lift、post lead drive，但没有严格约束“必须按阶段完成”。结果 policy 可以学到局部取巧动作，例如前腿异常抬高、单侧挂边、动作很黏、上台后仍保持准备上台姿态。

## 关键链路与现象

### 2026-06-30_01-45-31 / model_133597

视频：

```text
Kazam_screencast_00112_2026-06-30_01-45-31_model_133597
```

用户观察：

- 爬坡/爬高台能力有肉眼可见显著提升。
- 说明重新设计任务逻辑、训练目标和阶段状态机方向比之前盲目拧权重更合理。
- 但仍有两个次要问题：
  1. 蹬上高台后动作可能不正常，给向前指令时有时会往下蹲，不像正常行走。
  2. 没上高台之前，如果 height scanner 检测到高度变化，机器人会提前有向前运动和抬脚趋势。

### 2026-06-30_18-52-50 / model_136596

基于 `model_133597.pt` 做过一次修正训练，主要加入：

```text
post_clear_recovery.weight = 0.30
scanner_pretrigger_penalty.weight = -0.08
```

训练指向：

- 尝试让机器人上台后恢复正常姿态。
- 尝试抑制 scanner 过早触发导致的台前预备动作。

视频：

```text
Kazam_screencast_00112_2026-06-30_18-52-50_model_136596
```

结果：

- 上高台动作很好，很流畅。
- 但是阶段判定明显有问题：机器人很多时候像是一直处于“准备上高台/正在上高台”的 phase。
- 远离高台或刚检测到高度变化时，左前腿会自己抬起。
- 已经上高台后，机器人仍可能姿态上扬、后腿下沉，像还在准备上台。

## 136596 的数据解释

关键爬台指标保持较强：

```text
rear_feet_highstep_clearance avg200 ~= 0.427
rear_first_foot_highstep_preclearance avg200 ~= 0.159
rear_second_foot_highstep_clearance avg200 ~= 0.199
lead_rear_support_drive avg200 ~= 0.103
second_clear_rate avg200 ~= 0.995
bad_orientation avg200 ~= 0.00129
terrain_levels avg200 ~= 2.274
```

这说明当前上高台主能力不能随意动坏。

但两个修正项没有完全解决阶段错误：

```text
scanner_pretrigger_front_lift_mean avg200 ~= 0.760
scanner_pretrigger_signal_mean avg200 ~= 0
scanner_pretrigger_gate_mean avg200 ~= 0
```

原因：原来的 `scanner_pretrigger_penalty` 被 `low_cmd_gate` 关掉，前进指令下不生效，所以无法解决“前进接近高台时过早抬前腿”的问题。

```text
post_clear_recovery_signal_mean avg200 ~= 0.151
post_clear_recovery_gate_mean avg200 ~= 0.351
post_clear_recovery_posture_mean avg200 ~= 0.045
```

原因：`post_clear_recovery` 可以通过 base clearance / forward progress 部分拿到分数，posture 很低时仍可能有 reward，不能强制“上台后真正恢复正常姿态”。

## 当前核心诊断

`_forward_highstep_terrain_gate()` 主要根据 height scanner 的 front/rear 高度差来判断 highstep terrain：

```text
front_z - rear_z
```

这个信号无法区分：

1. 高台真的在前方，应该进入 highstep approach。
2. 机器人已经上台，身后的 scanner 仍扫到台下地面，导致 rear_z 低、front_z 高，看起来仍像高台在前方。
3. 机器人还离高台比较远，但前方 rays 已经扫到高台，提前触发准备动作。

因此，当前更应该修的是 phase exit / done gate，而不是继续乱加前腿惩罚。

## 当前正确修改方向

优先实现阶段退出和误触发抑制：

1. 增加 `on_top / done` 判断：
   - 两条后腿已经高于高台面附近；
   - base 已经达到高台面上方合理高度；
   - 此时应认为当前 highstep 已完成。

2. 当 `done / on_top` 成立时：
   - 关闭或衰减前腿 reach、front clearance、horse rearing、box phase prior 等“准备上台/正在上台”奖励；
   - 不再让 policy 通过继续保持上台姿态拿分；
   - 不直接惩罚后腿爬台主能力。

3. 修改 `post_clear_recovery`：
   - 只能在真正 done/on_top 后给分；
   - posture 必须作为乘法门控或强约束，不能让 base clearance / forward progress 绕过姿态恢复。

4. 修改 `scanner_pretrigger_penalty`：
   - 不再完全依赖 `low_cmd_gate`；
   - 前进接近高台但还没 front commit 时，如果过早抬前腿，也应该能产生诊断和轻量约束；
   - 权重必须低，避免压掉真实前腿搭台。

## 禁止事项

- 禁止为了平地步态直接加全局前腿抬高惩罚。
- 禁止降低当前已经证明有效的后腿 clearance / support 主 reward。
- 禁止把 branch balance、`rl_minus_rr_first_rate`、普通姿态美观放在爬台成功率之前。
- 禁止在没有视频和 event 指标对应关系时宣称修改成功。
- 禁止从前腿已经异常但爬台强的 checkpoint 继续时，假装它是干净起点；必须明确这是“保护爬台能力优先”的风险选择。

## 下一步验证标准

修改后必须重点观察：

```text
Episode_Reward/post_clear_recovery
Curriculum/highstep_rear_branch_metrics/post_clear_recovery_signal_mean
Curriculum/highstep_rear_branch_metrics/post_clear_recovery_posture_mean
Curriculum/highstep_rear_branch_metrics/scanner_pretrigger_signal_mean
Curriculum/highstep_rear_branch_metrics/scanner_pretrigger_gate_mean
Curriculum/highstep_rear_branch_metrics/scanner_pretrigger_front_lift_mean
Episode_Reward/front_legs_reach
Episode_Reward/front_feet_highstep_clearance
Episode_Reward/rear_feet_highstep_clearance
Episode_Reward/rear_first_foot_highstep_preclearance
Episode_Reward/rear_second_foot_highstep_clearance
Episode_Reward/lead_rear_support_drive
Episode_Reward/post_lead_body_drive
Curriculum/highstep_rear_branch_metrics/second_clear_rate
Curriculum/terrain_levels
Episode_Termination/bad_orientation
```

成功不能只看曲线。最终必须 play 验证：

1. 上高台能力是否保持或增强。
2. 上台后是否减少“仍在准备上台”的姿态。
3. 台前远距离 scanner 触发时，是否减少无意义前腿抬起。

## 2026-07-01 实际代码修改

备份路径：

```text
docs/robotlab_memory_zh/library/__backups__/2026-07-01_highstep_phase_priority_protocol/
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-07-01_phase_exit_gate_from_2026-06-30_18-52-50_model_136596/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__backups__/2026-07-01_phase_exit_gate_from_2026-06-30_18-52-50_model_136596/highstep_env_cfg.py
```

修改文件：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
```

修改内容：

1. 新增 `_highstep_on_top_phase_gates()`：
   - 通过 `task_gate * second_ready_gate * base_clearance_score` 判断机器人是否已经完成上高台；
   - `second_ready_gate` 要求两条后腿都已经清过高台面附近；
   - `base_clearance_score` 要求 base 已经在高台面上方；
   - 输出 `entry_allowed_gate = 1 - on_top_gate`，用于关闭完成后的准备/上台中 reward。

2. 给以下准备/上台中 reward 增加 `entry_allowed_gate`：
   - `front_feet_highstep_clearance_bonus`
   - `highstep_forward_progress_bonus`
   - `highstep_body_lift_bonus`
   - `highstep_base_advance_lift_bonus`
   - `highstep_leg_support_contact_bonus`
   - `horse_rearing_posture_bonus`
   - `highstep_box_phase_prior_alignment_bonus`

   意义：机器人已经 on_top/done 后，不再通过这些奖励继续保持“准备上台/正在上台”的姿态。

3. 修改 `post_clear_recovery_bonus()`：
   - `done_gate = ctx["task_gate"] * second_ready * base_clearance_score`；
   - `recovery_score` 改为 posture 参与乘法，姿态差时不能再靠 base clearance / forward score 绕过恢复目标。

4. 修改 `scanner_pretrigger_penalty()`：
   - 前腿提前抬高部分不再被 `low_cmd_gate` 完全关掉；
   - 低指令下的无故前冲仍然由 `low_cmd_gate` 控制；
   - 整体只在 `terrain_active * precommit_gate * entry_allowed_gate` 下生效，front commit 后不碰真实上台。

5. 配置只给 `front_feet_highstep_clearance` 增加：

```python
"rear_foot_names": ["RL_foot", "RR_foot"]
```

没有修改后腿主 reward 权重，没有修改 terrain 分布。

验证：

```text
python3 -m py_compile rewards.py highstep_env_cfg.py: 通过
git diff --check: 通过
```

本轮风险：

- 如果 `on_top_gate` 判定过早，可能提前关闭少量 front/progress/body lift 辅助奖励。
- 为降低这个风险，`on_top_gate` 同时要求两条后腿清台和 base 已经到台面上方，不使用单一 scanner height delta。
- 如果短训后 `rear_feet_highstep_clearance`、`rear_first_foot_highstep_preclearance`、`rear_second_foot_highstep_clearance`、`lead_rear_support_drive` 明显下降，应判为失败并回滚本轮 phase-exit 修改。

## 2026-07-01 目标切换：Highstep-only 策略 + 部署端 FSM 切换

用户 play `2026-07-01_01-46-33/model_137500.pt` 后确认：

- 上高台能力有保持。
- 但台前预触发、上台后仍像准备上台等问题仍然存在。

用户决定切换思路：

```text
不再要求同一个策略同时负责平地行走、上高台、上台后恢复全部行为。
部署端后续用有限状态机手动切换：
平地策略 <-> 上高台专用策略。
```

新的目标边界：

1. 上高台专用策略的第一优先级是把上台能力训练到尽可能强。
2. 平地正常行走、上台后正常行走、长时间保持普通状态，交给其它策略文件和部署端 FSM。
3. 但上高台策略的切入/切出姿态不能太离谱，否则从平地策略切到上高台策略、再切回其它策略时会出现过大姿态跳变。

因此，2026-07-01 后续代码方向调整为：

- 不再用 `on_top/done` 的 `entry_allowed_gate` 关闭主爬台 reward。
- `front_feet_highstep_clearance_bonus`、`highstep_forward_progress_bonus`、`highstep_body_lift_bonus`、`highstep_base_advance_lift_bonus`、`highstep_leg_support_contact_bonus`、`horse_rearing_posture_bonus`、`highstep_box_phase_prior_alignment_bonus` 必须继续完整服务爬台能力。
- `post_clear_recovery` 只保留为轻量姿态兼容项，不作为主要训练目标。
- `scanner_pretrigger_penalty` 只保留为轻量台前兼容项，避免切入前姿态过于离谱，但不能压制真实上台准备动作。

实际代码修改：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py
```

备份：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-07-01_highstep_only_pose_compat_from_2026-07-01_01-46-33_model_137500/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__backups__/2026-07-01_highstep_only_pose_compat_from_2026-07-01_01-46-33_model_137500/highstep_env_cfg.py
docs/robotlab_memory_zh/library/__backups__/2026-07-01_highstep_only_pose_compat/
```

关键改动：

```text
post_clear_recovery.weight: 0.30 -> 0.12
scanner_pretrigger_penalty.weight: -0.08 -> -0.02
```

并撤掉主爬台 reward 上的 `phase["entry_allowed_gate"]` 乘法门控，只保留在 `scanner_pretrigger_penalty` 里作为“已经上台后不再惩罚台前预触发”的保护。

验证：

```text
python3 -m py_compile rewards.py highstep_env_cfg.py: 通过
git diff --check rewards.py highstep_env_cfg.py: 通过
```

后续训练优先从 `2026-07-01_01-46-33/model_137500.pt` 继续，因为用户刚 play 确认其上台能力有保持；本轮代码的目的正是移除该分支中可能限制继续强化爬台的 phase-exit 主 reward 限制，同时保留很轻的切换姿态兼容约束。
