# Highstep 目标函数对齐与评估闭环方案

更新时间：2026-06-30，Asia/Hong_Kong。

本文件把最近两次结论固定成可审计方案：`terrain_levels`、平均 reward、teacher/student MSE 都不能直接等价于用户真实目标。后续 highstep teacher/student 的每次修改，都必须能映射到本文件中的阶段、指标和通过/失败条件。

## 真实目标

目标不是“平均地形上表现不错”，而是在固定高台任务中稳定完成以下动作链：

```text
approach -> front_commit -> rear_first_clear -> lead_rear_support -> second_rear_clear -> recover
```

动作链解释：

- `approach`：身体大致正向或小 yaw 角接近高台。
- `front_commit`：前腿搭上高台，身体进入真实上台阶段。
- `rear_first_clear`：第一条后腿足端足够高并搭上高台。
- `lead_rear_support`：已经搭上高台的后腿向下/向后支撑，把身体撑起来。
- `second_rear_clear`：第二条后腿在有限时间内清过台阶边缘。
- `recover`：上台后恢复稳定姿态；平地行走不能出现左前腿异常高抬。

## 为什么旧目标不等价

旧训练目标主要是：

```text
teacher PPO: 训练分布下的平均累计 reward
student distill: 平均 Teacher_Action_MSE / latent MSE / box prior loss
```

真实目标是窄阶段动作链。平均指标会把关键高台阶段稀释掉：

- 普通行走样本数量远多于高台后半段样本。
- 平均 MSE 降低可能把高台瞬时强动作磨平。
- `terrain_levels` 是训练环境 terrain row 平均值，不是固定 30/35 cm 高台成功率。
- 单个 reward 上升可能来自错误替代动作，例如前腿异常高抬、单后腿挂边、box joint 外溢。

## 分阶段实施

### Phase 1：固定评估闭环，不改训练行为

目的：停止用“最后 checkpoint / 曲线好看 / 肉眼印象”直接做结论。

已实现文件：

```text
tools/highstep_checkpoint_eval_suite.py
docs/robotlab_memory_zh/library/highstep_objective_alignment_plan.md
```

用户应该做：

1. 每次 teacher 或 student 训练后，先用 `tools/highstep_checkpoint_eval_suite.py` 对候选 checkpoint 打分。
2. 对分数最高和关键指标最高的 checkpoint 再 play。
3. Play 固定场景，不随意改变高度、terrain type、出生点、命令方式。

Phase 1 指标：

| 阶段 | 指标 | 作用 |
| --- | --- | --- |
| approach/front_commit | `front_legs_reach`, `front_feet_highstep_clearance` | 确认前腿进入高台阶段，但不能单独判成功 |
| rear_first_clear | `rear_feet_highstep_clearance`, `rear_first_foot_highstep_preclearance` | 第一条后腿搭台能力 |
| lead_rear_support | `lead_rear_support_drive`, `post_lead_body_drive`, `rear_legs_drive_bonus` | 已搭台后腿把身体撑起来 |
| second_rear_clear | `rear_second_foot_highstep_clearance`, `second_clear_rate`, `one_sided_stall_ratio` | 第二后腿清台和单侧卡边 |
| box action | `highstep_rear_box_push`, `highstep_box_phase_prior` | 伸缩腿是否参与上台 |
| stability/gait | `bad_orientation`, `time_out`, `front_lift_guard_*`, `fl_forward_flat_*` | 稳定性和平地前腿异常 |
| student only | `Teacher_Action_MSE`, `Distill_Latent_MSE`, `Mu_Out_Of_Bounds_Ratio` | 蒸馏误差和动作越界，但不能单独判定 highstep 成功 |

`tools/highstep_checkpoint_eval_suite.py` 必须同时看两类分数：

```text
mean_score: checkpoint 前一段窗口的平均分，反映整体训练分布表现。
highstep_last: checkpoint 末端高台动作强度分，反映该模型附近的具体爬台动作是否被磨平。
```

解释：

- 如果 `mean_score` 高但 `highstep_last` 低，说明平均训练指标更好，但关键高台动作可能退化。
- 如果 student 的 `student_distill` 分高但 `highstep_last` 低，说明蒸馏更贴近平均 teacher，但可能磨平高台强动作。
- 这种情况必须优先 play 验证，不能直接选 `mean_score` 最高的 checkpoint。

Phase 1 通过条件：

- 评估工具能找到候选 checkpoint 的关键指标。
- 输出中没有关键阶段全部 `MISSING`。
- 选出来的候选 checkpoint 必须再通过固定 play 验证。

Phase 1 失败条件：

- 工具提示某阶段指标缺失，不能把该阶段说成已验证。
- 只凭 final score 或某个单项最高直接决定部署。

### Phase 2：固定 play/eval 场景

目的：把肉眼 play 变成可重复试验。

固定场景：

| 场景 | terrain type | level | 目标 |
| --- | --- | --- | --- |
| flat_forward | flat/默认平地 | none | 检查平地左前腿异常和普通步态 |
| box_30cm_forward | box | 对应 30 cm 附近 | 检查真机目标高度 |
| box_35cm_forward | box | level10/最高高台 | 检查仿真极限余量 |
| box_35cm_yaw_left | box | level10/最高高台 | 检查小 yaw 角上台 |
| box_35cm_yaw_right | box | level10/最高高台 | 检查小 yaw 角上台 |

每个 checkpoint 至少记录：

```text
success_count / total_attempts
first_rear_clear_count / total_attempts
lead_rear_support_success_count / first_rear_clear_count
second_rear_clear_count / first_rear_clear_count
flat_left_front_abnormal_count / flat_trials
bad_orientation_count / total_attempts
```

Phase 2 通过条件：

- 30 cm 和 35 cm 场景都能给出固定尝试次数下的成功率。
- 每个失败能归因到第一后腿、支撑、第二后腿、平地前腿或姿态终止之一。

### Phase 3：把真实目标写回 teacher reward

目的：不是再简单加权，而是把上台动作写成阶段任务。

后续允许修改的方向：

1. `approach/front_commit` 阶段只允许前腿进入高台，不把前腿异常高抬当成功。
2. `rear_first_clear` 阶段强化第一条后腿足端抬高和前向越边。
3. `lead_rear_support` 阶段奖励已上台后腿支撑身体上升。
4. `second_rear_clear` 阶段对第二后腿限时清台，惩罚长时间单侧卡边。
5. `recover/flat` 阶段约束平地异常前腿动作和非高台 box 外溢。

Phase 3 指标：

```text
EvalHighstep/rear_first_clear_rate
EvalHighstep/lead_rear_support_rate
EvalHighstep/second_rear_clear_rate
EvalHighstep/one_sided_stall_rate
EvalHighstep/flat_left_front_lift_abnormal_rate
```

Phase 3 禁止：

- 只因为 `terrain_levels` 上升就判定成功。
- 只因为 `rear_feet_highstep_clearance` 上升就忽略平地前腿异常。
- 把前腿异常姿态、单后腿挂边当作完成上高台。

### Phase 4：把真实目标写回 student 蒸馏

目的：防止 student 后期把高台强动作磨平。

后续允许修改的方向：

1. 高台阶段样本加权：front commit、rear first clear、lead rear support、second rear clear 的样本权重大于普通行走样本。
2. 动作维度加权：后腿关节和 box joint 在高台阶段的 teacher action loss 权重大于普通动作。
3. Student early stopping：当 `Teacher_Action_MSE` 继续下降但 highstep eval 分数下降时停止。
4. 输出 student/teacher 同场景退化比，而不是只看 student 自己。

Phase 4 指标：

```text
Loss/Teacher_Action_MSE
Loss/Highstep_Phase_Teacher_Action_MSE
Loss/Highstep_Rear_Box_Action_MSE
Debug/Mu_Out_Of_Bounds_Ratio
EvalHighstep/student_teacher_success_gap
EvalHighstep/student_teacher_lead_support_gap
```

Phase 4 失败条件：

- `Mu_Out_Of_Bounds_Ratio` 降低但 highstep play 退化，不能判成功。
- `Teacher_Action_MSE` 降低但 `rear_first_clear_rate` 或 `lead_rear_support_rate` 下降，判为蒸馏磨平。

### Phase 5：部署前门槛

目的：避免把“训练曲线能看”误认为“真机可用”。

部署前必须通过：

```text
teacher fixed eval -> teacher play video -> student fixed eval -> student play video -> Mujoco sim-to-sim -> 真机低风险测试
```

真机前最低要求：

- 30 cm 高台成功率优先于 35 cm 极限高度。
- 平地左前腿异常不能明显影响起步和接近高台。
- 后腿第一条腿搭台和搭台后支撑必须在 student 中保住，而不是只在 teacher 中出现。

## 覆盖检查表

后续每次修改前必须填：

| 问题 | 必须回答 |
| --- | --- |
| 这次改哪个 phase？ | Phase 1/2/3/4/5 |
| 这次改哪个行为？ | 第一后腿、lead support、第二后腿、平地前腿、student 磨平等 |
| 对应哪些指标？ | 必须列完整指标名 |
| 哪些指标不能退化？ | 必须列硬保护指标 |
| 如何回滚？ | 指定 checkpoint 或备份文件 |
| 训练多少轮先验收？ | 明确 `--max_iterations` 是额外训练轮数 |

如果不能填完，不允许继续改 reward 或 distill loss。
