# 失败模式与经验教训

## 有依据地进入 Student 不等于可以跳过真机可行性门槛（2026-07-12）

用户催进度时，正确的加速方式是删除没有证据的重复训练，不是降低证据门槛。本轮冻结 Teacher 在 Standard 与实测 gains 中心 core9 都为 9/9，所以直接开始一次 Robust Student 蒸馏可以 defend；继续为了“更 Robust”盲改 Teacher reward 反而可能破坏已满足两后足上台的动作。但最初只准备评估最后一个 Student checkpoint，并且没有验证仿真执行器容量是否超过真机，属于真实遗漏。

固定规则：

```text
Teacher 已通过用户物理终点时，Student 是部署必经阶段，不是投机跳步。
Student 的 loss/训练 reward 不能证明继承成功；至少比较早/中/末 checkpoint，防止蒸馏后期把动作平滑掉。
Isaac asset 的配置上限不等于 policy 的实际力矩需求，不能只比较 80 与 44.4 就升级成阻断门槛。
先查真实 sim-to-sim 链：本机 MuJoCo hip/thigh/calf 实际上限是 23.7/23.7/45.43 N·m，gear=1，仍能轻松上35 cm；这与人提机身卸载后仍动作失败共同排除了单纯力矩不足主假设。
effort/velocity telemetry 可以保留为可选诊断，但不得脱离行为证据变成自动重训理由；主线继续看后腿轨迹、相位、中线塌缩和边缘卡住。
```

本轮还发现 Stage2 Student 的独立 `vae_optimizer` 和 `student_distill_update_count` 没有进入旧 checkpoint。为了看起来“谨慎”而中断当前连续进程，会实际制造无法无缝恢复的断点。规则是：

```text
不要把有 checkpoint 文件误写成可无缝续训。
新 checkpoint 必须版本化保存算法自有 optimizer/count；full-resume 缺状态默认 fail closed。
旧 Student 只能显式迁移精确 count，并明确 Adam moments 已丢失；否则 weights-only/reset 新分支。
当前连续运行健康时，不为形式审查中断一个无法完整恢复的旧进程。
```

## 成功定义错位和聚合器失败不能冒充 policy 失败（2026-07-12）

用户真正需要的是整机上台：RL/RR 双后足在顶面承载、机身进入并稳定到可以手动切 `fixed stand`。FL 随后是否放下、双前足持续承载、严格接触时序和后足继续深入 `0.18 m` 可以是改进目标，但不是这轮最低成功条件。旧门槛把这些诊断项升成硬条件，曾把实际能完成用户目标的 `model_172300.pt` 错报为失败，导致方案变复杂并延误推进。

固定规则：

```text
先把用户的物理终点写成直接可观测门槛，再训练或淘汰 checkpoint。
代理 reward、kinematic hold、前足姿态和严格事件链不得替代物理终点。
低优先级缺陷不得阻塞已经满足最高目标的候选。
```

同轮 real-gain 评估又出现一次基础设施假失败：9 场 raw 均 rc0、schema7、rear-platform pass，但 aggregate 因 legacy schedule 分支只接受 `role=teacher`、漏掉合法 `teacher_robust` 而给出 `valid_count=0`。后续所有聚合失败统一遵守：

```text
先检查 raw 行为、checkpoint SHA、task/role、schedule/runtime contract；
定位 valid_eval 的第一个实际拒绝条件；
只有 raw 行为也失败时才修改训练；
基础设施修复必须窄绑定任务/角色/固定 checkpoint/run-dir/SHA，并补正反例；
已有完整 raw 可安全重聚合时，不重复 GPU 评估。
```

本轮还确认：实测 gains 门禁应对同一冻结 checkpoint 单变量复评，固定 delay=0 并关闭额外随机事件；通过后直接进入 Robust Student，不为“流程完整”强制再训 Robust Teacher。`action_scale`、`joint_pos.clip`、`default_dof_pos` 和 observation/action 顺序是永久兼容合同，任何微调都不得触碰。

## ActionScore 失败：诚实指标不是有效训练机制

2026-07-03 失败 run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-03_05-40-48
source checkpoint:
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_141000.pt
last saved:
model_141500.pt
```

观察：

```text
entry_score tail20 ~= 0.975
support_score tail20 ~= 0.344 < support_floor 0.45
support_floor_violation_rate tail20 ~= 0.992
terrain_levels tail20 ~= 0.185
post_lead_body_drive tail20 ~= 0.00627
```

经验：

```text
support_floor_violation_rate 能诚实暴露失败，但它本身不能训练出支撑动作。
如果 support reward 仍允许 forward progress、second score 或 floor 项补偿，策略会学到往前蹭/局部抬升，而不是第一后腿搭台后支撑身体。
terrain 降低后 total 变好不是高台能力变好。
```

硬规则：

```text
新 ActionScore 或支撑指标上线前，必须先用正负样本做零训练评分校准。
正样本至少包括 2026-07-01_06-52-29/model_141000.pt。
负样本至少包括后期退化 checkpoint 和 2026-07-03_05-40-48/model_141500.pt。
若指标不能把正样本判高、负样本判低，不准开训。
```

修改边界：

```text
下一版先只修 support 判定/支撑奖励；
不要同时改 yaw、lin_vel、terrain 分布、普通 locomotion 权重和 student 蒸馏；
forward progress 不能补偿 support 主分；
teacher 稳定前不要用蒸馏验证 teacher 是否正确。
```

## Terrain 曲线会骗人

不要只优化 `terrain_levels`。

已观察到：

- 有些 run terrain 曲线更好，但 play 姿态很差；
- 机器人可能统计上在推进，但平地步态已经趴下、扭曲或乱抬腿。

规则：

```text
重要训练判断必须同时看曲线和 play/video。
```

## 不要频繁乱改 Terrain Curriculum

频繁改 terrain level 判断逻辑会让对比失效。只有 reset 分布或任务定义明显错误时，才改 terrain/curriculum。

Highstep 中有用的地形调整包括：

- 增加 inverse/低处接近高处的情况；
- 让机器人真正从低处爬上高台；
- 避免大多数情况只是从高处往低处走。

但不要为了让曲线好看反复重写 terrain level 评分。

## 训练中录视频容易卡死

`--video` 或频繁 Isaac/Replicator 录制多次造成系统卡死。优先：

- 长训不录视频；
- 单独 `play.py` 录视频；
- 只做短诊断视频。

## Highstep Reward 方向

有效 highstep 姿态应包括：

- 前腿伸到平台上；
- 后腿积极推进；
- 后侧 box joint 在推进阶段伸长；
- 机体越过障碍。

只奖励前腿搭上去不够，会形成“前腿在上面，身体卡住，后腿带不上来”的失败动作。

## Hip 外展不是 Highstep 目标

Sidestep 中 hip 外展用于防侧翻。Highstep 需要前向上台自由度。hip 外展和足端宽度约束会损害 highstep 能力。

规则：

```text
除非明确做侧向台阶混合任务，否则不要把 sidestep 的横向稳定目标搬到 highstep。
```

## Student 不是修补不稳定 Teacher 的工具

不要用 student 去修一个还没稳定的 teacher。teacher 不稳定，student 通常更弱。

No-prior student 注意：

- 先保 locomotion；
- 不要为了学 action prior 把原始步态毁掉；
- 不能随便打开 PPO 或改变 actor 可训练范围，必须明确理由。

## Student 的 OOD command 会比 Teacher 先崩

2026-06-18 highstep student 已观察到：

- teacher 虽然没显式训练负 `lin_vel_x`，但键盘后退时通常还能勉强稳住；
- student `model_69998.pt` 在平地后退时更容易后仰并触发 `bad_orientation`；
- 全局 WandB 平均值不容易暴露这个问题，因为键盘后退是很窄的训练外场景。

经验：

```text
如果 play/部署会使用训练 command 范围外的指令，student 蒸馏必须覆盖这些指令；不能指望 student 自动继承 teacher 的 OOD 泛化。
```

当前具体案例：

- 原 highstep command 范围：`lin_vel_x=(0.20, 0.85)`；
- 最先尝试 student-only 增强：`lin_vel_x=(-0.30, 0.85)`；
- 后来 teacher 侧也改为 `lin_vel_x=(-0.30, 0.85)`；
- 配合最终 command multiplier `0.75`，大约覆盖到 `-0.225 m/s` 的轻微后退。

不要指望继续同一分布的 student 训练能自动修复后退稳定性。先补 command 分布；如果 student 仍无法继承稳定 fallback，再回头改 teacher。

2026-06-18/19 后续经验：

```text
student-first 修复不够。如果 teacher 的好表现只是脆弱的 OOD fallback，盲走 student 不一定能继承。部署会用到的重要 command 必须先进 teacher 训练分布，再重新蒸馏。
```

当前 highstep command 已经在 teacher 侧覆盖：

```text
lin_vel_x=(-0.30, 0.85)
```

## Box Joint 默认位是独立目标

后退稳定修复让机器人后退更安全，但也导致平地/正常运动时 box joint 位置异常。这不可接受：box joint 应该只在高台阶段明显参与，平地静止和普通行走时不应乱伸缩。

当前修复函数：

```text
highstep_box_default_position_penalty()
```

逻辑：

- 计算 box joint 相对默认位置的偏差；
- 用前/后足几何位置和地形高度构造 highstep gate；
- 再乘以前进 command gate；
- 非主动上高台时使用强 `hold_scale`，主动前进上高台时使用低 `highstep_scale`。

经验：

```text
不能靠让 box joint 到处乱漂来解决爬高。box joint 必须有任务门控：平地/非前进保持默认，前进高台释放。
```

## Resume 门控规则

fresh highstep 训练可以保留 command terrain gate，避免太早放开完整 command 范围。但从成熟 checkpoint resume/refine 时，继续卡 gate 会妨碍精修。

当前 `scripts/rsl_rl/base/train.py` 行为：

```text
if highstep teacher and --resume:
    command_levels.params["terrain_gate_level"] = 0.0
```

不要忘记这个区别：

- fresh run：保留门控；
- mature resume/refine：放宽门控。

## Terrain Levels 不是高台高度

用户之前痛苦点：play 里机器人已经能爬 0.35 m 高台，但 `terrain_levels` 仍在 2.7-2.8 左右。正确解释是：

```text
terrain_levels = 所有训练 env 经过 curriculum 升降级后的 terrain row 平均值
```

它不是：

```text
机器人能爬的最大高台高度
```

原因：

- highstep 训练混合了 box、坑、正反楼梯、坡、粗糙地形、静止、后退、侧向等场景；
- box 高台只占训练分布的一部分；
- play 可以主动选择高台，训练曲线是所有场景平均；
- bodyflat 5.x 曲线来自不同任务、地形混合、command 范围和 curriculum 函数。

规则：

```text
terrain_levels 只能作为训练分布/进度信号，不能单独证明上高台能力。判断 highstep 能力要看指定高台高度的 play/eval 成功率、稳定性和 box joint 行为。
```

## 平均化后腿奖励会掩盖分支崩坏

2026-06-23 highstep teacher 精修中观察到：

- `model_97300` 仍能上高台，可作为对照 checkpoint，但重新核对继承关系后，不应作为新奖励修复的主父节点。
- `model_97300` 是从 `2026-06-22_12-41-28/model_95398.pt` 在旧奖励组合下继续约 1900 update 得到的。
- 如果新修复的原因正是“旧奖励组合可能把策略引到坏分支”，除非视频证据压倒性证明更晚 checkpoint 更好，否则优先选择这段旧训练之前的干净 checkpoint。
- `model_105397` 并非完全不会上；补充视频确认它仍有右后腿先登台的成功分支，但主导行为太容易变成更弱的左后腿先登台分支。
- 左后腿先登台时，另一条后腿容易卡在台阶边缘下方，机器人会持续蹬腿但无法完成上台。
- 成功上台常常依赖已经登上高台的那条后腿作为支撑/驱动腿，把身体顶起来，再把第二条后腿带上去。

曲线陷阱：

```text
rear clearance / rear drive / box push 奖励上升，不代表真实上台动作更可靠。
```

原因：

```text
如果后腿 clearance 部分奖励来自 max/单脚成功，而后腿卡台惩罚又对两条后腿取平均，那么“一条后腿上去了、另一条后腿还卡住”的失败会被奖励函数稀释。
```

规则：

```text
Highstep 必须分开看 RL-first 和 RR-first。
卡台/清台项要使用最差后腿或第二后腿逻辑。
选择 checkpoint 前要确认它是否已经包含当前正在修复的旧奖励逻辑训练段。
不要因为后期 checkpoint 标量奖励略高，就从视频已经显示分支崩坏的 checkpoint 继续。
```

## 备份纪律

这个仓库里有很多实验性脏改。不要 revert 用户改动。不要无备份覆盖。

改代码前：

```text
1. 找到相关最新 log 后缀。
2. 复制原文件为 *_<suffix>.py。
3. 只做必要修改。
4. 记录建议从哪个 run/checkpoint 接着训。
```

## ActionScore bodyflat 迁移失败：不要只补后段，先打通第一后腿链路

2026-07-03 检查并停止的失败 run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-03_00-48-49
source checkpoint:
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
stop checkpoint:
model_50100.pt
event final:
50127
```

失败经验：

```text
1. bodyflat -> highstep ActionScore 迁移时，不能只把 post-lead reward 提早打开。
   如果 commit_gate 长期很低、rear_first_preclearance 近 0，后段 reward 的输入条件本身就是空的。

2. 这次 post-lead 在 50100 后刚开始非零，但 stage_ramp_updates=350，
   到 50120 左右只打开了很小一段。以后不能把“刚过 stage_start 后 signal 很小”
   直接等同于“post-lead 阶段完全失败”；必须同时看 stage/ramp 进度。

3. 更硬的失败证据是：
   commit_gate_mean tail 很低（约 0.016），rear_first_foot_highstep_preclearance 几乎为 0，
   support_score 仍在 1e-4 量级，support_floor_violation_rate=1，
   bad_orientation 已经升到约 0.015-0.017。

4. terrain warmup 不能太长。support 为 0 时 terrain_levels 已经升到约 1.7，
   这会在主链未形成前增加难度，并复现“terrain 先涨、真实动作没接上”的早期版本。

5. `2026-07-01_18-19-06` 链路仍然是重要正例：流畅高台动作能训出来。
   这次失败不能解释为任务不可学，而应解释为 ActionScore 迁移期 gate/stage/诊断设计还没对齐。
```

修正原则：

```text
1. 保留后期 support bottleneck，不回到 terrain/reward 虚高。
2. 迁移早期给第一后腿预清台一个 terrain-triggered soft commit 通道，
   避免 commit_gate 太低时 rear_first reward 完全归零。
3. action prior 更早 ramp，帮助 bodyflat checkpoint 进入高台 box-joint 相位。
4. terrain 早期最高 level 更保守，score warmup 更早结束。
5. post-lead watchdog 必须 ramp-aware。
```

## 2026-07-03 01:31 后 ActionScore 修改失败：禁止自动副作用补丁

失败 run：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-03_02-53-22
source checkpoint:
logs/rsl_rl/arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
```

本地 event / wandb 关键证据：

```text
Curriculum/highstep_action_score/support_score tail100 ~= 0.106
Curriculum/highstep_action_score/support_floor_violation_rate tail100 = 1.0
Curriculum/highstep_action_score/post_lead_drive_gate_mean tail100 ~= 0.00039
Curriculum/highstep_action_score/post_lead_drive_height_mean tail100 ~= 0.0106
Curriculum/highstep_action_score/score_drop_from_best tail100 ~= 0.756
Episode_Reward/highstep_action_score tail100 ~= 0.00030
Train/mean_reward tail100 ~= 54.0
```

结论：

```text
这次修改不能算成功。
普通 locomotion/步态 reward 仍能给出较高总 reward，但真正的 post-lead support 几乎没有学出来。
warmup/fake bottleneck gate 会让 total 在早期看起来不低，但 hard_total/support_score 暴露真实支撑仍失败。
```

必须吸取的教训：

```text
1. 禁止再使用会自动修改代码、自动杀训练、自动重启训练的 watchdog 脚本。
   这类工具副作用太大，会制造新的不可审计变量。

2. support_bottleneck_warmup_min_gate 不能用来制造“假通过”。
   support 不达标时，total/hard_total 必须诚实地低。

3. bodyflat -> ActionScore 迁移不能靠普通行走 reward 过渡。
   如果 Train/mean_reward 上升而 highstep_action_score/support_score 仍接近 0，
   说明目标函数仍被普通 locomotion 占主导。

4. 继续从 bodyflat checkpoint 短训可以作为诊断，但不能再当成已验证有效路线。
   `2026-07-01_18-19-06` 及其 teacher `model_141000` 仍是证明流畅动作可训出的正例。
```

本轮代码处理：

```text
删除 scripts/tools/highstep_action_score_watchdog.py。
ActionScore 配置中 support_bottleneck_warmup_min_gate 改为 0.0。
降低 track_lin_vel_xy_exp / track_ang_vel_z_exp / feet_air_time / feet_gait 对专用高台任务的竞争。
提高 rear_first、rear_second、lead_rear_support_drive、post_lead_body_drive 的密度和权重。
post_lead_body_drive 更偏向 body height，而不是让 forward progress 掩盖身体没有上台。
```

## 2026-07-03 复核另一个对话补丁：撤回 fake bottleneck warmup

另一个对话把 ActionScore 配置改为：

```text
support_gate_floor: 0.22 -> 0.08
support_bottleneck_warmup_min_gate: 0.0 -> 0.35
track_ang_vel_z_exp.weight: 1.20 -> 1.55
lin_vel_x: (-0.08, 0.70) -> (-0.28, 0.72)
```

复核结论：

```text
1. support_gate_floor=0.08 可以暂时 defend。
   它只是在 support_score 刚超过 0.08 后给 composite score 一点稀疏梯度；
   support_floor 仍是 0.45，support_floor_violation_rate 仍会诚实显示失败。

2. support_bottleneck_warmup_min_gate=0.35 不能 defend。
   这会让 total / Episode_Reward/highstep_action_score 在 support 不达标时产生假分，
   和 2026-07-03_02-53-22 的失败教训冲突。

3. 恢复 yaw 跟踪和后退能力方向合理，但 lin_vel_x=-0.28 会让短训中过多样本不触发 highstep reward。
   当前折中为 lin_vel_x=(-0.18, 0.72)：保留后退控制，同时让训练更集中在上台。
```

本轮处理：

```text
highstep_env_cfg.py:
  reward / terrain curriculum / metrics 三处 support_bottleneck_warmup_min_gate 均改回 0.0。
  support_gate_floor 保持 0.08。
  track_ang_vel_z_exp.weight 保持 1.55，后续监控其是否压过 highstep_action_score。
  commands.base_velocity.ranges.lin_vel_x 改为 (-0.18, 0.72)。
```
