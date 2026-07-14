# Highstep 0707 真机视频/rosbag 审计与暂停断点交接（2026-07-12）

更新时间：2026-07-12 11:45，Asia/Hong_Kong。

## 2026-07-12 11:45 首轮 Student 终评与 actor-frozen refine

首轮 Robust Student `2026-07-12_04-41-42_robust_student_distill_20260712_044124` 已正常完成 `5000/5000`，三 checkpoint × core9 共 27 场也完整结束，基础设施无缺场。行为未达候选：`model_900=3/9`、`model_2900=0/9`、`model_4999=1/9`；机器选择 `model_900.pt`，但其 rear-platform hold 最坏值为 0，Teacher width/center/hold retention 均失败，decision=`student_refine_required`，禁止导出真机候选。三点证明中后期蒸馏退化；loss 下降不能覆盖行为下降。

Codex 对照权重与标量确认 actor adaptation 从约 update 300 开始，并持续改动后髋/box 输出；本次只改一个机制：把 ActionScore Student 的 `student_actor_warmup_updates` 提到 `100000`，使 5000-update 有界轮内 actor 完全冻结，只继续 blind estimator 适配。没有改 Teacher reward、`action_scale`、`joint_pos.clip`、`default_dof_pos`、obs/action order 或部署端。

旧 `model_900.pt` 的 `infos=None`，不能冒充 full resume。先以 weights-only/reset 做 1-iteration smoke，得到带 versioned `vae_optimizer_state_dict + student_distill_update_count` 的：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
2026-07-12_11-37-32_robust_student_actor_frozen_smoke_20260712_1138/model_0.pt
```

一次直接 weights-only 长训在第 26 轮遇到 Isaac/Carb recursive mutex assertion；Codex 已停止该进程和旧 monitor，不得恢复或计入正式结果。当前替代正式轮从上述 smoke `model_0.pt` 做真实 `full + preserve`：

```text
run: /home/lxq/Softwares/robot_lab/logs/rsl_rl/
  arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
  2026-07-12_11-41-02_robust_student_actor_frozen_full_refine_20260712_1142
train PID: 2537524
train unit: highstep-robust-student-actor-frozen-full-refine-20260712_1142.service
monitor PID: 2542349
monitor unit: highstep-robust-student-actor-frozen-full-refine-20260712_1142-monitor-x3s2000.service
W&B: https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/mptni7y1
```

首次 Codex 判断恰好触及 600 s timeout，留下了已停止临时 PID 的 stale handoff；主对话只修复控制元数据并重启 auto-loop，没有触碰新训练/monitor。11:44 后 `LIVE_HANDOFF_ADOPTED`，`validate_codex_handoff=True`、`training_control_health=True`，三个 unit 均 active；约 iteration `40/5000`、actor-adapt debug 恒为 0、distill count 连续、ETA 约 `6 h 15 min`。automation `60/60`、定向 pytest `70/70`、py_compile/bash/JSON/git diff check 全通过。结束后仍按 count=3/stride=2000 跑三 checkpoint core9。

## 2026-07-12 06:25 睡前自动化巡检

训练、三-checkpoint post-finish monitor 和 `highstep-auto-loop.service` 均为 active/running，`NRestarts=0`；schema-v4 handoff 与 `training_control_health` 均为 True。正式 Student 已到约 `1310/5000`，ETA 约 `4 h 42 min`，最新 `model_1300.pt` 正常落盘；唯一 train、唯一 monitor，无并发 play。monitor 每 120 秒更新且已持续识别 checkpoint。GPU 约 50°C、79% 利用、7.05/16.38 GiB，日志无运行期 fatal/CUDA OOM/PhysX/NaN；开头三条 Warp driver entry 警告没有阻止实际 CUDA 训练，监督器也判定 `fatal=False`。

Codex 决策链已经接入且健康：auto-loop 每 300 秒巡检，显式绑定当前 thread `019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c`、`gpt-5.6-sol/max`，Codex CLI 与代理端口可用，当前 5 h 额度约 82% 可用。活跃训练期间不周期唤醒 Codex是预期规则；训练结束后 monitor 先自动完成三 checkpoint × core9，评估进程结束后由 `train_not_running` 事件唤醒 Codex，并强制验证原子 JSON handoff；失败每 600 秒重试。三项服务均持有 sleep/idle inhibitor。磁盘可用约 22 GiB；低于 15 GiB 触发 Codex 资源处理，低于 5 GiB 即使额度不足也会优先 SIGINT 安全保存/停止训练。

## 2026-07-12 06:12 MuJoCo 执行链终核与监督器复验

已从默认 launch 一直追踪到 `mj_loadXML` 和编译后的 actuator：实际 sim-to-sim 加载的是 `arcdog_adjustable_leg_description/xml/scene.xml`，正前方启用的单个平台顶面精确为 `0.350 m`；安装文件与源码是同一符号链接/SHA。四腿 16 个 actuator 都是直接 joint motor，不存在 tendon/site/slider-crank transmission，也没有自定义 `gear` 或 gain 放大；默认 `gear=1, gain=1, bias=0`。最终硬控制上限为每条腿 hip `23.7 N·m`、thigh `23.7 N·m`、calf `45.43 N·m`，伸缩 box 为 `200 N`。因此 35 cm MuJoCo 成功动作本身已经排除“必须依赖 80 N·m calf 才能上台”的解释。

mandatory effort gate 已从自动策略与 handoff 验证彻底撤销；effort/velocity 只可选记录，不得阻断 core9、导出、视频或真机候选。06:12 只重启了 `highstep-auto-loop.service` 以加载修正规则，没有中断训练 PID `3043663` 或 monitor PID `3686062`。重启后 `validate_codex_handoff=True`，`training_control_health=True`，监督器仍绑定三 checkpoint `count=3/stride=2000`；回归为 automation `60/60`、定向 pytest `70/70`。训练约 `1156/5000`，日志无 fatal/CUDA/PhysX，ETA 约 `4 h 53 min`，磁盘可用约 `22 GiB`。

## 2026-07-12 05:59 Student 路线对抗审计与 sim-to-sim 反证（最新恢复入口）

用户质疑是否因催促而跳过关键步骤后，已按“先证明路线、再允许真机结论”做对抗审计。结论不是全盘辩护：启动一轮受控 Robust Student 有证据，因为冻结的 `model_172300.pt` 在 Standard core9 与只改变 rosbag 实测 gains 中心的 Robust core9 均为 9/9，继续无依据地改 Teacher reward 反而可能破坏已经满足“两后足上台、机身进入并可切 fixed stand”的动作；但原来的“只评最后 checkpoint”确实会漏掉蒸馏后期退化，已改成早/中/末三点。不能把当前训练指标直接称为真机成功证据。

当前正式 Student 仍是唯一 GPU 训练，PID `3043663`，run 与 W&B 不变；05:59 已推进到约 iteration 1000/5000，无 fatal/CUDA/PhysX 错误。不要为了形式上的审查中断它：当前进程启动于 checkpoint 修复落盘之前，现有 `model_N.pt` 没有保存独立 `vae_optimizer` Adam state 和 `student_distill_update_count`，中途停止后无法真正无缝恢复。继续本次 uninterrupted run 比停下再假装 full-resume 更严谨。

Student 结束后的评估已从单一末点升级为 3 个 checkpoint × core9，共 27 场。live monitor 合同为：

```text
unit: highstep-robust-student-distill-20260712_044124-monitor-x3s2000.service
monitor PID: 3686062
out: /home/lxq/Softwares/robot_lab/tmp/
  highstep_robust_student_distill_20260712_044124_core9x3s2000_monitor
EVAL_CHECKPOINT_COUNT=3
EVAL_CHECKPOINT_STRIDE=2000
EVAL_CHECKPOINT_PATH=(empty)
expected selection: approximately model_900 / model_2900 / model_4999
```

新 handoff 已同步 `count=3/stride=2000`，新源码验证 `validate_codex_handoff=True`，train/monitor/auto-loop 三个 unit 均 active。Student 候选门槛至少为 8/9、最坏 rear-platform hold `>=100` steps、保留 Teacher 最坏 hold 的 `>=50%`（当前 parent 为 505 steps，因此实际约 `>=252.5`）、后足宽度/离中线 q05 保留 `>=80%`，并且必须完成三个互异 checkpoint。通过后按主线直接导出/视频；真机仍需人工安全确认，但不得被低优先级 FL/strict-sequence 诊断阻塞。

曾一度把 Isaac asset 的 calf `80 N·m` 与真机 URDF `44.4 N·m` 差异升级成 mandatory effort gate；用户指出真机实验一直由人提着机身、已经显著卸载 knee/calf，但动作仍无法自然完成，这首先反驳“单纯力矩不足”。随后对真实 sim-to-sim 链做了源文件与 MuJoCo 3.2.6 编译模型双重核验：launch 实际加载 installed `scene.xml`，它与 source SHA 完全一致并启用了 `highstep_box_L10_h0350`；16 个 actuator 均 `gear=1, ctrllimited=1`，四腿 hip/thigh/calf 上限分别只有 `23.7/23.7/45.43 N·m`。控制桥输出 `kp*(q_des-q)+kd*(dq_des-dq)+tau`，消息里的二次 torque clamp 被注释，但 XML ctrlrange 真实生效。用户已经能在这套更低 hip/thigh、近似同 calf 上限的 MuJoCo 模型中轻松爬 35 cm，因此“策略需要 80 N·m”没有依据，mandatory effort gate 已撤销。

保留的 `play.py --eval_effort_limits` 与 torque/velocity telemetry 只可作为可选诊断，不接自动主线、不阻断 core9 候选、也不得因单独失败触发重训。核心原因重新锁回真实动作差异：接触后后腿轨迹/相位、向中线收窄、边缘卡住以及 Student 是否忠实继承 Teacher。`action_scale`、`joint_pos.clip`、`default_dof_pos` 与 observation/action order 继续绝对不改；physical target envelope 暂时只读审计，不把 FL 超限升为本轮淘汰门槛。

后续新启动的 Student 已补版本化 checkpoint state：`VAEPPO` 保存/恢复 `vae_optimizer_state_dict + student_distill_update_count`，runner full-resume 缺此状态默认 fail closed；旧 Student 只有显式 `--legacy_student_distill_update_count=N` 才能迁移，并必须承认 Adam moments 已丢失。该补丁只作用于未来进程，没有注入当前 PID。

## 2026-07-12 04:43 rear-platform 门禁完成并进入正式 Robust Student（当前实时入口）

用户最高目标已经落实为机器门禁：两只后足 RL/RR 都在高台顶面产生真实向上承载，后足和机身进入平台并连续稳定至少 25 control steps，使操作者有时间手动切换 `fixed stand`。FL 是否随后落回台面、双前足持续接触、严格事件顺序、`rear_advance >= 0.18 m` 均只作诊断，不得覆盖这一成功定义。`action_scale`、`joint_pos.clip`、`default_dof_pos`、observation/action 顺序仍绝对锁定。

当前冻结 parent：

```text
checkpoint:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/
  arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/
  2026-07-11_11-22-23/model_172300.pt
SHA256:
  dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35
Robust parent manifest:
  /home/lxq/Softwares/robot_lab/tmp/
  highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json
decision:
  robust_teacher_candidate_for_student
```

两级 core9 均已完成：Standard gate 为 9/9；同一 checkpoint 在 rosbag 实测 revolute gains 中心、delay=0、关闭 play-time random events/persistent bias 的 Robust gate 也是 9/9。Robust gate 最差 rear-platform 连续保持 505 步（约 10.1 s），最差后足顶面余量 `0.12176 m`，最差机身余量 `0.34011 m`，最差后足宽度/离中线距离 `0.35349/0.11649 m`，无 early termination。前足 bilateral hold 为 0、strict sequence 为 0/9、rear advance 为 1/9，但按用户目标均为诊断项。

首次 Robust 聚合曾错误得到 `valid_count=0`。九场 raw 实际全部通过；唯一根因是 monitor 的 explicit-legacy 分支硬编码 `role == teacher`，漏掉了唯一合法的 `teacher_robust` real-gain bootstrap gate。修复后同时绑定标准/Robust 角色-任务组合、固定 bootstrap 绝对路径、run directory、payload/manifest/磁盘 checkpoint SHA 和评估代码 SHA，保持 fail closed；已有 raw 离线重聚合为 9/9，没有重复 GPU play。经验固定为：聚合失败必须先审 raw 与最后一个排除条件，不能把 `valid_count=0` 直接解释成 policy 行为失败，也不能据此盲目重训。

Robust Student 2-iteration smoke 已通过：

```text
task: RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0
run_dir: /home/lxq/Softwares/robot_lab/logs/rsl_rl/
  arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
  2026-07-12_04-40-47_robust_student_smoke_20260712_044028
log: /home/lxq/Softwares/robot_lab/tmp/
  highstep_robust_student_smoke_20260712_044028/smoke.log
checkpoint: model_1.pt
```

日志确认从 parent `model_172300.pt` 只恢复 policy weights，optimizer fresh、learning iteration `172300 -> 0`、schedule reset，并完成 2/2 更新，无致命错误。它只作集成证明，不是候选。

正式 Robust Student 蒸馏已经运行并由自动化接管：

```text
task: RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0
iterations: 5000
num_envs: 4096
seed: 42
run_dir: /home/lxq/Softwares/robot_lab/logs/rsl_rl/
  arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
  2026-07-12_04-41-42_robust_student_distill_20260712_044124
train_log: /home/lxq/Softwares/robot_lab/tmp/
  highstep_robust_student_distill_20260712_044124/train.log
train_unit: highstep-robust-student-distill-20260712_044124.service
monitor_unit: highstep-robust-student-distill-20260712_044124-monitor.service
monitor_out: /home/lxq/Softwares/robot_lab/tmp/
  highstep_robust_student_distill_20260712_044124_core9_monitor
W&B: https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/qej66dgm
```

启动时三个 unit `train/monitor/highstep-auto-loop` 均为 active；正式 handoff 已通过 `validate_codex_handoff=True, ok`，自动 controller 已记录 `LIVE_HANDOFF_ADOPTED`。启动时 ETA 约 6 h 39 min。训练结束后 monitor 自动做 Robust Student core9；Student 至少 8/9 rear-platform hold 且无严重后腿内收才发布候选，否则按策略进入一次有边界的 Student refine。当前不需要用户提供新信息。

若插件或对话再次中断：先只读检查上述三个 unit、`tmp/highstep_goal_orchestrator/handoff.json`、train log 尾部和 monitor heartbeat；不要重跑 real-gain core9，不要重做 smoke，不要从其他 checkpoint 新开分支。当前 CPU 回归为 `103/103`，`bash -n`、`py_compile`、JSON 校验均通过。

## 2026-07-12 04:20 最高目标、经验教训与自动化覆盖（当前最高优先级）

用户再次明确，唯一最高目标是尽快得到能在真机上把整机爬上高台的 policy。对本轮任务，“整机已上台”的首要可操作定义是：RL/RR 两只后足都在高台顶面产生真实向上接触，后足中心进入平台近边缘至少 `0.04 m`，机身也已进入平台并保持足够长的稳定窗口，使操作者可以手动切换 FSM 到 `fixed stand`。左前腿是否在整机上台后主动放回台面不是最高优先级；FL 抬腿、前足双侧持续接触和严格前足事件顺序继续记录，但不得单独阻断 Teacher/Student 晋级。训练仍以把后足进一步送入平台（目标余量 `0.18 m`）作为优选方向，`0.18 m` 不再冒充“后腿是否已经上台”的最低定义。

本轮固定经验与教训：

- 0707 真机日志暴露了关节越限、约 60 ms 切换掉控、实际 gains 漂移和 policy 身份不可追溯等严重工程问题，但这些不能解释或掩盖核心动作没有训好的事实；真机主要失败链仍是接触后后腿收窄、卡边和无法把两后腿稳定送上台。
- 不能再用训练 reward、kinematic top hold、前足姿态或宽松代理替代用户真正需要的结果。主门槛必须直接验证 RL/RR 顶面接触、后足/机身平台余量、连续稳定窗口、无提前终止和无严重后腿中线塌缩。
- FL 在整机上台后仍抬起只作为低优先级诊断；它可以在后续专项修复，当前允许操作者用 `fixed stand` 复位。不得因此拖延一个已经能把后腿送上台的候选。
- `action_scale`、`joint_pos.clip`、`default_dof_pos` 语义和 observation/action 顺序继续绝对锁定；旧 checkpoint 与平地 baseline 的动作合同不得破坏。
- real gains 与 Robust Teacher 从未被用户禁止。此前把“不要让泛化支线拖慢核心动作”扩大成关闭 Robust 主线是错误。正确顺序是先得到 rear-on-platform Standard Teacher，再用同 checkpoint 做固定 delay0、关闭 play 随机事件的实测 revolute gains core9；通过后进入 Robust Student，失败才允许一次有界 Robust refine。额外 delay/bias/force 必须与 gains-only 结论分开。
- 每一轮训练必须有明确、单一、可验证的行为变量。若已有 checkpoint 在新的 rear-on-platform 门槛下通过，应先重评并复用，不得为了完成既定流程而无谓再训 2000 iterations。

当前候选锚点改为：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
  arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/
  2026-07-11_11-22-23/model_172300.pt
SHA256: dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35
```

旧 core9 已显示该模型 9/9 双后足 clear、第二后足之后双后足顶面接触率最差仍为 `1.0`、顶部保持约 517 步、机身平台余量最差约 `0.262 m`、后足余量最差约 `0.049 m`；它失败的是旧 `0.18 m rear_advance/dwell` 目标和前足严格事件链。由于新主门槛需要新增逐步连续 hold 计数，旧聚合结果只能作为强证据，正式晋级仍须用新 tracker 快速重跑 core9。若新 Standard 与 real-gain core9 均通过，直接进入 Robust Student；只有失败项才触发对应的有界修正。

截至写入时正式训练、play、monitor 均未启动；`highstep-auto-loop.service` 为 enabled/inactive，GPU 空闲。自动化代码正在按上述门槛合并和跑 CPU 回归，完成前不得盲目启动旧恢复命令。

## 2026-07-12 02:02 当前对话恢复与终审完成（中断时从这里继续）

- 本对话已经恢复到插件中断前的完整工作态。正式训练、正式评估矩阵、W&B、monitor 和自动 controller 均未运行；`highstep-auto-loop.service` 仍为 `disabled/inactive`。`tmp/highstep_goal_orchestrator/handoff.json` 中旧 `training_started` 只是不再存活的历史控制面记录，不能据此恢复。
- 唯一可靠旧断点仍是 `2026-07-11_23-45-31/model_600.pt`，大小 `11425355` bytes，SHA256 `3515ee8aa29b5e91ca7813798cbfe6b820895d0e87217f331fe790682fe6d94c`。schema6 nominal 600-step 预检已证明它不能作为 preserve-safe parent 或部署候选：严格事件链失败，raw target 越限率 `0.274375`、越限步率 `0.976667`、最大超限 `0.541321 rad`、最长连续越限 `574` 步。
- 最终补丁后又做了一次**非正式** 256-env×2-iteration action-safe migration smoke：Robust task、`model_600.pt` weights-only、fresh optimizer、schedule reset、tensorboard only。新 run 为 `2026-07-12_02-00-50_action_safe_migration_sha_smoke_20260712`；train manifest 固定 source model600 SHA，runtime sidecar schema2 的 model0/model1 SHA 均与文件一致，`model_1.pt` play 精确恢复 `runner_iteration=1 -> schedule_update=2` 并通过 runtime SHA 校验。它不是正式训练、不是候选、也不能替代长训练。
- schema6 核心严格门禁已完成：真实 delay、level/type/低地 patch/台阶与初始几何、当前帧向上法向接触、16 维 action order/scale/offset/asset IDs、checkpoint 与评估代码 SHA、source manifest、冻结 runtime clock、独立 600-step 计数、双前足→首后足→双后足→rear advance 事件链均 fail closed。旧 model600 的两步 `nominal_delay1` GPU 探针也已实测 `requested=1/supported=true/runtime_observed=true/runtime=1/match=true`；台阶 `0.326007 m`，高度门禁已从 `0.20..0.65` 收紧为 `0.28..0.40 m`。
- 终审新增的 schedule provenance 修复已落盘并通过 targeted 22/22：action/terrain/support/reward-stage cadence 进入 `schedule_definition`；train/play manifest 统一记录 source checkpoint SHA；每个新 checkpoint 的 runtime snapshot 记录 checkpoint SHA，同名同 iteration 替换会被拒绝。旧无 SHA sidecar 不能 preserve，只能显式 migration/reset。
- 终审发现的 Student parent lineage 高风险项已闭环：首次 Student 必须把实际 Robust Teacher parent manifest 的 resolved path+SHA 及 selected Teacher checkpoint path+SHA 写入 train manifest；实际加载 checkpoint 必须完全相同；四字段祖先跨 Student 续训传播。monitor 和 automation policy finalize 都会重新读取/哈希，换 parent、同内容换路径、重复 selected summary、runtime checkpoint 替换均 fail closed。旧 Student manifest 无 lineage 不能直接严格续训。
- 最终纯 Python 回归 `71/71`：schedule `22/22`、Student lineage `6/6`、target contract `5/5`、automation reliability `38/38`；独立 Student 3×3×6 fixture 为 18/18/18 valid 且 parent/source lineage/policy binding 全真；`py_compile`、`bash -n`、`git diff --check` 均通过。
- 最终关键源码 SHA256：`train.py=8ec336e77a70...`、`play.py=ec7e0fb3396b...`、`highstep_schedule.py=027f26f9a8ea...`、`monitor=880d013352c2...`、`automation_policy=786318d7e56d...`、`orchestrator=af247dc119f1...`。补丁后 smoke 总结为 `tmp/highstep_action_safe_migration_sha_smoke_20260712/summary.json`，SHA256 `744f128fc2d078412fb6e98a12b768bcb3add1ecdc925647dfd2934e6291e76d`。
- 部署端 current-q hold、完整命令原子提交、policy epoch/mutex、严格 4×4/16 YAML、controller joint order、mandatory physical envelope、resolved policy identity/manifest、callback→RealtimeBuffer 控制输入快照均已完成；clean build 1/1、定向 CTest 3/3（行为断言 10+4+1）通过。没有启动 ROS/controller/电机。真机前仍必须做无电机 FSM switch、机载坐标/限位核对和人工安全评审；运行时 FNV 不能冒充 SHA256。
- 磁盘约 23 GiB 可用（96% used），GPU 空闲。原始两组视频与 rosbag 未修改；第一组派生 `recovered.db3` 已在 compact 产物和哈希确认后删除。

如果本轮再次中断，先只读核验 service/process/checkpoint SHA，再读本节与 `tmp/highstep_action_safe_migration_sha_smoke_20260712/summary.json`。不得把两个 2-iteration smoke 当正式 run，也不得直接启用旧 controller。安全的下一条正式 lineage 只能是经用户确认后从 `model_600.pt` 做显式 `weights_only + migration + schedule reset` 的 Robust action-safe 新分支；首次 Student 还必须携带冻结的 `--highstep_parent_teacher_manifest`。

## 2026-07-12 01:02 插件恢复后的最新进度

- 训练/评估/自动 controller 仍全部停止；`highstep-auto-loop.service` disabled/inactive，可靠断点和 SHA 未变。
- 第二组精细分析已完成：`bag_rel = video_rel + 55.688 s ±0.030 s`；同一 policy158797、同一实测 gains、约 60.13 ms 切换全零窗、后足中线塌缩、calf/FL-hip unsafe target 和约 48.5 ms 转动关节相位滞后均独立复现。完整说明见 `tmp/rosbag_0707_alignment/second_analysis/README.md`。
- 第一组派生 `recovered.db3` 已在 compact 产物/脚本/哈希确认后删除，回收约 3.55 GB；原始 bag 大小复核前后均为 `3555594240` bytes，未修改。当前磁盘约 23 GB 可用。
- 新增完整报告 [highstep_0707_real_data_analysis_20260712.md](highstep_0707_real_data_analysis_20260712.md)，含两组对齐、完整训练链和有效/失效/副作用分类。
- 已落盘：统一 physical target clip、raw target penalty（actual limit 与 soft margin 分开统计）、8-step 首触 grace、Robust Teacher 实测 gain 中心、checkpoint-continuous schedule、train/play manifest、schema-5 eval tracker、三段 rear-transfer dwell、bilateral/slip/target-limit 硬门禁、6 场景×3 seed 独立矩阵、未来 run 名 task/role/argv 匹配。
- 已通过：schedule 11/11、automation reliability 18/18、static target contract 4/4、相关 `py_compile`/`bash -n`。
- StateRL current-q hold、policy1/2 strict key、epoch/mutex、mandatory envelope、manifest/debug 已完成第一轮 clean build 与 5/5 测试，但独立审查又发现三项必须封堵：半帧命令写入、YAML 转置前 exact-size、controller 实际 joint order 校验。部署支线正在修，禁止启动真机节点。
- schedule 独立审查又发现：patched sidecar 损坏不能静默 fallback，runtime prior/support 缺观测不能算 match，以及 model600 command curriculum 动态 range 会从约 `x=(-0.162, 0.464), y=(-0.054,0.054), yaw=(-0.3875,0.3875)` 重置到初始范围。schedule 支线正在把这些改成显式、可恢复/可审计语义；完成前禁止续训。
- 尚未做 Isaac GPU preflight；必须先等上述审查修复、复跑静态测试，再做 model600 单场 clamp/schedule 预检。不得直接恢复 54 场矩阵或训练。

用途：这是当前对话因插件重启、额度耗尽或上下文压缩后恢复工作的**最高优先级入口**。它记录两组 0707 真机视频与 rosbag 的对齐证据、从真机失败到当前训练的修改链、当前安全暂停点，以及已经完成的离线整改与仍需人工授权的正式训练/真机门槛。当前没有正式训练、评估或自动化进程在运行。

## 0. 恢复时先做什么

新对话按以下顺序读取：

1. 本文；
2. [Highstep Live Context Handoff](highstep_live_context_handoff.md)；
3. [Highstep 对话迁移：clean B](highstep_dialogue_migration_clean_b_2026-07-11.md)；
4. [部署链路记忆](sim_to_sim_deployment.md)；
5. [责任协议](accountability_protocol.md)。

然后只做只读核验：

```bash
systemctl --user is-active highstep-auto-loop.service highstep-monitor-resume-20260711_234523.service || true
systemctl --user is-enabled highstep-auto-loop.service || true
pgrep -af 'train.py|play.py|highstep.*monitor|wandb' || true
sha256sum /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_23-45-31/model_600.pt
```

预期：两个 unit 都 inactive，controller disabled，无 highstep train/play/W&B 进程；checkpoint SHA256 为 `3515ee8aa29b5e91ca7813798cbfe6b820895d0e87217f331fe790682fe6d94c`。

**禁止直接重启训练。** 先完成本文第 9 节的 schedule、严格评估、部署安全与监督器修正，并验证后才能决定是否从 `model_600.pt` 恢复。

## 1. 当前运行状态与可靠断点

- `highstep-auto-loop.service`：disabled/inactive。
- `highstep-monitor-resume-20260711_234523.service`：inactive；监控日志已记录 `monitor received interrupt`，没有进入评估。
- 训练、play、W&B 和 GPU compute：均无活动进程。
- 已执行 `sync`。
- 被停止的 run：
  `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_23-45-31`
- 训练日志：
  `/home/lxq/Softwares/robot_lab/tmp/highstep_display_reboot_resume_train_20260711_234523/train.log`
- W&B：`https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/srxfqe9o`
- 日志最后推进到约 iteration 685，但 601--685 没有可靠 checkpoint，已放弃。
- 唯一可靠断点：`model_600.pt`，大小 `11425355` bytes；`torch.load` 已确认包含 model/optimizer/iter/infos，iter=600，39 个模型 tensor、23 个 optimizer state。

暂停原因不是显示器问题，而是代码审计发现当前跨进程 clean B/自动续训会重置关键 schedule，继续训练会污染实验解释；真机日志又揭示了未被当前训练目标覆盖的限位与部署切换问题。

## 2. 用户提供的两组真机证据

第一组（视频只剪出上台动作）：

```text
video: /home/lxq/Videos/histep_real_robot_test_0707.mp4
bag:   /home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_52_07
```

第二组（较完整，作为独立复现）：

```text
video: /home/lxq/Videos/histep_real_robot_test_2026_07_07-00_28_27.mp4
bag:   /home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_28_27
```

视频与 rosbag 由不同机器记录，绝对时间戳不能直接对应，必须用手柄命令、IMU 冲击和可见接触事件对齐。

用户还提供一份部署端参考 YAML：`model_name_policy2` 指向
`policy_student_2026-07-05_00-13-46_highstep_model_158797.pt`，参考 gains 为
`50/55/70/2`、`1.5/1.5/2.5/3`。用户随后明确：YAML 只供参考，实际测试值以 rosbag 为准。

两份 bag 的 rosout 均直接确认 RL policy2 是 `158797`。两份 bag 的 `/joint_commands` 在有效 RL2 阶段均恒定记录：

```text
按 hip / thigh / calf / box：
Kp = 55 / 65 / 80 / 2
Kd = 1.5 / 1.5 / 2.5 / 3
```

因此本轮分析以这组 wire-level 实测 gains 为真值。参考 YAML、本机当前 YAML 和旧 backup 均不能覆盖 bag。

## 3. 已执行操作的逐步记录

1. 读取责任协议、live handoff、short-term、clean-B 迁移、显示卡死紧急暂停和 automation-v4 记忆。
2. 核验重启后的训练/监督器，曾从 iter401 派生锚点恢复 clean B。
3. 审计发现 schedule 跨进程重置后，先停 controller，再停 monitor，最后向训练发 SIGINT；确认所有进程退出并保存 `model_600.pt`。
4. 对第一份无 `metadata.yaml` 且尾部截断的 bag 做只读结构检查；保留原始 db 不写，建立派生恢复库与紧凑 NPZ/CSV。
5. 提取第一段视频 10 fps 帧、接触表、音频/IMU 冲击锚点，并完成视频到 bag 的事件对齐。
6. 从第一份 bag 提取 joystick、joint state/command、双 IMU、Euler、TF、debug topic、rosout、gains、关节目标越限和 qdes→qactual 响应。
7. 对第二份完整 bag 执行 SQLite quick check、元数据盘点并提取同一套信号；提取第二段视频帧和接触表。
8. 用第二组重复验证 policy 身份、gains、切换命令归零、FL hip 越限和后髋内收，不再把第一组当孤例。
9. 读取部署记忆并审计 `StateRL.cpp/.h`、硬件 `/joint_commands` 发布链和 `config.yaml`；确认 `/joint_commands` 是控制器实际写入硬件接口的 q/kp/kd，而不是离线推测值。
10. 重构 0707 真机失败到 Student、Teacher、SWAP support、centerline、stall、delay、Robust Teacher 和 clean B 的完整训练链。
11. 审计 `actions.py`、`curriculums.py`、`train.py`、`play.py`；确认 prior/curriculum/eval iteration reset 是当前自动化的实验有效性缺陷。
12. 已为本轮记忆修改建立备份，后缀为 `.before_0707_real_log_audit_20260712`；本文写入前尚未修改训练或部署源代码。

## 4. 第一组 bag 的完整性与对齐

原始 db：`rosbag2_2026_07_07-00_52_07_0.db3`，大小 `3555594240` bytes。SQLite header 宣称的尾部比实际文件多 `6688` pages，即缺 `27394048` bytes（约 26.125 MiB）；标准 SQLite 报 malformed。

只读 B-tree 扫描和 `.recover` 派生结果：

- 可访问 message 共 `3,424,259`，ID 1..3,424,259 连续；
- 时间 `2026-07-07 00:52:08.291156` 到 `01:03:24.182441` HKT，持续 675.891 s；
- 仅 32 个缺页指针；`lost_and_found` 4 行均为 messages 的精确重复，没有发现逻辑消息丢失；
- 原始 db 保持不动；派生库：
  `/home/lxq/Softwares/robot_lab/tmp/rosbag_0707_recovery/recovered.db3`。

第一段 RL2：

- switch：bag 相对 `+40.461929 s`；
- rosout enter `158797`：`+40.469196 s`；
- exit：`+59.147898 s`；
- joystick 前进动作段：`+47.714862..55.690009` 和 `+56.330641..57.066680`。

第一段视频 12.933333 s、30 fps、388 帧。最佳事件映射：

```text
bag_relative_time = video_relative_time + 46.086 s
保守不确定度 ±0.025 s
```

关键锚点：

- 视频 2.185 s 首次清楚的前足撞击竖直面/边缘 → bag 48.269954 s；
- 视频 3.687/3.991/4.570 s 的连续冲击/摆动 → bag 49.775543/50.079560/50.653579 s；
- 视频 5.882/6.016 s → bag 51.973833/52.109308 s；
- 视频 10.731 s 末段冲击 → bag 56.813612 s；
- 视频运动起点 1.55--1.80 s 映射 bag 47.636--47.886 s，包含 joystick 47.714862 s 起点。

因此 policy switch 发生在剪辑开头约 5.624 s 之前。剪辑中的主要可见失败不是切换瞬态；但 bag 另行发现的切换瞬态仍是独立部署缺陷。

视频动作：1.8--2.2 s 后腿开始收窄；2.185 s 前足撞台面/边缘并停留；3.35--4.60 s roll/yaw/pitch 大幅重试；约 4.05--4.15 s 才首次清楚搭上；后腿重构和卡边持续至约 10.7 s；11.7--12.0 s 才四足在台上。安全绳约 3.5 s 起明显主动拉拽，最终成功不能算独立稳定成功。

## 5. 第二组独立复现

视频：30.266667 s、30 fps、908 帧。bag 完整、`quick_check=ok`：

- db 大小 `641216512` bytes；
- 开始 `2026-07-07 00:28:27.937708912`；持续 130.270669 s；628,588 messages；
- policy2 switch bag rel `+54.987194 s`；enter `158797` `+54.995119 s`；exit `+84.403547 s`；
- debug topic 有效约 29.2896 s；
- joystick 在进入 policy 后约 10 s 开始前进，和视频约 12 s 开始运动一致。

第二组重复出现：

- policy2=158797；
- 实际 gains 完全同第一组；
- policy 切换后约 70 ms 的 q/kp/kd 归零窗口；
- FL hip qdes/actual 峰约 `2.550/2.259 rad`，超过当前部署 URDF `±1.22173 rad`；
- RL/RR 后髋命令同时落入 `|q|<0.06` 的累计比例约 34.7%，最长约 2.69 s；实际后髋相同风险约 43.8%，最长约 4.10 s；
- 视频呈现长时间安全绳保护、前足搭台后后部推进困难和辅助上台。

第二组的最终精细视频 offset/逐帧事件表尚需在下一轮补齐，但 rosout、joystick、IMU、关节与视频运动起点已把整段 RL2 对应关系锁定，足以作为独立复现证据。

## 6. 真机数据揭示的根因

### 6.1 真实动作失败：后髋/足端向中线塌缩

第一组 TF 足端相对 trunk：

- policy 前 rear width median 约 0.412 m；
- policy idle 0--6.5 s 约 0.472 m；
- 运动起点 6.5--8.5 s median 0.450 m、min 0.264 m；
- 首前足接触窗口 8.5--12 s median 0.330 m；RR `min_abs_y` 在 policy-relative 9.3169 s 塌到 `0.02162 m`，rear width 最低约 0.268 m；
- 后续 12--18.57 s median 仍仅约 0.325 m，min width 约 0.261 m。

这和视频“首前足接触时后腿内收、单前足承载、roll/yaw 摆动、后腿卡边”的链路一致。rear-hip guard 的方向有局部价值，但它只解决症状的一部分。

### 6.2 更严重的新证据：部署目标越物理限位

第一组 RL 动作段：

- FL hip qdes 超 `±1.22173` 的比例约 42.4%，actual 超限约 40.9%；
- qdes max `2.5284 rad`，actual max `2.2317 rad`；
- 连续越限区间 bag rel `52.4048..55.690` 和 `56.6129..57.0656`，映射视频约 `6.319..9.604` 和 `10.527..10.980 s`；
- RR calf qdes 超 URDF 范围约 27.1%，actual 没超；RL calf qdes 约 2.25%，FL calf约 0.75%。

第二组 FL hip 也达到 qdes/actual `2.550/2.259 rad`。这不是单次解析噪声。

0707 部署 config 的 `clip_actions ±60`、revolute `output_dof_pos ±60` 等于没有物理保护。训练也使用 raw action ±60，只有 box target 被限制。策略可能利用饱和/机械限位形成仿真中看似成功、真机中边缘硬撑的动作。必须把训练、严格评估和部署端的 revolute target contract 对齐。

### 6.3 policy 切换时真实控制掉线

第一组：switch 后 `/joint_commands` 全部 qdes=0 且 kp/kd=0，从 bag rel `40.481397` 到首个有效命令约 `40.540718`，约 59.3 ms；IMU 随后出现加速度范数约 35.3 m/s² 的冲击。第二组约 70 ms，重复出现。

代码原因：`StateRL::enter()` 初始化 `output_dof_pos_`，但没有同步填充 `robot_command_`；异步推理线程尚未完成第一次 inference，而控制循环的 `run()->setCommand()` 已把默认零 q/kp/kd 写出。

正确修复是 command-valid 原子交接：进入 RL 时先用当前实测关节位置（或前一状态最后安全命令）加 RL gains 形成 hold command；只有第一次完整 inference 原子发布后才切到 policy command。不能只靠 sleep。

### 6.4 gains/配置追溯漂移

两份 bag 的 wire-level gains 一致，却同时不同于用户参考 YAML、本机当前 YAML和 0707 backup YAML。说明真正运行的 resolved config 没有被完整保存。普通 `model_name` 经常切换，也不能代表 policy2。

后续每次真机启动必须发布并写入 rosbag：resolved policy1/policy2 绝对路径、policy SHA256、resolved config SHA256、逐关节 q-limit/action-scale/Kp/Kd、controller git commit/build ID。启动门禁应将 `/joint_commands` 前若干有效样本与 manifest 对比，不一致就禁止进入运动测试。

### 6.5 延迟不能混为一个数

- bag topic header→record 延迟 median 约 0.12--0.17 ms，p99 小于约 0.8 ms；这不是主要 action transport delay。
- qdes→qactual 交叉相关最佳滞后多数 36--58 ms；FR hip 约 97 ms，box 约 85--107 ms。这主要包含机械/PD 闭环响应。

训练的 0/1 control-step delay 只应用来覆盖推理/命令离散延迟；机械响应应通过实际 55/65/80 gains、执行器动力学和独立 gain/dynamics 场景校准，不能直接把 100 ms 全塞进 action delay，否则重复建模。

## 7. 0707 到当前训练的修改链与重新判定

1. `158797 -> 159900` Student rear-hip 小修：仿真 rear width 有改善，但未部署，也不是当前 lineage 父节点。方向局部有效，无法解决 FL hip 越限/切换掉线/支撑时序。
2. Teacher rear-width `151200 -> 155600`：加入 approach/motion width 和 DR，sim width 改善；有效但证据只在仿真，不能单独证明真机修复。
3. Student `155600 -> 159700`：宽度比中段好但 score 更低，Student-first 被放弃；不应继续作为主线。
4. SWAP-inspired Teacher `155600 -> 160300`：双侧接触、slip、roll/yaw、rear lag/support width 方向合理；但旧 row pass/full climb 没把这些做硬门禁，所谓“通过”可能仍是危险支撑。
5. centerline `160300 -> 165999`：修正中心线 hard risk 与 curriculum gate；旧 eval pass 0→0.133，但 instant min_abs 仍近 0，部分有效。
6. reward alignment `165999 -> 167998`：把 centerline gate 真正乘进 ActionScore，是有效代码修复；但 pass 仅 0.2、dwell 551，没解决卡边。
7. post-clear latch：单独无效，因为真实/仿真瓶颈常发生在第二后足 clear 之前，触发太晚。
8. bilateral rear advance 到 stall penalty：中间 checkpoint 曾把 pass 提到 0.733、avg dwell 降至约 113，但 nominal worst 仍约 533--548；改善均值，未消除确定性长卡边模式。
9. standard delay 0/1 mix：pass 0.733→0.667、avg dwell 113→151，未证明有效；且 delay1 与其它扰动绑定，缺少独立 factorial。
10. Robust Teacher persistent encoder bias：只覆盖 observation position bias，不覆盖低层 PD feedback drift/dropout、IMU bias、执行器响应或切换掉线；方向有限，覆盖面不足。
11. clean B：原本想比较 optimizer reset，但实际同时重置 action prior 与 terrain/support curriculum schedule，实验被混杂；当前暂停是正确选择。

潜在副作用：

- `highstep_support_stability_penalty` 从 commit 后立即生效，没有首次前足接触的自然顺序 grace，可能惩罚合理的单前足过渡；
- post-clear 类奖励/惩罚触发太晚，可能让策略先学会危险撞边再在后段“恢复”；
- rear width/centerline 约束过强会把策略推向宽站但不推进，且掩盖前足动作越限；
- deployment clamp 若只在真机端加、训练不加，会让策略在真机持续饱和，虽安全但动作失真；必须训练/评估/部署一致；
- final success 指标目前允许至少 2 contacts 且不硬门禁 bilateral contact/slip，可能把安全绳式/边缘硬撑等价地算成功。

## 8. 当前自动化为何不允许恢复

`PhasedHighstepBoxBiasJointPositionAction` 使用：

```text
update_count = env.common_step_counter / 24
prior 80 开始、520 完全展开
```

ActionScore terrain/support curriculum 也用进程本地 `common_step_counter`，阶段阈值 `650/1300/2200/3200`、最大 level `1/2/3/5/8`。每次新 train/play 进程都从 0 计数：

- 2000-update 分段永远到不了 level 5/8；
- 800/1200 refinement 更严重；
- weights-only clean B 同时重置 optimizer、prior、support bottleneck 和 terrain stage；
- full resume 恢复 runner iteration，却没有把它同步给 environment schedule。

`play.py` 加载 checkpoint 后也没有把 checkpoint iteration 注入 env；每场 600 steps 仅形成 update_count≤25，所以旧“strict eval”中的 Teacher/Robust Teacher action prior 实际为 0。没有显式 `disable_action_prior` 不代表 prior 在工作。

必须建立单一 `highstep_global_update`：train 在 load 后将 runner iteration 写入 env；weights-only 需明确选择 `preserve_schedule` 或 `reset_schedule` 并写 manifest；play 必须显式指定/从 checkpoint 恢复 schedule update，并在 JSON 输出 `prior_scale/allowed_max_level/support_bottleneck_blend`。监督器缺任一字段即判 eval invalid。

## 9. 下一轮必须按顺序完成

### A. 先完成分析产物

1. 补齐第二段视频↔bag 的最终 offset、±误差和逐事件表；
2. 输出两组并排的 switch dropout、rear width/min_abs、FL hip/calf target-limit、gains、IMU、qdes→qactual 表；
3. 把“视觉接触为推断、bag 无 contact/odom/base diagnostics topic”的证据限制写清楚。

### B. 修训练与严格评估 schedule

1. 为 `actions.py`、`curriculums.py` 增加统一读取 `env._highstep_global_update` 的 helper，回退本地 counter 仅作兼容；
2. `train.py` 在 runner load 后注入 checkpoint iteration/schedule mode，写 `highstep_schedule_manifest.json`；
3. `play.py` 恢复 checkpoint iteration并输出实际 prior scale、terrain stage、support blend；
4. 加纯 Python 单元测试，验证新进程 resume iter600/175800 时 schedule 不再归零；
5. 监督器将 schedule manifest 缺失或不一致列为硬失败。

### C. 修目标限位 contract

1. 从实际部署 URDF/真机确认每个 revolute 安全 target 范围；当前审计依据为 hip ±1.22173、thigh [-1.5708,3.4907]、calf [-2.77507,-0.64577]；
2. 在训练 action term、strict eval 与部署 `output_dof_pos` 使用同一 target limits；
3. 增加 raw target violation fraction/max delta/连续持续时间指标；任何 nominal/robust 场景有持续越限即 row fail；
4. clamp 触发必须发布完整 16 维前后值与计数，不能只打每 200 次一条 max delta；
5. 先离线评估旧 checkpoint 在 clamp 下能力，再决定从 model600 小步 refine 还是回更早 Teacher。

### D. 修部署切换和可追溯性

1. 备份 `StateRL.cpp/.h/config.yaml`；
2. 实现 current-position hold + command-valid atomic handoff，禁止初始 q/kp/kd 归零；
3. 启动时打印并发布 resolved policy path/SHA256、config SHA256、逐关节 gains/limits/scales；
4. 增加 `/rl_deployment_manifest` 和 `/rl_command_safety_debug`，确保 rosbag 自动记录；
5. 编译、静态检查和不接电机的 switch test 通过前，不部署真机。

### E. 修评估指标与自动化门禁

1. 把 dwell 拆成 `front_support→first_rear_top`、`first→second_rear_top`、`second_clear→rear_advance`；
2. 记录 first-front-contact 对齐窗口的 rear width/min_abs；
3. bilateral front/rear contact、slip、roll/yaw、target-limit、switch-valid 设硬门禁，并给首次接触合理 grace；
4. delay0/delay1 与 nominal/force/bias 独立 factorial，不再绑定；
5. supervisor run 名称匹配使用 task/stage/manifest，不依赖日期字符串或固定 run 名，保证以后新训练名称也能匹配；
6. disk 空间不足时禁止自动开训；当时剩余约 20 GB/96%，派生 recovered db 约 3.55 GB，等最终紧凑产物确认后可删除派生库，不能删原始 bag。

### F. 再决定是否恢复训练

恢复候选仍为 `model_600.pt`，但只有以下条件同时满足才允许：

- schedule 连续性测试通过；
- old checkpoint 在统一 physical clamp 下完成小矩阵评估；
- supervisor 新硬门禁可解析；
- deployment warm-start patch 至少编译/离线 switch test 通过；
- disk 和 GPU 状态安全；
- 新 run manifest 明确 parent SHA、optimizer load mode、global schedule update、target-limit contract、实际 robust gains 分布。

若 clamp 后旧策略大幅失能，不得硬续 clean B；应从旧 Teacher 做 action-safe refinement，先 Teacher→Robust Teacher→只用真机可得观测的 Robust Student→严格 eval→export。

## 10. 已生成的分析产物

```text
/home/lxq/Softwares/robot_lab/tmp/rosbag_0707_recovery/analysis/
  alignment_anchor_keyframes.csv
  candidate_event_timeline.csv
  deployment_limit_audit.csv
  extraction_summary.json
  joint_tracking_delay_summary.csv
  joy_samples.csv
  policy2_highstep_50hz.csv
  raw_policy2_imu_peak_anchors.csv
  raw_policy2_joint_extrema.csv
  rl_active_gain_summary.csv
  signal_extrema_policy2.csv
  tf_rear_width.csv
  topic_timing_quality.csv

/home/lxq/Softwares/robot_lab/tmp/rosbag_0707_alignment/
  extract_recoverable_window.py
  candidate_window/
  second_candidate_window/
  video_contact_2fps.jpg
  second_video_contact_1fps.jpg
  video_frames/
  second_video_frames/
```

`extract_recoverable_window.py` 已通过 `py_compile`。第一组 compact artifact 约 5 MB；第二组已提取 joint/tf/euler/imu/command/debug/joy NPZ。原始 bag 和视频不得修改。

## 11. 源代码整改与备份规则

以下列表是 01:02 初稿写入时的历史待办；到 02:02 已全部完成离线实现和回归：

- `robot_lab` schedule continuity；
- play strict schedule/fingerprint；
- supervisor 新指标/名称匹配；
- training physical target limits/real-gain Robust 环境；
- deployment StateRL warm start、target clamp 与 manifest。

最终状态以本文顶部 `02:02` 节为准：纯 Python `71/71`、补丁后 migration/play provenance smoke、部署 clean build 与定向 CTest 均通过；正式训练和真机运行仍未启动。

工作树本来已有大量用户/前序对话改动，不能 reset。修改前逐文件备份，统一后缀：

```text
.before_0707_real_log_audit_20260712
```

使用 `apply_patch` 做源文件修改；每次改动后先做语法/静态测试，再做最小无训练 smoke，最后才允许启动昂贵评估或训练。

## 12. 当前总判断

两组独立真机证据支持同一主失败链：`158797` 在前足接触高台后出现后支撑向中线塌缩、单前足/边缘承载和长时间后腿推进困难；安全绳明显介入，不能算独立成功。同时，rosbag 新增了三项此前训练链未覆盖的系统问题：revolute target 大幅越物理限位、RL 切换约 60--70 ms 的 q/kp/kd 全零掉线、真实运行配置缺少可验证指纹。

因此此前 rear width、centerline、support、stall 修改并非全部无效，但它们只优化了局部仿真指标，且 strict eval 的 action prior/schedule 实际被进程重置。当前最优动作是保持训练暂停，先修实验有效性与部署安全 contract，再以统一门禁重评旧 checkpoint；不是继续盲训，也不是把全部问题重新归因到 policy switch。
