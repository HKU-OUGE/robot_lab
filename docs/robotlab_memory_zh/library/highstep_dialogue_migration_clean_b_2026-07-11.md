# Highstep 对话迁移交接：Robust clean B 决策与恢复期经验（2026-07-11）

更新时间：2026-07-12 11:45，Asia/Hong_Kong。

用途：把 clean B 的判断、实验依据、恢复事实和自动化责任迁移到当前对话。本文保留稳定决策；实时状态仍须由经校验的 schema-v4 handoff、systemd unit 与实际进程共同确认。

> **2026-07-12 11:45 Student 终评覆盖：** 首轮三点 core9 为 `model_900=3/9`、`model_2900=0/9`、`model_4999=1/9`，未达候选且中后期退化。当前唯一新机制是冻结 Student actor（warmup `100000`），从 legacy model900 经 1-iteration migration smoke 生成完整 VAE optimizer/count state，再从 smoke model0 做 `full+preserve` 的 5000-update estimator-only refine。正式 PID `2537524`、monitor PID `2542349`、W&B `mptni7y1`；handoff/control health 已恢复为 True，回归 `60/60+70/70`。一次遇到 Carb mutex 的临时 weights-only 长训已停止，不得恢复。详细读 [0707 审计交接](highstep_0707_real_log_audit_handoff_20260712.md) 顶部 11:45 节。

> **2026-07-12 06:12 MuJoCo/监督器终核：** 默认 sim-to-sim 的 35 cm 单平台确实启用；16 个 actuator 无额外 transmission/gear/gain 放大，四腿 hip/thigh/calf 最终 ctrlrange 为 `23.7/23.7/45.43 N·m`，伸缩关节为 `200 N`。mandatory effort gate 已撤销，监督器已在不触碰训练/monitor 的情况下重新加载；handoff 与 control health 均通过，回归 `60/60 + 70/70`，正式 Student 约 `1156/5000`、ETA 约 `4 h 53 min`。

> **2026-07-12 05:59 对抗审计覆盖：** 启动 Robust Student 本身有两级 core9 9/9 作为依据，不需要为了流程完整再改/再训已经通过用户物理目标的 Teacher；结束后强制评估约 `model_900/model_2900/model_4999` 三点（3×core9=27场）。曾计划的 mandatory effort gate 已被用户人力卸载证据与实际 sim-to-sim 模型反证：MuJoCo 编译后的四腿 hip/thigh/calf 上限为 `23.7/23.7/45.43 N·m`、gear1，35 cm box 确实启用，而用户在该链上能轻松上台。因此 effort 只留作可选 telemetry，不阻断候选、不触发重训；主因仍是后腿轨迹/相位/中线塌缩与 Student 继承质量。当前训练 PID `3043663` 不应中断，因为旧进程 checkpoint 缺独立 VAE optimizer/count，停止会丢 Adam 连续性；未来进程已补版本化保存/恢复并对旧 full-resume fail closed。live monitor 已换为 `highstep-robust-student-distill-20260712_044124-monitor-x3s2000.service`，PID `3686062`，handoff count=3/stride=2000 且验证通过。所有训练与评估继续锁定 `action_scale`、`joint_pos.clip`、`default_dof_pos`、obs/action order；物理目标范围只读审计，不恢复曾经的训练 clip 修改。最新细节读 [0707 审计交接](highstep_0707_real_log_audit_handoff_20260712.md) 的 05:59 节。

> **2026-07-12 04:43 实时覆盖：** `model_172300.pt` 已按新的用户目标通过 Standard rear-platform core9 9/9，并在同 checkpoint、实测 gains 中心、delay0/no-random-events 条件下通过 Robust core9 9/9；冻结 parent manifest 为 `/home/lxq/Softwares/robot_lab/tmp/highstep_rear_platform_realgain_core9_20260712_042927/evaluation_manifest.json`。Robust Student smoke 2/2 更新通过，正式 5000-iteration run `2026-07-12_04-41-42_robust_student_distill_20260712_044124` 已运行，train/monitor/auto-loop 三个 unit active，W&B 为 `qej66dgm`。首次 real-gain 聚合的 `valid_count=0` 是 legacy 分支遗漏 `teacher_robust` 的基础设施错误，raw 行为实际 9/9；已用固定 bootstrap 路径、任务/角色、SHA 和 run-dir 窄绑定修复并离线重聚合，无需重跑 GPU。当前一切恢复都必须先读 `tmp/highstep_goal_orchestrator/handoff.json` 和 [0707 审计交接](highstep_0707_real_log_audit_handoff_20260712.md) 的 04:43 节，不得再恢复旧 clean-B/model600 路线。

> **2026-07-12 04:20 目标覆盖：** 用户把最高成功标准进一步明确为“真机两后腿都真实、稳定地上到高台，机身已进入平台，并能留出手动切换 `fixed stand` 的窗口”。左前腿在整机上台后是否主动放下、前足是否持续双侧承重不再是晋级硬门槛，只保留为诊断。最低 rear-platform 余量采用 `0.04 m`，训练优选目标仍可保持 `0.18 m`。动作 scale/clip/default/order 继续锁定。自动路线恢复为 Standard rear-platform gate → 同 checkpoint 的实测 gains/delay0/no-randomization core9 → Robust Student；real-gain gate 失败才做一次有界 Robust refine。当前优先读取 [0707 真机数据审计与暂停断点](highstep_0707_real_log_audit_handoff_20260712.md) 顶部 04:20 节。

> **2026-07-12 02:02 状态覆盖：** 两组 0707 真机视频/rosbag 审计后，clean B 已在 `model_600.pt` 安全暂停；controller disabled，monitor/train/play 均 inactive。schema6 600-step 预检确认该模型严格事件链失败且 raw target 长时间越限，所以它只保留为可靠旧权重锚点，不再是可 full/preserve 续训或部署的 parent。当前先读 [0707 真机数据审计与暂停断点](highstep_0707_real_log_audit_handoff_20260712.md)；下文“当前训练运行中”均只作历史记录。

> **02:02 终审补充：** 第二组对齐和跨试验复现、统一 physical target contract、schema6 dwell/bilateral/slip/raw-target/地形/真实 delay 门禁、checkpoint-continuous schedule、runtime checkpoint SHA、Student→Robust Teacher lineage、未来 run-name 匹配，以及部署端 current-q/原子命令/RealtimeBuffer 补丁均已完成。纯 Python `71/71`、部署定向 CTest `3/3` 通过；补丁后 2-iteration migration+play provenance smoke 也通过，但它不是正式训练。旧 clean B 不恢复；下一条正式路线只能经用户确认后从 model600 做 `weights_only + migration + schedule reset` 的 Robust action-safe 新 lineage。

> **迁移已经完成。** 当前自动化 thread ID 为
> `019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c`（本对话）；旧 thread
> `019f49b9-78e2-7610-972c-0834d70f279f` 不再承接后续决策。

## 0. 新对话读取顺序与冲突规则

开始任何操作前按以下顺序读取：

1. [Highstep 显示链卡死紧急暂停与恢复交接](highstep_display_hang_emergency_pause_handoff_2026-07-11.md)：当前最高优先级，包含暂停状态、可靠断点和恢复脚本。
2. 本文：解释为什么建立 clean B、如何理解恢复期，以及本次对话新增了什么。
3. [Highstep Automation V4 Short-Term Handoff](highstep_automation_v4_handoff_20260711.md)：自动化契约、Teacher/Robust/Student 顺序和 schema-v4 规则。
4. 恢复训练后生成的新 `handoff.json`、`decision.txt`、`evaluation_manifest.json`：只在进程和 schema-v4 完整性校验通过后作为实时机器事实。

冲突时采用以下优先级：

```text
用户最新明确指令
  > 紧急暂停交接
  > 恢复后经校验的实时 schema-v4 产物
  > 本文的实验决策与经验
  > 同日更早文件中的 PID/“训练仍运行”描述
```

用户已于 `2026-07-11 23:45 HKT` 明确允许恢复。恢复脚本已写出并校验新
`training_started` handoff，event 为
`display_reboot_resume:401:2026-07-11_23-45-31`；旧 PID `2797557/2803242`
和旧 W&B `lsoxm39v` 只作 lineage 证据。当前必须以新 handoff、实际 unit/argv
和进程存活校验为准，不得再次执行恢复脚本。

本文单独成文件的原因：现有 live/automation/紧急交接各自承担实时状态或自动化协议，直接合并会把稳定经验、历史实验和易过期 PID 混在一起。本文不加入现有 INDEX，避免与当前未提交索引修改冲突；迁移时直接使用本文绝对路径即可。

## 1. 不变目标与真机故障锚点

- 最高目标不是训练曲线好看，而是得到可冻结、可导出、可部署到真机的固定 policy，使真机尽量复现仿真中的上台动作和能力。
- 0707 真机证据固定为 `/home/lxq/Videos/histep_real_robot_test_0707.mp4`。
- highstep 已经切入并开始运动。失败发生在第一只前腿抬起并首次接触高台时：后腿向中线内收，随后出现单前足承载、边缘碰撞/停留、roll/yaw 摇摆和后腿卡边。不得重新解释为策略切换瞬态。
- 普通 Teacher 达标只允许进入 Robust Teacher；Robust Teacher 达标后进入只使用真机可得观测的 Robust Student；Student 严格通过后才能导出 policy、做接口一致性检查和验证视频。普通 Teacher 的 60% 门槛不是上真机许可。
- 固定 checkpoint 的完整多 seed、扰动、Robust、Student 和导出一致性验收高于 mean reward、末点单调性和“最后 checkpoint”。后续退化不得污染已经保存的更优 checkpoint。

## 2. 本次对话确认的训练恢复经验

### 2.1 每次接 checkpoint 都有恢复期

用户特别提醒并要求保留此前经验：每次新进程从 checkpoint 接训，开头约几百轮会受到 episode 统计、环境/curriculum 重新积累以及策略/critic 对新奖励或分布适应的影响。此前多轮都观察到前约 100 轮 reward 可暂时落在约 44，约 500 轮后才回到约 95 的平台。

因此后续统一遵守：

- 新分支至少训练约 2000 iterations；
- 前 500 轮只作为适应/恢复期，不用于判断收敛或宣布退化；
- 至少保留 300 至 500 轮稳定平台；
- 使用相同部署矩阵比较约 1000/1500/2000 的 checkpoint，而不是只看末点；
- 若最后约 300 轮部署指标仍系统改善，可延长约 1000 轮后重评；
- 若曲线已稳定但安全评估无改善，应修改目标、分布或策略结构，不能只机械延长。

恢复期开头的低 reward 不等于 checkpoint 的策略瞬间失效，但也不能反过来把训练 reward 恢复当作部署成功。

### 2.2 高 iteration 编号不是单独的污染证据

`175k/176k` 只是累计 runner iteration。actor 已学到的上台能力是真实资产，不能因为编号大就从随机初始化重训。

真正的链路污染风险来自：

- 反复 full resume 同一 optimizer 和策略习惯；
- 中途改变 reward、动作延迟或 Robust 分布；
- global update/curriculum prior 长期处于完全展开状态；
- 在恢复期内频繁改代码或再次换 checkpoint。

判断 lineage 是否继续，必须看恢复期之后多个 checkpoint 的同一部署矩阵斜率，而不是只看全局编号。

### 2.3 clean A/B 的正确含义

当 full-resume lineage 平台化时，合理对照为：

```text
A: full resume，保留 actor/critic、optimizer 和 runner iteration。
B: 只加载 actor/critic 权重，重置 optimizer、runner iteration 和新环境/curriculum。
```

两条分支都应给足约 2000 轮，前 500 轮都不判定，并在相对 1000/1500/2000 位置使用同一 45 场矩阵比较。B 不是随机重训，也不是只训 300 至 500 轮。

## 3. 到 clean B 之前发生了什么

普通 Teacher 阶段曾把 pass rate 提升到 `0.7333`，但 rear-edge dwell worst 长期停在约 `533-548`，之后回落并触发 strategy escalation。自动化据此切到 Robust Teacher，而不是继续重复普通 Teacher reward，也没有把普通 Teacher 门槛当作可部署结论。

第一轮 Robust Teacher 完整 schema-v4 评估选中：

```text
checkpoint: model_174299.pt
pass_rate: 0.3333
full_climb_rate: 0.8667
rear_edge_dwell_steps_avg: 273.93
rear_edge_dwell_steps_worst: 545
valid runs: 15/15
```

随后进行一轮约 2000 iterations 的 Robust full-resume refine，完成恢复期并比较三个 checkpoint。首次评估因 play 基础设施 `rc=137` 中断，`evaluation_complete=false`，该无效结果没有被用于改 reward 或启动训练；改用无 DISPLAY 的 xless 评估后，得到完整 schema-v4 `45/45`：

| 相对位置 | checkpoint | pass rate | full climb | dwell avg | dwell worst |
|---|---|---:|---:|---:|---:|
| 约 1100 | `model_175400.pt` | 0.2667 | 1.0000 | 293.13 | 548 |
| 约 1500 | `model_175800.pt` | 0.3333 | 1.0000 | 234.27 | 533 |
| 约 2000 | `model_176298.pt` | 0.0667 | 1.0000 | 448.27 | 548 |

完整证据：

```text
/home/lxq/Softwares/robot_lab/tmp/highstep_robust_teacher_refine_xless_recovery_schema4_monitor_20260711_213851/evaluation_manifest.json
/home/lxq/Softwares/robot_lab/tmp/highstep_robust_teacher_refine_xless_recovery_schema4_monitor_20260711_213851/checkpoint_summary.csv
/home/lxq/Softwares/robot_lab/tmp/highstep_robust_teacher_refine_xless_recovery_schema4_monitor_20260711_213851/eval_runs.jsonl
```

结论：

- 恢复后中点 `model_175800` 最好，说明不是“开头恢复看起来低”造成误判；
- 到末点 pass rate 从 `0.3333` 降到 `0.0667`，dwell avg 从 `234.27` 恶化到 `448.27`；
- worst dwell 仍在 `533-548` 平台，未解决真机对应的后腿卡边；
- 继续同一 full-resume lineage 缺乏依据，但已有上台能力值得保留；
- 因此本次对话选择 clean B，而不是立刻再加 reward、随机重训或只延长同一 lineage。

## 4. clean B 实现与验证

`scripts/rsl_rl/base/train.py` 新增了作用域受限的参数：

```text
--highstep_checkpoint_load_mode full|weights_only
```

- 默认值是 `full`，不改变既有训练行为；
- `weights_only` 仅允许 resumed highstep `OnPolicyRunner`；
- 调用 `runner.load(checkpoint, load_optimizer=False)` 后把 `runner.current_learning_iteration` 重置为 `0`；
- 新环境实例自然重置 episode/curriculum 状态；
- 日志必须出现 `Highstep clean resume: restored policy weights only; optimizer is fresh ... reset ... to 0`。

该修改已通过 `py_compile`。Robust smoke 使用 16 env、`max_iterations=1`、xless、W&B offline，返回 0 并生成 checkpoint：

```text
smoke log:
  /home/lxq/Softwares/robot_lab/tmp/highstep_robust_clean_weights_smoke_20260711_221140/smoke.log
smoke checkpoint:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-12-03_clean_weights_smoke_20260711_221140/model_0.pt
task:
  RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0
```

smoke 启动部分包含已知 NVIDIA Warp/driver 兼容警告，但训练正常完成、生成 checkpoint，日志尾部没有 orchestrator 定义的致命模式。不要把这组启动警告误写成 smoke 失败，也不要在没有新证据时重做 Kit 验证。

## 5. clean B 曾因显示故障暂停，现已恢复

clean B 从完整评估选出的 `model_175800.pt` 启动：

```text
task: RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0
seed: 42
num_envs: 4096
planned iterations: 2000
initial load mode: weights_only
run_dir: /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-29-20
train log: /home/lxq/Softwares/robot_lab/tmp/highstep_robust_clean_weights_train_20260711_222857/train.log
old W&B: https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/lsoxm39v
old monitor out: /home/lxq/Softwares/robot_lab/tmp/highstep_robust_clean_weights_schema4_monitor_20260711_222857
```

暂停前日志到 iteration `427/2000`，最后可靠周期 checkpoint 为 `model_400.pt`。iteration 427 的训练指标显示环境与策略正在恢复：

```text
mean reward: 87.55
highstep_action_score: 0.8066
terrain_levels: 0.9503
support_score: 0.6718
rear_approach_center_violation_rate: 0.2141
rear_motion_center_violation_rate: 0.0993
support_floor_violation_rate: 0.0000
bad_orientation: 0.0037
```

这些只证明恢复过程在推进。它仍未越过 500 轮恢复边界，也没有做 45 场评估，不能据此判断 clean B 成败。

暂停是用户针对 AMD 显示链卡死发出的明确指令。显示故障从 18:56 已存在，早于 22:29 的 clean B，NVIDIA 计算没有 Xid；不得归因于本轮训练。目前已确认：

```text
highstep train/play/monitor: none
NVIDIA compute process: none
highstep-auto-loop.service: disabled/inactive
highstep-xray-main.service: enabled/active
```

上面代码块是停机后的历史快照。机器已用新 boot ID
`1a8ba203-7373-4c93-b2d2-3b2cea299bcc` 重启，Xorg/显示链恢复响应；clean B
已从派生 iter401 断点以 `full` 模式续训。当前 run、monitor、W&B 和对话路由见
本文第 9 节。

## 6. 恢复时最容易混淆的一点

`weights_only` 只用于从旧 full-resume lineage 的 `model_175800.pt` 创建 clean B。clean B 已经训练到 400 并形成了自己的 optimizer 状态；显示故障后的续训必须沿 clean B 使用 `full`，不能再次 `weights_only`，否则会无意义地第二次重置 optimizer。

权威恢复锚点：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-29-20/model_400_resume_iter401.pt
SHA-256: 042429662c2e519647f744c55651578feef06c6b62c06ca5e036445cdd140ce4
checkpoint iter: 401
```

它与 `model_400.pt` 的模型参数和 optimizer state 完全一致，只把顶层 `iter` 从 400 修正为 401，避免 RSL runner 重跑 iteration 400。以下脚本已成功执行一次，只保留作审计，不得再次运行：

```bash
cd /home/lxq/Softwares/robot_lab
bash -n tmp/resume_highstep_after_display_reboot_20260711.sh
./tmp/resume_highstep_after_display_reboot_20260711.sh
```

脚本 SHA-256：

```text
2a18691be45864f947191cdc3c82525621c81b1c389296843798fa19822b7c99
```

脚本会使用 `--highstep_checkpoint_load_mode full --max_iterations 1599`，准确执行 `401..1999`。由于进程重启会再次初始化环境统计，恢复后也不要立即按训练 reward 下结论；第一个部署比较点仍放在 1100，距离恢复约 699 updates，已经为第二次进程恢复保留足够适应区间。

## 7. 恢复后的自动决策顺序

1. 用户已完成数据处理并明确允许恢复；当前保持唯一 GPU 训练，不再回到暂停态。
2. 恢复脚本已经建立唯一 train、唯一 schema-v4 monitor、新 W&B run 和经校验的新 handoff；不要再次运行脚本或手工并行启动另一条 A/B 分支。
3. 训练完成后比较 clean B 的 `1100/1500/1999`，执行 `3 checkpoints x 3 seeds x 5 scenarios = 45` 场。
4. 若 `evaluation_complete=false`、任一 `reset_valid=false`、场次缺失或字段非有限值，只修评估链，禁止据此加 reward 或开新训练。
5. 若完整评估达到 Robust Teacher 门槛，冻结选中 checkpoint，自动进入 Robust Student；不得继续 Teacher，也不得直接上真机。
6. 若未达到但相对 A 在 pass rate、dwell avg/worst 和 0707 对应安全指标上仍有系统改善，可依据末段斜率决定有限延长。
7. 若三个 checkpoint 平台化且 worst dwell 仍约 500，停止该 lineage，分析 action delay、phase trigger 与 post-clear 双后足推进耦合；若 policy 标记 `strategy_escalation_required=true`，禁止重复同一权重和同一训练。
8. Student 必须使用 Student task 和 parent Teacher manifest 完成自己的严格矩阵。Student 严格通过后才导出 policy、录制仿真视频并核对部署 observation/action contract。

部署指标优先级仍是：后腿关键阶段宽度/中心线、单前足承载与接触延迟、前足撞墙、roll/yaw、rear-edge dwell、完整上台和顶部保持。训练 reward 只作辅助诊断。

## 8. 新对话首轮检查清单

新对话不要凭本文的历史 PID 行动。先执行纯静态/只读检查：

```bash
cd /home/lxq/Softwares/robot_lab
sed -n '1,260p' docs/robotlab_memory_zh/library/highstep_display_hang_emergency_pause_handoff_2026-07-11.md
systemctl --user is-enabled highstep-auto-loop.service
systemctl --user is-active highstep-auto-loop.service
ps -eo pid,stat,args | rg 'train.py|play.py|highstep_centerline_guard_monitor' | rg -v 'rg '
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

暂停期预期 `disabled/inactive` 已被用户最新恢复指令覆盖。当前预期是 controller
`enabled/active`、唯一 Robust Teacher train 和唯一 schema-v4 monitor 存活。若和预期
不同，以实际进程、用户最新指令及经校验的新 handoff 为准；禁止盲目重跑旧恢复命令。

代码库信息：

```text
branch: dev_lxq_new
HEAD at pause: effd19945034e609f4f5221cf106114852f32882
working tree: dirty，包含用户与自动化累积修改，禁止 reset/checkout 覆盖
```

## 9. 当前对话恢复事实（2026-07-11 23:49 HKT）

```text
current dialogue / automation thread:
  019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c
retired automation thread:
  019f49b9-78e2-7610-972c-0834d70f279f
task / role / phase:
  RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0
  teacher_robust / teacher_robustification
run_dir:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_23-45-31
train_log:
  /home/lxq/Softwares/robot_lab/tmp/highstep_display_reboot_resume_train_20260711_234523/train.log
train unit / startup PID:
  highstep-train-resume-20260711_234523.service / 207212
monitor unit / startup PID:
  highstep-monitor-resume-20260711_234523.service / 210939
monitor out:
  /home/lxq/Softwares/robot_lab/tmp/highstep_display_reboot_resume_schema4_monitor_20260711_234523
W&B:
  https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/srxfqe9o
resume contract:
  full, iter401 checkpoint, 1599 updates, execute 401..1999
comparison / evaluation:
  model_1100.pt, model_1500.pt, model_1999.pt; 3 x 3 x 5 = 45
```

验收时训练已从 `401` 推进到 `442`，monitor 已完成第二个 120 秒心跳，live handoff
再次验证为 `ok`。控制器服务文件已改绑本对话；只重启控制器时 train 与 monitor PID
保持不变。当前根分区约剩 `22.5 GiB`，高于 `15 GiB` 自动资源处理门槛，但必须持续
监督。当前 goals 数据库中 `blocked` 条目属于已被用户覆盖的旧“等待评估后重启”目标，
orchestrator 只把它作为提示文本，不作为运行门禁；不得据此停止当前恢复训练。

本次迁移没有修改 reward、task、policy、monitor schema 或训练代码，只更新了自动化
thread 路由、实时 handoff 关联和交接记忆。
