# Highstep Automation V4 Short-Term Handoff

> **本文的 V4 协议仍有效，但以下 PID/实时描述是历史状态。** 机器已于
> `2026-07-11 23:45 HKT` 从 iter401 恢复 clean B；当前先读
> [紧急暂停与恢复交接](highstep_display_hang_emergency_pause_handoff_2026-07-11.md)和
> [clean B 对话迁移交接](highstep_dialogue_migration_clean_b_2026-07-11.md)，再校验实时
> schema-v4 handoff。不得按本文旧 PID 行动。

更新时间：2026-07-11 03:58，Asia/Hong_Kong。

用途：上下文压缩后的第一读取入口。继续操作前必须先核对本文件中的实时 PID、路径和未完成事项；实时状态变化后覆盖更新本文件，不把它当追加日志。

## 用户目标与不可偏离的现象

- 最高优先级真机证据：`/home/lxq/Videos/histep_real_robot_test_0707.mp4`。
- 问题不是站立策略切换到 highstep 的瞬态。机器人已经进入 highstep 并前进，在第一只前腿刚抬起、首次搭到高台时，两只后腿向中线内收，随后伴随单前足承载、边缘停留、roll/yaw 摇摆和后腿卡边。
- 当前目标是保留现有全自动闭环，并加入关键升级：标准 Teacher -> Robust Teacher -> Student -> 严格多 seed 评估 -> 自动导出/video。常规流程不要求用户手动干预；真实机器人上机本身仍必须有人在场。

## 2026-07-11 当前实时状态

- 训练 PID：`3399670`，不得中断。
- task：`RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0`。
- run_dir：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_01-47-52`。
- train_log：`/home/lxq/Softwares/robot_lab/tmp/highstep_postclear_bilateral_advance_train_20260711_014746/train.log`。
- W&B：`https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/h5kwms11`。
- 最终 schema-v4 monitor PID：`4167202`；out：`/home/lxq/Softwares/robot_lab/tmp/highstep_postclear_bilateral_advance_schema4_final_monitor_20260711_035607`。进程运行该目录内不可变脚本快照，不再读取共享脚本。
- systemd controller MainPID：`4173496`；实际 orchestrator PID：`4173498`；unit：`highstep-auto-loop.service`。
- orchestrator thread：`019f49b9-78e2-7610-972c-0834d70f279f`。
- 03:58 验收：训练到 `171676/171997`，ETA 约 24 分钟；W&B core PID `3401360`；5h 额度 `47%`，阈值 `20%`；磁盘可用约 `33.9 GiB`。

## 已完成的 V4 改动

1. 新增 `tmp/highstep_automation_policy.json` 和 `tmp/highstep_automation_policy.py`：
   - 精确区分 standard/robust Teacher 与 standard/robust Student；
   - 记录跨轮 `evaluation_rounds.jsonl`；
   - 连续两轮无实质进步时触发 Codex strategy escalation；
   - Student 必须相对 parent Teacher 通过严格保持率和安全阈值。
2. `tmp/highstep_centerline_guard_monitor_20260709.sh` 升级 schema-v4：
   - 参数为 `train_pid run_dir train_log out_dir task role [parent_teacher_manifest]`；
   - 动态 task，3 checkpoints x 3 seeds x 5 scenarios；
   - 四个扰动场景增加 1 control-step action delay；
   - 评估结束调用 policy helper finalize，写 manifest 和跨轮 ledger。
3. 新增 robust 任务：
   - `RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0`；
   - `RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0`；
   - 包含 0-1 step 动作延迟、每 episode 持久关节编码器偏置、post-clear 后腿推进停滞惩罚。
4. `tmp/highstep_goal_orchestrator.py` 已在磁盘升级：
   - 校验 task/role/schema-v4/parent Teacher manifest；
   - 使用 policy context 和跨轮无进展判断；
   - 要求 Teacher -> Robust Teacher -> Student -> strict eval -> export/video；
   - 机器可验证 handoff，禁止 Codex rc=0 但没有实际交接产物。
5. 已通过：目标 Python 文件 `py_compile`、monitor `bash -n`、目标 diff `git diff --check`；policy synthetic dry-run 已验证升级路径和两轮无进展触发。

## 评估基线与本轮依据

- 上一候选 `model_169997.pt`：15/15 完整上台且首步/中心线检查通过。
- 仍有 12/15 rear-edge dwell 不通过（309-551 steps），另有 2 场 roll-rate 问题，因此不能直接认定真机安全。
- 当前训练中 rear advance 信号约由前 100 步 `0.149` 升至末 100 步 `0.273`，recovery signal `0.064 -> 0.159`，approach center violation `0.453 -> 0.366`，motion center violation `0.303 -> 0.223`。方向有效，应让本轮完整结束并评估。
- ledger 已种入两轮历史：`model_167998` streak 0，`model_169997` streak 1；本轮若仍无实质进步，将自动进入 streak 2 并要求 Codex 做结构性策略升级。

## 当前等待事件

1. 当前 Teacher refine 继续训练到 `171997`；不要中断，也不在 GPU 被占用时启动 Robust smoke。
2. PID `4167202` 在训练退出后自动评估末段 3 checkpoints x 3 seeds x 5 scenarios，共 45 场，并写 schema-v4 manifest/decision/ledger。
3. 评估进程退出后，`highstep-auto-loop.service` 内的 orchestrator（当前 PID `4173498`，允许 systemd 自动更换）在下一次监督周期调用 Codex；若本轮无实质进步，ledger streak 将由 1 到 2，强制 strategy escalation；若达普通 Teacher 门槛，则进入 Robust Teacher，而不是直接上真机或直接宣告完成。
4. 首次长时 Robust 训练前，Codex 提示已强制要求在 GPU 空闲时先执行小规模、`max_iterations=1` task smoke；通过后才允许后台长训并挂新 schema-v4 monitor。

## 已完成的在线安全切换

- 旧四参数 monitor PID `3403319` 及进程组已停止，训练未受影响。
- `tmp/highstep_current_postfinish_monitor` 已原子指向最终不可变 schema-v4 monitor 目录；以后每次 handoff 采用时由 orchestrator 自动修复。
- `handoff.json` 已原子升级，包含 task、role、phase、schema=4 和新 monitor PID，并经新版校验器验证为 `ok`。
- 旧 `nohup` orchestrator/inhibitor（含中间 PID `3974044/3974156`）均已停止；权威控制入口改为启用且带 linger 的 `highstep-auto-loop.service`，不要依赖固定 orchestrator PID。
- 目标 Python `py_compile`、monitor `bash -n`、目标 diff check、orchestrator dry-run 和 dashboard one-shot 均通过。

## 2026-07-11 可靠性审计与修复

用户要求批判性检查，不能等自动化静默终止后再处理。审计确认并修复以下真实单点：

1. controller 原先是无上级守护的 `nohup` 进程，退出后 inhibitor 也退出：
   - 已安装并启用 `~/.config/systemd/user/highstep-auto-loop.service`；
   - `Restart=always`、45 分钟 watchdog、`KillMode=control-group`、5 分钟检查；
   - `loginctl` linger 已启用，终端关闭、注销或 controller 崩溃后仍能重启；
   - 实际 SIGKILL 故障注入通过，systemd 自动生成新 PID，训练和 monitor 未中断。
2. future Codex 子进程若继承 controller cgroup，controller 重启会误杀它启动的训练：
   - `codex resume` 现在自动放入独立 `highstep-codex-*.scope`；
   - handoff 拒绝 controller cgroup 内的 train/monitor；
   - sibling scope 在 controller restart 后继续存活的故障注入已通过。
3. 活跃训练缺 monitor、monitor 心跳卡死、train PID 活着但日志不推进、W&B core 消失过去会静默等待：
   - 现在均触发 `control_plane_incomplete`；
   - train log 与 monitor heartbeat 阈值均 600 秒，evaluation heartbeat 阈值 1500 秒；
   - 进程识别改为 `/proc/<pid>/cmdline` 独立 argv token，Codex prompt 文本不会再被误识别为训练。
4. `manual_review_required` / `no_action_complete` 过去可能永久 hold：
   - `no_action_complete` 只允许活跃训练的 periodic review；
   - manual review 只允许机器外部 blocker 且必须有 evidence，并每 600 秒重试；
   - 只有完成仿真候选 `simulation_candidate_ready` 才永久 hold 等真实上机。
5. monitor/评估可靠性：
   - 每轮启动即原子复制并 exec 不可变 monitor 快照；
   - task 必须显式且在四个支持 task 白名单，task/role/Student parent 必须一致；
   - 严格验证 3 checkpoints x 3 seeds x 5 scenarios 唯一矩阵、scenario 参数、action delay、关键字段和有限数值；
   - 第一场基础设施失败即 fail-fast，不再浪费余下 44 场；play 使用独立进程组，monitor 退出会清理 GPU child；
   - 同一 run 重评会原子覆盖 ledger 旧记录，不再保留过期结论；current monitor 软链接由 orchestrator 自动修复。
6. Robust 任务提前验证：
   - Kit 级配置加载真实通过 Robust Teacher 与 Robust Student；Gym 注册、action class、0-1 delay、persistent encoder bias、stall reward 均已验证；
   - Robust `training_started` handoff 强制要求真实 smoke task/rc/log/checkpoint 证据，缺失或 fatal 直接拒绝。
7. 磁盘保护：
   - 可用空间低于 15 GiB 时触发 Codex 资源处理；
   - 即使额度不足，可用空间低于 5 GiB 也会只执行一次有日志依据的训练进程组 SIGINT；
   - monitor 在低于 5 GiB 时拒绝启动 45 场评估，避免额度暂停期间写满磁盘。

验证产物：`tmp/test_highstep_automation_reliability.py` 共 12 个测试通过；Robust Kit 结果为 `/tmp/highstep_robust_cfg_validation_result.json`；备份目录为 `tmp/highstep_reliability_audit_backup_20260711_033601`。

## 禁止事项

- 不停止当前训练，不回退用户已有脏工作树，不删除无关修改。
- 不直接跳过完整评估去启动 Robust 或 Student。
- 不把真实机器人视频中的问题重新解释成策略切换问题。
- 不把物理上机测试描述为可由软件全自动完成。

## 2026-07-11 04:58 网络与评估恢复交接

- Teacher refine 已正常结束，最终 checkpoint 为 `model_171996.pt`；训练最终 mean reward 为 `107.52`，末 20 次均值为 `98.75`。
- 首轮 schema-v4 评估在 `left_offset + action_delay=1` 暴露 `DelayBuffer` 的 `torch.int32/torch.int64` dtype 不匹配。已让随机 delay 直接使用 buffer 自身 dtype，并用原失败场景实测通过。
- monitor 对 Codex 安装目录中的 `rg` 存在隐式运行时依赖，也已移除。当前完整重评由独立 systemd unit `highstep-eval-171996-20260711_0447.service` 执行，monitor PID `325283`，输出目录为 `tmp/highstep_eval_runtime_deps_repair_postfinish_monitor_20260711_0447`；04:58 已完成 `19/45`，无缺失 JSON 或异常退出。
- `highstep-auto-loop.service` 当前为权威 orchestrator，已成功采纳 `evaluation_started` handoff；Codex 子任务使用独立 scope、强制 `ALL_PROXY=socks5h://127.0.0.1:10808`、代理端口预检、CPU/IO 低权重，网络卡死上限由 1800 秒缩短为 600 秒。
- 主代理改为启用的 `highstep-xray-main.service`，`Restart=always`、`RestartSec=1`、启动前验证配置；orchestrator 对它有 `Requires/After` 硬依赖。当前节点为日本 NTT2，出口位于 Tokyo，台湾 HiNet 候选虽有旧 `46 ms` 记录，但实测对 ChatGPT/OpenAI/IP 查询全部超时，禁止使用。

两次网络现象必须严格区分：

1. 04:27 的后台 Codex 超时是 systemd 服务没有继承 `ALL_PROXY`，进程绕过本地 Xray 直接外连；不是训练、GPU 或节点本身崩溃。该问题已由显式代理注入和服务依赖修复。
2. 后续人工切节点时，旧 Xray 被先停止、新 Xray 尚未接管 `10808`，造成主代理短暂空窗和当前远程对话断连。这是操作顺序错误。以后活动 Codex 对话、W&B 同步或自动评估期间禁止切换主代理；候选只能在独立端口测试。确需切换时必须预告会重连、原子写配置、验证 systemd unit 可启动，并保留可立即回滚的旧配置。

资源规则：任何导入 AppLauncher/Isaac/Kit 的“配置验证”也算 GPU 工作负载；train/play/eval 活跃时不得并发启动。已为 robust 配置验证加入进程、GPU 和可用内存守卫。当前 swap 仍接近满载，但 MemAvailable 约 17 GiB；不要在完整评估结束前启动第二个 Kit。

## 部署 Policy 最高目标（用户 2026-07-11 明确）

- 所有训练、评估、Robust、Student、导出和视频验证的最高目标，是选出一个可冻结并部署到真机的固定 policy，使真机尽量复现仿真中的上高台动作与能力。
- checkpoint 邻域是否稳定、训练曲线是否单调、最后 checkpoint 是否最好，都只属于训练诊断；它们不能取代固定 checkpoint 的结构化部署验收。后续训练退化不会污染已经保存的更优 checkpoint。
- 部署排序优先看 0707 真机故障对应指标：第一步后腿宽度/中心线、前足单侧承载与接触延迟、前足撞墙、roll/yaw、后腿卡边停留、完整上台与台面保持；训练 reward 只作为辅助证据。
- 普通 Teacher 达标只允许进入 Robust Teacher；Robust Teacher 达标后进入只使用真机可用观测的 Robust Student；Student 严格达标后才导出并做仿真视频与接口一致性验证。普通 Teacher 的偶然峰值绝不等于可上真机。
- 不把“相邻 checkpoint 都稳定”新增为硬部署门槛，除非用户以后明确改变目标；它可用于判断优化是否系统性漂移以及何时 strategy escalation。

## 真机盲走约束与待取日志（用户 2026-07-11 明确）

- 真机最终是盲走上高台，部署 policy 不获得任何特权地形信息。最终必须部署只使用真机机载可得本体感知的 Robust Student；Teacher 的高度扫描、地形几何、仿真接触真值和未来相位只允许作为训练监督或 critic 信息，禁止进入导出 actor 的 observation contract。
- 普通/Robust Teacher 的成功只能证明可教能力，不能证明盲走可部署。Student 必须单独使用 Student task 完成严格矩阵、导出一致性和视频验证。
- 盲走策略不能依赖仿真评估固定的 `front_step_eval_edge_gap=0.55 m` 形成开环计时。部署审计必须覆盖随机接近距离、横向偏置、偏航、速度、控制延迟和接触时刻，验证其依靠本体感知/接触反馈维持第一步后腿支撑并完成恢复。
- 0707 真机测试的机载 log 仍在机器人电脑上，用户计划次日上班后拷贝。收到后优先对齐视频与 command/observation/action/joint/IMU/control-loop 时间戳，辨别 policy、观测延迟、动作缩放、执行器响应或接口不一致；在日志到手前不臆造真实随机化范围。
