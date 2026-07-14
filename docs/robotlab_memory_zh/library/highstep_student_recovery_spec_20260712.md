# Highstep Student 恢复与自动化规范 v1.11

- 状态：用户已批准
- 批准日期：2026-07-12
- v1.1 修订批准日期：2026-07-12
- v1.1.1 修订批准日期：2026-07-12
- v1.2 预先批准日期：2026-07-13（用户明确授权先形成正式草案并按草案自主推进，醒后复核）
- v1.3 批准日期：2026-07-13（用户明确批准修改现有 spec，并授权新训练和自动化流程直接推进）
- v1.5 批准日期：2026-07-13（用户撤销 B500 真机候选资格，要求完整复原并保留 0707 Teacher→Student 成功蒸馏与防后期退化机制）
- v1.5.1 批准日期：2026-07-13（用户重申长期有效的单侧工作域验收要求；纠正将特定 B500 人工否决错误扩大为废止整个 directional 候选机制的过度解释）
- v1.5.2 批准日期：2026-07-13（用户确认新 Teacher、0707 Teacher→Student→真机工作流和蒸馏参考均已构成可靠前提；要求所有评分与附加项服从“尽快交付至少一个方向可用于真机测试的 Student”）
- v1.6 批准日期：2026-07-13（用户依据 E300→E500 行为退化，停止未经 0707 验证的 post-prior warmup 路线，要求以新 Teacher 严格重跑 0707 Stage-2 蒸馏）
- v1.6.1 批准日期：2026-07-13（仅增加最小变更纪律，防止再次叠加未经证明的训练改造；不改变 v1.6 的目标、路线、参数、门禁或授权）
- v1.7 批准日期：2026-07-14（0707 原始 agent.yaml、运行时代码快照和 TensorBoard 闭环纠错；恢复唯一 historical_0707_exact 路线）
- v1.7.1 批准日期：2026-07-14（仅补齐 authority 与运行基础设施原子迁移；训练语义、Teacher、optimizer、保存点和行为门禁完全不变）
- v1.8 批准日期：2026-07-14（纠正 v1.7.1 未复现 0707 Student 环境的控制变量错误；批准“0707 环境 bootstrap → 同一 Student 完整鲁棒环境续训”双阶段课程）
- v1.9 批准日期：2026-07-14（v1.8 E1400 按预注册失败后封顶；先做同 checkpoint oracle prior-delta A/B，只有行为恢复才允许训练独立 4 维 blind prior residual head）
- v1.10 批准日期：2026-07-15（用户指出 E1400 与 terrain/support/score/warmup 多个切换边界重合，批准从同一 E1400 完整 checkpoint 做受限的 estimator-only 与历史 post-warmup box-row continuation A/B；两支最多各增加 300 updates，并在每 100 updates 做同规格评估）
- v1.11 批准日期：2026-07-15（暂停 v1.10 continuation，不启动训练；优先执行旧 Teacher model_151399 与新 Teacher model_172300 的零训练、同环境、同物理快照鲁棒性 A/B）
- 最高目标：得到能够完成约 35 cm 高台动作、可用于下一次真机实验的 Student policy
- 适用范围：Highstep Student 蒸馏、仿真行为评估、候选导出与视频

### v1.11：新旧 Teacher 零训练鲁棒性 A/B

本节是当前最高执行依据。v1.10 workflow 必须保持 `user_paused_pending_teacher_comparison`，其 checkpoint、optimizer、W&B、state、preregistration 和 handoff 不修改、不覆盖、不启动训练。v1.10 及以下章节均为 `historical-only`，除本节明确引用的固定行为合同外不再拥有启动 authority。

#### 唯一变量与只读约束

1. 唯一实验变量是 Teacher checkpoint：A=`2026-07-04_01-45-21/model_151399.pt`，SHA256=`d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d`；B=`2026-07-11_11-22-23/model_172300.pt`，SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`。
2. 两者必须使用同一个 Robust Teacher task、同一套当前 real gains、启用且运行时 scale 相同的 Teacher action prior、delay=`0`、固定命令 `[0.45,0,0]`、`box` level 9 平台、front-step reset、seeds=`11,22,33` 和逐场完全相同的初始物理 snapshot。普通 reset/force/observation 随机事件全部关闭。
3. 只允许预注册的后腿初始内收与单次机身横向速度冲量。禁止训练、`runner.learn`、`backward`、`optimizer.step`、checkpoint 写入和任何模型参数修改。每场必须记录 checkpoint 与加载 tensor 的前后 SHA 并证明不变。
4. 每个场景先由 Teacher A 生成并冻结初始 root pose/velocity、joint pose/velocity、terrain/reset context 与干预合同 snapshot；Teacher B 必须只读恢复同一 snapshot 并逐字段验证。不能以相同 seed 代替 snapshot 同一性证明。

#### 冻结矩阵与指标

1. 无干预 nominal：三个 seeds。
2. 后腿内收矩阵：初始 rear width=`0.34/0.30/0.26/0.22 m`，每级包含 `symmetric/RL-only/RR-only`，三个 seeds。只允许通过后 hip 初始关节状态机械实现目标宽度，误差不得超过 `0.002 m`。
3. 横向冲量矩阵：`approach/front-support/first-rear` 三阶段，左右两个方向，`delta-v=0.10/0.20/0.30 m/s`，三个 seeds；每场只施加一次 root lateral velocity increment。
4. 组合压力矩阵固定九场：rear width=`0.26 m`，三个 seeds；`symmetric` 绑定 seed 11/33 为左、seed 22 为右，`RL-only` 固定左向 `0.20 m/s`，`RR-only` 固定右向 `0.20 m/s`，冲量均在 first-rear 触发。该绑定只为得到预注册的 3 模式×3 seed 九场，不得看结果后换向。
5. 全部场景记录 `full_climb`、`rear_hold`、逐帧 rear width、rear min-abs-y、中心线穿越、恢复时间、roll/yaw、冲量实际阶段/方向/幅度和跌倒。恢复定义和能力等级判定必须写入不可修改 preregistration，结果出现后不得修改。

#### 预先固定的明显改善门禁

1. `nominal` 不退化：Teacher B 的 valid、full climb、rear hold 和 recovery 均不得低于 Teacher A，且跌倒数和中心线穿越数不得增加。
2. 后腿内收能力以通过的最难宽度等级定义；同一级 9 场必须 `valid=9/9` 且 `full/rear_hold/recovery>=7/9`。Teacher B 必须比 A 至少提高一个 `0.04 m` 等级。
3. 抗扰动能力以通过的最大 delta-v 等级定义；同一级 18 场必须全部 valid 且 `full/rear_hold/recovery>=14/18`。Teacher B 必须比 A 至少提高一个 `0.10 m/s` 等级。
4. Teacher B 的组合压力测试必须 `valid=9/9` 且 `full/rear_hold/recovery>=7/9`。
5. 以上四项必须同时满足才能写 `new_teacher_robustness_clearly_improved`；否则写 `new_teacher_robustness_not_clearly_improved`。阈值、矩阵、恢复定义和比较方式禁止看结果后放宽。

#### authority 与终态

1. 本轮建立独立 workflow/state/lock/heartbeat/handoff、只读 preregistration、代码 rebinding 和 disabled-on-terminal systemd user service。看板必须显式指向本轮 workflow，禁止按日志 mtime 回选 v1.10。
2. supervisor 必须先完成 Teacher A 全矩阵，再以同 snapshot 完成 Teacher B；只补缺失或基础设施损坏场次，禁止训练或用行为失败触发重跑。
3. 完成后只汇报新旧 Teacher 对照并停止。不得启动 Student 或 Teacher 训练，不得自动恢复 v1.10 continuation。

### v1.10：E1400 受限延续 A/B

本节从 v1.11 起为 `historical-only`。v1.9 的 oracle prior-delta 未恢复、residual 训练被禁止以及 v1.8 的全部历史结果保持事实有效。v1.10 continuation 保持用户暂停，不得自动恢复；旧 checkpoint、optimizer、日志、W&B、评估、state、preregistration 和 handoff 不得删除、覆盖或改判。

#### 固定起点与唯一控制变量

1. 两支都必须从 v1.8 E1400 完整 checkpoint 独立恢复：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_environment_curriculum_v18_Student/2026-07-14_21-03-31_v18_stage_a_E1400_20260714_210326/model_1394.pt`，SHA256=`92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9`。必须完整恢复 Student、独立 Teacher 绑定、原 Adam、effective update、schedule 和 runtime state；两支不得相互恢复。
2. 唯一 Teacher 继续固定为 `model_172300.pt`，SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`。
3. 两支唯一训练语义差异是 `post_warmup_box_row_adaptation`：
   - A=`estimator_only_extension`：将 actor/box-row warmup 边界固定延长到 E1700，使 E1400→E1700 期间只有 estimator 更新，四个 box action rows/bias 保持不变；
   - B=`historical_post_warmup_box_adaptation`：保持原 historical 0707 边界 E1400，从恢复后的第一个新 update 起允许四个 box action rows/bias 以原 `1e-5` 学习率有界适配，同时 estimator 继续以原 `1e-3` 学习率训练。
4. 除第 3 条外，Teacher、Student 起点、环境 profile、pre-prior 主 target、phase scale=`2.0`、rear-box scale=`1.5`、prior-box loss、其它 loss、VAE epochs、optimizer state、schedule、reward、DR、history、网络、前 12 个 action rows、action contract、`action_scale`、`joint_pos.clip`、default pose、部署输入和真机 gains 全部相同且不可修改。PPO 永久关闭。

#### 训练预算、评估与停止

1. 两支绝对上限均为 global E1700，即各自最多从 E1400 增加 300 effective updates；匹配保存点固定为 E1500、E1600、E1700。不得在看到结果后追加 E1800 或更晚点。
2. 每个匹配点必须先完成 A、B 两支的候选自主同规格 core9，再作比较；不得用 loss、same-state 或单场结果替代行为评估。保存全部中间 checkpoint，禁止再次只选择单个课程边界点作结论。
3. Stage-A 通过线继续固定为 `valid=9/9`、`full_climb>=7/9`、`rear_hold>=7/9` 且无经证据确认的真机破坏风险。若任一支在某个匹配点通过，必须等另一支同点评估完成后停止新增 update，并按 `min(full_climb,rear_hold)`、两者总和、`front_top_support`、`no_severe_inward` 的预注册顺序选择；不得因单项 loss 更低改选。
4. 若两支到 E1700 都未通过，状态写 `continuation_ab_stopped_by_gate` 并停止，不得自动扩大训练预算。若通过，状态写 `continuation_branch_selected_pending_stage_b_authority`；本节不自动启动 Stage B，也不自动部署或发布真机候选。
5. 每支每个保存阶段使用独立 W&B run，group 绑定同一 v1.10 workflow；W&B、评估和网络故障属于基础设施故障，不得改判行为失败。supervisor 必须记录 PID、lock、state、heartbeat、checkpoint/optimizer scope 和完整 handoff，禁止静默停止。

### v1.9：oracle prior-delta 因果验证与条件式 blind residual 路线

本节从 v1.10 起为 `historical-only`。其 oracle 未恢复和 residual 禁止事实继续保留；v1.8 及以下章节同样只作历史证据。旧 checkpoint、optimizer、W&B、日志、评估、state/handoff 和门禁事实必须完整保留，不删除、不覆盖、不改判。

#### v1.8 封顶事实与固定模型

1. v1.8 已按预注册完成 E1400 并以 `stopped_by_gate` 正常停止；禁止继续增加 pre-prior estimator iteration，禁止进入 v1.8 Stage B，禁止从 v1.8 optimizer 恢复任何后续路线。
2. v1.8 E1400 的候选自主 core9 固定记录为：`valid=9/9`、`full_climb=5/9`、`rear_hold=5/9`、`front_top_support=8/9`、`no_severe_inward=9/9`。按 v1.8 已冻结选择规则，最佳 Student 固定为 `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_environment_curriculum_v18_Student/2026-07-14_21-03-31_v18_stage_a_E1400_20260714_210326/model_1394.pt`，SHA256=`92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9`。
3. 唯一 Teacher 继续固定为 `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt`，SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`；禁止重训、替换或修改 Teacher。

#### 第一阶段：同 checkpoint oracle prior-delta A/B（只读、零训练）

1. 在任何新训练前，必须建立独立、只读、不可事后修改的 A/B preregistration。A 与 B 必须使用同一个 E1400 checkpoint、同一个任务、core9 九场矩阵、seeds=`11,22,33`、nominal/left-offset/right-offset reset、real gains、delay=`0`、随机事件关闭、schedule/runtime snapshot 和评估代码。
2. A=`control`：完全执行原 Student 16 维动作，虽然同帧计算 Teacher prior delta，但注入量严格为零。B=`oracle_prior_delta`：仅把同一状态下的 `Teacher post-prior - Teacher pre-prior` 加到 Student 最后 4 个 box action 维度；Student 前 12 维动作逐帧必须与 A 的算法输出完全一致。不得用 Teacher post-prior 整段替换 Student，不得替换非 box 动作，不得使用第二个 Teacher 环境，不得更新任何参数。
3. 每场必须绑定 Student/Teacher/checkpoint/code/spec/preregistration SHA，记录每帧 Teacher pre/post-prior、4 维 delta、注入前后动作和首 12 维零差证明；A/B 前后 Student checkpoint、Teacher checkpoint 和加载模型 tensor SHA 必须不变。
4. oracle 恢复门禁在观察结果前固定为：B `valid=9/9`、`full_climb>=7/9`、`rear_hold>=7/9`、`no_severe_inward>=8/9`；A 未通过同一行为门禁；且 B 的 `full_climb` 与 `rear_hold` 都严格高于 A。`front_top_support` 完整记录但不单独否决整机上台恢复。阈值不得看结果后放宽。
5. 若第 4 条不满足，状态写 `oracle_prior_delta_not_recovered`，停止且禁止启动 residual 训练。若满足，状态写 `oracle_prior_delta_recovery_verified`，才允许进入第二阶段。

#### 第二阶段：条件式 4 维 blind prior residual head

1. 只允许从第 2 条固定的最佳 E1400 Student 以 `weights-only` 加载；旧 optimizer 永久丢弃，创建只拥有 residual head 参数的 fresh optimizer。禁止继承或继续 v1.8 estimator optimizer。
2. 在原 Student 之外增加唯一一个 4 维 blind residual head：输入只能是原有 570 维 Student observation history，输出只能是 4 维 box prior residual；权重和 bias 必须全零初始化。最终 policy 输入维度仍为 570，最终部署动作仍为 16 维，前 12 维来自冻结 Student，最后 4 维为冻结 Student box 动作加 residual。
3. 监督 target 只能是同一状态下精确的 `Teacher post-prior - Teacher pre-prior` 最后 4 维。Teacher actor、Teacher privileged encoder、Student actor、Student estimator/latent、critic、privileged encoder 和其它所有参数永久冻结；optimizer 中不得出现 residual head 之外的参数。
4. 禁止同时修改 Teacher、reward、domain randomization、570 维 history/顺序/scale/normalization、全 actor、网络其它部分、`action_scale`、`joint_pos.clip`、default pose、部署输入、真机 gains 或 action contract。该路线唯一训练变量是零初始化的 4 维 blind prior residual head。
5. residual 训练前必须另写不可事后修改的 optimizer/LR/update-budget/save-point/行为门禁 preregistration，并以 static、tensor binding 和 1--5 update smoke 证明：仅 residual head 获得梯度和参数更新；冻结 Student/Teacher tensor SHA 不变；target 确为 4 维 Teacher prior delta；最终输出仍为 16 维。未完成这些证据不得启动训练。
6. 第一个恢复第 1 阶段行为门禁且无明确真机破坏风险的 residual checkpoint 必须停止新增 update，随后再按既有 directional 目标评估并进入用户视觉复核；禁止自动真机部署。

#### v1.9 authority 与自动化

1. oracle A/B 必须使用独立 workflow/state/lock/heartbeat/handoff/service；v1.8 service 保留 unit 和历史结果但必须 disabled。看板必须显式指向 v1.9 workflow，不得按旧日志 mtime 选回 v1.8。
2. supervisor 只可先完成 A，再完成 B，并按冻结门禁自动决策。评估/W&B/网络属于基础设施故障，不得改判行为失败；规范门禁终态不得自动绕过。
3. 只有 oracle 恢复后，才允许在安全 authority 边界迁移到第二阶段 residual workflow。任何 residual 训练都必须使用独立 W&B run，并完整记录起点 Student、Teacher、spec/preregistration、零初始化、optimizer scope、输出 checkpoint 和门禁。

### v1.8：0707 Student 环境 bootstrap → 当前完整鲁棒环境续训

本节从 v1.9 起为 `historical-only`。v1.7.1 及以下章节同样只保留 checkpoint、optimizer、W&B、诊断、门禁和失败事实，不再拥有启动、恢复或训练语义 authority。v1.7.1 E700 的 `latent_observability_isolation_required`/`stopped_by_gate` 事实必须完整保留，不删除、不覆盖、不改判；但该结果只能证明“新 Teacher + 从 update 0 使用 v1.7.1 完整 Student 环境”未复现 0707，不能证明 0707 基础蒸馏失败。

#### v1.8 唯一目标、Teacher 与单一控制变量

1. 唯一目标仍是尽快得到至少在一个预标定单侧工作域内稳定完成整机上高台、可供用户观看并用于有人保护真机实验的 Student。
2. 唯一 Teacher 固定为 `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt`，SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`。该 Teacher 已接受随机化、外力扰动、command/terrain curriculum 和阶段 schedule 训练；禁止重训 Standard/Robust Teacher。
3. 相对 v1.7.1，唯一训练机制变量为 `student_environment_schedule`：从 `robust_from_update_zero` 改为 `exact_0707_bootstrap_then_current_robust`。
4. Student 必须从 `model_172300` fresh、weights-only 初始化；独立 frozen Teacher、PPO 永久关闭、warmup=`1400`、主 action target=`teacher_pre_prior`、`phase_scale=2.0`、`rear_box_scale=1.5`、warmup prior-box loss=`0`、loss/LR/VAE epochs/冻结范围/网络/reward/action contract/`action_scale`/`joint_pos.clip`/default pose/部署输入/真机 gains 全部保持不变。禁止从 v1.7.1 E700 Student 或 optimizer 恢复。

#### v1.8 冻结环境 profile 与训练前验证

1. Stage A Student 环境必须由 0707 deployed Student 保存快照机械复现，不允许凭印象逐项关闭随机化：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46/params/env.yaml`，SHA256=`f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636`。
2. Stage B 当前完整鲁棒 Student 环境固定为 v1.7.1 E700 保存快照：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_historical_0707_exact_new_teacher_Student/2026-07-14_05-34-54_historical_0707_exact_E700_20260714_053449/params/env.yaml`，SHA256=`61d70655405a49ad8fd72377aed3e193b0d8435898ca1fd39316d08f1989a729`。
3. Teacher bootstrap profile 同时绑定 0707 Teacher 保存快照：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/params/env.yaml`，SHA256=`25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7`。Teacher 验证必须保留 `PhasedHighstepBoxBiasJointPositionAction` action prior；禁止把 Teacher checkpoint 塞进 plain Student action task。
4. 每个 profile 必须在构造环境前校验源文件 SHA；运行时必须把保存快照投影到 env cfg、禁用快照中不存在的 manager terms，并逐字段证明 observation/action/关节顺序/scale/clip 与快照一致。profile manifest 及运行时 binding 必须只读保存。
5. 使用 `model_172300` 在 Teacher bootstrap profile 做候选自身驱动的最小行为 preflight；若 Teacher 自身不能完成上台，状态写 `teacher_bootstrap_environment_incompatible` 并停止，不启动 Student。
6. code-to-spec、static、Student/Teacher tensor binding、环境合同及 1--5 effective update smoke 全部通过后才允许启动 Stage A。smoke 必须证明 Student fresh、Teacher 独立、actor warmup 冻结、PPO关闭、1400/2.0/1.5/pre-prior/box-loss=0 生效、optimizer/full checkpoint 可恢复；smoke checkpoint 永久丢弃。

#### v1.8 Stage A：0707 环境基础动作 bootstrap

1. 使用冻结的 0707 Student profile；保存点固定为 global effective updates `100→300→500→700→900→1400`。
2. 每个保存点必须使用候选 Student 自己驱动的 rollout；same-state 只作诊断，不能证明上台。
3. Stage A 固定通过门禁为：`valid=9/9`、`full_climb>=7/9`、`rear_hold>=7/9`，且没有经证据确认的真机破坏风险。该阶段不是最终鲁棒候选，禁止直接导出真机结论。
4. 第一个通过点立即停止 Stage A，保存包含 Student、独立 Teacher 绑定、完整 optimizer、effective update、schedule/runtime state 的 full checkpoint，并进入 Stage B；不得为寻找更高 Stage A 分数继续增加 iteration。
5. 到 E1400 仍未通过则状态写 `stage_a_stopped_by_gate` 并停止整条 v1.8 路线；禁止切换完整鲁棒环境继续“补回来”。

#### v1.8 Stage B：同一 Student 与 optimizer 的完整鲁棒环境续训

1. 必须从 Stage A 第一个通过 checkpoint 以 full mode 精确恢复同一 Student、Teacher binding、optimizer、effective update、warmup 与 schedule/runtime state；只允许把 Student environment profile 从 Stage A SHA 切换为 Stage B SHA。
2. 切换前必须写只读 zero-other-semantics diff manifest，证明 Teacher、target、loss、LR、VAE epochs、冻结范围、optimizer state、effective update、network、reward 与 action contract 未变化；不得重置 update/warmup，不得 fresh 初始化。
3. Stage B 保存点按切换点的固定相对增量 `+100,+300,+500,+900,+1400,+1800,+2400` 执行，只保留 global effective updates 小于等于 `2500` 的点；若最后一个合法相对点小于 2500，追加固定 global E2500。global E2500 是绝对上限，禁止看到结果后追加。
4. 每个保存点执行候选自主 probe/core9/directional。任一预标定单侧达到：`valid=9/9`、`full_climb>=8/9`、`rear_hold>=8/9`、对应 half-offset `full/rear_hold=3/3`，且无明确真机破坏风险，即立即停止新增 iteration。
5. 通过后生成机器人全程清晰可见的 Teacher/Student 同状态成对视频与俯视摆位卡，状态只能写 `student_directional_candidate_pending_user_visual_review`。未经用户观看接受不得称真机候选；禁止自动真机部署。

#### v1.8 authority、W&B 与自动恢复

1. 必须建立独立 workflow/state/lock/heartbeat/handoff、双阶段不可修改 preregistration 与显式 dashboard declaration；旧 v1.7.1 service 必须 disabled，旧结果保持只读。
2. Stage A 与 Stage B 的每个训练保存阶段必须使用独立 W&B run，`group` 绑定同一 v1.8 workflow；config/summary 必须记录 spec/prereg/profile SHA、起点 checkpoint、Teacher、冻结范围、optimizer、预算、输出 checkpoint、行为门禁和 manifest。远端同步未完成前禁止进入下一训练阶段。
3. supervisor 必须区分训练机制、W&B、评估、视频和外部故障；基础设施故障不得改判行为失败，并须在规范范围内自动修复/重试。只有无法自动恢复的外部故障或等待用户视觉复核时才停止。
4. heartbeat 至少每 60 秒记录 supervisor PID、active child PID、phase/status、checkpoint、effective updates、environment stage/profile SHA 和故障分类。每个 child 启动必须写 PID/PPID/process-group/state/heartbeat 见证。
5. 训练启动后必须核验 supervisor PID、真实 train PID、lock、连续 heartbeat、effective updates 增长、W&B run 与实际 environment profile SHA；禁止静默停止。
6. 未经用户再次确认，禁止改变两套冻结 Student profile、Teacher profile、双阶段保存点、门禁或加入其他训练变量。

### v1.7.1：仅基础设施 amendment 与整体 authority rebinding

本节是当前最高执行依据。它只修订 v1.7 的 authority、服务、看板、heartbeat、W&B/评估恢复、PID 见证与视频可见性基础设施；**不得改变** v1.7 已冻结的 Teacher、Student 初始化、warmup=`1400`、pre-prior 主 target、`phase_scale=2.0`、`rear_box_scale=1.5`、warmup prior-box loss=`0`、optimizer/LR、冻结范围、schedule、保存点、E700 比较、E1400 absolute cap 或行为/directional 门禁。

1. v1.7→v1.7.1 必须在无 active child 的安全阶段边界进行原子迁移。旧 spec、旧 preregistration、已完成 checkpoint、optimizer、W&B、日志及评估证据保持只读；通过单独 amendment/rebinding manifest 绑定旧/新 SHA 和唯一恢复 checkpoint。禁止仅替换 spec SHA、修改运行中 checkpoint，或从旧 optimizer 之外重新开始。
2. 当前唯一 service、state、lock、heartbeat、handoff 与 dashboard declaration 必须共同指向 `highstep_historical_0707_exact_new_teacher_20260714`。所有被 v1.7.1 取代的旧训练/评估 supervisor service 必须保持 `disabled`/`not-found` 且无进程；不得修旧 spec SHA 使旧路线复活。W&B 网络代理等共享只读基础设施不属于旧训练路线。
3. `tmp/highstep_dashboard_active_workflow.json` 必须显式绑定当前 workflow、state、spec SHA 与 preregistration SHA；禁止按旧日志 mtime 选中 zero-scale/v1.5/R2/R3。`~/tlog` 与 `~/mon` 必须按状态 schema 显示：training、imitation、probe、core9、directional、wandb_sync、视频/人工复核；training 时 evaluation=idle 是正常互斥状态，非 training 阶段不得误报“训练异常停止”。
4. heartbeat 在 supervisor 等待网络、W&B、评估重试及阶段切换时仍必须至少每 60 秒更新，包含 supervisor PID、active child PID、phase、status、checkpoint、effective updates、authority 版本和故障分类。heartbeat 超时或 PID/phase 不一致必须 fail-closed，不得静默等待。
5. 故障必须分为：`training_mechanism_error`、`wandb_infrastructure_retryable`、`evaluation_infrastructure_retryable`、`video_infrastructure_retryable`、`external_fault_requires_user` 与规范门禁终态。W&B/评估/视频基础设施故障应保存本地证据并由同一 supervisor 自动重试；不得改判为训练/行为失败，也不得借故修改训练语义。规范门禁终态不得自动绕过。
6. 每次启动下一 train/eval/video 子阶段后必须立即验证 child PID 存活、PPID/进程组归属、phase/state/heartbeat 一致，并写只读 PID 见证；验证失败时禁止继续或伪报运行。阶段完成后必须确认 active PID 已清空再进入下一阶段。
7. 每个 W&B 阶段仍必须独立 run，完整远端同步及 marker 通过后才可进入下一训练保存点；旧 run 只作历史证据。评估器仍须保留 checkpoint SHA、lineage、schedule、runtime snapshot、reset、schema、seed/matrix 和唯一合法 task alias 的 fail-closed 检查。
8. 候选视频交付前必须逐条执行视频可见性 QC：`side_top` env0 camera runtime witness、分辨率、帧数、时长和输出文件完整性全部通过；QC 结果和 SHA 写入只读视频 manifest。缺少机器人/相机绑定或动作不完整时只能分类为视频基础设施故障并自动重录，禁止把空视频提交用户，也禁止据此否决 checkpoint。
9. 从本节以下开始，v1.7、v1.6.x、v1.5.x、v1.4、v1.3、v1.2、v1.1.x 及原始第 1--15 节均为 `historical-only`：仅用于解释旧 checkpoint、旧门禁和经验，不再拥有当前启动、恢复或修改 authority。与 v1.7.1/v1.7 冲突时一律不得执行。
10. v1.7.1 rebinding 完成后，必须从迁移前最近完整 checkpoint/阶段证据继续；端到端验证至少覆盖新 supervisor PID、lock、heartbeat、显式 dashboard workflow、下一子阶段 PID witness 与 W&B/评估 phase。只有全部一致才允许重新 enable 当前 service。

### v1.7：0707 历史语义纠错与唯一 `historical_0707_exact` 路线

本节受 v1.7.1 基础设施 amendment 约束并保留为当前训练合同来源；它覆盖 v1.6/v1.6.1 以及更早章节中与本节冲突的“0707 exact”解释。历史原始证据已经闭环：此前把 `phase_scale=2.0`、`rear_box_scale=1.5` 和 warmup=`1400` 归类为后来新增差异是错误的。

#### v1.7 已冻结的历史事实

1. 0707 early 和 deployed Student 保存的 `agent.yaml` 均为 `student_actor_warmup_updates=1400`、`student_highstep_phase_loss_scale=2.0`、`student_highstep_rear_box_loss_scale=1.5`。
2. 两次运行保存的 `robot_lab.diff` 均证明主 action error 使用 Student 与 Teacher 原始/pre-prior action；`phase_scale=2.0` 和 `rear_box_scale=1.5` 从训练开始参与 estimator 的 Teacher action loss。
3. TensorBoard 证明 phase 权重从首个 update 已实际大于 1；actor adaptation 与非零 prior-box weight 均恰好在相对起点 `+1400` update 首次出现。
4. warmup 内额外 prior-box loss 必须为 0；warmup 后才允许最终层四个 box rows/bias 有界适配。
5. 完整证据及文件 SHA 冻结在 `/home/lxq/Softwares/robot_lab/tmp/highstep_historical_0707_exact_20260714/historical_evidence_correction.json`。

#### v1.7 旧 zero-scale ablation 的身份与终态

1. workflow `highstep_0707_exact_new_teacher_20260713` 实际采用 warmup=`1200`、`phase_scale=0`、`rear_box_scale=0`，从本版起只能称为 `zero_scale_ablation`，不得再称为 `0707_exact` 或历史复现。
2. 不得在该路线当前 optimizer 中途切换为 1400/2.0/1.5；其 checkpoint、optimizer、W&B、评估、日志和失败事实完整保留，只作消融证据。
3. 已在最近完整 E900 checkpoint 原子边界温和停止并封存；未完成的 E1200 运行不得作为恢复源，禁止从该 Student/optimizer 状态恢复 historical 路线。

#### v1.7 唯一 historical 路线

1. 新 workflow 固定命名为 `highstep_historical_0707_exact_new_teacher_20260714`，route 固定为 `historical_0707_exact`。
2. 从唯一受保护 Teacher `model_172300.pt`（SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`）fresh、weights-only 初始化 Student；独立 frozen Teacher、PPO 关闭、action contract 和 v1.6 其它未冲突保护项保持不变。
3. 固定训练语义：warmup=`1400`；主 action target=`teacher_pre_prior`；`phase_scale=2.0`；`rear_box_scale=1.5`；二者从 update 0 参与 estimator action loss；warmup prior-box loss=`0`；update 1400 后才允许最后四个 box rows/bias 以历史 LR 有界适配。
4. static、tensor binding 和 1--5 update smoke 必须逐项证明第 3 条真实生效；新 Student 必须 fresh，不得继承 zero-scale ablation 的 optimizer。
5. 保存点固定为 `100→300→500→700→900→1400`；E1400 是绝对上限，禁止 E1800/E2500。每点继续执行候选自身驱动的行为评估、模仿诊断、W&B 独立 run 和既有 directional 门禁。首次达到稳定单侧上台门禁即停止新增 iteration并进入人工视频复核。
6. E700 必须与 zero-scale E700 对 first rear、second rear、rear hold 的 latent 和 12 个非 box pre-prior action 作同状态逐阶段比较；若没有预注册定义的明确改善，停止新增 iteration并转入新 Teacher latent 可观测性定位，禁止机械推进 E900/E1400。
7. 本轮根因审计仅封顶完成三项：velocity target 与 `/2` 倍率合同；estimator observation history 的顺序、scale、normalization；pre-prior action loss 到 estimator 的实际梯度。禁止继续扩大审计。
8. 未经预注册的单变量 A/B 和用户批准，今后禁止再次改变 warmup=`1400`、`phase_scale=2.0`、`rear_box_scale=1.5`，也禁止把多个训练语义变化捆绑为一次“修复”。
9. 不修改 Teacher、reward、Teacher prior、网络、`action_scale`、`joint_pos.clip`、default pose、部署输入、真机 gains 或动作 contract；禁止自动真机部署。

### v1.6（historical-only）：`0707_exact_new_teacher` 严格复现路线

本节是当前最高执行依据。它覆盖 v1.5.2 及更早版本中要求 warmup 使用 post-prior 16 维主动作 target、`phase_scale=2`、`rear_box_scale=1.5`、warmup=1400 或允许从 E100/E300/E500/E900 恢复的冲突条款。当前唯一任务是：使用已经接受的新 Teacher，严格复用 0707 已证明有效的 Stage-2 蒸馏语义，尽快得到可观看并可用于有人保护真机实验的 Student。

#### v1.6 旧路线终态

1. v1.5/v1.5.2 路线的 E100、E300、E500、E900、optimizer、日志、W&B、manifest、state、handoff 和失败事实必须完整保留，不删除、不覆盖、不改判。
2. E300→E500 的行为退化证明该路线没有复现 0707 有效蒸馏；禁止继续追加 iteration，禁止从其任何 Student/optimizer 状态恢复。
3. 旧路线失败不影响新 Teacher 资格，也不得打开 Teacher、reward、网络、动作 contract、部署端或 sim-to-real 分支。

#### v1.6 唯一 Teacher 与 fresh 初始化

- Teacher：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt`
- SHA256：`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`
- Student 必须从该 checkpoint fresh、weights-only 初始化；Teacher 必须独立加载、冻结并与 Student 使用不同 tensor storage。

永久保持不变：PPO关闭；estimator LR=`1e-3`；每个 effective update 执行 4 个 VAE epochs；velocity/latent/action/recon/KL loss 分别为 `10/50/20/0.5/0.1`；actor、critic、privileged encoder 冻结；`action_scale`、`joint_pos.clip`、网络、reward、Teacher prior、部署输入和真机 gains 不变。

#### v1.6 精确 0707 蒸馏语义

1. warmup=`1200` effective updates；warmup 期间唯一可训练范围为 `estimator.*`。
2. warmup 主 action target 必须是 Teacher 原始/pre-prior 16 维动作；post-prior 动作不得进入 warmup 主 action loss。
3. warmup 期间 prior-box loss 必须严格为 0；`phase_scale=0`；`rear_box_scale=0`。
4. update 1200 之后才允许 actor 最后一层的 4 个 box action rows `[12:16]` 及 bias 以 Adam LR=`1e-5` 有界适配；其他 actor rows永久冻结。
5. update 1200 之后只允许 0707 历史条件式 post-prior box loss；不得把 post-prior 重新变成全 16 维主 action target，不得加入新的 phase/rear box 加权。
6. full checkpoint 必须保存并精确恢复 estimator/box optimizer、effective update、Student/Teacher 双 checkpoint、schedule 和训练范围。

#### v1.6 验证与保存点

训练前只完成必要的 static、tensor binding 和 1--5 update smoke，并直接证明：warmup 梯度只进入 estimator；主 target 是 pre-prior；warmup post-prior/box loss 为零；Teacher/Student storage 独立；optimizer/effective update 可完整恢复。禁止借 smoke 扩大研究范围。

fresh 训练保存点固定为：`100→300→500→700→900→1200`。700 必须保存。1200 之后只有在尚无候选且规范内证据支持时才按既有固定上限继续；不得默认追加。checkpoint 排序以候选 Student 自己驱动的上台行为为第一依据，不按总 loss、latent 或 0707 Student 驱动的 same-state 轨迹选择。

#### v1.6 行为与候选门禁

1. same-state 数据只用于动作误差诊断，禁止用 0707 Student 驱动的轨迹声称候选完成上台。
2. 行为结论只能来自候选 Student 自己驱动的 probe/core9/directional rollout 和机器人全程清晰可见的视频。
3. directional 矩阵、稳定上台门禁、真机安全诊断和人工视觉复核继续使用 v1.5.2 固定定义。
4. 第一个在任一预标定单侧工作域达到稳定门禁且无明确真机破坏风险的 checkpoint 出现后，立即停止新增 iteration，导出精确 checkpoint，生成 Teacher/Student 成对视频和俯视摆位卡，状态只能写 `student_directional_candidate_pending_user_visual_review`。
5. 禁止自动真机部署。

#### v1.6 自动化与 W&B

每个保存点使用独立 W&B run，`group=0707_exact_new_teacher` workflow id；进入下一保存点前必须完成同步。监督器必须自动区分机制错误、行为失败、安全风险、普通诊断和外部故障；禁止静默停止。规范范围内的代码修复、训练、评估、W&B、导出和视频已获授权。禁止增加 reward、随机化、sim-to-real、其它实验分支或未经 0707 验证的蒸馏“改进”。

### v1.6.1：最小变更纪律

本节与 v1.6 共同构成当前最高执行依据；旧版本章节中的“当前最高”只保留其发布时含义。本节只限制今后对训练语义的新增修改，不重新解释或覆盖 v1.6 已明确规定的内容，也不改判任何历史结果。

1. 自动推进授权继续包括进程恢复、监督器、日志、W&B、训练、评估、导出和视频；但运行故障修复不得顺带改变 Teacher、target、loss/权重、warmup、冻结范围、optimizer/LR、schedule、prior、网络、reward 或动作 contract。
2. v1.6 明确写出的 0707 蒸馏语义是当前受保护基线。新 Teacher 以及 v1.6 已逐项写明的差异是允许项；未写入 v1.6 的训练语义差异一律不得静默加入。
3. 若受保护路线尚未达到行为目标，先以候选 Student 自己驱动的 rollout 和视频确认失败；loss、latent、same-state 轨迹或其它诊断指标不得单独触发新的训练改造。
4. 确需改变训练语义时，一次只能提出一个变量，并在实施前写明失败证据、改动内容、固定预算、成功/失败条件和回滚点；未经用户明确批准不得实施，禁止同时叠加多个“改进”或看到结果后修改理由和门槛。
5. 不改变训练语义的正确性修复可以按现有自动化授权直接执行；无法确认是否影响训练语义时必须按训练语义修改处理，不得扩大解释。

### v1.5.2：以最短路径交付可真机测试 Student

本节是当前最高执行依据，覆盖 v1.5.1、v1.5 及更早版本中将辅助评分、诊断项或双侧鲁棒性置于实际交付目标之上的冲突文字。v1.5 的 0707-faithful Stage-2 蒸馏机制、唯一新 Teacher、永久动作 contract、保存点、checkpoint/lineage/schedule 真实性、W&B 证据保留和禁止自动真机部署继续有效；但所有指标的解释必须服务于本节的交付顺序，不得再次把指标本身变成项目目标。

#### v1.5.2 已确认的可靠前提

1. 用户已人工接受唯一新 Teacher `model_172300.pt` 的上高台效果；禁止因 Student 失败重新打开 Teacher、reward 或 Teacher action prior 方向。
2. 0707 已存在完整且实际有效的 Teacher→Student→真机部署链路，证明当前网络/动作 contract 下 Stage-2 蒸馏可行。
3. 冻结的 0707 early/deployed 配对、历史配置、视频和模仿统计是恢复蒸馏机制、诊断退化和选择 checkpoint 的参考；其作用是帮助得到可用 Student，而不是要求新 Student 机械复制旧策略的每个非关键数值。
4. 当前唯一交付目标是：尽快得到一个在预先标定的至少一个单侧工作域内稳定完成整机上台、动作经用户观看可接受、可用于有人保护真机实验的 Student checkpoint。

#### v1.5.2 固定优先级

所有自动化决策必须按以下顺序处理冲突：

1. Teacher/Student 双 checkpoint、冻结范围、PPO关闭、post-prior 16维 target、optimizer/effective update、schedule 和动作 contract 正确；
2. Student 在 nominal 或任一固定单侧 corridor 内稳定完成 `full_climb` 与 `rear_hold`；
3. 不存在足以造成真机损坏、失控或明显无法进行有人保护实验的严重动作；
4. Student 关键阶段动作与新 Teacher/0707 有效蒸馏机制一致，且视频中机器人与完整动作清晰；
5. 用户人工观看并接受；
6. 双侧鲁棒性、front-top 细节、逐元素模仿阈值、loss、latent、velocity、terrain、全局 MAE 和其他诊断指标。

低优先级项目不得覆盖高优先级项目。尤其禁止仅因另一方向失败、少量非关键逐元素模仿项超限、loss/latent 不理想、左前腿上台后未立即复位或附加评分未满，而永久否决已经满足前五项的单侧真机试验候选。

#### v1.5.2 硬阻断项与诊断项

只有以下情况允许自动化阻止候选交付：

- checkpoint、Teacher/Student、冻结范围、PPO状态、target、optimizer、schedule、action contract 或导出绑定不真实/不一致；
- 在预注册评估中没有任何一个 nominal/单侧 corridor 达到稳定整机上台；
- 出现经动作、关节、姿态或接触证据确认的真机破坏/失控高风险；
- 视频缺少机器人、动作不完整、Teacher/Student 不同状态或用户人工明确拒绝；
- 无法恢复的外部故障。

以下项目默认是诊断、回归定位和 checkpoint 排序证据，不得单独形成永久停止：另一方向失败、完整双侧 core9 未达标、非关键关节/阶段的逐元素 MAE/q95/sign、全局 action MAE、latent/velocity/reconstruction loss、terrain/curriculum、front-top 复位细节、W&B 图形趋势和不影响动作/安全/真实性的附加评分。诊断项异常时，监督器必须先判断它是否已经实质破坏蒸馏正确性、上台行为或真机安全；若没有，只记录并继续候选交付，不得机械 fail-closed。

冻结的 0707 imitation envelope 继续作为强制诊断基准。若出现广泛、关键阶段或关键动作组退化，必须阻止盲目续训并修复蒸馏机制；若仅少量非关键元素未通过，而候选已稳定上台、关键阶段与 Teacher 一致且视频可接受，则状态应进入用户目标复核，而不是永久 `stopped_by_gate`。最终是否接受该例外只由用户人工视觉复核决定，自动化不得自行宣称真机候选。

#### v1.5.2 最短自动推进路线

1. 保护并只读保留所有旧结果。当前 v1.5 E100/E300 的失败是蒸馏差异诊断证据，不是重新训练 Teacher、修改 reward 或废止整条路线的理由。
2. 对照0707成功链，只审计和修复会影响 Stage-2 蒸馏正确性的差异：Student/Teacher初始化与独立绑定、冻结/可训练张量、PPO、post-prior target、历史有效 loss 权重、optimizer、effective update、schedule和checkpoint恢复。禁止扩展到Teacher、reward、prior、网络、`action_scale`、`joint_pos.clip`、部署输入、真机gains或非必要sim-to-real修改。
3. 修复后从受保护 `model_172300.pt` fresh 启动，或仅在完整证据证明现有 checkpoint 的机制和 optimizer 均正确时精确恢复；不得为了节省少量 iteration 继承已确认错误的蒸馏状态。
4. 继续使用预注册保存点 `100→300→500→900→1400→1800→2500`，但每个保存点的首要判断是关键动作模仿趋势和实际上台行为，不得仅按总 loss、latent 或元素失败数量排序。
5. 在要求 real-gain core9 的保存点执行完整 core9，并按 v1.5.1 固定矩阵复用或补齐可能成功方向的 half-offset。完整双侧门禁通过可直接进入视觉复核；未通过时必须继续独立检查左右 directional corridor，另一侧失败不否决成功侧。
6. 单侧 corridor 的自动稳定证据保持 `valid=9/9、full_climb>=8/9、rear_hold>=8/9`，且对应 half-offset `full_climb=3/3、rear_hold=3/3`。`front_top_support`、`no_severe_inward` 和0707逐元素结果必须完整报告并用于安全/动作诊断；只有被证明确实破坏动作完整性或真机安全时才阻断，否则转入用户目标复核。
7. 第一个满足单侧稳定上台且无明确破坏风险的 checkpoint 出现时，立即停止新增 iteration，导出该精确 checkpoint，生成成功方向的 Teacher/Student 同状态成对视频和俯视摆位卡，状态写 `student_directional_candidate_pending_user_visual_review`。
8. 用户人工观看接受后才允许写 `directional_real_robot_trial_candidate` 并交付真机部署 manifest；禁止自动部署。用户若拒绝视频，只否决该 checkpoint，并依据证据恢复下一个最短合法动作，不得扩大为废止 Teacher、蒸馏主线或 directional 机制。

#### v1.5.2 自动化停止与恢复原则

1. `loss`、latent、另一侧失败、附加评分或连续诊断退化不得直接把整个 workflow 永久终止。监督器必须自动区分：蒸馏机制错误、行为失败、安全风险、普通诊断异常和外部基础设施故障。
2. 蒸馏机制错误时，保存 checkpoint/optimizer/log/manifest/handoff，自动完成 code-to-0707 差异定位和规范内最小修复，再从正确边界恢复；禁止无证据追加 iteration。
3. 行为尚未成功但机制正确时，按保存点和2500绝对上限继续；达到上限仍无任何单侧候选时才以行为失败终止并提交完整证据。
4. 外部网络、W&B、视频、Isaac进程或资源故障应在本地证据保护下自动恢复；只有无法自动恢复或需要用户观看候选视频时才等待用户。
5. 每次撤销、停止或改变路线必须把影响范围写清楚；禁止从一个 checkpoint、一个方向或一个指标的失败推导出用户未明确授权的更高层机制废止。

### v1.5.1：恢复单侧工作域快速真机试验候选通道

本节在 v1.5.1 发布时是最高执行依据，现受 v1.5.2 的交付优先级、硬阻断/诊断分类和用户目标复核条款约束。它仅覆盖 v1.5、v1.4 及更早版本中与 directional 候选资格、单侧工作域验收和候选发布相关的冲突文字；v1.5 的 0707-faithful 蒸馏机制、冻结范围、Teacher、保存点、完整 core9 主目标、W&B、视频人工复核和永久禁止项全部继续有效。

用户长期有效且不得再次扩大解释的最高验收要求是：**最高目标是真机完成整机上高台；如果 Student 只在一个预先标定的横向偏置/偏航方向稳定成功，而另一方向不成功，用户接受在真机实验中使用成功方向作为初始摆位。相反方向失败不得单独否决该单侧真机试验候选。**

#### v1.5.1 纠错边界

1. 旧 B500 checkpoint 的 `user_rejected_non_deployable` 状态保持不变。该状态只表示用户已人工否决这个具体 checkpoint 的动作质量，禁止恢复其候选资格、继续旧 B500/R5 训练或重新解释旧视频。
2. 旧 B500 的具体人工否决不得再被解释为用户否决 directional 验收思想、单侧初始摆位方案或所有未来新 Student 的单侧候选资格。
3. v1.4 的旧 B500/R5 实施路线仍为历史归档；v1.5.1 恢复的是**面向 v1.5 及后续新 Student checkpoint 的 directional 验收通道**，不是复活旧 checkpoint 或旧 optimizer。
4. 只有用户今后再次作出明确、直接、限定范围的指令，才允许取消这一单侧候选通道。不得依据某个 checkpoint 失败、某段视频被拒绝或某一方向失败，自行推导为机制永久废止。

#### v1.5.1 主目标与快速候选的关系

1. 完整双侧 real-gain core9 的最终主目标保持：`valid=9/9、full_climb>=8/9、rear_hold>=8/9、front_top_support>=8/9、no_severe_inward>=8/9`。
2. directional 是并列保留的**快速真机试验验收通道**，不是通用双侧鲁棒结论，也不得降低、改判或覆盖完整 core9 的统计结果。
3. 新 Student 只要通过任意一个固定方向的 directional 门禁，即应停止继续消耗训练预算，进入导出、成对视频和用户人工视觉复核；不得仅因相反方向失败而继续无限训练。
4. directional 候选只能称为 `directional_real_robot_trial_candidate`，只允许用于预标定摆位、有人保护的真机实验；禁止称为通用鲁棒候选，禁止自动真机部署。

#### v1.5.1 固定单侧评估矩阵

每个方向均使用 `seeds=11,22,33`，评估参数继续固定为 real-gain、delay=0、关闭 play-time 随机事件、`box level 9`、速度命令 `[0.45,0,0]`、600 steps：

- nominal：`lateral=0.00 m, yaw=0°`；
- left corridor：half-left `(+0.06 m,+2°)`，left-offset `(+0.12 m,+4°)`；
- right corridor：half-right `(-0.06 m,-2°)`，right-offset `(-0.12 m,-4°)`。

每个 directional corridor 固定为 nominal 3 场、对应 half-offset 3 场、对应 full-offset 3 场，共 9 场。已有 core9 场次只有在 checkpoint、checkpoint SHA、评估代码 SHA、agent/env 配置、real-gain、delay、随机事件、schedule 和 reset contract 全部一致时才允许复用；否则完整重跑。左右方向使用同一固定阈值独立判定，允许一个通过、另一个失败；不得看到结果后改变偏置符号、幅度、seed 或阈值。

#### v1.5.1 directional 固定门禁

任一方向必须同时满足：

- `valid=9/9`；
- `full_climb>=8/9`；
- `rear_hold>=8/9`；
- `front_top_support>=8/9`；
- `no_severe_inward>=8/9`；
- 对应 half-offset 必须 `full_climb=3/3` 且 `rear_hold=3/3`；
- 通过冻结的 0707 early/deployed 独立模仿包络，不得以行为分、全局平均 action MAE、loss 或 terrain 改善代替；
- 完成独立 Teacher-action regression、lineage、schedule、双 checkpoint 和动作 contract 复核；reference 必须是冻结的 0707 独立参考，不得使用候选自身循环自证。

满足数值门禁后，自动状态只能先写 `student_directional_candidate_pending_user_visual_review`。必须生成 Teacher/Student 同状态成对视频以及成功方向的摆位卡；视频中机器人和完整动作必须清晰可见。只有用户人工观看并明确接受后，状态才允许变为 `directional_real_robot_trial_candidate`。

#### v1.5.1 自动推进与防止再次过度废止

1. 在 v1.5 每个要求 real-gain core9 的保存点，先执行完整 core9。若完整主门禁未通过，但 nominal 加任一 full-offset 方向在现有证据上仍可能达到 directional `8/9`，只补该方向缺失的 half-offset 3 场；若两个方向都可能通过，则两侧都补并独立报告。
2. 任一方向通过 directional 数值门禁后，立即停止新增训练 iteration，完成导出、视频和摆位卡并等待用户视觉复核。相反方向失败只记录，不否决，不得以“继续追求双侧对称”为由自动覆盖这一加速停止条件。
3. 若两个方向都未通过，则按 v1.5 原保存点、止损和 2500 absolute cap 继续；不得放宽 directional 阈值或复活旧 B500/R5 路线。
4. 以后任何“撤销候选”必须分别写明：撤销的是具体 checkpoint、具体证据结论、具体训练路线，还是整个验收机制。未被用户明确点名撤销的更高层机制必须保留，禁止扩大解释。

### v1.5：0707 完整蒸馏机制恢复与新 Teacher 忠实继承主线

本节在 v1.5 发布时是最高执行依据，取代 v1.4、v1.3-R4/R5 及更早版本中与之冲突的候选、训练和门禁路线；现受 v1.5.2 的交付优先级以及 v1.5.1 的 directional 纠错条款共同约束。旧 checkpoint、日志、manifest、W&B、视频和数值结果全部只读保留，不删除、不覆盖、不改判；但用户已人工确认 B500 play 动作无法用于真机，故**该 B500 checkpoint** 的 `directional_real_robot_trial_candidate` 资格永久撤销，状态固定为 `user_rejected_non_deployable`。原 8/9 directional 数值仍是历史行为统计，不得再解释为该 checkpoint 的 Teacher 模仿到位或真机候选；不得据此否决未来新 Student 的 v1.5.1/v1.5.2 directional 验收通道。

#### v1.5 已确认事实与根因

1. 0707 实际部署 Student 的完整形成链不是从 `model_153898` 才开始，而是：

```text
早期结构证明：Teacher 2026-07-01_06-52-29/model_141000
  -> Student 2026-07-01_18-19-06/model_141700--142200

正式 ActionScore Teacher：2026-07-04_01-45-21/model_151399
  -> 首次 Stage-2 no-prior Student 2026-07-04_06-32-23
  -> 行为最佳中间点 model_152100
  -> 首轮终点 model_153898
  -> 继续 Stage-2 蒸馏 2026-07-05_00-13-46/model_158797
  -> 0707 真机部署
```

2. 该链路已经证明 Student 可以基本继承 Teacher 的完整上台动作。有效机制为：从 Teacher actor 初始化 Student 和独立 frozen teacher actor；PPO 始终关闭；前 1400 effective updates 完整冻结 Student actor、只训练 estimator/latent；监督目标为 Teacher post-prior 最终 16 维动作；warmup 后只允许最终层 box action rows 有界适配；按早/中/切换点/后期保存并比较 checkpoint，不默认选择最终点。

3. ActionScore 任务曾专门修正旧 highstep 的灾难性后期退化：阶段瓶颈 score、support gate、support bottleneck warmup、best-regression curriculum、success hold 和 `late/best` 止损共同把旧链路约 18%--69% 的动作/支撑塌缩，降低到首次正式 Student 蒸馏末段约 6%--7%。继续到 `model_158797` 后相对该段最佳窗口仍有约 12%--13% 回落，因此属于“已显著缓解、尚未彻底解决”，不得继续训练到最终点后才选择。

4. 当前 B500 不是完整蒸馏：它从已退化的 `model_900/B300` 出发，仅训练 `RL_hip/RR_hip` 两个最终输出行，actor body、estimator/latent、box rows 和其余动作均冻结，机制上无法恢复整套 Teacher 动作。

5. 现有 directional Teacher-action gate 对第一候选把 B500 自己作为 reference，只检查 5400 帧完整性；B500 的全局 action MAE `0.2830786397`、关键阶段 latent MAE 约 `0.266--0.555` 和显著 box action 误差未参与否决。该 gate 结论无效，修复前禁止发布任何新候选。

#### v1.5 永久保护与禁止项

冻结并保护唯一新 Teacher：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt
SHA256=dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35
```

该 Teacher 已完成 Standard core9 9/9 和 0707 rosbag 实测 revolute gains 中心、delay=0、关闭 play-time 随机事件的 real-gain core9 9/9。v1.5 禁止重训或修改 Teacher、reward、Teacher action prior、网络结构、部署输入、观测/动作/关节 contract、`action_scale`、`joint_pos.clip`、`default_dof_pos`、真机 gains、部署端代码或 sim-to-real 随机化。禁止恢复 B500/R2/R3/R4/R5 optimizer，禁止继续局部两行/四 tensor 修补，禁止自动真机部署。

#### v1.5-A：0707 独立 Teacher→Student 模仿基准

任何新参数更新前，必须先对以下只读配对建立独立参考基准：

```text
paired Teacher:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/model_151399.pt
  SHA256=d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d

early behavior-best Student:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-04_06-32-23/model_152100.pt
  SHA256=3c7a335bdc6c1b952e3b0ddbfdd35f0e4311ca1a9fb86cb9ae9a21de2f63a8d6

0707 deployed Student:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt
  SHA256=7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284
```

审计必须使用各自 run 保存的 agent/env 配置、正确 ActionScore Teacher/StudentNoPrior task、明确恢复的 action-prior/schedule 语义和相同状态；不得用当前配置静默覆盖历史配置。固定覆盖 approach、front lift、front top support、first rear top、second rear top、rear hold，至少包含 seeds `11/22/33` 的 nominal 轨迹；若历史 deterministic reset 无法复现，必须预注册等价固定状态集并记录限制，不得伪造同口径。

对 `152100↔151399` 与 `158797↔151399` 分别输出：逐阶段/逐关节 Student→Teacher post-prior bias、MAE、q95、sign mismatch、饱和/裁剪；latent/velocity estimate；动作一阶差分和阶段事件时序；front/rear/box 动作组；Teacher/Student 同初始状态同步视频。以两个旧 Student 中各指标较差但仍被历史视频证明可模仿 Teacher 的包络，形成只读 `0707_imitation_reference_manifest`。该包络只能由旧配对生成，在观察新 Student 之前冻结 SHA，禁止事后修改。

#### v1.5-B：新 Teacher 的 0707-faithful Stage-2 Student 蒸馏

新 workflow 必须从唯一 `model_172300` fresh 初始化新的 ActionScore StudentNoPrior，禁止从 `model_900`、B500 或任何 R 路线恢复。代码差异审计必须证明：

- `model_172300` actor 权重同时初始化 Student actor 和独立 frozen Teacher actor，Teacher privileged encoder 独立加载并保持冻结；
- `distill_stage=2`，Student PPO 永久关闭，critic/privileged encoder 不参与 RL 更新；
- `student_actor_warmup_updates=1400`；warmup 内完整 Student actor 冻结，只训练 estimator/VAE latent；
- `student_vae_epochs=4`、基础 optimizer/learning rate 与 0707 正式蒸馏一致，默认 `learning_rate=1e-4`；
- Student 目标为同状态 `model_172300` Teacher post-prior 最终 16 维动作；
- 恢复 0707 已验证配置：`student_teacher_action_loss_coef=20.0`、`student_prior_box_loss_coef=5.0`、`student_highstep_phase_loss_scale=2.0`、`student_highstep_rear_box_loss_scale=1.5`；
- warmup 后只允许 0707 历史机制中的最终层四个 box action rows 及 bias 有界适配，PPO、actor body、hip/thigh/calf rows 继续冻结；若当前代码无法严格复原该范围，必须先做最小正确性修复和 tensor-binding 测试，不得扩大可训练范围；
- optimizer、effective update count、Teacher/Student 双 checkpoint、schedule 和 W&B 均可完整恢复且 fail-closed。

训练前必须写不可事后修改的 preregistration manifest。static、tensor binding、1--5 iteration smoke 通过后，按以下固定保存点自主推进：

```text
100 -> 300 -> 500 -> 900 -> 1400 -> 1800 -> 2500 effective updates
```

`100--1400` 为 estimator/latent-only；只有 1400 门禁通过才允许进入 box-head adaptation。2500 是本轮绝对上限；禁止因 terrain/loss 上升追加到 5000 或从最终点盲目续训。每一保存点均保留，最终选择行为和模仿共同最好的 checkpoint，不默认选择 2500。

#### v1.5 分阶段模仿、防退化与行为门禁

每个保存点必须先执行同状态 imitation gate，再执行最小行为 probe；`500/900/1400/1800/2500` 还必须执行同规格 real-gain、delay=0、关闭随机事件的 core9。门禁固定为：

- lineage、Teacher、schedule、action contract 和帧完整性全部有效；
- gate reference 必须是冻结的 `0707_imitation_reference_manifest`，候选不得与自己比较；
- 每个关键阶段的 latent、velocity、16-D action、四个 action group 和关键 box joint 指标均不得劣于0707参考包络；禁止只用全局平均 MAE 抵消单阶段/单关节失败；
- approach、front support、first/second rear 和 rear hold 的事件时序偏差不得超过0707参考包络；
- warmup 阶段若连续两个保存点 imitation 或行为恶化且无 latent 改善，立即停止并保留较早最佳点；
- `late_window_score >= 0.90 * best_score`、`late_window_support >= 0.90 * best_support` 是最低防塌缩线；若跌破则不得继续下一阶段。候选发布还要求相对最佳保存点不低于 `0.93`；
- loss、terrain、latent单项改善不能覆盖真实完整上台、Teacher模仿或support退化；
- core9 的最终自动行为线仍为 `valid=9/9`、`full_climb>=8/9`、`rear_hold>=8/9`、`front_top_support>=8/9`、`no_severe_inward>=8/9`；未达到时不得发布候选；
- 任一保存点同时达到模仿和最终行为门禁，立即停止后续训练并进入视频复核。

#### v1.5 视频、候选与用户复核

候选必须为同 checkpoint 的 Teacher/Student 同初始状态成对视频，至少覆盖 nominal 和预注册的左右偏置，机器人必须从首帧到终止清晰可见；自动流程在写 candidate 前必须抽帧验证非空、相机跟随和机器人像素占比。视频必须展示完整 approach、前足支撑、两后足上台和 rear hold，不得只录成功片段。

自动化达到数值门禁后只能写 `student_candidate_pending_user_visual_review`。用户明确观看并接受动作前，不得写 `directional_real_robot_trial_candidate`、不得生成可部署结论、不得自动部署或启动真机。用户拒绝视频时状态为 `user_rejected_non_deployable`，保留全部证据并停止，不得用旧数值门禁覆盖人工结论。

#### v1.5 自动化、W&B 与非静默恢复

v1.4.1 的每阶段独立 W&B run、config/summary、`pending_sync` fail-closed 和历史补传规则继续有效。v1.5 使用独立 workflow/state/lock/heartbeat/handoff，不覆盖旧状态。合法状态至少包括：`auditing_0707_reference`、`training_estimator_warmup`、`evaluating_warmup_gate`、`training_box_head_adaptation`、`evaluating_candidate`、`student_candidate_pending_user_visual_review`、`user_rejected_non_deployable`、`stopped_by_gate`、`external_fault_needs_user`。

supervisor 必须在训练、评估或视频异常停止后自动定位并在规范范围内恢复；不得静默停在 `not running` 或 `evaluation 0/N`。只有规范未覆盖的重大方向变化、无法恢复的外部故障或用户视觉复核才需要用户介入。每次停止必须写明当前 checkpoint、effective updates、模仿门禁、core9、最佳 checkpoint、停止原因和下一步。

### v1.4：预标定单侧工作域真机试验候选

本节由用户于 2026-07-13 正式批准，取代 v1.3 中“只有双侧 core9 达到 8/9 才能形成任何真机试验候选”的冲突要求，但不改变或重判 v1.3 的结果。v1.3 仍永久记录为双侧鲁棒目标 `stopped_by_gate`；新增状态名固定为 `directional_real_robot_trial_candidate`，只表示在预先标定的单侧初始摆位工作域内，可用于有人保护真机实验的 Student，不得称为通用鲁棒候选，不得自动部署或启动真机。

#### v1.4 永久锁定项

禁止修改 Teacher、reward、Teacher action prior、网络结构、`action_scale`、`joint_pos.clip`、`default_dof_pos`、观测/动作/关节 contract、部署输入、部署端代码、真机 gains 和 sim-to-real 随机化。评估固定使用 real-gain、delay=0、关闭 play-time 随机事件、box level 9、速度命令 `[0.45,0,0]`、600 steps。所有阈值在评估前固定，看到结果后不得修改。

#### v1.4 第一优先候选：B500 正向单侧走廊

第一优先 checkpoint 固定为：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-12_23-55-22_student_recovery_v111_B500_20260712_235515/model_498.pt
SHA256=f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159
```

在 seeds `11/22/33` 上执行 3×3 固定矩阵：nominal=`lateral 0.00 m, yaw 0°`，half-left=`lateral +0.06 m, yaw +2°`，left-offset=`lateral +0.12 m, yaw +4°`。`left` 仅为场景名；正式几何定义为：面向平台、沿平台边缘法向向高台前进时，机器人中心相对平台中心线沿仿真世界/机器人初始横轴的正向位移，正偏航与之配套。摆位卡必须以带坐标箭头的俯视图表达真实方向，禁止只写 left/right。

B500 历史 nominal/left-offset 结果只有在 checkpoint、评估代码全部 SHA、real-gain、delay、随机事件、schedule manifest/runtime snapshot SHA 完全相同时才允许逐场复用；任一不同即完整重跑 9 场并保存不复用证据。

#### v1.4 directional 固定门禁

同一 checkpoint、同一预注册单侧矩阵必须同时满足：

- `valid=9/9`；
- `full_climb>=8/9`；
- `rear_hold>=8/9`；
- `front_top_support>=8/9`；
- `no_severe_inward>=8/9`；
- half-offset 三场必须 `full/rear_hold=3/3`；
- 在该单侧矩阵上完成并通过 Teacher-action regression、checkpoint/Teacher lineage、schedule/runtime 和动作 contract 复核。

若 B500 通过，立即停止所有训练，导出该精确 checkpoint，录制 nominal/half-left/left-offset 三种摆位（每个 seed 均保留）的完整视频和汇总视频，生成只读真机部署 manifest 与摆位卡。部署 manifest 只授权人工审核后的有人保护实验，不授权自动真机部署。摆位卡俯视图必须明确：平台中心线、平台前缘及其法向、机器人中心、真实横移方向、正偏航方向、起步距离 `2.05 m`（平台边缘净距 `0.55 m` 的当前 reset 几何）和速度命令 `[0.45,0,0]`。

#### v1.4 第二候选与有条件 R5 延续

仅当 B500 directional 门禁失败，才评估现有 R5_1：

```text
/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v13_20260713/r5_runs/main1_attempt_20260713_122952/model_498.pt
SHA256=fc17cf1d5e1623c49a4d7a39becc9ec4476cf74f2b4dbd4b555d030d28808347
```

R5_1 使用对称反向单侧矩阵：nominal=`0.00 m,0°`，half-right=`-0.06 m,-2°`，right-offset=`-0.12 m,-4°`，seeds 仍为 `11/22/33`，门禁、审计、导出、视频和摆位卡要求完全相同。

只有 B500 与 R5_1 两个现有 checkpoint 均未通过 directional 门禁，才允许从 R5_1 的完整 checkpoint/optimizer 继续已预注册 R5_3/R5_5；epoch 绝对上限仍为 5，optimizer、数据、loss、Teacher 和四个可训练 tensor 均不得改变。probe 保护项改为 nominal 加已选定的右侧工作域；另一侧必须记录但不得否决 `directional_real_robot_trial_candidate`。每个保存点仍必须执行同规格 directional 3×3 门禁，达到门禁立即停止；R5_5 未通过则 v1.4 终止并保留全部证据，不得追加 epoch 或引入新机制。

#### v1.4 自动化与状态

本路线使用独立 workflow/state、预注册、lock、heartbeat、handoff 和 systemd supervisor，不覆盖 v1.3。合法状态包括 `evaluating_directional_corridor`、`directional_real_robot_trial_candidate`、`directional_failed_continue_r5` 和 `directional_stopped_by_gate`。训练、评估、导出和视频均已授权；只有达到上述门禁才可写入 `directional_real_robot_trial_candidate`。任何失败或停止必须非静默写明当前 checkpoint、矩阵、结果、下一分支和是否需要用户操作。

### v1.3：固定 core9 同状态 replay 的证据对齐恢复路线

本节建立在 v1.2-R3 已正式 `stopped_by_gate` 的事实之上，并取代 v1.2 中“R3 失败后没有第四路线”的冲突文字。v1.2 及更早路线的 checkpoint、日志、manifest、state 和 handoff 全部只读保留，不得覆盖、删除或解释为获准续训。永久动作 contract、Teacher、最终 `8/9` 行为门槛和禁止自动真机部署仍保持不变。

#### v1.3 已确认失败证据

- R3-10 checkpoint 仅改变预注册的 `actor.6.weight/bias`，Adam、冻结范围、lineage、schedule 和 action contract 均通过审计；因此本次停止不是 checkpoint 恢复或 supervisor 故障；
- R3-10 probe3 为：nominal 失败、left-offset 成功、right-offset 失败；三场均 valid 且无严重后腿内收；
- R2-50 的相同三场为：nominal 成功、left-offset 成功、right-offset 失败；R3 未修复目标 right-offset，却破坏了原成功 nominal；
- R3 训练中 lateral gate mean 从约 `0.052` 增至 `0.132`，但在精确 probe3 轨迹上重新计算的 gate mean 为 nominal=`0`、left-offset=`0`、right-offset≈`0.0012`；
- `abs(left_scan_mean-right_scan_mean)` 既无法识别 3 m 宽平台上的 ±0.12 m/±4° core9 场景，也丢失左右方向，因此它选中的训练状态与目标失败场景不一致；
- R3-10 相对 R2-50 的 actor head weight relative-L2 drift 约 `0.00239`，三场同状态 raw-action MAE drift 约 `0.0067--0.0071`；该变化虽小，仍足以改变接触敏感的 nominal 成功轨迹；
- right-offset 的 Student→Teacher 16-D action MAE 仅由约 `0.2652` 降至 `0.2610`，没有形成有效纠偏。

因此永久禁止：恢复 R3 optimizer、继续 R3-25/50/100、调低 R3 学习率后改名重跑、放宽 R3 probe3、继续使用当前 lateral gate，或把 loss 下降解释为行为改善。

#### v1.3-R4：固定 core9 replay 的约束最小二乘动作头拟合

唯一优先新路线为 R4。它使用正式行为最佳选择规则选出的 B500 作为起点与冻结 anchor：B500 与 R2-50 均为 full/rear_hold=`6/9`，但 B500 front support=`8/9` 高于 R2-50 的 `6/9`。受保护根仍为 `model_900`，Teacher 仍为唯一 `model_172300`。

R4 不是新的 on-policy RL/DAgger 循环，而是对现有最后一层线性 action head 的确定性固定数据拟合：

- 先在完全相同的 real-gain、delay=0、关闭 play-time 随机事件的 core9 中，为 B500 采集 9 条各 600 帧的同状态轨迹；
- 每帧必须保存 policy/estimator/critic observation、B500 action、Teacher pre/post-prior action、phase、场景、物理状态 digest 和 runner/manual parity；
- B500 已成功的 6 条轨迹以 B500 action 为逐帧保持 target；失败的 3 条轨迹以同状态 Teacher post-prior 16-D action 为纠偏 target；
- 训练采样按 trajectory 和 phase 均衡；scenario/phase 只用于离线采样和 loss 权重，不进入 policy、estimator、critic 或部署输入；
- Student estimator、actor body、critic、privileged encoder、action std、Teacher 和 B500 anchor 全部冻结；R4 只允许改变现有 `actor.6 Linear(128,16)` 的 weight/bias；
- 从冻结 actor body 提取 128-D penultimate feature，附加 bias 列后，对 16-D target 执行以原 B500 head 为中心的加权 ridge least-squares；不使用 Adam、PPO reward、latent loss、velocity loss、height-scan gate 或手写关节目标；
- 成功轨迹保持权重固定为失败纠偏权重的 `20` 倍；每个 trajectory 总权重相等、每个已出现 phase 在该 trajectory 内总权重相等；
- ridge λ 候选在看到行为结果前固定为 `[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`；
- 对每个 λ 生成的纯离线候选，先执行有限性、冻结范围、lineage、训练集 action 统计和成功轨迹保持审计；不得根据 core9 结果修改 λ 集合或权重。

训练前的可行性门槛固定为：至少一个 λ 候选同时满足：

- 6 条原成功轨迹上的 candidate→B500 raw-action MAE `<=0.002`；
- 3 条原失败轨迹上的 candidate→Teacher post-prior raw-action MAE 相对 B500 至少降低 `10%`；
- 任一关节 candidate→B500 raw-action absolute error q99 `<=0.02`；
- 所有输出有限，只有 actor.6 weight/bias 改变。

若无候选通过，可行性审计必须判定 R4 head-only 不可解，禁止强行训练或放宽阈值，并进入下述 R5；若有多个候选通过，按 `(失败 Teacher MAE 降幅, -成功 anchor MAE, -q99 drift, λ)` 固定排序。

#### v1.3-R4 行为门禁

离线安全候选按上述固定排序逐个执行 probe3，每个候选的 probe3 必须完整保存，最多评估 3 个候选：

- `valid=3/3`；
- seed11 nominal 必须 full/rear_hold；
- seed11 left-offset 必须 full/rear_hold；
- seed22 right-offset 必须至少达到 front-top-support；
- no severe inward=`3/3`。

第一个通过 probe3 的候选进入 core9。core9 要求 `valid=9/9、full>=7/9、rear_hold>=7/9、front>=8/9、no_severe>=8/9` 才允许成为 R4 中间候选；若直接达到最终 `8/9` 门槛则停止并执行 Teacher-action regression、lineage/schedule 复核、导出和三场视频。未达到中间门槛时，可继续评估剩余离线安全候选，但总数仍不得超过 3；所有候选均失败则 R4 永久停止并进入 R5。

#### v1.3-R5：仅在 R4 线性头不可解或全部行为失败时启用

R5 不改变网络结构和部署输入，只允许解冻现有 `actor.4 Linear(256,128)` 与 `actor.6 Linear(128,16)` 的 weight/bias 四个 tensor。R5 使用完全相同的固定 core9 replay、成功/失败 target 和 20:1 保持权重，不重新采集或改写数据。

R5 必须先写独立只读预注册，固定 Adam 为 `lr=1e-6、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、1 epoch、4 mini-batches、max_grad_norm=1`。执行顺序固定为 discarded 1-step smoke → 1 epoch probe3 → 3 epochs probe3 → 5 epochs probe3，且每个阶段从上一完整 checkpoint full-resume。任一阶段只要 nominal/left 原成功场景退化即永久停止；只有 probe3 满足与 R4 相同门槛才进入同规格 core9。5 epochs 是绝对上限，禁止追加或改名重跑。

R5 未达到最终门槛时，本轮 v1.3 机制永久结束，保存最佳 checkpoint 和全部证据，等待用户基于失败报告决定是否改变更高层 Teacher/观测/网络方向；禁止自动修改 action contract、Teacher、reward、prior、部署端或 sim-to-real。

#### v1.3 自动化与非静默要求

- R4 trace、replay dataset、可行性报告、候选 checkpoint、probe3/core9、R5（若触发）都必须有独立 SHA256、只读 manifest 和可恢复 stage result；
- 任何 trace 必须恰好 600 帧、9/9 矩阵完整、runner/manual parity、forward digest 和 post-prior decomposition 全部通过；不完整 trace 只能按基础设施故障重跑，完整但无效 trace 不得重跑；
- 自动 supervisor 不调用 Codex；若外部自动化未来调用 Codex，只允许 `gpt-5.6-sol + max`，禁止 `ultra`；
- systemd 对可恢复基础设施故障最多重试 3 次；spec/checkpoint/code/dataset SHA、冻结范围和行为门禁失败均不可按基础设施故障重试；
- state、heartbeat、lock、handoff 必须持续可读；任何终态都必须明确最佳 checkpoint、失败原因、候选状态和是否需要用户操作；
- 自动导出和录制获授权；禁止自动部署到真机或启动真机实验。

### v1.2：R2 失败后的最后一条证据化恢复路线

本节建立在 v1.1.1 R2 已正式 `stopped_by_gate` 的事实之上，并取代第 8--10 节中“不得再启动新路线”的冲突文字。其余永久动作契约、最终 `8/9` 门槛、导出/真机边界全部保持不变。

已完成且不得重跑的证据：

- B500：`valid/full/rear_hold/front/no_severe = 9/6/6/8/9`；
- R2-50：`9/6/6/6/9`，三个失败全部为 right-offset；
- R2-100：`9/2/2/2/9`，R2 硬门禁失败；
- R2-50 的三个 nominal 和三个 left-offset 全部完整上台，三个 right-offset 全部在前足支撑前失败；
- R2 的 latent MSE 从第 0 次约 `0.2005` 到第 100 次约 `0.1975`，没有实质改善；post-prior action MSE 虽降至约 `0.0615`，行为却从 6/9 退化为 2/9；
- R2-100 的成功动作变得更早、更激进且 roll rate 上升，证明 loss/reward 改善不能代表稳健上台进展；
- R2-50 相对起点的 estimator 参数和四个 box 输出行已发生全局漂移，继续训练同一机制会扩大横向场景退化。

R2-50 只读收尾证据固定为：

```text
checkpoint = /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_recovery_r2_Student/2026-07-13_03-37-24_student_recovery_r2_main_50_20260713_033718_2942949/model_49.pt
checkpoint_sha256 = e875424ed69eaaa474a9ac6849a6066dd264ce0eaf04ad63a418c3af1d66918c
core9_manifest = /home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v11_20260712/poststop_diagnostics/R2_50_core9/evaluation_manifest.json
core9_manifest_sha256 = b86c62d9c0fe95d17fa723fede4bbdfb1f42dcdc7afc5a7545dd97bfcde0da20
```

#### v1.2-R3：冻结 estimator 的横向门控全动作头 DAgger

唯一合法新路线为 R3。它不是 R2 的改名重跑：

- live 起点与冻结行为锚点均为上述 R2-50；受保护根仍为 `model_900`，B500 继续作为只读行为基准；
- Teacher 仍为唯一 `model_172300`，监督 target 仍为同状态 Teacher post-prior 最终 16 维动作；
- Student estimator、actor body、critic、privileged encoder、动作标准差全部冻结；
- 只训练现有 `actor.6 Linear(128,16)` 的完整 weight 和 bias 两个 tensor，不改变网络结构；
- 沿用现有 Robust Student on-policy 数据、现有地形/随机化和确定性 actor mean，不增加 sim-to-real 随机化或新的部署输入；
- 从 critic 已有 height scan 计算 `lateral_delta = left_scan_mean - right_scan_mean`，再计算 `offset_gate = smoothstep((abs(lateral_delta)-0.02)/0.06)`；该量只用于训练 loss 权重，不进入 policy/estimator/critic 输入；
- `teacher_offset_loss` 为 16 维 Teacher post-prior action MSE，按 `offset_gate` 加权；
- `anchor_center_loss` 为 live Student 相对冻结 R2-50 actor 的 16 维 action MSE，按 `1-offset_gate` 加权；
- 固定总损失为 `teacher_offset_loss + 4 * anchor_center_loss`，禁止 latent、velocity、PPO reward、KL、reconstruction 或手写关节目标；
- Adam 只拥有上述两个 actor head tensor，`lr=5e-6`、`betas=(0.9,0.999)`、`eps=1e-8`、无 weight decay、1 epoch、4 mini-batches、`max_grad_norm=1.0`、fixed schedule；
- fresh R3 必须 weights-only 加载 R2-50 并从空 Adam、R3 effective count 0 开始；只有带精确 R3 绑定的 checkpoint 才允许 full resume。

训练前必须写入只读 preregistration 和第二份只读 launch manifest，绑定 spec、R2-50、B500、model_900、Teacher、R2-50/B500/R2-100 core9、代码、配置和测试 SHA256。fresh4 + full-resume1 smoke 必须验证：

- optimizer 只拥有 actor.6 weight/bias 两个 tensor；
- estimator 和其余全部 Student/Teacher/anchor tensor byte-identical；
- deterministic action、Teacher post-prior parity、24→0 rollout buffer、Adam step/count 连续性全部通过；
- lateral/center gate 均有有限样本，gate 值有限且位于 `[0,1]`。

R3 固定执行顺序与门禁：

```text
static -> fresh4/full+1 discarded smoke
-> 10 effective updates probe3
-> 25 core9
-> 50 core9（仅在 25 通过时）
-> 100 core9（仅在 50 严格改善时，绝对上限）
```

- probe3：3/3 valid；seed11 nominal 与 left-offset 必须继续 full/rear_hold；front support >=2/3；no severe=3/3。probe3 不能晋级为候选；
- R3-25：`valid=9/9`、`full>=7/9`、`rear_hold>=7/9`、`front>=7/9`、`no_severe>=8/9`，否则立即永久停止 R3；
- R3-50：必须通过上述门槛，且行为 rank `(min(full,rear_hold), full+rear_hold, front)` 严格优于 R3-25；否则停止；
- 只有 R3-50 尚未达到最终门槛但 rank 严格改善时才允许 R3-100；
- R3-100 是绝对上限，未达到最终门槛即停止，不得追加 iteration、改名重跑或修改阈值；
- 任一 core9 达到 `valid=9/9、full>=8/9、rear_hold>=8/9、no_severe>=8/9`，立即停止训练并执行同状态 Teacher-action regression、lineage/schedule 复核、导出和三场视频；
- 行为最佳选择集合必须包含 B500、R2-50 以及所有已执行 R3 core9；按 `(min(full,rear_hold), full+rear_hold, no_severe, front, -effective_updates)` 选取，不默认最后 checkpoint。

R3 失败后没有第四路线；保存全部证据并停止等待新的用户审核。

#### v1.2 自动恢复与非静默监督

- 独立 systemd supervisor 不依赖 Codex 在线；
- 每个 train/eval 原子阶段先写 launch，再写阶段结果，重启后只复用 SHA、checkpoint audit 和完整矩阵均通过的结果；
- 可恢复基础设施故障使用专用退出码并由 systemd 最多连续重启 3 次；每次保留原日志和 attempt 目录；
- checkpoint/spec/code 绑定错误、行为门禁失败和完整但无效的 core9 均不可重试；
- systemd 必须区分可恢复退出与永久 fail-closed，禁止无限重启；
- state、heartbeat、lock、handoff 必须持续可读；终态必须明确候选/门禁失败/绑定故障和用户是否需要操作；
- 禁止通过 shell 命令字符串误判并发进程；直接脚本导入必须由隔离环境测试覆盖；
- 所有预注册保存点必须及时评估，不能只看最后 checkpoint；
- 自动化中的任何 Codex 决策（若存在）固定 `gpt-5.6-sol + max`，禁止 `ultra`；本 supervisor 默认不调用 Codex。

### v1.1 修订原则

- 最终验收条件和永久动作契约不变；
- v1.0 的 B/C/A 固定路线不再是必须走完的流程；
- 中间蒸馏机制、冻结范围、训练数据和优化方式可依据可复现实验证据替换；
- 禁止把“继续运行”、loss 下降或局部指标改善冒充完整上台进展；
- 每个新方案必须预注册假设、唯一变量、预算和停止线，失败立即淘汰；
- 本修订中的第 6--10、13--14 节取代 v1.0 对应条款；发生冲突时以 v1.1 为准。

### v1.1.1 一次性收尾修订

- 用户不批准修改旧 B300 门禁后直接续跑 B500；
- 但批准在进入 v1.1 同状态审计和 R1/R2 之前，对 corrected B300 执行一次参数完全不变、预算严格封顶的恢复验证；
- 第 6.1 节是旧两行 hip 路线唯一一次例外，取代与之冲突的“禁止续跑 B500”条款；
- 第 6.1 节结束后，除非达到最终候选门槛，否则必须回到第 7 节同状态审计；不得再次恢复、改名或扩展旧路线；
- 本次阈值和分支必须在训练前写入预注册 manifest，看到结果后不得修改；
- v1.1 已开始但尚未执行的审计代码与独立 workflow 状态作为安全断点保留，不回滚、不删除。

## 1. 唯一目标

候选 Student 必须做到：

- 两只后足进入高台顶面并真实承载；
- 机身进入平台；
- 连续稳定保持；
- 后腿不得持续向中线塌缩。

左前腿最终是否落回平台只作诊断。整机上台后允许人工切换 fixed stand。

## 2. 永久锁定项与本轮可变项

以下内容禁止自动化修改：

- `action_scale`；
- `joint_pos.clip`；
- `default_dof_pos`；
- 观测维度/顺序、动作维度/顺序和关节顺序；
- Teacher action prior 的结构、参数和方向；
- 网络主体结构；
- 部署端控制器、gains 和 guard；
- 新增 sim-to-real 随机化；
- Teacher reward 或训练任务。

本轮允许依据第 7 节审计结论修改：

- 现有 Student 网络中哪些参数可训练，包括 estimator、actor 末端层或必要的现有 actor block；
- 蒸馏 target、loss 组成、阶段权重和 baseline 行为保持项；
- on-policy 数据采集、DAgger 式 Teacher 标注和采样分布；
- optimizer、学习率和每轮有效更新数；
- 仅用于训练标注的 privileged 阶段标签，但部署 policy 不得新增输入或依赖 privileged 信息。

本轮不允许重训 Teacher、修改 reward、改变网络结构或新增 sim-to-real 项。只有第 10 节定义的两条证据化路线均失败后，才可由用户另行批准新的规范修订；不得由自动化自行解锁。

FL hip 超限、target-limit、slip、roll/yaw 和 effort 仅作诊断，不得凌驾于完整上台行为。

## 3. 固定基准与术语

### 3.1 Teacher 基准

唯一可靠 Teacher：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt
```

- action prior 始终开启；
- checkpoint 和 prior 永久保留、不覆盖；
- 标准与 real-gain core9 均已达到 `9/9`；
- 禁止再次训练 no-prior Teacher。

### 3.2 Student 恢复起点

新一轮恢复的受保护起点：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt
```

历史表现约为：

- `pass = 3/9`；
- `full climb / rear hold = 4/9`。

“Student NoPrior”只表示部署时不额外运行依赖高度扫描的 prior 程序；Student 必须学习 Teacher 加完 prior 后的动作效果。

## 4. 双 checkpoint 强绑定

R1/R2 训练与第 7 节同状态审计必须同时绑定：

```text
Student 受保护根 checkpoint = model_900
当前路线起点 = model_900 或第 8 节允许的行为最佳合格 checkpoint
Teacher 监督目标 = model_172300
```

要求：

- Student actor、estimator/VAE 从当前路线起点加载，且 lineage 必须可追溯到 `model_900`；
- frozen Teacher actor 和 privileged encoder 从 `model_172300` 单独加载；
- 两个 checkpoint 的绝对路径和 SHA256 写入 manifest；
- 禁止把当前 Student actor 复制成 Teacher；
- Teacher target 绑定失败立即终止；
- parent manifest 只能证明谱系，不能代替真实 Teacher 权重加载。

这是正确性修复，不计作新的训练变量。

## 5. 启动前审计

任何参数更新前必须：

1. 锁定并记录代码版本、checkpoint SHA256、任务和评估配置。
2. 若代码、任务、评估配置和 checkpoint SHA256 与第 6 节已保存 baseline manifest 完全一致，直接复用该 core9；任一绑定发生变化时才重新评估 `model_900`。
3. baseline 至少复现 `pass >= 2/9`、`full climb / rear hold >= 3/9`；低于下限时停止排查加载或环境差异，禁止用训练“补回来”。
4. 验证 Teacher target 确实来自 `model_172300`。
5. 验证所有锁定动作 contract 与原 checkpoint 一致。
6. 验证新 checkpoint 能保存本轮 optimizer、更新计数和完整恢复状态。

## 6. v1.0 路线结论与新一轮起点

v1.0 中“仅训练 RL/RR hip 最终两行”的路线不得因为局部几何或 MSE 改善而默认续跑。新一轮启动前必须：

1. 温和结束或完成当时唯一在运行的只读评估，禁止并发 train/eval/play；
2. 保存并汇总 `model_100`、`model_200`、`model_299/300` 的完整 core9；
3. 只有某个 checkpoint 的完整上台和 rear hold 不低于 `model_900` baseline，才可把它列入候选起点；
4. 若所有 checkpoint 均低于 baseline，则正式判定“只训练后 hip 两行”机制失败，保护 `model_900` 为恢复起点，不得续跑 B500；
5. 旧 checkpoint、optimizer、日志、manifest 和失败原因全部保留，不覆盖、不删除。

截至 2026-07-12 23:26 HKT，遗漏的中间 checkpoint core9 已补齐，结果为：

| checkpoint | valid | full climb | rear hold | front top support |
|---|---:|---:|---:|---:|
| `model_900` baseline | 9/9 | 4/9 | 4/9 | 7/9 |
| corrected `model_100` | 9/9 | 0/9 | 0/9 | 1/9 |
| corrected `model_200` | 9/9 | 0/9 | 0/9 | 3/9 |
| corrected `model_299`（300 effective updates） | 9/9 | 3/9 | 3/9 | 7/9 |

因此 v1.0 的“两行 hip 蒸馏”机制按 v1.1 原判定为失败，`model_900` 仍是后续 R1/R2 的唯一受保护起点。除第 6.1 节新批准的一次性严格封顶收尾验证外，禁止续跑 B500。主代理只需校验原始 manifest 与上表一致，不得重复执行这些 core9。

证据位置：

- baseline：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/baseline_core9_retry2/evaluation_manifest.json`；
- corrected B100/B200：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/diagnostics_corrected/`；
- corrected B300：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/evaluations/B300_20260712_224547/evaluation_manifest.json`。

旧 supervisor 中任何比本规范更宽松或更严格的未定义门槛均不得直接沿用；必须先完成 code-to-spec 差异审计。

### 6.1 v1.1.1：corrected B300 的一次性严格封顶验证

本节优先于第 6 节中与“禁止续跑 B500”冲突的文字，但不解锁任何其他旧路线修改。唯一合法恢复点为：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-12_22-23-52_student_recovery_B300_20260712_222346/model_299.pt
```

其 SHA256 必须为：

```text
5dfb5f36a06e155d97eb4da84def2d91802d6749a5944c72166246257bd147f7
```

启动前必须验证：

- 这是共享 ELU / actor-prefix 修复后的 corrected B300，而不是 20:06 的错误 B300；
- checkpoint 中 `effective_update_count = 300`；
- 原 Adam state、参数顺序、step 和 moments 完整且有限；
- schedule 从 checkpoint-paired runtime state 精确恢复；
- Student 根 checkpoint 仍为受保护 `model_900`，Teacher 仍为独立 `model_172300`；
- 双 checkpoint 路径、SHA、component hash 和 no-shared-storage 检查全部通过；
- `list(actor)` 的共享 ELU/actor-prefix 数值等价检查继续通过；禁止重新使用会去重共享 ELU 的 `actor.children()`。

本次唯一变量是从 corrected B300 **追加 200 effective iterations**，到总计 B500。必须完整恢复原 optimizer 与计数，禁止 weights-only、fresh optimizer 或从 `model_900` 重开旧路线。

以下内容全部锁定且不得修改：

- Teacher；
- actor learning rate；
- loss 和 target；
- 仅 RL/RR hip 最终两行可训练的冻结范围；
- reward、Teacher action prior、网络结构；
- `action_scale`、`joint_pos.clip`、`default_dof_pos` 和全部动作/观测/关节 contract；
- schedule 语义和训练数据流程。

B500 后执行与既有 baseline/corrected B300 完全相同规格的 core9，不得重复旧 core9。定义：

```text
P = min(full_climb_count, rear_hold_count)
```

B500 core9 同时必须报告并检查：

- `valid = 9/9`；
- `front_top_support >= 7/9`；
- `first_rear_top >= 4/9`；
- `no_severe_inward >= 8/9`。

训练前固定的唯一决策表：

- `P < 4`：旧路线永久失败，禁止继续；恢复第 7 节 v1.1 同状态审计；
- `P = 4`：只恢复 baseline，没有行为提升；禁止 B1000；恢复第 7 节 v1.1 同状态审计；
- `P >= 5`：证明出现真实完整上台行为提升，才允许用**完全相同机制**从 B500 完整恢复并追加 500 effective iterations，到总计 B1000。

若 B500 的 `valid/front_top_support/first_rear_top/no_severe_inward` 任一伴随条件不满足，即使 `P >= 5` 也不得进入 B1000，必须停止旧路线并恢复第 7 节审计。

B1000 后必须执行同规格 core9：

- `P >= 6` 是最低有效门槛；
- 只有同一 checkpoint 同时达到第 11 节最终 `8/9` 候选门槛，才进入候选复核、导出和视频；
- 若未达到最终 `8/9`，无论 `P` 是小于 6 还是 6--7，均停止旧路线并恢复第 7 节审计；
- B1000 是旧路线绝对上限，不再追加 iteration。

本节所有阈值、checkpoint、SHA、唯一变量和分支必须在 B500 启动前写入不可事后修改的预注册 manifest。任一恢复绑定不成立时 fail closed，不得用重新训练或重建 optimizer 代替精确恢复。

## 7. 训练前的同状态证据审计

新参数更新前必须在同一批仿真状态上同时运行固定 Teacher 和 Student，至少覆盖：

- approach；
- 前足抬起与顶面支撑；
- 第一只后足上台；
- 第二只后足上台；
- rear hold 或对应失败阶段。

审计必须输出：

- Teacher post-prior 最终 16 维动作与 Student 16 维动作的逐关节误差；
- estimator/latent 差异与速度估计误差；
- 每个行为阶段的 bias、MAE、符号错误和饱和/裁剪情况；
- 失败轨迹最早出现的可复现分歧；
- 必要时最多执行固定三场的小规模因果替换测试：只在诊断 play 中将一个预注册动作组替换为 Teacher post-prior 输出，判断该动作组是否真正恢复后足上台；诊断结果不得作为最终候选。

审计结束后必须生成一份短报告和机器可读 manifest，并据此只选择第 8 节的一条路线。不能证明根因时不得训练。

## 8. 允许的两条证据化恢复路线

每条路线都从行为最好的合格 checkpoint 开始；若第 6 节没有中间 checkpoint 保持 baseline，则从受保护的 `model_900` 开始。Teacher 始终为 `model_172300`，监督 target 始终采用同状态下 Teacher post-prior 最终动作。

### 8.1 路线 R1：行为保持的定向 DAgger 蒸馏

仅当第 7 节证明误差集中在有限动作组或有限行为阶段时使用：

- 使用 Student on-policy 状态并由 Teacher 同状态标注；
- 只训练审计证明必要的现有输出行或 actor 最后层；
- 使用训练期阶段标签加权目标，但部署时不增加任何输入或规则；
- 对 approach 和已成功的前足阶段加入相对起点 checkpoint 的行为保持约束，防止修复后腿时破坏前半段；
- 不得以手工固定外展量、部署 guard 或 play-time prior 代替学习。

### 8.2 路线 R2：latent 与动作联合蒸馏

仅当第 7 节证明主要问题来自 Student estimator/latent，或误差跨越多个动作组、R1 因此不适用或已失败时使用：

- 允许训练 Student estimator 和审计证明必要的现有 actor block；
- 同时使用 Teacher privileged latent/可比表征监督和 Teacher post-prior 动作监督；
- 保留起点 checkpoint 的 approach/前足行为约束；
- 网络结构、部署输入和动作 contract 保持不变。

自动化最多依次执行 R1、R2 各一次；审计明确排除某条路线时必须跳过，不得为了凑次数启动。禁止增加第三条试验分支。

## 9. 每条路线的预注册与硬门禁

启动任一路线前，manifest 必须预先写明：

- 根因假设及第 7 节证据；
- 起点 checkpoint、Teacher、代码和配置 SHA256；
- 唯一训练机制、精确可训练 tensor、target 和 loss；
- optimizer、学习率、数据来源和最大更新预算；
- 不得在看到结果后修改的晋级与停止条件。

执行顺序：

```text
static audit
-> 1--5 iteration smoke
-> 最多 50 iteration 的 probe3 灾难性退化筛查
-> 100 effective iterations core9
-> 300 core9
-> 500 core9
-> 必要时最多 1000 core9
```

门禁：

- probe3 只允许提前淘汰明显崩溃方案，不能用于晋级或候选验收；
- 100 时必须 `valid=9/9`、完整上台和 rear hold 均不低于 baseline `4/9`、前足顶面支撑不低于 baseline `7/9`，否则立即淘汰；
- 300 时完整上台或 rear hold 必须至少达到 `5/9`，且严重后腿内收不得比 baseline 增加，否则立即淘汰；
- 500 时完整上台和 rear hold 必须至少达到 `6/9`，否则立即淘汰；
- 只有行为指标逐级改善才可到 1000；loss、latent loss、几何、卡边或宽度改善不能抵消完整上台退化；
- 任意 checkpoint 达到第 11 节门槛，立即停止训练并进入复核；
- 选择行为最好的 checkpoint，不默认选择最后一个；
- 同一门禁失败后禁止改名重跑、放宽阈值或追加 iteration。

## 10. 路线失败后的停止线

- R1 失败：只有第 7 节证据支持 R2 时才可自动进入 R2；
- R2 失败，或审计不能支持 R1/R2：保存全部证据并停止，明确报告“本轮允许的 Student recovery 机制未达到目标”；
- 停止后不得自动修改 Teacher、reward、prior、网络结构、部署端或 sim-to-real；
- 若仍要继续，必须由用户审核失败证据并批准下一版规范；不得用“再试一次”代替新证据。

## 11. 最终 Student 候选门槛

同一个 checkpoint 必须在 real-gain、delay=0、关闭 play-time 随机事件的 Robust Student core9 中满足：

- 9 个运行全部有效；
- `full climb >= 8/9`；
- `rear-on-platform stable hold >= 8/9`；
- 至少 8/9 不发生严重后腿内收；
- Student 相对 Teacher 不出现明显动作退化；
- schedule、checkpoint 和 parent lineage 全部有效。

左前足是否最终落下、FL hip/target limit、effort/torque、严格接触时序、slip、roll 和 yaw 仅作诊断，不得单独否决。

## 12. 导出与视频

达到候选门槛后自动：

- 导出 Student policy；
- 保存 evaluation manifest；
- 记录 checkpoint、Teacher、代码和配置 SHA256；
- 保存完整 core9 原始结果；
- 保存 nominal、left offset、right offset 视频；
- 不得只保留成功样本，失败场景也必须标注保存；
- 生成重复上台动作的汇总视频。

允许自动导出和录制；禁止自动部署或启动真机，真机测试必须等待用户审核。

## 13. 自动化权限

规范批准后，自动化只允许：

- 完成本文明确要求的正确性代码修复；
- 第 7 节同状态审计和限定的诊断因果替换；
- static audit、smoke 和 probe3；
- R1/R2 中由审计选择且已预注册的训练；
- 依据门禁自动延长或停止；
- core9 评估；
- 导出、录制视频和写入可验证 handoff。

禁止自动化：

- 启动 R1/R2 以外的第三条机制；
- 重训 Teacher；
- 修改 prior、reward 或网络结构；
- 修改锁定 action contract；
- 修改部署端；
- 加入新 sim-to-real 项；
- 因 loss 好看而越过行为门禁；
- 在失败后以“再试一下”、改名或放宽事后门槛为由启动新分支。

任一路线失败时保存 checkpoint、日志、manifest 和失败原因。若规范允许的下一路线有充分证据则继续，否则立即停止等待审核。

## 14. 运行与监督要求

- 一次只允许一个 train/eval/play；
- 训练后台运行并持续 heartbeat；
- 不使用固定 600 秒无条件杀死正常评估；
- watchdog 只能在持续无进展、进程异常或资源故障时终止；
- 每阶段报告唯一修改变量、起点、Teacher、有效更新数和门禁结果；
- 监督输出必须清楚显示当前阶段、checkpoint、core9 和停止原因。
- 训练或评估停止后，主代理必须立即向用户报告：是否达到目标、为何停止、当前最佳 checkpoint、用户是否需要操作以及下一项动作；禁止静默停止。
- 监督器必须评估所有预注册保存点并选择行为最好 checkpoint，禁止只评估最后一个 checkpoint。

## 15. 冲突处理顺序

1. 永久锁定 action contract；
2. 真实完整上台行为；
3. 后腿防内收；
4. 阶段门禁；
5. 训练 loss 和其他诊断指标。
# v1.4.1 amendment：后续自定义训练的强制 W&B 可观测性门禁

本节在 v1.4 `directional_real_robot_trial_candidate` 已写入终态 handoff 后生效。它只增加后续训练的可观测性门禁，不改变、不重判也不要求重跑 v1.4 已完成的行为评估、Teacher-action regression、视频、导出或候选交付。上述证据继续绑定 v1.4 spec SHA `1fbb17ab6c23a24005e8e34dadc11a6e4592a7be671a575206fd24819e5f3f1c`；新旧 authority 的关系必须由只读 amendment/rebinding manifest 记录。

后续每个自定义 B、R、R5 训练 stage/attempt 必须分别建立独立 W&B run，禁止复用 run id。run name 必须同时包含 `workflow_id`、route、stage、attempt，且 `group=workflow_id`。开始训练前写入的 W&B config 至少包含：本版 spec SHA、对应 preregistration SHA、起点 checkpoint 路径和 SHA、冻结 Teacher 路径和 SHA、冻结/可训练张量范围、优化器完整配置、iteration/epoch 预算与训练 task。

训练与评估完成后，W&B summary 至少包含：输出 checkpoint 路径和 SHA、effective updates/epochs、probe/core9 或该路线等价行为门禁指标、门禁结论、训练/评估 manifest 路径及 SHA。历史补传必须设置 `historical_sync=true`；新训练必须为 false。

网络或 W&B 服务暂时失败时允许训练在完整本地日志、checkpoint 与只读 stage manifest 保护下完成，状态必须显式为 `pending_sync`，不得声称上传成功。进入下一训练 stage 前必须先将本 workflow 所有 `pending_sync` stage 同步并远端核验为 `synced`；同步失败必须 fail-closed，禁止静默跳过。评估、导出和视频不是新的训练 stage，不需要各自新建 W&B run。

已确认的 B500 历史同步记录为：project `xinqili551-the-university-of-hong-kong/isaaclab`，run id `b500best`，name `highstep_student_B500_behavior_best_20260713`，URL `https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/b500best`，checkpoint SHA `f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159`，且必须标记 `historical_sync=true`。
