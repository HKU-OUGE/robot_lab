# Highstep 0707 真机数据复盘、训练链审计与整改报告（2026-07-12）

状态：分析结论已由两组独立视频/rosbag 互证；schema6、schedule/checkpoint provenance、Student parent lineage 与部署端离线整改已在 2026-07-12 02:02 完成终审回归。正式训练仍停止。本文是技术报告，紧急恢复入口仍是
[highstep_0707_real_log_audit_handoff_20260712.md](highstep_0707_real_log_audit_handoff_20260712.md)。

> 2026-07-12 05:59 后续决策覆盖：本文 02:02 的“正式训练仍停止”和第 173 行“训练 clip 已改成物理 envelope”已被用户后续批准的主线覆盖。当前 Robust Student 正式训练已启动；用户明确锁定并禁止修改 `joint_pos.clip` 与 `action_scale`，训练/评估 action target clip 已恢复为兼容旧 checkpoint 的 `±60` 语义。物理 envelope 只用于部署端强制安全保护与评估只读审计，不得把旧结论误读为可以修改训练 clip。最新实时状态与门槛只看上面的紧急恢复入口。

## 1. 结论先行

0707 真机问题不是单一的“后腿收窄”，而是四条相互叠加的失效链：

1. policy 158797 在前足碰台、后足迁移阶段反复把后髋/后足推向中线，形成单侧支撑、roll/yaw 摆动和后足卡边；
2. policy 请求的 FL hip 与部分 calf 目标长期超出机器人描述中的关节范围，旧训练与部署的 `±60` target clip 实际没有提供物理保护；
3. policy1/policy2 切换时控制器会连续约 59--60 ms 发布 `q_des=0, Kp=0, Kd=0`，两次独立测试均复现；
4. 旧自动化把 checkpoint iteration、环境 schedule 和严格评估分开计时，导致续训/评估进程重启后 action prior、terrain stage 和 support bottleneck 回到起点，clean B 与旧 strict eval 均被混杂。

因此，单独继续 rear-width reward、单独增加 action delay、或只在真机端 clamp 都不足以闭环。正确顺序是：统一训练/评估/部署 target contract，修 policy 切换交接，恢复全局 schedule 连续性，再用能直接卡住接触时序、滑移、卡边和 raw-target 越限的新监督器评估。

当前训练保持停止。唯一可靠的旧权重断点与审计锚点是：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/
  2026-07-11_23-45-31/model_600.pt
SHA256: 3515ee8aa29b5e91ca7813798cbfe6b820895d0e87217f331fe790682fe6d94c
```

它已不再是可直接 full/preserve 续训的安全 parent：schema6 600-step 预检确认严格事件链失败，且 raw target 越限率 `27.44%`、越限步率 `97.67%`、最大超限 `0.541 rad`。后续若启正式分支，只能把它作为显式 `weights_only + migration + schedule reset` 的权重来源。

## 2. 数据、真值优先级与证据边界

第一组：

```text
video: /home/lxq/Videos/histep_real_robot_test_0707.mp4
bag:   /home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_52_07
```

第二组：

```text
video: /home/lxq/Videos/histep_real_robot_test_2026_07_07-00_28_27.mp4
bag:   /home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_28_27
```

两段视频和 bag 来自不同机器，绝对时钟不能直接相减。本报告用 joystick 动作起点、视频/音频冲击、raw IMU 峰和可见动作序列做相对时间配准。policy 身份和实际 gains 以 rosout、`/joint_commands` 为最高优先级；用户提供的 YAML 与本机当前 config 只作为历史参考，不能覆盖 bag 的 wire-level 记录。

两份 bag 都缺少直接足端接触力、base odometry、外部安全绳拉力和电机底层诊断 topic。因此：

- “碰台/上台/承载”由视频、音频、IMU 和 TF 联合推断，不伪装成直接力传感器测量；
- TF 足端位置能证明几何收窄，不能单独证明承载接触；
- 两段视频中安全绳都被操作员持握，爬升后段存在明显辅助，最终到达台面不能记作独立 policy 成功；
- qdes→qactual 相位滞后包含 PD、机械和接触动力学，不能等同于纯推理延迟。

## 3. 视频与 bag 对齐

### 3.1 第一组剪辑视频

第一组 bag 尾部截断且无 metadata；原始文件保持只读。通过 B-tree 扫描和派生 `.recover` 库获得连续的 `3,424,259` 条消息，message ID 1..3,424,259 无逻辑缺口。原始视频 12.933 s，只包含爬台片段。

配准结果：

```text
bag_relative_time = video_relative_time + 46.086 s
保守不确定度：±0.025 s
```

关键锚点包括视频 2.185 s 的首次前足撞击、3.687/3.991/4.570 s 的连续冲击、5.882/6.016 s 的后段重组，以及 10.731 s 的后足卡边再冲击。policy2 在视频开始前约 5.624 s 已完成切换，所以第一段画面中的主要失败不是切换瞬态；切换掉控是独立且可复现的部署缺陷。

### 3.2 第二组较完整视频

第二组 bag SQLite quick check 正常，视频长 30.267 s，包含切换后静止、前进、完整爬升和退出 policy2。

配准结果：

```text
bag_relative_time = video_relative_time + 55.688 s
保守不确定度：±0.030 s
六个强冲击锚的 offset 范围：55.6786..55.6988 s
```

policy2 switch 位于视频开始前约 0.701 s；首个持续向前 joystick 命令约在视频 9.75 s；首次强前缘碰撞簇约 12.62 s；清晰前足上台约 13.35 s；17.13--17.75 s 出现大幅失稳和连续冲击；20.75--21.57 s 后髋/足端收窄达到全段最严重；约 25 s 后在安全绳持续辅助下进入台面稳定段。

## 4. 两组独立复现的量化证据

| 证据 | 第一组 | 第二组 | 判断 |
|---|---:|---:|---|
| policy2 | 158797 | 158797 | 同一策略 |
| 实测 Kp（hip/thigh/calf/box） | 55/65/80/2 | 55/65/80/2 | 完全一致 |
| 实测 Kd（hip/thigh/calf/box） | 1.5/1.5/2.5/3 | 1.5/1.5/2.5/3 | 完全一致 |
| 切换全零控制窗 | 约 59.3 ms | 约 59.0 ms；旧命令到首有效约 60.1 ms | 稳定复现 |
| FL hip qdes max | 2.528 rad | 2.550 rad | 均远超 ±1.222 rad |
| FL hip actual max | 2.232 rad | 2.259 rad | 实际关节也越界 |
| FL hip qdes 越限率 | 约 42.4% | 交互窗约 34.4% | 非偶发尖峰 |
| 后足最靠近中线 | RR abs(y) 约 0.0216 m | min abs(y) 约 0.0225 m | 几乎相同 |
| 后足宽度最低 | 约 0.261--0.268 m（首碰/后段） | 全段最低 0.149 m | 第二组后段更严重 |
| 转动关节相位滞后 | 多数 36--58 ms，离群到 97 ms | 38--77 ms，中位约 48.5 ms | 响应尺度复现 |
| box 相位滞后 | 约 85--107 ms | 约 81--94 ms | 机械响应较慢 |

第一组 high-rate TF 的全片单点 `min_abs_y` 最低为 `0.00123 m`（视频约 3.749 s），但它是短暂越中线：全片 `<0.04 m` 累计约 `0.088 s`、最长连续约 `0.049 s`，visible q05 为 `0.0767 m`。既有 50 Hz 表中的 `0.02162 m` 是同一类短瞬态的低采样视图。第二组最低 `0.02246 m`，`<0.04 m` 累计约 `0.278 s`、最长连续约 `0.276 s`，visible q05 `0.0671 m`，说明第二组的危险低间隙更持续。两组首触窗 q05 很接近（约 `0.0747/0.0735 m`），所以“首触时向中线逼近”是稳定复现，而不是依赖某一个极小单帧。

第二组提供了重要的时间因果区分：首次前缘接触时后足已经收窄；FL calf 有一段短暂 unsafe target 从视频约 12.413 s 开始，RR/RL calf unsafe target 约从 13.387/13.460 s 开始，而幅度更大的 FL hip 越限约从 17.566 s 才开始。因此，FL hip 越限不是第二组“首次碰撞”的根因；calf target 错误已经可能放大初碰/恢复，FL hip 饱和则与后段大幅失稳、卡边和反复冲击同期。后髋/足端中线塌缩从首次接触阶段开始，并在后段进一步恶化。

切换掉控也不是推测。第二组记录为：最后旧 fixed-stand 命令 `55.005786 s`，全零 qdes/Kp/Kd 从 `55.006953` 到 `55.064080 s`，首个有效 RL 命令 `55.065913 s`；随后 raw IMU 加速度范数从 `0.431` 跳到 `30.459 m/s²`。第一组出现相同命令模式和约 `35.3 m/s²` 冲击。

## 5. 从 0707 失败到当前训练的完整工作链

### 5.1 修改链及重新判定

| 阶段 | 原意 | 加入真机数据后的判断 |
|---|---|---|
| Student 158797→159900 rear-hip refine | 抑制后髋内收 | 方向局部有效，但没有部署验证，也不覆盖 FL hip/calf 越限、切换掉控和支撑时序；不是当前 lineage 父节点 |
| Teacher 151200→155600 rear-width | approach/motion width + DR | sim width 改善有效，但只能证明几何代理改善，不能证明真机承载稳定 |
| Student 155600→159700 | 继续 Student-first | score 更差，已正确放弃；不能因后足宽度局部变好而恢复为主线 |
| SWAP-inspired 155600→160300 | bilateral contact、slip、roll/yaw、rear lag/support width | 目标方向正确；旧 row pass 没把这些设硬门禁，因此旧“通过”不能作为安全结论 |
| centerline 160300→165999 | 修正中心线 hard risk/curriculum gate | 部分有效，pass 0→约 0.133；instant min_abs 仍可近 0，未消除塌缩 |
| reward alignment 165999→167998 | 真正把 centerline gate 乘入 ActionScore | 有效代码修复；pass 约 0.2、rear-edge dwell 仍约 551，未解决卡边 |
| post-clear latch | 第二后足 clear 后恢复推进 | 单独无效；实际瓶颈常发生在第二后足 clear 之前，触发过晚 |
| bilateral advance / stall penalty | 缩短单后足卡边 | 中间 checkpoint 把 pass 提到约 0.733、平均 dwell 降至约 113，但 nominal worst 仍约 533--548；改善均值，未消除确定性坏模态 |
| delay 0/1 mix | 加强时延鲁棒性 | pass/平均 dwell 反而变差，且 delay1 与其它扰动绑定；实验混杂，不能判有效 |
| Robust Teacher persistent encoder bias | 覆盖真机 observation bias | 只覆盖 encoder position bias；不覆盖 PD drift、IMU bias、执行器响应或切换掉控，覆盖面有限 |
| clean B optimizer reset | 与原优化器状态做 A/B | 同时重置 action prior、terrain/support schedule，已不再是 clean A/B；暂停正确 |

### 5.2 仍然成立的修改

- Teacher→Robust Teacher→只使用真机可得观测的 Robust Student 的大方向仍成立；
- rear width/centerline、bilateral support、slip、roll/yaw、rear-lag 都是必要代理；
- reward alignment 和 supervisor run-name 基于 task/role/manifest 的匹配是有效工程修复；
- delay0 与 delay1 应保留为独立场景，但不能把机械相位滞后整体建模成 action delay；
- post-clear recovery 可作为后段辅助目标，但不能承担首个后足/第二后足时序的主要责任。

### 5.3 已失效或证据不足的判断

- “full climb success” 不等于安全爬升；旧定义允许仅两个接触点，也不硬限制 slip、bilateral support、raw target 越限；
- “rear width 改善”不等于真机问题已修复；第二组说明最严重塌缩可能出现在后段；
- 旧 strict eval 并不严格：play 新进程没有恢复 checkpoint schedule，600 step 评估中 action prior 基本为 0；
- clean B 不能回答 optimizer reset 是否有效；它同时改变了多个 schedule；
- 第二组最终上台不能记作 policy 独立成功，安全绳从开始到结束持续参与。

### 5.4 可能带来副作用的修改

- support penalty 若从首次前足接触立即全强度生效，会惩罚自然的顺序落足，诱导僵硬或撞击式双足同步；需要短 grace 和平滑 ramp；
- rear-width/centerline 权重过强会得到“站得宽但不推进”的局部最优，并掩盖前腿目标越限；
- 只在部署端 clamp 会让 policy 长期撞限，虽然避免命令越界，却会造成 sim-to-real 动作失真；训练、评估、部署必须同一 contract；
- 只增加 delay 会重复建模 PD/机械响应，可能把 policy 推向更激进的预测动作；
- 只奖励 post-clear recovery 会纵容策略先危险撞边，再尝试后段补救。

## 6. 根因到整改的对应关系

```text
进程本地 schedule ──→ prior/terrain/support 被重置 ──→ 全局 checkpoint clock + manifest
训练/部署 ±60 target ─→ raw target 长期越物理范围 ──→ 同一 joint target contract + raw violation gate
StateRL 初始空命令 ──→ 切换后约 60 ms q/Kp/Kd 全零 ─→ current-q hold + command-valid/epoch 原子交接
旧 full-climb 判定 ───→ 卡边/滑移/单侧支撑仍可 pass ─→ 分段 dwell + bilateral/slip/limit 硬门禁
配置与文件频繁切换 ─→ model_name/config 无法追溯 ───→ resolved policy key/path/fingerprint + runtime manifest
```

## 7. 新的训练与评估原则

1. action term 的 clip 必须作用于 affine-mapped joint target，并与部署端安全 envelope 一致；不能只 clip network raw action；
2. reward 既惩罚 raw target 超界，也记录 actual-limit violation rate、最大超界量和连续超界步数；安全 margin 可用于软惩罚，但不能和“实际越界”统计混为一列；
3. eval 必须恢复 checkpoint 的 global update，并在每条 JSON 中输出 schedule 来源、update、prior scale、terrain stage/max level、support blend；缺失即 invalid；
4. 爬升时序至少拆成：front bilateral support→first rear top、first→second rear top、second rear top→rear advance；
5. 前足初触允许短 grace，之后 bilateral front/rear top support、top-contact slip、rear-edge dwell、roll/yaw 和 raw-target 越限均做 row-level hard gate；
6. nominal delay0、nominal delay1、左右偏置、快速命令和 random-force 独立组成场景，不再把 delay1 绑定到所有扰动；
7. 任何导出/部署候选必须绑定 parent checkpoint SHA、schedule manifest、target-limit contract、policy 文件 fingerprint 和训练 gains/DR 配置。

## 8. 已实施整改与验证状态

本节只登记已经落盘且验证过的内容；未通过测试的修改不写成“完成”。

### 8.1 已完成

- 自动 controller run-name 验证改为 task root + role + live argv/handoff，不再要求固定日期或时间戳 basename；新 run 名带任意 suffix 仍可匹配；4 个命名单元测试通过。
- 监督器评估矩阵改为 6 个独立场景 × 3 seeds：nominal delay0、nominal delay1、left、right、fast、random-force；不再把 delay1 与全部扰动混在一起；现有可靠性测试通过。
- highstep 训练 action clip 已改为统一 target contract：hip `[-1.22173, 1.22173]`、thigh `[-1.5708, 3.4907]`、calf `[-2.77507, -0.64577]`、box `[0, 0.06]`。
- ActionScore 增加 raw target-limit penalty；Robust Teacher 的执行器中心 gains 以两份 bag 的实测 `55/65/80`、`1.5/1.5/2.5` 为中心，standard Teacher 保持不动，避免改写基线。
- support stability penalty 增加 8-step grace + 16-step ramp，避免首次顺序落足立即受到满强度惩罚。

### 8.2 离线验证与终审已完成

- schema6 tracker/aggregator 已闭合真实 delay、level-9 box、verified low patch、`0.28..0.40 m` 台阶、初始几何、当前帧向上法向接触、16 维 action order/scale/offset/asset IDs、checkpoint/代码 SHA、source train manifest、冻结 runtime clock、独立 600-step 计数及严格事件顺序；`nominal_delay1` 两步 GPU 探针实测 `requested=1/supported=true/runtime_observed=true/runtime=1/match=true`。
- schedule manifest 绑定 action/terrain/support/reward-stage cadence；train/play 均写 source checkpoint SHA。新 runtime sidecar schema2 为每个 checkpoint 写 SHA，同名同 iteration 替换、缺 sidecar、坏 sidecar、定义漂移均 fail closed；旧无 SHA sidecar只允许显式 migration/reset。
- Student 首次训练新增 `--highstep_parent_teacher_manifest`：实际加载 checkpoint 必须与 Robust Teacher manifest 的 selected path+SHA 完全一致；四字段祖先跨 Student 续训传播。monitor 与 policy finalize 再次重读/重哈希，换 parent、同内容换路径、重复 selected summary、runtime checkpoint 替换均拒绝。
- 纯 Python 回归共 `71/71`：schedule `22/22`、Student lineage `6/6`、target contract `5/5`、automation reliability `38/38`；相关 `py_compile`、`bash -n`、`git diff --check` 通过。
- 旧 `model_600.pt` schema6 nominal 600-step 诊断：kinematic top hold 为真，但 strict full climb 为假；没有双前足顶面支撑事件，rear advance 未达，rear-edge dwell `529`，raw target violation fraction `0.274375`、step rate `0.976667`、max delta `0.541321 rad`、最长连续 `574`。结论是旧模型不能作为部署候选或 preserve-safe parent。
- 2026-07-12 02:00 又执行一次**非正式** 256 env×2 iteration Robust migration 集成烟测：`model_600.pt` weights-only、fresh optimizer、schedule reset、tensorboard only。train manifest 固定 source SHA；两个新 checkpoint 的 sidecar SHA 均与文件一致；`model_1.pt` play 精确恢复 `runner_iteration=1 -> schedule_update=2`，runtime SHA 验证为真。它只证明保存/恢复合同，不是正式训练或策略候选。
- 部署端已完成 current-q hold、完整命令校验后原子提交、policy epoch/mutex、严格 4×4/16 YAML、controller joint order、mandatory physical envelope、resolved policy identity/manifest，以及 callback mutex→RealtimeBuffer 完整控制输入快照。clean build `1/1`、定向 CTest `3/3`（10+4+1 行为断言）通过；未启动 ROS/controller/电机。

## 9. 恢复训练的门槛

`highstep-auto-loop.service` 当前必须保持 disabled/inactive。旧 model600 已在统一合同下失败，因此不再走 full/preserve clean B。下一条正式路线只能在用户确认后新建 Robust action-safe migration lineage：`weights_only + migration + schedule reset + fresh optimizer`，并把 source SHA、runtime sidecar、代码合同和正式 run 身份全部写入 manifest；先完成长训练和 schema6 6场景×3 seed 严格矩阵，再决定是否进入 Student。

首次 Student 必须由严格合格且已经冻结的 Robust Teacher evaluation manifest 启动，并携带 `--highstep_parent_teacher_manifest`；旧无 lineage 的 Student run 不能直接严格续训。通过仿真门槛也只代表 simulation candidate，不代表可直接部署真机。真机前仍需无负载/悬空 FSM switch test、机载坐标/限位与 `/joint_commands` 一致性检查、真实 SHA/fingerprint 复核以及人工安全评审。

## 10. 可复核产物

第一组恢复与分析：

```text
tmp/rosbag_0707_recovery/analysis/
tmp/rosbag_0707_alignment/candidate_window/
```

第二组精细配准与交叉验证：

```text
tmp/rosbag_0707_alignment/second_analysis/second_trial_summary.json
tmp/rosbag_0707_alignment/second_analysis/alignment_anchors.csv
tmp/rosbag_0707_alignment/second_analysis/video_bag_event_timeline.csv
tmp/rosbag_0707_alignment/second_analysis/policy_switch_dropout_events.csv
tmp/rosbag_0707_alignment/second_analysis/joint_limit_audit_by_window.csv
tmp/rosbag_0707_alignment/second_analysis/joint_tracking_lag.csv
tmp/rosbag_0707_alignment/second_analysis/rear_width_event_windows.csv
```

最终 schema6/provenance 诊断：

```text
tmp/model600_strict_preflight_20260712_schema6/summary.json
tmp/model600_strict_preflight_20260712_schema6/nominal_seed11.log
tmp/model600_strict_preflight_20260712_schema6_delay1/summary.json
tmp/highstep_action_safe_migration_sha_smoke_20260712/summary.json
tmp/highstep_action_safe_migration_sha_smoke_20260712/train.log
tmp/highstep_action_safe_migration_sha_smoke_20260712/play_model1_provenance.log
```

原始视频和 bag 未被修改。第一组 `recovered.db3` 只是派生中间库；在 compact CSV/NPZ/JSON、脚本和哈希确认后可删除以回收空间，原始 bag 不得删除。
