# Highstep 06-17 至 v1.12.1 完整链路回忆、Teacher 对比与手动 Play 交接

更新时间：2026-07-15（Asia/Hong_Kong）

## 0. 文件性质与新对话读取规则

本文是历史回忆、证据索引、因果分析和人工 play 交接，**不是训练执行 authority**。

新对话不得仅凭本文重启训练、评估、service 或真机部署。首先必须执行：

```bash
cd /home/lxq/Softwares/robot_lab
python3 .agents/skills/highstep-control-variable-guardian/scripts/highstep_guard.py audit --json
```

然后以 `tmp/highstep_dashboard_active_workflow.json` 声明的 state/spec/preregistration/handoff 及其 SHA 为唯一实时 authority。本文中的旧 checkpoint、旧参数和旧结论只是证据。

### 写入本文时的实时 authority 快照

```text
workflow_id: highstep_teacher_front_placement_v1121_20260715
authority_version: v1.12.1
phase/status: implementation_ready_pending_training_authorization
spec:
  /home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md
spec_sha256:
  a68c168e7bb812e2e497b11e9d8a003b2a91ce0e2d8fcaf8249e9023c5534ffd
preregistration:
  /home/lxq/Softwares/robot_lab/tmp/highstep_teacher_front_placement_v1121_20260715/preregistration_v1121_code_only.json
preregistration_sha256:
  f85014abf17cb1d0e43b07545a97ce61611605b5a3bd8d2cdab2c74ec936c73b
active_pid: null
supervisor_pid: null
requires_user_action: true
```

这个快照的含义是：v1.12.1 代码与回归测试已完成，但没有新训练、没有 optimizer step、没有新 checkpoint，也没有 Student 训练或真机部署。

---

## 1. 用户的一贯最高目标

从第一次真机高台实验到现在，最高目标没有改变：

```text
真机两条后腿都能上到高台，机身进入平台，
形成足够稳定的时间窗，可由人在保护下切换 fixed stand。
```

优先级一直是：

1. 两后腿真正上台，不是只有前腿搭台。
2. 第一条后腿上台后能支撑/推起机身，带动第二条后腿清台。
3. 后腿不向中心线严重塌缩，不长时间卡在台沿。
4. 整机上台后左前腿是否主动落下，优先级低于“两后腿上台”；但前足在初次越过台沿时不应撞上立面。

始终保持不变的边界：`action_scale`、`joint_pos.clip`、`default_dof_pos`、observation/action 顺序和部署动作合同不得为了调高台动作而随意改动。

---

## 2. 精确 checkpoint 链路与 SHA

### 2.1 06-17 第一条 Highstep Teacher → Student 链

Teacher：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_vae_Teacher/
2026-06-17_05-07-05/model_68199.pt
SHA256=355e6fcb757f0f188034c6a17e2c1f06a6da5ea4b4a816faf7bd0e917f07ff3c
```

Student：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/
2026-06-17_19-54-10/model_69998.pt
SHA256=bc90d36802a561b8e268e241c8fb15b6f46d72ce9d687ca54cc4c385d8af8657
```

Student 的 `params/agent.yaml` 明确绑定 Teacher `model_68199.pt`。

### 2.2 06-19 首次真机高台实验相关 Teacher → Student 链

Teacher：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_vae_Teacher/
2026-06-19_01-42-04/model_87399.pt
SHA256=a41663a3283af20c1b26a4ed72fa3c48315c42c710832b162f0ec5077bf22664
```

Student：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/
2026-06-19_06-03-11/model_103398.pt
SHA256=1453172208e843d79461a9d398c1904a52ebe2a840cbdfa63e6dd9159919e81a
```

用户回忆的 `06-19_06_03_01` 在磁盘上的实际 run 名为 `2026-06-19_06-03-11`。该 Student 的 `params/agent.yaml` 明确绑定 `2026-06-19_01-42-04/model_87399.pt`。

### 2.3 0707 ActionScore Teacher → Student → 真机链

Teacher：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/
2026-07-04_01-45-21/model_151399.pt
SHA256=d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d
```

early behavior-best Student：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
2026-07-04_06-32-23/model_152100.pt
SHA256=3c7a335bdc6c1b952e3b0ddbfdd35f0e4311ca1a9fb86cb9ae9a21de2f63a8d6
```

0707 真机部署 Student：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/
2026-07-05_00-13-46/model_158797.pt
SHA256=7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284
```

两份 0707 rosbag 的 rosout 都直接确认 policy2 身份为 `158797`，这比部署 YAML 中可能过时的 `model_name` 更可靠。

### 2.4 0707 之后的上一轮 Teacher

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/
2026-07-11_11-22-23/model_172300.pt
SHA256=dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35
```

v1.12 为它制作的同 SHA 只读 schedule/runtime 镜像：

```text
/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_rear_support_v112_20260715/
source_model_172300_v112/model_172300.pt
SHA256=dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35
```

### 2.5 最新 v1.12 后腿支撑动作 Teacher

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/
arclab_arcdog_adjustable_leg_highstep_rear_support_v112_Teacher/
2026-07-15_10-52-45_v112_long_attempt1_20260715_105240/model_173200.pt
SHA256=962fd3ce3983e4a478092be8f7a636e87ed872b79496c4fa3134191838d9b80d
```

这是从 `model_172300` 完整恢复后的 v1.12 E1000，也是预注册评估选出的 `selected_behavior_best`。

### 2.6 v1.12.1 当前还没有新 checkpoint

v1.12.1 只完成了“左前足预接触回收”的单变量代码实现。它的行为起点仍是上述 `model_173200.pt`；不存在可以 play 的“v1.12.1 训练结果”。

---

## 3. 从 06-17/06-19 到现在，为什么一直修改

### 3.1 06-17 Student：能勉强上台，但盲走蒸馏先丢掉 OOD 稳定性

`model_69998` 的历史结论：

- 仿真中上高台动作勉强可用；
- `terrain_levels ≈ 3.0`，command 达到 `0.6375`；
- Student `bad_orientation` last100 约 `0.0144`，高于 Teacher 约 `0.0083`；
- 键盘后退容易后仰翻倒。

当时 Teacher 的 command 主要是纯前进 `lin_vel_x=(0.20,0.85)`。Teacher 依靠 PPO/privileged latent 对训练外后退有一定泛化，但盲走 Student 在平均蒸馏中先丢掉了这部分。Student-first 加轻微后退样本后，play 没有达到预期，因此路线转回 Teacher 稳定性与 box 姿态精修。

### 3.2 06-19 链：更稳、原始仿真更流畅，但真机后腿清台余量不足

06-19 Teacher/Student 原始环境比后来温和：

```text
box: 25%, height 0.04-0.35 m
没有 box_hard 0.30-0.38 m
pyramid_stairs: 30%
pyramid_stairs_inv: 15%
pit: 15%
random_rough: 9%
```

历史标量中，06-19 Student `terrain_levels` tail200 约 `2.920`，高于后来更硬 terrain 下的多个 Teacher。但 06-19 Teacher 后腿相关动作奖励比后来弱，例如历史 tail 中 `rear_feet_highstep_clearance ≈ 0.048`。

因此这条链在较温和仿真地形上可以看起来流畅，但并没有充分解决 35 cm 真机高台的后腿高度、接触、支撑和第二后腿清台余量。这是第一次真机实验后必须继续修改的核心原因，不是因为当时整体 gait 完全失败。

### 3.3 06-22/06-23：“一条后腿上台”和“整机上台”被 reward 错误等价

仿真逐渐出现清晰的双模态：

```text
右后腿先上：通常更顺，已上台后腿可支撑机身，第二后腿更容易跟上。
左后腿先上：更容易长时间卡住，另一后腿在台沿下蹬腿。
```

平均化的 rear clearance、box push、rear drive 和 phase prior 可以在“一条后腿很强，另一条后腿失败”时仍然得到高分。所以曲线变好不等于整机上台变好。后续才开始增加：

- 第二后腿清台；
- worst-rear 而非 bilateral average；
- one-sided stall；
- 已登台后腿的支撑/body drive；
- 后腿宽度和中心线安全余量。

### 3.4 06-25 到 0707 前：调强第一后腿与保留后半段支撑之间不断交易

历史中曾反复出现：

- 第一后腿踏台能力变强，但一条后腿上台后更容易单侧卡边；
- 后半段支撑/body drive 变强，但第一后腿的进台概率变弱；
- 激进后腿 reward 通过 front commit gate 反向鼓励过强前半身 highstep 姿态，造成左前腿在平地/台前过度高抬；
- 只继续拧 reward 权重，没有固定候选自主 rollout 门禁，容易被平均 reward 和 terrain level 误导。

这些经历最终导向 ActionScore、阶段化后腿指标、独立 core9/directional 评估以及更明确的 rear-platform 成功定义。

### 3.5 0707 真机：真正失败链并不是单纯“力矩不够”

0707 的两组视频/rosbag 由不同机器记录，经 joystick、IMU 冲击和可见接触事件对齐后，重复得到同一主链：

```text
前足接触高台
→ 后腿/后髋向中心线收窄
→ 单前足/台沿承载、roll/yaw 重试
→ 后腿长时间卡边、无法把两后腿稳定送上台
→ 安全绳明显介入，最终结果不能算独立稳定成功
```

第一组 rear width 从 idle 约 `0.472 m` 降到首前足接触窗口 median 约 `0.330 m`，RR `min_abs_y` 一度约 `0.0216 m`。这和视频中的内收、卡边和摆动一致。

真机人为提拉已经大幅减小 knee/calf 负担，机器仍然无法自然完成动作；同时 MuJoCo 35 cm 平台在 hip/thigh/calf `23.7/23.7/45.43 N·m`、gear=1 下仍能完成。因此“需要 80 N·m 才能上台”被排除，主因仍是动作轨迹、接触相位、后腿中线塌缩与 Student 继承质量。

rosbag 还揭示了三类独立工程问题：

1. FL hip qdes/actual 峰值约 `2.53/2.23 rad`，超过当时部署 URDF `±1.22173 rad`；
2. policy switch 后约 `59-70 ms` 的 q/kp/kd 全零命令窗口；
3. 实际 policy/config/gains 缺少完整可追溯指纹。

这些问题值得修复，但不能用它们掩盖“两后腿本身没有训好”的核心事实。用户也明确表示，约 60 ms 切换掉控并没有在当次实验中造成足以解释主失败的实质破坏。

### 3.6 0707 后的修改：局部有效，但长期被评估/蒸馏真实性问题干扰

从 0707 到 `model_172300` 的大致方向包括：

- rear width/centerline 安全余量；
- 双侧接触、slip、roll/yaw 和 rear lag/support；
- one-sided stall 与 bilateral rear advance；
- delay/real-gain/Robust 评估；
- Student actor/estimator/latent、prior target、optimizer 恢复和评估 lineage 审计。

审计结论不是“全部无效”：

- rear width、centerline、support、stall 曾改善局部仿真指标或平均 dwell；
- 但一些后段 latch 触发太晚，无法解决第二后足 clear 之前的卡边；
- width/centerline 过强可能推出“宽站但不推进”的副作用；
- 评估器 task/lineage/schedule 映射错误曾把基础设施无效错报为行为失败；
- Student 平均 loss 下降多次未转化为自主上台，证明必须用候选自己驱动的 rollout，不能用 same-state 或全局 MSE 代替行为。

---

## 4. 三代关键 Teacher 对比：151399 vs 172300 vs 173200

### 4.1 可以严格证明的部分：151399 对 172300

v1.11 做了零训练、同物理 snapshot、同 Robust Teacher task、同 action prior/real gains/delay/command/platform/seed 的 204 场 A/B，每个 Teacher 102 场。

| 项目 | 0707 Teacher A `151399` | 上一轮 Teacher B `172300` |
| --- | ---: | ---: |
| valid | 102/102 | 102/102 |
| full climb | 102/102 | 96/102 |
| rear hold | 102/102 | 96/102 |
| recovery | 102/102 | 102/102 |
| nominal | 3/3 | 3/3 |
| combined pressure | 9/9 | 9/9 |
| 最强内收等级 | 0.22 m | 0.22 m |
| 最强横向冲量等级 | 0.30 m/s | 0.30 m/s |

`172300` 的 6 场 full/rear-hold 失败分布在：

- `0.10 m/s` 冲量 4 场；
- `0.20 m/s` 冲量 1 场；
- `0.26 m RR-only` 内收 1 场。

固定结论：

```text
model_172300 不能被称为比 model_151399 明显更鲁棒。
v1.11 status = new_teacher_robustness_not_clearly_improved
```

这不证明 `172300` 不会上台；它在 nominal 和用户 rear-platform 目标下有很强仿真能力。它只证明“更鲁棒”这个前提不成立。

### 4.2 173200 相对 172300 已经改变了什么

v1.12 只替换一个后腿动作目标族：

```text
旧：后腿绝对功率 + 左右平均姿态（总权重 2.30）
新：双后腿支撑锚定、镜像姿态、共同抬身和安全顺序上台的 rear_support_motion_contract（总权重仍 2.30）
```

这个变更不修改 action prior、网络、DR、optimizer、action scale/clip 或真机 gains。

v1.12 训练后固定评估：

| 保存点 | valid | full climb | rear hold | no severe inward |
| --- | ---: | ---: | ---: | ---: |
| E500 `172700` | 3/3 | 2/3 | 2/3 | 3/3 |
| E1000 `173200` | 3/3 | 3/3 | 3/3 | 3/3 |
| E2000 `174200` | 3/3 | 3/3 | 3/3 | 3/3 |
| E4000 `176200` | 3/3 | 3/3 | 3/3 | 3/3 |
| E6000 `178299` | 3/3 | 3/3 | 3/3 | 3/3 |

预注册聚合选出 E1000 `model_173200` 为 behavior-best。用户用 `v112_play_best` 实际观看 `Kazam_screencast_00154.mp4` 后明确评价：

```text
后腿已经是想要的动作。
```

因此可以说：**相对 `172300`，`173200` 的后腿动作结构在用户人工视觉复核中已改到目标方向**。

但不能越界声称：

- 还没有用 v1.11 的同一 204 场 snapshot 矩阵对 `173200` 与 `151399/172300` 重做全局鲁棒性 A/B；
- 因此不能说 `173200` 已严格证明“整体比 151399 或 172300 更鲁棒”；
- E1000 的 3/3 和用户视频是重要候选证据，不是真机通过证据；
- 它仍是 Teacher，还没有对应的最终 Student。

### 4.3 173200 的当前剩余问题

用户观看 `Kazam_screencast_00154.mp4` 时发现：

```text
后腿动作符合要求；
左前足刚上高台时有概率碰到立面/台沿并卡顿；
希望预接触时左前腿更向后回收，越过台沿后的首次落点更偏后。
```

这是相对独立、范围更小的前足预接触几何问题，不应再通过改后腿动作合同解决。

---

## 5. v1.12.1 已实现但未训练的单变量修改

旧的前腿 reach 总权重 `0.45` 被等总量拆分为：

```text
front_legs_reach = 0.35
left_front_precontact_retraction = 0.10
```

新项只在以下条件同时满足时生效：

- highstep 高台场景；
- 向前 command；
- pre-commit；
- 左前足尚未越过台面；
- 左前足已经抬起。

目标 body-frame x 从 `0.36 m` 回收到 `0.30 m`，足端高出台面 `0.04 m` 后释放。没有修改 FR、后腿、action prior、网络、optimizer、其他 reward、`action_scale` 或 `joint_pos.clip`。

已完成：

- Python compile；
- v1.12.1 + 旧 v1.12 定向回归：`16 passed`；
- Isaac task registry 可列出新 task；
- 新 task：`RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-ArcdogAdjustableLeg-v0`。

尚未完成：

- E1000 → v1.12.1 task-only full-resume rebinding；
- 有限训练预注册；
- 1-5 update smoke；
- 正式训练；
- 新 checkpoint 和人工视频 A/B。

因此新对话不得把“代码已实现”写成“前足问题已解决”。

---

## 6. 有效、部分有效、无效或有副作用的经验汇总

| 方向 | 当前判定 | 理由 |
| --- | --- | --- |
| 保留 action prior | 有效，必须保留 | 0707/sidestep 经验表明 prior 对结构化大动作重要；不应为蒸馏方便就拆掉。 |
| 第二后腿清台、worst-rear 和 one-sided stall | 方向正确，局部有效 | 弥补“一条后腿上台就得高分”的漏洞，但不是单项就能解决完整动作。 |
| rear width/centerline | 部分有效 | 可以改善内收风险，但过强可能变成宽站不推进。 |
| post-clear latch | 对主瓶颈单独无效 | 很多失败发生在第二后足 clear 之前，触发太晚。 |
| 普通 reward 权重继续微调 | 收益低 | 多次证明曲线改善不等于后腿支撑与第二后腿清台改善。 |
| 把 100 ms 全部当 action delay | 错误 | 它混合了机械/PD 响应；真实控制传输 delay 与执行器动力学应分开。 |
| 用全局 MSE/loss 选 Student | 无效 | 关键高台相位样本占比小，平均更像 Teacher 不等于高台强动作被保留。 |
| 用 same-state Teacher 轨迹声称 Student 会上台 | 错误 | same-state 只能做因果动作误差诊断，行为必须由 Student 自主驱动 rollout 证明。 |
| 把评估 lineage/task/SHA 错误当行为 0/9 | 严重错误 | 基础设施 invalid 必须与行为失败分类。 |
| v1.12 rear-support motion contract | 已得到人工目标动作证据 | `model_173200` 的后腿动作被用户接受；但尚未完成跨 Teacher 的全矩阵 A/B 或真机验证。 |
| v1.12.1 左前足预接触回收 | 代码正确性通过，行为未证明 | 它是当前待证明的单变量，不得预先称为有效。 |

---

## 7. 手动 play 对比文件

已生成可在 Linux 终端 `source` 的纯文本：

```text
/home/lxq/highstep_chain_play_comparison_20260715.txt
SHA256=ca357365e710d371f27deb6a73832f0475175c8e1bf8cec00739cc6b92d3002b
```

加载：

```bash
source /home/lxq/highstep_chain_play_comparison_20260715.txt
hsplay_status
```

`hsplay_status` 显示 `IDLE` 后，一次只运行一个：

```bash
hsplay_0617_teacher
hsplay_0617_student

hsplay_0619_teacher
hsplay_0619_student

hsplay_0707_teacher
hsplay_0707_student_early
hsplay_0707_student_deployed

hsplay_prev_teacher
hsplay_latest_teacher
```

所有 9 个 checkpoint 在文件生成时均已现场重新校验 SHA256 通过。

手动 play 的统一交互口径：

- canonical `scripts/rsl_rl/base/play.py`；
- 已修复的 `R` 同高台口径重生；
- 键盘手动操作，非固定视角；
- `--debug`；
- `box_hard level 9`；
- seed 11；
- 高台近边缘、正对高台；
- delay=0；
- Teacher 保留各自 action prior，Student 使用各自 NoPrior task；
- 不传 `--disable_action_prior`。

键位：

```text
↑/↓: 前后
←/→: 横移
Z/X: 转向
L: 清零命令
R: 在同一高台口径重生
Ctrl+C 或关闭窗口: 退出
```

### 如何解读手动对比

1. 06-17/06-19 策略原始训练环境没有现在的 `box_hard`；把它们放到当前 box_hard level 9 是压力可视化，不是原始历史环境完全复现。
2. 文件为每条链使用原生 task/action contract；不能把 06 月 plain Highstep、0707 ActionScore 和 v1.12 RearSupport checkpoint 全塞进同一个错误 task。
3. 手动 play 用于观察 gait、前足接触、第一/第二后腿时序、支撑、卡边和人类可接受性；它不代替同 snapshot 定量 A/B。
4. 每个 play 完全退出后再启动下一个；若 `hsplay_status` 显示 `BUSY`，不得为了手动观看强停活动流程。

---

## 8. 建议的人工观察表

每个策略至少重生 3 次，不要只记录“上/没上”：

| 策略 | 平地/接近步态 | 前足撞台 | 第一后腿 | 已上台后腿支撑 | 第二后腿 | 后腿内收 | 整机上台/保持 | 其他异常 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 06-17 Teacher 68199 |  |  |  |  |  |  |  |  |
| 06-17 Student 69998 |  |  |  |  |  |  |  |  |
| 06-19 Teacher 87399 |  |  |  |  |  |  |  |  |
| 06-19 Student 103398 |  |  |  |  |  |  |  |  |
| 0707 Teacher 151399 |  |  |  |  |  |  |  |  |
| 0707 Student early 152100 |  |  |  |  |  |  |  |  |
| 0707 deployed Student 158797 |  |  |  |  |  |  |  |  |
| previous Teacher 172300 |  |  |  |  |  |  |  |  |
| latest Teacher 173200 |  |  |  |  |  |  |  |  |

推荐观看顺序：

```text
06-17 Teacher → 06-17 Student
06-19 Teacher → 06-19 Student
0707 Teacher → 0707 early Student → 0707 deployed Student
172300 → 173200
```

这样可以分开看到：

- 同一时期 Teacher 本身动作好不好；
- Student 蒸馏后丢掉了什么；
- 0707 的 action prior/ActionScore 链相比 06 月链有什么视觉变化；
- v1.12 后腿支撑合同相比 `172300` 真正改变了哪些动作细节。

---

## 9. 当前不应被混淆的最终结论

1. **06-17/06-19 不是完全失败链。** 它们奠定了可上台的基础 gait，但原始地形较温和，Student 的 OOD 稳定性与真机后腿清台余量不足。
2. **0707 `model_151399` 仍是强 Teacher 基准。** 它在 v1.11 受控 102 场中 full/rear/recovery 全通过，不能因为时间较早就当作已被后来 Teacher 超越。
3. **`model_172300` 会上台，但没有证明比 151399 更鲁棒。** 受控 A/B 反而是 151399 的 full/rear hold 更好。
4. **`model_173200` 的主要新成果是用户认可的后腿动作。** 这是重要进步，但不等于已证明全局鲁棒性最强，更不等于已完成真机验证。
5. **v1.12.1 还是待验证修改。** 它只针对左前足预接触碰台，必须保住 `173200` 已经被用户接受的后腿动作。
6. **下一步不能因为本回忆文件就自动开训。** 必须先按当前 v1.12.1 authority 做 task-only full-resume rebinding，冻结有限预算/保存点/视觉 A/B 门禁，再做 1-5 update smoke。

---

## 10. 主要证据文件索引

```text
历史主训练链：
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_training_chain.md

06-25 至 07-03 历史思路：
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_historical_reasoning_notes_20260625_20260703.md

0707 真机视频/rosbag 审计：
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_0707_real_log_audit_handoff_20260712.md
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_0707_real_data_analysis_20260712.md

v1.11 Teacher A/B 冻结决策：
/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_robustness_ab_20260715/manifests/teacher_robustness_ab_decision.json
SHA256=f1b1ef087f8701f59d0b99316e66ed80c2ab343429bafbee9c9830aed5ce4a9c

v1.12 路线决策：
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_teacher_rear_support_v112_decision_20260715.md

v1.12 训练后选模聚合：
/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_rear_support_v112_20260715/evaluations/posttrain_summary.json

v1.12.1 当前代码交接：
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_v1121_front_placement_handoff_20260715.md

手动 play 命令：
/home/lxq/highstep_chain_play_comparison_20260715.txt
```

## 11. 新对话的一句话接管摘要

```text
先用 highstep_guard 确认实时 authority。历史上 06-17/06-19 奠定基础上台动作，
0707 Teacher 151399 在受控鲁棒性 A/B 中仍强于 172300；0707 真机主失败是前足接触后
后腿向中线收窄、卡边和无法把两后腿稳定送上台。v1.12 从 172300 单变量改为
rear-support motion contract，E1000 model_173200 的后腿动作已被用户人工接受；当前唯一待验证
问题是左前足初次越台沿的偶发碰台。v1.12.1 只完成了左前足预接触回收代码和 16 项回归，
尚未训练。不得恢复旧 v1.12 E6000 service，不得破坏 173200 后腿动作，不得改 action_scale/
joint_pos.clip，也不得把本历史记忆当成新的训练授权。
```
