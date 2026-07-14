# Highstep GitHub 可移植交接（2026-07-15）

## 目的

本文件让另一位协作者在自己的电脑上检出 `HKU-OUGE/robot_lab` 的 `dev_lxq_new` 分支后，通过 Codex 理解这段 highstep 工作的代码、决策、失败历史、当前进度和真实性边界。

它是 GitHub 发布时刻的只读交接，不是原工作站的实时 supervisor，也不授予自动训练或部署权限。

可直接外发的排版版手册：`docs/robotlab_memory_zh/deliverables/highstep_collaborator_handoff_20260715.pdf`。PDF SHA256：`f203b56ca93cf7c9932fe74a30a3e0e027a01fd98f070f5e9d2e4a685dc4dba3`。同目录 HTML 是可维护源文件。

## 当前正式 authority

- workflow：`highstep_teacher_robustness_ab_20260715`
- authority：v1.11
- spec：`docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md`
- spec SHA256：`445b322302f1e1b64a621208670c7fa8736d7f1a64dec2d000f6b629e936076a`
- preregistration 原路径：`tmp/highstep_teacher_robustness_ab_20260715/preregistration_v111.json`
- preregistration SHA256：`1b2bd47960d9111cf1b98676ece01524d89a889fc4b55cfc1ed08174782f2a55`
- Git 内冻结副本：`docs/robotlab_memory_zh/snapshots/highstep_teacher_robustness_ab_20260715/preregistration_v111.json`

v1.11 只比较 Teacher checkpoint。环境、action prior、real gains、delay、命令、平台、seed、物理 snapshot 和干预必须相同；禁止训练、backward、optimizer step 或 checkpoint 写入。v1.10 continuation 保持 `user_paused_pending_teacher_comparison`，不得自动恢复。

## 两个 Teacher

| 角色 | checkpoint | SHA256 |
|---|---|---|
| Teacher A，0707 旧 Teacher 对照 | `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-04_01-45-21/model_151399.pt` | `d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d` |
| Teacher B，新 Teacher 候选 | `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt` | `dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35` |

checkpoint 不进入 Git。需要在新机器继续 rollout 时，必须单独传输并现场重算 SHA；不能创建同名空文件或用别的 checkpoint 代替。

## 发布期间的实时状态

2026-07-15 05:36 HKT 的只读观测：

- authority audit：通过；
- service：`highstep-teacher-robustness-ab.service` active/running；
- supervisor PID：`1800946`（仅原工作站有效）；
- Teacher A：`87/102`；Teacher B：`0/102`；合计 `87/204`；
- 当前场：`impulse_first_rear_right_20cms_seed11`；
- infrastructure invalid：`0`；retry：`0`；failure class：`none`；
- 旧 pre-fix smoke 的 repo-root import 错误已单独归类为基础设施失败，不计入204场行为结果；
- A/B smoke 均证明 checkpoint 与 policy tensor SHA 前后不变，训练/backward/optimizer/runner.learn 调用数为零。

仓库中的 state snapshot 是这个时刻的证据。原工作站流程继续运行，因此接收者需要实时状态时必须重新读取原工作站 dashboard，而不是把该快照当 heartbeat。

提交前的第二个只读快照（2026-07-15 05:50 HKT）确认流程继续正常推进：Teacher A 已完成 `102/102`，Teacher B 为 `7/102`，总计 `109/204`；service 仍为 active/running、restart counter 为0、基础设施无效与补跑均为0。对应文件为 `state_snapshot_20260715_055016.json`。前一个 `05:36` 快照继续保留，用来证明发布整理期间进度从87增长到109且没有被 Git 操作中断。

用户已明确确认将本交接范围公开发布到公开仓库 `HKU-OUGE/robot_lab` 的 `dev_lxq_new` 分支。

## 本分支同步的代码范围

### 训练、恢复与 checkpoint 合同

- `scripts/rsl_rl/base/train.py`
- `scripts/rsl_rl/base/algorithm_checkpoint.py`
- `source/.../agents/vae_ppo.py`
- `source/.../agents/rsl_rl_ppo_cfg.py`
- `tools/highstep_*supervisor.py`、W&B stage gate、resume/rebinding 工具

### 环境、动作、reward、curriculum 与 schedule

- `highstep_env_cfg.py`
- `mdp/actions.py`、`curriculums.py`、`events.py`、`observations.py`、`rewards.py`
- `mdp/highstep_schedule.py`
- adjustable-leg task 注册和 terrain 配置

### 评估、same-state、诊断与视频真实性

- `scripts/rsl_rl/base/highstep_*play.py`
- frozen training-buffer / fc_mu observability diagnostics
- core9、directional、Teacher A/B、oracle prior-delta 工具
- `highstep_video_visibility_qc.py` 和交付纠错工具

### 自动化安全和 Codex 上下文

- `AGENTS.md`
- `.agents/skills/highstep-control-variable-guardian/`
- `.codex/hooks.json`
- `scripts/systemd/` 中的服务定义（包含原机器绝对路径，只作模板/历史证据；不得在新机器直接 enable）
- `tests/test_highstep_*.py`

## 历史链路怎么读

1. 0707 真机事实：`highstep_0707_real_data_analysis_20260712.md` 与 `highstep_0707_real_log_audit_handoff_20260712.md`。
2. 自动化、显示卡死和恢复经验：`highstep_automation_v4_handoff_20260711.md`、`highstep_display_hang_emergency_pause_handoff_2026-07-11.md`。
3. Student 恢复路线的完整版本链：正式 spec 内 v1.1→v1.11；顶部高版本拥有优先级，旧章节只解释 checkpoint 和失败事实。
4. v1.5.2 自动化事故：`highstep_v152_automation_failure_postmortem_20260713.md`。
5. v1.8 环境课程控制变量错误：`highstep_v18_student_environment_curriculum_decision_20260714.md`。
6. v1.9 oracle prior-delta 与 v1.10 continuation：对应 decision 文件。
7. 跨阶段通用经验：`failure_modes_and_lessons.md`。
8. 早期分散回答中的 Teacher/terrain/reward 证据已集中到 `highstep_historical_reasoning_notes_20260625_20260703.md`，并明确标为历史分析。

`highstep_live_context_handoff.md` 很长，保留了大量旧对话事实，但其旧“当前”、PID 和恢复命令已经失效。只把它当历史索引。

## Git 不包含的外部证据

- `logs/` 中的 `.pt` checkpoint、optimizer 和参数快照；
- `tmp/` 中的实时 state、heartbeat、逐场 snapshot、frames JSONL 和 manifests；
- `wandb/` 离线 run；
- 0707 rosbag、真机视频和自动评估视频；
- 用户 home 下的 systemd unit、部署端仓库和真机机载配置。

原因是体积、机器绑定或隐私，而不是这些证据不重要。所有分析结论仍应以文档记录的绝对路径、SHA 和 W&B run 交叉核验。若要继续实验，应通过受控文件传输或 artifact storage 补齐，不能把 GitHub 代码本身当作 checkpoint 证据。

## 发布前验证

当前 v1.11 和 authority guard 定向测试：

```text
14 passed
```

全量历史 highstep 测试：

```text
250 passed, 10 failed
```

10个失败均保留并可复现：9个历史测试把 v1.3/v1.5/v1.8/v1.9 的旧 spec、dashboard 或 play SHA 当成当前 authority；另1个是旧 R2 canonical-helper authority 冲突。它们不能通过修改当前 v1.11 spec SHA、放宽 canonical helper 或伪造旧 dashboard 来“修绿”。接收方应把这些失败视为待做的历史测试隔离/fixture 可移植性维护，而不是 v1.11 Teacher A/B 行为失败。

## 新电脑上的推荐动作

```bash
git clone git@github.com:HKU-OUGE/robot_lab.git
cd robot_lab
git checkout dev_lxq_new
sed -n '1,240p' AGENTS.md
sed -n '1,260p' docs/robotlab_memory_zh/CODEX_START_HERE.md
```

然后让 Codex：

1. 只读审查 `git status` 和 commit；
2. 阅读本 handoff、spec 顶部 v1.11、decision memory 和 snapshot manifest；
3. 运行不依赖 Isaac Sim 的纯 Python tests；
4. 列出缺失 checkpoint/配置/数据，而不是自动启动旧 service；
5. 只有在外部 artifact 完整、SHA 匹配并建立新机器自己的 preregistration/authority 后，才考虑恢复评估。

## 不可丢失的经验

- infrastructure invalid 不能改判为行为失败；行为失败也不能用重跑掩盖。
- smoke 证明机制可运行，不证明上台行为。
- same-state 证明因果动作差异，不证明候选自主上台。
- 逐场 checkpoint、schedule、runtime snapshot、lineage 和 matrix 必须 fail-closed。
- Teacher、reward、prior、网络、action scale、joint clipping 和训练路线不能在一次“修复”中捆绑改变。
- 数值门禁不能覆盖用户要求的视觉复核，任何候选都禁止自动真机部署。
- 旧 service 自动复活、W&B 多片段同步、remote scan API、task compatibility mapping 和 checkpoint scope 都曾造成基础设施循环；对应测试和修复代码已保留。
