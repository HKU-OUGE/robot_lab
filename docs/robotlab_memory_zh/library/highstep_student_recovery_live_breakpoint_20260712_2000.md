# Highstep Student recovery 实时断点（2026-07-12 20:00 HKT）

已按照要求提前检索记忆和约束｜HLC-OK

## 1. 唯一执行依据

- 正式规范：`/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md`
- 规范 SHA256：`ae31b2da2d6c1a058a608da12c8ce500f98db28bab4dd32f4c0240b36ce9daff`
- 与旧计划、旧 supervisor 或旧自动化策略冲突时，必须以该规范为准。
- 永久锁定项继续生效：不得修改 `action_scale`、`joint_pos.clip`、`default_dof_pos`、观测/动作/关节顺序、Teacher prior、网络主体、部署端、Teacher reward/task 或新增 sim-to-real。

## 2. 本断点的真实运行状态

- 参数更新：**尚未开始**；本断点前没有执行 B 阶段更新。
- baseline：`baseline_core9_retry2` 已完成 9/9 原始运行；aggregate 表明 `evaluation_complete=true`、`matrix_complete=true`、`parent_teacher_manifest_valid=true`、`student_source_lineage_valid=true`。
- baseline 目录：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/baseline_core9_retry2`
- 旧 `highstep-auto-loop.service` 保持 disabled/inactive；不得恢复旧循环。
- 当前唯一接力进程：PID `3061368`，PGID `3061368`。
- 接力脚本：`/home/lxq/Softwares/robot_lab/tools/launch_highstep_recovery_after_quota_handoff.sh`
- 接力日志：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/quota_bridge_launcher.log`
- 接力器行为：等待 baseline manifest -> 要求 Stage B 关键文件连续 120 秒字节不变 -> 确认无其他 train/play -> py_compile -> `exec` 正式 supervisor。
- 正式 supervisor：`/home/lxq/Softwares/robot_lab/tools/highstep_student_recovery_supervisor.py`
- supervisor 启动后权威状态：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/state.json`
- supervisor heartbeat：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/heartbeat.json`
- fail-closed / 完成 handoff：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/handoff.json`

## 3. 双 checkpoint 强绑定

Student 初始 actor + estimator/VAE：

`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt`

SHA256：`9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229`

独立 frozen Teacher actor + privileged encoder：

`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_11-22-23/model_172300.pt`

SHA256：`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`

验证方法不是比较文件名：训练启动时分别读取两个绝对路径并校验 SHA；逐项验证 Student actor/estimator 来源；Teacher actor/privileged encoder 由 Teacher checkpoint strict load；manifest 和 checkpoint extra state 同时写入两个路径/SHA/component hash；禁止 Student actor 被复制为 Teacher。任一不一致 fail closed。

## 4. 冻结范围与 smoke 验证

- 唯一 optimizer 参数是独立的 `actor.6` RL/RR hip 最终输出两行（action rows 2、3）的 weight/bias 叶张量。
- estimator/VAE、actor 主体、critic、privileged encoder、FL/FR hip、所有 thigh/calf、四个 box 输出和其他输出全部冻结。
- checkpoint 保存前把允许变化的两行写回标准 `actor.6`，保证普通 play/export 可直接读取。
- smoke 为 4 次 weights-only 更新 + 1 次 full-resume 更新；验证 frozen tensors bitwise zero-delta、仅 rows 2/3 变化、有限 loss/grad/output、optimizer 恰好两个参数、Adam moments 有限且第 5 次的 step 连续增加、model state 与 extra row state 一致。

## 5. 已完成的最小正确性修改

- `highstep_env_cfg.py`：撤销错误的 no-prior Teacher override；Robust Student 仍保持部署时 no-prior。
- `rsl_rl_ppo_cfg.py`：Stage B warm-up=0、actor LR=1e-5、canonical 双 checkpoint 路径/SHA；旧 min-abs/phase/box loss 归零。
- `vae_ppo.py`：独立 Teacher 强绑定；仅后 hip 两行 optimizer；精确 Teacher raw hip target；完整 optimizer/effective-count/full-resume；冻结张量逐位审计。
- `train.py`：Stage B 禁止 legacy migration、要求 resume、写 binding manifest；运行时 action/default/observation contract 的最终补强正在当前代码稳定窗口前完成。
- `play.py`：只做 play 必需的 `load_optimizer=False`，当前 SHA `8013ce5187a9a9b066246a9bacf8a9e857af492585fc84ba035767d318d66766`；为避免 baseline SHA 失效，独立 Teacher action MSE 审计设计已记录但尚未落地。
- `highstep_centerline_guard_monitor_20260709.sh`：新 workflow/ledger namespace、parent manifest 强校验、持续无进展 watchdog，移除固定墙钟 timeout。
- `highstep_student_recovery_supervisor.py`：baseline/smoke/300->500->1000->2000、core9、best checkpoint、门禁、heartbeat、checkpoint/handoff；已补 baseline 参与 best、front/first-rear 与 full/hold 非退化门、Adam resume 审计和关键文件 hash 集合。

## 6. 额度恢复后的第一组只读检查

依次查看，不得因没有读到对话上下文而重开分支：

```bash
ps -p 3061368 -o pid,ppid,pgid,stat,etime,cmd
tail -n 100 /home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/quota_bridge_launcher.log
python -m json.tool /home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/state.json
python -m json.tool /home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/handoff.json
```

文件不存在时按语义处理：`state.json` 不存在且 PID 仍活着表示接力器尚在稳定窗口；PID 已退出则读取接力日志的最后一条 fail-closed 原因。禁止未审计就删除 state 或隐式重试。

## 7. 恢复顺序

1. 先读正式规范、本文件、`state.json`、`handoff.json` 和接力日志。
2. 确认同一时间最多一个 train/eval/play。
3. 若 supervisor 正常运行，只做只读监督，不修改其 critical files。
4. 若 supervisor fail closed，保留 checkpoint/log/manifest，针对明确原因做最小修复；不得以“再试一次”另开实验分支。
5. B 按 core9 的 300 -> 500 -> 1000 -> 2000 行为门禁推进；loss 单独下降不能延长。
6. 达到同一 checkpoint 的 valid9、full>=8、rear hold>=8、no severe inward>=8 且 lineage/schedule/Teacher 对照有效后，才可导出和录制；不得自动部署真机。

## 8. 尚未允许绕过的事项

- Candidate 的“Student 相对 Teacher 无明显动作退化”仍需独立、可审计的动作对照证据，不能用 `full>=8` 循环定义替代。
- 若 B 到 C 门满足，C 启动前仍必须证明 Teacher post-prior rear-box 精确 target；无法证明就 fail closed，不能用 pitch 近似。
- candidate export 必须只接受本次生成的 Student artifact，并对 9 个视频逐场保留标签；旧 artifact、ffmpeg 失败或缺视频不得标记 candidate-ready。
- worktree 原本包含大量用户修改；不得回滚、覆盖或清理无关 dirty/untracked 文件。最终运行的代码 SHA 以 supervisor 的 `preflight_manifest.json` 为权威。

## 9. 20:07 HKT 接力后的权威增量

- 接力器已在 `20:06:15` 成功 `exec` 正式 supervisor；监督器 PID/PGID 为 `3061368`。
- 13 项 static/CPU tests 已通过。
- smoke 已通过：4 次更新后 Adam step 为 `16/16`，full-resume 第 5 次后为 `20/20`；moments 有限且 runner/algorithm 两份 optimizer state 一致。
- smoke 只改变 `actor.6.weight[2:4]` 和 `actor.6.bias[2:4]`，RL/RR 两行均变化；冻结张量违反数为 0。
- Student SHA 与 Teacher SHA 均在 smoke audit 中验证为 canonical 值。
- 正式 `B300` 已于 `20:06:53` 启动；活动训练 PID/PGID 在启动时为 `3125928`。
- B300 run name：`student_recovery_B300_20260712_200653`。
- B300 训练日志：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/training/student_recovery_B300_20260712_200653/train.log`。
- smoke audit：`/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_20260712/smoke_audit.json`。
- 后续 PID 会随 eval/下一训练段变化；恢复时应以 `state.json.active_pid` 而不是本段历史 PID 为准。

## 10. 22:24 HKT 正确性修复与显式恢复

- 原流程没有因 token 中断：它在 `20:35` 完成错误实现下的 B300 和 core9，因 `full/rear hold 4/9 -> 0/9` 按门禁停止。
- 失败 checkpoint `model_299.pt` SHA256 为 `9a56654427b1687342a46633719c839048fac25ecb2b9f4869900beaf881ce41`，已明确禁止作为恢复起点。
- `model_100.pt` 额外诊断 core9 也是 `valid=9, full=0, rear_hold=0`；诊断目录为 `tmp/highstep_student_recovery_20260712/diagnostics/B100_core9`。
- 发现并复现致命实现 bug：`vae_ppo.py` 用 `model.actor.children()` 重建 actor prefix；RSL MLP 在位置 1/3/5 复用同一个 ELU，`children()` 会去重，因此错误路径只执行一次 ELU。
- canonical model900 CPU 数值合同：旧路径相对真实 actor rows 2/3 最大 raw-action 误差 `16.91664695739746`；改为 `list(model.actor)` 后最大误差 `1.430511474609375e-06`。
- 修复后的 `vae_ppo.py` SHA256：`fac67b21ebb03a89b88b50088c5e6aafc7dd4cad730fdaf47a2c1d44655d7ee1`。
- 修复后的 Stage B tests SHA256：`fdde231e0734ce97ce53c2f21d79c5e467b2976001f8a2ca46f0ce05f8af003e`；14 项测试通过，并新增共享 ELU 三次的回归测试。
- 原失败 state/handoff/smoke/preflight/ledger 已完整归档到：`tmp/highstep_student_recovery_20260712/failed_attempts/20260712_203509_actor_prefix_children_bug`。
- 显式恢复谱系：`tmp/highstep_student_recovery_20260712/correctness_fix_resume_manifest.json`；它强绑定旧 handoff、无效 checkpoint、修复文件 SHA 和 canonical model900 恢复起点。
- 当前恢复 supervisor PID/PGID：`236735`。
- 修复后 smoke 再次通过，随后从 canonical model900 启动 B300；启动时训练 PID/PGID 为 `242702`。
- 当前 B300 日志：`tmp/highstep_student_recovery_20260712/training/student_recovery_B300_20260712_222346/train.log`。
- 修复前/后首轮 Stage B MSE：`18.6569 -> 0.0554`；grad norm：`723.6128 -> 3.9449`；修复后的 actor-prefix runtime equivalence 为通过。
- Teacher、Student、action contract、LR、冻结范围和训练门禁均未改变；这次属于规范允许的实现正确性修复，不是新机制或失败后盲目重试。

## 11. 23:22 HKT 修正后 B300 与遗漏 checkpoint 补评

- 修正后 B300 已完整训练并完成 core9；不是因 token 或外部故障中断，而是 supervisor 在 `22:51:42` 依据当前严格门禁停止。
- baseline `model_900.pt`：`valid=9, pass=3, full=4, rear_hold=4, front_support=7, first_rear=4, no_severe=7`。
- 修正后 B300 `model_299.pt`：`valid=9, pass=3, full=3, rear_hold=3, front_support=7, first_rear=3, no_severe=8`；rear abs-y worst `0.00848 -> 0.01391`，rear width worst `0.21182 -> 0.34386`，dwell avg `156.56 -> 71.78`，dwell worst `469 -> 218`。
- 原 supervisor 把 full/hold 任意 `1/9` 下降都判为退化，因此 state 为 `stopped_by_gate`，best 仍为 canonical baseline `model_900.pt`。
- 为补齐规范“选择行为最好的 checkpoint”要求，停止后只读完成了同一修正 run 的中间 checkpoint core9；没有启动新训练或绕过门禁：
  - B100 `model_100.pt`：`valid=9, pass/full/hold=0, front_support=1, first_rear=0, no_severe=9`；
  - B200 `model_200.pt`：`valid=9, pass/full/hold=0, front_support=3, first_rear=0, no_severe=9`。
- B100/B200 均不合格；训练 checkpoint 中没有比 baseline 更好的可恢复点。权威补评记录：`tmp/highstep_student_recovery_20260712/diagnostics_corrected/poststop_checkpoint_audit.json`。
- 当前没有 train/eval/play，checkpoint、optimizer、日志、manifest 均保留。
- 尚未批准的唯一门禁歧义：正式规范写“完整上台明显减少”才停止；当前代码把任何 `1/9` 下降都算明显。若要从修正后 B300 full-resume 到预批准总 500，必须先明确 `1/9` 是否允许在 pass/front 不退、no-severe 与 geometry/dwell 改善时视为非明显退化。不得自行修改 LR、Teacher target、action contract、网络或另开机制。
