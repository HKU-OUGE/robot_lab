# Highstep 显示链卡死紧急暂停与恢复交接（2026-07-11）

更新时间：2026-07-11 23:49，Asia/Hong_Kong。

> **当前最高优先级状态：机器已重启，用户已明确允许恢复，clean B 已从 iter401 续训。**
> 恢复脚本已在 `2026-07-11 23:45 HKT` 成功执行一次，禁止再次运行。
> 当前实时事实以经校验的 schema-v4 `tmp/highstep_goal_orchestrator/handoff.json`、
> [clean B 对话迁移交接](highstep_dialogue_migration_clean_b_2026-07-11.md)和实际进程为准；
> 本文后续暂停细节保留为 checkpoint lineage 与故障审计证据。

## 〇、重启后恢复完成（当前入口）

```text
boot_id: 1a8ba203-7373-4c93-b2d2-3b2cea299bcc
resume time: 2026-07-11 23:45 HKT
task: RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0
role / phase: teacher_robust / teacher_robustification
checkpoint load mode: full
checkpoint source: model_400_resume_iter401.pt
planned execution: iteration 401..1999 (1599 updates)
new run_dir: /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_23-45-31
new train_log: /home/lxq/Softwares/robot_lab/tmp/highstep_display_reboot_resume_train_20260711_234523/train.log
train unit / startup PID: highstep-train-resume-20260711_234523.service / 207212
monitor unit / startup PID: highstep-monitor-resume-20260711_234523.service / 210939
monitor out: /home/lxq/Softwares/robot_lab/tmp/highstep_display_reboot_resume_schema4_monitor_20260711_234523
new W&B: https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/srxfqe9o
handoff event: display_reboot_resume:401:2026-07-11_23-45-31
automation thread: 019f4c6b-0fbd-73d2-91cc-6c72e8c8c35c (本对话)
```

启动验收确认 iteration `401 -> 442` 连续推进、monitor 第二次 120 秒心跳正常、
live handoff 再校验为 `ok`，且 orchestrator 已采纳新 run。`highstep-auto-loop.service`
已改绑本对话并为 `enabled/active`；控制器重启时 train/monitor PID 均未变化。
上述 PID 只代表启动快照，后续必须以实际 unit、argv 与 handoff 校验为准。

## 一、停机前最终状态

- 用户因 Linux 显示器卡死，要求温和中断训练与自动化、保留可靠断点、写交接后重启。
- `highstep-auto-loop.service` 已执行 `disable --now`：当前 `disabled/inactive`，重启后不会自动抢跑。
- 紧急重启守卫 `highstep-display-recovery-guard.service` 已停止。
- schema-v4 monitor PID `2803242` 于 `23:01:56` 收到 `TERM` 并记录 `monitor received interrupt`，先于训练退出，因此没有误启动 45 场评估。
- 训练 PID `2797557` 于 `23:02:17` 收到 `SIGINT`，约 2 秒内退出。
- W&B core PID `2799349` 随后退出；最后一次 HTTP 上传成功约在 `23:02:08`，本地 run 包写到 `23:02:17`。由于 `train.py` 没有信号 finally，日志没有标准 `wandb.finish` 尾句，远端应在恢复联网后复核，不得误报为 SDK 完整 finish。
- 停止后已确认：无 highstep `train.py`、`play.py`、schema-v4 monitor、W&B core 或 Codex resume 进程；NVIDIA 无计算进程。
- 训练日志最后完整输出 iteration `427/2000`；最后可靠定期保存断点为 iteration `400`，因此舍弃 401–427 共 27 个未 checkpoint 更新。
- 没有执行 45 场结构化评估；该评估必须等恢复训练完成至 iteration 1999 后由新 monitor 执行。

权威路径：

```text
task:
  RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0
role / phase:
  teacher_robust / teacher_robustification
paused run_dir:
  /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-29-20
train log:
  /home/lxq/Softwares/robot_lab/tmp/highstep_robust_clean_weights_train_20260711_222857/train.log
old W&B lineage:
  https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/lsoxm39v
old monitor output:
  /home/lxq/Softwares/robot_lab/tmp/highstep_robust_clean_weights_schema4_monitor_20260711_222857
```

停止前 iteration 427 的关键指标仅用于恢复对照，不代表通过评估：

```text
Mean reward                                      87.55
Episode_Reward/highstep_action_score              0.8066
Curriculum/terrain_levels                         0.9503
Curriculum/highstep_action_score/total            0.3116
Curriculum/highstep_action_score/support_score    0.6718
Curriculum/highstep_action_score/safety_score     0.4174
rear_approach_center_violation_rate               0.2141
rear_motion_center_violation_rate                 0.0993
support_floor_violation_rate                      0.0000
Episode_Termination/time_out                      0.9963
Episode_Termination/bad_orientation               0.0037
```

## 二、权威断点与 off-by-one 修正

原始定期断点必须永久保留：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-29-20/model_400.pt
size: 11425355 bytes
SHA-256: 9e45fe765c17ce1e631af4f1a087b7091237914a2ee32295d5b5a21654f47c33
checkpoint iter: 400
```

CPU `torch.load` 已验证它是完整 dict，含：

```text
model_state_dict
optimizer_state_dict
iter
infos
```

当前 RSL runner 载入 `iter=400` 后会从 `range(400, ...)` 再跑一次 400。为避免重复，已原子生成只修改顶层迭代游标的派生恢复锚点：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/2026-07-11_22-29-20/model_400_resume_iter401.pt
size: 11426255 bytes
SHA-256: 042429662c2e519647f744c55651578feef06c6b62c06ca5e036445cdd140ce4
checkpoint iter: 401
```

逐 tensor 深比较已确认：派生文件和原文件的所有模型参数、optimizer state、param groups 完全一致；唯一有意变化是顶层 `iter: 400 -> 401`。恢复时必须：

```text
--highstep_checkpoint_load_mode full
--max_iterations 1599
```

这会准确执行 iteration `401..1999`，最终生成 `model_1999.pt`；不得使用：

- `weights_only`：会丢 optimizer 与迭代连续性；
- 原 `model_400.pt + 1599`：会重复 400 且只到 1998；
- `full + 2000`：会错误跑到 2400 左右。

现有 checkpoint 不保存仿真环境状态、terrain/command curriculum 内部状态或 RNG 状态。因此可保证 policy、optimizer 和 iteration 连续，但不能声称位级完全无缝；环境会由现有框架重新初始化。

## 三、重启后恢复方法（已执行，禁止重复）

以下命令已于 `2026-07-11 23:45 HKT` 成功执行，只保留作审计记录。当前不得再次执行；
后续由现有 train、schema-v4 monitor 和 auto-loop 接管：

```bash
cd /home/lxq/Softwares/robot_lab
bash -n tmp/resume_highstep_after_display_reboot_20260711.sh
./tmp/resume_highstep_after_display_reboot_20260711.sh
```

恢复脚本：

```text
/home/lxq/Softwares/robot_lab/tmp/resume_highstep_after_display_reboot_20260711.sh
SHA-256: 2a18691be45864f947191cdc3c82525621c81b1c389296843798fa19822b7c99
```

脚本会按以下顺序 fail-closed 执行：

1. 校验原始/派生 checkpoint、schema-v4 monitor、Robust smoke 和本交接文件的存在与哈希；
2. 拒绝在已有 highstep train/play 或已启动 controller 时运行；
3. 用独立 systemd user service 启动 continuation，参数为 seed 42、4096 env、`full`、1599 updates、headless、W&B；
4. 不复用旧 W&B ID，而是创建新的在线 run，并在 handoff 中保留 `parent_wandb=lsoxm39v`。旧 W&B 已记录到 427，而恢复锚点为 401；强行复用会造成 step 401–427 重复/乱序；
5. 从训练日志解析严格的 `YYYY-MM-DD_HH-MM-SS` 新 run 名，禁止 `--run_name` 后缀，确保 orchestrator 的 canonical name 校验通过；
6. 启动独立 schema-v4 monitor，比较 checkpoints 仍为 `1100/1500/1999`，最终执行 3 checkpoints × 3 seeds × 5 scenarios = 45 场；
7. 原子写入带新 PID/run/log/W&B/monitor 的 schema-v4 `handoff.json`，先调用 orchestrator 校验器验证；
8. 只有交接验证通过后才重新 `enable --now highstep-auto-loop.service`；若中途失败，脚本会停止自己启动的 train/monitor，避免孤儿训练。

恢复成功后必须验证：

```bash
python3 tmp/highstep_status_dashboard.py --no-color
systemctl --user is-active highstep-auto-loop.service
tail -30 tmp/highstep_goal_orchestrator/orchestrator.log
```

应看到：

```text
LIVE_HANDOFF_ADOPTED
checkpoint_load_mode=full
run_dir 为新 YYYY-MM-DD_HH-MM-SS
train PID、monitor PID、train_log 和 handoff 完全一致
```

## 四、代码与配置指纹

```text
branch: dev_lxq_new
HEAD: effd19945034e609f4f5221cf106114852f32882
工作树：dirty，必须原样保留，不得 reset/checkout 覆盖
训练启动 diff snapshot:
  run/git/robot_lab.diff
  SHA-256 129c33a336013031f245469b0c289bcd2f449c829ac4d2459e48647540b1975e
agent.yaml:
  SHA-256 00882d07ab0ea580820ec105daf92b3ecf445f8840bf1cac13d6fef4fe72cf81
env.yaml:
  SHA-256 ed3e9158577c5e6ea5938f29e3474c55f535ffe0e648de07a1ee137f04a4e905
schema-v4 monitor:
  SHA-256 3f963e31dfd9263b391f48faae0c70bd52e82416afb2f9e1f279b6588e5b0929
Robust smoke log:
  SHA-256 562f0f860d699cd1f8d7d71802480e54a779cf6bb6280766379dcd7fdcb42091
Robust smoke model_0:
  SHA-256 96f3ef57273b911a6301e31f0a26dea729e3cd172af8f8624f60471b481927b2
```

本轮配置摘要：

```text
seed: 42
num_envs: 4096
save_interval: 100
optimizer learning_rate: 0.0001
schedule: adaptive
comparison checkpoints after resume: 1100, 1500, 1999
```

最终清理复核（`2026-07-11 23:14 HKT`）：两个遗留 `tail -f` 日志跟随器与一个
`highstep_status_dashboard.py --follow` 只读看板均已停止；训练、评估、schema-v4
监督器、W&B、自动恢复子任务及 NVIDIA GPU 计算进程仍全部为零。随后再次执行了
`sync`。此清理不改变上述 checkpoint、恢复边界或恢复命令。

## 五、显示故障与重启边界

显示器卡死不是整机、训练进程、内存、磁盘或 NVIDIA 计算卡死：

- AMD iGPU 承担实际显示；NVIDIA RTX 4060 Ti 的 display active 为 disabled，计算正常且无 NVRM/Xid；
- `2026-07-11 18:56:05` 起，Xorg 子进程 `1388539` 长期处于不可中断 D-state，`wchan=drm_sched_entity_flush`；
- 同秒 Firefox 报 GPU forced reset、VS Code GPU process 退出、GNOME Shell 重启但 D-Bus 对象未完整恢复；
- `22:39:32` 内核记录：`amdgpu CRTC:79 hw_done or flip_done timed out`；
- 根因高置信为 AMD+NVIDIA 混合 X11 路径中的 AMD DRM hang。

不要再读取卡死 PID 的 `/proc/.../syscall`/`stack`，不要 `lsof -p`，不要热 reset AMD GPU；这些操作会产生新的 D-state 等待者。

当前账户 `CanReboot=challenge`，Codex 无法安全提供 sudo 密码。所有训练产物已同步并执行 `sync` 后，使用已认证 SSH 终端执行：

`23:08` 已实际尝试 `systemctl reboot --no-wall`，logind/polkit 明确返回 `Interactive authentication required`；这不是训练或系统故障，而是正常权限边界。不要把 sudo 密码发送到对话中。

```bash
sudo reboot
```

重启维护建议：

1. 更新 Ubuntu HWE kernel（本机当前 `6.8.0-124`，APT 已有 `6.8.0-134`）与 `linux-firmware`/Mesa；
2. 长期将显示器线只接 RTX 4060 Ti，BIOS 设 PEG/PCIe 主显并关闭 iGPU multi-monitor；确认物理接线后再考虑 `sudo prime-select nvidia`；
3. 根分区使用率约 94%、剩余约 32 GiB，恢复训练前后安排旧 run/cache 清理，但不得删除本文件列出的 lineage、checkpoint、manifest、smoke 和当前 run。

## 六、禁止事项

- 恢复脚本已经成功执行一次，不要再次运行或并行启动另一条训练；以当前新 handoff 为准。
- 不要从 `model_427` 恢复：该文件不存在；日志到 427 不等于有 427 断点。
- 不要覆盖或删除原始 `model_400.pt` 与派生 `model_400_resume_iter401.pt`。
- 不要复用旧 W&B ID `lsoxm39v`，避免恢复步数重叠。
- 不要在用户完成数据查看/输入前自动启动 GPU 训练。
- 不要把显示卡死归因于当前 22:29 启动的训练；显示故障在 18:56 已发生。
