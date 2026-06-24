# 2026-06-24 highstep 当前交接状态

## 当前目标
- 伸缩腿高台 highstep teacher 继续调参。
- 用户重点观察现象：
  - `model_110000` 与 `model_112700` 视频差异不大，`112700` 稍好但不多。
  - 大多数情况下左后腿先上高台；卡死频率降低，但仍会慢慢卡一会再上去。
  - 手动改变 yaw 后，有时能让右后腿先上；这只是现象线索，不能直接加 yaw reward/constraint。
  - 用户怀疑 `112700` 左后腿优先比 `110000` 更严重。

## 当前代码策略
- 不再新增 yaw/approach/branch-choice 训练约束。
- 不再保留 speculative rescue/stall reward。
- 当前应该是“主上台能力 baseline + 纯诊断”：
  - `rear_feet_highstep_clearance` 使用 `max(rear feet)` 和 `max(rear_x)`，保留“一条后腿先登台撑起身体”的成功路径。
  - `rear_second_foot_highstep_clearance.weight = 0.95`。
  - `highstep_rear_box_push.weight = 1.10`。
  - `highstep_box_phase_prior.weight = 0.85`。
  - `rear_one_sided_highstep_stall` 已从当前 highstep config、stage terms、resume staged reward list 中移除，并从 `rewards.py` 删除。
  - `rescue_only` 已删除，不应再出现在当前 config/train/rewards 中。
  - 保留 `highstep_rear_branch_metrics` 作为纯诊断，记录 RL/RR 先上、单后腿卡住、第二后腿补上等指标。

## 已知不能再犯
- `--max_iterations` 是“本次运行训练多少 iteration”，不是“训练到哪个 model 编号”。
- 从 `model_108000.pt` 短训到 `model_108500` 附近，应使用 `--max_iterations 500`。
- 从 `model_108000.pt` 短训到 `model_109500` 附近，应使用 `--max_iterations 1500`。
- 严禁再给 `--max_iterations 109500` 这种错误命令。
- 每次修改训练相关代码前必须先备份，备份文件名必须包含对应训练包/起点。

## 当前备份
- `highstep_env_cfg_backup_after_failed_2026-06-24_19-30-09_from_2026-06-24_00-36-54_model_108000.py`
- `rewards_backup_after_failed_2026-06-24_19-30-09_from_2026-06-24_00-36-54_model_108000.py`
- `curriculums_backup_after_failed_2026-06-24_19-30-09_from_2026-06-24_00-36-54_model_108000.py`
- `train_backup_after_failed_2026-06-24_19-30-09_from_2026-06-24_00-36-54_model_108000.py`

## 训练判断
- 不建议长训。
- 如果用户决定验证，只能短训。
- 当前推荐短训命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-24_00-36-54/model_108000.pt \
  --max_iterations 500
```

## 短训后只看这些
- `Curriculum/highstep_rear_branch_metrics/valid_rate` 必须非零，否则诊断无效。
- `rl_first_rate / rr_first_rate / rl_minus_rr_first_rate` 用来判断左/右后腿先上偏置，不要凭感觉。
- `one_sided_stall_ratio` 用来判断单后腿卡住趋势。
- `rear_box_push / highstep_box_phase_prior / rear_legs_drive_bonus / rear_feet_highstep_clearance` 不能明显低于 `2026-06-24_00-36-54` 同阶段，否则主上台能力仍受损。
- `bad_orientation` 不能上升。

## 下一步禁令
- 不要再大改 reward。
- 不要再因为“手动 yaw 能改变后腿先后”就添加 yaw reward/constraint。
- 如果要改，必须先逐条映射用户现象、先备份、只做一个小改动、写清失败回滚条件。
