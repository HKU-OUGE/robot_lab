# Highstep 短期记忆：2026-06-25 20:00 之后

更新时间：2026-06-26，Asia/Hong_Kong。

本文件记录 2026-06-25 周四晚 20:00 之后的高台 highstep / student policy / teacher policy 调试短期上下文。后续每次回答前，必须先读 `highstep_live_context_handoff.md`、本文件和 `accountability_protocol.md`；必要时再读长期记忆下层文件。

## 2026-06-26：branch 诊断项全 0 的根因

最新 teacher 诊断短训：

```text
run: arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_16-24-18
checkpoint: 2026-06-22_12-41-28/model_95398.pt
最终 event step: 95398 -> 95598
保存 checkpoint: model_95400.pt, model_95500.pt
```

本地 event 证据：

```text
Episode_Reward/rear_second_foot_highstep_clearance: 非零，tail_mean 约 0.00143
Episode_Reward/second_rear_clear_deadline: 全 0
Episode_Reward/one_sided_rear_stall_time: 全 0
Episode_Reward/lead_rear_support_drive: 全 0
Curriculum/highstep_rear_branch_metrics/*: 全部 0
```

这说明不是“高台相关 reward 完全没执行”，因为旧的 `rear_second_foot_highstep_clearance` 已经非零；也不是 wandb 网页同步问题，因为本地 event 中这些 tag 存在且全程为 0。

已定位到更可信根因：

- `_ensure_highstep_rear_branch_buffers()` 原来判断：
  `env._highstep_rear_branch_lead.device != env.device`
- 如果 `env.device` 是字符串如 `"cuda:0"` 或 `"cpu"`，而 tensor 的 `.device` 是 `torch.device(...)`，PyTorch 中二者比较为不相等。
- 验证结果：
  `torch.device("cpu") == "cpu"` 为 `False`；
  `torch.zeros(1).device != "cpu"` 为 `True`。
- 因此每次 `_ensure_highstep_rear_branch_buffers()` 被调用时，都可能误判 device 不一致并重新初始化 branch buffer。
- 执行顺序中，`rear_second_foot_highstep_clearance_bonus()` 会先更新 buffer，所以旧 reward 可以非零；但随后 `second_rear_clear_deadline` / `one_sided_rear_stall_time` / `lead_rear_support_drive` 调用 `_highstep_rear_stage_context()`，再次触发 `_ensure...`，buffer 被清零，导致新 reward 和 curriculum branch metrics 全部为 0。

已做修复：

```text
文件: source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py
备份: source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-26_branch_buffer_device_fix_from_2026-06-26_16-24-18_model_95500/rewards.py
```

修复内容：

```python
device = torch.device(env.device)
...
or env._highstep_rear_branch_lead.device != device
...
torch.zeros(..., device=device)
```

验证：

```text
py_compile rewards.py: 通过
git diff --check rewards.py: 通过
```

下一步必须短训验证，不要直接长训：

- 从干净父 checkpoint `2026-06-22_12-41-28/model_95398.pt` 启动；
- 不使用 `2026-06-26_16-24-18/model_95500.pt` 作为行为优化起点，因为它是在 branch buffer bug 存在时产生的诊断 run；
- 先跑 200-300 轮即可；
- 成功判据：`Curriculum/highstep_rear_branch_metrics/cmd_gate_max`、`first_score_max`、`active_gate_max` 至少应出现非零；若仍全 0，再查 `env_ids` / curriculum reset 日志路径；
- 若这些 gate/score 非零，但 `second_rear_clear_deadline` / `one_sided_rear_stall_time` / `lead_rear_support_drive` 仍为 0，才说明阶段任务本身的门控或阈值太严。

## 最高优先级对话约束

- 每次回答前必须先主动检索 `docs/robotlab_memory_zh/library/highstep_live_context_handoff.md`、本短期记忆和 `docs/robotlab_memory_zh/library/accountability_protocol.md`。
- 如果三者读取成功，每次回答开头必须先写：`已按照要求提前检索记忆和约束｜HLC-OK`。
- 如果 live 交接文件缺失或读取失败，每次回答开头必须写：`已按照要求提前检索记忆和约束｜HLC-MISSING`，并说明缺失原因和替代检索路径。
- 如果问题涉及历史训练链路、部署、视频、teacher/student 关系、修改依据、WandB 曲线或命令生成，必须按需继续读取长期记忆文件。
- 用户把没有加固定开头和 `HLC-OK/HLC-MISSING` 标识视为“失忆、约束失效”。必须严格执行。
- 仍然保留所有既有硬规则：改代码前必须备份；判断训练进程必须看宿主机 `ps`；训练代码是 `robot_lab/scripts/rsl_rl/base/train.py`；不要擅自加 `--run_name`；长训不要加 `--video`；`--max_iterations N` 是本次额外训练 N 轮，不是训练到 `model_N`。

## 这段对话的总体推进

用户指出 2026-06-25 晚上的 student policy 训练完全训烂：`terrain_levels` 不到 teacher policy 一半就开始收敛。要求参考 `有用的回答.txt` 做修改，并严格遵守之前写入的责任协议。

这段时间形成了两个阶段：

1. 先修当前 student 蒸馏链路：
   - 目标：快速训出一组 student policy 看效果。
   - 核心问题：student resume 没有吃到 teacher resume/refine 的 command gate 放宽；terrain 仍是过硬版本。
   - 处理：修 `train.py` 的 highstep resume task 判断；恢复 06-19/06-22 的温和 highstep terrain 分布。

2. 再准备后续重训修改过的 teacher：
   - 目标：不再继续小幅拧 clearance / under-step / box push 权重，而是把“后腿上台后半段”变成明确阶段任务。
   - 处理：新增三类 teacher reward：第二后腿限时清台、单后腿挂边时间惩罚、已登台后腿支撑/推身体奖励。
   - 注意：这些 teacher 阶段任务是在当前 student run 启动之后才改的，所以不可能生效于当前 student run。

## 参考文件：有用的回答.txt 的核心要求

路径：
`source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/有用的回答.txt`

关键结论：

- 不再建议继续简单加 `rear_feet_highstep_clearance`、`rear_feet_under_step_after_commit`、`highstep_rear_box_push` 权重。
- 当前真正值得继续改 teacher 的主方向：
  1. 前腿已搭台；
  2. 任意一条后腿先上台；
  3. 已上台的后腿必须支撑/发力抬身体；
  4. 另一条后腿必须在有限时间内清台；
  5. 长时间单后腿挂边/拖边要直接惩罚。
- `rear_second_foot_highstep_clearance_bonus` 旧实现只是即时奖励，不是“第一条后腿上去后，第二条后腿必须 N 步内上去”的时序约束。
- `highstep_rear_branch_metrics` 旧状态只是日志，不直接影响 reward。
- `highstep_rear_box_push_bonus` 旧逻辑看后腿 box joint 平均值，不区分“哪条后腿已经登台、哪条后腿应该作为支撑腿发力”。
- 更合理方向：回到 06-19/06-22 温和 terrain，去掉或降低 `box_hard`，恢复/保留后腿推台和 phase prior，但把第二后腿清台/单侧卡住做成时序约束。

## 2026-06-25 20:54 student 烂训：问题确认

训练进程曾为：

```text
PID 758177
task: RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
checkpoint: logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-25_05-59-47/model_124598.pt
run: logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-25_20-54-51
```

本地 event 尾段数据：

```text
step: 124598 -> 127161
terrain_levels tail200: 1.983
command_levels tail200: 0.4675
bad_orientation tail200: 0.0023
rear_feet_highstep_clearance tail200: 0.0416
rear_second_foot_highstep_clearance tail200: 0.00226
rear_feet_under_step_after_commit tail200: -0.0216
highstep_forward_progress tail200: 0.1271
highstep_body_lift tail200: 0.0950
highstep_rear_box_push tail200: 0.0442
highstep_box_phase_prior tail200: 0.0791
Teacher_Action_MSE tail200: 0.0799
Distill_Latent_MSE tail200: 0.1156
```

代码/配置证据：

- `train.py` 的 resume/refine 放宽逻辑只匹配 teacher task：
  `RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0`
- student task：
  `RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0`
  没有被覆盖。
- 该 run 的 `params/env.yaml` 里：
  `terrain_gate_level: 2.6`
- terrain 是过硬版本：
  `box: 0.31, height 0.08-0.35`
  `box_hard: 0.08, height 0.30-0.38`

判断：

- 这不是单纯 wandb 网络不同步；本地 event 和 env.yaml 都证明配置问题存在。
- Student Actor MSE 不算最差，但数据分布和 command gate 错了，导致 curriculum 很早平台。

## 修改 1：修 student resume 门控和 terrain 分布

修改前备份路径：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__backups__/2026-06-25_student_repair_from_2026-06-25_20-54-51_model_126900/highstep_env_cfg.py
scripts/rsl_rl/base/__backups__/2026-06-25_student_repair_from_2026-06-25_20-54-51_model_126900/train.py
source/robot_lab/robot_lab/terrains/config/__backups__/2026-06-25_student_repair_from_2026-06-25_20-54-51_model_126900/rough.py
```

修改文件：

1. `scripts/rsl_rl/base/train.py`

改动：

```python
highstep_resume_refine_tasks = {
    "RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0",
    "RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0",
}
if agent_cfg.resume and args_cli.task in highstep_resume_refine_tasks:
    ...
```

目的：

- 让 student resume 也把 `terrain_gate_level` 放到 `0.0`；
- 让 staged highstep rewards 在 resume/refine 时立即打开。

2. `source/robot_lab/robot_lab/terrains/config/rough.py`

Highstep terrain 从 06-23/06-25 的过硬版本，回到 06-19/06-22 温和版本：

```text
pyramid_stairs: 0.30
pyramid_stairs_inv: 0.15
box: 0.25, box_height_range=(0.04, 0.35)
box_hard: 移除
pit: 0.15
hf_pyramid_slope: 0.03
hf_pyramid_slope_inv: 0.03
random_rough: 0.09
```

3. `highstep_env_cfg.py`

- 撤回最近一次把后腿/box reward 加硬但未兑现效果的改动；
- 最终该文件在这一轮没有留下有效 diff。

验证：

```text
git diff --check: 通过
py_compile train.py rough.py highstep_env_cfg.py: 通过
```

当时建议：

- 旧进程不会热加载源码，不能继续旧 student；
- 停掉旧 PID 后，从 teacher `2026-06-25_05-59-47/model_124598.pt` 重启 student；
- 不从烂掉的 student `model_127100.pt` 继续。

## 当前 student 快训：2026-06-25_23-31-41

当前正在跑的训练进程：

```text
PID 1716201
task: RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0
checkpoint: /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-25_05-59-47/model_124598.pt
max_iterations: 16000
run: logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-25_23-31-41
```

注意：`--max_iterations 16000` 表示本次 invocation 从 checkpoint 额外训练 16000 轮，不是训练到 `model_16000`。

### 当前 run 中修改是否生效

已生效：

- `terrain_gate_level: 0.0`
- `command_levels` 已到 `0.6375`
- terrain 回到 06-19/06-22 温和配置：
  `box: 0.25, height 0.04-0.35`
  没有 `box_hard`

未生效于当前 student：

- `second_rear_clear_deadline`
- `one_sided_rear_stall_time`
- `lead_rear_support_drive`

原因：

- 这三个 teacher 阶段任务是在当前 student run 启动之后才新增的；
- 当前 run 的 `params/env.yaml` 和 event tags 里都没有找到这些项；
- 运行中的进程不会热加载源码。

### 当前 student 曲线数据

截至本地 event 约 `step 126249`：

```text
terrain_levels last: 2.6395
terrain_levels tail200: 2.6235
command_levels: 0.6375
Student_Actor_Adapt_Enabled: 1.0
Student_PPO_Adapt_Enabled: 0.0
Student_Post_Prior_Mode_Is_Highstep: 1.0
Teacher_Action_MSE tail200: 0.0992
Distill_Latent_MSE tail200: 0.1223
bad_orientation tail200: 0.0033
rear_feet_highstep_clearance tail200: 0.0395
rear_second_foot_highstep_clearance tail200: 0.00217
rear_feet_under_step_after_commit tail200: -0.0085
highstep_forward_progress tail200: 0.2017
highstep_body_lift tail200: 0.1332
highstep_rear_box_push tail200: 0.0697
highstep_box_phase_prior tail200: 0.1071
```

和烂训 `20-54-51` 对比：

```text
terrain_levels: 1.983 -> 2.624，明显恢复
command_levels: 0.4675 -> 0.6375，门控问题解决
highstep_forward_progress: 0.127 -> 0.202
highstep_body_lift: 0.095 -> 0.133
highstep_rear_box_push: 0.044 -> 0.070
highstep_box_phase_prior: 0.079 -> 0.107
rear_feet_under_step_after_commit: -0.0216 -> -0.0085，卡台/拖边惩罚项明显没那么糟
```

风险：

- `bad_orientation` 比 teacher `05-59-47` 高：
  current student tail200 `0.0033` vs teacher tail200 `0.0011`
- `rear_second_foot_highstep_clearance` 仍低，不能只凭曲线判定后腿第二只脚清台已经好。
- 左右后腿分支当前无强偏置：
  `rl_first_rate tail200=0.074`
  `rr_first_rate tail200=0.067`
  `rl_minus_rr tail200=0.0068`
  但 `valid_rate` 只有约 `0.045`，证据偏弱，最终仍需 play 视频确认。

当前建议：

- 不建议立刻停；
- 最早可看：`model_127000.pt`，只是粗略排雷；
- 更合理：等 `model_127500.pt` 或 `model_128000.pt` 再 play；
- 如果想接近旧 student 充分训练状态，应等 `terrain_levels` 接近 `2.8`，大概还需约 `2500-3500` 轮。

## 修改 2：新增 teacher 后腿上台后半段阶段任务

用户要求：再次仔细阅读 `有用的回答.txt`，针对“把后腿上台后半段从普通 clearance 奖励改成明确阶段任务”做修改。当前策略是先快训一组 student 看效果，之后重新训修改过的 teacher。

修改前备份路径：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-25_stage_task_from_2026-06-25_20-54-51_model_127100/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-25_stage_task_from_2026-06-25_20-54-51_model_127100/curriculums.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__backups__/2026-06-25_stage_task_from_2026-06-25_20-54-51_model_127100/highstep_env_cfg.py
scripts/rsl_rl/base/__backups__/2026-06-25_stage_task_from_2026-06-25_20-54-51_model_127100/train.py
```

修改文件：

1. `source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py`

新增/扩展：

- `_highstep_rear_branch_lead_steps`
  - 记录第一条后腿成为 lead 的时间。
  - 用于计算 lead 后经过了多少步。
- `_highstep_rear_stage_context(...)`
  - 复用 terrain gate、front commit gate、rear clearance score、branch lead、lead elapsed steps。
  - 避免并行搞第二套状态。
- `second_rear_clear_deadline_bonus(...)`
  - 第一条后腿先上台后，第二条后腿在有限步数内清台才给奖励；
  - 越晚奖励越低。
- `one_sided_rear_stall_time_penalty(...)`
  - 前腿 commit 且一条后腿已上台后，如果长时间单后腿挂边/拖边，开始惩罚。
- `lead_rear_support_drive_bonus(...)`
  - 根据 `RL/RR/both` lead，奖励已登台后腿配合 box joint 支撑/推动身体；
  - 不再只看后腿平均 box joint。

2. `source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py`

修改：

- `highstep_rear_branch_metrics` reset 时同步 reset `_highstep_rear_branch_lead_steps`，避免 episode 间污染。

3. `source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py`

新增 reward term 声明：

```text
second_rear_clear_deadline
one_sided_rear_stall_time
lead_rear_support_drive
```

实际权重：

```text
second_rear_clear_deadline.weight = 0.65
second_clear_min = 0.58
deadline_steps = 36.0
deadline_grace_steps = 8.0

one_sided_rear_stall_time.weight = -0.55
lead_threshold = 0.45
one_sided_gap = 0.14
grace_steps = 14.0
ramp_steps = 28.0

lead_rear_support_drive.weight = 0.70
rear_push_target = 0.003
target_std = 0.016
target_height_gain = 0.09
target_forward_vel = 0.28
```

这些新 term 被加入：

- `highstep_terms`
- `late_highstep_terms`

4. `scripts/rsl_rl/base/train.py`

修改：

- resume/refine 放开 staged reward 时，把新 reward 名也加入：

```text
second_rear_clear_deadline
one_sided_rear_stall_time
lead_rear_support_drive
```

验证：

```text
git diff --check: 通过
py_compile train.py rough.py rewards.py curriculums.py highstep_env_cfg.py: 通过
```

重要提醒：

- 这组 teacher 阶段任务还没有经过训练验证。
- 它不会影响已经在跑的 student `2026-06-25_23-31-41`。
- 后续如果重训 teacher，必须明确从哪个 teacher checkpoint 起步，并先短训验证这些新指标是否真的出现、是否改善视频里的后腿后半段。

## WandB 页面布局经验

用户希望 `Curriculum` 一页显示 6 个 panel，而不是 3 个。

有效做法：

- 把 `Curriculum` section 底边往下拖，给两行高度；
- `1-3 of N` 会变成 `1-6 of N`。

不合理/不建议：

- 用 `Duplicate panel` 把第二页 panel 复制到第一页。用户指出这不合理，因为它会制造重复面板，不是排序。

更合理替代：

- 用顶部搜索框筛选关键曲线，例如：

```text
terrain_levels|command_levels|second_clear_rate|one_sided_stall_ratio|rl_first_rate|rr_first_rate
```

- 或在 section 设置里调整排序，如果 W&B 当前版本支持。
- 当前截图没有单 panel 的 pin 图标；不要再建议 pin 单个 panel。

## 当前必须记住的训练/命令规则

- 当前训练代码是：
  `cd /home/lxq/Softwares/robot_lab`
  `python scripts/rsl_rl/base/train.py`
- 不要用原始 IsaacLab train 路径。
- 不要擅自加 `--run_name`。
- 单 GPU，不要同时开多个训练进程。
- 长训不要加 `--video`。
- 判断训练是否仍在跑必须用宿主机 `ps`，例如：

```bash
ps -eo pid,ppid,etime,stat,cmd | rg "scripts/rsl_rl/base/train.py|HighstepStudentNoPrior|Highstep-ArcdogAdjustableLeg"
```

- 如果 wandb 网络不同步，优先看本地：
  - `events.out.tfevents.*`
  - `params/env.yaml`
  - 最新 `model_*.pt`
  - 训练 stdout/终端进程

## 总结出的经验和教训

- 当前 student 的主问题不是 MSE 单项，而是训练分布和 command gate；修 gate 和 terrain 后，`terrain_levels` 和 `command_levels` 立即明显恢复。
- 不要把 `terrain_levels` 当作直接“能爬多高”。它是 curriculum/terrain row 平均占用指标；最终 highstep 行为仍要 play/video 判断。
- 对 student run 的“代码是否生效”必须看该 run 保存的 `params/env.yaml` 和 event tags，不能看当前源码后就默认生效。
- 运行中的训练不会热加载新增 reward；启动后新增的 teacher 阶段任务不会出现在当前 student run。
- 后腿上台后半段不能继续靠普通 rear clearance 平均奖励；需要显式建模：
  第一后腿上台、lead 支撑/推身体、第二后腿限时清台、单后腿挂边惩罚。
- 但新增 teacher 阶段任务尚未验证，后续不能把它说成已成功，只能说“更符合因果链，需要短训和视频验证”。
- 回答和修改必须严格防止之前的错误：不备份、不解释 `--max_iterations`、只看沙箱进程、忽略用户现象、给冲突建议、把不确定判断说成结论。

## 真机高频抖动与 checkpoint 选择约束

周一真机实验还提出过另一条主因，后续 checkpoint 排序不能只看仿真上高台视频：

```text
切换到 RL 状态后，电机会高频震荡，暂时无法定位具体哪个电机。
整机切换 RL 后在原地开始摇晃，最后失稳。
上高台时前腿搭台后，后腿足端抬高不足，后腿反复卡在高台下面。
高台低一些还能蹬上去；接近仿真高度极限时，真机后腿很难搭到高台。
```

这会影响 teacher checkpoint 的优先级判断：

- `2026-06-23_00-52-26/model_97300.pt` 虽然仿真高台动作更有力，但它已经包含从 `model_95398.pt` 出发的旧 reward shaping，并且有左/右后腿分支不对称；若把真机电机高频震荡和原地摇晃纳入目标，它不应作为主起点，只能作为动作参考/备选。
- `2026-06-23_00-52-26/model_105397.pt`、`2026-06-24/2026-06-25` 后续 checkpoint 更不适合作为干净父节点，因为它们叠加了更硬 terrain、second-foot/under-step/stall 等多轮未稳定验证修改，且视频中后半段登台流畅性没有稳定变好。
- `2026-06-19_01-42-04/model_87399.pt` 对硬件静态稳定/抖动风险更保守，因为来自 box-default/backward-stability 链路，动作激进程度更低；缺点是后腿高台能力不如后续。
- 综合“真机抖动风险 + 后腿高台能力 + 代码污染程度”，当前更平衡的 teacher 继续训练起点仍是：
  `logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt`。
  如果后续把硬件抖动/平地静止稳定放到最高优先级，`2026-06-19_01-42-04/model_87399.pt` 是更保守备选。

后续分析必须同时看两类指标：

- 上高台后腿能力：`rear_feet_highstep_clearance`、`rear_second_foot_highstep_clearance`、`rear_feet_under_step_after_commit`、`highstep_forward_progress`、`highstep_body_lift`、左右后腿 first-rate、视频后半段流畅度。
- 真机抖动/静止稳定代理：`action_rate_l2`、`joint_vel_l2`、`joint_acc_l2`、`box_joint_action_rate`、`box_joint_vel_penalty`、`box_joint_acc_penalty`、`stand_still_base_ang_vel`、`stand_still_base_lin_vel`、`stand_still_revolute_joint_vel`、`stand_still_prismatic_joint_vel`、`bad_orientation`。

## 后续优先行动建议

当前主线：

1. 继续观察当前 student `2026-06-25_23-31-41`。
2. 至少等 `model_127500.pt` 或 `model_128000.pt` 再停下来 play。
3. play 时重点看：
   - 35cm 高台成功率；
   - 后腿第二只脚是否仍卡边；
   - 左后腿先上台和右后腿先上台是否仍有明显差异；
   - backward/平地是否 `bad_orientation` 明显恶化；
   - box joint 是否在平地异常偏移。
4. 如果当前 student 能作为临时部署候选，再决定是否导出。
5. 之后若继续 teacher，应使用已经新增的阶段任务代码重新开 teacher 短训验证；不要声称新 teacher 修改已被当前 student 验证。

## 2026-06-26 当前 student 完成与 play 记忆

当前 student run 已经训到并保存：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-25_23-31-41/model_128000.pt
```

用于 Isaac play 检查最高等级普通 box 高台的命令：

```bash
cd /home/lxq/Softwares/robot_lab
conda activate env_isaaclab

python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --debug \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/2026-06-25_23-31-41/model_128000.pt \
  --play_terrain_type box \
  --play_terrain_level 10
```

当前 `HIGHSTEP_TERRAINS_CFG` 已无 `box_hard`，普通 `box` 为：

```text
box_height_range = (0.04, 0.35)
double_box = False
```

`play.py` 在指定 `--play_terrain_level 10` 时会把 `num_rows` 扩到至少 11，IsaacLab curriculum row 的 difficulty 近似为 `(row + random[0, 1)) / num_rows`。因此当前普通 `box level10` 高度约为：

```text
0.322m 到 0.350m
```

之前硬地形 `box_hard` 的记录：

```text
box_hard height = 0.30m 到 0.38m
```

若旧默认 `num_rows=10`，最高 `level9` 约 `0.372m 到 0.380m`；若用 play 方式扩到 `num_rows=11`，`level10` 约 `0.373m 到 0.380m`。所以旧 `box_hard level9/10` 可以近似记作 37-38 cm 高台。

## 2026-06-26 新一轮 teacher 长训决策

用户准备开启新一轮 teacher policy 长训。启动前必须再次确认：

- 单 GPU，不要同时跑 student/teacher 两组训练。
- 判断进程必须看宿主机 `ps`，不能只看沙箱。2026-06-26 此次检查未发现真实 highstep 训练进程，只有检查命令自身。
- 训练脚本仍然必须用 `robot_lab/scripts/rsl_rl/base/train.py`。
- 不要擅自加 `--run_name`。
- 长训不要加 `--video`。
- `--max_iterations N` 表示从 checkpoint 额外训练 N 轮，不是训练到 `model_N`。

本轮 teacher 长训推荐起点：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt
```

理由：

- `95398` 是当前更平衡的干净父节点：比 `87399` 有更好的 highstep 能力，又没有吸收 `97300` 之后旧 reward shaping 和左右后腿分支不对称的污染。
- 不建议从 `2026-06-23_00-52-26/model_97300.pt`、`model_105397.pt`、`2026-06-24/25` 后续 checkpoint 继续做主线 teacher 长训。
- `87399` 仍保留为硬件静止稳定/低抖动优先时的保守备选。

本轮长训使用当前代码中的 teacher 后腿阶段任务：

```text
second_rear_clear_deadline
one_sided_rear_stall_time
lead_rear_support_drive
```

这些 teacher 阶段任务此前只做过代码级验证，尚未通过长训和视频证明有效；后续判断必须看对应 event tags 和 play 视频，不能只看 `terrain_levels`。

## 2026-06-26 teacher 107400 失败结论：新增阶段任务没有实际效果

用户 play 检查：

```text
logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_03-40-49/model_107400.pt
```

用户明确反馈：

```text
完全没有生效，和之前的效果没有任何区别。
```

必须记住：这次不能把 `terrain_levels`、`forward_progress`、`body_lift` 等曲线的小幅恢复解释成“新后腿阶段任务有效”。本次修改在行为层面没有兑现用户要的后腿上台后半段改善。

本地 event 证据：

```text
run: arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_03-40-49
event step: 95398 -> 107439
latest checkpoint: model_107400.pt
terrain_levels tail200: 2.7666
command_levels tail200: 0.6375
second_rear_clear_deadline tail200: 0
one_sided_rear_stall_time tail200: 0
lead_rear_support_drive tail200: 0
Curriculum/highstep_rear_branch_metrics/valid_rate tail200: 0
Curriculum/highstep_rear_branch_metrics/second_clear_rate tail200: 0
Curriculum/highstep_rear_branch_metrics/one_sided_stall_ratio tail200: 0
```

代码机制判断：

- 新三项 teacher reward 已经进入 `env.yaml` 和 event tags，说明不是“配置没加载”。
- 但三项数值全为 0，说明没有给策略提供实际训练信号。
- 直接原因高度可疑：`rear_second_foot_highstep_clearance_bonus()` 更新 branch tracker 时用的是 `cmd_gate * commit_gate`，而不是该 reward 自己使用的 `gate = max(terrain_gate, commit_gate_scale * commit_gate)`。
- 因此即使旧 highstep reward 有数值，branch tracker 仍可能长期没有 `lead_known`，导致：
  `second_rear_clear_deadline`、`one_sided_rear_stall_time`、`lead_rear_support_drive` 全部被乘成 0。

后续必须执行的判断规则：

1. 不能再声称这三项阶段任务已经改善了后腿上台后半段。
2. 如果继续修 teacher，第一优先级不是继续加权重，而是先修 branch tracker 的触发和诊断，让 `valid_rate`、`rl_first_rate`、`rr_first_rate`、`second_clear_rate` 有非零值。
3. 只有当这些新指标非零，并且 play 视频看到后腿后半段动作变化，才能说“新阶段任务开始生效”。
4. 若继续从 `model_107400.pt` 往后训练，在不修 tracker 的前提下，不能期待这三项新 reward 自己变有效。

## 2026-06-26 branch tracker 修复：先让阶段任务产生非零信号

用户要求“按照你的思路修改”。这次只修导致 107400 无效的关键链路，不改 terrain、不改 reward 权重、不加 yaw 约束。

修改前备份：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-26_branch_tracker_fix_from_2026-06-26_03-40-49_model_107400/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-26_branch_tracker_fix_from_2026-06-26_03-40-49_model_107400/curriculums.py
```

修改文件：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py
```

具体修改：

1. `rear_second_foot_highstep_clearance_bonus()` 中 branch tracker 更新从：

```python
cmd_gate * commit_gate
```

改成：

```python
cmd_gate * gate
```

其中 `gate = max(terrain_gate, commit_gate_scale * commit_gate)`，与该 reward 自身的 highstep 任务门一致。目的：避免旧的 front commit gate 太窄，导致 branch lead 永远抓不到，新三项 reward 全为 0。

2. `_update_highstep_rear_branch_state()` 增加 `active_threshold` 参数，`rear_second_foot_highstep_clearance_bonus()` 默认使用 `branch_active_threshold=0.12`。目的：只微放宽 tracker 激活，不改 reward 权重。

3. `highstep_rear_branch_metrics()` 增加两个诊断：

```text
Curriculum/highstep_rear_branch_metrics/active_rate
Curriculum/highstep_rear_branch_metrics/lead_capture_rate
```

用途：

- `active_rate > 0` 但 `valid_rate = 0`：说明任务门进来了，但 lead 阈值/左右分支判断仍抓不到。
- `active_rate = 0`：说明 highstep tracker 仍没进入任务阶段。
- `valid_rate > 0` 但三项 reward 仍为 0：说明下一步要看 `second_clear_rate`、`lead_rear_support_drive` 的下游条件。

验证：

```text
py_compile rewards.py curriculums.py: 通过
git diff --check rewards.py curriculums.py: 通过
```

下一轮短训不能只看 `terrain_levels`。必须先看：

```text
Curriculum/highstep_rear_branch_metrics/active_rate
Curriculum/highstep_rear_branch_metrics/valid_rate
Curriculum/highstep_rear_branch_metrics/lead_capture_rate
Curriculum/highstep_rear_branch_metrics/rl_first_rate
Curriculum/highstep_rear_branch_metrics/rr_first_rate
Curriculum/highstep_rear_branch_metrics/second_clear_rate
Episode_Reward/second_rear_clear_deadline
Episode_Reward/one_sided_rear_stall_time
Episode_Reward/lead_rear_support_drive
```

成功的最低前提：`active_rate`、`valid_rate`、三项新 reward 至少出现非零值。若仍全 0，则这次 tracker 修复也不能算生效，不能继续盲训。

## 2026-06-26 branch diagnostics：让失败位置可判定

用户观察最新短训 `2026-06-26_15-42-45`，指出几个值仍然全是 0。本地 event 证实：

```text
run: arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-26_15-42-45
checkpoint: model_107400.pt -> model_107700.pt 左右
actual command: --max_iterations 5000
Episode_Reward/rear_second_foot_highstep_clearance tail100: 0.00243
Curriculum/highstep_rear_branch_metrics/active_rate tail100: 0
Curriculum/highstep_rear_branch_metrics/valid_rate tail100: 0
Curriculum/highstep_rear_branch_metrics/lead_capture_rate tail100: 0
Episode_Reward/second_rear_clear_deadline tail100: 0
Episode_Reward/one_sided_rear_stall_time tail100: 0
Episode_Reward/lead_rear_support_drive tail100: 0
```

结论：旧 highstep reward 有微弱信号，但 branch tracker 仍没进 active。这次 `cmd_gate * gate` 修复不能算成功；不能继续盲训。

为解决“到底卡在哪里不可判定”的问题，新增 branch 诊断指标。修改前备份：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-26_branch_diagnostics_from_2026-06-26_15-42-45_zero_metrics/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/__backups__/2026-06-26_branch_diagnostics_from_2026-06-26_15-42-45_zero_metrics/curriculums.py
```

修改文件：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py
source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py
```

新增可判定指标：

```text
Curriculum/highstep_rear_branch_metrics/active_step_rate
Curriculum/highstep_rear_branch_metrics/soft_active_step_rate
Curriculum/highstep_rear_branch_metrics/lead_candidate_step_rate
Curriculum/highstep_rear_branch_metrics/active_gate_mean
Curriculum/highstep_rear_branch_metrics/active_gate_max
Curriculum/highstep_rear_branch_metrics/task_gate_mean
Curriculum/highstep_rear_branch_metrics/task_gate_max
Curriculum/highstep_rear_branch_metrics/terrain_gate_mean
Curriculum/highstep_rear_branch_metrics/terrain_gate_max
Curriculum/highstep_rear_branch_metrics/commit_gate_mean
Curriculum/highstep_rear_branch_metrics/commit_gate_max
Curriculum/highstep_rear_branch_metrics/cmd_gate_mean
Curriculum/highstep_rear_branch_metrics/cmd_gate_max
Curriculum/highstep_rear_branch_metrics/first_score_mean
Curriculum/highstep_rear_branch_metrics/first_score_max
Curriculum/highstep_rear_branch_metrics/second_score_mean
Curriculum/highstep_rear_branch_metrics/second_score_max
```

下一轮短训的判读规则：

1. `task_gate_max = 0` 或极低：说明 terrain/commit 任务门没有进入，问题在高台检测或 commit gate。
2. `task_gate_max > 0` 但 `cmd_gate_max = 0`：说明 command 阶段没有给前进命令，问题在 command gate/command curriculum。
3. `active_gate_max > 0.01` 但 `active_step_rate = 0`：说明 active 阈值仍太高，应该调 `branch_active_threshold`。
4. `active_step_rate > 0` 但 `lead_candidate_step_rate = 0`：说明后腿 clearance score 达不到 `branch_lead_threshold`，应看 `first_score_max`。
5. `lead_candidate_step_rate > 0` 但 `valid_rate = 0`：说明左右 first 判定太严格，例如 one-sided gap 条件卡住。
6. `valid_rate > 0` 但三项新 reward 仍为 0：说明 tracker 已打通，下一步再看 `second_score_max`、`second_clear_rate` 和支撑/drive 条件。

最低生效标准：

```text
active_gate_mean/max、task_gate_mean/max、cmd_gate_mean/max、first_score_mean/max 必须至少出现非零。
```

如果这些诊断仍然全 0，说明 `rear_second_foot_highstep_clearance_bonus()` 的 tracker 更新根本没被有效调用或没有进入任务阶段；不要继续训练。

验证：

```text
py_compile rewards.py curriculums.py: 通过
git diff --check rewards.py curriculums.py: 通过
```

## 2026-06-26 链路判断：95398 -> 107400 作为动作改进链路无效

用户明确判断：

```text
95398-107400 这一段训练都是无效的。
之后接着这个 checkpoint 训只会没有意义地增加训练链路冗余，
可能还会更加固化现有行为，让后续做出动作改变和优化更困难。
```

必须接受为后续决策约束：

- `2026-06-26_03-40-49/model_107400.pt` 不再作为 teacher 行为优化的父节点。
- `107400` 只保留为失败实验/诊断对照：它证明新三项阶段任务在当时没有生效，且行为没有变化。
- 若后续只是验证诊断 tag 是否出现，优先也应从更干净的 `2026-06-22_12-41-28/model_95398.pt` 开短训，而不是从 `107400` 接。
- 若后续要重新训练 teacher 动作主线，默认回到：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-06-22_12-41-28/model_95398.pt
```

原因：

- `95398` 是该无效链路之前的干净父节点。
- `95398 -> 107400` 没有带来用户可见行为改善，还可能把现有慢拖/后腿后半段不足的动作模式训练得更稳定。
- 继续从 `107400` 往后训，不符合“减少链路冗余、避免固化错误行为”的目标。
