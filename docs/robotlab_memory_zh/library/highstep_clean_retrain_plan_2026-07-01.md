# Highstep 从 0 到当前 checkpoint 完整历史链路与重训流程报告

更新时间：2026-07-01，Asia/Hong_Kong。

## 当前结论

- 当前最新保存 checkpoint：`logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_148700.pt`。
- 当前数据/视频综合更推荐的行为候选：`model_142300.pt`，备选 `model_142100.pt`。
- 关键原因：`148000/148700` 的 `terrain_levels` 更高，但 `Support` 类评分明显崩掉；高台动作不如 `142100/142300`。
- 本文区分三件事：从 0 到当前 checkpoint 的真实历史、历史中不应复刻的试错分支、重新从 0 跑一条更干净链路的方案。

## 逐阶段历史链路

| 阶段 | 分组 | run | 输入 | 输出 | max_iterations | 是否当前父链 | 目的 | 结果/教训 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| 0 | Bodyflat 基础 | `arclab_arcdog_adjustable_leg_bodyflat_vae/2026-05-08_08-59-28` | `fresh / resume=false / 0 轮随机初始化` | `model_12400.pt` | 17000 | 是 | 从 0 建立可伸缩腿机器狗的基础 locomotion：平地、普通地形、box joint 基础协同。 | 尾段 terrain≈5.682、command=1.0、time_out≈0.995；这是后续所有 teacher 的最早祖先。 不要从 0 直接 highstep；历史上先 bodyflat 预训练明显更稳。 |
| 1 | Bodyflat 基础 | `bodyflat_vae_Teacher/2026-05-11_09-20-46` | `2026-05-08_08-59-28/model_12400.pt` | `model_25300.pt` | 17000 | 是 | 迁移到 Teacher 实验名并继续稳定基础步态。 | 继续积累稳定步态和 terrain curriculum 能力。 这是命名/runner 迁移阶段，不是 highstep 专项。 |
| 2 | Bodyflat 基础 | `bodyflat_vae_Teacher/2026-05-11_21-33-05` | `2026-05-11_09-20-46/model_25300.pt` | `model_27000.pt` | 17000 | 是 | 短接续，筛选更稳 bodyflat 父模型。 | 作为后续 bodyflat/sidestep 链路中间节点。 历史链路不是一次完整长训，而是多次人工筛选 checkpoint。 |
| 3 | Bodyflat 基础 | `bodyflat_vae_Teacher/2026-05-31_16-58-47` | `2026-05-11_21-33-05/model_27000.pt` | `model_29600.pt` | 17000 | 是 | 延长 bodyflat teacher 稳定训练。 | 给后续 bodyflat/sidestep 和 highstep 共同提供动作基础。 稳定性指标比某个单轮 terrain 数字更重要。 |
| 4 | Bodyflat 基础 | `bodyflat_vae_Teacher/2026-06-03_21-47-54` | `2026-05-31_16-58-47/model_29600.pt` | `model_45500.pt` | 17000 | 是 | 继续 bodyflat 稳定化，形成较成熟的普通运动能力。 | 该 run 最后到 46599，本链路选 45500；用于下一段 bodyflat。 选择 checkpoint 可以基于行为质量，不必机械取最后一个。 |
| 5 | Bodyflat 基础 | `bodyflat_vae_Teacher/2026-06-04_13-18-36` | `2026-06-03_21-47-54/model_45500.pt` | `model_49600.pt` | 17000 | 是 | 形成 highstep 迁移前的 bodyflat 父模型。 | 尾段 terrain≈5.391、command=1.0、time_out≈0.993、mean_reward≈195.91。 这就是 highstep 主链最重要的 bodyflat 父节点。 |
| 6 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-15_23-16-43` | `bodyflat_Teacher/2026-06-04_13-18-36/model_49600.pt` | `model_52200.pt` | 4500 | 是 | 从 bodyflat 切入 Highstep teacher，开始学习前腿搭台、后腿抬高、box joint 上台动作。 | 尾段 terrain≈2.516、command≈0.4675；高台能力刚起步。 highstep terrain_levels 和 bodyflat 5.x 不能直接比较。 |
| 7 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-16_02-44-57` | `2026-06-15_23-16-43/model_52200.pt` | `model_53200.pt` | 4500 | 是 | 短接续，稳定 highstep 初始动作。 | 作为 06-16 后续强化的中间节点。 这一时期主要解决“能不能开始上台”，不是精调后腿。 |
| 8 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-16_04-22-56` | `2026-06-16_02-44-57/model_53200.pt` | `model_57699.pt` | 4500 | 是 | 继续 highstep 迁移。 | 记忆中标记为视觉上开始能上 highstep 的阶段。 视频/play 已经开始比 terrain 数字更关键。 |
| 9 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-16_22-07-48` | `2026-06-16_04-22-56/model_57699.pt` | `model_60400.pt` | 4500 | 是 | 加强接近高台和逆向地形分布。 | 高台接近动作继续增强。 地形分布会显著改变 curriculum 曲线含义。 |
| 10 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-17_00-56-56` | `2026-06-16_22-07-48/model_60400.pt` | `model_61900.pt` | 4500 | 是 | 继续提升 terrain levels，命令等级仍偏保守。 | terrain levels 上升，但 command 仍低。 只看 terrain 不够，command 释放也影响行为。 |
| 11 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-17_03-05-16` | `2026-06-17_00-56-56/model_61900.pt` | `model_63700.pt` | 4500 | 是 | 放宽 command 到约 0.6375，使策略不只会低速上台。 | command 覆盖更接近后续部署需求。 command gate 会影响上台动作和退化判断。 |
| 12 | Highstep 迁移 | `highstep_vae_Teacher/2026-06-17_05-07-05` | `2026-06-17_03-05-16/model_63700.pt` | `model_68199.pt` | 4500 | 是 | 形成第一版较强 highstep teacher。 | 尾段 terrain≈3.274、command≈0.6375；视频中运动能力基本满足初期要求。 后续 student 蒸馏从这里暴露后退 bad_orientation 和真机部署差距。 |
| 13 | 稳定性与部署前修正 | `2026-06-18_05-10-44_teacher_backward_stability_long_from_68199` | `2026-06-17_05-07-05/model_68199.pt` | `model_80198.pt` | 12000 | 是 | 加入/强化后退 command 覆盖和 pitch 稳定，处理 student/teacher 后退容易后仰的问题。 | 后退稳定性改善，但 terrain levels 不一定比 06-17 更高。 真机/部署问题会改变优化重点，不能只追求爬台曲线。 |
| 14 | 稳定性与部署前修正 | `2026-06-18_21-19-11_teacher_box_default_gate_from_80198` | `2026-06-18_05-10-44/model_80198.pt` | `model_82100.pt` | 5000 | 是 | 修复平地/普通运动中 box joint 位置异常偏移；box joint 应主要在上高台阶段起作用。 | box default gate 进入主线。 伸缩腿不能为了高台动作长期偏离默认位，否则部署风险高。 |
| 15 | 稳定性与部署前修正 | `2026-06-18_23-20-22_teacher_box_default_resume_gate_free_from_82100` | `2026-06-18_21-19-11/model_82100.pt` | `model_84400.pt` | 3000 | 是 | resume/refine 时放宽 command terrain gate，避免从已有 checkpoint 继续时重新被早期门控卡住。 | resume gate-free 机制成为后续 train.py 的重要经验。 --resume 时 staged reward 和 command gate 必须单独处理。 |
| 16 | 稳定性与部署前修正 | `2026-06-19_01-42-04` | `2026-06-18_23-20-22/model_84400.pt` | `model_87399.pt` | 3000 | 是 | 形成真机测试前相对稳定的 teacher 来源。 | 尾段 terrain≈2.812、bad_orientation≈0.00073、time_out≈0.999；仿真效果可用。 真机测试暴露电机高频抖动、整机摇晃和后腿抬高不足，说明仿真目标仍不等价于真实任务。 |
| 17 | 稳定性与部署前修正 | `2026-06-22_12-41-28` | `2026-06-19_01-42-04/model_87399.pt` | `model_95398.pt` | 8000 | 是 | 继续 teacher 精修，成为后续多条分支的干净父节点。 | 尾段 terrain≈2.834、command≈0.6375、bad_orientation≈0.00132；被反复作为较干净 anchor。 95398 比 97300 更干净，因为尚未吸收后续旧 reward shaping 的偏置。 |
| 17A | 对照分支 | `2026-06-23_00-52-26` | `2026-06-22_12-41-28/model_95398.pt` | `model_97300.pt / model_105397.pt` | 10000 | 不是当前父链 | 旧 reward mix 下尝试继续增强高台动作。 | 97300 仍能上台，右后腿先登台更顺；105397 左后腿先登台失败倾向增强。 该分支是重要对照，但不作为当前 07-01 链路父节点。 |
| 18 | 通往 124598 的直接基底 | `2026-06-24_00-36-54` | `2026-06-22_12-41-28/model_95398.pt` | `model_108000.pt（后续父节点）；另有 110000/112700 视频对比` | 110000 configured / 实际手动中止 | 是，通往124598 | 从 95398 开启另一条高台强化基底，观察 110000 和 112700 行为差异。 | 用户观察 112700 比 110000 略好但差异不大；左后腿先登台仍更常见。 `max_iterations=110000` 是额外训练上限，历史上这里曾造成理解混乱；实际选用 108000 接下一段。 |
| 19 | 通往 124598 的直接基底 | `2026-06-24_22-59-55` | `2026-06-24_00-36-54/model_108000.pt` | `model_111800.pt` | 4000 | 是，通往124598 | 继续从 108000 调整高台能力。 | 111800 曾用于和 06-17/06-19 链路对比；整体稳定些但后腿后半段力量感仍不如早期部分策略。 不能只追求更稳，后腿后半段支撑动作质量也必须保留。 |
| 20 | 通往 124598 的直接基底 | `2026-06-25_04-35-54` | `2026-06-24_22-59-55/model_111800.pt` | `model_112599.pt` | 800 | 是，通往124598 | 短训微调 111800。 | 作为 06-25 长段的直接父 checkpoint。 短训可用来验证方向，但不能把未收敛结果当最终。 |
| 21 | 通往 124598 的直接基底 | `2026-06-25_05-59-47` | `2026-06-25_04-35-54/model_112599.pt` | `model_124598.pt` | 12000 | 是，通往当前阶段任务重构 | 形成 06-30 阶段任务重构前的重要可用基底。 | 上台能力勉强可用，后续 student 蒸馏也曾基于它；但第一后腿/后半段动作仍不够理想。 124598 不是从 0 的干净点，但它是后续阶段任务重构的实际起点。 |
| 22 | 06-28 试错分支 | `2026-06-28_06-05-11 / 08-23-04 / 09-53-57 / 12-46-05 等` | `多次从 2026-06-25_05-59-47/model_124598.pt 重启` | `多组 125k-127k checkpoint` | 多为4000 | 不是最终直接父链 | 围绕第一后腿搭台、post-lead 支撑、前腿异常高抬进行多轮试错。 | 08-23 除前腿异常高抬外，第一后腿搭台和撑身体很有希望；09-53 前腿护栏压掉后腿能力；support_pose_scale=0 分支 hard fail 后被 supervisor 回滚。 全局前腿护栏和过强姿态约束会牺牲爬台主能力；后续改成阶段任务逻辑。 |
| 23 | 阶段任务重构 | `2026-06-30_01-45-31` | `2026-06-25_05-59-47/model_124598.pt` | `model_133597.pt` | 9000 | 是，07-01主父链 | 把 highstep 从普通 dense reward 调参改为阶段任务：approach -> front_commit -> rear_first_clear -> lead_rear_support -> second_rear_clear -> post_clear_recovery。 | 视频 00112_2026-06-30_01-45-31_model_133597 显示爬高台能力肉眼显著提升。 这是当前最关键的方法论转折：目标函数必须贴近真实动作链。 |
| 24 | 阶段任务重构 | `2026-06-30_18-52-50` | `2026-06-30_01-45-31/model_133597.pt` | `model_136596.pt` | 3000 | 是，07-01主父链 | 加入 post_clear_recovery=0.30、scanner_pretrigger_penalty=-0.08，尝试减弱上台后仍像准备上台和台前预触发。 | 上台动作很好且流畅，但阶段判定异常仍明显：远处会抬前腿，上台后仍保持上台姿态。 不能让姿态恢复/预触发约束压过爬台能力；后续切换为 highstep-only + FSM。 |
| 25 | 阶段任务重构 | `2026-07-01_01-46-33` | `2026-06-30_18-52-50/model_136596.pt` | `model_137500.pt / model_137600.pt` | 4000 | 是，07-01主父链 | 继续 136596，用户 play 后决定：上高台能力优先，平地/上台后普通行走后续用部署端 FSM 切换其它策略。 | 用户确认上台能力有保持，但非高台状态问题仍在。 目标边界改变后，不能再为了平地步态牺牲高台主能力。 |
| 26 | 阶段任务重构 | `2026-07-01_03-28-38` | `2026-07-01_01-46-33/model_137500.pt` | `model_140400.pt` | 4000 | 是，07-01主父链 | 撤掉主爬台 reward 上过强 phase-exit 门控；post_clear_recovery 0.30->0.12，scanner_pretrigger -0.08->-0.02，保留轻量切入/切出姿态兼容。 | 视频 00112_2026-07-01_03-28-38_model_140400：上高台能力可以接受，用户决定可轻量加回 box_hard。 上台能力专用策略路线成立，但仍必须用视频+score 选 checkpoint。 |
| 27 | 轻量 hard refine | `2026-07-01_06-52-29` | `2026-07-01_03-28-38/model_140400.pt` | `最新保存 model_148700.pt；推荐行为窗口 model_142100/model_142300` | 13400 | 是，当前最新 | terrain 从 warm 切到 final：box 0.58 + box_hard 0.04，高度 0.30-0.38m，轻量增加硬高台样本。 | 142100/142300 score 高；148000/148700 terrain 更高但 support 崩。 当前目标函数仍不保证越训越好；hard refine 应短训并选择早期高分 checkpoint，不默认取最后。 |

## 当前 hard refine 评分证据

| checkpoint | HighstepScore | Entry | Support | terrain | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| `model_142100` | 74.6 | 62.2 | 87.3 | 2.24 | early hard refine，肉眼比 148000 流畅 |
| `model_142300` | 82.0 | 82.5 | 89.7 | 2.36 | 数据评分最高，当前推荐候选之一 |
| `model_148000` | 38.0 | 70.9 | 8.9 | 4.27 | terrain 高，但支撑动作崩 |
| `model_148700` | 27.9 | 56.1 | 4.9 | 4.38 | 最新保存，但不等于最佳动作 |

## 新的干净重训方案

1. Phase A：使用 warm terrain 从 0 开始训练主体上台能力，`box=0.62`，禁用 `box_hard`，保留当前阶段任务 reward。建议 `--max_iterations 100000`。
2. Gate A：不要只看最后 checkpoint，优先选 `Entry + Support` 综合最好的 warm checkpoint。
3. Phase B：从 Phase A 最优 checkpoint 切 final terrain，`box=0.58`，`box_hard=0.04`，只额外 refine 约 2500 轮。
4. Gate B：重点比较 start+1600、start+1900、start+2200；不要默认取最后一个。
5. Phase C：只有 play 中第一后腿搭台、已搭台后腿撑身体、第二后腿清台都满足后，再进入 student 蒸馏。

## 关键命令模板

```bash
cd /home/lxq/Softwares/robot_lab

/home/lxq/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --max_iterations 100000
```

```bash
/home/lxq/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --checkpoint <PHASE_A_BEST_CHECKPOINT> \
  --max_iterations 2500
```
