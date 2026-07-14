# Highstep 历史分析与回答摘录（2026-06-25 至 2026-07-03）

> 从源码目录迁入记忆库。以下内容保留当时的判断、证据和命令，全部属于历史分析；其中“当前”“建议启动”和绝对路径不得覆盖最新正式 spec。

你这句批评成立。我不能把“teacher 不是完美 teacher”说得像轻描淡写，事实是：这两周继续沿着 reward 权重微调，没有把你最关心的真机问题调出来。

严格按约束，我现在能 defend 的判断是：

**结论**
如果继续改 teacher，我不会再建议“继续加后腿 clearance / under-step penalty / box push 权重”这种小修小补。
这条线已经被 `model_124598` 的视频和曲线证明收益不足。

还有价值的 teacher 思路只有一个主方向：

把“后腿上台后半段”从普通 clearance 奖励，改成明确的阶段任务：

1. 前腿已搭台；
2. 任意一条后腿先上台；
3. 已上台的后腿必须支撑/发力抬身体；
4. 另一条后腿必须在有限时间内清台；
5. 长时间单后腿挂边/拖边要直接惩罚。

现在代码没有真正把这个过程约束清楚。

**依据**
视频 `model_124598`：没有明显翻倒，能上台，但后半段仍然慢、拖、靠磨过去，和你说的“后腿后半段力量感不如之前”一致。

wandb 尾段数据：
`bad_orientation` 尾段约 `0.0011`，稳定性没崩；
`rear_second_foot_highstep_clearance` 仍然很低，尾段约 `0.0034`；
`rear_feet_under_step_after_commit` 变得更负，尾段约 `-0.0227`，说明“台下卡/拖”的惩罚没有把行为真正纠正回来；
`one_sided_stall_ratio` 尾段约 `0.108`，单侧卡住仍存在。

代码依据：
[rear_second_foot_highstep_clearance_bonus](/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py:1878) 是即时奖励，不是“第一条后腿上去后，第二条后腿必须在 N 步内上去”的时序约束。
[highstep_rear_branch_metrics](/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/curriculums.py:283) 只是日志，不直接影响 reward。
[highstep_rear_box_push_bonus](/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py:2553) 奖励后腿 box joint 的平均目标，不区分“哪条后腿已经登台、哪条后腿应该作为支撑腿发力”。

所以核心问题不是“再调一点权重”，而是 teacher 现在没有被明确教会：**一条后腿登台后，要用这条腿撑起身体，把另一条腿带上去。**

**如果继续改 teacher，我只认可这三个方向**
1. 加一个真正参与 reward 的 `one_sided_stall_time_penalty`：
   当前 `one_sided_stall_ratio` 只是诊断。应该在前腿 commit 且一条后腿已清台后，累计单侧卡住时间，超过短窗口就惩罚。

2. 加一个 `second_rear_clear_deadline_bonus`：
   第一条后腿清台后，第二条后腿在 N 步内清台才给额外奖励；越晚奖励越低。这个比单纯提高 `rear_second_foot_highstep_clearance.weight` 更对症。

3. 加一个“已登台后腿支撑/推身体”的奖励：
   不再只看 RL/RR box joint 平均值，而是根据哪条后腿先上台，奖励那条后腿形成支撑并带动 base height / forward progress。这个才对应你观察到的成功案例：已经登台的后腿把身体撑起来，另一条腿才被带上去。

**我不会再建议的方向**
不建议继续加 yaw 约束。你通过手动 yaw 能触发右后腿先上台，只说明姿态会影响后腿分支，不等于应该加 yaw reward。

不建议继续单纯加 `rear_feet_under_step_after_commit`。这次它变更负了，但视频没明显变好。

不建议从 `124598` 继续硬训 teacher。它稳定，但已经把当前不够有效的 reward 又训练了 12000 轮。

**如果真的要继续 teacher**
我会从 `2026-06-25_04-35-54/model_112599.pt` 开新分支，而不是从 `124598` 开。理由是：`124598` 稳定但已经吸收了这次 under-step 压力的无效导向；`112599` 更接近改动前状态，后腿 clearance 指标不差，污染更少。

但我必须明确：这不是保证能调出来。它只是目前唯一比“继续拧权重”更有因果对应关系的 teacher 修改方向。当前如果你已经决定不再硬调 teacher，直接 student 蒸馏是更省时间的选择。




对，加上 `2026-06-19_06-03-11` 后，结论更清楚了：**你最初部署真机那条链路，和 `2026-06-22_12-41-28` 用的是同一套更温和 terrain；`2026-06-23` 之后 terrain 明显变硬了。**

**链路关系**
`2026-06-19_06-03-11` 不是 teacher，是 student：

```text
student: 2026-06-19_06-03-11/model_103398.pt
加载 teacher: 2026-06-19_01-42-04/model_87399.pt
最终导出: policy_student_2026-06-19_06-03-11.pt
```

**Terrain 对比**
`06-19 teacher/student` 和 `06-22 teacher`：

```text
box: 25%, height 0.04-0.35
没有 box_hard
pyramid_stairs: 30%
pyramid_stairs_inv: 15%
pit: 15%
random_rough: 9%
```

`06-23 teacher` 和现在 `06-25 teacher`：

```text
box: 31%, height 0.08-0.35
box_hard: 8%, height 0.30-0.38
pyramid_stairs: 25%
pyramid_stairs_inv: 12%
pit: 12%
random_rough: 7%
```

所以，**之前部署真机那版不是在当前这套 terrain 上训出来的**。它没有 `box_hard 0.30-0.38m`，普通 box 也从 `0.04m` 起步；现在低难度 box 也从 `0.08m` 起步，还额外混入了 30-38cm 的硬高台。

**Terrain Levels 数值**
```text
06-19 teacher:  tail200 terrain_levels = 2.808
06-19 student:  tail200 terrain_levels = 2.920
06-22 teacher:  tail200 terrain_levels = 2.836

06-23 teacher:  tail200 terrain_levels = 2.450
06-25 teacher:  tail200 terrain_levels = 2.505
```

这说明你记忆里“之前 terrain levels 更高”是对的，尤其是 `06-19 student` 和 `06-22 teacher`。主要原因不是 curriculum 函数变了，因为这些 teacher 的 `terrain_levels_vel_highstep` 参数一样；主要是 **terrain 分布变难了**。

**但是为什么真机后腿高度不够**
`06-19` 的 sim 看起来还可以，但它的后腿相关 reward 其实比后来弱很多：

```text
06-19 teacher:
rear_feet_highstep_clearance tail = 0.048
highstep_rear_box_push tail = 0.036
highstep_box_phase_prior tail = 0.102

06-23 teacher:
rear_feet_highstep_clearance tail = 0.099
highstep_rear_box_push tail = 0.107
highstep_box_phase_prior tail = 0.155
```

所以你真机观察“06-19 后腿抬腿高度不够”并不奇怪。那版 sim 流畅，很可能是因为 terrain 更温和、没有 `box_hard`，动作不用真的解决高台后腿清台极限问题。

**这几天为什么没明显提升**
我们后来确实把后腿高度、box push、body lift 这些指标一度推高了，尤其 `06-23`，但又引入了两个问题：

1. `box_hard` 让训练环境更难，terrain levels 数字天然更难涨。
2. 后来为了防卡边，加重了 `rear_feet_under_step_after_commit`，但当前 `06-25` 结果显示它变成更负：`-0.0227`，视频却没有明显变好。这说明这个惩罚没有转化成更流畅的后腿清台，反而削弱了后腿推台/phase prior：

```text
06-23 -> 06-25
rear_feet_highstep_clearance: 0.099 -> 0.051
highstep_rear_box_push:       0.107 -> 0.059
highstep_box_phase_prior:     0.155 -> 0.101
```

**我的修正判断**
如果目标是“回到 06-19 那种流畅性，同时比它多一点后腿清台高度”，那现在这套方向过头了：terrain 变硬 + 惩罚变重 + 后腿推台奖励反而被削弱。

更合理的 teacher 方向不是继续在当前基础上硬堆，而是：

```text
回到 06-19/06-22 那套 terrain 分布
去掉或暂时降低 box_hard
恢复 06-23 那组较强的 rear_box_push / box_phase_prior
不要继续加重 under_step penalty
再单独加“第二后腿限时清台/单侧卡住时间”这种时序约束
```

也就是说：**之前真机部署那版的问题不是整体 gait 烂，而是高台后腿余量不够；我们后面把训练环境和惩罚一起加硬，导致流畅性被破坏，但没有稳定换来后腿清台质量。**



已按照要求提前检索记忆和约束｜HLC-OK

可以，而且我认为**这不是“调不出来”**。你这个直觉大方向是对的：前腿异常高抬和“第一后腿踏台、后腿撑身体”本来不是同一个动作目标，理论上可以拆开处理。现在最痛苦的地方是：我们已经看到后腿方向有希望，但前腿外溢把结果毁了。

但我也要稍微反驳一点：“解决前腿异常完全不会影响其他功能”这个不能直接保证。因为当前后腿奖励依赖 `front commit gate`，而这个 gate 本身看前脚高度差、前脚前伸、pitch。也就是说，策略可能发现“把前腿抬得更夸张”能帮它更早触发后腿阶段奖励。所以如果只是粗暴压前腿，很可能又把后腿希望压没。

正确方向不是撤掉后腿增强，而是加一个**只针对异常前腿外溢的边界条件**：

1. 保留现在已经有效的后腿项
   `rear_first_foot_highstep_preclearance`、`rear_second_foot_highstep_clearance`、`lead_rear_support_drive`、`post_lead_body_drive` 这些不能全撤，否则就回到之前“后腿没希望”的状态。

2. 惩罚“没有真实高台需求时的前腿高抬”
   平地、台前还没 commit、或者只有单侧前腿异常抬高时，惩罚前脚相对机身/地形的过高高度和 FL/FR 不对称。

3. 不惩罚“真实上高台时的前腿搭台”
   当前腿确实已经在高台上、并且高台 terrain gate/commit gate 成立时，这个保护项要自动减弱，否则会破坏上台动作。

4. 把现有 `front_legs_quiet_penalty` 改得更合理
   现在它只在“两只前脚同时高于绝对高度阈值”时惩罚前膝速度，抓不住“左前腿单独异常高抬”。这就是明显漏洞。

所以我的判断是：**有必要继续修，而且这次问题比之前更清晰**。
不是再去乱加后腿 reward，而是给已经出现希望的后腿策略加“护栏”：限制前腿异常外溢，但不压合法上高台前腿动作。

下一步如果你让我改，我会按这个边界来做：
保留后腿增强项，新增/加强前腿单侧高抬保护和台前/平地保护，并加明确 wandb 指标，例如 `front_unilateral_lift_penalty`、`front_lift_asymmetry`、`front_lift_guard_active_rate`。这样训练 300-600 轮就能判断：前腿崩坏有没有被压住，同时后腿相关指标有没有保住。

已按照要求提前检索记忆和约束｜HLC-OK

你抓到的“目标函数不等价于真实行为”是现在最关键的问题。我的错误不是单一某次判断错，而是之前没有先把这个问题制度化解决，就反复调 reward、看曲线、play、再调，这导致很多判断像在追影子。

**我犯的核心错误**
| 错误 | 后果 | 修正方式 |
|---|---|---|
| 把训练曲线当成任务效果 | `terrain_levels`、MSE 好看，但 play 退化 | 必须建立独立 highstep eval，不用训练 reward 直接判成败 |
| 没有固定选 checkpoint 标准 | “挑前期模型”看起来像事后找借口 | 固定评估场景、固定指标、固定权重，自动选 best |
| 没有把视频现象转成指标 | 前腿异常、后腿撑身体只能靠肉眼发现 | 加 `EvalHighstep/*` 指标，训练后自动记录 |
| teacher 和 student 问题混在一起 | 不知道是 teacher 不好，还是 student 磨平动作 | teacher 先独立 eval，student 再对同一套 eval 做退化比 |
| 蒸馏只看平均误差 | `145799` MSE 更低但高台动作更差 | student 必须加高台阶段加权蒸馏/early stopping |

**为什么目标函数不等价**
teacher PPO 实际优化的是：

```text
训练分布下的平均累计 reward
```

student 蒸馏实际优化的是：

```text
盲走观测下，student 动作/latent 和 teacher 的平均差异
```

但你真正要的是：

```text
在固定高台任务里，机器人能完成一串阶段动作：
前腿搭台 -> 第一条后腿抬上台 -> 已上台后腿撑起身体 -> 第二后腿清台 -> 平地步态不能崩
```

这三者不是同一个东西。

比如这次 `145799`：

- `Teacher_Action_MSE` 比 `134100` 更低；
- `Mu_Out_Of_Bounds_Ratio` 更低；
- `terrain_levels` 更高；

但高台动作里：

- `rear_feet_highstep_clearance` 单点更低；
- `rear_legs_drive_bonus` 更低；
- `highstep_rear_box_push` 更低；
- `lead_rear_support_drive` 更低。

这说明 student 更“平均贴近 teacher”，但把你最需要的高台瞬时强动作磨平了。平均 MSE 会偏向占比更大的普通行走样本，关键高台阶段即使行为差很多，在总 loss 里也可能只占很小比例。

**完整解决方案**
第一步，不再直接靠肉眼和训练曲线选模型。建立固定 eval suite：

```text
EvalHighstep/success_rate_30cm
EvalHighstep/success_rate_35cm
EvalHighstep/rear_first_clear_rate
EvalHighstep/lead_rear_support_rate
EvalHighstep/second_rear_clear_rate
EvalHighstep/one_sided_stall_rate
EvalHighstep/flat_left_front_lift_abnormal_rate
EvalHighstep/bad_orientation_rate
EvalHighstep/final_score
```

评估场景固定：

```text
平地前进
30cm 高台正向
35cm 高台正向
35cm 高台 yaw -8/+8 度
出生点离高台近/中/远
每个 checkpoint 跑固定种子
```

第二步，teacher 训练继续保存多个 checkpoint，但不叫“随便挑”。用固定评分选：

```text
硬失败：
flat_left_front_lift_abnormal_rate 超阈值
bad_orientation 明显升高
35cm success_rate 明显低

主评分：
35cm success_rate
rear_first_clear_rate
lead_rear_support_rate
second_rear_clear_rate
one_sided_stall_rate
```

第三步，student 蒸馏不能只靠平均 MSE。要么先用 early stopping，发现高台 eval 下降就停；要么进一步改蒸馏 loss：高台阶段样本、后腿关节、box joint、第一后腿搭台后支撑阶段给更高权重。这样 student 不会为了降低普通动作误差，把高台爆发动作磨掉。

第四步，所有后续判断分离成三层：

```text
teacher 是否足够好：只看 teacher eval
student 是否学到了 teacher：看同一 eval 下 student/teacher 差距
部署是否可用：最后看 Mujoco/真机
```

我的主建议很明确：**先补 eval/选模流程，再决定是否继续改 reward。**
现在继续调 reward 仍然可能有效，但没有独立 eval，就会继续陷入“曲线似乎合理、play 才发现退化”的循环。下一步最该做的是把 highstep 任务的真实目标变成可重复、可比较的评估指标。

已按照要求提前检索记忆和约束｜HLC-OK

你这个怀疑是合理的，而且我要先承认一半：**只加 eval suite，确实不能根本解决目标函数和真实行为不匹配。** 它只能让我们更早、更清楚地发现不匹配，避免继续被 `terrain_levels`、MSE、reward 均值骗。真正解决问题，必须把 eval 里的成功定义反向写回训练目标、训练分布和蒸馏权重里。

但我也要反驳另一半：**这不说明 PPO / RobotLab 这套框架已经完全不匹配上高台动作。** 更准确地说，是当前任务定义太“泛”，而你真正要的是一个很窄、很阶段化的动作技能。框架还能用，但不能继续按普通 velocity locomotion + 混合地形 curriculum 的方式硬磨。

**为什么 eval 能产生正反馈**
它不是单纯“多几个曲线让人看”。正确用法应该是闭环：

```text
固定场景 eval
-> 自动给每个 checkpoint 打分
-> 找出哪个阶段退化
-> 只把退化阶段对应地改进 reward / curriculum / distill loss
-> 下一轮训练必须在同一 eval 上证明改善
```

也就是说，eval 的价值不是“继续手动分析”，而是把以前模糊的 play 现象变成硬判据：

```text
第一条后腿没搭上：失败原因 A
第一条后腿搭上但身体没撑起：失败原因 B
第二后腿卡边：失败原因 C
平地左前腿异常：失败原因 D
```

这样后续修改不能再用“terrain 高了”“MSE 低了”来糊弄过去。它必须在 A/B/C/D 中至少一个明确改善，同时其它关键项不退化。

**真正需要改的是训练目标**
你要的行为不是“平均越过各种地形”，而是：

```text
正向接近高台
前腿搭台
第一条后腿抬高并搭上
已搭台后腿向下/向后发力，把身体撑上去
第二后腿清过边缘
恢复正常步态
```

这应该被写成一个阶段任务，而不是散落成几个普通 dense reward。现在的问题就是：reward 里有后腿 clearance、box push、support drive，但它们还没有完全组成“必须按顺序完成”的任务链。结果 policy 可以学到局部替代动作，比如前腿异常高抬、单侧挂边、动作被蒸馏磨平。

我建议的根本方案是三层：

1. **收窄训练分布**
   如果你的真实目标就是正向或小 yaw 角爬 30-35cm 高台，不要再让训练长期承担任意方向、任意起点、复杂混合地形的泛化压力。先训专门 highstep skill，这是合理的，不是作弊。

2. **把 reward 改成阶段状态机**
   用阶段 gate 明确区分：
   `approach -> front_commit -> rear_first_clear -> lead_rear_support -> second_rear_clear -> recover`。
   每个阶段只奖励当前该做的事，不让前腿异常姿态、单后腿挂边、普通 clearance 在错误阶段拿到高分。

3. **蒸馏改成关键阶段加权**
   student 不能再只最小化平均 `Teacher_Action_MSE`。高台阶段样本、后腿关节、box joint、lead rear support 阶段必须加权，否则后期训练会继续把关键动作磨平。`134100` 比 `145799` 好就是这个问题的证据。

所以我的判断是：

```text
框架不是彻底不匹配；
当前任务定义和目标函数不匹配。
```

如果继续按旧方式只调几个 reward 权重，我不 defend，那确实可能继续浪费时间。
我 defend 的路线是：**先用 eval suite 固化真实目标，再把真实目标写回阶段 reward、训练分布和 student 加权蒸馏。**

这才是能产生正反馈的方案：每次修改都必须让固定 eval 的某个失败阶段变好，而不是让我们继续在曲线和肉眼之间来回猜。










已按照要求提前检索记忆和约束｜HLC-OK

我检查完了。另一个对话的修改里，我能 defense 两点：`support_gate_floor=0.08` 暂时合理，因为它只让 support 出现一点苗头后有稀疏梯度；`track_ang_vel_z_exp.weight=1.55` 也可以先保留，因为手动上台确实需要 yaw 微调。

我不能 defense 的是 `support_bottleneck_warmup_min_gate=0.35`。这会让 `total / Episode_Reward/highstep_action_score` 在 support 不达标时拿假分，正好复现 `2026-07-03_02-53-22` 那种“总 reward 看着还行，但 post-lead support 完全没学出来”的失败。`2026-07-01_18-19-06` 链路证明流畅上台能训出来，所以这里不能继续用 warmed total 掩盖支撑阶段失败。

我已修改：

- [highstep_env_cfg.py](/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py:1860)
  - 三处 `support_bottleneck_warmup_min_gate` 全部改回 `0.0`
  - `support_gate_floor` 保持 `0.08`
  - `track_ang_vel_z_exp.weight` 保持 `1.55`
  - `lin_vel_x` 从 `(-0.28, 0.72)` 收到 `(-0.18, 0.72)`，保留后退能力，但减少短训里不触发 highstep reward 的样本比例
- 更新了记忆：
  - [failure_modes_and_lessons.md](/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/failure_modes_and_lessons.md)
  - [highstep_live_context_handoff.md](/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_live_context_handoff.md)

备份已做：

```text
source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__backups__/2026-07-03_review_other_thread_bottleneck_honesty_from_2026-07-03_02-53-22/
docs/robotlab_memory_zh/library/__backups__/2026-07-03_review_other_thread_bottleneck_honesty_from_2026-07-03_02-53-22/
```

验证结果：

```text
py_compile: highstep_env_cfg.py / rewards.py / curriculums.py / train.py 通过
git diff --check: 通过
当前没有 train.py 训练进程，也没有 watchdog 进程
源码中没有残留 support_bottleneck_warmup_min_gate=0.35
```

建议你手动短训从已经证明过强上台能力的 teacher 起点开始，不要再把 bodyflat migration 当主验证路线：

```bash
cd /home/lxq/Softwares/robot_lab
/home/lxq/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless \
  --resume \
  --highstep_resume_mode refine \
  --checkpoint /home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/2026-07-01_06-52-29/model_141000.pt \
  --max_iterations 800
```

这次判断训练是否有效，重点看 `support_score`、`post_lead_body_drive`、`support_floor_violation_rate`、`hard_total` 和 `score_drop_from_best`。`total` 现在不应该再靠 warmup 虚高。
