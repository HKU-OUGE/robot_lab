# 失败模式与经验教训

## Terrain 曲线会骗人

不要只优化 `terrain_levels`。

已观察到：

- 有些 run terrain 曲线更好，但 play 姿态很差；
- 机器人可能统计上在推进，但平地步态已经趴下、扭曲或乱抬腿。

规则：

```text
重要训练判断必须同时看曲线和 play/video。
```

## 不要频繁乱改 Terrain Curriculum

频繁改 terrain level 判断逻辑会让对比失效。只有 reset 分布或任务定义明显错误时，才改 terrain/curriculum。

Highstep 中有用的地形调整包括：

- 增加 inverse/低处接近高处的情况；
- 让机器人真正从低处爬上高台；
- 避免大多数情况只是从高处往低处走。

但不要为了让曲线好看反复重写 terrain level 评分。

## 训练中录视频容易卡死

`--video` 或频繁 Isaac/Replicator 录制多次造成系统卡死。优先：

- 长训不录视频；
- 单独 `play.py` 录视频；
- 只做短诊断视频。

## Highstep Reward 方向

有效 highstep 姿态应包括：

- 前腿伸到平台上；
- 后腿积极推进；
- 后侧 box joint 在推进阶段伸长；
- 机体越过障碍。

只奖励前腿搭上去不够，会形成“前腿在上面，身体卡住，后腿带不上来”的失败动作。

## Hip 外展不是 Highstep 目标

Sidestep 中 hip 外展用于防侧翻。Highstep 需要前向上台自由度。hip 外展和足端宽度约束会损害 highstep 能力。

规则：

```text
除非明确做侧向台阶混合任务，否则不要把 sidestep 的横向稳定目标搬到 highstep。
```

## Student 不是修补不稳定 Teacher 的工具

不要用 student 去修一个还没稳定的 teacher。teacher 不稳定，student 通常更弱。

No-prior student 注意：

- 先保 locomotion；
- 不要为了学 action prior 把原始步态毁掉；
- 不能随便打开 PPO 或改变 actor 可训练范围，必须明确理由。

## Student 的 OOD command 会比 Teacher 先崩

2026-06-18 highstep student 已观察到：

- teacher 虽然没显式训练负 `lin_vel_x`，但键盘后退时通常还能勉强稳住；
- student `model_69998.pt` 在平地后退时更容易后仰并触发 `bad_orientation`；
- 全局 WandB 平均值不容易暴露这个问题，因为键盘后退是很窄的训练外场景。

经验：

```text
如果 play/部署会使用训练 command 范围外的指令，student 蒸馏必须覆盖这些指令；不能指望 student 自动继承 teacher 的 OOD 泛化。
```

当前具体案例：

- 原 highstep command 范围：`lin_vel_x=(0.20, 0.85)`；
- 最先尝试 student-only 增强：`lin_vel_x=(-0.30, 0.85)`；
- 后来 teacher 侧也改为 `lin_vel_x=(-0.30, 0.85)`；
- 配合最终 command multiplier `0.75`，大约覆盖到 `-0.225 m/s` 的轻微后退。

不要指望继续同一分布的 student 训练能自动修复后退稳定性。先补 command 分布；如果 student 仍无法继承稳定 fallback，再回头改 teacher。

2026-06-18/19 后续经验：

```text
student-first 修复不够。如果 teacher 的好表现只是脆弱的 OOD fallback，盲走 student 不一定能继承。部署会用到的重要 command 必须先进 teacher 训练分布，再重新蒸馏。
```

当前 highstep command 已经在 teacher 侧覆盖：

```text
lin_vel_x=(-0.30, 0.85)
```

## Box Joint 默认位是独立目标

后退稳定修复让机器人后退更安全，但也导致平地/正常运动时 box joint 位置异常。这不可接受：box joint 应该只在高台阶段明显参与，平地静止和普通行走时不应乱伸缩。

当前修复函数：

```text
highstep_box_default_position_penalty()
```

逻辑：

- 计算 box joint 相对默认位置的偏差；
- 用前/后足几何位置和地形高度构造 highstep gate；
- 再乘以前进 command gate；
- 非主动上高台时使用强 `hold_scale`，主动前进上高台时使用低 `highstep_scale`。

经验：

```text
不能靠让 box joint 到处乱漂来解决爬高。box joint 必须有任务门控：平地/非前进保持默认，前进高台释放。
```

## Resume 门控规则

fresh highstep 训练可以保留 command terrain gate，避免太早放开完整 command 范围。但从成熟 checkpoint resume/refine 时，继续卡 gate 会妨碍精修。

当前 `scripts/rsl_rl/base/train.py` 行为：

```text
if highstep teacher and --resume:
    command_levels.params["terrain_gate_level"] = 0.0
```

不要忘记这个区别：

- fresh run：保留门控；
- mature resume/refine：放宽门控。

## Terrain Levels 不是高台高度

用户之前痛苦点：play 里机器人已经能爬 0.35 m 高台，但 `terrain_levels` 仍在 2.7-2.8 左右。正确解释是：

```text
terrain_levels = 所有训练 env 经过 curriculum 升降级后的 terrain row 平均值
```

它不是：

```text
机器人能爬的最大高台高度
```

原因：

- highstep 训练混合了 box、坑、正反楼梯、坡、粗糙地形、静止、后退、侧向等场景；
- box 高台只占训练分布的一部分；
- play 可以主动选择高台，训练曲线是所有场景平均；
- bodyflat 5.x 曲线来自不同任务、地形混合、command 范围和 curriculum 函数。

规则：

```text
terrain_levels 只能作为训练分布/进度信号，不能单独证明上高台能力。判断 highstep 能力要看指定高台高度的 play/eval 成功率、稳定性和 box joint 行为。
```

## 平均化后腿奖励会掩盖分支崩坏

2026-06-23 highstep teacher 精修中观察到：

- `model_97300` 仍能上高台，可作为对照 checkpoint，但重新核对继承关系后，不应作为新奖励修复的主父节点。
- `model_97300` 是从 `2026-06-22_12-41-28/model_95398.pt` 在旧奖励组合下继续约 1900 update 得到的。
- 如果新修复的原因正是“旧奖励组合可能把策略引到坏分支”，除非视频证据压倒性证明更晚 checkpoint 更好，否则优先选择这段旧训练之前的干净 checkpoint。
- `model_105397` 并非完全不会上；补充视频确认它仍有右后腿先登台的成功分支，但主导行为太容易变成更弱的左后腿先登台分支。
- 左后腿先登台时，另一条后腿容易卡在台阶边缘下方，机器人会持续蹬腿但无法完成上台。
- 成功上台常常依赖已经登上高台的那条后腿作为支撑/驱动腿，把身体顶起来，再把第二条后腿带上去。

曲线陷阱：

```text
rear clearance / rear drive / box push 奖励上升，不代表真实上台动作更可靠。
```

原因：

```text
如果后腿 clearance 部分奖励来自 max/单脚成功，而后腿卡台惩罚又对两条后腿取平均，那么“一条后腿上去了、另一条后腿还卡住”的失败会被奖励函数稀释。
```

规则：

```text
Highstep 必须分开看 RL-first 和 RR-first。
卡台/清台项要使用最差后腿或第二后腿逻辑。
选择 checkpoint 前要确认它是否已经包含当前正在修复的旧奖励逻辑训练段。
不要因为后期 checkpoint 标量奖励略高，就从视频已经显示分支崩坏的 checkpoint 继续。
```

## 备份纪律

这个仓库里有很多实验性脏改。不要 revert 用户改动。不要无备份覆盖。

改代码前：

```text
1. 找到相关最新 log 后缀。
2. 复制原文件为 *_<suffix>.py。
3. 只做必要修改。
4. 记录建议从哪个 run/checkpoint 接着训。
```
