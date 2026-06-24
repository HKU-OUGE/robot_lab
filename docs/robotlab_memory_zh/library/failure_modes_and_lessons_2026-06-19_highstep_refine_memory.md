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
- student-only 增强：`lin_vel_x=(-0.30, 0.85)`；
- 配合最终 command multiplier `0.75`，大约覆盖到 `-0.225 m/s` 的轻微后退。

不要指望继续同一分布的 student 训练能自动修复后退稳定性。先补 command 分布；如果 student 仍无法继承稳定 fallback，再回头改 teacher。

## 备份纪律

这个仓库里有很多实验性脏改。不要 revert 用户改动。不要无备份覆盖。

改代码前：

```text
1. 找到相关最新 log 后缀。
2. 复制原文件为 *_<suffix>.py。
3. 只做必要修改。
4. 记录建议从哪个 run/checkpoint 接着训。
```
