# Bodyflat 与 Sidestep 记忆

## Bodyflat 目标

任务：

```text
RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0
```

原始目标：

- 机器人通过 `box_joint` 和原有关节配合，在斜坡/楼梯上保持 body 水平；
- 静止和运动时都尽量 body flat；
- 静止时不能有明显低频晃动。

重要 teacher 参考：

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-04_13-18-36/model_49600.pt
```

重要 sidestep/bodyflat 后期 teacher：

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-11_00-12-35/model_45699.pt
```

用户后来确认过可用的 bodyflat/sidestep 候选：

```text
arclab_arcdog_adjustable_leg_bodyflat_vae_Teacher/2026-06-12_21-52-29/model_47400.pt
```

这个 checkpoint 在 play 中带/不带 `--disable_action_prior` 动作都还可以。

## Sidestep 目标

目标场景：

- 机器人一侧腿在台阶/平台上；
- 另一侧腿在低处平地；
- 防止机器人向低侧侧翻；
- 低侧腿需要更明显外展；
- box joints 需要有可见长度差异。

重要经验：

```text
不要把 sidestep 的 hip 外展和足端宽度目标搬到 highstep。
```

Highstep 需要前向上台自由度，这些横向稳定约束会破坏爬高台能力。

## Action Prior 方向经验

Sidestep 中曾多次把 box joint 方向写反，用户通过 play 图像和 action mapping 抓出来。

永久约定：

```text
box_joint 数值越小，腿越长。
box_joint 数值越大，腿越短。
当前可用范围 0.00 到 0.06。
```

所有 reward、action prior、student distillation target、部署端 clamp 都必须按这个方向检查。

## Student 蒸馏经验

No-prior student 很敏感。

关键教训：

- 如果 teacher 好效果依赖环境侧 action prior，部署时不带 prior 的 student 不能只靠普通 latent 蒸馏就自动获得最终动作。
- 但如果过强地追 post-prior action，可能 box 行为学到了，原始步态却被破坏。
- 是否在 student 阶段打开 PPO 必须非常谨慎，不能把它当成万能修补。

安全原则：

```text
先保 locomotion，再有限度内化 action prior 行为。
```

Highstep 当前应先等 teacher 稳定，再考虑 student。

