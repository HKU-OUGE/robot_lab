# RobotLab 仓库使用规范

## 工作空间

仓库：

```text
/home/lxq/Softwares/robot_lab
```

部署仓库：

```text
/home/lxq/colcon_ws/src/quadruped_control_ros2
```

Highstep 常规训练命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/train.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --logger wandb \
  --headless
```

Highstep play 命令：

```bash
cd /home/lxq/Softwares/robot_lab

python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --checkpoint /absolute/path/to/model.pt
```

Bodyflat/sidestep teacher 不带 action prior 的 play：

```bash
python scripts/rsl_rl/base/play.py \
  --task RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0 \
  --num_envs 1 \
  --real-time \
  --keyboard \
  --disable_action_prior \
  --checkpoint /absolute/path/to/model.pt
```

## 备份规则

改代码前必须备份所有要改的文件。

备份后缀优先使用：

```text
_<相关上一条或当前训练log文件夹名>
```

例子：

```text
highstep_env_cfg_2026-06-17_03-05-16.py
rewards_2026-06-17_03-05-16.py
rough_2026-06-17_03-05-16.py
```

不要在没有备份的情况下大改 reward、terrain 或 curriculum。

## 长训规则

长训尽量不要带 `--video`。训练中 Isaac/Replicator 录视频多次造成系统卡死或 VSCode/Isaac 一起崩。

推荐：

- 长训不录视频；
- 单独用 `play.py` 看效果；
- 只在短诊断 run 中录视频。

如果训练在视频后崩溃：

- 查最新完整 checkpoint；
- 查 event 文件更新时间；
- 从最新完整 checkpoint 恢复，不要用不存在或不完整的 checkpoint。

## NVIDIA/GPU 注意

用户真实终端中 `nvidia-smi` 正常。助手沙箱可能看不到 `/dev/nvidia*`，所以助手工具里 `nvidia-smi` 失败不能说明真实驱动坏了。

判断 GPU 状态时，以用户真实终端的 `nvidia-smi` 为准。

## Hydra Override 注意

viewer 覆盖曾报错。直接写 `viewer.*` 不一定对，可能需要 `env.viewer.*`，但有些字段类型是 `NoneType`，会拒绝字符串覆盖。

长训不要使用复杂 viewer override。

## 强制终止卡死训练

如果终端 Ctrl-C 无效：

```bash
pgrep -af 'train.py|play.py|isaac|kit'
kill <pid>
kill -9 <pid>
```

只有真正卡死时才用 `kill -9`。

