# Highstep v1.12.1 左前足预接触回收：新对话接管记忆

更新时间：2026-07-15（Asia/Hong_Kong）

## 一句话状态

v1.12 E1000 的后腿动作已被用户人工接受；针对视频 `Kazam_screencast_00154.mp4` 中偶发的左前足碰台，独立 v1.12.1 task 和回归测试已经实现，但没有启动新训练。当前应从 `implementation_ready_pending_training_authorization` 继续，不得恢复旧 v1.12 6000-update supervisor。

## 冻结证据

- 用户观看命令：`v112_play_best`
- 接受的后腿行为 checkpoint：
  `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_rear_support_v112_Teacher/2026-07-15_10-52-45_v112_long_attempt1_20260715_105240/model_173200.pt`
- checkpoint SHA256：`962fd3ce3983e4a478092be8f7a636e87ed872b79496c4fa3134191838d9b80d`
- 人工视频：`/home/lxq/Videos/Kazam_screencast_00154.mp4`
- 视频 SHA256：`714290599671e7bd69047628a27d6598af096a09e5659e5c36b5a305761c3412`
- 用户结论：后腿动作符合目标；前腿登台时左前足有概率碰台并短暂停顿，希望预接触后扬/回收更大、首次越过台面的位置更靠后。

## 已实现的唯一语义变量

旧的前腿 reach 总权重 `0.45` 被等总量拆分为：

- `front_legs_reach=0.35`
- `left_front_precontact_retraction=0.10`

新项只在高台、向前命令、pre-commit、左前足尚未越过台面且已经抬起时生效；目标 body-frame x 从 `0.36 m` 回收到 `0.30 m`，高出台面 `0.04 m` 后释放。FR、后腿、action prior、网络、optimizer、其他 reward、`action_scale` 和 `joint_pos.clip` 均未修改。

## 代码位置

- reward：`source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py`
  - `left_front_highstep_precontact_retraction_bonus`
- env/task：`source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py`
  - `ArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121RewardsCfg`
  - `ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121EnvCfg`
- runner：`source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/agents/rsl_rl_ppo_cfg.py`
- registry：`source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/__init__.py`
- tests：`tests/test_highstep_teacher_front_placement_v1121.py`

Task：

`RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-ArcdogAdjustableLeg-v0`

## 已完成验证

- Python compile：通过。
- 新 v1.12.1 + 旧 v1.12 回归：`16 passed`。
- Isaac task registry：已能列出新 task。
- 未执行训练、optimizer.step、checkpoint 写入或 Student 启动。

## 新对话唯一下一步

1. 运行 highstep guard audit，确认 dashboard 指向 v1.12.1 workflow 且无 train/eval/play。
2. 校验 spec、preregistration、state/handoff、E1000 checkpoint 和本文件 SHA。
3. 只做 E1000 → V1121 的 task-only full-resume rebinding，证明 actor/critic/optimizer/iteration/schedule 全部完整且仅环境 reward 变量改变。
4. 固定有限训练预算、保存点和视觉/行为 A/B 门禁后，先做 1–5 update smoke；未经新的正式训练 amendment 不得启动长训。

禁止从 E6000 继续、禁止恢复旧 v1.12 service、禁止同时调整后腿合同或其它 reward。
