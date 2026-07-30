# Highstep 固定条件显式阶段 Direct-Action 实验规范 v0.1

本规范只约束独立实验 `highstep_fixed_condition_phase_direct_action_20260717`。它不修改、不取代正式 v1.13.1，也不改写已经终止的 residual 分支。

## 1. 目标与单一变化

目标仍是约 0.378 m `box_hard level 9`、固定摆位、盲观测条件下连续 15 次完整上台。唯一机制变化为：

```text
Student obs[570] + explicit phase[8]
  -> 直接预测 Teacher 的最终安全 16 维 joint target
```

相对于已终止 residual 分支，取消 `target-reference` 的 ±0.20 rad / ±0.006 m 表示瓶颈。reference 只保留为 296 步零命令 preroll、显式阶段时钟和回退证据；active phase 的控制目标完全来自 direct-action model。

## 2. 冻结条件

- Teacher、Teacher prior、任务、平台、命令、摆位、delay、关节顺序和物理 limits 不变；
- Teacher checkpoint SHA：`d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`；
- 输入仍是候选自身 exact Student obs 570，其中 action history 是候选最终安全 target 的原始 affine 逆映射，再追加冻结的 8 维 phase；
- 输出通过冻结的 physical center/half-range 与 `tanh` 映射，任何时候都不得超出 16 关节物理包络；
- 生产 action prior 必须被精确抵消，最终 `processed_actions` 必须等于 model safe target；
- 不修改 action_scale、joint_pos.clip、default_dof_pos、gains、reward、现有 Student/Teacher 网络或正式 supervisor。

## 3. 冻结数据

训练只使用 residual 分支已只读保存的 4904 个同状态样本：15 条成功 Teacher rollout、reference-only DAgger 有效前缀、round-1 residual DAgger 有效前缀。终止后的 496 个重置帧均已剔除。数据 manifest：

`/home/lxq/Softwares/robot_lab/tmp/highstep_fixed_condition_phase_residual_20260717/dagger_round2_aggregate_retry1/dataset_manifest.json`

SHA256=`bf156fbb712b360a28147e4762b8d94c0f3060ae1e144fa01021e34e09cb720a`。

训练/验证继续按完整 rollout 分离；禁止帧泄漏。target 固定为 `safe_teacher_target_16`，不得使用 raw 越界 target。

## 4. 模型与优化

- MLP：578 -> 512 -> 512 -> 256 -> 16，ELU + final tanh；
- 输入使用 train split 冻结 mean/std，标准化后 clamp 到 [-10, 10]；
- 输出按关节 physical center/half-range 映射；
- loss 在归一化 target 空间计算：stage-balanced Huber + 0.1 相邻帧一阶差分；
- Adam，LR=1e-3，weight decay=0，最多 500 epochs，patience=50；
- 从零初始化；PPO、reward、latent、velocity、recon、KL 全部禁用；
- 独立 W&B run，远端 summary/checkpoint SHA 验证完成前禁止行为评估。

## 5. 门禁和停止

固定 Isaac 15 环境：valid=15/15、full_climb=15/15、rear_hold=15/15。未达到不得进入 MuJoCo 或部署。

初始 direct-action 失败时最多允许两轮同固定条件 DAgger；Teacher 只在候选同一 pre-step 状态提供 safe post-prior target，绝不驱动物理。两轮后仍失败则本 direct-action 分支停止。

达到 Isaac 门禁后才生成视频并交给用户视觉复核；未经用户批准不得真机部署。

