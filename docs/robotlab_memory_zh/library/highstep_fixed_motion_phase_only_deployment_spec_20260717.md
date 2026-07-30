# Highstep 固定动作 Phase-Only 部署适配规范 v1.0

## 1. 独立范围

本规范只约束 `highstep_fixed_motion_phase_only_deployment_20260717`。它引用已获用户视觉批准的 DAgger Round 2 Student，不覆盖正式 v1.13.1、raw-action 训练分支或任何历史结果。禁止自动真机部署。

## 2. 单一语义变量

| 字段 | 视觉批准时的 Isaac 合同 | 本部署适配 | 改变 | 证据 |
|---|---|---|---|---|
| highstep prior 门控来源 | 平台高度差、前后足高度差和 command | 摇杆积分 phase 的冻结分段 | 是，唯一变量 | 真机只能使用现有 IMU/关节/摇杆输入；用户 2026-07-17 STEER |
| Student/Teacher/570维观测/8维phase/16维raw输出/action_scale/default_dof_pos/clip/kp/kd/tau_ff | 固定 | 完全相同 | 否 | 用户视觉批准记录与原 raw-action preregistration |

不得增加 raycast、相机、平台高度传感器或任何外部硬件。固定平台、预标定摆位、盲走和人工保护是唯一部署场景。

## 3. 冻结 phase 与 prior

phase 初值为0。每个策略推理步按 `clamp(vx, min=0) / 0.7200000286 / 59` 累加并截断到 `[0,1]`；与1相差不超过 `1e-6` 时精确吸附为1，避免浮点累加使终态无法到达。松开或反向时保持，不回退。8维特征仍为 global phase、六段 one-hot、段内进度，边界 `[0,.15,.35,.55,.70,.85,1]`。

phase-only prior 只修改四个 box joint 的标准映射后物理目标：

- `phase == 0`：零偏置；
- `0 < phase < 0.15`：front `-0.020 m`，rear `+0.002 m`；
- `0.15 <= phase < 1`：front `+0.002 m`，rear `-0.022 m`；
- `phase == 1`：零偏置，等待人工切换 fixed stand；
- 最终 box target 继续截断到 `[0.000, 0.060] m`。

禁止改变16维raw action或绕过标准 `raw clip -> action_scale + default_dof_pos -> phase-only prior -> physical target clip` 通道。

## 4. 验证顺序与门禁

1. 静态、纯函数、边界和Python/C++黄金向量测试。
2. 使用视觉批准 checkpoint 在 Isaac 固定条件下重新运行15场；valid=15/15、full_climb>=12/15、rear_hold>=12/15、safe=15/15才允许进入MuJoCo。
3. Isaac失败时停止部署适配；只允许在本规范真实部署合同下 fresh 训练，不得增加传感器或恢复旧optimizer。
4. Isaac通过后，MuJoCo和真机都必须调用同一个 policy2 C++ phase/prior适配实现；逐帧输出合同和MuJoCo行为通过后才可提出有人保护真机测试。
5. 真机部署、复制模型到机载电脑、修改活动config或电机使能均需用户另行确认。

## 5. 一次性适配目标

policy2适配完成并验证后，phase、prior、观测和控制合同固定在部署代码中。后续同合同模型只替换经过SHA绑定的policy2 TorchScript文件，不重复修改控制逻辑。
