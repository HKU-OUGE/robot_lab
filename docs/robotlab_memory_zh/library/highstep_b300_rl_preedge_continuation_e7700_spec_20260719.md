# Highstep B300 RL Pre-edge Continuation E7700 Spec v1.0

本规范仅授权从受保护的 E5700 完整 checkpoint 及其原 Adam 状态精确恢复，追加 2000 effective updates 到 E7700。旧 A+B 路线及其所有结果完整保留。

## 冻结恢复源

- checkpoint: `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_Student/2026-07-19_06-43-15_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_attempt2_from_173499/model_179198.pt`
- SHA256: `31fe19c7d1ba9872c0b714d594fe6e66c549a52e88fe856ed62206680d37c498`
- `student_distill_update_count=5700`
- 必须完整恢复 estimator 与 box rows/bias 两组 Adam 参数及 state；estimator LR=1e-3，box LR=1e-5。
- 禁止使用 E5700 之后任何未形成完整 checkpoint 的临时 update。

## 唯一续训变化

采样固定为 25% FR 关键转换窗口、25% RL pre-edge 窗口、50% 原始分布。

- FR 关键转换及其窗口保持原路线定义（事件前后各 10 policy steps）。
- RL pre-edge 谓词固定为：`vx>=0.65`、两前腿已支撑、RL 未上台且进入冻结 first-rear pre-clearance 边界。
- RL pre-edge 窗口固定为 `[-30,+10]` policy steps，同 env、同 episode，done 严格隔离。
- 训练标签仅进入 rollout buffer，不得进入 570 维 Student history、critic、actor、TorchScript 或部署输入。

## 完全保持不变

- 固定 B300 Teacher model_173499 及 Teacher prior；
- 570维输入、64维 latent、16维输出、网络；
- warmup=1400，不得重新经历；
- Teacher pre-prior 主 action target，prior-box/post-prior box 监督；
- phase_scale=2.0、rear_box_scale=1.5、全部 loss 系数、VAE epochs；
- estimator 与最后4个box rows/bias自首个续训 update 可训练；actor前12行及其余模块冻结；
- FR=[1,5,9]、RL=[2,6,10] 的阶段专用3倍逐元素权重；不得追加 RL-only loss；
- 0707 Student environment、delay=0、action_scale、joint_pos.clip、default_dof_pos、gains及部署合同。

## 运行与终态

- 单进程追加 2000 effective updates，目标 E7700；每100 updates保存完整 checkpoint。
- W&B 独立 continuation run，首个 optimizer update 前 online 初始化完成。
- 恢复前 fail-closed 校验 checkpoint/SHA、双绑定、5700计数、两组optimizer及state、box LR、schedule/runtime。
- E7700 完成后停止，状态写 `training_complete_pending_user_manual_play_6`。
- 禁止自动15场门禁、自动play、视频、core9、继续追加或自动真机部署；由用户手动连续满推 play 6 次审核。
