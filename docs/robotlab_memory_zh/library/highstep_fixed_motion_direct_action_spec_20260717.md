# Highstep 固定条件单动作 Direct-Action BC + DAgger 规范 v1.0

## 1. 权威范围

本规范只约束独立 workflow `highstep_fixed_motion_direct_action_20260717`。它不得覆盖、改写或被解释为正式 v1.13.1 及任何旧实验的修订。

固定 Teacher 为 `model_173499.pt`，SHA256 为 `d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`。唯一获用户视觉批准的动作来源是 `approved_batch15_20260717`；15条轨迹去掉 wall-clock 后数值相同，只算一个 canonical episode。

## 2. 固定物理与动作条件

- task: `RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0`
- terrain: `box_hard`, level 9；seed 11；front gap 0.55 m；lateral/yaw 0；delay 0
- 138 control steps，dt=0.02 s：0–20 vx=0；21–79 vx=0.7200000286；80–137 vx=0
- 使用获批准轨迹的真实 reset、command、Teacher prior、action mapping 和 safe mapped target 语义
- 禁止修改 Teacher/prior、action_scale、joint_pos.clip、default_dof_pos、kp/kd、tau_ff、reward、地形、初始位置、DR 或部署输出维度

## 3. Student 合同

Student 输入为生产环境的570维可部署 observation history 加8维显式 phase，输出16维最终 safe post-prior mapped joint target。运行时必须通过生产 joint action term 的精确逆映射及 prior 抵消，使 `processed_actions` 与网络目标逐步一致；禁止把物理 mapped target 直接冒充 raw policy action。

禁用 VAE、latent、critic、PPO 和 reward 训练。网络 fresh 初始化，禁止恢复旧 direct-action optimizer。

相位由实时正向 vx 积分：每步增量为 `max(vx,0)/0.7200000286/59`，截断到1。松开时保持，重新前推继续；负向输入不倒退；最终 phase 保持，等待人工切换 fixed stand。8维为 global phase、六段 one-hot 和段内 phase。六段边界固定为 `[0,.15,.35,.55,.70,.85,1]`。正式训练和门禁只使用标准 full-push 时序。

## 4. 零训练 safe oracle

在候选实际 pre-step 状态查询 Teacher raw action，经生产 action term 得到最终 post-prior mapped target，再经同一生产通道逆映射执行该 safe target。逐步要求执行后的 `processed_actions` 与 safe target 最大误差不超过 1e-6，且无 NaN、越限或非法输出。oracle 必须完成整机上台和 rear hold；失败即安全终止，禁止训练和禁止自动切换 raw 越界目标。

oracle 同时直接记录每个 pre-step 的真实570维生产 observation、8维 phase 和16维 safe mapped target，形成唯一初始 canonical episode。禁止从15条数值重复轨迹制造15份训练信息或虚假验证集。

## 5. BC 与 DAgger

初始BC仅使用canonical episode。按完整episode去重；只有一个唯一episode时不伪造holdout，行为rollout是唯一门禁。

若初始BC未通过，最多执行3轮Student-driven DAgger。Student驱动物理；Teacher只在Student到达的同一pre-step状态提供 safe post-prior mapped 标签。每轮独立数据、checkpoint、manifest和W&B run；禁止恢复旧分支optimizer或加入其它变量。

## 6. 固定门禁

每阶段15场都必须形成有效行为结果；跌倒是行为失败，不是基础设施invalid。要求：

- valid=15/15
- full_climb>=12/15
- rear_hold>=12/15
- 15/15无NaN、非法输出、target越限或暴力关节动作

12–14/15立即停止新增训练并进入用户视觉复核；15/15优选但不是硬门禁；<12/15进入下一轮；第3轮仍<12/15则停止。

每阶段必须使用独立在线W&B run，远端config/history/summary及checkpoint SHA核验完成前禁止下一阶段。基础设施故障可以补损坏场次，不得改判行为失败。

## 7. 交付边界

通过门禁后生成15条独立Student视频及编号合集，与Teacher逐阶段对照，状态只能为 `student_fixed_motion_candidate_pending_user_visual_review`。未经用户视觉批准，禁止MuJoCo部署结论和真机部署。
