# Highstep 固定条件单动作 Raw-Action BC + DAgger 规范 v1.0

## 1. 独立范围与唯一变量

本规范只约束 `highstep_fixed_motion_raw_action_20260717`，不得覆盖正式 v1.13.1 或任何旧结果。

| 字段 | 已封存mapped-target路线 | 本路线 | 改变 | 证据 |
|---|---|---|---|---|
| 监督输出 | Teacher final post-prior mapped target | Teacher原始16维policy_raw | 是，唯一训练语义变化 | mapped-target safe-oracle失败；用户2026-07-17 STEER |
| Teacher/环境/输入/phase/网络维度/门禁 | 固定 | 完全相同 | 否 | 本规范与preregistration |

固定Teacher为 `model_173499.pt`，SHA256 `d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`。批准参考根目录为 `approved_batch15_20260717`。15条数值相同轨迹只计一个canonical episode。

## 2. 动作与输入合同

Student输入为生产570维observation history加8维显式phase，输出16维raw policy action。raw输出必须原样送入普通production action term，再经过原有action_scale、default_dof_pos、Teacher action prior、joint_pos.clip及物理安全限幅。禁止把raw输出作为关节角或绕过production action term。

实时phase由正向vx积分：每步 `max(vx,0)/0.7200000286/59`，截断到1；松开保持，重新前推继续，最终保持等待人工fixed stand。8维为global phase、六段one-hot和段内phase，边界固定 `[0,.15,.35,.55,.70,.85,1]`。

禁止修改Teacher、prior、reward、环境、DR、570维部署观测、输入输出维度、kp/kd、tau_ff、action_scale、default_dof_pos或joint_pos.clip。fresh初始化Student和optimizer，禁止恢复任何旧direct-action/mapped-target optimizer。

## 3. Canonical数据与raw passthrough smoke

批准轨迹的 `policy_raw.*` 是唯一初始target。训练前必须证明全部15条中policy_raw与action_manager_raw逐帧完全一致。重复轨迹不制造独立样本或虚假holdout。

只运行一次最小raw passthrough smoke：完全复用批准预览的reset、box_hard level9、seed11、gap0.55、delay0和21/59/58命令时序。Teacher policy_raw经过新Student入口后必须逐元素不变，并经正常production prior/scale/clip执行，完成full climb及rear hold。失败只定位入口接线并安全停止。

smoke同时直接采集每个pre-step的生产570维obs、8维phase和Teacher policy_raw，形成唯一canonical BC episode。Student运行时obs action history必须由其真实raw输出自然产生，禁止合成替换。

## 4. BC、DAgger与W&B

初始canonical BC后执行15场Student自主行为门禁。若不足，最多3轮Student-driven DAgger：Student驱动物理，Teacher只在Student相同pre-step状态提供policy_raw标签。每阶段独立数据、checkpoint、manifest、fresh阶段optimizer和在线W&B run；远端config/history/summary/checkpoint SHA核验完成前禁止进入下一阶段。

固定门禁：valid=15/15、full_climb>=12/15、rear_hold>=12/15、15/15无NaN/非法输出/暴力关节动作。跌倒是行为失败。达到12/15立即停止新增训练并生成15条独立Student视频及合集；状态只能为 `student_fixed_motion_candidate_pending_user_visual_review`。第3轮仍不足则停止，不创建新分支。

未经用户视觉批准，禁止MuJoCo部署结论或真机部署。
