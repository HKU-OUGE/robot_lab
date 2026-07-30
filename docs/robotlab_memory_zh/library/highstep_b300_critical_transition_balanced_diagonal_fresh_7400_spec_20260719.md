# Highstep B300 critical-transition balanced diagonal fresh 7400 规范

- 版本：v1.0
- 日期：2026-07-19
- 状态：user_approved_pending_atomic_rebinding
- workflow：`highstep_b300_critical_transition_balanced_diagonal_fresh_7400_20260719`

## 1. 目标、起点与历史保护

从固定 B300 Teacher weights-only fresh 初始化 Student，使用 fresh optimizer、独立 frozen Teacher 和 relative update 0，执行一次连续单进程 7400 effective updates。当前 E1600 diagonal 路线、00169/00170、正式 v1.13.1 及全部历史 checkpoint/optimizer/W&B/state/handoff 均为只读历史证据，禁止覆盖、删除或作为本轮恢复源。

固定 Teacher：

`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt`

SHA256：`d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`

## 2. 唯一批准的组合训练语义 A+B

### A. 关键转换窗口均衡采样

- 使用 rollout-only 四维训练标签：`front_lift/front_support/first_rear/second_rear`；标签来自当前真实地形扫描、足端状态和接触张量。
- 标签不得进入 570 维 policy/estimator 输入、162 维 critic 输入、actor、TorchScript 或部署端。
- FR 转换为 front_lift 进入与 front_support 建立；RL 转换为 first_rear 进入与向 second_rear 转换。
- 每个转换保留同一 env、同一 episode 内事件前后各 10 个 policy control steps。
- 每个训练 minibatch 固定由 25% FR 窗口、25% RL 窗口、50% 原始未修改 rollout 分布组成。
- 可在 minibatch 内有放回重采样；禁止把重复物理帧登记为新增独立数据。
- history 不得跨 episode；某类窗口完全缺失时 fail-closed，不得伪造或改比例。
- 禁止导入旧 Student、旧 DAgger 或旧 canonical 轨迹。

### B. FR/RL 阶段专用对角模仿加权

冻结关节顺序：

`[FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf, FL_box, FR_box, RL_box, RR_box]`

- FR：`[1,5,9]`
- RL：`[2,6,10]`
- 基础 nonbox 项保持 `2.0 * mean(error²[:,0:12])`。
- `front_diag_loss = 0.5 * mean(error²[:,[1,5,9]])`
- `rear_diag_loss = 0.5 * mean(error²[:,[2,6,10]])`
- 新增项为：

```python
per_sample_teacher_action_loss += (
    2.0 * front_phase_gate * front_diag_loss
    + 2.0 * rear_phase_gate * rear_diag_loss
)
```

rear 项必须为正向增加；负号只会降低 RL 监督，与用户固定的“rear gate 增加 RL 且达到 3 倍”合同冲突，禁止实现为负 loss。

- front gate 只覆盖 front_lift/front_support，且只增加 FR 三维。
- rear gate 只覆盖 first_rear/second_rear，且只增加 RL 三维。
- gate 外新增 loss/梯度严格为 0。
- 0.5 归一化不可删除：门控关节有效逐元素总权重必须恰为普通 nonbox 的 3 倍，不得变为 5 倍。
- box 四维、所有 target 及旧的六维统一 diagonal loss 均不改变；旧统一 diagonal scale 在本路线为 0。

## 3. 完全冻结的其余合同

- 冻结 0707 Student env/config/runtime；
- 570 维输入、64 维 latent、16 维输出；
- Teacher pre-prior 主 action target；
- warmup=1400；phase_scale=2.0；rear_box_scale=1.5；warmup prior-box loss=0；
- update 1400 后原最后四个 box rows/bias 适配合同；
- 原 loss 系数、LR、VAE epochs、冻结范围、PPO/critic/reward 关闭合同；
- action_scale、joint_pos.clip、default_dof_pos、Teacher prior、网络、delay=0、部署输入输出合同；
- 一个 optimizer 生命周期、一个 warmup 生命周期、Student/Teacher 独立存储。

## 4. 训练生命周期

- 单进程连续 7400 effective updates，禁止拆段或自动恢复；
- absolute runner/W&B step 从 173499 接续，relative distill update 从 0 开始；
- 每 100 relative updates 保存完整 Student、Teacher binding、optimizer、effective update、runtime/schedule state；
- 训练期间禁止 play、视频、core9、行为 probe、评估、早停和参数修改；
- 训练意外退出时保存现场并等待用户决定，不得自动重启；
- 完成后状态只写 `training_complete_pending_user_manual_play`。

## 5. W&B

- online 模式必须在首个 optimizer update 前取得独立 run id/URL；失败停在 update 0；
- entity=`xinqili551-the-university-of-hong-kong`，project=`isaaclab`；
- group 与新 workflow 一致；
- config 绑定本 spec/preregistration、Teacher、A+B、冻结范围和训练预算；
- 禁止写入 Teacher、E1600 或任何旧 run。

## 6. 唯一允许的训练前验证

仅允许 py_compile、一个 CPU 张量合同断言、Student/Teacher 独立及冻结范围检查、1–5 update smoke 和完整 checkpoint 恢复。CPU 断言必须证明索引、gate 隔离、3 倍非 5 倍、25/25/50 采样比例以及 episode 边界不被窗口跨越。禁止 Isaac 行为 smoke、额外 A/B、reward/target/history/网络/DAgger/gains/sim-to-real 修改。

## 7. 部署限制

本轮仅训练；不得自动评估、筛选、MuJoCo、真机部署或自动真机部署。E7400 后等待用户人工 play。
