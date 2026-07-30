# Highstep B300 diagonal imitation fresh 7400 规范

- 版本：v1.0
- 日期：2026-07-19
- 状态：user_approved_pending_atomic_rebinding
- workflow：`highstep_b300_diagonal_imitation_fresh_7400_20260719`

## 1. 唯一目标与单变量

从固定 B300 Teacher fresh 初始化新的 0707-derived Stage-2 Student，执行一个连续单进程 7400 effective-updates 长训。本轮只改变一个训练语义：

`student_highstep_diagonal_action_loss_scale: 0.0 -> 1.0`

基线为已完成 workflow `highstep_b300_0707_derived_single_run_7400_20260718`，其规范：

- `/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_b300_0707_single_run_7400_spec_20260718.md`
- SHA256：`b73470a6b15a9fb5595b7153c7cb155878c2a2c4c63590b178b7b3f640af5fd3`

旧 E7400 Student、checkpoint、optimizer、W&B、日志、state 和 handoff 全部保留；禁止将其作为本轮初始化或恢复源。

## 2. 固定 Teacher

- checkpoint：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt`
- SHA256：`d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`
- runner iteration：`173499`

Student 必须 weights-only fresh 从该 Teacher 初始化；独立 frozen Teacher；fresh optimizer；relative distill update 从 0 开始。禁止加载任何 Student optimizer。

## 3. 对角模仿项精确定义

仅当现有 `post_prior_gate/highstep phase` 有效时生效。固定索引：

`[1, 2, 5, 6, 9, 10]`

对应 FR/RL hip、thigh、calf。对现有 Teacher pre-prior action squared error：

```python
diagonal_loss = mean(squared_action_error[:, [1, 2, 5, 6, 9, 10]], dim=-1)
per_sample_teacher_action_loss += (
    1.0 * post_prior_gate.detach() * diagonal_loss
)
```

现有基础 nonbox 项保持 `2.0 * mean(error[:, 0:12])`。所以门控阶段六个指定元素的有效逐元素权重为其他 nonbox 元素的 2 倍。禁止修改最后 4 个 box 动作的 target、基础权重、rear-box 权重或 prior-box loss。

## 4. 完全冻结的其余训练合同

以下全部与基线规范和 preregistration 相同：

- 精确 0707 Student env/config/runtime，env SHA256：`f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636`；
- delay=0；
- 570 维 Student history、64 维 latent、634 维 actor 输入、16 维输出；
- actor `[512,256,128]`、VAE `[256,128]`、ELU；
- PPO/critic/reward 学习永久关闭；
- 每 update 4 个 Student/VAE epochs；
- estimator LR `1e-3`、algorithm LR `1e-4`、adaptive schedule；
- velocity/latent/action/prior-box/recon/KL loss 系数分别为 `10/50/20/5/0.5/0.1`；
- warmup=1400；warmup 主 target=Teacher pre-prior 16维；
- phase scale=2.0、rear-box scale=1.5；
- warmup prior-box loss=0；
- update 1400 后只允许最后4个box rows/bias以LR `1e-5`适配，其余actor永久冻结；
- hard latent clamp；
- Teacher prior、Student plain JointPositionAction、observation/action contract；
- action_scale、joint_pos.clip、default_dof_pos、reward、环境、randomization、网络和部署合同。

禁止把 box 极端 action 作为本轮附加训练变量或改 target。

## 5. 连续训练生命周期

- 一个训练进程；
- 一个 optimizer 生命周期；
- 一个 1400-update warmup；
- 连续 7400 effective updates；
- 每 100 relative updates 保存一个完整 checkpoint；
- absolute runner/W&B step 从 `173499` 接续；
- relative distill update 从 `0` 开始；
- Student schedule 使用冻结 0707 runtime，从 relative update 0 开始；
- 禁止拆段、自动恢复、自动重启、提前停止或中途更改参数。

训练意外退出时保存现场并等待用户决定，不得自动从 checkpoint 恢复。

## 6. W&B

- `WANDB_MODE=online`；
- entity：`xinqili551-the-university-of-hong-kong`；
- project：`isaaclab`；
- group：`highstep_b300_diagonal_imitation_fresh_7400_20260719`；
- run name：`highstep_b300_diagonal_imitation_fresh_7400_student_from_173499`；
- 独立 run，禁止写入 Teacher 或基线 Student run；
- 在第一个 optimizer update 前必须取得 run id 和 URL；失败则停止在 update 0；
- 训练开始后仅使用 W&B SDK 原生在线同步与重试，不添加 remote gate。

## 7. 训练前允许的最小验证

只允许：

1. spec、preregistration、Teacher、0707环境证据及代码 SHA；
2. 路径、GPU空闲和启动参数静态展开；
3. `py_compile`；
4. 一个 CPU tensor assertion，证明：
   - gate=0 时所有权重与基线一致；
   - gate=1 时仅索引 `[1,2,5,6,9,10]` 的 nonbox 逐元素权重变为其他 nonbox 的 2 倍；
   - box 四维权重不变；
5. W&B online 初始化。

禁止 Isaac smoke、Teacher rollout、Student 1–5 update smoke、pytest 全量测试及其它诊断。

## 8. 训练中禁止事项

第一个 optimizer update 到第 7400 个 update 原子保存完成期间，禁止：

- 行为评估、probe、core9、directional、15场评估；
- play、视频、MuJoCo、部署、真机；
- loss/latent/action/terrain 门禁或自动早停；
- 新增实验、A/B、参数修改、自定义自动修复或自动重启。

heartbeat 只记录进程存活、absolute step 和 relative update，不读取指标作门禁。

## 9. 完成状态

7400 完成并保存最终 checkpoint 后停止，状态必须写：

`training_complete_pending_user_manual_play`

不得自动评估、播放、续训或修改变量。下一步只由用户手动 play 检查。

禁止自动真机部署。
