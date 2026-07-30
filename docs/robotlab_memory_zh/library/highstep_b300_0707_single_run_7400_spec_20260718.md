# Highstep B300 Teacher → 0707-derived Student 单次 7400 长训规范

- 版本：v1.0
- 日期：2026-07-18
- 状态：`draft_approved_by_user_but_inactive_until_main_thread_rebinding`
- 用途：供主对话创建独立 workflow/preregistration 并原子迁移 authority。
- 本文件写入本身不启动训练、不修改现有 dashboard/state/service，也不取代当前正式 authority。

## 1. 唯一目标

使用当前已接受的 B300 Teacher，沿用 0707 已实际完成真机测试的 Stage-2 Student 蒸馏配置与 Student 环境，执行一次不被中途检查打断的连续 7400 effective updates 长训。

本轮只回答一个问题：在不加入后来 recovery、robust、STE、clamp-gradient、direct-action、虚拟地形或部署端改造的前提下，0707 蒸馏机制配合当前 B300 Teacher 能否训练出可供训练结束后人工检查的 Student。

本路线正式名称必须包含：

`highstep_b300_0707_derived_single_run_7400`

禁止称为 `0707_exact`，因为用户批准的单次 7400 进程与历史 0707 的两个训练进程并不完全相同。

## 2. 冻结输入与证据

### 2.1 当前 B300 Teacher

- checkpoint：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt`
- SHA256：`d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431`
- Teacher W&B run id：`v1123be30017841497641160009`
- Teacher W&B URL：`https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/v1123be30017841497641160009`
- Teacher runner iteration：`173499`

Teacher checkpoint、Teacher actor、Teacher prior、Teacher action contract均禁止修改或重训。

### 2.2 0707 Student 配置证据

初始 0707 Student：

- run：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-04_06-32-23`
- `params/agent.yaml` SHA256：`c1941ecca30859aa7ad29fa8e2558bef099234beb83ec8a1e63e72549e1deed4`
- `params/env.yaml` SHA256：`f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636`
- runtime diff SHA256：`97c779bf06cb00f759df0c98b40b86409c3735a649979fa89cd251ca9f15defe`

0707 部署 Student 续训：

- run：`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/2026-07-05_00-13-46`
- `params/agent.yaml` SHA256：`bb01ed1c7a8574363e9cea0d9d1dc4a0828d8453a097715ae2e4b49fa51be9ca`
- `params/env.yaml` SHA256：`f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636`
- runtime diff SHA256：`81181c32436cc6c0b4a869efd45b372a718205a6b4b4403e5f56a8ec3ebd6bf9`

历史训练语义纠错证据：

- `/home/lxq/Softwares/robot_lab/tmp/highstep_historical_0707_exact_20260714/historical_evidence_correction.json`
- SHA256：`eb561df82d5f116577497eb7c3f553bb82ae81493258599212da59d4d0c6a175`

## 3. 两项明确且仅此两项的训练语义差异

相对于历史 0707 完整链，本轮存在两项用户明确批准的训练语义差异：

1. Teacher 从历史 `model_151399.pt` 更换为本规范第 2.1 节的当前 B300 `model_173499.pt`。
2. 历史链为 `2500 + 4900` 两个进程，第二个进程会重新开始一次 1400-update warmup；本轮改为一个连续 7400-update 进程，只执行一次 1400-update warmup，不在中途重置蒸馏计数或 optimizer。

除此之外，训练环境、网络、loss、LR、冻结范围、action/observation contract 和 Teacher 标签语义必须以冻结的 0707 证据复现。路径、run 名、W&B 元数据和日志目录属于基础设施差异，不属于训练语义变量。

## 4. 冻结训练合同

### 4.1 Student 环境与输入输出

- 精确使用 SHA256 为 `f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636` 的 0707 Student `env.yaml` 及其对应运行时代码语义。
- 禁止继承后来加入的 robust-from-update-zero 环境、额外随机化、持续偏置、额外外力、额外 reset、恢复门禁或 curriculum 修改。
- Student 输入历史为 570 维；估计器输出 64 维 latent；actor 输入为 `570 + 64 = 634` 维；输出为 16 维。
- actor hidden dims：`[512, 256, 128]`，activation：ELU。
- VAE/estimator hidden dims：`[256, 128]`。
- observation history、关节顺序、scale、clip、action scale、`joint_pos.clip`、`default_dof_pos` 必须与冻结 0707 配置一致。
- Student 使用 0707 plain `JointPositionAction` 生产执行通道；Teacher 标签必须保留 0707 对应的 action-prior 语义。
- 本轮固定使用 0707 的 `delay=0` 训练语义，不引入 B300 后来新增的 0–1 delay 分布。

### 4.2 Stage-2 蒸馏机制

- `distill_stage=2`。
- PPO、critic/reward 学习永久关闭；不得以 PPO 更新 Student。
- 每个 update 执行 4 个 VAE/Student epochs。
- estimator learning rate：`1e-3`。
- 0707 基础 algorithm learning rate：`1e-4`，adaptive schedule 语义保持不变。
- velocity loss weight：`10`。
- latent loss weight：`50`。
- Teacher action loss weight：`20`。
- prior-box loss weight：`5`。
- reconstruction loss weight：`0.5`。
- KL loss weight：`0.1`。
- phase scale：`2.0`。
- rear-box scale：`1.5`。
- warmup：相对蒸馏 update `0–1399`，共 1400 effective updates。
- warmup 主 action target：Teacher `pre-prior` 16 维动作。
- phase/rear-box 加权从训练开始进入 estimator 的 Teacher action loss。
- warmup 期间额外 prior-box loss 必须为 0。
- warmup 期间 actor、critic 和 privileged encoder 完全冻结，只有 0707 原机制允许的 estimator/VAE 参数训练。
- 相对 update 1400 后，才允许最后 4 个 box output rows 及其 bias 按 0707 原有条件式 post-prior box loss 和 LR `1e-5` 有界适配。
- 其余 actor rows 永久冻结；禁止全 actor 解冻。

### 4.3 初始化、双 checkpoint 与 optimizer

- Student/独立 frozen Teacher 的初始化和加载顺序必须复用 0707 初始 Student run 的 Stage-2 运行时代码语义，不得按后来路线重新解释。
- Student 与 Teacher 参数存储必须独立；Teacher 全程 frozen。
- 不得从任何后来 Student、direct-action、v1.7/v1.8、recovery 或虚拟地形路线的 Student/optimizer 恢复。
- optimizer 的创建与是否加载 Teacher checkpoint 内已有 optimizer 字段，严格服从冻结的 0707 初始 Stage-2 实现；禁止为了“看起来更干净”擅自改成另一套 fresh/load 规则。
- `relative_distill_update_count` 从 0 开始且只初始化一次。
- `absolute_runner_iteration` 从 Teacher checkpoint 的 173499 接续，禁止重置为 0。

## 5. 单次连续 7400-update 生命周期

- 一个训练进程。
- 一个 Student W&B run。
- 一个 optimizer 生命周期。
- 连续完成 7400 effective updates。
- 禁止在 update 2500 或其他位置人为拆段、重启、重置 warmup、重置 optimizer 或重置 relative update count。
- `save_interval=100`；每 100 update 保存一次完整 checkpoint，并在 7400 完成时保存最终完整 checkpoint。
- 完整 checkpoint 至少包含 Student、独立 Teacher 绑定、optimizer、absolute runner iteration、relative distill update、schedule/runtime state。
- 所有中间 checkpoint、日志和 W&B 本地文件均保留，不覆盖、不删除。

## 6. 训练中绝对禁止的操作

从第一个 optimizer update 开始到第 7400 个 update 原子写入完成，禁止：

- 行为评估、probe、core9、directional corridor 或 15 场评估；
- play、视频录制、MuJoCo、sim-to-sim、部署或真机测试；
- latent/action/terrain/loss 门禁；
- 依据 W&B 曲线人工或自动提前停止；
- 阶段切换、用户视觉复核等待或候选选择；
- 修改 Teacher、Student 网络、loss、LR、冻结范围、环境、reward、randomization、prior、schedule 或动作合同；
- 自动重启训练、从中间 checkpoint 自动恢复、自动换路线；
- 新增自定义 NaN guard、process guardian、行为 watchdog、诊断采样或自动修复分支。

不得关闭训练框架本身已有的异常与文件写入错误；本条只禁止为了本轮额外开发自定义监控/门禁。

## 7. 启动前允许的最小机械检查

启动前只允许执行以下不启动 Isaac、不产生 optimizer update 的机械检查：

1. 重新校验本规范、Teacher checkpoint、冻结 0707 agent/env/runtime 证据的 SHA256。
2. 确认路径可读、输出目录可写、GPU 当前未被 train/eval/play 占用。
3. 确认最终启动参数的静态展开值与本规范一致。
4. 完成第 8 节 W&B 在线初始化并取得 run id/URL。

明确禁止 Teacher preflight rollout、Student smoke rollout、1–5 update smoke、行为测试和额外全量测试。若实现代码无法通过最基本的 Python import/compile，应在第一个 optimizer update 前退出并报告，不得边训练边修。

## 8. W&B 在线同步与横轴接续合同

### 8.1 固定在线配置

- `WANDB_MODE=online`，禁止 offline-first。
- entity：`xinqili551-the-university-of-hong-kong`
- project：`isaaclab`
- group：`highstep_b300_0707_single_run_7400_20260718`
- run name：`highstep_b300_0707_single_run_7400_student_from_173499`
- 只允许一个 Student run；禁止按 100-update checkpoint 拆分多个 W&B run。

### 8.2 曲线接续方式

不得复用或写入 Teacher 原 run id。Student 必须是独立 run，但 W&B `step` 使用 absolute runner iteration：

- Teacher parent step：`173499`。
- Student `relative_distill_update_origin=0`。
- Student `absolute_runner_step_origin=173499`。
- Student 第一批日志从 absolute step 173499 附近开始，随后连续推进 7400 updates；禁止从 W&B step 0 开始。
- 最终绝对 step 以 checkpoint metadata 的实际 off-by-one 语义为准，预期约为 180899。

这样在 W&B 中叠加 Teacher 和 Student 曲线时，Student 会从 Teacher 的 173499 位置接续，而不是从 0 重新绘制。不得通过伪造训练数据或回填 Teacher history 来实现视觉接续。

W&B config 必须至少记录：

- `parent_teacher_checkpoint_path`
- `parent_teacher_checkpoint_sha256`
- `parent_teacher_wandb_run_id`
- `parent_teacher_wandb_url`
- `absolute_runner_step_origin=173499`
- `relative_distill_update_origin=0`
- 本规范路径与 SHA256
- preregistration 路径与 SHA256
- 0707 agent/env/runtime 证据路径与 SHA256
- `max_effective_updates=7400`
- `save_interval=100`

### 8.3 唯一 W&B 启动门禁

- 在第一个 optimizer update 前必须成功在线初始化 W&B，并取得可访问的 run id 和 URL。
- 若 W&B 在启动阶段无法连接、认证、创建 run 或返回 URL，禁止开始训练；安全退出并把唯一具体错误报告给用户，由用户判断重试还是处理网络/VPN。
- 训练已经开始后，不设置额外 remote-completeness 门禁，不因短暂网页刷新、上传延迟或某个 summary 字段暂缺而中断 7400-update 训练。
- 允许 W&B SDK 使用其原生缓冲与重试；禁止为此开发新的阶段 supervisor 或多片段补传分支。

## 9. 异常退出规则

- 本轮不设置自动恢复、自动重启或无限 retry。
- 若唯一训练进程异常退出，保留最后完整 checkpoint、日志、W&B 本地目录和错误现场，状态写 `unexpected_training_exit_requires_user_decision`。
- 禁止 systemd `Restart=on-failure` 反复重训或重复创建 W&B run。
- 禁止把基础设施退出解释成训练行为失败。
- 未经用户决定，不得从最后 checkpoint 自动继续，也不得从头重训。

## 10. 7400 完成后的唯一下一步

只有第 7400 个 effective update 和最终 checkpoint 原子保存完成后，才允许：

1. 核验 checkpoint、optimizer、双绑定、relative/absolute counters 和 W&B 记录完整性。
2. 对每 100-update checkpoint 进行统一的训练后评估与人工视频筛选。
3. 选择行为最佳 checkpoint；不得默认选择最终 checkpoint。
4. 是否进入 MuJoCo、部署或真机测试由用户另行批准。

本规范不预注册训练后的行为阈值，不允许训练期间因任何自动评分提前停止；评估规范须在长训完成后另行建立，不能反向改变本轮训练事实。

## 11. 明确非目标

- 不证明 B300 Teacher 一定可蒸馏。
- 不保证 7400 updates 一定优于历史 0707 Student。
- 不解决 box joint sim-to-real gap、真机 gains、延迟、FSM 或 passive 切换。
- 不训练多地形、多指令或通用盲走策略之外的新部署合同。
- 不自动导出、部署或上真机。
- 不修改现有正式 v1.13.1 或任何历史结果。

## 12. 主对话激活要求

主对话实施时必须：

1. 基于本规范创建独立只读 preregistration，冻结本规范全部路径、SHA、W&B metadata 与 7400 单进程合同。
2. 创建独立 workflow/state/lock/heartbeat/handoff；heartbeat 只记录训练进程存活和 absolute/relative update，不读取指标做门禁。
3. 禁用任何会自动重启该训练的 service；若使用 launcher，必须是单次启动且不具备自动恢复逻辑。
4. 原子迁移 dashboard authority 后，再执行第 7、8 节的最小启动检查。
5. 汇报 W&B run id/URL、真实 train PID、输出目录、首个 absolute/relative step 后，让训练不受干扰地连续完成。

任何实现若需要改变本规范的 Teacher、0707 环境、loss、warmup、训练分段、7400 预算、W&B absolute-step 合同或中途检查禁令，必须先停止在第一个 optimizer update 之前并重新取得用户批准。
