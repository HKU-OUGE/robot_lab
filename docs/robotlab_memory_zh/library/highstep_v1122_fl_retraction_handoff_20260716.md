# Highstep v1.12.2：左前足 6 cm 预接触回收交接

更新时间：2026-07-16 02:05 +08:00

## 当前结论

- v1.12 behavior-best E1000 的后腿动作已经由用户接受，必须保持，不再扩散修改范围。
- 正确人工证据是 `/home/lxq/Videos/Kazam_screencast_00155.mp4`，SHA256=`5350ab46fa2cfe1e966f99e6717b738b60b32b9edcbd1f4a323205a55f2c128e`，不是 00154。
- 视频约 74.22 秒；14 次完整尝试中 6 次完整成功，8 次由同一种 FL 足端撞台阶立面/顶沿下方并卡住而 reset；末尾另有 1 次未完成尝试。
- 用户确认：只处理 FL；任何 FL 立面接触即失败，即使随后恢复也不改判；首轮正式门禁是 0/15。
- 用户选择 A 型动作：准备上台时 FL 短暂向 body-frame 后方回收，幅度固定 6 cm，允许约 0.1–0.2 秒后再恢复既有前伸。FR、RL、RR 不加新目标。

## 14 次完整尝试冻结标注

| 尝试 | 视频区间（秒） | 判定 |
|---:|---:|---|
| 1 | 0.000–4.600 | 完整成功 |
| 2 | 4.600–8.983 | FL 卡立面，失败 |
| 3 | 8.983–14.133 | 完整成功 |
| 4 | 14.133–17.600 | FL 卡立面，失败 |
| 5 | 17.600–23.200 | 完整成功 |
| 6 | 23.200–30.917 | 完整成功 |
| 7 | 30.917–35.633 | FL 卡立面，失败 |
| 8 | 35.633–40.017 | FL 卡立面，失败 |
| 9 | 40.017–46.783 | 完整成功 |
| 10 | 46.783–52.867 | 完整成功 |
| 11 | 52.867–58.000 | FL 卡立面，失败 |
| 12 | 58.000–61.667 | FL 卡立面，失败 |
| 13 | 61.667–66.167 | FL 卡立面，失败 |
| 14 | 66.167–71.967 | FL 卡立面，失败 |
| 未完成 | 71.967–74.217 | 视频结束，不计分母 |

典型失败链：FL 抬起后足尖过早向前，先碰台阶竖直面或顶沿下方；机身仍向前/上运动，使 FL 远端链折叠并被顶住；未出现可靠再次抬腿，用户 reset。没有证据支持同时修改 FR 或后腿。

## v1.12.1 错误与 v1.12.2 修正

v1.12.1 草案把 `front_legs_reach` 从 0.45 降为 0.35，再增加 FL 项 0.10。虽然总数仍为 0.45，但这会减弱同时作用于两条前腿的原奖励，间接改变 FR，违反用户批准的 FL-only 范围。该草案从未训练。

v1.12.2 固定为：

- `front_legs_reach.weight = 0.45`，与 v1.12 完全相同；
- `left_front_precontact_retraction.weight = 0.10`，唯一新增训练语义；
- `retraction_start_x=0.36 m`、`retraction_target_x=0.30 m`，即 6 cm；
- 仅在高台、正向 command、front precommit、FL 已抬起且尚未越过台面释放线时生效；
- FL 越过台面后立即衰减，由原有 reach/support 完成落足；
- FR 和后腿只保留原有合同，不接受新动作/关节/足端目标；
- reward 其余项、Teacher prior、网络、observation/action、optimizer/LR/PPO/schedule、DR、`action_scale`、`joint_pos.clip`、default pose、真机 gains 均冻结。

修改文件：

- `/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py`
- `/home/lxq/Softwares/robot_lab/source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py`
- `/home/lxq/Softwares/robot_lab/tests/test_highstep_teacher_front_placement_v1121.py`
- `/home/lxq/Softwares/robot_lab/tests/test_highstep_teacher_rear_support_v112.py`
- `/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md`

目标 task 仍为 `RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-ArcdogAdjustableLeg-v0`；v1.12.2 通过新的 spec/code/preregistration SHA 与未训练的 v1.12.1 草案严格区分。

## 固定起点与训练边界

唯一 source checkpoint：

`/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_rear_support_v112_Teacher/2026-07-15_10-52-45_v112_long_attempt1_20260715_105240/model_173200.pt`

SHA256=`962fd3ce3983e4a478092be8f7a636e87ed872b79496c4fa3134191838d9b80d`

当前没有 train/eval/play/supervisor。不能恢复旧 v1.12 6000-update service，也不能从 E6000 继续。下一步必须由主对话先冻结有限训练预算、保存点与停止条件 amendment；然后从 E1000 做 task-only full resume，验证 actor、critic、optimizer、iteration、schedule/runtime，完成 1–5 update smoke 后才启动正式训练。

## W&B 实时同步硬门禁

- 每个有限训练阶段必须是独立 W&B online run，`group=highstep_teacher_front_placement_v1122_20260716`。
- config 必须记录 spec/prereg/code/source checkpoint SHA、唯一变量、optimizer、iteration budget 和 task。
- state/heartbeat 必须记录 run id、URL、sync status 和远端最新 step。
- 远端 history step 必须随 effective updates 实时增长；网络或 W&B 故障时保留本地日志和完整 checkpoint，并在安全边界 fail-closed，不得进入下一阶段或正式评估。
- 阶段结束 summary 必须记录 output checkpoint SHA、effective updates、0/15 卡台结果、完整上台/rear-hold 指标、门禁结论和 manifest。

## 正式评估与终态

- 15 场候选自主 rollout；valid=15/15；FL 立面或顶沿下方接触=0/15。
- 完整上台与 rear hold 不得低于同规格冻结 E1000 baseline。
- FR/后腿只做非退化检查；后腿巩固或轻微增强只能在 FL 门禁相同的候选间作 tie-breaker。
- 数值通过后只可写 `teacher_fl_retraction_candidate_pending_user_visual_review`，生成机器人全程可见且不丢弃失败尝试的视频；禁止自动 Student 或真机部署。

## 当前唯一下一步

主对话读取正式 spec、当前 preregistration、state 和本文件并校验 SHA；先写不可修改的有限预算 amendment，再做 full-resume rebinding、static/binding、1–5 update smoke。正式训练必须在启动时就验证 W&B online run、URL和远端 step 增长，不能先离线训练后补传。
