# Highstep 阶段归档：E7700 真机成功上台（2026-07-30）

## 归档结论

本轮 highstep 工作在 2026-07-23 完成了一次有人保护的真机实验。用户确认并经视频复核：

- E7700 Student 完成整机上高台；
- 四条腿均进入台面，并在 `policy2` 阶段保持站立姿态；
- 随后用户主动切换到 `Fixed Down`，下压动作使一条后腿被台边挤出台面；
- 该后续现象不属于上台失败，也不改写已经完成整机上台的事实。

本归档表示当前阶段工作结束，不表示已经获得无人保护、统计鲁棒或自动部署许可。后续若继续开发，仍在当前 `dev_lxq_new` 分支进行，并由新的用户任务明确范围。

## 最终模型链

| 角色 | 路径 | SHA256 |
| --- | --- | --- |
| B300 Teacher | `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt` | `d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431` |
| E5700 合法续训源 | `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_Student/2026-07-19_06-43-15_highstep_b300_critical_transition_balanced_diagonal_fresh_7400_attempt2_from_173499/model_179198.pt` | `31fe19c7d1ba9872c0b714d594fe6e66c549a52e88fe856ed62206680d37c498` |
| E7700 最终完整 checkpoint | `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_rl_preedge_continuation_e7700_Student/2026-07-20_02-14-00_highstep_b300_rl_preedge_continuation_e7700_attempt9_repaired/model_181198.pt` | `86d7ee987f8d421fdc18ad14474887699a40abbf725ca6d91ac191a31e88abd2` |
| E7700 TorchScript | `/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_rl_preedge_continuation_e7700_Student/2026-07-20_02-14-00_highstep_b300_rl_preedge_continuation_e7700_attempt9_repaired/exported/policy_student.pt` | `ed2d13516ce47ba6b8442469f29d5be1d64b4f6ce9a3af2bf39601b0715d7937` |

训练终态：

```text
workflow=highstep_b300_rl_preedge_continuation_e7700_20260719
effective_updates=7700
absolute_runner_step=185793
W&B run=126d9b3f
W&B URL=https://wandb.ai/xinqili551-the-university-of-hong-kong/isaaclab/runs/126d9b3f
```

## 真机证据

视频：

```text
/home/lxq/Videos/3e6c115b13f9ba83fcbe23a224ab3bcd.mp4
SHA256=068fe2e6395b091fa44e63f51213cb383f412eec2a39d52efcae7e7b41a99bfe
duration=46.800 s
resolution=1280x720
codec=HEVC
frames=1403
```

rosbag：

```text
/home/lxq/log/rosbag2_2026_07_23-15_25_01
duration=63.588121156 s
messages=305686
metadata SHA256=f5ea4a71a7199ca08ab6493b19aa95a93523a6328eecb400bd0cd22883722000
db3 SHA256=9282a3951e43913d61f220702e7ee59beee3432a359d3ceca2384cea23c56363
```

视频关键时序复核：

```text
约 37--39 s：前腿、后腿依次完成登台。
约 39 s：四条腿均在台面，机器人保持站立。
约 40 s 后：用户切换 Fixed Down；机器人下压，一条后腿随后被台边挤出。
```

bag 中的 FSM 顺序与这一解释一致：

```text
Fixed Stand -> policy2 -> Fixed Down -> passive
```

视频和 bag 没有共同的绝对录制时间戳，因此这里只做事件顺序和动作形态对齐，不声称逐帧严格同步。

## 真机实际部署合同

bag 的 `/rl_deployment_manifest` 共记录 6 条内容一致的 manifest。它绑定：

```text
policy2 model:
policy_student_b300_rl_preedge_E7700_model_181198.pt

model fingerprint:
fnv1a64:7677c5120d02eb7b:bytes:3535560

config fingerprint:
fnv1a64:dd18f7e06b513bd1:bytes:4049

input/output:
570 -> 16

disabled:
rear_hip_guard
fixed_motion
one_shot
teacher_virtual_terrain
```

精确 gains、joint order、scale 和限位已冻结在：

[config_e7700_real_test_20260723_reference.yaml](config_e7700_real_test_20260723_reference.yaml)

该文件是证据参考，不覆盖任一电脑正在使用的 `config.yaml`。

## 仍需保留的诊断事实

- policy2 阶段共记录 474 次推理提交，最终 `deployment_fault=0`。
- `/rl_target_clamp_debug` 记录 104 个物理目标限幅事件，主要集中在约 2.12 s 的连续窗口。
- 最大记录限幅差为 `0.444215 rad`。
- 这些限幅值得未来复盘，但当前证据不足以把它们直接定性为失稳原因，更不能覆盖真机已经成功上台的视觉事实。
- `/rl_policy2_contract_trace` 在该 bag 中没有消息；后续不能假称已有逐帧 contract trace。

## 调试经验

1. **必须按 FSM 分段解释视频。** `policy2` 已完成目标后切换到 `Fixed Down` 引起的姿态变化，不能倒算成策略上台失败。
2. **用户视觉确认是动作结论的最终证据。** 自动指标、缩略图和孤立关键帧不能替代完整视频时序。
3. **训练、部署和真机配置必须分别绑定。** checkpoint SHA、TorchScript SHA、机载 model/config 指纹和 bag manifest 都应记录，不能用本机当前 `config.yaml` 代替真机事实。
4. **基础设施错误不能冒充行为失败。** W&B、评估映射、稀疏采样和 checkpoint 命名错误都应单独修复并保留失败事实。
5. **有效 update 与完整恢复状态必须一致。** Student、Teacher 绑定、optimizer、effective update、runtime/schedule 缺一不可称为完整续训。
6. **大文件不进入普通 Git。** 视频、bag、checkpoint 和 W&B 原件保留在工作站；Git 只保存报告、路径、指纹和可复现代码。

## Source control 范围

本阶段只整理 `/home/lxq/Softwares/robot_lab`：

- 保留并提交 highstep/adjustable-leg 源码、针对性测试、spec、工具和经验文档；
- 已终止路线的入口明确视为 historical，不自动启动；
- 排除日志、checkpoint、W&B、视频、rosbag、生成文件和机器本地临时配置；
- 不修改或提交 `/home/lxq/colcon_ws/src/quadruped_control_ros2`。

## 当前状态与未来入口

```text
phase=archived_after_successful_real_robot_test
status=archived_after_successful_real_robot_test
automatic_training=false
automatic_evaluation=false
automatic_restart=false
automatic_real_robot_deployment=false
```

未来如果继续 highstep：

1. 先读本文件；
2. 校验上述模型和证据 SHA；
3. 明确新的开发目标；
4. 不自动恢复任何旧训练、评估或部署流程。
