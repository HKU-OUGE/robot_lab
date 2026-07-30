# Highstep 固定条件 Teacher phase-residual 模仿实验分支规范 v0.3

- 创建日期：2026-07-17
- 状态：用户已批准创建独立实验分支、使用现有约 0.38 m Teacher 动作，并直接采用 `Student obs + explicit phase -> bounded trajectory residual`；尚未成为 active workflow authority
- 分支 ID：`highstep_fixed_condition_phase_residual_20260717`
- 路线性质：纯监督模仿、固定条件、显式动作阶段、无泛化目标、可独立废弃
- 最高目标：在预先标定的固定摆位下，让当前机器人完整复现已接受 Teacher 的约 0.38 m 上高台动作，供有人保护的真机实验

## 1. 与正式主线的隔离关系

本文件从下列正式规范分支，但不修改、不覆盖、不重绑定该规范：

```text
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md
parent version: v1.13.1
parent SHA256: 0307b9e5c5c9c81999499ce2c3f05c0beabaa98caedde24aac5ac7843d73d768
```

隔离规则固定如下：

1. v1.13.1 的 spec、preregistration、dashboard、state、service、checkpoint、W&B 和人工结论保持原样。
2. 本分支使用新的 workflow/state/artifact 目录；不得复用正式主线 state 或 service 名。
3. 本分支不得自动成为正式 highstep authority，不得自动部署真机。
4. 本分支失败时只把自身状态写为 `experimental_branch_rejected`，随后可整体废弃；不得把失败扩大为否定 v1.13.1、Teacher 或现有 Student 结果。
5. 本分支成功时也只能写 `fixed_condition_replay_pending_user_real_robot_approval`；它不是通用 Student，也不声称具备泛化能力。

## 2. 唯一 Teacher 与动作合同

唯一 Teacher 固定为：

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt
SHA256: d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431
```

本路线不修改 Teacher、reward、网络、action prior、`action_scale`、`joint_pos.clip`、default joint offset、部署输入或真机 gains。Teacher 只在 Isaac 仿真中生成最终的 16 维 post-prior mapped joint target；部署端不运行 Teacher，也不需要 privileged observation。

固定 16 维关节顺序为：

```text
FL/FR/RL/RR hip,
FL/FR/RL/RR thigh,
FL/FR/RL/RR calf,
FL/FR/RL/RR box
```

物理目标范围固定为：

```text
hip:   [-1.2217304764,  1.2217304764] rad
thigh: [-1.5708,        3.4907] rad
calf:  [-2.7750735107, -0.6457718232] rad
box:   [ 0.0,           0.06] m
```

源 Teacher mapped target 允许保留越限事实，但导出的每一帧部署 reference target 都必须在上述范围内。源 target 到部署 reference 的物理范围投影必须是显式、可哈希和可复核的离线变换，随后重新通过完整 replay 门禁；禁止依靠运行时隐藏 clamp 把未经验证的越限轨迹改判为可用。

## 3. 固定任务条件

第一版只服务于下列单一条件：

```text
reference/evaluation platform height: approximately 0.380 m (box_hard level 9)
platform width: 3.0 m
approach side: x-
front edge clearance at reset: 0.550 m
lateral offset: 0.000 m
yaw offset: 0.0 deg
recorded Teacher command during the retained active segment: vx=+0.72 m/s, vy=0, wz=0
action delay in the retained recording: 0 step
ordinary random events: disabled
```

真机摆位必须使用物理量具标定平台高度、前缘净距、横向偏移和偏航；不得只凭肉眼判断。进入轨迹状态后，外部速度命令只作为 arm/start 触发，不再实时改变参考轨迹。若最终真机平台实测仍为 0.350 m，则必须在真机前用同一 0.38 m reference 额外完成一次 0.350 m Isaac/MuJoCo replay；不重新采 Teacher，也不得隐瞒这 3 cm 接触时序差。

现有 `box_hard level 9` 范围为 `0.30..0.38 m`，最高级约 0.38 m。用户已批准直接采用这组约 0.38 m 记录，因此本分支不再重新采 0.350 m Teacher reference。

## 4. 现有记录的用途和边界

现有 B-E300 Teacher 记录：

```text
manifest:
/home/lxq/Softwares/robot_lab/logs/play_joint_records/20260717_072307_teacher_b300_box_hard_seed11_pid877984/manifest.json
SHA256: a88f0bebdc3343f9d9465f0f2a17f93e42c553f19fe94e216ca49f4bff32c3f0

joint trace:
/home/lxq/Softwares/robot_lab/logs/play_joint_records/20260717_072307_teacher_b300_box_hard_seed11_pid877984/joint_trace.csv
SHA256: 2b076e3972c9d7f40873e4c94b5d82e250ab874a950019a7be1676656b70368d
```

该记录包含 12 次完整 episode；12 次最终机身高度约为 0.79--0.80 m，证明当前 Teacher 在固定高台附近能稳定产生可提取的完整动作。用户已批准约 0.38 m 和实际 `vx=0.72` 的运动段作为正式 source candidate。

现场审计发现 source mapped target 的 FL hip、FR calf 和 RR calf 存在短时物理 envelope 超限，但实际仿真 q 未越界。因此禁止原样播放 CSV；必须从现有 12 条 episode 生成有界 reference candidate，并绑定：Teacher checkpoint SHA、环境快照 SHA、play 代码 SHA、动作项类型、关节顺序、地形高度、摆位、命令、delay、每帧 source mapped target、物理投影结果、实测 q/dq 和采集文件 SHA。

## 5. 真机 rosbag 与部署端可用信号审计

审计依据：

```text
/home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_28_27
/home/lxq/Softwares/robot_lab/tmp/rosbag_0707_recovery/recovered.db3
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_0707_real_log_audit_handoff_20260712.md
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/highstep_0707_real_data_analysis_20260712.md
/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/sim_to_sim_deployment.md
```

### 5.1 可用于在线闭环的信号

| 信号 | 部署端来源 | rosbag 证据 | 第一版用途 |
|---|---|---|---|
| 16 关节位置 q | ros2_control position state | `/joint_states`、`/dynamic_joint_states` | 起始姿态校验、跟踪误差、阶段推进和安全停止 |
| 16 关节速度 dq | ros2_control velocity state | 同上 | 跟踪稳定性、异常速度保护 |
| IMU quaternion | IMU state interface | `/imu_sensor_broadcaster/imu`、`/imu` | roll/pitch 姿态与倾倒保护 |
| IMU gyro | IMU state interface | 同上 | 角速度保护、阶段稳定判断 |
| IMU acceleration | IMU state interface | 同上 | 冲击记录；第一版不单独据此判接触 |
| 手柄/FSM command | realtime control input | `/joy` | arm、start、abort 和切换 fixed stand/passive |
| 单调控制时间 | controller update clock | 新 debug topic | 固定轨迹相位与 timeout |
| 实际 q_des/Kp/Kd/tau_ff | command interfaces | `/joint_commands` | 运行真实性、跟踪和配置审计 |

### 5.2 只允许诊断、不得作为第一版阶段门禁的信号

- `joint_states.effort` / `tauEst`：字段存在，但 0707 bag 的 box effort 数值量纲明显可疑；未经悬空标定和已知载荷标定，不得作为接触检测或相位推进条件。
- IMU acceleration：可记录碰撞峰，但单独不能区分足端接触、机身冲击和安全绳作用。
- `/tf`：可在离线复盘中重建几何，但不是当前真机控制器的可靠外部位姿传感器。

### 5.3 当前拿不到的信号

- 没有足端接触/足端力 topic；
- `/NED_odometry`、`/system_speed`、`/gps/fix` 在 0707 bag 中均为 0 条；
- 没有平台相对位姿、相机、深度或高度图；
- `RobotState.cur` 虽在结构中声明，但当前 `getState()` 没有填充，不能使用；
- 没有安全绳拉力。

结论：第一版不能假装拥有接触传感器。它必须依赖固定摆位、单调时间、q/dq 跟踪和 IMU 姿态完成盲爬。

## 6. 实现选择

第一版采用独立 ROS2 FSM 状态 `HIGHSTEPPHASERESIDUAL`，同时加载一条有界 reference 和一个小型 TorchScript residual model。它不是现有 Student，也不从现有 Student checkpoint 恢复。

固定计算为：

```text
student_obs_570 = existing blind deployment observation contract
phase_features_8 = global phase + six-stage one-hot + within-stage phase
delta_q_16 = bounded_residual_model(student_obs_570, phase_features_8)
q_command_16 = physical_clamp(q_reference_16(phase) + delta_q_16)
```

选择 residual 而不是从零预测完整 action，理由是：

1. reference 保证基本动作顺序和幅值，网络只纠正同阶段的小偏差；
2. residual 可逐关节限制，模型异常或禁用时令 `delta_q=0` 即可退回基础轨迹；
3. explicit phase 直接消除盲观测难以区分 `front_support/first_rear/second_rear` 的主要歧义；
4. 不需要 estimator、privileged latent、PPO、reward 或把 Teacher action prior 内化到完整 actor；
5. 失败时可删除 reference/residual/FSM，不影响 RL/Student 主线。

Reference 资产是 50 Hz 的有界 joint target 序列及阶段标记。部署控制循环使用线性插值提升到控制频率；禁止对多个接触轨迹逐帧平均。

Residual model 固定为小型 MLP，输入 578、输出 16。逐关节 residual bound 在训练前由训练集 `q99(abs(target-reference))*1.25` 冻结，且 revolute 上限不得超过 `0.20 rad`、box 上限不得超过 `0.006 m`。最终 physical clamp 仍强制执行。

## 7. 固定六阶段与在线推进

离线借助 Isaac 接触/根状态把 reference 标记为：

1. `approach`
2. `front_lift`
3. `front_support`
4. `first_rear`
5. `second_rear`
6. `rear_hold`

真机在线不重新估计足端接触。每阶段采用：

- 主时钟：reference 单调时间；
- 辅助条件：q 跟踪误差在允许范围、dq 无异常、roll/pitch/gyro 无安全越界；
- 若跟踪暂时落后：允许在预先固定的最大等待时间内减慢或暂停相位；
- 若超时或姿态越界：停止向后推进，进入 current-q hold，并等待人工切换 fixed stand/passive；
- 到达 `rear_hold`：保持最终 reference，不自动部署、不自动继续行走；用户可手动切换 fixed stand 复位。

第一版不得使用未经标定的 effort 伪造接触门控。在线只允许 residual model 在冻结 bound 内修正 reference，不得改变 reference、phase 定义或 residual bound。

## 8. Reference 与监督数据生成

1. 只读解析现有 12 条约 0.38 m 成功 episode，不重新运行 Teacher。
2. 预先固定三个 source candidate：episode 0（最小横向漂移）、episode 4（最少 source target 超限帧）及按全局姿态/跟踪距离得到的 medoid；禁止看 replay 结果后增加候选。
3. 对每个 candidate 同时生成两种有界 seed：显式物理范围投影后的 mapped target，以及仿真实测 q 轨迹。两者都绑定变换代码与输出 SHA。
4. 从命令启动前的稳定姿态开始裁剪，到 rear hold 稳定结束；保留短 current-q 到 reference 起点的 quintic blend。
5. 先选择与原 Teacher 实测 q 最接近、且能完成固定场景 replay 的有界 reference；它是 residual model 的零残差基线。
6. 现有 CSV 没有保存精确 570 维 Student observation，因此必须在相同 0.38 m 固定场景重新运行 Teacher 15 次，但只新增 observer，不改变 Teacher、环境或动作。每帧直接保存 exact `student_obs_570`、explicit phase、Teacher post-prior mapped target、物理投影后 target、q/dq 和事件阶段。
7. 570 维 observation 的 `actions` 历史固定为最终安全 joint target 经原 `default_dof_pos/action_scale` 逆映射得到的 no-prior raw action；禁止把 Teacher prior 的内部 raw action直接塞入 Student action history。
8. dataset 按完整 rollout 切分为 12 train / 3 validation，禁止同一 rollout 帧泄漏到两侧；batch 在六阶段间均衡采样，避免 rear transition 被 approach/rear_hold 长时样本稀释。
9. 监督 target 固定为 `safe_teacher_target - q_reference(phase)`。只训练 residual MLP，loss 只包含逐关节 Huber residual loss 和相邻帧一阶差分 loss；不得加入 PPO、reward、latent、velocity、recon 或 KL。
10. 初始行为克隆后由 hybrid 自主驱动固定场景，Teacher 在同一状态只提供 target。若未达到 Isaac 门禁，最多允许两轮固定场景 DAgger 追加；每轮数据和模型独立留档，禁止扩大环境分布。
11. 生成只读 dataset/reference/training manifest，绑定 source trace SHA、Teacher SHA、环境 SHA、observation/action contract、phase 定义、rollout split、residual bound、代码和最终 model/reference SHA。

## 9. 验证门禁

门禁按以下顺序执行，上一层失败不得进入下一层：

### A. 离线真实性

- Teacher、环境、trace、reference、部署配置 SHA 全部匹配；
- 16 维顺序完全一致；
- reference 非有限值数为 0，物理 target 超限帧数为 0；
- 轨迹切换首帧与当前站立姿态通过有界 quintic blend，无全零 q/Kp/Kd 窗口。

### B. Isaac 固定场景 hybrid rollout

- 同一约 0.380 m `box_hard level 9` 场景连续 15 次；
- valid=15/15、完整上台=15/15、rear hold=15/15；
- residual-disabled reference-only 结果与 residual-enabled 结果必须同时保留；
- 失败不得靠重跑删除，任何失败都必须保留并分析。

### C. MuJoCo sim-to-sim

- 使用部署端相同 FSM、reference、residual TorchScript、578 维输入合同、关节顺序、limits 和已冻结 gains；
- 同一约 0.380 m 固定摆位连续 15 次；
- valid=15/15、完整上台=15/15、rear hold=15/15；
- `/joint_commands`、`/joint_states`、IMU、trajectory debug 和 manifest 全部录制；
- 不允许通过改 action scale、关节 offset、Teacher 或全局 gains 让结果过门。

### D. 真机前低风险验证

- clean build、单元/回归测试、FSM switch test 通过；
- 悬空执行完整 reference，确认 q_des、Kp/Kd、joint order、limits、abort 和 fixed-stand 切换；
- 真机实际 wire-level gains 与冻结 manifest 一致；
- 用户观看 Isaac/MuJoCo 视频并单独批准后，才可进行有人保护真机实验。

## 10. rosbag 与 debug 合同

新增只读发布：

```text
/highstep_phase_residual_manifest  std_msgs/String
/highstep_phase_residual_debug     std_msgs/Float64MultiArray
```

manifest 至少包含：workflow、spec/reference/config/code SHA、关节顺序、limits、gains、控制频率、当前运行编号和固定摆位。

debug 固定 schema 至少包含：sequence、elapsed time、stage、phase、reference index、q_ref[16]、delta_q[16]、q_command[16]、q[16]、dq[16]、IMU quaternion[4]、gyro[3]、acceleration[3]、最大/均值 tracking error、phase hold 原因和 safety flags。

每次仿真和真机实验必须同步录制：

```text
/highstep_phase_residual_manifest
/highstep_phase_residual_debug
/joint_commands
/joint_states
/dynamic_joint_states
/imu_sensor_broadcaster/imu
/imu
/euler_angles
/joy
/rosout
```

若某个必需 topic 缺失，运行只能记为基础设施无效，不得作为成功或行为失败样本。

## 11. 预计代码范围

实施时只允许新增/最小接线：

```text
controller_common/include/controller_common/common/enumClass.h
rl_quadruped_adjustable_leg_controller/include/.../FSM/StateHighstepPhaseResidual.h
rl_quadruped_adjustable_leg_controller/src/FSM/StateHighstepPhaseResidual.cpp
rl_quadruped_adjustable_leg_controller/src/RlQuadrupedControllerAdjustableLeg.h
rl_quadruped_adjustable_leg_controller/src/RlQuadrupedControllerAdjustableLeg.cpp
rl_quadruped_adjustable_leg_controller/CMakeLists.txt
robot_description/arcdog_adjustable_leg_description/config/highstep_phase_residual.yaml
robot_description/arcdog_adjustable_leg_description/config/highstep_phase_residual/reference_*.csv
robot_description/arcdog_adjustable_leg_description/config/highstep_phase_residual/residual_*.pt
对应 C++/Python contract tests
robot_lab 中仅新增 exact Student observation recorder、phase-residual dataset/training、固定 rollout 和测试
```

禁止为了该实验重写 `StateRL`、覆盖 `model_name_policy2`、改变现有手动 play 脚本默认行为，或修改正式 highstep supervisor。

## 12. 失败与废弃规则

- 两轮固定场景 DAgger 后 Isaac hybrid rollout 仍未达到 15/15：分支停止，不进入 MuJoCo。
- Isaac 通过、MuJoCo 未达到 15/15：只允许在不改关节幅值和 gains 的前提下，预注册一次“分阶段时间缩放/相位等待”单变量修正；仍失败则废弃本分支。
- 任意真机前检查发现 joint order、目标限位、gains、reference SHA 或 FSM 原子切换不一致：fail-closed，禁止上电实验。
- 废弃只影响 `highstep_fixed_condition_phase_residual_20260717`，现有 v1.13.1 主线保持可继续使用。

## 13. 当前下一步

本 v0.3 只建立独立分支合同，不启动正式训练、评估、部署或服务。唯一下一步是：

```text
从现有 12 次 0.38 m Teacher 成功记录生成有界 reference candidates，
在同一 0.38 m 场景采集 exact Student obs + explicit phase + Teacher target，
训练 bounded residual MLP 并完成固定 Isaac hybrid rollout，
冻结 reference/model manifest，
再实现独立 StateHighstepPhaseResidual。
```
