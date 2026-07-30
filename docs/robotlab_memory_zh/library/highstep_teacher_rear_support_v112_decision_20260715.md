# Highstep v1.12 后腿支撑动作合同路线决策

- 日期：2026-07-15
- 状态：用户已批准并要求直接启动长训
- workflow：`highstep_teacher_rear_support_v112_20260715`
- 正式 authority：`highstep_student_recovery_spec_20260712.md` 顶部 v1.12

## 1. 为什么现在回到 Teacher 动作

v1.11 对旧 Teacher `model_151399` 和新 Teacher `model_172300` 完成了 204 场零训练、同 snapshot 鲁棒性 A/B。冻结决策文件是：

`/home/lxq/Softwares/robot_lab/tmp/highstep_teacher_robustness_ab_20260715/manifests/teacher_robustness_ab_decision.json`

结论为 `new_teacher_robustness_not_clearly_improved`。新 Teacher 的 nominal 和组合压力测试通过，但在部分 inward/impulse 场次出现失败，没有同时满足预注册的“内收能力提升一级、抗扰动能力提升一级”标准。这不否定 model_172300 已学会仿真上台；它说明当前最缺的不是继续微调 box 响应，而是后腿支撑动作本身在扰动下缺少结构稳定性。

## 2. 真机—仿真证据边界

只读输入：

- 0707 真机视频：`/home/lxq/Videos/histep_real_robot_test_0707.mp4`
- 0707 较完整视频：`/home/lxq/Videos/histep_real_robot_test_2026_07_07-00_28_27.mp4`
- 对应真机 rosbag：`/home/lxq/log/rosbag_2026_0707/rosbag2_2026_07_07-00_52_07` 与 `.../rosbag2_2026_07_07-00_28_27`
- 校准后的 MuJoCo 视频：`/home/lxq/Videos/Kazam_screencast_00153.mp4`
- 校准后的 MuJoCo rosbag：`/home/lxq/colcon_ws/rosbag2_box_calibrated_2026_07_15-07_33_22`
- SWAP 只读材料：`/home/lxq/Softwares/robot_lab/tmp/swap_parkour_read/`

已有时间对齐分析显示：

1. 真机 approach 时后腿仍基本对称：rear width 约 `0.424 m`，左右 thigh 差约 `0.007 rad`、calf 差约 `0.024 rad`。
2. 前支撑建立至第一后腿阶段，真机后足前后差曾扩大到约 `0.378 m`，后足横向中心偏移约 `0.121 m`；后续左右 rear calf 差达到约 `1.54–1.63 rad`，rear width 最低约 `0.149 m`、rear min-abs-y 最低约 `0.022 m`。
3. MuJoCo 也会出现一定左右时序差，但理想接触、模型与较少未建模扰动使它仍能上台。box 闭环减速后，00153 仍可完成约 35 cm 上台，只在右后腿接续时略有卡顿；因此 box 速度/响应不是解释巨大 sim-to-real 差异的首要变量。
4. 不再使用“前腿弹射/腾空”作为事实前提。现有视频不能严格证明该动作；本轮前腿只作为阶段触发，不作为优先重塑对象。

## 3. 单一变量选择

| 字段 | 接受的 model_172300 路线 | v1.12 | 是否改变 |
|---|---|---|---|
| 起点/optimizer/schedule | model_172300 full checkpoint | 完全相同 | 否 |
| Robust env/DR/外力/curriculum | 原值 | 原值 | 否 |
| action prior/网络/obs/action | 原值 | 原值 | 否 |
| box response/action scale/clip/gains | 原值 | 原值 | 否 |
| 前腿 reward 与其它 reward | 原值 | 原值 | 否 |
| 后腿动作目标 | 后腿绝对功率 + 左右平均姿态 | 双支撑锚定、同步抬身、镜像动作、安全顺序上台 | **是，唯一变量** |
| 被替换 reward 总权重 | `1.10 + 1.20 = 2.30` | `2.30` | 否 |

旧绝对功率奖励允许一条后腿用更大功率补偿另一条腿，旧平均姿态奖励会掩盖 RL/RR 的相反误差。v1.12 把两项整体替换成一个有阶段语义的 `rear_support_motion_contract`，而不是继续叠加 reward。

## 4. 新动作合同

1. 前足还未稳定 commit、第一条后腿尚未离地：两后足保持接触与世界系低滑移；后足相对 body 中线镜像；hip 采用反号镜像，thigh/calf 采用同号镜像；左右接触力尽量平衡；两腿共同抬升 body，并限制 roll/roll-rate/yaw-rate。
2. 第一后腿开始上台：停止要求所有后关节同相，允许顺序动作；保留支撑后足接触、宽度/min-abs-y 和机身姿态约束。
3. 第二后腿上台及 hold：保留中心线安全余量和稳定姿态。
4. 不要求“任何时刻都绝对不内收”。为跨越台沿允许必要的足端路径，但不得进入危险的过窄/穿越中心线状态。

## 5. 长训与边界

- 从受保护 model_172300 完整恢复，追加 6000 effective updates，4096 env，seed 42，每 100 updates 保存。
- 训练中不插入抢 GPU 的评估，也不因中间 loss 或单场行为停止；基础设施异常由独立 supervisor 自动从最近完整 checkpoint 恢复。
- 长训完成后统一评估 E500/E1000/E2000/E4000/E6000，选择后腿动作与完整上台行为共同最佳点。
- 本轮不自动蒸馏 Student、不发布真机候选、不自动部署。下一步必须先复核后腿动作证据。

