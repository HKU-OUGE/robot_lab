# B300 E1600 Student：MuJoCo sim-to-sim → 异机 sim-to-real 交接记忆

更新时间：2026-07-19，Asia/Hong_Kong  
用途：把本机已经验证过的 MuJoCo 经验、精确策略绑定和部署注意事项交给另一台真机电脑上的 Codex。  
性质：只读交接证据，不是自动部署授权，也不是要求另一台电脑照抄本机仿真参数。

## 0. 先读结论

1. 当前最适合带到另一台电脑复现的策略是 **E1600 Student**，不是当前尚未获得部署结论的新 A+B 训练路线。
2. E1600 已在本机 MuJoCo 的 **0.350 m 实体高台**上表现出明确的完整上台能力；这证明策略动作能力存在，是好消息。
3. 这仍不能直接证明真机可用。00170 中存在三个重要有利条件：
   - 人工先用 policy1 / Fixed Stand 调整到了靠近平台、甚至前腿已接近支撑的位置；
   - 使用平滑模拟摇杆，policy2 的正向命令峰值约 0.43–0.50，而 Isaac 键盘测试是阶跃到 0.72；
   - MuJoCo 的关节止挡、接触、box 执行器响应和本机仿真 gains 与真机不同。
4. **绝对不要把本机 MuJoCo 的 `config.yaml` 整份覆盖真机配置。** 真机 gains、电机限位、硬件接口和安全参数必须保留，并由另一台电脑上的 Codex逐项审计。
5. 本次没有执行 git push，没有修改另一台电脑，也没有授权自动真机启动。

## 1. 当前流程状态与防止拿错模型

2026-07-19 现场 authority 审计：

```text
active workflow = highstep_b300_critical_transition_balanced_diagonal_fresh_7400_20260719
phase           = manual_visual_review_prepared
status          = stopped_by_user_for_manual_visual_review
active train    = none
active eval     = none
supervisor      = none
```

这条较新的 A+B 路线当前只是停在人工复核边界，**没有取代 E1600 的 MuJoCo 证据，也不得在另一台电脑上被自动选作 policy2**。

本交接包只绑定下面的 E1600：

```text
训练路线：highstep_b300_diagonal_imitation_fresh_7400_20260719
相对蒸馏 update：1600
训练保存名：model_175098.pt
```

其中 `175098` 是 runner 的绝对编号；对应本路线从 Teacher fresh 初始化后的相对 update 是 1600。

## 2. 精确策略与 lineage

### 2.1 真正给部署控制器加载的 TorchScript

```text
源文件：
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_diagonal_imitation_fresh_7400_Student/2026-07-19_02-26-51_highstep_b300_diagonal_imitation_fresh_7400_student_from_173499/exported/policy_student.pt

部署文件名：
policy_student_b300_diagonal_E1600_model_175098.pt

SHA256：
878a265de70644d4bdb997520004147c6d8f80420880627b9152d326fcad071b

输入：[1, 570] float tensor
输出：[1, 16] float tensor
CPU finite-output smoke：通过
```

压缩包中的位置：

```text
policies/policy_student_b300_diagonal_E1600_model_175098.pt
```

### 2.2 训练完整 checkpoint（不能直接当部署 TorchScript）

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_b300_diagonal_imitation_fresh_7400_Student/2026-07-19_02-26-51_highstep_b300_diagonal_imitation_fresh_7400_student_from_173499/model_175098.pt

SHA256：
f2855f996aa43af710253b79405503e5800db068a4a09336c17f32adbcbcced7
```

压缩包中的位置：

```text
policies/E1600_full_checkpoint_model_175098.pt
```

它用于 lineage、恢复和重新导出，不应填入真机 `model_name_policy2`。

### 2.3 固定 Teacher lineage（仅审计，不给真机加载）

```text
/home/lxq/Softwares/robot_lab/logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt

SHA256：
d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431
```

Teacher 需要 privileged observation，不是 570→16 的部署策略。为避免误用，本压缩包没有复制 Teacher 二进制，只记录其 lineage 和哈希。

## 3. E1600 的生产部署合同

### 3.1 输入

每帧 57 维，10 帧 history，总计 570 维：

```text
ang_vel + gravity_vec + commands + dof_pos + dof_vel + last_actions
history indices = [0,1,2,3,4,5,6,7,8,9]
clip_obs = 60
```

不得增加 phase、raycast、地形扫描、世界坐标或其它传感器输入。

### 3.2 输出和关节顺序

策略输出是 16 维普通 Student raw action。控制器内部顺序为：

```text
[FL_hip, FR_hip, RL_hip, RR_hip,
 FL_thigh, FR_thigh, RL_thigh, RR_thigh,
 FL_calf, FR_calf, RL_calf, RR_calf,
 FL_box, FR_box, RL_box, RR_box]
```

注意：YAML 为便于填写，常以每条腿 `[hip, thigh, calf, box]` 组织；`StateRL` 会转置成上面的控制器顺序。另一台电脑必须验证这一步，不能按 YAML 表面顺序直接索引 policy 输出。

本机 00170 使用的动作合同：

```text
action_scale：12 个转动关节 0.1；4 个 box 关节 0.02
default_dof_pos：hip=0，thigh=0.7，calf=-1.3，box=0.03
box 最终目标范围：[0.00, 0.06] m
clip_actions：[-60, 60]（16维）
```

输出按现有 production 通道执行：raw action → action clamp → action_scale + default_dof_pos → 物理目标 clamp。不得把 raw action 直接当关节角。

### 3.3 明确禁用的旧实验路径

00170 的普通 Student policy2 配置是：

```text
policy2_fixed_motion_enabled = false
policy2_one_shot_enabled = false
teacher_virtual_terrain_enabled = false
rear_hip_guard_enabled = false
```

因此它没有：固定动作回放、一次触发、8维 phase、虚拟地形 Teacher、隐藏 action prior 或 rear-hip guard。另一台电脑不得把这些历史实验路径误开。

### 3.4 policy1 必须保留

平地 policy1 与 highstep policy2 是两个独立模型键：

```text
model_name         = 平地 policy1
model_name_policy2 = 本交接包 E1600 Student
```

只替换 policy2，禁止覆盖或删除 policy1。

## 4. 00170 MuJoCo 证据

### 4.1 原始证据与哈希

视频：

```text
/home/lxq/Videos/Kazam_screencast_00170.mp4
SHA256=de54b411875d271e6ef0333f345ddf4fee24438a3fc8fe38d796f451ac6043a0
duration=117.133 s
```

日志目录：

```text
/home/lxq/Softwares/robot_lab/tmp/highstep_student_mujoco_E1600_20260719/manual/20260719_050012
```

关键文件：

```text
controller.log
SHA256=da6ffc12eeba6bd9ffc92b156885ee01ec61d639d5bb39126e0a08842b7c296c

mujoco.log
SHA256=bd4c824d12aae2c105a9ee6b2df6d866046c684661a093e0e2c172c667f22d30

rosbag/metadata.yaml
SHA256=1e0a5e9c8e0df0a4e6fca2513c702cf5124e23ea076d86fa28a1df8db51bec82

rosbag/rosbag_0.db3
SHA256=d6ac1987a311fb9cd75050d2865381d97307d0590ca02b37aa1410126221d191
```

视频 112 MiB、rosbag 941 MiB，没有放入轻量交接压缩包；本机原文件继续保留。

### 4.2 时间对齐

```text
video_s = ROS_unix_s - 1784408445.593463
```

该偏移已用 FSM 的 policy2 上升沿核对。视频中七段 policy2 约为：

```text
11.02–15.34 s
30.90–34.23 s
45.65–49.63 s
54.49–58.27 s
67.93–72.35 s
82.38–89.03 s
98.67–103.45 s
```

至少四段清楚完成后腿上台；另外几段在进入 policy2 时已经处于平台上或前腿支撑较深的位置，因此不能把七段都当作独立 nominal 成功率。

### 4.3 客观解释：为什么 MuJoCo 看起来明显好于 Isaac 手动 play

最有解释力的不是“MuJoCo 给了无限力矩”，而是以下组合：

1. **进入状态不同。** 00170 多次由 policy1 / Fixed Stand 先靠近和调姿，再切 policy2；Isaac 00169 多从统一、对称、距边缘约 0.55 m 的 reset 开始。前腿已经靠近或支撑时，Student 不必自己解决完整 approach。
2. **命令波形不同。** MuJoCo 是平滑模拟摇杆，第一段峰值约 0.434，后续不超过 0.5；Isaac 键盘是 0→0.72 的突变。历史观测包含 commands 和 last action，因此命令轨迹会改变动作相位触发。
3. **接触与止挡不同。** MuJoCo 的关节硬止挡、碰撞求解和平台边缘接触可提供额外机械约束；它可能把擦边动作“导向”平台，而 Isaac 或真机不一定一样。
4. **box 执行器在仿真中较理想。** 本机 MuJoCo 使用 box 目标范围 `[0,0.06] m` 和 `0.080 m/s` 目标速率限制，但仿真 `kp=8000,kd=1`，与真机参考中的低增益 box 执行器完全不同。
5. **最终目标 clamp 很活跃。** 既有分析中，policy2 推理帧约有 34.6%–77.9% 出现至少一个目标被 physical clamp，常见于 FL hip、FR calf、RL box。MuJoCo 的机械止挡与控制器 clamp 共同改变了实际动作。

另一方面，MuJoCo 的转动关节电机并非无限强：

```text
hip / thigh ctrlrange ≈ ±23.7 Nm
calf ctrlrange        ≈ ±45.43 Nm
```

它们没有高于 Isaac 常见的约 35 / 80 Nm。因此“单纯扭矩过大使动作虚假成功”不是首要解释。

平台几何与摩擦：

```text
高台高度：0.350 m
高台 geom friction：2.0
足端 geom priority=1，friction="1.0 0.04 0.01"
```

平台 `friction=2.0` 不能简单等价为所有接触都使用 2.0；MuJoCo 的 priority/contact 合成仍需看双方 geom。真机台面材料必须单独记录。

### 4.4 这条结果是好消息还是坏消息

结论是“**有条件的好消息**”：

- 好消息：E1600 的 570→16 Student 确实包含完整后腿上台动作，不是完全没学会。
- 风险：目前尚未证明它能从统一 nominal 起点、按真机实际摇杆映射与真实执行器动态稳定触发。
- 因此下一步应先让另一台电脑严格复现 00170 的代码与模型身份，再逐项把仿真假设替换成真机已有参数；不能靠直接改 gains 来追视频。

## 5. 本机两个仓库的真实 Git 状态

### 5.1 训练仓库 robot_lab

```text
path   = /home/lxq/Softwares/robot_lab
branch = dev_lxq_new
HEAD   = 7c6557f31a81c90d9bc80bf7437c8a18f62f0fac
remote = git@github.com:HKU-OUGE/robot_lab.git
state  = dirty，近期训练代码尚未 push
```

### 5.2 部署仓库 quadruped_control_ros2

```text
path   = /home/lxq/colcon_ws/src/quadruped_control_ros2
branch = dev_new_arcdog
HEAD   = ff5e3131e9496e536d35c1df5d0c01c7de4a3a98
remote = https://github.com/ruihuang1124/quadruped_control_ros2.git
state  = dirty，00170 相关部署/MuJoCo改动尚未 push
tracked diff SHA256 = fcc3de54db7c26e66b93aef6b1e00ab77107103c34dcf8723f0c47de198e4d4c
```

所以“两个电脑都在 `dev_new_arcdog`”只适用于部署仓库；训练仓库本机不是该分支。另一台电脑 Codex必须先确认自己正在审计哪一个仓库。

E1600 TorchScript 在部署仓库中是 **untracked 文件**，单纯 `git pull` 不会得到它；必须从本压缩包复制并校验 SHA。

## 6. 00170 运行时绑定的部署文件

以下 SHA 是 00170 当时和现存部署 manifest 一致的参考：

```text
robot_description/arcdog_adjustable_leg_description/config/rl_policy/config.yaml
93e683e67f486c62292b7287dd2da9eafcb3c41c6ee9e7f1b8f5994fca45c211

robot_description/arcdog_adjustable_leg_description/config/robot_control.yaml
46d1bede521e5aec41d430a3930bbbd52b572a37ccd378c8f41ba20de5557925

rl_quadruped_adjustable_leg_controller/launch/mujoco_adjustable_leg.launch.py
648f9f062d132dbe9c89314c13e6611398d1dbd5a29e4f6d6b4441586e7f13ad

rl_quadruped_adjustable_leg_controller/src/FSM/StateRL.cpp
8170380a8e6db087ea595e34f260959fc6062da99cbc337073447e6f6853f24d

mujoco_simulator/src/adjustable_leg_mujoco_msg_handler.cpp
143d767dfd24bc5cbc402072b4b296749ce0921db855c77cfafe44467ee96b76

robot_description/arcdog_adjustable_leg_description/xml/arcdog_adjustable_leg.xml
cd0e2d2bbc486570dcf36bc3b840121012fd7ee5b3e2505e9652cf59b1746066

robot_description/arcdog_adjustable_leg_description/xml/scene.xml
f575cd4031b61e1e8279ddb8e16ed57baff10295a7769a7153357f94bef453e8
```

这些 SHA 用来判断代码是否一致，不代表真机端必须使用相同 `config.yaml`。

## 7. 真机端已知差异：必须保留并现场审计

用户此前给过一份机载参考配置，其中转动关节 gains 约为：

```text
kp: hip=50, thigh=55, calf=70
kd: hip=1.5, thigh=1.5, calf=2.5
box: kp=2, kd=3
```

而本机 00170 仿真配置为：

```text
kp: hip=45, thigh=50, calf=70, box=8000
kd: hip=1.5, thigh=1.5, calf=2.0, box=1.0
```

用户已明确说明真机 YAML 只作参考，实际测试数据/机载现状优先。因此另一台电脑 Codex应：

1. 备份并哈希真实机载 `config.yaml`、硬件描述和当前 policy1/policy2；
2. 不从本包覆盖 `rl_kp`、`rl_kd`、`torque_limits`、硬件限位、`tau_ff`；
3. 只在确认 570维输入、16维输出、joint order、scale、offset 和 clamp 合同一致后绑定新 policy2；
4. 对所有差异先输出表格和最小修改方案，等用户明确批准后再写入。

## 8. 推荐的跨电脑执行顺序

### 阶段 A：在 pull 之前，只读审计

1. 解压本包并运行 `sha256sum -c SHA256SUMS`。
2. 阅读 `README_FIRST_zh.md`、`manifest.json` 和 `PROMPT_FOR_REAL_ROBOT_CODEX.txt`。
3. 在另一台电脑执行：

```bash
cd <真机电脑的 quadruped_control_ros2 路径>
git branch --show-current
git rev-parse HEAD
git remote -v
git status --short
```

4. 找到并哈希实际真机：

```text
config/rl_policy/config.yaml
robot_control.yaml
StateRL.cpp / StateRL.h
real_robot_adjustable_leg.launch.py
硬件接口源码与真实robot description
当前 policy1 / policy2 文件
```

5. 比较本包的仿真参考配置与真机现状，但不修改、不 pull、不 build、不启动电机。
6. 向用户汇报：当前分支/commit、dirty 状态、参数差异、需要从本机 push 的最小代码文件、真机本地必须保留的参数。

### 阶段 B：用户在本机选定并 push 后

1. 用户提供明确的部署仓库 commit SHA。
2. 真机电脑 Codex确认工作树安全，禁止 `git reset --hard` 或覆盖本地真机参数。
3. 先 `git fetch` 并审查该 commit 的文件级 diff。
4. 若本地真机配置与远端冲突，先制作保留真机参数的移植补丁；不要整文件接受远端仿真配置。
5. 只有在用户确认 diff 后才 `git pull --ff-only` 或按明确方案移植。
6. 从本包复制 TorchScript 到 policy 目录，校验 SHA，再只修改 `model_name_policy2`；policy1 不变。
7. build 后先做模型身份、570→16、joint order、scale/offset/clamp 和无电机输出检查。
8. 先在另一台电脑 MuJoCo复现，保存视频、controller log、deployment manifest 和 rosbag。
9. MuJoCo复现通过后，再提出有人保护真机实验方案；禁止自动真机部署。

## 9. 真机前最低限度核对清单

```text
[ ] policy2 TorchScript SHA = 878a265d...071b
[ ] policy2 input = 570，output = 16
[ ] policy1 文件和绑定未变
[ ] history顺序、时间方向和last action语义一致
[ ] joint order转置已验证
[ ] action_scale / default_dof_pos / joint_pos.clip未被误改
[ ] 真机kp/kd/torque/tau_ff保持机载已验证值
[ ] physical box范围与真实机构一致
[ ] fixed-motion / one-shot / virtual-terrain / rear guard全关
[ ] 手柄满推实际model vx及符号已记录
[ ] 切policy2前站姿与距平台边缘已记录
[ ] 急停、passive、fixed stand和人工保护可用
[ ] rosbag与控制器日志已启动
```

## 10. 本压缩包内容和排除项

包含：

```text
README_FIRST_zh.md
PROMPT_FOR_REAL_ROBOT_CODEX.txt
manifest.json
SHA256SUMS
policies/E1600_full_checkpoint_model_175098.pt
policies/policy_student_b300_diagonal_E1600_model_175098.pt
training_reference/{agent.yaml,env.yaml,binding.json,runtime_state.json,schedule_manifest.json}
deployment_reference/E1600_deployment_manifest.json
deployment_reference/simulation_config_DO_NOT_OVERWRITE.yaml
deployment_reference/simulation_robot_control_DO_NOT_OVERWRITE.yaml
evidence/00170_controller.log
evidence/00170_rosbag_metadata.yaml
```

不包含：

- 112 MiB 视频和 941 MiB rosbag（二者本机原路径和 SHA 已记录）；
- 未提交源代码 patch（避免在另一台电脑误套用仿真代码）；
- Teacher 二进制（避免误当 570→16 Student 部署）；
- 当前尚未完成部署复核的新 A+B Student。

## 11. 最重要的禁止项

- 不得看到 MuJoCo 上台成功就直接启动真机电机。
- 不得把仿真 gains 覆盖到真机。
- 不得把训练 checkpoint 当 TorchScript。
- 不得删除/覆盖 policy1。
- 不得自动选择目录中“数字更大”的 checkpoint。
- 不得启用历史 fixed-motion、one-shot、phase-only 或 virtual-terrain 分支。
- 不得在未校验 branch/commit/dirty state 前 pull。
- 不得使用 `git reset --hard` 清理真机电脑的本地配置。

