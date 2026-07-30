# Highstep fixed-motion one-shot deployment contract (2026-07-18)

本文件是独立部署适配分支，不修改或取代正式 v1.13.1，也不授权真机实验。

## 唯一语义变化

policy2 从“实时摇杆幅度推进 command/phase”改为“从 Fixed Stand 合法进入 policy2 后，一次性内部执行固定 138 个推理步”。摇杆轴不再参与动作时钟，只保留 FSM 触发与人工急停。

固定序列（推理周期 `dt=0.02 s`）：

- step 0–20：`vx=0`
- step 21–79：`vx=0.7200000286`
- step 80–137：`vx=0`

phase 仍按已经通过 Isaac 15/15 门禁的 59 个正向步推进；phase-only box prior、570+8 输入、raw 16 维输出、production action scale/default/clip/physical clamp 全部不变。

## 触发与安全

- 唯一入口：机器人已在 `FIXEDSTANDADJUSTABLELEG`，Fixed Stand 收敛门已满足，随后按 `RB+LB+Y`。
- policy2 每次进入只接受一次动作；运行中或完成后再次发送同一触发必须拒绝，必须先人工切回 Fixed Stand 才能重新进入。
- 进入时必须通过有限值、关节姿态/速度门；否则 fail-closed 到 Passive。
- `RB+B` 人工急停、现有 estimator 姿态安全、完整帧 NaN/限位/力矩检查和底层通信安全保持有效。
- step 137 后保持最终合法输出并等待人工切换 Fixed Stand，不自动真机部署。

## 验证顺序

1. 静态、适配器和编译 smoke。
2. 使用同一 checkpoint、phase-only prior、raw-action production mapping，在 Isaac 录制一条新 one-shot 视频。
3. 使用同一部署控制器在 MuJoCo 录制一条新 one-shot 视频和 rosbag。
4. 只先交付单条视频、日志和 manifest 给用户；未经视觉批准不运行 15 轮，不进行真机实验。

