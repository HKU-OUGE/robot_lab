# v1.11 Teacher A/B Git 快照

这些文件是原工作站运行时 authority 的只读副本，用于代码审查和跨机器分析。

- `preregistration_v111.json`：不可事后修改的实验矩阵与门禁。
- `runtime_code_rebinding.json`：启动时绑定的 executor/supervisor/monitor/test SHA。
- `state_snapshot_20260715_053600.json`：发布期间某一时刻的 state；不是实时 heartbeat。
- `state_snapshot_20260715_055016.json`：提交前第二次只读 state，证明 Teacher A 已完成且流程已进入 Teacher B。

原始运行时目录为 `tmp/highstep_teacher_robustness_ab_20260715/`，它被 Git 忽略。新机器不得用这些副本伪造活动 lock、heartbeat 或 resume authority。

源文件 SHA256：

```text
1b2bd47960d9111cf1b98676ece01524d89a889fc4b55cfc1ed08174782f2a55  preregistration_v111.json
8245b6f1a41916a93741bfab0d1a101608a642d1e8f899810e5d617a9a2353f5  runtime_code_rebinding.json
9e8d980e9220494a79e0aba91895d0dd63a8273cdbda94388d56229a84926ba1  state_snapshot_20260715_053600.json
bf084a93f8b3fdbf53cb4bf56ee025cd33d4270ce7c1c927d313efd61f727e9b  state_snapshot_20260715_055016.json
```
