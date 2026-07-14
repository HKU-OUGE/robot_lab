# Highstep v1.9 oracle prior-delta 路线决策记忆

- v1.8 已在 E1400 按预注册封顶，终态为 `stopped_by_gate`；不得延长 pre-prior estimator 路线。
- 唯一 Teacher 仍是 `model_172300.pt`，SHA256=`dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`。
- v1.8 最佳 Student 固定为 E1400 `model_1394.pt`，SHA256=`92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9`；其 core9 为 full/rear=`5/9`。
- 下一步不是继续训练 estimator，而是同 checkpoint、同 core9 的只读 oracle prior-delta A/B。oracle 只把 Teacher `post-prior - pre-prior` 加到 Student 最后四维；前十二维不得改变。
- 只有 oracle 达到预注册恢复门禁且严格优于 control，才允许从最佳 Student weights-only、fresh optimizer 启动零初始化 `570→4` blind prior residual head；只训练该 head，最终仍为 16 维动作。
- 若 oracle 不恢复，禁止训练 residual head；该因果分支以 `oracle_prior_delta_not_recovered` 终止。
- 禁止同时改 Teacher、reward、DR、history、全 actor、action scale、joint clip、部署输入或真机 gains。

正式 authority 以 `highstep_student_recovery_spec_20260712.md` v1.9 和只读 preregistration 的 SHA 为准，本文件只做路线记忆。
