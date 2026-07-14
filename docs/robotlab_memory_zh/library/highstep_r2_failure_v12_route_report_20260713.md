# Highstep R2 失败结论与 v1.2-R3 路线依据（2026-07-13）

## 结论

R2 已按 v1.1.1 在 100 effective updates 永久失败。当前行为最佳仍为 B500；R2-50 不是候选，但它把残余失败收缩成了可复现的 right-offset 单一场景族，为 v1.2 的有限范围修复提供了证据。

| checkpoint | valid | full | rear hold | front top | no severe |
|---|---:|---:|---:|---:|---:|
| B500 | 9 | 6 | 6 | 8 | 9 |
| R2-50 | 9 | 6 | 6 | 6 | 9 |
| R2-100 | 9 | 2 | 2 | 2 | 9 |

R2-50 的 nominal 3/3、left-offset 3/3 成功，right-offset 0/3；R2-100 只剩 nominal 2/3 成功。R2-50 与 B500 的成功集合不同，但都没有超过 6/9。

## 可证实的训练教训

1. R2 latent MSE 从约 0.2005 到 0.1975，100 次更新没有实质下降；不能把 latent 联合蒸馏继续当作有效机制。
2. post-prior action MSE 降至约 0.0615、训练 reward 升至约 45，但 core9 从 6/9 降到 2/9；loss/reward 不能代替完整上台门禁。
3. R2-100 成功场景的第一后足上台提前到约 104--105 步，roll rate 同时升高；学到的是更激进、更窄的动作模式，而不是更稳健的策略。
4. R2-100 相对 B500 的 estimator `fc_mu.weight` 相对 L2 漂移约 5.53%，四个 box 输出行 weight 漂移约 11.46%；这种全局漂移与六个 offset 场景前足支撑消失同时出现。
5. R2-50 已将失败严格集中到 right-offset，说明下一步应停止修改 estimator，保留其六个成功场景，只在已有 actor head 中学习横向偏置状态的 Teacher 最终动作。

## v1.2-R3 的唯一变量

- 起点/行为锚点：R2-50 `model_49.pt`；
- 冻结：完整 estimator、actor body、critic、Student privileged encoder、action std、Teacher 和 anchor；
- 训练：现有 `actor.6` 的完整 weight/bias 两个 tensor；
- target：同状态 `model_172300` Teacher post-prior 最终 16 维动作；
- sampling：不增加随机化，使用现有 on-policy rollout；从已有 critic height scan 得到横向不对称门控；
- loss：偏置状态 Teacher action MSE，加四倍居中状态 R2-50 action anchor；
- 最大预算：100 effective updates；10 probe3、25/50/100 core9；看到结果后不得更改门禁。

## 自动化故障教训

- 确定性 rollout 仍需为 RSL logger 初始化只读 distribution；smoke 必须真正跨过第一次日志输出。
- 进程扫描只能匹配 argv 的真实脚本 token，不能匹配 shell 命令正文。
- 按绝对路径运行 `tools/*.py` 时必须在隔离环境验证包导入。
- 预注册保存点必须及时 core9；R2-50 的完整矩阵不能拖到 R2-100 失败后才补。
- systemd 必须自动恢复明确的基础设施故障，同时对行为门禁和绑定错误保持永久 fail-closed；任何停止都必须写 handoff，不能等待用户主动询问。

## 证据绑定

- v1.2 spec SHA256：`053da1c6d30f9d5ed9aa4c2d5bedbab8d98130edb8a60f8e2657d4a8d0d99352`
- B500 core9 manifest SHA256：`e21cfcfe3b9dbb5984015c81e042fb279f3ba4dae7e1ff4e5a8f010a75ba1e73`
- R2-50 checkpoint SHA256：`e875424ed69eaaa474a9ac6849a6066dd264ce0eaf04ad63a418c3af1d66918c`
- R2-50 core9 manifest SHA256：`b86c62d9c0fe95d17fa723fede4bbdfb1f42dcdc7afc5a7545dd97bfcde0da20`
- R2-100 checkpoint SHA256：`06b0f4a98b89dc00beb63367039d404d5fb6cd2dd181b92242e62b0fa94da7bb`
- R2-100 core9 manifest SHA256：`b80a30105a56f1c350318595edbffb037bb353a54df59d2631ddcdb138381bca`
- R2 terminal handoff SHA256：`019c77394c34ada81ad3e9c0f4a46479ae1b347b93aabf1b2867cdc693c32e81`
