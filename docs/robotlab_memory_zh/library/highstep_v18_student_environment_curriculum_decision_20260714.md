# Highstep v1.8 Student 环境课程路线决策（2026-07-14）

状态：用户正式批准；只读路线记忆。

## 决策

- 唯一 Teacher 继续使用 `model_172300.pt`，SHA256 `dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35`。该 Teacher 已在随机化、扰动和 curriculum 下训练，不重训 Teacher。
- v1.7.1 E700 的 checkpoint、完整 optimizer、W&B、诊断、`latent_observability_isolation_required` 与 stopped-by-gate 事实永久保留，不删除、不覆盖、不改判。
- v1.7.1 的控制变量错误是：路线名称宣称 historical 0707，但 Student 从 update 0 使用的实际 env 快照 SHA 为 `61d706...`，而 0707 deployed Student env SHA 是 `f83b0e...`。
- 因此 v1.7.1 只证明“新 Teacher + 完整当前 Student 环境从 update 0”没有复现 0707，不能证明 0707 基础环境中的 Stage-2 蒸馏失败。
- v1.8 唯一路线是：`0707 Student 环境 bootstrap → 同一 Student 和完整 optimizer 切换到当前完整鲁棒环境续训`。
- Stage A 只能证明基础动作 bootstrap，不具备真机候选资格；最终必须通过 Stage B 完整鲁棒环境 directional 门禁和用户视觉复核。

## 单一训练机制变量

| 字段 | v1.7.1 接受基线 | v1.8 批准值 | 是否变化 | 证据 |
|---|---|---|---|---|
| `student_environment_schedule` | `robust_from_update_zero` | `exact_0707_bootstrap_then_current_robust` | 是（唯一变量） | 两份保存 env.yaml 的 SHA 和 72 个叶级差异 |
| Teacher | `model_172300` | 相同 | 否 | Teacher SHA |
| fresh Student 初始化 | `model_172300` | 相同 | 否 | 双 checkpoint binding |
| warmup/target/phase/rear scale | `1400/pre-prior/2.0/1.5` | 相同 | 否 | v1.7.1 prereg + v1.8 prereg |
| loss/LR/VAE epochs/冻结范围 | v1.7.1 | 相同 | 否 | tensor/smoke manifest |
| network/reward/action contract | v1.7.1 | 相同 | 否 | profile contract audit |

## 冻结 profile

- Stage A Student：`2026-07-05_00-13-46/params/env.yaml`，SHA256 `f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636`。
- Stage B Student：v1.7.1 E700 `params/env.yaml`，SHA256 `61d70655405a49ad8fd72377aed3e193b0d8435898ca1fd39316d08f1989a729`。
- Teacher bootstrap：`2026-07-04_01-45-21/params/env.yaml`，SHA256 `25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7`；必须保留 action prior。

## 不可扩大解释

Stage A 失败只否决本轮环境课程路线；不得重训 Teacher、修改 reward/prior/网络/action contract，亦不得把 v1.7.1 或 v1.8 的一个 Student 失败扩大为 Teacher 无效。Stage B 数值通过后仍只能进入 `student_directional_candidate_pending_user_visual_review`，禁止自动真机部署。

正式 authority：`highstep_student_recovery_spec_20260712.md` v1.8，SHA256 `e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4`。
