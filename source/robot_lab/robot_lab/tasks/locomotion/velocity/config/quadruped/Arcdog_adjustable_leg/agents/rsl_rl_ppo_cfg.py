# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import field
from rsl_rl.algorithms import PPO
# 【关键修复 1】：导入 rsl_rl 的 runner 模块
import rsl_rl.runners.on_policy_runner as on_policy_runner


@configclass
class ArclabArcdogAdjustableLegRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 17000
    save_interval = 100
    experiment_name = "arclab_arcdog_adjustable_leg_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ArclabArcdogAdjustableLegFlatPPORunnerCfg(ArclabArcdogAdjustableLegRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "arclab_arcdog_adjustable_leg_flat"

# ---------------------------------------------------------
# 1. 声明 VAE 专属的网络配置
# ---------------------------------------------------------
@configclass
class VAEActorCriticCfg(RslRlPpoActorCriticCfg):
    # 注意：这里的 class_name 必须能让 RSL-RL 找到你的自定义类。
    # 如果你的算法逻辑文件叫 vae_algo.py，通常需要写完整的模块路径，
    # 例如: class_name = "your_package.vae_algo:VAEActorCritic"
    # 如果你已经在主脚本里 import 了 VAEActorCritic，保持原样即可。
    class_name: str = "VAEActorCritic"
    distill_stage: int = 1  # 【新增】配置项：1 为 Teacher, 2 为 Student
    vae_latent_dim: int = 64
    vae_hidden_dims: list = field(default_factory=lambda: [256, 128])
    student_actor_warmup_updates: int = 800
    student_vae_epochs: int = 4
    student_low_speed_threshold: float = 0.10
    student_prior_fade_speed: float = 0.45
    student_vel_loss_coef: float = 10.0
    student_latent_loss_coef: float = 50.0
    student_teacher_action_loss_coef: float = 20.0
    student_prior_box_loss_coef: float = 5.0
    student_recon_loss_coef: float = 0.5
    student_kl_loss_coef: float = 0.1
    student_post_prior_mode: str = "lateral"

# ---------------------------------------------------------
# 2. 声明 VAE 专属的算法配置
# ---------------------------------------------------------
@configclass
class VAEPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "VAEPPO"

# ---------------------------------------------------------
# 3. 你的主 Runner 配置
# ---------------------------------------------------------
@configclass
class ArclabArcdogAdjustableLegBodyflatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 17000
    save_interval = 100
    experiment_name = "arclab_arcdog_adjustable_leg_bodyflat_vae" # 改个名字区分一下
    empirical_normalization = False

    # 【新增】：告诉 Runner 我们有三个观测组
    obs_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"]
    }

    # 【修改】：使用 VAE 的 Policy 配置
    policy = VAEActorCriticCfg(
        distill_stage=1,  # <========= 【核心修改】：1 代表第一阶段 (训练 Teacher)
        # distill_stage=2,  # <========= 【核心修改】：2 代表第二阶段 (训练 Student)
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # VAE 专属参数
        vae_latent_dim=64,
        vae_hidden_dims=[256, 128],
    )

    # 【修改】：使用 VAE 的 Algorithm 配置
    algorithm = VAEPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # ==========================================
    # 【核心修复】：利用 __post_init__ 自动动态修改实验名称
    # 保证 Teacher 和 Student 的权重、TensorBoard 日志绝对隔离！
    # ==========================================
    def __post_init__(self):
        # 如果父类有 __post_init__，先调用它保证基础初始化完整
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        # 自动读取 policy 中的 distill_stage，并追加到 experiment_name 后缀
        if self.policy.distill_stage == 1:
            self.experiment_name += "_Teacher"
        elif self.policy.distill_stage == 2:
            self.experiment_name += "_Student"

    # ==========================================
        # 【新增】：实时 Debug 面板，证明配置修改有效
        # ==========================================
        print("\n" + "="*65)
        print("====== [REAL-TIME CONFIG DEBUG PANEL] ======")
        print(f"  [Distill Stage] : Stage {self.policy.distill_stage} ({'Teacher' if self.policy.distill_stage==1 else 'Student'})")
        print(f"  [Target Log Dir]: logs/rsl_rl/{self.experiment_name}")
        if self.policy.distill_stage == 2:
            print("  [✅ 状态确认]   : 当前为 Student 阶段，日志已隔离至 _Student 目录。")
            print("  [✅ 路径机制]   : 显式绝对路径加载机制已就绪，可直接跨目录读取 Teacher 权重！")
        print("="*65 + "\n")


@configclass
class ArclabArcdogAdjustableLegBodyflatStudentNoPriorPPORunnerCfg(ArclabArcdogAdjustableLegBodyflatPPORunnerCfg):
    experiment_name = "arclab_arcdog_adjustable_leg_bodyflat_vae_student_no_prior_quick_deploy"
    max_iterations = 1800
    save_interval = 50

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_actor_warmup_updates = 800
        self.policy.student_vae_epochs = 4
        self.policy.student_low_speed_threshold = 0.10
        self.policy.student_prior_fade_speed = 0.45
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4500
    save_interval = 100
    experiment_name = "arclab_arcdog_adjustable_leg_highstep_vae"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"],
    }

    policy = VAEActorCriticCfg(
        distill_stage=1,
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        vae_latent_dim=64,
        vae_hidden_dims=[256, 128],
    )

    algorithm = VAEPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if self.policy.distill_stage == 1:
            self.experiment_name += "_Teacher"
        elif self.policy.distill_stage == 2:
            self.experiment_name += "_Student"

        print("\n" + "="*65)
        print("====== [HIGHSTEP CONFIG DEBUG PANEL] ======")
        print(f"  [Distill Stage] : Stage {self.policy.distill_stage} ({'Teacher' if self.policy.distill_stage==1 else 'Student'})")
        print(f"  [Target Log Dir]: logs/rsl_rl/{self.experiment_name}")
        print("="*65 + "\n")


@configclass
class ArclabArcdogAdjustableLegHighstepStudentNoPriorPPORunnerCfg(ArclabArcdogAdjustableLegHighstepPPORunnerCfg):
    experiment_name = "arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior"
    max_iterations = 6000
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_actor_warmup_updates = 1200
        self.policy.student_vae_epochs = 4
        self.policy.student_low_speed_threshold = 0.10
        self.policy.student_prior_fade_speed = 0.45
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        self.policy.student_post_prior_mode = "highstep"
        super().__post_init__()

# @configclass
# class ArclabArcdogAdjustableLegBodyflatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 24
#     max_iterations = 17000
#     save_interval = 100
#     experiment_name = "arclab_arcdog_adjustable_leg_bodyflat"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[512, 256, 128],
#         critic_hidden_dims=[512, 256, 128],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.01,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         learning_rate=1.0e-3,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
