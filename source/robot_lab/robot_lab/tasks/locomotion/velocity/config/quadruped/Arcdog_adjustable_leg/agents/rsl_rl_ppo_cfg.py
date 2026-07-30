# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import field
import os
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
    student_highstep_phase_loss_scale: float = 0.0
    student_highstep_rear_box_loss_scale: float = 0.0
    student_highstep_diagonal_action_loss_scale: float = 0.0
    student_critical_transition_balanced_sampling: bool = False
    student_critical_transition_window_radius: int = 10
    student_rl_preedge_sampling: bool = False
    student_rl_preedge_pre_steps: int = 30
    student_rl_preedge_post_steps: int = 10
    student_front_diagonal_action_loss_scale: float = 0.0
    student_rear_diagonal_action_loss_scale: float = 0.0
    student_highstep_rear_hip_loss_scale: float = 0.0
    student_highstep_rear_hip_min_abs: float = 0.0
    student_highstep_rear_hip_action_scale: float = 0.1
    student_actor_latent_clamp_backward: str = "hard"
    # Approved 2026-07-12 highstep Student recovery.  "none" preserves the
    # legacy distillation path for unrelated tasks; stage B is enabled only by
    # the dedicated highstep ActionScore Student config below.
    student_recovery_stage: str = "none"
    student_recovery_actor_lr: float = 1.0e-5
    student_recovery_epochs: int = 1
    student_recovery_source_checkpoint: str = ""
    student_recovery_source_sha256: str = ""
    student_recovery_teacher_checkpoint: str = ""
    student_recovery_teacher_sha256: str = ""
    student_recovery_r2_preregistration_path: str = ""
    student_recovery_r2_preregistration_sha256: str = ""
    student_recovery_r3_preregistration_path: str = ""
    student_recovery_r3_preregistration_sha256: str = ""
    student_recovery_v15_preregistration_path: str = ""
    student_recovery_v15_preregistration_sha256: str = ""
    student_recovery_0707_exact_preregistration_path: str = ""
    student_recovery_0707_exact_preregistration_sha256: str = ""
    student_recovery_historical_0707_exact_preregistration_path: str = ""
    student_recovery_historical_0707_exact_preregistration_sha256: str = ""
    student_recovery_be300_0707_preregistration_path: str = ""
    student_recovery_be300_0707_preregistration_sha256: str = ""
    student_recovery_b300_hybrid_preregistration_path: str = ""
    student_recovery_b300_hybrid_preregistration_sha256: str = ""
    student_recovery_v18_preregistration_path: str = ""
    student_recovery_v18_preregistration_sha256: str = ""

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


@configclass
class ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg(ArclabArcdogAdjustableLegHighstepPPORunnerCfg):
    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae"
    max_iterations = 12000
    save_interval = 100

    def __post_init__(self):
        self.algorithm.entropy_coef = 0.0015
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.006
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepRearSupportV112PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Long-run Teacher continuation for the v1.12 rear-support contract."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_rear_support_v112"
    max_iterations = 6000
    save_interval = 100


@configclass
class ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepRearSupportV112PPORunnerCfg
):
    """Isolated output namespace for the v1.12.1 FL placement experiment."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_rear_support_front_placement_v1121"
    )


@configclass
class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121PPORunnerCfg
):
    """Treatment namespace for the paired v1.12.3 reward-geometry experiment."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment"
    max_iterations = 300
    save_interval = 100


@configclass
class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123ControlPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepFrontGeometryV1123PPORunnerCfg
):
    """Control namespace for the paired v1.12.3 reward-geometry experiment."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_control"
    max_iterations = 100


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior"
    max_iterations = 2000
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        # Formal recovery spec B: reuse model_900's estimator and actor body,
        # and train only the final RL/RR hip rows against the independently
        # loaded model_172300 Teacher.  No warm-up or hand-written hip target.
        self.policy.student_recovery_stage = "B"
        self.policy.student_actor_warmup_updates = 0
        self.policy.student_recovery_actor_lr = 1.0e-5
        self.policy.student_recovery_epochs = 1
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_no_prior_Student/"
            "2026-07-12_04-41-42_robust_student_distill_20260712_044124/model_900.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "9bbd5b597d9c195ecf9afb141152b8f0749dc599a54868674b299107fb40a229"
        )
        self.policy.student_recovery_teacher_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-11_11-22-23/model_172300.pt"
        )
        self.policy.student_recovery_teacher_sha256 = (
            "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
        )
        self.policy.student_highstep_phase_loss_scale = 0.0
        self.policy.student_highstep_rear_box_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR2PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Dedicated immutable-v1.1.1 R2 Student runner; the archived Stage-B cfg is untouched."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_recovery_r2"
    max_iterations = 1000
    save_interval = 100
    obs_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"],
        "teacher_context": ["teacher_context"],
    }

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "R2"
        self.policy.student_recovery_r2_preregistration_path = (
            "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v11_20260712/"
            "r2_preregistration.json"
        )
        self.policy.student_recovery_r2_preregistration_sha256 = (
            "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
        )
        self.policy.student_actor_warmup_updates = 0
        self.policy.student_recovery_epochs = 1
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 4
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.max_grad_norm = 1.0
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR3PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Spec-locked v1.2 R3 Student runner with a frozen estimator and full action head."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_recovery_r3"
    max_iterations = 100
    save_interval = 25
    obs_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"],
        "teacher_context": ["teacher_context"],
    }

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "R3"
        self.policy.student_recovery_r3_preregistration_path = (
            "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v12_20260713/"
            "r3_preregistration.json"
        )
        self.policy.student_recovery_r3_preregistration_sha256 = (
            "13c184e4b6e2c3b514f52466dac1e27ff18256433716fd0c6b5d0890d93945e6"
        )
        self.policy.student_actor_warmup_updates = 0
        self.policy.student_recovery_epochs = 1
        self.algorithm.num_learning_epochs = 1
        self.algorithm.num_mini_batches = 4
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.max_grad_norm = 1.0
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Spec-locked v1.5 fresh Stage-2 distillation from model_172300."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_v15"
    max_iterations = 2500
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "V15"
        self.policy.student_recovery_v15_preregistration_path = os.environ.get(
            "HIGHSTEP_V15_PREREGISTRATION_PATH",
            "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v15_20260713/"
            "preregistration_v6.json",
        )
        # The preregistration binds this source file, so its own digest cannot
        # be embedded here without a hash cycle.  The supervisor supplies the
        # already-frozen digest; an absent/incorrect value fails closed in VAEPPO.
        self.policy.student_recovery_v15_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_V15_PREREGISTRATION_SHA256", ""
        )
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v15_20260713/"
            "source_model_172300_v3/model_172300.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
        )
        self.policy.student_recovery_teacher_checkpoint = self.policy.student_recovery_source_checkpoint
        self.policy.student_recovery_teacher_sha256 = self.policy.student_recovery_source_sha256
        self.policy.student_actor_warmup_updates = 1400
        self.policy.student_vae_epochs = 4
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_post_prior_mode = "highstep"
        self.policy.student_highstep_phase_loss_scale = 2.0
        self.policy.student_highstep_rear_box_loss_scale = 1.5
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPrior0707ExactPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Strict replay of the proven 0707 Stage-2 semantics with model_172300."""

    experiment_name = "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_0707_exact_new_teacher"
    max_iterations = 1200
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "0707_EXACT"
        self.policy.student_recovery_0707_exact_preregistration_path = os.environ.get(
            "HIGHSTEP_0707_EXACT_PREREGISTRATION_PATH", ""
        )
        self.policy.student_recovery_0707_exact_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_0707_EXACT_PREREGISTRATION_SHA256", ""
        )
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/tmp/highstep_0707_exact_new_teacher_20260713/"
            "source_model_172300/model_172300.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
        )
        self.policy.student_recovery_teacher_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-11_11-22-23/model_172300.pt"
        )
        self.policy.student_recovery_teacher_sha256 = self.policy.student_recovery_source_sha256
        self.policy.student_actor_warmup_updates = 1200
        self.policy.student_vae_epochs = 4
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        self.policy.student_post_prior_mode = "highstep"
        self.policy.student_highstep_phase_loss_scale = 0.0
        self.policy.student_highstep_rear_box_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorHistorical0707ExactPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """Evidence-frozen historical 0707 Stage-2 semantics with model_172300."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
        "historical_0707_exact_new_teacher"
    )
    max_iterations = 1400
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "HISTORICAL_0707_EXACT"
        self.policy.student_recovery_historical_0707_exact_preregistration_path = os.environ.get(
            "HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_PATH", ""
        )
        self.policy.student_recovery_historical_0707_exact_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_SHA256", ""
        )
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/tmp/"
            "highstep_historical_0707_exact_new_teacher_20260714/"
            "source_model_172300/model_172300.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
        )
        self.policy.student_recovery_teacher_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-11_11-22-23/model_172300.pt"
        )
        self.policy.student_recovery_teacher_sha256 = self.policy.student_recovery_source_sha256
        self.policy.student_actor_warmup_updates = 1400
        self.policy.student_vae_epochs = 4
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        self.policy.student_post_prior_mode = "highstep"
        self.policy.student_highstep_phase_loss_scale = 2.0
        self.policy.student_highstep_rear_box_loss_scale = 1.5
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """v1.13.1: B-E300 fresh Student with the proven 0707 Stage-2 contract."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_student_no_prior"
    )
    max_iterations = 4000
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        self.policy.student_recovery_stage = "BE300_0707"
        self.policy.student_recovery_be300_0707_preregistration_path = os.environ.get(
            "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH", ""
        )
        self.policy.student_recovery_be300_0707_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256", ""
        )
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/"
            "2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
        )
        self.policy.student_recovery_teacher_checkpoint = (
            self.policy.student_recovery_source_checkpoint
        )
        self.policy.student_recovery_teacher_sha256 = (
            self.policy.student_recovery_source_sha256
        )
        self.policy.student_actor_warmup_updates = 1400
        self.policy.student_vae_epochs = 4
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        self.policy.student_post_prior_mode = "highstep"
        self.policy.student_highstep_phase_loss_scale = 2.0
        self.policy.student_highstep_rear_box_loss_scale = 1.5
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
        super().__post_init__()


@configclass
class ArclabArcdogAdjustableLegHighstepFrontGeometryV114StudentNoPriorPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorPPORunnerCfg
):
    """v1.14: v1.13.1 contract with STE clamp backward as the sole change."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_front_geometry_v114_ste_student_no_prior"
    )
    max_iterations = 2500
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_actor_latent_clamp_backward = "straight_through"


@configclass
class ArclabArcdogAdjustableLegHighstepB3000707DerivedSingleRun7400PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorPPORunnerCfg
):
    """One uninterrupted 7400-update B300 -> 0707-derived Stage-2 run."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_b300_0707_derived_single_run_7400"
    )
    max_iterations = 7400
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_recovery_be300_0707_preregistration_path = os.environ.get(
            "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH", ""
        )
        self.policy.student_recovery_be300_0707_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256", ""
        )


@configclass
class ArclabArcdogAdjustableLegHighstepB300DiagonalImitationFresh7400PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepB3000707DerivedSingleRun7400PPORunnerCfg
):
    """Fresh B300 7400 run with the one preregistered diagonal loss scale."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_b300_diagonal_imitation_fresh_7400"
    )

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_highstep_diagonal_action_loss_scale = 1.0


@configclass
class ArclabArcdogAdjustableLegHighstepB300CriticalTransitionBalancedDiagonalFresh7400PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepB3000707DerivedSingleRun7400PPORunnerCfg
):
    """Fresh B300 7400 run with the approved transition sampler and phase loss."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_b300_critical_transition_balanced_"
        "diagonal_fresh_7400"
    )

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_highstep_diagonal_action_loss_scale = 0.0
        self.policy.student_critical_transition_balanced_sampling = True
        self.policy.student_critical_transition_window_radius = 10
        self.policy.student_front_diagonal_action_loss_scale = 2.0
        self.policy.student_rear_diagonal_action_loss_scale = 2.0


@configclass
class ArclabArcdogAdjustableLegHighstepB300RLPreEdgeContinuationE7700PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepB300CriticalTransitionBalancedDiagonalFresh7400PPORunnerCfg
):
    """Exact E5700 continuation with RL pre-edge sampling."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_b300_rl_preedge_continuation_e7700"
    )
    max_iterations = 2000
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_rl_preedge_sampling = True
        self.policy.student_rl_preedge_pre_steps = 30
        self.policy.student_rl_preedge_post_steps = 10


@configclass
class ArclabArcdogAdjustableLegHighstepB300CanonicalHybridStudentNoPriorPPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepFrontGeometryV114StudentNoPriorPPORunnerCfg
):
    """Independent canonical B300 mixed pre/post-prior latent distillation."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_b300_canonical_hybrid_student_no_prior"
    )
    max_iterations = 700
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_recovery_stage = "B300_CANONICAL_HYBRID"
        self.policy.student_recovery_b300_hybrid_preregistration_path = os.environ.get(
            "HIGHSTEP_B300_HYBRID_PREREGISTRATION_PATH", ""
        )
        self.policy.student_recovery_b300_hybrid_preregistration_sha256 = os.environ.get(
            "HIGHSTEP_B300_HYBRID_PREREGISTRATION_SHA256", ""
        )
        # The last four rows and estimator learn from the first effective update.
        self.policy.student_actor_warmup_updates = 0


@configclass
class ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18PPORunnerCfg(
    ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg
):
    """v1.8 two-stage environment curriculum with unchanged 0707 distillation contract."""

    experiment_name = (
        "arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
        "environment_curriculum_v18"
    )
    max_iterations = 2500
    save_interval = 100

    def __post_init__(self):
        self.policy.distill_stage = 2
        continuation_branch = os.environ.get("HIGHSTEP_E1400_CONTINUATION_BRANCH", "").upper()
        if continuation_branch:
            if continuation_branch not in {"A", "B"}:
                raise ValueError(
                    "HIGHSTEP_E1400_CONTINUATION_BRANCH must be A or B, got "
                    f"{continuation_branch!r}"
                )
            self.policy.student_recovery_stage = "E1400_CONTINUATION"
            self.policy.student_recovery_e1400_continuation_branch = continuation_branch
            self.policy.student_recovery_e1400_continuation_preregistration_path = os.environ.get(
                "HIGHSTEP_E1400_CONTINUATION_PREREGISTRATION_PATH", ""
            )
            self.policy.student_recovery_e1400_continuation_preregistration_sha256 = os.environ.get(
                "HIGHSTEP_E1400_CONTINUATION_PREREGISTRATION_SHA256", ""
            )
        else:
            self.policy.student_recovery_stage = "ENV_CURRICULUM_V18"
            self.policy.student_recovery_v18_preregistration_path = os.environ.get(
                "HIGHSTEP_V18_PREREGISTRATION_PATH", ""
            )
            self.policy.student_recovery_v18_preregistration_sha256 = os.environ.get(
                "HIGHSTEP_V18_PREREGISTRATION_SHA256", ""
            )
        self.policy.student_recovery_source_checkpoint = (
            "/home/lxq/Softwares/robot_lab/tmp/"
            "highstep_historical_0707_exact_new_teacher_20260714/"
            "source_model_172300/model_172300.pt"
        )
        self.policy.student_recovery_source_sha256 = (
            "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
        )
        self.policy.student_recovery_teacher_checkpoint = (
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-11_11-22-23/model_172300.pt"
        )
        self.policy.student_recovery_teacher_sha256 = self.policy.student_recovery_source_sha256
        self.policy.student_actor_warmup_updates = (
            1700 if continuation_branch == "A" else 1400
        )
        self.policy.student_vae_epochs = 4
        self.policy.student_vel_loss_coef = 10.0
        self.policy.student_latent_loss_coef = 50.0
        self.policy.student_teacher_action_loss_coef = 20.0
        self.policy.student_prior_box_loss_coef = 5.0
        self.policy.student_recon_loss_coef = 0.5
        self.policy.student_kl_loss_coef = 0.1
        self.policy.student_post_prior_mode = "highstep"
        self.policy.student_highstep_phase_loss_scale = 2.0
        self.policy.student_highstep_rear_box_loss_scale = 1.5
        self.policy.student_highstep_rear_hip_loss_scale = 0.0
        self.policy.student_highstep_rear_hip_min_abs = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
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
