from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from .rsl_rl_ppo_cfg import ArclabArcdogAdjustableLegBodyflatPPORunnerCfg

# ==========================================================
# 🌟 1. 定义带有 symmetry_coef 的自定义算法配置类
# ==========================================================
@configclass
class SymmetricPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the Symmetric PPO algorithm."""
    symmetry_coef: float = 2.0  # 添加对称性权重参数，默认值为 2.0


# ==========================================================
# 🌟 2. 定义 Runner 配置类
# ==========================================================
@configclass
class ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg(ArclabArcdogAdjustableLegBodyflatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # 修改实验名称，方便在 TensorBoard/WandB 中区分
        self.experiment_name = "arclab_arcdog_adjustable_leg_bodyflat_symmetric"

        # 使用我们刚刚定义的 SymmetricPpoAlgorithmCfg 替换掉原来的 algorithm 配置
        self.algorithm = SymmetricPpoAlgorithmCfg(
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
            symmetry_coef=2.0,  # <--- 🌟 在这里自由调整对称性 Loss 的权重
        )