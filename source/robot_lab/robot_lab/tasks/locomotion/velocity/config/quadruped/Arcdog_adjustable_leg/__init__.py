# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ArclabArcdogAdjustableLegFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:ArclabArcdogAdjustableLegRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Bodyflat-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodyflat_env_cfg:ArclabArcdogAdjustableLegBodyflatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegBodyflatPPORunnerCfg",
        # ==========================================================
        # 🌟 新增下面这一行，将 symmetric_ppo_cfg 映射到你的自定义配置类
        # ==========================================================
        "symmetric_ppo_cfg": f"{agents.__name__}.symmetric_ppo_cfg:ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-BodyflatStudentNoPrior-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bodyflat_env_cfg:ArclabArcdogAdjustableLegBodyflatStudentNoPriorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegBodyflatStudentNoPriorPPORunnerCfg",
        "symmetric_ppo_cfg": f"{agents.__name__}.symmetric_ppo_cfg:ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepStudentNoPriorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepStudentNoPriorPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepActionScoreRobustEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepRearSupportV112-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepRearSupportV112EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepRearSupportV112PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepRearSupportFrontPlacementV1121-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepRearSupportFrontPlacementV1121PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123Control-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123ControlEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123ControlPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepFrontGeometryV114STEStudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV114StudentNoPriorPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepB300CanonicalHybridStudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepFrontGeometryV1123StudentNoPriorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300CanonicalHybridStudentNoPriorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.highstep_env_cfg:ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR2-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorR2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR2PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR3-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorR2EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR3PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV15-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15PPORunnerCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior0707Exact-ArcdogAdjustableLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPrior0707ExactPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorHistorical0707Exact-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV15EnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorHistorical0707ExactPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepB3000707DerivedSingleRun7400StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18BootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepB3000707DerivedSingleRun7400PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepB300DiagonalImitationFresh7400StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18BootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300DiagonalImitationFresh7400PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepB300CriticalTransitionBalancedDiagonalFresh7400StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300CriticalTransitionBalancedEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300CriticalTransitionBalancedDiagonalFresh7400PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300RLPreEdgeContinuationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepB300RLPreEdgeContinuationE7700PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Bootstrap-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18BootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18RobustEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorV18PPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "RobotLab-Isaac-Velocity-HighstepActionScoreTeacherV18Bootstrap-"
        "ArcdogAdjustableLeg-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.highstep_env_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScoreTeacherV18BootstrapEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "ArclabArcdogAdjustableLegHighstepActionScorePPORunnerCfg"
        ),
    },
)
