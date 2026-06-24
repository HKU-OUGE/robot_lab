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
