# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-Velocity-Flat-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CUHKLRLSiriusWFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:CUHKLRLSiriusWRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWRoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Stand-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stand_env_cfg:CUHKLRLSiriusWStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWStandPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWStandPPORunnerWithSymmetryCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-BackFlip-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.backflip_env_cfg:CUHKLRLSiriusWBackFlipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWBackFlipPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Pit-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pit_env_cfg:CUHKLRLSiriusWPitEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWPitPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Ring-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ring_env_cfg:CUHKLRLSiriusWRingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWRingPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWRingPPORunnerWithSymmetryCfg"
        ),
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Slip-Flat-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slip_flat_env_cfg:CUHKLRLSiriusWSlipFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWSlipFlatPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-WheelEXP-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_exp_env_cfg:CUHKLRLSiriusWWheelEXPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWWheelEXPPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-LegEXP-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leg_exp_env_cfg:CUHKLRLSiriusWLegEXPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWLegEXPPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-SlopeLegEXP-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.slope_leg_exp_env_cfg:CUHKLRLSiriusWSlopeLegEXPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWSlopeLegEXPPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-CMPLegEXP-CUHKLRL-SiriusW-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.compare_legexp_env_cfg:CUHKLRLSiriusWCMPLegEXPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWCMPLegEXPPPORunnerCfg",
    },
)

###############################
# Teacher-Student Distillation #
###############################


gym.register(
    id="RobotLab-Isaac-Velocity-LegEXP-CUHKLRL-SiriusW-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leg_exp_env_cfg:CUHKLRLSiriusWLegEXPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CUHKLRLSiriusWLegEXPDistillationRunnerCfg",
    },
)



###############################
# MoE #
###############################
gym.register(
    id="RobotLab-Isaac-Velocity-SiriusW-MoE-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.siriusw_moe_env_cfg:CUHKLRLSiriusWMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_terrain:SiriusSharedMoEPPOCfg", 
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-SiriusW-MoECMP-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.siriusw_moe_env_cfg:CUHKLRLSiriusWMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SiriusMoECMPPPOCfg", 
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-SiriusW-MoESimple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.siriusw_moe_env_cfg:CUHKLRLSiriusWMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_simple:SiriusRobustMoEPPOCfg", 
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-SiriusW-MoECheck-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.siriusw_moe_env_cfg:CUHKLRLSiriusWMoEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.moe_simple:SiriusMoESanityCheckCfg", 
    },
)

