# Copyright (c) 2024-2025 Tianyang TANG
# SPDX-License-Identifier: Apache-2.0
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim.schemas import CollisionPropertiesCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from .rough_env_cfg import CUHKLRLSiriusWRoughEnvCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import RigidObjectCfg
@configclass
class CUHKLRLSiriusWFlatEnvCfg(CUHKLRLSiriusWRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
            # spawn a usd file of a table into the scene
        # cfg = sim_utils.UsdFileCfg(usd_path="/home/ouge/Iveco_Daily_Van_2014_origin.usda", scale=(0.01, 0.01, 0.01), collision_props=CollisionPropertiesCfg(collision_enabled=True), rigid_props=RigidBodyPropertiesCfg(rigid_body_enabled=True))
        # cfg.func("/World/Objects/Van", cfg, translation=(10.0, 10.0, 0.55))
        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"

        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CUHKLRLSiriusWFlatEnvCfg":
            self.disable_zero_weight_rewards()
