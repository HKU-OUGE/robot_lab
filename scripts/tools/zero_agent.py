# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--init_height", type=float, default=2.0, help="Initial robot base height (meters).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import robot_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import configclass
import robot_lab.tasks.locomotion.velocity.mdp as mdp

@configclass
class EmptyEventsCfg:
    pass

@configclass
class EmptyCurriculumCfg:
    pass

@configclass
class EmptyRewardsCfg:
    pass

@configclass
class EmptyCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0, 0)
        ),
    )


# PLACEHOLDER: Extension template (do not remove this comment)
def _noop_event(env, env_ids, **kwargs):
    return None

def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # disable events
    env_cfg.events = EmptyEventsCfg()
    env_cfg.curriculum = EmptyCurriculumCfg()
    env_cfg.commands = EmptyCommandsCfg()
    env_cfg.rewards = EmptyRewardsCfg()
    if hasattr(env_cfg.scene, "robot") and hasattr(env_cfg.scene.robot, "init_state"):
        x, y, _ = env_cfg.scene.robot.init_state.pos
        env_cfg.scene.robot.init_state.pos = (x, y, args_cli.init_height)
        print(f"[INFO] Set robot initial height to {args_cli.init_height} m (from {x}, {y})")
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        print("[INFO] Disabled global gravity in the simulation.")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
