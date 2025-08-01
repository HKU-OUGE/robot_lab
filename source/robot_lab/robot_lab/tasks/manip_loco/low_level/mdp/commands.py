# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import omni.logs
import isaaclab.utils.math as math_utils
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_apply, quat_inv
# from omni.isaac.lab.assets import Articulation
# from omni.isaac.lab.managers import CommandTerm
# from omni.isaac.lab.markers import VisualizationMarkers
# from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_apply, quat_inv


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from commands_cfg import UniformPoseCommandCfg
    from commands_cfg import UniformVelocityCommandCfg


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """

class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.curr_ee_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.env = env


    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame

        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        if self.cfg.is_QuadrupedARM or self.cfg.is_QuadrupedARM_Play == True:
            self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0] 

        # compute the error    
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        
        # -- position
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])        
        r = torch.empty(len(env_ids), device=self.device)
        r_1 = torch.empty(1, device=self.device)

        if self.cfg.is_QuadrupedARM == True:

            coeff = 0.01 / self.cfg.curriculum_coeff   # TODO:only for 4096 envs, please change "0.01" for other envs. It may be a multiple relationship.
            coeff = torch.tensor(coeff, device=self.device)
            count = torch.tensor(self.env._sim_step_counter * coeff, device=self.device)
            self.pose_command_b[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1)
            self.pose_command_b[env_ids, 1] = (r.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
            self.pose_command_w_z[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                              (r.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)
            self.pose_command_b[env_ids, 2] = self.pose_command_w_z[env_ids, 0]  - self.robot.data.root_pos_w[env_ids, 2] 
                                              
            for i in range(len(env_ids)):
                length_arm = torch.norm(torch.stack([self.pose_command_b[i, 0], 
                                                     self.pose_command_b[i, 1], 
                                                     self.pose_command_b[i, 2] 
                                                    ])) 
                while((length_arm > 0.7) or (length_arm < 0.3) or (self.pose_command_b[i, 0] < 0.45 and torch.abs(self.pose_command_b[i, 1]) < 0.2)):
                        self.pose_command_b[i, 0] = (r_1.uniform_(*self.cfg.ranges_init.pos_x))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_x)) * torch.clamp((count), 0, 1) 
                        self.pose_command_b[i, 1] = (r_1.uniform_(*self.cfg.ranges_init.pos_y))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_y)) * torch.clamp((count), 0, 1)
                        self.pose_command_w_z[i, 0] = (r_1.uniform_(*self.cfg.ranges_init.pos_z))  * torch.clamp((1 - count), 0, 1) + \
                                                        (r_1.uniform_(*self.cfg.ranges_final.pos_z)) * torch.clamp((count), 0, 1)
                        self.pose_command_b[i, 2] = self.pose_command_w_z[i, 0] - self.robot.data.root_pos_w[i, 2]                     
                        length_arm = torch.norm(torch.stack([self.pose_command_b[i, 0], 
                                                        self.pose_command_b[i, 1], 
                                                         self.pose_command_b[i, 2]
                                                        ])) 
            
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges_init.roll) * torch.clamp((1 - count), 0, 1) + \
                                 r.uniform_(*self.cfg.ranges_final.roll) * torch.clamp((count), 0, 1)
                                 
            delta_x = self.pose_command_b[env_ids, 0] 
            delta_y = self.pose_command_b[env_ids, 1] 
            delta_z = self.pose_command_b[env_ids, 2]

            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + \
                                    r.uniform_(*self.cfg.ranges.pitch) * torch.clamp((1 - count), 0, 1) + \
                                    r.uniform_(*self.cfg.ranges_final.pitch) * torch.clamp((count), 0, 1)   
                                    
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + \
                                r.uniform_(*self.cfg.ranges_init.yaw) * torch.clamp((1 - count), 0, 1) + \
                                r.uniform_(*self.cfg.ranges_final.yaw) * torch.clamp((count), 0, 1) 
            # roll useless 
          
        elif self.cfg.is_QuadrupedARM_Play == True:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_w_z[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_z)
            self.pose_command_b[env_ids, 2] =  self.pose_command_w_z[env_ids, 0] - self.robot.data.root_pos_w[env_ids, 2] 

            delta_x = self.pose_command_b[env_ids, 0] 
            delta_y = self.pose_command_b[env_ids, 1] 
            delta_z = self.pose_command_b[env_ids, 2]         
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = - torch.atan2(delta_z, torch.sqrt(delta_x**2 + delta_y**2)) + r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = torch.atan2(delta_y, delta_x) + r.uniform_(*self.cfg.ranges.yaw)
        else:
            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)    
            euler_angles[:, 0] = r.uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1] = r.uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2] = r.uniform_(*self.cfg.ranges.yaw)

        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat     


    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # self.env = env ###
        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.curr_base_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.env = env

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        
        r = torch.empty(len(env_ids), device=self.device)
        coeff = 0.01 / self.cfg.curriculum_coeff # TODO:only for 4096 envs, please change "0.01" for other envs. It may be a multiple relationship.
        coeff = torch.tensor(coeff, device=self.device)
        if self.cfg.is_QuadrupedARM:
            count = torch.tensor(self.env._sim_step_counter * coeff, device=self.device)
            self.vel_command_b[env_ids, 0] = (r.uniform_(*self.cfg.ranges_init.lin_vel_x)) * torch.clamp((1 - count), 0, 1) + (r.uniform_(*self.cfg.ranges_final.lin_vel_x)) * torch.clamp(count, 0, 1)
            self.vel_command_b[env_ids, 1] = (r.uniform_(*self.cfg.ranges_init.lin_vel_y)) * torch.clamp((1 - count), 0, 1) + (r.uniform_(*self.cfg.ranges_final.lin_vel_y)) * torch.clamp(count, 0, 1)
            self.vel_command_b[env_ids, 2] = (r.uniform_(*self.cfg.ranges_init.ang_vel_z)) * torch.clamp((1 - count), 0, 1) + (r.uniform_(*self.cfg.ranges_final.ang_vel_z)) * torch.clamp(count, 0, 1)

        else:
            self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)    

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)

            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat