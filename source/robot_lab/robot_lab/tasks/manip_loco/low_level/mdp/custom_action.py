# custom_actions.py

from __future__ import annotations

import torch
from collections.abc import Sequence
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

# 引入必要的数学工具进行坐标变换
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation


class CommandDrivenIKAction(ActionTerm):
    """
    一个自定义动作项，它忽略策略网络的输入，直接使用环境的 Command 来驱动 IK 控制器。
    它在机器人的基座坐标系（Base Frame）下执行所有的 IK 计算。
    """

    cfg: CommandDrivenIKActionCfg
    _asset: Articulation
    _controller: DifferentialIKController

    def __init__(self, cfg: CommandDrivenIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        
        # 1. 获取受控关节索引
        if cfg.joint_names is None:
            self._joint_indices = list(range(self._asset.num_joints))
        else:
            # 这里保存了机械臂关节在全身关节列表中的索引
            self._joint_indices = [self._asset.joint_names.index(name) for name in cfg.joint_names]
        
        # 2. 获取相关 Body 的索引
        self._ee_body_index = self._asset.body_names.index(cfg.body_name)
        self._robot_base_body_index = self._asset.body_names.index(cfg.robot_base_body_name)
        
        # 获取用于从 PhysX 视图提取雅可比的索引
        if self._ee_body_index > 0:
             self._ee_jacobi_index = self._ee_body_index - 1
        else:
             self._ee_jacobi_index = 0

        # 初始化控制器
        self._controller = DifferentialIKController(cfg.controller, num_envs=self.num_envs, device=self.device)
        self._command_name = cfg.command_name_to_follow

        # Buffer
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._asset.num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return len(self._joint_indices)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """
        在基座坐标系下执行 IK 计算的核心方法。
        """
        self._raw_actions[:] = actions

        # 1. 获取目标位姿 (Target Pose)
        target_pose_b = self._env.command_manager.get_command(self._command_name)

        # 2. 设置控制器的目标
        self._controller.set_command(target_pose_b)

        # 3. 获取当前状态信息 (世界坐标系)
        root_pose_w = self._asset.data.body_state_w[..., self._robot_base_body_index, :7]
        root_quat_w = root_pose_w[..., 3:]
        ee_pose_w = self._asset.data.body_state_w[..., self._ee_body_index, :7]
        
        # 注意：这里只需要获取受控关节（机械臂）的当前位置
        current_joint_pos_arm = self._asset.data.joint_pos[..., self._joint_indices]

        # 4. 计算相对于基座的当前末端位姿 (World -> Base Frame)
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[..., :3], root_quat_w,
            ee_pose_w[..., :3], ee_pose_w[..., 3:]
        )

        # 5. 获取并转换雅可比矩阵 (World -> Base Frame)
        full_jacobian_w = self._asset.root_physx_view.get_jacobians()
        # 只提取末端关于机械臂关节的雅可比
        jacobian_w_arm = full_jacobian_w[:, self._ee_jacobi_index, :, :][..., self._joint_indices]

        base_rot_matrix_inv = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
        
        jacobian_b_arm = torch.empty_like(jacobian_w_arm)
        jacobian_b_arm[:, :3, :] = torch.bmm(base_rot_matrix_inv, jacobian_w_arm[:, :3, :])
        jacobian_b_arm[:, 3:, :] = torch.bmm(base_rot_matrix_inv, jacobian_w_arm[:, 3:, :])

        # 6. 调用 compute 方法
        # 注意：传入的 joint_pos 和 jacobian 都必须只包含受控关节的数据
        joint_pos_targets_arm = self._controller.compute(
            ee_pos=ee_pos_b,
            ee_quat=ee_quat_b,
            jacobian=jacobian_b_arm,
            joint_pos=current_joint_pos_arm
        )

        # 7. 应用缩放和偏移，并填入动作张量
        scaled_targets_arm = joint_pos_targets_arm * self.cfg.scale + self.cfg.offset
        
        # 为了内部状态一致性，我们仍然先填满默认值（可选，但推荐）
        self._processed_actions[:] = self._asset.data.default_joint_pos
        # 将计算出的机械臂目标填入对应位置
        self._processed_actions[..., self._joint_indices] = scaled_targets_arm

    # def apply_actions(self):
    #     """将计算出的关节目标应用到物理引擎。"""
    #     # --- 关键修正 ---
    #     # 1. 只从完整的 processed_actions 中提取出属于机械臂的目标值
    #     # shape: [num_envs, num_arm_joints]
    #     arm_targets = self._processed_actions[..., self._joint_indices]
        
    #     # 2. 调用 set_joint_position_target 时，显式传入 joint_indices
    #     # 这样物理引擎就只会更新这些特定关节的目标，而不会触碰其他关节（如腿部）
    #     self._asset.set_joint_position_target(arm_targets, joint_ids=self._joint_indices)
    #     def reset(self, env_ids: Sequence[int] | None = None):
    #         """在环境重置时重置控制器状态。"""
    #         self._controller.reset(env_ids)
    #     # ----------------

    def apply_actions(self):
        """将计算出的关节目标应用到物理引擎。
        
        注意：此版本将机械臂目标位置硬编码为固定值。
        """
        
        # 固定的目标关节位置（弧度）
        # 假设您的机械臂有 6 个关节，按顺序对应 "joint1" 到 "joint6"
        fixed_arm_targets = [
            0.0,    # "joint1"
            0.7,    # "joint2"
            -1.2,   # "joint3"
            0.0,    # "joint4"
            1.0,    # "joint5"
            0.0     # "joint6"
            ###############
            # 0.0,    # "joint1"
            # 0.0,    # "joint2"
            # 0.0,   # "joint3"
            # 0.0,    # "joint4"
            # 0.0,    # "joint5"
            # -1.57     # "joint6"
            # 0.0,
            # 2.2,
            # -1.5,
            # 0.0,
            # -0.6, # -0.6 0.7
            # 0.0
        ]
        
        # 1. 将固定目标转换为适合环境数量的张量 (Tensor)
        # 假设 self._env.num_envs 是环境的数量
        # 确保 fixed_arm_targets 是一个张量
        fixed_targets_tensor = torch.tensor(
            fixed_arm_targets, 
            dtype=self._processed_actions.dtype, # 保持与动作张量相同的类型
            device=self._processed_actions.device # 保持与动作张量相同的设备 (CPU/GPU)
        )
        
        # 将形状从 [num_arm_joints] 扩展到 [num_envs, num_arm_joints]
        # 使用 unsqueeze(0) 在第一个维度添加一个批次维度，然后 repeat 复制 num_envs 次。
        # shape: [num_envs, num_arm_joints]
        arm_targets = fixed_targets_tensor.unsqueeze(0).repeat(self._env.num_envs, 1)

        # 2. 调用 set_joint_position_target，显式传入 joint_indices
        # 物理引擎只会更新这些特定关节的目标。
        self._asset.set_joint_position_target(arm_targets, joint_ids=self._joint_indices)
            
        def reset(self, env_ids: Sequence[int] | None = None):
            """在环境重置时重置控制器状态。"""
            self._controller.reset(env_ids)

@configclass
class CommandDrivenIKActionCfg(ActionTermCfg):
    class_type: type = CommandDrivenIKAction
    asset_name: str = "robot"
    joint_names: Sequence[str] | None = None
    robot_base_body_name: str = "base"
    body_name: str = "ee_link"
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    command_name_to_follow: str = "ee_pose"
    scale: float = 1.0
    offset: float = 0.0