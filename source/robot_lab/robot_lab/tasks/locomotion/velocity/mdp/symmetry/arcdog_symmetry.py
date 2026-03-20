import torch

class ArcdogSymmetry:
    def __init__(self, device):
        self.device = device
        
        # 16自由度关节镜像交换索引
        # 你的顺序: [FL, FR, RL, RR] * 4组 (Hip, Thigh, Calf, Box)
        # 交换逻辑: FL(0)<->FR(1), RL(2)<->RR(3), FL_th(4)<->FR_th(5)...
        self.swap_indices_16 = [
            1, 0, 3, 2,    # Hip
            5, 4, 7, 6,    # Thigh
            9, 8, 11, 10,  # Calf
            13, 12, 15, 14 # Box (变长腿)
        ]
        
        # 只有 Hip (HAA) 关节在左右镜像时，角度/速度/动作需要取反 (侧摆角左右相反)
        # 假设 Thigh(HFE), Calf(KFE), Box(Prismatic) 左右伸展方向是一致的，不需要取反
        self.negate_indices = [0, 1, 2, 3]

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        """对 16 维动作进行左右镜像"""
        mirrored = action.clone()
        # 1. 交换左右腿
        mirrored = mirrored[:, self.swap_indices_16]
        # 2. Hip 关节取反
        mirrored[:, self.negate_indices] *= -1.0
        return mirrored

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        对 57 维观测进行左右镜像
        结构: ang_vel(0:3), gravity(3:6), commands(6:9), dof_pos(9:25), dof_vel(25:41), actions(41:57)
        """
        mirrored = obs.clone()
        
        # 1. ang_vel (0:3): 基座角速度 (Roll-x, Pitch-y, Yaw-z)
        # 左右镜像时，Roll(x) 和 Yaw(z) 取反
        mirrored[:, 0] *= -1.0 
        mirrored[:, 2] *= -1.0
        
        # 2. gravity_vec (3:6): 重力投影 (gx, gy, gz)
        # 左右镜像时，y轴分量取反
        mirrored[:, 4] *= -1.0
        
        # 3. commands (6:9): 速度指令 (vx, vy, yaw_rate)
        # 左右镜像时，侧向速度(vy) 和 偏航角速度(yaw_rate) 取反
        mirrored[:, 7] *= -1.0
        mirrored[:, 8] *= -1.0
        
        # 4. dof_pos (9:25): 16维关节位置
        dof_pos = mirrored[:, 9:25]
        dof_pos = dof_pos[:, self.swap_indices_16]
        dof_pos[:, self.negate_indices] *= -1.0
        mirrored[:, 9:25] = dof_pos
        
        # 5. dof_vel (25:41): 16维关节速度
        dof_vel = mirrored[:, 25:41]
        dof_vel = dof_vel[:, self.swap_indices_16]
        dof_vel[:, self.negate_indices] *= -1.0
        mirrored[:, 25:41] = dof_vel
        
        # 6. actions (41:57): 16维历史动作
        prev_actions = mirrored[:, 41:57]
        prev_actions = prev_actions[:, self.swap_indices_16]
        prev_actions[:, self.negate_indices] *= -1.0
        mirrored[:, 41:57] = prev_actions

        return mirrored