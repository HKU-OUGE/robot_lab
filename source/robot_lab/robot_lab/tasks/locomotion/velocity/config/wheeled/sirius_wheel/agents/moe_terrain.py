# source/robot_lab/algo/moe/policy.py

import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ELU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HierarchicalMoEActorCritic(ActorCritic):
    """
    [Teacher Policy] 端到端双层混合专家 (H-MoE) 架构
    
    架构层级:
    1. Shared Backbone (GRU): 处理观测序列，提取时序 Latent。
    2. Layer 1 (Mode Gate): 宏观决策，输出 [轮组权重, 腿组权重]。
    3. Layer 2 (Expert Groups):
       - 轮式专家组 (Wheel Group): N 个专家，专注于轮式主导的移动 (如巡航、爬坡)。
       - 腿式专家组 (Leg Group): M 个专家，专注于腿式主导的移动 (如迈步、脱困)。
    4. Fusion: 
       Action = w_wheel * Sum(Wheel_Experts) + w_leg * Sum(Leg_Experts)
       注意：所有专家均输出全维度动作 (num_actions=16)，实现全身协调控制。

    [改进特性]:
    - 集成 Temperature Annealing (温度退火) 防止路由崩塌 (0.5/0.5)。
    - 集成 Noisy Gating (噪声门控) 保持探索性，防止专家饿死。
    - [New] Action Masking: 强制轮/腿专家组只控制对应的关节。
    """
    def __init__(self, 
                 obs, 
                 obs_groups, 
                 num_actions, 
                 actor_hidden_dims=[512, 256, 128], 
                 critic_hidden_dims=[512, 256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 # === H-MoE 特定参数 ===
                 num_wheel_experts=3, # 轮组专家数量
                 num_leg_experts=3,   # 腿组专家数量
                 latent_dim=None,     # GRU 输出维度
                 # === 训练稳定性参数 ===
                 gating_start_temp=1.0,    # 初始温度 (较高以鼓励随机探索)
                 gating_end_temp=0.1,      # 结束温度 (较低以鼓励尖锐决策)
                 gating_anneal_steps=2000, # 退火持续的迭代次数 (Iterations)
                 gating_noise_std=0.5,     # 门控噪声标准差
                 # === 动作掩码参数 (New) ===
                 wheel_indices=None,       # 轮关节索引列表 (e.g., [0,1,2,3])
                 leg_indices=None,         # 腿关节索引列表 (e.g., [4,5,...,15])
                 **kwargs):
        
        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **kwargs)

        # -----------------------------------------------------
        # 1. 输入处理与维度计算
        # -----------------------------------------------------
        self.policy_keys = None 
        self.critic_keys = None 
        
        is_dict_like = isinstance(obs, dict) or hasattr(obs, "keys")

        if is_dict_like:
            # Policy Keys
            policy_keys = obs_groups.get("policy", None)
            if policy_keys is None and "actor" in obs_groups:
                policy_keys = obs_groups["actor"]
            self.policy_keys = policy_keys

            if policy_keys:
                num_actor_obs = sum(obs[k].shape[-1] for k in policy_keys)
            else:
                first_key = next(iter(obs.keys()))
                self.policy_keys = [first_key]
                num_actor_obs = obs[first_key].shape[-1]
            
            # Critic Keys
            self.critic_keys = obs_groups.get("critic", None)
        else:
            num_actor_obs = obs.shape[-1]

        # 确定 Latent 维度 (GRU Hidden Size)
        self.hidden_state_dim = latent_dim if latent_dim is not None else num_actor_obs
        
        # 初始化 RNN 状态
        self.is_recurrent = True
        self._valid_batch_size = None
        
        if is_dict_like:
             ref_tensor = obs[self.policy_keys[0] if self.policy_keys else next(iter(obs.keys()))]
        else:
             ref_tensor = obs
        
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        self.active_hidden_states = torch.zeros(1, batch_size, self.hidden_state_dim, device=device)
        
        # === 门控训练参数初始化 ===
        self.gating_start_temp = gating_start_temp
        self.gating_end_temp = gating_end_temp
        self.gating_anneal_steps = gating_anneal_steps
        self.gating_noise_std = gating_noise_std
        
        self.current_gating_temp = gating_start_temp
        self.anneal_counter = 0
        
        # === 动作掩码初始化 (New) ===
        # 创建 buffer 以便自动处理 device 和 precision
        self.register_buffer('wheel_action_mask', torch.ones(num_actions))
        self.register_buffer('leg_action_mask', torch.ones(num_actions))
        
        self.use_action_masking = False
        if wheel_indices is not None and leg_indices is not None:
            self.use_action_masking = True
            # 重置为全0
            self.wheel_action_mask.fill_(0.0)
            self.leg_action_mask.fill_(0.0)
            
            # 激活对应关节
            # 轮专家组：只控制轮关节 (indices设为1)
            # [Fix] 将 tuple 转为 list，避免 PyTorch 将其误判为多维索引
            wheel_indices_list = list(wheel_indices)
            self.wheel_action_mask[wheel_indices_list] = 1.0
            
            # 腿专家组：只控制腿关节 (indices设为1)
            # [Fix] 将 tuple 转为 list
            leg_indices_list = list(leg_indices)
            self.leg_action_mask[leg_indices_list] = 1.0
            
            print(f"[H-MoE] Action Masking Enabled:")
            print(f"        Wheel Experts Active Indices: {wheel_indices}")
            print(f"        Leg Experts Active Indices:   {leg_indices}")
        else:
            print("[H-MoE] Action Masking Disabled (Indices not provided). Experts control all joints.")

        print(f"[H-MoE] Input Dim: {num_actor_obs} | Latent Dim: {self.hidden_state_dim}")
        print(f"[H-MoE] Config: {num_wheel_experts} Wheel Experts, {num_leg_experts} Leg Experts")
        print(f"[H-MoE] Gating: Temp={gating_start_temp}->{gating_end_temp}, NoiseStd={gating_noise_std}")

        # -----------------------------------------------------
        # 2. 共享骨干 (Shared Backbone)
        # -----------------------------------------------------
        self.gru = nn.GRU(input_size=num_actor_obs, 
                          hidden_size=self.hidden_state_dim, 
                          batch_first=False) 

        # -----------------------------------------------------
        # 3. 第1层: 模态门控 (Mode Gating)
        # -----------------------------------------------------
        # 输出 [Wheel_Weight, Leg_Weight]
        self.mode_gate = nn.Sequential(
            nn.Linear(self.hidden_state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2) 
        )

        # -----------------------------------------------------
        # 4. 第2层: 专家组 (Expert Groups)
        # -----------------------------------------------------
        
        # A. 轮式专家组 (Wheel Experts)
        # 专家网络
        self.wheel_experts = nn.ModuleList([
            MLP(self.hidden_state_dim, num_actions, hidden_dims=actor_hidden_dims)
            for _ in range(num_wheel_experts)
        ])
        # 轮组内部门控
        self.wheel_gate = nn.Linear(self.hidden_state_dim, num_wheel_experts)

        # B. 腿式专家组 (Leg Experts)
        # 专家网络
        self.leg_experts = nn.ModuleList([
            MLP(self.hidden_state_dim, num_actions, hidden_dims=actor_hidden_dims)
            for _ in range(num_leg_experts)
        ])
        # 腿组内部门控
        self.leg_gate = nn.Linear(self.hidden_state_dim, num_leg_experts)

        print("\n" + "="*50)
        print("[H-MoE] End-to-End Hierarchical MoE Architecture Initialized.")
        print("        Action Space: Full 16-Dim Whole Body Control.")
        print("="*50 + "\n")

    @property
    def is_recurrent(self):
        return True
    
    @is_recurrent.setter
    def is_recurrent(self, value):
        pass

    # ----------------------------------------------------------------
    # [New] 门控辅助函数：处理噪声和温度
    # ----------------------------------------------------------------
    def update_annealing(self):
        """
        更新门控温度。
        建议在每次 PPO 迭代(Iteration)结束时调用此函数。
        """
        self.anneal_counter += 1
        if self.anneal_counter < self.gating_anneal_steps:
            # 线性衰减
            alpha = self.anneal_counter / self.gating_anneal_steps
            self.current_gating_temp = self.gating_start_temp - alpha * (self.gating_start_temp - self.gating_end_temp)
        else:
            self.current_gating_temp = self.gating_end_temp

    def _gated_softmax(self, logits):
        """
        应用噪声门控 (Noisy Gating) 和 温度退火 (Temperature Annealing)。
        
        Args:
            logits: 门控网络的原始输出
        Returns:
            weights: 经过 Softmax 归一化后的权重
        """
        # 1. 噪声注入 (仅在训练模式下)
        if self.training and self.gating_noise_std > 0.0:
            noise = torch.randn_like(logits) * self.gating_noise_std
            logits = logits + noise
        
        # 2. 温度缩放 (训练初期温度高->分布平滑，后期温度低->分布尖锐)
        # 避免除以0
        temp = max(self.current_gating_temp, 1e-4)
        scaled_logits = logits / temp
        
        return torch.softmax(scaled_logits, dim=-1)

    # ----------------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------------
    def forward(self, obs, masks=None, hidden_states=None, save_dist=True, valid_batch_size=None):
        # --- 1. 数据准备 (与原代码一致) ---
        if self.policy_keys is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in self.policy_keys]
            x_in = torch.cat(tensors, dim=-1)
        else:
            x_in = obs

        if hidden_states is None:
            batch_size = x_in.shape[0] 
            hidden_states = torch.zeros(1, batch_size, self.hidden_state_dim, device=x_in.device)

        # 数据切片处理 (Slicing for efficient buffer usage)
        if valid_batch_size is not None:
            if x_in.ndim == 3 and x_in.shape[1] > valid_batch_size: 
                x_in = x_in[:, :valid_batch_size, :]
            elif x_in.ndim == 2 and x_in.shape[0] > valid_batch_size:
                x_in = x_in[:valid_batch_size, :]
            
            if hidden_states is not None and hidden_states.shape[1] > valid_batch_size:
                hidden_states = hidden_states[:, :valid_batch_size, :]
                
            if masks is not None:
                if masks.ndim == 3 and masks.shape[1] > valid_batch_size:
                     masks = masks[:, :valid_batch_size, :]
                elif masks.ndim == 2 and masks.shape[1] > valid_batch_size:
                     masks = masks[:, :valid_batch_size]

        if hidden_states.shape[0] > hidden_states.shape[1] and hidden_states.shape[1] == self.gru.num_layers:
             hidden_states = hidden_states.transpose(0, 1).contiguous()
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        # --- 2. Shared Backbone (GRU) -> Latent ---
        latent = None
        
        if x_in.ndim == 3: # Sequence
            is_input_transposed = False
            if x_in.shape[0] == hidden_states.shape[1]:
                x_gru = x_in.transpose(0, 1)
                is_input_transposed = True
            elif x_in.shape[1] == hidden_states.shape[1]:
                x_gru = x_in
                is_input_transposed = False
            else:
                x_gru = x_in.transpose(0, 1)
                is_input_transposed = True

            gru_out, next_hidden_states = self.gru(x_gru, hidden_states)
            latent = gru_out.transpose(0, 1) if is_input_transposed else gru_out

        elif x_in.ndim == 2: # Step
            num_envs_hidden = hidden_states.shape[1]
            batch_size_in = x_in.shape[0]

            if batch_size_in == num_envs_hidden:
                x_gru = x_in.unsqueeze(0)
                if masks is not None:
                     hidden_states = hidden_states * masks.reshape(1, -1, 1)
                gru_out, next_hidden_states = self.gru(x_gru, hidden_states)
                latent = gru_out[0]
            else:
                # Handle flattened batch from PPO buffers
                if batch_size_in % num_envs_hidden == 0:
                    seq_len = batch_size_in // num_envs_hidden
                    x_gru = x_in.reshape(seq_len, num_envs_hidden, -1)
                    gru_out, next_hidden_states = self.gru(x_gru, hidden_states)
                    latent = gru_out.reshape(-1, self.hidden_state_dim)
                else:
                    x_gru = x_in.unsqueeze(0)
                    gru_out, next_hidden_states = self.gru(x_gru, hidden_states)
                    latent = gru_out[0]
        else:
            latent = x_in
            next_hidden_states = hidden_states

        # ============================================================
        # H-MoE 核心逻辑 (Updated with Noise & Temperature & Masking)
        # ============================================================
        
        # --- Layer 1: Mode Gating (宏观模态) ---
        mode_logits = self.mode_gate(latent)
        # [Use Gated Softmax]
        mode_weights = self._gated_softmax(mode_logits) # [Batch, 2]
        
        w_wheel_group = mode_weights[..., 0].unsqueeze(-1)
        w_leg_group   = mode_weights[..., 1].unsqueeze(-1)

        # --- Layer 2A: Wheel Expert Group (轮组) ---
        wheel_gate_logits = self.wheel_gate(latent)
        # [Use Gated Softmax]
        wheel_expert_weights = self._gated_softmax(wheel_gate_logits) # [Batch, N_wheel]
        
        wheel_actions_sum = 0
        for i, expert in enumerate(self.wheel_experts):
            out = expert(latent) # 全维度输出
            # [New] 应用轮组 Mask: 强制腿关节输出为0
            if self.use_action_masking:
                out = out * self.wheel_action_mask
            
            wheel_actions_sum += out * wheel_expert_weights[..., i].unsqueeze(-1)

        # --- Layer 2B: Leg Expert Group (腿组) ---
        leg_gate_logits = self.leg_gate(latent)
        # [Use Gated Softmax]
        leg_expert_weights = self._gated_softmax(leg_gate_logits) # [Batch, N_leg]
        
        leg_actions_sum = 0
        for i, expert in enumerate(self.leg_experts):
            out = expert(latent) # 全维度输出
            # [New] 应用腿组 Mask: 强制轮关节输出为0
            if self.use_action_masking:
                out = out * self.leg_action_mask
                
            leg_actions_sum += out * leg_expert_weights[..., i].unsqueeze(-1)

        # --- Fusion: 全身控制融合 ---
        # 轮组动作与腿组动作按模态权重混合
        # 由于我们施加了 Mask，这里实际上变成了：
        # Wheel_Joints = w_wheel * Wheel_Experts(wheel_part) + w_leg * 0
        # Leg_Joints   = w_wheel * 0 + w_leg * Leg_Experts(leg_part)
        # 这实现了完美的解耦。
        actions_mean = w_wheel_group * wheel_actions_sum + w_leg_group * leg_actions_sum
        
        if save_dist:
            self.distribution = torch.distributions.Normal(actions_mean, self.std)

        return actions_mean, self.std, next_hidden_states

    # ----------------------------------------------------------------
    # 辅助函数 (保持不变)
    # ----------------------------------------------------------------
    def get_hidden_states(self):
        return self.active_hidden_states, self.active_hidden_states

    def get_actions_log_prob(self, actions):
        if actions is not None:
             self._valid_batch_size = actions.shape[1] 
        
        if self.distribution is not None:
            if self.distribution.batch_shape[1] != actions.shape[1]:
                target_batch = actions.shape[1]
                mean = self.distribution.loc[:, :target_batch, :]
                std = self.distribution.scale[:, :target_batch, :]
                self.distribution = torch.distributions.Normal(mean, std)

        return super().get_actions_log_prob(actions)

    def act(self, obs, masks=None, hidden_state=None):
        self._valid_batch_size = None
        if self.actor_obs_normalization:
            obs = self.norm_actor_obs(obs)
        
        if hidden_state is None:
            current_actor_state = self.active_hidden_states
        else:
            if isinstance(hidden_state, tuple):
                current_actor_state = hidden_state[0]
            else:
                current_actor_state = hidden_state
        
        actions_mean, actions_std, next_hidden_states = self.forward(obs, masks, current_actor_state, save_dist=True)
        
        if hidden_state is None:
            self.active_hidden_states = next_hidden_states.detach() 
        
        dist = torch.distributions.Normal(actions_mean, actions_std)
        self.distribution = dist
        
        actions = dist.sample()
        return actions

    def act_inference(self, obs, masks=None, hidden_states=None):
        self._valid_batch_size = None
        if self.actor_obs_normalization:
            obs = self.norm_actor_obs(obs)
        if hidden_states is None:
            hidden_states = self.active_hidden_states
        actions_mean, _, next_hidden_states = self.forward(obs, masks, hidden_states, save_dist=False)
        self.active_hidden_states = next_hidden_states
        return actions_mean
    
    def evaluate(self, obs, masks=None, hidden_states=None, hidden_state=None):
        if hidden_states is None and hidden_state is not None:
             hidden_states = hidden_state

        valid_batch = None
        if (masks is not None or hidden_states is not None) and self._valid_batch_size is not None:
            valid_batch = self._valid_batch_size

        self.forward(obs, masks, hidden_states, save_dist=False, valid_batch_size=valid_batch)

        if self.critic_keys is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in self.critic_keys]
            x_critic = torch.cat(tensors, dim=-1)
        else:
            x_critic = obs
            
        if valid_batch is not None:
            if x_critic.ndim == 3 and x_critic.shape[1] > valid_batch:
                x_critic = x_critic[:, :valid_batch, :]
            elif x_critic.ndim == 2 and x_critic.shape[0] > valid_batch:
                x_critic = x_critic[:valid_batch, :]
            
        return self.critic(x_critic)

# === 定义自定义 Config 类以接纳额外的参数 ===
@configclass
class HierarchicalMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "HierarchicalMoEActorCritic"
    num_wheel_experts: int = 3
    num_leg_experts: int = 3
    latent_dim: int = 256
    # [New] 训练稳定性参数默认值
    gating_start_temp: float = 1.0
    gating_end_temp: float = 0.1
    gating_anneal_steps: int = 2000
    gating_noise_std: float = 0.5
    # [New] 动作空间分解/掩码
    # 注意：这里需要传入 Tuple 或 List，例如 (0, 1, 2, 3) 代表前4个关节是轮子
    wheel_indices: tuple = None 
    leg_indices: tuple = None

@configclass
class SiriusMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000 
    save_interval = 50
    experiment_name = "sirius_h_moe_end2end"
    empirical_normalization = False
    
    # [Fix] 使用继承后的 Config 类并传入参数
    policy = HierarchicalMoEActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # H-MoE 参数
        num_wheel_experts=3,
        num_leg_experts=3,
        latent_dim=256,
        # 稳定性参数:
        gating_start_temp=1.0,    # 开始时 Temp=1.0，使分布平滑 (防止早期崩塌)
        gating_end_temp=0.05,     # 结束时 Temp=0.05，使分布尖锐 (防止平庸解)
        gating_anneal_steps=2500, # 在前 2500 次迭代中逐渐降温
        gating_noise_std=0.5,     # 加入噪声防止专家饿死
        
        # [New] 技能解耦 Action Masking
        # 假设 16 维动作：前 4 维是轮子，后 12 维是腿 (Hip, Thigh, Calf * 4)
        # 请根据您的实际机器人 URDF 修改这里！
        wheel_indices=(12, 13, 14, 15), 
        leg_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )