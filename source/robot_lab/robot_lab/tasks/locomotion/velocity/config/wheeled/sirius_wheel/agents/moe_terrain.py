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

# === Debug Switch ===
ENABLE_DEBUG_PRINT = False

def debug_print(tag, tensor_info):
    if ENABLE_DEBUG_PRINT:
        print(f"[DEBUG][{tag}] {tensor_info}")

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

class SharedBackboneMoEActorCritic(ActorCritic):
    """
    [Teacher Policy] 共享主干 (Shared Backbone & Gate) H-MoE 架构
    适用于 Teacher 策略：Actor 和 Critic 输入一致（都包含特权信息），共享 GRU 和 Gate。
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
                 num_wheel_experts=2, 
                 num_leg_experts=2,   
                 latent_dim=None,     
                 # === 训练稳定性参数 ===
                 gating_start_temp=1.0,    
                 gating_end_temp=0.1,      
                 gating_anneal_steps=2000, 
                 gating_noise_std=0.5,     
                 # === 动作掩码参数 ===
                 wheel_indices=None,       
                 leg_indices=None,         
                 **kwargs):
        
        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **kwargs)

        # 1. 处理输入维度 (假设 Actor 和 Critic 输入一致)
        self.input_keys = None 
        is_dict_like = isinstance(obs, dict) or hasattr(obs, "keys")

        if is_dict_like:
            # 优先查找 policy 键，如果没有则使用全部
            keys = obs_groups.get("policy", None)
            if keys is None and "actor" in obs_groups:
                keys = obs_groups["actor"]
            
            # 如果配置里没有明确指定，默认使用第一个键（通常是 'obs'）
            if keys is None:
                keys = [next(iter(obs.keys()))]
            
            self.input_keys = keys
            num_obs = sum(obs[k].shape[-1] for k in keys)
        else:
            num_obs = obs.shape[-1]

        self.latent_dim = latent_dim if latent_dim is not None else 256
        # [修改] hidden_state_dim 只有一份，因为共享 GRU
        self.hidden_state_dim = self.latent_dim
        
        self._valid_batch_size = None
        
        if is_dict_like:
             ref_tensor = obs[self.input_keys[0] if self.input_keys else next(iter(obs.keys()))]
        else:
             ref_tensor = obs
        
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        # [修改] 只需要初始化一份 hidden state
        self.active_hidden_states = torch.zeros(1, batch_size, self.hidden_state_dim, device=device)
        
        # Gating annealing 参数
        self.gating_start_temp = gating_start_temp
        self.gating_end_temp = gating_end_temp
        self.gating_anneal_steps = gating_anneal_steps
        self.gating_noise_std = gating_noise_std
        self.current_gating_temp = gating_start_temp
        self.anneal_counter = 0
        
        # Action Masks
        self.register_buffer('wheel_action_mask', torch.ones(num_actions))
        self.register_buffer('leg_action_mask', torch.ones(num_actions))
        self.use_action_masking = False
        if wheel_indices is not None and leg_indices is not None:
            self.use_action_masking = True
            self.wheel_action_mask.fill_(0.0)
            self.leg_action_mask.fill_(0.0)
            wheel_indices_list = list(wheel_indices)
            self.wheel_action_mask[wheel_indices_list] = 1.0
            leg_indices_list = list(leg_indices)
            self.leg_action_mask[leg_indices_list] = 1.0

        # === SHARED BACKBONE & GATES (共享部分) ===
        self.gru = nn.GRU(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)
        
        # 共享的 Gate 网络
        self.mode_gate = nn.Sequential(nn.Linear(self.latent_dim, 64), nn.ELU(), nn.Linear(64, 2))
        self.wheel_gate = nn.Linear(self.latent_dim, num_wheel_experts)
        self.leg_gate = nn.Linear(self.latent_dim, num_leg_experts)

        # === ACTOR EXPERTS (独立部分) ===
        self.actor_wheel_experts = nn.ModuleList([MLP(self.latent_dim, num_actions, hidden_dims=actor_hidden_dims) for _ in range(num_wheel_experts)])
        self.actor_leg_experts = nn.ModuleList([MLP(self.latent_dim, num_actions, hidden_dims=actor_hidden_dims) for _ in range(num_leg_experts)])

        # === CRITIC EXPERTS (独立部分) ===
        # Critic experts 输出维度为 1 (Value)
        self.critic_wheel_experts = nn.ModuleList([MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims) for _ in range(num_wheel_experts)])
        self.critic_leg_experts = nn.ModuleList([MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims) for _ in range(num_leg_experts)])
        
        self.critic = nn.Identity() 

    @property
    def is_recurrent(self):
        return True
    
    @is_recurrent.setter
    def is_recurrent(self, value):
        pass

    def update_annealing(self):
        self.anneal_counter += 1
        if self.anneal_counter < self.gating_anneal_steps:
            alpha = self.anneal_counter / self.gating_anneal_steps
            self.current_gating_temp = self.gating_start_temp - alpha * (self.gating_start_temp - self.gating_end_temp)
        else:
            self.current_gating_temp = self.gating_end_temp

    def _gated_softmax(self, logits):
        if self.training and self.gating_noise_std > 0.0:
            noise = torch.randn_like(logits) * self.gating_noise_std
            logits = logits + noise
        temp = max(self.current_gating_temp, 1e-4)
        return torch.softmax(logits / temp, dim=-1)

    # ----------------------------------------------------------------
    # Helper: 计算权重 (Gate Forward)
    # ----------------------------------------------------------------
    def _compute_weights(self, latent):
        """根据 latent 计算共享的 gating weights"""
        mode_logits = self.mode_gate(latent)
        mode_weights = self._gated_softmax(mode_logits)
        w_wheel_group = mode_weights[..., 0].unsqueeze(-1)
        w_leg_group   = mode_weights[..., 1].unsqueeze(-1)

        wheel_gate_logits = self.wheel_gate(latent)
        wheel_expert_weights = self._gated_softmax(wheel_gate_logits)

        leg_gate_logits = self.leg_gate(latent)
        leg_expert_weights = self._gated_softmax(leg_gate_logits)
        
        return w_wheel_group, w_leg_group, wheel_expert_weights, leg_expert_weights

    # ----------------------------------------------------------------
    # Helper: 应用专家 (Apply Experts)
    # ----------------------------------------------------------------
    def _apply_experts(self, latent, w_wheel_group, w_leg_group, w_wheel_experts, w_leg_experts, 
                       wheel_experts_list, leg_experts_list, is_actor=True):
        """应用权重到指定的专家列表"""
        
        # 1. Wheel Experts Weighted Sum
        wheel_sum = 0
        for i, expert in enumerate(wheel_experts_list):
            out = expert(latent)
            if is_actor and self.use_action_masking:
                out = out * self.wheel_action_mask
            wheel_sum += out * w_wheel_experts[..., i].unsqueeze(-1)

        # 2. Leg Experts Weighted Sum
        leg_sum = 0
        for i, expert in enumerate(leg_experts_list):
            out = expert(latent)
            if is_actor and self.use_action_masking:
                out = out * self.leg_action_mask
            leg_sum += out * w_leg_experts[..., i].unsqueeze(-1)

        # 3. Mode Weighted Sum
        output = w_wheel_group * wheel_sum + w_leg_group * leg_sum
        return output

    # ----------------------------------------------------------------
    # Helper: GRU Forward (复用之前的逻辑)
    # ----------------------------------------------------------------
    def _run_gru(self, gru_module, x_in, hidden_states, masks, valid_batch_size):
        # 保持与原始代码一致的切片和转置逻辑
        if valid_batch_size is not None:
            if x_in.ndim == 3 and x_in.shape[1] > valid_batch_size: 
                x_in = x_in[:, :valid_batch_size, :]
            elif x_in.ndim == 2 and x_in.shape[0] > valid_batch_size:
                x_in = x_in[:valid_batch_size, :]
            if hidden_states.shape[1] > valid_batch_size:
                hidden_states = hidden_states[:, :valid_batch_size, :]
            if masks is not None:
                if masks.ndim == 3 and masks.shape[1] > valid_batch_size:
                     masks = masks[:, :valid_batch_size, :]
                elif masks.ndim == 2 and masks.shape[1] > valid_batch_size:
                     masks = masks[:, :valid_batch_size]

        if hidden_states.shape[0] > hidden_states.shape[1] and hidden_states.shape[1] == gru_module.num_layers:
             hidden_states = hidden_states.transpose(0, 1).contiguous()
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        latent = None
        next_hidden = None

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
            
            gru_out, next_hidden = gru_module(x_gru, hidden_states)
            latent = gru_out.transpose(0, 1) if is_input_transposed else gru_out

        elif x_in.ndim == 2: # Step
            num_envs_hidden = hidden_states.shape[1]
            batch_size_in = x_in.shape[0]
            if batch_size_in == num_envs_hidden:
                x_gru = x_in.unsqueeze(0)
                if masks is not None:
                     hidden_states = hidden_states * masks.reshape(1, -1, 1)
                gru_out, next_hidden = gru_module(x_gru, hidden_states)
                latent = gru_out[0]
            else:
                if batch_size_in % num_envs_hidden == 0:
                    seq_len = batch_size_in // num_envs_hidden
                    x_gru = x_in.reshape(seq_len, num_envs_hidden, -1)
                    gru_out, next_hidden = gru_module(x_gru, hidden_states)
                    latent = gru_out.reshape(-1, self.latent_dim)
                else:
                    x_gru = x_in.unsqueeze(0)
                    gru_out, next_hidden = gru_module(x_gru, hidden_states)
                    latent = gru_out[0]
        else:
            latent = x_in
            next_hidden = hidden_states
            
        return latent, next_hidden

    # ----------------------------------------------------------------
    # Actor Forward
    # ----------------------------------------------------------------
    def forward(self, obs, masks=None, hidden_states=None, save_dist=True, valid_batch_size=None):
        # 准备输入
        if self.input_keys is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in self.input_keys]
            x_in = torch.cat(tensors, dim=-1)
        else:
            x_in = obs

        if hidden_states is None:
            hidden_states = torch.zeros(1, x_in.shape[0], self.latent_dim, device=x_in.device)
            
        # 1. 运行共享 GRU
        latent, next_hidden = self._run_gru(self.gru, x_in, hidden_states, masks, valid_batch_size)
        
        # 2. 计算共享权重
        w_mode_0, w_mode_1, w_wheel, w_leg = self._compute_weights(latent)
        
        # 3. 使用 Actor 专家计算动作
        actions_mean = self._apply_experts(latent, w_mode_0, w_mode_1, w_wheel, w_leg,
                                           self.actor_wheel_experts, self.actor_leg_experts,
                                           is_actor=True)
        
        if save_dist:
            self.distribution = torch.distributions.Normal(actions_mean, self.std)

        return actions_mean, self.std, next_hidden

    # ----------------------------------------------------------------
    # Evaluate (Critic)
    # ----------------------------------------------------------------
    def evaluate(self, obs, masks=None, hidden_states=None, hidden_state=None):
        if hidden_states is None and hidden_state is not None:
             hidden_states = hidden_state

        valid_batch_size = None
        if (masks is not None or hidden_states is not None) and self._valid_batch_size is not None:
            valid_batch_size = self._valid_batch_size

        # 准备输入 (逻辑与 forward 相同，因为输入一致)
        if self.input_keys is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in self.input_keys]
            x_in = torch.cat(tensors, dim=-1)
        else:
            x_in = obs
        
        if hidden_states is None:
             hidden_states = self.active_hidden_states

        # 1. 运行共享 GRU
        latent, next_hidden = self._run_gru(self.gru, x_in, hidden_states, masks, valid_batch_size)

        # 2. 计算共享权重 (Crucial: 这里使用与 Actor 完全相同的逻辑和参数)
        w_mode_0, w_mode_1, w_wheel, w_leg = self._compute_weights(latent)

        # 3. 使用 Critic 专家计算价值
        value = self._apply_experts(latent, w_mode_0, w_mode_1, w_wheel, w_leg,
                                    self.critic_wheel_experts, self.critic_leg_experts,
                                    is_actor=False)
        
        # 更新 hidden states 缓存 (如果是 active_hidden_states 引用)
        if hidden_states is self.active_hidden_states:
             self.active_hidden_states = next_hidden.detach()

        return value

    def get_hidden_states(self):
        # 现在只需要返回一份 hidden states
        return self.active_hidden_states, self.active_hidden_states

    def get_actions_log_prob(self, actions):
        if actions is not None:
             if actions.ndim == 3: 
                 self._valid_batch_size = actions.shape[1]
             else: 
                 pass 

        if actions is not None and self.distribution is not None:
            dist_batch_size = self.distribution.batch_shape[1]
            action_batch_size = actions.shape[1]
            
            if dist_batch_size != action_batch_size:
                target_batch = min(dist_batch_size, action_batch_size)
                mean = self.distribution.loc[:, :target_batch, :]
                std = self.distribution.scale[:, :target_batch, :]
                self.distribution = torch.distributions.Normal(mean, std)

        return super().get_actions_log_prob(actions)

    def act(self, obs, masks=None, hidden_state=None):
        self._valid_batch_size = None
        if self.actor_obs_normalization:
            obs = self.norm_actor_obs(obs)
        
        if hidden_state is None:
            current_state = self.active_hidden_states
        else:
            current_state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
        
        actions_mean, actions_std, next_hidden_states = self.forward(obs, masks, current_state, save_dist=True)
        
        if hidden_state is None:
            self.active_hidden_states = next_hidden_states.detach() 
        
        dist = torch.distributions.Normal(actions_mean, actions_std)
        self.distribution = dist
        return dist.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        self._valid_batch_size = None
        if self.actor_obs_normalization:
            obs = self.norm_actor_obs(obs)
        if hidden_states is None:
            hidden_states = self.active_hidden_states
        
        actions_mean, _, next_hidden_states = self.forward(obs, masks, hidden_states, save_dist=False)
        self.active_hidden_states = next_hidden_states
        return actions_mean

# === Config ===
@configclass
class SharedBackboneMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SharedBackboneMoEActorCritic"
    num_wheel_experts: int = 2
    num_leg_experts: int = 2
    latent_dim: int = 256
    gating_start_temp: float = 1.0
    gating_end_temp: float = 0.1
    gating_anneal_steps: int = 2000
    gating_noise_std: float = 0.5
    wheel_indices: tuple = None 
    leg_indices: tuple = None

@configclass
class SiriusSharedMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 5000 
    save_interval = 50
    experiment_name = "sirius_h_moe_shared_backbone"
    empirical_normalization = False
    
    policy = SharedBackboneMoEActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        num_wheel_experts=2,
        num_leg_experts=2,
        latent_dim=256,
        gating_start_temp=1.0,    
        gating_end_temp=0.05,     
        gating_anneal_steps=2500, 
        gating_noise_std=0.5,     
        wheel_indices=(12, 13, 14, 15), 
        leg_indices=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )