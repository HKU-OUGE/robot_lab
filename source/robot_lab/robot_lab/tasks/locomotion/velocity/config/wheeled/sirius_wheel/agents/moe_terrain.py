# source/robot_lab/algo/moe/policy_split_moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories
from rsl_rl.algorithms import PPO
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from dataclasses import field
import numpy as np

# === 1. 基础组件 (参考 moe_simple.py) ===

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128], activation="elu", output_gain=1.0):
        super().__init__()
        layers = []
        prev_dim = input_dim
        act_fn = nn.ELU() if activation == "elu" else nn.ReLU()
        
        for h_dim in hidden_dims:
            layer = nn.Linear(prev_dim, h_dim)
            orthogonal_init(layer, gain=np.sqrt(2)) # 标准初始化
            layers.append(layer)
            layers.append(act_fn)
            prev_dim = h_dim
        
        out_layer = nn.Linear(prev_dim, output_dim)
        orthogonal_init(out_layer, gain=output_gain) # 输出层增益控制
        layers.append(out_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SplitMoEActorCritic(ActorCritic):
    """
    [Split-MoE] 解耦并行混合专家架构
    结构：GRU -> (Leg Gate + Leg Experts) || (Wheel Gate + Wheel Experts) -> Concat
    """
    is_recurrent = True

    def __init__(self, 
                 obs, 
                 obs_groups, 
                 num_actions, 
                 actor_hidden_dims=[256, 128, 128], 
                 critic_hidden_dims=[512, 256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 # === Split MoE 参数 ===
                 num_wheel_experts=2, 
                 num_leg_experts=2,
                 num_leg_actions=12,   # 显式指定腿部动作维度 (通常为12)
                 latent_dim=256,
                 rnn_type="gru",
                 aux_loss_coef=0.01,
                 **kwargs):
        
        super().__init__(obs, obs_groups, num_actions, 
                         actor_hidden_dims=actor_hidden_dims, 
                         critic_hidden_dims=critic_hidden_dims, 
                         activation=activation, 
                         init_noise_std=init_noise_std, 
                         **kwargs)

        # 1. 输入处理
        self.input_keys = None 
        if isinstance(obs, dict) or hasattr(obs, "keys"):
            keys = obs_groups.get("policy", None)
            self.input_keys = keys
            num_obs = sum(obs[k].shape[-1] for k in keys)
        else:
            num_obs = obs.shape[-1]

        self.latent_dim = latent_dim
        self.rnn_type = rnn_type.lower()
        self.aux_loss_coef = aux_loss_coef
        

        # === 动作空间拆分 ===
        self.num_leg_actions = num_leg_actions
        self.num_wheel_actions = num_actions - num_leg_actions
        
        if self.num_wheel_actions < 0:
            raise ValueError(f"num_leg_actions ({num_leg_actions}) cannot be larger than total actions ({num_actions})")

        print(f"[SplitMoE] Actions Split: Legs={self.num_leg_actions}, Wheels={self.num_wheel_actions}")

        # === 2. 共享 RNN Backbone ===
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)
        else:
            self.rnn = nn.GRU(input_size=num_obs, hidden_size=self.latent_dim, batch_first=False)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)
            elif 'bias' in name: nn.init.constant_(param, 0)

        # === 3. 并行 Gating Networks ===
        # 使用 LayerNorm 稳定 Gate 输入
        self.gate_input_norm = nn.LayerNorm(self.latent_dim)

        # Leg Gate
        self.leg_gate = nn.Sequential(
            nn.Linear(self.latent_dim, 64), 
            nn.ELU(), 
            nn.Linear(64, num_leg_experts)
        )
        
        # Wheel Gate
        self.wheel_gate = nn.Sequential(
            nn.Linear(self.latent_dim, 64), 
            nn.ELU(), 
            nn.Linear(64, num_wheel_experts)
        )

        self._init_gate(self.leg_gate)
        self._init_gate(self.wheel_gate)

        # === 4. 并行专家组 ===
        # 注意：输出维度分别为 num_leg_actions 和 num_wheel_actions
        
        self.actor_leg_experts = nn.ModuleList([
            MLP(self.latent_dim, self.num_leg_actions, hidden_dims=actor_hidden_dims, activation=activation, output_gain=0.01) 
            for _ in range(num_leg_experts)
        ])

        self.actor_wheel_experts = nn.ModuleList([
            MLP(self.latent_dim, self.num_wheel_actions, hidden_dims=actor_hidden_dims, activation=activation, output_gain=0.01) 
            for _ in range(num_wheel_experts)
        ])

        # === 5. 独立 Critic (保持原设计) ===
        self.critic_mlp = MLP(self.latent_dim, 1, hidden_dims=critic_hidden_dims, activation=activation, output_gain=1.0)

        # 运行时状态
        if isinstance(obs, dict): ref_tensor = obs[list(obs.keys())[0]]
        else: ref_tensor = obs
        batch_size = ref_tensor.shape[0]
        device = ref_tensor.device
        
        self.active_hidden_states = self._init_rnn_state(batch_size, device)
        
        # Logging & Loss
        self.latest_weights = {}
        self.active_aux_loss = 0.0
        # === 6. 初始化动作噪声 (分腿轮设置不同噪声) ===
        new_std = torch.ones(num_actions)
        self.num_wheel_experts = num_wheel_experts
        self.num_leg_experts = num_leg_experts
        noise_legs = kwargs.get("init_noise_legs", 1.0)
        noise_wheels = kwargs.get("init_noise_wheels", 0.4) # 轮子默认给小点
        print(f"[SplitMoE] Overriding Noise: Legs={noise_legs}, Wheels={noise_wheels}")
        if num_leg_actions <= num_actions:
            new_std[:num_leg_actions] = noise_legs
            new_std[num_leg_actions:] = noise_wheels
        else:
            print("[Warning] num_leg_actions > num_actions, skipping noise override.")
        self.std.data.copy_(new_std.to(device))
    def _init_rnn_state(self, batch_size, device):
        if self.rnn_type == "lstm":
            h = torch.zeros(1, batch_size, self.latent_dim, device=device)
            c = torch.zeros(1, batch_size, self.latent_dim, device=device)
            return (h, c)
        else:
            return torch.zeros(1, batch_size, self.latent_dim, device=device)

    def _init_gate(self, gate_net):
        orthogonal_init(gate_net[0], gain=np.sqrt(2))
        orthogonal_init(gate_net[2], gain=0.01) # 最后一层小增益，使初始权重接近均匀

    def _prepare_input(self, obs, key_list):
        if key_list is not None and (isinstance(obs, dict) or hasattr(obs, "keys")):
            tensors = [obs[k] for k in key_list]
            return torch.cat(tensors, dim=-1)
        return obs
    
    def _run_rnn(self, rnn_module, x_in, hidden_states, masks):
        if hidden_states is None:
            B = x_in.shape[1] if x_in.ndim == 3 else x_in.shape[0]
            rnn_state = self._init_rnn_state(B, x_in.device)
        else:
            rnn_state = hidden_states

        if x_in.ndim == 3: # Training (Batch Mode)
            rnn_out, next_rnn_state = rnn_module(x_in, rnn_state)
            if masks is not None:
                # 使用 unpad_trajectories 处理掩码
                latent = unpad_trajectories(rnn_out, masks)
            else:
                latent = rnn_out
        elif x_in.ndim == 2: # Inference Mode
            x_rnn = x_in.unsqueeze(0)
            if masks is not None:
                m = masks.view(1, -1, 1)
                if self.rnn_type == "lstm":
                    rnn_state = (rnn_state[0] * m, rnn_state[1] * m)
                else:
                    rnn_state = rnn_state * m
            rnn_out, next_rnn_state = rnn_module(x_rnn, rnn_state)
            latent = rnn_out[0]
        
        return latent, next_rnn_state

    def _compute_actor_output(self, latent):
        gate_in = self.gate_input_norm(latent)
        
        # === 1. Calculate Gate Weights ===
        leg_logits = self.leg_gate(gate_in)
        w_leg = F.softmax(leg_logits, dim=-1)
        
        wheel_logits = self.wheel_gate(gate_in)
        w_wheel = F.softmax(wheel_logits, dim=-1)

        # === 2. Calculate Expert Outputs ===
        
        # Leg Branch (Action dim: num_leg_actions)
        leg_act_sum = 0
        for i, expert in enumerate(self.actor_leg_experts):
            leg_act_sum += expert(latent) * w_leg[..., i].unsqueeze(-1)
            
        # Wheel Branch (Action dim: num_wheel_actions)
        wheel_act_sum = 0
        for i, expert in enumerate(self.actor_wheel_experts):
            wheel_act_sum += expert(latent) * w_wheel[..., i].unsqueeze(-1)
            
        # === 3. Fusion (Concatenation) ===
        # 假设动作空间顺序是 [Legs..., Wheels...]，这通常是 Isaac Lab 的默认顺序
        # 如果反了，交换这里的顺序即可
        total_action = torch.cat([leg_act_sum, wheel_act_sum], dim=-1)

        # === 4. Logging & Aux Loss ===
        if self.training:
            self.active_aux_loss = self._calculate_load_balancing_loss(w_leg, w_wheel) * self.aux_loss_coef
        
        with torch.no_grad():
            def flat_mean(w): return w.reshape(-1, w.shape[-1]).mean(dim=0).detach()
            self.latest_weights = {
                "leg": flat_mean(w_leg),
                "wheel": flat_mean(w_wheel)
            }

        return total_action

    def _calculate_load_balancing_loss(self, w_leg, w_wheel):
        loss = 0.0
        # 目标：均匀分布
        leg_usage = w_leg.reshape(-1, w_leg.shape[-1]).mean(dim=0)
        target_leg = torch.full_like(leg_usage, 1.0 / self.num_leg_experts)
        loss += (leg_usage - target_leg).pow(2).sum()

        wheel_usage = w_wheel.reshape(-1, w_wheel.shape[-1]).mean(dim=0)
        target_wheel = torch.full_like(wheel_usage, 1.0 / self.num_wheel_experts)
        loss += (wheel_usage - target_wheel).pow(2).sum()
        return loss

    def _prepare_hidden_state(self, hidden, device):
        if hidden is None: return None
        if isinstance(hidden, (tuple, list)):
            if len(hidden) == 2 and not self.rnn_type == "lstm": hidden = hidden[0]
            elif self.rnn_type == "lstm" and len(hidden) == 2:
                if isinstance(hidden[0], (tuple, list)): hidden = hidden[0]
        def to_dev(h):
            if isinstance(h, (tuple, list)): return tuple(x.to(device).contiguous() for x in h)
            return h.to(device).contiguous()
        return to_dev(hidden)

    # === Standard Interface ===

    def act(self, obs, masks=None, hidden_state=None):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_state is None: self.active_hidden_states = next_state
            
        mean = self._compute_actor_output(latent)
        self.distribution = torch.distributions.Normal(mean, self.std)
        return self.distribution.sample()

    def act_inference(self, obs, masks=None, hidden_states=None):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
             
        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        if hidden_states is None: self.active_hidden_states = next_state
        return self._compute_actor_output(latent)

    def evaluate(self, obs, masks=None, hidden_state=None):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)
        
        current_state = self._prepare_hidden_state(hidden_state, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)
        
        latent, _ = self._run_rnn(self.rnn, x_in, current_state, masks)
        return self.critic_mlp(latent)
    
    def forward(self, obs, masks=None, hidden_states=None, save_dist=True):
        x_in = self._prepare_input(obs, self.input_keys)
        if self.actor_obs_normalization: x_in = self.actor_obs_normalizer(x_in)

        current_state = self._prepare_hidden_state(hidden_states, x_in.device)
        if current_state is None: current_state = self._prepare_hidden_state(self.active_hidden_states, x_in.device)

        latent, next_state = self._run_rnn(self.rnn, x_in, current_state, masks)
        action_mean = self._compute_actor_output(latent)
        
        if save_dist: self.distribution = torch.distributions.Normal(action_mean, self.std)
        return action_mean, self.std, next_state

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def get_hidden_states(self):
        return self.active_hidden_states, self.active_hidden_states
    
    def reset(self, dones=None):
        if dones is None: return
        def reset_hidden(h, mask):
            if isinstance(h, tuple): return tuple(reset_hidden(x, mask) for x in h)
            else: 
                h[:, mask, :] = 0.0
                return h
        if self.active_hidden_states is not None:
            self.active_hidden_states = reset_hidden(self.active_hidden_states, dones)

# === PPO Class with Logging ===
class SplitMoEPPO(PPO):
    def update(self):
        loss_dict = super().update()
        if hasattr(self.policy, "latest_weights") and self.policy.latest_weights:
            w = self.policy.latest_weights
            if "leg" in w:
                for i, val in enumerate(w["leg"]): loss_dict[f"Gate/Leg_Expert_{i}"] = val.item()
            if "wheel" in w:
                for i, val in enumerate(w["wheel"]): loss_dict[f"Gate/Wheel_Expert_{i}"] = val.item()
        if hasattr(self.policy, "active_aux_loss"):
            if isinstance(self.policy.active_aux_loss, torch.Tensor):
                 loss_dict["Loss/Load_Balancing"] = self.policy.active_aux_loss.item()
        if hasattr(self.policy, "std"):
            # 转移到 CPU 计算
            std_np = self.policy.std.detach().cpu().numpy()
            
            # 获取切分点 (默认 12)
            n_legs = getattr(self.policy, "num_leg_actions", 12)
            
            # 安全切片计算
            if len(std_np) >= n_legs:
                leg_val = std_np[:n_legs].mean()
                wheel_val = std_np[n_legs:].mean() if len(std_np) > n_legs else 0.0
                
                loss_dict["Noise/Leg_Std"] = leg_val
                loss_dict["Noise/Wheel_Std"] = wheel_val
        return loss_dict

# === Configs ===

@configclass
class SplitMoEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "SplitMoEActorCritic"
    num_wheel_experts: int = 6
    num_leg_experts: int = 6
    # === 关键设置：请根据机器人实际关节数修改 ===
    num_leg_actions: int = 12
    init_noise_std: float = 1.0
    init_noise_legs: float = 1.0
    init_noise_wheels: float = 0.5
    latent_dim: int = 256
    rnn_type: str = "gru"
    aux_loss_coef: float = 0.01
    
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    
    actor_hidden_dims: list = field(default_factory=lambda: [256, 128, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    def get_std(self):
            """
            Returns the current mean std for legs and wheels separately.
            Assuming actions are ordered [Legs..., Wheels...]
            """
            # self.std 形状是 [num_actions]
            # 确保数据在 CPU 上以便打印
            std_np = self.std.detach().cpu().numpy()
            
            # 1. 腿部噪声 (前 num_leg_actions 维)
            if self.num_leg_actions > 0:
                leg_std = std_np[:self.num_leg_actions].mean()
            else:
                leg_std = 0.0
                
            # 2. 轮子噪声 (剩余维度)
            if self.num_leg_actions < len(std_np):
                wheel_std = std_np[self.num_leg_actions:].mean()
            else:
                wheel_std = 0.0
                
            return leg_std, wheel_std
@configclass
class SiriusSplitMoEPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 50000
    save_interval = 200
    experiment_name = "sirius_split_moe_parallel" 
    empirical_normalization = False
    
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    
    policy = SplitMoEActorCriticCfg(
        init_noise_std=1.0,  # 作为一个默认基准（会被下面覆盖）
        init_noise_legs=0.8,  # 腿部建议保持 0.8 ~ 1.0
        init_noise_wheels=0.5,   # 轮子建议给小一点，配合大 Scale
        actor_hidden_dims=[256, 128, 128], 
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # === 并行专家设置 ===
        num_wheel_experts=6,
        num_leg_experts=6,
        num_leg_actions=12, # 12个腿部关节，剩余的自动分配给轮子
        latent_dim=256,
        rnn_type="gru",
        aux_loss_coef=0.01,
        actor_obs_normalization=True, 
        critic_obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="SplitMoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )