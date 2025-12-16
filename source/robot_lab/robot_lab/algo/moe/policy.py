# source/robot_lab/algo/moe/policy.py

import torch
import torch.nn as nn
import os
from rsl_rl.modules import ActorCritic

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

class MoEActorCritic(ActorCritic):
    """
    自定义的 MoE 策略类，继承自 RSL-RL 的标准 ActorCritic。
    支持加载预训练的专家权重并进行冻结/解冻操作。
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 128], 
                 critic_hidden_dims=[256, 128], 
                 activation='elu', 
                 init_noise_std=1.0,
                 # === MoE 特定参数 ===
                 checkpoint_wheel=None, 
                 checkpoint_leg=None,   
                 freeze_experts=True,
                 **kwargs):
        
        super().__init__(num_actor_obs, num_critic_obs, num_actions, 
                         actor_hidden_dims, critic_hidden_dims, 
                         activation, init_noise_std, **kwargs)

        # 1. 重构 Actor 部分
        # 假设 GRU 的 Hidden State 维度等于输入 Observation 维度，以便兼容预训练专家
        self.hidden_state_dim = num_actor_obs 
        
        # 共享 GRU
        self.gru = nn.GRU(input_size=num_actor_obs, 
                          hidden_size=self.hidden_state_dim, 
                          batch_first=True)

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(self.hidden_state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 2) # [Wheel, Leg]
        )

        # 专家网络 (结构必须与预训练的 actor_hidden_dims 一致)
        self.expert_wheel = MLP(self.hidden_state_dim, num_actions, hidden_dims=actor_hidden_dims)
        self.expert_leg = MLP(self.hidden_state_dim, num_actions, hidden_dims=actor_hidden_dims)

        print(f"[MoE] Initialized. Freeze Experts: {freeze_experts}")

        # 2. 加载权重
        if checkpoint_wheel:
            self._load_expert_weights(self.expert_wheel, checkpoint_wheel, "Wheel")
        if checkpoint_leg:
            self._load_expert_weights(self.expert_leg, checkpoint_leg, "Leg")

        # 3. 冻结参数
        if freeze_experts:
            self.freeze_module(self.expert_wheel)
            self.freeze_module(self.expert_leg)
            print("[MoE] Experts are FROZEN. Only GRU and Router will update.")

    def _load_expert_weights(self, target_module, checkpoint_path, name):
        if not os.path.exists(checkpoint_path):
            print(f"[MoE] Warning: {name} Checkpoint not found at {checkpoint_path}")
            return
        
        try:
            loaded_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = loaded_dict.get('model_state_dict', loaded_dict)
            
            # 键名映射: actor.0.weight -> net.0.weight
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('actor.'):
                    name_in_mlp = k.replace('actor.', 'net.') 
                    new_state_dict[name_in_mlp] = v
            
            target_module.load_state_dict(new_state_dict, strict=True)
            print(f"[MoE] Successfully loaded {name} Expert weights.")
        except Exception as e:
            print(f"[MoE] Error loading {name} weights: {e}")

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_experts(self):
        print("[MoE] Unfreezing Experts for Finetuning...")
        for param in self.expert_wheel.parameters():
            param.requires_grad = True
        for param in self.expert_leg.parameters():
            param.requires_grad = True

    def forward(self, obs, masks=None, hidden_states=None):
        # GRU Forward
        x = obs.unsqueeze(1)
        if hidden_states is None:
            hidden_states = torch.zeros(1, obs.shape[0], self.hidden_state_dim, device=obs.device)
        if masks is not None:
             hidden_states = hidden_states * masks.view(1, -1, 1)
        
        gru_out, next_hidden_states = self.gru(x, hidden_states)
        latent = gru_out[:, -1, :] 

        # Router Forward
        router_logits = self.router(latent)
        weights = torch.softmax(router_logits, dim=-1)
        
        # Experts Forward
        act_wheel = self.expert_wheel(latent)
        act_leg = self.expert_leg(latent)

        # Mixture
        w_wheel = weights[:, 0].unsqueeze(-1)
        w_leg   = weights[:, 1].unsqueeze(-1)
        actions_mean = w_wheel * act_wheel + w_leg * act_leg

        return actions_mean, self.std, next_hidden_states

    # RSL-RL 必需接口
    def act(self, obs, masks=None, hidden_states=None):
        actions_mean, _, next_hidden_states = self.forward(obs, masks, hidden_states)
        return actions_mean, next_hidden_states

    def act_inference(self, obs, masks=None, hidden_states=None):
        actions_mean, _, next_hidden_states = self.forward(obs, masks, hidden_states)
        return actions_mean, next_hidden_states
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        # 简单起见，Critic 依然使用标准 MLP，不涉及 MoE
        return self.critic(critic_observations)