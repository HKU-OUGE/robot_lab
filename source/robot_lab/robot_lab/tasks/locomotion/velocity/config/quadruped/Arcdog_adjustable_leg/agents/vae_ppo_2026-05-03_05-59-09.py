import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from dataclasses import field

class ProprioVAE(nn.Module):
    """变分自编码器：负责将历史观测压缩，并预测当前线速度"""
    def __init__(self, input_dim, vel_dim=3, latent_dim=64, hidden_dims=[256, 128]):
        super().__init__()
        # Encoder
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
        self.encoder = nn.Sequential(*layers)
        
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)
        self.vel_estimator = nn.Linear(curr_dim, vel_dim)

        # Decoder
        dec_layers = []
        curr_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(curr_dim, h))
            dec_layers.append(nn.ELU())
            curr_dim = h
        dec_layers.append(nn.Linear(curr_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h), self.vel_estimator(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, vel_pred = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon_x = self.decoder(z) if self.training else None
        return vel_pred, recon_x, mu, logvar, z


class VAEActorCritic(ActorCritic):
    """带有 VAE 状态估计器的 ActorCritic 网络"""
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        vae_latent_dim = kwargs.pop("vae_latent_dim", 64)
        vae_hidden_dims = kwargs.pop("vae_hidden_dims", [256, 128])
        
        policy_keys = obs_groups.get("policy", ["policy"])
        estimator_keys = obs_groups.get("estimator", ["estimator"])
        critic_keys = obs_groups.get("critic", ["critic"])

        policy_obs_dim = obs[policy_keys[0]].shape[-1]
        estimator_obs_dim = obs[estimator_keys[0]].shape[-1]

        # 重新计算 Actor 的输入维度 (Policy Obs + Vel Pred + Latent Z)
        actor_input_dim = policy_obs_dim + 3 + vae_latent_dim
        
        # 构建一个与原始 obs 结构相同的 mock 字典，避免 TensorDict 限制
        obs_mock = {}
        if hasattr(obs, "items"):
            for k, v in obs.items():
                obs_mock[k] = torch.zeros_like(v)
        else:
            obs_mock[policy_keys[0]] = torch.zeros_like(obs)
            
        # 仅仅把 policy 对应的观测维度替换为加上 VAE 特征后的新维度
        batch_size = obs[policy_keys[0]].shape[0] if hasattr(obs, "items") else obs.shape[0]
        device = obs[policy_keys[0]].device if hasattr(obs, "items") else obs.device
        obs_mock[policy_keys[0]] = torch.zeros((batch_size, actor_input_dim), device=device)
        
        # 调用父类初始化
        super().__init__(obs_mock, obs_groups, num_actions, **kwargs)
        
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.policy_keys = policy_keys
        self.estimator_keys = estimator_keys
        self.critic_keys = critic_keys

        # 初始化 VAE
        self.estimator = ProprioVAE(
            input_dim=estimator_obs_dim, 
            vel_dim=3, 
            latent_dim=self.vae_latent_dim, 
            hidden_dims=self.vae_hidden_dims
        ).to(device)

    def _process_obs(self, obs_dict):
        """将观测字典拆分，过 VAE，并拼接给 Actor"""
        policy_obs = obs_dict[self.policy_keys[0]]
        estimator_obs = obs_dict[self.estimator_keys[0]]
        
        vel_pred, recon_x, mu, logvar, z = self.estimator(estimator_obs)
        
        if self.training:
            self.active_vae_recon = recon_x
            self.active_vae_mu = mu
            self.active_vae_logvar = logvar
            self.active_vae_vel_pred = vel_pred
            self.active_vae_input = estimator_obs
            
        actor_input = torch.cat([policy_obs, vel_pred, z], dim=-1)
        return actor_input

    def act(self, obs, **kwargs):
        actor_input = self._process_obs(obs)
        
        # 【修复】：将 TensorDict 转换为普通的 Python 字典，
        # 避免因为 actor_input 维度变长而触发 TensorDict 的 Shape 校验报错
        if hasattr(obs, "items"):
            mock_obs = {k: v for k, v in obs.items()}
        else:
            mock_obs = {}
            
        mock_obs[self.policy_keys[0]] = actor_input
        return super().act(mock_obs, **kwargs)

    def act_inference(self, obs, **kwargs):
        actor_input = self._process_obs(obs)
        
        if hasattr(obs, "items"):
            mock_obs = {k: v for k, v in obs.items()}
        else:
            mock_obs = {}
            
        mock_obs[self.policy_keys[0]] = actor_input
        return super().act_inference(mock_obs, **kwargs)


class VAEPPO(PPO):
    """重写 PPO 的更新逻辑，加入 VAE 的 Loss 计算"""
    def update(self):
        # 1. 先执行标准的 PPO 更新 (Actor 和 Critic)
        loss_dict = super().update()
        
        model = self.policy
        
        if model.estimator is not None and self.num_learning_epochs > 0:
            self.optimizer.zero_grad()
            
            obs_storage = self.storage.observations
            
            # 【修复】：使用 hasattr(..., "keys") 兼容 dict 和 TensorDict
            # 这样就能正确提取出内部的 Tensor，而不是把整个 TensorDict 传给网络
            if hasattr(obs_storage, "keys"):
                est_input = obs_storage[model.estimator_keys[0]].flatten(0, 1)
                critic_obs = obs_storage[model.critic_keys[0]].flatten(0, 1)
            else:
                obs_batch = obs_storage.flatten(0, 1)
                est_input = obs_batch
                critic_obs = obs_batch
            
            # 假设 critic 组的前 3 维就是真实的 base_lin_vel
            target_vel = critic_obs[..., 0:3] 
            
            # VAE 前向传播
            vel_pred, recon_x, mu, logvar, z = model.estimator(est_input)
            
            # 计算 Loss
            vel_loss = (vel_pred - target_vel).pow(2).mean()
            recon_loss = (recon_x - est_input).pow(2).mean()
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            
            # 权重系数可以自己调
            beta = 0.01 
            vae_loss = 1.0 * vel_loss + 1.0 * recon_loss + beta * kl_loss
            
            # 反向传播 VAE 独有的 Loss
            vae_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # 记录日志
            loss_dict["Loss/VAE_Vel_MSE"] = vel_loss.item()
            loss_dict["Loss/VAE_Recon_MSE"] = recon_loss.item()
            loss_dict["Loss/VAE_KL"] = kl_loss.item()
            
        return loss_dict


@configclass
class VAEActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "VAEActorCritic"
    vae_latent_dim: int = 64
    vae_hidden_dims: list = field(default_factory=lambda: [256, 128])


@configclass
class VAEPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "VAEPPO"