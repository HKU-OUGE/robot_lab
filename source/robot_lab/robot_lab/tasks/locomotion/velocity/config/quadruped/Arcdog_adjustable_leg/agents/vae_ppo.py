import copy
import hashlib
import io
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO


HIGHSTEP_DIAGONAL_ACTION_INDICES = (1, 2, 5, 6, 9, 10)
HIGHSTEP_FRONT_DIAGONAL_ACTION_INDICES = (1, 5, 9)
HIGHSTEP_REAR_DIAGONAL_ACTION_INDICES = (2, 6, 10)


def highstep_diagonal_action_loss(
    squared_action_error: torch.Tensor,
    post_prior_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return the preregistered gated FR/RL six-joint pre-prior loss."""
    if squared_action_error.ndim != 2 or squared_action_error.shape[-1] < 12:
        raise ValueError("diagonal action loss requires a [batch, >=12] squared-error tensor")
    indices = torch.as_tensor(
        HIGHSTEP_DIAGONAL_ACTION_INDICES,
        device=squared_action_error.device,
        dtype=torch.long,
    )
    return (
        float(scale)
        * post_prior_gate.detach()
        * torch.mean(torch.index_select(squared_action_error, dim=-1, index=indices), dim=-1)
    )


def highstep_phase_diagonal_action_loss(
    squared_action_error: torch.Tensor,
    front_phase_gate: torch.Tensor,
    rear_phase_gate: torch.Tensor,
    front_scale: float,
    rear_scale: float,
) -> torch.Tensor:
    """Return the preregistered positive FR/RL phase-specific add-on.

    The explicit 0.5 before each three-element mean is essential: with the
    frozen ``2 * mean(error[:12])`` base loss and scale 2.0, a gated joint has
    exactly three times the per-element coefficient of an ordinary nonbox
    joint, never five times.
    """
    if squared_action_error.ndim != 2 or squared_action_error.shape[-1] < 12:
        raise ValueError("phase diagonal loss requires [batch, >=12] squared errors")
    front_indices = torch.as_tensor(
        HIGHSTEP_FRONT_DIAGONAL_ACTION_INDICES,
        device=squared_action_error.device,
        dtype=torch.long,
    )
    rear_indices = torch.as_tensor(
        HIGHSTEP_REAR_DIAGONAL_ACTION_INDICES,
        device=squared_action_error.device,
        dtype=torch.long,
    )
    front_loss = 0.5 * torch.mean(
        torch.index_select(squared_action_error, -1, front_indices), dim=-1
    )
    rear_loss = 0.5 * torch.mean(
        torch.index_select(squared_action_error, -1, rear_indices), dim=-1
    )
    return (
        float(front_scale) * front_phase_gate.detach() * front_loss
        + float(rear_scale) * rear_phase_gate.detach() * rear_loss
    )


def highstep_critical_transition_window_masks(
    stage_context: torch.Tensor,
    dones: torch.Tensor,
    radius: int = 10,
    threshold: float = 0.5,
    require_complete_window: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build episode-safe +/- ``radius`` windows around four stage entries."""
    if stage_context.ndim != 3 or stage_context.shape[-1] != 4:
        raise ValueError("stage_context must have shape [time, env, 4]")
    done = dones.squeeze(-1).to(dtype=torch.bool)
    if done.shape != stage_context.shape[:2]:
        raise ValueError("dones must match stage_context [time, env]")
    if radius < 0:
        raise ValueError("transition radius must be non-negative")

    active = stage_context >= float(threshold)
    episode_id = torch.zeros_like(done, dtype=torch.long)
    if done.shape[0] > 1:
        episode_id[1:] = torch.cumsum(done[:-1].to(dtype=torch.long), dim=0)
    previous = torch.zeros_like(active)
    if active.shape[0] > 1:
        same_episode = episode_id[1:] == episode_id[:-1]
        previous[1:] = active[:-1] & same_episode.unsqueeze(-1)
    rising = active & ~previous
    # The first row has no previous frame in this rollout and is therefore not
    # promoted into a synthetic transition event.
    rising[0] = False
    front_event = torch.any(rising[..., :2], dim=-1)
    rear_event = torch.any(rising[..., 2:], dim=-1)

    if require_complete_window:
        complete = torch.zeros_like(done)
        if done.shape[0] >= 2 * radius + 1:
            center = slice(radius, done.shape[0] - radius)
            # Episode ids are monotonic. Equal ids at both endpoints prove
            # every frame in the full +/- radius interval belongs to the same
            # env episode as the event center.
            complete[center] = (
                (episode_id[center] == episode_id[: done.shape[0] - 2 * radius])
                & (episode_id[center] == episode_id[2 * radius :])
            )
        front_event &= complete
        rear_event &= complete

    def expand(events: torch.Tensor) -> torch.Tensor:
        window = torch.zeros_like(events)
        time = events.shape[0]
        for offset in range(-radius, radius + 1):
            if offset < 0:
                source = slice(-offset, time)
                target = slice(0, time + offset)
            elif offset > 0:
                source = slice(0, time - offset)
                target = slice(offset, time)
            else:
                source = slice(0, time)
                target = slice(0, time)
            same_episode = episode_id[source] == episode_id[target]
            window[target] |= events[source] & same_episode
        return window

    return expand(front_event), expand(rear_event)


def highstep_cross_rollout_transition_windows(
    stage_context: torch.Tensor,
    dones: torch.Tensor,
    radius: int,
    carry_stage_context: torch.Tensor | None = None,
    carry_dones: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build full windows from a bounded prior-rollout carry plus live rollout.

    The carry is always prepended along time for the same env slots. Only
    events with all ``2 * radius + 1`` real frames available in one episode
    are promoted. Events near the live rollout tail are therefore deferred to
    the next update instead of producing truncated windows.
    """
    if (carry_stage_context is None) != (carry_dones is None):
        raise ValueError("critical transition carry context and dones must be paired")
    carry_steps = 0
    if carry_stage_context is not None:
        if carry_stage_context.ndim != 3 or carry_stage_context.shape[-1] != 4:
            raise ValueError("carry stage context must have shape [time, env, 4]")
        if carry_stage_context.shape[1:] != stage_context.shape[1:]:
            raise ValueError("critical transition carry/live env axes disagree")
        if carry_dones.shape[:2] != carry_stage_context.shape[:2]:
            raise ValueError("critical transition carry dones axes disagree")
        if carry_stage_context.shape[0] > 2 * radius:
            raise ValueError("critical transition carry exceeds the bounded 2*radius FIFO")
        carry_steps = int(carry_stage_context.shape[0])
        stage_context = torch.cat((carry_stage_context, stage_context), dim=0)
        dones = torch.cat((carry_dones, dones), dim=0)
    front, rear = highstep_critical_transition_window_masks(
        stage_context,
        dones,
        radius=radius,
        require_complete_window=True,
    )
    return front, rear, carry_steps


def highstep_rl_preedge_window_mask(
    stage_context: torch.Tensor,
    dones: torch.Tensor,
    pre_steps: int = 30,
    post_steps: int = 10,
) -> torch.Tensor:
    """Return complete, episode-safe ``[-pre_steps,+post_steps]`` RL pre-edge windows."""
    if stage_context.ndim != 3 or stage_context.shape[-1] != 5:
        raise ValueError("RL pre-edge context must have shape [time, env, 5]")
    done = dones.squeeze(-1).to(dtype=torch.bool)
    if done.shape != stage_context.shape[:2]:
        raise ValueError("RL pre-edge dones must match context time/env axes")
    if pre_steps < 0 or post_steps < 0:
        raise ValueError("RL pre-edge window extents must be non-negative")
    episode_id = torch.zeros_like(done, dtype=torch.long)
    if done.shape[0] > 1:
        episode_id[1:] = torch.cumsum(done[:-1].to(dtype=torch.long), dim=0)
    active = stage_context[..., 4] >= 0.5
    previous = torch.zeros_like(active)
    if active.shape[0] > 1:
        same_episode = episode_id[1:] == episode_id[:-1]
        previous[1:] = active[:-1] & same_episode
    event = active & ~previous
    event[0] = False
    complete = torch.zeros_like(event)
    if event.shape[0] >= pre_steps + post_steps + 1:
        center = slice(pre_steps, event.shape[0] - post_steps)
        complete[center] = (
            (episode_id[center] == episode_id[: event.shape[0] - pre_steps - post_steps])
            & (episode_id[center] == episode_id[pre_steps + post_steps :])
        )
    event &= complete
    window = torch.zeros_like(event)
    time = event.shape[0]
    for offset in range(-pre_steps, post_steps + 1):
        if abs(offset) >= time:
            continue
        if offset < 0:
            source, target = slice(-offset, time), slice(0, time + offset)
        elif offset > 0:
            source, target = slice(0, time - offset), slice(offset, time)
        else:
            source, target = slice(0, time), slice(0, time)
        window[target] |= event[source] & (episode_id[source] == episode_id[target])
    return window


def highstep_balanced_transition_indices(
    front_window: torch.Tensor,
    rear_window: torch.Tensor,
    batch_size: int,
    original_pool: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw exactly 25% front, 25% rear and 50% original-distribution samples."""
    if front_window.shape != rear_window.shape or front_window.ndim != 1:
        raise ValueError("front/rear windows must be same-shape flat masks")
    if batch_size <= 0 or batch_size % 4:
        raise ValueError("balanced transition batch_size must be positive and divisible by four")
    front_pool = torch.nonzero(front_window, as_tuple=False).flatten()
    rear_pool = torch.nonzero(rear_window, as_tuple=False).flatten()
    if front_pool.numel() == 0 or rear_pool.numel() == 0:
        raise RuntimeError(
            "critical transition class is absent from the live rollout; refusing an unbalanced update"
        )
    quarter = batch_size // 4
    half = batch_size // 2
    front = front_pool[torch.randint(front_pool.numel(), (quarter,), device=front_pool.device)]
    rear = rear_pool[torch.randint(rear_pool.numel(), (quarter,), device=rear_pool.device)]
    if original_pool is None:
        original_pool = torch.arange(front_window.numel(), device=front_pool.device)
    if original_pool.ndim != 1 or original_pool.numel() == 0:
        raise ValueError("original-distribution pool must be a non-empty flat index tensor")
    original_pool = original_pool.to(device=front_pool.device, dtype=torch.long)
    if torch.any(original_pool < 0) or torch.any(original_pool >= front_window.numel()):
        raise ValueError("original-distribution pool contains an out-of-range index")
    original = original_pool[
        torch.randint(original_pool.numel(), (half,), device=front_pool.device)
    ]
    indices = torch.cat((front, rear, original), dim=0)
    source = torch.cat(
        (
            torch.zeros(quarter, device=indices.device, dtype=torch.long),
            torch.ones(quarter, device=indices.device, dtype=torch.long),
            torch.full((half,), 2, device=indices.device, dtype=torch.long),
        )
    )
    permutation = torch.randperm(batch_size, device=indices.device)
    return indices[permutation], source[permutation]

class PrivilegedEncoder(nn.Module):
    """特权信息编码器 (Teacher)：将上帝视角的观测压缩为 Latent 向量"""
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ELU())
            curr_dim = h
            
        # 1. 独立出最后一层
        last_layer = nn.Linear(curr_dim, latent_dim)
        
        # 2. 【核心数学修复】：近零初始化
        # 保证 t=0 时刻输出接近全 0，消除高方差噪声，让策略平滑起步
        with torch.no_grad():
            nn.init.uniform_(last_layer.weight, -1e-5, 1e-5)
            nn.init.zeros_(last_layer.bias)
            
        layers.append(last_layer)
        
        # 3. 【架构保障】：加入 Tanh
        # 将 Teacher 的特征空间严格界定在 [-1, 1]，保证第二阶段 VAE 蒸馏的 MSE Loss 处于合理量级
        # ==========================================
        # 【本次修改】：注释掉 Tanh。
        # 理由：防止高难度地形下特征需求过大导致 Tanh 饱和，引发梯度消失。
        # 必须让 Teacher 保持线性输出，释放最高性能上限。
        # ==========================================
        # 【修复 2】：必须加回 Tanh！Teacher 必须有界，否则 PPO 扰动会导致输出爆炸，Actor 激活值瞬间崩溃。
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

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
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # ==========================================
        # 【新增修改】：强制限制 logvar 的范围，防止方差坍塌导致 KL 爆炸
        # 限制在 [-4.0, 4.0] 之间，避免网络输出极小的方差死记硬背
        # ==========================================
        logvar = torch.clamp(logvar, min=-4.0, max=4.0)
        
        vel_pred = self.vel_estimator(h)
        return mu, logvar, vel_pred

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
        # 【新增】获取当前训练阶段 (1: Teacher, 2: Student)
        self.distill_stage = kwargs.pop("distill_stage", 1) 
        self.student_actor_warmup_updates = kwargs.pop("student_actor_warmup_updates", 800)
        self.student_vae_epochs = kwargs.pop("student_vae_epochs", 4)
        self.student_low_speed_threshold = kwargs.pop("student_low_speed_threshold", 0.10)
        self.student_prior_fade_speed = kwargs.pop("student_prior_fade_speed", 0.45)
        self.student_vel_loss_coef = kwargs.pop("student_vel_loss_coef", 10.0)
        self.student_latent_loss_coef = kwargs.pop("student_latent_loss_coef", 50.0)
        self.student_teacher_action_loss_coef = kwargs.pop("student_teacher_action_loss_coef", 20.0)
        self.student_prior_box_loss_coef = kwargs.pop("student_prior_box_loss_coef", 5.0)
        self.student_recon_loss_coef = kwargs.pop("student_recon_loss_coef", 0.5)
        self.student_kl_loss_coef = kwargs.pop("student_kl_loss_coef", 0.1)
        self.student_post_prior_mode = kwargs.pop("student_post_prior_mode", "lateral")
        self.student_highstep_phase_loss_scale = kwargs.pop("student_highstep_phase_loss_scale", 0.0)
        self.student_highstep_rear_box_loss_scale = kwargs.pop("student_highstep_rear_box_loss_scale", 0.0)
        self.student_highstep_diagonal_action_loss_scale = kwargs.pop(
            "student_highstep_diagonal_action_loss_scale", 0.0
        )
        self.student_critical_transition_balanced_sampling = kwargs.pop(
            "student_critical_transition_balanced_sampling", False
        )
        self.student_critical_transition_window_radius = kwargs.pop(
            "student_critical_transition_window_radius", 10
        )
        self.student_rl_preedge_sampling = kwargs.pop("student_rl_preedge_sampling", False)
        self.student_rl_preedge_pre_steps = kwargs.pop("student_rl_preedge_pre_steps", 30)
        self.student_rl_preedge_post_steps = kwargs.pop("student_rl_preedge_post_steps", 10)
        self.student_front_diagonal_action_loss_scale = kwargs.pop(
            "student_front_diagonal_action_loss_scale", 0.0
        )
        self.student_rear_diagonal_action_loss_scale = kwargs.pop(
            "student_rear_diagonal_action_loss_scale", 0.0
        )
        self.student_highstep_rear_hip_loss_scale = kwargs.pop("student_highstep_rear_hip_loss_scale", 0.0)
        self.student_highstep_rear_hip_min_abs = kwargs.pop("student_highstep_rear_hip_min_abs", 0.0)
        self.student_highstep_rear_hip_action_scale = kwargs.pop("student_highstep_rear_hip_action_scale", 0.1)
        self.student_actor_latent_clamp_backward = kwargs.pop(
            "student_actor_latent_clamp_backward", "hard"
        )
        self.student_recovery_stage = kwargs.pop("student_recovery_stage", "none")
        self.student_recovery_actor_lr = kwargs.pop("student_recovery_actor_lr", 1.0e-5)
        self.student_recovery_epochs = kwargs.pop("student_recovery_epochs", 1)
        self.student_recovery_source_checkpoint = kwargs.pop("student_recovery_source_checkpoint", "")
        self.student_recovery_source_sha256 = kwargs.pop("student_recovery_source_sha256", "")
        self.student_recovery_teacher_checkpoint = kwargs.pop("student_recovery_teacher_checkpoint", "")
        self.student_recovery_teacher_sha256 = kwargs.pop("student_recovery_teacher_sha256", "")
        self.student_recovery_r2_preregistration_path = kwargs.pop(
            "student_recovery_r2_preregistration_path", ""
        )
        self.student_recovery_r2_preregistration_sha256 = kwargs.pop(
            "student_recovery_r2_preregistration_sha256", ""
        )
        self.student_recovery_r3_preregistration_path = kwargs.pop(
            "student_recovery_r3_preregistration_path", ""
        )
        self.student_recovery_r3_preregistration_sha256 = kwargs.pop(
            "student_recovery_r3_preregistration_sha256", ""
        )
        self.student_recovery_v15_preregistration_path = kwargs.pop(
            "student_recovery_v15_preregistration_path", ""
        )
        self.student_recovery_v15_preregistration_sha256 = kwargs.pop(
            "student_recovery_v15_preregistration_sha256", ""
        )
        self.student_recovery_0707_exact_preregistration_path = kwargs.pop(
            "student_recovery_0707_exact_preregistration_path", ""
        )
        self.student_recovery_0707_exact_preregistration_sha256 = kwargs.pop(
            "student_recovery_0707_exact_preregistration_sha256", ""
        )
        self.student_recovery_historical_0707_exact_preregistration_path = kwargs.pop(
            "student_recovery_historical_0707_exact_preregistration_path", ""
        )
        self.student_recovery_historical_0707_exact_preregistration_sha256 = kwargs.pop(
            "student_recovery_historical_0707_exact_preregistration_sha256", ""
        )
        self.student_recovery_be300_0707_preregistration_path = kwargs.pop(
            "student_recovery_be300_0707_preregistration_path", ""
        )
        self.student_recovery_be300_0707_preregistration_sha256 = kwargs.pop(
            "student_recovery_be300_0707_preregistration_sha256", ""
        )
        self.student_recovery_b300_hybrid_preregistration_path = kwargs.pop(
            "student_recovery_b300_hybrid_preregistration_path", ""
        )
        self.student_recovery_b300_hybrid_preregistration_sha256 = kwargs.pop(
            "student_recovery_b300_hybrid_preregistration_sha256", ""
        )
        self.student_recovery_v18_preregistration_path = kwargs.pop(
            "student_recovery_v18_preregistration_path", ""
        )
        self.student_recovery_v18_preregistration_sha256 = kwargs.pop(
            "student_recovery_v18_preregistration_sha256", ""
        )
        self.student_recovery_e1400_continuation_branch = kwargs.pop(
            "student_recovery_e1400_continuation_branch", ""
        )
        self.student_recovery_e1400_continuation_preregistration_path = kwargs.pop(
            "student_recovery_e1400_continuation_preregistration_path", ""
        )
        self.student_recovery_e1400_continuation_preregistration_sha256 = kwargs.pop(
            "student_recovery_e1400_continuation_preregistration_sha256", ""
        )
        
        vae_latent_dim = kwargs.pop("vae_latent_dim", 64)
        vae_hidden_dims = kwargs.pop("vae_hidden_dims", [256, 128])
        
        policy_keys = obs_groups.get("policy", ["policy"])
        estimator_keys = obs_groups.get("estimator", ["estimator"])
        critic_keys = obs_groups.get("critic", ["critic"])
        teacher_context_keys = obs_groups.get("teacher_context", ["teacher_context"])

        self.policy_obs_dim = obs[policy_keys[0]].shape[-1] if hasattr(obs, "items") else obs.shape[-1]
        estimator_obs_dim = obs[estimator_keys[0]].shape[-1] if hasattr(obs, "items") else obs.shape[-1]
        critic_obs_dim = obs[critic_keys[0]].shape[-1] if hasattr(obs, "items") else obs.shape[-1]

        # ==========================================
        # 【核心修复 1】：移除 '+ 3' (真实速度维度)
        # Actor 的输入只能是: 基础观测 + Latent 向量
        # 绝对不能让 Actor 直接看到线速度！
        # ==========================================
        actor_input_dim = self.policy_obs_dim + vae_latent_dim
        
        device = obs[policy_keys[0]].device if hasattr(obs, "items") else obs.device
        
        # =====================================================================
        # 【终极同构替换修复】：绝对不欺骗父类！直接传入原始 obs！
        # 保证 Actor 的后续层、Critic 网络、Action Std 的 RNG 初始化状态与 Baseline 100% 完全一致！
        # =====================================================================
        # 调用父类初始化
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        # =====================================================================
        # 【外科手术：同构替换 Actor 第一层】
        # 将前 N 维完美复制，后 64 维强行置 0，在数学上绝对等价于 Baseline
        # 保证 Actor 起步平滑，不会被未训练的 64 维 Latent 噪声带崩
        # =====================================================================
        if hasattr(self, 'actor') and isinstance(self.actor[0], nn.Linear):
            old_layer = self.actor[0]
            new_layer = nn.Linear(actor_input_dim, old_layer.out_features).to(device)
            
            with torch.no_grad():
                # 1. 完美复制前 N 维的权重
                new_layer.weight[:, :self.policy_obs_dim] = old_layer.weight
                # 2. 将新增的 64 维权重强制设为 0.0！
                new_layer.weight[:, self.policy_obs_dim:] = 0.0
                # 3. 完美复制 Bias
                new_layer.bias = old_layer.bias
                
            self.actor[0] = new_layer
        
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dims = vae_hidden_dims
        self.policy_keys = policy_keys
        self.estimator_keys = estimator_keys
        self.critic_keys = critic_keys
        self.teacher_context_keys = teacher_context_keys

        # 【新增】初始化 Teacher 编码器
        # <=== STRICT FIX 1: critic_obs_dim 包含了前 3 维的 base_lin_vel。
        # 为了防止信息泄露导致步态崩溃，PrivEncoder 的输入维度必须减去这 3 维。
        self.priv_encoder = PrivilegedEncoder(
            input_dim=critic_obs_dim - 3, 
            latent_dim=self.vae_latent_dim,
            hidden_dims=self.vae_hidden_dims
        ).to(device)

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
        
        # 【修复 1 补充】：防止 rsl_rl 回放池中的 obs 被无限拼接导致维度爆炸
        if policy_obs.shape[-1] >= self.policy_obs_dim + self.vae_latent_dim:
            policy_obs = policy_obs[..., :self.policy_obs_dim]
            
        if self.distill_stage == 1:
            # ==========================================
            # 阶段 1 (Teacher): 使用真实的特权信息
            # ==========================================
            critic_obs = obs_dict[self.critic_keys[0]]
            
            # <=== STRICT FIX 2: 严格切除前 3 维的 base_lin_vel。
            # Teacher 只能看到无噪声的本体感觉，绝对不能看到真实速度，否则必将触发拖拽作弊！
            priv_input = critic_obs[..., 3:]
            
            true_latent = self.priv_encoder(priv_input)
            
            # ==========================================
            # 【新增修改】：Stage 1 注入 Latent 噪声
            # 理由：让 Actor 在 Stage 1 习惯不完美的 Latent，防止 Stage 2 遇到 VAE 误差时直接崩溃。
            # ==========================================
            if self.training:
                noise = torch.randn_like(true_latent) * 0.05
                true_latent = true_latent + noise
                # 保证注入噪声后依然在 [-1, 1] 范围内，与 Tanh 保持一致
                true_latent = torch.clamp(true_latent, -1.0, 1.0)
            
            # ==========================================
            # 【核心修复 2】：移除 true_vel
            # 强迫 Actor 依赖 policy_obs 和 Teacher 提取的 latent 来学习步态
            # ==========================================
            actor_input = torch.cat([policy_obs, true_latent], dim=-1)
        else:
            # ==========================================
            # 阶段 2 (Student): 使用 VAE 的预测信息
            # ==========================================
            estimator_obs = obs_dict[self.estimator_keys[0]]
            if str(self.student_recovery_stage).upper() in {"R2", "R3"}:
                # R2 rollout is deterministic by preregistration: the actor consumes
                # estimator ``mu`` and must not draw a latent reparameterization sample
                # merely to populate reconstruction-only legacy fields.
                mu, logvar, vel_pred = self.estimator.encode(estimator_obs)
                recon_x = None
            else:
                vel_pred, recon_x, mu, logvar, z = self.estimator(estimator_obs)
            
            if self.training:
                self.active_vae_recon = recon_x
                self.active_vae_mu = mu
                self.active_vae_logvar = logvar
                self.active_vae_vel_pred = vel_pred
                self.active_vae_input = estimator_obs
                
                # ==========================================
                # 【新增 Debug 记录】：记录 mu 越界的比例
                # ==========================================
                self.active_mu_oob_ratio = (mu.abs() > 1.0).float().mean()
                
            # ==========================================
            # 【核心修复 3】：移除 vel_pred.detach()
            # 第二阶段同样不能把预测的速度喂给 Actor
            # VAE 的 vel_pred 只用于在 VAEPPO 中计算 Loss，不参与前向控制
            # ==========================================
            # 【极其关键的修复】：将 z.detach() 改为 mu.detach()
            # 保证 PPO 在 Rollout 和 Update 阶段看到的 Actor 输入是确定且一致的！
            
            # ==========================================
            # 【本次核心修复】：强制截断 VAE 的 mu 输出
            # 理由：Teacher 最后一层是 Tanh，Actor 的权重是基于 [-1, 1] 的输入训练的。
            # VAE 的 mu 是无界的，一旦输出 >1 或 <-1，Frozen Actor 的激活值就会爆炸，导致直接翻倒。
            # ==========================================
            safe_mu = torch.clamp(mu, min=-1.0, max=1.0)
            
            actor_input = torch.cat([policy_obs, safe_mu.detach()], dim=-1)
            
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

    # =====================================================================
    # 【修复 1 核心】：必须重写 evaluate 和 evaluate_actions！
    # rsl_rl 在 PPO update 时调用的是这些函数来计算 Loss，而不是 act。
    # 如果不重写，Teacher 前向传播不会被触发，梯度就永远传不过去！
    # =====================================================================
    def evaluate(self, critic_observations, **kwargs):
        actor_input = self._process_obs(critic_observations)
        if hasattr(critic_observations, "items"):
            mock_obs = {k: v for k, v in critic_observations.items()}
        else:
            mock_obs = {}
        mock_obs[self.policy_keys[0]] = actor_input
        return super().evaluate(mock_obs, **kwargs)
        
    def evaluate_actions(self, obs, actions, **kwargs):
        actor_input = self._process_obs(obs)
        if hasattr(obs, "items"):
            mock_obs = {k: v for k, v in obs.items()}
        else:
            mock_obs = {}
        mock_obs[self.policy_keys[0]] = actor_input
        return super().evaluate_actions(mock_obs, actions, **kwargs)


class VAEPPO(PPO):
    """重写 PPO 的更新逻辑，加入 VAE 的 Loss 计算"""

    EXTRA_CHECKPOINT_STATE_SCHEMA_VERSION = 1

    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        self.distill_stage = getattr(self.policy, "distill_stage", 1)
        self.student_recovery_stage = str(
            getattr(self.policy, "student_recovery_stage", "none")
        ).upper()
        self._teacher_actor_synced = self.distill_stage != 2
        
        # 【核心修复 2】：为 VAE 创建完全独立的优化器
        # 避免 VAE 的巨大 Loss 污染 Actor/Critic 的 Adam 动量状态
        if hasattr(self.policy, 'estimator') and self.policy.estimator is not None:
            if self.distill_stage == 2:
                if self.student_recovery_stage == "B":
                    self._initialize_student_recovery_stage_b()
                    return
                if self.student_recovery_stage == "R2":
                    self._initialize_student_recovery_r2()
                    return
                if self.student_recovery_stage == "R3":
                    self._initialize_student_recovery_r3()
                    return
                if self.student_recovery_stage in {
                    "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
                    "E1400_CONTINUATION", "BE300_0707", "B300_CANONICAL_HYBRID",
                }:
                    self._initialize_student_recovery_v15()
                    return

                # Legacy Student path retained only for non-recovery tasks.
                self.teacher_actor = copy.deepcopy(self.policy.actor).to(self.device).eval()
                for param in self.teacher_actor.parameters():
                    param.requires_grad = False

                for param in self.policy.actor.parameters():
                    param.requires_grad = False
                for param in self.policy.critic.parameters():
                    param.requires_grad = False
                for param in self.policy.priv_encoder.parameters():
                    param.requires_grad = False
                
                self.student_distill_update_count = 0
                self.student_actor_warmup_updates = getattr(self.policy, "student_actor_warmup_updates", 800)
                self.student_vae_epochs = getattr(self.policy, "student_vae_epochs", 4)
                self.student_low_speed_threshold = getattr(self.policy, "student_low_speed_threshold", 0.10)
                self.student_prior_fade_speed = getattr(self.policy, "student_prior_fade_speed", 0.45)
                self.student_vel_loss_coef = getattr(self.policy, "student_vel_loss_coef", 10.0)
                self.student_latent_loss_coef = getattr(self.policy, "student_latent_loss_coef", 50.0)
                self.student_teacher_action_loss_coef = getattr(
                    self.policy, "student_teacher_action_loss_coef", 20.0
                )
                self.student_prior_box_loss_coef = getattr(self.policy, "student_prior_box_loss_coef", 5.0)
                self.student_recon_loss_coef = getattr(self.policy, "student_recon_loss_coef", 0.5)
                self.student_kl_loss_coef = getattr(self.policy, "student_kl_loss_coef", 0.1)
                self.student_post_prior_mode = getattr(self.policy, "student_post_prior_mode", "lateral")
                self.student_highstep_phase_loss_scale = getattr(
                    self.policy, "student_highstep_phase_loss_scale", 0.0
                )
                self.student_highstep_rear_box_loss_scale = getattr(
                    self.policy, "student_highstep_rear_box_loss_scale", 0.0
                )
                self.student_highstep_diagonal_action_loss_scale = getattr(
                    self.policy, "student_highstep_diagonal_action_loss_scale", 0.0
                )
                self.student_highstep_rear_hip_loss_scale = getattr(
                    self.policy, "student_highstep_rear_hip_loss_scale", 0.0
                )
                self.student_highstep_rear_hip_min_abs = getattr(
                    self.policy, "student_highstep_rear_hip_min_abs", 0.0
                )
                self.student_highstep_rear_hip_action_scale = getattr(
                    self.policy, "student_highstep_rear_hip_action_scale", 0.1
                )

                vae_lr = 1e-3
                actor_lr = 1e-5
                self.vae_optimizer = torch.optim.Adam(
                    [
                        {"params": self.policy.estimator.parameters(), "lr": vae_lr},
                        {"params": self.policy.actor.parameters(), "lr": actor_lr},
                    ]
                )
            else:
                # 【关键修复】：大幅降低 VAE 的学习率 (例如固定为 1e-4)
                # 防止 VAE 训练过快导致 Actor 输入特征剧烈震荡
                vae_lr = 1e-4 
                self.vae_optimizer = torch.optim.Adam(self.policy.estimator.parameters(), lr=vae_lr)

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _require_sha256(value: object, label: str) -> str:
        text = str(value)
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise RuntimeError(f"{label} must be a lowercase SHA256, got {value!r}")
        return text

    def _load_r2_preregistration(self) -> tuple[dict, str, str]:
        path = os.path.realpath(
            str(getattr(self.policy, "student_recovery_r2_preregistration_path", ""))
        )
        expected_sha = self._require_sha256(
            getattr(self.policy, "student_recovery_r2_preregistration_sha256", ""),
            "R2 preregistration SHA256",
        )
        if not path or not os.path.isabs(path):
            raise RuntimeError(f"R2 preregistration path must be absolute, got {path!r}")
        actual_sha = self._sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"R2 preregistration SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError("R2 preregistration must contain one JSON object")
        if payload.get("kind") != "highstep_student_recovery_r2_preregistration":
            raise RuntimeError("R2 preregistration kind mismatch")
        if payload.get("route", {}).get("selected") != "R2":
            raise RuntimeError("R2 preregistration does not select R2")
        if payload.get("authority", {}).get("approved_version") != "v1.1.1":
            raise RuntimeError("R2 preregistration is not bound to spec v1.1.1")
        return payload, path, actual_sha

    @staticmethod
    def _r2_exact_optimizer_names() -> tuple[str, ...]:
        return (
            "estimator.encoder.0.weight",
            "estimator.encoder.0.bias",
            "estimator.encoder.2.weight",
            "estimator.encoder.2.bias",
            "estimator.fc_mu.weight",
            "estimator.fc_mu.bias",
            "algorithm.r2_box_weight",
            "algorithm.r2_box_bias",
        )

    def _initialize_student_recovery_r2(self) -> None:
        """Initialize the immutable v1.1.1 R2 optimizer without changing policy structure."""
        prereg, prereg_path, prereg_sha = self._load_r2_preregistration()
        self._r2_preregistration = prereg
        self._r2_preregistration_path = prereg_path
        self._r2_preregistration_sha256 = prereg_sha

        training_data = prereg.get("training_data", {})
        context = training_data.get("extra_rollout_group", {})
        if context.get("name") != "teacher_context" or context.get("shape") != [3]:
            raise RuntimeError("R2 preregistration teacher_context contract changed")
        if context.get("order") != ["height_delta", "command_x", "front_rear_delta"]:
            raise RuntimeError("R2 preregistration teacher_context order changed")
        if list(getattr(self.policy, "teacher_context_keys", [])) != ["teacher_context"]:
            raise RuntimeError("R2 policy must bind exactly teacher_context:[teacher_context]")

        scope = prereg.get("trainable_scope", {})
        expected_live = list(self._r2_exact_optimizer_names()[:6])
        if scope.get("allowed_live_tensors") != expected_live:
            raise RuntimeError("R2 preregistration live trainable tensor list changed")
        if scope.get("optimizer_parameter_tensor_count") != 8:
            raise RuntimeError("R2 preregistration must own exactly eight optimizer tensors")
        rows = scope.get("algorithm_owned_materialized_rows", {})
        if rows.get("actor_final_weight_rows") != [12, 13, 14, 15] or rows.get(
            "actor_final_bias_rows"
        ) != [12, 13, 14, 15]:
            raise RuntimeError("R2 preregistration box output rows changed")

        for parameter in self.policy.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        if not isinstance(self.policy.estimator.encoder, nn.Sequential):
            raise RuntimeError("R2 requires the locked sequential estimator encoder")
        encoder_modules = list(self.policy.estimator.encoder)
        if len(encoder_modules) != 4 or not isinstance(encoder_modules[0], nn.Linear) or not isinstance(
            encoder_modules[2], nn.Linear
        ):
            raise RuntimeError("R2 estimator encoder architecture differs from the preregistered 2-layer MLP")
        estimator_parameters = list(self.policy.estimator.encoder.parameters()) + list(
            self.policy.estimator.fc_mu.parameters()
        )
        if len(estimator_parameters) != 6:
            raise RuntimeError(
                f"R2 expected six encoder/fc_mu tensors, got {len(estimator_parameters)}"
            )
        for parameter in estimator_parameters:
            parameter.requires_grad = True

        if not isinstance(self.policy.actor, nn.Sequential):
            raise RuntimeError("R2 requires the locked sequential actor")
        self._r2_actor_modules = list(self.policy.actor)
        first_linear = next(
            (module for module in self._r2_actor_modules if isinstance(module, nn.Linear)), None
        )
        last_name, last_linear = self._find_actor_last_linear()
        if (
            not self._r2_actor_modules
            or first_linear is None
            or first_linear.in_features != 634
            or first_linear.out_features != 512
            or self._r2_actor_modules[-1] is not last_linear
            or last_name != "6"
            or last_linear.in_features != 128
            or last_linear.out_features != 16
        ):
            raise RuntimeError(
                "R2 requires the locked Linear(634,512) actor input and actor.6 Linear(128,16) head"
            )
        self._r2_first_linear = first_linear
        self._r2_last_linear_name = last_name
        self._r2_last_linear = last_linear
        self._r2_box_rows = (12, 13, 14, 15)
        self.r2_box_weight = nn.Parameter(
            last_linear.weight.detach()[12:16].clone(), requires_grad=True
        )
        self.r2_box_bias = nn.Parameter(
            last_linear.bias.detach()[12:16].clone(), requires_grad=True
        )

        optimizer_cfg = prereg.get("optimizer", {})
        groups = optimizer_cfg.get("groups")
        if not isinstance(groups, list) or len(groups) != 2:
            raise RuntimeError("R2 preregistration optimizer groups changed")
        if groups[0].get("name") != "estimator_encoder_and_mu" or groups[0].get(
            "parameter_tensors"
        ) != 6:
            raise RuntimeError("R2 estimator optimizer group changed")
        if groups[1].get("name") != "box_output_rows" or groups[1].get(
            "parameter_tensors"
        ) != 2:
            raise RuntimeError("R2 box optimizer group changed")
        estimator_lr = float(groups[0].get("learning_rate"))
        box_lr = float(groups[1].get("learning_rate"))
        betas = tuple(float(value) for value in optimizer_cfg.get("betas", []))
        if (
            estimator_lr != 1.0e-4
            or box_lr != 1.0e-5
            or betas != (0.9, 0.999)
            or float(optimizer_cfg.get("eps")) != 1.0e-8
            or float(optimizer_cfg.get("weight_decay")) != 0.0
            or optimizer_cfg.get("amsgrad") is not False
            or int(optimizer_cfg.get("epochs_per_effective_update")) != 1
            or int(optimizer_cfg.get("mini_batches")) != 4
            or float(optimizer_cfg.get("max_grad_norm")) != 1.0
            or optimizer_cfg.get("learning_rate_schedule") != "fixed"
        ):
            raise RuntimeError("R2 optimizer preregistration changed")
        if int(self.num_mini_batches) != 4 or float(self.max_grad_norm) != 1.0:
            raise RuntimeError("R2 runtime mini-batch/max-grad settings differ from preregistration")
        self.r2_epochs = 1
        self.learning_rate = estimator_lr
        self.optimizer = torch.optim.Adam(
            [
                {"params": estimator_parameters, "lr": estimator_lr, "name": groups[0]["name"]},
                {
                    "params": [self.r2_box_weight, self.r2_box_bias],
                    "lr": box_lr,
                    "name": groups[1]["name"],
                },
            ],
            betas=betas,
            eps=1.0e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.vae_optimizer = self.optimizer
        self.student_distill_update_count = 0
        self._r2_optimizer_names = self._r2_exact_optimizer_names()
        self._r2_binding_ready = False
        self._r2_extra_restored = False
        self._r2_binding_manifest = None
        self._r2_post_prior_function = None
        self.r2_teacher_actor = None
        self.r2_teacher_priv_encoder = None
        self.r2_anchor_actor = None
        self.r2_anchor_estimator = None
        self._teacher_actor_synced = False
        self._validate_r2_optimizer_scope()
        self._validate_r2_optimizer_state(0)

    def _validate_r2_optimizer_scope(self) -> None:
        groups = self.optimizer.param_groups
        if len(groups) != 2:
            raise RuntimeError(f"R2 optimizer must have exactly two parameter groups, got {len(groups)}")
        expected_group_contract = (
            ("estimator_encoder_and_mu", 6, 1.0e-4),
            ("box_output_rows", 2, 1.0e-5),
        )
        for group, (expected_name, expected_count, expected_lr) in zip(
            groups, expected_group_contract, strict=True
        ):
            if group.get("name") != expected_name or len(group.get("params", [])) != expected_count:
                raise RuntimeError(f"R2 optimizer group structure changed: {expected_name}")
            if (
                float(group.get("lr")) != expected_lr
                or tuple(float(value) for value in group.get("betas", ())) != (0.9, 0.999)
                or float(group.get("eps")) != 1.0e-8
                or float(group.get("weight_decay")) != 0.0
                or group.get("amsgrad") is not False
            ):
                raise RuntimeError(f"R2 optimizer hyperparameters changed: {expected_name}")
        parameters = [parameter for group in groups for parameter in group["params"]]
        expected = list(self.policy.estimator.encoder.parameters()) + list(
            self.policy.estimator.fc_mu.parameters()
        ) + [self.r2_box_weight, self.r2_box_bias]
        if len(parameters) != 8 or any(actual is not wanted for actual, wanted in zip(parameters, expected, strict=True)):
            raise RuntimeError("R2 optimizer does not own exactly the eight preregistered tensors in order")
        trainable_policy = [
            name for name, parameter in self.policy.named_parameters() if parameter.requires_grad
        ]
        if trainable_policy != list(self._r2_exact_optimizer_names()[:6]):
            raise RuntimeError(f"R2 trainable policy scope changed: {trainable_policy}")
        if self.optimizer is not self.vae_optimizer:
            raise RuntimeError("R2 requires one shared runner/VAEPPO Adam optimizer")

    def _load_r3_preregistration(self) -> tuple[dict, str, str]:
        path = os.path.realpath(
            str(getattr(self.policy, "student_recovery_r3_preregistration_path", ""))
        )
        expected_sha = self._require_sha256(
            getattr(self.policy, "student_recovery_r3_preregistration_sha256", ""),
            "R3 preregistration SHA256",
        )
        if not path or not os.path.isabs(path):
            raise RuntimeError(f"R3 preregistration path must be absolute, got {path!r}")
        actual_sha = self._sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"R3 preregistration SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError("R3 preregistration must contain one JSON object")
        if payload.get("kind") != "highstep_student_recovery_r3_preregistration":
            raise RuntimeError("R3 preregistration kind mismatch")
        if payload.get("workflow_id") != "highstep_student_recovery_v12_20260713":
            raise RuntimeError("R3 preregistration workflow changed")
        if payload.get("approved_version") != "v1.2":
            raise RuntimeError("R3 preregistration is not bound to spec v1.2")
        if payload.get("route", {}).get("selected") != "R3":
            raise RuntimeError("R3 preregistration does not select R3")
        if payload.get("immutable_before_parameter_update") is not True:
            raise RuntimeError("R3 preregistration is not immutable before parameter update")
        return payload, path, actual_sha

    @staticmethod
    def _r3_exact_optimizer_names() -> tuple[str, str]:
        return ("actor.6.weight", "actor.6.bias")

    def _initialize_student_recovery_r3(self) -> None:
        """Initialize the v1.2 R3 full action-head optimizer with every other tensor frozen."""
        prereg, prereg_path, prereg_sha = self._load_r3_preregistration()
        self._r3_preregistration = prereg
        self._r3_preregistration_path = prereg_path
        self._r3_preregistration_sha256 = prereg_sha
        # Reuse only the already-audited runtime/action/post-prior helpers.  R3
        # owns separate optimizer, checkpoint and lineage state below.
        self._r2_preregistration = prereg

        context = prereg.get("training_data", {}).get("extra_rollout_group", {})
        if context.get("name") != "teacher_context" or context.get("shape") != [3]:
            raise RuntimeError("R3 teacher_context contract changed")
        if context.get("order") != ["height_delta", "command_x", "front_rear_delta"]:
            raise RuntimeError("R3 teacher_context order changed")
        if list(getattr(self.policy, "teacher_context_keys", [])) != ["teacher_context"]:
            raise RuntimeError("R3 policy must bind exactly teacher_context:[teacher_context]")
        if prereg.get("trainable_scope", {}).get("allowed_live_tensors") != list(
            self._r3_exact_optimizer_names()
        ):
            raise RuntimeError("R3 trainable tensor scope changed")

        for parameter in self.policy.parameters():
            parameter.requires_grad = False
            parameter.grad = None
        if not isinstance(self.policy.actor, nn.Sequential):
            raise RuntimeError("R3 requires the locked sequential actor")
        self._r3_actor_modules = list(self.policy.actor)
        first_linear = next(
            (module for module in self._r3_actor_modules if isinstance(module, nn.Linear)), None
        )
        last_name, last_linear = self._find_actor_last_linear()
        if (
            first_linear is None
            or first_linear.in_features != 634
            or first_linear.out_features != 512
            or self._r3_actor_modules[-1] is not last_linear
            or last_name != "6"
            or last_linear.in_features != 128
            or last_linear.out_features != 16
            or last_linear.bias is None
        ):
            raise RuntimeError(
                "R3 requires the locked Linear(634,512) actor and actor.6 Linear(128,16) head"
            )
        self._r3_last_linear_name = last_name
        self._r3_last_linear = last_linear
        last_linear.weight.requires_grad = True
        last_linear.bias.requires_grad = True

        optimizer_cfg = prereg.get("optimizer", {})
        if (
            optimizer_cfg.get("type") != "Adam"
            or int(optimizer_cfg.get("parameter_tensors", -1)) != 2
            or float(optimizer_cfg.get("learning_rate", -1.0)) != 5.0e-6
            or tuple(float(value) for value in optimizer_cfg.get("betas", [])) != (0.9, 0.999)
            or float(optimizer_cfg.get("eps", -1.0)) != 1.0e-8
            or float(optimizer_cfg.get("weight_decay", -1.0)) != 0.0
            or optimizer_cfg.get("amsgrad") is not False
            or int(optimizer_cfg.get("epochs_per_effective_update", -1)) != 1
            or int(optimizer_cfg.get("mini_batches", -1)) != 4
            or float(optimizer_cfg.get("max_grad_norm", -1.0)) != 1.0
            or optimizer_cfg.get("learning_rate_schedule") != "fixed"
        ):
            raise RuntimeError("R3 optimizer preregistration changed")
        if int(self.num_mini_batches) != 4 or float(self.max_grad_norm) != 1.0:
            raise RuntimeError("R3 runtime mini-batch/max-grad settings changed")
        self.r3_epochs = 1
        self.learning_rate = 5.0e-6
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [last_linear.weight, last_linear.bias],
                    "lr": 5.0e-6,
                    "name": "full_action_head",
                }
            ],
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
            amsgrad=False,
        )
        self.vae_optimizer = self.optimizer
        self.student_distill_update_count = 0
        self._r3_optimizer_names = self._r3_exact_optimizer_names()
        self._r3_binding_ready = False
        self._r3_extra_restored = False
        self._r3_binding_manifest = None
        self._r2_post_prior_function = None
        self.r3_teacher_actor = None
        self.r3_teacher_priv_encoder = None
        self.r3_anchor_actor = None
        self.r3_anchor_estimator = None
        self._teacher_actor_synced = False
        self._validate_r3_optimizer_scope()
        self._validate_r3_optimizer_state(0)

    def _validate_r3_optimizer_scope(self) -> None:
        groups = self.optimizer.param_groups
        if len(groups) != 1:
            raise RuntimeError(f"R3 optimizer must have one group, got {len(groups)}")
        group = groups[0]
        parameters = list(group.get("params", []))
        expected = [self._r3_last_linear.weight, self._r3_last_linear.bias]
        if group.get("name") != "full_action_head" or len(parameters) != 2 or any(
            actual is not wanted for actual, wanted in zip(parameters, expected, strict=True)
        ):
            raise RuntimeError("R3 optimizer does not own exactly actor.6 weight/bias in order")
        if (
            float(group.get("lr")) != 5.0e-6
            or tuple(float(value) for value in group.get("betas", ())) != (0.9, 0.999)
            or float(group.get("eps")) != 1.0e-8
            or float(group.get("weight_decay")) != 0.0
            or group.get("amsgrad") is not False
        ):
            raise RuntimeError("R3 optimizer hyperparameters changed")
        trainable = [name for name, parameter in self.policy.named_parameters() if parameter.requires_grad]
        if trainable != list(self._r3_optimizer_names):
            raise RuntimeError(f"R3 trainable policy scope changed: {trainable}")
        if self.optimizer is not self.vae_optimizer:
            raise RuntimeError("R3 requires one shared runner/VAEPPO Adam optimizer")

    def _validate_r3_optimizer_state(self, effective_update_count: int) -> None:
        self._validate_r3_optimizer_scope()
        count = self._validate_student_distill_update_count(effective_update_count)
        parameters = [self._r3_last_linear.weight, self._r3_last_linear.bias]
        state = self.optimizer.state
        if count == 0:
            if state:
                raise RuntimeError("Fresh R3 count=0 must have empty Adam state")
            return
        if len(state) != 2 or set(state) != set(parameters):
            raise RuntimeError("Resumed R3 Adam must contain exactly two state entries")
        expected_step = 4 * count
        for name, parameter in zip(self._r3_optimizer_names, parameters, strict=True):
            parameter_state = state.get(parameter)
            if not isinstance(parameter_state, dict) or set(parameter_state) != {
                "step", "exp_avg", "exp_avg_sq"
            }:
                raise RuntimeError(f"Malformed R3 Adam state for {name}")
            step = torch.as_tensor(parameter_state["step"]).detach().cpu().reshape(-1)
            if step.numel() != 1 or not bool(torch.isfinite(step).all().item()):
                raise RuntimeError(f"R3 Adam step is invalid for {name}")
            step_value = float(step.item())
            if not step_value.is_integer() or int(step_value) != expected_step:
                raise RuntimeError(
                    f"R3 Adam step/count mismatch for {name}: {step_value} != {expected_step}"
                )
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = parameter_state[moment_name]
                if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"R3 Adam {moment_name} shape mismatch for {name}")
                if not bool(torch.isfinite(moment).all().item()):
                    raise RuntimeError(f"R3 Adam {moment_name} is non-finite for {name}")

    def _validate_r2_optimizer_state(self, effective_update_count: int) -> None:
        """Fail closed on malformed or discontinuous R2 Adam moments/count."""
        self._validate_r2_optimizer_scope()
        count = self._validate_student_distill_update_count(effective_update_count)
        parameters = [
            parameter for group in self.optimizer.param_groups for parameter in group["params"]
        ]
        state = self.optimizer.state
        if count == 0:
            if state:
                raise RuntimeError("Fresh R2 count=0 must have an empty Adam state")
            return
        if len(state) != 8 or set(state) != set(parameters):
            raise RuntimeError(
                "Resumed R2 Adam must contain exactly eight state entries mapped in optimizer order"
            )
        expected_step = 4 * count
        for name, parameter in zip(self._r2_optimizer_names, parameters, strict=True):
            parameter_state = state.get(parameter)
            if not isinstance(parameter_state, dict) or set(parameter_state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise RuntimeError(f"Malformed R2 Adam state fields for {name}")
            step = parameter_state["step"]
            step_tensor = torch.as_tensor(step).detach().cpu().reshape(-1)
            if step_tensor.numel() != 1 or not bool(torch.isfinite(step_tensor).all().item()):
                raise RuntimeError(f"R2 Adam step is non-finite or non-scalar for {name}")
            step_value = float(step_tensor.item())
            if step_value <= 0.0 or not step_value.is_integer() or int(step_value) != expected_step:
                raise RuntimeError(
                    f"R2 Adam step/count mismatch for {name}: step={step_value}, "
                    f"expected={expected_step}"
                )
            for moment_name in ("exp_avg", "exp_avg_sq"):
                moment = parameter_state[moment_name]
                if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"R2 Adam {moment_name} shape mismatch for {name}: "
                        f"{getattr(moment, 'shape', None)} != {tuple(parameter.shape)}"
                    )
                if not bool(torch.isfinite(moment).all().item()):
                    raise RuntimeError(f"R2 Adam {moment_name} is non-finite for {name}")

    def _sync_r2_box_rows_to_actor(self) -> None:
        with torch.no_grad():
            self._r2_last_linear.weight[12:16].copy_(self.r2_box_weight)
            self._r2_last_linear.bias[12:16].copy_(self.r2_box_bias)

    def _reset_r2_box_rows_from_actor(self) -> None:
        with torch.no_grad():
            self.r2_box_weight.copy_(self._r2_last_linear.weight[12:16])
            self.r2_box_bias.copy_(self._r2_last_linear.bias[12:16])

    def _r2_actor_forward(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, float]:
        modules = list(self.policy.actor)
        if len(modules) != len(self._r2_actor_modules) or any(
            current is not expected
            for current, expected in zip(modules, self._r2_actor_modules, strict=True)
        ):
            raise RuntimeError("R2 actor module sequence changed")
        hidden = actor_input
        for module in modules[:-1]:
            hidden = module(hidden)
        revolute = F.linear(
            hidden,
            self._r2_last_linear.weight[:12].detach(),
            self._r2_last_linear.bias[:12].detach(),
        )
        box = F.linear(hidden, self.r2_box_weight, self.r2_box_bias)
        action = torch.cat((revolute, box), dim=-1)
        with torch.no_grad():
            reference = self.policy.actor(actor_input)
            max_error = float(torch.max(torch.abs(action.detach() - reference)).item())
            if not torch.allclose(action.detach(), reference, rtol=1.0e-6, atol=1.0e-6):
                raise RuntimeError(
                    "R2 materialized box-row actor differs from live actor "
                    f"(max_abs_error={max_error})"
                )
        return action, max_error

    def _find_actor_last_linear(self) -> tuple[str, nn.Linear]:
        result = None
        for name, module in self.policy.actor.named_modules():
            if isinstance(module, nn.Linear):
                result = (name, module)
        if result is None:
            raise RuntimeError("Student recovery requires a final nn.Linear actor layer")
        return result

    def _initialize_student_recovery_stage_b(self) -> None:
        """Create the strict two-row optimizer used by approved recovery stage B."""
        for parameter in self.policy.parameters():
            parameter.requires_grad = False

        if not isinstance(self.policy.actor, nn.Sequential):
            raise RuntimeError(
                "Stage B requires the locked sequential actor so its prefix can be reproduced exactly"
            )
        # Do not use ``children()`` here.  RSL's MLP registers the same ELU
        # instance at several positions, while Module.children() deliberately
        # de-duplicates shared module objects.  Iterating the Sequential itself
        # preserves all three activation applications used by actor.forward().
        self._recovery_actor_modules = list(self.policy.actor)
        last_name, last_linear = self._find_actor_last_linear()
        if last_linear.out_features != 16 or last_linear.in_features <= 0:
            raise RuntimeError(
                "Stage B requires the locked 16-action actor head; got "
                f"{last_linear.out_features}x{last_linear.in_features}"
            )
        if not self._recovery_actor_modules or self._recovery_actor_modules[-1] is not last_linear:
            raise RuntimeError("Stage B actor traversal does not end at the locked final action layer")

        self._recovery_last_linear_name = last_name
        self._recovery_last_linear = last_linear
        self._recovery_rows = (2, 3)
        self.recovery_rear_hip_weight = nn.Parameter(
            last_linear.weight.detach()[2:4].clone(), requires_grad=True
        )
        self.recovery_rear_hip_bias = nn.Parameter(
            last_linear.bias.detach()[2:4].clone(), requires_grad=True
        )
        actor_lr = float(getattr(self.policy, "student_recovery_actor_lr", 1.0e-5))
        if actor_lr <= 0.0:
            raise ValueError(f"student_recovery_actor_lr must be positive, got {actor_lr}")
        self.learning_rate = actor_lr
        self.optimizer = torch.optim.Adam(
            [self.recovery_rear_hip_weight, self.recovery_rear_hip_bias],
            lr=actor_lr,
        )
        # Keep the existing process-independent checkpoint hook interface.
        self.vae_optimizer = self.optimizer
        self.student_distill_update_count = 0
        self.student_actor_warmup_updates = 0
        self.student_recovery_epochs = int(getattr(self.policy, "student_recovery_epochs", 1))
        if self.student_recovery_epochs < 1:
            raise ValueError("student_recovery_epochs must be >= 1")
        self._recovery_binding_ready = False
        self._recovery_extra_restored = False
        self._recovery_binding_manifest = None
        self.teacher_actor = None
        self.teacher_priv_encoder = None
        self._teacher_actor_synced = False
        self._validate_stage_b_optimizer_scope()

    def _validate_stage_b_optimizer_scope(self) -> None:
        owned = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(owned) != 2 or owned[0] is not self.recovery_rear_hip_weight or owned[1] is not self.recovery_rear_hip_bias:
            raise RuntimeError("Stage B optimizer must own exactly the RL/RR hip row weight and bias")
        expected_weight_shape = (2, self._recovery_last_linear.in_features)
        if tuple(self.recovery_rear_hip_weight.shape) != expected_weight_shape:
            raise RuntimeError(
                f"Stage B rear-hip weight shape mismatch: expected {expected_weight_shape}, "
                f"got {tuple(self.recovery_rear_hip_weight.shape)}"
            )
        if tuple(self.recovery_rear_hip_bias.shape) != (2,):
            raise RuntimeError("Stage B rear-hip bias must have shape (2,)")

    def _stage_b_actor_prefix(self, actor_input: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Run the exact frozen actor prefix and prove equivalence to actor.forward()."""
        current_modules = list(self.policy.actor)
        if len(current_modules) != len(self._recovery_actor_modules) or any(
            current is not expected
            for current, expected in zip(current_modules, self._recovery_actor_modules, strict=True)
        ):
            raise RuntimeError("Stage B actor module sequence changed after optimizer construction")
        if current_modules[-1] is not self._recovery_last_linear:
            raise RuntimeError("Stage B actor final layer changed after optimizer construction")

        with torch.no_grad():
            hidden = actor_input
            for module in current_modules[:-1]:
                hidden = module(hidden)
            reconstructed = F.linear(
                hidden,
                self._recovery_last_linear.weight[2:4],
                self._recovery_last_linear.bias[2:4],
            )
            reference = self.policy.actor(actor_input)[..., 2:4]
            max_error = float(torch.max(torch.abs(reconstructed - reference)).item())
            if not torch.allclose(reconstructed, reference, rtol=1.0e-6, atol=1.0e-6):
                raise RuntimeError(
                    "Stage B actor-prefix reconstruction differs from the live Student actor "
                    f"(max_abs_error={max_error})"
                )
        return hidden, max_error

    @staticmethod
    def _load_hashed_checkpoint(path: str, expected_sha256: str) -> tuple[dict, str]:
        if not path or not os.path.isabs(path):
            raise RuntimeError(f"Recovery checkpoint path must be absolute, got {path!r}")
        with open(path, "rb") as checkpoint_file:
            payload = checkpoint_file.read()
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checkpoint SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
            )
        checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise RuntimeError(f"Checkpoint has no model_state_dict: {path}")
        return checkpoint, actual_sha256

    @staticmethod
    def _component_state(model_state: dict, prefix: str) -> dict:
        component = {
            key[len(prefix):]: value
            for key, value in model_state.items()
            if key.startswith(prefix)
        }
        if not component:
            raise RuntimeError(f"Checkpoint is missing required component {prefix!r}")
        return component

    @staticmethod
    def _component_sha256(component_state: dict) -> str:
        digest = hashlib.sha256()
        for key in sorted(component_state):
            tensor = component_state[key]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Non-tensor value in checkpoint component: {key}")
            cpu_tensor = tensor.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(cpu_tensor.dtype).encode("ascii"))
            digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
            digest.update(cpu_tensor.numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _assert_module_matches_checkpoint(module: nn.Module, component_state: dict, label: str) -> None:
        current = module.state_dict()
        if set(current) != set(component_state):
            missing = sorted(set(current) - set(component_state))
            extra = sorted(set(component_state) - set(current))
            raise RuntimeError(f"{label} checkpoint keys mismatch: missing={missing}, extra={extra}")
        for key, current_tensor in current.items():
            source_tensor = component_state[key]
            if tuple(current_tensor.shape) != tuple(source_tensor.shape):
                raise RuntimeError(
                    f"{label} shape mismatch at {key}: {tuple(current_tensor.shape)} != {tuple(source_tensor.shape)}"
                )
            if not torch.equal(current_tensor.detach().cpu(), source_tensor.detach().cpu()):
                raise RuntimeError(f"{label} weights do not match the bound Student checkpoint at {key}")

    def _sync_recovery_rows_to_actor(self) -> None:
        with torch.no_grad():
            self._recovery_last_linear.weight[2:4].copy_(self.recovery_rear_hip_weight)
            self._recovery_last_linear.bias[2:4].copy_(self.recovery_rear_hip_bias)

    def _reset_recovery_rows_from_actor(self) -> None:
        with torch.no_grad():
            self.recovery_rear_hip_weight.copy_(self._recovery_last_linear.weight[2:4])
            self.recovery_rear_hip_bias.copy_(self._recovery_last_linear.bias[2:4])

    def _capture_recovery_frozen_snapshot(self) -> None:
        self._recovery_frozen_snapshot = {
            key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()
        }

    def _assert_recovery_frozen_unchanged(self) -> None:
        if not hasattr(self, "_recovery_frozen_snapshot"):
            raise RuntimeError("Stage B frozen snapshot has not been captured")
        weight_key = f"actor.{self._recovery_last_linear_name}.weight"
        bias_key = f"actor.{self._recovery_last_linear_name}.bias"
        current_state = self.policy.state_dict()
        if set(current_state) != set(self._recovery_frozen_snapshot):
            raise RuntimeError("Stage B policy state keys changed after binding")
        for key, current in current_state.items():
            before = self._recovery_frozen_snapshot[key]
            current_cpu = current.detach().cpu()
            if key == weight_key:
                frozen_rows = [row for row in range(current_cpu.shape[0]) if row not in self._recovery_rows]
                unchanged = torch.equal(current_cpu[frozen_rows], before[frozen_rows])
            elif key == bias_key:
                frozen_rows = [row for row in range(current_cpu.shape[0]) if row not in self._recovery_rows]
                unchanged = torch.equal(current_cpu[frozen_rows], before[frozen_rows])
            else:
                unchanged = torch.equal(current_cpu, before)
            if not unchanged:
                raise RuntimeError(f"Frozen Stage B tensor changed: {key}")
        if not torch.equal(
            self._recovery_last_linear.weight.detach()[2:4],
            self.recovery_rear_hip_weight.detach(),
        ) or not torch.equal(
            self._recovery_last_linear.bias.detach()[2:4],
            self.recovery_rear_hip_bias.detach(),
        ):
            raise RuntimeError("Materialized actor RL/RR hip rows differ from optimizer-owned rows")

    def _verify_r2_preregistered_files(self) -> None:
        prereg = self._r2_preregistration
        checks = [
            (
                prereg["authority"]["spec_path"],
                prereg["authority"]["spec_sha256"],
                "R2 spec",
            ),
            (
                prereg["route"]["same_state_aggregate"],
                prereg["route"]["same_state_aggregate_sha256"],
                "R2 same-state aggregate",
            ),
            (
                prereg["route"]["route_selection"],
                prereg["route"]["route_selection_sha256"],
                "R2 route selection",
            ),
        ]
        binding = prereg["checkpoint_binding"]
        for key, label in (
            ("protected_student_root", "protected Student root"),
            ("behavior_start_and_anchor", "B500 behavior anchor"),
            ("frozen_teacher", "frozen Teacher"),
        ):
            checks.append((binding[key]["path"], binding[key]["sha256"], label))
        anchor = binding["behavior_start_and_anchor"]
        checks.extend(
            [
                (
                    anchor["evaluation_manifest"],
                    anchor["evaluation_manifest_sha256"],
                    "B500 evaluation manifest",
                ),
                (anchor["capped_handoff"], anchor["capped_handoff_sha256"], "B500 capped handoff"),
            ]
        )
        teacher = binding["frozen_teacher"]
        checks.append((teacher["env_yaml"], teacher["env_yaml_sha256"], "Teacher env YAML"))
        helper_path = str(prereg["permanent_contract"]["shared_post_prior_helper"]).split("::", 1)[0]
        checks.append(
            (
                helper_path,
                prereg["permanent_contract"]["shared_post_prior_helper_file_sha256"],
                "shared post-prior helper",
            )
        )
        for raw_path, raw_sha, label in checks:
            path = os.path.realpath(str(raw_path))
            expected_sha = self._require_sha256(raw_sha, f"{label} SHA256")
            if not os.path.isabs(path):
                raise RuntimeError(f"{label} path must be absolute")
            actual_sha = self._sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"{label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
                )

    @staticmethod
    def _module_storage_pointers(module: nn.Module) -> set[int]:
        pointers: set[int] = set()
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.numel() > 0:
                pointers.add(int(tensor.data_ptr()))
        return pointers

    @classmethod
    def _assert_modules_do_not_share_storage(cls, modules: dict[str, nn.Module]) -> None:
        names = list(modules)
        pointers = {name: cls._module_storage_pointers(module) for name, module in modules.items()}
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = pointers[left] & pointers[right]
                if overlap:
                    raise RuntimeError(f"R2 modules share tensor storage: {left} and {right}")

    @staticmethod
    def _r2_vector(value, length: int, label: str) -> list[float]:
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if tensor.ndim > 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        tensor = tensor.reshape(-1)
        if tensor.numel() != length or not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"R2 runtime {label} must contain {length} finite values")
        return [float(item) for item in tensor.tolist()]

    def _bind_r2_runtime_contract(self, runtime_contract: dict) -> dict:
        if not isinstance(runtime_contract, dict):
            raise RuntimeError("R2 runtime_contract must be a dict")
        action = runtime_contract.get("action", runtime_contract)
        if not isinstance(action, dict):
            raise RuntimeError("R2 runtime action contract is missing")
        joint_order = action.get("joint_order", action.get("joint_names"))
        expected_order = self._r2_preregistration["permanent_contract"]["joint_order"]
        if joint_order != expected_order:
            raise RuntimeError("R2 runtime joint order differs from the permanent contract")
        scale = self._r2_vector(action.get("action_scale", action.get("scale")), 16, "action scale")
        expected_scale = [
            float(value) for value in self._r2_preregistration["permanent_contract"]["action_scale"]
        ]
        if scale != expected_scale:
            raise RuntimeError(f"R2 runtime action scale changed: {scale}")
        offset = self._r2_vector(action.get("action_offset", action.get("offset")), 16, "action offset")
        expected_offset = [0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4
        if not torch.allclose(
            torch.tensor(offset), torch.tensor(expected_offset), rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError(f"R2 runtime default_dof_pos/action offset changed: {offset}")
        clip_value = action.get("joint_pos_clip", action.get("clip"))
        clip = torch.as_tensor(clip_value, dtype=torch.float64)
        if clip.ndim == 3 and clip.shape[0] == 1:
            clip = clip[0]
        expected_clip = torch.tensor(
            self._r2_preregistration["permanent_contract"]["joint_pos_clip"],
            dtype=torch.float64,
        )
        if clip.shape != (16, 2) or not torch.equal(clip, expected_clip):
            raise RuntimeError("R2 runtime joint_pos.clip changed")
        context_shape = runtime_contract.get("teacher_context_shape")
        context_order = runtime_contract.get("teacher_context_order")
        if context_shape not in ([3], (3,)) or context_order != [
            "height_delta",
            "command_x",
            "front_rear_delta",
        ]:
            raise RuntimeError("R2 runtime teacher_context contract changed")
        self._r2_action_scale = torch.tensor(scale, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._r2_action_offset = torch.tensor(offset, device=self.device, dtype=torch.float32).unsqueeze(0)
        self._r2_action_clip = clip.to(device=self.device, dtype=torch.float32).unsqueeze(0)
        return {
            "joint_order": joint_order,
            "action_scale": scale,
            "action_offset": offset,
            "joint_pos_clip": clip.tolist(),
            "teacher_context_shape": [3],
            "teacher_context_order": context_order,
        }

    def _load_r2_teacher_prior_contract(self) -> dict:
        import yaml

        teacher = self._r2_preregistration["checkpoint_binding"]["frozen_teacher"]
        with open(teacher["env_yaml"], encoding="utf-8") as handle:
            payload = yaml.load(handle, Loader=yaml.BaseLoader)
        action = payload["actions"]["joint_pos"]
        expected_class = (
            "robot_lab.tasks.locomotion.velocity.mdp.actions:"
            "PhasedHighstepBoxBiasJointPositionAction"
        )
        if action.get("class_type") != expected_class:
            raise RuntimeError("Teacher env YAML no longer uses the phased highstep prior")
        if action.get("joint_names") != self._r2_preregistration["permanent_contract"]["joint_order"]:
            raise RuntimeError("Teacher env YAML joint order changed")
        values = (
            "height_threshold",
            "height_gate_width",
            "min_forward_command",
            "command_gate_width",
            "commit_height_delta_min",
            "commit_height_delta_target",
            "commit_gate_floor",
            "front_reach_box_bias",
            "rear_approach_box_bias",
            "front_support_box_bias",
            "rear_push_box_bias",
            "min_box_target",
            "max_box_target",
        )
        contract = {name: float(action[name]) for name in values}
        contract.update(
            {
                "box_action_ids": torch.tensor([12, 13, 14, 15], device=self.device, dtype=torch.long),
                "terrain_gate_available": True,
                "update_count": float(teacher["teacher_update_count_for_prior"]),
                "prior_start_update": float(action["prior_start_update"]),
                "prior_full_update": float(action["prior_full_update"]),
            }
        )
        if contract["update_count"] < contract["prior_full_update"]:
            raise RuntimeError("Teacher prior was not at its preregistered full scale")
        return contract

    def _capture_r2_frozen_snapshot(self) -> None:
        self._r2_frozen_policy_snapshot = {
            key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()
        }
        self._r2_frozen_module_snapshots = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in {
                "teacher_actor": self.r2_teacher_actor,
                "teacher_priv": self.r2_teacher_priv_encoder,
                "anchor_actor": self.r2_anchor_actor,
                "anchor_estimator": self.r2_anchor_estimator,
            }.items()
        }

    def _assert_r2_live_lineage_scope(self, anchor_model_state: dict) -> None:
        """Prove a resumed R2 policy differs from B500 only in preregistered tensors/rows."""
        current = self.policy.state_dict()
        if set(current) != set(anchor_model_state):
            raise RuntimeError("R2 live/B500 policy state keys differ")
        allowed = set(self._r2_exact_optimizer_names()[:6])
        weight_key = "actor.6.weight"
        bias_key = "actor.6.bias"
        for key, value in current.items():
            reference = anchor_model_state[key].detach().cpu()
            actual = value.detach().cpu()
            if key in allowed:
                if not bool(torch.isfinite(actual).all().item()):
                    raise RuntimeError(f"R2 live trainable tensor is non-finite: {key}")
                continue
            if key == weight_key:
                equal = torch.equal(actual[:12], reference[:12])
            elif key == bias_key:
                equal = torch.equal(actual[:12], reference[:12])
            else:
                equal = torch.equal(actual, reference)
            if not equal:
                raise RuntimeError(f"R2 full-resume scope escaped B500 at {key}")

    def _assert_r2_frozen_unchanged(self) -> None:
        if not hasattr(self, "_r2_frozen_policy_snapshot"):
            raise RuntimeError("R2 frozen snapshot has not been captured")
        allowed = set(self._r2_exact_optimizer_names()[:6])
        weight_key = f"actor.{self._r2_last_linear_name}.weight"
        bias_key = f"actor.{self._r2_last_linear_name}.bias"
        current = self.policy.state_dict()
        if set(current) != set(self._r2_frozen_policy_snapshot):
            raise RuntimeError("R2 policy state keys changed")
        for key, value in current.items():
            before = self._r2_frozen_policy_snapshot[key]
            now = value.detach().cpu()
            if key in allowed:
                if not bool(torch.isfinite(now).all().item()):
                    raise RuntimeError(f"R2 trainable tensor became non-finite: {key}")
                continue
            if key == weight_key:
                unchanged = torch.equal(now[:12], before[:12])
            elif key == bias_key:
                unchanged = torch.equal(now[:12], before[:12])
            else:
                unchanged = torch.equal(now, before)
            if not unchanged:
                raise RuntimeError(f"Frozen R2 Student tensor changed: {key}")
        if not torch.equal(
            self._r2_last_linear.weight.detach()[12:16], self.r2_box_weight.detach()
        ) or not torch.equal(
            self._r2_last_linear.bias.detach()[12:16], self.r2_box_bias.detach()
        ):
            raise RuntimeError("R2 materialized box rows differ from optimizer-owned leaves")
        modules = {
            "teacher_actor": self.r2_teacher_actor,
            "teacher_priv": self.r2_teacher_priv_encoder,
            "anchor_actor": self.r2_anchor_actor,
            "anchor_estimator": self.r2_anchor_estimator,
        }
        for name, module in modules.items():
            reference = self._r2_frozen_module_snapshots[name]
            state = module.state_dict()
            if set(state) != set(reference) or any(
                not torch.equal(value.detach().cpu(), reference[key]) for key, value in state.items()
            ):
                raise RuntimeError(f"Frozen R2 module changed: {name}")

    def bind_student_recovery_r2_checkpoints(
        self,
        *,
        loaded_student_checkpoint: str,
        checkpoint_load_mode: str,
        schedule_resume_mode: str,
        runtime_contract: dict,
    ) -> dict:
        """Bind B500, independent Teacher, exact prior contract and R2 resume state."""
        if self.student_recovery_stage != "R2":
            raise RuntimeError("R2 checkpoint binding requires student_recovery_stage=R2")
        if schedule_resume_mode != "preserve":
            raise RuntimeError("R2 requires highstep schedule preserve")
        self._verify_r2_preregistered_files()
        prereg_binding = self._r2_preregistration["checkpoint_binding"]
        root_cfg = prereg_binding["protected_student_root"]
        anchor_cfg = prereg_binding["behavior_start_and_anchor"]
        teacher_cfg = prereg_binding["frozen_teacher"]
        root_checkpoint, root_sha = self._load_hashed_checkpoint(root_cfg["path"], root_cfg["sha256"])
        anchor_checkpoint, anchor_sha = self._load_hashed_checkpoint(
            anchor_cfg["path"], anchor_cfg["sha256"]
        )
        teacher_checkpoint, teacher_sha = self._load_hashed_checkpoint(
            teacher_cfg["path"], teacher_cfg["sha256"]
        )
        del root_checkpoint
        loaded_path = os.path.realpath(loaded_student_checkpoint)
        if checkpoint_load_mode == "weights_only":
            if loaded_path != os.path.realpath(anchor_cfg["path"]):
                raise RuntimeError("Fresh R2 must load the preregistered B500 checkpoint weights-only")
            if self.student_distill_update_count != 0 or self.optimizer.state:
                raise RuntimeError("Fresh R2 must start with count=0 and empty Adam state")
            self._assert_module_matches_checkpoint(
                self.policy.actor,
                self._component_state(anchor_checkpoint["model_state_dict"], "actor."),
                "R2 live Student actor",
            )
            self._assert_module_matches_checkpoint(
                self.policy.estimator,
                self._component_state(anchor_checkpoint["model_state_dict"], "estimator."),
                "R2 live Student estimator",
            )
            self._reset_r2_box_rows_from_actor()
        elif checkpoint_load_mode == "full":
            if not self._r2_extra_restored or not isinstance(self._r2_binding_manifest, dict):
                raise RuntimeError("R2 full resume did not restore R2 Adam/count/binding state")
            restored = self._r2_binding_manifest
            required = {
                "preregistration_sha256": self._r2_preregistration_sha256,
                "initial_student_checkpoint": os.path.realpath(anchor_cfg["path"]),
                "initial_student_sha256": anchor_sha,
                "teacher_checkpoint": os.path.realpath(teacher_cfg["path"]),
                "teacher_sha256": teacher_sha,
            }
            for key, expected in required.items():
                if restored.get(key) != expected:
                    raise RuntimeError(f"R2 full-resume binding mismatch at {key}")
            if loaded_path == os.path.realpath(anchor_cfg["path"]):
                raise RuntimeError("R2 full resume cannot reinterpret B500 as an R2 checkpoint")
            self._sync_r2_box_rows_to_actor()
        else:
            raise RuntimeError(f"Unsupported R2 checkpoint load mode: {checkpoint_load_mode!r}")

        parent_extra = (anchor_checkpoint.get("infos") or {}).get(
            "robot_lab_algorithm_checkpoint_state"
        )
        parent_recovery = parent_extra.get("student_recovery") if isinstance(parent_extra, dict) else None
        if (
            not isinstance(parent_recovery, dict)
            or parent_recovery.get("stage") != "B"
            or parent_recovery.get("effective_update_count") != 500
            or parent_extra.get("student_distill_update_count") != 500
        ):
            raise RuntimeError("B500 checkpoint lacks the preregistered corrected Stage-B count/binding")
        parent_manifest = parent_recovery.get("binding_manifest")
        if not isinstance(parent_manifest, dict) or parent_manifest.get(
            "initial_student_checkpoint"
        ) != os.path.realpath(root_cfg["path"]) or parent_manifest.get(
            "initial_student_sha256"
        ) != root_sha:
            raise RuntimeError("B500 checkpoint lineage does not bind the protected model_900 root")

        anchor_actor_state = self._component_state(anchor_checkpoint["model_state_dict"], "actor.")
        anchor_estimator_state = self._component_state(
            anchor_checkpoint["model_state_dict"], "estimator."
        )
        teacher_actor_state = self._component_state(teacher_checkpoint["model_state_dict"], "actor.")
        teacher_priv_state = self._component_state(
            teacher_checkpoint["model_state_dict"], "priv_encoder."
        )
        teacher_actor_sha = self._component_sha256(teacher_actor_state)
        teacher_priv_sha = self._component_sha256(teacher_priv_state)
        expected_parent_teacher = {
            "teacher_checkpoint": os.path.realpath(teacher_cfg["path"]),
            "teacher_sha256": teacher_sha,
            "teacher_actor_component_sha256": teacher_actor_sha,
            "teacher_privileged_encoder_component_sha256": teacher_priv_sha,
        }
        for key, expected in expected_parent_teacher.items():
            if parent_manifest.get(key) != expected:
                raise RuntimeError(f"B500 parent Teacher binding mismatch at {key}")
        self._assert_r2_live_lineage_scope(anchor_checkpoint["model_state_dict"])
        self.r2_anchor_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.r2_anchor_actor.load_state_dict(anchor_actor_state, strict=True)
        self.r2_anchor_estimator = copy.deepcopy(self.policy.estimator).to(self.device)
        self.r2_anchor_estimator.load_state_dict(anchor_estimator_state, strict=True)
        self.r2_teacher_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.r2_teacher_actor.load_state_dict(teacher_actor_state, strict=True)
        self.r2_teacher_priv_encoder = copy.deepcopy(self.policy.priv_encoder).to(self.device)
        self.r2_teacher_priv_encoder.load_state_dict(teacher_priv_state, strict=True)
        frozen_modules = (
            self.r2_anchor_actor,
            self.r2_anchor_estimator,
            self.r2_teacher_actor,
            self.r2_teacher_priv_encoder,
        )
        for module in frozen_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
                parameter.grad = None
        self._assert_modules_do_not_share_storage(
            {
                "live_actor": self.policy.actor,
                "live_estimator": self.policy.estimator,
                "live_priv": self.policy.priv_encoder,
                "anchor_actor": self.r2_anchor_actor,
                "anchor_estimator": self.r2_anchor_estimator,
                "teacher_actor": self.r2_teacher_actor,
                "teacher_priv": self.r2_teacher_priv_encoder,
            }
        )
        runtime_manifest = self._bind_r2_runtime_contract(runtime_contract)
        self._r2_prior_contract = self._load_r2_teacher_prior_contract()
        self._validate_r2_optimizer_scope()
        probe = torch.linspace(
            -0.25,
            0.25,
            steps=self._r2_first_linear.in_features,
            device=self.device,
            dtype=self._r2_first_linear.weight.dtype,
        ).unsqueeze(0)
        _, prefix_error = self._r2_actor_forward(probe)
        self._r2_binding_manifest = {
            "schema_version": 1,
            "stage": "R2",
            "preregistration_path": self._r2_preregistration_path,
            "preregistration_sha256": self._r2_preregistration_sha256,
            "protected_student_root": os.path.realpath(root_cfg["path"]),
            "protected_student_root_sha256": root_sha,
            "initial_student_checkpoint": os.path.realpath(anchor_cfg["path"]),
            "initial_student_sha256": anchor_sha,
            "loaded_student_checkpoint": loaded_path,
            "checkpoint_load_mode": checkpoint_load_mode,
            "schedule_resume_mode": schedule_resume_mode,
            "teacher_checkpoint": os.path.realpath(teacher_cfg["path"]),
            "teacher_sha256": teacher_sha,
            "teacher_env_yaml": os.path.realpath(teacher_cfg["env_yaml"]),
            "teacher_env_yaml_sha256": teacher_cfg["env_yaml_sha256"],
            "effective_update_count": int(self.student_distill_update_count),
            "optimizer_parameter_names": list(self._r2_optimizer_names),
            "optimizer_parameter_tensor_count": 8,
            "trainable_live_tensors": list(self._r2_optimizer_names[:6]),
            "materialized_actor_rows": [12, 13, 14, 15],
            "teacher_independent_storage": True,
            "anchor_independent_storage": True,
            "actor_prefix_equivalence_max_error": prefix_error,
            "runtime_contract": runtime_manifest,
        }
        self._r2_binding_ready = True
        self._teacher_actor_synced = True
        self._capture_r2_frozen_snapshot()
        self._assert_r2_frozen_unchanged()
        return copy.deepcopy(self._r2_binding_manifest)

    def _verify_r3_preregistered_files(self) -> None:
        prereg = self._r3_preregistration
        checks = [
            (prereg["authority"]["spec_path"], prereg["authority"]["spec_sha256"], "spec"),
            (
                prereg["authority"]["route_report"],
                prereg["authority"]["route_report_sha256"],
                "route report",
            ),
        ]
        for record in prereg["evidence"].values():
            checks.append((record["path"], record["sha256"], "R3 evidence"))
        binding = prereg["checkpoint_binding"]
        for key in (
            "protected_student_root",
            "behavior_reference_b500",
            "r3_start_and_anchor",
            "frozen_teacher",
        ):
            checks.append((binding[key]["path"], binding[key]["sha256"], key))
        teacher = binding["frozen_teacher"]
        checks.append((teacher["env_yaml"], teacher["env_yaml_sha256"], "Teacher env"))
        helper_path = prereg["permanent_contract"]["shared_post_prior_helper"].split("::", 1)[0]
        checks.append(
            (
                helper_path,
                prereg["permanent_contract"]["shared_post_prior_helper_file_sha256"],
                "post-prior helper",
            )
        )
        for raw_path, raw_sha, label in checks:
            path = os.path.realpath(str(raw_path))
            expected_sha = self._require_sha256(raw_sha, f"R3 {label} SHA256")
            if not os.path.isabs(path) or not os.path.isfile(path):
                raise RuntimeError(f"R3 {label} is unavailable: {path}")
            actual_sha = self._sha256_file(path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"R3 {label} SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
                )

    def _assert_r3_live_lineage_scope(self, anchor_model_state: dict) -> None:
        current = self.policy.state_dict()
        if set(current) != set(anchor_model_state):
            raise RuntimeError("R3 live/anchor policy state keys differ")
        allowed = set(self._r3_optimizer_names)
        for key, value in current.items():
            actual = value.detach().cpu()
            reference = anchor_model_state[key].detach().cpu()
            if key in allowed:
                if not bool(torch.isfinite(actual).all().item()):
                    raise RuntimeError(f"R3 live action head is non-finite: {key}")
            elif not torch.equal(actual, reference):
                raise RuntimeError(f"R3 full-resume scope escaped the R2-50 anchor at {key}")

    def _capture_r3_frozen_snapshot(self) -> None:
        self._r3_frozen_policy_snapshot = {
            key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()
        }
        self._r3_frozen_module_snapshots = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in {
                "teacher_actor": self.r3_teacher_actor,
                "teacher_priv": self.r3_teacher_priv_encoder,
                "anchor_actor": self.r3_anchor_actor,
                "anchor_estimator": self.r3_anchor_estimator,
            }.items()
        }

    def _assert_r3_frozen_unchanged(self) -> None:
        if not hasattr(self, "_r3_frozen_policy_snapshot"):
            raise RuntimeError("R3 frozen snapshot has not been captured")
        current = self.policy.state_dict()
        if set(current) != set(self._r3_frozen_policy_snapshot):
            raise RuntimeError("R3 policy state keys changed")
        allowed = set(self._r3_optimizer_names)
        for key, value in current.items():
            now = value.detach().cpu()
            before = self._r3_frozen_policy_snapshot[key]
            if key in allowed:
                if not bool(torch.isfinite(now).all().item()):
                    raise RuntimeError(f"R3 trainable action head became non-finite: {key}")
            elif not torch.equal(now, before):
                raise RuntimeError(f"Frozen R3 Student tensor changed: {key}")
        for name, module in {
            "teacher_actor": self.r3_teacher_actor,
            "teacher_priv": self.r3_teacher_priv_encoder,
            "anchor_actor": self.r3_anchor_actor,
            "anchor_estimator": self.r3_anchor_estimator,
        }.items():
            reference = self._r3_frozen_module_snapshots[name]
            state = module.state_dict()
            if set(state) != set(reference) or any(
                not torch.equal(value.detach().cpu(), reference[key])
                for key, value in state.items()
            ):
                raise RuntimeError(f"Frozen R3 module changed: {name}")

    def bind_student_recovery_r3_checkpoints(
        self,
        *,
        loaded_student_checkpoint: str,
        checkpoint_load_mode: str,
        schedule_resume_mode: str,
        runtime_contract: dict,
    ) -> dict:
        """Bind the R2-50 anchor, protected lineage and independent Teacher for R3."""
        if self.student_recovery_stage != "R3":
            raise RuntimeError("R3 binding requires student_recovery_stage=R3")
        if schedule_resume_mode != "preserve":
            raise RuntimeError("R3 requires highstep schedule preserve")
        self._verify_r3_preregistered_files()
        binding = self._r3_preregistration["checkpoint_binding"]
        root_cfg = binding["protected_student_root"]
        b500_cfg = binding["behavior_reference_b500"]
        anchor_cfg = binding["r3_start_and_anchor"]
        teacher_cfg = binding["frozen_teacher"]
        _, root_sha = self._load_hashed_checkpoint(root_cfg["path"], root_cfg["sha256"])
        _, b500_sha = self._load_hashed_checkpoint(b500_cfg["path"], b500_cfg["sha256"])
        anchor_checkpoint, anchor_sha = self._load_hashed_checkpoint(
            anchor_cfg["path"], anchor_cfg["sha256"]
        )
        teacher_checkpoint, teacher_sha = self._load_hashed_checkpoint(
            teacher_cfg["path"], teacher_cfg["sha256"]
        )
        loaded_path = os.path.realpath(loaded_student_checkpoint)
        anchor_path = os.path.realpath(anchor_cfg["path"])
        if checkpoint_load_mode == "weights_only":
            if loaded_path != anchor_path:
                raise RuntimeError("Fresh R3 must weights-only load the preregistered R2-50")
            if self.student_distill_update_count != 0 or self.optimizer.state:
                raise RuntimeError("Fresh R3 must start at count=0 with empty Adam state")
            self._assert_module_matches_checkpoint(
                self.policy.actor,
                self._component_state(anchor_checkpoint["model_state_dict"], "actor."),
                "R3 live actor",
            )
            self._assert_module_matches_checkpoint(
                self.policy.estimator,
                self._component_state(anchor_checkpoint["model_state_dict"], "estimator."),
                "R3 live estimator",
            )
        elif checkpoint_load_mode == "full":
            if not self._r3_extra_restored or not isinstance(self._r3_binding_manifest, dict):
                raise RuntimeError("R3 full resume did not restore Adam/count/binding state")
            required = {
                "preregistration_sha256": self._r3_preregistration_sha256,
                "initial_student_checkpoint": anchor_path,
                "initial_student_sha256": anchor_sha,
                "teacher_checkpoint": os.path.realpath(teacher_cfg["path"]),
                "teacher_sha256": teacher_sha,
            }
            for key, expected in required.items():
                if self._r3_binding_manifest.get(key) != expected:
                    raise RuntimeError(f"R3 full-resume binding mismatch at {key}")
            if loaded_path == anchor_path:
                raise RuntimeError("R3 full resume cannot reinterpret R2-50 as an R3 checkpoint")
        else:
            raise RuntimeError(f"Unsupported R3 checkpoint load mode: {checkpoint_load_mode!r}")

        parent_extra = (anchor_checkpoint.get("infos") or {}).get(
            "robot_lab_algorithm_checkpoint_state"
        )
        parent_recovery = parent_extra.get("student_recovery") if isinstance(parent_extra, dict) else None
        parent_manifest = (
            parent_recovery.get("binding_manifest") if isinstance(parent_recovery, dict) else None
        )
        if (
            not isinstance(parent_recovery, dict)
            or parent_recovery.get("stage") != "R2"
            or parent_recovery.get("effective_update_count") != 50
            or parent_extra.get("student_distill_update_count") != 50
            or not isinstance(parent_manifest, dict)
            or parent_manifest.get("protected_student_root") != os.path.realpath(root_cfg["path"])
            or parent_manifest.get("protected_student_root_sha256") != root_sha
            or parent_manifest.get("initial_student_checkpoint") != os.path.realpath(b500_cfg["path"])
            or parent_manifest.get("initial_student_sha256") != b500_sha
            or parent_manifest.get("teacher_checkpoint") != os.path.realpath(teacher_cfg["path"])
            or parent_manifest.get("teacher_sha256") != teacher_sha
        ):
            raise RuntimeError("R2-50 parent lineage does not bind model_900/B500/Teacher exactly")

        anchor_actor_state = self._component_state(anchor_checkpoint["model_state_dict"], "actor.")
        anchor_estimator_state = self._component_state(
            anchor_checkpoint["model_state_dict"], "estimator."
        )
        teacher_actor_state = self._component_state(teacher_checkpoint["model_state_dict"], "actor.")
        teacher_priv_state = self._component_state(
            teacher_checkpoint["model_state_dict"], "priv_encoder."
        )
        self._assert_r3_live_lineage_scope(anchor_checkpoint["model_state_dict"])
        self.r3_anchor_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.r3_anchor_actor.load_state_dict(anchor_actor_state, strict=True)
        self.r3_anchor_estimator = copy.deepcopy(self.policy.estimator).to(self.device)
        self.r3_anchor_estimator.load_state_dict(anchor_estimator_state, strict=True)
        self.r3_teacher_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.r3_teacher_actor.load_state_dict(teacher_actor_state, strict=True)
        self.r3_teacher_priv_encoder = copy.deepcopy(self.policy.priv_encoder).to(self.device)
        self.r3_teacher_priv_encoder.load_state_dict(teacher_priv_state, strict=True)
        for module in (
            self.r3_anchor_actor,
            self.r3_anchor_estimator,
            self.r3_teacher_actor,
            self.r3_teacher_priv_encoder,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
                parameter.grad = None
        self._assert_modules_do_not_share_storage(
            {
                "live_actor": self.policy.actor,
                "live_estimator": self.policy.estimator,
                "live_priv": self.policy.priv_encoder,
                "anchor_actor": self.r3_anchor_actor,
                "anchor_estimator": self.r3_anchor_estimator,
                "teacher_actor": self.r3_teacher_actor,
                "teacher_priv": self.r3_teacher_priv_encoder,
            }
        )
        runtime_manifest = self._bind_r2_runtime_contract(runtime_contract)
        self._r2_prior_contract = self._load_r2_teacher_prior_contract()
        self._validate_r3_optimizer_scope()
        self._r3_binding_manifest = {
            "schema_version": 1,
            "stage": "R3",
            "preregistration_path": self._r3_preregistration_path,
            "preregistration_sha256": self._r3_preregistration_sha256,
            "protected_student_root": os.path.realpath(root_cfg["path"]),
            "protected_student_root_sha256": root_sha,
            "behavior_reference_b500": os.path.realpath(b500_cfg["path"]),
            "behavior_reference_b500_sha256": b500_sha,
            "initial_student_checkpoint": anchor_path,
            "initial_student_sha256": anchor_sha,
            "loaded_student_checkpoint": loaded_path,
            "checkpoint_load_mode": checkpoint_load_mode,
            "schedule_resume_mode": schedule_resume_mode,
            "teacher_checkpoint": os.path.realpath(teacher_cfg["path"]),
            "teacher_sha256": teacher_sha,
            "teacher_env_yaml": os.path.realpath(teacher_cfg["env_yaml"]),
            "teacher_env_yaml_sha256": teacher_cfg["env_yaml_sha256"],
            "effective_update_count": int(self.student_distill_update_count),
            "optimizer_parameter_names": list(self._r3_optimizer_names),
            "optimizer_parameter_tensor_count": 2,
            "teacher_independent_storage": True,
            "anchor_independent_storage": True,
            "runtime_contract": runtime_manifest,
        }
        self._r3_binding_ready = True
        self._teacher_actor_synced = True
        self._capture_r3_frozen_snapshot()
        self._assert_r3_frozen_unchanged()
        return copy.deepcopy(self._r3_binding_manifest)

    def bind_student_recovery_checkpoints(
        self,
        *,
        loaded_student_checkpoint: str,
        checkpoint_load_mode: str,
    ) -> dict:
        """Fail-closed binding of model_900 Student and model_172300 Teacher."""
        if self.student_recovery_stage != "B":
            raise RuntimeError("Student recovery checkpoint binding is only valid for stage B")

        source_path = os.path.realpath(str(self.policy.student_recovery_source_checkpoint))
        source_expected_sha = str(self.policy.student_recovery_source_sha256)
        teacher_path = os.path.realpath(str(self.policy.student_recovery_teacher_checkpoint))
        teacher_expected_sha = str(self.policy.student_recovery_teacher_sha256)
        loaded_path = os.path.realpath(loaded_student_checkpoint)
        source_checkpoint, source_sha = self._load_hashed_checkpoint(source_path, source_expected_sha)
        teacher_checkpoint, teacher_sha = self._load_hashed_checkpoint(teacher_path, teacher_expected_sha)

        if checkpoint_load_mode == "weights_only":
            if loaded_path != source_path:
                raise RuntimeError(
                    f"Fresh stage B must load the canonical model_900: {loaded_path} != {source_path}"
                )
            self._assert_module_matches_checkpoint(
                self.policy.actor,
                self._component_state(source_checkpoint["model_state_dict"], "actor."),
                "Student actor",
            )
            self._assert_module_matches_checkpoint(
                self.policy.estimator,
                self._component_state(source_checkpoint["model_state_dict"], "estimator."),
                "Student estimator",
            )
            self._reset_recovery_rows_from_actor()
            self.student_distill_update_count = 0
        elif checkpoint_load_mode == "full":
            if not self._recovery_extra_restored:
                raise RuntimeError("Full stage B resume did not restore recovery optimizer/count state")
            restored = self._recovery_binding_manifest or {}
            if restored.get("initial_student_checkpoint") != source_path or restored.get("initial_student_sha256") != source_sha:
                raise RuntimeError("Full stage B resume has a different initial Student binding")
            if restored.get("teacher_checkpoint") != teacher_path or restored.get("teacher_sha256") != teacher_sha:
                raise RuntimeError("Full stage B resume has a different Teacher binding")
            self._sync_recovery_rows_to_actor()
        else:
            raise RuntimeError(f"Unsupported stage B checkpoint load mode: {checkpoint_load_mode!r}")

        teacher_actor_state = self._component_state(teacher_checkpoint["model_state_dict"], "actor.")
        teacher_priv_state = self._component_state(
            teacher_checkpoint["model_state_dict"], "priv_encoder."
        )
        self.teacher_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.teacher_actor.load_state_dict(teacher_actor_state, strict=True)
        self.teacher_actor.eval()
        self.teacher_priv_encoder = copy.deepcopy(self.policy.priv_encoder).to(self.device)
        self.teacher_priv_encoder.load_state_dict(teacher_priv_state, strict=True)
        self.teacher_priv_encoder.eval()
        for module in (self.teacher_actor, self.teacher_priv_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False

        student_actor_param = next(self.policy.actor.parameters())
        teacher_actor_param = next(self.teacher_actor.parameters())
        student_priv_param = next(self.policy.priv_encoder.parameters())
        teacher_priv_param = next(self.teacher_priv_encoder.parameters())
        if student_actor_param.data_ptr() == teacher_actor_param.data_ptr() or student_priv_param.data_ptr() == teacher_priv_param.data_ptr():
            raise RuntimeError("Teacher modules share storage with the live Student")

        with torch.no_grad():
            priv_input = torch.zeros(
                2,
                next(self.teacher_priv_encoder.parameters()).shape[1],
                device=self.device,
            )
            target_latent = self.teacher_priv_encoder(priv_input)
            actor_input = torch.zeros(
                2,
                next(self.teacher_actor.parameters()).shape[1],
                device=self.device,
            )
            actor_input[:, -target_latent.shape[1]:] = target_latent
            fixed_output = self.teacher_actor(actor_input)
            if not torch.isfinite(target_latent).all() or not torch.isfinite(fixed_output).all():
                raise RuntimeError("Teacher fixed-observation binding check produced non-finite output")

            first_linear = next(
                (module for module in self._recovery_actor_modules if isinstance(module, nn.Linear)),
                None,
            )
            if first_linear is None:
                raise RuntimeError("Stage B actor has no input linear layer")
            prefix_probe = torch.linspace(
                -0.25,
                0.25,
                steps=first_linear.in_features,
                device=self.device,
                dtype=first_linear.weight.dtype,
            ).unsqueeze(0).repeat(2, 1)
            _, actor_prefix_max_error = self._stage_b_actor_prefix(prefix_probe)

        teacher_actor_sha = self._component_sha256(teacher_actor_state)
        teacher_priv_sha = self._component_sha256(teacher_priv_state)
        student_actor_sha = self._component_sha256(
            self._component_state(source_checkpoint["model_state_dict"], "actor.")
        )
        if teacher_actor_sha == student_actor_sha:
            raise RuntimeError("Teacher actor unexpectedly equals the Student source actor")

        self._recovery_binding_manifest = {
            "schema_version": 1,
            "stage": "B",
            "initial_student_checkpoint": source_path,
            "initial_student_sha256": source_sha,
            "loaded_student_checkpoint": loaded_path,
            "teacher_checkpoint": teacher_path,
            "teacher_sha256": teacher_sha,
            "teacher_actor_component_sha256": teacher_actor_sha,
            "teacher_privileged_encoder_component_sha256": teacher_priv_sha,
            "student_actor_component_sha256": student_actor_sha,
            "trainable_action_rows": [2, 3],
            "trainable_joint_names": ["RL_hip_joint", "RR_hip_joint"],
            "optimizer_parameter_shapes": [
                list(self.recovery_rear_hip_weight.shape),
                list(self.recovery_rear_hip_bias.shape),
            ],
            "actor_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "warmup_updates": 0,
            "effective_update_count": int(self.student_distill_update_count),
            "estimator_frozen": True,
            "actor_body_frozen": True,
            "critic_frozen": True,
            "student_privileged_encoder_frozen": True,
            "box_rows_frozen": True,
            "teacher_independent_storage": True,
            "fixed_observation_check_finite": True,
            "actor_prefix_equivalence_verified": True,
            "actor_prefix_equivalence_max_error": actor_prefix_max_error,
        }
        self._recovery_binding_ready = True
        self._teacher_actor_synced = True
        self._validate_stage_b_optimizer_scope()
        self._capture_recovery_frozen_snapshot()
        self._assert_recovery_frozen_unchanged()
        return dict(self._recovery_binding_manifest)

    def _load_v15_preregistration(self) -> tuple[dict, str, str]:
        zero_scale_ablation = self.student_recovery_stage == "0707_EXACT"
        historical_0707_exact = self.student_recovery_stage == "HISTORICAL_0707_EXACT"
        be300_0707 = self.student_recovery_stage == "BE300_0707"
        b300_hybrid = self.student_recovery_stage == "B300_CANONICAL_HYBRID"
        environment_curriculum_v18 = self.student_recovery_stage == "ENV_CURRICULUM_V18"
        e1400_continuation = self.student_recovery_stage == "E1400_CONTINUATION"
        if b300_hybrid:
            path_attr = "student_recovery_b300_hybrid_preregistration_path"
            sha_attr = "student_recovery_b300_hybrid_preregistration_sha256"
            authority_label = "B300 canonical hybrid preregistration SHA256"
        elif e1400_continuation:
            path_attr = "student_recovery_e1400_continuation_preregistration_path"
            sha_attr = "student_recovery_e1400_continuation_preregistration_sha256"
            authority_label = "E1400 continuation v1.10 preregistration SHA256"
        elif environment_curriculum_v18:
            path_attr = "student_recovery_v18_preregistration_path"
            sha_attr = "student_recovery_v18_preregistration_sha256"
            authority_label = "v1.8 environment curriculum preregistration SHA256"
        elif historical_0707_exact:
            path_attr = "student_recovery_historical_0707_exact_preregistration_path"
            sha_attr = "student_recovery_historical_0707_exact_preregistration_sha256"
            authority_label = "historical 0707 exact preregistration SHA256"
        elif be300_0707:
            path_attr = "student_recovery_be300_0707_preregistration_path"
            sha_attr = "student_recovery_be300_0707_preregistration_sha256"
            authority_label = "B-E300 0707 distillation preregistration SHA256"
        elif zero_scale_ablation:
            path_attr = "student_recovery_0707_exact_preregistration_path"
            sha_attr = "student_recovery_0707_exact_preregistration_sha256"
            authority_label = "zero-scale ablation preregistration SHA256"
        else:
            path_attr = "student_recovery_v15_preregistration_path"
            sha_attr = "student_recovery_v15_preregistration_sha256"
            authority_label = "v1.5 preregistration SHA256"
        path = os.path.realpath(
            str(getattr(self.policy, path_attr, ""))
        )
        expected_sha = self._require_sha256(
            getattr(self.policy, sha_attr, ""),
            authority_label,
        )
        if not path or not os.path.isabs(path):
            raise RuntimeError(f"v1.5 preregistration path must be absolute, got {path!r}")
        actual_sha = self._sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"v1.5 preregistration SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        authority = payload.get("authority", {}) if isinstance(payload, dict) else {}
        accepted_authorities = {
            (
                "highstep_student_recovery_v15_preregistration",
                "v1.5",
                "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a",
            ),
            (
                "highstep_student_recovery_v152_preregistration",
                "v1.5.2",
                "c892c0d8bc811228b34bb3c439e2977d5f473ecf1374453f3032509e2776f63e",
            ),
        }
        if zero_scale_ablation:
            accepted_authorities = {
                (
                    "highstep_0707_exact_new_teacher_preregistration",
                    "v1.6.1",
                    "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425",
                )
            }
        elif historical_0707_exact:
            accepted_authorities = {
                (
                    "highstep_historical_0707_exact_preregistration",
                    "v1.7",
                    "eff246af70dbfca7b3a6661f0a29f71bb1aa3b769a944351431f8b589663437a",
                ),
                (
                    "highstep_historical_0707_exact_v171_preregistration",
                    "v1.7.1",
                    "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917",
                ),
            }
        elif b300_hybrid:
            accepted_authorities = {
                (
                    "highstep_b300_canonical_hybrid_prior_latent_preregistration",
                    "1.0",
                    "aff25865dadf6d191984c454d050517f1a8b793b381fe150d79abc2b54e78f8c",
                )
            }
        elif be300_0707:
            accepted_authorities = {
                (
                    "highstep_be300_0707_distillation_preregistration",
                    "v1.13.1",
                    "0307b9e5c5c9c81999499ce2c3f05c0beabaa98caedde24aac5ac7843d73d768",
                ),
                (
                    "highstep_be300_clamp_gradient_repair_preregistration",
                    "v1.14",
                    "d3e69743993008a27bf1d30e5f78f634b8710a9dde1239b102cad4370ff84108",
                ),
                (
                    "highstep_b300_0707_derived_single_run_7400_preregistration",
                    "v1.0",
                    "b73470a6b15a9fb5595b7153c7cb155878c2a2c4c63590b178b7b3f640af5fd3",
                ),
                (
                    "highstep_b300_diagonal_imitation_fresh_7400_preregistration",
                    "v1.0",
                    "7c8ddc84a5653274981da6d4fc118a8d3c02156e6ec5adcc4d43c996085811e7",
                ),
                (
                    "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_preregistration",
                    "v1.0",
                    "4d23ca975cfe5f91d6536b911af369d0807cee027a8e5f54bfacee83fda68575",
                ),
                (
                    "highstep_b300_rl_preedge_continuation_e7700_preregistration",
                    "v1.0",
                    "b1a854ebc0c8ccf674bc25d869b6e088532cf8911e64a1404f0b365fdda59f08",
                ),
            }
        elif environment_curriculum_v18:
            accepted_authorities = {
                (
                    "highstep_student_environment_curriculum_v18_preregistration",
                    "v1.8",
                    "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4",
                )
            }
        elif e1400_continuation:
            accepted_authorities = {
                (
                    "highstep_e1400_continuation_ab_v110_runtime_preregistration",
                    "v1.10",
                    "3b526ddb88b60193d5fee1bb30f6473b5322df96eca393106ad680c6b245ffcf",
                )
            }
        identity = (
            payload.get("kind"),
            authority.get("version"),
            authority.get("spec_sha256"),
        )
        if identity not in accepted_authorities:
            raise RuntimeError("v1.5/v1.5.2 preregistration authority mismatch")
        return payload, path, actual_sha

    def _validate_v152_resume_rebinding(
        self,
        source_preregistration_sha256: str,
        *,
        effective_updates: int,
        loaded_checkpoint: str | None = None,
    ) -> dict:
        """Validate the one-way, hash-bound v1.5 E300 -> v1.5.2 authority transition."""
        prereg = getattr(self, "_v15_preregistration", {})
        if prereg.get("kind") != "highstep_student_recovery_v152_preregistration":
            raise ValueError("v1.5 checkpoint preregistration SHA mismatch")
        rebinding = prereg.get("resume_rebinding")
        if not isinstance(rebinding, dict):
            raise ValueError("v1.5.2 resume rebinding contract is missing")
        required = {
            "source_preregistration_sha256": source_preregistration_sha256,
            "source_effective_updates": effective_updates,
            "preserve_optimizer": True,
        }
        if any(rebinding.get(key) != value for key, value in required.items()):
            raise ValueError("v1.5.2 resume rebinding contract mismatch")
        audit_path = os.path.realpath(str(rebinding.get("audit_path", "")))
        audit_sha = str(rebinding.get("audit_sha256", ""))
        if not audit_path or not os.path.isabs(audit_path) or self._sha256_file(audit_path) != audit_sha:
            raise ValueError("v1.5.2 resume rebinding audit mismatch")
        if loaded_checkpoint is not None:
            loaded_path = os.path.realpath(loaded_checkpoint)
            if loaded_path != os.path.realpath(str(rebinding.get("source_checkpoint", ""))):
                raise ValueError("v1.5.2 rebound checkpoint path mismatch")
            if self._sha256_file(loaded_path) != rebinding.get("source_checkpoint_sha256"):
                raise ValueError("v1.5.2 rebound checkpoint SHA mismatch")
        return copy.deepcopy(rebinding)

    def _validate_historical_v171_resume_rebinding(
        self,
        source_preregistration_sha256: str,
        *,
        effective_updates: int,
        loaded_checkpoint: str | None = None,
    ) -> dict:
        """Validate the infrastructure-only v1.7 -> v1.7.1 E100 authority migration."""
        prereg = getattr(self, "_v15_preregistration", {})
        if prereg.get("kind") != "highstep_historical_0707_exact_v171_preregistration":
            raise ValueError("historical v1.7.1 checkpoint preregistration SHA mismatch")
        rebinding = prereg.get("resume_rebinding")
        if not isinstance(rebinding, dict):
            raise ValueError("historical v1.7.1 resume rebinding contract is missing")
        required = {
            "source_preregistration_sha256": source_preregistration_sha256,
            "source_spec_sha256": (
                "eff246af70dbfca7b3a6661f0a29f71bb1aa3b769a944351431f8b589663437a"
            ),
            "source_effective_updates": effective_updates,
            "preserve_optimizer": True,
            "training_contract_changed": False,
        }
        if any(rebinding.get(key) != value for key, value in required.items()):
            raise ValueError("historical v1.7.1 resume rebinding contract mismatch")
        audit_path = os.path.realpath(str(rebinding.get("audit_path", "")))
        audit_sha = str(rebinding.get("audit_sha256", ""))
        if not audit_path or not os.path.isabs(audit_path) or self._sha256_file(audit_path) != audit_sha:
            raise ValueError("historical v1.7.1 resume rebinding audit mismatch")
        if loaded_checkpoint is not None:
            loaded_path = os.path.realpath(loaded_checkpoint)
            source_checkpoint = os.path.realpath(str(rebinding.get("source_checkpoint", "")))
            if loaded_path != source_checkpoint:
                raise ValueError("historical v1.7.1 rebound checkpoint path mismatch")
            if self._sha256_file(loaded_path) != rebinding.get("source_checkpoint_sha256"):
                raise ValueError("historical v1.7.1 rebound checkpoint SHA mismatch")
        return copy.deepcopy(rebinding)

    def _validate_v18_resume_rebinding(
        self,
        source_preregistration_sha256: str,
        *,
        effective_updates: int,
        loaded_checkpoint: str | None = None,
    ) -> dict:
        """Validate the v1.8-only E100 infrastructure authority migration.

        This does not widen the archived v1.5/v1.5.2 checkpoint authority.  It
        accepts exactly one SHA-bound v1.8 checkpoint and preserves its full
        optimizer/update state across infrastructure-only preregistration
        amendments.
        """
        prereg = getattr(self, "_v15_preregistration", {})
        if prereg.get("kind") != "highstep_student_environment_curriculum_v18_preregistration":
            raise ValueError("v1.8 checkpoint preregistration SHA mismatch")
        rebinding = prereg.get("resume_rebinding")
        if not isinstance(rebinding, dict):
            raise ValueError("v1.8 resume rebinding contract is missing")
        required = {
            "kind": "highstep_v18_full_checkpoint_resume_rebinding",
            "workflow_id": "highstep_student_env_curriculum_v18_20260714",
            "source_preregistration_sha256": source_preregistration_sha256,
            "source_spec_sha256": (
                "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"
            ),
            "source_effective_updates": effective_updates,
            "preserve_optimizer": True,
            "training_contract_changed": False,
        }
        if any(rebinding.get(key) != value for key, value in required.items()):
            raise ValueError("v1.8 resume rebinding contract mismatch")
        audit_path = os.path.realpath(str(rebinding.get("audit_path", "")))
        audit_sha = str(rebinding.get("audit_sha256", ""))
        if not audit_path or not os.path.isabs(audit_path) or self._sha256_file(audit_path) != audit_sha:
            raise ValueError("v1.8 resume rebinding audit mismatch")
        source_checkpoint = os.path.realpath(str(rebinding.get("source_checkpoint", "")))
        source_checkpoint_sha = str(rebinding.get("source_checkpoint_sha256", ""))
        if (
            not source_checkpoint
            or not os.path.isabs(source_checkpoint)
            or self._sha256_file(source_checkpoint) != source_checkpoint_sha
        ):
            raise ValueError("v1.8 rebound checkpoint SHA mismatch")
        if loaded_checkpoint is not None and os.path.realpath(loaded_checkpoint) != source_checkpoint:
            raise ValueError("v1.8 rebound checkpoint path mismatch")
        return copy.deepcopy(rebinding)

    def _validate_v15_resume_rebinding(
        self,
        source_preregistration_sha256: str,
        *,
        effective_updates: int,
        loaded_checkpoint: str | None = None,
    ) -> dict:
        if getattr(self, "_v15_preregistration", {}).get("kind") == (
            "highstep_b300_rl_preedge_continuation_e7700_preregistration"
        ):
            prereg = self._v15_preregistration
            rebinding = prereg.get("resume_rebinding", {})
            required = {
                "kind": "highstep_b300_e5700_to_e7700_full_resume_rebinding",
                "source_preregistration_sha256": source_preregistration_sha256,
                "source_effective_updates": 5700,
                "preserve_optimizer": True,
                "preserve_schedule_runtime": True,
            }
            if effective_updates != 5700 or any(
                rebinding.get(key) != value for key, value in required.items()
            ):
                raise ValueError("E5700 continuation resume rebinding contract mismatch")
            source_path = os.path.realpath(str(rebinding.get("source_checkpoint", "")))
            source_sha = str(rebinding.get("source_checkpoint_sha256", ""))
            if self._sha256_file(source_path) != source_sha:
                raise ValueError("E5700 continuation checkpoint SHA mismatch")
            if loaded_checkpoint is not None and os.path.realpath(loaded_checkpoint) != source_path:
                raise ValueError("E5700 continuation checkpoint path mismatch")
            audit_path = os.path.realpath(str(rebinding.get("audit_path", "")))
            if self._sha256_file(audit_path) != rebinding.get("audit_sha256"):
                raise ValueError("E5700 continuation resume audit mismatch")
            return copy.deepcopy(rebinding)
        if getattr(self, "_v15_preregistration", {}).get("kind") == (
            "highstep_e1400_continuation_ab_v110_runtime_preregistration"
        ):
            prereg = self._v15_preregistration
            source = prereg.get("source", {})
            if not (
                source_preregistration_sha256
                == "fd38dd7e3270fd257c211ca8c23d15d9719cbef3e817dd1281b73f122563fac1"
                and effective_updates == 1400
                and source.get("effective_updates") == 1400
                and source.get("checkpoint_sha256")
                == "92bf3d0612f0f9ea1f85b379af8754a0df2710febb9b0067fd86e17487c810f9"
                and source.get("resume_mode") == "full_checkpoint_with_original_optimizer"
                and source.get("schedule_resume_mode") == "preserve"
            ):
                raise ValueError("E1400 continuation resume rebinding contract mismatch")
            source_path = os.path.realpath(str(source.get("checkpoint", "")))
            if self._sha256_file(source_path) != source.get("checkpoint_sha256"):
                raise ValueError("E1400 continuation source checkpoint SHA mismatch")
            if loaded_checkpoint is not None and os.path.realpath(loaded_checkpoint) != source_path:
                raise ValueError("E1400 continuation rebound checkpoint path mismatch")
            audit = prereg.get("runtime_rebinding", {})
            audit_path = os.path.realpath(str(audit.get("audit_path", "")))
            audit_sha = str(audit.get("audit_sha256", ""))
            if not audit_path or self._sha256_file(audit_path) != audit_sha:
                raise ValueError("E1400 continuation runtime rebinding audit mismatch")
            return copy.deepcopy(audit)
        if getattr(self, "_v15_preregistration", {}).get("kind") == (
            "highstep_student_environment_curriculum_v18_preregistration"
        ):
            return self._validate_v18_resume_rebinding(
                source_preregistration_sha256,
                effective_updates=effective_updates,
                loaded_checkpoint=loaded_checkpoint,
            )
        if getattr(self, "_v15_preregistration", {}).get("kind") == (
            "highstep_historical_0707_exact_v171_preregistration"
        ):
            return self._validate_historical_v171_resume_rebinding(
                source_preregistration_sha256,
                effective_updates=effective_updates,
                loaded_checkpoint=loaded_checkpoint,
            )
        return self._validate_v152_resume_rebinding(
            source_preregistration_sha256,
            effective_updates=effective_updates,
            loaded_checkpoint=loaded_checkpoint,
        )

    def _initialize_student_recovery_v15(self) -> None:
        """Build the strict 0707-faithful estimator + four-box-row optimizer."""
        zero_scale_ablation = self.student_recovery_stage == "0707_EXACT"
        e1400_continuation = self.student_recovery_stage == "E1400_CONTINUATION"
        prereg, prereg_path, prereg_sha = self._load_v15_preregistration()
        self._v15_preregistration = prereg
        self._v15_preregistration_path = prereg_path
        self._v15_preregistration_sha256 = prereg_sha
        expected_clamp_backward = (
            "straight_through"
            if prereg.get("kind") in {
                "highstep_be300_clamp_gradient_repair_preregistration",
                "highstep_b300_canonical_hybrid_prior_latent_preregistration",
            }
            else "hard"
        )
        if self.policy.student_actor_latent_clamp_backward != expected_clamp_backward:
            raise RuntimeError(
                "Student actor-facing latent clamp backward contract changed: "
                f"expected {expected_clamp_backward!r}, got "
                f"{self.policy.student_actor_latent_clamp_backward!r}"
            )

        for parameter in self.policy.parameters():
            parameter.requires_grad = False
        for parameter in self.policy.estimator.parameters():
            parameter.requires_grad = True

        if not isinstance(self.policy.actor, nn.Sequential):
            raise RuntimeError("v1.5 requires the locked sequential actor")
        self._v15_actor_modules = list(self.policy.actor)
        self._v15_last_linear_name, self._v15_last_linear = self._find_actor_last_linear()
        if (
            not self._v15_actor_modules
            or self._v15_actor_modules[-1] is not self._v15_last_linear
            or self._v15_last_linear.out_features != 16
            or self._v15_last_linear.bias is None
        ):
            raise RuntimeError("v1.5 requires the locked 16-D final actor Linear with bias")
        self.v15_box_weight = nn.Parameter(
            self._v15_last_linear.weight.detach()[12:16].clone(), requires_grad=True
        )
        self.v15_box_bias = nn.Parameter(
            self._v15_last_linear.bias.detach()[12:16].clone(), requires_grad=True
        )
        estimator_params = list(self.policy.estimator.parameters())
        if not estimator_params:
            raise RuntimeError("v1.5 Student estimator has no parameters")
        # These are the exact effective 0707 Stage-2 rates.  The saved runner's
        # base algorithm LR remains 1e-4; VAEPPO historically used 1e-3 for the
        # estimator and 1e-5 for the bounded box-head adaptation.
        self.vae_optimizer = torch.optim.Adam(
            [
                {"params": estimator_params, "lr": 1.0e-3, "v15_role": "estimator"},
                {
                    "params": [self.v15_box_weight, self.v15_box_bias],
                    "lr": 1.0e-5,
                    "v15_role": "box_rows",
                },
            ]
        )
        self.optimizer = self.vae_optimizer
        self.student_distill_update_count = 0
        self.student_actor_warmup_updates = int(
            getattr(
                self.policy,
                "student_actor_warmup_updates",
                1200 if zero_scale_ablation else 1400,
            )
        )
        self.student_vae_epochs = int(getattr(self.policy, "student_vae_epochs", 4))
        self.student_low_speed_threshold = getattr(self.policy, "student_low_speed_threshold", 0.10)
        self.student_prior_fade_speed = getattr(self.policy, "student_prior_fade_speed", 0.45)
        self.student_vel_loss_coef = getattr(self.policy, "student_vel_loss_coef", 10.0)
        self.student_latent_loss_coef = getattr(self.policy, "student_latent_loss_coef", 50.0)
        self.student_teacher_action_loss_coef = getattr(
            self.policy, "student_teacher_action_loss_coef", 20.0
        )
        self.student_prior_box_loss_coef = getattr(self.policy, "student_prior_box_loss_coef", 5.0)
        self.student_recon_loss_coef = getattr(self.policy, "student_recon_loss_coef", 0.5)
        self.student_kl_loss_coef = getattr(self.policy, "student_kl_loss_coef", 0.1)
        self.student_post_prior_mode = getattr(self.policy, "student_post_prior_mode", "highstep")
        self.student_highstep_phase_loss_scale = getattr(
            self.policy, "student_highstep_phase_loss_scale", 2.0
        )
        self.student_highstep_rear_box_loss_scale = getattr(
            self.policy, "student_highstep_rear_box_loss_scale", 1.5
        )
        self.student_highstep_diagonal_action_loss_scale = getattr(
            self.policy, "student_highstep_diagonal_action_loss_scale", 0.0
        )
        self.student_critical_transition_balanced_sampling = bool(getattr(
            self.policy, "student_critical_transition_balanced_sampling", False
        ))
        self.student_critical_transition_window_radius = int(getattr(
            self.policy, "student_critical_transition_window_radius", 10
        ))
        self.student_rl_preedge_sampling = bool(getattr(
            self.policy, "student_rl_preedge_sampling", False
        ))
        self.student_rl_preedge_pre_steps = int(getattr(
            self.policy, "student_rl_preedge_pre_steps", 30
        ))
        self.student_rl_preedge_post_steps = int(getattr(
            self.policy, "student_rl_preedge_post_steps", 10
        ))
        self.student_front_diagonal_action_loss_scale = float(getattr(
            self.policy, "student_front_diagonal_action_loss_scale", 0.0
        ))
        self.student_rear_diagonal_action_loss_scale = float(getattr(
            self.policy, "student_rear_diagonal_action_loss_scale", 0.0
        ))
        self._critical_transition_fifo = None
        self.student_highstep_rear_hip_loss_scale = 0.0
        self.student_highstep_rear_hip_min_abs = 0.0
        critical_balanced_route = prereg.get("kind") in {
            "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_preregistration",
            "highstep_b300_rl_preedge_continuation_e7700_preregistration",
        }
        if critical_balanced_route:
            sampling = prereg.get("critical_transition_sampling", {})
            extra_group = sampling.get("extra_rollout_group", {})
            phase_loss = prereg.get("phase_diagonal_loss", {})
            continuation = prereg.get("kind") == (
                "highstep_b300_rl_preedge_continuation_e7700_preregistration"
            )
            expected_group = {
                "name": "critical_transition",
                "shape": [5] if continuation else [4],
                "order": (
                    ["front_lift", "front_support", "first_rear", "second_rear", "rl_preedge"]
                    if continuation else
                    ["front_lift", "front_support", "first_rear", "second_rear"]
                ),
                "policy_estimator_critic_torchscript_input": False,
            }
            expected_fractions = {
                "front_transition": 0.25,
                ("rl_preedge" if continuation else "rear_transition"): 0.25,
                "original_distribution": 0.5,
            }
            if not (
                extra_group == expected_group
                and (
                    sampling.get("front_window") == {"pre_steps": 10, "post_steps": 10}
                    if continuation else sampling.get("window_radius_policy_steps") == 10
                )
                and sampling.get("minibatch_source_fractions") == expected_fractions
                and sampling.get("episode_safe_windows") is True
                and phase_loss.get("front_indices") == [1, 5, 9]
                and phase_loss.get("rear_indices") == [2, 6, 10]
                and phase_loss.get("front_scale") == 2.0
                and phase_loss.get("rear_scale") == 2.0
                and phase_loss.get("rear_term_sign") == "positive_addition"
                and phase_loss.get("gated_total_per_element_weight_ratio") == 3.0
                and phase_loss.get("legacy_unified_diagonal_scale") == 0.0
                and phase_loss.get("box_target_or_weight_changed") is False
                and (
                    not continuation
                    or (
                        sampling.get("rl_preedge_window") == {"pre_steps": 30, "post_steps": 10}
                        and sampling.get("rl_preedge_predicate") == {
                            "vx_min": 0.65,
                            "both_front_support": True,
                            "rl_not_on_platform": True,
                            "rl_near_edge_semantics": "frozen_first_rear_preclearance_boundary",
                        }
                        and phase_loss.get("additional_rl_only_weight") is False
                    )
                )
            ):
                raise RuntimeError("critical-transition balanced A+B contract changed")
        expected_warmup = (
            0 if self.student_recovery_stage == "B300_CANONICAL_HYBRID"
            else 1200 if zero_scale_ablation else 1400
        )
        if e1400_continuation:
            branch = str(getattr(self.policy, "student_recovery_e1400_continuation_branch", ""))
            branch_contract = prereg.get("branches", {}).get(branch, {})
            expected_warmup = int(branch_contract.get("actor_warmup_updates", -1))
            if not (
                branch in {"A", "B"}
                and expected_warmup == (1700 if branch == "A" else 1400)
                and branch_contract.get("box_rows_adaptation") is (branch == "B")
            ):
                raise RuntimeError("E1400 continuation branch contract changed")
            self._e1400_continuation_branch = branch
        if (
            self.student_actor_warmup_updates != expected_warmup
            or self.student_vae_epochs != 4
            or self.student_vel_loss_coef != 10.0
            or self.student_latent_loss_coef != 50.0
            or self.student_teacher_action_loss_coef != 20.0
            or self.student_prior_box_loss_coef != 5.0
            or self.student_recon_loss_coef != 0.5
            or self.student_kl_loss_coef != 0.1
            or self.student_highstep_phase_loss_scale != (0.0 if zero_scale_ablation else 2.0)
            or self.student_highstep_rear_box_loss_scale != (0.0 if zero_scale_ablation else 1.5)
            or self.student_highstep_diagonal_action_loss_scale
            != (
                1.0
                if prereg.get("kind")
                == "highstep_b300_diagonal_imitation_fresh_7400_preregistration"
                else 0.0
            )
            or self.student_critical_transition_balanced_sampling
            != (
                prereg.get("kind") in {
                    "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_preregistration",
                    "highstep_b300_rl_preedge_continuation_e7700_preregistration",
                }
            )
            or self.student_critical_transition_window_radius != 10
            or self.student_rl_preedge_sampling
            != (
                prereg.get("kind")
                == "highstep_b300_rl_preedge_continuation_e7700_preregistration"
            )
            or self.student_rl_preedge_pre_steps != 30
            or self.student_rl_preedge_post_steps != 10
            or self.student_front_diagonal_action_loss_scale
            != (
                2.0
                if prereg.get("kind") in {
                    "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_preregistration",
                    "highstep_b300_rl_preedge_continuation_e7700_preregistration",
                }
                else 0.0
            )
            or self.student_rear_diagonal_action_loss_scale
            != (
                2.0
                if prereg.get("kind") in {
                    "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_preregistration",
                    "highstep_b300_rl_preedge_continuation_e7700_preregistration",
                }
                else 0.0
            )
            or self.student_post_prior_mode != "highstep"
        ):
            raise RuntimeError("fixed Stage-2 hyperparameters changed")
        self.teacher_actor = None
        self.teacher_priv_encoder = None
        self._v15_binding_manifest = None
        self._v15_binding_ready = False
        self._v15_extra_restored = False
        self._teacher_actor_synced = False
        self._validate_v15_optimizer_scope()
        if self.student_recovery_stage == "B300_CANONICAL_HYBRID":
            self._load_b300_canonical_hybrid_dataset()

    def _load_b300_canonical_hybrid_dataset(self) -> None:
        prereg = self._v15_preregistration
        override_manifest = os.environ.get("HIGHSTEP_B300_HYBRID_TRAINING_DATASET_MANIFEST", "")
        override_sha = os.environ.get("HIGHSTEP_B300_HYBRID_TRAINING_DATASET_MANIFEST_SHA256", "")
        if override_manifest:
            manifest_path = os.path.realpath(override_manifest)
            manifest_sha = override_sha
            if self._sha256_file(manifest_path) != manifest_sha:
                raise RuntimeError("B300 DAgger aggregate manifest SHA mismatch")
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not (
                manifest.get("kind") == "highstep_b300_hybrid_dagger_aggregate"
                and manifest.get("workflow_id") == "highstep_b300_canonical_hybrid_prior_latent_20260718"
                and manifest.get("preregistration_sha256") == self._v15_preregistration_sha256
                and manifest.get("canonical_tensor_dataset_sha256")
                == prereg.get("canonical_tensor_dataset_sha256")
                and 1 <= int(manifest.get("round", 0)) <= 3
            ):
                raise RuntimeError("B300 DAgger aggregate authority mismatch")
            dataset_path = os.path.realpath(str(manifest.get("dataset_path", "")))
            dataset_sha = str(manifest.get("dataset_sha256", ""))
        else:
            manifest_path = os.path.realpath(str(prereg.get("canonical_tensor_manifest", "")))
            manifest_sha = str(prereg.get("canonical_tensor_manifest_sha256", ""))
            dataset_path = os.path.realpath(str(prereg.get("canonical_tensor_dataset", "")))
            dataset_sha = str(prereg.get("canonical_tensor_dataset_sha256", ""))
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        if (
            not manifest_path
            or not dataset_path
            or self._sha256_file(manifest_path) != manifest_sha
            or self._sha256_file(dataset_path) != dataset_sha
        ):
            raise RuntimeError("B300 canonical tensor dataset binding changed")
        if not override_manifest and not (
            manifest.get("kind") == "highstep_b300_canonical_hybrid_tensor_trajectory"
            and manifest.get("status") == "completed_and_centerline_verified"
            and manifest.get("sample_count") == 138 and manifest.get("unique_episode_count") == 1
            and manifest.get("dataset_sha256") == dataset_sha
        ):
            raise RuntimeError("B300 canonical tensor manifest contract changed")
        payload = torch.load(dataset_path, map_location="cpu", weights_only=True)
        sample_count = int(manifest.get("sample_count", 0))
        expected = {
            "student_obs_570": (sample_count, 570), "critic_obs": (sample_count, 162),
            "teacher_latent_raw_64": (sample_count, 64),
            "teacher_latent_clamped_64": (sample_count, 64),
            "teacher_pre_prior_action_16": (sample_count, 16),
            "teacher_post_prior_policy_action_16": (sample_count, 16),
        }
        dataset = {}
        for name, shape in expected.items():
            value = payload.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or not bool(torch.all(torch.isfinite(value)).item())
            ):
                raise RuntimeError(f"B300 canonical tensor invalid: {name}")
            dataset[name] = value.to(self.device).detach().contiguous()
        mixed = torch.cat(
            (
                dataset["teacher_pre_prior_action_16"][:, :12],
                dataset["teacher_post_prior_policy_action_16"][:, 12:16],
            ),
            dim=-1,
        )
        dataset["mixed_action_target_16"] = mixed
        self._b300_canonical_dataset = dataset
        self._b300_canonical_dataset_path = dataset_path
        self._b300_canonical_dataset_sha256 = dataset_sha
        self._b300_canonical_manifest_path = manifest_path
        self._b300_canonical_manifest_sha256 = manifest_sha
        self._b300_training_unique_episode_count = int(manifest.get("unique_episode_count", 1))
        self._b300_training_round = int(manifest.get("round", 0))

    def _update_b300_canonical_hybrid(self) -> dict[str, float]:
        """Offline canonical update; live rollout storage is never used as supervision."""
        if not self._v15_binding_ready or not hasattr(self, "_v15_source_snapshot"):
            raise RuntimeError("B300 canonical update requires completed dual-checkpoint binding")
        self._validate_v15_optimizer_scope()
        model = self.policy
        data = self._b300_canonical_dataset
        num_samples = int(data["student_obs_570"].shape[0])
        batch_size = (num_samples + int(self.num_mini_batches) - 1) // int(self.num_mini_batches)
        totals = {
            "velocity": 0.0,
            "latent": 0.0,
            "reconstruction": 0.0,
            "kl": 0.0,
            "action": 0.0,
            "prior_box": 0.0,
            "nonbox": 0.0,
            "box": 0.0,
            "estimator_grad": 0.0,
            "box_grad": 0.0,
            "mu_oob": 0.0,
        }
        optimizer_steps = 0
        for _ in range(self.student_vae_epochs):
            order = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                index = order[start : start + batch_size]
                obs = data["student_obs_570"][index]
                critic = data["critic_obs"][index]
                target_latent = data["teacher_latent_raw_64"][index]
                target_action = data["mixed_action_target_16"][index]
                target_pre = data["teacher_pre_prior_action_16"][index]
                target_post = data["teacher_post_prior_policy_action_16"][index]
                target_velocity = critic[:, :3]

                self.vae_optimizer.zero_grad(set_to_none=True)
                velocity, reconstruction, mu, logvar, _ = model.estimator(obs)
                velocity_loss = torch.mean(torch.square(velocity - target_velocity))
                latent_loss = torch.mean(torch.square(mu - target_latent))
                reconstruction_loss = torch.mean(torch.square(reconstruction - obs))
                kl_loss = -0.5 * torch.sum(
                    1 + logvar - torch.square(mu) - torch.exp(logvar), dim=-1
                ).mean()
                clamped = torch.clamp(mu, -1.0, 1.0)
                safe_mu = mu + (clamped - mu).detach()
                student_action = self._v15_actor_forward(torch.cat((obs, safe_mu), dim=-1))
                squared = torch.square(student_action - target_action)
                nonbox_per_sample = torch.mean(squared[:, :12], dim=-1)
                box_per_sample = torch.mean(squared[:, 12:16], dim=-1)
                rear_box_per_sample = torch.mean(squared[:, 14:16], dim=-1)
                post_prior_gate = (
                    torch.max(torch.abs(target_post[:, 12:16] - target_pre[:, 12:16]), dim=-1).values
                    > 1.0e-6
                ).to(squared.dtype)
                command_speed = torch.linalg.norm(critic[:, 9:12], dim=-1)
                moving_weight = 1.0 + torch.clamp(
                    command_speed / self.student_prior_fade_speed, 0.0, 1.0
                )
                phase_weight = 1.0 + self.student_highstep_phase_loss_scale * post_prior_gate
                per_sample_action = (
                    2.0 * nonbox_per_sample
                    + 0.5 * box_per_sample
                    + self.student_highstep_rear_box_loss_scale
                    * post_prior_gate
                    * rear_box_per_sample
                )
                action_loss = torch.mean(per_sample_action * moving_weight * phase_weight)
                low_speed_gate = torch.clamp(
                    (self.student_prior_fade_speed - command_speed)
                    / max(self.student_prior_fade_speed - self.student_low_speed_threshold, 1.0e-6),
                    0.0,
                    1.0,
                )
                is_static = (command_speed < self.student_low_speed_threshold).to(squared.dtype)
                gravity = critic[:, 6:9]
                tilt = torch.clamp(torch.linalg.norm(gravity[:, :2], dim=-1) / 0.25, 0.0, 1.0)
                prior_weight = post_prior_gate * (
                    0.15 + 1.25 * low_speed_gate + 1.0 * is_static + 0.75 * tilt
                )
                prior_box_loss = self._weighted_mean(box_per_sample, prior_weight)
                total_loss = (
                    self.student_vel_loss_coef * velocity_loss
                    + self.student_latent_loss_coef * latent_loss
                    + self.student_teacher_action_loss_coef * action_loss
                    + self.student_prior_box_loss_coef * prior_box_loss
                    + self.student_recon_loss_coef * reconstruction_loss
                    + self.student_kl_loss_coef * kl_loss
                )
                if not torch.isfinite(total_loss):
                    raise RuntimeError("B300 canonical hybrid loss is non-finite")
                total_loss.backward()
                estimator_grad = torch.sqrt(
                    sum(
                        torch.sum(torch.square(parameter.grad))
                        for parameter in model.estimator.parameters()
                        if parameter.grad is not None
                    )
                )
                box_grad = torch.sqrt(
                    sum(
                        torch.sum(torch.square(parameter.grad))
                        for parameter in (self.v15_box_weight, self.v15_box_bias)
                        if parameter.grad is not None
                    )
                )
                if not (float(estimator_grad) > 0.0 and float(box_grad) > 0.0):
                    raise RuntimeError("B300 estimator/box rows did not both receive gradients")
                for parameter in model.actor.parameters():
                    if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() != 0:
                        raise RuntimeError("B300 frozen actor tensor accumulated a gradient")
                nn.utils.clip_grad_norm_(
                    list(model.estimator.parameters()) + [self.v15_box_weight, self.v15_box_bias],
                    self.max_grad_norm,
                )
                self.vae_optimizer.step()
                self._sync_v15_box_rows_to_actor()

                totals["velocity"] += float(velocity_loss.detach())
                totals["latent"] += float(latent_loss.detach())
                totals["reconstruction"] += float(reconstruction_loss.detach())
                totals["kl"] += float(kl_loss.detach())
                totals["action"] += float(action_loss.detach())
                totals["prior_box"] += float(prior_box_loss.detach())
                totals["nonbox"] += float(torch.mean(nonbox_per_sample).detach())
                totals["box"] += float(torch.mean(box_per_sample).detach())
                totals["estimator_grad"] += float(estimator_grad.detach())
                totals["box_grad"] += float(box_grad.detach())
                totals["mu_oob"] += float(torch.mean((torch.abs(mu) > 1.0).to(mu.dtype)).detach())
                optimizer_steps += 1

        if optimizer_steps == 0:
            raise RuntimeError("B300 canonical hybrid performed no optimizer step")
        storage_before = int(self.storage.step)
        self.storage.clear()
        self.student_distill_update_count += 1
        self._sync_v15_box_rows_to_actor()
        self._assert_v15_frozen_unchanged()
        self._validate_v15_optimizer_scope()
        average = {name: value / optimizer_steps for name, value in totals.items()}
        return {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "Loss/VAE_Vel_MSE": average["velocity"],
            "Loss/Distill_Latent_MSE": average["latent"],
            "Loss/VAE_Recon_MSE": average["reconstruction"],
            "Loss/VAE_KL": average["kl"],
            "Loss/Teacher_Action_MSE": average["action"],
            "Loss/Prior_Box_Loss": average["prior_box"],
            "Loss/B300_Hybrid_Nonbox_PrePrior_MSE": average["nonbox"],
            "Loss/B300_Hybrid_Box_PostPrior_Policy_MSE": average["box"],
            "Debug/B300_Canonical_Unique_Episodes": float(self._b300_training_unique_episode_count),
            "Debug/B300_Canonical_Samples": float(num_samples),
            "Debug/B300_Estimator_Grad_Norm": average["estimator_grad"],
            "Debug/B300_Box_Rows_Grad_Norm": average["box_grad"],
            "Debug/B300_Frozen_Nonbox_Grad_Norm": 0.0,
            "Debug/B300_Box_Adapt_From_Update_Zero": 1.0,
            "Debug/B300_Mixed_Target_Policy_Units": 1.0,
            "Debug/Mu_Out_Of_Bounds_Ratio": average["mu_oob"],
            "Debug/Student_Distill_Update_Count": float(self.student_distill_update_count),
            "Debug/Buffer_Step_Before_Clear": float(storage_before),
            "Debug/Buffer_Step_After_Clear": float(self.storage.step),
        }

    def _v15_optimizer_parameter_names(self) -> tuple[str, ...]:
        names = tuple(f"estimator.{name}" for name, _ in self.policy.estimator.named_parameters())
        return names + ("algorithm.v15_box_weight", "algorithm.v15_box_bias")

    def _validate_v15_optimizer_scope(self) -> None:
        groups = self.vae_optimizer.param_groups
        if len(groups) != 2 or groups[0].get("v15_role") != "estimator" or groups[1].get("v15_role") != "box_rows":
            raise RuntimeError("v1.5 optimizer group roles changed")
        estimator_params = list(self.policy.estimator.parameters())
        if len(groups[0]["params"]) != len(estimator_params) or any(
            owned is not expected
            for owned, expected in zip(groups[0]["params"], estimator_params, strict=True)
        ):
            raise RuntimeError("v1.5 optimizer does not own the complete estimator in order")
        expected_box_params = (self.v15_box_weight, self.v15_box_bias)
        if len(groups[1]["params"]) != len(expected_box_params) or any(
            owned is not expected
            for owned, expected in zip(groups[1]["params"], expected_box_params, strict=True)
        ):
            raise RuntimeError("v1.5 optimizer must own exactly the four box rows and bias leaves")
        if float(groups[0]["lr"]) != 1.0e-3 or float(groups[1]["lr"]) != 1.0e-5:
            raise RuntimeError("v1.5 effective 0707 optimizer learning rates changed")
        trainable_policy = [name for name, value in self.policy.named_parameters() if value.requires_grad]
        expected_policy = [f"estimator.{name}" for name, _ in self.policy.estimator.named_parameters()]
        if trainable_policy != expected_policy:
            raise RuntimeError(f"v1.5 live policy trainable scope changed: {trainable_policy}")

    def _sync_v15_box_rows_to_actor(self) -> None:
        with torch.no_grad():
            self._v15_last_linear.weight[12:16].copy_(self.v15_box_weight)
            self._v15_last_linear.bias[12:16].copy_(self.v15_box_bias)

    def _reset_v15_box_rows_from_actor(self) -> None:
        with torch.no_grad():
            self.v15_box_weight.copy_(self._v15_last_linear.weight[12:16])
            self.v15_box_bias.copy_(self._v15_last_linear.bias[12:16])

    def _v15_actor_forward(self, actor_input: torch.Tensor) -> torch.Tensor:
        modules = list(self.policy.actor)
        if len(modules) != len(self._v15_actor_modules) or any(
            current is not expected
            for current, expected in zip(modules, self._v15_actor_modules, strict=True)
        ):
            raise RuntimeError("v1.5 actor module sequence changed")
        hidden = actor_input
        for module in modules[:-1]:
            hidden = module(hidden)
        revolute = F.linear(
            hidden,
            self._v15_last_linear.weight[:12].detach(),
            self._v15_last_linear.bias[:12].detach(),
        )
        box = F.linear(hidden, self.v15_box_weight, self.v15_box_bias)
        return torch.cat((revolute, box), dim=-1)

    def _assert_v15_lineage_scope(self, source_model_state: dict) -> None:
        current = self.policy.state_dict()
        if set(current) != set(source_model_state):
            raise RuntimeError("v1.5 Student/Teacher-root state keys differ")
        for key, value in current.items():
            actual = value.detach().cpu()
            source = source_model_state[key].detach().cpu()
            if key.startswith("estimator."):
                if not bool(torch.isfinite(actual).all().item()):
                    raise RuntimeError(f"v1.5 estimator tensor is non-finite: {key}")
                continue
            if key == f"actor.{self._v15_last_linear_name}.weight":
                equal = torch.equal(actual[:12], source[:12])
            elif key == f"actor.{self._v15_last_linear_name}.bias":
                equal = torch.equal(actual[:12], source[:12])
            else:
                equal = torch.equal(actual, source)
            if not equal:
                raise RuntimeError(f"v1.5 frozen lineage escaped at {key}")

    def _capture_v15_frozen_snapshot(self) -> None:
        self._v15_source_snapshot = {
            key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()
        }
        self._v15_teacher_snapshot = {
            "actor": {
                key: value.detach().cpu().clone() for key, value in self.teacher_actor.state_dict().items()
            },
            "priv_encoder": {
                key: value.detach().cpu().clone()
                for key, value in self.teacher_priv_encoder.state_dict().items()
            },
        }

    def _assert_v15_frozen_unchanged(self) -> None:
        if not hasattr(self, "_v15_source_snapshot"):
            raise RuntimeError("v1.5 frozen snapshot is missing")
        current = self.policy.state_dict()
        allow_box_delta = (
            self.student_recovery_stage == "B300_CANONICAL_HYBRID"
            or self.student_distill_update_count > self.student_actor_warmup_updates
        )
        for key, before in self._v15_source_snapshot.items():
            now = current[key].detach().cpu()
            if key.startswith("estimator."):
                if not bool(torch.isfinite(now).all().item()):
                    raise RuntimeError(f"v1.5 estimator became non-finite: {key}")
                continue
            if allow_box_delta and key == f"actor.{self._v15_last_linear_name}.weight":
                unchanged = torch.equal(now[:12], before[:12])
            elif allow_box_delta and key == f"actor.{self._v15_last_linear_name}.bias":
                unchanged = torch.equal(now[:12], before[:12])
            else:
                unchanged = torch.equal(now, before)
            if not unchanged:
                raise RuntimeError(f"v1.5 frozen Student tensor changed: {key}")
        for name, module in (
            ("actor", self.teacher_actor),
            ("priv_encoder", self.teacher_priv_encoder),
        ):
            reference = self._v15_teacher_snapshot[name]
            if any(
                not torch.equal(value.detach().cpu(), reference[key])
                for key, value in module.state_dict().items()
            ):
                raise RuntimeError(f"v1.5 frozen Teacher {name} changed")
        if not torch.equal(
            self._v15_last_linear.weight.detach()[12:16], self.v15_box_weight.detach()
        ) or not torch.equal(
            self._v15_last_linear.bias.detach()[12:16], self.v15_box_bias.detach()
        ):
            raise RuntimeError("v1.5 materialized actor box rows differ from optimizer leaves")

    def bind_student_recovery_v15_checkpoints(
        self, *, loaded_student_checkpoint: str, checkpoint_load_mode: str
    ) -> dict:
        if self.student_recovery_stage not in {
            "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
            "E1400_CONTINUATION", "BE300_0707", "B300_CANONICAL_HYBRID",
        }:
            raise RuntimeError(
                "isolated checkpoint binding requires V15, zero-scale ablation, "
                "or HISTORICAL_0707_EXACT"
            )
        zero_scale_ablation = self.student_recovery_stage == "0707_EXACT"
        historical_0707_exact = self.student_recovery_stage == "HISTORICAL_0707_EXACT"
        be300_0707 = self.student_recovery_stage == "BE300_0707"
        b300_hybrid = self.student_recovery_stage == "B300_CANONICAL_HYBRID"
        environment_curriculum_v18 = self.student_recovery_stage == "ENV_CURRICULUM_V18"
        e1400_continuation = self.student_recovery_stage == "E1400_CONTINUATION"
        pre_prior_route = (
            zero_scale_ablation or historical_0707_exact
            or environment_curriculum_v18 or e1400_continuation or be300_0707
            or b300_hybrid
        )
        source_path = os.path.realpath(str(self.policy.student_recovery_source_checkpoint))
        source_expected = str(self.policy.student_recovery_source_sha256)
        teacher_path = os.path.realpath(str(self.policy.student_recovery_teacher_checkpoint))
        teacher_expected = str(self.policy.student_recovery_teacher_sha256)
        loaded_path = os.path.realpath(loaded_student_checkpoint)
        source_checkpoint, source_sha = self._load_hashed_checkpoint(source_path, source_expected)
        teacher_checkpoint, teacher_sha = self._load_hashed_checkpoint(teacher_path, teacher_expected)
        source_state = source_checkpoint["model_state_dict"]

        if source_sha != teacher_sha or (not pre_prior_route and source_path != teacher_path):
            raise RuntimeError("Student root and independent Teacher must bind the same checkpoint bytes")
        if checkpoint_load_mode == "weights_only":
            if loaded_path != source_path:
                raise RuntimeError("fresh Stage-2 must weights-only load the canonical Teacher checkpoint")
            for component_name, module in (
                ("actor", self.policy.actor),
                ("estimator", self.policy.estimator),
                ("priv_encoder", self.policy.priv_encoder),
            ):
                self._assert_module_matches_checkpoint(
                    module,
                    self._component_state(source_state, f"{component_name}."),
                    f"fresh v1.5 Student {component_name}",
                )
            self.student_distill_update_count = 0
            self._reset_v15_box_rows_from_actor()
        elif checkpoint_load_mode == "full":
            if not self._v15_extra_restored:
                raise RuntimeError("full v1.5 resume did not restore optimizer/count state")
            binding = self._v15_binding_manifest or {}
            authority_unchanged = (
                binding.get("initial_student_checkpoint") != source_path
                or binding.get("initial_student_sha256") != source_sha
                or binding.get("teacher_checkpoint") != teacher_path
                or binding.get("teacher_sha256") != teacher_sha
            )
            if authority_unchanged:
                raise RuntimeError("full v1.5 resume authority binding changed")
            if e1400_continuation:
                prior_stage = binding.get("stage")
                prior_branch = binding.get("continuation_branch")
                current_branch = self._e1400_continuation_branch
                if prior_stage == "E1400_CONTINUATION" and prior_branch != current_branch:
                    raise RuntimeError("E1400 continuation branch checkpoint crossover forbidden")
                if prior_stage not in {"ENV_CURRICULUM_V18", "E1400_CONTINUATION"}:
                    raise RuntimeError("E1400 continuation source stage mismatch")
            prior_preregistration_sha = binding.get("preregistration_sha256")
            rebinding = None
            if prior_preregistration_sha != self._v15_preregistration_sha256:
                rebinding = self._validate_v15_resume_rebinding(
                    prior_preregistration_sha,
                    effective_updates=int(self.student_distill_update_count),
                    loaded_checkpoint=loaded_path,
                )
            self._sync_v15_box_rows_to_actor()
            self._assert_v15_lineage_scope(source_state)
        else:
            raise RuntimeError(f"unsupported v1.5 checkpoint load mode: {checkpoint_load_mode!r}")

        teacher_actor_state = self._component_state(teacher_checkpoint["model_state_dict"], "actor.")
        teacher_priv_state = self._component_state(
            teacher_checkpoint["model_state_dict"], "priv_encoder."
        )
        self.teacher_actor = copy.deepcopy(self.policy.actor).to(self.device)
        self.teacher_actor.load_state_dict(teacher_actor_state, strict=True)
        self.teacher_priv_encoder = copy.deepcopy(self.policy.priv_encoder).to(self.device)
        self.teacher_priv_encoder.load_state_dict(teacher_priv_state, strict=True)
        for module in (self.teacher_actor, self.teacher_priv_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        student_ptrs = {
            value.untyped_storage().data_ptr()
            for module in (self.policy.actor, self.policy.priv_encoder)
            for value in list(module.parameters()) + list(module.buffers())
            if value.numel()
        }
        teacher_ptrs = {
            value.untyped_storage().data_ptr()
            for module in (self.teacher_actor, self.teacher_priv_encoder)
            for value in list(module.parameters()) + list(module.buffers())
            if value.numel()
        }
        if student_ptrs & teacher_ptrs:
            raise RuntimeError("v1.5 Teacher and Student share tensor storage")
        self._assert_v15_lineage_scope(source_state)
        self._sync_v15_box_rows_to_actor()
        self._validate_v15_optimizer_scope()
        self._v15_binding_manifest = {
            "schema_version": 1,
            "stage": self.student_recovery_stage,
            "initial_student_checkpoint": source_path,
            "initial_student_sha256": source_sha,
            "loaded_student_checkpoint": loaded_path,
            "teacher_checkpoint": teacher_path,
            "teacher_sha256": teacher_sha,
            "student_actor_component_sha256_at_root": self._component_sha256(
                self._component_state(source_state, "actor.")
            ),
            "teacher_actor_component_sha256": self._component_sha256(teacher_actor_state),
            "teacher_privileged_encoder_component_sha256": self._component_sha256(
                teacher_priv_state
            ),
            "preregistration_path": self._v15_preregistration_path,
            "preregistration_sha256": self._v15_preregistration_sha256,
            "recovery_spec_sha256": (
                "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"
                if zero_scale_ablation else
                self._v15_preregistration.get("authority", {}).get("spec_sha256")
                if historical_0707_exact or environment_curriculum_v18 or e1400_continuation or be300_0707 or b300_hybrid
                else None
            ),
            "checkpoint_load_mode": checkpoint_load_mode,
            "effective_update_count": int(self.student_distill_update_count),
            "warmup_updates": self.student_actor_warmup_updates,
            "optimizer_parameter_names": list(self._v15_optimizer_parameter_names()),
            "optimizer_group_learning_rates": [1.0e-3, 1.0e-5],
            "student_teacher_storage_independent": True,
            "student_actor_and_teacher_actor_same_root_weights": True,
            "student_ppo_permanently_disabled": True,
            "critical_transition_balanced_sampling": bool(
                self.student_critical_transition_balanced_sampling
            ),
            "critical_transition_window_radius": int(
                self.student_critical_transition_window_radius
            ),
            "critical_transition_minibatch_fractions": (
                [0.25, 0.25, 0.50]
                if self.student_critical_transition_balanced_sampling else None
            ),
            "front_diagonal_indices": (
                [1, 5, 9] if self.student_critical_transition_balanced_sampling else None
            ),
            "rear_diagonal_indices": (
                [2, 6, 10] if self.student_critical_transition_balanced_sampling else None
            ),
            "phase_diagonal_weight_ratio": (
                3.0 if self.student_critical_transition_balanced_sampling else None
            ),
            "actor_body_frozen": True,
            "critic_frozen": True,
            "student_privileged_encoder_frozen": True,
            "post_warmup_trainable_action_rows": [12, 13, 14, 15],
            "trainable_action_rows_from_update_zero": (
                [12, 13, 14, 15] if b300_hybrid else []
            ),
            "student_actor_latent_clamp_backward": str(
                self.policy.student_actor_latent_clamp_backward
            ),
            "post_warmup_trainable_joint_names": [
                "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint"
            ],
            "warmup_main_action_target": (
                "teacher_pre_prior_nonbox_plus_post_prior_policy_unit_box"
                if b300_hybrid else
                "teacher_pre_prior" if pre_prior_route else "teacher_post_prior"
            ),
            "warmup_prior_box_loss_coefficient": (
                self.student_prior_box_loss_coef if b300_hybrid else 0.0
            ),
            "phase_scale": 0.0 if zero_scale_ablation else 2.0,
            "rear_box_scale": 0.0 if zero_scale_ablation else 1.5,
        }
        if b300_hybrid:
            self._v15_binding_manifest.update(
                {
                    "canonical_tensor_dataset": self._b300_canonical_dataset_path,
                    "canonical_tensor_dataset_sha256": self._b300_canonical_dataset_sha256,
                    "canonical_tensor_manifest": self._b300_canonical_manifest_path,
                    "canonical_tensor_manifest_sha256": self._b300_canonical_manifest_sha256,
                    "canonical_unique_episode_count": self._b300_training_unique_episode_count,
                    "canonical_sample_count": int(self._b300_canonical_dataset["student_obs_570"].shape[0]),
                    "dagger_round": self._b300_training_round,
                    "frozen_nonbox_action_rows": list(range(12)),
                }
            )
        if e1400_continuation:
            self._v15_binding_manifest["continuation_branch"] = self._e1400_continuation_branch
        if checkpoint_load_mode == "full" and rebinding is not None:
            self._v15_binding_manifest["authority_rebound_from_preregistration_sha256"] = (
                prior_preregistration_sha
            )
            self._v15_binding_manifest["authority_rebinding_audit"] = {
                "path": rebinding["audit_path"],
                "sha256": rebinding["audit_sha256"],
            }
        self._v15_binding_ready = True
        self._teacher_actor_synced = True
        self._capture_v15_frozen_snapshot()
        self._assert_v15_frozen_unchanged()
        return copy.deepcopy(self._v15_binding_manifest)

    @staticmethod
    def _validate_student_distill_update_count(value) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "student_distill_update_count must be a non-negative integer, "
                f"got {value!r}"
            )
        return value

    def extra_checkpoint_state_dict(self) -> dict:
        """Return VAEPPO state that is not owned by the stock RSL-RL runner."""
        if not hasattr(self, "vae_optimizer"):
            raise RuntimeError("VAEPPO has no vae_optimizer to checkpoint")

        state = {
            "schema_version": self.EXTRA_CHECKPOINT_STATE_SCHEMA_VERSION,
            "algorithm_class": type(self).__name__,
            "distill_stage": int(self.distill_stage),
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
        }
        if self.distill_stage == 2:
            state["student_distill_update_count"] = self._validate_student_distill_update_count(
                self.student_distill_update_count
            )
        if getattr(self, "student_recovery_stage", "NONE") == "B":
            if not self._recovery_binding_ready or not isinstance(self._recovery_binding_manifest, dict):
                raise RuntimeError("Cannot checkpoint stage B before dual-checkpoint binding succeeds")
            self._assert_recovery_frozen_unchanged()
            binding = dict(self._recovery_binding_manifest)
            binding["effective_update_count"] = int(self.student_distill_update_count)
            state["student_recovery"] = {
                "schema_version": 1,
                "stage": "B",
                "effective_update_count": int(self.student_distill_update_count),
                "rear_hip_weight": self.recovery_rear_hip_weight.detach().clone(),
                "rear_hip_bias": self.recovery_rear_hip_bias.detach().clone(),
                "binding_manifest": binding,
            }
        elif getattr(self, "student_recovery_stage", "NONE") == "R2":
            if not self._r2_binding_ready or not isinstance(self._r2_binding_manifest, dict):
                raise RuntimeError("Cannot checkpoint R2 before all bindings succeed")
            self._sync_r2_box_rows_to_actor()
            self._assert_r2_frozen_unchanged()
            self._validate_r2_optimizer_state(self.student_distill_update_count)
            binding = copy.deepcopy(self._r2_binding_manifest)
            binding["effective_update_count"] = int(self.student_distill_update_count)
            state["student_recovery"] = {
                "schema_version": 1,
                "stage": "R2",
                "effective_update_count": int(self.student_distill_update_count),
                "preregistration_sha256": self._r2_preregistration_sha256,
                "optimizer_parameter_names": list(self._r2_optimizer_names),
                "box_weight": self.r2_box_weight.detach().clone(),
                "box_bias": self.r2_box_bias.detach().clone(),
                "binding_manifest": binding,
            }
        elif getattr(self, "student_recovery_stage", "NONE") == "R3":
            if not self._r3_binding_ready or not isinstance(self._r3_binding_manifest, dict):
                raise RuntimeError("Cannot checkpoint R3 before all bindings succeed")
            self._assert_r3_frozen_unchanged()
            self._validate_r3_optimizer_state(self.student_distill_update_count)
            binding = copy.deepcopy(self._r3_binding_manifest)
            binding["effective_update_count"] = int(self.student_distill_update_count)
            state["student_recovery"] = {
                "schema_version": 1,
                "stage": "R3",
                "effective_update_count": int(self.student_distill_update_count),
                "preregistration_sha256": self._r3_preregistration_sha256,
                "optimizer_parameter_names": list(self._r3_optimizer_names),
                "binding_manifest": binding,
            }
        elif getattr(self, "student_recovery_stage", "NONE") in {
            "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
            "E1400_CONTINUATION", "BE300_0707", "B300_CANONICAL_HYBRID",
        }:
            if not self._v15_binding_ready or not isinstance(self._v15_binding_manifest, dict):
                raise RuntimeError("Cannot checkpoint v1.5 before all bindings succeed")
            self._sync_v15_box_rows_to_actor()
            self._assert_v15_frozen_unchanged()
            self._validate_v15_optimizer_scope()
            binding = copy.deepcopy(self._v15_binding_manifest)
            binding["effective_update_count"] = int(self.student_distill_update_count)
            state["student_recovery"] = {
                "schema_version": 1,
                "stage": self.student_recovery_stage,
                "effective_update_count": int(self.student_distill_update_count),
                "preregistration_sha256": self._v15_preregistration_sha256,
                "optimizer_parameter_names": list(self._v15_optimizer_parameter_names()),
                "box_weight": self.v15_box_weight.detach().clone(),
                "box_bias": self.v15_box_bias.detach().clone(),
                "binding_manifest": binding,
            }
        return state

    def load_extra_checkpoint_state_dict(self, state: dict) -> None:
        """Restore a versioned VAEPPO state saved by :meth:`extra_checkpoint_state_dict`."""
        if not isinstance(state, dict):
            raise TypeError(f"VAEPPO extra checkpoint state must be a dict, got {type(state).__name__}")
        schema_version = state.get("schema_version")
        if schema_version != self.EXTRA_CHECKPOINT_STATE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported VAEPPO extra checkpoint schema_version "
                f"{schema_version!r}; expected {self.EXTRA_CHECKPOINT_STATE_SCHEMA_VERSION}"
            )
        if state.get("algorithm_class") != type(self).__name__:
            raise ValueError(
                "VAEPPO extra checkpoint algorithm mismatch: "
                f"checkpoint={state.get('algorithm_class')!r}, runtime={type(self).__name__!r}"
            )
        checkpoint_stage = state.get("distill_stage")
        if checkpoint_stage != int(self.distill_stage):
            raise ValueError(
                "VAEPPO extra checkpoint distill_stage mismatch: "
                f"checkpoint={checkpoint_stage!r}, runtime={int(self.distill_stage)!r}"
            )
        optimizer_state = state.get("vae_optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("VAEPPO extra checkpoint is missing vae_optimizer_state_dict")
        if not hasattr(self, "vae_optimizer"):
            raise RuntimeError("VAEPPO has no vae_optimizer to restore")

        student_update_count = None
        if self.distill_stage == 2:
            student_update_count = self._validate_student_distill_update_count(
                state.get("student_distill_update_count")
            )

        if getattr(self, "student_recovery_stage", "NONE") == "B":
            recovery_state = state.get("student_recovery")
            if not isinstance(recovery_state, dict) or recovery_state.get("schema_version") != 1:
                raise ValueError("Stage B checkpoint is missing versioned student_recovery state")
            if recovery_state.get("stage") != "B":
                raise ValueError("Stage B checkpoint recovery phase mismatch")
            recovery_count = self._validate_student_distill_update_count(
                recovery_state.get("effective_update_count")
            )
            if recovery_count != student_update_count:
                raise ValueError("Stage B checkpoint update counts disagree")
            weight = recovery_state.get("rear_hip_weight")
            bias = recovery_state.get("rear_hip_bias")
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(self.recovery_rear_hip_weight.shape):
                raise ValueError("Stage B checkpoint rear_hip_weight is missing or has the wrong shape")
            if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != tuple(self.recovery_rear_hip_bias.shape):
                raise ValueError("Stage B checkpoint rear_hip_bias is missing or has the wrong shape")
            binding = recovery_state.get("binding_manifest")
            if not isinstance(binding, dict) or binding.get("stage") != "B":
                raise ValueError("Stage B checkpoint binding manifest is missing")
            if not torch.equal(
                self._recovery_last_linear.weight.detach()[2:4].cpu(), weight.detach().cpu()
            ) or not torch.equal(
                self._recovery_last_linear.bias.detach()[2:4].cpu(), bias.detach().cpu()
            ):
                raise ValueError(
                    "Stage B checkpoint model_state rear-hip rows disagree with algorithm recovery state"
                )
            self.vae_optimizer.load_state_dict(optimizer_state)
            with torch.no_grad():
                self.recovery_rear_hip_weight.copy_(weight.to(self.device))
                self.recovery_rear_hip_bias.copy_(bias.to(self.device))
            self._sync_recovery_rows_to_actor()
            self.student_distill_update_count = recovery_count
            self._recovery_binding_manifest = dict(binding)
            self._recovery_extra_restored = True
            self._recovery_binding_ready = False
            self._teacher_actor_synced = False
            self._validate_stage_b_optimizer_scope()
            return

        if getattr(self, "student_recovery_stage", "NONE") == "R2":
            recovery_state = state.get("student_recovery")
            if not isinstance(recovery_state, dict) or recovery_state.get("schema_version") != 1:
                raise ValueError("R2 checkpoint is missing versioned student_recovery state")
            if recovery_state.get("stage") != "R2":
                raise ValueError("R2 checkpoint recovery stage mismatch")
            if recovery_state.get("preregistration_sha256") != self._r2_preregistration_sha256:
                raise ValueError("R2 checkpoint preregistration SHA mismatch")
            if recovery_state.get("optimizer_parameter_names") != list(self._r2_optimizer_names):
                raise ValueError("R2 checkpoint optimizer parameter order mismatch")
            recovery_count = self._validate_student_distill_update_count(
                recovery_state.get("effective_update_count")
            )
            if recovery_count != student_update_count:
                raise ValueError("R2 checkpoint update counts disagree")
            weight = recovery_state.get("box_weight")
            bias = recovery_state.get("box_bias")
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(
                self.r2_box_weight.shape
            ):
                raise ValueError("R2 checkpoint box_weight is missing or has the wrong shape")
            if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != tuple(
                self.r2_box_bias.shape
            ):
                raise ValueError("R2 checkpoint box_bias is missing or has the wrong shape")
            binding = recovery_state.get("binding_manifest")
            if not isinstance(binding, dict) or binding.get("stage") != "R2":
                raise ValueError("R2 checkpoint binding manifest is missing")
            if not torch.equal(
                self._r2_last_linear.weight.detach()[12:16].cpu(), weight.detach().cpu()
            ) or not torch.equal(
                self._r2_last_linear.bias.detach()[12:16].cpu(), bias.detach().cpu()
            ):
                raise ValueError("R2 model_state box rows disagree with algorithm recovery state")
            self.optimizer.load_state_dict(optimizer_state)
            self._validate_r2_optimizer_state(recovery_count)
            with torch.no_grad():
                self.r2_box_weight.copy_(weight.to(self.device))
                self.r2_box_bias.copy_(bias.to(self.device))
            self._sync_r2_box_rows_to_actor()
            self.student_distill_update_count = recovery_count
            self._r2_binding_manifest = copy.deepcopy(binding)
            self._r2_extra_restored = True
            self._r2_binding_ready = False
            self._teacher_actor_synced = False
            self._validate_r2_optimizer_scope()
            return

        if getattr(self, "student_recovery_stage", "NONE") == "R3":
            recovery_state = state.get("student_recovery")
            if not isinstance(recovery_state, dict) or recovery_state.get("schema_version") != 1:
                raise ValueError("R3 checkpoint is missing versioned student_recovery state")
            if recovery_state.get("stage") != "R3":
                raise ValueError("R3 checkpoint recovery stage mismatch")
            if recovery_state.get("preregistration_sha256") != self._r3_preregistration_sha256:
                raise ValueError("R3 checkpoint preregistration SHA mismatch")
            if recovery_state.get("optimizer_parameter_names") != list(self._r3_optimizer_names):
                raise ValueError("R3 checkpoint optimizer parameter order mismatch")
            recovery_count = self._validate_student_distill_update_count(
                recovery_state.get("effective_update_count")
            )
            if recovery_count != student_update_count:
                raise ValueError("R3 checkpoint update counts disagree")
            binding = recovery_state.get("binding_manifest")
            if not isinstance(binding, dict) or binding.get("stage") != "R3":
                raise ValueError("R3 checkpoint binding manifest is missing")
            self.optimizer.load_state_dict(optimizer_state)
            self.student_distill_update_count = recovery_count
            self._r3_binding_manifest = copy.deepcopy(binding)
            self._r3_extra_restored = True
            self._r3_binding_ready = False
            self._teacher_actor_synced = False
            self._validate_r3_optimizer_scope()
            self._validate_r3_optimizer_state(recovery_count)
            return

        if getattr(self, "student_recovery_stage", "NONE") in {
            "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
            "E1400_CONTINUATION", "BE300_0707", "B300_CANONICAL_HYBRID",
        }:
            recovery_state = state.get("student_recovery")
            if not isinstance(recovery_state, dict) or recovery_state.get("schema_version") != 1:
                raise ValueError("v1.5 checkpoint is missing versioned student_recovery state")
            checkpoint_recovery_stage = recovery_state.get("stage")
            stage_migration = (
                self.student_recovery_stage == "E1400_CONTINUATION"
                and checkpoint_recovery_stage == "ENV_CURRICULUM_V18"
                and student_update_count == 1400
            )
            if checkpoint_recovery_stage != self.student_recovery_stage and not stage_migration:
                raise ValueError("v1.5 checkpoint recovery stage mismatch")
            checkpoint_preregistration_sha = recovery_state.get("preregistration_sha256")
            if recovery_state.get("optimizer_parameter_names") != list(
                self._v15_optimizer_parameter_names()
            ):
                raise ValueError("v1.5 checkpoint optimizer parameter order mismatch")
            recovery_count = self._validate_student_distill_update_count(
                recovery_state.get("effective_update_count")
            )
            if recovery_count != student_update_count:
                raise ValueError("v1.5 checkpoint update counts disagree")
            if checkpoint_preregistration_sha != self._v15_preregistration_sha256:
                self._validate_v15_resume_rebinding(
                    checkpoint_preregistration_sha,
                    effective_updates=recovery_count,
                )
            weight = recovery_state.get("box_weight")
            bias = recovery_state.get("box_bias")
            if not isinstance(weight, torch.Tensor) or tuple(weight.shape) != tuple(
                self.v15_box_weight.shape
            ):
                raise ValueError("v1.5 checkpoint box_weight is missing or invalid")
            if not isinstance(bias, torch.Tensor) or tuple(bias.shape) != tuple(
                self.v15_box_bias.shape
            ):
                raise ValueError("v1.5 checkpoint box_bias is missing or invalid")
            if not torch.equal(
                self._v15_last_linear.weight.detach()[12:16].cpu(), weight.detach().cpu()
            ) or not torch.equal(
                self._v15_last_linear.bias.detach()[12:16].cpu(), bias.detach().cpu()
            ):
                raise ValueError("v1.5 model_state box rows disagree with recovery leaves")
            binding = recovery_state.get("binding_manifest")
            if not isinstance(binding, dict) or (
                binding.get("stage") != self.student_recovery_stage and not stage_migration
            ):
                raise ValueError("v1.5 checkpoint binding manifest is missing")
            self.vae_optimizer.load_state_dict(optimizer_state)
            with torch.no_grad():
                self.v15_box_weight.copy_(weight.to(self.device))
                self.v15_box_bias.copy_(bias.to(self.device))
            self._sync_v15_box_rows_to_actor()
            self.student_distill_update_count = recovery_count
            self._v15_binding_manifest = copy.deepcopy(binding)
            self._v15_extra_restored = True
            self._v15_binding_ready = False
            self._teacher_actor_synced = False
            self._validate_v15_optimizer_scope()
            return

        self.vae_optimizer.load_state_dict(optimizer_state)
        if student_update_count is not None:
            self.student_distill_update_count = student_update_count
            # runner.load() has just replaced policy.actor.  Force the frozen
            # teacher copy to synchronize from those restored policy weights.
            self._teacher_actor_synced = False

    def migrate_legacy_extra_checkpoint_state(self, *, student_distill_update_count: int) -> None:
        """Explicitly migrate a legacy Student checkpoint with irrecoverable Adam state loss."""
        if getattr(self, "student_recovery_stage", "NONE") in {
            "B", "R2", "R3", "V15", "0707_EXACT", "HISTORICAL_0707_EXACT",
            "ENV_CURRICULUM_V18", "E1400_CONTINUATION", "BE300_0707",
            "B300_CANONICAL_HYBRID",
        }:
            raise RuntimeError(
                f"Stage {self.student_recovery_stage} forbids legacy full-resume migration"
            )
        if self.distill_stage != 2:
            raise RuntimeError("Legacy student checkpoint migration is only valid for distill_stage=2")
        if not hasattr(self, "vae_optimizer"):
            raise RuntimeError("VAEPPO has no vae_optimizer for legacy checkpoint migration")
        if self.vae_optimizer.state:
            raise RuntimeError("Legacy checkpoint migration requires a fresh vae_optimizer")
        self.student_distill_update_count = self._validate_student_distill_update_count(
            student_distill_update_count
        )
        self._teacher_actor_synced = False

    def _sync_teacher_actor_from_policy_once(self):
        """Sync frozen teacher actor after runner.load() has restored checkpoint weights."""
        if getattr(self, "student_recovery_stage", "NONE") in {
            "B", "R2", "R3", "V15", "0707_EXACT", "HISTORICAL_0707_EXACT",
            "ENV_CURRICULUM_V18", "E1400_CONTINUATION", "BE300_0707",
            "B300_CANONICAL_HYBRID",
        }:
            if (
                self.student_recovery_stage in {
                    "V15", "0707_EXACT", "HISTORICAL_0707_EXACT",
                    "ENV_CURRICULUM_V18", "E1400_CONTINUATION", "BE300_0707",
                    "B300_CANONICAL_HYBRID",
                }
                and self._teacher_actor_synced
                and self._v15_binding_ready
            ):
                return
            raise RuntimeError(
                f"Stage {self.student_recovery_stage} must never synchronize Teacher weights from the Student"
            )
        if self._teacher_actor_synced or self.distill_stage != 2 or not hasattr(self, "teacher_actor"):
            return

        self.teacher_actor.load_state_dict(self.policy.actor.state_dict())
        self.teacher_actor.to(self.device)
        self.teacher_actor.eval()
        for param in self.teacher_actor.parameters():
            param.requires_grad = False
        self._teacher_actor_synced = True

    @staticmethod
    def _estimate_lateral_height_from_scan(height_scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate left/right terrain clearance from the critic height scan."""
        num_rays = height_scan.shape[-1]
        if num_rays >= 6 and num_rays % 6 == 0:
            # Current bodyflat scanner: size=[1.6, 0.5], resolution=0.1 -> 6 y rows.
            scan = height_scan.reshape(height_scan.shape[0], 6, num_rays // 6)
            right_scan = scan[:, :3, :].mean(dim=(1, 2))
            left_scan = scan[:, 3:, :].mean(dim=(1, 2))
        else:
            half = max(num_rays // 2, 1)
            right_scan = height_scan[:, :half].mean(dim=1)
            left_scan = height_scan[:, half:].mean(dim=1)
        return left_scan, right_scan

    def _apply_lateral_step_box_prior_to_raw_action(
        self, raw_action: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert teacher raw actions into raw-equivalent actions after the box prior."""
        if raw_action.shape[-1] < 16 or critic_obs.shape[-1] <= 60:
            zeros = raw_action.new_zeros(raw_action.shape[0])
            return raw_action, zeros, zeros, raw_action[..., -4:], zeros

        height_scan = critic_obs[:, 60:]
        left_scan, right_scan = self._estimate_lateral_height_from_scan(height_scan)

        # height_scan = sensor_z - hit_z - offset. Larger scan values mean lower terrain.
        # The scan side order is mirrored against the action-vector FL/RL vs FR/RR side labels
        # in this task, so the raw scan delta is inverted before assigning lower/upper box bias.
        scan_delta = left_scan - right_scan
        gate = torch.clamp((torch.abs(scan_delta) - 0.02) / 0.06, min=0.0, max=1.0)
        left_is_lower = scan_delta < 0.0

        lower_bias = torch.full_like(gate, 0.018)
        upper_bias = torch.full_like(gate, -0.014)
        left_bias = torch.where(left_is_lower, lower_bias, upper_bias)
        right_bias = torch.where(left_is_lower, upper_bias, lower_bias)
        box_bias = torch.stack((left_bias, right_bias, left_bias, right_bias), dim=1) * gate.unsqueeze(1)

        target_action = raw_action.clone()
        box_default = 0.03
        box_scale = 0.02
        box_targets = raw_action[:, -4:] * box_scale + box_default + box_bias
        box_targets = torch.clamp(box_targets, min=0.0, max=0.06)
        target_action[:, -4:] = (box_targets - box_default) / box_scale
        left_box_mean = 0.5 * (box_targets[:, 0] + box_targets[:, 2])
        right_box_mean = 0.5 * (box_targets[:, 1] + box_targets[:, 3])
        lower_box_mean = torch.where(left_is_lower, left_box_mean, right_box_mean)
        upper_box_mean = torch.where(left_is_lower, right_box_mean, left_box_mean)
        lower_minus_upper_box = lower_box_mean - upper_box_mean
        return target_action, gate, scan_delta, box_targets, lower_minus_upper_box

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @staticmethod
    def _estimate_forward_highstep_delta_from_scan(height_scan: torch.Tensor) -> torch.Tensor:
        """Estimate front-high/rear-low terrain delta from the critic height scan."""
        num_rays = height_scan.shape[-1]
        if num_rays >= 6 and num_rays % 6 == 0:
            # Current highstep scanner: size=[1.6, 0.5], resolution=0.1 -> 6 y rows.
            scan = height_scan.reshape(height_scan.shape[0], 6, num_rays // 6)
            num_x = scan.shape[-1]
            x = torch.linspace(-0.8, 0.8, num_x, device=height_scan.device, dtype=height_scan.dtype)
            front_mask = x >= 0.25
            rear_mask = x <= -0.20
            if bool(torch.any(front_mask).item()) and bool(torch.any(rear_mask).item()):
                front_scan = scan[:, :, front_mask].mean(dim=(1, 2))
                rear_scan = scan[:, :, rear_mask].mean(dim=(1, 2))
            else:
                third = max(num_x // 3, 1)
                rear_scan = scan[:, :, :third].mean(dim=(1, 2))
                front_scan = scan[:, :, -third:].mean(dim=(1, 2))
        else:
            third = max(num_rays // 3, 1)
            rear_scan = height_scan[:, :third].mean(dim=1)
            front_scan = height_scan[:, -third:].mean(dim=1)

        # height_scan = sensor_z - hit_z - offset. Higher terrain gives a smaller scan value.
        return rear_scan - front_scan

    def _apply_phased_highstep_box_prior_to_raw_action(
        self, raw_action: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert teacher raw actions into raw-equivalent actions after the highstep box prior."""
        if raw_action.shape[-1] < 16 or critic_obs.shape[-1] <= 60:
            zeros = raw_action.new_zeros(raw_action.shape[0])
            return raw_action, zeros, zeros, raw_action[..., -4:], zeros

        height_scan = critic_obs[:, 60:]
        height_delta = self._estimate_forward_highstep_delta_from_scan(height_scan)
        height_gate = self._smoothstep((height_delta - 0.060) / 0.14)

        command = critic_obs[:, 9:12] if critic_obs.shape[-1] >= 12 else critic_obs.new_zeros(raw_action.shape[0], 3)
        cmd_gate = torch.clamp((command[:, 0] - 0.10) / 0.25, min=0.0, max=1.0)
        gate = height_gate * cmd_gate

        gravity_b = critic_obs[:, 6:9] if critic_obs.shape[-1] >= 9 else critic_obs.new_zeros(raw_action.shape[0], 3)
        pitch_metric = torch.abs(gravity_b[:, 0])
        commit_gate = self._smoothstep((pitch_metric - 0.08) / 0.17)
        reach_gate = gate * (1.0 - commit_gate)
        push_gate = gate * commit_gate

        reach_bias = torch.tensor([-0.020, -0.020, 0.002, 0.002], device=raw_action.device, dtype=raw_action.dtype)
        push_bias = torch.tensor([0.002, 0.002, -0.022, -0.022], device=raw_action.device, dtype=raw_action.dtype)
        box_bias = reach_gate.unsqueeze(1) * reach_bias + push_gate.unsqueeze(1) * push_bias

        target_action = raw_action.clone()
        box_default = 0.03
        box_scale = 0.02
        box_targets = raw_action[:, -4:] * box_scale + box_default + box_bias
        box_targets = torch.clamp(box_targets, min=0.0, max=0.06)
        target_action[:, -4:] = (box_targets - box_default) / box_scale

        front_box_mean = 0.5 * (box_targets[:, 0] + box_targets[:, 1])
        rear_box_mean = 0.5 * (box_targets[:, 2] + box_targets[:, 3])
        rear_minus_front_box = rear_box_mean - front_box_mean
        return target_action, torch.maximum(reach_gate, push_gate), height_delta, box_targets, rear_minus_front_box

    @staticmethod
    def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Mean with a zero-safe denominator for optional distillation terms."""
        return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0)

    def _student_main_action_reference(
        self, teacher_pre_prior: torch.Tensor, teacher_post_prior: torch.Tensor
    ) -> torch.Tensor:
        """Select the immutable Stage-2 main target without hiding route semantics."""
        if getattr(self, "student_recovery_stage", "NONE") == "V15":
            return teacher_post_prior
        return teacher_pre_prior

    def _configure_student_actor_trainable(
        self, train_box_head: bool, train_rear_hip_head: bool = False
    ) -> None:
        """Freeze the actor except selected final-layer action rows."""
        for param in self.policy.actor.parameters():
            param.requires_grad = False

        if not train_box_head and not train_rear_hip_head:
            return

        last_linear = None
        for module in self.policy.actor.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            return

        last_linear.weight.requires_grad = True
        if last_linear.bias is not None:
            last_linear.bias.requires_grad = True

        hook_key = (bool(train_box_head), bool(train_rear_hip_head), last_linear.out_features)
        if (
            getattr(self, "_student_action_head_hook_layer", None) is last_linear
            and getattr(self, "_student_action_head_hook_key", None) == hook_key
        ):
            return

        for handle in getattr(self, "_student_box_head_grad_hooks", []):
            handle.remove()

        trainable_rows = set()
        if train_rear_hip_head and last_linear.out_features >= 4:
            trainable_rows.update((2, 3))
        if train_box_head:
            box_rows = min(4, last_linear.out_features)
            trainable_rows.update(range(last_linear.out_features - box_rows, last_linear.out_features))

        weight_mask = torch.zeros_like(last_linear.weight)
        for row in trainable_rows:
            if 0 <= row < last_linear.out_features:
                weight_mask[row, :] = 1.0
        bias_mask = None
        if last_linear.bias is not None:
            bias_mask = torch.zeros_like(last_linear.bias)
            for row in trainable_rows:
                if 0 <= row < last_linear.out_features:
                    bias_mask[row] = 1.0

        self._student_box_head_grad_hooks = [
            last_linear.weight.register_hook(
                lambda grad, mask=weight_mask: grad * mask.to(device=grad.device, dtype=grad.dtype)
            )
        ]
        if last_linear.bias is not None:
            self._student_box_head_grad_hooks.append(
                last_linear.bias.register_hook(
                    lambda grad, mask=bias_mask: grad * mask.to(device=grad.device, dtype=grad.dtype)
                )
            )
        self._student_action_head_hook_layer = last_linear
        self._student_action_head_hook_key = hook_key

    def _highstep_rear_hip_min_action_loss(
        self, student_action: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep rear hip raw actions from collapsing inward during highstep approach."""
        if student_action.shape[-1] < 4 or critic_obs.shape[-1] <= 60:
            zero = student_action.new_zeros(())
            return zero, zero

        min_abs = float(getattr(self, "student_highstep_rear_hip_min_abs", 0.0))
        action_scale = max(float(getattr(self, "student_highstep_rear_hip_action_scale", 0.1)), 1.0e-6)
        if min_abs <= 0.0:
            zero = student_action.new_zeros(())
            return zero, zero

        height_scan = critic_obs[:, 60:]
        height_delta = self._estimate_forward_highstep_delta_from_scan(height_scan)
        height_gate = self._smoothstep((height_delta - 0.060) / 0.14)
        command = critic_obs[:, 9:12] if critic_obs.shape[-1] >= 12 else critic_obs.new_zeros(student_action.shape[0], 3)
        cmd_gate = torch.clamp((command[:, 0] - 0.06) / 0.20, min=0.0, max=1.0)
        gravity_b = critic_obs[:, 6:9] if critic_obs.shape[-1] >= 9 else critic_obs.new_zeros(student_action.shape[0], 3)
        commit_gate = self._smoothstep((torch.abs(gravity_b[:, 0]) - 0.08) / 0.17)
        approach_gate = (height_gate * cmd_gate * (1.0 - commit_gate)).detach()

        raw_min_abs = min_abs / action_scale
        rl_deficit = torch.clamp(raw_min_abs - student_action[:, 2], min=0.0)
        rr_deficit = torch.clamp(raw_min_abs + student_action[:, 3], min=0.0)
        per_sample_loss = 0.5 * (torch.square(rl_deficit) + torch.square(rr_deficit))
        return self._weighted_mean(per_sample_loss, approach_gate), approach_gate.mean()

    def _resolve_r2_post_prior_function(self):
        if self._r2_post_prior_function is None:
            from robot_lab.tasks.locomotion.velocity.mdp.actions import (
                compute_phased_highstep_post_prior,
            )

            self._r2_post_prior_function = compute_phased_highstep_post_prior
        return self._r2_post_prior_function

    def _r2_teacher_post_prior_target(
        self,
        teacher_raw_action: torch.Tensor,
        teacher_context: torch.Tensor,
    ):
        if teacher_context.ndim != 2 or teacher_context.shape[1] != 3:
            raise RuntimeError(
                f"R2 teacher_context must have shape [B,3], got {tuple(teacher_context.shape)}"
            )
        batch = teacher_raw_action.shape[0]
        scale = self._r2_action_scale.to(
            device=teacher_raw_action.device, dtype=teacher_raw_action.dtype
        ).expand(batch, -1)
        offset = self._r2_action_offset.to(
            device=teacher_raw_action.device, dtype=teacher_raw_action.dtype
        ).expand(batch, -1)
        clip = self._r2_action_clip.to(
            device=teacher_raw_action.device, dtype=teacher_raw_action.dtype
        ).expand(batch, -1, -1)
        processed = torch.clamp(
            teacher_raw_action * scale + offset,
            min=clip[..., 0],
            max=clip[..., 1],
        )
        contract = dict(self._r2_prior_contract)
        box_action_ids = contract.pop("box_action_ids").to(device=teacher_raw_action.device)
        result = self._resolve_r2_post_prior_function()(
            raw_actions=teacher_raw_action,
            processed_actions=processed,
            box_action_ids=box_action_ids,
            action_scale=scale,
            action_offset=offset,
            height_delta=teacher_context[:, 0],
            command_x=teacher_context[:, 1],
            front_rear_delta=teacher_context[:, 2],
            **contract,
        )
        target = getattr(result, "raw_equivalent_actions", None)
        push_gate = getattr(result, "push_gate", None)
        if (
            not isinstance(target, torch.Tensor)
            or target.shape != teacher_raw_action.shape
            or not isinstance(push_gate, torch.Tensor)
            or push_gate.shape != (batch,)
        ):
            raise RuntimeError("Shared R2 post-prior helper returned an invalid contract")
        if not bool(torch.isfinite(target).all().item()) or not bool(
            torch.isfinite(push_gate).all().item()
        ):
            raise RuntimeError("Shared R2 post-prior helper returned non-finite output")
        return target, push_gate.detach(), result

    @staticmethod
    def _r2_loss_terms(
        *,
        student_mu: torch.Tensor,
        teacher_latent: torch.Tensor,
        velocity_prediction: torch.Tensor,
        true_velocity: torch.Tensor,
        student_action: torch.Tensor,
        teacher_post_prior_action: torch.Tensor,
        b500_action: torch.Tensor,
        push_gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent_loss = torch.mean(torch.square(student_mu - teacher_latent))
        velocity_loss = torch.mean(torch.square(velocity_prediction - true_velocity))
        revolute_error = torch.mean(
            torch.square(student_action[:, :12] - teacher_post_prior_action[:, :12]), dim=1
        )
        box_error = torch.mean(
            torch.square(student_action[:, 12:16] - teacher_post_prior_action[:, 12:16]), dim=1
        )
        action_numerator = torch.sum(12.0 * push_gate * revolute_error + 20.0 * box_error)
        action_denominator = torch.clamp(torch.sum(12.0 * push_gate + 20.0), min=1.0)
        action_loss = action_numerator / action_denominator
        anchor_weight = torch.clamp(1.0 - push_gate, min=0.0, max=1.0)
        anchor_error = torch.mean(
            torch.square(student_action[:, :12] - b500_action[:, :12]), dim=1
        )
        anchor_loss = torch.sum(anchor_weight * anchor_error) / torch.clamp(
            torch.sum(anchor_weight), min=1.0
        )
        total = 50.0 * latent_loss + 10.0 * velocity_loss + 20.0 * action_loss + 5.0 * anchor_loss
        return {
            "total": total,
            "latent": latent_loss,
            "velocity": velocity_loss,
            "action": action_loss,
            "revolute_action": torch.mean(revolute_error),
            "box_action": torch.mean(box_error),
            "anchor": anchor_loss,
            "push_gate": torch.mean(push_gate),
            "anchor_weight": torch.mean(anchor_weight),
        }

    def act(self, obs):
        stage = getattr(self, "student_recovery_stage", "NONE")
        if stage not in {"R2", "R3"}:
            return super().act(obs)
        binding_ready = self._r2_binding_ready if stage == "R2" else self._r3_binding_ready
        if not binding_ready:
            raise RuntimeError(f"{stage} deterministic rollout attempted before checkpoint binding")
        if self.policy.is_recurrent:
            raise RuntimeError(f"{stage} preregistration does not permit a recurrent policy")
        actions = self.policy.act_inference(obs).detach()
        # R2 applies the deterministic actor mean and must never draw a Gaussian
        # action sample.  RSL's logger nevertheless reads ``policy.action_std``
        # after every update; that property requires a populated distribution.
        # Materialize the diagnostic distribution from the already-computed mean
        # and the frozen scalar std without calling ``sample()`` or changing the
        # action sent to the environment.
        if bool(getattr(self.policy, "state_dependent_std", False)):
            raise RuntimeError(f"{stage} requires the locked state-independent action std")
        if str(getattr(self.policy, "noise_std_type", "scalar")) != "scalar":
            raise RuntimeError(f"{stage} requires the locked scalar action std")
        policy_std = getattr(self.policy, "std", None)
        if not isinstance(policy_std, torch.Tensor) or policy_std.requires_grad:
            raise RuntimeError(f"{stage} action std is missing or unexpectedly trainable")
        action_sigma = policy_std.detach().to(device=actions.device, dtype=actions.dtype).expand_as(actions)
        if not bool(torch.isfinite(actions).all().item()) or not bool(
            torch.isfinite(action_sigma).all().item()
        ) or not bool(torch.all(action_sigma > 0.0).item()):
            raise RuntimeError(f"{stage} deterministic mean or logging std is invalid")
        self.policy.distribution = torch.distributions.Normal(actions, action_sigma)
        if not torch.equal(self.policy.distribution.mean, actions):
            raise RuntimeError(f"{stage} logging distribution changed the deterministic action mean")
        batch = actions.shape[0]
        self.transition.actions = actions
        self.transition.values = torch.zeros(batch, 1, device=actions.device, dtype=actions.dtype)
        self.transition.actions_log_prob = torch.zeros(
            batch, 1, device=actions.device, dtype=actions.dtype
        )
        self.transition.action_mean = actions.clone()
        self.transition.action_sigma = action_sigma.clone()
        self.transition.observations = obs
        return actions

    def compute_returns(self, last_critic_obs):
        if getattr(self, "student_recovery_stage", "NONE") in {"R2", "R3"}:
            # Recovery DAgger stages have no PPO/reward objective.
            return None
        return super().compute_returns(last_critic_obs)

    def _update_student_recovery_r2(self) -> dict:
        if not self._r2_binding_ready:
            raise RuntimeError("R2 update attempted before all bindings succeeded")
        self._validate_r2_optimizer_scope()
        self._assert_r2_frozen_unchanged()
        step_before_clear = int(self.storage.step)
        if step_before_clear != int(self.storage.num_transitions_per_env):
            raise RuntimeError(
                "R2 requires one complete rollout before update: "
                f"{step_before_clear} != {self.storage.num_transitions_per_env}"
            )
        observations = self.storage.observations
        required_keys = {
            self.policy.policy_keys[0],
            self.policy.estimator_keys[0],
            self.policy.critic_keys[0],
            self.policy.teacher_context_keys[0],
        }
        if not hasattr(observations, "keys") or not required_keys.issubset(set(observations.keys())):
            raise RuntimeError(f"R2 rollout is missing observation groups {sorted(required_keys)}")
        policy_obs = observations[self.policy.policy_keys[0]][:step_before_clear].flatten(0, 1)
        estimator_obs = observations[self.policy.estimator_keys[0]][:step_before_clear].flatten(0, 1)
        critic_obs = observations[self.policy.critic_keys[0]][:step_before_clear].flatten(0, 1)
        teacher_context = observations[self.policy.teacher_context_keys[0]][:step_before_clear].flatten(0, 1)
        if policy_obs.shape[-1] >= self.policy.policy_obs_dim + self.policy.vae_latent_dim:
            policy_obs = policy_obs[..., : self.policy.policy_obs_dim]
        dimensions = {
            "policy": (policy_obs.shape[-1], 570),
            "estimator": (estimator_obs.shape[-1], 570),
            "critic": (critic_obs.shape[-1], 162),
            "teacher_context": (teacher_context.shape[-1], 3),
        }
        mismatches = [f"{name}:{actual}!={expected}" for name, (actual, expected) in dimensions.items() if actual != expected]
        if mismatches:
            raise RuntimeError("R2 rollout dimensions changed: " + ", ".join(mismatches))
        tensors = (policy_obs, estimator_obs, critic_obs, teacher_context)
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
            raise RuntimeError("R2 rollout contains non-finite observation/context values")
        num_samples = policy_obs.shape[0]
        if num_samples == 0:
            raise RuntimeError("R2 received an empty rollout")

        indices = torch.randperm(num_samples, device=policy_obs.device)
        batches = [batch for batch in torch.tensor_split(indices, int(self.num_mini_batches)) if batch.numel()]
        if len(batches) != int(self.num_mini_batches) or sum(batch.numel() for batch in batches) != num_samples:
            raise RuntimeError("R2 mini-batching lost rollout samples")
        metric_sums = {
            "total": 0.0,
            "latent": 0.0,
            "velocity": 0.0,
            "action": 0.0,
            "revolute_action": 0.0,
            "box_action": 0.0,
            "anchor": 0.0,
            "push_gate": 0.0,
            "anchor_weight": 0.0,
        }
        total_grad_norm = 0.0
        actor_prefix_max_error = 0.0
        optimizer_steps = 0
        for _ in range(self.r2_epochs):
            for batch_indices in batches:
                batch_policy = policy_obs[batch_indices]
                batch_estimator = estimator_obs[batch_indices]
                batch_critic = critic_obs[batch_indices]
                batch_context = teacher_context[batch_indices]
                with torch.no_grad():
                    teacher_latent = self.r2_teacher_priv_encoder(batch_critic[:, 3:])
                    teacher_raw_action = self.r2_teacher_actor(
                        torch.cat((batch_policy, teacher_latent), dim=-1)
                    )
                    teacher_post, push_gate, _ = self._r2_teacher_post_prior_target(
                        teacher_raw_action, batch_context
                    )
                    anchor_mu, _, _ = self.r2_anchor_estimator.encode(batch_estimator)
                    anchor_action = self.r2_anchor_actor(
                        torch.cat((batch_policy, torch.clamp(anchor_mu, -1.0, 1.0)), dim=-1)
                    )
                hidden = self.policy.estimator.encoder(batch_estimator)
                student_mu = self.policy.estimator.fc_mu(hidden)
                velocity_prediction = self.policy.estimator.vel_estimator(hidden)
                student_action, prefix_error = self._r2_actor_forward(
                    torch.cat((batch_policy, torch.clamp(student_mu, -1.0, 1.0)), dim=-1)
                )
                actor_prefix_max_error = max(actor_prefix_max_error, prefix_error)
                losses = self._r2_loss_terms(
                    student_mu=student_mu,
                    teacher_latent=teacher_latent,
                    velocity_prediction=velocity_prediction,
                    true_velocity=batch_critic[:, :3],
                    student_action=student_action,
                    teacher_post_prior_action=teacher_post,
                    b500_action=anchor_action,
                    push_gate=push_gate,
                )
                if not all(bool(torch.isfinite(value).all().item()) for value in losses.values()):
                    raise RuntimeError("R2 loss became non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                owned = [parameter for group in self.optimizer.param_groups for parameter in group["params"]]
                if any(parameter.grad is None for parameter in owned):
                    raise RuntimeError("R2 optimizer-owned tensor has no gradient")
                if any(not bool(torch.isfinite(parameter.grad).all().item()) for parameter in owned):
                    raise RuntimeError("R2 gradient became non-finite")
                grad_norm = torch.sqrt(
                    sum(torch.sum(torch.square(parameter.grad)) for parameter in owned)
                )
                nn.utils.clip_grad_norm_(owned, self.max_grad_norm)
                self.optimizer.step()
                self._sync_r2_box_rows_to_actor()
                batch_count = int(batch_indices.numel())
                for name in metric_sums:
                    metric_sums[name] += float(losses[name].detach().item()) * batch_count
                total_grad_norm += float(grad_norm.detach().item()) * batch_count
                optimizer_steps += 1

        if optimizer_steps != self.r2_epochs * int(self.num_mini_batches):
            raise RuntimeError("R2 performed an unexpected number of Adam steps")
        self.storage.clear()
        self.student_distill_update_count += 1
        self._r2_binding_manifest["effective_update_count"] = int(
            self.student_distill_update_count
        )
        self._assert_r2_frozen_unchanged()
        self._validate_r2_optimizer_state(self.student_distill_update_count)
        denominator = float(num_samples * self.r2_epochs)
        return {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "Loss/R2_Total": metric_sums["total"] / denominator,
            "Loss/R2_Latent_MSE": metric_sums["latent"] / denominator,
            "Loss/R2_Velocity_Preservation_MSE": metric_sums["velocity"] / denominator,
            "Loss/R2_Post_Prior_Action_MSE": metric_sums["action"] / denominator,
            "Loss/R2_Revolute_Action_MSE": metric_sums["revolute_action"] / denominator,
            "Loss/R2_Box_Action_MSE": metric_sums["box_action"] / denominator,
            "Loss/R2_B500_Anchor_MSE": metric_sums["anchor"] / denominator,
            "Debug/R2_Push_Gate_Mean": metric_sums["push_gate"] / denominator,
            "Debug/R2_Anchor_Weight_Mean": metric_sums["anchor_weight"] / denominator,
            "Debug/R2_Grad_Norm": total_grad_norm / denominator,
            "Debug/R2_Actor_Prefix_Equivalence_Max_Error": actor_prefix_max_error,
            "Debug/R2_Effective_Update_Count": float(self.student_distill_update_count),
            "Debug/R2_Optimizer_Parameter_Tensors": 8.0,
            "Debug/R2_Adam_Steps_Per_Effective_Update": float(optimizer_steps),
            "Debug/R2_Deterministic_Mean_Rollout": 1.0,
            "Debug/R2_Buffer_Step_Before_Clear": float(step_before_clear),
            "Debug/R2_Buffer_Step_After_Clear": float(self.storage.step),
        }

    @staticmethod
    def _r3_loss_terms(
        *,
        student_action: torch.Tensor,
        teacher_post_prior_action: torch.Tensor,
        anchor_action: torch.Tensor,
        lateral_gate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if lateral_gate.ndim != 1 or lateral_gate.shape[0] != student_action.shape[0]:
            raise RuntimeError("R3 lateral gate shape changed")
        if not bool(torch.all((lateral_gate >= 0.0) & (lateral_gate <= 1.0)).item()):
            raise RuntimeError("R3 lateral gate escaped [0,1]")
        teacher_error = torch.mean(
            torch.square(student_action - teacher_post_prior_action), dim=1
        )
        anchor_error = torch.mean(torch.square(student_action - anchor_action), dim=1)
        center_gate = 1.0 - lateral_gate
        teacher_offset = torch.sum(lateral_gate * teacher_error) / torch.clamp(
            torch.sum(lateral_gate), min=1.0
        )
        anchor_center = torch.sum(center_gate * anchor_error) / torch.clamp(
            torch.sum(center_gate), min=1.0
        )
        total = teacher_offset + 4.0 * anchor_center
        return {
            "total": total,
            "teacher_offset": teacher_offset,
            "anchor_center": anchor_center,
            "teacher_all": torch.mean(teacher_error),
            "anchor_all": torch.mean(anchor_error),
            "lateral_gate": torch.mean(lateral_gate),
            "center_gate": torch.mean(center_gate),
        }

    def _update_student_recovery_r3(self) -> dict:
        if not self._r3_binding_ready:
            raise RuntimeError("R3 update attempted before all bindings succeeded")
        self._validate_r3_optimizer_scope()
        self._assert_r3_frozen_unchanged()
        step_before_clear = int(self.storage.step)
        if step_before_clear != int(self.storage.num_transitions_per_env):
            raise RuntimeError(
                "R3 requires one complete rollout before update: "
                f"{step_before_clear} != {self.storage.num_transitions_per_env}"
            )
        observations = self.storage.observations
        required_keys = {
            self.policy.policy_keys[0],
            self.policy.estimator_keys[0],
            self.policy.critic_keys[0],
            self.policy.teacher_context_keys[0],
        }
        if not hasattr(observations, "keys") or not required_keys.issubset(set(observations.keys())):
            raise RuntimeError(f"R3 rollout is missing observation groups {sorted(required_keys)}")
        policy_obs = observations[self.policy.policy_keys[0]][:step_before_clear].flatten(0, 1)
        estimator_obs = observations[self.policy.estimator_keys[0]][:step_before_clear].flatten(0, 1)
        critic_obs = observations[self.policy.critic_keys[0]][:step_before_clear].flatten(0, 1)
        teacher_context = observations[self.policy.teacher_context_keys[0]][:step_before_clear].flatten(0, 1)
        if policy_obs.shape[-1] >= self.policy.policy_obs_dim + self.policy.vae_latent_dim:
            policy_obs = policy_obs[..., : self.policy.policy_obs_dim]
        dimensions = {
            "policy": (policy_obs.shape[-1], 570),
            "estimator": (estimator_obs.shape[-1], 570),
            "critic": (critic_obs.shape[-1], 162),
            "teacher_context": (teacher_context.shape[-1], 3),
        }
        mismatches = [
            f"{name}:{actual}!={expected}"
            for name, (actual, expected) in dimensions.items()
            if actual != expected
        ]
        if mismatches:
            raise RuntimeError("R3 rollout dimensions changed: " + ", ".join(mismatches))
        tensors = (policy_obs, estimator_obs, critic_obs, teacher_context)
        if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
            raise RuntimeError("R3 rollout contains non-finite observations")
        num_samples = policy_obs.shape[0]
        if num_samples == 0:
            raise RuntimeError("R3 received an empty rollout")

        with torch.no_grad():
            left_scan, right_scan = self._estimate_lateral_height_from_scan(critic_obs[:, 60:])
            lateral_delta = left_scan - right_scan
            lateral_gate_all = self._smoothstep((torch.abs(lateral_delta) - 0.02) / 0.06)
        if (
            not bool(torch.isfinite(lateral_gate_all).all().item())
            or float(torch.sum(lateral_gate_all).item()) < 1.0
            or float(torch.sum(1.0 - lateral_gate_all).item()) < 1.0
        ):
            raise RuntimeError("R3 rollout lacks finite lateral and centered gate support")

        indices = torch.randperm(num_samples, device=policy_obs.device)
        batches = [
            batch
            for batch in torch.tensor_split(indices, int(self.num_mini_batches))
            if batch.numel()
        ]
        if len(batches) != int(self.num_mini_batches) or sum(
            batch.numel() for batch in batches
        ) != num_samples:
            raise RuntimeError("R3 mini-batching lost rollout samples")
        metric_sums = {
            "total": 0.0,
            "teacher_offset": 0.0,
            "anchor_center": 0.0,
            "teacher_all": 0.0,
            "anchor_all": 0.0,
            "lateral_gate": 0.0,
            "center_gate": 0.0,
        }
        total_grad_norm = 0.0
        optimizer_steps = 0
        for _ in range(self.r3_epochs):
            for batch_indices in batches:
                batch_policy = policy_obs[batch_indices]
                batch_estimator = estimator_obs[batch_indices]
                batch_critic = critic_obs[batch_indices]
                batch_context = teacher_context[batch_indices]
                batch_lateral_gate = lateral_gate_all[batch_indices]
                with torch.no_grad():
                    teacher_latent = self.r3_teacher_priv_encoder(batch_critic[:, 3:])
                    teacher_raw_action = self.r3_teacher_actor(
                        torch.cat((batch_policy, teacher_latent), dim=-1)
                    )
                    teacher_post, _, _ = self._r2_teacher_post_prior_target(
                        teacher_raw_action, batch_context
                    )
                    anchor_mu, _, _ = self.r3_anchor_estimator.encode(batch_estimator)
                    anchor_input = torch.cat(
                        (batch_policy, torch.clamp(anchor_mu, -1.0, 1.0)), dim=-1
                    )
                    anchor_action = self.r3_anchor_actor(anchor_input)
                    student_mu, _, _ = self.policy.estimator.encode(batch_estimator)
                    student_input = torch.cat(
                        (batch_policy, torch.clamp(student_mu, -1.0, 1.0)), dim=-1
                    )
                student_action = self.policy.actor(student_input)
                losses = self._r3_loss_terms(
                    student_action=student_action,
                    teacher_post_prior_action=teacher_post,
                    anchor_action=anchor_action,
                    lateral_gate=batch_lateral_gate,
                )
                if not all(bool(torch.isfinite(value).all().item()) for value in losses.values()):
                    raise RuntimeError("R3 loss became non-finite")
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                owned = [self._r3_last_linear.weight, self._r3_last_linear.bias]
                if any(parameter.grad is None for parameter in owned):
                    raise RuntimeError("R3 optimizer-owned tensor has no gradient")
                if any(not bool(torch.isfinite(parameter.grad).all().item()) for parameter in owned):
                    raise RuntimeError("R3 gradient became non-finite")
                grad_norm = torch.sqrt(
                    sum(torch.sum(torch.square(parameter.grad)) for parameter in owned)
                )
                nn.utils.clip_grad_norm_(owned, self.max_grad_norm)
                self.optimizer.step()
                batch_count = int(batch_indices.numel())
                for name in metric_sums:
                    metric_sums[name] += float(losses[name].detach().item()) * batch_count
                total_grad_norm += float(grad_norm.detach().item()) * batch_count
                optimizer_steps += 1

        if optimizer_steps != self.r3_epochs * int(self.num_mini_batches):
            raise RuntimeError("R3 performed an unexpected number of Adam steps")
        self.storage.clear()
        self.student_distill_update_count += 1
        self._r3_binding_manifest["effective_update_count"] = int(
            self.student_distill_update_count
        )
        self._assert_r3_frozen_unchanged()
        self._validate_r3_optimizer_state(self.student_distill_update_count)
        denominator = float(num_samples * self.r3_epochs)
        return {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "Loss/R3_Total": metric_sums["total"] / denominator,
            "Loss/R3_Teacher_Offset_MSE": metric_sums["teacher_offset"] / denominator,
            "Loss/R3_Anchor_Center_MSE": metric_sums["anchor_center"] / denominator,
            "Loss/R3_Teacher_All_MSE": metric_sums["teacher_all"] / denominator,
            "Loss/R3_Anchor_All_MSE": metric_sums["anchor_all"] / denominator,
            "Debug/R3_Lateral_Gate_Mean": metric_sums["lateral_gate"] / denominator,
            "Debug/R3_Center_Gate_Mean": metric_sums["center_gate"] / denominator,
            "Debug/R3_Grad_Norm": total_grad_norm / denominator,
            "Debug/R3_Effective_Update_Count": float(self.student_distill_update_count),
            "Debug/R3_Optimizer_Parameter_Tensors": 2.0,
            "Debug/R3_Adam_Steps_Per_Effective_Update": float(optimizer_steps),
            "Debug/R3_Deterministic_Mean_Rollout": 1.0,
            "Debug/R3_Buffer_Step_Before_Clear": float(step_before_clear),
            "Debug/R3_Buffer_Step_After_Clear": float(self.storage.step),
        }

    def _update_student_recovery_stage_b(self) -> dict:
        """Exact model_172300 RL/RR hip supervision with every Student tensor frozen."""
        if not self._recovery_binding_ready:
            raise RuntimeError("Stage B update attempted before dual-checkpoint binding")
        self._validate_stage_b_optimizer_scope()
        self._assert_recovery_frozen_unchanged()
        for parameter in self.policy.parameters():
            if parameter.requires_grad:
                raise RuntimeError("Stage B found a trainable policy parameter outside the row-only optimizer")
            parameter.grad = None

        model = self.policy
        obs_storage = self.storage.observations
        if hasattr(obs_storage, "keys"):
            policy_obs_full = obs_storage[model.policy_keys[0]].flatten(0, 1)
            estimator_obs_full = obs_storage[model.estimator_keys[0]].flatten(0, 1)
            critic_obs_full = obs_storage[model.critic_keys[0]].flatten(0, 1)
        else:
            obs_batch = obs_storage.flatten(0, 1)
            policy_obs_full = obs_batch
            estimator_obs_full = obs_batch
            critic_obs_full = obs_batch
        if policy_obs_full.shape[-1] >= model.policy_obs_dim + model.vae_latent_dim:
            policy_obs_full = policy_obs_full[..., :model.policy_obs_dim]
        if policy_obs_full.shape[0] == 0:
            raise RuntimeError("Stage B received an empty rollout")

        actor_modules = self._recovery_actor_modules
        if not actor_modules or actor_modules[-1] is not self._recovery_last_linear:
            raise RuntimeError("Stage B actor final layer changed after optimizer construction")

        with torch.no_grad():
            student_mu_full, _, _ = model.estimator.encode(estimator_obs_full)
            student_mu_full = torch.clamp(student_mu_full, min=-1.0, max=1.0)
            teacher_latent_full = self.teacher_priv_encoder(critic_obs_full[..., 3:])
            teacher_latent_full = torch.clamp(teacher_latent_full, min=-1.0, max=1.0)
            teacher_action_full = self.teacher_actor(
                torch.cat([policy_obs_full, teacher_latent_full], dim=-1)
            )
            teacher_rear_hip_full = teacher_action_full[..., 2:4]
            if not torch.isfinite(student_mu_full).all() or not torch.isfinite(teacher_rear_hip_full).all():
                raise RuntimeError("Stage B Student/Teacher forward pass produced non-finite values")
            probe_count = min(32, policy_obs_full.shape[0])
            prefix_probe = torch.cat(
                [policy_obs_full[:probe_count], student_mu_full[:probe_count]], dim=-1
            )
            _, actor_prefix_max_error = self._stage_b_actor_prefix(prefix_probe)

        weight_before = self.recovery_rear_hip_weight.detach().clone()
        bias_before = self.recovery_rear_hip_bias.detach().clone()
        num_samples = policy_obs_full.shape[0]
        num_batches = max(1, min(int(self.num_mini_batches), num_samples))
        batch_size = (num_samples + num_batches - 1) // num_batches
        total_loss = 0.0
        total_grad_norm = 0.0
        optimizer_steps = 0

        for _ in range(self.student_recovery_epochs):
            indices = torch.randperm(num_samples, device=policy_obs_full.device)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                with torch.no_grad():
                    actor_input = torch.cat(
                        [policy_obs_full[batch_indices], student_mu_full[batch_indices]], dim=-1
                    )
                    hidden = actor_input
                    for module in actor_modules[:-1]:
                        hidden = module(hidden)
                student_rear_hip = F.linear(
                    hidden,
                    self.recovery_rear_hip_weight,
                    self.recovery_rear_hip_bias,
                )
                target_rear_hip = teacher_rear_hip_full[batch_indices]
                loss = torch.mean(torch.square(student_rear_hip - target_rear_hip))
                if not torch.isfinite(loss):
                    raise RuntimeError("Stage B rear-hip supervision loss is non-finite")

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = (
                    self.recovery_rear_hip_weight.grad,
                    self.recovery_rear_hip_bias.grad,
                )
                if any(gradient is None for gradient in gradients):
                    raise RuntimeError("Stage B RL/RR hip row gradient is missing")
                if any(not torch.isfinite(gradient).all() for gradient in gradients):
                    raise RuntimeError("Stage B RL/RR hip row gradient is non-finite")
                grad_norm = torch.sqrt(sum(torch.sum(torch.square(gradient)) for gradient in gradients))
                nn.utils.clip_grad_norm_(
                    [self.recovery_rear_hip_weight, self.recovery_rear_hip_bias],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self._sync_recovery_rows_to_actor()
                total_loss += float(loss.item())
                total_grad_norm += float(grad_norm.item())
                optimizer_steps += 1

        if optimizer_steps == 0:
            raise RuntimeError("Stage B performed no optimizer step")
        row_weight_changed = not torch.equal(weight_before, self.recovery_rear_hip_weight.detach())
        row_bias_changed = not torch.equal(bias_before, self.recovery_rear_hip_bias.detach())
        if not row_weight_changed and not row_bias_changed:
            raise RuntimeError("Stage B optimizer step produced no RL/RR hip row change")

        step_before_clear = int(self.storage.step)
        self.storage.clear()
        self.student_distill_update_count += 1
        self._recovery_binding_manifest["effective_update_count"] = int(
            self.student_distill_update_count
        )
        self._assert_recovery_frozen_unchanged()
        for parameter in self.policy.parameters():
            if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() != 0:
                raise RuntimeError("Frozen Stage B policy parameter accumulated a non-zero gradient")

        return {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "Loss/StageB_Rear_Hip_Teacher_MSE": total_loss / optimizer_steps,
            "Debug/StageB_Rear_Hip_Grad_Norm": total_grad_norm / optimizer_steps,
            "Debug/StageB_Effective_Update_Count": float(self.student_distill_update_count),
            "Debug/StageB_Optimizer_Parameter_Tensors": 2.0,
            "Debug/StageB_Trainable_Action_Row_RL": 2.0,
            "Debug/StageB_Trainable_Action_Row_RR": 3.0,
            "Debug/StageB_Frozen_Tensor_Violation_Count": 0.0,
            "Debug/StageB_Teacher_Binding_Ready": 1.0,
            "Debug/StageB_Actor_Prefix_Equivalence_Max_Error": actor_prefix_max_error,
            "Debug/Student_Warmup_Updates": 0.0,
            "Debug/StageB_Buffer_Step_Before_Clear": float(step_before_clear),
            "Debug/StageB_Buffer_Step_After_Clear": float(self.storage.step),
        }

    def update(self):
        if getattr(self, "student_recovery_stage", "NONE") == "B300_CANONICAL_HYBRID":
            return self._update_b300_canonical_hybrid()
        if getattr(self, "student_recovery_stage", "NONE") == "R2":
            return self._update_student_recovery_r2()
        if getattr(self, "student_recovery_stage", "NONE") == "R3":
            return self._update_student_recovery_r3()
        if getattr(self, "student_recovery_stage", "NONE") == "B":
            return self._update_student_recovery_stage_b()
        if self.distill_stage == 1:
            # 1. 先执行标准的 PPO 更新 (Actor 和 Critic)
            # 阶段 1: 正常的 PPO 更新 (训练 Actor, Critic, PrivEncoder)，不训练 VAE
            loss_dict = super().update()
            
            # 保留核心 Debug 监控，用于验证有效修改是否生效
            with torch.no_grad():
                # 监控 1：检查 Teacher 是否收到了梯度 (证明 Teacher 正在学习)
                grad_norm = sum(p.grad.norm().item() for p in self.policy.priv_encoder.parameters() if p.grad is not None)
                loss_dict["Debug/Teacher_Grad_Norm"] = grad_norm
                
                # 监控 2：检查 Actor 第一层新增的 64 维权重 (证明零初始化有效且 Actor 正在利用 Latent)
                if hasattr(self.policy, 'actor') and isinstance(self.policy.actor[0], nn.Linear):
                    zero_weight_sum = self.policy.actor[0].weight[:, self.policy.policy_obs_dim:].abs().sum().item()
                    loss_dict["Debug/Actor_Layer1_Zero_Weight_Sum"] = zero_weight_sum
                
            return loss_dict
            
        else:
            # ==========================================
            # Stage 2: blind student distillation with box-prior isolation.
            # Warmup aligns the VAE with the teacher while the actor is fully frozen.
            # After warmup, only selected final-layer action rows can adapt; no PPO is
            # opened here because full actor/RL updates polluted the teacher gait.
            # ==========================================
            model = self.policy
            self._sync_teacher_actor_from_policy_once()
            if not hasattr(self, "student_distill_update_count"):
                self.student_distill_update_count = 0
            warmup_updates = getattr(self, "student_actor_warmup_updates", 800)
            actor_adapt_enabled = self.student_distill_update_count >= warmup_updates
            recovery_stage = getattr(self, "student_recovery_stage", "NONE")
            v15_active = recovery_stage in {
                "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
                "E1400_CONTINUATION", "BE300_0707", "B300_CANONICAL_HYBRID",
            }
            pre_prior_0707_active = recovery_stage in {
                "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
                "E1400_CONTINUATION", "BE300_0707",
            }
            rear_hip_adapt_enabled = (
                actor_adapt_enabled
                and getattr(self, "student_highstep_rear_hip_loss_scale", 0.0) > 0.0
                and getattr(self, "student_highstep_rear_hip_min_abs", 0.0) > 0.0
            )
            if v15_active:
                self._validate_v15_optimizer_scope()
            else:
                self._configure_student_actor_trainable(
                    train_box_head=actor_adapt_enabled,
                    train_rear_hip_head=rear_hip_adapt_enabled,
                )
            for param in model.critic.parameters():
                param.requires_grad = False
            
            if model.estimator is not None and self.num_learning_epochs > 0:
                obs_storage = self.storage.observations
                rollout_steps = int(self.storage.step)
                current_policy_time = None
                current_est_input_time = None
                current_critic_time = None
                if hasattr(obs_storage, "keys"):
                    current_policy_time = obs_storage[model.policy_keys[0]][:rollout_steps]
                    current_est_input_time = obs_storage[model.estimator_keys[0]][:rollout_steps]
                    current_critic_time = obs_storage[model.critic_keys[0]][:rollout_steps]
                    policy_obs_full = current_policy_time.flatten(0, 1)
                    est_input_full = current_est_input_time.flatten(0, 1)
                    critic_obs_full = current_critic_time.flatten(0, 1)
                else:
                    obs_batch = obs_storage[:rollout_steps].flatten(0, 1)
                    policy_obs_full = obs_batch
                    est_input_full = obs_batch
                    critic_obs_full = obs_batch

                if policy_obs_full.shape[-1] >= model.policy_obs_dim + model.vae_latent_dim:
                    policy_obs_full = policy_obs_full[..., :model.policy_obs_dim]
                    if current_policy_time is not None:
                        current_policy_time = current_policy_time[..., :model.policy_obs_dim]

                current_num_samples = est_input_full.shape[0]
                critical_context_full = None
                front_transition_window = None
                rear_transition_window = None
                original_distribution_pool = None
                if self.student_critical_transition_balanced_sampling:
                    if not hasattr(obs_storage, "keys") or "critical_transition" not in obs_storage.keys():
                        raise RuntimeError(
                            "balanced critical-transition route is missing its rollout-only label group"
                        )
                    stage_context_time = obs_storage["critical_transition"][:rollout_steps]
                    if tuple(stage_context_time.shape[:2]) != tuple(self.storage.dones[:rollout_steps].shape[:2]):
                        raise RuntimeError("critical-transition context/storage time axes disagree")
                    current_dones_time = self.storage.dones[:rollout_steps]
                    carry = self._critical_transition_fifo
                    if self.student_rl_preedge_sampling:
                        combined_context_time = (
                            torch.cat((carry["stage_context"], stage_context_time), dim=0)
                            if carry is not None else stage_context_time
                        )
                        combined_dones_time = (
                            torch.cat((carry["dones"], current_dones_time), dim=0)
                            if carry is not None else current_dones_time
                        )
                        front_window_time, _ = highstep_critical_transition_window_masks(
                            combined_context_time[..., :4],
                            combined_dones_time,
                            radius=self.student_critical_transition_window_radius,
                            require_complete_window=True,
                        )
                        rear_window_time = highstep_rl_preedge_window_mask(
                            combined_context_time,
                            combined_dones_time,
                            pre_steps=self.student_rl_preedge_pre_steps,
                            post_steps=self.student_rl_preedge_post_steps,
                        )
                        carry_steps = int(carry["stage_context"].shape[0]) if carry is not None else 0
                    else:
                        front_window_time, rear_window_time, carry_steps = highstep_cross_rollout_transition_windows(
                            stage_context_time,
                            current_dones_time,
                            radius=self.student_critical_transition_window_radius,
                            carry_stage_context=(carry or {}).get("stage_context"),
                            carry_dones=(carry or {}).get("dones"),
                        )
                    if carry is not None:
                        policy_obs_full = torch.cat(
                            (carry["policy_obs"], current_policy_time), dim=0
                        ).flatten(0, 1)
                        est_input_full = torch.cat(
                            (carry["est_input"], current_est_input_time), dim=0
                        ).flatten(0, 1)
                        critic_obs_full = torch.cat(
                            (carry["critic_obs"], current_critic_time), dim=0
                        ).flatten(0, 1)
                        critical_context_full = torch.cat(
                            (carry["stage_context"], stage_context_time), dim=0
                        ).flatten(0, 1)
                    else:
                        critical_context_full = stage_context_time.flatten(0, 1)
                    if self.student_rl_preedge_sampling and stage_context_time.shape[-1] != 5:
                        raise RuntimeError("RL pre-edge continuation requires five rollout labels")
                    front_transition_window = front_window_time.flatten()
                    rear_transition_window = rear_window_time.flatten()
                    if self.student_rl_preedge_sampling and (
                        not bool(front_transition_window.any().item())
                        or not bool(rear_transition_window.any().item())
                    ):
                        # Seed/advance the bounded FIFO without an optimizer or
                        # effective update until both complete sampling classes
                        # exist. This never substitutes truncated windows.
                        keep = self.student_rl_preedge_pre_steps + self.student_rl_preedge_post_steps
                        def _prefill_retain(key: str, current: torch.Tensor) -> torch.Tensor:
                            combined = (
                                torch.cat((carry[key], current), dim=0)
                                if carry is not None else current
                            )
                            return combined[-keep:].detach().clone()
                        self._critical_transition_fifo = {
                            "policy_obs": _prefill_retain("policy_obs", current_policy_time),
                            "est_input": _prefill_retain("est_input", current_est_input_time),
                            "critic_obs": _prefill_retain("critic_obs", current_critic_time),
                            "stage_context": _prefill_retain("stage_context", stage_context_time),
                            "dones": _prefill_retain("dones", current_dones_time),
                        }
                        self.storage.clear()
                        return {
                            "value_function": 0.0,
                            "surrogate": 0.0,
                            "entropy": 0.0,
                            "Debug/RL_PreEdge_FIFO_Prefill": 1.0,
                            "Debug/RL_PreEdge_Current_Active_Fraction": float(
                                stage_context_time[..., 4].float().mean().item()
                            ),
                            "Debug/RL_PreEdge_Front_Window_Count": float(
                                front_transition_window.sum().item()
                            ),
                            "Debug/RL_PreEdge_Rear_Window_Count": float(
                                rear_transition_window.sum().item()
                            ),
                            "Debug/Student_Distill_Update_Count": float(
                                self.student_distill_update_count
                            ),
                        }
                    num_samples = est_input_full.shape[0]
                    if critical_context_full.shape[0] != num_samples:
                        raise RuntimeError("critical-transition labels do not align with rollout samples")
                    current_start = carry_steps * int(stage_context_time.shape[1])
                    original_distribution_pool = torch.arange(
                        current_start,
                        num_samples,
                        device=est_input_full.device,
                        dtype=torch.long,
                    )
                    if original_distribution_pool.numel() != current_num_samples:
                        raise RuntimeError("critical-transition original pool is not the live rollout")
                else:
                    num_samples = current_num_samples

                # The minibatch budget remains exactly the unmodified live
                # rollout budget. FIFO frames are eligible only for the 25/25
                # transition portions and never inflate independent data size.
                batch_size = current_num_samples // self.num_mini_batches
                # 假设 critic 组的前 3 维就是真实的 base_lin_vel
                target_vel_full = critic_obs_full[..., 0:3]
                # 获取 Teacher 的目标 (Teacher Latent)
                with torch.no_grad():
                    # <=== STRICT FIX 4: 蒸馏时，Teacher 依然只吃切除速度后的观测
                    target_latent_full = model.priv_encoder(critic_obs_full[..., 3:])
                if self.student_critical_transition_balanced_sampling:
                    if batch_size % 4:
                        raise RuntimeError(
                            "critical-transition minibatch size must be divisible by four for 25/25/50"
                        )
                
                total_vel_loss = 0.0
                total_recon_loss = 0.0
                total_kl_loss = 0.0
                total_latent_loss = 0.0
                total_action_loss = 0.0
                total_teacher_action_loss = 0.0
                total_prior_box_loss = 0.0
                total_static_action_loss = 0.0
                total_box_action_loss = 0.0
                total_post_prior_gate = 0.0
                total_prior_box_weight = 0.0
                total_post_prior_box_delta = 0.0
                total_post_prior_lower_minus_upper_box = 0.0
                total_actor_grad_norm = 0.0
                total_highstep_phase_weight = 0.0
                total_highstep_phase_action_loss = 0.0
                total_highstep_rear_box_action_loss = 0.0
                total_highstep_rear_hip_action_loss = 0.0
                total_highstep_rear_hip_gate = 0.0
                total_front_sampling_fraction = 0.0
                total_rear_sampling_fraction = 0.0
                total_original_sampling_fraction = 0.0
                total_front_phase_gate = 0.0
                total_rear_phase_gate = 0.0
                update_steps = 0

                # 【核心修复 3】：使用 Epoch 和 Mini-batch 训练 VAE
                # 【新增优化】：解耦 VAE 与 PPO 的 Epoch，强制 VAE 每次 PPO 更新只训练 1-2 个 Epoch
                # 阶段2蒸馏可以适当增加 epoch，这里设为 5 以加速蒸馏
                vae_epochs = getattr(self, "student_vae_epochs", 4)
                for epoch in range(vae_epochs):
                    # Baseline route keeps the original random permutation.
                    indices = (
                        None
                        if self.student_critical_transition_balanced_sampling
                        else torch.randperm(num_samples, device=est_input_full.device)
                    )
                    
                    for i in range(self.num_mini_batches):
                        if self.student_critical_transition_balanced_sampling:
                            batch_idx, sampling_source = highstep_balanced_transition_indices(
                                front_transition_window,
                                rear_transition_window,
                                batch_size,
                                original_pool=original_distribution_pool,
                            )
                            total_front_sampling_fraction += (sampling_source == 0).float().mean().item()
                            total_rear_sampling_fraction += (sampling_source == 1).float().mean().item()
                            total_original_sampling_fraction += (sampling_source == 2).float().mean().item()
                            stage_context = critical_context_full[batch_idx]
                            front_phase_gate = (
                                torch.max(stage_context[:, :2], dim=-1).values >= 0.5
                            ).to(dtype=est_input_full.dtype)
                            rear_phase_gate = (
                                torch.max(stage_context[:, 2:4], dim=-1).values >= 0.5
                            ).to(dtype=est_input_full.dtype)
                        else:
                            start = i * batch_size
                            end = (i + 1) * batch_size
                            batch_idx = indices[start:end]
                            front_phase_gate = est_input_full.new_zeros(batch_size)
                            rear_phase_gate = est_input_full.new_zeros(batch_size)
                        
                        est_input = est_input_full[batch_idx]
                        policy_obs = policy_obs_full[batch_idx]
                        critic_obs = critic_obs_full[batch_idx]
                        target_vel = target_vel_full[batch_idx]
                        target_latent = target_latent_full[batch_idx]
                        
                        # 使用 VAE 专属优化器清空梯度
                        self.vae_optimizer.zero_grad()
                        
                        # VAE 前向传播
                        vel_pred, recon_x, mu, logvar, z = model.estimator(est_input)
                        
                        # 计算 Loss
                        vel_loss = (vel_pred - target_vel).pow(2).mean()
                        recon_loss = (recon_x - est_input).pow(2).mean()
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                        
                        # ==========================================
                        # 【本次修改】：核心蒸馏 Loss 归一化处理
                        # 理由：Teacher 移除了 Tanh，输出无界；而 VAE 的 mu 受 KL 约束在 0 附近。
                        # 直接算 MSE 会导致 Loss 爆炸。通过 L2 归一化，将 MSE 转化为余弦相似度约束。
                        # ==========================================
                        # 【修复 3】：绝对不能使用 L2 Normalize！因为我们已经加回了 Tanh，Teacher 输出有界。
                        # L2 归一化会丢弃向量的模长信息，导致 Student 学不到真实的特征尺度。直接计算 MSE Loss。
                        latent_loss = (mu - target_latent).pow(2).mean()

                        # The route authority selects the main target explicitly.  The historical
                        # 0707 and archived zero-scale routes use Teacher pre-prior action; v1.5
                        # uses the post-prior target.  The separate box-prior term stays zero
                        # throughout warmup for both 0707 routes.
                        clamped_mu = torch.clamp(mu, min=-1.0, max=1.0)
                        if model.student_actor_latent_clamp_backward == "straight_through":
                            # v1.14 changes only the backward graph. Forward remains
                            # exactly equal to the deployment hard clamp.
                            safe_mu = mu + (clamped_mu - mu).detach()
                        elif model.student_actor_latent_clamp_backward == "hard":
                            safe_mu = clamped_mu
                        else:
                            raise RuntimeError(
                                "unsupported student_actor_latent_clamp_backward: "
                                f"{model.student_actor_latent_clamp_backward!r}"
                            )
                        with torch.no_grad():
                            safe_target_latent = torch.clamp(target_latent, min=-1.0, max=1.0)
                            teacher_actor = getattr(self, "teacher_actor", model.actor)
                            teacher_action = teacher_actor(torch.cat([policy_obs, safe_target_latent], dim=-1))
                            if getattr(self, "student_post_prior_mode", "lateral") == "highstep":
                                (
                                    teacher_action_target,
                                    post_prior_gate,
                                    _,
                                    post_prior_box_targets,
                                    post_prior_lower_minus_upper_box,
                                ) = self._apply_phased_highstep_box_prior_to_raw_action(teacher_action, critic_obs)
                            else:
                                (
                                    teacher_action_target,
                                    post_prior_gate,
                                    _,
                                    post_prior_box_targets,
                                    post_prior_lower_minus_upper_box,
                                ) = self._apply_lateral_step_box_prior_to_raw_action(teacher_action, critic_obs)

                        student_actor_input = torch.cat([policy_obs, safe_mu], dim=-1)
                        student_action = (
                            self._v15_actor_forward(student_actor_input)
                            if v15_active and actor_adapt_enabled
                            else model.actor(student_actor_input)
                        )
                        command = critic_obs[:, 9:12]
                        command_speed = torch.norm(command, dim=-1)
                        low_speed_threshold = getattr(self, "student_low_speed_threshold", 0.12)
                        prior_fade_speed = getattr(self, "student_prior_fade_speed", 0.45)
                        is_static = (command_speed < low_speed_threshold).float()
                        low_speed_gate = torch.clamp(
                            (prior_fade_speed - command_speed) / max(prior_fade_speed - low_speed_threshold, 1e-6),
                            min=0.0,
                            max=1.0,
                        )
                        gravity_b = critic_obs[:, 6:9]
                        tilt_scale = torch.clamp(torch.norm(gravity_b[:, :2], dim=-1) / 0.25, min=0.0, max=1.0)

                        # Movement anchor: always imitate the teacher actor before the hand-coded action prior.
                        # This keeps the blind student from sacrificing gait just to satisfy box posture.
                        teacher_action_reference = self._student_main_action_reference(
                            teacher_action, teacher_action_target
                        )
                        teacher_action_error = torch.square(
                            student_action - teacher_action_reference
                        )
                        if teacher_action_error.shape[-1] >= 16:
                            non_box_loss = torch.mean(teacher_action_error[:, :12], dim=-1)
                            raw_box_loss = torch.mean(teacher_action_error[:, -4:], dim=-1)
                            rear_box_action_loss = torch.mean(teacher_action_error[:, -2:], dim=-1)
                            # Keep gait as the dominant target. Box raw-action imitation is relaxed only
                            # on clear post-prior samples after the isolated box head starts adapting.
                            if actor_adapt_enabled:
                                box_raw_weight = torch.clamp(1.0 - post_prior_gate, min=0.0, max=1.0)
                            else:
                                box_raw_weight = torch.ones_like(post_prior_gate)
                            per_sample_teacher_action_loss = 2.0 * non_box_loss + 0.5 * raw_box_loss * box_raw_weight
                            per_sample_teacher_action_loss = (
                                per_sample_teacher_action_loss
                                + highstep_diagonal_action_loss(
                                    teacher_action_error,
                                    post_prior_gate,
                                    getattr(
                                        self,
                                        "student_highstep_diagonal_action_loss_scale",
                                        0.0,
                                    ),
                                )
                                + highstep_phase_diagonal_action_loss(
                                    teacher_action_error,
                                    front_phase_gate,
                                    rear_phase_gate,
                                    self.student_front_diagonal_action_loss_scale,
                                    self.student_rear_diagonal_action_loss_scale,
                                )
                            )
                        else:
                            per_sample_teacher_action_loss = torch.mean(teacher_action_error, dim=-1)
                            rear_box_action_loss = per_sample_teacher_action_loss
                        moving_weight = 1.0 + torch.clamp(command_speed / prior_fade_speed, min=0.0, max=1.0)
                        highstep_phase_scale = getattr(self, "student_highstep_phase_loss_scale", 0.0)
                        rear_box_phase_scale = getattr(self, "student_highstep_rear_box_loss_scale", 0.0)
                        highstep_phase_weight = 1.0 + highstep_phase_scale * post_prior_gate.detach()
                        per_sample_teacher_action_loss = (
                            per_sample_teacher_action_loss
                            + rear_box_phase_scale * post_prior_gate.detach() * rear_box_action_loss
                        )
                        teacher_action_loss = torch.mean(
                            per_sample_teacher_action_loss * moving_weight * highstep_phase_weight
                        )
                        highstep_phase_action_loss = self._weighted_mean(
                            torch.mean(teacher_action_error, dim=-1), post_prior_gate.detach()
                        )
                        highstep_rear_box_action_loss = self._weighted_mean(
                            rear_box_action_loss, post_prior_gate.detach()
                        )

                        # Post-prior loss is now a conditional box-only add-on. It is strong for static/low-speed
                        # prior samples, weak for moving samples, and disabled during the initial actor warmup.
                        if student_action.shape[-1] >= 16:
                            prior_actor_input = torch.cat([policy_obs, safe_mu.detach()], dim=-1)
                            student_action_for_prior = (
                                self._v15_actor_forward(prior_actor_input)
                                if v15_active and actor_adapt_enabled
                                else model.actor(prior_actor_input)
                            )
                            per_sample_prior_box_loss = torch.mean(
                                torch.square(student_action_for_prior[:, -4:] - teacher_action_target[:, -4:]), dim=-1
                            )
                            prior_box_weight = post_prior_gate * (
                                0.15 + 1.25 * low_speed_gate + 1.0 * is_static + 0.75 * tilt_scale
                            )
                            if not actor_adapt_enabled:
                                prior_box_weight = torch.zeros_like(prior_box_weight)
                            prior_box_loss = self._weighted_mean(per_sample_prior_box_loss, prior_box_weight)
                            box_action_loss = (
                                torch.zeros((), device=student_action.device)
                                if pre_prior_0707_active and not actor_adapt_enabled
                                else torch.mean(per_sample_prior_box_loss)
                            )
                        else:
                            prior_box_weight = torch.zeros_like(command_speed)
                            prior_box_loss = torch.zeros((), device=student_action.device)
                            box_action_loss = torch.mean(torch.square(student_action - teacher_action_target))

                        action_loss = teacher_action_loss + prior_box_loss
                        if rear_hip_adapt_enabled:
                            rear_hip_action_loss, rear_hip_gate_mean = self._highstep_rear_hip_min_action_loss(
                                student_action, critic_obs
                            )
                        else:
                            rear_hip_action_loss = torch.zeros((), device=student_action.device)
                            rear_hip_gate_mean = torch.zeros((), device=student_action.device)
                        static_action_loss = self._weighted_mean(
                            torch.mean(torch.square(student_action - teacher_action_target), dim=-1), is_static
                        )
                        
                        # 权重系数：因为速度只有3维，重构维度很大，必须放大速度 Loss 的权重
                        # ==========================================
                        # 【新增修改】：将 beta 从 0.01 调大到 0.1，进一步强迫网络降低 KL
                        # ==========================================
                        beta = getattr(self, "student_kl_loss_coef", 0.1)
                        
                        # ==========================================
                        # 【本次核心修复】：大幅提升 latent_loss 的权重！
                        # 理由：当前 latent_loss 权重为 1.0，完全被高维度的 recon_loss 淹没。
                        # 蒸馏的核心是模仿 Teacher，必须强制 VAE 优先降低 latent_loss。
                        # 将权重从 1.0 提升到 50.0。
                        # ==========================================
                        vae_loss = (
                            getattr(self, "student_vel_loss_coef", 10.0) * vel_loss
                            + getattr(self, "student_latent_loss_coef", 50.0) * latent_loss
                            + getattr(self, "student_teacher_action_loss_coef", 20.0) * teacher_action_loss
                            + getattr(self, "student_prior_box_loss_coef", 5.0) * prior_box_loss
                            + getattr(self, "student_highstep_rear_hip_loss_scale", 0.0) * rear_hip_action_loss
                            + getattr(self, "student_recon_loss_coef", 0.5) * recon_loss
                            + beta * kl_loss
                        )
                        
                        # 反向传播并使用 VAE 专属优化器更新
                        vae_loss.backward()
                        actor_grad_norm = sum(
                            p.grad.detach().norm().item() for p in model.actor.parameters() if p.grad is not None
                        )
                        if v15_active:
                            actor_grad_norm += sum(
                                parameter.grad.detach().norm().item()
                                for parameter in (self.v15_box_weight, self.v15_box_bias)
                                if parameter.grad is not None
                            )
                            distill_params = list(model.estimator.parameters()) + [
                                self.v15_box_weight,
                                self.v15_box_bias,
                            ]
                        else:
                            distill_params = list(model.estimator.parameters()) + [
                                param for param in model.actor.parameters() if param.requires_grad
                            ]
                        nn.utils.clip_grad_norm_(distill_params, self.max_grad_norm)
                        self.vae_optimizer.step()
                        if v15_active and actor_adapt_enabled:
                            self._sync_v15_box_rows_to_actor()
                        
                        total_vel_loss += vel_loss.item()
                        total_recon_loss += recon_loss.item()
                        total_kl_loss += kl_loss.item()
                        total_latent_loss += latent_loss.item()
                        total_action_loss += action_loss.item()
                        total_teacher_action_loss += teacher_action_loss.item()
                        total_prior_box_loss += prior_box_loss.item()
                        total_static_action_loss += static_action_loss.item()
                        total_box_action_loss += box_action_loss.item()
                        total_post_prior_gate += post_prior_gate.mean().item()
                        total_prior_box_weight += prior_box_weight.mean().item()
                        total_post_prior_box_delta += torch.mean(
                            torch.abs(post_prior_box_targets - (teacher_action[:, -4:] * 0.02 + 0.03))
                        ).item()
                        total_post_prior_lower_minus_upper_box += post_prior_lower_minus_upper_box.mean().item()
                        total_actor_grad_norm += actor_grad_norm
                        total_highstep_phase_weight += highstep_phase_weight.mean().item()
                        total_highstep_phase_action_loss += highstep_phase_action_loss.item()
                        total_highstep_rear_box_action_loss += highstep_rear_box_action_loss.item()
                        total_highstep_rear_hip_action_loss += rear_hip_action_loss.item()
                        total_highstep_rear_hip_gate += rear_hip_gate_mean.item()
                        total_front_phase_gate += front_phase_gate.mean().item()
                        total_rear_phase_gate += rear_phase_gate.mean().item()
                        update_steps += 1
                
                # ==========================================
                # 【新增修改】：记录清空前的 buffer 指针位置，用于证明修复有效
                # ==========================================
                step_before_clear = self.storage.step

                if self.student_critical_transition_balanced_sampling:
                    keep_steps = min(
                        (
                            int(current_policy_time.shape[0])
                            + int((self._critical_transition_fifo or {}).get(
                                "policy_obs", current_policy_time[:0]
                            ).shape[0])
                        ),
                        (
                            self.student_rl_preedge_pre_steps
                            + self.student_rl_preedge_post_steps
                            if self.student_rl_preedge_sampling else
                            2 * self.student_critical_transition_window_radius
                        ),
                    )
                    if keep_steps <= 0:
                        raise RuntimeError("critical-transition FIFO cannot retain zero steps")
                    # Clone before storage.clear/next rollout overwrites the
                    # underlying tensors. This is a bounded, same-env carry;
                    # it is not counted as new independent dataset material.
                    prior_fifo = self._critical_transition_fifo
                    def _retain(key: str, current: torch.Tensor) -> torch.Tensor:
                        combined = (
                            torch.cat((prior_fifo[key], current), dim=0)
                            if prior_fifo is not None else current
                        )
                        return combined[-keep_steps:].detach().clone()
                    self._critical_transition_fifo = {
                        "policy_obs": _retain("policy_obs", current_policy_time),
                        "est_input": _retain("est_input", current_est_input_time),
                        "critic_obs": _retain("critic_obs", current_critic_time),
                        "stage_context": _retain("stage_context", stage_context_time),
                        "dones": _retain("dones", current_dones_time),
                    }
                
                # Stage 2 intentionally does not run PPO. The previous PPO path let the
                # current reward mix overwrite teacher gait while chasing box posture.
                self.storage.clear()
                
                # ==========================================
                # 【新增修改】：记录清空后的 buffer 指针位置，用于证明修复有效
                # ==========================================
                step_after_clear = self.storage.step

                loss_dict = {
                    "value_function": 0.0,
                    "surrogate": 0.0,
                    "entropy": 0.0,
                    "Loss/VAE_Vel_MSE": total_vel_loss / update_steps,
                    "Loss/VAE_Recon_MSE": total_recon_loss / update_steps,
                    "Loss/VAE_KL": total_kl_loss / update_steps,
                    "Loss/Distill_Latent_MSE": total_latent_loss / update_steps,
                    "Loss/Post_Prior_Action_MSE": total_action_loss / update_steps,
                    "Loss/Teacher_Action_MSE": total_teacher_action_loss / update_steps,
                    "Loss/Prior_Box_Loss": total_prior_box_loss / update_steps,
                    "Loss/Static_Action_MSE": total_static_action_loss / update_steps,
                    "Loss/Post_Prior_Box_Action_MSE": total_box_action_loss / update_steps,
                    "Loss/Highstep_Phase_Teacher_Action_MSE": (
                        total_highstep_phase_action_loss / update_steps
                    ),
                    "Loss/Highstep_Rear_Box_Action_MSE": (
                        total_highstep_rear_box_action_loss / update_steps
                    ),
                    "Loss/Highstep_Rear_Hip_Action_Min_MSE": (
                        total_highstep_rear_hip_action_loss / update_steps
                    ),
                    "Debug/Highstep_Support_Phase_Weight_Mean": total_highstep_phase_weight / update_steps,
                    "Debug/Highstep_Rear_Hip_Approach_Gate": total_highstep_rear_hip_gate / update_steps,
                    "Debug/Critical_Transition_Sampler_Enabled": float(
                        self.student_critical_transition_balanced_sampling
                    ),
                    "Debug/Critical_Transition_Front_Sample_Fraction": (
                        total_front_sampling_fraction / update_steps
                    ),
                    "Debug/Critical_Transition_Rear_Sample_Fraction": (
                        total_rear_sampling_fraction / update_steps
                    ),
                    "Debug/Critical_Transition_Original_Sample_Fraction": (
                        total_original_sampling_fraction / update_steps
                    ),
                    "Debug/Critical_Transition_Front_Phase_Gate": total_front_phase_gate / update_steps,
                    "Debug/Critical_Transition_Rear_Phase_Gate": total_rear_phase_gate / update_steps,
                    "Debug/Post_Prior_Box_Gate": total_post_prior_gate / update_steps,
                    "Debug/Prior_Box_Loss_Weight": total_prior_box_weight / update_steps,
                    "Debug/Post_Prior_Box_Target_Delta": total_post_prior_box_delta / update_steps,
                    "Debug/Post_Prior_LowerMinusUpper_Box_Target": (
                        total_post_prior_lower_minus_upper_box / update_steps
                    ),
                    "Debug/Student_Post_Prior_Mode_Is_Highstep": float(
                        getattr(self, "student_post_prior_mode", "lateral") == "highstep"
                    ),
                    "Debug/Student_Actor_Adapt_Enabled": float(actor_adapt_enabled),
                    "Debug/Student_Box_Head_Adapt_Enabled": float(actor_adapt_enabled),
                    "Debug/Student_Rear_Hip_Head_Adapt_Enabled": float(rear_hip_adapt_enabled),
                    "Debug/Student_PPO_Adapt_Enabled": 0.0,
                    "Debug/Student_Distill_Update_Count": float(self.student_distill_update_count),
                    "Debug/Student_Warmup_Updates": float(warmup_updates),
                    "Debug/Student_VAE_Epochs": float(vae_epochs),
                    "Debug/Student_Actor_Grad_Norm": total_actor_grad_norm / update_steps,
                    "Debug/0707Exact_Main_Target_Is_Pre_Prior": float(pre_prior_0707_active),
                    "Debug/0707Exact_Warmup_Prior_Box_Loss_Zero": float(
                        pre_prior_0707_active and not actor_adapt_enabled
                        and total_prior_box_loss == 0.0
                    ),
                    "Debug/0707Exact_Warmup_Post_Prior_Main_Loss_Zero": float(
                        pre_prior_0707_active and not actor_adapt_enabled
                    ),
                    "Debug/Buffer_Step_Before_Clear": float(step_before_clear), # 证明清空前已满
                    "Debug/Buffer_Step_After_Clear": float(step_after_clear),   # 证明清空后归零
                    # ==========================================
                    # 【新增 Debug 证明】：记录 mu 越界的比例
                    # ==========================================
                    "Debug/Mu_Out_Of_Bounds_Ratio": getattr(model, "active_mu_oob_ratio", torch.tensor(0.0)).item(),
                }
                self.student_distill_update_count += 1
                if v15_active:
                    self._sync_v15_box_rows_to_actor()
                    self._assert_v15_frozen_unchanged()
                    self._validate_v15_optimizer_scope()
                
            return loss_dict
