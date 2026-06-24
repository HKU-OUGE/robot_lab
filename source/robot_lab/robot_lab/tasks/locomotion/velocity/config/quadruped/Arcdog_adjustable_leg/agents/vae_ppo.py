import torch
import torch.nn as nn
import copy
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO

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
        
        vae_latent_dim = kwargs.pop("vae_latent_dim", 64)
        vae_hidden_dims = kwargs.pop("vae_hidden_dims", [256, 128])
        
        policy_keys = obs_groups.get("policy", ["policy"])
        estimator_keys = obs_groups.get("estimator", ["estimator"])
        critic_keys = obs_groups.get("critic", ["critic"])

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

    def __init__(self, *args, **kwargs):
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        self.distill_stage = getattr(self.policy, "distill_stage", 1)
        self._teacher_actor_synced = self.distill_stage != 2
        
        # 【核心修复 2】：为 VAE 创建完全独立的优化器
        # 避免 VAE 的巨大 Loss 污染 Actor/Critic 的 Adam 动量状态
        if hasattr(self.policy, 'estimator') and self.policy.estimator is not None:
            if self.distill_stage == 2:
                # Stage 2 keeps a frozen teacher actor as the movement anchor.
                # The student first learns to reproduce the teacher's blind-motion
                # behavior through the VAE. Only after a warmup do we allow a small
                # actor update for the side-step box prior.
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

    def _sync_teacher_actor_from_policy_once(self):
        """Sync frozen teacher actor after runner.load() has restored checkpoint weights."""
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

    def _configure_student_actor_trainable(self, train_box_head: bool) -> None:
        """Freeze the actor except, optionally, the final-layer rows for box actions."""
        for param in self.policy.actor.parameters():
            param.requires_grad = False

        if not train_box_head:
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

        if getattr(self, "_student_box_head_hook_layer", None) is last_linear:
            return

        for handle in getattr(self, "_student_box_head_grad_hooks", []):
            handle.remove()

        box_rows = min(4, last_linear.out_features)
        weight_mask = torch.zeros_like(last_linear.weight)
        weight_mask[-box_rows:, :] = 1.0
        bias_mask = None
        if last_linear.bias is not None:
            bias_mask = torch.zeros_like(last_linear.bias)
            bias_mask[-box_rows:] = 1.0

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
        self._student_box_head_hook_layer = last_linear

    def update(self):
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
            # After warmup, only the final-layer box action rows can adapt; no PPO is
            # opened here because full actor/RL updates polluted the teacher gait.
            # ==========================================
            model = self.policy
            self._sync_teacher_actor_from_policy_once()
            if not hasattr(self, "student_distill_update_count"):
                self.student_distill_update_count = 0
            warmup_updates = getattr(self, "student_actor_warmup_updates", 800)
            actor_adapt_enabled = self.student_distill_update_count >= warmup_updates
            self._configure_student_actor_trainable(train_box_head=actor_adapt_enabled)
            for param in model.critic.parameters():
                param.requires_grad = False
            
            if model.estimator is not None and self.num_learning_epochs > 0:
                obs_storage = self.storage.observations
                
                if hasattr(obs_storage, "keys"):
                    policy_obs_full = obs_storage[model.policy_keys[0]].flatten(0, 1)
                    est_input_full = obs_storage[model.estimator_keys[0]].flatten(0, 1)
                    critic_obs_full = obs_storage[model.critic_keys[0]].flatten(0, 1)
                else:
                    obs_batch = obs_storage.flatten(0, 1)
                    policy_obs_full = obs_batch
                    est_input_full = obs_batch
                    critic_obs_full = obs_batch

                if policy_obs_full.shape[-1] >= model.policy_obs_dim + model.vae_latent_dim:
                    policy_obs_full = policy_obs_full[..., :model.policy_obs_dim]
                
                # 假设 critic 组的前 3 维就是真实的 base_lin_vel
                target_vel_full = critic_obs_full[..., 0:3] 
                # 获取 Teacher 的目标 (Teacher Latent)
                with torch.no_grad():
                    # <=== STRICT FIX 4: 蒸馏时，Teacher 依然只吃切除速度后的观测
                    target_latent_full = model.priv_encoder(critic_obs_full[..., 3:])
                
                num_samples = est_input_full.shape[0]
                batch_size = num_samples // self.num_mini_batches
                
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
                update_steps = 0

                # 【核心修复 3】：使用 Epoch 和 Mini-batch 训练 VAE
                # 【新增优化】：解耦 VAE 与 PPO 的 Epoch，强制 VAE 每次 PPO 更新只训练 1-2 个 Epoch
                # 阶段2蒸馏可以适当增加 epoch，这里设为 5 以加速蒸馏
                vae_epochs = getattr(self, "student_vae_epochs", 4)
                for epoch in range(vae_epochs):
                    # 打乱数据
                    indices = torch.randperm(num_samples, device=est_input_full.device)
                    
                    for i in range(self.num_mini_batches):
                        start = i * batch_size
                        end = (i + 1) * batch_size
                        batch_idx = indices[start:end]
                        
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

                        # 动作蒸馏：目标是 teacher actor 经过任务对应 box prior 后的 raw-equivalent action。
                        safe_mu = torch.clamp(mu, min=-1.0, max=1.0)
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

                        student_action = model.actor(torch.cat([policy_obs, safe_mu], dim=-1))
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
                        teacher_action_error = torch.square(student_action - teacher_action)
                        if teacher_action_error.shape[-1] >= 16:
                            non_box_loss = torch.mean(teacher_action_error[:, :12], dim=-1)
                            raw_box_loss = torch.mean(teacher_action_error[:, -4:], dim=-1)
                            # Keep gait as the dominant target. Box raw-action imitation is relaxed only
                            # on clear post-prior samples after the isolated box head starts adapting.
                            if actor_adapt_enabled:
                                box_raw_weight = torch.clamp(1.0 - post_prior_gate, min=0.0, max=1.0)
                            else:
                                box_raw_weight = torch.ones_like(post_prior_gate)
                            per_sample_teacher_action_loss = 2.0 * non_box_loss + 0.5 * raw_box_loss * box_raw_weight
                        else:
                            per_sample_teacher_action_loss = torch.mean(teacher_action_error, dim=-1)
                        moving_weight = 1.0 + torch.clamp(command_speed / prior_fade_speed, min=0.0, max=1.0)
                        teacher_action_loss = torch.mean(per_sample_teacher_action_loss * moving_weight)

                        # Post-prior loss is now a conditional box-only add-on. It is strong for static/low-speed
                        # prior samples, weak for moving samples, and disabled during the initial actor warmup.
                        if student_action.shape[-1] >= 16:
                            student_action_for_prior = model.actor(torch.cat([policy_obs, safe_mu.detach()], dim=-1))
                            per_sample_prior_box_loss = torch.mean(
                                torch.square(student_action_for_prior[:, -4:] - teacher_action_target[:, -4:]), dim=-1
                            )
                            prior_box_weight = post_prior_gate * (
                                0.15 + 1.25 * low_speed_gate + 1.0 * is_static + 0.75 * tilt_scale
                            )
                            if not actor_adapt_enabled:
                                prior_box_weight = torch.zeros_like(prior_box_weight)
                            prior_box_loss = self._weighted_mean(per_sample_prior_box_loss, prior_box_weight)
                            box_action_loss = torch.mean(per_sample_prior_box_loss)
                        else:
                            prior_box_weight = torch.zeros_like(command_speed)
                            prior_box_loss = torch.zeros((), device=student_action.device)
                            box_action_loss = torch.mean(torch.square(student_action - teacher_action_target))

                        action_loss = teacher_action_loss + prior_box_loss
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
                            + getattr(self, "student_recon_loss_coef", 0.5) * recon_loss
                            + beta * kl_loss
                        )
                        
                        # 反向传播并使用 VAE 专属优化器更新
                        vae_loss.backward()
                        actor_grad_norm = sum(
                            p.grad.detach().norm().item() for p in model.actor.parameters() if p.grad is not None
                        )
                        distill_params = list(model.estimator.parameters()) + [
                            param for param in model.actor.parameters() if param.requires_grad
                        ]
                        nn.utils.clip_grad_norm_(distill_params, self.max_grad_norm)
                        self.vae_optimizer.step()
                        
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
                        update_steps += 1
                
                # ==========================================
                # 【新增修改】：记录清空前的 buffer 指针位置，用于证明修复有效
                # ==========================================
                step_before_clear = self.storage.step
                
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
                    "Debug/Student_PPO_Adapt_Enabled": 0.0,
                    "Debug/Student_Distill_Update_Count": float(self.student_distill_update_count),
                    "Debug/Student_Warmup_Updates": float(warmup_updates),
                    "Debug/Student_VAE_Epochs": float(vae_epochs),
                    "Debug/Student_Actor_Grad_Norm": total_actor_grad_norm / update_steps,
                    "Debug/Buffer_Step_Before_Clear": float(step_before_clear), # 证明清空前已满
                    "Debug/Buffer_Step_After_Clear": float(step_after_clear),   # 证明清空后归零
                    # ==========================================
                    # 【新增 Debug 证明】：记录 mu 越界的比例
                    # ==========================================
                    "Debug/Mu_Out_Of_Bounds_Ratio": getattr(model, "active_mu_oob_ratio", torch.tensor(0.0)).item(),
                }
                self.student_distill_update_count += 1
                
            return loss_dict
