import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

# 引入刚刚写的对称性工具
from robot_lab.tasks.locomotion.velocity.mdp.symmetry.arcdog_symmetry import ArcdogSymmetry

class SymmetricPPO(PPO):
    def __init__(self, policy, device="cpu", symmetry_coef=2.0, **kwargs):
        super().__init__(policy, device, **kwargs)
        self.symmetry_coef = symmetry_coef
        self.symmetry_tool = ArcdogSymmetry(device)
        self.mean_symmetry_loss = 0.0
        self.net = policy

    def update(self):
        """
        采用 PyTorch 原生 Hook 机制。
        直接在 actor 神经网络层拦截输入，避开所有 TensorDict 和复杂字典的解包问题。
        """
        intercepted_flat_obs = []
        
        # 1. 拦截器：挂载到 actor 神经网络上。
        # 当 rsl_rl 准备好纯正的 2D Tensor 并送入网络时，我们把它偷存下来。
        def actor_pre_hook(module, args):
            # args[0] 就是输入给 actor MLP 的展平 tensor
            intercepted_flat_obs.append(args[0].detach())
            
        # 注册 hook
        hook_handle = self.net.actor.register_forward_pre_hook(actor_pre_hook)
        
        original_step = self.optimizer.step
        symmetry_losses = []
        
        # 2. 拦截器：在原版 PPO 更新权重前，计算并注入对称性梯度
        def hooked_step(*args, **kwargs):
            if len(intercepted_flat_obs) > 0:
                # 拿到的绝对是标准的 2D Tensor
                flat_obs = intercepted_flat_obs.pop(0)
                
                # 获取镜像观测
                flat_mirrored_obs = self.symmetry_tool.mirror_obs(flat_obs)
                
                # 计算对称性 Loss
                mirrored_action_mean = self.net.actor(flat_mirrored_obs)
                with torch.no_grad():
                    original_action_mean = self.net.actor(flat_obs)
                    
                original_action_mirrored = self.symmetry_tool.mirror_action(original_action_mean)
                sym_loss = nn.functional.mse_loss(mirrored_action_mean, original_action_mirrored)
                
                # 反向传播：这会将对称性梯度【叠加】到原版 PPO 的梯度上
                (self.symmetry_coef * sym_loss).backward()
                
                # 重新裁剪梯度，防止叠加后梯度爆炸
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                
                symmetry_losses.append(sym_loss.item())
                
            # 执行真正的权重更新
            return original_step(*args, **kwargs)
            
        self.optimizer.step = hooked_step
        
        # 3. 运行原版 update！(它会自己完美处理所有复杂的 batch 解包和 TensorDict)
        try:
            result = super().update()
        finally:
            # 4. 无论成功失败，务必移除 hook 并还原原始方法，防止内存泄漏或影响 Rollout
            hook_handle.remove()
            self.optimizer.step = original_step
            intercepted_flat_obs.clear()
        
        # 5. 记录并返回 Loss
        if len(symmetry_losses) > 0:
            self.mean_symmetry_loss = sum(symmetry_losses) / len(symmetry_losses)
        else:
            self.mean_symmetry_loss = 0.0
            
        # 兼容新版 rsl_rl (返回 dict) 和老版 (返回 tuple)
        if isinstance(result, dict):
            result["symmetry_loss"] = self.mean_symmetry_loss
            
        return result


# ==========================================================
# 🌟 自定义 Runner
# ==========================================================
class SymmetricOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, env_info, config, log_dir=None, device="cpu"):
        self.env_cfg = config  
        
        extracted_symmetry_coef = 2.0 
        
        # 安全提取 symmetry_coef
        if isinstance(env_info, dict) and "algorithm" in env_info:
            if "symmetry_coef" in env_info["algorithm"]:
                extracted_symmetry_coef = env_info["algorithm"].pop("symmetry_coef")
        elif hasattr(env_info, "algorithm"):
            if "symmetry_coef" in env_info.algorithm:
                extracted_symmetry_coef = env_info.algorithm.symmetry_coef
                del env_info.algorithm["symmetry_coef"]

        # 1. 调用父类初始化
        super().__init__(env, env_info, log_dir, device)  
        
        # 2. 动态类替换与属性自适应
        net = getattr(self.alg, 'actor_critic', getattr(self.alg, 'policy', None))
        if net is None:
            raise AttributeError(f"无法在 PPO 实例中找到神经网络。当前 PPO 的属性有: {dir(self.alg)}")
        
        self.alg.__class__ = SymmetricPPO
        self.alg.net = net
        self.alg.symmetry_coef = extracted_symmetry_coef
        self.alg.symmetry_tool = ArcdogSymmetry(self.device)
        self.alg.mean_symmetry_loss = 0.0