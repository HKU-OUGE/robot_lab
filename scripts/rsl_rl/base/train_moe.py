import argparse
import sys
import os

# 1. 启动 App (必须最先执行，且在 import torch 之前)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train H-MoE Policy (End-to-End)")
# 添加标准参数
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
# parser.add_argument("--headless", action="store_true", default=False, help="Force display off")

# === [New] H-MoE 专用参数 ===
# 不再需要 checkpoints，因为是端到端训练
parser.add_argument("--num_wheel_experts", type=int, default=3, help="Number of wheel experts")
parser.add_argument("--num_leg_experts", type=int, default=3, help="Number of leg experts")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入其余库 (必须在 simulation_app 启动之后！)
import torch
from datetime import datetime
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# [Fix] 引入 RSL-RL 环境包装器
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === 关键：导入新的 H-MoE 策略类 ===
from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import HierarchicalMoEActorCritic

# === [Fix] 核心修正：将自定义类注入到 rsl_rl 的命名空间中 ===
# 1. 注入到 rsl_rl.modules
import rsl_rl.modules as rsl_modules
rsl_modules.HierarchicalMoEActorCritic = HierarchicalMoEActorCritic

# 2. 注入到 OnPolicyRunner 所在的模块命名空间 (解决 NameError)
import rsl_rl.runners.on_policy_runner as runner_module
runner_module.HierarchicalMoEActorCritic = HierarchicalMoEActorCritic

def main():
    # 解析环境配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # 加载 PPO 配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # === [Fix] 核心修正：配置处理 ===
    if hasattr(train_cfg, "to_dict"):
        train_cfg_dict = train_cfg.to_dict()
    else:
        train_cfg_dict = train_cfg

    # 2. 在字典中注入 H-MoE 参数和类名
    train_cfg_dict["policy"]["class_name"] = "HierarchicalMoEActorCritic"
    
    # [Change] 注入新的架构参数
    train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    # [Clean] 清理旧参数 (防止报错或混淆)
    train_cfg_dict["policy"].pop("checkpoint_wheel", None)
    train_cfg_dict["policy"].pop("checkpoint_leg", None)
    train_cfg_dict["policy"].pop("freeze_experts", None)

    # 设置 Log
    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    log_dir = os.path.join("logs", "moe_training", experiment_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # [Fix] 使用 RSLRlVecEnvWrapper 包装环境
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # === [Fix] 实例化 Runner ===
    runner = OnPolicyRunner(
        env,
        train_cfg_dict,
        log_dir=log_dir,
        device="cuda:0"
    )

    # === [Debug] 打印完整的实际网络架构 ===
    print("\n" + "="*80)
    print("[Debug] Full Policy Architecture (Actual Runtime Model):")
    try:
        # 尝试直接获取模型，优先尝试 'actor_critic'，然后是 'policy'
        model = getattr(runner.alg, "actor_critic", None)
        if model is None:
            model = getattr(runner.alg, "policy", None)
        
        if model is not None:
            print(model)
        else:
            # Fallback: 使用 get_inference_policy，但保持在 GPU 上以避免副作用
            # 注意：这里使用 "cuda:0" 而不是 "cpu"，防止将训练权重移动到 CPU 导致后续 learn() 失败
            inference_policy = runner.get_inference_policy(device="cuda:0")
            if hasattr(inference_policy, "__self__"):
                print(inference_policy.__self__)
            else:
                print("Could not retrieve model instance via inference policy introspection.")
    except Exception as e:
        print(f"Error printing architecture: {e}")
        # 打印 alg 的属性以便调试
        print(f"Available attributes in runner.alg: {dir(runner.alg)}")
    print("="*80 + "\n")

    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()