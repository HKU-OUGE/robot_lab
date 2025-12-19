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

# === [Fix] H-MoE 专用参数 ===
# 将默认值设为 None，避免意外覆盖配置文件中的设置
parser.add_argument("--num_wheel_experts", type=int, default=None, help="Number of wheel experts (overrides config if set)")
parser.add_argument("--num_leg_experts", type=int, default=None, help="Number of leg experts (overrides config if set)")

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

# 引入 RSL-RL 环境包装器
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === 关键：导入新的 H-MoE 策略类 ===
from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import SharedBackboneMoEActorCritic

# === 核心修正：将自定义类注入到 rsl_rl 的命名空间中 ===
import rsl_rl.modules as rsl_modules
rsl_modules.SharedBackboneMoEActorCritic = SharedBackboneMoEActorCritic

import rsl_rl.runners.on_policy_runner as runner_module
runner_module.SharedBackboneMoEActorCritic = SharedBackboneMoEActorCritic

def main():
    # 解析环境配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # 加载 PPO 配置 (从 moe_terrain.py 中加载)
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # === 配置处理 ===
    if hasattr(train_cfg, "to_dict"):
        train_cfg_dict = train_cfg.to_dict()
    else:
        train_cfg_dict = train_cfg

    # 2. 在字典中注入 H-MoE 参数和类名
    train_cfg_dict["policy"]["class_name"] = "SharedBackboneMoEActorCritic"
    
    # [Fix] 仅当命令行显式指定时才覆盖配置
    if args.num_wheel_experts is not None:
        print(f"[Info] Overriding num_wheel_experts from command line: {args.num_wheel_experts}")
        train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    
    if args.num_leg_experts is not None:
        print(f"[Info] Overriding num_leg_experts from command line: {args.num_leg_experts}")
        train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    # [Clean] 清理旧参数
    train_cfg_dict["policy"].pop("checkpoint_wheel", None)
    train_cfg_dict["policy"].pop("checkpoint_leg", None)
    train_cfg_dict["policy"].pop("freeze_experts", None)

    # 设置 Log
    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    log_dir = os.path.join("logs", "moe_training", experiment_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # 使用 RSLRlVecEnvWrapper 包装环境
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # === 实例化 Runner ===
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
        model = getattr(runner.alg, "actor_critic", None)
        if model is None:
            model = getattr(runner.alg, "policy", None)
        
        if model is not None:
            print(model)
            # 简单检查一下专家数量是否正确
            n_wheel = len(model.actor_wheel_experts)
            n_leg = len(model.actor_leg_experts)
            print(f"\n[Check] Wheel Experts: {n_wheel}, Leg Experts: {n_leg}")
        else:
            inference_policy = runner.get_inference_policy(device="cuda:0")
            if hasattr(inference_policy, "__self__"):
                print(inference_policy.__self__)
            else:
                print("Could not retrieve model instance via inference policy introspection.")
    except Exception as e:
        print(f"Error printing architecture: {e}")
    print("="*80 + "\n")

    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()