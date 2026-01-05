import argparse
import sys
import os

# 1. 启动 App (必须最先执行)
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Train Improved MoE Policy (Noisy Top-K)")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoESimple-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# === Improved MoE 参数 ===
parser.add_argument("--num_experts", type=int, default=None, help="Number of experts")
parser.add_argument("--top_k", type=int, default=None, help="Top-K experts to activate")

# === Resume 参数 (新增) ===
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint filename (e.g. model_1400.pt)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入依赖
import torch
from datetime import datetime
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === 3. 关键：导入 RobustMoEActorCritic ===
# 尝试多个可能的路径导入
try:
    # 路径 1: 完整的任务配置路径 (根据你的报错调整)
    from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import RobustMoEActorCritic
    print("[Info] Imported RobustMoEActorCritic from agents config path.")
except ImportError:
    try:
        # 路径 2: 算法目录
        from robot_lab.algo.moe.moe_simple import RobustMoEActorCritic
        print("[Info] Imported RobustMoEActorCritic from algo path.")
    except ImportError:
        # 路径 3: 尝试当前目录 (fallback)
        try:
            sys.path.append(os.getcwd())
            from moe_simple import RobustMoEActorCritic
            print("[Info] Imported RobustMoEActorCritic from current directory.")
        except ImportError as e:
            raise ImportError(
                "Could not import 'RobustMoEActorCritic'. \n"
                "Please make sure you saved 'moe_improved.py' in one of the following locations:\n"
                "1. robot_lab/tasks/locomotion/velocity/config/wheeled/sirius_wheel/agents/\n"
                "2. robot_lab/algo/moe/\n"
                f"Error details: {e}"
            )

# === 4. 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

rsl_modules.RobustMoEActorCritic = RobustMoEActorCritic
runner_module.RobustMoEActorCritic = RobustMoEActorCritic
# 为了兼容某些硬编码检查，也注入旧名字（可选）
rsl_modules.SimpleMoEActorCritic = RobustMoEActorCritic 

# === 新增：注入自定义算法类 ===
# 确保 moesimple 模块中有 MoEPPO 类
try:
    from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import MoEPPO
    # 将 MoEPPO 注入到 on_policy_runner 模块的全局命名空间中
    # 这样 runner 内部执行 eval("MoEPPO") 时就能找到它
    runner_module.MoEPPO = MoEPPO
    print("[Info] Registered custom algorithm: MoEPPO")
except ImportError:
    print("[Warning] Could not import MoEPPO from moe_simple. Gate logging might fail if config uses it.")

def main():
    # 解析环境
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # 加载配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"):
        train_cfg_dict = train_cfg.to_dict()
    else:
        train_cfg_dict = train_cfg

    # === 配置覆写 ===
    print(f"\n[Info] Switching Policy Class to: RobustMoEActorCritic")
    train_cfg_dict["policy"]["class_name"] = "RobustMoEActorCritic"
    
    # 注入命令行参数
    if args.num_experts is not None:
        train_cfg_dict["policy"]["num_experts"] = args.num_experts
    if args.top_k is not None:
        train_cfg_dict["policy"]["top_k"] = args.top_k
    
    # 清理旧参数防止报错
    keys_to_remove = ["num_wheel_experts", "num_leg_experts", "wheel_indices", "leg_indices", 
                      "checkpoint_wheel", "checkpoint_leg", "freeze_experts"]
    for k in keys_to_remove:
        train_cfg_dict["policy"].pop(k, None)

    # 设置 Log 目录
    experiment_name = train_cfg_dict.get("experiment_name", "improved_moe_run")
    # 使用绝对路径以避免多重嵌套问题
    log_root_path = os.path.abspath(os.path.join("logs", "moe_training", experiment_name))
    
    # === Resume 逻辑处理 ===
    resume_path = None
    if args.resume:
        # 增强的路径搜索逻辑
        possible_roots = [
            log_root_path,                                      # logs/moe_training/{exp_name}
            os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name)), # logs/rsl_rl/{exp_name}
            os.path.abspath(os.path.join("logs", experiment_name))            # logs/{exp_name}
        ]
        
        found = False
        for search_root in possible_roots:
            if not os.path.exists(search_root):
                continue
            try:
                # get_checkpoint_path 能够处理 load_run 为空(找最新)或指定名称的情况
                resume_path = get_checkpoint_path(search_root, args.load_run, args.checkpoint)
                print(f"\n[Info] Found checkpoint at: {resume_path}")
                found = True
                break
            except Exception:
                continue
        
        if not found:
            print(f"\n[Error] Could not find run '{args.load_run}' with checkpoint '{args.checkpoint}'")
            print(f"Searched in:")
            for r in possible_roots:
                print(f"  - {r}")
            sys.exit(1)

    # 创建本次新的 Run 目录
    log_dir = os.path.join(log_root_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # 包装环境
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # 启动 Runner
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")

    # === 加载 Checkpoint ===
    if resume_path:
        runner.load(resume_path)

    # [Debug] 打印网络结构确认
    print("\n" + "="*80)
    print("[Debug] Active Policy Architecture:")
    try:
        model = getattr(runner.alg, "actor_critic", getattr(runner.alg, "policy", None))
        if model:
            print(model)
            if hasattr(model, "gating_net"):
                print(f"\n[Check] Gating Mechanism: {type(model.gating_net).__name__}")
                print(f"[Check] Top-K: {getattr(model.gating_net, 'top_k', 'Unknown')}")
    except Exception as e:
        print(f"Error inspecting model: {e}")
    print("="*80 + "\n")

    runner.learn(num_learning_iterations=train_cfg_dict["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()