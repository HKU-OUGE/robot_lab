import argparse
import sys
import os
import glob
from isaaclab.app import AppLauncher

# === 1. 启动 App (必须最先执行) ===
parser = argparse.ArgumentParser(description="Play/Evaluate H-MoE Policy")

# 标准参数
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments")
parser.add_argument("--seed", type=int, default=None, help="Random seed")

# [Fix] 设为 None，仅在显式指定时覆盖 config
parser.add_argument("--num_wheel_experts", type=int, default=None, help="Number of wheel experts")
parser.add_argument("--num_leg_experts", type=int, default=None, help="Number of leg experts")

# 加载参数
parser.add_argument("--load_run", type=str, default=None, help="Name of the experiment folder (default: latest run)")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint file pattern (default: model_*.pt)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入其余库
import torch
import torch.nn.functional as F
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === MoE 类注入 ===
from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import SharedBackboneMoEActorCritic
import rsl_rl.modules as rsl_modules
rsl_modules.SharedBackboneMoEActorCritic = SharedBackboneMoEActorCritic
import rsl_rl.runners.on_policy_runner as runner_module
runner_module.SharedBackboneMoEActorCritic = SharedBackboneMoEActorCritic

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    """智能解析模型路径"""
    run_dir = None
    if run_name_or_path is None:
        print(f"[Info] No run specified. Searching for latest run in: {root_log_dir}")
        if not os.path.exists(root_log_dir):
             raise FileNotFoundError(f"Log directory not found: {root_log_dir}")
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if not all_runs:
            raise FileNotFoundError(f"No runs found in {root_log_dir}")
        all_runs.sort(key=os.path.getmtime)
        run_dir = all_runs[-1]
        print(f"[Info] Auto-selected latest run: {os.path.basename(run_dir)}")
    elif os.path.isabs(run_name_or_path):
        run_dir = run_name_or_path
    else:
        potential_path = os.path.join(root_log_dir, run_name_or_path)
        run_dir = potential_path if os.path.exists(potential_path) else potential_path # Fallback

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    search_pattern = os.path.join(run_dir, checkpoint_pattern)
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint matching '{checkpoint_pattern}' found in {run_dir}")
    
    def extract_iter(f):
        try:
            return int(f.split("_")[-1].split(".")[0])
        except:
            return os.path.getmtime(f)
            
    files.sort(key=extract_iter)
    return files[-1], run_dir

def main():
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"):
        train_cfg_dict = train_cfg.to_dict()
    else:
        train_cfg_dict = train_cfg

    train_cfg_dict["policy"]["class_name"] = "SharedBackboneMoEActorCritic"
    
    # [Fix] 仅在显式指定时覆盖
    if args.num_wheel_experts is not None:
        train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts is not None:
        train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    root_log_dir = os.path.join("logs", "moe_training", experiment_name)

    try:
        model_path, log_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
        print(f"\n[Success] Loading model from: {model_path}")
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    
    if hasattr(policy, "__self__"):
        model_instance = policy.__self__
    else:
        model_instance = policy

    monitor_data = {}
    def get_activation(name):
        def hook(model, input, output):
            monitor_data[name] = output.detach()
        return hook

    print(f"\n[Info] Registering hooks on: {type(model_instance).__name__}")
    try:
        if hasattr(model_instance, "mode_gate"):
            model_instance.mode_gate.register_forward_hook(get_activation("Mode_Gate"))
        if hasattr(model_instance, "wheel_gate"):
            model_instance.wheel_gate.register_forward_hook(get_activation("Wheel_Gate"))
        if hasattr(model_instance, "leg_gate"):
            model_instance.leg_gate.register_forward_hook(get_activation("Leg_Gate"))
    except Exception as e:
        print(f"[Error] Failed to register hooks: {e}")

    # === [State] 用于记录上次打印了多少行，以便回退 ===
    last_printed_lines = 0

    def visualize_weights(obs_idx=0):
        nonlocal last_printed_lines
        
        # 收集所有要打印的行
        lines = []
        lines.append("="*20 + " H-MoE Realtime Weights " + "="*20)

        # --- 1. Mode Gate (Wheel vs Leg) ---
        if "Mode_Gate" in monitor_data:
            logits = monitor_data["Mode_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0) # [2]
            
            p_wheel = probs[0].item()
            p_leg = probs[1].item()
            
            bar_len = 20
            w_bar = int(p_wheel * bar_len)
            l_bar = int(p_leg * bar_len)
            
            c_reset = "\033[0m"
            c_wheel = "\033[92m" if p_wheel > p_leg else "\033[90m" 
            c_leg   = "\033[92m" if p_leg > p_wheel else "\033[90m"
            
            lines.append(f"Top Mode:  {c_wheel}Wheel {p_wheel:.2f}{c_reset} vs {c_leg}Leg {p_leg:.2f}{c_reset}")
            lines.append(f"  Wheel: |{c_wheel}{'█'*w_bar:<{bar_len}}{c_reset}|")
            lines.append(f"  Leg:   |{c_leg}{'█'*l_bar:<{bar_len}}{c_reset}|")
        else:
            lines.append("[Wait] Mode Gate data missing.")

        # --- 2. Wheel Experts (Routing) ---
        if "Wheel_Gate" in monitor_data:
            logits = monitor_data["Wheel_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0)
            lines.append(f"Wheel Experts:")
            for i, p in enumerate(probs):
                val = p.item()
                bar = int(val * 20)
                if val > 0.1: # 仅高亮活跃的专家
                    lines.append(f"  Exp {i}: \033[93m{val:.2f}\033[0m |{'#'*bar:<20}|")
                else:
                    lines.append(f"  Exp {i}: {val:.2f} |{'#'*bar:<20}|")

        # --- 3. Leg Experts (Routing) ---
        if "Leg_Gate" in monitor_data:
            logits = monitor_data["Leg_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0)
            lines.append(f"Leg Experts:")
            for i, p in enumerate(probs):
                val = p.item()
                bar = int(val * 20)
                if val > 0.1:
                    lines.append(f"  Exp {i}: \033[96m{val:.2f}\033[0m |{'#'*bar:<20}|")
                else:
                    lines.append(f"  Exp {i}: {val:.2f} |{'#'*bar:<20}|")
        lines.append("="*60)
        
        # === 原地更新逻辑 ===
        # 1. 如果上次有打印内容，先回退光标并清除下方
        if last_printed_lines > 0:
            # \033[nA: 光标上移 n 行
            # \033[J: 清除光标及以下的内容
            sys.stdout.write(f"\033[{last_printed_lines}A")
            sys.stdout.write("\033[J")
        
        # 2. 打印新内容
        print("\n".join(lines))
        
        # 3. 更新行数计数
        last_printed_lines = len(lines)
    
    obs, _ = env.reset()
    # 打印个空行让光标有起始位置，避免吞掉前面的Log
    print("\nStarting inference loop...")
    
    step_counter = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step_counter += 1
            if step_counter % 10 == 0:
                visualize_weights(obs_idx=0)
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()