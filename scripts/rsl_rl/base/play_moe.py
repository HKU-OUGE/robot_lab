import argparse
import sys
import os
import glob
from isaaclab.app import AppLauncher
# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# === 1. 启动 App ===
parser = argparse.ArgumentParser(description="Play H-MoE Policy")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoE-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
# H-MoE Params
parser.add_argument("--num_wheel_experts", type=int, default=None)
parser.add_argument("--num_leg_experts", type=int, default=None)
# Checkpoint
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_*.pt")
# Keyboard
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === 新增导入：Keyboard & Utils ===
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
import rsl_rl_utils  # 确保这个模块在 python path 中，通常位于 rsl_rl 仓库根目录

# === 关键：导入自定义模块 ===
try:
    sys.path.append(os.getcwd())
    from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
    print("[Info] Imported H-MoE classes from current directory.")
except ImportError:
    try:
        from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import SplitMoEActorCritic, SplitMoEPPO
    except ImportError:
        raise ImportError("Could not import classes from moe_terrain.py")

# === 注入到 RSL-RL ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

rsl_modules.SplitMoEActorCritic = SplitMoEActorCritic
runner_module.SplitMoEActorCritic = SplitMoEActorCritic
rsl_modules.SharedBackboneMoEActorCritic = SplitMoEActorCritic 
runner_module.SplitMoEPPO = SplitMoEPPO

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    run_dir = None
    if run_name_or_path is None:
        if not os.path.exists(root_log_dir):
             parent = os.path.dirname(root_log_dir)
             if os.path.exists(parent): root_log_dir = parent
             
        if not os.path.exists(root_log_dir):
             raise FileNotFoundError(f"Log dir not found: {root_log_dir}")
             
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if not all_runs: raise FileNotFoundError("No runs found")
        all_runs.sort(key=os.path.getmtime)
        run_dir = all_runs[-1]
        print(f"[Info] Auto-selected run: {os.path.basename(run_dir)}")
    elif os.path.isabs(run_name_or_path):
        run_dir = run_name_or_path
    else:
        run_dir = os.path.join(root_log_dir, run_name_or_path)

    search_pattern = os.path.join(run_dir, checkpoint_pattern)
    files = glob.glob(search_pattern)
    if not files: raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    
    files.sort(key=os.path.getmtime)
    return files[-1], run_dir

def main():
    # 1. 解析基础配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)

    # === Keyboard 配置逻辑 (仿照 play.py) ===
    controller = None
    if args.keyboard:
        print("[Info] Enabling Keyboard Control")
        # 强制单环境
        env_cfg.scene.num_envs = 1
        # 去除超时
        env_cfg.terminations.time_out = None
        # 启用命令可视化
        env_cfg.commands.base_velocity.debug_vis = True

        # 配置键盘控制器
        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        controller = Se2Keyboard(kb_cfg)

        # 覆盖 Observation 中的 velocity_commands
        # 注意：这里假设 observation group 名字是 "policy"，且 velocity_commands 在其中
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )

    # 2. 创建环境
    env = gym.make(args.task, cfg=env_cfg)

    # === Keyboard Reset 回调 (必须在 env 创建后定义) ===
    if args.keyboard and controller is not None:
        def reset_env_callback():
            print("[INFO] 'R' key pressed: Resetting environment.")
            nonlocal obs
            # === [Debug] Checking Normalizer Stats (Corrected) ===
            print("\n[Debug] Checking Normalizer Stats:")
            if hasattr(model_instance, "actor_obs_normalizer"):
                norm = model_instance.actor_obs_normalizer
                print(f"  - Count: {norm.count}")
                if norm.count > 0:
                    # [修改] 使用 .mean 和 .var
                    print(f"  - Mean (first 5): {norm.mean[:5].cpu().numpy()}") 
                    print(f"  - Var  (first 5): {norm._var[:5].cpu().numpy()}")
            else:
                print("⚠️ [WARNING] No actor_obs_normalizer found in policy!")
            obs, _ = env.reset()
        
        # 注册回调
        controller.add_callback("R", reset_env_callback)

    # 3. 加载训练配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg

    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"

    if args.num_wheel_experts: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    
    search_paths = [
        os.path.join("logs", "moe_training", experiment_name),
        os.path.join("logs", "rsl_rl", experiment_name),
        os.path.join("logs", experiment_name)
    ]
    root_log_dir = search_paths[0]
    for p in search_paths:
        if os.path.exists(p): 
            root_log_dir = p
            break

    try:
        model_path, log_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
        print(f"\n[Success] Loading model from: {model_path}")
    except Exception as e:
        print(f"\n[Error] {e}")
        sys.exit(1)

    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    # === 检测是否启用了 Hard Mode ===
    use_hard_mode = getattr(model_instance, "use_hard_mode_switch", False)
    print(f"\n[Info] Mode Gate Strategy: {'HARD (Top-1)' if use_hard_mode else 'SOFT (Softmax)'}")

    # === 可视化 Hooks ===
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            # output 是 logits
            monitor_data[name] = output.detach()
        return _hook

    if hasattr(model_instance, "mode_gate"): model_instance.mode_gate.register_forward_hook(hook_fn("Mode"))
    if hasattr(model_instance, "wheel_gate"): model_instance.wheel_gate.register_forward_hook(hook_fn("Wheel"))
    if hasattr(model_instance, "leg_gate"): model_instance.leg_gate.register_forward_hook(hook_fn("Leg"))

    # === 辅助打印函数 (对齐 play_moesimple 风格) ===
    def print_expert_bars(probs, expert_names=None):
        lines = []
        for i, p in enumerate(probs):
            val = p.item()
            bar_len = int(val * 40)
            bar = '█' * bar_len
            
            # 状态判定
            if val > 0.9:
                color = "\033[92m" # Green (Dominant)
                status = "DOMINANT"
            elif val > 0.1:
                color = "\033[96m" # Cyan (Active)
                status = "ACTIVE"
            elif val > 0.01:
                color = "\033[93m" # Yellow (Weak)
                status = " WEAK "
            else:
                color = "\033[90m" # Grey (Dead)
                status = " DEAD "
            
            name = expert_names[i] if expert_names else f"Exp {i}"
            lines.append(f"  {name:<8}: {color}{val:.3f} | {bar:<40} | {status}\033[0m")
        return lines

    last_printed_lines = 0
    def visualize(obs_idx=0):
        nonlocal last_printed_lines
        lines = ["="*20 + " H-MoE Activation " + "="*20]
        
        # 1. Mode Gate
        if "Mode" in monitor_data:
            logits = monitor_data["Mode"]
            if logits.ndim == 3: logits = logits[-1]
            
            # === 核心逻辑修改：匹配训练时的 Hard/Soft 行为 ===
            probs = F.softmax(logits[obs_idx], dim=0)
            
            if use_hard_mode:
                # 如果是 Hard Mode，可视化也强制显示 Top-1
                # 找到最大值的索引
                idx = probs.argmax()
                probs_vis = torch.zeros_like(probs)
                probs_vis[idx] = 1.0
                mode_str = "HARD (Top-1)"
            else:
                probs_vis = probs
                mode_str = "SOFT"

            lines.append(f"Top Mode ({mode_str}):")
            lines.extend(print_expert_bars(probs_vis, expert_names=["Wheel", "Leg"]))
        else:
            lines.append("Waiting for Mode data...")

        lines.append("-" * 60)

        # 2. Sub-Gates (始终是 Softmax)
        for name in ["Wheel", "Leg"]:
            if name in monitor_data:
                logits = monitor_data[name]
                if logits.ndim == 3: logits = logits[-1]
                
                probs = F.softmax(logits[obs_idx], dim=0)
                lines.append(f"{name} Experts:")
                lines.extend(print_expert_bars(probs))
        
        lines.append("="*60)

        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        print("\n".join(lines))
        last_printed_lines = len(lines)
    # === [Debug] Checking Normalizer Stats (Corrected) ===
    print("\n[Debug] Checking Normalizer Stats:")
    if hasattr(model_instance, "actor_obs_normalizer"):
        norm = model_instance.actor_obs_normalizer
        print(f"  - Count: {norm.count}")
        if norm.count > 0:
            # [修改] 使用 .mean 和 .var
            print(f"  - Mean (first 5): {norm.mean[:5].cpu().numpy()}") 
            print(f"  - Var  (first 5): {norm._var[:5].cpu().numpy()}")
    else:
        print("⚠️ [WARNING] No actor_obs_normalizer found in policy!")
    obs, _ = env.reset()
    print("\nStarting inference...")
    
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            if step % 10 == 0:
                visualize(obs_idx=0)
            
            # === Keyboard Camera Follow ===
            if args.keyboard:
                rsl_rl_utils.camera_follow(env)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()