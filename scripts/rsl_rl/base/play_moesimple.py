import argparse
import sys
import os
import glob
import time  # 导入 time
from isaaclab.app import AppLauncher
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === 1. 启动 App ===
parser = argparse.ArgumentParser(description="Play Improved MoE Policy")
parser.add_argument("--task", type=str, default="RobotLab-Isaac-Velocity-SiriusW-MoESimple-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
# MoE Params
parser.add_argument("--num_experts", type=int, default=None)
parser.add_argument("--top_k", type=int, default=None)
# Checkpoint
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder (e.g. 'sirius_moe_run')")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint pattern")
# Play Options (Added)
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard control")
parser.add_argument("--video", action="store_true", default=False, help="Record videos (dummy arg for compatibility)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 导入 Isaac Lab 和 RSL-RL 工具
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
import rsl_rl_utils  # 确保 rsl_rl_utils 在 python path 中 (通常在 IsaacLab 环境中可用)

# === 导入自定义模块 (Policy 和 Algorithm) ===
try:
    sys.path.append(os.getcwd())
    from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import RobustMoEActorCritic, MoEPPO
    print("[Info] Successfully imported RobustMoEActorCritic and MoEPPO from current directory.")
except ImportError as e:
    print(f"[Warning] Local import failed: {e}")
    try:
        from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import RobustMoEActorCritic, MoEPPO
        print("[Info] Imported from robot_lab.algo.moe.moe_simple")
    except ImportError:
        # Fallback provided for RobustMoEActorCritic if imports fail (Assuming user environment might be tricky)
        print("[Warning] Could not import RobustMoEActorCritic. Ensure moe_simple.py is accessible.")

# === 注入到 RSL-RL 运行环境 ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

try:
    rsl_modules.RobustMoEActorCritic = RobustMoEActorCritic
    runner_module.RobustMoEActorCritic = RobustMoEActorCritic
    runner_module.MoEPPO = MoEPPO 
    rsl_modules.SimpleMoEActorCritic = RobustMoEActorCritic 
except NameError:
    pass # If imports failed above

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    run_dir = None
    if run_name_or_path is None:
        if not os.path.exists(root_log_dir): 
            parent_dir = os.path.dirname(root_log_dir)
            if os.path.exists(parent_dir):
                print(f"[Info] Log dir {root_log_dir} not found, searching in {parent_dir}")
                root_log_dir = parent_dir
        if not os.path.exists(root_log_dir): 
            raise FileNotFoundError(f"Log dir not found: {root_log_dir}")
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if not all_runs: 
            raise FileNotFoundError(f"No runs found in {root_log_dir}")
        all_runs.sort(key=os.path.getmtime)
        run_dir = all_runs[-1]
        print(f"[Info] Auto-selected latest run: {os.path.basename(run_dir)}")
    elif os.path.isabs(run_name_or_path):
        run_dir = run_name_or_path
    else:
        run_dir = os.path.join(root_log_dir, run_name_or_path)

    search_pattern = os.path.join(run_dir, checkpoint_pattern)
    files = glob.glob(search_pattern)
    if not files: 
        raise FileNotFoundError(f"No checkpoint found matching '{checkpoint_pattern}' in {run_dir}")
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[-1], run_dir

def main():
    # 1. 解析环境配置
    # 如果启用了 keyboard，强制单环境
    num_envs = 1 if args.keyboard else args.num_envs
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=num_envs)

    # === Keyboard Controller Setup (Pre-Env) ===
    controller = None
    if args.keyboard:
        print("[Info] Setting up Keyboard Controller...")
        # 禁用超时，防止驾驶中途重置
        if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        
        # 启用速度指令可视化
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = True
            
            # 配置键盘映射灵敏度
            # 尝试从配置中读取范围，如果不存在则使用默认值
            ranges = env_cfg.commands.base_velocity.ranges
            kb_cfg = Se2KeyboardCfg(
                v_x_sensitivity=float(getattr(ranges, "lin_vel_x", [0.0, 1.0])[1]),
                v_y_sensitivity=float(getattr(ranges, "lin_vel_y", [0.0, 1.0])[1]),
                omega_z_sensitivity=float(getattr(ranges, "ang_vel_z", [0.0, 1.0])[1]),
            )
            controller = Se2Keyboard(kb_cfg)

            # 替换观测中的指令项，改为从 controller 读取
            # 注意：这里的 'policy' 和 'velocity_commands' 需要与你的 EnvCfg 结构匹配
            if hasattr(env_cfg.observations, "policy"):
                env_cfg.observations.policy.velocity_commands = ObsTerm(
                    func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
                )
    
    # 创建环境
    env = gym.make(args.task, cfg=env_cfg)

    # === Keyboard Callback Setup (Post-Env) ===
    if args.keyboard and controller is not None:
        def reset_env_callback():
            print("[INFO] 'R' key pressed: Resetting environment.")
            env.reset()
        controller.add_callback("R", reset_env_callback)

    # 2. 加载训练配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg

    # === 强制覆写 Policy 配置 ===
    train_cfg_dict["policy"]["class_name"] = "RobustMoEActorCritic"
    if args.num_experts: train_cfg_dict["policy"]["num_experts"] = args.num_experts
    if args.top_k: train_cfg_dict["policy"]["top_k"] = args.top_k
    
    # 清理参数
    for k in ["num_wheel_experts", "num_leg_experts", "wheel_indices", "leg_indices", 
              "checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # 3. 确定 Checkpoint 路径
    experiment_name = train_cfg_dict.get("experiment_name", "improved_moe_run")
    search_paths = [
        os.path.join("logs", "moe_training", experiment_name),
        os.path.join("logs", "rsl_rl", experiment_name),
        os.path.join("logs", experiment_name)
    ]
    
    root_log_dir = search_paths[0]
    # 简单的路径查找逻辑
    for path in search_paths:
        if os.path.exists(path):
            root_log_dir = path
            break

    try:
        model_path, log_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
        print(f"\n[Success] Loading model from: {model_path}")
    except Exception as e:
        print(f"\n[Error] {e}")
        print(f"Searched in: {root_log_dir}")
        sys.exit(1)

    # 4. 启动 Runner 并加载模型
    env = RslRlVecEnvWrapper(env, clip_actions=train_cfg_dict.get("clip_actions", True))
    runner = OnPolicyRunner(env, train_cfg_dict, log_dir=log_dir, device="cuda:0")
    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    # === 可视化 Hook ===
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            if isinstance(output, tuple): data = output[0]
            else: data = output
            monitor_data[name] = data.detach()
        return _hook

    hook_registered = False
    if hasattr(model_instance, "actor_gate"): 
        model_instance.actor_gate.register_forward_hook(hook_fn("weights"))
        hook_registered = True

    last_printed_lines = 0
    def visualize(obs_idx=0):
        nonlocal last_printed_lines
        lines = ["="*20 + " MoE Policy Activation " + "="*20]
        if hook_registered and "weights" in monitor_data:
            weights_data = monitor_data["weights"]
            if weights_data.ndim == 3: weights = weights_data[-1, obs_idx]
            else: weights = weights_data[obs_idx]

            num_experts = len(weights)
            lines.append(f"Experts (Total: {num_experts}) | Env {obs_idx}")
            lines.append("-" * 50)
            for idx in range(num_experts):
                val = weights[idx].item()
                bar_len = int(val * 40)
                bar = '█' * bar_len
                if val > 0.1:
                    color = "\033[92m" # Green
                    status = "ACTIVE"
                elif val > 0.01:
                    color = "\033[93m" # Yellow
                    status = " WEAK "
                else:
                    color = "\033[90m" # Grey
                    status = " DEAD "
                lines.append(f"Exp {idx:02d}: {color}{val:.3f} | {bar:<40} | {status}\033[0m")
        elif not hook_registered:
            lines.append("Visualization unavailable (Hook failed).")
        lines.append("="*63)
        
        # 键盘模式下额外打印一些指令信息
        if args.keyboard and controller is not None:
             # 获取当前的指令 (vx, vy, w)
            cmd = controller.advance()
            lines.append(f"[Keyboard CMD] Vx: {cmd[0]:.2f}, Vy: {cmd[1]:.2f}, Yaw: {cmd[2]:.2f}")

        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        print("\n".join(lines))
        last_printed_lines = len(lines)

    # 5. 开始推理循环
    obs, _ = env.reset()
    print("\nStarting inference... Press Ctrl+C to stop.")
    if args.keyboard:
        print("[Info] Use WASD to move, Q/E to rotate, R to reset.")
    
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            
            # 可视化 (键盘模式下每帧刷新更流畅，普通模式降频)
            refresh_rate = 2 if args.keyboard else 10
            if step % refresh_rate == 0: 
                visualize(obs_idx=0)

            # 相机跟随
            if args.keyboard:
                rsl_rl_utils.camera_follow(env)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()