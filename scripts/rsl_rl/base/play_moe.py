import argparse
import sys
import os
import glob
import json
import pickle
import yaml
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === 1. 启动 App ===
parser = argparse.ArgumentParser(description="Play H-MoE Policy and Export")
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
# Export
parser.add_argument("--export", action="store_true", default=True, help="Whether to export ONNX/TorchScript and Configs.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ==============================================================================
#  所有的 Torch/Gym/IsaacLab 导入必须放在 simulation_app 启动之后
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.devices import Se2Keyboard
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
import rsl_rl_utils

# === 导入自定义模块 ===
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

# ==============================================================================
#  Define Export Classes
# ==============================================================================

class ExportablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()

    def forward(self, obs, hidden_states):
        batch_size = obs.shape[0]
        masks = torch.ones(batch_size, dtype=torch.bool, device=obs.device)
        action_mean, _, next_state = self.policy.forward(
            obs, masks=masks, hidden_states=hidden_states, save_dist=False
        )
        return action_mean, next_state

class StatefulWrapper(nn.Module):
    def __init__(self, policy_scripted, hidden_shape):
        super().__init__()
        self.policy = policy_scripted
        self.register_buffer("hidden_state", torch.zeros(hidden_shape))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action, next_hidden = self.policy(obs, self.hidden_state)
        self.hidden_state = next_hidden
        return action

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()

class ExportableEstimator(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.estimator = policy.estimator
        self.normalizer = policy.estimator_obs_normalizer
        self.estimator.eval()
        if self.normalizer: self.normalizer.eval()

    def forward(self, obs):
        if self.normalizer is not None: obs = self.normalizer(obs)
        return self.estimator(obs)

# ==============================================================================
#  Helper Functions
# ==============================================================================

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

def save_configs(log_dir, env_cfg, train_cfg_dict):
    """Saves configuration files to log_dir/params."""
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    print(f"\n[Export] Saving configs to: {params_dir}")

    with open(os.path.join(params_dir, "train_cfg.json"), "w") as f:
        json.dump(train_cfg_dict, f, indent=4, default=str)

    def sanitize_for_yaml(obj):
        if isinstance(obj, dict): return {k: sanitize_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [sanitize_for_yaml(v) for v in obj]
        elif callable(obj): return f"<function {getattr(obj, '__name__', 'lambda')}>"
        elif hasattr(obj, "to_dict"): return sanitize_for_yaml(obj.to_dict())
        elif hasattr(obj, "__dict__"): return sanitize_for_yaml(vars(obj))
        return obj

    try:
        env_dict = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else vars(env_cfg)
        env_dict_clean = sanitize_for_yaml(env_dict)
        with open(os.path.join(params_dir, "env_cfg.yaml"), "w") as f:
            yaml.dump(env_dict_clean, f, default_flow_style=False, sort_keys=False)
        print("  - Saved env_cfg.yaml")
    except Exception as e:
        print(f"[Warning] Failed to save env_cfg.yaml: {e}")
        with open(os.path.join(params_dir, "env_cfg_dump.txt"), "w") as f:
            f.write(str(env_cfg))

    with open(os.path.join(params_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

def export_model_files(policy, log_dir, obs_dim, device):
    exported_dir = os.path.join(log_dir, "exported")
    os.makedirs(exported_dir, exist_ok=True)
    print(f"\n[Export] Exporting models to: {exported_dir}")

    # 1. Base Model
    base_model = ExportablePolicy(policy).to(device)
    base_model.eval()
    
    batch_size = 1
    dummy_obs = torch.zeros(batch_size, obs_dim, device=device)
    with torch.no_grad():
        _, dummy_hidden = base_model(dummy_obs, None)
    
    # 2. Stateful JIT
    try:
        print("  - Tracing inner policy...")
        traced_inner = torch.jit.trace(base_model, (dummy_obs, dummy_hidden))
        stateful_model = StatefulWrapper(traced_inner, dummy_hidden.shape).to(device)
        scripted_stateful = torch.jit.script(stateful_model)
        
        save_path = os.path.join(exported_dir, "policy_stateful.pt")
        scripted_stateful.save(save_path)
        print(f"  - [Recommended] Stateful JIT saved: {save_path}")
    except Exception as e:
        print(f"  - [Error] Stateful JIT export failed: {e}")

    # 3. ONNX
    try:
        onnx_path = os.path.join(exported_dir, "policy.onnx")
        input_names = ["obs", "hidden_states"]
        output_names = ["action", "next_hidden_states"]
        if isinstance(dummy_hidden, tuple):
             input_names = ["obs", "h_in", "c_in"]
             output_names = ["action", "h_out", "c_out"]
        
        torch.onnx.export(
            base_model, (dummy_obs, dummy_hidden), onnx_path, verbose=False,
            input_names=input_names, output_names=output_names, opset_version=13,
            dynamic_axes={
                "obs": {0: "batch_size"},
                input_names[1]: {1: "batch_size"}, 
                output_names[0]: {0: "batch_size"},
                output_names[1]: {1: "batch_size"},
            }
        )
        print(f"  - ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"  - [Error] ONNX export failed: {e}")

    # 4. Estimator
    if hasattr(policy, "estimator") and policy.estimator is not None:
        print("\n  - Found Estimator, exporting separately...")
        try:
            est_input_indices = getattr(policy, "estimator_input_indices", list(range(3, 32)))
            est_input_dim = len(est_input_indices)
            dummy_est_obs = torch.zeros(1, est_input_dim, device=device)
            est_wrapper = ExportableEstimator(policy).to(device)
            
            traced_estimator = torch.jit.trace(est_wrapper, dummy_est_obs)
            est_pt_path = os.path.join(exported_dir, "estimator_jit.pt")
            traced_estimator.save(est_pt_path)
            print(f"    - Estimator JIT saved: {est_pt_path}")
            
            est_onnx_path = os.path.join(exported_dir, "estimator.onnx")
            torch.onnx.export(
                est_wrapper, dummy_est_obs, est_onnx_path, verbose=False,
                input_names=["proprioception"], output_names=["estimated_state"], opset_version=13,
                dynamic_axes={"proprioception": {0: "batch_size"}, "estimated_state": {0: "batch_size"}}
            )
            print(f"    - Estimator ONNX saved: {est_onnx_path}")
        except Exception as e:
            print(f"    - [Error] Estimator export failed: {e}")

# ==============================================================================
#  Main Loop
# ==============================================================================

def main():
    # 1. 配置与环境
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    
    # Keyboard setup
    controller = None
    if args.keyboard:
        print("[Info] Enabling Keyboard Control")
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True
        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
        )
        controller = Se2Keyboard(kb_cfg)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )

    env = gym.make(args.task, cfg=env_cfg)
    
    # 获取 obs_dim
    try:
        obs_sample, _ = env.reset()
        if isinstance(obs_sample, dict):
            policy_obs = obs_sample["policy"]
            obs_dim = policy_obs.shape[-1]
        else:
            obs_dim = obs_sample.shape[-1]
    except:
        obs_dim = 48 

    if args.keyboard and controller is not None:
        def reset_env_callback():
            print("[INFO] 'R' key pressed: Resetting environment.")
            nonlocal obs
            obs, _ = env.reset()
        controller.add_callback("R", reset_env_callback)

    # 2. 加载模型
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg

    train_cfg_dict["policy"]["class_name"] = "SplitMoEActorCritic"
    if args.num_wheel_experts: train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    if args.num_leg_experts: train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]: train_cfg_dict["policy"].pop(k, None)

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

    # Export
    if args.export:
        print("\n" + "="*80)
        print(" STARTING EXPORT PROCESS ")
        print("="*80)
        save_configs(log_dir, env_cfg, train_cfg_dict)
        export_model_files(model_instance, log_dir, obs_dim, device="cuda:0")
        print("="*80 + "\n")

    # Hard/Soft Mode
    use_hard_mode = getattr(model_instance, "use_hard_mode_switch", False)
    print(f"\n[Info] Mode Gate Strategy: {'HARD (Top-1)' if use_hard_mode else 'SOFT (Softmax)'}")

    # Hooks
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            monitor_data[name] = output.detach()
        return _hook

    if hasattr(model_instance, "mode_gate"): model_instance.mode_gate.register_forward_hook(hook_fn("Mode"))
    if hasattr(model_instance, "wheel_gate"): model_instance.wheel_gate.register_forward_hook(hook_fn("Wheel"))
    if hasattr(model_instance, "leg_gate"): model_instance.leg_gate.register_forward_hook(hook_fn("Leg"))

    # === [关键新增] 获取仿真器底层 Robot 对象以便读取 GT ===
    # 注意：env.unwrapped 是 RslRlVecEnvWrapper -> ManagerBasedRslEnv
    try:
        # 尝试获取 Robot 对象 (通常在场景中名为 "robot" 或 "actor")
        # 直接访问 scene
        robot_entity = env.unwrapped.scene["robot"]
        print("[Info] Successfully connected to Robot entity for GT verification.")
    except Exception as e:
        robot_entity = None
        print(f"[Warning] Could not find 'robot' in scene for GT verification: {e}")

    # === 辅助打印函数 ===
    def print_expert_bars(probs, expert_names=None):
        lines = []
        for i, p in enumerate(probs):
            val = p.item()
            bar_len = int(val * 40)
            bar = '█' * bar_len
            
            if val > 0.9: color, status = "\033[92m", "DOMINANT" # Green
            elif val > 0.1: color, status = "\033[96m", "ACTIVE" # Cyan
            elif val > 0.01: color, status = "\033[93m", " WEAK " # Yellow
            else: color, status = "\033[90m", " DEAD " # Grey
            
            name = expert_names[i] if expert_names else f"Exp {i}"
            lines.append(f"  {name:<8}: {color}{val:.3f} | {bar:<40} | {status}\033[0m")
        return lines

    def print_estimator_diff(est_vec, gt_vec):
        """可视化 Estimator 误差"""
        lines = []
        labels = ["Vx", "Vy", "Wz"]
        # 假设输出是 3 维 (lin_x, lin_y, ang_z)
        # 如果维度不对，做简单适配
        dim = min(len(est_vec), len(gt_vec), 3)
        
        for i in range(dim):
            e = est_vec[i].item()
            g = gt_vec[i].item()
            diff = abs(e - g)
            
            # 颜色编码误差
            if diff < 0.1: err_color = "\033[92m" # Green (Good)
            elif diff < 0.3: err_color = "\033[93m" # Yellow (Warning)
            else: err_color = "\033[91m" # Red (Bad)
            
            # 误差条
            bar_len = min(int(diff * 50), 40)
            bar = '▒' * bar_len
            
            lines.append(f"  {labels[i]}: Est={e:6.3f} | GT={g:6.3f} | Err={err_color}{diff:6.3f} {bar}\033[0m")
        return lines

    last_printed_lines = 0
    
    # 增加参数：传入 est 和 gt
    def visualize(obs_idx=0, est_state=None, gt_state=None):
        nonlocal last_printed_lines
        lines = ["="*25 + " H-MoE Dashboard " + "="*25]
        
        # 1. Mode Gate
        if "Mode" in monitor_data:
            logits = monitor_data["Mode"]
            if logits.ndim == 3: logits = logits[-1]
            probs = F.softmax(logits[obs_idx], dim=0)
            if use_hard_mode:
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

        lines.append("-" * 65)

        # 2. Sub-Gates
        for name in ["Wheel", "Leg"]:
            if name in monitor_data:
                logits = monitor_data[name]
                if logits.ndim == 3: logits = logits[-1]
                probs = F.softmax(logits[obs_idx], dim=0)
                lines.append(f"{name} Experts:")
                lines.extend(print_expert_bars(probs))
        
        lines.append("-" * 65)
        
        # 3. Estimator Visualization (New)
        if est_state is not None and gt_state is not None:
            lines.append("State Estimator (Velocities):")
            lines.extend(print_estimator_diff(est_state[obs_idx], gt_state[obs_idx]))
        elif est_state is None:
            lines.append("Estimator: Not active (or None).")

        lines.append("="*65)

        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        print("\n".join(lines))
        last_printed_lines = len(lines)
        
    obs, _ = env.reset()
    print("\nStarting inference...")
    
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            
            # === 新增：获取 Estimator 预测值 ===
            est_state = None
            if hasattr(model_instance, "get_estimated_state"):
                # 注意：get_estimated_state 内部会切片 obs 获取 proprioception
                est_state = model_instance.get_estimated_state(obs)
            
            # === 新增：获取 Ground Truth ===
            gt_state = None
            if robot_entity is not None:
                # 获取基座线速度 (Body Frame)
                # robot_entity.data.root_lin_vel_b 形状为 [num_envs, 3]
                gt_lin = robot_entity.data.root_lin_vel_b
                # 如果 estimator 输出也包含角速度 (通常是 3维: vx, vy, wz)，我们这里暂时只取线速度前2维 + 角速度z
                # 需要根据你的 estimator_output_dim 来定
                # 假设 Estimator 输出 3 维 [vx, vy, wz] (常见配置)
                # GT 构造:
                gt_ang = robot_entity.data.root_ang_vel_b
                gt_state = torch.cat([gt_lin[:, :2], gt_ang[:, 2:3]], dim=-1) # [vx, vy, wz]

            # Step
            obs, _, _, _ = env.step(actions)
            step += 1
            
            if step % 10 == 0:
                visualize(obs_idx=0, est_state=est_state, gt_state=gt_state)
            
            if args.keyboard:
                rsl_rl_utils.camera_follow(env)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()