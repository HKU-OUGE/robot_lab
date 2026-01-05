import argparse
import sys
import os
import glob
from isaaclab.app import AppLauncher

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

# === 导入自定义模块 (Policy 和 Algorithm) ===
# 尝试从多个路径导入，确保能找到 moe_simple.py
try:
    # 路径 1: 尝试本地直接导入
    sys.path.append(os.getcwd())
    from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import RobustMoEActorCritic, MoEPPO
    print("[Info] Successfully imported RobustMoEActorCritic and MoEPPO from current directory.")
except ImportError as e:
    print(f"[Warning] Local import failed: {e}")
    try:
        # 路径 2: 尝试从项目路径导入 (根据你的实际项目结构调整)
        from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_simple import RobustMoEActorCritic, MoEPPO
        print("[Info] Imported from robot_lab.algo.moe.moe_simple")
    except ImportError:
        raise ImportError(
            "Could not import 'RobustMoEActorCritic' or 'MoEPPO'. \n"
            "Please ensure 'moe_simple.py' is in the current directory or python path."
        )

# === 注入到 RSL-RL 运行环境 ===
import rsl_rl.modules as rsl_modules
import rsl_rl.runners.on_policy_runner as runner_module

# 1. 注入 Policy 类
rsl_modules.RobustMoEActorCritic = RobustMoEActorCritic
runner_module.RobustMoEActorCritic = RobustMoEActorCritic

# 2. 注入 Algorithm 类 (必须注入，否则加载 train_cfg 时会报错找不到 MoEPPO)
runner_module.MoEPPO = MoEPPO 

# 为了兼容旧的 checkpoint 名字（如果有的话）
rsl_modules.SimpleMoEActorCritic = RobustMoEActorCritic 

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    run_dir = None
    # 如果没有指定具体的 run，自动寻找最新的
    if run_name_or_path is None:
        if not os.path.exists(root_log_dir): 
            # 尝试在上一级目录找 (有时 log_dir 结构不同)
            parent_dir = os.path.dirname(root_log_dir)
            if os.path.exists(parent_dir):
                print(f"[Info] Log dir {root_log_dir} not found, searching in {parent_dir}")
                root_log_dir = parent_dir
        if not os.path.exists(root_log_dir): 
            raise FileNotFoundError(f"Log dir not found: {root_log_dir}")
        # 寻找最近修改的运行文件夹
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
    # 按时间排序取最新的模型
    files.sort(key=lambda f: os.path.getmtime(f))
    return files[-1], run_dir

def main():
    # 1. 创建环境
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # 2. 加载配置
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"): train_cfg_dict = train_cfg.to_dict()
    else: train_cfg_dict = train_cfg

    # === 强制覆写 Policy 配置 ===
    # 确保 class_name 与我们导入的类一致
    train_cfg_dict["policy"]["class_name"] = "RobustMoEActorCritic"
    # 如果配置中使用了 MoEPPO，确保 runner 也能识别 (已通过上面的注入解决)
    # 但如果为了保险，可以将算法改回 PPO (仅推理时不影响，因为推理只用 Policy 网络)
    # train_cfg_dict["algorithm"]["class_name"] = "PPO" 

    if args.num_experts: train_cfg_dict["policy"]["num_experts"] = args.num_experts
    if args.top_k: train_cfg_dict["policy"]["top_k"] = args.top_k
    # 清理多余参数
    for k in ["num_wheel_experts", "num_leg_experts", "wheel_indices", "leg_indices", 
              "checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # 3. 确定 Checkpoint 路径
    experiment_name = train_cfg_dict.get("experiment_name", "improved_moe_run")
    # 默认路径列表
    search_paths = [
        os.path.join("logs", "moe_training", experiment_name), # 训练脚本中自定义的路径
        os.path.join("logs", "rsl_rl", experiment_name),       # 标准路径
        os.path.join("logs", experiment_name)
    ]
    root_log_dir = search_paths[0]
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
    # 获取底层 Model 实例 (处理可能的包装器)
    model_instance = policy.__self__ if hasattr(policy, "__self__") else policy

    # === 可视化 Hook ===
    monitor_data = {}
    def hook_fn(name):
        def _hook(model, input, output):
            # output 是权重张量 (Batch, Num_Experts) 或 (Seq, Batch, Num_Experts)
            if isinstance(output, tuple):
                data = output[0]
            else:
                data = output
            monitor_data[name] = data.detach()
        return _hook

    # 注册 Hook 到 Gate 网络
    hook_registered = False
    if hasattr(model_instance, "actor_gate"): 
        print("[Info] Hooking into 'actor_gate' for visualization")
        model_instance.actor_gate.register_forward_hook(hook_fn("weights"))
        hook_registered = True
    else:
        print(f"[Warning] 'actor_gate' not found in policy. Available keys: {model_instance.__dict__.keys()}")

    last_printed_lines = 0
    def visualize(obs_idx=0):
        nonlocal last_printed_lines
        lines = ["="*20 + " MoE Policy Activation " + "="*20]
        if hook_registered and "weights" in monitor_data:
            # 获取当前帧的权重
            weights_data = monitor_data["weights"]
            # 如果是 RNN 序列 (Seq, Batch, Exp)，取最后一个时间步
            if weights_data.ndim == 3:
                weights = weights_data[-1, obs_idx]
            else:
                weights = weights_data[obs_idx] # (Batch, Exp) -> (Exp,)

            num_experts = len(weights)
            lines.append(f"Experts (Total: {num_experts}) | Env {obs_idx}")
            lines.append("-" * 50)
            for idx in range(num_experts):
                val = weights[idx].item()
                # 简单的条形图
                bar_len = int(val * 40)
                bar = '█' * bar_len
                # 高亮主要专家 (权重 > 0.1)
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
            lines.append("Visualization unavailable.")
        else:
            lines.append("Waiting for model execution...")
        lines.append("="*63)

        # 刷新控制台输出
        if last_printed_lines > 0:
            sys.stdout.write(f"\033[{last_printed_lines}A\033[J")
        print("\n".join(lines))
        last_printed_lines = len(lines)

    # 5. 开始推理循环
    obs, _ = env.reset()
    print("\nStarting inference... Press Ctrl+C to stop.")
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            # 每 10 步刷新一次可视化，避免闪烁过快
            if step % 10 == 0: 
                visualize(obs_idx=0) # 默认观察第 0 个环境

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()