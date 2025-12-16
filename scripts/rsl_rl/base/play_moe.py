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

# MoE 架构参数 (必须与训练一致)
parser.add_argument("--num_wheel_experts", type=int, default=3, help="Number of wheel experts")
parser.add_argument("--num_leg_experts", type=int, default=3, help="Number of leg experts")

# === [Change] 加载参数改为可选 ===
parser.add_argument("--load_run", type=str, default=None, help="Name of the experiment folder (default: latest run)")
parser.add_argument("--checkpoint", type=str, default="model_*.pt", help="Checkpoint file pattern (default: model_*.pt)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 2. 导入其余库
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# === MoE 类注入 ===
from robot_lab.tasks.locomotion.velocity.config.wheeled.sirius_wheel.agents.moe_terrain import HierarchicalMoEActorCritic
import rsl_rl.modules as rsl_modules
rsl_modules.HierarchicalMoEActorCritic = HierarchicalMoEActorCritic
import rsl_rl.runners.on_policy_runner as runner_module
runner_module.HierarchicalMoEActorCritic = HierarchicalMoEActorCritic

def resolve_checkpoint_path(root_log_dir, run_name_or_path, checkpoint_pattern):
    """
    智能解析模型路径：
    1. 如果 run_name_or_path 为空 -> 找 root_log_dir 下最新的文件夹
    2. 如果 run_name_or_path 是相对路径名 -> 拼接到 root_log_dir 下
    3. 如果 run_name_or_path 是绝对路径 -> 直接使用
    """
    
    # 1. 确定实验运行目录 (run_dir)
    run_dir = None
    
    if run_name_or_path is None:
        # 情况 A: 未指定 run，自动寻找最新的
        print(f"[Info] No run specified. Searching for latest run in: {root_log_dir}")
        if not os.path.exists(root_log_dir):
             raise FileNotFoundError(f"Log directory not found: {root_log_dir}")
        
        # 获取所有子文件夹
        all_runs = [os.path.join(root_log_dir, d) for d in os.listdir(root_log_dir) if os.path.isdir(os.path.join(root_log_dir, d))]
        if not all_runs:
            raise FileNotFoundError(f"No runs found in {root_log_dir}")
        
        # 按修改时间排序，取最新的
        all_runs.sort(key=os.path.getmtime)
        run_dir = all_runs[-1]
        print(f"[Info] Auto-selected latest run: {os.path.basename(run_dir)}")
        
    elif os.path.isabs(run_name_or_path):
        # 情况 B: 绝对路径
        run_dir = run_name_or_path
    else:
        # 情况 C: 相对路径 (也就是具体的日期文件夹名)
        # 尝试直接拼接
        potential_path = os.path.join(root_log_dir, run_name_or_path)
        if os.path.exists(potential_path):
            run_dir = potential_path
        else:
            # 最后的尝试：也许用户把 run_name 当作了 experiment_name 的一部分？
            # 简单起见，这里假设用户传的就是日期文件夹名或者 train_moe 生成的那个目录
            run_dir = potential_path

    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    # 2. 在 run_dir 中寻找 checkpoint 文件
    # 常见结构是 run_dir/model_*.pt 或者 run_dir/checkpoints/model_*.pt
    # rsl_rl 通常直接放在 run_dir 下
    
    search_pattern = os.path.join(run_dir, checkpoint_pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        raise FileNotFoundError(f"No checkpoint matching '{checkpoint_pattern}' found in {run_dir}")
    
    # 排序取数字最大的 (通常文件名包含迭代次数)
    # 假设文件名是 model_1000.pt, model_2000.pt
    def extract_iter(f):
        try:
            return int(f.split("_")[-1].split(".")[0])
        except:
            return os.path.getmtime(f)
            
    files.sort(key=extract_iter)
    final_model_path = files[-1]
    
    return final_model_path, run_dir

def main():
    # 1. 解析环境配置
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # 2. 加载并修改 Train Config
    train_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    if hasattr(train_cfg, "to_dict"):
        train_cfg_dict = train_cfg.to_dict()
    else:
        train_cfg_dict = train_cfg

    # 注入类名和参数 (必须步骤)
    train_cfg_dict["policy"]["class_name"] = "HierarchicalMoEActorCritic"
    train_cfg_dict["policy"]["num_wheel_experts"] = args.num_wheel_experts
    train_cfg_dict["policy"]["num_leg_experts"] = args.num_leg_experts
    
    # 清理参数
    for k in ["checkpoint_wheel", "checkpoint_leg", "freeze_experts"]:
        train_cfg_dict["policy"].pop(k, None)

    # 3. 确定 Log 根目录
    # train_moe.py 中的逻辑是: logs/moe_training/<experiment_name>/<date>
    experiment_name = train_cfg_dict.get("experiment_name", "h_moe_end2end")
    root_log_dir = os.path.join("logs", "moe_training", experiment_name)

    # 4. 自动解析模型路径
    try:
        model_path, log_dir = resolve_checkpoint_path(root_log_dir, args.load_run, args.checkpoint)
        print(f"\n[Success] Loading model from: {model_path}")
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print(f"Please check your directories or specify --load_run explicitly.")
        sys.exit(1)

    # 5. 包装环境
    clip_actions = train_cfg_dict.get("clip_actions", True) 
    env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)

    # 6. 加载 Runner
    runner = OnPolicyRunner(
        env,
        train_cfg_dict,
        log_dir=log_dir, 
        device="cuda:0"
    )

    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    # ================= [修正版] H-MoE 权重监听逻辑 =================
    
    # 1. [关键修复] 获取真正的模型实例
    # policy 是 act_inference 方法，policy.__self__ 才是 HierarchicalMoEActorCritic 实例
    if hasattr(policy, "__self__"):
        model_instance = policy.__self__
    else:
        model_instance = policy # 备用

    monitor_data = {}

    def get_activation(name):
        """Hook 函数：捕获输出并detach"""
        def hook(model, input, output):
            monitor_data[name] = output.detach()
        return hook

    # 2. 注册 Hooks (根据你提供的 Log 精确匹配层级)
    print(f"\n[Info] Registering hooks on: {type(model_instance).__name__}")
    
    try:
        # (1) 顶层 Mode Gate
        if hasattr(model_instance, "mode_gate"):
            model_instance.mode_gate.register_forward_hook(get_activation("Mode_Gate"))
            print("  |-- Hooked: mode_gate")
        
        # (2) 轮子专家 Gate (根据 Log，它是 model_instance 的直接子模块)
        if hasattr(model_instance, "wheel_gate"):
            model_instance.wheel_gate.register_forward_hook(get_activation("Wheel_Gate"))
            print("  |-- Hooked: wheel_gate")
            
        # (3) 腿部专家 Gate
        if hasattr(model_instance, "leg_gate"):
            model_instance.leg_gate.register_forward_hook(get_activation("Leg_Gate"))
            print("  |-- Hooked: leg_gate")
            
    except Exception as e:
        print(f"[Error] Failed to register hooks: {e}")

    # 3. 可视化函数
    def visualize_weights(obs_idx=0):
        import torch.nn.functional as F
        
        # ANSI 转义码：\033[F 上移一行，\033[K 清除行 (实现原地刷新，不刷屏)
        # 如果终端不支持，可以注释掉下面这行
        # print("\033c", end="") 

        print("\n" + "="*20 + " H-MoE Realtime Weights " + "="*20)

        # --- 1. Mode Gate (Wheel vs Leg) ---
        if "Mode_Gate" in monitor_data:
            logits = monitor_data["Mode_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0) # [2]
            
            # 假设 0:Wheel, 1:Leg (取决于你训练时的 label 定义)
            p_wheel = probs[0].item()
            p_leg = probs[1].item()
            
            bar_len = 20
            w_bar = int(p_wheel * bar_len)
            l_bar = int(p_leg * bar_len)
            
            # 颜色高亮 (可选)
            c_reset = "\033[0m"
            c_wheel = "\033[92m" if p_wheel > p_leg else "\033[90m" # Green vs Grey
            c_leg   = "\033[92m" if p_leg > p_wheel else "\033[90m"
            
            print(f"Top Mode:  {c_wheel}Wheel {p_wheel:.2f}{c_reset} vs {c_leg}Leg {p_leg:.2f}{c_reset}")
            print(f"  Wheel: |{c_wheel}{'█'*w_bar:<{bar_len}}{c_reset}|")
            print(f"  Leg:   |{c_leg}{'█'*l_bar:<{bar_len}}{c_reset}|")
        else:
            print("[Wait] Mode Gate data missing.")

        # --- 2. Wheel Experts (Routing) ---
        if "Wheel_Gate" in monitor_data:
            logits = monitor_data["Wheel_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0)
            print(f"\nWheel Experts:")
            for i, p in enumerate(probs):
                val = p.item()
                bar = int(val * 20)
                if val > 0.1: # 仅高亮活跃的专家
                    print(f"  Exp {i}: \033[93m{val:.2f}\033[0m |{'#'*bar:<20}|")
                else:
                    print(f"  Exp {i}: {val:.2f} |{'#'*bar:<20}|")

        # --- 3. Leg Experts (Routing) ---
        if "Leg_Gate" in monitor_data:
            logits = monitor_data["Leg_Gate"][obs_idx]
            probs = F.softmax(logits, dim=0)
            print(f"\nLeg Experts:")
            for i, p in enumerate(probs):
                val = p.item()
                bar = int(val * 20)
                if val > 0.1:
                    print(f"  Exp {i}: \033[96m{val:.2f}\033[0m |{'#'*bar:<20}|")
                else:
                    print(f"  Exp {i}: {val:.2f} |{'#'*bar:<20}|")
        print("="*60)
    
    # ================= [结束] =================
    # 7. 推理循环
    obs, _ = env.reset()
    print("\nStarting inference loop... Press Ctrl+C to stop.")
    step_counter = 0

    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            
            # [Change] 这里的返回值必须是 4 个，如上个问题所述
            obs, _, _, _ = env.step(actions)
            
            # === [新增] 每 N 帧打印一次权重 ===
            step_counter += 1
            if step_counter % 10 == 0:  # 每10帧刷新一次，防止刷屏太快
                visualize_weights(obs_idx=0) # 只看第0个机器人的权重
            
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()