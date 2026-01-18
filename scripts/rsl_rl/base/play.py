# ==============================================================================
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Modified by: Tianyang TANG
# ==============================================================================

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


# import json

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--debug", action="store_true", default=False, help="Print debug information (env config, action and observation spaces).")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
parser.add_argument("--moe", action="store_true", default=False, help="Whether to use MoE.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
if hydra_args:
    print("[INFO] Ignoring Hydra-style overrides in play.py:", hydra_args)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

import rsl_rl_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.devices.keyboard.se2_keyboard import Se2KeyboardCfg 
import robot_lab.tasks  # noqa: F401
# --- MoE Actor that can drop-in replace the base policy's actor MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.utils.dict import print_dict
def _make_mlp(in_dim: int, hidden: list[int], out_dim: int, act: nn.Module):
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)

class _MoEActor(nn.Module):
    """Deterministic MoE actor head: softmax routing over expert MLPs.

    - No randomness inside the actor mean; all exploration still comes from base policy's Normal(mean, std).
    - Supports top-k sparse routing by zeroing non-topk logits before softmax (still deterministic).
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: list[int],
        act: type[nn.Module],
        num_experts: int = 4,
        topk: int = 1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self._in_features = int(in_dim)     # NEW: 供导出器探测
        self._out_features = int(out_dim)   # NEW: 供导出器探测
        assert num_experts >= 1
        assert 1 <= topk <= num_experts
        self.num_experts = num_experts
        self.topk = topk
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.experts = nn.ModuleList([_make_mlp(in_dim, hidden, out_dim, act) for _ in range(num_experts)])
        # 一个小 gating MLP（与基类 actor 的规模同一量级即可）
        gate_hidden = max(64, (hidden[0] if hidden else 128) // 2)
        self.gate = _make_mlp(in_dim, [gate_hidden], num_experts, act)
    # --- 以下三个方法是“顺序模块”兼容探针 ---
    class _FakeLayer:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    def __len__(self):                      # NEW
        # 让导出器可以 len(self.actor)
        return 2

    def __getitem__(self, idx):             # NEW
        # 导出器会读 [0].in_features，可能也会读 [-1].out_features
        if idx in (0, -2):
            return self._FakeLayer(in_features=self._in_features)
        if idx in (-1, 1):
            return self._FakeLayer(out_features=self._out_features)
        raise IndexError(idx)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # [B, E]
        if self.topk < self.num_experts:
            # 稀疏 top-k：非 top-k 位置置为 -inf，再 softmax
            topk_vals, topk_idx = logits.topk(self.topk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, topk_idx, True)
            logits = logits.masked_fill(~mask, float("-inf"))
        probs = F.softmax(logits / self.temperature.clamp(min=1e-6), dim=-1)  # [B, E]
        means = torch.stack([e(x) for e in self.experts], dim=1)              # [B, E, A]
        out = torch.einsum("be,bea->ba", probs, means)                        # [B, A]
        return out

# -------- Policy that reuses ALL rsl-rl logic and just swaps the actor --------
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except Exception:
    import rsl_rl.modules.actor_critic as _ac_mod
    _BaseActorCritic = _ac_mod.ActorCritic

import torch
import torch.nn as nn

class ActorCriticMoE(_BaseActorCritic):
    """RSL-RL v3 兼容：基于基类的 MoE 策略。
    - 复用基类：obs 归一化 / log_std / act() / evaluate() / evaluate_actions() / 导出等
    - 仅替换 actor 的 MLP 为 MoE 头（确定性均值；探索仍由基类的 Normal(mean, std) 完成）
    """
    def __init__(
        self,
        # ★ v3 签名：先给 obs、obs_groups，再给 num_actions 与其余 cfg ★
        obs,                      # dict 或 TensorDict：来自环境的一个样本观测（含各组）
        obs_groups: dict,         # 形如 {'actor': ['policy'], 'critic': ['critic']}
        num_actions: int,
        *,
        actor_hidden_dims = [256, 256],
        critic_hidden_dims = [512, 256],
        activation = "elu",
        init_noise_std = 0.8,
        noise_std_type = "scalar",
        actor_obs_normalization = True,
        critic_obs_normalization = True,
        # MoE 相关
        num_experts: int = 4,
        topk: int = 1,
        moe_temperature: float = 1.0,
        **kwargs,
    ):
        # 先让基类按 v3 流程把一切搭好（包含 obs 归一化、log_std 等）
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims = actor_hidden_dims,
            critic_hidden_dims = critic_hidden_dims,
            activation = activation,
            init_noise_std = init_noise_std,
            noise_std_type = noise_std_type,
            actor_obs_normalization = actor_obs_normalization,
            critic_obs_normalization = critic_obs_normalization,
            **kwargs,
        )

        act = dict(relu=nn.ReLU, elu=nn.ELU, gelu=nn.GELU)[activation.lower()]

        # 1) 从基类已构建好的 actor MLP 里取首层 Linear 的 in_features（最稳妥）
        base_actor = self.actor
        actor_in_dim = None
        for m in base_actor.modules():
            if isinstance(m, nn.Linear):
                actor_in_dim = m.in_features
                break

        # 2) 兜底：如果意外没取到（几乎不会发生），再用 obs_groups 的 'policy' 写法；若还没有就取第一个键
        if actor_in_dim is None:
            def _lastdim(t):
                return int(t.shape[-1])
            keys_for_actor = obs_groups.get("policy", None)
            if keys_for_actor is None and "actor" in obs_groups:  # 兼容别处用过的命名
                keys_for_actor = obs_groups["actor"]
            if keys_for_actor is None:
                keys_for_actor = ["policy"] if (isinstance(obs, dict) and "policy" in obs) else [next(iter(obs.keys()))]
            actor_in_dim = sum(_lastdim(obs[k]) for k in keys_for_actor)

        # 3) 构建 MoE 头并替换
        self.actor = _MoEActor(
            in_dim = actor_in_dim,
            out_dim = num_actions,
            hidden = list(actor_hidden_dims),
            act    = act,
            num_experts = num_experts,
            topk        = topk,
            temperature = float(moe_temperature),
        )

        # 4) （可选）继续给 MoE 线性层加谱归一化，与你现有风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

        # （可选）给 MoE 里的 Linear 上谱归一化，与你之前的 SN 风格一致
        try:
            from torch.nn.utils.parametrizations import spectral_norm as _sn
            for m in self.actor.modules():
                if isinstance(m, nn.Linear):
                    _sn(m, n_power_iterations=1)
        except Exception:
            pass

    # 便于监控 gating 的平均熵（可 wandb.log）
# 让 Isaac Lab 用 getattr(...) 能找到：rsl_rl.modules.actor_critic.ActorCriticMoE

# === NEW: debug helper to print root pose & target height ===
def _print_root_and_target(env):
    """
    Print root_pos_w (x,y,z), configured target_height, and (if available)
    ground_z from height scanner plus adjusted target = ground_z + target_height.
    """
    try:
        asset = env.unwrapped.scene["robot"]
        rp = asset.data.root_pos_w[0].detach().cpu().numpy()
        msg = f"[ROOT_POS_W] x={rp[0]:+.3f}  y={rp[1]:+.3f}  z={rp[2]:+.3f}"
    except Exception as e:
        print(f"[WARN] Cannot read root_pos_w: {e}", flush=True)
        return

    # read target_height from env config if present
    th = None
    try:
        th = float(env.unwrapped.cfg.rewards.base_height_l2.params["target_height"])
        msg += f"  | target_height(cfg)={th:+.3f}"
    except Exception:
        pass

    # try to read ground estimate from a RayCaster named "height_scanner_base"
    try:
        sensor = env.unwrapped.scene.sensors.get("height_scanner_base", None)
        if sensor is not None:
            z_hits = sensor.data.ray_hits_w[0, :, 2]
            import torch
            valid = torch.isfinite(z_hits)
            if valid.any():
                ground_z = z_hits[valid].mean().item()
                msg += f"  | ground_z≈{ground_z:+.3f}"
                if th is not None:
                    msg += f"  | adjusted≈{ground_z + th:+.3f}"
    except Exception as e:
        msg += f"  | ground_z=N/A ({e})"

    print(msg, flush=True)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # if args_cli.debug:
    #     print("\n==== [env_cfg 配置结构] ====\n")
    #     print_dict(env_cfg.to_dict(), nesting=4)
    # with open("env_cfg_debug.json", "w") as f:
    #     json.dump(env_cfg.to_dict(), f, indent=4)
    agent_cfg: RslRlBaseRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.moe:
        import rsl_rl.modules.actor_critic as ac
        ac.ActorCriticMoE = ActorCriticMoE
        import rsl_rl.runners.on_policy_runner as _opr
        _opr.ActorCriticMoE = ActorCriticMoE
        agent_cfg.policy.class_name = "ActorCriticMoE"
        # 2) 强制对齐训练时的网络维度 / 激活 / 探索噪声
        #    （Hydra 风格的 'agent.policy.*' 在这个脚本里会被忽略，所以在代码里直接改 cfg）
        agent_cfg.policy.actor_hidden_dims = [512, 256, 128]
        agent_cfg.policy.critic_hidden_dims = [512, 256, 128]
        agent_cfg.policy.activation = "elu"
        agent_cfg.policy.init_noise_std = 0.8
    # make a smaller scene for play
    env_cfg.scene.num_envs = args_cli.num_envs
    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 10
        env_cfg.scene.terrain.terrain_generator.num_cols = 14
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = True

        kb_cfg = Se2KeyboardCfg(
            v_x_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1]),
            v_y_sensitivity=float(env_cfg.commands.base_velocity.ranges.lin_vel_y[1]),
            omega_z_sensitivity=float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1]),
            # sim_device 默认即可；需要的话可传 env_cfg.sim.device
        )
        controller = Se2Keyboard(kb_cfg)  # ← 用配置类构造

        # 返回形状 [1, 3] 的 (vx, vy, wz)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: controller.advance().unsqueeze(0).to(env.device, dtype=torch.float32),
        )


        def reset_env_callback():
            print("[INFO] 'R' key pressed: Resetting environment.")
            nonlocal obs
            obs, _ = env.reset()
        controller.add_callback("R", reset_env_callback)


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # --- De-parametrize all layers before TorchScript export ---
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    def _deparametrize_all(m):
        # 遍历所有子模块，若存在任何参数化(如 spectral_norm)则移除
        for mod in m.modules():
            if is_parametrized(mod):
                # 逐个把所有被参数化的参数（通常是 "weight"）恢复成普通参数
                if hasattr(mod, "parametrizations"):
                    for pname in list(mod.parametrizations.keys()):
                        try:
                            remove_parametrizations(mod, pname, leave_parametrized=False)
                        except Exception:
                            pass

    # 对策略网络做去参数化
    _deparametrize_all(policy_nn)
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx", verbose=True)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    # # --- 构建观测切片索引：名字 -> slice(start, end) ---
    # def build_group_index_map(obs_mgr, group_name="policy"):
    #     names = obs_mgr._group_obs_term_names[group_name]
    #     shapes = obs_mgr._group_obs_term_dim[group_name]  # e.g. (171,), (3,), ...
    #     idx_map, start = {}, 0
    #     for name, shape in zip(names, shapes):
    #         # shape 可能是 int 或 tuple，做个通用乘积
    #         if isinstance(shape, (list, tuple)):
    #             n = 1
    #             for s in shape:
    #                 n *= int(s)
    #         else:
    #             n = int(shape)
    #         idx_map[name] = slice(start, start + n)
    #         start += n
    #     return idx_map

    # # 在获取到第一帧 obs 之后构建一次映射
    # obs_mgr = env.unwrapped.observation_manager
    # idx_map = build_group_index_map(obs_mgr, group_name="policy")

    # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
    # env_id = 0
    # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
    # print(f"[height_scan] env#{env_id} len={hs.size}:")
    # print(hs)

    # # 如果你想按网格显示（171=9*19 很常见），可 reshape 看看
    # try:
    #     hs_grid = hs.reshape(9, 19)   # 若你的配置不是 9x19，把 9,19 换成你的行列
    #     print("[height_scan as grid 9x19]:")
    #     print(hs_grid)
    # except Exception:
    #     pass

    timestep = 0
    debug_print = False
    if args_cli.debug and not debug_print:
        # print obs & action dim
        print("\n========== [OBSERVATION / ACTION SHAPE INFO] ==========", flush=True)
        try:
            # print observation vector shape
            if isinstance(obs, dict):
                flat_obs_shape = sum([v.numel() for v in obs.values()])
                print(f"[OBS] Total flattened shape: {flat_obs_shape} (from {len(obs)} components)", flush=True)
            else:
                print(f"[OBS] shape: {tuple(obs.shape)}", flush=True)
            
            # print action vector shape
            action_tensor = env.unwrapped.action_manager.action
            print(f"[ACTION] shape: {tuple(action_tensor.shape)}", flush=True)
        except Exception as e:
            print(f"[WARN] Cannot access shape info: {e}", flush=True)
        print("========================================================\n", flush=True)

        # print obs group -> term list
        print("\n========== [OBS GROUP MEMBERS LIST & INFO] ==========", flush=True)
        try:
            obs_mgr = env.unwrapped.observation_manager
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                print(f"[OBS GROUP] {group_name}: {term_names}", flush=True)
                for idx, name in enumerate(term_names):
                    term_cfg = obs_mgr._group_obs_term_cfgs[group_name][idx]
                    shape = obs_mgr._group_obs_term_dim[group_name][idx]
                    func_name = getattr(term_cfg.func, '__name__', str(term_cfg.func))
                    noise_type = type(term_cfg.noise).__name__ if term_cfg.noise else None
                    # print detailed info
                    print(f"  [OBS NAME] {name}", flush=True)
                    print(f"    [FUNC]        {func_name}", flush=True)
                    print(f"    [SHAPE]       {shape}", flush=True)
                    print(f"    [HISTORY]     len={term_cfg.history_length} flatten={term_cfg.flatten_history_dim}", flush=True)
                    print(f"    [CLIP]        {term_cfg.clip}", flush=True)
                    print(f"    [SCALE]       {term_cfg.scale}", flush=True)
                    print(f"    [NOISE]       {noise_type}", flush=True)
        except Exception as e:
            print(f"[WARN] Observation manager terms not accessible: {e}", flush=True)
        print("======================================================\n", flush=True)

        # print action space vector
        print("\n====== [Action Vector Mapping] ======", flush=True)
        idx = 0
        for group_name, term in env.unwrapped.action_manager._terms.items():
            print(f"[ACTION GROUP] {group_name}", flush=True)
            joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
            term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
            for i, val in enumerate(term_actions):
                joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
            idx += term.action_dim
        print("=====================================\n", flush=True)

        debug_print = True
        time.sleep(0.1)  # avoid stdout loss
    # # 导出后切 JIT
    # jit_path = os.path.join(export_model_dir, "policy.pt")
    # del runner           # 不再需要含 critic 的 runner
    # torch.cuda.empty_cache() # 回收显存

    # policy_jit = torch.jit.load(jit_path, map_location=env.unwrapped.device)
    # policy_jit.eval()
    # simulate environment
    while simulation_app.is_running():
        # print action space vector
        if args_cli.debug and args_cli.keyboard:
            print("\n====== [Action Vector Mapping] ======", flush=True)
            idx = 0
            for group_name, term in env.unwrapped.action_manager._terms.items():
                print(f"[ACTION GROUP] {group_name}", flush=True)
                joint_names = term._joint_names if hasattr(term, "_joint_names") else [f"joint_{i}" for i in range(term.action_dim)]
                term_actions = env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
                for i, val in enumerate(term_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                    print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
                idx += term.action_dim
            print("=====================================\n", flush=True)
            # === NEW: also print root_pos_w & (optional) ground/adjusted target
            _print_root_and_target(env)
            # # 取出并打印某个 env 的 height_scan（这里以 env_id = 0 为例）
            # env_id = 0
            # hs = obs[env_id, idx_map["height_scan"]].detach().cpu().numpy()
            # def print_height_scan_col_major(grid, precision=6, sep=" "):
            #     """
            #     按列打印：先 [0][0] [1][0] ... [8][0]，换行；
            #     然后 [0][1] [1][1] ... [8][1]，以此类推。
            #     """
            #     rows, cols = grid.shape
            #     fmt = f"{{:+.{precision}f}}"
            #     for c in range(cols):
            #         line = sep.join(fmt.format(float(grid[r, cols-1-c])) for r in range(rows))
            #         print(line, flush=True)

            # # 已有的 hs -> (9, 19)
            # hs_grid = hs.reshape(9, 19)
            # print_height_scan_col_major(hs_grid, precision=3)

        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # actions = torch.zeros_like(actions)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.keyboard:
            rsl_rl_utils.camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
