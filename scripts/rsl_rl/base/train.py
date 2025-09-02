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

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import importlib.metadata as metadata
import platform
from packaging import version
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add obs&action dict
obs_action_info = {
    "observation_groups": {},
    "action_groups": {}
}

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--recovery_mode", action="store_true", default=False, help="Whether to use recovery mode.")
parser.add_argument("--debug", action="store_true", default=False, help="Print debug information (env config, action and observation spaces).")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# ---- torchrun read distributed----
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = (WORLD_SIZE > 1) or bool(args_cli.distributed)
IS_MASTER = (LOCAL_RANK == 0)

# enable camera only on master
args_cli.enable_cameras = bool(args_cli.video and IS_MASTER)

# disable W&B in slaves
if IS_DISTRIBUTED and not IS_MASTER:
    os.environ["WANDB_MODE"] = "disabled"  # 等价于 wandb.init(mode="disabled")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)
"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# --- CustomRecordVideo: PyAV + W&B---
from typing import Callable
try:
    import wandb
except Exception:
    wandb = None
try:
    import av  # optional
except Exception:
    av = None

from gymnasium.wrappers.rendering import RecordVideo
from gymnasium import logger

class CustomRecordVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        enable_wandb: bool = True,
        wandb_key: str = "train/video",
        video_resolution: tuple[int, int] = (1280, 720),
        video_crf: int = 30,
    ):
        # robustness
        super().__init__(
            env=env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=step_trigger,
            video_length=video_length,
            name_prefix=name_prefix,
            disable_logger=disable_logger,
        )
        # Gymnasium  RecordVideoV0 will set self.frames_per_sec（if fps=None， env.metadata.render_fps & 30）
        if fps is not None:
            self.frames_per_sec = fps  

        self.enable_wandb = bool(enable_wandb and (wandb is not None))
        self.wandb_key = wandb_key
        self.video_resolution = tuple(video_resolution)
        self.video_crf = int(video_crf)

    def _write_with_pyav(self, frames, path):
        # PyAV->h264 + yuv420p（for web use）
        if av is None:
            raise RuntimeError("PyAV not available")
        container = av.open(path, "w")
        stream = container.add_stream("libx264", rate=round(self.frames_per_sec))
        stream.width, stream.height = self.video_resolution
        stream.pix_fmt = "yuv420p"
        # CRF defines video quality
        stream.options = {"crf": str(self.video_crf), "preset": "veryslow"}
        for fr in frames:
            vf = av.VideoFrame.from_ndarray(fr, format="rgb24")
            vf = vf.reformat(width=self.video_resolution[0], height=self.video_resolution[1])
            packet = stream.encode(vf)
            if packet:
                container.mux(packet)
        # flush
        packet = stream.encode(None)
        if packet:
            container.mux(packet)
        container.close()

    def stop_recording(self):
        """write to disk then upload to W&B。"""
        assert self.recording, "stop_recording was called, but no recording was started"

        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")

        if len(self.recorded_frames) == 0:
            logger.warn("Ignored saving a video as there were zero frames to save.")
        else:
            try:
                # PyAV 
                self._write_with_pyav(self.recorded_frames, path)
            except Exception:
                # Roll back to moviepy
                super().stop_recording()
            else:
                # Reset
                self.recorded_frames = []
                self.recording = False
                self._video_name = None

            # Try Upload
            if self.enable_wandb and os.path.exists(path) and (wandb is not None):
                try:
                    # key for bounding
                    wandb.log({self.wandb_key: wandb.Video(path, format="mp4")}, commit=True)
                    print(f"[W&B] Logged video: {path}")
                except Exception as e:
                    print(f"[WARN] wandb video log failed: {e}")

def make_serializable(info: dict):
    """Convert tensor and object fields into serializable types for YAML."""
    def tensor_to_list(val):
        if isinstance(val, torch.Tensor):
            return val.cpu().tolist()
        return val

    serializable_info = {}
    for key, value in info.items():
        if isinstance(value, dict):
            serializable_info[key] = make_serializable(value)
        elif isinstance(value, list):
            serializable_info[key] = [make_serializable(v) if isinstance(v, dict) else tensor_to_list(v) for v in value]
        else:
            serializable_info[key] = tensor_to_list(value)
    return serializable_info

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    if IS_DISTRIBUTED:
        env_cfg.sim.device = f"cuda:{LOCAL_RANK}"
        agent_cfg.device = f"cuda:{LOCAL_RANK}"
        seed = (agent_cfg.seed or 0) + LOCAL_RANK
        env_cfg.seed = seed
        agent_cfg.seed = seed
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # multi-gpu / multi-node (torch.distributed via torchrun)
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # random seed for each gpu
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

        # （Optional）Average num_envs： If you want --num_envs means “Total envs”
        # if args_cli.num_envs is not None:
        #     # world_size = GPUs * nodes
        #     world_size = app_launcher.world_size
        #     per_rank_envs = max(1, args_cli.num_envs // world_size)
        #     env_cfg.scene.num_envs = per_rank_envs

    # set recovery mode
    if args_cli.recovery_mode:
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 1.0),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        env_cfg.rewards.upward.weight = 0.5
        env_cfg.terminations.illegal_contact = None
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video and IS_MASTER:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "enable_wandb": (agent_cfg.logger == "wandb"),
            "wandb_key": "train/video",
            "video_resolution": (1280, 720),
            "video_crf": 30,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if args_cli.debug:
        import time
        print("\n========== [OBSERVATION / ACTION SHAPE INFO] ==========", flush=True)
        try:
            obs, _ = runner.env.reset()
            if isinstance(obs, dict):
                total_shape = sum([v.numel() for v in obs.values()])
                print(f"[OBS] Total shape (dict): {total_shape}", flush=True)
                for k, v in obs.items():
                    print(f"  - {k:20s}: shape = {tuple(v.shape)}", flush=True)
            elif isinstance(obs, torch.Tensor):
                print(f"[OBS] shape: {tuple(obs.shape)}", flush=True)
            else:
                print(f"[OBS] type={type(obs)}, content={obs}", flush=True)
            # print action vector shape
            action_tensor = env.unwrapped.action_manager.action
            print(f"[ACTION] shape: {tuple(action_tensor.shape)}", flush=True)
        except Exception as e:
            print(f"[WARN] Cannot access shape info: {e}", flush=True)
        print("========================================================\n", flush=True)

        obs_action_info = {"observation_groups": {}, "action_groups": {}}

        print("\n========== [OBS GROUP MEMBERS LIST & INFO] ==========", flush=True)
        try:
            obs_mgr = runner.env.env.unwrapped.observation_manager
            for group_name, term_names in obs_mgr._group_obs_term_names.items():
                print(f"[OBS GROUP] {group_name}: {term_names}", flush=True)
                obs_action_info["observation_groups"][group_name] = []

                for idx, name in enumerate(term_names):
                    term_cfg = obs_mgr._group_obs_term_cfgs[group_name][idx]
                    shape = obs_mgr._group_obs_term_dim[group_name][idx]
                    func_name = getattr(term_cfg.func, '__name__', str(term_cfg.func))
                    noise_type = type(term_cfg.noise).__name__ if term_cfg.noise else None

                    info = {
                        "name": name,
                        "shape": shape,
                        "func": func_name,
                        "history_length": term_cfg.history_length,
                        "flatten_history_dim": term_cfg.flatten_history_dim,
                        "clip": term_cfg.clip,
                        "scale": term_cfg.scale,
                        "noise": noise_type
                    }

                    obs_action_info["observation_groups"][group_name].append(info)

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


        print("\n====== [Action Vector Mapping] ======", flush=True)
        try:
            idx = 0
            for group_name, term in runner.env.unwrapped.action_manager._terms.items():
                action_group = {
                    "action_dim": term.action_dim,
                    "joint_names": getattr(term, "_joint_names", [f"joint_{i}" for i in range(term.action_dim)])
                }
                obs_action_info["action_groups"][group_name] = action_group
                print(f"[ACTION GROUP] {group_name}", flush=True)
                joint_names = getattr(term, "_joint_names", [f"joint_{i}" for i in range(term.action_dim)])
                term_actions = runner.env.unwrapped.action_manager.action[0, idx : idx + term.action_dim].cpu().numpy()
                for i, val in enumerate(term_actions):
                    joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
                    print(f"  action[{idx+i:02d}] {joint_name:>12s}: {val:+.4f}", flush=True)
                idx += term.action_dim
        except Exception as e:
            print(f"[WARN] Action manager info not available: {e}", flush=True)
        print("=====================================\n", flush=True)
        safe_obs_action_info = make_serializable(obs_action_info)
        dump_yaml(os.path.join(log_dir, "params", "obs_action.yaml"), safe_obs_action_info)
        dump_pickle(os.path.join(log_dir, "params", "obs_action.pkl"), safe_obs_action_info)
        time.sleep(0.1)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # Optional: Force commit
    try:
        import wandb
        if wandb and wandb.run is not None:
            wandb.log({}, commit=True)
            wandb.finish()
    except Exception:
        pass
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
