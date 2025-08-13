# ==============================================================================
# Zero-Pose Inspector (robot_lab friendly)
# ==============================================================================

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="Keep initial pose (no gravity) and print observations.")
parser.add_argument("--task", type=str, required=True,
                    help="Gym task name, e.g. RobotLab-Isaac-Velocity-Rough-CUHKLRL-SiriusW-v0")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs")
parser.add_argument("--steps", type=int, default=10000, help="How many simulation steps to run")
parser.add_argument("--print_every", type=int, default=50, help="Print observation every N steps")
parser.add_argument("--save_obs", action="store_true", help="Save last observation to logs/obs_latest.pkl")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable Fabric and use USD I/O operations.")
parser.add_argument("--disable_collision", action="store_true",
                    help="Disable collisions for all scene actors (if supported).")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---------------- Launch App ----------------
app = AppLauncher(args)
simulation_app = app.app

# ---------------- Imports after app launch ----------------
import gymnasium as gym
import torch
import pickle
from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_tasks  # noqa: F401
import robot_lab       # noqa: F401
import robot_lab.tasks # noqa: F401


def _resolve_task(name: str) -> str:
    def _ok(n):
        try:
            gym.spec(n)
            return True
        except Exception:
            return False
    if _ok(name):
        return name
    if not any(name.endswith(f"-v{i}") for i in range(10)):
        cand = f"{name}-v0"
        if _ok(cand):
            print(f"[INFO] Using '{cand}' (auto-added -v0).")
            return cand
    raise RuntimeError(f"Task '{name}' not found. Make sure robot_lab is imported and name matches registration.")

def _try_disable_events(env_cfg):
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        try:
            for k in list(env_cfg.events.__dict__.keys()):
                setattr(env_cfg.events, k, None)
            print("[INFO] Disabled env random events (if any).")
        except Exception:
            pass

def _print_obs(step_idx, obs):
    print(f"\n[OBS @ step {step_idx}]")
    if isinstance(obs, dict):
        # (num_envs, dim)
        for k, v in obs.items():
            try:
                shape = tuple(v.shape)
                head = v[0].detach().cpu() if torch.is_tensor(v) else v[0]
                print(f"  - {k:20s} shape={shape} first_env_sample[0:8]={head.flatten()[:8]}")
            except Exception:
                print(f"  - {k}: type={type(v)}")
    elif torch.is_tensor(obs):
        print(f"  Tensor shape={tuple(obs.shape)} first_env_sample[0:16]={obs[0].detach().cpu()[:16]}")
    else:
        print(f"  type={type(obs)} value={obs}")
def _try_disable_collision(env_cfg):
    if hasattr(env_cfg, "scene"):
        try:
            for actor_name, actor_cfg in env_cfg.scene.items():
                if hasattr(actor_cfg, "disable_collisions"):
                    actor_cfg.disable_collisions = True
            print("[INFO] Disabled collisions in scene config.")
        except Exception as e:
            print(f"[WARN] Could not disable collisions: {e}")
def main():
    task_name = _resolve_task(args.task)

    # config env ：gravity=0，disable events
    env_cfg = parse_env_cfg(
        task_name,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )

    # gravity = 0
    if hasattr(env_cfg.sim, "gravity"):
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)
        print("[INFO] Gravity set to (0, 0, 0).")

    # close base_velocity visualization
    try:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = False
            if hasattr(env_cfg.commands.base_velocity, "preview_in_viewer"):
                env_cfg.commands.base_velocity.preview_in_viewer = False
    except Exception:
        pass

    _try_disable_events(env_cfg)

    env = gym.make(task_name, cfg=env_cfg)

    print(f"[INFO] Observation space: {env.observation_space}")
    print(f"[INFO] Action space     : {env.action_space}")

    obs, _ = env.reset()
    _print_obs(step_idx=0, obs=obs)

    last_obs = obs
    for t in range(1, args.steps + 1):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            # (num_envs, action_dim)
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            last_obs, _, _, _, _ = env.step(actions)

        # if t % args.print_every == 0:
        #     _print_obs(t, last_obs)

    if args.save_obs:
        os.makedirs("logs", exist_ok=True)
        with open(os.path.join("logs", "obs_latest.pkl"), "wb") as f:
            pickle.dump(last_obs, f)
        print("[INFO] Saved last observation to logs/obs_latest.pkl")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
