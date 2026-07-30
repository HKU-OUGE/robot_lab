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
import hashlib
import json
import os
import sys
import importlib.metadata as metadata
import platform
from packaging import version
from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args
from algorithm_checkpoint import install_algorithm_checkpoint_state_hook


def _setup_hash_bound_wandb_config(logger: str) -> None:
    """Load one supervisor-provided W&B config without an ambiguous env sequence.

    W&B 0.22 validates ``config_paths`` as ``Sequence[str]`` but environment
    variables are necessarily strings.  The workflow therefore transports a
    non-W&B path plus SHA256 and supplies the required one-element sequence to
    ``wandb.setup`` programmatically before RSL-RL calls ``wandb.init``.
    """
    path_text = os.environ.get("ROBOT_LAB_WANDB_CONFIG_PATH", "")
    expected_sha = os.environ.get("ROBOT_LAB_WANDB_CONFIG_SHA256", "")
    if not path_text and not expected_sha:
        return
    if logger != "wandb" or not path_text or not expected_sha:
        raise RuntimeError("hash-bound W&B config transport is incomplete")
    if not os.path.isabs(path_text):
        raise RuntimeError("hash-bound W&B config path must be absolute")
    config_path = os.path.realpath(path_text)
    if not os.path.isfile(config_path):
        raise RuntimeError("hash-bound W&B config file is missing")
    if os.stat(config_path).st_mode & 0o222:
        raise RuntimeError("hash-bound W&B config file must be read-only")
    with open(config_path, "rb") as stream:
        actual_sha = hashlib.sha256(stream.read()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError("hash-bound W&B config SHA mismatch")

    import wandb

    wandb.setup(settings=wandb.Settings(config_paths=[config_path]))
    print(f"[INFO] Hash-bound W&B config installed: {config_path} sha256={actual_sha}")
    if os.environ.get("ROBOT_LAB_WANDB_PREINITIALIZE", "") == "1":
        required = {
            "entity": os.environ.get("WANDB_ENTITY", ""),
            "project": os.environ.get("WANDB_PROJECT", ""),
            "name": os.environ.get("ROBOT_LAB_WANDB_RUN_NAME", ""),
            "group": os.environ.get("WANDB_RUN_GROUP", ""),
            "id": os.environ.get("WANDB_RUN_ID", ""),
            "ready_path": os.environ.get("ROBOT_LAB_WANDB_READY_PATH", ""),
        }
        if any(not value for value in required.values()):
            raise RuntimeError("single-run W&B preinitialization environment is incomplete")
        run = wandb.init(
            entity=required["entity"],
            project=required["project"],
            name=required["name"],
            group=required["group"],
            id=required["id"],
            resume="never",
        )
        if run is None or run.id != required["id"] or not run.url:
            raise RuntimeError("W&B online initialization returned no bound run id/URL")
        ready_path = os.path.realpath(required["ready_path"])
        if not os.path.isabs(ready_path):
            raise RuntimeError("W&B ready witness path must be absolute")
        temporary = ready_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": 1,
                    "run_id": run.id,
                    "run_url": run.url,
                    "run_name": required["name"],
                    "group": required["group"],
                    "online_initialized_before_first_optimizer_update": True,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        os.replace(temporary, ready_path)
        print(f"[INFO] W&B online preinitialization passed: id={run.id} url={run.url}")

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
parser.add_argument(
    "--highstep_v18_environment_profile_manifest",
    type=str,
    default=None,
    help="Hash-bound v1.8 Stage-A/Stage-B environment profile manifest.",
)
parser.add_argument(
    "--highstep_v18_environment_profile_manifest_sha256",
    type=str,
    default=None,
    help="Expected SHA256 for --highstep_v18_environment_profile_manifest.",
)
parser.add_argument(
    "--highstep_v18_schedule_anchor_effective_update",
    type=int,
    default=None,
    help=(
        "For the one preregistered Stage-A to Stage-B environment transition, "
        "anchor the new environment schedule at the full checkpoint's effective Student update."
    ),
)
parser.add_argument(
    "--highstep_v18_stage_transition_manifest",
    type=str,
    default=None,
    help="Read-only v1.8 environment-only Stage-A to Stage-B transition manifest.",
)
parser.add_argument(
    "--highstep_v18_stage_transition_manifest_sha256",
    type=str,
    default=None,
    help="Expected SHA256 for the v1.8 Stage-A to Stage-B transition manifest.",
)
parser.add_argument(
    "--frozen_diagnostic_output",
    type=str,
    default=None,
    help=(
        "Run one read-only frozen training-buffer diagnostic after full checkpoint/runtime binding, "
        "write its manifest here, and exit without runner.learn() or optimizer.step()."
    ),
)
parser.add_argument(
    "--frozen_diagnostic_burn_in_steps",
    type=int,
    default=480,
    help="Read-only policy rollout steps used to stationarize the fresh simulator before one captured buffer.",
)
parser.add_argument(
    "--frozen_fc_mu_diagnostic_output",
    type=str,
    default=None,
    help="Run the preregistered read-only fc_mu/observability diagnostic and write its immutable manifest.",
)
parser.add_argument(
    "--frozen_fc_mu_diagnostic_preregistration",
    type=str,
    default=None,
    help="Immutable preregistration JSON for --frozen_fc_mu_diagnostic_output.",
)
parser.add_argument("--v114_frozen_gate_dataset", type=str, default=None)
parser.add_argument("--v114_frozen_gate_report", type=str, default=None)
parser.add_argument("--v114_create_frozen_gate_dataset", action="store_true", default=False)
parser.add_argument("--v114_expected_frozen_gate_dataset_sha256", type=str, default=None)
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument(
    "--highstep_resume_mode",
    type=str,
    default="auto",
    choices=("auto", "refine", "migration"),
    help=(
        "Controls highstep resume gate handling. 'refine' immediately relaxes "
        "highstep staged rewards/command gate, 'migration' keeps the staged "
        "curriculum for bodyflat/sidestep -> highstep transfer, and 'auto' "
        "infers this from the checkpoint path."
    ),
)
parser.add_argument(
    "--highstep_checkpoint_load_mode",
    type=str,
    default="full",
    choices=("full", "weights_only"),
    help=(
        "Controls checkpoint state restoration for highstep refinement. "
        "'full' restores policy, optimizer, and learning iteration; "
        "'weights_only' restores policy weights into a fresh optimizer and resets the runner iteration to zero; "
        "its independent schedule decision is required via --highstep_schedule_resume_mode."
    ),
)
parser.add_argument(
    "--highstep_schedule_resume_mode",
    type=str,
    default="auto",
    choices=("auto", "preserve", "reset"),
    help=(
        "Controls the process-independent high-step action/reward/curriculum schedule. "
        "For a full checkpoint load, auto preserves same-task refinement and resets a migration. "
        "A weights_only load must explicitly choose preserve or reset; auto is rejected."
    ),
)
parser.add_argument(
    "--highstep_absolute_runner_step_origin",
    type=int,
    default=None,
    help=(
        "Exact absolute runner/W&B step origin for the preregistered B300 0707-derived "
        "single-run route. It does not alter the relative Student distillation counter."
    ),
)
parser.add_argument(
    "--allow_legacy_highstep_schedule_fallback",
    action="store_true",
    default=False,
    help=(
        "Explicitly allow a verified legacy full checkpoint with no highstep schedule sidecar "
        "to use its checkpoint iteration as the schedule update. Never valid for weights_only+preserve."
    ),
)
parser.add_argument(
    "--highstep_parent_teacher_manifest",
    type=str,
    default=None,
    help=(
        "Required for the first high-step Student run. Pins the schema-v4 robust-Teacher "
        "evaluation manifest and its selected checkpoint; Student continuations inherit it."
    ),
)
parser.add_argument(
    "--legacy_student_distill_update_count",
    type=int,
    default=None,
    help=(
        "Explicit migration value for a verified legacy Student checkpoint that predates VAEPPO "
        "optimizer/count persistence. Only valid with a full checkpoint load. The unavailable VAE "
        "Adam moments remain fresh, so this is compatible but not a seamless resume."
    ),
)
parser.add_argument(
    "--verified_legacy_teacher_algorithm_state_manifest",
    type=str,
    default=None,
    help=(
        "Read-only, SHA-bound proof that one exact legacy Stage-1 Teacher checkpoint has a fully "
        "restorable PPO Adam and a provably never-stepped empty VAE Adam. This is not a generic override."
    ),
)
parser.add_argument(
    "--verified_legacy_teacher_algorithm_state_manifest_sha256",
    type=str,
    default=None,
    help="Expected SHA256 for --verified_legacy_teacher_algorithm_state_manifest.",
)

# ==========================================
# 🌟 修改点：在 argparse 中明确 agent 的 choices
# ==========================================
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point",
                    help="Name of the RL agent configuration entry point. Can be 'symmetric_ppo_cfg'.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# ---- torchrun read distributed----
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = (WORLD_SIZE > 1) or bool(args_cli.distributed)
IS_MASTER = (LOCAL_RANK == 0)

# enable camera only on master
if args_cli.enable_cameras:
    args_cli.enable_cameras = True
elif args_cli.video:
    args_cli.enable_cameras = bool(args_cli.video and IS_MASTER)
# always enable cameras to record video


# disable W&B in slaves
if IS_DISTRIBUTED and not IS_MASTER:
    os.environ["WANDB_MODE"] = "disabled"  # 等价于 wandb.init(mode="disabled")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
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
import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401
from robot_lab.tasks.locomotion.velocity.mdp.highstep_schedule import (
    RUNTIME_STATE_NAME,
    SCHEDULE_MANIFEST_NAME,
    ScheduleContinuityError,
    append_runtime_snapshot,
    assert_schedule_definition_compatible,
    build_schedule_manifest,
    checkpoint_sha256,
    checkpoint_iteration_from_mapping,
    global_update,
    install_global_update,
    load_source_manifest,
    resolve_student_parent_lineage,
    resolve_schedule_resume_mode,
    resolve_loaded_schedule_update,
    schedule_definition_from_configs,
    write_manifest,
)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
# --- MoE Actor that can drop-in replace the base policy's actor MLP ---
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        assert num_experts >= 1
        assert 1 <= topk <= num_experts
        self.num_experts = num_experts
        self.topk = topk
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.experts = nn.ModuleList([_make_mlp(in_dim, hidden, out_dim, act) for _ in range(num_experts)])
        # 一个小 gating MLP（与基类 actor 的规模同一量级即可）
        gate_hidden = max(64, (hidden[0] if hidden else 128) // 2)
        self.gate = _make_mlp(in_dim, [gate_hidden], num_experts, act)
        # 暴露一个路由熵指标，便于 wandb 打点
        self.last_router_probs: torch.Tensor | None = None

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
        self.last_router_probs = probs
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
    @property
    def routing_entropy(self):
        p = getattr(self.actor, "last_router_probs", None)
        if p is None:
            return None
        eps = 1e-8
        return (-(p * (p + eps).log()).sum(dim=-1)).mean()
# 让 Isaac Lab 用 getattr(...) 能找到：rsl_rl.modules.actor_critic.ActorCriticMoE
import rsl_rl.modules.actor_critic as ac
ac.ActorCriticMoE = ActorCriticMoE
import rsl_rl.runners.on_policy_runner as _opr
_opr.ActorCriticMoE = ActorCriticMoE


def _resolve_highstep_resume_mode(args_cli, agent_cfg):
    """Infer whether --resume is a highstep refine or a base-locomotion migration."""
    requested_mode = getattr(args_cli, "highstep_resume_mode", "auto")
    if requested_mode != "auto":
        return requested_mode, f"explicit --highstep_resume_mode={requested_mode}"

    source_text = " ".join(
        str(item or "") for item in (getattr(agent_cfg, "load_checkpoint", None), getattr(agent_cfg, "load_run", None))
    )
    source_text_lower = source_text.lower()

    if "bodyflat" in source_text_lower or "sidestep" in source_text_lower:
        return "migration", "checkpoint/load_run looks like a base locomotion policy"
    if "highstep" in source_text_lower:
        return "refine", "checkpoint/load_run looks like an existing highstep policy"

    checkpoint = getattr(agent_cfg, "load_checkpoint", None)
    if checkpoint and os.path.isabs(str(checkpoint)):
        return "migration", "absolute checkpoint path is not recognized as highstep; keeping staged curriculum"

    return "refine", "relative/default highstep resume is assumed to be same-task refine"


def _read_rsl_checkpoint_iteration(checkpoint_path: str) -> int:
    """Read the checkpoint clock before the RSL wrapper performs its initial reset."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as safe_load_error:
        print(
            "[WARN] weights_only checkpoint metadata read failed; retrying the trusted local checkpoint "
            f"with the legacy loader: {safe_load_error}"
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        return checkpoint_iteration_from_mapping(checkpoint)
    finally:
        del checkpoint


def _read_student_recovery_effective_update(checkpoint_path: str) -> int:
    """Read the algorithm-owned Student update count for a v1.8 full resume."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        recovery = (
            checkpoint.get("infos", {})
            .get("robot_lab_algorithm_checkpoint_state", {})
            .get("student_recovery")
        )
        if not isinstance(recovery, dict):
            raise RuntimeError("v1.8 transition source lacks Student recovery state")
        value = int(recovery.get("effective_update_count", -1))
        if value < 0:
            raise RuntimeError("v1.8 transition source has invalid effective update")
        return value
    finally:
        del checkpoint


def _load_verified_legacy_teacher_algorithm_state_contract(
    *, manifest_path: str, manifest_sha256: str, checkpoint_path: str, task: str
) -> dict:
    """Load the one narrow Stage-1 Teacher legacy-state proof used by v1.12."""
    resolved_manifest = os.path.realpath(str(manifest_path))
    resolved_checkpoint = os.path.realpath(str(checkpoint_path))
    if not os.path.isabs(resolved_manifest) or not os.path.isfile(resolved_manifest):
        raise RuntimeError("Verified legacy Teacher algorithm-state manifest path is invalid")
    if checkpoint_sha256(resolved_manifest) != str(manifest_sha256):
        raise RuntimeError("Verified legacy Teacher algorithm-state manifest SHA256 mismatch")
    with open(resolved_manifest, encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise RuntimeError("Verified legacy Teacher algorithm-state manifest must contain one object")
    if not (
        payload.get("kind") == "highstep_v112_verified_legacy_teacher_algorithm_state_migration"
        and payload.get("workflow_id") == "highstep_teacher_rear_support_v112_20260715"
        and payload.get("authority_version") == "v1.12"
        and payload.get("task") == task
        and payload.get("training_semantics_changed") is False
        and payload.get("weights_only") is False
        and payload.get("fresh_ppo_optimizer") is False
    ):
        raise RuntimeError("Verified legacy Teacher algorithm-state authority contract mismatch")

    contract = payload.get("algorithm_state_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Verified legacy Teacher manifest lacks algorithm_state_contract")
    if os.path.realpath(str(contract.get("checkpoint_path", ""))) != resolved_checkpoint:
        raise RuntimeError("Verified legacy Teacher manifest is bound to a different checkpoint path")
    if checkpoint_sha256(resolved_checkpoint) != contract.get("checkpoint_sha256"):
        raise RuntimeError("Verified legacy Teacher checkpoint SHA256 mismatch")

    evidence = payload.get("source_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("Verified legacy Teacher manifest lacks source evidence")
    for item in evidence:
        if not isinstance(item, dict):
            raise RuntimeError("Verified legacy Teacher source evidence entry is invalid")
        evidence_path = os.path.realpath(str(item.get("path", "")))
        if not os.path.isabs(evidence_path) or not os.path.isfile(evidence_path):
            raise RuntimeError("Verified legacy Teacher source evidence path is invalid")
        if checkpoint_sha256(evidence_path) != item.get("sha256"):
            raise RuntimeError("Verified legacy Teacher source evidence SHA256 changed")
    return dict(contract)


def _tensor_or_sequence_list(value) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    result = [float(item) for item in value]
    if not result or not all(torch.isfinite(torch.tensor(result)).tolist()):
        raise ScheduleContinuityError(f"Invalid command curriculum vector: {result}")
    return result


def _stage_b_tensor_rows(label: str, value, expected, *, num_envs: int) -> list:
    """Validate an initialized per-environment tensor and return its first row for the manifest."""
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Stage B runtime {label} is not a tensor: {type(value).__name__}")
    actual = value.detach().to(device="cpu", dtype=torch.float64)
    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    expected_shape = (num_envs, *expected_tensor.shape)
    if tuple(actual.shape) != expected_shape:
        raise RuntimeError(
            f"Stage B runtime {label} shape changed: {tuple(actual.shape)} != {expected_shape}"
        )
    if not bool(torch.all(torch.isfinite(actual)).item()):
        raise RuntimeError(f"Stage B runtime {label} contains non-finite values")
    expected_rows = expected_tensor.unsqueeze(0).expand_as(actual)
    if not torch.allclose(actual, expected_rows, rtol=0.0, atol=1.0e-7):
        max_delta = float(torch.max(torch.abs(actual - expected_rows)).item())
        raise RuntimeError(f"Stage B runtime {label} changed (max_abs_delta={max_delta})")
    return actual[0].tolist()


def _assert_stage_b_runtime_contract(
    env, policy, *, allow_critical_transition_context: bool = False,
    critical_transition_context_dim: int = 4,
) -> dict:
    """Fail closed unless the initialized Stage B action/robot/observation contract is canonical."""
    expected_joint_order = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    expected_joint_ids = list(range(16))
    expected_action_scale = [0.1] * 12 + [0.02] * 4
    expected_default_joint_pos = [0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4
    expected_action_clip = [[-60.0, 60.0] for _ in expected_joint_order]

    runtime_env = env.unwrapped
    num_envs = int(runtime_env.num_envs)
    action_manager = getattr(runtime_env, "action_manager", None)
    if action_manager is None:
        raise RuntimeError("Stage B runtime action manager is unavailable")
    action_terms = getattr(action_manager, "_terms", None)
    if not isinstance(action_terms, dict) or list(action_terms) != ["joint_pos"]:
        raise RuntimeError(
            f"Stage B runtime action terms changed: {list(action_terms) if isinstance(action_terms, dict) else action_terms}"
        )
    action_term = action_terms["joint_pos"]
    action_joint_names = list(getattr(action_term, "_joint_names", []))
    raw_joint_ids = getattr(action_term, "_joint_ids", None)
    if isinstance(raw_joint_ids, torch.Tensor):
        action_joint_ids = [int(item) for item in raw_joint_ids.detach().cpu().flatten().tolist()]
    elif isinstance(raw_joint_ids, (list, tuple)):
        action_joint_ids = [int(item) for item in raw_joint_ids]
    else:
        raise RuntimeError(f"Stage B runtime action joint IDs are unresolved: {raw_joint_ids!r}")
    if action_joint_names != expected_joint_order:
        raise RuntimeError(f"Stage B runtime action joint order changed: {action_joint_names}")
    if action_joint_ids != expected_joint_ids:
        raise RuntimeError(f"Stage B runtime action joint IDs changed: {action_joint_ids}")
    if int(getattr(action_term, "action_dim", -1)) != 16:
        raise RuntimeError(f"Stage B runtime action dimension changed: {getattr(action_term, 'action_dim', None)}")
    if int(getattr(action_manager, "total_action_dim", -1)) != 16:
        raise RuntimeError(
            f"Stage B runtime total action dimension changed: {getattr(action_manager, 'total_action_dim', None)}"
        )

    robot = runtime_env.scene["robot"]
    robot_joint_names = list(getattr(robot, "joint_names", []))
    if robot_joint_names != expected_joint_order:
        raise RuntimeError(f"Stage B runtime robot joint order changed: {robot_joint_names}")
    if getattr(action_term, "_asset", None) is not robot:
        raise RuntimeError("Stage B runtime action term is not bound to the robot articulation")

    runtime_scale = _stage_b_tensor_rows(
        "action scale", getattr(action_term, "_scale", None), expected_action_scale, num_envs=num_envs
    )
    runtime_default_joint_pos = _stage_b_tensor_rows(
        "robot default_joint_pos",
        robot.data.default_joint_pos[:, action_joint_ids],
        expected_default_joint_pos,
        num_envs=num_envs,
    )
    runtime_offset = _stage_b_tensor_rows(
        "action offset", getattr(action_term, "_offset", None), expected_default_joint_pos, num_envs=num_envs
    )
    if not torch.equal(
        action_term._offset.detach(),
        robot.data.default_joint_pos[:, action_joint_ids].detach(),
    ):
        raise RuntimeError("Stage B runtime action offset is not the robot's actual default_joint_pos")
    runtime_clip = _stage_b_tensor_rows(
        "action clip", getattr(action_term, "_clip", None), expected_action_clip, num_envs=num_envs
    )

    expected_observations = {
        "policy": {
            "terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"],
            "dims": [(30,), (30,), (30,), (160,), (160,), (160,)],
            "history": [10, 10, 10, 10, 10, 10],
            "group_dim": (570,),
        },
        "estimator": {
            "terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"],
            "dims": [(30,), (30,), (30,), (160,), (160,), (160,)],
            "history": [10, 10, 10, 10, 10, 10],
            "group_dim": (570,),
        },
        "critic": {
            "terms": [
                "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos", "joint_vel", "actions", "height_scan",
            ],
            "dims": [(3,), (3,), (3,), (3,), (16,), (16,), (16,), (102,)],
            "history": [0, 0, 0, 0, 0, 0, 0, 0],
            "group_dim": (162,),
        },
    }
    if allow_critical_transition_context:
        expected_observations["critical_transition"] = {
            "terms": ["stage_contact_context"],
            "dims": [(critical_transition_context_dim,)],
            "history": [0],
            "group_dim": (critical_transition_context_dim,),
        }
    observation_manager = getattr(runtime_env, "observation_manager", None)
    if observation_manager is None:
        raise RuntimeError("Stage B runtime observation manager is unavailable")
    actual_groups = list(getattr(observation_manager, "_group_obs_term_names", {}))
    if actual_groups != list(expected_observations):
        raise RuntimeError(f"Stage B runtime observation groups changed: {actual_groups}")

    observation_manifest = {}
    for group_name, expected in expected_observations.items():
        term_names = list(observation_manager._group_obs_term_names[group_name])
        term_dims = [tuple(int(item) for item in dim) for dim in observation_manager._group_obs_term_dim[group_name]]
        term_cfgs = list(observation_manager._group_obs_term_cfgs[group_name])
        history = [int(term_cfg.history_length) for term_cfg in term_cfgs]
        flatten_history = [bool(term_cfg.flatten_history_dim) for term_cfg in term_cfgs]
        group_dim = tuple(int(item) for item in observation_manager._group_obs_dim[group_name])
        if term_names != expected["terms"]:
            raise RuntimeError(f"Stage B runtime {group_name} observation order changed: {term_names}")
        if term_dims != expected["dims"] or group_dim != expected["group_dim"]:
            raise RuntimeError(
                f"Stage B runtime {group_name} observation dimensions changed: terms={term_dims}, group={group_dim}"
            )
        if history != expected["history"] or flatten_history != [True] * len(term_names):
            raise RuntimeError(
                f"Stage B runtime {group_name} observation history changed: "
                f"lengths={history}, flatten={flatten_history}"
            )
        if not bool(observation_manager._group_obs_concatenate[group_name]):
            raise RuntimeError(f"Stage B runtime {group_name} observations are no longer concatenated")
        history_buffers = observation_manager._group_obs_term_history_buffer[group_name]
        expected_buffer_terms = {
            name for name, history_length in zip(term_names, history, strict=True) if history_length > 0
        }
        if set(history_buffers) != expected_buffer_terms:
            raise RuntimeError(
                f"Stage B runtime {group_name} history buffers changed: {list(history_buffers)}"
            )
        for term_name in expected_buffer_terms:
            if int(history_buffers[term_name].max_length) != 10:
                raise RuntimeError(
                    f"Stage B runtime {group_name}/{term_name} history buffer length changed: "
                    f"{history_buffers[term_name].max_length}"
                )
        observation_manifest[group_name] = {
            "term_names": term_names,
            "term_dims": [list(dim) for dim in term_dims],
            "history_length": history,
            "flatten_history_dim": flatten_history,
            "group_dim": list(group_dim),
        }

    expected_policy_keys = {
        "policy_keys": ["policy"],
        "estimator_keys": ["estimator"],
        "critic_keys": ["critic"],
    }
    for attribute, expected_keys in expected_policy_keys.items():
        actual_keys = list(getattr(policy, attribute, []))
        if actual_keys != expected_keys:
            raise RuntimeError(f"Stage B runtime policy {attribute} changed: {actual_keys}")
    if int(getattr(policy, "policy_obs_dim", -1)) != 570:
        raise RuntimeError(f"Stage B runtime policy observation dimension changed: {policy.policy_obs_dim}")
    actual_policy_groups = {
        key: list(value) for key, value in getattr(policy, "obs_groups", {}).items()
    }
    if allow_critical_transition_context and any(
        "critical_transition" in values for values in actual_policy_groups.values()
    ):
        raise RuntimeError("critical-transition labels leaked into a policy observation route")

    return {
        "validated": True,
        "action": {
            "term_names": ["joint_pos"],
            "joint_names": action_joint_names,
            "joint_ids": action_joint_ids,
            "scale": runtime_scale,
            "offset": runtime_offset,
            "clip": runtime_clip,
        },
        "robot": {
            "joint_names": robot_joint_names,
            "default_joint_pos": runtime_default_joint_pos,
        },
        "observations": observation_manifest,
        "policy_observation_keys": expected_policy_keys,
        "policy_obs_dim": int(policy.policy_obs_dim),
    }


_R2_PREREGISTRATION_PATH = (
    "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v11_20260712/"
    "r2_preregistration.json"
)
_R2_PREREGISTRATION_SHA256 = "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
_R2_SPEC_PATH = (
    "/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/"
    "highstep_student_recovery_spec_20260712.md"
)
_R2_SPEC_SHA256 = "7dbb47d76b5eafab0e425be7ac20b5a0e9487466c6512831600f22fb7509bbe5"


def _load_r2_preregistration_contract(policy) -> tuple[dict, dict]:
    """Load and independently bind the immutable R2 authority artifacts."""
    configured_path = os.path.realpath(
        str(getattr(policy, "student_recovery_r2_preregistration_path", ""))
    )
    configured_sha = str(
        getattr(policy, "student_recovery_r2_preregistration_sha256", "")
    )
    if configured_path != os.path.realpath(_R2_PREREGISTRATION_PATH):
        raise RuntimeError(
            "R2 preregistration path differs from the approved immutable artifact: "
            f"{configured_path!r}"
        )
    if configured_sha != _R2_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "R2 policy preregistration SHA256 differs from the approved immutable SHA"
        )
    if not os.path.isfile(configured_path):
        raise RuntimeError(f"R2 preregistration is unavailable: {configured_path}")
    if os.stat(configured_path).st_mode & 0o222:
        raise RuntimeError("R2 preregistration must be read-only before any parameter update")
    actual_prereg_sha = checkpoint_sha256(configured_path)
    if actual_prereg_sha != _R2_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "R2 preregistration SHA256 mismatch: "
            f"expected {_R2_PREREGISTRATION_SHA256}, got {actual_prereg_sha}"
        )
    with open(configured_path, encoding="utf-8") as preregistration_file:
        preregistration = json.load(preregistration_file)
    if not isinstance(preregistration, dict):
        raise RuntimeError("R2 preregistration root must be a JSON object")
    if preregistration.get("schema_version") != 1:
        raise RuntimeError("R2 preregistration schema_version must be 1")
    if preregistration.get("kind") != "highstep_student_recovery_r2_preregistration":
        raise RuntimeError("R2 preregistration kind changed")
    if preregistration.get("workflow_id") != "highstep_student_recovery_v11_20260712":
        raise RuntimeError("R2 preregistration workflow_id changed")
    if preregistration.get("immutable_before_parameter_update") is not True:
        raise RuntimeError("R2 preregistration is not marked immutable before parameter update")
    route = preregistration.get("route")
    if not isinstance(route, dict) or route.get("selected") != "R2" or route.get("r1_excluded") is not True:
        raise RuntimeError("R2 preregistration route decision changed")

    authority = preregistration.get("authority")
    if not isinstance(authority, dict):
        raise RuntimeError("R2 preregistration authority is missing")
    spec_path = os.path.realpath(str(authority.get("spec_path", "")))
    spec_sha = str(authority.get("spec_sha256", ""))
    if spec_path != os.path.realpath(_R2_SPEC_PATH) or spec_sha != _R2_SPEC_SHA256:
        raise RuntimeError("R2 preregistration does not bind the approved v1.1.1 spec")
    actual_spec_sha = checkpoint_sha256(spec_path)
    if actual_spec_sha != _R2_SPEC_SHA256:
        raise RuntimeError(
            f"R2 spec SHA256 mismatch: expected {_R2_SPEC_SHA256}, got {actual_spec_sha}"
        )

    checkpoint_binding = preregistration.get("checkpoint_binding")
    if not isinstance(checkpoint_binding, dict):
        raise RuntimeError("R2 checkpoint binding is missing from preregistration")

    def bind_artifact(record_name: str, *, extra_path_key: str = "path", extra_sha_key: str = "sha256"):
        record = checkpoint_binding.get(record_name)
        if not isinstance(record, dict):
            raise RuntimeError(f"R2 preregistration is missing {record_name}")
        path = os.path.realpath(str(record.get(extra_path_key, "")))
        expected_sha = str(record.get(extra_sha_key, ""))
        if not os.path.isabs(path) or not os.path.isfile(path):
            raise RuntimeError(f"R2 bound artifact is unavailable for {record_name}: {path}")
        actual_sha = checkpoint_sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"R2 bound artifact SHA256 mismatch for {record_name}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        return path, actual_sha

    protected_root_path, protected_root_sha = bind_artifact("protected_student_root")
    behavior_anchor_path, behavior_anchor_sha = bind_artifact("behavior_start_and_anchor")
    teacher_path, teacher_sha = bind_artifact("frozen_teacher")
    teacher_yaml_path, teacher_yaml_sha = bind_artifact(
        "frozen_teacher", extra_path_key="env_yaml", extra_sha_key="env_yaml_sha256"
    )
    if len({protected_root_path, behavior_anchor_path, teacher_path}) != 3:
        raise RuntimeError("R2 protected root, B500 behavior anchor and Teacher must be distinct files")

    permanent_contract = preregistration.get("permanent_contract")
    if not isinstance(permanent_contract, dict):
        raise RuntimeError("R2 permanent action contract is missing")
    expected_joint_order = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    expected_scale = [0.1] * 12 + [0.02] * 4
    expected_clip = [[-60.0, 60.0] for _ in range(16)]
    if permanent_contract.get("joint_order") != expected_joint_order:
        raise RuntimeError("R2 preregistered joint order changed")
    if permanent_contract.get("action_scale") != expected_scale:
        raise RuntimeError("R2 preregistered action_scale changed")
    if permanent_contract.get("joint_pos_clip") != expected_clip:
        raise RuntimeError("R2 preregistered joint_pos.clip changed")
    if permanent_contract.get("use_default_offset") is not True:
        raise RuntimeError("R2 preregistered default-offset contract is disabled")
    if permanent_contract.get("network_and_deployment_observations_unchanged") is not True:
        raise RuntimeError("R2 preregistration no longer protects deployment observations")

    extra_group = preregistration.get("training_data", {}).get("extra_rollout_group")
    expected_context = {
        "name": "teacher_context",
        "shape": [3],
        "order": ["height_delta", "command_x", "front_rear_delta"],
        "policy_estimator_critic_input": False,
    }
    if not isinstance(extra_group, dict) or any(
        extra_group.get(key) != value for key, value in expected_context.items()
    ):
        raise RuntimeError("R2 teacher_context preregistration changed")

    binding = {
        "workflow_id": preregistration["workflow_id"],
        "preregistration_path": configured_path,
        "preregistration_sha256": actual_prereg_sha,
        "recovery_spec_path": spec_path,
        "recovery_spec_sha256": actual_spec_sha,
        "protected_student_root": protected_root_path,
        "protected_student_root_sha256": protected_root_sha,
        "behavior_start_and_anchor": behavior_anchor_path,
        "behavior_start_and_anchor_sha256": behavior_anchor_sha,
        "teacher_checkpoint": teacher_path,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_env_yaml": teacher_yaml_path,
        "teacher_env_yaml_sha256": teacher_yaml_sha,
    }
    return preregistration, binding


def _validate_r2_load_request(
    preregistration: dict,
    *,
    loaded_checkpoint: str,
    checkpoint_load_mode: str,
    schedule_resume_mode: str,
    resume_enabled: bool,
    legacy_student_count,
) -> None:
    """Fail closed on any R2 start/resume mode not approved by preregistration."""
    if not resume_enabled:
        raise RuntimeError("R2 requires an explicit bound checkpoint")
    if legacy_student_count is not None:
        raise RuntimeError("R2 forbids legacy Student count migration")
    if schedule_resume_mode != "preserve":
        raise RuntimeError("R2 fresh start and full resume both require schedule preserve")
    binding = preregistration["checkpoint_binding"]
    loaded_path = os.path.realpath(str(loaded_checkpoint))
    behavior_path = os.path.realpath(binding["behavior_start_and_anchor"]["path"])
    protected_paths = {
        os.path.realpath(binding["protected_student_root"]["path"]),
        behavior_path,
        os.path.realpath(binding["frozen_teacher"]["path"]),
    }
    if checkpoint_load_mode == "weights_only":
        if loaded_path != behavior_path:
            raise RuntimeError(
                f"Fresh R2 must load the preregistered B500 weights_only: {loaded_path} != {behavior_path}"
            )
        expected_sha = binding["behavior_start_and_anchor"]["sha256"]
        actual_sha = checkpoint_sha256(loaded_path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Fresh R2 B500 SHA256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        return
    if checkpoint_load_mode == "full":
        if loaded_path in protected_paths:
            raise RuntimeError(
                "R2 full resume requires an R2 checkpoint, not model_900, B500 or the Teacher"
            )
        return
    raise RuntimeError(f"Unsupported R2 checkpoint load mode: {checkpoint_load_mode!r}")


_R3_PREREGISTRATION_PATH = (
    "/home/lxq/Softwares/robot_lab/tmp/highstep_student_recovery_v12_20260713/"
    "r3_preregistration.json"
)
_R3_PREREGISTRATION_SHA256 = "13c184e4b6e2c3b514f52466dac1e27ff18256433716fd0c6b5d0890d93945e6"
_R3_SPEC_SHA256 = "053da1c6d30f9d5ed9aa4c2d5bedbab8d98130edb8a60f8e2657d4a8d0d99352"


def _load_r3_preregistration_contract(policy) -> tuple[dict, dict]:
    """Load the user-preapproved immutable v1.2 R3 authority."""
    configured_path = os.path.realpath(
        str(getattr(policy, "student_recovery_r3_preregistration_path", ""))
    )
    configured_sha = str(
        getattr(policy, "student_recovery_r3_preregistration_sha256", "")
    )
    if configured_path != os.path.realpath(_R3_PREREGISTRATION_PATH):
        raise RuntimeError("R3 preregistration path differs from the approved artifact")
    if configured_sha != _R3_PREREGISTRATION_SHA256:
        raise RuntimeError("R3 policy preregistration SHA differs from the approved SHA")
    if not os.path.isfile(configured_path) or os.stat(configured_path).st_mode & 0o222:
        raise RuntimeError("R3 preregistration must exist and remain read-only")
    actual_prereg_sha = checkpoint_sha256(configured_path)
    if actual_prereg_sha != _R3_PREREGISTRATION_SHA256:
        raise RuntimeError("R3 preregistration content SHA mismatch")
    with open(configured_path, encoding="utf-8") as handle:
        preregistration = json.load(handle)
    if (
        not isinstance(preregistration, dict)
        or preregistration.get("schema_version") != 1
        or preregistration.get("kind") != "highstep_student_recovery_r3_preregistration"
        or preregistration.get("workflow_id") != "highstep_student_recovery_v12_20260713"
        or preregistration.get("approved_version") != "v1.2"
        or preregistration.get("immutable_before_parameter_update") is not True
        or preregistration.get("route", {}).get("selected") != "R3"
    ):
        raise RuntimeError("R3 preregistration identity/approval changed")
    authority = preregistration.get("authority", {})
    spec_path = os.path.realpath(str(authority.get("spec_path", "")))
    if (
        str(authority.get("spec_sha256", "")) != _R3_SPEC_SHA256
        or checkpoint_sha256(spec_path) != _R3_SPEC_SHA256
    ):
        raise RuntimeError("R3 spec binding changed")

    def bind_file(record: dict, path_key="path", sha_key="sha256") -> tuple[str, str]:
        path = os.path.realpath(str(record.get(path_key, "")))
        expected_sha = str(record.get(sha_key, ""))
        if not os.path.isabs(path) or not os.path.isfile(path):
            raise RuntimeError(f"R3 bound artifact is unavailable: {path}")
        actual_sha = checkpoint_sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"R3 bound artifact SHA mismatch: {path}")
        return path, actual_sha

    report_path, report_sha = bind_file(
        authority, path_key="route_report", sha_key="route_report_sha256"
    )
    for record in preregistration.get("evidence", {}).values():
        bind_file(record)
    binding = preregistration.get("checkpoint_binding", {})
    required = (
        "protected_student_root",
        "behavior_reference_b500",
        "r3_start_and_anchor",
        "frozen_teacher",
    )
    resolved = {name: bind_file(binding.get(name, {})) for name in required}
    teacher_yaml, teacher_yaml_sha = bind_file(
        binding["frozen_teacher"], path_key="env_yaml", sha_key="env_yaml_sha256"
    )
    if len({path for path, _ in resolved.values()}) != len(required):
        raise RuntimeError("R3 root/B500/start/Teacher checkpoints must be distinct")
    permanent = preregistration.get("permanent_contract", {})
    if (
        permanent.get("joint_order")
        != [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
        ]
        or permanent.get("action_scale") != [0.1] * 12 + [0.02] * 4
        or permanent.get("joint_pos_clip") != [[-60.0, 60.0] for _ in range(16)]
        or permanent.get("use_default_offset") is not True
        or permanent.get("network_structure_unchanged") is not True
        or permanent.get("deployment_observations_unchanged") is not True
    ):
        raise RuntimeError("R3 permanent action/network contract changed")
    extra = preregistration.get("training_data", {}).get("extra_rollout_group", {})
    if extra != {
        "name": "teacher_context",
        "shape": [3],
        "order": ["height_delta", "command_x", "front_rear_delta"],
        "policy_estimator_critic_input": False,
    }:
        raise RuntimeError("R3 training-only Teacher context changed")
    return preregistration, {
        "workflow_id": preregistration["workflow_id"],
        "preregistration_path": configured_path,
        "preregistration_sha256": actual_prereg_sha,
        "recovery_spec_path": spec_path,
        "recovery_spec_sha256": _R3_SPEC_SHA256,
        "route_report": report_path,
        "route_report_sha256": report_sha,
        "protected_student_root": resolved["protected_student_root"][0],
        "protected_student_root_sha256": resolved["protected_student_root"][1],
        "behavior_reference_b500": resolved["behavior_reference_b500"][0],
        "behavior_reference_b500_sha256": resolved["behavior_reference_b500"][1],
        "r3_start_and_anchor": resolved["r3_start_and_anchor"][0],
        "r3_start_and_anchor_sha256": resolved["r3_start_and_anchor"][1],
        "teacher_checkpoint": resolved["frozen_teacher"][0],
        "teacher_checkpoint_sha256": resolved["frozen_teacher"][1],
        "teacher_env_yaml": teacher_yaml,
        "teacher_env_yaml_sha256": teacher_yaml_sha,
    }


def _validate_r3_load_request(
    preregistration: dict,
    *,
    loaded_checkpoint: str,
    checkpoint_load_mode: str,
    schedule_resume_mode: str,
    resume_enabled: bool,
    legacy_student_count,
) -> None:
    if not resume_enabled or legacy_student_count is not None:
        raise RuntimeError("R3 requires an explicit checkpoint and forbids legacy count migration")
    if schedule_resume_mode != "preserve":
        raise RuntimeError("R3 requires schedule preserve")
    binding = preregistration["checkpoint_binding"]
    loaded_path = os.path.realpath(str(loaded_checkpoint))
    start_path = os.path.realpath(binding["r3_start_and_anchor"]["path"])
    protected = {
        os.path.realpath(binding[name]["path"])
        for name in (
            "protected_student_root", "behavior_reference_b500", "r3_start_and_anchor", "frozen_teacher"
        )
    }
    if checkpoint_load_mode == "weights_only":
        if loaded_path != start_path:
            raise RuntimeError("Fresh R3 must weights-only load the preregistered R2-50")
        if checkpoint_sha256(loaded_path) != binding["r3_start_and_anchor"]["sha256"]:
            raise RuntimeError("Fresh R3 start checkpoint SHA mismatch")
        return
    if checkpoint_load_mode == "full":
        if loaded_path in protected:
            raise RuntimeError("R3 full resume requires an R3 checkpoint")
        return
    raise RuntimeError(f"Unsupported R3 checkpoint load mode: {checkpoint_load_mode!r}")


def _assert_r2_action_config_contract(action_cfg, preregistration: dict) -> dict:
    """Validate the unresolved action configuration before binding initialized tensors."""
    permanent_contract = preregistration["permanent_contract"]
    joint_order = list(getattr(action_cfg, "joint_names", []))
    scale = dict(getattr(action_cfg, "scale", {}))
    clip = dict(getattr(action_cfg, "clip", {}))
    if joint_order != permanent_contract["joint_order"]:
        raise RuntimeError(f"R2 action/joint order contract changed: {joint_order}")
    if scale != {
        ".*_box_joint": 0.02,
        ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
    }:
        raise RuntimeError(f"R2 action_scale config contract changed: {scale}")
    if clip != {".*": (-60.0, 60.0)}:
        raise RuntimeError(f"R2 joint_pos.clip config contract changed: {clip}")
    if not bool(getattr(action_cfg, "use_default_offset", False)):
        raise RuntimeError("R2 default_dof_pos offset contract is disabled")
    if not bool(getattr(action_cfg, "preserve_order", False)):
        raise RuntimeError("R2 preserve_order contract is disabled")
    return {
        "joint_order": joint_order,
        "scale_patterns": scale,
        "clip_patterns": {key: list(value) for key, value in clip.items()},
        "use_default_offset": True,
        "preserve_order": True,
    }


def _r2_tensor_rows(label: str, value, expected, *, num_envs: int) -> list:
    """Validate a finite initialized per-environment R2 tensor and return one row."""
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"R2 runtime {label} is not a tensor: {type(value).__name__}")
    actual = value.detach().to(device="cpu", dtype=torch.float64)
    expected_tensor = torch.tensor(expected, dtype=torch.float64)
    expected_shape = (num_envs, *expected_tensor.shape)
    if tuple(actual.shape) != expected_shape:
        raise RuntimeError(f"R2 runtime {label} shape changed: {tuple(actual.shape)} != {expected_shape}")
    if not bool(torch.all(torch.isfinite(actual)).item()):
        raise RuntimeError(f"R2 runtime {label} contains non-finite values")
    expected_rows = expected_tensor.unsqueeze(0).expand_as(actual)
    if not torch.allclose(actual, expected_rows, rtol=0.0, atol=1.0e-7):
        max_delta = float(torch.max(torch.abs(actual - expected_rows)).item())
        raise RuntimeError(f"R2 runtime {label} changed (max_abs_delta={max_delta})")
    # The initialized tensor was just proven equivalent.  Persist the canonical
    # preregistered decimal values rather than device/dtype round-off noise.
    return expected_tensor.tolist()


def _assert_r2_runtime_contract(env, policy, preregistration: dict) -> dict:
    """Bind exact R2 action tensors and the training-only three-value Teacher context."""
    permanent_contract = preregistration["permanent_contract"]
    extra_group = preregistration["training_data"]["extra_rollout_group"]
    expected_joint_order = list(permanent_contract["joint_order"])
    expected_joint_ids = list(range(16))
    expected_scale = list(permanent_contract["action_scale"])
    expected_clip = list(permanent_contract["joint_pos_clip"])
    expected_default_joint_pos = [0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4

    runtime_env = env.unwrapped
    num_envs = int(runtime_env.num_envs)
    action_manager = getattr(runtime_env, "action_manager", None)
    action_terms = getattr(action_manager, "_terms", None)
    if not isinstance(action_terms, dict) or list(action_terms) != ["joint_pos"]:
        raise RuntimeError(
            f"R2 runtime action terms changed: {list(action_terms) if isinstance(action_terms, dict) else action_terms}"
        )
    action_term = action_terms["joint_pos"]
    joint_names = list(getattr(action_term, "_joint_names", []))
    raw_joint_ids = getattr(action_term, "_joint_ids", None)
    if isinstance(raw_joint_ids, torch.Tensor):
        joint_ids = [int(item) for item in raw_joint_ids.detach().cpu().flatten().tolist()]
    elif isinstance(raw_joint_ids, (list, tuple)):
        joint_ids = [int(item) for item in raw_joint_ids]
    else:
        raise RuntimeError(f"R2 runtime action joint IDs are unresolved: {raw_joint_ids!r}")
    if joint_names != expected_joint_order or joint_ids != expected_joint_ids:
        raise RuntimeError(f"R2 runtime action joint binding changed: names={joint_names}, ids={joint_ids}")
    if int(getattr(action_term, "action_dim", -1)) != 16:
        raise RuntimeError("R2 runtime action dimension must be 16")
    if int(getattr(action_manager, "total_action_dim", -1)) != 16:
        raise RuntimeError("R2 runtime total action dimension must be 16")

    robot = runtime_env.scene["robot"]
    robot_joint_names = list(getattr(robot, "joint_names", []))
    if robot_joint_names != expected_joint_order:
        raise RuntimeError(f"R2 runtime robot joint order changed: {robot_joint_names}")
    if getattr(action_term, "_asset", None) is not robot:
        raise RuntimeError("R2 runtime action term is not bound to the robot articulation")
    runtime_scale = _r2_tensor_rows(
        "action scale", getattr(action_term, "_scale", None), expected_scale, num_envs=num_envs
    )
    runtime_default = _r2_tensor_rows(
        "robot default_joint_pos",
        robot.data.default_joint_pos[:, joint_ids],
        expected_default_joint_pos,
        num_envs=num_envs,
    )
    runtime_offset = _r2_tensor_rows(
        "action offset", getattr(action_term, "_offset", None), expected_default_joint_pos, num_envs=num_envs
    )
    if not torch.equal(
        action_term._offset.detach(), robot.data.default_joint_pos[:, joint_ids].detach()
    ):
        raise RuntimeError("R2 runtime action offset is not the robot's actual default_joint_pos")
    runtime_clip = _r2_tensor_rows(
        "action clip", getattr(action_term, "_clip", None), expected_clip, num_envs=num_envs
    )

    expected_observations = {
        "policy": {
            "terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"],
            "dims": [(30,), (30,), (30,), (160,), (160,), (160,)],
            "history": [10, 10, 10, 10, 10, 10],
            "group_dim": (570,),
        },
        "estimator": {
            "terms": ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"],
            "dims": [(30,), (30,), (30,), (160,), (160,), (160,)],
            "history": [10, 10, 10, 10, 10, 10],
            "group_dim": (570,),
        },
        "critic": {
            "terms": [
                "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
                "joint_pos", "joint_vel", "actions", "height_scan",
            ],
            "dims": [(3,), (3,), (3,), (3,), (16,), (16,), (16,), (102,)],
            "history": [0, 0, 0, 0, 0, 0, 0, 0],
            "group_dim": (162,),
        },
        "teacher_context": {
            "terms": ["prior_context"],
            "dims": [(3,)],
            "history": [0],
            "group_dim": (3,),
        },
    }
    observation_manager = getattr(runtime_env, "observation_manager", None)
    if observation_manager is None:
        raise RuntimeError("R2 runtime observation manager is unavailable")
    actual_groups = list(getattr(observation_manager, "_group_obs_term_names", {}))
    if actual_groups != list(expected_observations):
        raise RuntimeError(f"R2 runtime observation groups changed: {actual_groups}")

    observation_manifest = {}
    for group_name, expected in expected_observations.items():
        term_names = list(observation_manager._group_obs_term_names[group_name])
        term_dims = [
            tuple(int(item) for item in dim)
            for dim in observation_manager._group_obs_term_dim[group_name]
        ]
        term_cfgs = list(observation_manager._group_obs_term_cfgs[group_name])
        history = [int(term_cfg.history_length) for term_cfg in term_cfgs]
        flatten_history = [bool(term_cfg.flatten_history_dim) for term_cfg in term_cfgs]
        group_dim = tuple(int(item) for item in observation_manager._group_obs_dim[group_name])
        if term_names != expected["terms"]:
            raise RuntimeError(f"R2 runtime {group_name} observation order changed: {term_names}")
        if term_dims != expected["dims"] or group_dim != expected["group_dim"]:
            raise RuntimeError(
                f"R2 runtime {group_name} dimensions changed: terms={term_dims}, group={group_dim}"
            )
        if history != expected["history"] or flatten_history != [True] * len(term_names):
            raise RuntimeError(
                f"R2 runtime {group_name} observation history changed: "
                f"lengths={history}, flatten={flatten_history}"
            )
        if not bool(observation_manager._group_obs_concatenate[group_name]):
            raise RuntimeError(f"R2 runtime {group_name} observations are not concatenated")
        history_buffers = observation_manager._group_obs_term_history_buffer[group_name]
        expected_buffer_terms = {
            name for name, history_length in zip(term_names, history, strict=True) if history_length > 0
        }
        if set(history_buffers) != expected_buffer_terms:
            raise RuntimeError(
                f"R2 runtime {group_name} history buffers changed: {list(history_buffers)}"
            )
        for term_name in expected_buffer_terms:
            if int(history_buffers[term_name].max_length) != 10:
                raise RuntimeError(
                    f"R2 runtime {group_name}/{term_name} history buffer length changed: "
                    f"{history_buffers[term_name].max_length}"
                )
        observation_manifest[group_name] = {
            "term_names": term_names,
            "term_dims": [list(dim) for dim in term_dims],
            "history_length": history,
            "flatten_history_dim": flatten_history,
            "group_dim": list(group_dim),
        }

    expected_policy_groups = {
        "policy": ["policy"],
        "estimator": ["estimator"],
        "critic": ["critic"],
        "teacher_context": ["teacher_context"],
    }
    actual_policy_groups = {
        key: list(value) for key, value in getattr(policy, "obs_groups", {}).items()
    }
    if actual_policy_groups != expected_policy_groups:
        raise RuntimeError(f"R2 policy observation routing changed: {actual_policy_groups}")
    for attribute, expected_keys in {
        "policy_keys": ["policy"],
        "estimator_keys": ["estimator"],
        "critic_keys": ["critic"],
        "teacher_context_keys": ["teacher_context"],
    }.items():
        actual_keys = list(getattr(policy, attribute, []))
        if actual_keys != expected_keys:
            raise RuntimeError(f"R2 policy {attribute} changed: {actual_keys}")
    if any(
        "teacher_context" in actual_policy_groups[group_name]
        for group_name in ("policy", "estimator", "critic")
    ):
        raise RuntimeError("R2 teacher_context leaked into a deployment observation input")
    if int(getattr(policy, "policy_obs_dim", -1)) != 570:
        raise RuntimeError("R2 policy observation dimension must remain 570")

    teacher_context = observation_manager.compute_group("teacher_context", update_history=False)
    if not isinstance(teacher_context, torch.Tensor):
        raise RuntimeError("R2 teacher_context runtime output is not a tensor")
    if tuple(teacher_context.shape) != (num_envs, 3):
        raise RuntimeError(
            f"R2 teacher_context runtime shape changed: {tuple(teacher_context.shape)} != {(num_envs, 3)}"
        )
    if not bool(torch.all(torch.isfinite(teacher_context)).item()):
        raise RuntimeError("R2 teacher_context runtime output contains non-finite values")

    return {
        "validated": True,
        "joint_order": joint_names,
        "joint_ids": joint_ids,
        "action_scale": runtime_scale,
        "action_offset": runtime_offset,
        "joint_pos_clip": runtime_clip,
        "default_joint_pos": runtime_default,
        "teacher_context_shape": list(extra_group["shape"]),
        "teacher_context_order": list(extra_group["order"]),
        "teacher_context_runtime_finite": True,
        "policy_estimator_critic_input": False,
        "shared_post_prior_function": permanent_contract["shared_post_prior_helper"],
        "observations": observation_manifest,
        "policy_observation_groups": actual_policy_groups,
        "policy_obs_dim": int(policy.policy_obs_dim),
    }


def _highstep_reward_stage_definition(env_cfg) -> dict[str, dict[str, int]]:
    definition: dict[str, dict[str, int]] = {}
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return definition
    for name in dir(rewards_cfg):
        if name.startswith("_"):
            continue
        term_cfg = getattr(rewards_cfg, name, None)
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict) or "stage_start_update" not in params:
            continue
        definition[name] = {
            "stage_start_update": int(params["stage_start_update"]),
            "stage_ramp_updates": int(params.get("stage_ramp_updates", 1)),
            "num_steps_per_update": int(params.get("num_steps_per_update", 24)),
        }
    return dict(sorted(definition.items()))


def _highstep_fresh_initial_terrain_distribution(env) -> dict:
    """Fingerprint the fresh per-env terrain population used by paired training."""
    terrain = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)
    if not isinstance(levels, torch.Tensor) or not isinstance(types, torch.Tensor):
        raise ScheduleContinuityError(
            "Paired v1.12.3 training requires runtime terrain_levels and terrain_types"
        )

    def record(name: str, value: torch.Tensor) -> dict:
        cpu = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(cpu.dtype).encode("utf-8"))
        digest.update(json.dumps(list(cpu.shape)).encode("utf-8"))
        digest.update(cpu.numpy().tobytes())
        keys, counts = torch.unique(cpu, sorted=True, return_counts=True)
        histogram = {
            str(int(key.item())): int(count.item()) for key, count in zip(keys, counts, strict=True)
        }
        return {
            "name": name,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
            "sha256": digest.hexdigest(),
            "histogram": histogram,
        }

    levels_record = record("terrain_levels", levels)
    types_record = record("terrain_types", types)
    joint_sha = hashlib.sha256(
        json.dumps(
            {"levels": levels_record["sha256"], "types": types_record["sha256"]},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "claim": "same_fresh_initial_training_distribution_not_process_exact_resume",
        "terrain_levels": levels_record,
        "terrain_types": types_record,
        "joint_sha256": joint_sha,
    }


def _capture_highstep_command_state(env, *, enabled: bool) -> dict:
    if not enabled:
        return {"enabled": False}
    unwrapped = env.unwrapped
    required_attrs = (
        "_highstep_original_vel_x",
        "_highstep_original_vel_y",
        "_highstep_original_yaw",
        "_highstep_initial_vel_x",
        "_highstep_initial_vel_y",
        "_highstep_initial_yaw",
        "_highstep_final_vel_x",
        "_highstep_final_vel_y",
        "_highstep_final_yaw",
        "_highstep_gated_vel_x",
        "_highstep_gated_vel_y",
        "_highstep_gated_yaw",
    )
    missing = [name for name in required_attrs if not hasattr(unwrapped, name)]
    if missing:
        raise ScheduleContinuityError(f"Command curriculum state was not initialized: {missing}")
    ranges = unwrapped.command_manager.get_term("base_velocity").cfg.ranges
    state = {"enabled": True}
    for name in required_attrs:
        state[name.removeprefix("_highstep_")] = _tensor_or_sequence_list(getattr(unwrapped, name))
    state["current_lin_vel_x"] = _tensor_or_sequence_list(ranges.lin_vel_x)
    state["current_lin_vel_y"] = _tensor_or_sequence_list(ranges.lin_vel_y)
    state["current_ang_vel_z"] = _tensor_or_sequence_list(ranges.ang_vel_z)
    return state


def _capture_highstep_moving_best(env, *, required: bool) -> dict:
    unwrapped = env.unwrapped
    action_best = getattr(unwrapped, "_highstep_action_score_curriculum_best", None)
    support_best = getattr(unwrapped, "_highstep_support_score_curriculum_best", None)
    observed = action_best is not None and support_best is not None
    if required and not observed:
        raise ScheduleContinuityError("ActionScore terrain moving-best state was not initialized before checkpoint save")
    return {
        "required": bool(required),
        "observed": bool(observed),
        "action_score_best": float(action_best.detach().cpu()) if action_best is not None else None,
        "support_score_best": float(support_best.detach().cpu()) if support_best is not None else None,
    }


def _restore_highstep_runtime_snapshot(
    env,
    snapshot: dict,
    *,
    command_required: bool,
    moving_best_required: bool,
) -> tuple[bool, bool]:
    unwrapped = env.unwrapped
    command_state = snapshot.get("command_curriculum", {})
    command_restored = False
    if command_required:
        if command_state.get("enabled") is not True:
            raise ScheduleContinuityError("Preserve resume requires an enabled command curriculum snapshot")
        ranges = unwrapped.command_manager.get_term("base_velocity").cfg.ranges
        tensor_keys = (
            "original_vel_x",
            "original_vel_y",
            "original_yaw",
            "initial_vel_x",
            "initial_vel_y",
            "initial_yaw",
            "final_vel_x",
            "final_vel_y",
            "final_yaw",
            "gated_vel_x",
            "gated_vel_y",
            "gated_yaw",
        )
        for key in tensor_keys:
            values = command_state.get(key)
            if not isinstance(values, list) or not values:
                raise ScheduleContinuityError(f"Command curriculum snapshot is missing {key}")
            tensor = torch.tensor(values, device=unwrapped.device, dtype=torch.float32)
            if not bool(torch.all(torch.isfinite(tensor)).item()):
                raise ScheduleContinuityError(f"Command curriculum snapshot contains invalid {key}")
            setattr(unwrapped, f"_highstep_{key}", tensor)
        ranges.lin_vel_x = [float(value) for value in command_state["current_lin_vel_x"]]
        ranges.lin_vel_y = [float(value) for value in command_state["current_lin_vel_y"]]
        ranges.ang_vel_z = [float(value) for value in command_state["current_ang_vel_z"]]
        unwrapped._highstep_command_state_restored = True
        command_restored = True

    moving_best = snapshot.get("moving_best", {})
    moving_best_restored = False
    if moving_best_required:
        if moving_best.get("observed") is not True:
            raise ScheduleContinuityError("Preserve resume requires observed ActionScore moving-best state")
        action_best = float(moving_best["action_score_best"])
        support_best = float(moving_best["support_score_best"])
        if not all(torch.isfinite(torch.tensor([action_best, support_best])).tolist()):
            raise ScheduleContinuityError("ActionScore moving-best snapshot contains non-finite values")
        unwrapped._highstep_action_score_curriculum_best = torch.tensor(action_best, device=unwrapped.device)
        unwrapped._highstep_support_score_curriculum_best = torch.tensor(support_best, device=unwrapped.device)
        unwrapped._highstep_moving_best_state_restored = True
        moving_best_restored = True
    return command_restored, moving_best_restored


def _install_highstep_checkpoint_state_hook(
    runner,
    env,
    *,
    runtime_state_path: str,
    num_steps_per_update: int,
    command_curriculum_enabled: bool,
    moving_best_required: bool,
) -> None:
    original_save = runner.save

    def save_with_runtime_state(path: str, infos: dict | None = None):
        original_save(path, infos)
        snapshot = {
            "created_at": datetime.now().astimezone().isoformat(),
            "checkpoint_file": os.path.basename(path),
            "checkpoint_sha256": checkpoint_sha256(path),
            "runner_iteration": int(runner.current_learning_iteration),
            "schedule_update": global_update(env, num_steps_per_update),
            "local_common_step": int(env.unwrapped.common_step_counter),
            "command_curriculum": _capture_highstep_command_state(
                env, enabled=command_curriculum_enabled
            ),
            "moving_best": _capture_highstep_moving_best(env, required=moving_best_required),
        }
        append_runtime_snapshot(runtime_state_path, snapshot)

    runner.save = save_with_runtime_state


def _install_relative_student_checkpoint_cadence(
    runner, *, absolute_origin: int, interval: int
) -> None:
    """Save on the relative Student clock while W&B uses an absolute runner clock."""
    if absolute_origin < 0 or interval <= 0:
        raise ValueError("invalid relative Student checkpoint cadence")
    original_update = runner.alg.update
    # Disable the stock absolute-iteration modulo saves. The final stock save
    # remains active; relative checkpoints below call the fully wrapped save().
    runner.save_interval = 10**12

    def update_with_relative_checkpoint():
        previous_relative_update = int(runner.alg.student_distill_update_count)
        result = original_update()
        relative_update = int(runner.alg.student_distill_update_count)
        if (
            relative_update > previous_relative_update
            and relative_update % interval == 0
        ):
            expected_absolute = absolute_origin + relative_update - 1
            previous_iteration = int(runner.current_learning_iteration)
            runner.current_learning_iteration = expected_absolute
            try:
                runner.save(os.path.join(runner.log_dir, f"model_{expected_absolute}.pt"))
            finally:
                runner.current_learning_iteration = previous_iteration
        return result

    runner.alg.update = update_with_relative_checkpoint


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
        stream.options = {"crf": str(self.video_crf), "preset": "ultrafast"}
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
# === Add: Spectral-Normalized ActorCritic defined inline in train.py ===
import torch.nn as nn
# 兼容两种导入路径（不同 PyTorch 版本）
try:
    from torch.nn.utils.parametrizations import spectral_norm as _spectral_norm
except Exception:
    from torch.nn.utils import spectral_norm as _spectral_norm

# rsl-rl 的 ActorCritic 基类
try:
    from rsl_rl.modules.actor_critic import ActorCritic as _BaseActorCritic
except Exception:
    import rsl_rl.modules.actor_critic as _ac_mod
    _BaseActorCritic = _ac_mod.ActorCritic

def _apply_sn(module: nn.Module, n_power_iterations: int = 1):
    """给模块里所有 Linear 施加谱归一化。"""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            _spectral_norm(m, n_power_iterations=n_power_iterations)
    return module

class ActorCriticSN(_BaseActorCritic):
    """Actor-Critic with Spectral Normalization on all Linear layers."""
    def __init__(self, *args, sn_on=("actor", "critic"), n_power_iterations: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if "actor" in sn_on and hasattr(self, "actor"):
            _apply_sn(self.actor, n_power_iterations)
        if "critic" in sn_on and hasattr(self, "critic"):
            _apply_sn(self.critic, n_power_iterations)

# ---- 最关键的一行：把默认类名映射到我们的 SN 版本（无需改任何 cfg）----
# import rsl_rl.modules.actor_critic as _ac
# _ac.ActorCritic = ActorCriticSN
# （可选）如果你在 cfg 里把 class_name 改成了 "ActorCriticSN"：
# import rsl_rl.runners.on_policy_runner as _opr
# _opr.ActorCriticSN = ActorCriticSN
# 这样 eval("ActorCriticSN") 也能解析到这个类。
# === End Add ===

# =========================================================================
# 🌟 注册 VAEActorCritic 和 VAEPPO 到 RSL-RL 命名空间
# =========================================================================
try:
    # 1. 从你的实际路径导入 VAEActorCritic 和 VAEPPO
    from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.vae_ppo import VAEActorCritic, VAEPPO

    # 2. 导入 rsl_rl 的相关模块
    import rsl_rl.modules.actor_critic as _ac
    import rsl_rl.algorithms.ppo as _ppo
    import rsl_rl.runners.on_policy_runner as _opr

    # 3. 强行注入到 rsl_rl 的命名空间中，这样 eval() 就能找到它们了！
    _ac.VAEActorCritic = VAEActorCritic
    _ppo.VAEPPO = VAEPPO
    _opr.VAEActorCritic = VAEActorCritic
    _opr.VAEPPO = VAEPPO
    print("[INFO] Successfully registered VAEActorCritic and VAEPPO to RSL-RL.")
except ImportError as e:
    print(f"[WARN] Could not import VAE classes: {e}")
# =========================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""

    # =========================================================================
    # 🌟 修改点：根据 args_cli.agent 动态拦截并覆盖 agent_cfg
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg":
        print("[INFO] Using Symmetric PPO Algorithm and Config!")
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo_cfg import ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg

        # 覆盖 config
        agent_cfg = ArclabArcdogAdjustableLegBodyflatSymmetricPPORunnerCfg()
        # 强制指定 class_name，以便后续逻辑识别
        agent_cfg.class_name = "SymmetricOnPolicyRunner"
    else:
        print("[INFO] Using Standard RSL-RL Config!")
    # =========================================================================

    if IS_DISTRIBUTED:
        env_cfg.sim.device = f"cuda:{LOCAL_RANK}"
        agent_cfg.device = f"cuda:{LOCAL_RANK}"
        seed = (agent_cfg.seed or 0) + LOCAL_RANK
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # override configurations with non-hydra CLI arguments
    # 注意：因为我们在上面覆盖了 agent_cfg，这里的 cli_args.update_rsl_rl_cfg 依然能正常工作！
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    highstep_resume_refine_tasks = {
        "RobotLab-Isaac-Velocity-Highstep-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV15-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior0707Exact-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorHistorical0707Exact-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123StudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepFrontGeometryV114STEStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300CanonicalHybridStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300DiagonalImitationFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300CriticalTransitionBalancedDiagonalFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-ArcdogAdjustableLeg-v0",
    }
    is_highstep_schedule_task = (
        args_cli.task in highstep_resume_refine_tasks
        or "highstep" in str(args_cli.task or "").lower()
    )
    v114_historical_baseline_capture = os.environ.get(
        "HIGHSTEP_BE300_V114_HISTORICAL_BASELINE_CAPTURE", ""
    ) == "1"
    if v114_historical_baseline_capture:
        expected_e4000 = os.path.realpath(
            "/home/lxq/Softwares/robot_lab/logs/rsl_rl/"
            "arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_student_no_prior_Student/"
            "2026-07-16_16-03-20_highstep_be300_0707_distill_long_E4000_recovery_r3_model1000_20260716/"
            "model_3998.pt"
        )
        expected_prereg = os.path.realpath(
            "/home/lxq/Softwares/robot_lab/tmp/highstep_be300_0707_distill_20260716/"
            "preregistration_v1131.json"
        )
        if not (
            args_cli.task
            == "RobotLab-Isaac-Velocity-HighstepFrontGeometryV1123StudentNoPrior-ArcdogAdjustableLeg-v0"
            and args_cli.v114_create_frozen_gate_dataset
            and args_cli.v114_frozen_gate_dataset
            and args_cli.v114_frozen_gate_report
            and not args_cli.v114_expected_frozen_gate_dataset_sha256
            and args_cli.resume
            and args_cli.highstep_checkpoint_load_mode == "full"
            and args_cli.highstep_schedule_resume_mode == "preserve"
            and os.path.realpath(str(args_cli.checkpoint or "")) == expected_e4000
            and os.path.realpath(
                os.environ.get("HIGHSTEP_BE300_0707_PREREGISTRATION_PATH", "")
            ) == expected_prereg
            and os.environ.get("HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256", "")
            == "2660ad2ba78719018c0d1ba1d926ee6fc6e997aebd8bfabdefcd1e18196b1cc6"
        ):
            raise RuntimeError(
                "v1.14 historical baseline authority is restricted to one read-only E4000 "
                "frozen-buffer capture"
            )
    if agent_cfg.resume and is_highstep_schedule_task:
        highstep_resume_mode, highstep_resume_reason = _resolve_highstep_resume_mode(args_cli, agent_cfg)
        print(
            "[INFO] Highstep resume mode: "
            f"{highstep_resume_mode} ({highstep_resume_reason})."
        )
    else:
        highstep_resume_mode = None

    highstep_checkpoint_load_mode = args_cli.highstep_checkpoint_load_mode
    highstep_schedule_requested_mode = args_cli.highstep_schedule_resume_mode
    legacy_student_distill_update_count = args_cli.legacy_student_distill_update_count
    absolute_runner_step_origin = args_cli.highstep_absolute_runner_step_origin
    single_run_7400_tasks = {
        "RobotLab-Isaac-Velocity-HighstepB3000707DerivedSingleRun7400StudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300DiagonalImitationFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepB300CriticalTransitionBalancedDiagonalFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0",
    }
    if absolute_runner_step_origin is not None and not (
        args_cli.task in single_run_7400_tasks
        and absolute_runner_step_origin == 173499
        and args_cli.max_iterations == 7400
        and agent_cfg.resume
        and highstep_checkpoint_load_mode == "weights_only"
        and highstep_schedule_requested_mode == "reset"
    ):
        raise RuntimeError("absolute runner step origin is outside the frozen single-run 7400 contract")
    if args_cli.task in single_run_7400_tasks and absolute_runner_step_origin != 173499:
        raise RuntimeError("single-run 7400 requires absolute runner step origin 173499")
    verified_teacher_manifest = args_cli.verified_legacy_teacher_algorithm_state_manifest
    verified_teacher_manifest_sha = args_cli.verified_legacy_teacher_algorithm_state_manifest_sha256
    verified_legacy_teacher_checkpoint = None
    if bool(verified_teacher_manifest) != bool(verified_teacher_manifest_sha):
        raise ValueError(
            "Verified legacy Teacher algorithm-state manifest path and SHA256 must be supplied together"
        )
    if verified_teacher_manifest:
        if legacy_student_distill_update_count is not None:
            raise ValueError("Legacy Student and verified legacy Teacher migrations are mutually exclusive")
        if highstep_checkpoint_load_mode != "full" or not agent_cfg.resume:
            raise ValueError("Verified legacy Teacher migration requires a full resumed checkpoint load")
        if args_cli.task != (
            "RobotLab-Isaac-Velocity-HighstepRearSupportV112-ArcdogAdjustableLeg-v0"
        ):
            raise ValueError("Verified legacy Teacher migration is only valid for the exact v1.12 Teacher task")
    if legacy_student_distill_update_count is not None:
        if legacy_student_distill_update_count < 0:
            raise ValueError("--legacy_student_distill_update_count must be non-negative")
        if highstep_checkpoint_load_mode != "full":
            raise ValueError(
                "--legacy_student_distill_update_count is only valid with "
                "--highstep_checkpoint_load_mode=full"
            )
        if not agent_cfg.resume:
            raise ValueError("--legacy_student_distill_update_count requires --resume")
        if "student" not in str(args_cli.task or "").lower():
            raise ValueError("--legacy_student_distill_update_count is only valid for a Student task")
    if highstep_checkpoint_load_mode == "weights_only":
        if not agent_cfg.resume or not is_highstep_schedule_task:
            raise ValueError("--highstep_checkpoint_load_mode=weights_only requires a resumed highstep task")
        if agent_cfg.class_name != "OnPolicyRunner":
            raise ValueError("--highstep_checkpoint_load_mode=weights_only currently supports OnPolicyRunner only")
    if is_highstep_schedule_task:
        will_load_checkpoint = agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
        if highstep_schedule_requested_mode == "preserve" and not will_load_checkpoint:
            raise ValueError("Cannot preserve a high-step schedule without loading a checkpoint")
        highstep_schedule_resolved_mode = resolve_schedule_resume_mode(
            checkpoint_load_mode=highstep_checkpoint_load_mode,
            requested_mode=highstep_schedule_requested_mode,
            highstep_resume_kind=highstep_resume_mode,
        )
        if args_cli.allow_legacy_highstep_schedule_fallback and not (
            will_load_checkpoint
            and highstep_checkpoint_load_mode == "full"
            and highstep_schedule_resolved_mode == "preserve"
        ):
            raise ValueError(
                "--allow_legacy_highstep_schedule_fallback is only valid for a full checkpoint "
                "whose resolved schedule mode is preserve"
            )
        print(
            "[INFO] Highstep schedule resume mode: "
            f"{highstep_schedule_resolved_mode} (requested={highstep_schedule_requested_mode}, "
            f"checkpoint_load={highstep_checkpoint_load_mode})."
        )
    else:
        if highstep_schedule_requested_mode != "auto" or args_cli.allow_legacy_highstep_schedule_fallback:
            raise ValueError("High-step schedule options are only valid for a high-step task")
        highstep_schedule_resolved_mode = "reset"

    v18_schedule_anchor = args_cli.highstep_v18_schedule_anchor_effective_update
    v18_transition = None
    if v18_schedule_anchor is not None:
        if not (
            args_cli.task
            == "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV18Robust-ArcdogAdjustableLeg-v0"
            and agent_cfg.resume
            and highstep_checkpoint_load_mode == "full"
            and highstep_schedule_resolved_mode == "reset"
            and v18_schedule_anchor >= 0
        ):
            raise RuntimeError(
                "v1.8 schedule anchor is only valid for the full Stage-A to Stage-B reset transition"
            )
        transition_path = os.path.realpath(
            str(args_cli.highstep_v18_stage_transition_manifest or "")
        )
        transition_sha = str(args_cli.highstep_v18_stage_transition_manifest_sha256 or "")
        if not transition_path or not os.path.isabs(transition_path) or not transition_sha:
            raise RuntimeError("v1.8 schedule anchor requires a hash-bound transition manifest")
        if checkpoint_sha256(transition_path) != transition_sha:
            raise RuntimeError("v1.8 Stage-A to Stage-B transition manifest SHA mismatch")
        with open(transition_path, encoding="utf-8") as stream:
            v18_transition = json.load(stream)
        source_effective_update = _read_student_recovery_effective_update(
            agent_cfg.load_checkpoint
        )
        if not (
            v18_transition.get("kind") == "highstep_v18_environment_only_stage_transition"
            and v18_transition.get("source_checkpoint") == os.path.realpath(agent_cfg.load_checkpoint)
            and v18_transition.get("source_checkpoint_sha256") == checkpoint_sha256(agent_cfg.load_checkpoint)
            and v18_transition.get("source_effective_updates") == v18_schedule_anchor
            and source_effective_update == v18_schedule_anchor
            and v18_transition.get("resume_mode") == "full_checkpoint_with_original_optimizer"
            and v18_transition.get("effective_updates_reset") is False
            and v18_transition.get("only_changed_training_mechanism") == "student_environment_profile"
        ):
            raise RuntimeError("v1.8 Stage-A to Stage-B transition contract mismatch")
    elif (
        args_cli.highstep_v18_stage_transition_manifest
        or args_cli.highstep_v18_stage_transition_manifest_sha256
    ):
        raise RuntimeError("v1.8 transition manifest supplied without an effective-update anchor")

    if (
        agent_cfg.resume
        and is_highstep_schedule_task
        and highstep_resume_mode == "refine"
        and highstep_schedule_resolved_mode == "preserve"
    ):
        print(
            "[INFO] Highstep resume/refine: preserving reward-stage endpoints and command curriculum "
            "definition; checkpoint-paired runtime state will be restored before the first wrapper reset."
        )
    elif agent_cfg.resume and is_highstep_schedule_task:
        print(
            "[INFO] Highstep resume with reset/migration schedule: keeping staged highstep "
            "curriculum and command terrain gate unchanged."
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
        # =================================================================
        # ====== [DEBUG-CURVE: CHECKPOINT PATH RESOLUTION] ======
        # 实时查看路径解析数据，证明修改是否有效
        # =================================================================
        print("\n" + "="*65)
        print("====== [DEBUG-CURVE: CHECKPOINT PATH RESOLUTION] ======")
        print(f"  [Data Point 1] Target log_root_path: {log_root_path}")
        print(f"  [Data Point 2] Input checkpoint arg: {agent_cfg.load_checkpoint}")

        _is_abs = agent_cfg.load_checkpoint and os.path.isabs(str(agent_cfg.load_checkpoint))
        print(f"  [Data Point 3] Is absolute path?   : {_is_abs}")

        if _is_abs:
            # 修改核心 1：如果是绝对路径（跨目录读取 Teacher），直接绕过 get_checkpoint_path
            resume_path = agent_cfg.load_checkpoint
            print(f"  [Data Point 4] Action Taken      : Bypassed get_checkpoint_path, using absolute path directly.")
        else:
            # 修改核心 2：如果是相对路径，确保 log_root_path 存在，防止 os.scandir 崩溃
            if not os.path.exists(log_root_path):
                os.makedirs(log_root_path, exist_ok=True)
                print(f"  [Data Point 4] Action Taken      : Created missing log_root_path to prevent FileNotFoundError.")
            else:
                print(f"  [Data Point 4] Action Taken      : log_root_path exists, proceeding with standard scan.")

            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        print(f"  [Data Point 5] Final resume_path : {resume_path}")

        _path_exists = os.path.exists(resume_path)
        print(f"  [Data Point 6] Path exists on disk?: {'✅ YES' if _path_exists else '❌ NO'}")
        if _path_exists:
            _file_size_mb = os.path.getsize(resume_path) / (1024 * 1024)
            print(f"  [Data Point 7] Checkpoint Size   : {_file_size_mb:.2f} MB (Validating file integrity)")
        print("=================================================================\n")
        # =================================================================

        if verified_teacher_manifest:
            verified_legacy_teacher_checkpoint = (
                _load_verified_legacy_teacher_algorithm_state_contract(
                    manifest_path=verified_teacher_manifest,
                    manifest_sha256=verified_teacher_manifest_sha,
                    checkpoint_path=resume_path,
                    task=str(args_cli.task),
                )
            )

    student_parent_lineage = None
    if is_highstep_schedule_task:
        lineage_source_manifest = None
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            _, lineage_source_manifest = load_source_manifest(resume_path)
        student_parent_lineage = resolve_student_parent_lineage(
            task=args_cli.task,
            checkpoint_path=(
                resume_path
                if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
                else None
            ),
            parent_teacher_manifest_path=args_cli.highstep_parent_teacher_manifest,
            source_manifest=lineage_source_manifest,
        )
    elif args_cli.highstep_parent_teacher_manifest:
        raise ValueError("--highstep_parent_teacher_manifest is only valid for a high-step Student task")

    highstep_action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    highstep_terrain_term_cfg = getattr(getattr(env_cfg, "curriculum", None), "terrain_levels", None)
    highstep_support_term_cfg = getattr(getattr(env_cfg, "curriculum", None), "highstep_action_score", None)
    highstep_command_term_cfg = getattr(getattr(env_cfg, "curriculum", None), "command_levels", None)
    highstep_command_curriculum_enabled = highstep_command_term_cfg is not None
    highstep_reward_stage_contract = _highstep_reward_stage_definition(env_cfg)
    terrain_func_name = getattr(getattr(highstep_terrain_term_cfg, "func", None), "__name__", "")
    highstep_moving_best_required = "action_score" in terrain_func_name
    highstep_num_steps_per_update = int(
        getattr(highstep_action_cfg, "num_steps_per_update", 24) or 24
    )
    highstep_runtime_state_path = os.path.join(log_dir, "params", RUNTIME_STATE_NAME)
    command_state_restored = False
    moving_best_state_restored = False

    # RslRlVecEnvWrapper resets the task in its constructor.  Install a
    # provisional checkpoint clock before that reset so reset-time reward and
    # curriculum terms never observe a false update zero.  The runner-loaded
    # value is checked and re-anchored below.
    if is_highstep_schedule_task:
        pre_checkpoint_iteration = None
        pre_checkpoint_schedule_update = 0.0
        pre_schedule_source = {"method": "fresh_training"}
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            pre_checkpoint_iteration = _read_rsl_checkpoint_iteration(resume_path)
            pre_checkpoint_schedule_update, pre_schedule_source = resolve_loaded_schedule_update(
                resume_path,
                pre_checkpoint_iteration,
                checkpoint_load_mode=highstep_checkpoint_load_mode,
                schedule_resume_mode=highstep_schedule_resolved_mode,
                allow_legacy_checkpoint_fallback=args_cli.allow_legacy_highstep_schedule_fallback,
            )
            if v18_schedule_anchor is not None:
                pre_checkpoint_schedule_update = float(v18_schedule_anchor)
                pre_schedule_source = {
                    "method": "v18_environment_transition_effective_update_anchor",
                    "transition_manifest": transition_path,
                    "transition_manifest_sha256": transition_sha,
                    "source_effective_updates": v18_schedule_anchor,
                }
        if pre_checkpoint_iteration is None:
            pre_schedule_update_at_anchor = 0.0
        else:
            pre_schedule_update_at_anchor = pre_checkpoint_schedule_update
        pre_runner_iteration = (
            int(absolute_runner_step_origin)
            if highstep_checkpoint_load_mode == "weights_only"
            and absolute_runner_step_origin is not None
            else 0
            if highstep_checkpoint_load_mode == "weights_only"
            else int(pre_checkpoint_iteration or 0)
        )
        install_global_update(
            env,
            pre_schedule_update_at_anchor,
            source=str(pre_schedule_source.get("method", "unknown")),
            resume_mode=highstep_schedule_resolved_mode,
            runner_iteration=pre_runner_iteration,
        )
        if pre_checkpoint_iteration is not None and highstep_schedule_resolved_mode == "preserve":
            runtime_snapshot = pre_schedule_source.get("runtime_snapshot")
            if runtime_snapshot is None:
                if highstep_command_curriculum_enabled or highstep_moving_best_required:
                    raise ScheduleContinuityError(
                        "Legacy checkpoint fallback has no checkpoint-paired command/moving-best state. "
                        "Use an explicit schedule reset/migration, or reconstruct valid v3 schedule and runtime sidecars."
                    )
            else:
                source_manifest_path, source_manifest = load_source_manifest(resume_path)
                if source_manifest_path is None or source_manifest is None:
                    raise ScheduleContinuityError("Preserve resume has runtime state but no source schedule manifest")
                current_definition = schedule_definition_from_configs(
                    action_cfg=highstep_action_cfg,
                    terrain_term_cfg=highstep_terrain_term_cfg,
                    support_term_cfg=highstep_support_term_cfg,
                    command_term_cfg=highstep_command_term_cfg,
                    command_curriculum_enabled=highstep_command_curriculum_enabled,
                    reward_stage_definition=highstep_reward_stage_contract,
                )
                assert_schedule_definition_compatible(
                    source_manifest,
                    current_definition,
                    allow_be300_teacher_prior_removal=(
                        str(getattr(agent_cfg.policy, "student_recovery_stage", "")).upper()
                        == "BE300_0707"
                    ),
                )
                command_state_restored, moving_best_state_restored = _restore_highstep_runtime_snapshot(
                    env,
                    runtime_snapshot,
                    command_required=highstep_command_curriculum_enabled,
                    moving_best_required=highstep_moving_best_required,
                )
        print(
            "[INFO] Installed pre-wrapper highstep schedule: "
            f"update={pre_schedule_update_at_anchor}, checkpoint_iter={pre_checkpoint_iteration}."
        )

    # wrap for video recording
    if args_cli.video and IS_MASTER:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "enable_wandb": (agent_cfg.logger == "wandb"),
            "wandb_key": "train/video",
            "video_resolution": (640, 360),
            "video_crf": 30,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # create runner from rsl-rl

    # # =========================================================================
    # # [Sim-to-Real 安全补丁] 强制修复 Action Clip (Hard Clip)
    # # =========================================================================
    # if hasattr(env.unwrapped, "action_manager"):
    #     for group_name, action_term in env.unwrapped.action_manager._terms.items():
    #         if hasattr(action_term, "_joint_names"):
    #             # 如果底层没有初始化 clip tensor，则先初始化为无限制
    #             if not hasattr(action_term, "_clip_min") or action_term._clip_min is None:
    #                 action_term._clip_min = torch.full((1, action_term.action_dim), -float('inf'), device=env.unwrapped.device)
    #                 action_term._clip_max = torch.full((1, action_term.action_dim), float('inf'), device=env.unwrapped.device)

    #             # 动态寻找包含 "box_joint" 的关节索引
    #             box_indices = [i for i, name in enumerate(action_term._joint_names) if "box_joint" in name]

    #             if box_indices:
    #                 # 强制写入绝对安全的 Raw Action 截断范围 [-1.0, 1.0]
    #                 action_term._clip_min[:, box_indices] = -1.0
    #                 action_term._clip_max[:, box_indices] = 1.0
    #                 print(f"\n!!! [SAFETY WARNING] 已强制锁定 {group_name} 中的 box_joint (索引: {box_indices}) 的动作范围为 [-1.0, 1.0] !!!\n", flush=True)
    # # =========================================================================

    # # =========================================================================
    # 🌟 修改点：动态切换 RunnerClass
    # =========================================================================
    if args_cli.agent == "symmetric_ppo_cfg" or agent_cfg.class_name == "SymmetricOnPolicyRunner":
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.agents.symmetric_ppo import SymmetricOnPolicyRunner
        runner = SymmetricOnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            config=env_cfg,       # <==== 补充缺失的 config 参数！
            log_dir=log_dir,
            device=agent_cfg.device
        )
    elif agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # =========================================================================

    # Install before any runner.load().  The high-step schedule hook installed
    # later wraps this enhanced save method, so its SHA covers the final .pt
    # containing both stock RSL-RL and algorithm-owned state.
    install_algorithm_checkpoint_state_hook(
        runner,
        legacy_student_distill_update_count=legacy_student_distill_update_count,
        verified_legacy_teacher_checkpoint=verified_legacy_teacher_checkpoint,
    )

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    checkpoint_iteration = None
    checkpoint_schedule_update = 0.0
    schedule_source = {"method": "fresh_training"}
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if highstep_checkpoint_load_mode == "weights_only":
            runner.load(resume_path, load_optimizer=False)
            loaded_iteration = int(runner.current_learning_iteration)
            checkpoint_iteration = loaded_iteration
            checkpoint_schedule_update, schedule_source = resolve_loaded_schedule_update(
                resume_path,
                checkpoint_iteration,
                checkpoint_load_mode=highstep_checkpoint_load_mode,
                schedule_resume_mode=highstep_schedule_resolved_mode,
                allow_legacy_checkpoint_fallback=args_cli.allow_legacy_highstep_schedule_fallback,
            )
            runner.current_learning_iteration = int(absolute_runner_step_origin or 0)
            print(
                "[INFO] Highstep clean resume: restored policy weights only; "
                "optimizer is fresh and learning iteration set from "
                f"{loaded_iteration} to {runner.current_learning_iteration}."
            )
        else:
            runner.load(resume_path)
            checkpoint_iteration = int(getattr(runner, "current_learning_iteration", 0))
            checkpoint_schedule_update, schedule_source = resolve_loaded_schedule_update(
                resume_path,
                checkpoint_iteration,
                checkpoint_load_mode=highstep_checkpoint_load_mode,
                schedule_resume_mode=highstep_schedule_resolved_mode,
                allow_legacy_checkpoint_fallback=args_cli.allow_legacy_highstep_schedule_fallback,
            )
        if v18_schedule_anchor is not None:
            checkpoint_schedule_update = float(v18_schedule_anchor)
            schedule_source = {
                "method": "v18_environment_transition_effective_update_anchor",
                "transition_manifest": transition_path,
                "transition_manifest_sha256": transition_sha,
                "source_effective_updates": v18_schedule_anchor,
            }

    student_recovery_stage = str(
        getattr(runner.alg, "student_recovery_stage", "none")
    ).upper()
    if student_recovery_stage == "B":
        if not agent_cfg.resume:
            raise RuntimeError("Highstep Student recovery stage B requires an explicit checkpoint")
        if legacy_student_distill_update_count is not None:
            raise RuntimeError("Stage B forbids legacy Student count migration")
        expected_joint_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
        ]
        action_joint_order = list(getattr(highstep_action_cfg, "joint_names", []))
        action_scale_contract = dict(getattr(highstep_action_cfg, "scale", {}))
        action_clip_contract = dict(getattr(highstep_action_cfg, "clip", {}))
        if action_joint_order != expected_joint_order:
            raise RuntimeError(f"Stage B action/joint order contract changed: {action_joint_order}")
        if action_scale_contract != {
            ".*_box_joint": 0.02,
            ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
        }:
            raise RuntimeError(f"Stage B action_scale contract changed: {action_scale_contract}")
        if action_clip_contract != {".*": (-60.0, 60.0)}:
            raise RuntimeError(f"Stage B joint_pos.clip contract changed: {action_clip_contract}")
        if not bool(getattr(highstep_action_cfg, "use_default_offset", False)):
            raise RuntimeError("Stage B default_dof_pos offset contract is disabled")
        if not bool(getattr(highstep_action_cfg, "preserve_order", False)):
            raise RuntimeError("Stage B preserve_order contract is disabled")

        stage_b_runtime_contract = _assert_stage_b_runtime_contract(runner.env, runner.alg.policy)
        recovery_manifest = runner.alg.bind_student_recovery_checkpoints(
            loaded_student_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
        )
        recovery_spec_path = (
            "/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/"
            "highstep_student_recovery_spec_20260712.md"
        )
        recovery_manifest.update(
            {
                "workflow_id": "highstep_student_recovery_20260712_v1",
                "recovery_spec_path": recovery_spec_path,
                "recovery_spec_sha256": checkpoint_sha256(recovery_spec_path),
                "task": args_cli.task,
                "checkpoint_load_mode": highstep_checkpoint_load_mode,
                "schedule_resume_mode": highstep_schedule_resolved_mode,
                "parent_teacher_manifest": args_cli.highstep_parent_teacher_manifest,
                "action_joint_order": action_joint_order,
                "action_scale": action_scale_contract,
                "joint_pos_clip": {key: list(value) for key, value in action_clip_contract.items()},
                "use_default_offset": True,
                "preserve_order": True,
                "runtime_contract": stage_b_runtime_contract,
            }
        )
        recovery_manifest_path = write_manifest(
            os.path.join(log_dir, "params", "highstep_student_recovery_binding.json"),
            recovery_manifest,
        )
        print(f"[INFO] Stage B dual-checkpoint binding manifest: {recovery_manifest_path}")

    elif student_recovery_stage == "R2":
        preregistration, r2_authority = _load_r2_preregistration_contract(runner.alg.policy)
        _validate_r2_load_request(
            preregistration,
            loaded_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
            schedule_resume_mode=highstep_schedule_resolved_mode,
            resume_enabled=bool(agent_cfg.resume),
            legacy_student_count=legacy_student_distill_update_count,
        )
        r2_action_config_contract = _assert_r2_action_config_contract(
            highstep_action_cfg, preregistration
        )
        r2_runtime_contract = _assert_r2_runtime_contract(
            runner.env, runner.alg.policy, preregistration
        )
        r2_binding_manifest = runner.alg.bind_student_recovery_r2_checkpoints(
            loaded_student_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
            schedule_resume_mode=highstep_schedule_resolved_mode,
            runtime_contract=r2_runtime_contract,
        )
        if not isinstance(r2_binding_manifest, dict):
            raise RuntimeError(
                "R2 algorithm checkpoint binding must return a JSON-compatible manifest dict"
            )
        r2_train_binding = {
            **r2_authority,
            "stage": "R2",
            "task": args_cli.task,
            "checkpoint_load_mode": highstep_checkpoint_load_mode,
            "schedule_resume_mode": highstep_schedule_resolved_mode,
            "loaded_checkpoint": os.path.realpath(resume_path),
            "loaded_checkpoint_sha256": checkpoint_sha256(resume_path),
            "parent_teacher_manifest": args_cli.highstep_parent_teacher_manifest,
            "action_config_contract": r2_action_config_contract,
            "train_runtime_contract": r2_runtime_contract,
        }
        for key, expected_value in r2_train_binding.items():
            if key in r2_binding_manifest and r2_binding_manifest[key] != expected_value:
                raise RuntimeError(
                    f"R2 algorithm/train binding disagreement at {key}: "
                    f"algorithm={r2_binding_manifest[key]!r}, train={expected_value!r}"
                )
            r2_binding_manifest[key] = expected_value
        r2_binding_manifest_path = write_manifest(
            os.path.join(log_dir, "params", "highstep_student_recovery_r2_binding.json"),
            r2_binding_manifest,
        )
        print(f"[INFO] R2 immutable dual-anchor binding manifest: {r2_binding_manifest_path}")

    elif student_recovery_stage == "R3":
        preregistration, r3_authority = _load_r3_preregistration_contract(runner.alg.policy)
        _validate_r3_load_request(
            preregistration,
            loaded_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
            schedule_resume_mode=highstep_schedule_resolved_mode,
            resume_enabled=bool(agent_cfg.resume),
            legacy_student_count=legacy_student_distill_update_count,
        )
        r3_action_config_contract = _assert_r2_action_config_contract(
            highstep_action_cfg, preregistration
        )
        r3_runtime_contract = _assert_r2_runtime_contract(
            runner.env, runner.alg.policy, preregistration
        )
        r3_binding_manifest = runner.alg.bind_student_recovery_r3_checkpoints(
            loaded_student_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
            schedule_resume_mode=highstep_schedule_resolved_mode,
            runtime_contract=r3_runtime_contract,
        )
        if not isinstance(r3_binding_manifest, dict):
            raise RuntimeError("R3 algorithm checkpoint binding must return a manifest dict")
        r3_train_binding = {
            **r3_authority,
            "stage": "R3",
            "task": args_cli.task,
            "checkpoint_load_mode": highstep_checkpoint_load_mode,
            "schedule_resume_mode": highstep_schedule_resolved_mode,
            "loaded_checkpoint": os.path.realpath(resume_path),
            "loaded_checkpoint_sha256": checkpoint_sha256(resume_path),
            "parent_teacher_manifest": args_cli.highstep_parent_teacher_manifest,
            "action_config_contract": r3_action_config_contract,
            "train_runtime_contract": r3_runtime_contract,
        }
        for key, expected_value in r3_train_binding.items():
            if key in r3_binding_manifest and r3_binding_manifest[key] != expected_value:
                raise RuntimeError(
                    f"R3 algorithm/train binding disagreement at {key}: "
                    f"algorithm={r3_binding_manifest[key]!r}, train={expected_value!r}"
                )
            r3_binding_manifest[key] = expected_value
        r3_binding_manifest_path = write_manifest(
            os.path.join(log_dir, "params", "highstep_student_recovery_r3_binding.json"),
            r3_binding_manifest,
        )
        print(f"[INFO] R3 immutable dual-anchor binding manifest: {r3_binding_manifest_path}")

    elif student_recovery_stage in {
        "V15", "0707_EXACT", "HISTORICAL_0707_EXACT", "ENV_CURRICULUM_V18",
        "BE300_0707", "B300_CANONICAL_HYBRID",
    }:
        zero_scale_ablation = student_recovery_stage == "0707_EXACT"
        historical_0707_exact = student_recovery_stage == "HISTORICAL_0707_EXACT"
        environment_curriculum_v18 = student_recovery_stage == "ENV_CURRICULUM_V18"
        be300_0707 = student_recovery_stage == "BE300_0707"
        b300_hybrid = student_recovery_stage == "B300_CANONICAL_HYBRID"
        route_label = (
            "B300 canonical hybrid-prior latent distillation" if b300_hybrid
            else "B-E300 0707 distillation" if be300_0707
            else "v1.8 environment curriculum" if environment_curriculum_v18
            else "historical 0707 exact" if historical_0707_exact
            else "zero-scale ablation" if zero_scale_ablation
            else "v1.5"
        )
        if not agent_cfg.resume:
            raise RuntimeError(f"{route_label} Stage-2 requires an explicit model_172300 checkpoint load")
        if legacy_student_distill_update_count is not None:
            raise RuntimeError(f"{route_label} forbids legacy Student count migration")
        if b300_hybrid:
            allowed_b300_schedule = (
                highstep_checkpoint_load_mode == "weights_only"
                and highstep_schedule_resolved_mode == "reset"
            ) or (
                highstep_checkpoint_load_mode == "full"
                and highstep_schedule_resolved_mode == "preserve"
            )
            if not allowed_b300_schedule:
                raise RuntimeError(
                    "B300 hybrid requires fresh weights-only/reset or full/preserve resume"
                )
        elif (
            args_cli.task in single_run_7400_tasks
            and not (
                highstep_checkpoint_load_mode == "weights_only"
                and highstep_schedule_resolved_mode == "reset"
            )
        ):
            raise RuntimeError(
                "single-run 7400 requires fresh weights-only load with frozen 0707 Student schedule reset"
            )
        elif (
            args_cli.task not in single_run_7400_tasks
            and not environment_curriculum_v18
            and highstep_schedule_resolved_mode != "preserve"
        ):
            raise RuntimeError(f"{route_label} requires checkpoint-paired schedule preserve")
        if environment_curriculum_v18:
            allowed_v18_schedule = (
                highstep_checkpoint_load_mode == "weights_only"
                and highstep_schedule_resolved_mode == "reset"
                and v18_schedule_anchor is None
            ) or (
                highstep_checkpoint_load_mode == "full"
                and highstep_schedule_resolved_mode == "preserve"
                and v18_schedule_anchor is None
            ) or (
                highstep_checkpoint_load_mode == "full"
                and highstep_schedule_resolved_mode == "reset"
                and v18_schedule_anchor is not None
                and v18_transition is not None
            )
            if not allowed_v18_schedule:
                raise RuntimeError("v1.8 schedule resume request is not a preregistered course transition")
        if highstep_checkpoint_load_mode == "weights_only":
            expected_root = os.path.realpath(
                str(runner.alg.policy.student_recovery_source_checkpoint)
            )
            if os.path.realpath(resume_path) != expected_root:
                raise RuntimeError(f"fresh {route_label} must weights-only load model_172300")
        expected_joint_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
        ]
        action_joint_order = list(getattr(highstep_action_cfg, "joint_names", []))
        action_scale_contract = dict(getattr(highstep_action_cfg, "scale", {}))
        action_clip_contract = dict(getattr(highstep_action_cfg, "clip", {}))
        if action_joint_order != expected_joint_order:
            raise RuntimeError(f"{route_label} action/joint order contract changed: {action_joint_order}")
        if action_scale_contract != {
            ".*_box_joint": 0.02,
            ".*_(hip_joint|thigh_joint|calf_joint)$": 0.1,
        }:
            raise RuntimeError(f"{route_label} action_scale contract changed: {action_scale_contract}")
        if action_clip_contract != {".*": (-60.0, 60.0)}:
            raise RuntimeError(f"{route_label} joint_pos.clip contract changed: {action_clip_contract}")
        if not bool(getattr(highstep_action_cfg, "use_default_offset", False)):
            raise RuntimeError(f"{route_label} default_dof_pos offset contract is disabled")
        if not bool(getattr(highstep_action_cfg, "preserve_order", False)):
            raise RuntimeError(f"{route_label} preserve_order contract is disabled")
        v15_runtime_contract = _assert_stage_b_runtime_contract(
            runner.env,
            runner.alg.policy,
            allow_critical_transition_context=(
                args_cli.task in {
                    "RobotLab-Isaac-Velocity-HighstepB300CriticalTransitionBalancedDiagonalFresh7400StudentNoPrior-ArcdogAdjustableLeg-v0",
                    "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-ArcdogAdjustableLeg-v0",
                }
            ),
            critical_transition_context_dim=(
                5
                if args_cli.task == (
                    "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-"
                    "ArcdogAdjustableLeg-v0"
                )
                else 4
            ),
        )
        v15_binding_manifest = runner.alg.bind_student_recovery_v15_checkpoints(
            loaded_student_checkpoint=resume_path,
            checkpoint_load_mode=highstep_checkpoint_load_mode,
        )
        v15_binding_manifest.update(
            {
                "workflow_id": (
                    runner.alg._v15_preregistration.get("workflow_id")
                    if be300_0707 or b300_hybrid else
                    "highstep_student_env_curriculum_v18_20260714"
                    if environment_curriculum_v18 else
                    "highstep_historical_0707_exact_new_teacher_20260714"
                    if historical_0707_exact else
                    "highstep_0707_exact_new_teacher_20260713"
                    if zero_scale_ablation else
                    "highstep_student_recovery_v15_20260713"
                ),
                "recovery_spec_path": (
                    runner.alg._v15_preregistration.get("spec_path")
                    if b300_hybrid else
                    "/home/lxq/Softwares/robot_lab/docs/robotlab_memory_zh/library/"
                    "highstep_student_recovery_spec_20260712.md"
                ),
                "recovery_spec_sha256": (
                    runner.alg._v15_preregistration.get("authority", {}).get("spec_sha256")
                    if be300_0707 or b300_hybrid else
                    "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"
                    if environment_curriculum_v18 else
                    "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917"
                    if historical_0707_exact else
                    "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"
                    if zero_scale_ablation else
                    "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
                ),
                "task": args_cli.task,
                "checkpoint_load_mode": highstep_checkpoint_load_mode,
                "schedule_resume_mode": highstep_schedule_resolved_mode,
                "parent_teacher_manifest": args_cli.highstep_parent_teacher_manifest,
                "action_joint_order": action_joint_order,
                "action_scale": action_scale_contract,
                "joint_pos_clip": {
                    key: list(value) for key, value in action_clip_contract.items()
                },
                "runtime_contract": v15_runtime_contract,
                "v18_schedule_anchor_effective_update": v18_schedule_anchor,
                "v18_stage_transition_manifest": (
                    transition_path if environment_curriculum_v18 and v18_transition is not None else None
                ),
            }
        )
        v15_binding_manifest_path = write_manifest(
            os.path.join(
                log_dir,
                "params",
                "highstep_historical_0707_exact_binding.json"
                if historical_0707_exact else
                "highstep_b300_canonical_hybrid_binding.json"
                if b300_hybrid else
                "highstep_be300_0707_binding.json"
                if be300_0707 else
                "highstep_student_environment_curriculum_v18_binding.json"
                if environment_curriculum_v18 else
                "highstep_zero_scale_ablation_binding.json"
                if zero_scale_ablation else
                "highstep_student_recovery_v15_binding.json",
            ),
            v15_binding_manifest,
        )
        print(f"[INFO] {route_label} strict dual-checkpoint binding manifest: {v15_binding_manifest_path}")

    if is_highstep_schedule_task:
        if student_parent_lineage is not None:
            # Re-read both the source schedule sidecar and the pinned Teacher
            # files after runner.load.  Any replacement between initial path
            # resolution and the actual checkpoint load fails closed.
            _, current_lineage_source_manifest = load_source_manifest(resume_path)
            current_student_parent_lineage = resolve_student_parent_lineage(
                task=args_cli.task,
                checkpoint_path=resume_path,
                parent_teacher_manifest_path=args_cli.highstep_parent_teacher_manifest,
                source_manifest=current_lineage_source_manifest,
            )
            if current_student_parent_lineage != student_parent_lineage:
                raise ScheduleContinuityError(
                    "Student parent Teacher lineage changed while the checkpoint was loading"
                )
        if checkpoint_iteration is not None and checkpoint_iteration != pre_checkpoint_iteration:
            raise RuntimeError(
                "Checkpoint iteration changed between pre-wrapper metadata read and runner.load: "
                f"{pre_checkpoint_iteration} != {checkpoint_iteration}"
            )
        runner_iteration_at_anchor = int(getattr(runner, "current_learning_iteration", 0))
        if checkpoint_iteration is None:
            schedule_update_at_anchor = 0.0
        else:
            schedule_update_at_anchor = checkpoint_schedule_update

        install_global_update(
            env,
            schedule_update_at_anchor,
            source=str(schedule_source.get("method", "unknown")),
            resume_mode=highstep_schedule_resolved_mode,
            runner_iteration=runner_iteration_at_anchor,
        )
        schedule_manifest = build_schedule_manifest(
            context="train",
            task=args_cli.task,
            env=env,
            action_cfg=highstep_action_cfg,
            terrain_term_cfg=highstep_terrain_term_cfg,
            support_term_cfg=highstep_support_term_cfg,
            terrain_curriculum_enabled=highstep_terrain_term_cfg is not None,
            support_metric_enabled=highstep_support_term_cfg is not None,
            checkpoint_path=resume_path if checkpoint_iteration is not None else None,
            checkpoint_iteration=checkpoint_iteration,
            checkpoint_load_mode=(highstep_checkpoint_load_mode if checkpoint_iteration is not None else "fresh"),
            requested_resume_mode=highstep_schedule_requested_mode,
            resolved_resume_mode=highstep_schedule_resolved_mode,
            runner_iteration_at_anchor=runner_iteration_at_anchor,
            schedule_update_at_anchor=schedule_update_at_anchor,
            schedule_source=schedule_source,
            command_term_cfg=highstep_command_term_cfg,
            command_curriculum_enabled=highstep_command_curriculum_enabled,
            runtime_state_path=highstep_runtime_state_path,
            command_state_restored=command_state_restored,
            moving_best_state_restored=moving_best_state_restored,
            clean_ab_comparison_allowed=False,
            reward_stage_definition=highstep_reward_stage_contract,
            student_parent_lineage=student_parent_lineage,
        )
        if any(
            marker in str(args_cli.task)
            for marker in ("HighstepFrontGeometryV1123", "HighstepFrontGeometryV114")
        ):
            schedule_manifest["fresh_initial_training_distribution"] = (
                _highstep_fresh_initial_terrain_distribution(env)
            )
        schedule_manifest_path = write_manifest(
            os.path.join(log_dir, "params", SCHEDULE_MANIFEST_NAME), schedule_manifest
        )
        print(f"[INFO] Highstep schedule manifest: {schedule_manifest_path}")
        print(
            "[INFO] Highstep schedule active: "
            + json.dumps(
                {
                    "update": schedule_manifest["schedule_update_at_anchor"],
                    "prior_scale": schedule_manifest["action_prior"]["actual_prior_scale"],
                    "terrain_stage": schedule_manifest["terrain_schedule"]["configured_stage_index"],
                    "terrain_max_level": schedule_manifest["terrain_schedule"]["configured_allowed_max_level"],
                    "support_blend": schedule_manifest["support_bottleneck"]["actual_blend"],
                },
                sort_keys=True,
            )
        )
        print(
            "[INFO] Highstep clean A/B eligibility: false "
            "(checkpoint-paired runtime state is tracked, but simulator/RNG/terrain population state is not)."
        )
        if IS_MASTER:
            _install_highstep_checkpoint_state_hook(
                runner,
                env,
                runtime_state_path=highstep_runtime_state_path,
                num_steps_per_update=highstep_num_steps_per_update,
                command_curriculum_enabled=highstep_command_curriculum_enabled,
                moving_best_required=highstep_moving_best_required,
            )
            if args_cli.task in single_run_7400_tasks:
                _install_relative_student_checkpoint_cadence(
                    runner,
                    absolute_origin=int(absolute_runner_step_origin),
                    interval=100,
                )
            # The sparse RL-pre-edge continuation uses its own effective-update
            # driver below.  Installing the generic update wrapper here would
            # reintroduce stock learn(1) final-save path collisions.

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
        time.sleep(0.1)
    if args_cli.frozen_fc_mu_diagnostic_output:
        if not args_cli.frozen_fc_mu_diagnostic_preregistration:
            raise RuntimeError("fc_mu diagnostic requires its immutable preregistration")
        from frozen_fc_mu_observability_diagnostic import run_frozen_fc_mu_observability_diagnostic

        run_frozen_fc_mu_observability_diagnostic(
            runner=runner,
            checkpoint_path=resume_path,
            preregistration_path=args_cli.frozen_fc_mu_diagnostic_preregistration,
            output_path=args_cli.frozen_fc_mu_diagnostic_output,
            burn_in_steps=args_cli.frozen_diagnostic_burn_in_steps,
        )
        env.close()
        return

    if args_cli.v114_frozen_gate_report:
        if not args_cli.v114_frozen_gate_dataset:
            raise RuntimeError("v1.14 frozen gate requires --v114_frozen_gate_dataset")
        from highstep_v114_frozen_gate import run_v114_frozen_gate

        run_v114_frozen_gate(
            runner=runner,
            checkpoint_path=resume_path,
            dataset_path=args_cli.v114_frozen_gate_dataset,
            report_path=args_cli.v114_frozen_gate_report,
            create_dataset=args_cli.v114_create_frozen_gate_dataset,
            burn_in_steps=args_cli.frozen_diagnostic_burn_in_steps,
            expected_dataset_sha256=args_cli.v114_expected_frozen_gate_dataset_sha256,
        )
        env.close()
        return

    if args_cli.frozen_diagnostic_output:
        from frozen_training_buffer_diagnostic import run_frozen_training_buffer_diagnostic

        run_frozen_training_buffer_diagnostic(
            runner=runner,
            checkpoint_path=resume_path,
            output_path=args_cli.frozen_diagnostic_output,
            burn_in_steps=args_cli.frozen_diagnostic_burn_in_steps,
            workflow_id="highstep_historical_0707_exact_new_teacher_20260714",
            spec_sha256="140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917",
            preregistration_sha256="7b1bb49a9640e8614313423fce7e66fadc40e38bfb1ce551b78fb02dfbb9c035",
        )
        env.close()
        return

    # Bind the v1.8 environment course to a read-only profile manifest.  The
    # task config has already restored and verified the source snapshot before
    # environment construction; this records the exact runtime witness next to
    # every checkpoint.
    v18_profile = None
    is_v18_task = "V18Bootstrap" in str(args_cli.task) or "V18Robust" in str(args_cli.task)
    if is_v18_task:
        manifest_path = os.path.realpath(
            str(args_cli.highstep_v18_environment_profile_manifest or "")
        )
        expected_manifest_sha = str(
            args_cli.highstep_v18_environment_profile_manifest_sha256 or ""
        )
        if not manifest_path or not os.path.isabs(manifest_path) or not expected_manifest_sha:
            raise RuntimeError("v1.8 training requires a hash-bound environment profile manifest")
        with open(manifest_path, "rb") as stream:
            actual_manifest_sha = hashlib.sha256(stream.read()).hexdigest()
        if actual_manifest_sha != expected_manifest_sha:
            raise RuntimeError("v1.8 environment profile manifest SHA mismatch")
        with open(manifest_path, encoding="utf-8") as stream:
            profile_manifest = json.load(stream)
        from robot_lab.tasks.locomotion.velocity.config.quadruped.Arcdog_adjustable_leg.highstep_env_cfg import (
            v18_environment_profile_contract,
        )
        v18_profile = v18_environment_profile_contract(str(args_cli.task))
        if not (
            profile_manifest.get("task") == str(args_cli.task)
            and profile_manifest.get("source_env_yaml") == v18_profile["source_env_yaml"]
            and profile_manifest.get("source_env_yaml_sha256")
            == v18_profile["source_env_yaml_sha256"]
            and profile_manifest.get("immutable") is True
        ):
            raise RuntimeError("v1.8 environment task/profile binding mismatch")
    elif args_cli.highstep_v18_environment_profile_manifest:
        raise RuntimeError("v1.8 environment profile manifest supplied to a non-v1.8 task")

    # dump the configuration into log-directory
    runtime_env_yaml = os.path.join(log_dir, "params", "env.yaml")
    dump_yaml(runtime_env_yaml, env_cfg)
    if v18_profile is not None:
        with open(runtime_env_yaml, "rb") as stream:
            runtime_env_sha = hashlib.sha256(stream.read()).hexdigest()
        binding_path = os.path.join(log_dir, "params", "highstep_v18_environment_binding.json")
        binding = {
            "schema_version": 1,
            "kind": "highstep_v18_runtime_environment_binding",
            "task": str(args_cli.task),
            "profile_manifest": manifest_path,
            "profile_manifest_sha256": expected_manifest_sha,
            **v18_profile,
            "runtime_env_yaml": os.path.realpath(runtime_env_yaml),
            "runtime_env_yaml_sha256": runtime_env_sha,
            "source_snapshot_projection_verified_before_environment_creation": True,
        }
        temporary = binding_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(binding, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, binding_path)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # run training
    ac = runner.alg.policy  # 某些版本也叫 runner.alg.actor_critic
    # print(">> Actor type:", ac.actor.__class__.__name__)  # 期望看到 _MoEActor
    _setup_hash_bound_wandb_config(agent_cfg.logger)
    if args_cli.task == (
        "RobotLab-Isaac-Velocity-HighstepB300RLPreEdgeContinuationE7700StudentNoPrior-"
        "ArcdogAdjustableLeg-v0"
    ):
        from highstep_effective_update_driver import run_sparse_effective_update_continuation

        run_sparse_effective_update_continuation(
            runner,
            target_effective_update=7700,
            checkpoint_interval=100,
            checkpoint_absolute_origin=173499,
            progress_path=os.environ.get("HIGHSTEP_EFFECTIVE_PROGRESS_PATH"),
        )
    else:
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
