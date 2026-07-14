"""Read-only frozen training-buffer diagnosis for historical_0707_exact.

This module is deliberately called only after ``train.py`` has restored and
validated the full Student/Teacher/optimizer/schedule binding.  It never calls
``runner.learn()``, ``backward()``, or an optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


PHASES = (
    "approach",
    "front_lift",
    "front_support",
    "first_rear",
    "second_rear",
    "rear_hold",
)
CRITICAL_LATENT_DIMS = (29, 35, 39, 42, 47, 52)
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
REAR_FOCUS_JOINTS = (6, 7, 10, 11)


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_hash(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _update_hash(digest, str(key))
            _update_hash(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode() + b"\0")
        for item in value:
            _update_hash(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode() + b"\0")
        digest.update(repr(value).encode())
        digest.update(b"\0")
    else:
        raise TypeError(f"unsupported hash value: {type(value).__name__}")


def _state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, state)
    return digest.hexdigest()


def _module_sha256(module: torch.nn.Module) -> str:
    return _state_sha256(dict(module.state_dict()))


def _optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    return _state_sha256(optimizer.state_dict())


def _summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"count": 0, "mean": None, "q95": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "q95": float(torch.quantile(values, 0.95).item()),
        "max": float(values.max().item()),
    }


def _sign_agreement(target: torch.Tensor, prediction: torch.Tensor) -> float | None:
    target = target.detach().float().reshape(-1)
    prediction = prediction.detach().float().reshape(-1)
    valid = (target.abs() > 1.0e-6) | (prediction.abs() > 1.0e-6)
    if not bool(valid.any().item()):
        return None
    return float((torch.sign(target[valid]) == torch.sign(prediction[valid])).float().mean().item())


class _VectorizedPhaseTracker:
    """Training-terrain phase tracker using the live action-term sensors."""

    def __init__(self, env: Any):
        self.env = env.unwrapped
        terms = self.env.action_manager._terms
        if list(terms) != ["joint_pos"]:
            raise RuntimeError(f"unexpected action terms: {list(terms)}")
        self.term = terms["joint_pos"]
        self.asset = self.env.scene["robot"]
        self.height_sensor = self.env.scene["height_scanner"]
        ray_starts = self.height_sensor.ray_starts[0]
        side = torch.abs(ray_starts[:, 1]) <= 0.30
        self.front_ray_mask = (ray_starts[:, 0] >= 0.25) & side
        self.rear_ray_mask = (ray_starts[:, 0] <= -0.20) & side
        if not bool(self.front_ray_mask.any().item()) or not bool(self.rear_ray_mask.any().item()):
            raise RuntimeError("training height scanner has empty canonical front/rear masks")
        self.front_foot_ids, front_names = self.asset.find_bodies(
            ["FL_foot", "FR_foot"], preserve_order=True
        )
        self.rear_foot_ids, rear_names = self.asset.find_bodies(
            ["RL_foot", "RR_foot"], preserve_order=True
        )
        if list(front_names) != ["FL_foot", "FR_foot"] or list(rear_names) != ["RL_foot", "RR_foot"]:
            raise RuntimeError(f"robot foot order changed: front={front_names}, rear={rear_names}")
        try:
            self.contact = self.env.scene.sensors["contact_forces"]
        except (KeyError, AttributeError) as error:
            raise RuntimeError("training environment has no contact_forces sensor") from error
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.contact_ids, resolved = self.contact.find_bodies(foot_names, preserve_order=True)
        if list(resolved) != foot_names:
            raise RuntimeError(f"contact foot order changed: {resolved}")
        self.num_envs = int(self.env.num_envs)
        device = self.env.device
        self.stage = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.seen_step = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self.front_lift_streak = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.front_top_streak = torch.zeros_like(self.front_lift_streak)
        self.rear_top_streak = torch.zeros(self.num_envs, 2, dtype=torch.long, device=device)
        self.rear_hold_streak = torch.zeros_like(self.front_lift_streak)
        self.stage_age = torch.zeros_like(self.front_lift_streak)
        self.last_stage_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    @staticmethod
    def _streak(previous: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return torch.where(condition, previous + 1, torch.zeros_like(previous))

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        selected = values[:, mask]
        valid = torch.isfinite(selected) & (torch.abs(selected) < 1.0e6)
        count = valid.float().sum(dim=1)
        total = torch.where(valid, selected, torch.zeros_like(selected)).sum(dim=1)
        return torch.where(count > 0.0, total / torch.clamp(count, min=1.0), fallback)

    def reset(self, dones: torch.Tensor) -> None:
        mask = dones.reshape(-1).bool()
        if not bool(mask.any().item()):
            return
        self.stage[mask] = 0
        self.seen_step[mask] = False
        self.front_lift_streak[mask] = 0
        self.front_top_streak[mask] = 0
        self.rear_top_streak[mask] = 0
        self.rear_hold_streak[mask] = 0
        self.stage_age[mask] = 0
        self.last_stage_entered[mask] = False

    def sample(self) -> torch.Tensor:
        ray_z = self.height_sensor.data.ray_hits_w[..., 2]
        sensor_z = self.height_sensor.data.pos_w[:, 2]
        front_ground = self._masked_mean(ray_z, self.front_ray_mask, sensor_z)
        rear_ground = self._masked_mean(ray_z, self.rear_ray_mask, front_ground)
        height_delta = front_ground - rear_ground
        command_x = self.env.command_manager.get_command("base_velocity")[:, 0]
        visible = (height_delta >= 0.060) & (command_x >= 0.06)
        self.seen_step |= visible

        foot_ids = list(self.front_foot_ids) + list(self.rear_foot_ids)
        foot_z = self.asset.data.body_pos_w[:, foot_ids, 2]
        top_z = torch.maximum(front_ground, rear_ground).unsqueeze(1)
        low_z = torch.minimum(front_ground, rear_ground).unsqueeze(1)
        top_height = (foot_z >= top_z - 0.04) & (foot_z <= top_z + 0.09)

        data = self.contact.data
        if getattr(data, "net_forces_w", None) is not None:
            force = data.net_forces_w[:, self.contact_ids, :]
        elif getattr(data, "net_forces_w_history", None) is not None:
            force = data.net_forces_w_history[:, 0, self.contact_ids, :]
        else:
            raise RuntimeError("contact_forces has no current force tensor")
        norm = torch.linalg.norm(force, dim=-1)
        upward = torch.clamp(force[..., 2], min=0.0)
        loaded = (upward > 5.0) & (upward / torch.clamp(norm, min=1.0e-6) >= 0.45)
        top_loaded = top_height & loaded & self.seen_step.unsqueeze(1)

        front_lift_now = torch.any(foot_z[:, :2] > low_z + 0.08, dim=1) & self.seen_step
        front_top_now = torch.any(top_loaded[:, :2], dim=1)
        rear_top_now = top_loaded[:, 2:]
        self.front_lift_streak = self._streak(self.front_lift_streak, front_lift_now)
        self.front_top_streak = self._streak(self.front_top_streak, front_top_now)
        self.rear_top_streak = self._streak(self.rear_top_streak, rear_top_now)
        rear_count = torch.sum(self.rear_top_streak >= 2, dim=1)
        rear_hold_now = rear_count == 2
        self.rear_hold_streak = self._streak(self.rear_hold_streak, rear_hold_now)

        current = torch.zeros_like(self.stage)
        current = torch.where(self.front_lift_streak >= 2, 1, current)
        current = torch.where(self.front_top_streak >= 2, 2, current)
        current = torch.where(rear_count >= 1, 3, current)
        current = torch.where(rear_count >= 2, 4, current)
        current = torch.where(self.rear_hold_streak >= 25, 5, current)
        next_stage = torch.maximum(self.stage, current)
        entered = next_stage > self.stage
        self.stage_age = torch.where(entered, torch.zeros_like(self.stage_age), self.stage_age + 1)
        self.last_stage_entered = entered
        self.stage = next_stage
        # Samples that have not yet seen a high step remain approach samples;
        # the manifest separately records seen-step coverage.
        return self.stage.clone()


def _loss_terms(alg: Any, policy_obs: torch.Tensor, est_input: torch.Tensor, critic_obs: torch.Tensor):
    model = alg.policy
    vel_pred, recon, mu, logvar, _ = model.estimator(est_input)
    with torch.no_grad():
        target_vel = critic_obs[:, :3]
        target_latent = alg.teacher_priv_encoder(critic_obs[:, 3:])
        safe_target = torch.clamp(target_latent, -1.0, 1.0)
        teacher_action = alg.teacher_actor(torch.cat((policy_obs, safe_target), dim=-1))
        _, post_prior_gate, _, _, _ = alg._apply_phased_highstep_box_prior_to_raw_action(
            teacher_action, critic_obs
        )
    safe_mu = torch.clamp(mu, -1.0, 1.0)
    student_action = model.actor(torch.cat((policy_obs, safe_mu), dim=-1))
    error = torch.square(student_action - teacher_action)
    non_box = torch.mean(error[:, :12], dim=1)
    raw_box = torch.mean(error[:, 12:], dim=1)
    rear_box = torch.mean(error[:, 14:], dim=1)
    command_speed = torch.linalg.norm(critic_obs[:, 9:12], dim=1)
    moving_weight = 1.0 + torch.clamp(command_speed / float(alg.student_prior_fade_speed), 0.0, 1.0)
    phase_weight = 1.0 + float(alg.student_highstep_phase_loss_scale) * post_prior_gate.detach()
    per_sample_action = 2.0 * non_box + 0.5 * raw_box
    per_sample_action = per_sample_action + float(alg.student_highstep_rear_box_loss_scale) * post_prior_gate.detach() * rear_box
    action = torch.mean(per_sample_action * moving_weight * phase_weight)
    terms = {
        "action": action,
        "latent": torch.mean(torch.square(mu - target_latent)),
        "velocity": torch.mean(torch.square(vel_pred - target_vel)),
        "recon": torch.mean(torch.square(recon - est_input)),
        "kl": -0.5 * torch.sum(1.0 + logvar - mu.square() - logvar.exp(), dim=1).mean(),
    }
    return terms, {
        "mu": mu,
        "target_latent": target_latent,
        "student_action": student_action,
        "teacher_action": teacher_action,
        "post_prior_gate": post_prior_gate,
    }


def _gradient_evidence(
    alg: Any,
    policy_obs: torch.Tensor,
    est_input: torch.Tensor,
    critic_obs: torch.Tensor,
    phase_ids: torch.Tensor,
) -> dict[str, Any]:
    torch.manual_seed(701407)
    terms, tensors = _loss_terms(alg, policy_obs, est_input, critic_obs)
    coefficients = {"action": 20.0, "latent": 50.0, "velocity": 10.0, "recon": 0.5, "kl": 0.1}
    parameters = tuple(alg.policy.estimator.parameters())
    vectors: dict[str, torch.Tensor] = {}
    raw_norms: dict[str, float] = {}
    weighted_norms: dict[str, float] = {}
    for name in ("action", "latent", "velocity", "recon", "kl"):
        raw = torch.autograd.grad(terms[name], parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([
            (torch.zeros_like(parameter) if item is None else item).reshape(-1)
            for parameter, item in zip(parameters, raw, strict=True)
        ])
        raw_norms[name] = float(torch.linalg.vector_norm(vector).item())
        vectors[name] = vector * coefficients[name]
        weighted_norms[name] = float(torch.linalg.vector_norm(vectors[name]).item())
    cosines: dict[str, float | None] = {}
    names = tuple(vectors)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            denom = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            cosines[f"{left}__{right}"] = (
                None if float(denom.item()) == 0.0
                else float(torch.dot(vectors[left], vectors[right]).div(denom).item())
            )
    rear_mask = phase_ids >= 3
    rear_gradient = None
    if bool(rear_mask.any().item()):
        rear_action = torch.mean(torch.square(
            tensors["student_action"][rear_mask, :12] - tensors["teacher_action"][rear_mask, :12]
        ))
        rear_raw = torch.autograd.grad(rear_action, parameters, retain_graph=False, allow_unused=True)
        rear_vector = torch.cat([
            (torch.zeros_like(parameter) if item is None else item).reshape(-1)
            for parameter, item in zip(parameters, rear_raw, strict=True)
        ]) * 20.0
        global_action = vectors["action"]
        denom = torch.linalg.vector_norm(rear_vector) * torch.linalg.vector_norm(global_action)
        rear_gradient = {
            "samples": int(rear_mask.sum().item()),
            "weighted_norm": float(torch.linalg.vector_norm(rear_vector).item()),
            "cosine_with_global_action": (
                None if float(denom.item()) == 0.0
                else float(torch.dot(rear_vector, global_action).div(denom).item())
            ),
        }
    return {
        "method": "torch.autograd.grad only; no backward and no optimizer.step",
        "sample_count": int(policy_obs.shape[0]),
        "coefficients": coefficients,
        "raw_loss_values": {name: float(value.detach().item()) for name, value in terms.items()},
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norms": weighted_norms,
        "pairwise_weighted_gradient_cosines": cosines,
        "rear_nonbox_action_gradient": rear_gradient,
    }


def _select_gradient_indices(phases: torch.Tensor, cap: int = 8192) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    per_phase = max(1, cap // len(PHASES))
    generator = torch.Generator(device=phases.device).manual_seed(701407)
    for phase_id in range(len(PHASES)):
        indices = torch.nonzero(phases == phase_id, as_tuple=False).reshape(-1)
        if indices.numel() > per_phase:
            order = torch.randperm(indices.numel(), generator=generator, device=indices.device)[:per_phase]
            indices = indices[order]
        selected.append(indices)
    result = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=phases.device)
    if result.numel() < min(cap, phases.numel()):
        used = torch.zeros(phases.numel(), dtype=torch.bool, device=phases.device)
        used[result] = True
        remaining = torch.nonzero(~used, as_tuple=False).reshape(-1)
        need = min(cap - result.numel(), remaining.numel())
        if need:
            order = torch.randperm(remaining.numel(), generator=generator, device=remaining.device)[:need]
            result = torch.cat((result, remaining[order]))
    return result


def _metric_evidence(alg: Any, observations: Any, phase_ids: torch.Tensor) -> dict[str, Any]:
    model = alg.policy
    policy_obs = observations[model.policy_keys[0]].flatten(0, 1)
    est_input = observations[model.estimator_keys[0]].flatten(0, 1)
    critic_obs = observations[model.critic_keys[0]].flatten(0, 1)
    if policy_obs.shape[1] >= model.policy_obs_dim + model.vae_latent_dim:
        policy_obs = policy_obs[:, :model.policy_obs_dim]
    flat_phases = phase_ids.flatten().to(policy_obs.device)
    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    chunk = 8192
    model.estimator.eval()
    with torch.no_grad():
        for start in range(0, policy_obs.shape[0], chunk):
            end = min(start + chunk, policy_obs.shape[0])
            est = est_input[start:end]
            pol = policy_obs[start:end]
            critic = critic_obs[start:end]
            mu, _, _ = model.estimator.encode(est)
            target = alg.teacher_priv_encoder(critic[:, 3:])
            teacher_action = alg.teacher_actor(torch.cat((pol, torch.clamp(target, -1.0, 1.0)), dim=1))
            student_action = model.actor(torch.cat((pol, torch.clamp(mu, -1.0, 1.0)), dim=1))
            collected["mu"].append(mu.cpu())
            collected["target"].append(target.cpu())
            collected["student_action"].append(student_action.cpu())
            collected["teacher_action"].append(teacher_action.cpu())
    model.estimator.train()
    values = {key: torch.cat(items) for key, items in collected.items()}
    cpu_phases = flat_phases.cpu()
    by_phase: dict[str, Any] = {}
    for phase_id, phase in enumerate(PHASES):
        mask = cpu_phases == phase_id
        count = int(mask.sum().item())
        phase_result: dict[str, Any] = {"samples": count}
        if count:
            latent_dims: dict[str, Any] = {}
            for dim in CRITICAL_LATENT_DIMS:
                target = values["target"][mask, dim]
                prediction = values["mu"][mask, dim]
                latent_dims[str(dim)] = {
                    "target_mean": float(target.mean().item()),
                    "prediction_mean": float(prediction.mean().item()),
                    "absolute_error": _summary(torch.abs(prediction - target)),
                    "sign_agreement": _sign_agreement(target, prediction),
                }
            action_error = torch.abs(values["student_action"][mask] - values["teacher_action"][mask])
            phase_result["critical_latent_dimensions"] = latent_dims
            phase_result["nonbox_12_joint_action_mae"] = float(action_error[:, :12].mean().item())
            focus: dict[str, Any] = {}
            for joint in REAR_FOCUS_JOINTS:
                target = values["teacher_action"][mask, joint]
                prediction = values["student_action"][mask, joint]
                focus[JOINT_NAMES[joint]] = {
                    "absolute_error": _summary(torch.abs(prediction - target)),
                    "sign_agreement": _sign_agreement(target, prediction),
                }
            phase_result["rear_thigh_calf_pre_prior"] = focus
        by_phase[phase] = phase_result

    gradient_indices = _select_gradient_indices(flat_phases)
    model.estimator.train()
    gradient = _gradient_evidence(
        alg,
        policy_obs[gradient_indices],
        est_input[gradient_indices],
        critic_obs[gradient_indices],
        flat_phases[gradient_indices],
    )
    return {
        "pre_prior_only": True,
        "post_prior_action_excluded_from_actor_estimator_metrics": True,
        "phase_metrics": by_phase,
        "gradient_evidence": gradient,
    }


def run_frozen_training_buffer_diagnostic(
    *,
    runner: Any,
    checkpoint_path: str,
    output_path: str,
    burn_in_steps: int,
    workflow_id: str,
    spec_sha256: str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    if burn_in_steps < 0:
        raise ValueError("burn_in_steps must be non-negative")
    alg = runner.alg
    if str(getattr(alg, "student_recovery_stage", "")).upper() != "HISTORICAL_0707_EXACT":
        raise RuntimeError("frozen diagnostic requires HISTORICAL_0707_EXACT")
    if int(getattr(alg, "student_distill_update_count", -1)) != 700:
        raise RuntimeError("frozen diagnostic requires the full E700 update counter")
    if not bool(getattr(alg, "_v15_binding_ready", False)):
        raise RuntimeError("Student/Teacher checkpoint binding is not ready")
    if int(alg.storage.step) != 0:
        raise RuntimeError("rollout storage must be empty before frozen collection")

    optimizers = {"algorithm": alg.optimizer, "vae": alg.vae_optimizer}
    unique_optimizers = {id(value): value for value in optimizers.values()}
    original_steps = {}
    for identity, optimizer in unique_optimizers.items():
        original_steps[identity] = optimizer.step

        def forbidden_step(*args: Any, **kwargs: Any):
            raise RuntimeError("optimizer.step is forbidden in frozen diagnostic mode")

        optimizer.step = forbidden_step

    before = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "actor_tensor_sha256": _module_sha256(alg.policy.actor),
        "estimator_tensor_sha256": _module_sha256(alg.policy.estimator),
        "optimizer_state_sha256": _optimizer_sha256(alg.vae_optimizer),
    }
    phase_tracker = _VectorizedPhaseTracker(runner.env)
    phase_steps: list[torch.Tensor] = []
    runner.train_mode()
    obs = runner.env.get_observations().to(runner.device)
    runner.env.episode_length_buf = torch.randint_like(
        runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
    )
    try:
        with torch.inference_mode():
            for _ in range(burn_in_steps):
                phase_tracker.sample()
                actions = alg.policy.act(obs)
                obs, _, dones, _ = runner.env.step(actions.to(runner.env.device))
                obs, dones = obs.to(runner.device), dones.to(runner.device)
                alg.policy.update_normalization(obs)
                alg.policy.reset(dones)
                phase_tracker.reset(dones)

            for _ in range(runner.num_steps_per_env):
                phase_steps.append(phase_tracker.sample().cpu())
                actions = alg.act(obs)
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                alg.process_env_step(obs, rewards, dones, extras)
                phase_tracker.reset(dones)
        if int(alg.storage.step) != int(runner.num_steps_per_env):
            raise RuntimeError("frozen rollout did not fill exactly one training buffer")
        phases = torch.stack(phase_steps, dim=0)
        counts = {phase: int((phases == index).sum().item()) for index, phase in enumerate(PHASES)}
        total = int(phases.numel())
        phase_distribution = {
            phase: {"samples": counts[phase], "fraction": counts[phase] / total}
            for phase in PHASES
        }
        metrics = _metric_evidence(alg, alg.storage.observations, phases)
    finally:
        for identity, optimizer in unique_optimizers.items():
            optimizer.step = original_steps[identity]

    after = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "actor_tensor_sha256": _module_sha256(alg.policy.actor),
        "estimator_tensor_sha256": _module_sha256(alg.policy.estimator),
        "optimizer_state_sha256": _optimizer_sha256(alg.vae_optimizer),
    }
    unchanged = {key: before[key] == after[key] for key in before}
    if not all(unchanged.values()):
        raise RuntimeError(f"frozen diagnostic mutated protected state: {unchanged}")

    rear_counts = [counts[phase] for phase in ("first_rear", "second_rear", "rear_hold")]
    per_phase_floor = 256
    total_rear_floor = max(1024, math.ceil(total * 0.01))
    if sum(rear_counts) < total_rear_floor or min(rear_counts) < per_phase_floor:
        conclusion = "rear_phase_sample_scarcity"
        rationale = {
            "preregistered_thresholds": {
                "minimum_total_rear_samples": total_rear_floor,
                "minimum_samples_per_rear_phase": per_phase_floor,
            },
            "observed_total_rear_samples": sum(rear_counts),
            "observed_per_rear_phase": dict(zip(("first_rear", "second_rear", "rear_hold"), rear_counts, strict=True)),
        }
        proposal = {
            "single_variable_only": True,
            "status": "proposal_not_executed",
            "variable": "training-buffer sampling weight for first_rear/second_rear/rear_hold samples",
            "change": "preregister one phase-balanced replay/sampling multiplier while keeping all model, target, loss coefficients, optimizer, and environment semantics fixed",
        }
    else:
        rear_gradient = metrics["gradient_evidence"]["rear_nonbox_action_gradient"]
        action_norm = metrics["gradient_evidence"]["weighted_gradient_norms"]["action"]
        rear_norm = None if rear_gradient is None else rear_gradient["weighted_norm"]
        if rear_norm is not None and rear_norm > 0.0 and action_norm / rear_norm < 0.10:
            conclusion = "critical_latent_supervision_dilution"
            rationale = {
                "rear_samples_sufficient": True,
                "global_to_rear_action_gradient_norm_ratio": action_norm / rear_norm,
                "fixed_dilution_threshold": 0.10,
            }
            proposal = {
                "single_variable_only": True,
                "status": "proposal_not_executed",
                "variable": "critical rear-phase sample weight",
                "change": "preregister one rear-phase weighting multiplier; keep loss targets, coefficients, optimizer, and architecture fixed",
            }
        else:
            conclusion = "diagnosis_not_closed"
            rationale = {
                "first_unclosed_tensor": "estimator.fc_mu output contribution for latent dims 29/35/39/42/47/52 on rear_hold samples",
                "rear_samples_sufficient": True,
                "global_action_gradient_norm": action_norm,
                "rear_action_gradient_norm": rear_norm,
            }
            proposal = None

    manifest = {
        "schema_version": 1,
        "kind": "historical_0707_exact_E700_frozen_training_buffer_diagnostic",
        "created_at": datetime.now().astimezone().isoformat(),
        "workflow_id": workflow_id,
        "authority": "v1.7.1",
        "spec_sha256": spec_sha256,
        "preregistration_sha256": preregistration_sha256,
        "checkpoint": os.path.realpath(checkpoint_path),
        "checkpoint_expected_sha256": "8f02861037d2f24370cfc406bab32c8945c7ea09ec7f42cd5636c8423824ebff",
        "frozen_contract": {
            "effective_updates": int(alg.student_distill_update_count),
            "optimizer_step_guard_installed": True,
            "optimizer_step_calls": 0,
            "backward_calls": 0,
            "runner_learn_calls": 0,
            "captured_training_buffers": 1,
            "num_envs": int(runner.env.num_envs),
            "steps_per_buffer": int(runner.num_steps_per_env),
            "burn_in_steps_without_storage_or_updates": int(burn_in_steps),
            "training_action_sampling_used": True,
            "training_environment_used": True,
        },
        "phase_classifier": {
            "source": "live training action-term height scanner + four-foot contact sensor",
            "monotonic_per_episode": True,
            "front_contact_streak": 2,
            "rear_contact_streak": 2,
            "rear_hold_streak": 25,
        },
        "sample_distribution": phase_distribution,
        "analysis": metrics,
        "protected_state_before": before,
        "protected_state_after": after,
        "protected_state_unchanged": unchanged,
        "all_protected_state_unchanged": all(unchanged.values()),
        "conclusion": conclusion,
        "conclusion_evidence": rationale,
        "single_variable_proposal": proposal,
        "training_started": False,
        "next_stage_started": False,
    }
    if before["checkpoint_sha256"] != manifest["checkpoint_expected_sha256"]:
        raise RuntimeError("E700 checkpoint SHA changed before diagnostic")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    manifest_sha = _sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{manifest_sha}  {output.name}\n", encoding="utf-8")
    output.chmod(0o444)
    sidecar.chmod(0o444)
    print(json.dumps({
        "frozen_diagnostic_manifest": str(output),
        "frozen_diagnostic_manifest_sha256": manifest_sha,
        "conclusion": conclusion,
        "sample_distribution": phase_distribution,
        "all_protected_state_unchanged": True,
    }, sort_keys=True), flush=True)
    return manifest
