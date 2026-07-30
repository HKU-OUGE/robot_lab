"""Frozen comparison-buffer support for the approved v1.14 gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from frozen_training_buffer_diagnostic import (
    PHASES,
    _VectorizedPhaseTracker,
    _module_sha256,
    _optimizer_sha256,
    _sha256_file,
)

CRITICAL = ("first_rear", "second_rear", "rear_hold")


def _sha(path: str | os.PathLike[str]) -> str:
    return _sha256_file(path)


def _protected(alg: Any, checkpoint: str) -> dict[str, str]:
    return {
        "checkpoint": _sha(checkpoint),
        "actor": _module_sha256(alg.policy.actor),
        "estimator": _module_sha256(alg.policy.estimator),
        "teacher_actor": _module_sha256(alg.teacher_actor),
        "teacher_priv_encoder": _module_sha256(alg.teacher_priv_encoder),
        "student_priv_encoder": _module_sha256(alg.policy.priv_encoder),
        "optimizer": _optimizer_sha256(alg.vae_optimizer),
    }


def _metrics(alg: Any, data: dict[str, torch.Tensor]) -> dict[str, Any]:
    device = alg.device
    output: dict[str, Any] = {}
    for phase in CRITICAL:
        mask = data["phase_name"] == PHASES.index(phase)
        policy = data["policy"][mask].to(device)
        estimator = data["estimator"][mask].to(device)
        critic = data["critic"][mask].to(device)
        with torch.no_grad():
            raw_mu, _, _ = alg.policy.estimator.encode(estimator)
            clamped_mu = raw_mu.clamp(-1.0, 1.0)
            target = alg.teacher_priv_encoder(critic[:, 3:]).clamp(-1.0, 1.0)
            teacher_action = alg.teacher_actor(torch.cat((policy, target), dim=1))
            teacher_post_prior, _, _, _, _ = (
                alg._apply_phased_highstep_box_prior_to_raw_action(teacher_action, critic)
            )
            student_action = alg.policy.actor(torch.cat((policy, clamped_mu), dim=1))
            latent_abs = (clamped_mu - target).abs()
            action_delta = student_action[:, :12] - teacher_action[:, :12]
            box_delta = student_action[:, 12:16] - teacher_post_prior[:, 12:16]
            teacher_prior_delta = teacher_post_prior[:, 12:16] - teacher_action[:, 12:16]
        output[phase] = {
            "samples": int(mask.sum()),
            "clamped_latent_mae": float(latent_abs.mean()),
            "clamped_latent_mse": float(latent_abs.square().mean()),
            "clamped_latent_q95": float(torch.quantile(latent_abs, 0.95)),
            "nonbox_pre_prior_action_mae": float(action_delta.abs().mean()),
            "nonbox_pre_prior_action_mse": float(action_delta.square().mean()),
            "nonbox_pre_prior_action_max_abs": float(action_delta.abs().max()),
            "teacher_post_minus_pre_prior_box_mae": float(teacher_prior_delta.abs().mean()),
            "post_prior_box_action_mae": float(box_delta.abs().mean()),
            "student_box_vs_teacher_post_prior_mae": float(box_delta.abs().mean()),
            "student_box_vs_teacher_post_prior_mse": float(box_delta.square().mean()),
            "student_box_vs_teacher_post_prior_max_abs": float(box_delta.abs().max()),
            "raw_mu_oob_fraction": float((raw_mu.abs() > 1.0).float().mean()),
            "saturated_wrong_elements": int(
                ((raw_mu.abs() > 1.0) & ((clamped_mu - target).abs() > 1.0e-3)).sum()
            ),
        }
    return output


def _write_json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def run_v114_frozen_gate(
    *, runner: Any, checkpoint_path: str, dataset_path: str, report_path: str,
    create_dataset: bool, burn_in_steps: int, expected_dataset_sha256: str | None,
) -> dict[str, Any]:
    alg = runner.alg
    before = _protected(alg, checkpoint_path)
    dataset_file = Path(dataset_path).resolve()
    if create_dataset:
        tracker = _VectorizedPhaseTracker(runner.env)
        runner.train_mode()
        obs = runner.env.get_observations().to(runner.device)
        runner.env.episode_length_buf = torch.randint_like(
            runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
        )
        phases = []
        with torch.inference_mode():
            for _ in range(burn_in_steps):
                tracker.sample()
                action = alg.policy.act(obs)
                obs, _, dones, _ = runner.env.step(action.to(runner.env.device))
                obs, dones = obs.to(runner.device), dones.to(runner.device)
                alg.policy.update_normalization(obs)
                alg.policy.reset(dones)
                tracker.reset(dones)
            for _ in range(runner.num_steps_per_env):
                phases.append(tracker.sample().cpu())
                action = alg.act(obs)
                obs, rewards, dones, extras = runner.env.step(action.to(runner.env.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                alg.process_env_step(obs, rewards, dones, extras)
                tracker.reset(dones)
        model = alg.policy
        stored = alg.storage.observations
        policy = stored[model.policy_keys[0]].flatten(0, 1)
        if policy.shape[1] >= model.policy_obs_dim + model.vae_latent_dim:
            policy = policy[:, :model.policy_obs_dim]
        estimator = stored[model.estimator_keys[0]].flatten(0, 1)
        critic = stored[model.critic_keys[0]].flatten(0, 1)
        phase_ids = torch.stack(phases).flatten().to(policy.device)
        chosen = []
        for phase in CRITICAL:
            indices = torch.nonzero(phase_ids == PHASES.index(phase), as_tuple=False).flatten()
            if not indices.numel():
                raise RuntimeError(f"frozen comparison buffer has no {phase} samples")
            chosen.append(indices)
        selected = torch.cat(chosen)
        data = {
            "policy": policy[selected].cpu(),
            "estimator": estimator[selected].cpu(),
            "critic": critic[selected].cpu(),
            "phase_name": phase_ids[selected].cpu(),
        }
        dataset_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, dataset_file)
    else:
        if not expected_dataset_sha256:
            raise RuntimeError("v1.14 gate requires the preregistered frozen dataset SHA256")
        actual_dataset_sha = _sha(dataset_file)
        if actual_dataset_sha != expected_dataset_sha256:
            raise RuntimeError(
                "v1.14 frozen comparison dataset SHA256 mismatch: "
                f"expected {expected_dataset_sha256}, got {actual_dataset_sha}"
            )
        data = torch.load(dataset_file, map_location="cpu", weights_only=True)

    dataset_sha = _sha(dataset_file)
    if expected_dataset_sha256 and dataset_sha != expected_dataset_sha256:
        raise RuntimeError("v1.14 frozen comparison dataset changed while loading")
    result = {
        "schema_version": 1,
        "kind": "highstep_v114_frozen_comparison",
        "checkpoint": os.path.realpath(checkpoint_path),
        "checkpoint_sha256": before["checkpoint"],
        "dataset": str(dataset_file),
        "dataset_sha256": dataset_sha,
        "dataset_created_in_this_run": bool(create_dataset),
        "statistics": _metrics(alg, data),
        "protected_before": before,
        "protected_after": _protected(alg, checkpoint_path),
        "optimizer_step_calls": 0,
        "runner_learn_calls": 0,
    }
    result["protected_unchanged"] = result["protected_before"] == result["protected_after"]
    if not result["protected_unchanged"]:
        raise RuntimeError("v1.14 frozen gate changed protected state")
    report = Path(report_path).resolve()
    report_sha = _write_json(report, result)
    print(json.dumps({"report": str(report), "sha256": report_sha, "dataset_sha256": dataset_sha}), flush=True)
    return result
