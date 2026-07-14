"""Preregistered, read-only E700 fc_mu and observability isolation.

Student tensors and optimizer state are immutable.  All Student derivatives use
``torch.autograd.grad``.  The optional capacity probe owns independent modules
and an independent optimizer and is discarded after held-out evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from frozen_training_buffer_diagnostic import (
    CRITICAL_LATENT_DIMS,
    JOINT_NAMES,
    PHASES,
    REAR_FOCUS_JOINTS,
    _VectorizedPhaseTracker,
    _loss_terms,
    _module_sha256,
    _optimizer_sha256,
    _sha256_file,
)


LOSS_NAMES = ("action", "latent", "velocity", "recon", "kl")
REAR_PHASE_IDS = (3, 4, 5)


def _environment_split_codes(env_ids: torch.Tensor, split_seed: int) -> torch.Tensor:
    """Return deterministic 70/10/20 codes without leaking an env across splits.

    Hashing the decimal bucket directly with a linear congruential multiplier can
    collapse the modulo-10 output (the previous multiplier ended in 5).  Use a
    stable cryptographic digest per environment instead; 0..6=train,
    7=validation, and 8..9=test exactly preserve the preregistered semantics.
    """
    unique_envs = torch.unique(env_ids).detach().cpu().tolist()
    code_by_env = {
        int(env_id): int.from_bytes(
            hashlib.sha256(f"{split_seed}:{int(env_id)}".encode("ascii")).digest()[:8],
            "little",
        ) % 10
        for env_id in unique_envs
    }
    buckets = torch.tensor(
        [code_by_env[int(env_id)] for env_id in env_ids.detach().cpu().tolist()],
        dtype=torch.long,
        device=env_ids.device,
    )
    return torch.where(buckets <= 6, 0, torch.where(buckets == 7, 1, 2))


def _read_preregistration(path: str) -> tuple[dict[str, Any], str, str]:
    resolved = os.path.realpath(path)
    payload = json.loads(Path(resolved).read_text(encoding="utf-8"))
    actual_sha = _sha256_file(resolved)
    authority = payload["authority"]
    expected = {
        "workflow_id": "highstep_historical_0707_exact_new_teacher_20260714",
        "spec_sha256": "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917",
        "training_preregistration_sha256": "7b1bb49a9640e8614313423fce7e66fadc40e38bfb1ce551b78fb02dfbb9c035",
        "checkpoint_sha256": "8f02861037d2f24370cfc406bab32c8945c7ea09ec7f42cd5636c8423824ebff",
        "prior_diagnostic_sha256": "47fc72bd803fa44b51170e405dfc32f57db89ba91e6190af1986d99ee4738e5d",
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise RuntimeError(f"diagnostic preregistration authority mismatch at {key}")
    if payload.get("immutable") is not True or payload.get("schema_version") != 1:
        raise RuntimeError("diagnostic preregistration is not immutable schema 1")
    return payload, resolved, actual_sha


def _protected_state(alg: Any, checkpoint_path: str) -> dict[str, str]:
    return {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "actor_tensor_sha256": _module_sha256(alg.policy.actor),
        "estimator_tensor_sha256": _module_sha256(alg.policy.estimator),
        "optimizer_state_sha256": _optimizer_sha256(alg.vae_optimizer),
    }


def _zero_or_flatten(gradient: torch.Tensor | None, parameter: torch.Tensor) -> torch.Tensor:
    return (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)


def _group_row_gradients(
    alg: Any,
    policy_obs: torch.Tensor,
    est_input: torch.Tensor,
    critic_obs: torch.Tensor,
    indices: torch.Tensor,
    *,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    fc_mu = alg.policy.estimator.fc_mu
    parameters = (fc_mu.weight, fc_mu.bias)
    row_vectors = {
        dim: {loss: torch.zeros(fc_mu.in_features + 1, dtype=torch.float64) for loss in LOSS_NAMES}
        for dim in CRITICAL_LATENT_DIMS
    }
    losses = {loss: 0.0 for loss in LOSS_NAMES}
    count = int(indices.numel())
    if count == 0:
        return {"samples": 0, "losses": losses, "row_vectors": row_vectors}
    generator = torch.Generator(device=indices.device).manual_seed(seed)
    order = torch.randperm(count, generator=generator, device=indices.device)
    ordered = indices[order]
    torch.manual_seed(seed)
    alg.policy.estimator.train()
    for start in range(0, count, chunk_size):
        selected = ordered[start:start + chunk_size]
        fraction = selected.numel() / count
        terms, _ = _loss_terms(
            alg,
            policy_obs[selected],
            est_input[selected],
            critic_obs[selected],
        )
        for loss_index, loss_name in enumerate(LOSS_NAMES):
            gradients = torch.autograd.grad(
                terms[loss_name],
                parameters,
                retain_graph=loss_index < len(LOSS_NAMES) - 1,
                allow_unused=True,
            )
            weight_gradient = (
                torch.zeros_like(fc_mu.weight) if gradients[0] is None else gradients[0]
            )
            bias_gradient = (
                torch.zeros_like(fc_mu.bias) if gradients[1] is None else gradients[1]
            )
            for dim in CRITICAL_LATENT_DIMS:
                vector = torch.cat((weight_gradient[dim], bias_gradient[dim:dim + 1]))
                row_vectors[dim][loss_name] += vector.detach().cpu().double() * fraction
            losses[loss_name] += float(terms[loss_name].detach().item()) * fraction
    return {"samples": count, "losses": losses, "row_vectors": row_vectors}


def _summarize_row_vectors(
    vectors: dict[str, torch.Tensor], coefficients: dict[str, float]
) -> dict[str, Any]:
    weighted = {name: vector * coefficients[name] for name, vector in vectors.items()}
    raw_norms = {name: float(torch.linalg.vector_norm(vector).item()) for name, vector in vectors.items()}
    weighted_norms = {
        name: float(torch.linalg.vector_norm(vector).item()) for name, vector in weighted.items()
    }
    action = weighted["action"]
    action_norm = torch.linalg.vector_norm(action)
    cosines: dict[str, float | None] = {}
    projections: dict[str, float | None] = {}
    for left_index, left in enumerate(LOSS_NAMES):
        for right in LOSS_NAMES[left_index + 1:]:
            denominator = torch.linalg.vector_norm(weighted[left]) * torch.linalg.vector_norm(weighted[right])
            cosines[f"{left}__{right}"] = (
                None if float(denominator.item()) == 0.0
                else float(torch.dot(weighted[left], weighted[right]).div(denominator).item())
            )
    for name in LOSS_NAMES:
        projections[name] = (
            None if float(action_norm.item()) == 0.0
            else float(torch.dot(weighted[name], action).div(action_norm).item())
        )
    vector_sum = sum(weighted.values(), torch.zeros_like(action))
    norm_sum = sum(torch.linalg.vector_norm(value) for value in weighted.values())
    cancellation_ratio = (
        None if float(norm_sum.item()) == 0.0
        else float(torch.linalg.vector_norm(vector_sum).div(norm_sum).item())
    )
    return {
        "raw_gradient_norms": raw_norms,
        "weighted_gradient_norms": weighted_norms,
        "pairwise_weighted_cosines": cosines,
        "weighted_projection_on_action_direction": projections,
        "weighted_total_cancellation_ratio": cancellation_ratio,
    }


def _combine_group_vectors(
    groups: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    for dim in CRITICAL_LATENT_DIMS:
        result[dim] = {}
        for loss in LOSS_NAMES:
            template = next(iter(groups.values()))["row_vectors"][dim][loss]
            combined = torch.zeros_like(template)
            for name, group in groups.items():
                combined += group["row_vectors"][dim][loss] * weights[name]
            result[dim][loss] = combined
    return result


def _actor_jacobian(
    alg: Any,
    policy_obs: torch.Tensor,
    est_input: torch.Tensor,
    indices: torch.Tensor,
    *,
    sample_cap: int,
    seed: int,
) -> dict[str, Any]:
    if indices.numel() == 0:
        return {"samples": 0}
    generator = torch.Generator(device=indices.device).manual_seed(seed)
    if indices.numel() > sample_cap:
        chosen = indices[torch.randperm(indices.numel(), generator=generator, device=indices.device)[:sample_cap]]
    else:
        chosen = indices
    with torch.no_grad():
        mu, _, _ = alg.policy.estimator.encode(est_input[chosen])
    mu = mu.detach().requires_grad_(True)
    actions = alg.policy.actor(torch.cat((policy_obs[chosen], torch.clamp(mu, -1.0, 1.0)), dim=1))
    by_dim: dict[str, Any] = {str(dim): {} for dim in CRITICAL_LATENT_DIMS}
    for joint in range(12):
        gradient = torch.autograd.grad(
            actions[:, joint].sum(), mu, retain_graph=joint < 11, allow_unused=False
        )[0]
        for dim in CRITICAL_LATENT_DIMS:
            values = gradient[:, dim].detach().float()
            by_dim[str(dim)][JOINT_NAMES[joint]] = {
                "signed_mean": float(values.mean().item()),
                "mean_abs": float(values.abs().mean().item()),
                "q95_abs": float(torch.quantile(values.abs(), 0.95).item()),
                "rear_thigh_or_calf": joint in REAR_FOCUS_JOINTS,
            }
    return {"samples": int(chosen.numel()), "critical_dim_to_nonbox_action_jacobian": by_dim}


def _prediction_metrics(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, Any]:
    target = target.detach().float().cpu()
    prediction = prediction.detach().float().cpu()
    if target.shape[0] == 0:
        return {
            "samples": 0,
            "dimensions": {
                str(dim): {"r2": None, "mae": None} for dim in CRITICAL_LATENT_DIMS
            },
            "macro_r2": None,
            "macro_mae": None,
            "dims_r2_at_least_0_25": 0,
            "dims_r2_nonpositive": 0,
        }
    error = prediction - target
    mean_target = target.mean(dim=0, keepdim=True)
    denominator = torch.sum(torch.square(target - mean_target), dim=0)
    r2 = 1.0 - torch.sum(torch.square(error), dim=0) / torch.clamp(denominator, min=1.0e-12)
    mae = torch.mean(torch.abs(error), dim=0)
    return {
        "samples": int(target.shape[0]),
        "dimensions": {
            str(dim): {"r2": float(r2[index].item()), "mae": float(mae[index].item())}
            for index, dim in enumerate(CRITICAL_LATENT_DIMS)
        },
        "macro_r2": float(r2.mean().item()),
        "macro_mae": float(mae.mean().item()),
        "dims_r2_at_least_0_25": int(torch.sum(r2 >= 0.25).item()),
        "dims_r2_nonpositive": int(torch.sum(r2 <= 0.0).item()),
    }


def _predict_critical(module: Any, inputs: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    if inputs.shape[0] == 0:
        return torch.empty(
            (0, len(CRITICAL_LATENT_DIMS)), device=inputs.device, dtype=inputs.dtype
        )
    predictions = []
    module.eval()
    with torch.no_grad():
        for start in range(0, inputs.shape[0], chunk_size):
            mu, _, _ = module.encode(inputs[start:start + chunk_size])
            predictions.append(mu[:, CRITICAL_LATENT_DIMS])
    return torch.cat(predictions)


def _capacity_probe(
    alg: Any,
    est_input: torch.Tensor,
    target: torch.Tensor,
    phase_ids: torch.Tensor,
    transition_mask: torch.Tensor,
    env_ids: torch.Tensor,
    split_codes: torch.Tensor,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    config = prereg["capacity_probe"]
    torch.manual_seed(int(config["seed"]))
    probe = type(alg.policy.estimator)(
        input_dim=int(est_input.shape[1]),
        vel_dim=3,
        latent_dim=int(alg.policy.vae_latent_dim),
        hidden_dims=list(config["hidden_dims"]),
    ).to(est_input.device)
    student_ptrs = {
        value.untyped_storage().data_ptr()
        for value in list(alg.policy.estimator.parameters()) + list(alg.policy.estimator.buffers())
        if value.numel()
    }
    probe_ptrs = {
        value.untyped_storage().data_ptr()
        for value in list(probe.parameters()) + list(probe.buffers())
        if value.numel()
    }
    if student_ptrs & probe_ptrs:
        raise RuntimeError("capacity probe shares storage with Student estimator")
    probe_parameters = tuple(probe.encoder.parameters()) + tuple(probe.fc_mu.parameters())
    optimizer = torch.optim.Adam(probe_parameters, lr=float(config["learning_rate"]))
    train_mask = split_codes == 0
    validation_mask = split_codes == 1
    test_mask = split_codes == 2
    split_sizes = {
        "train": int(train_mask.sum().item()),
        "validation": int(validation_mask.sum().item()),
        "test": int(test_mask.sum().item()),
    }
    if any(size == 0 for size in split_sizes.values()):
        raise RuntimeError(f"capacity probe env split contains an empty partition: {split_sizes}")
    if not torch.isfinite(est_input).all() or not torch.isfinite(target).all():
        raise RuntimeError("capacity probe received non-finite observation history or target")
    train_envs = set(env_ids[train_mask].detach().cpu().tolist())
    validation_envs = set(env_ids[validation_mask].detach().cpu().tolist())
    test_envs = set(env_ids[test_mask].detach().cpu().tolist())
    if train_envs & validation_envs or train_envs & test_envs or validation_envs & test_envs:
        raise RuntimeError("capacity probe env split leaked across train/validation/test")
    train_by_phase = [torch.nonzero(train_mask & (phase_ids == phase), as_tuple=False).reshape(-1) for phase in range(6)]
    if any(indices.numel() == 0 for indices in train_by_phase):
        raise RuntimeError("capacity probe phase-balanced training split has an empty phase")
    generator = torch.Generator(device=est_input.device).manual_seed(int(config["seed"]))
    samples_per_phase = int(config["samples_per_phase_per_epoch"])
    batch_size = int(config["batch_size"])
    best_validation = float("inf")
    best_epoch = -1
    best_state = None
    optimizer_steps = 0
    for epoch in range(int(config["epochs"])):
        epoch_indices = []
        for available in train_by_phase:
            choice = torch.randint(
                available.numel(), (samples_per_phase,), generator=generator, device=available.device
            )
            epoch_indices.append(available[choice])
        epoch_indices = torch.cat(epoch_indices)
        order = torch.randperm(epoch_indices.numel(), generator=generator, device=epoch_indices.device)
        epoch_indices = epoch_indices[order]
        probe.train()
        for start in range(0, epoch_indices.numel(), batch_size):
            selected = epoch_indices[start:start + batch_size]
            mu, _, _ = probe.encode(est_input[selected])
            loss = torch.mean(torch.square(mu[:, CRITICAL_LATENT_DIMS] - target[selected]))
            optimizer.zero_grad(set_to_none=True)
            gradients = torch.autograd.grad(loss, probe_parameters, allow_unused=False)
            for parameter, gradient in zip(probe_parameters, gradients, strict=True):
                parameter.grad = gradient
            optimizer.step()
            optimizer_steps += 1
        validation_indices = torch.nonzero(validation_mask, as_tuple=False).reshape(-1)
        validation_prediction = _predict_critical(probe, est_input[validation_indices])
        validation_loss = float(torch.mean(torch.square(
            validation_prediction - target[validation_indices]
        )).item())
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in probe.state_dict().items()})
    if best_state is None:
        raise RuntimeError("capacity probe did not produce a validation-selected state")
    probe.load_state_dict(best_state, strict=True)
    test_indices = torch.nonzero(test_mask, as_tuple=False).reshape(-1)
    student_prediction = _predict_critical(alg.policy.estimator, est_input[test_indices])
    probe_prediction = _predict_critical(probe, est_input[test_indices])
    result: dict[str, Any] = {
        "student_storage_independent": True,
        "student_optimizer_used": False,
        "probe_optimizer_steps": optimizer_steps,
        "selection": {"best_validation_epoch_zero_based": best_epoch, "best_validation_mse": best_validation},
        "test_all": {
            "student": _prediction_metrics(target[test_indices], student_prediction),
            "probe": _prediction_metrics(target[test_indices], probe_prediction),
        },
        "test_transition_windows": {},
        "split": {
            "train_envs": len(train_envs),
            "validation_envs": len(validation_envs),
            "test_envs": len(test_envs),
            "env_overlap": False,
        },
    }
    for phase_id in REAR_PHASE_IDS:
        local = test_mask & transition_mask & (phase_ids == phase_id)
        selected = torch.nonzero(local, as_tuple=False).reshape(-1)
        result["test_transition_windows"][PHASES[phase_id]] = {
            "student": _prediction_metrics(target[selected], _predict_critical(alg.policy.estimator, est_input[selected])),
            "probe": _prediction_metrics(target[selected], _predict_critical(probe, est_input[selected])),
        }
    combined = test_mask & transition_mask & (phase_ids >= 3)
    selected = torch.nonzero(combined, as_tuple=False).reshape(-1)
    result["test_rear_transition_combined"] = {
        "student": _prediction_metrics(target[selected], _predict_critical(alg.policy.estimator, est_input[selected])),
        "probe": _prediction_metrics(target[selected], _predict_critical(probe, est_input[selected])),
    }
    del optimizer, probe
    return result


def _nearest_neighbor_ambiguity(
    est_input: torch.Tensor,
    target: torch.Tensor,
    phase_ids: torch.Tensor,
    transition_mask: torch.Tensor,
    split_codes: torch.Tensor,
    prereg: dict[str, Any],
) -> dict[str, Any]:
    cfg = prereg["nearest_neighbor"]
    train = split_codes == 0
    test = split_codes == 2
    train_input = est_input[train]
    mean = train_input.mean(dim=0)
    std = train_input.std(dim=0).clamp(min=1.0e-4)
    target_std = target[train].std(dim=0).clamp(min=1.0e-4)
    generator = torch.Generator(device=est_input.device).manual_seed(int(cfg["seed"]))
    result: dict[str, Any] = {}
    for phase_id in REAR_PHASE_IDS:
        reference = torch.nonzero(train & transition_mask & (phase_ids == phase_id), as_tuple=False).reshape(-1)
        query = torch.nonzero(test & transition_mask & (phase_ids == phase_id), as_tuple=False).reshape(-1)
        if reference.numel() > int(cfg["reference_cap"]):
            order = torch.randperm(reference.numel(), generator=generator, device=reference.device)
            reference = reference[order[:int(cfg["reference_cap"])]]
        if query.numel() > int(cfg["query_cap"]):
            order = torch.randperm(query.numel(), generator=generator, device=query.device)
            query = query[order[:int(cfg["query_cap"])]]
        if reference.numel() == 0 or query.numel() == 0:
            result[PHASES[phase_id]] = {
                "reference_samples": int(reference.numel()),
                "query_samples": int(query.numel()),
                "near_pair_count": 0,
                "near_input_rms_max": None,
                "near_input_rms_mean": None,
                "near_normalized_target_rms_mean": None,
                "ambiguous_pair_fraction": None,
            }
            continue
        ref_x = (est_input[reference] - mean) / std
        query_x = (est_input[query] - mean) / std
        ref_norm = torch.sum(ref_x.square(), dim=1)
        nearest_distances = []
        nearest_targets = []
        for start in range(0, query.numel(), int(cfg["query_chunk_size"])):
            chunk = query_x[start:start + int(cfg["query_chunk_size"])]
            distance = torch.sum(chunk.square(), dim=1, keepdim=True) + ref_norm.unsqueeze(0) - 2.0 * chunk @ ref_x.T
            minimum, indices = torch.min(torch.clamp(distance, min=0.0), dim=1)
            nearest_distances.append(torch.sqrt(minimum / est_input.shape[1]))
            nearest_targets.append(target[reference[indices]])
        distances = torch.cat(nearest_distances)
        neighbor_target = torch.cat(nearest_targets)
        target_difference = torch.sqrt(torch.mean(torch.square(
            (target[query] - neighbor_target) / target_std
        ), dim=1))
        near_count = max(int(cfg["minimum_near_pairs"]), math.ceil(query.numel() * float(cfg["near_quantile"])))
        near_count = min(near_count, query.numel())
        near_order = torch.argsort(distances)[:near_count]
        ambiguity = target_difference[near_order] >= float(cfg["normalized_target_difference_threshold"])
        result[PHASES[phase_id]] = {
            "reference_samples": int(reference.numel()),
            "query_samples": int(query.numel()),
            "near_pair_count": int(near_count),
            "near_input_rms_max": float(distances[near_order].max().item()),
            "near_input_rms_mean": float(distances[near_order].mean().item()),
            "near_normalized_target_rms_mean": float(target_difference[near_order].mean().item()),
            "ambiguous_pair_fraction": float(ambiguity.float().mean().item()),
        }
    return result


def _write_immutable_json(path: str, payload: dict[str, Any]) -> tuple[str, str]:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    sha = _sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{sha}  {output.name}\n", encoding="utf-8")
    output.chmod(0o444)
    sidecar.chmod(0o444)
    return str(output), sha


def run_frozen_fc_mu_observability_diagnostic(
    *, runner: Any, checkpoint_path: str, preregistration_path: str, output_path: str, burn_in_steps: int
) -> dict[str, Any]:
    prereg, prereg_path, prereg_sha = _read_preregistration(preregistration_path)
    if burn_in_steps != int(prereg["collection"]["burn_in_steps"]):
        raise RuntimeError("diagnostic burn-in differs from preregistration")
    alg = runner.alg
    if str(getattr(alg, "student_recovery_stage", "")).upper() != "HISTORICAL_0707_EXACT":
        raise RuntimeError("fc_mu diagnosis requires HISTORICAL_0707_EXACT")
    if int(getattr(alg, "student_distill_update_count", -1)) != 700:
        raise RuntimeError("fc_mu diagnosis requires full E700 effective update state")
    if int(alg.storage.step) != 0:
        raise RuntimeError("rollout storage is not empty")

    unique_optimizers = {id(value): value for value in (alg.optimizer, alg.vae_optimizer)}
    original_steps = {identity: optimizer.step for identity, optimizer in unique_optimizers.items()}
    for optimizer in unique_optimizers.values():
        def forbidden_student_step(*args: Any, **kwargs: Any):
            raise RuntimeError("Student optimizer.step is forbidden in fc_mu diagnostic")
        optimizer.step = forbidden_student_step
    before = _protected_state(alg, checkpoint_path)
    tracker = _VectorizedPhaseTracker(runner.env)
    phase_steps = []
    age_steps = []
    entered_steps = []
    episode_steps = []
    episode_ids = torch.zeros(runner.env.num_envs, dtype=torch.long, device=runner.device)
    runner.train_mode()
    obs = runner.env.get_observations().to(runner.device)
    runner.env.episode_length_buf = torch.randint_like(
        runner.env.episode_length_buf, high=int(runner.env.max_episode_length)
    )
    try:
        with torch.inference_mode():
            for _ in range(burn_in_steps):
                tracker.sample()
                actions = alg.policy.act(obs)
                obs, _, dones, _ = runner.env.step(actions.to(runner.env.device))
                obs, dones = obs.to(runner.device), dones.to(runner.device)
                alg.policy.update_normalization(obs)
                alg.policy.reset(dones)
                episode_ids += dones.reshape(-1).long()
                tracker.reset(dones)
            for _ in range(runner.num_steps_per_env):
                phase_steps.append(tracker.sample().cpu())
                age_steps.append(tracker.stage_age.clone().cpu())
                entered_steps.append(tracker.last_stage_entered.clone().cpu())
                episode_steps.append(episode_ids.clone().cpu())
                actions = alg.act(obs)
                obs, rewards, dones, extras = runner.env.step(actions.to(runner.env.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                alg.process_env_step(obs, rewards, dones, extras)
                episode_ids += dones.reshape(-1).long()
                tracker.reset(dones)
        if int(alg.storage.step) != int(runner.num_steps_per_env):
            raise RuntimeError("diagnostic did not collect exactly one training buffer")
        phase_ids = torch.stack(phase_steps).flatten().to(runner.device)
        stage_age = torch.stack(age_steps).flatten().to(runner.device)
        stage_entered = torch.stack(entered_steps).flatten().to(runner.device)
        episode_flat = torch.stack(episode_steps).flatten().to(runner.device)
        env_flat = torch.arange(runner.env.num_envs, device=runner.device).repeat(runner.num_steps_per_env)
        window_steps = int(prereg["transition_windows"]["steps_after_entry"])
        transition_mask = (phase_ids >= 3) & (stage_age < window_steps)
        split_codes = _environment_split_codes(
            env_flat, int(prereg["capacity_probe"]["split_seed"])
        )

        model = alg.policy
        storage = alg.storage.observations
        policy_obs = storage[model.policy_keys[0]].flatten(0, 1)
        est_input = storage[model.estimator_keys[0]].flatten(0, 1)
        critic_obs = storage[model.critic_keys[0]].flatten(0, 1)
        if policy_obs.shape[1] >= model.policy_obs_dim + model.vae_latent_dim:
            policy_obs = policy_obs[:, :model.policy_obs_dim]
        target_parts = []
        with torch.no_grad():
            for start in range(0, critic_obs.shape[0], 8192):
                target_parts.append(alg.teacher_priv_encoder(critic_obs[start:start + 8192, 3:])[:, CRITICAL_LATENT_DIMS])
        critical_target = torch.cat(target_parts)

        coefficients = prereg["gradient"]["loss_coefficients"]
        phase_groups: dict[str, dict[str, Any]] = {}
        for phase_id, phase_name in enumerate(PHASES):
            indices = torch.nonzero(phase_ids == phase_id, as_tuple=False).reshape(-1)
            phase_groups[phase_name] = _group_row_gradients(
                alg, policy_obs, est_input, critic_obs, indices,
                seed=int(prereg["gradient"]["seed"]) + phase_id,
                chunk_size=int(prereg["gradient"]["chunk_size"]),
            )
        total_samples = phase_ids.numel()
        real_weights = {name: group["samples"] / total_samples for name, group in phase_groups.items()}
        nonempty = [name for name, group in phase_groups.items() if group["samples"]]
        balanced_weights = {name: (1.0 / len(nonempty) if name in nonempty else 0.0) for name in phase_groups}
        real_vectors = _combine_group_vectors(phase_groups, real_weights)
        balanced_vectors = _combine_group_vectors(phase_groups, balanced_weights)
        gradient_result: dict[str, Any] = {
            "loss_coefficients": coefficients,
            "real_training_distribution": {
                "weights": real_weights,
                "rows": {str(dim): _summarize_row_vectors(real_vectors[dim], coefficients) for dim in CRITICAL_LATENT_DIMS},
            },
            "phase_balanced_distribution": {
                "weights": balanced_weights,
                "rows": {str(dim): _summarize_row_vectors(balanced_vectors[dim], coefficients) for dim in CRITICAL_LATENT_DIMS},
            },
            "by_phase": {},
            "transition_windows": {},
        }
        for phase_name, group in phase_groups.items():
            gradient_result["by_phase"][phase_name] = {
                "samples": group["samples"],
                "losses": group["losses"],
                "rows": {
                    str(dim): _summarize_row_vectors(group["row_vectors"][dim], coefficients)
                    for dim in CRITICAL_LATENT_DIMS
                },
            }
        transition_counts = {}
        for phase_id in REAR_PHASE_IDS:
            phase_name = PHASES[phase_id]
            local = transition_mask & (phase_ids == phase_id)
            indices = torch.nonzero(local, as_tuple=False).reshape(-1)
            group = _group_row_gradients(
                alg, policy_obs, est_input, critic_obs, indices,
                seed=int(prereg["gradient"]["seed"]) + 100 + phase_id,
                chunk_size=int(prereg["gradient"]["chunk_size"]),
            )
            env_count = int(torch.unique(env_flat[local]).numel())
            episode_count = int(torch.unique(env_flat[local] * 100000 + episode_flat[local]).numel())
            test_local = local & (split_codes == 2)
            transition_counts[phase_name] = {
                "samples": int(indices.numel()),
                "envs": env_count,
                "episodes": episode_count,
                "test_samples": int(test_local.sum().item()),
                "test_envs": int(torch.unique(env_flat[test_local]).numel()),
            }
            gradient_result["transition_windows"][phase_name] = {
                **transition_counts[phase_name],
                "stage_age_range": [0, window_steps - 1],
                "rows": {
                    str(dim): _summarize_row_vectors(group["row_vectors"][dim], coefficients)
                    for dim in CRITICAL_LATENT_DIMS
                },
                "nonbox_action_jacobian": _actor_jacobian(
                    alg, policy_obs, est_input, indices,
                    sample_cap=int(prereg["jacobian"]["sample_cap_per_transition_phase"]),
                    seed=int(prereg["jacobian"]["seed"]) + phase_id,
                ),
            }

        probe_result = _capacity_probe(
            alg, est_input, critical_target, phase_ids, transition_mask, env_flat, split_codes, prereg
        )
        ambiguity_result = _nearest_neighbor_ambiguity(
            est_input, critical_target, phase_ids, transition_mask, split_codes, prereg
        )

        criteria = prereg["decision"]
        scarcity = any(
            row["samples"] < int(criteria["transition_min_samples_per_phase"])
            or row["envs"] < int(criteria["transition_min_envs_per_phase"])
            or row["test_samples"] < int(criteria["transition_min_test_samples_per_phase"])
            or row["test_envs"] < int(criteria["transition_min_test_envs_per_phase"])
            for row in transition_counts.values()
        )
        combined_probe = probe_result["test_rear_transition_combined"]["probe"]
        combined_student = probe_result["test_rear_transition_combined"]["student"]
        probe_pass = (
            combined_probe["macro_r2"] is not None
            and combined_probe["macro_r2"] >= float(criteria["probe_pass_macro_r2"])
            and combined_probe["dims_r2_at_least_0_25"] >= int(criteria["probe_pass_dims_r2_at_least_0_25"])
        )
        probe_fail = (
            combined_probe["macro_r2"] is not None
            and (combined_probe["macro_r2"] <= float(criteria["probe_fail_macro_r2"])
            or combined_probe["dims_r2_nonpositive"] >= int(criteria["probe_fail_dims_nonpositive"])
            )
        )
        ambiguity_support = sum(
            row["ambiguous_pair_fraction"] is not None
            and row["ambiguous_pair_fraction"] >= float(criteria["ambiguity_fraction_threshold"])
            for row in ambiguity_result.values()
        ) >= int(criteria["ambiguity_phases_required"])
        probe_improvement = (
            combined_probe["macro_mae"] is not None
            and combined_student["macro_mae"] is not None
            and (
            combined_probe["macro_mae"]
            <= float(criteria["probe_student_mae_ratio_max"]) * combined_student["macro_mae"]
            or combined_probe["macro_r2"] - combined_student["macro_r2"]
            >= float(criteria["probe_student_macro_r2_improvement_min"])
            )
        )
        conflict_count = 0
        conflict_rows = []
        for phase_name, phase_result in gradient_result["transition_windows"].items():
            for dim, row in phase_result["rows"].items():
                action_latent = row["pairwise_weighted_cosines"].get("action__latent")
                cancellation = row["weighted_total_cancellation_ratio"]
                strong = (
                    action_latent is not None
                    and action_latent <= float(criteria["strong_action_latent_conflict_cosine_max"])
                ) or (
                    cancellation is not None
                    and cancellation <= float(criteria["strong_cancellation_ratio_max"])
                )
                if strong:
                    conflict_count += 1
                    conflict_rows.append({"phase": phase_name, "dim": int(dim)})
        if scarcity:
            conclusion = "critical_transition_sample_scarcity"
            first_missing = next(
                name for name, row in transition_counts.items()
                if row["samples"] < int(criteria["transition_min_samples_per_phase"])
                or row["envs"] < int(criteria["transition_min_envs_per_phase"])
                or row["test_samples"] < int(criteria["transition_min_test_samples_per_phase"])
                or row["test_envs"] < int(criteria["transition_min_test_envs_per_phase"])
            )
            evidence = {"first_insufficient_transition_phase": first_missing}
        elif probe_fail and ambiguity_support:
            conclusion = "estimator_observation_unobservable_for_critical_latents"
            evidence = {"probe_failed": True, "nearest_neighbor_ambiguity_supported": True}
        elif probe_pass and (probe_improvement or conflict_count >= int(criteria["strong_conflict_rows_required"])):
            conclusion = "fc_mu_row_optimization_or_gradient_cancellation"
            evidence = {
                "probe_passed": True,
                "probe_improved_over_student": probe_improvement,
                "strong_conflict_row_count": conflict_count,
                "strong_conflict_rows": conflict_rows,
            }
        else:
            conclusion = "diagnosis_not_closed"
            if not probe_pass and not probe_fail:
                first_missing = "capacity_probe_result_falls_in_preregistered_gray_zone"
            elif probe_fail and not ambiguity_support:
                first_missing = "nearest_neighbor_ambiguity_does_not_support_probe_failure"
            elif probe_pass and not probe_improvement and conflict_count < int(criteria["strong_conflict_rows_required"]):
                first_missing = "probe_predicts_but_current_fc_mu_failure_or_gradient_cancellation_not_proven"
            else:
                first_missing = "A_B_C_joint_evidence_not_satisfied"
            evidence = {"first_missing_evidence": first_missing}
    finally:
        for identity, optimizer in unique_optimizers.items():
            optimizer.step = original_steps[identity]

    after = _protected_state(alg, checkpoint_path)
    unchanged = {key: before[key] == after[key] for key in before}
    if not all(unchanged.values()):
        raise RuntimeError(f"protected Student state changed: {unchanged}")
    manifest = {
        "schema_version": 1,
        "kind": "E700_fc_mu_row_and_observability_diagnostic",
        "created_at": datetime.now().astimezone().isoformat(),
        "authority": prereg["authority"],
        "diagnostic_preregistration": prereg_path,
        "diagnostic_preregistration_sha256": prereg_sha,
        "checkpoint": os.path.realpath(checkpoint_path),
        "collection": {
            "burn_in_steps": burn_in_steps,
            "captured_training_buffers": 1,
            "num_envs": int(runner.env.num_envs),
            "steps_per_buffer": int(runner.num_steps_per_env),
            "samples": int(phase_ids.numel()),
            "runner_learn_calls": 0,
            "student_backward_calls": 0,
            "student_optimizer_step_calls": 0,
        },
        "phase_distribution": {
            phase: int((phase_ids == phase_id).sum().item())
            for phase_id, phase in enumerate(PHASES)
        },
        "transition_window_counts": transition_counts,
        "gradient_analysis": gradient_result,
        "capacity_probe": probe_result,
        "nearest_neighbor_ambiguity": ambiguity_result,
        "decision_inputs": {
            "transition_scarcity": scarcity,
            "probe_pass": probe_pass,
            "probe_fail": probe_fail,
            "ambiguity_support": ambiguity_support,
            "probe_improvement": probe_improvement,
            "strong_conflict_row_count": conflict_count,
        },
        "conclusion": conclusion,
        "conclusion_evidence": evidence,
        "protected_state_before": before,
        "protected_state_after": after,
        "protected_state_unchanged": unchanged,
        "all_protected_state_unchanged": True,
        "training_stage_started": False,
        "E900_E1400_started": False,
        "training_semantics_changed": False,
    }
    resolved_output, manifest_sha = _write_immutable_json(output_path, manifest)
    print(json.dumps({
        "manifest": resolved_output,
        "manifest_sha256": manifest_sha,
        "conclusion": conclusion,
        "transition_window_counts": transition_counts,
        "protected_state_unchanged": True,
    }, sort_keys=True), flush=True)
    return manifest
