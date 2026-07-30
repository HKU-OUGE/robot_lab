#!/usr/bin/env python3
"""Train the frozen fixed-condition 570-D-observation + phase residual head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


WORKFLOW_ID = "highstep_fixed_condition_phase_residual_20260717"
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
HARD_CAP = torch.tensor([0.2] * 12 + [0.006] * 4, dtype=torch.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bound_dataset(manifest_path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = manifest_path.expanduser().resolve()
    actual = sha256_file(manifest_path)
    if actual != expected_sha256:
        raise RuntimeError(f"dataset manifest SHA mismatch: expected={expected_sha256} actual={actual}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    if kind == "highstep_fixed_condition_phase_residual_merged_teacher_dataset":
        valid_manifest = (
            manifest.get("status") == "frozen_teacher_dataset_ready"
            and int(manifest.get("rollout_count", -1)) == 15
            and int(manifest.get("train_rollouts", -1)) == 12
            and int(manifest.get("validation_rollouts", -1)) == 3
        )
    elif kind == "highstep_fixed_condition_phase_residual_dagger_aggregate":
        round_index = int(manifest.get("round", -1))
        valid_manifest = (
            round_index in (1, 2)
            and manifest.get("status") == f"frozen_dagger_round{round_index}_aggregate_ready"
            and int(manifest.get("post_termination_samples_used", -1)) == 0
            and int(manifest.get("train_rollouts", -1)) >= 12
            and int(manifest.get("validation_rollouts", -1)) >= 3
        )
    elif kind == "highstep_fixed_condition_phase_direct_action_dagger_aggregate":
        round_index = int(manifest.get("round", -1))
        valid_manifest = (
            manifest.get("workflow_id")
            == "highstep_fixed_condition_phase_direct_action_20260717"
            and round_index in (1, 2)
            and manifest.get("status")
            == f"frozen_direct_action_dagger_round{round_index}_aggregate_ready"
            and int(manifest.get("post_termination_samples_used", -1)) == 0
            and int(manifest.get("train_rollouts", -1)) >= 12
            and int(manifest.get("validation_rollouts", -1)) >= 3
        )
    else:
        valid_manifest = False
    if not valid_manifest:
        raise RuntimeError("dataset manifest is not a frozen Teacher/DAgger training contract")
    dataset_path = Path(manifest["dataset_path"]).expanduser().resolve()
    if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
        raise RuntimeError("teacher dataset SHA mismatch")
    archive = np.load(dataset_path, allow_pickle=False)
    arrays = {name: archive[name] for name in archive.files}
    samples = int(manifest.get("samples", -1))
    required_shapes = {
        "student_obs_570": (samples, 570),
        "phase_features_8": (samples, 8),
        "residual_target_16": (samples, 16),
        "stage": (samples,),
        "episode_id": (samples,),
        "step": (samples,),
        "split": (samples,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise RuntimeError(f"dataset array {name} shape changed: {arrays.get(name, np.empty(0)).shape}")
    train_episodes = set(arrays["episode_id"][arrays["split"] == 0].tolist())
    validation_episodes = set(arrays["episode_id"][arrays["split"] == 1].tolist())
    if len(train_episodes) != int(manifest["train_rollouts"]) or len(validation_episodes) != int(
        manifest["validation_rollouts"]
    ):
        raise RuntimeError("whole-rollout split count changed")
    if train_episodes & validation_episodes:
        raise RuntimeError("train/validation episode leakage detected")
    return manifest, arrays


def load_reference_binding(merged_manifest: dict[str, Any]) -> dict[str, str]:
    """Resolve the selected reference through every frozen source-batch manifest."""
    source_batches = merged_manifest.get("source_batches")
    if not isinstance(source_batches, list) or not source_batches:
        raise RuntimeError("merged dataset has no frozen source-batch manifests")
    bindings: list[dict[str, str]] = []
    for source in source_batches:
        manifest_path = Path(source["manifest_path"]).expanduser().resolve()
        actual_manifest_sha = sha256_file(manifest_path)
        if actual_manifest_sha != source.get("manifest_sha256"):
            raise RuntimeError(f"source dataset manifest SHA mismatch: {manifest_path}")
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        binding = {
            "selected_reference_manifest_path": str(
                Path(source_manifest["selected_reference_manifest_path"]).expanduser().resolve()
            ),
            "selected_reference_manifest_sha256": source_manifest[
                "selected_reference_manifest_sha256"
            ],
            "selected_reference_sha256": source_manifest["selected_reference_sha256"],
            "source_timing_amendment_sha256": source_manifest["metadata"][
                "source_timing_amendment_sha256"
            ],
        }
        reference_manifest_path = Path(binding["selected_reference_manifest_path"])
        if sha256_file(reference_manifest_path) != binding["selected_reference_manifest_sha256"]:
            raise RuntimeError("selected reference manifest SHA mismatch")
        bindings.append(binding)
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise RuntimeError("source datasets do not share one selected reference binding")
    if (
        bindings[0]["source_timing_amendment_sha256"]
        != merged_manifest.get("metadata", {}).get("source_timing_amendment_sha256")
    ):
        raise RuntimeError("merged dataset timing amendment binding changed")
    return bindings[0]


def compute_residual_bounds(target: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train = torch.abs(target[train_mask])
    if train.ndim != 2 or train.shape[1] != 16:
        raise RuntimeError("residual target shape changed")
    q99 = torch.quantile(train, 0.99, dim=0)
    return torch.minimum(1.25 * q99, HARD_CAP.to(q99)).clamp_min(1.0e-6)


class BoundedPhaseResidualMLP(nn.Module):
    def __init__(self, residual_bound: torch.Tensor) -> None:
        super().__init__()
        if tuple(residual_bound.shape) != (16,):
            raise ValueError("residual_bound must be 16-D")
        self.network = nn.Sequential(
            nn.Linear(578, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 16),
            nn.Tanh(),
        )
        self.register_buffer("residual_bound", residual_bound.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs) * self.residual_bound


def stage_balanced_indices(
    stages: torch.Tensor,
    train_mask: torch.Tensor,
    *,
    per_stage: int,
    generator: torch.Generator,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for stage in range(6):
        choices = torch.nonzero(train_mask & (stages == stage), as_tuple=False).flatten()
        if choices.numel() == 0:
            raise RuntimeError(f"training split has no samples for stage {stage}")
        draw = torch.randint(choices.numel(), (per_stage,), generator=generator)
        chunks.append(choices[draw])
    combined = torch.cat(chunks)
    order = torch.randperm(combined.numel(), generator=generator)
    return combined[order]


def temporal_pair_indices(
    episode_id: torch.Tensor, step: torch.Tensor, split: torch.Tensor, split_value: int
) -> tuple[torch.Tensor, torch.Tensor]:
    left: list[int] = []
    right: list[int] = []
    episodes = sorted(set(episode_id[split == split_value].tolist()))
    for episode in episodes:
        indices = torch.nonzero(
            (episode_id == int(episode)) & (split == split_value), as_tuple=False
        ).flatten()
        order = indices[torch.argsort(step[indices])]
        expected = torch.arange(order.numel())
        if order.numel() < 1 or not torch.equal(step[order].cpu(), expected):
            raise RuntimeError(f"episode {episode} is not one contiguous rollout prefix")
        if order.numel() == 1:
            continue
        left.extend(order[:-1].tolist())
        right.extend(order[1:].tolist())
    return torch.tensor(left, dtype=torch.long), torch.tensor(right, dtype=torch.long)


def _huber(prediction: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    return F.huber_loss(prediction, target, reduction="mean", delta=delta)


def evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pair_left: torch.Tensor,
    pair_right: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        residual_loss = _huber(prediction[mask], targets[mask])
        prediction_diff = prediction[pair_right] - prediction[pair_left]
        target_diff = targets[pair_right] - targets[pair_left]
        diff_loss = _huber(prediction_diff, target_diff)
        total = residual_loss + 0.1 * diff_loss
        absolute = torch.abs(prediction[mask] - targets[mask])
    return {
        "loss": float(total.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "first_difference_loss": float(diff_loss.detach().cpu()),
        "mae": float(torch.mean(absolute).detach().cpu()),
        "q95": float(torch.quantile(absolute, 0.95).detach().cpu()),
    }


def train_model(
    arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    seed: int = 20260717,
    max_epochs: int = 300,
    patience: int = 30,
    epoch_callback=None,
) -> tuple[BoundedPhaseResidualMLP, dict[str, Any], list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs = torch.from_numpy(arrays["student_obs_570"]).float()
    phase = torch.from_numpy(arrays["phase_features_8"]).float()
    inputs_cpu = torch.cat((obs, phase), dim=1)
    targets_cpu = torch.from_numpy(arrays["residual_target_16"]).float()
    stages = torch.from_numpy(arrays["stage"]).long()
    episode_id = torch.from_numpy(arrays["episode_id"]).long()
    step = torch.from_numpy(arrays["step"]).long()
    split = torch.from_numpy(arrays["split"]).long()
    train_mask_cpu = split == 0
    validation_mask_cpu = split == 1
    bounds = compute_residual_bounds(targets_cpu, train_mask_cpu)
    model = BoundedPhaseResidualMLP(bounds).to(device)
    inputs = inputs_cpu.to(device)
    targets = targets_cpu.to(device)
    train_mask = train_mask_cpu.to(device)
    validation_mask = validation_mask_cpu.to(device)
    train_left, train_right = temporal_pair_indices(episode_id, step, split, 0)
    val_left, val_right = temporal_pair_indices(episode_id, step, split, 1)
    train_left, train_right = train_left.to(device), train_right.to(device)
    val_left, val_right = val_left.to(device), val_right.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    best_loss = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        balanced_cpu = stage_balanced_indices(
            stages, train_mask_cpu, per_stage=682, generator=generator
        )
        balanced = balanced_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        residual_loss = _huber(prediction[balanced], targets[balanced])
        diff_loss = _huber(
            prediction[train_right] - prediction[train_left],
            targets[train_right] - targets[train_left],
        )
        loss = residual_loss + 0.1 * diff_loss
        loss.backward()
        optimizer.step()
        train_metrics = evaluate(
            model, inputs, targets, train_mask, train_left, train_right
        )
        validation_metrics = evaluate(
            model, inputs, targets, validation_mask, val_left, val_right
        )
        record = {
            "epoch": float(epoch),
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        if epoch_callback is not None:
            epoch_callback(epoch, record)
        validation_loss = validation_metrics["loss"]
        if validation_loss < best_loss - 1.0e-9:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("supervised optimization produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    result = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "residual_bound": [float(value) for value in bounds.tolist()],
        "train_metrics": evaluate(model, inputs, targets, train_mask, train_left, train_right),
        "validation_metrics": evaluate(
            model, inputs, targets, validation_mask, val_left, val_right
        ),
    }
    return model, result, history


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite training output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest, arrays = load_bound_dataset(args.dataset_manifest, args.dataset_manifest_sha256)
    reference_binding = load_reference_binding(manifest)
    device = torch.device(args.device)

    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"highstep_fixed_condition_phase_residual_{args.stage_name}",
        group=WORKFLOW_ID,
        job_type="supervised_phase_residual",
        mode="online",
        config={
            "workflow_id": WORKFLOW_ID,
            "route": "fixed_condition_phase_residual",
            "stage": args.stage_name,
            "attempt": args.attempt,
            "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_sha256": manifest["dataset_sha256"],
            **reference_binding,
            "input_dim": 578,
            "hidden_dims": [256, 256],
            "output_dim": 16,
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "batch_size": 4092,
            "stage_balanced": True,
            "max_epochs": 300,
            "patience": 30,
            "huber_delta": 0.05,
            "first_difference_loss_weight": 0.1,
            "ppo_enabled": False,
        },
    )
    try:
        model, result, history = train_model(
            arrays,
            device=device,
            epoch_callback=lambda epoch, record: run.log(record, step=epoch),
        )
        metrics_path = output_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for record in history:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        checkpoint_path = output_dir / "phase_residual_best.pt"
        torch.save(
            {
                "schema_version": 1,
                "kind": "bounded_phase_residual_mlp",
                "workflow_id": WORKFLOW_ID,
                "stage": args.stage_name,
                "attempt": args.attempt,
                "model_state_dict": model.state_dict(),
                "architecture": {"input_dim": 578, "hidden_dims": [256, 256], "output_dim": 16},
                "residual_bound": result["residual_bound"],
                "joint_names": JOINT_NAMES,
                "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
                "dataset_manifest_sha256": args.dataset_manifest_sha256,
                "dataset_sha256": manifest["dataset_sha256"],
                **reference_binding,
                "best_epoch": result["best_epoch"],
            },
            checkpoint_path,
        )
        checkpoint_sha = sha256_file(checkpoint_path)
        example = torch.zeros(1, 578, device=device)
        scripted = torch.jit.trace(model, example)
        script_path = output_dir / "phase_residual_best.jit.pt"
        scripted.save(str(script_path))
        artifact = wandb.Artifact(
            f"highstep-fixed-condition-phase-residual-{args.stage_name}",
            type="model",
            metadata={"checkpoint_sha256": checkpoint_sha},
        )
        artifact.add_file(str(checkpoint_path))
        artifact.add_file(str(script_path))
        run.log_artifact(artifact)
        training_manifest = {
            "schema_version": 1,
            "kind": "highstep_fixed_condition_phase_residual_training",
            "status": "training_complete_pending_wandb_remote_verification",
            "workflow_id": WORKFLOW_ID,
            "stage": args.stage_name,
            "attempt": args.attempt,
            "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_sha256": manifest["dataset_sha256"],
            **reference_binding,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "torchscript_path": str(script_path),
            "torchscript_sha256": sha256_file(script_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "wandb": {
                "entity": run.entity,
                "project": run.project,
                "run_id": run.id,
                "run_name": run.name,
                "group": WORKFLOW_ID,
                "url": run.url,
            },
            **result,
        }
        training_manifest_path = output_dir / "training_manifest.json"
        _atomic_json(training_manifest_path, training_manifest)
        training_manifest_sha = sha256_file(training_manifest_path)
        run.summary.update(
            {
                "output_checkpoint_sha256": checkpoint_sha,
                "effective_epochs": result["epochs_completed"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
                "training_manifest_path": str(training_manifest_path),
                "training_manifest_sha256": training_manifest_sha,
                "gate_conclusion": "training_complete_pending_behavior_gate",
            }
        )
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise

    remote_verified = False
    remote_error = None
    remote_path = f"{run.entity}/{run.project}/{run.id}"
    for _ in range(30):
        try:
            remote = wandb.Api(timeout=30).run(remote_path)
            remote_verified = (
                remote.summary.get("output_checkpoint_sha256") == checkpoint_sha
                and int(remote.summary.get("effective_epochs", -1)) == result["epochs_completed"]
            )
            if remote_verified:
                break
            remote_error = "remote summary is present but incomplete"
        except Exception as error:
            remote_error = repr(error)
        time.sleep(2.0)
    verification = {
        "schema_version": 1,
        "kind": "highstep_phase_residual_wandb_verification",
        "status": "verified" if remote_verified else "failed_closed",
        "run_path": remote_path,
        "run_url": run.url,
        "checkpoint_sha256": checkpoint_sha,
        "effective_epochs": result["epochs_completed"],
        "remote_verified": remote_verified,
        "error": remote_error,
    }
    verification_path = output_dir / "wandb_verification.json"
    _atomic_json(verification_path, verification)
    if not remote_verified:
        raise RuntimeError(f"W&B remote verification failed closed: {remote_error}")
    result_payload = {
        "training_manifest_path": str(training_manifest_path),
        "training_manifest_sha256": training_manifest_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "wandb_run_path": remote_path,
        "wandb_url": run.url,
        "wandb_verification_path": str(verification_path),
        "wandb_verification_sha256": sha256_file(verification_path),
        **result,
    }
    return result_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="xinqili551-the-university-of-hong-kong")
    parser.add_argument("--wandb-project", default="isaaclab")
    parser.add_argument("--stage-name", default="supervised_attempt1")
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_training(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
