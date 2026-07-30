#!/usr/bin/env python3
"""Train phase-indexed direct safe Teacher targets for the fixed-condition branch."""

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
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from highstep_phase_residual_train import (
    JOINT_NAMES,
    load_bound_dataset,
    load_reference_binding,
    stage_balanced_indices,
    temporal_pair_indices,
)


WORKFLOW_ID = "highstep_fixed_condition_phase_direct_action_20260717"
LOWER = torch.tensor(
    [-1.2217304764] * 4 + [-1.5708] * 4 + [-2.7750735107] * 4 + [0.0] * 4,
    dtype=torch.float32,
)
UPPER = torch.tensor(
    [1.2217304764] * 4 + [3.4907] * 4 + [-0.6457718232] * 4 + [0.06] * 4,
    dtype=torch.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def _load_preregistration(path: Path, expected_sha256: str, dataset_sha256: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("direct-action preregistration SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    round_index = int(payload.get("round", 0))
    initial_contract = (
        payload.get("kind")
        == "highstep_fixed_condition_phase_direct_action_preregistration"
        and payload.get("status") == "frozen_before_training"
        and round_index == 0
    )
    dagger_contract = (
        payload.get("kind")
        == "highstep_fixed_condition_phase_direct_action_dagger_training_preregistration"
        and round_index in (1, 2)
        and payload.get("status")
        == f"frozen_before_dagger_round{round_index}_training"
    )
    if (
        not (initial_contract or dagger_contract)
        or payload.get("workflow_id") != WORKFLOW_ID
        or payload.get("dataset_sha256") != dataset_sha256
        or payload.get("output_contract", {}).get("target")
        != "safe_teacher_post_prior_mapped_target_16"
        or payload.get("model_contract", {}).get("hidden_dims") != [512, 512, 256]
    ):
        raise RuntimeError("direct-action preregistration contract changed")
    spec_path = Path(payload["spec_path"]).expanduser().resolve()
    if sha256_file(spec_path) != payload.get("spec_sha256"):
        raise RuntimeError("direct-action spec SHA mismatch")
    teacher_path = Path(payload["teacher_checkpoint"]).expanduser().resolve()
    if sha256_file(teacher_path) != payload.get("teacher_sha256"):
        raise RuntimeError("direct-action Teacher SHA mismatch")
    return payload


class PhaseDirectActionMLP(nn.Module):
    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor) -> None:
        super().__init__()
        if tuple(input_mean.shape) != (578,) or tuple(input_std.shape) != (578,):
            raise ValueError("direct-action input normalization must be 578-D")
        self.register_buffer("input_mean", input_mean.detach().clone())
        self.register_buffer("input_std", input_std.detach().clone())
        self.register_buffer("target_center", ((LOWER + UPPER) * 0.5).clone())
        self.register_buffer("target_half_range", ((UPPER - LOWER) * 0.5).clone())
        self.network = nn.Sequential(
            nn.Linear(578, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 16), nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((inputs - self.input_mean) / self.input_std, -10.0, 10.0)
        return self.target_center + self.target_half_range * self.network(normalized)

    def normalized_target(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.target_center) / self.target_half_range


def _huber(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(prediction, target, reduction="mean", delta=0.05)


def _evaluate(
    model: PhaseDirectActionMLP,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pair_left: torch.Tensor,
    pair_right: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(inputs)
        pred_norm = model.normalized_target(prediction)
        target_norm = model.normalized_target(targets)
        target_loss = _huber(pred_norm[mask], target_norm[mask])
        diff_loss = _huber(
            pred_norm[pair_right] - pred_norm[pair_left],
            target_norm[pair_right] - target_norm[pair_left],
        )
        total = target_loss + 0.1 * diff_loss
        absolute = torch.abs(prediction[mask] - targets[mask])
    return {
        "loss": float(total.cpu()),
        "target_loss": float(target_loss.cpu()),
        "first_difference_loss": float(diff_loss.cpu()),
        "physical_mae": float(torch.mean(absolute).cpu()),
        "physical_q95": float(torch.quantile(absolute, 0.95).cpu()),
    }


def train_model(
    arrays: dict[str, np.ndarray],
    *,
    device: torch.device,
    seed: int = 20260717,
    max_epochs: int = 500,
    patience: int = 50,
    epoch_callback=None,
) -> tuple[PhaseDirectActionMLP, dict[str, Any], list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    obs = torch.from_numpy(arrays["student_obs_570"]).float()
    phase = torch.from_numpy(arrays["phase_features_8"]).float()
    inputs_cpu = torch.cat((obs, phase), dim=1)
    targets_cpu = torch.from_numpy(arrays["safe_teacher_target_16"]).float()
    lower, upper = LOWER.reshape(1, -1), UPPER.reshape(1, -1)
    if bool(torch.any((targets_cpu < lower - 1.0e-7) | (targets_cpu > upper + 1.0e-7)).item()):
        raise RuntimeError("direct-action training target exceeds the physical envelope")
    stage = torch.from_numpy(arrays["stage"]).long()
    episode = torch.from_numpy(arrays["episode_id"]).long()
    step = torch.from_numpy(arrays["step"]).long()
    split = torch.from_numpy(arrays["split"]).long()
    train_mask_cpu = split == 0
    validation_mask_cpu = split == 1
    input_mean = torch.mean(inputs_cpu[train_mask_cpu], dim=0)
    input_std = torch.std(inputs_cpu[train_mask_cpu], dim=0, unbiased=False).clamp_min(1.0e-4)
    model = PhaseDirectActionMLP(input_mean, input_std).to(device)
    inputs, targets = inputs_cpu.to(device), targets_cpu.to(device)
    train_mask, validation_mask = train_mask_cpu.to(device), validation_mask_cpu.to(device)
    train_left, train_right = temporal_pair_indices(episode, step, split, 0)
    val_left, val_right = temporal_pair_indices(episode, step, split, 1)
    train_left, train_right = train_left.to(device), train_right.to(device)
    val_left, val_right = val_left.to(device), val_right.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss, best_epoch, stale = math.inf, -1, 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train()
        balanced = stage_balanced_indices(
            stage, train_mask_cpu, per_stage=1024, generator=generator
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(inputs)
        pred_norm = model.normalized_target(prediction)
        target_norm = model.normalized_target(targets)
        target_loss = _huber(pred_norm[balanced], target_norm[balanced])
        diff_loss = _huber(
            pred_norm[train_right] - pred_norm[train_left],
            target_norm[train_right] - target_norm[train_left],
        )
        loss = target_loss + 0.1 * diff_loss
        loss.backward()
        optimizer.step()
        train_metrics = _evaluate(
            model, inputs, targets, train_mask, train_left, train_right
        )
        validation_metrics = _evaluate(
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
        if validation_metrics["loss"] < best_loss - 1.0e-9:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("direct-action optimization produced no checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    result = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_loss": best_loss,
        "train_metrics": _evaluate(model, inputs, targets, train_mask, train_left, train_right),
        "validation_metrics": _evaluate(
            model, inputs, targets, validation_mask, val_left, val_right
        ),
    }
    return model, result, history


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite direct-action output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_manifest, arrays = load_bound_dataset(
        args.dataset_manifest, args.dataset_manifest_sha256
    )
    prereg = _load_preregistration(
        args.preregistration, args.preregistration_sha256, dataset_manifest["dataset_sha256"]
    )
    if prereg.get("dataset_manifest_sha256") != args.dataset_manifest_sha256:
        raise RuntimeError("direct-action dataset manifest binding changed")
    reference_binding = load_reference_binding(dataset_manifest)
    round_index = int(prereg.get("round", 0))
    stage = "supervised_initial" if round_index == 0 else f"dagger_round{round_index}"
    import wandb
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"highstep_fixed_condition_phase_direct_action_{stage}",
        group=WORKFLOW_ID,
        job_type="supervised_phase_direct_action",
        mode="online",
        config={
            "workflow_id": WORKFLOW_ID,
            "stage": stage,
            "attempt": 1,
            "spec_path": prereg["spec_path"],
            "spec_sha256": prereg["spec_sha256"],
            "preregistration_path": str(args.preregistration.expanduser().resolve()),
            "preregistration_sha256": args.preregistration_sha256,
            "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            **reference_binding,
            "architecture": [578, 512, 512, 256, 16],
            "target": "safe_teacher_post_prior_mapped_target_16",
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "max_epochs": 500,
            "patience": 50,
            "stage_balanced": True,
            "ppo_enabled": False,
        },
    )
    try:
        model, result, history = train_model(
            arrays,
            device=torch.device(args.device),
            epoch_callback=lambda epoch, record: run.log(record, step=epoch),
        )
        metrics_path = output_dir / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            for record in history:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        checkpoint_path = output_dir / "phase_direct_action_best.pt"
        torch.save({
            "schema_version": 1,
            "kind": "bounded_phase_direct_action_mlp",
            "workflow_id": WORKFLOW_ID,
            "model_state_dict": model.state_dict(),
            "architecture": {"input_dim": 578, "hidden_dims": [512, 512, 256], "output_dim": 16},
            "joint_names": JOINT_NAMES,
            "physical_lower": LOWER.tolist(),
            "physical_upper": UPPER.tolist(),
            "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            "preregistration_path": str(args.preregistration.expanduser().resolve()),
            "preregistration_sha256": args.preregistration_sha256,
            **reference_binding,
            "best_epoch": result["best_epoch"],
        }, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        script_path = output_dir / "phase_direct_action_best.jit.pt"
        torch.jit.trace(model, torch.zeros(1, 578, device=args.device)).save(str(script_path))
        artifact = wandb.Artifact(
            f"highstep-fixed-condition-phase-direct-action-{stage}",
            type="model",
            metadata={"checkpoint_sha256": checkpoint_sha},
        )
        artifact.add_file(str(checkpoint_path)); artifact.add_file(str(script_path))
        run.log_artifact(artifact)
        manifest = {
            "schema_version": 1,
            "kind": "highstep_fixed_condition_phase_direct_action_training",
            "status": "training_complete_pending_wandb_remote_verification",
            "workflow_id": WORKFLOW_ID,
            "stage": stage,
            "attempt": 1,
            "spec_path": prereg["spec_path"],
            "spec_sha256": prereg["spec_sha256"],
            "preregistration_path": str(args.preregistration.expanduser().resolve()),
            "preregistration_sha256": args.preregistration_sha256,
            "dataset_manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "dataset_manifest_sha256": args.dataset_manifest_sha256,
            "dataset_sha256": dataset_manifest["dataset_sha256"],
            **reference_binding,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "torchscript_path": str(script_path),
            "torchscript_sha256": sha256_file(script_path),
            "metrics_path": str(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "teacher_checkpoint": prereg["teacher_checkpoint"],
            "teacher_sha256": prereg["teacher_sha256"],
            "wandb": {"entity": run.entity, "project": run.project, "run_id": run.id,
                      "run_name": run.name, "group": WORKFLOW_ID, "url": run.url},
            **result,
        }
        manifest_path = output_dir / "training_manifest.json"
        _atomic_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        run.summary.update({
            "output_checkpoint_sha256": checkpoint_sha,
            "effective_epochs": result["epochs_completed"],
            "best_epoch": result["best_epoch"],
            "best_validation_loss": result["best_validation_loss"],
            "training_manifest_path": str(manifest_path),
            "training_manifest_sha256": manifest_sha,
            "gate_conclusion": "training_complete_pending_behavior_gate",
        })
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    remote_verified, remote_error = False, None
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
        "kind": "highstep_phase_direct_action_wandb_verification",
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
        raise RuntimeError(f"W&B direct-action verification failed closed: {remote_error}")
    return {
        "training_manifest_path": str(manifest_path),
        "training_manifest_sha256": manifest_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "wandb_run_path": remote_path,
        "wandb_url": run.url,
        "wandb_verification_path": str(verification_path),
        "wandb_verification_sha256": sha256_file(verification_path),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="xinqili551-the-university-of-hong-kong")
    parser.add_argument("--wandb-project", default="isaaclab")
    args = parser.parse_args()
    print(json.dumps(run_training(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
