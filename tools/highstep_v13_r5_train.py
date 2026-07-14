#!/usr/bin/env python3
"""Run one bounded fixed-replay v1.3 R5 actor-tail training stage."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from tools import highstep_v13_r4_fit as common


ROOT = common.ROOT
PREREGISTRATION = ROOT / "tmp/highstep_student_recovery_v13_20260713/r5_preregistration.json"
PREREGISTRATION_SHA256 = "c222cbeafc5ca1dbd1cb94cdb93b588a25cc7b7d160af8385c4fdd8a14a48736"
B500 = common.B500
B500_SHA256 = common.B500_SHA256
TRAINABLE_NAMES = (
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
)


def _authority(dataset_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if common.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256 or PREREGISTRATION.stat().st_mode & 0o222:
        raise RuntimeError("R5 preregistration SHA/read-only binding failed")
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if common.sha256_file(B500) != B500_SHA256:
        raise RuntimeError("R5 B500 anchor changed")
    dataset_manifest = dataset_manifest.resolve(strict=True)
    if dataset_manifest.stat().st_mode & 0o222:
        raise RuntimeError("R5 dataset manifest is writable")
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if not (
        dataset.get("kind") == "highstep_v13_b500_core9_replay_dataset"
        and dataset.get("matrix_complete") is True
        and dataset.get("total_frames") == 5400
        and dataset.get("preregistration_sha256") == common.PREREGISTRATION_SHA256
    ):
        raise RuntimeError("R5 dataset is not the exact R4 replay dataset")
    return prereg, dataset


def _fixed_prefix(inputs: torch.Tensor, state: Mapping[str, torch.Tensor]) -> torch.Tensor:
    value = inputs
    for index in (0, 2):
        value = F.elu(F.linear(value, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"]))
    if value.shape[1] != 256:
        raise RuntimeError("R5 fixed actor prefix dimension changed")
    return value


def _dataset(dataset: Mapping[str, Any], anchor_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    weight_parts: list[torch.Tensor] = []
    for item in dataset["trajectories"]:
        frames = Path(item["frames"]).resolve(strict=True)
        if common.sha256_file(frames) != item["frames_sha256"]:
            raise RuntimeError(f"R5 frames changed: {frames}")
        records = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines()]
        if len(records) != 600:
            raise RuntimeError("R5 trajectory is not 600 frames")
        policy = torch.tensor([row["policy_observations"] for row in records], dtype=torch.float32)
        latent = torch.tensor([row["student_latent_clamped"] for row in records], dtype=torch.float32)
        anchor = torch.tensor([row["student_action"] for row in records], dtype=torch.float32)
        teacher = torch.tensor(
            [row["teacher_action_post_prior_raw_equivalent"] for row in records],
            dtype=torch.float32,
        )
        prefix = _fixed_prefix(torch.cat((policy, latent), dim=1), anchor_state)
        success = item["role"] == "success_anchor"
        target = anchor if success else teacher
        phases: dict[str, list[int]] = {}
        for index, row in enumerate(records):
            phases.setdefault(str(row["phase"]), []).append(index)
        weights = torch.zeros(600, dtype=torch.float32)
        for indices in phases.values():
            weights[indices] = 1.0 / (len(phases) * len(indices))
        if success:
            weights *= 20.0
        prefix_parts.append(prefix)
        target_parts.append(target)
        weight_parts.append(weights)
    return {
        "prefix": torch.cat(prefix_parts),
        "target": torch.cat(target_parts),
        "weights": torch.cat(weight_parts),
    }


class ActorTail(nn.Module):
    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        self.hidden = nn.Linear(256, 128)
        self.head = nn.Linear(128, 16)
        with torch.no_grad():
            self.hidden.weight.copy_(state["actor.4.weight"])
            self.hidden.bias.copy_(state["actor.4.bias"])
            self.head.weight.copy_(state["actor.6.weight"])
            self.head.bias.copy_(state["actor.6.bias"])

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.head(F.elu(self.hidden(prefix)))


def _optimizer(model: ActorTail) -> torch.optim.Adam:
    return torch.optim.Adam(
        [model.hidden.weight, model.hidden.bias, model.head.weight, model.head.bias],
        lr=1.0e-6,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )


def _audit_scope(
    anchor_state: Mapping[str, torch.Tensor], candidate_state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    changed: list[str] = []
    if set(anchor_state) != set(candidate_state):
        raise RuntimeError("R5 candidate/anchor state keys differ")
    for name in anchor_state:
        before, after = anchor_state[name], candidate_state[name]
        if before.shape != after.shape or before.dtype != after.dtype or not bool(torch.isfinite(after).all().item()):
            raise RuntimeError(f"R5 tensor contract failed: {name}")
        if not torch.equal(before, after):
            if name not in TRAINABLE_NAMES:
                raise RuntimeError(f"R5 changed frozen tensor: {name}")
            changed.append(name)
    if not changed:
        raise RuntimeError("R5 stage produced no permitted parameter change")
    return {"changed_tensors": sorted(changed), "frozen_tensor_violation_count": 0}


def _save(
    *,
    anchor_payload: Mapping[str, Any],
    model: ActorTail,
    optimizer: torch.optim.Adam,
    total_epochs: int,
    dataset_manifest: Path,
    output_dir: Path,
    scope: Mapping[str, Any],
    losses: list[float],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "params").mkdir()
    payload = copy.deepcopy(anchor_payload)
    state = payload["model_state_dict"]
    state["actor.4.weight"] = model.hidden.weight.detach().cpu().clone()
    state["actor.4.bias"] = model.hidden.bias.detach().cpu().clone()
    state["actor.6.weight"] = model.head.weight.detach().cpu().clone()
    state["actor.6.bias"] = model.head.bias.detach().cpu().clone()
    payload.setdefault("infos", {})["highstep_v13_r5"] = {
        "schema_version": 1,
        "stage": "R5",
        "workflow_id": "highstep_student_recovery_v13_20260713",
        "preregistration": str(PREREGISTRATION.resolve()),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": common.sha256_file(dataset_manifest),
        "anchor_checkpoint": str(B500.resolve()),
        "anchor_checkpoint_sha256": B500_SHA256,
        "effective_epochs": total_epochs,
        "trainable_tensors": list(TRAINABLE_NAMES),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
    }
    checkpoint = output_dir / "model_498.pt"
    temporary = output_dir / f".model_498.pt.{os.getpid()}.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint)
    checkpoint_sha = common.sha256_file(checkpoint)
    source_params = B500.parent / "params"
    for name in ("agent.yaml", "env.yaml", "highstep_schedule_manifest.json"):
        shutil.copyfile(source_params / name, output_dir / "params" / name)
    runtime = json.loads((source_params / "highstep_runtime_state.json").read_text())
    snapshot = copy.deepcopy(next(item for item in runtime["snapshots"] if item["checkpoint_file"] == B500.name))
    snapshot["checkpoint_file"] = checkpoint.name
    snapshot["checkpoint_sha256"] = checkpoint_sha
    snapshot["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    common.atomic_json(
        output_dir / "params/highstep_runtime_state.json",
        {"context": "train", "schema_version": 2, "snapshots": [snapshot]},
    )
    binding = {
        "schema_version": 1,
        "kind": "highstep_v13_r5_candidate_binding",
        "workflow_id": "highstep_student_recovery_v13_20260713",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "anchor_checkpoint": str(B500.resolve()),
        "anchor_checkpoint_sha256": B500_SHA256,
        "preregistration": str(PREREGISTRATION.resolve()),
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "dataset_manifest": str(dataset_manifest.resolve()),
        "dataset_manifest_sha256": common.sha256_file(dataset_manifest),
        "effective_epochs": total_epochs,
        "trainable_tensors": list(TRAINABLE_NAMES),
        "scope_audit": dict(scope),
        "losses": losses,
    }
    binding_path = output_dir / "params/highstep_v13_r5_binding.json"
    common.atomic_json(binding_path, binding, read_only=True)
    manifest = {
        **binding,
        "kind": "highstep_v13_r5_candidate_manifest",
        "run_dir": str(output_dir.resolve()),
        "runtime_state": str((output_dir / "params/highstep_runtime_state.json").resolve()),
        "runtime_state_sha256": common.sha256_file(output_dir / "params/highstep_runtime_state.json"),
        "binding": str(binding_path.resolve()),
        "binding_sha256": common.sha256_file(binding_path),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    common.atomic_json(output_dir / "candidate_manifest.json", manifest, read_only=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--load-mode", choices=("weights_only", "full"), required=True)
    parser.add_argument("--additional-epochs", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    wandb_run = None
    wandb_stage_manifest = os.environ.get("HIGHSTEP_WANDB_STAGE_MANIFEST")
    if wandb_stage_manifest:
        from tools import highstep_wandb_stage_gate
        stage_path = Path(wandb_stage_manifest).resolve(strict=True)
        stage = json.loads(stage_path.read_text())
        config = json.loads((stage_path.parent / "config.snapshot.json").read_text())
        highstep_wandb_stage_gate.validate_contract(config)
        import wandb
        wandb_run = wandb.init(
            entity=stage["entity"], project=stage["project"], id=stage["run_id"],
            name=stage["run_name"], group=stage["group"], resume="allow", config=config,
        )
    dataset_manifest = args.dataset_manifest.resolve(strict=True)
    prereg, dataset_manifest_payload = _authority(dataset_manifest)
    anchor_payload = torch.load(B500, map_location="cpu", weights_only=False)
    anchor_state = anchor_payload["model_state_dict"]
    source = args.checkpoint.resolve(strict=True)
    if args.load_mode == "weights_only":
        if source != B500.resolve() or args.additional_epochs != 1:
            raise RuntimeError("fresh R5 must run exactly one epoch from B500")
        source_payload = anchor_payload
        previous_epochs = 0
    else:
        source_payload = torch.load(source, map_location="cpu", weights_only=False)
        extra = (source_payload.get("infos") or {}).get("highstep_v13_r5")
        if not (
            isinstance(extra, Mapping)
            and extra.get("stage") == "R5"
            and extra.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and extra.get("dataset_manifest_sha256") == common.sha256_file(dataset_manifest)
            and extra.get("anchor_checkpoint_sha256") == B500_SHA256
            and extra.get("trainable_tensors") == list(TRAINABLE_NAMES)
        ):
            raise RuntimeError("R5 full-resume binding failed")
        previous_epochs = int(extra["effective_epochs"])
    total_epochs = previous_epochs + args.additional_epochs
    if args.additional_epochs <= 0 or total_epochs > int(prereg["absolute_epoch_cap"]):
        raise RuntimeError("R5 epoch budget exceeded")
    model = ActorTail(source_payload["model_state_dict"])
    optimizer = _optimizer(model)
    if args.load_mode == "full":
        optimizer.load_state_dict(source_payload["infos"]["highstep_v13_r5"]["optimizer_state_dict"])
    data = _dataset(dataset_manifest_payload, anchor_state)
    losses: list[float] = []
    for epoch in range(previous_epochs, total_epochs):
        generator = torch.Generator().manual_seed(130713 + epoch)
        permutation = torch.randperm(data["prefix"].shape[0], generator=generator)
        batches = torch.tensor_split(permutation, 4)
        epoch_loss = 0.0
        for indices in batches:
            predicted = model(data["prefix"][indices])
            error = torch.mean(torch.square(predicted - data["target"][indices]), dim=1)
            weights = data["weights"][indices]
            loss = torch.sum(weights * error) / torch.clamp(torch.sum(weights), min=1.0e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach().item()) / 4.0
        losses.append(epoch_loss)
        if wandb_run is not None:
            wandb_run.log({"train/epoch_loss": epoch_loss, "effective_epoch": epoch + 1})
    candidate_state = copy.deepcopy(anchor_state)
    candidate_state["actor.4.weight"] = model.hidden.weight.detach().cpu()
    candidate_state["actor.4.bias"] = model.hidden.bias.detach().cpu()
    candidate_state["actor.6.weight"] = model.head.weight.detach().cpu()
    candidate_state["actor.6.bias"] = model.head.bias.detach().cpu()
    scope = _audit_scope(anchor_state, candidate_state)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"R5 output directory already exists: {output_dir}")
    manifest = _save(
        anchor_payload=anchor_payload,
        model=model,
        optimizer=optimizer,
        total_epochs=total_epochs,
        dataset_manifest=dataset_manifest,
        output_dir=output_dir,
        scope=scope,
        losses=losses,
    )
    if wandb_run is not None:
        wandb_run.summary["local_training_complete"] = True
        wandb_run.summary["effective_updates"] = total_epochs
        wandb_run.finish()
    print(json.dumps({"checkpoint": manifest["checkpoint"], "effective_epochs": total_epochs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
