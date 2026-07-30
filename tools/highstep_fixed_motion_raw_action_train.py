#!/usr/bin/env python3
"""Train one fresh fixed-motion 578->16 raw-action Student stage."""

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


WORKFLOW_ID = "highstep_fixed_motion_raw_action_20260717"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path, sha: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != sha:
        raise RuntimeError(f"SHA mismatch: {path}")
    return json.loads(path.read_text())


class RawActionMLP(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("input_mean", mean.clone())
        self.register_buffer("input_std", std.clone())
        self.network = nn.Sequential(
            nn.Linear(578, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 16),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp((value - self.input_mean) / self.input_std, -10.0, 10.0)
        return self.network(normalized)


def train_model(arrays: dict[str, np.ndarray], device: torch.device, *, max_epochs: int = 1000,
                callback=None) -> tuple[RawActionMLP, dict[str, Any], list[dict[str, float]]]:
    random.seed(20260717); np.random.seed(20260717); torch.manual_seed(20260717)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(20260717)
    obs = torch.from_numpy(arrays["student_obs_570"]).float()
    phase = torch.from_numpy(arrays["phase_features_8"]).float()
    target = torch.from_numpy(arrays["teacher_policy_raw_16"]).float()
    inputs = torch.cat((obs, phase), dim=1)
    episode = torch.from_numpy(arrays["episode_id"]).long()
    if inputs.shape[1] != 578 or target.shape[1] != 16 or not torch.isfinite(target).all():
        raise RuntimeError("raw BC dataset contract changed")
    if int(np.unique(arrays["episode_id"]).size) < 1:
        raise RuntimeError("raw BC dataset has no episode")
    mean = inputs.mean(0)
    std = inputs.std(0, unbiased=False).clamp_min(1.0e-4)
    model = RawActionMLP(mean, std).to(device)
    x, y = inputs.to(device), target.to(device)
    adjacent = (episode[1:] == episode[:-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_loss, best_epoch, stale, best_state = math.inf, -1, 0, None
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        target_loss = F.huber_loss(prediction, y, delta=0.1)
        diff_loss = F.huber_loss((prediction[1:] - prediction[:-1])[adjacent],
                                 (y[1:] - y[:-1])[adjacent], delta=0.1)
        loss = target_loss + 0.1 * diff_loss
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(x); absolute = torch.abs(out - y)
            current = float((F.huber_loss(out, y, delta=0.1) + 0.1 * F.huber_loss(
                (out[1:] - out[:-1])[adjacent], (y[1:] - y[:-1])[adjacent], delta=0.1)).cpu())
            record = {"epoch": float(epoch), "train/loss": current,
                      "train/mae": float(absolute.mean().cpu()),
                      "train/q95": float(torch.quantile(absolute, 0.95).cpu()),
                      "train/max_abs_output": float(torch.abs(out).max().cpu())}
        history.append(record)
        if callback is not None: callback(epoch, record)
        if current < best_loss - 1.0e-8:
            best_loss, best_epoch, stale = current, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 100 and best_loss < 5.0e-4: break
    if best_state is None: raise RuntimeError("raw BC produced no checkpoint")
    model.load_state_dict(best_state); model.to(device).eval()
    with torch.no_grad():
        final = model(x); absolute = torch.abs(final - y)
    result = {"best_epoch": best_epoch, "epochs_completed": len(history), "best_loss": best_loss,
              "training_mae": float(absolute.mean().cpu()),
              "training_q95": float(torch.quantile(absolute, 0.95).cpu()),
              "maximum_raw_output": float(torch.abs(final).max().cpu())}
    return model, result, history


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = args.output_dir.expanduser().resolve()
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    prereg = load_json(args.preregistration, args.preregistration_sha256)
    smoke = load_json(args.smoke_manifest, args.smoke_manifest_sha256)
    if (prereg.get("workflow_id") != WORKFLOW_ID or smoke.get("passed") is not True
            or smoke.get("preregistration_sha256") != args.preregistration_sha256):
        raise RuntimeError("raw BC authority binding changed")
    dataset = Path(smoke["dataset_path"])
    if sha256_file(dataset) != smoke["dataset_sha256"]: raise RuntimeError("raw BC dataset SHA mismatch")
    arrays = dict(np.load(dataset, allow_pickle=False))
    import wandb
    run_name = f"{WORKFLOW_ID}_{args.stage}_attempt1"
    wb = wandb.init(entity=args.wandb_entity, project=args.wandb_project, name=run_name,
                    group=WORKFLOW_ID, job_type="raw_action_bc", mode="online", config={
        "workflow_id": WORKFLOW_ID, "stage": args.stage, "attempt": 1,
        "spec_sha256": prereg["spec_sha256"], "preregistration_sha256": args.preregistration_sha256,
        "teacher_checkpoint": prereg["teacher_checkpoint"], "teacher_sha256": prereg["teacher_sha256"],
        "dataset_sha256": smoke["dataset_sha256"], "unique_episode_count": int(np.unique(arrays["episode_id"]).size),
        "architecture": [578, 512, 512, 256, 16], "target": "teacher_policy_raw_action_16",
        "optimizer": "Adam", "learning_rate": 1.0e-3, "max_epochs": 1000,
        "ppo": False, "reward_training": False,
    })
    try:
        model, result, history = train_model(arrays, torch.device(args.device),
                                             callback=lambda step, data: wb.log(data, step=step))
        metrics = out / "metrics.jsonl"
        metrics.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in history))
        checkpoint = out / "raw_action_student_best.pt"
        torch.save({"kind": "fixed_motion_raw_action_student", "workflow_id": WORKFLOW_ID,
                    "stage": args.stage, "model_state_dict": model.state_dict(),
                    "architecture": [578, 512, 512, 256, 16], "fresh_optimizer": True,
                    "dataset_sha256": smoke["dataset_sha256"], **result}, checkpoint)
        script = out / "raw_action_student_best.jit.pt"
        torch.jit.trace(model, torch.zeros(1, 578, device=args.device)).save(str(script))
        manifest = {"schema_version": 1, "kind": "highstep_fixed_motion_raw_action_training",
                    "status": "completed", "workflow_id": WORKFLOW_ID, "stage": args.stage,
                    "preregistration_path": str(args.preregistration.resolve()),
                    "preregistration_sha256": args.preregistration_sha256,
                    "dataset_path": str(dataset.resolve()), "dataset_sha256": smoke["dataset_sha256"],
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                    "torchscript_path": str(script), "torchscript_sha256": sha256_file(script),
                    "metrics_path": str(metrics), "metrics_sha256": sha256_file(metrics),
                    "wandb": {"entity": wb.entity, "project": wb.project, "run_id": wb.id,
                              "name": wb.name, "group": WORKFLOW_ID, "url": wb.url}, **result}
        manifest_path = out / "training_manifest.json"; atomic_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        wb.summary.update({"output_checkpoint_sha256": manifest["checkpoint_sha256"],
                           "effective_epochs": result["epochs_completed"],
                           "training_manifest_path": str(manifest_path),
                           "training_manifest_sha256": manifest_sha,
                           "gate_conclusion": "training_complete_pending_behavior_gate"})
        wb.finish(exit_code=0)
    except BaseException:
        wb.finish(exit_code=1); raise
    remote_path = f"{wb.entity}/{wb.project}/{wb.id}"; error = None; verified = False
    for _ in range(30):
        try:
            remote = wandb.Api(timeout=30).run(remote_path)
            verified = remote.summary.get("output_checkpoint_sha256") == manifest["checkpoint_sha256"]
            if verified: break
            error = "remote summary incomplete"
        except Exception as exc: error = repr(exc)
        time.sleep(2)
    verification = {"schema_version": 1, "kind": "raw_action_wandb_verification",
                    "status": "verified" if verified else "failed_closed", "remote_verified": verified,
                    "run_path": remote_path, "run_url": wb.url,
                    "checkpoint_sha256": manifest["checkpoint_sha256"], "error": error}
    verification_path = out / "wandb_verification.json"; atomic_json(verification_path, verification)
    if not verified: raise RuntimeError(f"W&B verification failed closed: {error}")
    return {"training_manifest_path": str(manifest_path), "training_manifest_sha256": manifest_sha,
            "wandb_verification_path": str(verification_path),
            "wandb_verification_sha256": sha256_file(verification_path), "wandb_run": remote_path, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--smoke-manifest-sha256", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-entity", default="xinqili551-the-university-of-hong-kong")
    parser.add_argument("--wandb-project", default="isaaclab")
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
