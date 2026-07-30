#!/usr/bin/env python3
"""Freeze the round-1 aggregate: successful Teacher data + valid DAgger prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


WORKFLOW_ID = "highstep_fixed_condition_phase_residual_20260717"
TRAIN_ARRAYS = (
    "student_obs_570",
    "phase_features_8",
    "teacher_post_prior_mapped_target_16",
    "safe_teacher_target_16",
    "reference_target_16",
    "residual_target_16",
    "joint_pos_16",
    "joint_vel_16",
    "stage",
    "episode_id",
    "step",
    "split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, sha256: str, name: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != sha256:
        raise RuntimeError(f"{name} SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} is not a JSON object")
    return payload


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def merge_round1(
    *,
    teacher_manifest_path: Path,
    teacher_manifest_sha256: str,
    dagger_manifest_path: Path,
    dagger_manifest_sha256: str,
    preregistration_path: Path,
    preregistration_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite DAgger aggregate: {output_dir}")
    prereg = _load_json(preregistration_path, preregistration_sha256, "DAgger preregistration")
    teacher_manifest = _load_json(
        teacher_manifest_path, teacher_manifest_sha256, "Teacher dataset manifest"
    )
    dagger_manifest = _load_json(
        dagger_manifest_path, dagger_manifest_sha256, "DAgger dataset manifest"
    )
    if (
        prereg.get("kind") != "highstep_fixed_condition_phase_residual_dagger_preregistration"
        or prereg.get("status") != "frozen_before_dagger_round1_collection"
        or prereg.get("workflow_id") != WORKFLOW_ID
        or prereg.get("source_teacher_dataset_manifest_sha256") != teacher_manifest_sha256
        or teacher_manifest.get("kind")
        != "highstep_fixed_condition_phase_residual_merged_teacher_dataset"
        or teacher_manifest.get("status") != "frozen_teacher_dataset_ready"
        or dagger_manifest.get("kind")
        != "highstep_fixed_condition_phase_residual_dagger_dataset"
        or dagger_manifest.get("status") != "frozen_dagger_round1_dataset_ready"
        or dagger_manifest.get("preregistration_sha256") != preregistration_sha256
    ):
        raise RuntimeError("DAgger aggregate authority contract changed")

    teacher_path = Path(teacher_manifest["dataset_path"]).expanduser().resolve()
    dagger_path = Path(dagger_manifest["dataset_path"]).expanduser().resolve()
    if sha256_file(teacher_path) != teacher_manifest.get("dataset_sha256"):
        raise RuntimeError("Teacher dataset SHA mismatch")
    if sha256_file(dagger_path) != dagger_manifest.get("dataset_sha256"):
        raise RuntimeError("DAgger dataset SHA mismatch")
    teacher_npz = np.load(teacher_path, allow_pickle=False)
    dagger_npz = np.load(dagger_path, allow_pickle=False)
    teacher = {name: teacher_npz[name] for name in TRAIN_ARRAYS}
    dagger = {name: dagger_npz[name] for name in TRAIN_ARRAYS}
    valid = dagger_npz["sample_valid"].astype(bool)
    if valid.shape != (1800,) or int(np.sum(valid)) != int(dagger_manifest["valid_rows"]):
        raise RuntimeError("DAgger valid-mask contract changed")

    # Keep every Teacher rollout intact.  DAgger contributes only each
    # candidate rollout's contiguous pre-termination prefix; post-reset rows
    # remain preserved in the source archive but can never enter training.
    dagger_filtered: dict[str, np.ndarray] = {}
    for name in TRAIN_ARRAYS:
        values = dagger[name][valid].copy()
        if name == "episode_id":
            values = values + 15
        dagger_filtered[name] = values
    for env_id in range(15):
        episode = env_id + 15
        steps = dagger_filtered["step"][dagger_filtered["episode_id"] == episode]
        if steps.size and not np.array_equal(steps, np.arange(steps.size)):
            raise RuntimeError(f"DAgger env {env_id} valid prefix is not contiguous")

    aggregate = {
        name: np.concatenate((teacher[name], dagger_filtered[name]), axis=0)
        for name in TRAIN_ARRAYS
    }
    source = np.concatenate(
        (
            np.zeros(teacher["step"].shape[0], dtype=np.int64),
            np.ones(dagger_filtered["step"].shape[0], dtype=np.int64),
        )
    )
    aggregate["sample_source"] = source
    train_episodes = sorted(
        set(aggregate["episode_id"][aggregate["split"] == 0].astype(int).tolist())
    )
    validation_episodes = sorted(
        set(aggregate["episode_id"][aggregate["split"] == 1].astype(int).tolist())
    )
    if set(train_episodes[:12]) != set(range(12)) or set(validation_episodes[:3]) != {12, 13, 14}:
        raise RuntimeError("original whole-rollout 12/3 split changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "dagger_aggregate_dataset.npz"
    _atomic_npz(dataset_path, aggregate)
    manifest = {
        "schema_version": 1,
        "kind": "highstep_fixed_condition_phase_residual_dagger_aggregate",
        "status": "frozen_dagger_round1_aggregate_ready",
        "workflow_id": WORKFLOW_ID,
        "round": 1,
        "preregistration_path": str(preregistration_path.expanduser().resolve()),
        "preregistration_sha256": preregistration_sha256,
        "source_teacher_dataset_manifest_path": str(teacher_manifest_path.expanduser().resolve()),
        "source_teacher_dataset_manifest_sha256": teacher_manifest_sha256,
        "source_dagger_dataset_manifest_path": str(dagger_manifest_path.expanduser().resolve()),
        "source_dagger_dataset_manifest_sha256": dagger_manifest_sha256,
        "source_batches": teacher_manifest["source_batches"],
        "metadata": teacher_manifest["metadata"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "samples": int(aggregate["step"].shape[0]),
        "teacher_samples": int(teacher["step"].shape[0]),
        "dagger_valid_samples": int(dagger_filtered["step"].shape[0]),
        "dagger_invalid_rows_excluded": int(np.sum(~valid)),
        "train_rollouts": len(train_episodes),
        "validation_rollouts": len(validation_episodes),
        "train_episode_ids": train_episodes,
        "validation_episode_ids": validation_episodes,
        "array_shapes": {name: list(values.shape) for name, values in aggregate.items()},
        "post_termination_samples_used": 0,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-manifest-sha256", required=True)
    parser.add_argument("--dagger-manifest", type=Path, required=True)
    parser.add_argument("--dagger-manifest-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge_round1(
        teacher_manifest_path=args.teacher_manifest,
        teacher_manifest_sha256=args.teacher_manifest_sha256,
        dagger_manifest_path=args.dagger_manifest,
        dagger_manifest_sha256=args.dagger_manifest_sha256,
        preregistration_path=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
