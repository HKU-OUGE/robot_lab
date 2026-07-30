#!/usr/bin/env python3
"""Freeze one direct-action DAgger aggregate from valid candidate prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


WORKFLOW_ID = "highstep_fixed_condition_phase_direct_action_20260717"
TRAIN_ARRAYS = (
    "student_obs_570", "phase_features_8", "teacher_post_prior_mapped_target_16",
    "safe_teacher_target_16", "reference_target_16", "residual_target_16",
    "joint_pos_16", "joint_vel_16", "stage", "episode_id", "step", "split",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if sha256_file(path) != expected_sha256:
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


def merge_direct_round(
    *,
    round_index: int,
    base_manifest_path: Path,
    base_manifest_sha256: str,
    dagger_manifest_path: Path,
    dagger_manifest_sha256: str,
    preregistration_path: Path,
    preregistration_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    if round_index not in (1, 2):
        raise RuntimeError("direct-action DAgger round must be 1 or 2")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite direct DAgger aggregate: {output_dir}")
    base = _load_json(base_manifest_path, base_manifest_sha256, "base dataset manifest")
    dagger = _load_json(dagger_manifest_path, dagger_manifest_sha256, "DAgger dataset manifest")
    prereg = _load_json(preregistration_path, preregistration_sha256, "DAgger preregistration")
    expected_base = (
        base.get("kind") == "highstep_fixed_condition_phase_residual_dagger_aggregate"
        and int(base.get("round", -1)) == 2
        if round_index == 1
        else base.get("kind") == "highstep_fixed_condition_phase_direct_action_dagger_aggregate"
        and int(base.get("round", -1)) == 1
    )
    if (
        not expected_base
        or prereg.get("kind")
        != "highstep_fixed_condition_phase_direct_action_dagger_preregistration"
        or prereg.get("status") != f"frozen_before_dagger_round{round_index}_collection"
        or prereg.get("workflow_id") != WORKFLOW_ID
        or int(prereg.get("round", -1)) != round_index
        or prereg.get("source_base_dataset_manifest_sha256") != base_manifest_sha256
        or dagger.get("kind")
        != "highstep_fixed_condition_phase_direct_action_dagger_dataset"
        or dagger.get("status") != f"frozen_dagger_round{round_index}_dataset_ready"
        or int(dagger.get("round", -1)) != round_index
        or dagger.get("preregistration_sha256") != preregistration_sha256
    ):
        raise RuntimeError("direct-action DAgger aggregate authority contract changed")
    base_path = Path(base["dataset_path"]).expanduser().resolve()
    dagger_path = Path(dagger["dataset_path"]).expanduser().resolve()
    if sha256_file(base_path) != base.get("dataset_sha256"):
        raise RuntimeError("base DAgger dataset SHA mismatch")
    if sha256_file(dagger_path) != dagger.get("dataset_sha256"):
        raise RuntimeError("direct-action DAgger dataset SHA mismatch")
    base_npz = np.load(base_path, allow_pickle=False)
    dagger_npz = np.load(dagger_path, allow_pickle=False)
    base_arrays = {name: base_npz[name] for name in TRAIN_ARRAYS}
    valid = dagger_npz["sample_valid"].astype(bool)
    if valid.shape != (1800,) or int(np.sum(valid)) != int(dagger["valid_rows"]):
        raise RuntimeError("direct-action DAgger valid-mask contract changed")
    next_episode = int(np.max(base_arrays["episode_id"])) + 1
    filtered: dict[str, np.ndarray] = {}
    for name in TRAIN_ARRAYS:
        values = dagger_npz[name][valid].copy()
        if name == "episode_id":
            values += next_episode
        filtered[name] = values
    for env_id in range(15):
        steps = filtered["step"][filtered["episode_id"] == next_episode + env_id]
        if steps.size and not np.array_equal(steps, np.arange(steps.size)):
            raise RuntimeError(f"direct-action DAgger env {env_id} prefix is not contiguous")
    aggregate = {
        name: np.concatenate((base_arrays[name], filtered[name]), axis=0)
        for name in TRAIN_ARRAYS
    }
    base_source = (
        base_npz["sample_source"].astype(np.int64)
        if "sample_source" in base_npz.files
        else np.zeros(base_arrays["step"].shape[0], dtype=np.int64)
    )
    aggregate["sample_source"] = np.concatenate(
        (base_source, np.full(filtered["step"].shape[0], 2 + round_index, dtype=np.int64))
    )
    train_episodes = sorted(
        int(value) for value in set(aggregate["episode_id"][aggregate["split"] == 0])
    )
    validation_episodes = sorted(
        int(value) for value in set(aggregate["episode_id"][aggregate["split"] == 1])
    )
    if set(train_episodes) & set(validation_episodes):
        raise RuntimeError("direct-action DAgger train/validation episode leakage")
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / f"direct_action_dagger_round{round_index}_aggregate.npz"
    _atomic_npz(dataset_path, aggregate)
    manifest = {
        "schema_version": 1,
        "kind": "highstep_fixed_condition_phase_direct_action_dagger_aggregate",
        "status": f"frozen_direct_action_dagger_round{round_index}_aggregate_ready",
        "workflow_id": WORKFLOW_ID,
        "round": round_index,
        "preregistration_path": str(preregistration_path.expanduser().resolve()),
        "preregistration_sha256": preregistration_sha256,
        "source_base_dataset_manifest_path": str(base_manifest_path.expanduser().resolve()),
        "source_base_dataset_manifest_sha256": base_manifest_sha256,
        "source_dagger_dataset_manifest_path": str(dagger_manifest_path.expanduser().resolve()),
        "source_dagger_dataset_manifest_sha256": dagger_manifest_sha256,
        "source_batches": base["source_batches"],
        "metadata": base["metadata"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "samples": int(aggregate["step"].shape[0]),
        "base_samples": int(base_arrays["step"].shape[0]),
        "dagger_valid_samples": int(filtered["step"].shape[0]),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--dagger-manifest", type=Path, required=True)
    parser.add_argument("--dagger-manifest-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_direct_round(
        round_index=args.round,
        base_manifest_path=args.base_manifest,
        base_manifest_sha256=args.base_manifest_sha256,
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
