#!/usr/bin/env python3
"""Merge the preregistered earliest successful fixed-condition Teacher rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


EXPECTED_ARRAYS = (
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


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} SHA mismatch: expected={expected_sha256} actual={actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
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


def merge_datasets(
    *,
    recovery_amendment_path: Path,
    recovery_amendment_sha256: str,
    batch_bindings: list[tuple[Path, str]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged dataset: {output_dir}")
    recovery = _load_bound_json(
        recovery_amendment_path,
        recovery_amendment_sha256,
        "dataset recovery amendment",
    )
    if recovery.get("status") != "frozen_before_second_identical_batch":
        raise RuntimeError("dataset recovery amendment has the wrong frozen status")
    contract = recovery.get("recovery_contract", {})
    if (
        int(contract.get("additional_batches", -1)) != 1
        or int(contract.get("selected_rollouts_required", -1)) != 15
        or int(contract.get("train_rollouts", -1)) != 12
        or int(contract.get("validation_rollouts", -1)) != 3
    ):
        raise RuntimeError("dataset recovery selection contract changed")
    if len(batch_bindings) != 2:
        raise RuntimeError("recovery contract requires exactly two dataset batches")

    manifests: list[dict[str, Any]] = []
    loaded_arrays: list[dict[str, np.ndarray]] = []
    for batch_index, (manifest_path, expected_sha) in enumerate(batch_bindings):
        manifest = _load_bound_json(manifest_path, expected_sha, f"batch {batch_index} manifest")
        if manifest.get("kind") != "highstep_fixed_condition_phase_residual_teacher_dataset":
            raise RuntimeError(f"batch {batch_index} has the wrong dataset kind")
        if int(manifest.get("rollout_count", -1)) != 15 or int(manifest.get("steps_per_rollout", -1)) != 120:
            raise RuntimeError(f"batch {batch_index} shape contract changed")
        dataset_path = Path(manifest["dataset_path"]).expanduser().resolve()
        if sha256_file(dataset_path) != manifest.get("dataset_sha256"):
            raise RuntimeError(f"batch {batch_index} dataset SHA changed")
        archive = np.load(dataset_path, allow_pickle=False)
        arrays = {name: archive[name] for name in EXPECTED_ARRAYS}
        if any(arrays[name].shape[0] != 1800 for name in EXPECTED_ARRAYS):
            raise RuntimeError(f"batch {batch_index} sample count changed")
        manifests.append(manifest)
        loaded_arrays.append(arrays)

    invariant_keys = (
        "task",
        "checkpoint_sha256",
        "seed",
        "fixed_velocity_command",
        "terrain_type",
        "terrain_level",
        "front_edge_gap",
        "action_delay_steps",
        "settle_steps_before_dataset",
        "dataset_pre_active_steps",
        "source_timing_amendment_sha256",
    )
    first_metadata = manifests[0].get("metadata", {})
    second_metadata = manifests[1].get("metadata", {})
    changed = [key for key in invariant_keys if first_metadata.get(key) != second_metadata.get(key)]
    if changed:
        raise RuntimeError(f"dataset batch condition changed: {changed}")
    if manifests[0].get("selected_reference_sha256") != manifests[1].get("selected_reference_sha256"):
        raise RuntimeError("selected reference changed between dataset batches")

    candidates: list[tuple[int, int]] = []
    for batch_index, manifest in enumerate(manifests):
        hold = manifest.get("env_hold_success", [])
        full = manifest.get("env_full_climb_success", [])
        if len(hold) != 15 or len(full) != 15:
            raise RuntimeError(f"batch {batch_index} behavior mask length changed")
        for env_id in range(15):
            if bool(hold[env_id]) and bool(full[env_id]):
                candidates.append((batch_index, env_id))
    selected = candidates[:15]
    if len(selected) != 15:
        raise RuntimeError(f"only {len(selected)} successful Teacher rollouts are available")

    output_chunks: dict[str, list[np.ndarray]] = {name: [] for name in EXPECTED_ARRAYS}
    provenance: list[dict[str, int]] = []
    for new_episode_id, (batch_index, source_env_id) in enumerate(selected):
        source = loaded_arrays[batch_index]
        mask = source["episode_id"] == source_env_id
        indices = np.flatnonzero(mask)
        if len(indices) != 120:
            raise RuntimeError(
                f"batch {batch_index} env {source_env_id} has {len(indices)} samples, expected 120"
            )
        order = indices[np.argsort(source["step"][indices], kind="stable")]
        if not np.array_equal(source["step"][order], np.arange(120)):
            raise RuntimeError(f"batch {batch_index} env {source_env_id} step sequence changed")
        for name in EXPECTED_ARRAYS:
            chunk = source[name][order].copy()
            if name == "episode_id":
                chunk[...] = new_episode_id
            elif name == "split":
                chunk[...] = 0 if new_episode_id < 12 else 1
            output_chunks[name].append(chunk)
        provenance.append(
            {
                "new_episode_id": new_episode_id,
                "source_batch_index": batch_index,
                "source_env_id": source_env_id,
            }
        )
    merged = {name: np.concatenate(chunks, axis=0) for name, chunks in output_chunks.items()}
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / "teacher_dataset.npz"
    _atomic_npz(dataset_path, merged)
    manifest = {
        "schema_version": 1,
        "kind": "highstep_fixed_condition_phase_residual_merged_teacher_dataset",
        "status": "frozen_teacher_dataset_ready",
        "workflow_id": "highstep_fixed_condition_phase_residual_20260717",
        "recovery_amendment_path": str(recovery_amendment_path.expanduser().resolve()),
        "recovery_amendment_sha256": recovery_amendment_sha256,
        "source_batches": [
            {
                "manifest_path": str(path.expanduser().resolve()),
                "manifest_sha256": expected_sha,
                "dataset_path": manifest_payload["dataset_path"],
                "dataset_sha256": manifest_payload["dataset_sha256"],
                "successful_rollouts": int(sum(manifest_payload["env_hold_success"])),
            }
            for (path, expected_sha), manifest_payload in zip(batch_bindings, manifests)
        ],
        "selection_rule": contract["selection_rule"],
        "selected_rollout_provenance": provenance,
        "rollout_count": 15,
        "train_rollouts": 12,
        "validation_rollouts": 3,
        "steps_per_rollout": 120,
        "samples": 1800,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "array_shapes": {name: list(value.shape) for name, value in merged.items()},
        "all_selected_hold_success": True,
        "all_selected_full_climb_success": True,
        "failed_source_rollouts_preserved": True,
        "metadata": {key: first_metadata.get(key) for key in invariant_keys},
    }
    manifest_path = output_dir / "dataset_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-amendment", type=Path, required=True)
    parser.add_argument("--recovery-amendment-sha256", required=True)
    parser.add_argument("--batch", nargs=2, action="append", metavar=("MANIFEST", "SHA256"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge_datasets(
        recovery_amendment_path=args.recovery_amendment,
        recovery_amendment_sha256=args.recovery_amendment_sha256,
        batch_bindings=[(Path(path), sha) for path, sha in args.batch],
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
