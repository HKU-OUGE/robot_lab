from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "highstep_phase_residual_merge_dataset.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase_residual_merge", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _batch(tmp_path: Path, index: int, failed_env: int) -> tuple[Path, str]:
    folder = tmp_path / f"batch{index}"
    folder.mkdir()
    episode = np.tile(np.arange(15), 120)
    step = np.repeat(np.arange(120), 15)
    arrays = {
        "student_obs_570": np.full((1800, 570), index, np.float32),
        "phase_features_8": np.zeros((1800, 8), np.float32),
        "teacher_post_prior_mapped_target_16": np.zeros((1800, 16), np.float32),
        "safe_teacher_target_16": np.zeros((1800, 16), np.float32),
        "reference_target_16": np.zeros((1800, 16), np.float32),
        "residual_target_16": np.zeros((1800, 16), np.float32),
        "joint_pos_16": np.zeros((1800, 16), np.float32),
        "joint_vel_16": np.zeros((1800, 16), np.float32),
        "stage": np.zeros(1800, np.int64),
        "episode_id": episode,
        "step": step,
        "split": (episode >= 12).astype(np.int64),
    }
    dataset = folder / "teacher_dataset.npz"
    np.savez_compressed(dataset, **arrays)
    success = [env != failed_env for env in range(15)]
    metadata = {
        "task": "teacher", "checkpoint_sha256": "a" * 64, "seed": 11,
        "fixed_velocity_command": [0.72, 0.0, 0.0], "terrain_type": "box_hard",
        "terrain_level": 9, "front_edge_gap": 0.55, "action_delay_steps": 0,
        "settle_steps_before_dataset": 296, "dataset_pre_active_steps": 10,
        "source_timing_amendment_sha256": "b" * 64,
    }
    manifest = {
        "kind": "highstep_fixed_condition_phase_residual_teacher_dataset",
        "rollout_count": 15, "steps_per_rollout": 120,
        "dataset_path": str(dataset), "dataset_sha256": _sha(dataset),
        "selected_reference_sha256": "c" * 64,
        "env_hold_success": success, "env_full_climb_success": success,
        "metadata": metadata,
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, _sha(path)


def _amendment(tmp_path: Path) -> tuple[Path, str]:
    payload = {
        "status": "frozen_before_second_identical_batch",
        "recovery_contract": {
            "additional_batches": 1, "selected_rollouts_required": 15,
            "train_rollouts": 12, "validation_rollouts": 3,
            "selection_rule": "batch then env",
        },
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, _sha(path)


def test_selects_earliest_fifteen_successful_real_rollouts(tmp_path):
    module = _load_module()
    amendment, amendment_sha = _amendment(tmp_path)
    batches = [_batch(tmp_path, 0, 8), _batch(tmp_path, 1, 8)]
    result = module.merge_datasets(
        recovery_amendment_path=amendment,
        recovery_amendment_sha256=amendment_sha,
        batch_bindings=batches,
        output_dir=tmp_path / "merged",
    )
    assert result["status"] == "frozen_teacher_dataset_ready"
    assert result["selected_rollout_provenance"][-1] == {
        "new_episode_id": 14, "source_batch_index": 1, "source_env_id": 0
    }
    arrays = np.load(result["dataset_path"])
    assert set(arrays["episode_id"]) == set(range(15))
    assert set(arrays["episode_id"][arrays["split"] == 1]) == {12, 13, 14}


def test_wrong_batch_hash_fails_closed(tmp_path):
    module = _load_module()
    amendment, amendment_sha = _amendment(tmp_path)
    first = _batch(tmp_path, 0, 8)
    second = _batch(tmp_path, 1, 8)
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        module.merge_datasets(
            recovery_amendment_path=amendment,
            recovery_amendment_sha256=amendment_sha,
            batch_bindings=[first, (second[0], "0" * 64)],
            output_dir=tmp_path / "merged",
        )
