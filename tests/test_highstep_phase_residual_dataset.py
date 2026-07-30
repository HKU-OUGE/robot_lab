from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "scripts" / "rsl_rl" / "base" / "highstep_phase_residual_runtime.py"
DATASET_PATH = ROOT / "scripts" / "rsl_rl" / "base" / "highstep_phase_residual_dataset.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    runtime = _load_module(RUNTIME_PATH, "highstep_phase_residual_runtime")
    import sys
    sys.modules["highstep_phase_residual_runtime"] = runtime
    dataset = _load_module(DATASET_PATH, "highstep_phase_residual_dataset")
    target_fields = [f"reference_target.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    measured_fields = [f"measured_joint_pos.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reference_step", "global_phase", *target_fields, *measured_fields],
        )
        writer.writeheader()
        for step in range(3):
            values = [0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4
            row = {"reference_step": step, "global_phase": step / 2}
            row.update(dict(zip(target_fields, values)))
            row.update(dict(zip(measured_fields, values)))
            writer.writerow(row)
    reference = runtime.PhaseResidualReference(reference_path, expected_sha256=_sha(reference_path))
    manifest = {
        "status": "frozen_reference_replay_passed_pending_teacher_dataset_collection",
        "selected_reference": {"sha256": reference.sha256},
        "stage_boundaries_control_step": {
            "approach": [0, 0], "front_lift": [1, 1], "front_support": [2, 2],
            "first_rear": [3, 3], "second_rear": [4, 4], "rear_hold": [5, 5],
        },
    }
    manifest_path = tmp_path / "selected.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    Term = type("JointPositionAction", (), {})
    term = Term()
    term._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
    term._offset = torch.tensor([[0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4])
    collector = dataset.PhaseResidualDatasetCollector(
        output_dir=tmp_path / "dataset",
        selected_manifest_path=manifest_path,
        selected_manifest_sha256=_sha(manifest_path),
        reference=reference,
        action_term=term,
        joint_names=runtime.EXPECTED_JOINT_ORDER,
        num_envs=15,
    )
    return dataset, collector


def test_phase_features_are_contiguous_and_monotonic(tmp_path):
    dataset, collector = _fixture(tmp_path)
    features0, stage0 = collector.phase.features(0, device=torch.device("cpu"), dtype=torch.float32)
    features4, stage4 = collector.phase.features(4, device=torch.device("cpu"), dtype=torch.float32)
    features9, stage9 = collector.phase.features(9, device=torch.device("cpu"), dtype=torch.float32)
    assert stage0 == 0 and stage4 == 4 and stage9 == 5
    assert features0[1].item() == 1.0
    assert features4[5].item() == 1.0
    assert features9[6].item() == 1.0
    assert features0[0] <= features4[0] <= features9[0]


def test_collects_exact_shapes_and_whole_rollout_split(tmp_path):
    _, collector = _fixture(tmp_path)
    obs = torch.arange(15 * 570, dtype=torch.float32).reshape(15, 570)
    teacher = torch.tensor([[0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4]).repeat(15, 1)
    for step in range(2):
        student_obs = collector.prepare_observation(obs, step)
        if step == 0:
            assert torch.count_nonzero(student_obs[:, 410:]) == 0
        collector.record_after_step(
            teacher_mapped_target=teacher,
            joint_pos=teacher,
            joint_vel=torch.zeros_like(teacher),
            dones=torch.zeros(15, dtype=torch.bool),
        )
    result = collector.finalize(
        metadata={"test": True},
        env_hold_success=[True] * 15,
        env_full_climb_success=[True] * 15,
    )
    assert result["valid"] is True
    arrays = np.load(result["dataset_path"])
    assert arrays["student_obs_570"].shape == (30, 570)
    assert arrays["phase_features_8"].shape == (30, 8)
    assert set(arrays["episode_id"][arrays["split"] == 0]) == set(range(12))
    assert set(arrays["episode_id"][arrays["split"] == 1]) == {12, 13, 14}


def test_extracts_exact_policy_tensor_from_mapping_observation(tmp_path):
    _, collector = _fixture(tmp_path)
    obs = torch.zeros(15, 570)
    extracted = collector.prepare_observation({"policy": obs, "critic": torch.zeros(15, 1)}, 0)
    assert extracted.shape == (15, 570)


def test_settle_primes_safe_action_history_without_writing_samples(tmp_path):
    _, collector = _fixture(tmp_path)
    target = torch.tensor(
        [[0.1] * 4 + [0.8] * 4 + [-1.4] * 4 + [0.04] * 4]
    ).repeat(15, 1)
    for _ in range(10):
        collector.prime_action_history(target)
    obs = collector.prepare_observation(torch.zeros(15, 570), 0)
    assert torch.count_nonzero(obs[:, 410:]) > 0
    assert not collector._records["step"]


def test_any_termination_fails_dataset_closed(tmp_path):
    _, collector = _fixture(tmp_path)
    obs = torch.zeros(15, 570)
    target = torch.tensor([[0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4]).repeat(15, 1)
    collector.prepare_observation(obs, 0)
    dones = torch.zeros(15, dtype=torch.bool)
    dones[3] = True
    collector.record_after_step(
        teacher_mapped_target=target,
        joint_pos=target,
        joint_vel=torch.zeros_like(target),
        dones=dones,
    )
    result = collector.finalize(
        metadata={},
        env_hold_success=[True] * 15,
        env_full_climb_success=[True] * 15,
    )
    assert result["valid"] is False
    assert result["terminated_env_ids"] == [3]
