from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "rsl_rl" / "base"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dagger_records_terminal_state_but_excludes_post_reset_rows(tmp_path):
    runtime = _load(BASE / "highstep_phase_residual_runtime.py", "highstep_phase_residual_runtime")
    _load(BASE / "highstep_phase_residual_dataset.py", "highstep_phase_residual_dataset")
    dagger = _load(BASE / "highstep_phase_residual_dagger.py", "phase_residual_dagger_test")
    training = {
        "kind": "highstep_fixed_condition_phase_residual_training",
        "workflow_id": dagger.WORKFLOW_ID,
        "checkpoint_sha256": "checkpoint-sha",
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    prereg = {
        "kind": "highstep_fixed_condition_phase_residual_dagger_preregistration",
        "status": "frozen_before_dagger_round1_collection",
        "workflow_id": dagger.WORKFLOW_ID,
        "round": 1,
        "driver_mode": "reference_only",
        "teacher_label": "same_pre_step_state_post_prior_mapped_target",
        "post_termination_samples": "recorded_but_excluded_from_training",
        "parent_training_manifest_sha256": _sha(training_path),
        "parent_checkpoint_sha256": "checkpoint-sha",
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    collector = dagger.PhaseResidualDaggerCollector(
        output_dir=tmp_path / "out",
        preregistration_path=prereg_path,
        preregistration_sha256=_sha(prereg_path),
        training_manifest_path=training_path,
        training_manifest_sha256=_sha(training_path),
        joint_names=runtime.EXPECTED_JOINT_ORDER,
        num_envs=15,
        driver_mode="reference_only",
    )
    teacher = torch.tensor(
        [[0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4]
    ).repeat(15, 1)
    phase = torch.zeros(15, 8)
    phase[:, 1] = 1.0
    for step in range(120):
        sample = {
            "student_obs_570": torch.zeros(15, 570),
            "phase_features_8": phase,
            "reference_target_16": teacher,
            "candidate_residual_16": torch.zeros(15, 16),
            "candidate_target_16": teacher,
            "stage": min(step // 20, 5),
            "step": step,
        }
        collector.prepare_before_step(
            sample=sample,
            teacher_mapped_target=teacher,
            joint_pos=teacher,
            joint_vel=torch.zeros(15, 16),
        )
        dones = torch.zeros(15, dtype=torch.bool)
        if step == 2:
            dones[0] = True
        collector.record_after_step(dones)
    result = collector.finalize(metadata={"test": True})
    assert result["valid_rows"] == 14 * 120 + 3
    assert result["invalid_post_termination_rows"] == 117
    assert result["valid_rows_by_env"]["0"] == 3
    archive = np.load(result["dataset_path"], allow_pickle=False)
    env0 = archive["episode_id"] == 0
    assert int(np.sum(archive["sample_valid"][env0])) == 3
    assert int(np.sum(archive["terminated_after_step"][env0])) == 1


def test_dagger_fails_closed_on_parent_manifest_sha(tmp_path):
    runtime = _load(BASE / "highstep_phase_residual_runtime.py", "highstep_phase_residual_runtime")
    _load(BASE / "highstep_phase_residual_dataset.py", "highstep_phase_residual_dataset")
    dagger = _load(BASE / "highstep_phase_residual_dagger.py", "phase_residual_dagger_sha_test")
    training_path = tmp_path / "training.json"
    training_path.write_text("{}", encoding="utf-8")
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}", encoding="utf-8")
    try:
        dagger.PhaseResidualDaggerCollector(
            output_dir=tmp_path / "out",
            preregistration_path=prereg_path,
            preregistration_sha256="0" * 64,
            training_manifest_path=training_path,
            training_manifest_sha256=_sha(training_path),
            joint_names=runtime.EXPECTED_JOINT_ORDER,
            num_envs=15,
            driver_mode="reference_only",
        )
    except RuntimeError as error:
        assert "SHA mismatch" in str(error)
    else:
        raise AssertionError("wrong preregistration SHA must fail closed")

