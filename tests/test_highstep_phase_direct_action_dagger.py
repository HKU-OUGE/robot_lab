from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


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


def test_direct_action_dagger_uses_only_its_narrow_authority_profile(tmp_path):
    runtime = _load(BASE / "highstep_phase_residual_runtime.py", "highstep_phase_residual_runtime")
    _load(BASE / "highstep_phase_residual_dataset.py", "highstep_phase_residual_dataset")
    dagger = _load(BASE / "highstep_phase_residual_dagger.py", "direct_action_dagger_test")
    training = {
        "kind": "highstep_fixed_condition_phase_direct_action_training",
        "workflow_id": dagger.DIRECT_WORKFLOW_ID,
        "checkpoint_sha256": "direct-checkpoint",
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    prereg = {
        "kind": "highstep_fixed_condition_phase_direct_action_dagger_preregistration",
        "status": "frozen_before_dagger_round1_collection",
        "workflow_id": dagger.DIRECT_WORKFLOW_ID,
        "round": 1,
        "driver_mode": "phase_direct_action",
        "teacher_label": "same_pre_step_state_post_prior_mapped_target",
        "post_termination_samples": "recorded_but_excluded_from_training",
        "parent_training_manifest_sha256": _sha(training_path),
        "parent_checkpoint_sha256": "direct-checkpoint",
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
        driver_mode="phase_direct_action",
        authority_profile="direct_action",
    )
    assert collector.workflow_id == dagger.DIRECT_WORKFLOW_ID
    assert collector.dataset_kind == "highstep_fixed_condition_phase_direct_action_dagger_dataset"

