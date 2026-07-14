from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    path = ROOT / "tools/highstep_wandb_stage_gate.py"
    spec = importlib.util.spec_from_file_location("highstep_wandb_stage_gate_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(tmp_path: Path) -> dict:
    start = tmp_path / "start.pt"; teacher = tmp_path / "teacher.pt"
    start.write_bytes(b"start"); teacher.write_bytes(b"teacher")
    workflow = "wf_directional"
    return {
        "workflow_id": workflow, "route": "R5", "stage": "R5_3", "attempt": "attempt1",
        "run_name": f"{workflow}_R5_R5_3_attempt1", "group": workflow,
        "spec_sha256": "1" * 64, "preregistration_sha256": "2" * 64,
        "start_checkpoint": str(start), "start_checkpoint_sha256": sha(start),
        "teacher_checkpoint": str(teacher), "teacher_sha256": sha(teacher),
        "frozen_tensors": "all other", "trainable_tensors": ["actor.4.weight"],
        "optimizer": {"class": "Adam", "lr": 1e-6}, "budget": {"epochs": 2},
        "training_task": "fixed_replay", "historical_sync": False,
    }


def test_contract_requires_independent_named_grouped_run(tmp_path):
    gate = load_gate(); value = contract(tmp_path)
    assert gate.validate_contract(value)["group"] == value["workflow_id"]
    value["run_name"] = "shared"
    with pytest.raises(RuntimeError, match="does not contain"):
        gate.validate_contract(value)


@pytest.mark.parametrize("route", ["B", "R", "R5", "student_environment_curriculum"])
def test_gate_accepts_every_authorized_custom_training_route(tmp_path, route):
    gate = load_gate(); value = contract(tmp_path)
    value["route"] = route
    value["run_name"] = f"{value['workflow_id']}_{route}_{value['stage']}_{value['attempt']}"
    assert gate.validate_contract(value)["route"] == route


def test_pending_sync_blocks_next_stage(tmp_path):
    gate = load_gate(); stage = tmp_path / "wandb_stages/a"; stage.mkdir(parents=True)
    (stage / "stage_manifest.json").write_text(json.dumps({"sync_status": "pending_sync"}))
    with pytest.raises(RuntimeError, match="blocks the next"):
        gate.assert_all_prior_synced(tmp_path)


def test_r5_path_prepares_and_finalizes_wandb_stage():
    supervisor = (ROOT / "tools/highstep_directional_trial_supervisor.py").read_text()
    trainer = (ROOT / "tools/highstep_v13_r5_train.py").read_text()
    assert "prepare_stage(wandb_config, STATE_ROOT)" in supervisor
    assert "finalize_r5_wandb" in supervisor
    assert 'sync_status"] != "synced"' in supervisor
    assert "HIGHSTEP_WANDB_STAGE_MANIFEST" in trainer
    assert "wandb_run.log" in trainer


def test_amendment_preserves_old_evidence_sha_and_adds_wandb_gate():
    spec = (ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md").read_text()
    assert "v1.4.1 amendment" in spec
    assert "1fbb17ab6c23a24005e8e34dadc11a6e4592a7be671a575206fd24819e5f3f1c" in spec
    assert "pending_sync" in spec and "historical_sync=true" in spec
