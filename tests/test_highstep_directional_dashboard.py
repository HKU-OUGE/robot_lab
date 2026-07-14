from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_state_selector_includes_live_directional_workflow(tmp_path):
    module = load("dashboard_state_directional_test", ROOT / "tmp/highstep_dashboard_state.py")
    state_dir = tmp_path / "tmp/highstep_directional_real_robot_trial_20260713"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(json.dumps({
        "workflow_id": "highstep_directional_real_robot_trial_20260713",
        "status": "running", "phase": "trace_seed11_nominal", "supervisor_pid": 1,
    }))
    candidates = module.workflow_state_candidates(tmp_path)
    assert any(item[1]["workflow_id"].startswith("highstep_directional") for item in candidates)


def test_explicit_current_workflow_beats_old_service_mtime(tmp_path):
    module = load("dashboard_state_explicit_authority_test", ROOT / "tmp/highstep_dashboard_state.py")
    current = tmp_path / "tmp/highstep_0707_exact_new_teacher_20260713/state.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({
        "workflow_id": "highstep_0707_exact_new_teacher_20260713",
        "status": "mechanism_error_requires_repair", "effective_updates": 300,
    }))
    old = tmp_path / "tmp/highstep_student_recovery_v15_20260713/state.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({
        "workflow_id": "highstep_student_recovery_v15_20260713",
        "status": "running", "phase": "training_E300", "supervisor_pid": 1,
    }))
    (tmp_path / module.ACTIVE_WORKFLOW_DECLARATION).write_text(json.dumps({
        "schema_version": 1,
        "workflow_id": "highstep_0707_exact_new_teacher_20260713",
        "state_path": str(current),
    }))
    selected_path, selected = module.select_workflow_state(tmp_path)
    assert selected_path == current.resolve()
    assert selected["workflow_id"] == "highstep_0707_exact_new_teacher_20260713"


def test_v171_stage_semantics_are_explicit():
    module = load("dashboard_state_v171_semantics", ROOT / "tmp/highstep_dashboard_state.py")
    assert module.stage_semantics("training_300_attempt1") == (
        "training", "child_required_while_running"
    )
    assert module.stage_semantics("core9_E100_seed11") == (
        "evaluation", "child_expected_only_during_atomic_rollout"
    )
    assert module.stage_semantics("wandb_sync_E100") == (
        "infrastructure", "no_gpu_child_expected"
    )
    assert module.stage_semantics("E100_complete_adopted") == (
        "transition", "no_child_expected_at_safe_boundary"
    )


def test_status_dashboard_has_directional_9_run_contract():
    source = (ROOT / "tmp/highstep_status_dashboard.py").read_text()
    assert "Behavior gate:" in source
    assert "{behavior_done}/9" in source
    assert "Training: not running (EXPECTED" in source
    assert "Teacher-action regression:" in source


def test_status_dashboard_prioritizes_live_v15_reference_baseline(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("dashboard_v15_reference_test", ROOT / "tmp/highstep_status_dashboard.py")
    workflow = tmp_path / "highstep_student_recovery_v15_20260713"
    reference = workflow / "0707_current_rule_core9_v2"
    reference.mkdir(parents=True)
    (workflow / "state.json").write_text(json.dumps({
        "workflow_id": "highstep_student_recovery_v15_20260713",
        "status": "external_fault_needs_user",
        "effective_updates": 300,
        "checkpoint": "/tmp/model_298.pt",
    }))
    (reference / "heartbeat.json").write_text(json.dumps({
        "status": "running",
        "phase": "evaluating_seed11_nominal",
        "pid": 123,
        "active_pid": 124,
        "completed": 0,
        "total": 9,
        "timestamp": "2099-01-01T00:00:00+0000",
    }))
    monkeypatch.setattr(module, "V15_ROOT", workflow)
    monkeypatch.setattr(module, "pid_alive", lambda value: value in {123, 124})
    monkeypatch.setattr(
        module,
        "systemd_unit_status",
        lambda _unit: {"ActiveState": "activating", "SubState": "start"},
    )
    selected = module.active_v15_reference_baseline()
    assert selected is not None
    rendered = module.render_v15_reference_baseline(*selected, False)
    assert "REFERENCE BASELINE RUNNING" in rendered
    assert "0/9" in rendered
    assert "intentionally paused at E300" in rendered
    assert "not an external fault" in rendered
