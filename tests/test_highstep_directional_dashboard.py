from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def test_continuous_workflow_latest_log_is_exposed_as_train_log(tmp_path):
    module = load("dashboard_state_latest_log_test", ROOT / "tmp/highstep_dashboard_state.py")
    workflow = "highstep_be300_0707_distill_20260716"
    state_path = tmp_path / "tmp" / workflow / "state.json"
    state_path.parent.mkdir(parents=True)
    log_path = state_path.parent / "logs/formal_E4000.log"
    log_path.parent.mkdir()
    log_path.write_text("Learning iteration 167/4000\n", encoding="utf-8")
    state_path.write_text(json.dumps({
        "workflow_id": workflow,
        "status": "running",
        "phase": "formal_E4000_continuous_training",
        "active_pid": 1,
        "latest_log": str(log_path),
    }))
    (tmp_path / module.ACTIVE_WORKFLOW_DECLARATION).write_text(json.dumps({
        "schema_version": 1,
        "workflow_id": workflow,
        "state_path": str(state_path),
    }))

    _, selected = module.select_workflow_state(tmp_path)
    assert selected["train_log"] == str(log_path)
    assert module.field_value("train-log", tmp_path) == str(log_path)


def test_be300_continuous_tlog_uses_student_metrics_and_manual_review(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("be300_continuous_tlog_test", ROOT / "tmp/highstep_train_dashboard.py")
    log_path = tmp_path / "formal_E4000.log"
    log_path.write_text(
        " Learning iteration 204/4000 \n"
        "Mean Debug/Student_Distill_Update_Count loss: 204.0000\n"
        "Mean Debug/Student_Warmup_Updates loss: 1400.0000\n"
        "Mean Loss/Teacher_Action_MSE loss: 0.1980\n"
        "Time elapsed: 00:19:15\n"
        "ETA: 05:56:35\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "select_workflow_state", lambda: (None, {
        "workflow_id": "highstep_be300_0707_distill_20260716",
        "status": "running",
        "phase": "formal_E4000_continuous_training",
        "train_pid": 123,
        "train_log": str(log_path),
        "run_dir": str(tmp_path),
    }))
    monkeypatch.setattr(module, "pid_alive", lambda _value: True)

    rendered = module.render(False)
    assert "204/4000" in rendered
    assert "Distill updates" in rendered
    assert "Pre-prior target MSE" in rendered
    assert "manual play" in rendered
    assert "27-run play matrix" not in rendered


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


def test_b300_hybrid_dashboard_renders_training_without_legacy_v15(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("dashboard_b300_training_test", ROOT / "tmp/highstep_status_dashboard.py")
    root = tmp_path / "highstep_b300_canonical_hybrid_prior_latent_20260718"
    stage = root / "stages/E300"
    stage.mkdir(parents=True)
    (stage / "train.log").write_text(
        "Learning iteration 124/299\nETA: 00:15:00\n", encoding="utf-8"
    )
    prereg = root / "preregistration.json"
    prereg.write_text(json.dumps({
        "centerline": {
            "evaluation_runs": 15,
            "required": {"valid": 15, "full_climb": 12, "rear_hold": 12, "safe": 15},
        }
    }))
    state = {
        "workflow_id": "highstep_b300_canonical_hybrid_prior_latent_20260718",
        "status": "running", "phase": "training_E300", "supervisor_pid": 10,
        "active_pid": 11, "service": "highstep-b300-canonical-hybrid.service",
        "updated_at_epoch": 100.0, "preregistration_path": str(prereg),
        "centerline_counts": {"valid": 15, "full_climb": 0, "rear_hold": 0, "safe": 15},
    }
    monkeypatch.setattr(module, "pid_alive", lambda value: value in {10, 11})
    monkeypatch.setattr(
        module, "systemd_unit_status",
        lambda _unit: {"ActiveState": "active", "SubState": "running", "NRestarts": "0"},
    )
    monkeypatch.setattr(module.time, "time", lambda: 105.0)
    rendered = module.render_b300_canonical_hybrid(root / "state.json", state, False)
    assert "Highstep B300 Canonical Hybrid" in rendered
    assert "Training E300" in rendered
    assert "124/299" in rendered
    assert "Latest completed centerline: valid 15/15  full 0/15" in rendered
    assert "Student Recovery v1.5" not in rendered
    assert "imitation" not in rendered
    assert "core9" not in rendered


def test_b300_hybrid_dashboard_parses_live_centerline_counts(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "tmp"))
    module = load("dashboard_b300_eval_test", ROOT / "tmp/highstep_status_dashboard.py")
    root = tmp_path / "highstep_b300_canonical_hybrid_prior_latent_20260718"
    eval_dir = root / "stages/E300/centerline15"
    eval_dir.mkdir(parents=True)
    prereg = root / "preregistration.json"
    prereg.write_text(json.dumps({
        "centerline": {
            "evaluation_runs": 15,
            "required": {"valid": 15, "full_climb": 12, "rear_hold": 12, "safe": 15},
        }
    }))
    base = {
        "reset_valid": True, "schedule_valid": True,
        "target_action_order_valid": True, "target_limit_action_input_valid": True,
        "target_limit_invalid_action_steps": 0, "target_limit_violation_fraction": 0.0,
        "samples": 138, "full_climb_success": False,
        "rear_on_platform_hold_success": False,
    }
    (eval_dir / "run01_attempt1.log").write_text(
        "[HIGHSTEP_EVAL_JSON] " + json.dumps(base) + "\n", encoding="utf-8"
    )
    passed = dict(base, full_climb_success=True, rear_on_platform_hold_success=True)
    (eval_dir / "run02_attempt1.log").write_text(
        "[HIGHSTEP_EVAL_JSON] " + json.dumps(passed) + "\n", encoding="utf-8"
    )
    state = {
        "workflow_id": "highstep_b300_canonical_hybrid_prior_latent_20260718",
        "status": "evaluating", "phase": "E300_centerline_run03_attempt1",
        "supervisor_pid": 10, "active_pid": 11,
        "updated_at_epoch": 100.0, "preregistration_path": str(prereg),
    }
    monkeypatch.setattr(module, "pid_alive", lambda value: value in {10, 11})
    monkeypatch.setattr(module, "systemd_unit_status", lambda _unit: {
        "ActiveState": "active", "SubState": "running", "NRestarts": "0"
    })
    monkeypatch.setattr(module.time, "time", lambda: 105.0)
    rendered = module.render_b300_canonical_hybrid(root / "state.json", state, False)
    assert "Centerline E300" in rendered
    assert "2/15" in rendered
    assert "current run 03" in rendered
    assert "valid 2/15  full 1/15  rear-hold 1/15  safe 2/15" in rendered
