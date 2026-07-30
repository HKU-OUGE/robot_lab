from __future__ import annotations

import hashlib
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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dtlog_binds_active_attempt_and_current_ab_authority(monkeypatch, tmp_path):
    module = load("highstep_diagonal_7400_log_test", ROOT / "tools/highstep_diagonal_7400_log.py")
    workflow = "highstep_b300_critical_transition_balanced_diagonal_fresh_7400_20260719"
    spec_path = tmp_path / "spec.md"
    spec_path.write_text("frozen spec\n", encoding="utf-8")
    prereg_path = tmp_path / "preregistration.json"
    prereg = {
        "workflow_id": workflow,
        "authority": {"spec_sha256": digest(spec_path)},
        "critical_transition_sampling": {
            "window_radius_policy_steps": 10,
            "minibatch_source_fractions": {
                "front_transition": 0.25,
                "rear_transition": 0.25,
                "original_distribution": 0.5,
            },
        },
        "phase_diagonal_loss": {
            "front_indices": [1, 5, 9],
            "rear_indices": [2, 6, 10],
            "front_gate_stages": ["front_lift", "front_support"],
            "rear_gate_stages": ["first_rear", "second_rear"],
            "gated_total_per_element_weight_ratio": 3.0,
        },
        "distillation_contract": {
            "student_vae_epochs": 4,
            "phase_scale": 2.0,
            "rear_box_scale": 1.5,
        },
    }
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    state_path = tmp_path / "state.json"
    active_log = tmp_path / "train_attempt2.log"
    active_log.write_text(
        " Learning iteration 173510/180899 \n"
        "Mean Debug/Student_Distill_Update_Count loss: 11.0000\n"
        "Mean Loss/Teacher_Action_MSE loss: 0.3\n"
        "Time elapsed: 00:01:00\n"
        "ETA: 06:00:00\n",
        encoding="utf-8",
    )
    (tmp_path / "train.log").write_text(
        " Learning iteration 173500/180899 \n"
        "Mean Debug/Student_Distill_Update_Count loss: 1.0000\n"
        "Time elapsed: 00:00:10\nETA: 07:00:00\n",
        encoding="utf-8",
    )
    heartbeat = tmp_path / "heartbeat.json"
    heartbeat.write_text("{}\n", encoding="utf-8")
    state = {
        "workflow_id": workflow,
        "spec_path": str(spec_path),
        "spec_sha256": digest(spec_path),
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": digest(prereg_path),
        "active_pid": 12345,
        "supervisor_pid": 12346,
        "heartbeat_path": str(heartbeat),
        "relative_distill_update": 12,
        "absolute_runner_step_origin": 173499,
        "max_effective_updates": 7400,
        "wandb_online_initialized": True,
        "wandb_run_name": "route_attempt2_from_173499",
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    declaration = {
        "workflow_id": workflow,
        "state_path": str(state_path),
        "heartbeat_path": str(heartbeat),
        "spec_path": str(spec_path),
        "spec_sha256": digest(spec_path),
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": digest(prereg_path),
    }
    dashboard = tmp_path / "dashboard.json"
    dashboard.write_text(json.dumps(declaration), encoding="utf-8")

    monkeypatch.setattr(module, "DASHBOARD", dashboard)
    monkeypatch.setattr(module, "pid_alive", lambda _value: True)
    monkeypatch.setattr(
        module.os,
        "readlink",
        lambda path: str(active_log) if path == "/proc/12345/fd/1" else (_ for _ in ()).throw(OSError()),
    )

    rendered = module.render(False)

    assert "B300 critical-transition A+B fresh 7400" in rendered
    assert "Authority OK" in rendered
    assert "E12/7400" in rendered
    assert "同步=正常(state差+0)" in rendered
    assert "Critical sampling=0.25/0.25/0.5" in rendered
    assert "FR joints=[1, 5, 9]" in rendered
    assert "RL joints=[2, 6, 10]" in rendered
    assert str(active_log) in rendered
    assert "Diagonal scale=N/A" not in rendered


def test_dtlog_attempt_name_fallback_does_not_select_stale_train_log(tmp_path):
    module = load("highstep_diagonal_7400_log_fallback_test", ROOT / "tools/highstep_diagonal_7400_log.py")
    state_path = tmp_path / "state.json"
    stale = tmp_path / "train.log"
    active = tmp_path / "train_attempt2.log"
    stale.write_text("old\n", encoding="utf-8")
    active.write_text("new\n", encoding="utf-8")

    selected = module.resolve_train_log(
        state_path,
        {"active_pid": None, "wandb_run_name": "route_attempt2_from_173499"},
    )

    assert selected == active


def test_dtlog_keeps_continuation_runner_and_effective_clocks_independent():
    module = load("highstep_diagonal_7400_log_clock_test", ROOT / "tools/highstep_diagonal_7400_log.py")

    state_effective, log_effective = module.effective_clocks(
        {"effective_updates": 6381, "absolute_runner_step": 181337},
        {
            "absolute": 181337,
            "Debug/Student_Distill_Update_Count": 6381.0,
        },
    )

    assert (state_effective, log_effective) == (6381, 6381)
    assert module.target_effective_updates(
        {"target_effective_updates": 7700},
        {"training_budget": {"target_effective_update": 7700}},
    ) == 7700


def test_dtlog_retains_fresh_run_zero_based_count_compatibility():
    module = load("highstep_diagonal_7400_log_fresh_clock_test", ROOT / "tools/highstep_diagonal_7400_log.py")

    assert module.effective_clocks(
        {"relative_distill_update": 12},
        {"Debug/Student_Distill_Update_Count": 11.0},
    ) == (12, 12)


def test_dtlog_estimates_remaining_time_from_effective_updates_not_runner_steps():
    module = load("highstep_diagonal_7400_log_eta_test", ROOT / "tools/highstep_diagonal_7400_log.py")

    estimate = module.remaining_time_estimate(
        {"start_effective_updates": 5700, "absolute_runner_step": 181337},
        {"Time elapsed": "02:00:00", "absolute": 181337},
        current_effective=6300,
        target_effective=7700,
    )

    assert estimate == (16800, 5.0)
    assert module.format_duration(16800) == "04:40:00"
