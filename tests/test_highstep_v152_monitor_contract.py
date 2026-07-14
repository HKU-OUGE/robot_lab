"""Regression tests for the v1.5.2 train-to-core9 authority boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
SUPERVISOR = ROOT / "tools/highstep_student_recovery_v152_supervisor.py"
RECOVERY = ROOT / "tools/highstep_v152_recover_core9.py"
TASK_COMPAT = ROOT / "tools/highstep_core9_task_compat.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monitor_alias_is_exact_and_workflow_scoped() -> None:
    source = MONITOR.read_text()
    assert "from tools.highstep_core9_task_compat import source_authority_is_valid" in source
    assert source.count("and source_authority_valid") == 1


def test_exact_0707_and_v152_aliases_are_narrow_and_fail_closed() -> None:
    module = _load("highstep_core9_task_compat_test", TASK_COMPAT)
    common = {
        "role": "student",
        "eval_task": module.ROBUST_STUDENT_TASK,
        "checkpoint_sha_valid": True,
        "lineage_valid": True,
    }
    assert module.source_authority_is_valid(
        workflow_id="highstep_0707_exact_new_teacher_20260713",
        source_task=module.EXACT_0707_SOURCE_TASK,
        **common,
    )
    assert module.source_authority_is_valid(
        workflow_id="highstep_student_recovery_v152_20260713",
        source_task=module.V152_SOURCE_TASK,
        **common,
    )
    assert not module.source_authority_is_valid(
        workflow_id="wrong_workflow",
        source_task=module.EXACT_0707_SOURCE_TASK,
        **common,
    )
    assert not module.source_authority_is_valid(
        workflow_id="highstep_0707_exact_new_teacher_20260713",
        source_task=module.V152_SOURCE_TASK,
        **common,
    )
    assert not module.source_authority_is_valid(
        workflow_id="highstep_0707_exact_new_teacher_20260713",
        source_task=module.EXACT_0707_SOURCE_TASK,
        **{**common, "checkpoint_sha_valid": False},
    )
    assert not module.source_authority_is_valid(
        workflow_id="highstep_0707_exact_new_teacher_20260713",
        source_task=module.EXACT_0707_SOURCE_TASK,
        **{**common, "lineage_valid": False},
    )


def test_recovery_executes_same_embedded_fail_closed_aggregator() -> None:
    recovery = _load("highstep_v152_recovery_test", RECOVERY)
    aggregate = recovery.aggregate_source(MONITOR)
    assert "student_source_lineage_valid" in aggregate
    assert "runtime_snapshot_checkpoint_sha256_verified" in aggregate
    assert 'all(summary["valid_count"] == expected_per_checkpoint' in aggregate


def test_auxiliary_failure_cannot_stop_before_absolute_cap() -> None:
    module = _load("highstep_v152_supervisor_test", SUPERVISOR)
    supervisor = object.__new__(module.Supervisor)
    imitation = {
        "passed": False,
        "failed_element_count": 999,
        "phase_metrics": {"approach": {"latent_mae": 99.0}},
    }
    probe = {
        "counts": {"valid": 0}, "behavior_score": 0, "support_score": 0,
        "catastrophic_validity_passed": False,
    }
    decision = supervisor.decision(500, imitation, probe, None, None)
    assert decision["stop"] is False
    assert decision["classification"] == "behavior_not_yet_successful"


def test_core_rows_take_lateral_geometry_from_validated_ledger(tmp_path: Path) -> None:
    module = _load("highstep_v152_core_rows_test", SUPERVISOR)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "evaluation_manifest.json"
    manifest.write_text("{}")
    row = {
        "seed": 11, "scenario": "left_offset", "lateral": 0.12,
        "yaw_offset_deg": 4.0, "log": "evidence.log",
        "eval": {
            "yaw_offset_deg": 4.0, "full_climb_success": False,
            "rear_on_platform_hold_success": False, "front_top_support_reached": False,
            "critical_rear_min_abs_y_q05": 0.20, "critical_rear_width_q05": 0.45,
            "critical_center_violation_rate": 0.0, "critical_width_violation_rate": 0.0,
        },
    }
    (tmp_path / "eval_runs.jsonl").write_text(json.dumps(row) + "\n")
    supervisor = object.__new__(module.Supervisor)
    rows = supervisor._core_rows({"evaluation_manifest": str(manifest)}, checkpoint)
    assert rows[0]["lateral_offset_m"] == 0.12
    assert rows[0]["yaw_offset_deg"] == 4.0
