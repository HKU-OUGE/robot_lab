"""CPU-only contracts for the approved v1.3 fixed-replay R4 route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools import highstep_v13_r4_fit as fit
from tools import highstep_same_state_audit as audit


ROOT = Path("/home/lxq/Softwares/robot_lab")
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
PREREG = ROOT / "tmp/highstep_student_recovery_v13_20260713/r4_preregistration.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v13_authority_and_matrix_are_immutable_and_exact():
    payload = json.loads(PREREG.read_text())
    assert sha256(SPEC) == "e9375189896e2f6a23b1b8018102c39813076dd048ec8242346b175bb1bc2ef4"
    assert payload["authority"]["spec_sha256"] == "a1fc5e1283606b69bec7b7256c20abf53003601e9a127d303294a160cbb57223"
    assert "v1.3 仍永久记录为双侧鲁棒目标 `stopped_by_gate`" in SPEC.read_text()
    assert "historical-only" in SPEC.read_text()
    assert sha256(PREREG) == fit.PREREGISTRATION_SHA256
    assert PREREG.stat().st_mode & 0o222 == 0
    assert len(payload["core9_matrix"]) == 9
    assert sum(row["b500_role"] == "success_anchor" for row in payload["core9_matrix"]) == 6
    assert sum(row["b500_role"] == "failure_teacher" for row in payload["core9_matrix"]) == 3
    assert payload["permanent_contract"]["action_scale_unchanged"] is True
    assert payload["permanent_contract"]["joint_pos_clip_unchanged"] is True


def test_weighted_ridge_is_centered_on_anchor_and_finite():
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(64, 128, generator=generator, dtype=torch.float64)
    theta0 = torch.randn(129, 16, generator=generator, dtype=torch.float64)
    design = torch.cat((hidden, torch.ones(64, 1, dtype=torch.float64)), dim=1)
    target = design @ theta0
    weights = torch.ones(64, dtype=torch.float64)
    solved = fit._weighted_ridge(hidden, target, weights, theta0, 1.0)
    torch.testing.assert_close(solved, theta0, rtol=1.0e-9, atol=1.0e-9)


def test_offline_gate_requires_preservation_improvement_and_q99():
    anchor = torch.zeros(10, 16, dtype=torch.float64)
    teacher = torch.zeros_like(anchor)
    teacher[5:] = 0.1
    success = torch.tensor([True] * 5 + [False] * 5)
    data = {"anchor": anchor, "teacher": teacher, "success": success}
    candidate = anchor.clone()
    candidate[5:] = 0.011
    passed = fit._statistics(data, candidate)
    assert passed["offline_gate_passed"] is True
    candidate[:5] = 0.003
    failed = fit._statistics(data, candidate)
    assert failed["offline_gate_passed"] is False


def test_cross_device_anchor_reconstruction_tolerance_is_integrity_only():
    anchor = torch.zeros(2, 16)
    reconstructed = anchor.clone()
    reconstructed[0, 0] = 1.4e-5
    audit_row = fit._audit_anchor_reconstruction(reconstructed, anchor, "gpu_cpu_roundoff")
    assert audit_row["max_abs"] < fit.ANCHOR_RECONSTRUCTION_ATOL
    reconstructed[0, 0] = 2.1e-5
    with pytest.raises(RuntimeError, match="action reconstruction failed"):
        fit._audit_anchor_reconstruction(reconstructed, anchor, "real_mismatch")


def test_v13_wrapper_registers_every_preregistered_core9_identity():
    wrapper_path = ROOT / "scripts/rsl_rl/base/highstep_v13_core9_trace_play.py"
    module_spec = importlib.util.spec_from_file_location("test_v13_trace_wrapper", wrapper_path)
    assert module_spec is not None and module_spec.loader is not None
    wrapper = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = wrapper
    module_spec.loader.exec_module(wrapper)
    payload = json.loads(PREREG.read_text())

    class FakeRuntime:
        Trajectory = __import__("collections").namedtuple(
            "Trajectory", "run_id seed scenario lateral_offset_m yaw_offset_deg"
        )
        TRAJECTORIES = {}

        @staticmethod
        def _runtime_code_hashes():
            return {}

        @staticmethod
        def _ensure_runtime_binding(output_root, suite_plan_path, suite_plan_sha256):
            return suite_plan_path, suite_plan_sha256

    original_trajectories = audit.DEFAULT_AUDIT_TRAJECTORIES
    original_plan = audit.base_suite_plan_payload
    try:
        wrapper.install_v13_matrix(FakeRuntime, payload)
        for row in payload["core9_matrix"]:
            audit.AuditPreregistration(
                run_id=f"seed{row['seed']}_{row['scenario']}",
                seed=row["seed"],
                scenario=row["scenario"],
                suite_plan_path="/tmp/v13-core9-plan.json",
                suite_plan_sha256="0" * 64,
                lateral_offset_m=row["lateral_offset_m"],
                yaw_offset_deg=row["yaw_offset_deg"],
            )
    finally:
        audit.DEFAULT_AUDIT_TRAJECTORIES = original_trajectories
        audit.base_suite_plan_payload = original_plan


def test_deterministic_trace_contract_failure_is_not_retried(tmp_path):
    log = tmp_path / "trace_attempts/seed11_right_offset_attempt/trace.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "ValueError: base3 preregistration is not one of the fixed three trajectories\n"
    )
    script = f"""
from pathlib import Path
from types import SimpleNamespace
from tools import highstep_student_recovery_v13_supervisor as supervisor
supervisor.STATE_ROOT = Path({str(tmp_path)!r})
fake = SimpleNamespace(state={{'phase': 'trace_seed11_right_offset'}})
raise SystemExit(0 if supervisor._is_transient(fake, RuntimeError('phase exited return code 1')) is False else 1)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=False)
    assert result.returncode == 0


def test_deterministic_r4_child_traceback_is_not_retried(tmp_path):
    log = tmp_path / "r4_fit_attempts/attempt.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Traceback (most recent call last):\n"
        "RuntimeError: R4 B500 action reconstruction failed: seed11_nominal\n"
    )
    script = f"""
from pathlib import Path
from types import SimpleNamespace
from tools import highstep_student_recovery_v13_supervisor as supervisor
supervisor.STATE_ROOT = Path({str(tmp_path)!r})
fake = SimpleNamespace(state={{'phase': 'r4_fixed_replay_fit'}})
raise SystemExit(0 if supervisor._is_transient(fake, RuntimeError('phase exited return code 1')) is False else 1)
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=False)
    assert result.returncode == 0


def test_r5_entrypoint_imports_tools_from_outside_repository(tmp_path):
    script = ROOT / "tools/highstep_v13_r5_train.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert "dataset-manifest" in result.stdout
