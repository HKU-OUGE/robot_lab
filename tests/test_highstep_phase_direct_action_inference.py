from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

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


def _fixture(tmp_path: Path):
    runtime = _load(BASE / "highstep_phase_residual_runtime.py", "highstep_phase_residual_runtime")
    _load(BASE / "highstep_phase_residual_dataset.py", "highstep_phase_residual_dataset")
    _load(BASE / "highstep_phase_residual_inference.py", "highstep_phase_residual_inference")
    inference = _load(
        BASE / "highstep_phase_direct_action_inference.py",
        "phase_direct_action_inference_test",
    )
    target_fields = [f"reference_target.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    measured_fields = [f"measured_joint_pos.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    values = [0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4
    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reference_step", "global_phase", *target_fields, *measured_fields],
        )
        writer.writeheader()
        for step in range(6):
            row = {"reference_step": step, "global_phase": step / 5}
            row.update(dict(zip(target_fields, values)))
            row.update(dict(zip(measured_fields, values)))
            writer.writerow(row)
    reference = runtime.PhaseResidualReference(reference_path, expected_sha256=_sha(reference_path))
    preroll_path = tmp_path / "preroll.csv"
    with preroll_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["preroll_step", *target_fields])
        writer.writeheader()
        for step in range(296):
            row = {"preroll_step": step}
            row.update(dict(zip(target_fields, values)))
            writer.writerow(row)
    preroll_manifest = {
        "kind": "highstep_fixed_condition_teacher_zero_command_preroll",
        "status": "frozen_preroll_ready_before_behavior_retry",
        "workflow_id": "highstep_fixed_condition_phase_residual_20260717",
        "selected_reference_sha256": reference.sha256,
        "source_timing_amendment_sha256": "timing-sha",
        "command": [0.0, 0.0, 0.0],
        "frame_count": 296,
        "projection": {"exported_target_limit_violations": 0},
        "preroll_path": str(preroll_path),
        "preroll_sha256": _sha(preroll_path),
    }
    preroll_manifest_path = tmp_path / "preroll_manifest.json"
    preroll_manifest_path.write_text(json.dumps(preroll_manifest), encoding="utf-8")
    residual_inference = sys.modules["highstep_phase_residual_inference"]
    preroll = residual_inference.PhaseResidualPreroll(
        preroll_manifest_path,
        _sha(preroll_manifest_path),
        expected_reference_sha256=reference.sha256,
        expected_timing_sha256="timing-sha",
    )
    selected = {
        "selected_reference": {"sha256": reference.sha256},
        "stage_boundaries_control_step": {
            "approach": [0, 0], "front_lift": [1, 1], "front_support": [2, 2],
            "first_rear": [3, 3], "second_rear": [4, 4], "rear_hold": [5, 5],
        },
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"frozen-teacher")
    spec_path = tmp_path / "spec.md"
    spec_path.write_text("frozen direct action spec\n", encoding="utf-8")
    prereg = {
        "kind": "highstep_fixed_condition_phase_direct_action_preregistration",
        "status": "frozen_before_training",
        "workflow_id": inference.WORKFLOW_ID,
        "spec_path": str(spec_path),
        "spec_sha256": _sha(spec_path),
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": _sha(teacher_path),
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")

    class ConstantSafeTarget(torch.nn.Module):
        def forward(self, inputs):
            value = torch.tensor(values, device=inputs.device, dtype=inputs.dtype)
            return value.reshape(1, 16).expand(inputs.shape[0], -1)

    script_path = tmp_path / "direct.jit.pt"
    torch.jit.trace(ConstantSafeTarget(), torch.zeros(1, 578)).save(str(script_path))
    checkpoint_path = tmp_path / "direct.pt"
    torch.save(
        {
            "kind": "bounded_phase_direct_action_mlp",
            "workflow_id": inference.WORKFLOW_ID,
            "architecture": {"input_dim": 578, "hidden_dims": [512, 512, 256], "output_dim": 16},
            "dataset_manifest_sha256": "dataset-manifest",
            "selected_reference_sha256": reference.sha256,
        },
        checkpoint_path,
    )
    training = {
        "kind": "highstep_fixed_condition_phase_direct_action_training",
        "workflow_id": inference.WORKFLOW_ID,
        "dataset_manifest_sha256": "dataset-manifest",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "torchscript_path": str(script_path),
        "torchscript_sha256": _sha(script_path),
        "preregistration_path": str(prereg_path),
        "preregistration_sha256": _sha(prereg_path),
        "selected_reference_manifest_path": str(selected_path),
        "selected_reference_manifest_sha256": _sha(selected_path),
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    verification = {
        "kind": "highstep_phase_direct_action_wandb_verification",
        "status": "verified",
        "remote_verified": True,
        "checkpoint_sha256": training["checkpoint_sha256"],
    }
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    term = type("JointPositionAction", (), {})()
    term._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
    term._offset = torch.tensor([[0.0] * 4 + [0.4] * 4 + [-1.0] * 4 + [0.0] * 4])
    controller = inference.PhaseDirectActionController(
        training_manifest_path=training_path,
        training_manifest_sha256=_sha(training_path),
        wandb_verification_path=verification_path,
        wandb_verification_sha256=_sha(verification_path),
        reference=reference,
        preroll=preroll,
        environment_checkpoint_path=teacher_path,
        action_term=term,
        initial_joint_pos=torch.zeros(2, 16),
        joint_names=runtime.EXPECTED_JOINT_ORDER,
        settle_steps=296,
        pre_active_steps=1,
    )
    return controller, term


def test_direct_controller_uses_candidate_history_and_exact_safe_target(tmp_path):
    controller, term = _fixture(tmp_path)
    obs = torch.zeros(2, 570)
    action0 = controller.action_for_step(obs, 0)
    term.processed_actions = action0 * term._scale + term._offset
    assert controller.audit_processed_target()["processed_target_exact"] is True
    action296 = controller.action_for_step(obs, 296)
    sample = controller.dagger_sample()
    assert sample["stage"] == 0
    assert sample["step"] == 0
    assert torch.any(sample["student_obs_570"][:, 410:] != 0.0)
    term.processed_actions = action296 * term._scale + term._offset
    assert controller.audit_processed_target()["processed_target_exact"] is True
    assert controller.summary()["controller_mode"] == "phase_direct_action"
    assert controller.summary()["final_command_target_limit_violations"] == 0


def test_direct_controller_fails_closed_on_training_sha(tmp_path):
    controller, _ = _fixture(tmp_path)
    inference = sys.modules["phase_direct_action_inference_test"]
    try:
        inference._load_bound_json(
            controller.training_manifest_path, "0" * 64, name="training"
        )
    except RuntimeError as error:
        assert "SHA mismatch" in str(error)
    else:
        raise AssertionError("wrong evidence SHA must fail closed")
