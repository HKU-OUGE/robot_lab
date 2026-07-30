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
    inference = _load(BASE / "highstep_phase_residual_inference.py", "phase_residual_inference_test")
    target_fields = [f"reference_target.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    measured_fields = [f"measured_joint_pos.{name}" for name in runtime.EXPECTED_JOINT_ORDER]
    reference_path = tmp_path / "reference.csv"
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reference_step", "global_phase", *target_fields, *measured_fields],
        )
        writer.writeheader()
        for step in range(6):
            values = [0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4
            row = {"reference_step": step, "global_phase": step / 5}
            row.update(dict(zip(target_fields, values)))
            row.update(dict(zip(measured_fields, values)))
            writer.writerow(row)
    reference = runtime.PhaseResidualReference(reference_path, expected_sha256=_sha(reference_path))
    preroll_path = tmp_path / "preroll.csv"
    with preroll_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["preroll_step", *target_fields]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(296):
            values = [0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4
            row = {"preroll_step": step}
            row.update(dict(zip(target_fields, values)))
            writer.writerow(row)
    preroll_manifest = {
        "kind": "highstep_fixed_condition_teacher_zero_command_preroll",
        "status": "frozen_preroll_ready_before_behavior_retry",
        "workflow_id": inference.WORKFLOW_ID,
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
    preroll = inference.PhaseResidualPreroll(
        preroll_manifest_path,
        _sha(preroll_manifest_path),
        expected_reference_sha256=reference.sha256,
        expected_timing_sha256="timing-sha",
    )
    selected = {
        "selected_reference": {"sha256": reference.sha256},
        "preregistration": {},
        "stage_boundaries_control_step": {
            "approach": [0, 0], "front_lift": [1, 1], "front_support": [2, 2],
            "first_rear": [3, 3], "second_rear": [4, 4], "rear_hold": [5, 5],
        },
    }
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"frozen-teacher")
    prereg = {
        "unchanged_contract": {
            "teacher_checkpoint": str(teacher_path),
            "teacher_sha256": _sha(teacher_path),
        }
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    selected["preregistration"] = {"path": str(prereg_path), "sha256": _sha(prereg_path)}
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected), encoding="utf-8")

    class ZeroResidual(torch.nn.Module):
        def forward(self, inputs):
            return torch.zeros(inputs.shape[0], 16, device=inputs.device, dtype=inputs.dtype)

    script_path = tmp_path / "model.jit.pt"
    torch.jit.trace(ZeroResidual(), torch.zeros(1, 578)).save(str(script_path))
    checkpoint_path = tmp_path / "model.pt"
    checkpoint = {
        "kind": "bounded_phase_residual_mlp",
        "workflow_id": inference.WORKFLOW_ID,
        "dataset_manifest_sha256": "dataset-manifest",
        "selected_reference_sha256": reference.sha256,
        "architecture": {"input_dim": 578, "hidden_dims": [256, 256], "output_dim": 16},
        "residual_bound": [0.2] * 12 + [0.006] * 4,
    }
    torch.save(checkpoint, checkpoint_path)
    training = {
        "kind": "highstep_fixed_condition_phase_residual_training",
        "workflow_id": inference.WORKFLOW_ID,
        "dataset_manifest_sha256": "dataset-manifest",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "torchscript_path": str(script_path),
        "torchscript_sha256": _sha(script_path),
        "selected_reference_manifest_path": str(selected_path),
        "selected_reference_manifest_sha256": _sha(selected_path),
    }
    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    verification = {
        "status": "verified", "remote_verified": True,
        "checkpoint_sha256": training["checkpoint_sha256"],
    }
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(json.dumps(verification), encoding="utf-8")
    Term = type("JointPositionAction", (), {})
    term = Term()
    term._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
    term._offset = torch.tensor([[0.0] * 4 + [0.4] * 4 + [-1.0] * 4 + [0.0] * 4])
    controller = inference.PhaseResidualHybridController(
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
        residual_enabled=True,
        settle_steps=296,
        pre_active_steps=1,
    )
    return controller, term


def test_hybrid_controller_uses_zero_command_then_candidate_observation(tmp_path):
    controller, term = _fixture(tmp_path)
    assert controller.command_is_active(296) is False
    assert controller.command_is_active(297) is True
    obs = torch.zeros(2, 570)
    action0 = controller.action_for_step(obs, 0)
    term.processed_actions = action0 * term._scale + term._offset
    assert controller.audit_processed_target()["processed_target_exact"] is True
    action296 = controller.action_for_step(obs, 296)
    sample = controller.dagger_sample()
    assert sample["step"] == 0
    assert sample["stage"] == 0
    assert torch.equal(sample["student_obs_570"][:, :410], obs[:, :410])
    assert torch.any(sample["student_obs_570"][:, 410:] != 0.0)
    assert torch.equal(sample["candidate_residual_16"], torch.zeros(2, 16))
    term.processed_actions = action296 * term._scale + term._offset
    assert controller.audit_processed_target()["processed_target_exact"] is True
    assert controller.summary()["final_command_target_limit_violations"] == 0


def test_hybrid_controller_fails_closed_on_wandb_verification_sha(tmp_path):
    controller, _ = _fixture(tmp_path)
    assert controller.checkpoint_sha256
    try:
        inference = sys.modules["phase_residual_inference_test"]
        inference._load_bound_json(
            controller.training_manifest_path, "0" * 64, name="training"
        )
    except RuntimeError as error:
        assert "SHA mismatch" in str(error)
    else:
        raise AssertionError("wrong evidence SHA must fail closed")


def test_canonical_play_wires_hybrid_before_policy_and_keeps_a_b_mode():
    source = (BASE / "play.py").read_text(encoding="utf-8")
    assert "PhaseResidualHybridController" in source
    assert "--phase_residual_behavior_output" in source
    assert "--phase_residual_preroll_manifest" in source
    assert "--phase_residual_disable_residual" in source
    assert "--phase_residual_dagger_output" in source
    assert source.index("teacher_probe_action = policy(obs)") < source.index(
        "actions = phase_residual_hybrid_controller.action_for_step(obs, timestep)"
    )
    assert source.index("actions = phase_residual_hybrid_controller.action_for_step(obs, timestep)") < source.index(
        "obs, _, dones, _ = env.step(actions)"
    )
    assert source.index("if phase_residual_hybrid_controller is not None:\n                dagger_teacher_target") < source.index(
        "elif phase_residual_reference_controller is None:\n                dataset_step"
    )
    assert '"full_climb_count": hold_count' in source
    assert '"rear_hold_count": hold_count' in source
    assert "forbids the env0-only early-break metric path" in source
