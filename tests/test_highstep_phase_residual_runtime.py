from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/rsl_rl/base/highstep_phase_residual_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase_residual_runtime_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_reference(
    path: Path, module, *, violation: bool = False, full_state: bool = False
) -> str:
    target_fields = [
        f"reference_target.{name}" for name in module.EXPECTED_JOINT_ORDER
    ]
    measured_fields = [
        f"measured_joint_pos.{name}" for name in module.EXPECTED_JOINT_ORDER
    ]
    velocity_fields = [
        f"measured_joint_vel.{name}" for name in module.EXPECTED_JOINT_ORDER
    ]
    fields = ["reference_step", "global_phase"]
    if full_state:
        fields.extend(module.FULL_STATE_COLUMNS)
        fields.extend(velocity_fields)
    fields.extend((*target_fields, *measured_fields))
    rows = []
    for step, phase in enumerate((0.0, 0.5, 1.0)):
        values = [0.0] * 4 + [0.5] * 4 + [-1.2] * 4 + [0.03] * 4
        if violation and step == 1:
            values[0] = 2.0
        row = {"reference_step": step, "global_phase": phase}
        if full_state:
            row.update(
                {
                    "source_root_pos.x": 1.0,
                    "source_root_pos.y": 2.0,
                    "source_root_pos.z": 0.5,
                    "source_root_quat.w": 1.0,
                    "source_root_quat.x": 0.0,
                    "source_root_quat.y": 0.0,
                    "source_root_quat.z": 0.0,
                    "source_root_lin_vel_b.x": 0.2,
                    "source_root_lin_vel_b.y": 0.1,
                    "source_root_lin_vel_b.z": 0.0,
                    "source_root_ang_vel_b.x": 0.0,
                    "source_root_ang_vel_b.y": 0.0,
                    "source_root_ang_vel_b.z": 0.3,
                }
            )
            row.update({field: 0.01 for field in velocity_fields})
        row.update({field: value for field, value in zip(target_fields, values)})
        row.update({field: value for field, value in zip(measured_fields, values)})
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_hash_and_physical_envelope_are_fail_closed(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    assert reference.frame_count == 3
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        module.PhaseResidualReference(path, expected_sha256="0" * 64)

    bad = tmp_path / "bad.csv"
    bad_sha = _write_reference(bad, module, violation=True)
    with pytest.raises(RuntimeError, match="physical-envelope violations"):
        module.PhaseResidualReference(bad, expected_sha256=bad_sha)


def test_initial_measured_state_is_explicitly_projected_to_physical_envelope(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["measured_joint_pos.RR_calf_joint"] = str(module.PHYSICAL_LOWER[11] - 1.0e-5)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    assert reference.initial_measured_projection_count == 1
    assert reference.initial_measured_projection_max_abs == pytest.approx(1.0e-5)
    assert reference.initial_measured_joint_pos_cpu[11].item() == pytest.approx(
        module.PHYSICAL_LOWER[11]
    )


def test_controller_inverts_plain_joint_position_affine_map_exactly(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)

    JointPositionAction = type("JointPositionAction", (), {})
    term = JointPositionAction()
    term._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
    term._offset = torch.tensor([[0.0] * 4 + [0.4] * 4 + [-1.0] * 4 + [0.0] * 4])
    initial = torch.tensor([[0.1] * 4 + [0.6] * 4 + [-1.1] * 4 + [0.02] * 4])
    controller = module.PhaseResidualReferenceController(
        reference,
        action_term=term,
        initial_joint_pos=initial,
        joint_names=module.EXPECTED_JOINT_ORDER,
        blend_steps=2,
    )
    action = controller.action_for_step(2)
    term.processed_actions = action * term._scale + term._offset
    assert controller.audit_processed_target()["processed_target_exact"] is True
    assert torch.allclose(term.processed_actions, reference.targets_cpu[0].float().unsqueeze(0))


def test_full_state_restore_contract_and_next_frame_alignment(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module, full_state=True)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    assert reference.has_initial_full_state is True
    assert reference.initial_root_pose_cpu.tolist() == pytest.approx(
        [1.0, 2.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    )
    assert reference.initial_root_velocity_world_cpu.tolist() == pytest.approx(
        [0.2, 0.1, 0.0, 0.0, 0.0, 0.3]
    )

    JointPositionAction = type("JointPositionAction", (), {})
    term = JointPositionAction()
    term._scale = torch.ones(1, 16)
    term._offset = torch.zeros(1, 16)
    controller = module.PhaseResidualReferenceController(
        reference,
        action_term=term,
        initial_joint_pos=torch.zeros(1, 16),
        joint_names=module.EXPECTED_JOINT_ORDER,
        blend_steps=0,
        start_at_next_frame=True,
    )
    controller.target_for_step(0)
    assert controller.last_reference_index == 1


def test_controller_cancels_production_phased_prior_exactly(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    PriorTerm = type("PhasedHighstepBoxBiasJointPositionAction", (), {})
    term = PriorTerm()
    term._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
    term._offset = torch.tensor([[0.0] * 4 + [0.4] * 4 + [-1.0] * 4 + [0.0] * 4])
    term._action_delay_buffer = None
    box_bias = torch.tensor([[0.0] * 12 + [-0.004, -0.004, 0.004, 0.004]])
    term._box_action_ids = torch.tensor([12, 13, 14, 15])
    term._last_phased_highstep_box_bias = box_bias[:, 12:].clone()

    def process_actions(actions):
        term.processed_actions = actions * term._scale + term._offset + box_bias
        term._last_phased_highstep_box_bias[:] = box_bias[:, 12:]

    term.process_actions = process_actions
    controller = module.PhaseResidualReferenceController(
        reference,
        action_term=term,
        initial_joint_pos=torch.zeros(1, 16),
        joint_names=module.EXPECTED_JOINT_ORDER,
        blend_steps=2,
    )
    action = controller.action_for_step(2)
    term.process_actions(action)
    assert controller.audit_processed_target()["processed_target_exact"] is True
    assert controller.last_prior_probe_iterations <= 2


def test_controller_accepts_one_explicit_safe_composed_target(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    JointPositionAction = type("JointPositionAction", (), {})
    term = JointPositionAction()
    term._scale = torch.ones(1, 16)
    term._offset = torch.zeros(1, 16)
    controller = module.PhaseResidualReferenceController(
        reference,
        action_term=term,
        initial_joint_pos=torch.zeros(2, 16),
        joint_names=module.EXPECTED_JOINT_ORDER,
        blend_steps=0,
    )
    target = reference.targets_cpu[1].float().unsqueeze(0).expand(2, -1)
    action = controller.action_for_target(target, reference_index=1, global_phase=0.5)
    term.processed_actions = action
    assert controller.audit_processed_target()["processed_target_exact"] is True
    assert controller.last_reference_index == 1
    assert controller.last_global_phase == pytest.approx(0.5)


def test_controller_rejects_unknown_action_term(tmp_path):
    module = _load_module()
    path = tmp_path / "reference.csv"
    sha = _write_reference(path, module)
    reference = module.PhaseResidualReference(path, expected_sha256=sha)
    UnknownTerm = type("UnknownTerm", (), {})
    term = UnknownTerm()
    with pytest.raises(RuntimeError, match="exact plain or production phased"):
        module.PhaseResidualReferenceController(
            reference,
            action_term=term,
            initial_joint_pos=torch.zeros(1, 16),
            joint_names=module.EXPECTED_JOINT_ORDER,
        )
