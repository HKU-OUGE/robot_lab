from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/rsl_rl/base/play_joint_recorder.py"
PLAY_PATH = ROOT / "scripts/rsl_rl/base/play.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("play_joint_recorder_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _vector(length: int, offset: float = 0.0) -> list[float]:
    return [offset + index / 100.0 for index in range(length)]


def _student_tensors(module, offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        name: torch.arange(width, dtype=torch.float32) + offset
        for name, width in module.STUDENT_TENSOR_WIDTHS.items()
    }


def _load_play_tensor_helpers():
    tree = ast.parse(PLAY_PATH.read_text(encoding="utf-8"))
    wanted = {"_recording_observation_group", "_student_tensor_recording_snapshot"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PLAY_PATH), "exec"), namespace)
    return namespace["_student_tensor_recording_snapshot"]


def test_joint_recorder_streams_exact_16d_mapped_targets_and_reset_episodes(tmp_path):
    module = _load_module()
    joint_names = [f"joint_{index:02d}" for index in range(16)]
    output = tmp_path / "student_record"
    recorder = module.PlayJointRecorder(
        output,
        joint_names=joint_names,
        step_dt=0.02,
        metadata={"checkpoint_sha256": "a" * 64, "label": "student"},
        flush_every=1,
    )

    common = {
        "policy_raw": _vector(16, 0.1),
        "action_manager_raw": _vector(16, 0.2),
        "mapped_target": _vector(16, 0.3),
        "joint_pos": _vector(16, 0.4),
        "joint_vel": _vector(16, 0.5),
        "command": [0.45, 0.0, 0.0],
        "root_pos": [1.0, 2.0, 3.0],
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel_b": [0.1, 0.2, 0.3],
        "root_ang_vel_b": [0.4, 0.5, 0.6],
    }
    recorder.record(control_step=0, terminated_after_step=False, **common)
    recorder.mark_reset()
    recorder.record(control_step=1, terminated_after_step=False, **common)
    recorder.close("normal")

    with (output / "joint_trace.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["episode_id"] for row in rows] == ["0", "1"]
    assert [row["episode_step"] for row in rows] == ["0", "0"]
    assert [row["reset_before_step"] for row in rows] == ["1", "1"]
    mapped_columns = [column for column in rows[0] if column.startswith("mapped_target.")]
    assert mapped_columns == [f"mapped_target.{name}" for name in joint_names]
    assert float(rows[0]["mapped_target.joint_15"]) == pytest.approx(0.45)
    assert float(rows[0]["joint_pos.joint_15"]) == pytest.approx(0.55)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["sample_count"] == 2
    assert manifest["episode_count"] == 2
    assert manifest["joint_count"] == 16
    assert manifest["metadata"]["checkpoint_sha256"] == "a" * 64
    expected_sha = hashlib.sha256((output / "joint_trace.csv").read_bytes()).hexdigest()
    assert manifest["csv_sha256"] == expected_sha


def test_joint_recorder_fails_closed_on_wrong_action_contract(tmp_path):
    module = _load_module()
    with pytest.raises(ValueError, match="exact 16-action contract"):
        module.PlayJointRecorder(
            tmp_path / "bad",
            joint_names=["only_one"],
            step_dt=0.02,
            metadata={},
        )


def test_joint_recorder_never_overwrites_existing_directory(tmp_path):
    module = _load_module()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.PlayJointRecorder(
            output,
            joint_names=[f"joint_{index}" for index in range(16)],
            step_dt=0.02,
            metadata={},
        )


def test_joint_recorder_saves_aligned_student_observation_and_latent_tensors(tmp_path):
    module = _load_module()
    joint_names = [f"joint_{index:02d}" for index in range(16)]
    output = tmp_path / "student_tensors"
    recorder = module.PlayJointRecorder(
        output,
        joint_names=joint_names,
        step_dt=0.02,
        metadata={"checkpoint_sha256": "b" * 64, "label": "student"},
        flush_every=1,
        record_student_tensors=True,
    )
    common = {
        "policy_raw": _vector(16, 0.1),
        "action_manager_raw": _vector(16, 0.2),
        "mapped_target": _vector(16, 0.3),
        "joint_pos": _vector(16, 0.4),
        "joint_vel": _vector(16, 0.5),
        "command": [0.45, 0.0, 0.0],
        "root_pos": [1.0, 2.0, 3.0],
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel_b": [0.1, 0.2, 0.3],
        "root_ang_vel_b": [0.4, 0.5, 0.6],
    }
    recorder.record(
        control_step=7,
        terminated_after_step=False,
        student_tensors=_student_tensors(module, 0.0),
        **common,
    )
    recorder.mark_reset()
    recorder.record(
        control_step=8,
        terminated_after_step=False,
        student_tensors=_student_tensors(module, 1000.0),
        **common,
    )
    recorder.close("normal")

    payload = torch.load(output / "student_tensors.pt", weights_only=False)
    assert tuple(payload["student_obs_570"].shape) == (2, 570)
    assert tuple(payload["estimator_obs_570"].shape) == (2, 570)
    assert tuple(payload["latent_raw_mu_64"].shape) == (2, 64)
    assert tuple(payload["latent_clamped_mu_64"].shape) == (2, 64)
    assert tuple(payload["actor_input_634"].shape) == (2, 634)
    assert tuple(payload["velocity_pred_3"].shape) == (2, 3)
    assert payload["student_obs_570"][1, 569].item() == pytest.approx(1569.0)
    assert payload["control_step"].tolist() == [7, 8]
    assert payload["episode_id"].tolist() == [0, 1]
    assert payload["episode_step"].tolist() == [0, 0]
    assert payload["reset_before_step"].tolist() == [1, 1]

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["student_tensor_recording_enabled"] is True
    assert manifest["student_tensor_sample_count"] == 2
    assert manifest["student_tensor_shapes"]["actor_input_634"] == [2, 634]
    expected_sha = hashlib.sha256((output / "student_tensors.pt").read_bytes()).hexdigest()
    assert manifest["student_tensors_sha256"] == expected_sha


def test_joint_recorder_student_tensor_mode_fails_closed_on_missing_or_wrong_fields(tmp_path):
    module = _load_module()
    recorder = module.PlayJointRecorder(
        tmp_path / "bad_tensors",
        joint_names=[f"joint_{index:02d}" for index in range(16)],
        step_dt=0.02,
        metadata={},
        record_student_tensors=True,
    )
    common = {
        "policy_raw": _vector(16),
        "action_manager_raw": _vector(16),
        "mapped_target": _vector(16),
        "joint_pos": _vector(16),
        "joint_vel": _vector(16),
        "command": [0.0, 0.0, 0.0],
        "root_pos": [0.0, 0.0, 0.0],
        "root_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "root_lin_vel_b": [0.0, 0.0, 0.0],
        "root_ang_vel_b": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="no student tensors"):
        recorder.record(control_step=0, terminated_after_step=False, **common)
    broken = _student_tensors(module)
    broken["latent_raw_mu_64"] = torch.zeros(63)
    with pytest.raises(ValueError, match="latent_raw_mu_64 has 63 values"):
        recorder.record(
            control_step=0,
            terminated_after_step=False,
            student_tensors=broken,
            **common,
        )
    recorder.close("test_cleanup")


def test_play_records_processed_actions_after_environment_step():
    source = PLAY_PATH.read_text(encoding="utf-8")
    assert '"--record_joint_data"' in source
    assert 'mapped = getattr(term, "processed_actions", None)' in source
    step_offset = source.index("obs, _, dones, _ = env.step(actions)")
    record_offset = source.index("joint_recorder.record(", step_offset)
    assert record_offset > step_offset
    assert 'joint_recorder.mark_reset()' in source


def test_play_captures_student_tensors_before_policy_and_writes_after_step():
    source = PLAY_PATH.read_text(encoding="utf-8")
    assert '"--record_student_tensors"' in source
    snapshot = "pending_student_tensors = _student_tensor_recording_snapshot(obs, policy_nn)"
    snapshot_offset = source.index(snapshot)
    policy_offset = source.index("actions = policy(obs)", snapshot_offset)
    step_offset = source.index("obs, _, dones, _ = env.step(actions)", policy_offset)
    record_offset = source.index("student_tensors=pending_student_tensors", step_offset)
    assert snapshot_offset < policy_offset < step_offset < record_offset


def test_play_student_snapshot_is_pre_step_clamped_and_actor_normalized():
    snapshot = _load_play_tensor_helpers()

    class Estimator:
        def encode(self, observation):
            mu = observation[:, :64].clone()
            velocity = observation[:, 64:67].clone()
            return mu, torch.zeros_like(mu), velocity

    class Policy:
        estimator = Estimator()
        policy_keys = ("policy",)
        estimator_keys = ("estimator",)

        @staticmethod
        def actor_obs_normalizer(value):
            return value * 2.0

    policy_obs = torch.arange(2 * 570, dtype=torch.float32).reshape(2, 570) / 100.0
    estimator_obs = torch.arange(2 * 570, dtype=torch.float32).reshape(2, 570) / 10.0 - 2.0
    result = snapshot({"policy": policy_obs, "estimator": estimator_obs}, Policy())

    assert tuple(result["student_obs_570"].shape) == (570,)
    assert torch.equal(result["student_obs_570"], policy_obs[0])
    assert torch.equal(result["estimator_obs_570"], estimator_obs[0])
    assert torch.equal(result["latent_raw_mu_64"], estimator_obs[0, :64])
    expected_clamped = torch.clamp(estimator_obs[0, :64], -1.0, 1.0)
    assert torch.equal(result["latent_clamped_mu_64"], expected_clamped)
    assert torch.equal(result["velocity_pred_3"], estimator_obs[0, 64:67])
    expected_actor_input = torch.cat((policy_obs[0], expected_clamped), dim=0) * 2.0
    assert torch.equal(result["actor_input_634"], expected_actor_input)
