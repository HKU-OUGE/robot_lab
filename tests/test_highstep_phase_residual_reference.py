from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "highstep_phase_residual_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase_residual_reference", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
]
LOWER = [-1.0] * 4 + [-1.5] * 4 + [-2.7] * 4 + [0.0] * 4
UPPER = [1.0] * 4 + [3.4] * 4 + [-0.6] * 4 + [0.06] * 4


def _write_trace(path: Path) -> None:
    fields = [
        "episode_id", "control_step", "episode_time_s",
        "command.vx", "command.vy", "command.wz",
        "root_pos.x", "root_pos.y", "root_pos.z",
        "root_quat.w", "root_quat.x", "root_quat.y", "root_quat.z",
        "root_lin_vel_b.x", "root_lin_vel_b.y", "root_lin_vel_b.z",
        "root_ang_vel_b.x", "root_ang_vel_b.y", "root_ang_vel_b.z",
    ]
    for prefix in ("mapped_target", "joint_pos", "joint_vel"):
        fields.extend(f"{prefix}.{name}" for name in JOINTS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        control_step = 0
        for episode in range(3):
            for step in range(50):
                active = 10 <= step < 30
                progress = max(0.0, min(1.0, (step - 10) / 19.0))
                row = {
                    "episode_id": episode,
                    "control_step": control_step,
                    "episode_time_s": step * 0.02,
                    "command.vx": 0.7 if active else 0.0,
                    "command.vy": 0.0,
                    "command.wz": 0.0,
                    "root_pos.x": 1.0 + progress,
                    "root_pos.y": episode * 0.01,
                    "root_pos.z": 0.44 + 0.38 * progress,
                    "root_quat.w": 1.0,
                    "root_quat.x": 0.0,
                    "root_quat.y": 0.0,
                    "root_quat.z": 0.0,
                    "root_lin_vel_b.x": 0.2,
                    "root_lin_vel_b.y": 0.0,
                    "root_lin_vel_b.z": 0.0,
                    "root_ang_vel_b.x": 0.0,
                    "root_ang_vel_b.y": 0.0,
                    "root_ang_vel_b.z": 0.1,
                }
                for index, name in enumerate(JOINTS):
                    base = 0.0 if index < 8 else (-1.2 if index < 12 else 0.03)
                    target = base + 0.01 * episode + 0.02 * progress
                    if episode == 0 and step == 20 and index == 0:
                        target = 1.4
                    row[f"mapped_target.{name}"] = target
                    row[f"joint_pos.{name}"] = min(max(target, LOWER[index]), UPPER[index])
                    row[f"joint_vel.{name}"] = 0.0
                writer.writerow(row)
                control_step += 1


def _fixture(tmp_path: Path, *, bad_trace_sha: bool = False) -> Path:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"teacher")
    spec = tmp_path / "spec.md"
    spec.write_text("spec\n", encoding="utf-8")
    trace = tmp_path / "trace.csv"
    _write_trace(trace)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"joint_names_in_action_order": JOINTS}), encoding="utf-8"
    )
    prereg = {
        "status": "frozen_before_implementation",
        "workflow_id": "highstep_fixed_condition_phase_residual_20260717",
        "branch_spec": {"path": str(spec), "sha256": _sha(spec)},
        "unchanged_contract": {
            "teacher_checkpoint": str(teacher),
            "teacher_sha256": _sha(teacher),
            "joint_order": JOINTS,
            "physical_lower": LOWER,
            "physical_upper": UPPER,
        },
        "source_recording": {
            "manifest_path": str(manifest),
            "manifest_sha256": _sha(manifest),
            "trace_path": str(trace),
            "trace_sha256": "0" * 64 if bad_trace_sha else _sha(trace),
            "episode_count": 3,
            "source_candidates": [
                {"kind": "fixed_episode", "episode_id": 0},
                {"kind": "fixed_episode", "episode_id": 1},
                {"kind": "computed_medoid", "episode_id": None},
            ],
        },
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    return prereg_path


def test_builds_only_preregistered_bounded_candidates(tmp_path: Path):
    module = _load_module()
    prereg = _fixture(tmp_path)
    output = tmp_path / "out"
    result = module.build_references(prereg, output)

    assert len(result["candidates"]) == 6
    assert {row["label"] for row in result["candidates"]} >= {"episode0", "episode1", "medoid"}
    clamped = next(
        row for row in result["candidates"]
        if row["label"] == "episode0" and row["transform"] == "physical_clamped_mapped_target"
    )
    assert clamped["audit"]["source_target_violation_values"] == 1
    assert clamped["audit"]["exported_target_violation_values"] == 0
    assert Path(clamped["path"]).is_file()
    assert _sha(Path(clamped["path"])) == clamped["sha256"]
    with Path(clamped["path"]).open(encoding="utf-8", newline="") as handle:
        first = next(csv.DictReader(handle))
    assert float(first["source_root_pos.x"]) > 0.0
    assert float(first["source_root_lin_vel_b.x"]) == pytest.approx(0.2)


def test_wrong_source_hash_fails_closed(tmp_path: Path):
    module = _load_module()
    prereg = _fixture(tmp_path, bad_trace_sha=True)
    with pytest.raises(RuntimeError, match="source trace SHA256 mismatch"):
        module.build_references(prereg, tmp_path / "out")


def test_refuses_to_overwrite_output_directory(tmp_path: Path):
    module = _load_module()
    prereg = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.build_references(prereg, output)
