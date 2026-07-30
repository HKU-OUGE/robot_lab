from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/rsl_rl/base/highstep_recorded_command_replay.py"
PLAY_PATH = ROOT / "scripts/rsl_rl/base/play.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("recorded_command_replay_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _trace(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "control_step",
                "episode_id",
                "episode_step",
                "reset_before_step",
                "command.vx",
                "command.vy",
                "command.wz",
                "joint_pos.FL_hip_joint",
                "joint_vel.FL_hip_joint",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "control_step": 9,
                "episode_id": 3,
                "episode_step": 8,
                "reset_before_step": 0,
                "command.vx": 0.0,
                "command.vy": 0.0,
                "command.wz": 0.0,
                "joint_pos.FL_hip_joint": 0.125,
                "joint_vel.FL_hip_joint": -0.25,
            }
        )
        for step, vx in enumerate((0.0, 0.0, 0.72, 0.72, 0.0)):
            writer.writerow(
                {
                    "control_step": 10 + step,
                    "episode_id": 4,
                    "episode_step": step,
                    "reset_before_step": int(step == 0),
                    "command.vx": vx,
                    "command.vy": 0.0,
                    "command.wz": 0.0,
                    "joint_pos.FL_hip_joint": 0.0,
                    "joint_vel.FL_hip_joint": 0.0,
                }
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_recorded_command_replay_preserves_zero_history_press_and_release(tmp_path):
    module = _load_module()
    path, digest = _trace(tmp_path)
    replay = module.RecordedCommandReplay(path, expected_sha256=digest, episode_id=4)
    assert replay.frame_count == 5
    assert replay.pre_reset_joint_state(["FL_hip_joint"]) == ((0.125,), (-0.25,))
    assert replay.summary()["pre_reset_source_control_step"] == 9
    assert replay.summary()["transitions"] == [
        {"episode_step": 0, "command": [0.0, 0.0, 0.0]},
        {"episode_step": 2, "command": [0.72, 0.0, 0.0]},
        {"episode_step": 4, "command": [0.0, 0.0, 0.0]},
    ]
    for step, expected in enumerate((0.0, 0.0, 0.72, 0.72, 0.0)):
        replay.set_step(step)
        assert replay.command() == (expected, 0.0, 0.0)


def test_recorded_command_replay_fails_closed_on_sha_or_noncontiguous_episode(tmp_path):
    module = _load_module()
    path, digest = _trace(tmp_path)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        module.RecordedCommandReplay(path, expected_sha256="0" * 64, episode_id=4)
    text = path.read_text(encoding="utf-8").replace("4,1,0,", "4,7,0,")
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="not contiguous"):
        module.RecordedCommandReplay(path, expected_sha256=digest, episode_id=4)


def test_play_has_narrow_recorded_command_and_world_fixed_camera_paths():
    source = PLAY_PATH.read_text(encoding="utf-8")
    assert '"--recorded_command_trace"' in source
    assert (
        'choices=("none", "rear_top", "top", "side_top", "static_side_top", '
        '"static_robot_side")'
    ) in source
    static_start = source.index('if mode == "static_side_top":')
    robot_lookup = source.index('robot = env.unwrapped.scene["robot"]', static_start)
    static_return = source.index("return", static_start)
    assert static_return < robot_lookup
    robot_static_start = source.index('if mode == "static_robot_side":')
    robot_static_return = source.index("return", robot_static_start)
    assert robot_lookup < robot_static_start < robot_static_return
    assert '"static_robot_side",' in source
    assert "recorded_command_replay.set_step(timestep)" in source
    assert "recorded_command_replay.command()" in source
    assert "fixed_velocity_command and recorded command replay are mutually exclusive" in source
