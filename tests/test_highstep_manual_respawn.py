from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/rsl_rl/base"
sys.path.insert(0, str(BASE))

from highstep_manual_respawn import ManualHighstepStartGate, restore_default_joint_state  # noqa: E402


class _FakeRobotData:
    def __init__(self):
        self.joint_pos = torch.zeros(2, 4)
        self.joint_vel = torch.full((2, 4), 7.0)
        self.default_joint_pos = torch.tensor(
            [[0.0, 0.7, -1.3, 0.03], [0.1, 0.8, -1.2, 0.02]],
            dtype=torch.float32,
        )


class _FakeRobot:
    def __init__(self):
        self.data = _FakeRobotData()
        self.write = None

    def write_joint_state_to_sim(self, joint_pos, joint_vel, env_ids):
        self.write = (joint_pos.clone(), joint_vel.clone(), env_ids.clone())


def _observe_stable(gate: ManualHighstepStartGate, joint_pos=None):
    if joint_pos is None:
        joint_pos = torch.zeros(4)
    return gate.observe(
        joint_pos,
        torch.zeros(4),
        torch.zeros(3),
        torch.zeros(3),
    )


def test_restore_default_joint_state_writes_defaults_and_explicit_zero_velocity():
    robot = _FakeRobot()

    env_ids, joint_pos, joint_vel = restore_default_joint_state(robot)

    assert torch.equal(env_ids, torch.tensor([0, 1]))
    assert torch.equal(joint_pos, robot.data.default_joint_pos)
    assert torch.count_nonzero(joint_vel) == 0
    assert robot.write is not None
    written_pos, written_vel, written_env_ids = robot.write
    assert torch.equal(written_pos, robot.data.default_joint_pos)
    assert torch.count_nonzero(written_vel) == 0
    assert torch.equal(written_env_ids, env_ids)


def test_start_gate_holds_commands_for_two_seconds_then_requires_stability():
    gate = ManualHighstepStartGate(min_hold_steps=100, stable_steps_required=10)
    command = torch.tensor([0.72, 0.0, 0.0])

    for _ in range(99):
        assert not _observe_stable(gate)
        assert torch.equal(gate.filter_command(command), torch.zeros_like(command))

    assert _observe_stable(gate)
    assert gate.ready
    assert torch.equal(gate.filter_command(command), command)


def test_start_gate_does_not_open_on_time_alone_and_reset_rearms_it():
    gate = ManualHighstepStartGate(min_hold_steps=12, stable_steps_required=3)

    for step in range(12):
        joint_pos = torch.zeros(4)
        if step == 11:
            joint_pos[0] = 0.1
        assert not _observe_stable(gate, joint_pos)
    assert gate.blocked

    assert not _observe_stable(gate, torch.full((4,), 0.1))
    assert not _observe_stable(gate, torch.full((4,), 0.1))
    assert not _observe_stable(gate, torch.full((4,), 0.1))
    assert _observe_stable(gate, torch.full((4,), 0.1))
    assert gate.ready

    gate.reset()
    assert gate.blocked
    assert gate.steps_since_reset == 0


def test_play_binds_complete_respawn_and_same_gate_to_observation_and_command_manager():
    source = (BASE / "play.py").read_text(encoding="utf-8")

    assert source.count("_apply_manual_highstep_respawn(") >= 3
    assert "unwrapped.observation_manager.compute(update_history=True)" in source
    assert "manual_highstep_start_gate.reset()" in source
    assert source.count("_manual_keyboard_command()") >= 3
    assert "robot_state.root_lin_vel_b[0]" in source
    assert "robot_state.root_ang_vel_b[0]" in source
    assert "RESET RECOVERING / 请勿操作" in source
    assert "RESET READY / 现在可以推动方向键" in source
    assert "full recovery wait restarted" in source
