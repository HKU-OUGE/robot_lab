from pathlib import Path
import importlib.util
import sys

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "rsl_rl" / "base"
sys.path.insert(0, str(BASE))
SPEC = importlib.util.spec_from_file_location(
    "fixed_motion_raw_action_test", BASE / "highstep_fixed_motion_raw_action.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_push_phase_reaches_one_and_release_freezes():
    phase = MODULE.JoystickPhase(1, torch.device("cpu"), torch.float32)
    for _ in range(21):
        features, _ = phase.features(torch.tensor([0.0]))
    assert features[0, 0].item() == 0.0
    for _ in range(59):
        features, _ = phase.features(torch.tensor([MODULE.FULL_PUSH_VX]))
    assert features[0, 0].item() == 1.0
    frozen, _ = phase.features(torch.tensor([0.0]))
    assert torch.equal(features, frozen)


def test_partial_push_scales_phase_and_negative_never_rewinds():
    phase = MODULE.JoystickPhase(1, torch.device("cpu"), torch.float32)
    half, _ = phase.features(torch.tensor([MODULE.FULL_PUSH_VX / 2]))
    expected = 0.5 / MODULE.FULL_PUSH_STEPS
    assert abs(half[0, 0].item() - expected) < 1.0e-7
    negative, _ = phase.features(torch.tensor([-1.0]))
    assert negative[0, 0].item() == half[0, 0].item()


def test_manual_reset_clears_raw_student_phase():
    controller = object.__new__(MODULE.RawActionStudentController)
    controller.phase = MODULE.JoystickPhase(1, torch.device("cpu"), torch.float32)
    controller.one_shot_phase = None
    controller.phase.features(torch.tensor([MODULE.FULL_PUSH_VX]))
    assert controller.phase.value.item() > 0.0
    controller.reset()
    assert controller.phase.value.item() == 0.0

    play = (BASE / "play.py").read_text()
    assert "raw_motion_student_controller.reset()" in play


def test_raw_runtime_does_not_invert_or_bypass_production_action_term():
    runtime = (BASE / "highstep_fixed_motion_raw_action.py").read_text()
    play = (BASE / "play.py").read_text()
    assert "return teacher_raw" in runtime
    assert "raw_action_term.raw_actions" not in runtime
    assert "env.step(actions)" in play
    assert "RawPassthroughCollector" in play
    assert "SafeMappedTargetAdapter" not in runtime


def test_raw_runtime_contains_no_training_operation():
    source = (BASE / "highstep_fixed_motion_raw_action.py").read_text()
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
    assert "runner.learn" not in source


def test_phase_only_action_term_receives_the_exact_student_phase():
    class FakeActionTerm:
        phase = None

        def set_phase_only_deployment_phase(self, phase):
            self.phase = phase.clone()

    controller = object.__new__(MODULE.RawActionStudentController)
    controller.num_envs = 1
    controller.device = torch.device("cpu")
    controller.dtype = torch.float32
    controller.phase = MODULE.JoystickPhase(1, controller.device, controller.dtype)
    controller.one_shot_phase = None
    controller.phase_only_action_term = FakeActionTerm()
    controller.model = lambda value: torch.zeros((value.shape[0], 16), dtype=value.dtype)
    controller.illegal_outputs = 0
    controller.last_sample = None

    obs = torch.zeros((1, 570), dtype=torch.float32)
    controller.action(obs, torch.tensor([MODULE.FULL_PUSH_VX]))
    assert torch.equal(
        controller.phase_only_action_term.phase,
        controller.last_sample["phase_features_8"][:, 0],
    )


def test_one_shot_clock_ignores_joystick_and_executes_exact_138_steps():
    clock = MODULE.FixedSequenceClock(1, torch.device("cpu"), torch.float32)
    commands = []
    phases = []
    for step in range(138):
        commands.append(clock.command_for_step(step))
        features, _ = clock.features()
        phases.append(features[0, 0].item())
    assert commands[:21] == [0.0] * 21
    assert commands[21:80] == [MODULE.FULL_PUSH_VX] * 59
    assert commands[80:] == [0.0] * 58
    assert phases[20] == 0.0
    assert phases[79] == 1.0
    assert phases[-1] == 1.0
    assert clock.command_for_step(138) == 0.0


def test_one_shot_play_refreshes_observation_after_current_step_command():
    play = (BASE / "play.py").read_text()
    assert 'one_shot_obs_command_step = {"value": 0}' in play
    assert 'one_shot_obs_command_step["value"] = timestep' in play
    assert '0.7200000286 if 21 <= int(clock["value"]) < 80 else 0.0' in play
    command_write = play.index("cmd_term.command[:] = cur_cmd")
    refresh_guard = play.index("if one_shot_command_active:", command_write)
    refresh = play.index("obs = env.get_observations()", refresh_guard)
    inference = play.index("actions = raw_motion_student_controller.action", refresh)
    assert command_write < refresh_guard < refresh < inference


def test_phase_only_contract_is_sensor_free_and_fail_closed():
    action_source = (
        ROOT
        / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
    ).read_text()
    start = action_source.index("def compute_phase_only_highstep_post_prior")
    end = action_source.index("\n\n\nclass ", start)
    helper = action_source[start:end]
    assert "height_delta" not in helper
    assert "front_rear_delta" not in helper
    assert "raycast" in helper
    assert "phase >= (1.0 - terminal_epsilon)" in helper
    assert "front_reach_box_bias: float = -0.020" in helper
    assert "rear_push_box_bias: float = -0.022" in helper


def test_small_raw_bc_problem_learns_without_ppo():
    trainer_path = ROOT / "tools" / "highstep_fixed_motion_raw_action_train.py"
    spec = importlib.util.spec_from_file_location("raw_action_train_test", trainer_path)
    trainer = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(trainer)
    rng = np.random.default_rng(17)
    count = 48
    obs = rng.normal(0.0, 0.1, (count, 570)).astype(np.float32)
    phase = np.zeros((count, 8), dtype=np.float32)
    phase[:, 0] = np.linspace(0.0, 1.0, count)
    phase[:, 1] = 1.0
    target = np.zeros((count, 16), dtype=np.float32)
    target[:, 0] = phase[:, 0] * 2.0
    arrays = {
        "student_obs_570": obs,
        "phase_features_8": phase,
        "teacher_policy_raw_16": target,
        "episode_id": np.zeros(count, dtype=np.int64),
    }
    _, result, history = trainer.train_model(
        arrays, torch.device("cpu"), max_epochs=8
    )
    assert history[-1]["train/loss"] < history[0]["train/loss"]
    assert result["best_epoch"] >= 1
