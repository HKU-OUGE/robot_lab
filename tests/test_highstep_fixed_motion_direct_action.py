from pathlib import Path
import importlib.util
import sys

import torch


BASE = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "base"
sys.path.insert(0, str(BASE))
SPEC = importlib.util.spec_from_file_location(
    "fixed_motion_direct_action_test", BASE / "highstep_fixed_motion_direct_action.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PlainActionTerm:
    def __init__(self):
        self._scale = torch.tensor([[0.1] * 12 + [0.02] * 4])
        self._offset = torch.tensor([[0.0] * 12 + [0.03] * 4])
        self.processed_actions = torch.zeros(1, 16)
        self._clip = torch.tensor([[[-60.0, 60.0]] * 16])

    def process_actions(self, raw):
        self.processed_actions = raw * self._scale + self._offset


def test_full_push_phase_reaches_one_and_release_freezes():
    phase = MODULE.JoystickPhase(1, torch.device("cpu"), torch.float32)
    for _ in range(21):
        features, _ = phase.features(torch.tensor([0.0]))
    assert features[0, 0].item() == 0.0
    for _ in range(59):
        features, _ = phase.features(torch.tensor([MODULE.FULL_PUSH_VX]))
    assert abs(features[0, 0].item() - 1.0) < 1e-6
    frozen, _ = phase.features(torch.tensor([0.0]))
    assert torch.equal(features, frozen)


def test_partial_push_scales_phase_and_negative_never_rewinds():
    phase = MODULE.JoystickPhase(1, torch.device("cpu"), torch.float32)
    half, _ = phase.features(torch.tensor([MODULE.FULL_PUSH_VX / 2]))
    expected = 0.5 / MODULE.FULL_PUSH_STEPS
    assert abs(half[0, 0].item() - expected) < 1e-7
    negative, _ = phase.features(torch.tensor([-1.0]))
    assert negative[0, 0].item() == half[0, 0].item()


def test_safe_target_adapter_inverts_production_mapping():
    term = PlainActionTerm()
    adapter = MODULE.SafeMappedTargetAdapter(term, 1)
    target = torch.tensor([[0.1] * 4 + [0.3] * 4 + [-1.2] * 4 + [0.04] * 4])
    raw = adapter.raw_for_target(target)
    term.process_actions(raw)
    assert adapter.audit()["exact"] is True
    assert torch.allclose(term.processed_actions, target, atol=1e-7, rtol=0)


def test_runtime_contains_no_training_operation():
    source = (BASE / "highstep_fixed_motion_direct_action.py").read_text()
    assert "optimizer.step" not in source
    assert "loss.backward" not in source
    assert "runner.learn" not in source
