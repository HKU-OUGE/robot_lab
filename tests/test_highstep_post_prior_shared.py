"""CPU-only equivalence tests for the shared phased high-step post-prior."""

from __future__ import annotations

import ast
from collections.abc import Sequence
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
ACTIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
SCHEDULE_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"

SCHEDULE_SPEC = importlib.util.spec_from_file_location("highstep_post_prior_test_schedule", SCHEDULE_PATH)
assert SCHEDULE_SPEC is not None and SCHEDULE_SPEC.loader is not None
schedule = importlib.util.module_from_spec(SCHEDULE_SPEC)
SCHEDULE_SPEC.loader.exec_module(schedule)


class _StubJointPositionAction:
    """Minimal CPU stand-in for Isaac Lab's standard affine action mapping."""

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


class _UnusedDelayBuffer:
    pass


def _load_selected_production_code() -> dict[str, object]:
    tree = ast.parse(ACTIONS_PATH.read_text(encoding="utf-8"), filename=str(ACTIONS_PATH))
    wanted = {
        "PhasedHighstepPostPriorResult",
        "_phased_highstep_smoothstep",
        "compute_phased_highstep_post_prior",
        "PhasedHighstepBoxBiasJointPositionAction",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted
    ]
    namespace: dict[str, object] = {
        "__name__": "highstep_post_prior_selected_production",
        "torch": torch,
        "NamedTuple": NamedTuple,
        "Sequence": Sequence,
        "JointPositionAction": _StubJointPositionAction,
        "DelayBuffer": _UnusedDelayBuffer,
        "_global_update": schedule.global_update,
        "_prior_scale": schedule.prior_scale,
        "_reset_action_delay": lambda *_args, **_kwargs: None,
    }
    selected_module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(selected_module, str(ACTIONS_PATH), "exec"), namespace)
    return namespace


PRODUCTION = _load_selected_production_code()
COMPUTE_POST_PRIOR = PRODUCTION["compute_phased_highstep_post_prior"]
ACTION_TERM = PRODUCTION["PhasedHighstepBoxBiasJointPositionAction"]


def _smoothstep_reference(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _legacy_inline_reference(*args, **kwargs) -> dict[str, torch.Tensor | float]:
    """The pre-extraction live arithmetic, kept independent for regression."""

    (
        raw_actions,
        processed_actions,
        box_action_ids,
        action_scale,
        action_offset,
        height_delta,
        command_x,
        front_rear_delta,
    ) = args

    gate = torch.zeros_like(height_delta)
    if kwargs["terrain_gate_available"]:
        height_gate_width = max(kwargs["height_gate_width"], 1.0e-6)
        height_gate = _smoothstep_reference(
            (height_delta - kwargs["height_threshold"]) / height_gate_width
        )
        command_gate_width = max(kwargs["command_gate_width"], 1.0e-6)
        command_gate = torch.clamp(
            (command_x - kwargs["min_forward_command"]) / command_gate_width,
            0.0,
            1.0,
        )
        gate = height_gate * command_gate

    commit_gate = _smoothstep_reference(
        (front_rear_delta - kwargs["commit_height_delta_min"])
        / max(
            kwargs["commit_height_delta_target"] - kwargs["commit_height_delta_min"],
            1.0e-6,
        )
    )
    reach_gate = gate * (1.0 - commit_gate)
    push_gate = torch.maximum(
        gate * commit_gate,
        kwargs["commit_gate_floor"] * commit_gate,
    )

    reach_bias = torch.tensor(
        [
            kwargs["front_reach_box_bias"],
            kwargs["front_reach_box_bias"],
            kwargs["rear_approach_box_bias"],
            kwargs["rear_approach_box_bias"],
        ],
        device=processed_actions.device,
        dtype=processed_actions.dtype,
    )
    push_bias = torch.tensor(
        [
            kwargs["front_support_box_bias"],
            kwargs["front_support_box_bias"],
            kwargs["rear_push_box_bias"],
            kwargs["rear_push_box_bias"],
        ],
        device=processed_actions.device,
        dtype=processed_actions.dtype,
    )
    box_bias = reach_gate.unsqueeze(1) * reach_bias + push_gate.unsqueeze(1) * push_bias
    prior_scale = schedule.prior_scale(
        kwargs["update_count"],
        kwargs["prior_start_update"],
        kwargs["prior_full_update"],
    )
    box_bias = box_bias * prior_scale

    physical_targets = processed_actions.clone()
    unclamped_box_targets = processed_actions[:, box_action_ids] + box_bias
    box_clip_mask = (unclamped_box_targets < kwargs["min_box_target"]) | (
        unclamped_box_targets > kwargs["max_box_target"]
    )
    box_targets = torch.clamp(
        unclamped_box_targets,
        min=kwargs["min_box_target"],
        max=kwargs["max_box_target"],
    )
    physical_targets[:, box_action_ids] = box_targets

    box_scale = action_scale[:, box_action_ids] if isinstance(action_scale, torch.Tensor) else action_scale
    box_offset = action_offset[:, box_action_ids] if isinstance(action_offset, torch.Tensor) else action_offset
    raw_equivalent_actions = raw_actions.clone()
    raw_equivalent_actions[:, box_action_ids] = (box_targets - box_offset) / box_scale
    return {
        "raw_equivalent_actions": raw_equivalent_actions,
        "physical_targets": physical_targets,
        "box_bias": box_bias,
        "gate": torch.maximum(reach_gate, push_gate) * prior_scale,
        "reach_gate": reach_gate * prior_scale,
        "push_gate": push_gate * prior_scale,
        "box_clip_mask": box_clip_mask,
        "prior_scale": prior_scale,
    }


def _teacher_prior_cfg() -> SimpleNamespace:
    """Resolved ActionScore Teacher values, including its final overrides."""

    return SimpleNamespace(
        clip={".*": (-60.0, 60.0)},
        command_name="base_velocity",
        height_threshold=0.060,
        height_gate_width=0.14,
        min_forward_command=0.06,
        command_gate_width=0.22,
        commit_height_delta_min=0.04,
        commit_height_delta_target=0.18,
        commit_gate_floor=0.0,
        front_reach_box_bias=-0.020,
        rear_approach_box_bias=0.002,
        front_support_box_bias=0.002,
        rear_push_box_bias=-0.022,
        min_box_target=0.000,
        max_box_target=0.060,
        prior_start_update=80,
        prior_full_update=520,
        num_steps_per_update=24,
    )


def _make_live_term(
    *,
    sensor_available: bool,
    terrain_height_delta: float,
    command_x: float,
    front_rear_delta: float,
    box_physical_targets: tuple[float, float, float, float] = (0.03, 0.03, 0.03, 0.03),
    commit_gate_floor: float = 0.0,
    update_count: int = 520,
) -> tuple[object, torch.Tensor]:
    batch_size = 3
    box_ids = torch.tensor([12, 13, 14, 15], dtype=torch.long)
    scale = torch.full((batch_size, 16), 0.1, dtype=torch.float32)
    scale[:, box_ids] = 0.02
    offset = torch.zeros((batch_size, 16), dtype=torch.float32)
    offset[:, 4:8] = 0.7
    offset[:, 8:12] = -1.3
    offset[:, box_ids] = 0.03
    raw_actions = torch.linspace(-0.25, 0.25, batch_size * 16, dtype=torch.float32).reshape(batch_size, 16)
    wanted_box_targets = torch.tensor(box_physical_targets, dtype=torch.float32).repeat(batch_size, 1)
    raw_actions[:, box_ids] = (wanted_box_targets - offset[:, box_ids]) / scale[:, box_ids]

    command = torch.zeros(batch_size, 3, dtype=torch.float32)
    command[:, 0] = command_x
    env = SimpleNamespace(
        common_step_counter=update_count * 24,
        command_manager=SimpleNamespace(get_command=lambda _name: command),
    )

    body_pos_w = torch.zeros(batch_size, 4, 3, dtype=torch.float32)
    body_pos_w[:, 0:2, 2] = front_rear_delta
    asset = SimpleNamespace(data=SimpleNamespace(body_pos_w=body_pos_w))

    term = object.__new__(ACTION_TERM)
    term.num_envs = batch_size
    term.device = torch.device("cpu")
    term.cfg = _teacher_prior_cfg()
    term.cfg.commit_gate_floor = commit_gate_floor
    term._raw_actions = torch.zeros_like(raw_actions)
    term._processed_actions = torch.zeros_like(raw_actions)
    term._scale = scale
    term._offset = offset
    term._clip = torch.tensor([-60.0, 60.0], dtype=torch.float32).reshape(1, 1, 2).repeat(batch_size, 16, 1)
    term._box_action_ids = box_ids
    term._front_foot_ids = [0, 1]
    term._rear_foot_ids = [2, 3]
    term._asset = asset
    term._env = env
    term._front_ray_mask = None
    term._rear_ray_mask = None
    term._height_sensor = None
    if sensor_available:
        ray_hits_w = torch.zeros(batch_size, 4, 3, dtype=torch.float32)
        ray_hits_w[:, 0:2, 2] = terrain_height_delta
        term._height_sensor = SimpleNamespace(
            data=SimpleNamespace(
                ray_hits_w=ray_hits_w,
                pos_w=torch.zeros(batch_size, 3, dtype=torch.float32),
            )
        )
        term._front_ray_mask = torch.tensor([True, True, False, False])
        term._rear_ray_mask = torch.tensor([False, False, True, True])
    term._last_phased_highstep_box_bias = torch.zeros(batch_size, 4)
    term._last_phased_highstep_gate = torch.zeros(batch_size)
    term._last_phased_highstep_reach_gate = torch.zeros(batch_size)
    term._last_phased_highstep_push_gate = torch.zeros(batch_size)
    term._last_phased_highstep_height_delta = torch.zeros(batch_size)
    term._last_phased_highstep_prior_scale = 0.0
    term._action_delay_buffer = None
    return term, raw_actions


class SharedPostPriorTests(unittest.TestCase):
    def _assert_live_case(self, **case) -> object:
        term, raw_actions = _make_live_term(**case)
        raw_input_before = raw_actions.clone()
        captured: dict[str, object] = {}
        original_compute = PRODUCTION["compute_phased_highstep_post_prior"]

        def capture_compute(*args, **kwargs):
            captured["args"] = tuple(
                argument.clone() if isinstance(argument, torch.Tensor) else argument
                for argument in args
            )
            captured["kwargs"] = {
                name: value.clone() if isinstance(value, torch.Tensor) else value
                for name, value in kwargs.items()
            }
            result = original_compute(*args, **kwargs)
            captured["result"] = result
            return result

        PRODUCTION["compute_phased_highstep_post_prior"] = capture_compute
        try:
            term.process_actions(raw_actions)
        finally:
            PRODUCTION["compute_phased_highstep_post_prior"] = original_compute

        self.assertTrue(torch.equal(raw_actions, raw_input_before), "caller raw input was mutated")
        reference = _legacy_inline_reference(*captured["args"], **captured["kwargs"])
        result = captured["result"]
        for field in (
            "raw_equivalent_actions",
            "physical_targets",
            "box_bias",
            "gate",
            "reach_gate",
            "push_gate",
            "box_clip_mask",
        ):
            self.assertTrue(torch.equal(getattr(result, field), reference[field]), field)
        self.assertEqual(result.prior_scale, reference["prior_scale"])
        self.assertTrue(torch.equal(term._processed_actions, reference["physical_targets"]))
        self.assertTrue(torch.equal(term._last_phased_highstep_box_bias, reference["box_bias"]))
        self.assertTrue(torch.equal(term._last_phased_highstep_gate, reference["gate"]))
        self.assertTrue(torch.equal(term._last_phased_highstep_reach_gate, reference["reach_gate"]))
        self.assertTrue(torch.equal(term._last_phased_highstep_push_gate, reference["push_gate"]))
        return result

    def test_no_sensor_is_exact_zero_gate(self) -> None:
        result = self._assert_live_case(
            sensor_available=False,
            terrain_height_delta=0.30,
            command_x=0.40,
            front_rear_delta=0.0,
        )
        self.assertEqual(0, torch.count_nonzero(result.gate).item())
        self.assertEqual(0, torch.count_nonzero(result.box_bias).item())

    def test_sensor_available_but_below_threshold_is_exact_zero_gate(self) -> None:
        result = self._assert_live_case(
            sensor_available=True,
            terrain_height_delta=0.02,
            command_x=0.40,
            front_rear_delta=0.0,
        )
        self.assertEqual(0, torch.count_nonzero(result.gate).item())
        self.assertEqual(0, torch.count_nonzero(result.box_bias).item())

    def test_no_sensor_preserves_nonzero_commit_floor_push(self) -> None:
        result = self._assert_live_case(
            sensor_available=False,
            terrain_height_delta=0.0,
            command_x=0.0,
            front_rear_delta=0.22,
            commit_gate_floor=0.25,
        )
        self.assertEqual(0, torch.count_nonzero(result.reach_gate).item())
        self.assertTrue(torch.all(result.push_gate > 0.0))

    def test_reach_phase_is_bitwise_equal_to_legacy_inline_math(self) -> None:
        result = self._assert_live_case(
            sensor_available=True,
            terrain_height_delta=0.30,
            command_x=0.40,
            front_rear_delta=0.0,
            update_count=300,
        )
        self.assertEqual(0.5, result.prior_scale)
        self.assertTrue(torch.all(result.reach_gate > 0.0))
        self.assertEqual(0, torch.count_nonzero(result.push_gate).item())

    def test_commit_push_phase_is_bitwise_equal_to_legacy_inline_math(self) -> None:
        result = self._assert_live_case(
            sensor_available=True,
            terrain_height_delta=0.30,
            command_x=0.40,
            front_rear_delta=0.22,
        )
        self.assertEqual(0, torch.count_nonzero(result.reach_gate).item())
        self.assertTrue(torch.all(result.push_gate > 0.0))

    def test_prior_clamp_and_clip_mask_are_bitwise_equal(self) -> None:
        result = self._assert_live_case(
            sensor_available=True,
            terrain_height_delta=0.30,
            command_x=0.40,
            front_rear_delta=0.22,
            box_physical_targets=(0.059, 0.059, 0.010, 0.010),
        )
        self.assertTrue(torch.all(result.box_clip_mask))
        self.assertTrue(torch.equal(result.physical_targets[:, 12:14], torch.full((3, 2), 0.060)))
        self.assertEqual(0, torch.count_nonzero(result.physical_targets[:, 14:16]).item())

    def test_live_call_passes_every_prior_parameter_from_cfg(self) -> None:
        tree = ast.parse(ACTIONS_PATH.read_text(encoding="utf-8"), filename=str(ACTIONS_PATH))
        action_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PhasedHighstepBoxBiasJointPositionAction"
        )
        process = next(
            node
            for node in action_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "process_actions"
        )
        calls = [
            node
            for node in ast.walk(process)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "compute_phased_highstep_post_prior"
        ]
        self.assertEqual(1, len(calls))
        call = calls[0]
        expected_cfg_keywords = {
            "height_threshold",
            "height_gate_width",
            "min_forward_command",
            "command_gate_width",
            "commit_height_delta_min",
            "commit_height_delta_target",
            "commit_gate_floor",
            "front_reach_box_bias",
            "rear_approach_box_bias",
            "front_support_box_bias",
            "rear_push_box_bias",
            "min_box_target",
            "max_box_target",
            "prior_start_update",
            "prior_full_update",
        }
        keyword_values = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        for name in expected_cfg_keywords:
            self.assertEqual(f"self.cfg.{name}", keyword_values[name])
        self.assertEqual(
            [
                "self._raw_actions",
                "self._processed_actions",
                "self._box_action_ids",
                "self._scale",
                "self._offset",
                "height_delta",
                "command_x",
                "front_rear_delta",
            ],
            [ast.unparse(argument) for argument in call.args],
        )


if __name__ == "__main__":
    unittest.main()
