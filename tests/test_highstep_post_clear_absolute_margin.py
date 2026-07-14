"""CPU-only tests for the post-clear absolute platform-margin target mode."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
REWARDS = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/rewards.py"
HIGHSTEP_CFG = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg/highstep_env_cfg.py"
)


def _load_reward_functions() -> dict[str, object]:
    tree = ast.parse(REWARDS.read_text(encoding="utf-8"), filename=str(REWARDS))
    wanted = {
        "_post_clear_absolute_rear_margin",
        "_update_post_clear_near_edge_counter",
        "post_clear_rear_advance_stall_penalty",
    }
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise AssertionError("post-clear reward helper extraction is incomplete")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)]
        + functions,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "torch": torch,
        "_ensure_highstep_rear_branch_buffers": lambda env: None,
    }
    exec(compile(module, str(REWARDS), "exec"), namespace)
    return namespace


FUNCTIONS = _load_reward_functions()
absolute_margin = FUNCTIONS["_post_clear_absolute_rear_margin"]
update_counter = FUNCTIONS["_update_post_clear_near_edge_counter"]
stall_penalty = FUNCTIONS["post_clear_rear_advance_stall_penalty"]


class _FakeScene(dict):
    def __init__(self, asset, origins: torch.Tensor):
        super().__init__(robot=asset)
        self.env_origins = origins


class _FakeAsset:
    def __init__(self, rear_xy: torch.Tensor):
        zeros = torch.zeros((*rear_xy.shape[:-1], 1), dtype=rear_xy.dtype)
        self.data = SimpleNamespace(body_pos_w=torch.cat((rear_xy, zeros), dim=-1))

    def find_bodies(self, _names):
        return torch.tensor([0, 1]), ["RL_foot", "RR_foot"]


class _FakeCommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, _name):
        return self.command


class _FakeEnv:
    def __init__(self, rear_xy: torch.Tensor, heading: torch.Tensor, command_x: float = 0.33):
        count = rear_xy.shape[0]
        self.scene = _FakeScene(_FakeAsset(rear_xy), torch.zeros((count, 3)))
        command = torch.zeros((count, 3), dtype=rear_xy.dtype)
        command[:, 0] = command_x
        self.command_manager = _FakeCommandManager(command)
        self._highstep_post_clear_origin_valid = torch.ones(count, dtype=torch.bool)
        self._highstep_post_clear_heading_w = heading.clone()
        self._highstep_post_clear_stall_near_edge_counter = torch.zeros(count)
        self._highstep_post_clear_stall_sample_steps = torch.zeros(count)
        self._highstep_post_clear_stall_margin_sum = torch.zeros(count)
        self._highstep_post_clear_stall_margin_min = torch.full((count,), 10.0)
        self._highstep_post_clear_stall_counter_max = torch.zeros(count)
        self._highstep_post_clear_stall_signal_sum = torch.zeros(count)


def _rear_positions_for_margin(
    heading: torch.Tensor,
    margins: tuple[float, float],
    origin: torch.Tensor | None = None,
    half_width: float = 1.5,
) -> torch.Tensor:
    heading = heading / torch.linalg.vector_norm(heading)
    edge_distance = half_width / torch.amax(torch.abs(heading))
    origin = torch.zeros(2) if origin is None else origin
    outward_edge = origin - heading * edge_distance
    return torch.stack([outward_edge + heading * margin for margin in margins])


class PostClearAbsoluteMarginTensorTests(unittest.TestCase):
    def test_square_entry_geometry_for_x_and_y_both_directions(self) -> None:
        headings = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        origins = torch.tensor([[0.0, 0.0], [2.0, -3.0], [-1.0, 4.0], [3.0, 2.0]])
        rear = torch.stack(
            [
                _rear_positions_for_margin(heading, (0.07, 0.21), origin)
                for heading, origin in zip(headings, origins)
            ]
        )
        actual = absolute_margin(rear, origins, headings, 1.5)
        torch.testing.assert_close(actual, torch.full((4,), 0.07), atol=1.0e-6, rtol=0.0)

    def test_square_entry_geometry_is_exact_at_four_degree_yaw(self) -> None:
        angle = math.radians(4.0)
        heading = torch.tensor([[math.cos(angle), math.sin(angle)]])
        origin = torch.tensor([[3.0, -2.0]])
        rear = _rear_positions_for_margin(heading[0], (0.04, 0.18), origin[0])[None]
        actual = absolute_margin(rear, origin, heading, 1.5)
        torch.testing.assert_close(actual, torch.tensor([0.04]), atol=1.0e-6, rtol=0.0)

    def test_counter_requires_consecutive_valid_commanded_near_edge_steps(self) -> None:
        counter = torch.tensor([7.0, 7.0, 7.0, 7.0])
        updated = update_counter(
            counter,
            torch.tensor([True, False, True, True]),
            torch.tensor([True, True, False, True]),
            torch.tensor([0.05, 0.05, 0.05, 0.12]),
            0.12,
        )
        # The threshold is strict: exactly 0.12 m is no longer near-edge.
        torch.testing.assert_close(updated, torch.tensor([8.0, 0.0, 0.0, 0.0]))

    def test_safe_margin_has_zero_penalty_and_resets_counter(self) -> None:
        heading = torch.tensor([[1.0, 0.0]])
        rear = _rear_positions_for_margin(heading[0], (0.04, 0.06))[None]
        env = _FakeEnv(rear, heading)
        asset_cfg = SimpleNamespace(name="robot")

        signal = None
        for _ in range(40):
            signal = stall_penalty(env, "base_velocity", asset_cfg, ["RL_foot", "RR_foot"])
        self.assertIsNotNone(signal)
        self.assertAlmostEqual(1.0, float(signal.item()), places=6)
        self.assertEqual(40.0, float(env._highstep_post_clear_stall_near_edge_counter.item()))

        safe_rear = _rear_positions_for_margin(heading[0], (0.18, 0.24))[None]
        env.scene["robot"].data.body_pos_w[:, :, :2] = safe_rear
        safe_signal = stall_penalty(env, "base_velocity", asset_cfg, ["RL_foot", "RR_foot"])
        self.assertEqual(0.0, float(safe_signal.item()))
        self.assertEqual(0.0, float(env._highstep_post_clear_stall_near_edge_counter.item()))

    def test_counter_resets_when_command_or_post_clear_validity_disappears(self) -> None:
        heading = torch.tensor([[1.0, 0.0]])
        rear = _rear_positions_for_margin(heading[0], (0.04, 0.06))[None]
        env = _FakeEnv(rear, heading)
        asset_cfg = SimpleNamespace(name="robot")
        stall_penalty(env, "base_velocity", asset_cfg, ["RL_foot", "RR_foot"])
        self.assertEqual(1.0, float(env._highstep_post_clear_stall_near_edge_counter.item()))

        env.command_manager.command[:, 0] = 0.08
        stall_penalty(env, "base_velocity", asset_cfg, ["RL_foot", "RR_foot"])
        self.assertEqual(0.0, float(env._highstep_post_clear_stall_near_edge_counter.item()))

        env.command_manager.command[:, 0] = 0.33
        env._highstep_post_clear_origin_valid[:] = False
        stall_penalty(env, "base_velocity", asset_cfg, ["RL_foot", "RR_foot"])
        self.assertEqual(0.0, float(env._highstep_post_clear_stall_near_edge_counter.item()))


class PostClearAbsoluteMarginStaticTests(unittest.TestCase):
    def test_standard_action_score_configuration_uses_one_target_mode(self) -> None:
        source = HIGHSTEP_CFG.read_text(encoding="utf-8")
        self.assertNotIn('"min_rear_advance"', source)
        self.assertNotIn('"target_rear_advance"', source)
        self.assertGreaterEqual(source.count('"platform_half_width": 1.5'), 4)
        self.assertGreaterEqual(source.count('"min_rear_margin": 0.04'), 4)
        self.assertGreaterEqual(source.count('"target_rear_margin": 0.18'), 4)
        self.assertGreaterEqual(source.count('"near_edge_margin": 0.12'), 2)
        self.assertIn("self.rewards.post_clear_rear_advance_stall.weight = -0.50", source)

    def test_recovery_component_uses_absolute_margin_not_relative_displacement(self) -> None:
        tree = ast.parse(REWARDS.read_text(encoding="utf-8"), filename=str(REWARDS))
        recovery = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "post_clear_recovery_bonus"
        )
        recovery_source = ast.unparse(recovery)
        self.assertIn("_post_clear_absolute_rear_margin", recovery_source)
        self.assertNotIn("rear_displacement", recovery_source)
        self.assertIn("rear_advance_weight", recovery_source)


if __name__ == "__main__":
    unittest.main()
