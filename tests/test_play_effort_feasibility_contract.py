"""CPU-only contracts for the opt-in play effort-limit feasibility probe."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
SOURCE = PLAY.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(PLAY))


def _top_level(name: str, node_type=ast.FunctionDef):
    return next(
        node
        for node in TREE.body
        if isinstance(node, node_type) and getattr(node, "name", None) == name
    )


def _load_override_helpers():
    wanted_assignments = {
        "_EVAL_EFFORT_ACTUATOR_GROUPS",
        "_EVAL_EFFORT_EXPECTED_JOINTS",
    }
    selected = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_assignments
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_normalize_eval_effort_limits",
            "_apply_eval_effort_limit_override",
        }:
            selected.append(node)
    namespace = {"math": math, "json": json}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PLAY), "exec"), namespace)
    return namespace


def _fake_env_cfg():
    actuators = {
        "legs_hip": SimpleNamespace(
            effort_limit=35.0, saturation_effort=50.0, effort_limit_sim=None
        ),
        "legs_thigh": SimpleNamespace(
            effort_limit=35.0, saturation_effort=50.0, effort_limit_sim=None
        ),
        "legs_calf": SimpleNamespace(
            effort_limit=80.0, saturation_effort=100.0, effort_limit_sim=None
        ),
        "extensions": SimpleNamespace(effort_limit=1000.0),
    }
    return SimpleNamespace(
        scene=SimpleNamespace(robot=SimpleNamespace(actuators=actuators)),
        actions=SimpleNamespace(
            joint_pos=SimpleNamespace(scale=0.1, clip={".*": (-60.0, 60.0)})
        ),
    )


class EvalEffortOverrideUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.helpers = _load_override_helpers()
        cls.apply = staticmethod(cls.helpers["_apply_eval_effort_limit_override"])

    def test_default_none_is_a_strict_noop(self) -> None:
        cfg = _fake_env_cfg()
        before = {
            name: vars(actuator).copy()
            for name, actuator in cfg.scene.robot.actuators.items()
        }
        self.assertIsNone(self.apply(cfg, None))
        self.assertEqual(
            before,
            {name: vars(actuator) for name, actuator in cfg.scene.robot.actuators.items()},
        )

    def test_atomic_override_changes_only_three_effort_limit_fields(self) -> None:
        cfg = _fake_env_cfg()
        action_scale = cfg.actions.joint_pos.scale
        joint_clip = cfg.actions.joint_pos.clip.copy()
        requested = self.apply(cfg, [40.0, 42.0, 45.0])
        self.assertEqual({"hip": 40.0, "thigh": 42.0, "calf": 45.0}, requested)
        self.assertEqual(40.0, cfg.scene.robot.actuators["legs_hip"].effort_limit)
        self.assertEqual(42.0, cfg.scene.robot.actuators["legs_thigh"].effort_limit)
        self.assertEqual(45.0, cfg.scene.robot.actuators["legs_calf"].effort_limit)
        self.assertEqual(50.0, cfg.scene.robot.actuators["legs_hip"].saturation_effort)
        self.assertEqual(50.0, cfg.scene.robot.actuators["legs_thigh"].saturation_effort)
        self.assertEqual(100.0, cfg.scene.robot.actuators["legs_calf"].saturation_effort)
        self.assertIsNone(cfg.scene.robot.actuators["legs_hip"].effort_limit_sim)
        self.assertIsNone(cfg.scene.robot.actuators["legs_thigh"].effort_limit_sim)
        self.assertIsNone(cfg.scene.robot.actuators["legs_calf"].effort_limit_sim)
        self.assertEqual(1000.0, cfg.scene.robot.actuators["extensions"].effort_limit)
        self.assertEqual(action_scale, cfg.actions.joint_pos.scale)
        self.assertEqual(joint_clip, cfg.actions.joint_pos.clip)

    def test_invalid_or_partial_requests_fail_before_mutation(self) -> None:
        for values in (
            [40.0, 40.0],
            [40.0, 0.0, 40.0],
            [40.0, math.inf, 40.0],
            [51.0, 40.0, 40.0],
        ):
            cfg = _fake_env_cfg()
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.apply(cfg, values)
            self.assertEqual(35.0, cfg.scene.robot.actuators["legs_hip"].effort_limit)
            self.assertEqual(35.0, cfg.scene.robot.actuators["legs_thigh"].effort_limit)
            self.assertEqual(80.0, cfg.scene.robot.actuators["legs_calf"].effort_limit)

    def test_missing_actuator_group_fails_before_any_group_is_changed(self) -> None:
        cfg = _fake_env_cfg()
        del cfg.scene.robot.actuators["legs_thigh"]
        with self.assertRaises(ValueError):
            self.apply(cfg, [40.0, 40.0, 40.0])
        self.assertEqual(35.0, cfg.scene.robot.actuators["legs_hip"].effort_limit)
        self.assertEqual(80.0, cfg.scene.robot.actuators["legs_calf"].effort_limit)

    def test_invalid_actuator_type_fails_before_any_group_is_changed(self) -> None:
        cfg = _fake_env_cfg()
        del cfg.scene.robot.actuators["legs_thigh"].saturation_effort
        with self.assertRaises(ValueError):
            self.apply(cfg, [40.0, 40.0, 40.0])
        self.assertEqual(35.0, cfg.scene.robot.actuators["legs_hip"].effort_limit)
        self.assertEqual(35.0, cfg.scene.robot.actuators["legs_thigh"].effort_limit)
        self.assertEqual(80.0, cfg.scene.robot.actuators["legs_calf"].effort_limit)


class EvalEffortStaticContractTests(unittest.TestCase):
    def test_cli_is_explicit_atomic_and_disabled_by_default(self) -> None:
        call = next(
            node
            for node in ast.walk(TREE)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--eval_effort_limits"
        )
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertEqual(3, ast.literal_eval(keywords["nargs"]))
        self.assertIsNone(ast.literal_eval(keywords["default"]))

    def test_override_is_applied_before_environment_creation(self) -> None:
        main = _top_level("main")
        apply_call = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_apply_eval_effort_limit_override"
        )
        gym_make = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "gym"
            and node.func.attr == "make"
        )
        self.assertLess(apply_call.lineno, gym_make.lineno)

    def test_override_helper_has_only_effort_limit_attribute_assignment(self) -> None:
        helper = _top_level("_apply_eval_effort_limit_override")
        attribute_targets = [
            ast.unparse(target)
            for node in ast.walk(helper)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
            )
            if isinstance(target, ast.Attribute)
        ]
        self.assertEqual(["actuator_cfg.effort_limit"], attribute_targets)

    def test_runtime_tracker_exports_requested_verified_and_rear_torque_metrics(self) -> None:
        tracker = _top_level("_EvalEffortFeasibilityTracker", ast.ClassDef)
        tracker_source = ast.unparse(tracker)
        for required in (
            "eval_effort_limit_requested_nm",
            "eval_effort_limit_runtime_nm_by_joint",
            "eval_effort_limit_runtime_match",
            "eval_effort_limit_runtime_saturation_nm_by_joint",
            "applied_torque_abs_q95_nm_by_joint",
            "applied_torque_abs_max_nm_by_joint",
            "applied_torque_near_effort_limit_fraction_by_joint",
            "rear_applied_torque_abs_q95_nm_by_type",
            "rear_applied_torque_abs_max_nm_by_type",
            "rear_applied_torque_near_effort_limit_fraction_by_type",
            "RL_",
            "RR_",
            "getattr(self.asset.data, 'applied_torque'",
            "if not self.runtime_match:\n            return None",
            "self.NEAR_CAP_RATIO * self.runtime_by_joint[name]",
            "not the velocity-dependent instantaneous DC-motor cap",
            "getattr(self.asset.data, 'joint_vel'",
            "joint_velocity_cap_override_requested",
            "joint_velocity_reference_limits_are_diagnostic_only",
            "joint_velocity_abs_q95_rad_s_by_joint",
            "joint_velocity_abs_max_rad_s_by_joint",
            "joint_velocity_reference_exceedance_fraction_by_joint",
            "rear_joint_velocity_abs_q95_rad_s_by_type",
            "rear_joint_velocity_abs_max_rad_s_by_type",
            "rear_joint_velocity_reference_exceedance_fraction_by_type",
        ):
            self.assertIn(required, tracker_source)

    def test_torque_is_sampled_after_step_and_merged_into_eval_payload(self) -> None:
        main_source = ast.unparse(_top_level("main"))
        # Do not depend on ast.unparse choosing parenthesized or bare tuple targets.
        step_offset = main_source.index("env.step(actions)")
        sample_offset = main_source.index("effort_feasibility_tracker.sample_applied_torque()")
        merge_offset = main_source.index(
            "highstep_eval_summary.update(effort_feasibility_tracker.summary())"
        )
        json_offset = main_source.rindex("'[HIGHSTEP_EVAL_JSON] '")
        self.assertLess(step_offset, sample_offset)
        self.assertLess(sample_offset, merge_offset)
        self.assertLess(merge_offset, json_offset)
        self.assertIn("'eval_effort_limit_override_requested': False", main_source)
        self.assertIn("'eval_effort_limit_runtime_match': None", main_source)


if __name__ == "__main__":
    unittest.main()
