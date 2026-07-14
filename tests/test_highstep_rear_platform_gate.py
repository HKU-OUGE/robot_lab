"""CPU-only contract tests for the rear-on-platform release gate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
MONITOR = ROOT / "tmp/highstep_centerline_guard_monitor_20260709.sh"
POLICY = ROOT / "tmp/highstep_automation_policy.json"


def _tracker_method(name: str) -> ast.FunctionDef:
    tree = ast.parse(PLAY.read_text(encoding="utf-8"), filename=str(PLAY))
    tracker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_HighstepEvalTracker"
    )
    return next(
        node
        for node in tracker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class RearPlatformTrackerContractTests(unittest.TestCase):
    def test_handoff_condition_has_no_front_foot_dependency(self) -> None:
        update = _tracker_method("update")
        assignment = next(
            node
            for node in ast.walk(update)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "rear_on_platform" for target in node.targets)
        )
        source = ast.unparse(assignment.value)
        self.assertIn("rear_bilateral_top", source)
        self.assertIn("rear_edge_margin", source)
        self.assertIn("root_edge_margin", source)
        self.assertIn("root_h_top", source)
        self.assertIn("rear_width", source)
        self.assertIn("rear_min_abs_y", source)
        self.assertNotIn("front_", source)
        self.assertNotIn("rear_advance", source)

    def test_fixed_stand_window_uses_25_steps_and_four_centimeter_rear_margin(self) -> None:
        init_source = ast.unparse(_tracker_method("__init__"))
        self.assertIn("self.rear_on_platform_hold_confirm_steps = 25", init_source)
        self.assertIn("self.rear_on_platform_min_root_edge_margin = 0.2", init_source)
        self.assertIn("self.rear_on_platform_min_rear_edge_margin = 0.04", init_source)
        self.assertIn("self.rear_on_platform_min_root_h_top = 0.28", init_source)
        self.assertIn("self.rear_on_platform_min_rear_width = 0.18", init_source)
        self.assertIn("self.rear_on_platform_min_rear_abs_y = 0.04", init_source)

    def test_front_contact_is_exported_only_as_a_diagnostic(self) -> None:
        summary_source = ast.unparse(_tracker_method("summary"))
        self.assertIn("rear_on_platform_hold_success", summary_source)
        self.assertIn("front_bilateral_top_contact_rate_rear_on_platform", summary_source)


class RearPlatformAutomationGateTests(unittest.TestCase):
    def test_monitor_pass_gate_uses_rear_handoff_not_front_or_rear_advance(self) -> None:
        source = MONITOR.read_text(encoding="utf-8")
        start = source.index("def row_pass(ev):")
        end = source.index("\nsummaries = []", start)
        gate = source[start:end]
        self.assertIn('ev.get("rear_on_platform_hold_success") is True', gate)
        self.assertIn('ev.get("rear_clear_reached") is True', gate)
        self.assertIn('ev.get("terminated_early") is False', gate)
        self.assertNotIn("kinematic_top_hold_success", gate)
        self.assertNotIn("strict_full_climb_success", gate)
        self.assertNotIn("front_bilateral", gate)
        self.assertNotIn("rear_advance_reached", gate)
        self.assertNotIn("rear_edge_dwell_max_consecutive", gate)
        sort_start = source.index("summaries.sort(")
        sort_end = source.index("\nselected =", sort_start)
        selection = source[sort_start:sort_end]
        self.assertIn('s["rear_on_platform_hold_steps_worst"]', selection)
        self.assertNotIn('s["top_hold_rate"]', selection)

    def test_policy_has_explicit_teacher_and_student_rear_hold_rates(self) -> None:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        thresholds = policy["thresholds"]
        self.assertEqual(0.6, thresholds["teacher"]["rear_on_platform_hold_rate"])
        self.assertEqual(0.8, thresholds["student"]["rear_on_platform_hold_rate"])
        self.assertNotIn("max_rear_edge_dwell_average", thresholds["student"])
        self.assertNotIn("max_teacher_dwell_average_ratio", thresholds["student"])


if __name__ == "__main__":
    unittest.main()
