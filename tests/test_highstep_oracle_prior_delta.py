from __future__ import annotations

import unittest

import torch

from tools.highstep_oracle_prior_delta import (
    apply_oracle_prior_delta,
    oracle_recovery_decision,
    scenario_from_offsets,
)


class TestOraclePriorDelta(unittest.TestCase):
    def test_control_is_bit_exact_and_oracle_changes_only_box_dims(self) -> None:
        student = torch.arange(32, dtype=torch.float32).reshape(2, 16)
        teacher_pre = torch.zeros_like(student)
        teacher_post = torch.zeros_like(student)
        teacher_post[:, 12:] = torch.tensor([1.0, -2.0, 3.0, -4.0])
        control, delta = apply_oracle_prior_delta(
            student, teacher_pre, teacher_post, mode="control"
        )
        self.assertTrue(torch.equal(control, student))
        oracle, oracle_delta = apply_oracle_prior_delta(
            student, teacher_pre, teacher_post, mode="oracle_prior_delta"
        )
        self.assertTrue(torch.equal(delta, oracle_delta))
        self.assertTrue(torch.equal(oracle[:, :12], student[:, :12]))
        self.assertTrue(torch.equal(oracle[:, 12:], student[:, 12:] + oracle_delta))

    def test_shape_nonfinite_and_mode_fail_closed(self) -> None:
        good = torch.zeros(1, 16)
        with self.assertRaises(ValueError):
            apply_oracle_prior_delta(good[:, :15], good[:, :15], good[:, :15], mode="control")
        with self.assertRaises(ValueError):
            apply_oracle_prior_delta(good, good, good, mode="unknown")
        bad = good.clone()
        bad[0, 12] = float("nan")
        with self.assertRaises(RuntimeError):
            apply_oracle_prior_delta(good, good, bad, mode="oracle_prior_delta")

    def test_offsets_are_exact(self) -> None:
        self.assertEqual(scenario_from_offsets(0.0, 0.0), "nominal")
        self.assertEqual(scenario_from_offsets(0.12, 4.0), "left_offset")
        self.assertEqual(scenario_from_offsets(-0.12, -4.0), "right_offset")
        with self.assertRaises(ValueError):
            scenario_from_offsets(0.06, 2.0)

    def test_recovery_requires_oracle_gate_control_failure_and_strict_gain(self) -> None:
        control = {"valid": 9, "full_climb": 5, "rear_hold": 5, "no_severe_inward": 9}
        oracle = {"valid": 9, "full_climb": 7, "rear_hold": 8, "no_severe_inward": 8}
        self.assertTrue(oracle_recovery_decision(control, oracle)["recovered"])
        same = dict(oracle)
        self.assertFalse(oracle_recovery_decision(same, oracle)["recovered"])
        weak = dict(oracle, rear_hold=6)
        self.assertFalse(oracle_recovery_decision(control, weak)["recovered"])


if __name__ == "__main__":
    unittest.main()
