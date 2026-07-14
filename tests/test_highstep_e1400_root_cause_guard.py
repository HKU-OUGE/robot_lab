from __future__ import annotations

from tools.highstep_e1400_root_cause_guard import (
    _joint_error_metrics,
    behavior_comparison,
)


def test_e1400_componentwise_baseline_requires_both_full_and_hold() -> None:
    baseline = {"counts": {"valid": 9, "full_climb": 7, "rear_hold": 7}}
    result = {"core9": {"counts": {"valid": 9, "full_climb": 8, "rear_hold": 6}}}
    decision = behavior_comparison(result, baseline)
    assert decision["root_cause_isolation_required"] is True
    assert decision["actual_behavior_score"] == 14


def test_e1400_baseline_reproduced_only_componentwise() -> None:
    baseline = {"counts": {"valid": 9, "full_climb": 7, "rear_hold": 7}}
    result = {"core9": {"counts": {"valid": 9, "full_climb": 7, "rear_hold": 8}}}
    assert behavior_comparison(result, baseline)["reproduced_0707_behavior"] is True


def test_three_path_metrics_include_joint_q95_sign_and_first_difference() -> None:
    def row(step: int, left: list[float], right: list[float]):
        return {"step": step, "phase": "approach", "student_action": left,
                "teacher_action_pre_prior": right}
    zeros = [0.1] * 16
    runs = [[row(0, zeros, zeros), row(1, [0.2] * 16, zeros)]]
    metrics = _joint_error_metrics(
        runs, "approach", "student_action", "teacher_action_pre_prior"
    )
    assert len(metrics["joint_mae"]) == 16
    assert len(metrics["joint_q95_abs_error"]) == 16
    assert len(metrics["joint_sign_mismatch_rate"]) == 16
    assert len(metrics["action_first_difference_error_q95"]) == 16
    assert metrics["action_first_difference_error_mae"][0] == 0.1
