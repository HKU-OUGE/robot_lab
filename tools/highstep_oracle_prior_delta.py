#!/usr/bin/env python3
"""Fail-closed helpers for the v1.9 same-checkpoint prior-delta A/B.

This module contains no simulator launcher and no optimizer.  The runtime
wrapper calls :func:`apply_oracle_prior_delta` after independently computing
the frozen Teacher's pre/post-prior actions on the Student's current state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


MODES = ("control", "oracle_prior_delta")
BOX_SLICE = slice(12, 16)
MATRIX = tuple(
    (seed, scenario)
    for seed in (11, 22, 33)
    for scenario in ("nominal", "left_offset", "right_offset")
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_oracle_prior_delta(
    student_action: torch.Tensor,
    teacher_pre_prior: torch.Tensor,
    teacher_post_prior: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the diagnostic action and exact four-dimensional Teacher delta."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if student_action.ndim != 2 or student_action.shape[-1] != 16:
        raise ValueError(f"student action must have shape [B,16], got {tuple(student_action.shape)}")
    if teacher_pre_prior.shape != student_action.shape or teacher_post_prior.shape != student_action.shape:
        raise ValueError("Teacher pre/post-prior actions must exactly match the Student action shape")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (student_action, teacher_pre_prior, teacher_post_prior)
    ):
        raise RuntimeError("oracle prior-delta inputs contain non-finite values")

    delta4 = teacher_post_prior[:, BOX_SLICE] - teacher_pre_prior[:, BOX_SLICE]
    applied = student_action.clone()
    if mode == "oracle_prior_delta":
        applied[:, BOX_SLICE] = applied[:, BOX_SLICE] + delta4
    if not torch.equal(applied[:, :12], student_action[:, :12]):
        raise RuntimeError("oracle prior-delta changed one or more protected non-box action dimensions")
    if mode == "control" and not torch.equal(applied, student_action):
        raise RuntimeError("control mode changed the Student action")
    return applied, delta4


def scenario_from_offsets(lateral: float, yaw_deg: float) -> str:
    expected = {
        (0.0, 0.0): "nominal",
        (0.12, 4.0): "left_offset",
        (-0.12, -4.0): "right_offset",
    }
    for (wanted_lateral, wanted_yaw), name in expected.items():
        if abs(float(lateral) - wanted_lateral) <= 1.0e-9 and abs(float(yaw_deg) - wanted_yaw) <= 1.0e-9:
            return name
    raise ValueError(f"offset pair is outside the frozen core9 matrix: lateral={lateral}, yaw={yaw_deg}")


def counts_from_eval_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    identities = {(int(row["seed"]), str(row["scenario"])) for row in rows}
    if len(rows) != 9 or identities != set(MATRIX):
        raise RuntimeError("oracle A/B evaluation rows are not the exact frozen core9 matrix")
    evaluations = []
    for row in rows:
        item = row.get("eval")
        if row.get("return_code") != 0 or not isinstance(item, Mapping):
            raise RuntimeError("oracle A/B contains a failed or missing evaluation payload")
        if item.get("schema_version") != 7 or item.get("schedule_valid") is not True or item.get("reset_valid") is not True:
            raise RuntimeError("oracle A/B evaluation payload failed schema/schedule/reset validation")
        evaluations.append(item)
    return {
        "valid": 9,
        "full_climb": sum(item.get("full_climb_success") is True for item in evaluations),
        "rear_hold": sum(item.get("rear_on_platform_hold_success") is True for item in evaluations),
        "front_top_support": sum(item.get("front_top_support_reached") is True for item in evaluations),
        "no_severe_inward": sum(
            float(item.get("critical_center_violation_rate", 1.0)) <= 0.0
            and float(item.get("critical_width_violation_rate", 1.0)) <= 0.0
            for item in evaluations
        ),
    }


def oracle_recovery_decision(
    control: Mapping[str, int], oracle: Mapping[str, int]
) -> dict[str, Any]:
    required = {"valid", "full_climb", "rear_hold", "no_severe_inward"}
    if not required.issubset(control) or not required.issubset(oracle):
        raise ValueError("A/B counts lack required fields")
    control_passed = bool(
        int(control["valid"]) == 9
        and int(control["full_climb"]) >= 7
        and int(control["rear_hold"]) >= 7
        and int(control["no_severe_inward"]) >= 8
    )
    oracle_passed = bool(
        int(oracle["valid"]) == 9
        and int(oracle["full_climb"]) >= 7
        and int(oracle["rear_hold"]) >= 7
        and int(oracle["no_severe_inward"]) >= 8
    )
    strictly_improved = bool(
        int(oracle["full_climb"]) > int(control["full_climb"])
        and int(oracle["rear_hold"]) > int(control["rear_hold"])
    )
    recovered = bool(oracle_passed and not control_passed and strictly_improved)
    return {
        "control_gate_passed": control_passed,
        "oracle_gate_passed": oracle_passed,
        "oracle_strictly_improved_full_and_rear": strictly_improved,
        "recovered": recovered,
        "status": (
            "oracle_prior_delta_recovery_verified"
            if recovered
            else "oracle_prior_delta_not_recovered"
        ),
        "residual_training_allowed": recovered,
    }


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
