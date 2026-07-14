#!/usr/bin/env python3
"""Pure contracts for the v1.11 zero-training Teacher robustness A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SEEDS = (11, 22, 33)
INWARD_WIDTHS_M = (0.34, 0.30, 0.26, 0.22)
INWARD_MODES = ("symmetric", "rl_only", "rr_only")
IMPULSE_PHASES = ("approach", "front_support", "first_rear")
IMPULSE_DIRECTIONS = ("left", "right")
IMPULSE_LEVELS_MPS = (0.10, 0.20, 0.30)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def width_tag(width: float) -> str:
    return f"{int(round(width * 100)):02d}cm"


def dv_tag(value: float) -> str:
    return f"{int(round(value * 100)):02d}cms"


def build_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(
            {
                "run_id": f"nominal_seed{seed}",
                "family": "nominal",
                "seed": seed,
                "initial_rear_width_m": None,
                "inward_mode": "none",
                "impulse_phase": "none",
                "impulse_direction": "none",
                "impulse_delta_v_mps": 0.0,
            }
        )
    for width in INWARD_WIDTHS_M:
        for mode in INWARD_MODES:
            for seed in SEEDS:
                rows.append(
                    {
                        "run_id": f"inward_{width_tag(width)}_{mode}_seed{seed}",
                        "family": "inward",
                        "seed": seed,
                        "initial_rear_width_m": width,
                        "inward_mode": mode,
                        "impulse_phase": "none",
                        "impulse_direction": "none",
                        "impulse_delta_v_mps": 0.0,
                    }
                )
    for phase in IMPULSE_PHASES:
        for direction in IMPULSE_DIRECTIONS:
            for delta_v in IMPULSE_LEVELS_MPS:
                for seed in SEEDS:
                    rows.append(
                        {
                            "run_id": f"impulse_{phase}_{direction}_{dv_tag(delta_v)}_seed{seed}",
                            "family": "impulse",
                            "seed": seed,
                            "initial_rear_width_m": None,
                            "inward_mode": "none",
                            "impulse_phase": phase,
                            "impulse_direction": direction,
                            "impulse_delta_v_mps": delta_v,
                        }
                    )
    for mode in INWARD_MODES:
        for seed in SEEDS:
            if mode == "rl_only":
                direction = "left"
            elif mode == "rr_only":
                direction = "right"
            else:
                direction = "right" if seed == 22 else "left"
            rows.append(
                {
                    "run_id": f"combined_26cm_{mode}_{direction}_seed{seed}",
                    "family": "combined",
                    "seed": seed,
                    "initial_rear_width_m": 0.26,
                    "inward_mode": mode,
                    "impulse_phase": "first_rear",
                    "impulse_direction": direction,
                    "impulse_delta_v_mps": 0.20,
                }
            )
    run_ids = [str(row["run_id"]) for row in rows]
    if len(rows) != 102 or len(set(run_ids)) != 102:
        raise RuntimeError("v1.11 matrix must contain exactly 102 unique rows")
    return rows


def matrix_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["run_id"]): row for row in build_matrix()}


def _count(rows: Iterable[Mapping[str, Any]], key: str) -> int:
    return sum(row.get(key) is True for row in rows)


def _level_pass(rows: list[Mapping[str, Any]], *, denominator: int, threshold: int) -> bool:
    return bool(
        len(rows) == denominator
        and _count(rows, "valid") == denominator
        and _count(rows, "full_climb") >= threshold
        and _count(rows, "rear_hold") >= threshold
        and _count(rows, "recovery") >= threshold
    )


def aggregate_teacher(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    expected = matrix_by_id()
    actual = {str(row["run_id"]): row for row in rows}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"matrix mismatch: missing={missing}, extra={extra}")

    nominal = [row for row in rows if row["family"] == "nominal"]
    inward_levels: dict[str, dict[str, Any]] = {}
    hardest_inward_index = -1
    # Capability index grows with difficulty: 0=34 cm, 3=22 cm.
    for index, width in enumerate(INWARD_WIDTHS_M):
        selected = [
            row for row in rows
            if row["family"] == "inward"
            and abs(float(row["initial_rear_width_m"]) - width) <= 1.0e-9
        ]
        passed = _level_pass(selected, denominator=9, threshold=7)
        inward_levels[width_tag(width)] = {
            "valid": _count(selected, "valid"),
            "full_climb": _count(selected, "full_climb"),
            "rear_hold": _count(selected, "rear_hold"),
            "recovery": _count(selected, "recovery"),
            "passed": passed,
        }
        if passed:
            hardest_inward_index = max(hardest_inward_index, index)

    impulse_levels: dict[str, dict[str, Any]] = {}
    hardest_impulse_index = -1
    for index, delta_v in enumerate(IMPULSE_LEVELS_MPS):
        selected = [
            row for row in rows
            if row["family"] == "impulse"
            and abs(float(row["impulse_delta_v_mps"]) - delta_v) <= 1.0e-9
        ]
        passed = _level_pass(selected, denominator=18, threshold=14)
        impulse_levels[dv_tag(delta_v)] = {
            "valid": _count(selected, "valid"),
            "full_climb": _count(selected, "full_climb"),
            "rear_hold": _count(selected, "rear_hold"),
            "recovery": _count(selected, "recovery"),
            "passed": passed,
        }
        if passed:
            hardest_impulse_index = max(hardest_impulse_index, index)

    combined = [row for row in rows if row["family"] == "combined"]
    combined_counts = {
        "valid": _count(combined, "valid"),
        "full_climb": _count(combined, "full_climb"),
        "rear_hold": _count(combined, "rear_hold"),
        "recovery": _count(combined, "recovery"),
    }
    combined_counts["passed"] = bool(
        combined_counts["valid"] == 9
        and combined_counts["full_climb"] >= 7
        and combined_counts["rear_hold"] >= 7
        and combined_counts["recovery"] >= 7
    )
    return {
        "run_count": len(rows),
        "nominal": {
            "valid": _count(nominal, "valid"),
            "full_climb": _count(nominal, "full_climb"),
            "rear_hold": _count(nominal, "rear_hold"),
            "recovery": _count(nominal, "recovery"),
            "falls": _count(nominal, "fell"),
            "centerline_crossings": _count(nominal, "centerline_crossed"),
        },
        "inward_levels": inward_levels,
        "hardest_inward_level_index": hardest_inward_index,
        "hardest_inward_width_m": (
            INWARD_WIDTHS_M[hardest_inward_index] if hardest_inward_index >= 0 else None
        ),
        "impulse_levels": impulse_levels,
        "hardest_impulse_level_index": hardest_impulse_index,
        "hardest_impulse_delta_v_mps": (
            IMPULSE_LEVELS_MPS[hardest_impulse_index] if hardest_impulse_index >= 0 else None
        ),
        "combined": combined_counts,
    }


def improvement_decision(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_nominal = old["nominal"]
    new_nominal = new["nominal"]
    nominal_not_regressed = bool(
        all(int(new_nominal[key]) >= int(old_nominal[key]) for key in ("valid", "full_climb", "rear_hold", "recovery"))
        and int(new_nominal["falls"]) <= int(old_nominal["falls"])
        and int(new_nominal["centerline_crossings"]) <= int(old_nominal["centerline_crossings"])
    )
    inward_level_improved = bool(
        int(new["hardest_inward_level_index"]) >= int(old["hardest_inward_level_index"]) + 1
    )
    impulse_level_improved = bool(
        int(new["hardest_impulse_level_index"]) >= int(old["hardest_impulse_level_index"]) + 1
    )
    combined_passed = bool(new["combined"]["passed"])
    clearly_improved = bool(
        nominal_not_regressed
        and inward_level_improved
        and impulse_level_improved
        and combined_passed
    )
    return {
        "nominal_not_regressed": nominal_not_regressed,
        "inward_level_improved_by_at_least_4cm": inward_level_improved,
        "impulse_level_improved_by_at_least_0p10mps": impulse_level_improved,
        "new_teacher_combined_pressure_passed": combined_passed,
        "clearly_improved": clearly_improved,
        "status": (
            "new_teacher_robustness_clearly_improved"
            if clearly_improved
            else "new_teacher_robustness_not_clearly_improved"
        ),
    }
