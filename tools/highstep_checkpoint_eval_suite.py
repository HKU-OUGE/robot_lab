#!/usr/bin/env python3
"""Score highstep teacher/student checkpoints from local TensorBoard events.

This is a scalar proxy evaluator, not a replacement for fixed play/video eval.
Its job is to make checkpoint selection auditable and to mark missing coverage.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass(frozen=True)
class MetricSpec:
    key: str
    tag: str
    phase: str
    target: float
    direction: str
    weight: float
    hard_min: float | None = None
    hard_max: float | None = None


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("terrain", "Curriculum/terrain_levels", "curriculum", 2.7, "higher", 4.0),
    MetricSpec("command", "Curriculum/command_levels", "curriculum", 0.6375, "higher", 2.0),
    MetricSpec("action_total", "Curriculum/highstep_action_score/total", "curriculum", 0.55, "higher", 6.0),
    MetricSpec("bad_orientation", "Episode_Termination/bad_orientation", "stability_gait", 0.003, "lower", 12.0, hard_max=0.006),
    MetricSpec("time_out", "Episode_Termination/time_out", "stability_gait", 0.995, "higher", 6.0, hard_min=0.98),
    MetricSpec("front_reach", "Episode_Reward/front_legs_reach", "front_commit", 0.085, "higher", 2.0),
    MetricSpec("front_clear", "Episode_Reward/front_feet_highstep_clearance", "front_commit", 0.055, "higher", 2.0),
    MetricSpec("rear_clear", "Episode_Reward/rear_feet_highstep_clearance", "rear_first_clear", 0.14, "higher", 16.0, hard_min=0.08),
    MetricSpec(
        "rear_first",
        "Episode_Reward/rear_first_foot_highstep_preclearance",
        "rear_first_clear",
        0.018,
        "higher",
        12.0,
        hard_min=0.006,
    ),
    MetricSpec(
        "rear_second",
        "Episode_Reward/rear_second_foot_highstep_clearance",
        "second_rear_clear",
        0.032,
        "higher",
        14.0,
        hard_min=0.010,
    ),
    MetricSpec("under_step", "Episode_Reward/rear_feet_under_step_after_commit", "second_rear_clear", -0.018, "higher", 4.0),
    MetricSpec("deadline", "Episode_Reward/second_rear_clear_deadline", "second_rear_clear", 0.0018, "higher", 6.0),
    MetricSpec("stall_penalty", "Episode_Reward/one_sided_rear_stall_time", "second_rear_clear", -0.020, "higher", 5.0),
    MetricSpec("lead_drive", "Episode_Reward/lead_rear_support_drive", "lead_rear_support", 0.030, "higher", 16.0, hard_min=0.018),
    MetricSpec("post_drive", "Episode_Reward/post_lead_body_drive", "lead_rear_support", 0.0024, "higher", 8.0),
    MetricSpec("rear_drive", "Episode_Reward/rear_legs_drive_bonus", "lead_rear_support", 0.12, "higher", 10.0, hard_min=0.08),
    MetricSpec("box_push", "Episode_Reward/highstep_rear_box_push", "box_action", 0.080, "higher", 10.0, hard_min=0.050),
    MetricSpec("phase_prior", "Episode_Reward/highstep_box_phase_prior", "box_action", 0.115, "higher", 8.0, hard_min=0.070),
    MetricSpec("support_contact", "Episode_Reward/highstep_leg_support_contact", "lead_rear_support", 0.065, "higher", 4.0),
    MetricSpec("action_support", "Curriculum/highstep_action_score/support_score", "lead_rear_support", 0.45, "higher", 14.0, hard_min=0.08),
    MetricSpec(
        "action_support_violation",
        "Curriculum/highstep_action_score/support_floor_violation_rate",
        "lead_rear_support",
        0.35,
        "lower",
        8.0,
        hard_max=0.95,
    ),
    MetricSpec(
        "lead_support_height_active",
        "Curriculum/highstep_action_score/lead_support_drive_height_active_mean",
        "lead_rear_support",
        0.35,
        "higher",
        6.0,
    ),
    MetricSpec(
        "lead_support_body_active",
        "Curriculum/highstep_action_score/lead_support_drive_body_active_mean",
        "lead_rear_support",
        0.35,
        "higher",
        6.0,
    ),
    MetricSpec(
        "lead_support_signal",
        "Curriculum/highstep_action_score/lead_support_drive_signal_mean",
        "lead_rear_support",
        0.0015,
        "higher",
        5.0,
    ),
    MetricSpec(
        "second_clear_rate",
        "Curriculum/highstep_rear_branch_metrics/second_clear_rate",
        "second_rear_clear",
        0.88,
        "higher",
        14.0,
        hard_min=0.72,
    ),
    MetricSpec(
        "one_sided_stall_ratio",
        "Curriculum/highstep_rear_branch_metrics/one_sided_stall_ratio",
        "second_rear_clear",
        0.18,
        "lower",
        10.0,
        hard_max=0.30,
    ),
    MetricSpec("valid_rate", "Curriculum/highstep_rear_branch_metrics/valid_rate", "coverage", 0.85, "higher", 5.0),
    MetricSpec("rl_first", "Curriculum/highstep_rear_branch_metrics/rl_first_rate", "branch_balance", 0.25, "higher", 1.0),
    MetricSpec("rr_first", "Curriculum/highstep_rear_branch_metrics/rr_first_rate", "branch_balance", 0.20, "higher", 1.0),
    MetricSpec("rl_minus_rr", "Curriculum/highstep_rear_branch_metrics/rl_minus_rr_first_rate", "branch_balance", 0.10, "lower_abs", 3.0),
    MetricSpec("both_first", "Curriculum/highstep_rear_branch_metrics/both_first_rate", "branch_balance", 0.45, "higher", 2.0),
    MetricSpec("post_signal", "Curriculum/highstep_rear_branch_metrics/post_lead_drive_signal_mean", "lead_rear_support", 0.0020, "higher", 5.0),
    MetricSpec("fl_overlift_rate", "Curriculum/highstep_rear_branch_metrics/fl_forward_flat_overlift_rate", "stability_gait", 0.02, "lower", 8.0, hard_max=0.08),
    MetricSpec("fl_lift", "Curriculum/highstep_rear_branch_metrics/fl_forward_flat_lift_mean", "stability_gait", 0.08, "lower", 4.0),
    MetricSpec("fl_minus_fr", "Curriculum/highstep_rear_branch_metrics/fl_minus_fr_forward_flat_mean", "stability_gait", 0.025, "lower_abs", 5.0),
    MetricSpec("mu_oob", "Loss/Debug/Mu_Out_Of_Bounds_Ratio", "student_distill", 0.10, "lower", 6.0, hard_max=0.20),
    MetricSpec("teacher_mse", "Loss/Loss/Teacher_Action_MSE", "student_distill", 0.10, "lower", 4.0),
    MetricSpec("latent_mse", "Loss/Loss/Distill_Latent_MSE", "student_distill", 0.12, "lower", 3.0),
    MetricSpec("phase_teacher_mse", "Loss/Highstep_Phase_Teacher_Action_MSE", "student_phase_distill", 0.10, "lower", 8.0),
    MetricSpec("phase_rear_box_mse", "Loss/Highstep_Rear_Box_Action_MSE", "student_phase_distill", 0.05, "lower", 8.0),
)

STAGE_ORDER = (
    "curriculum",
    "front_commit",
    "rear_first_clear",
    "lead_rear_support",
    "second_rear_clear",
    "box_action",
    "branch_balance",
    "stability_gait",
    "student_distill",
    "student_phase_distill",
)

HIGHSTEP_ACTION_KEYS = {
    "front_reach",
    "front_clear",
    "rear_clear",
    "rear_first",
    "rear_second",
    "lead_drive",
    "post_drive",
    "rear_drive",
    "box_push",
    "phase_prior",
    "support_contact",
    "action_support",
    "action_support_violation",
    "lead_support_height_active",
    "lead_support_body_active",
    "lead_support_signal",
    "second_clear_rate",
    "one_sided_stall_ratio",
}

REAR_BRANCH_PREFIX = "Curriculum/highstep_rear_branch_metrics/"
ACTION_SCORE_PREFIX = "Curriculum/highstep_action_score/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="RSL-RL run directory containing event file(s).")
    parser.add_argument("--targets", type=int, nargs="*", default=None, help="Checkpoint steps to score, e.g. 134100 145799.")
    parser.add_argument("--window", type=int, default=200, help="Number of scalar samples used for tail mean before each target.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Score all model_*.pt files in the run directory instead of only the latest 12.",
    )
    return parser.parse_args()


def latest_event(run_dir: Path) -> Path:
    events = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    return events[-1]


def checkpoint_steps(run_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(set(steps))


def load_scalars(event_path: Path) -> dict[str, list[tuple[int, float]]]:
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    scalars: dict[str, list[tuple[int, float]]] = {}
    for spec in METRICS:
        for tag in candidate_tags(spec.tag):
            if tag in tags:
                scalars[spec.key] = [(event.step, float(event.value)) for event in acc.Scalars(tag)]
                break
    return scalars


def candidate_tags(tag: str) -> tuple[str, ...]:
    tags = [tag]
    if tag.startswith(REAR_BRANCH_PREFIX):
        tags.append(ACTION_SCORE_PREFIX + tag[len(REAR_BRANCH_PREFIX) :])
    return tuple(dict.fromkeys(tags))


def tail_mean(values: list[tuple[int, float]], target: int, window: int) -> tuple[int, float, float] | None:
    selected = [(step, value) for step, value in values if step <= target]
    if not selected:
        return None
    tail = selected[-window:]
    mean = sum(value for _, value in tail) / len(tail)
    last_step, last_value = selected[-1]
    return last_step, last_value, mean


def metric_score(spec: MetricSpec, value: float | None) -> float | None:
    if value is None:
        return None
    if spec.direction == "higher":
        if spec.target <= 0:
            return 1.0
        return max(0.0, min(value / spec.target, 1.25))
    if spec.direction == "lower":
        if value <= spec.target:
            return 1.0
        if spec.target <= 0:
            return 0.0
        return max(0.0, 1.0 - (value - spec.target) / spec.target)
    if spec.direction == "lower_abs":
        value = abs(value)
        if value <= spec.target:
            return 1.0
        return max(0.0, 1.0 - (value - spec.target) / max(spec.target, 1e-6))
    raise ValueError(f"Unknown direction: {spec.direction}")


def hard_failures(means: dict[str, float | None]) -> list[str]:
    failures: list[str] = []
    for spec in METRICS:
        value = means.get(spec.key)
        if value is None:
            continue
        if spec.hard_min is not None and value < spec.hard_min:
            failures.append(f"{spec.key}={value:.6g}<hard_min={spec.hard_min:.6g}")
        if spec.hard_max is not None and value > spec.hard_max:
            failures.append(f"{spec.key}={value:.6g}>hard_max={spec.hard_max:.6g}")
    return failures


def weighted_score(values: dict[str, float | None], keys: set[str] | None = None) -> float | None:
    total = 0.0
    total_weight = 0.0
    for spec in METRICS:
        if keys is not None and spec.key not in keys:
            continue
        score = metric_score(spec, values.get(spec.key))
        if score is None:
            continue
        total += score * spec.weight
        total_weight += spec.weight
    return None if total_weight == 0.0 else 100.0 * total / total_weight


def score_target(scalars: dict[str, list[tuple[int, float]]], target: int, window: int) -> dict[str, Any]:
    means: dict[str, float | None] = {}
    lasts: dict[str, float | None] = {}
    last_steps: dict[str, int | None] = {}
    for spec in METRICS:
        values = scalars.get(spec.key)
        result = tail_mean(values, target, window) if values else None
        if result is None:
            last_steps[spec.key] = None
            lasts[spec.key] = None
            means[spec.key] = None
        else:
            last_step, last_value, mean = result
            last_steps[spec.key] = last_step
            lasts[spec.key] = last_value
            means[spec.key] = mean

    phase_scores: dict[str, float | None] = {}
    phase_missing: dict[str, list[str]] = {}
    total = 0.0
    total_weight = 0.0
    for phase in STAGE_ORDER:
        specs = [spec for spec in METRICS if spec.phase == phase]
        weighted = 0.0
        weight = 0.0
        missing: list[str] = []
        for spec in specs:
            score = metric_score(spec, means.get(spec.key))
            if score is None:
                missing.append(spec.key)
                continue
            weighted += score * spec.weight
            weight += spec.weight
        phase_scores[phase] = None if weight == 0 else weighted / weight
        phase_missing[phase] = missing
        if weight > 0:
            total += weighted
            total_weight += weight

    return {
        "target": target,
        "score": None if total_weight == 0 else 100.0 * total / total_weight,
        "last_score": weighted_score(lasts),
        "highstep_action_mean_score": weighted_score(means, HIGHSTEP_ACTION_KEYS),
        "highstep_action_last_score": weighted_score(lasts, HIGHSTEP_ACTION_KEYS),
        "hard_failures": hard_failures(means),
        "means": means,
        "lasts": lasts,
        "last_steps": last_steps,
        "phase_scores": phase_scores,
        "phase_missing": phase_missing,
    }


def fmt(value: float | None) -> str:
    return "MISSING" if value is None else f"{value:.4g}"


def print_report(report: dict[str, Any]) -> None:
    print(f"run_dir: {report['run_dir']}")
    print(f"event: {report['event']}")
    print(f"window: {report['window']}")
    print()
    print(
        "checkpoint\tmean_score\tlast_score\thighstep_mean\thighstep_last\thard_failures\t"
        "rear_clear_mean\trear_clear_last\trear_first_last\tlead_drive_last\trear_second_last\t"
        "box_push_last\taction_support_mean\tsupport_violation_mean\tsecond_rate_mean\tone_sided_mean\t"
        "bad_ori_mean\tmu_oob_mean\tteacher_mse_mean"
    )
    for item in report["checkpoints"]:
        means = item["means"]
        lasts = item["lasts"]
        failures = ";".join(item["hard_failures"]) or "-"
        print(
            "\t".join(
                [
                    str(item["target"]),
                    fmt(item["score"]),
                    fmt(item["last_score"]),
                    fmt(item["highstep_action_mean_score"]),
                    fmt(item["highstep_action_last_score"]),
                    failures,
                    fmt(means.get("rear_clear")),
                    fmt(lasts.get("rear_clear")),
                    fmt(lasts.get("rear_first")),
                    fmt(lasts.get("lead_drive")),
                    fmt(lasts.get("rear_second")),
                    fmt(lasts.get("box_push")),
                    fmt(means.get("action_support")),
                    fmt(means.get("action_support_violation")),
                    fmt(means.get("second_clear_rate")),
                    fmt(means.get("one_sided_stall_ratio")),
                    fmt(means.get("bad_orientation")),
                    fmt(means.get("mu_oob")),
                    fmt(means.get("teacher_mse")),
                ]
            )
        )
    print()
    print("phase coverage and scores:")
    for item in report["checkpoints"]:
        print(f"- checkpoint {item['target']}:")
        for phase in STAGE_ORDER:
            missing = item["phase_missing"].get(phase, [])
            missing_text = "none" if not missing else ",".join(missing)
            print(f"  {phase}: score={fmt(item['phase_scores'].get(phase))}, missing={missing_text}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    event = latest_event(run_dir)
    scalars = load_scalars(event)
    targets = args.targets
    if not targets:
        targets = checkpoint_steps(run_dir)
        if not args.all_checkpoints:
            targets = targets[-12:]
    if not targets:
        raise SystemExit("No targets supplied and no model_*.pt checkpoints found.")

    report = {
        "run_dir": str(run_dir),
        "event": str(event),
        "window": args.window,
        "checkpoints": [score_target(scalars, target, args.window) for target in targets],
    }
    print_report(report)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\njson_out: {args.json_out}")


if __name__ == "__main__":
    main()
