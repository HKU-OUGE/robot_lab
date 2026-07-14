#!/usr/bin/env python3
"""Zero-training calibration gate for highstep ActionScore work.

This script intentionally reads only existing CSV/TensorBoard event files. It
does not load policies, step environments, or start training. CSV rows are
legacy segment-relative proxy scores; ActionScore event rows are current
curriculum scores. This gate can separate legacy positive/degraded CSV windows
and a current ActionScore failed run, but it cannot prove that the current
event support_score is calibrated on the positive anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEACHER_CSV = REPO_ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_Teacher/"
    "2026-07-01_06-52-29/analysis_highstep_score_model_140400_to_148700.csv"
)
DEFAULT_STUDENT_CSV = REPO_ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_vae_student_no_prior_Student/"
    "2026-07-01_18-19-06/analysis_student_highstep_score_current.csv"
)
DEFAULT_ACTION_RUN = REPO_ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-03_05-40-48"
)
DEFAULT_OUT_DIR = REPO_ROOT / "analysis/highstep_score_calibration_2026-07-03"


@dataclass
class GateThresholds:
    positive_score_min: float = 85.0
    positive_support_min: float = 90.0
    teacher_negative_score_max: float = 45.0
    teacher_negative_support_max: float = 50.0
    action_negative_support_max: float = 45.0
    action_negative_violation_min: float = 0.50


@dataclass
class CalibrationRow:
    label: str
    role: str
    checkpoints: str
    score_mean: float | None = None
    score_min: float | None = None
    score_max: float | None = None
    hard_score_mean: float | None = None
    entry_mean: float | None = None
    support_mean: float | None = None
    support_min: float | None = None
    support_violation_mean: float | None = None
    terrain_mean: float | None = None
    bad_orientation_mean: float | None = None
    post_lead_signal_mean: float | None = None
    post_lead_reward_mean: float | None = None
    pass_gate: bool | None = None
    evidence: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv", type=Path, default=DEFAULT_TEACHER_CSV)
    parser.add_argument("--student-csv", type=Path, default=DEFAULT_STUDENT_CSV)
    parser.add_argument("--action-run", type=Path, default=DEFAULT_ACTION_RUN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--teacher-positive", type=int, default=141000)
    parser.add_argument("--teacher-negative-start", type=int, default=146000)
    parser.add_argument("--teacher-negative-end", type=int, default=148700)
    parser.add_argument("--teacher-negative-single", type=int, default=148700)
    parser.add_argument("--student-positive-start", type=int, default=141700)
    parser.add_argument("--student-positive-end", type=int, default=142200)
    parser.add_argument("--student-info-start", type=int, default=142300)
    parser.add_argument("--student-info-end", type=int, default=144000)
    parser.add_argument("--action-target", type=int, default=141500)
    parser.add_argument("--action-window", type=int, default=80)
    parser.add_argument("--positive-score-min", type=float, default=85.0)
    parser.add_argument("--positive-support-min", type=float, default=90.0)
    parser.add_argument("--teacher-negative-score-max", type=float, default=45.0)
    parser.add_argument("--teacher-negative-support-max", type=float, default=50.0)
    parser.add_argument("--action-negative-support-max", type=float, default=45.0)
    parser.add_argument("--action-negative-violation-min", type=float, default=0.50)
    parser.add_argument("--no-write", action="store_true", help="Print only; do not write markdown/json reports.")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_score_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            checkpoint = as_float(parsed.get("checkpoint"))
            if checkpoint is None:
                continue
            parsed["checkpoint"] = int(checkpoint)
            for key, value in list(parsed.items()):
                if key in {"checkpoint", "source_run"}:
                    continue
                numeric = as_float(value)
                if numeric is not None:
                    parsed[key] = numeric
            rows.append(parsed)
    return rows


def select_range(rows: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if start <= int(row["checkpoint"]) <= end]
    if not selected:
        raise ValueError(f"No rows found for checkpoint range {start}-{end}")
    return selected


def select_one(rows: list[dict[str, Any]], checkpoint: int) -> list[dict[str, Any]]:
    return select_range(rows, checkpoint, checkpoint)


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def csv_summary(label: str, role: str, rows: list[dict[str, Any]]) -> CalibrationRow:
    checkpoints = [int(row["checkpoint"]) for row in rows]

    def avg(key: str) -> float | None:
        values = metric_values(rows, key)
        return None if not values else mean(values)

    def min_value(key: str) -> float | None:
        values = metric_values(rows, key)
        return None if not values else min(values)

    def max_value(key: str) -> float | None:
        values = metric_values(rows, key)
        return None if not values else max(values)

    ckpt_text = str(checkpoints[0]) if len(checkpoints) == 1 else f"{checkpoints[0]}-{checkpoints[-1]}"
    return CalibrationRow(
        label=label,
        role=role,
        checkpoints=ckpt_text,
        score_mean=avg("highstep_score"),
        score_min=min_value("highstep_score"),
        score_max=max_value("highstep_score"),
        entry_mean=avg("entry_score"),
        support_mean=avg("support_score"),
        support_min=min_value("support_score"),
        terrain_mean=avg("terrain"),
        bad_orientation_mean=avg("bad_orientation"),
        evidence="legacy segment-relative highstep CSV; not current ActionScore event support_score",
    )


def latest_event(run_dir: Path) -> Path:
    events = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.stat().st_mtime)
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")
    return events[-1]


def tail_mean(acc: EventAccumulator, tag: str, target: int, window: int) -> tuple[float | None, int, int | None]:
    if tag not in set(acc.Tags().get("scalars", [])):
        return None, 0, None
    selected = [(event.step, float(event.value)) for event in acc.Scalars(tag) if event.step <= target]
    if not selected:
        return None, 0, None
    tail = selected[-window:]
    return mean(value for _, value in tail), len(tail), tail[-1][0]


def action_score_summary(run_dir: Path, target: int, window: int) -> CalibrationRow:
    event = latest_event(run_dir)
    acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
    acc.Reload()
    tags = {
        "score_mean": "Curriculum/highstep_action_score/total",
        "hard_score_mean": "Curriculum/highstep_action_score/hard_total",
        "entry_mean": "Curriculum/highstep_action_score/entry_score",
        "support_mean": "Curriculum/highstep_action_score/support_score",
        "support_violation_mean": "Curriculum/highstep_action_score/support_floor_violation_rate",
        "terrain_mean": "Curriculum/terrain_levels",
        "post_lead_signal_mean": "Curriculum/highstep_action_score/post_lead_drive_signal_mean",
        "post_lead_reward_mean": "Episode_Reward/post_lead_body_drive",
    }
    values: dict[str, float | None] = {}
    samples: dict[str, int] = {}
    last_steps: dict[str, int | None] = {}
    for key, tag in tags.items():
        value, count, last_step = tail_mean(acc, tag, target, window)
        values[key] = value
        samples[key] = count
        last_steps[key] = last_step

    for key in ("score_mean", "hard_score_mean", "entry_mean", "support_mean"):
        if values[key] is not None:
            values[key] *= 100.0

    return CalibrationRow(
        label="action_score_failed",
        role="action_negative",
        checkpoints=str(target),
        score_mean=values["score_mean"],
        hard_score_mean=values["hard_score_mean"],
        entry_mean=values["entry_mean"],
        support_mean=values["support_mean"],
        support_violation_mean=values["support_violation_mean"],
        terrain_mean=values["terrain_mean"],
        post_lead_signal_mean=values["post_lead_signal_mean"],
        post_lead_reward_mean=values["post_lead_reward_mean"],
        evidence=f"{event}; tail window samples={samples}; last_steps={last_steps}",
    )


def apply_gate(row: CalibrationRow, thresholds: GateThresholds) -> CalibrationRow:
    score = row.score_mean
    support = row.support_mean
    violation = row.support_violation_mean
    if row.role == "positive":
        row.pass_gate = (
            score is not None
            and support is not None
            and score >= thresholds.positive_score_min
            and support >= thresholds.positive_support_min
        )
    elif row.role == "teacher_negative":
        row.pass_gate = (
            score is not None
            and support is not None
            and (
                score <= thresholds.teacher_negative_score_max
                or support <= thresholds.teacher_negative_support_max
            )
        )
    elif row.role == "action_negative":
        row.pass_gate = (
            support is not None
            and violation is not None
            and support < thresholds.action_negative_support_max
            and violation > thresholds.action_negative_violation_min
        )
    else:
        row.pass_gate = None
    return row


def format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4g}"


def markdown_report(rows: list[CalibrationRow], thresholds: GateThresholds) -> str:
    lines = [
        "# Highstep Score Calibration Gate",
        "",
        "This is a zero-training calibration report. It reads existing CSV/event data only.",
        "CSV positives are legacy segment-relative proxy scores, while ActionScore rows are current event metrics.",
        "Do not treat a PASS here as proof that current `Curriculum/highstep_action_score/support_score` is calibrated on the positive anchor.",
        "",
        "## Thresholds",
        "",
        (
            f"- Legacy CSV positive: score >= {thresholds.positive_score_min:g} "
            f"and support >= {thresholds.positive_support_min:g}"
        ),
        (
            f"- Legacy CSV teacher negative: score <= {thresholds.teacher_negative_score_max:g} "
            f"or support <= {thresholds.teacher_negative_support_max:g}"
        ),
        (
            f"- ActionScore negative: support < {thresholds.action_negative_support_max:g} "
            f"and support_floor_violation_rate > {thresholds.action_negative_violation_min:g}"
        ),
        "",
        "## Results",
        "",
        (
            "| label | role | checkpoints | pass | score | support | entry | support_violation | "
            "terrain | bad_orientation | post_lead_signal | post_lead_reward |"
        ),
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        passed = "INFO" if row.pass_gate is None else ("PASS" if row.pass_gate else "FAIL")
        lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    row.role,
                    row.checkpoints,
                    passed,
                    format_number(row.score_mean),
                    format_number(row.support_mean),
                    format_number(row.entry_mean),
                    format_number(row.support_violation_mean),
                    format_number(row.terrain_mean),
                    format_number(row.bad_orientation_mean),
                    format_number(row.post_lead_signal_mean),
                    format_number(row.post_lead_reward_mean),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The positive gate checks the known good teacher anchor and the student reference window only in the legacy CSV score family.",
            "- The teacher negative gate catches the known late degradation where terrain rises but legacy CSV support collapses.",
            "- The ActionScore negative gate catches the failed refine run where entry is high but support remains below floor.",
            "- Current ActionScore positive calibration still requires a play/event evaluation under the current task.",
            "- This report is not a video replacement and must not be used as proof of deployable behavior.",
            "",
            "## Evidence",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row.label}: {row.evidence}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = GateThresholds(
        positive_score_min=args.positive_score_min,
        positive_support_min=args.positive_support_min,
        teacher_negative_score_max=args.teacher_negative_score_max,
        teacher_negative_support_max=args.teacher_negative_support_max,
        action_negative_support_max=args.action_negative_support_max,
        action_negative_violation_min=args.action_negative_violation_min,
    )

    teacher_rows = load_score_csv(args.teacher_csv.resolve())
    student_rows = load_score_csv(args.student_csv.resolve())
    rows = [
        csv_summary("teacher_positive_anchor", "positive", select_one(teacher_rows, args.teacher_positive)),
        csv_summary(
            "student_reference_window",
            "positive",
            select_range(student_rows, args.student_positive_start, args.student_positive_end),
        ),
        csv_summary(
            "teacher_late_degraded_window",
            "teacher_negative",
            select_range(teacher_rows, args.teacher_negative_start, args.teacher_negative_end),
        ),
        csv_summary(
            "teacher_late_degraded_single",
            "teacher_negative",
            select_one(teacher_rows, args.teacher_negative_single),
        ),
        csv_summary(
            "student_later_reference_info",
            "info",
            select_range(student_rows, args.student_info_start, args.student_info_end),
        ),
        action_score_summary(args.action_run.resolve(), args.action_target, args.action_window),
    ]
    rows = [apply_gate(row, thresholds) for row in rows]
    required_rows = [row for row in rows if row.pass_gate is not None]
    overall_pass = all(row.pass_gate for row in required_rows)

    report = {
        "overall_pass": overall_pass,
        "thresholds": asdict(thresholds),
        "inputs": {
            "teacher_csv": str(args.teacher_csv.resolve()),
            "student_csv": str(args.student_csv.resolve()),
            "action_run": str(args.action_run.resolve()),
            "action_target": args.action_target,
            "action_window": args.action_window,
        },
        "score_family_note": (
            "Positive and teacher-negative rows use legacy segment-relative CSV scores; "
            "action_negative uses current TensorBoard ActionScore event metrics."
        ),
        "rows": [asdict(row) for row in rows],
    }

    print(markdown_report(rows, thresholds))
    print(f"overall_pass: {'PASS' if overall_pass else 'FAIL'}")

    if not args.no_write:
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "calibration_gate.json"
        md_path = out_dir / "calibration_gate.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md_path.write_text(markdown_report(rows, thresholds), encoding="utf-8")
        print(f"json_out: {json_path}")
        print(f"markdown_out: {md_path}")

    raise SystemExit(0 if overall_pass else 2)


if __name__ == "__main__":
    main()
