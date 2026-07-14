#!/usr/bin/env python3
"""Aggregate and freeze the independent 0707 imitation reference for v1.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
REFERENCE_ROOT = WORKFLOW / "0707_reference"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
PROFILES = ("early", "deployed")
SEEDS = (11, 22, 33)
PHASES = (
    "approach", "front_lift", "front_top_support",
    "first_rear_top", "second_rear_top", "rear_hold",
)
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
GROUPS = {
    "front_hips": (0, 1), "rear_hips": (2, 3),
    "front_thighs": (4, 5), "rear_thighs": (6, 7),
    "front_calves": (8, 9), "rear_calves": (10, 11),
    "front_boxes": (12, 13), "rear_boxes": (14, 15),
    "all_boxes": (12, 13, 14, 15),
}
EVENT_FIELDS = (
    "front_lift_step", "front_top_support_step", "first_rear_top_step",
    "second_rear_top_step", "rear_hold_step",
    "front_support_to_first_rear_top_steps", "first_to_second_rear_top_steps",
    "second_rear_top_to_rear_hold_steps",
)
CHECKPOINTS = {
    "teacher": {
        "path": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
            "2026-07-04_01-45-21/model_151399.pt"
        ),
        "sha256": "d34d560ee7c2b3d8c3df00b04c4514e6c69be4779ec38aeee6d176dec0640b1d",
    },
    "early": {
        "path": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-04_06-32-23/model_152100.pt"
        ),
        "sha256": "3c7a335bdc6c1b952e3b0ddbfdd35f0e4311ca1a9fb86cb9ae9a21de2f63a8d6",
    },
    "deployed": {
        "path": ROOT / (
            "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
            "student_no_prior_Student/2026-07-05_00-13-46/model_158797.pt"
        ),
        "sha256": "7ab180f579f549c35688e605149a8b1f1e5c18bf0e43ca46642abf30cce97284",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile")
    ordered = sorted(float(value) for value in values)
    location = (len(ordered) - 1) * q
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty mean")
    return sum(float(value) for value in values) / len(values)


def _per_dim(vectors: Sequence[Sequence[float]], fn) -> list[float]:
    if not vectors:
        raise ValueError("empty vector collection")
    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise RuntimeError("metric vector width changed")
    return [fn([float(vector[index]) for vector in vectors]) for index in range(width)]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_frames(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _run_paths(profile: str, seed: int) -> tuple[Path, Path]:
    root = REFERENCE_ROOT / f"reference_shared_driver_v2_{profile}" / f"seed{seed}_nominal"
    return root / "run_summary.json", root / "frames.jsonl"


def _phase_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors = [record["action_error_student_minus_teacher"] for record in records]
    abs_errors = [[abs(float(value)) for value in vector] for vector in errors]
    sign_eligible = [record["action_sign_eligible"] for record in records]
    sign_errors = [record["action_sign_error"] for record in records]
    latent = [record["latent_error_student_minus_teacher"] for record in records]
    velocity = [record["velocity_error_estimate_minus_critic_target"] for record in records]
    student_clip = [record["student_target_clip_mask"] for record in records]
    teacher_clip = [record["teacher_prior_box_clip_mask"] for record in records]

    eligible_count = [sum(bool(vector[i]) for vector in sign_eligible) for i in range(16)]
    mismatch_count = [sum(bool(vector[i]) for vector in sign_errors) for i in range(16)]
    sign_rate = [
        mismatch_count[i] / eligible_count[i] if eligible_count[i] else None for i in range(16)
    ]

    groups = {}
    for name, indices in GROUPS.items():
        values = [abs(float(vector[index])) for vector in errors for index in indices]
        groups[name] = {"mae": _mean(values), "q95": _quantile(values, 0.95)}

    return {
        "samples": len(records),
        "joint_bias_student_minus_teacher": _per_dim(errors, _mean),
        "joint_mae": _per_dim(abs_errors, _mean),
        "joint_q95_abs_error": _per_dim(abs_errors, lambda values: _quantile(values, 0.95)),
        "joint_sign_eligible_count": eligible_count,
        "joint_sign_mismatch_rate": sign_rate,
        "student_target_clip_rate": _per_dim(student_clip, _mean),
        "teacher_target_clip_rate": _per_dim(teacher_clip, _mean),
        "latent_mae": _mean([record["latent_mae"] for record in records]),
        "latent_q95_abs_error_by_dim": _per_dim(
            [[abs(float(value)) for value in vector] for vector in latent],
            lambda values: _quantile(values, 0.95),
        ),
        "velocity_abs_error_mean_by_dim": _per_dim(
            [[abs(float(value)) for value in vector] for vector in velocity], _mean
        ),
        "velocity_abs_error_q95_by_dim": _per_dim(
            [[abs(float(value)) for value in vector] for vector in velocity],
            lambda values: _quantile(values, 0.95),
        ),
        "action_groups": groups,
    }


def _delta_metrics(runs: Sequence[Sequence[Mapping[str, Any]]], phase: str) -> dict[str, Any]:
    student_deltas: list[list[float]] = []
    teacher_deltas: list[list[float]] = []
    for records in runs:
        previous = None
        for record in records:
            current = record
            if previous is not None and current["phase"] == phase and previous["phase"] == phase:
                student_deltas.append([
                    abs(float(a) - float(b))
                    for a, b in zip(current["student_action"], previous["student_action"], strict=True)
                ])
                teacher_deltas.append([
                    abs(float(a) - float(b))
                    for a, b in zip(
                        current["teacher_action_post_prior_raw_equivalent"],
                        previous["teacher_action_post_prior_raw_equivalent"], strict=True,
                    )
                ])
            previous = current
    if not student_deltas:
        raise RuntimeError(f"phase {phase!r} has no consecutive action samples")
    return {
        "student_abs_first_difference_mae_by_joint": _per_dim(student_deltas, _mean),
        "student_abs_first_difference_q95_by_joint": _per_dim(
            student_deltas, lambda values: _quantile(values, 0.95)
        ),
        "teacher_abs_first_difference_mae_by_joint": _per_dim(teacher_deltas, _mean),
        "teacher_abs_first_difference_q95_by_joint": _per_dim(
            teacher_deltas, lambda values: _quantile(values, 0.95)
        ),
    }


def _event_timing(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    first = {
        phase: next((float(record["step"]) for record in records if record["phase"] == phase), None)
        for phase in PHASES
    }
    result: dict[str, float | None] = {
        "front_lift_step": first["front_lift"],
        "front_top_support_step": first["front_top_support"],
        "first_rear_top_step": first["first_rear_top"],
        "second_rear_top_step": first["second_rear_top"],
        "rear_hold_step": first["rear_hold"],
    }
    for name, start, end in (
        ("front_support_to_first_rear_top_steps", "front_top_support", "first_rear_top"),
        ("first_to_second_rear_top_steps", "first_rear_top", "second_rear_top"),
        ("second_rear_top_to_rear_hold_steps", "second_rear_top", "rear_hold"),
    ):
        result[name] = None if first[start] is None or first[end] is None else first[end] - first[start]
    return result


def _profile(profile: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries = []
    all_runs = []
    inputs = []
    for seed in SEEDS:
        summary_path, frames_path = _run_paths(profile, seed)
        summary = _read_json(summary_path)
        frames = _read_frames(frames_path)
        if len(frames) != summary["frame_count"] or len(frames) != 600:
            raise RuntimeError(f"incomplete reference trace: {frames_path}")
        if summary["spec_sha256"] != SPEC_SHA:
            raise RuntimeError(f"reference spec mismatch: {summary_path}")
        if summary["frames_sha256"] != sha256_file(frames_path):
            raise RuntimeError(f"frames SHA mismatch: {frames_path}")
        summaries.append(summary)
        all_runs.append(frames)
        inputs.extend((summary_path, frames_path, summary_path.parent / "runtime_environment.json"))

    phase_payload = {}
    for phase in PHASES:
        records = [record for run in all_runs for record in run if record["phase"] == phase]
        if not records:
            raise RuntimeError(f"{profile} historical pair has no samples for required phase {phase}")
        phase_payload[phase] = _phase_metrics(records)
        phase_payload[phase]["action_first_difference"] = _delta_metrics(all_runs, phase)

    event_values = {field: [] for field in EVENT_FIELDS}
    event_missing_runs = {field: [] for field in EVENT_FIELDS}
    for summary, frames in zip(summaries, all_runs, strict=True):
        timing = _event_timing(frames)
        for field in EVENT_FIELDS:
            value = timing[field]
            if value is None:
                event_missing_runs[field].append(summary["run_id"])
            else:
                event_values[field].append(float(value))
    if any(not values for values in event_values.values()):
        missing = [field for field, values in event_values.items() if not values]
        raise RuntimeError(f"{profile} reference never observed required events: {missing}")
    events = {
        field: {
            "minimum": min(values), "maximum": max(values), "values": values,
            "observed_count": len(values), "missing_runs": event_missing_runs[field],
        }
        for field, values in event_values.items()
    }
    outcomes = [summary["outcome"] for summary in summaries]
    payload = {
        "checkpoint": str(CHECKPOINTS[profile]["path"].resolve()),
        "checkpoint_sha256": CHECKPOINTS[profile]["sha256"],
        "teacher_checkpoint": str(CHECKPOINTS["teacher"]["path"].resolve()),
        "teacher_checkpoint_sha256": CHECKPOINTS["teacher"]["sha256"],
        "seeds": list(SEEDS),
        "phase_metrics": phase_payload,
        "event_timing": events,
        "behavior_witness": {
            "valid": len(outcomes),
            "full_climb": sum(bool(outcome["full_climb_success"]) for outcome in outcomes),
            "rear_hold": sum(bool(outcome["rear_on_platform_hold_success"]) for outcome in outcomes),
            "front_top_support": sum(bool(outcome["front_top_support_reached"]) for outcome in outcomes),
        },
    }
    return payload, inputs


def _elementwise_max(left: Sequence[float], right: Sequence[float]) -> list[float]:
    if len(left) != len(right):
        raise RuntimeError("envelope vector widths changed")
    return [max(float(a), float(b)) for a, b in zip(left, right, strict=True)]


def _envelope(profiles: Mapping[str, Any]) -> dict[str, Any]:
    early = profiles["early"]
    deployed = profiles["deployed"]
    phases = {}
    for phase in PHASES:
        a = early["phase_metrics"][phase]
        b = deployed["phase_metrics"][phase]
        delta_a = a["action_first_difference"]
        delta_b = b["action_first_difference"]
        phases[phase] = {
            "minimum_reference_samples": min(a["samples"], b["samples"]),
            "joint_abs_bias_upper": _elementwise_max(
                [abs(value) for value in a["joint_bias_student_minus_teacher"]],
                [abs(value) for value in b["joint_bias_student_minus_teacher"]],
            ),
            "joint_mae_upper": _elementwise_max(a["joint_mae"], b["joint_mae"]),
            "joint_q95_abs_error_upper": _elementwise_max(
                a["joint_q95_abs_error"], b["joint_q95_abs_error"]
            ),
            "joint_sign_mismatch_rate_upper": [
                max(float(x or 0.0), float(y or 0.0))
                for x, y in zip(
                    a["joint_sign_mismatch_rate"], b["joint_sign_mismatch_rate"], strict=True
                )
            ],
            "student_target_clip_rate_upper": _elementwise_max(
                a["student_target_clip_rate"], b["student_target_clip_rate"]
            ),
            "latent_mae_upper": max(a["latent_mae"], b["latent_mae"]),
            "latent_q95_abs_error_by_dim_upper": _elementwise_max(
                a["latent_q95_abs_error_by_dim"], b["latent_q95_abs_error_by_dim"]
            ),
            "velocity_abs_error_mean_by_dim_upper": _elementwise_max(
                a["velocity_abs_error_mean_by_dim"], b["velocity_abs_error_mean_by_dim"]
            ),
            "velocity_abs_error_q95_by_dim_upper": _elementwise_max(
                a["velocity_abs_error_q95_by_dim"], b["velocity_abs_error_q95_by_dim"]
            ),
            "action_groups_upper": {
                group: {
                    "mae": max(a["action_groups"][group]["mae"], b["action_groups"][group]["mae"]),
                    "q95": max(a["action_groups"][group]["q95"], b["action_groups"][group]["q95"]),
                }
                for group in GROUPS
            },
            "student_abs_first_difference_mae_by_joint_upper": _elementwise_max(
                delta_a["student_abs_first_difference_mae_by_joint"],
                delta_b["student_abs_first_difference_mae_by_joint"],
            ),
            "student_abs_first_difference_q95_by_joint_upper": _elementwise_max(
                delta_a["student_abs_first_difference_q95_by_joint"],
                delta_b["student_abs_first_difference_q95_by_joint"],
            ),
        }
    events = {}
    for field in EVENT_FIELDS:
        values = early["event_timing"][field]["values"] + deployed["event_timing"][field]["values"]
        events[field] = {"minimum": min(values), "maximum": max(values)}
    return {
        "rule": "For every phase/joint/group metric, use the worse (larger error) of the independently paired early and deployed 0707 Students. Event timing must stay inside the union min/max. No global average may compensate a failed element.",
        "phase_metrics": phases,
        "event_timing": events,
    }


def build(video_manifest: Path | None) -> dict[str, Any]:
    if sha256_file(SPEC) != SPEC_SHA:
        raise RuntimeError("v1.5 spec SHA mismatch")
    for binding in CHECKPOINTS.values():
        if sha256_file(binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch: {binding['path']}")
    profiles = {}
    input_paths: list[Path] = [SPEC, Path(__file__).resolve()]
    for profile in PROFILES:
        profiles[profile], paths = _profile(profile)
        input_paths.extend(paths)
    videos = None
    if video_manifest is not None:
        videos = _read_json(video_manifest)
        if not bool(videos.get("visual_qa", {}).get("robot_visible_all_pairs", False)):
            raise RuntimeError("paired-video visual QA is not complete")
        input_paths.append(video_manifest)
        for entry in videos.get("videos", []):
            path = Path(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"video SHA mismatch: {path}")
            input_paths.append(path)
    return {
        "schema_version": 1,
        "kind": "highstep_v15_0707_imitation_reference_manifest",
        "status": "frozen_read_only" if video_manifest else "draft_waiting_for_paired_video",
        "immutable_after_write": bool(video_manifest),
        "authority": {"spec_path": str(SPEC), "spec_sha256": SPEC_SHA},
        "historical_runtime_contract": {
            "terrain_type": "box_hard", "terrain_level": 9,
            "seeds": list(SEEDS), "fixed_command": [0.45, 0.0, 0.0],
            "real_gains": {"hip": [45.0, 1.5], "thigh": [50.0, 1.5], "calf": [60.0, 2.0]},
            "action_delay_steps": 0, "random_events_enabled": False,
            "student_action_prior_enabled": False,
            "teacher_post_prior_source": "saved Teacher env config at update 151399",
            "shared_state_driver": {
                "role": "read_only_deployed_0707_student",
                "checkpoint": str(CHECKPOINTS["deployed"]["path"].resolve()),
                "checkpoint_sha256": CHECKPOINTS["deployed"]["sha256"],
            },
        },
        "joint_names": list(JOINT_NAMES),
        "profiles": profiles,
        "acceptance_envelope": _envelope(profiles),
        "paired_video_manifest": videos,
        "input_sha256": {str(path.resolve()): sha256_file(path) for path in sorted(set(input_paths))},
        "candidate_data_observed_before_freeze": False,
        "training_allowed_during_reference_generation": False,
    }


def _write(path: Path, payload: Mapping[str, Any], read_only: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read_only:
        raise FileExistsError(f"refusing to overwrite frozen manifest: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze != bool(args.video_manifest):
        raise RuntimeError("freeze requires the completed paired-video manifest, and vice versa")
    payload = build(args.video_manifest)
    _write(args.output, payload, read_only=args.freeze)
    print(json.dumps({
        "path": str(args.output.resolve()), "sha256": sha256_file(args.output),
        "status": payload["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
