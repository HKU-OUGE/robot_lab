#!/usr/bin/env python3
"""Build immutable, physically bounded highstep reference candidates.

This tool is deliberately independent of Isaac Lab.  It consumes the exact
interactive Teacher trace frozen by the phase-residual preregistration, verifies
all source hashes, selects only the preregistered episodes, and writes bounded
reference candidates without modifying the source recording.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
RESAMPLE_POINTS = 64
PRE_ACTIVE_FRAMES = 10
POST_ACTIVE_FRAMES = 25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def verify_file(path_text: str, expected_sha256: str, label: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected={expected_sha256}, actual={actual}, path={path}"
        )
    return path


def read_trace(path: Path) -> tuple[list[str], dict[int, list[dict[str, str]]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"trace has no header: {path}")
        episodes: dict[int, list[dict[str, str]]] = {}
        for row in reader:
            episode_id = int(row["episode_id"])
            episodes.setdefault(episode_id, []).append(row)
    if not episodes:
        raise RuntimeError(f"trace contains no samples: {path}")
    return list(reader.fieldnames), episodes


def active_crop(rows: Sequence[dict[str, str]]) -> tuple[int, int]:
    active = [
        index
        for index, row in enumerate(rows)
        if max(
            abs(float(row["command.vx"])),
            abs(float(row["command.vy"])),
            abs(float(row["command.wz"])),
        )
        > 0.05
    ]
    if not active:
        raise RuntimeError("episode has no active command segment")
    start = max(0, active[0] - PRE_ACTIVE_FRAMES)
    stop = min(len(rows), active[-1] + POST_ACTIVE_FRAMES + 1)
    if stop - start < 8:
        raise RuntimeError("active command crop is too short")
    return start, stop


def linear_resample(values: Sequence[Sequence[float]], points: int = RESAMPLE_POINTS) -> list[list[float]]:
    if len(values) < 2:
        raise ValueError("at least two source frames are required")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError("inconsistent row width while resampling")
    result: list[list[float]] = []
    for output_index in range(points):
        source = output_index * (len(values) - 1) / (points - 1)
        lower = int(math.floor(source))
        upper = min(lower + 1, len(values) - 1)
        fraction = source - lower
        result.append(
            [
                values[lower][column] * (1.0 - fraction) + values[upper][column] * fraction
                for column in range(width)
            ]
        )
    return result


def normalized_episode_signature(
    rows: Sequence[dict[str, str]],
    joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
) -> list[list[float]]:
    start, stop = active_crop(rows)
    cropped = rows[start:stop]
    root_y0 = float(cropped[0]["root_pos.y"])
    values: list[list[float]] = []
    for row in cropped:
        signature: list[float] = []
        for index, name in enumerate(joint_names):
            span = max(upper[index] - lower[index], 1.0e-6)
            signature.append((float(row[f"joint_pos.{name}"]) - lower[index]) / span)
        signature.extend(
            [
                float(row["root_pos.z"]) / 0.38,
                (float(row["root_pos.y"]) - root_y0) / 0.38,
                float(row["root_quat.w"]),
                float(row["root_quat.x"]),
                float(row["root_quat.y"]),
                float(row["root_quat.z"]),
            ]
        )
        values.append(signature)
    return linear_resample(values)


def signature_distance(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signature shapes do not match")
    total = 0.0
    count = 0
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            raise ValueError("signature widths do not match")
        for left_value, right_value in zip(left_row, right_row):
            total += abs(left_value - right_value)
            count += 1
    return total / max(count, 1)


def choose_medoid(
    episodes: dict[int, list[dict[str, str]]],
    joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[int, dict[int, float]]:
    signatures = {
        episode_id: normalized_episode_signature(rows, joint_names, lower, upper)
        for episode_id, rows in episodes.items()
    }
    totals: dict[int, float] = {}
    for left_id, left in signatures.items():
        totals[left_id] = sum(
            signature_distance(left, right)
            for right_id, right in signatures.items()
            if right_id != left_id
        )
    return min(totals, key=lambda episode_id: (totals[episode_id], episode_id)), totals


def target_violation(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def candidate_rows(
    source_rows: Sequence[dict[str, str]],
    joint_names: Sequence[str],
    lower: Sequence[float],
    upper: Sequence[float],
    transform: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start, stop = active_crop(source_rows)
    cropped = source_rows[start:stop]
    output: list[dict[str, Any]] = []
    source_violation_frames = 0
    source_violation_values = 0
    maximum_projection = 0.0
    per_joint_projection_max = {name: 0.0 for name in joint_names}

    for output_index, row in enumerate(cropped):
        source_target = [float(row[f"mapped_target.{name}"]) for name in joint_names]
        measured = [float(row[f"joint_pos.{name}"]) for name in joint_names]
        if transform == "physical_clamped_mapped_target":
            seed = source_target
        elif transform == "measured_joint_position":
            seed = measured
        else:
            raise ValueError(f"unsupported transform: {transform}")

        reference = [clamp(value, lower[index], upper[index]) for index, value in enumerate(seed)]
        source_deltas = [
            target_violation(value, lower[index], upper[index])
            for index, value in enumerate(source_target)
        ]
        projection = [abs(reference[index] - seed[index]) for index in range(len(joint_names))]
        if any(delta > 0.0 for delta in source_deltas):
            source_violation_frames += 1
        source_violation_values += sum(delta > 0.0 for delta in source_deltas)
        maximum_projection = max(maximum_projection, max(projection, default=0.0))
        for index, name in enumerate(joint_names):
            per_joint_projection_max[name] = max(per_joint_projection_max[name], projection[index])

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "reference_step": output_index,
            "source_episode_id": int(row["episode_id"]),
            "source_control_step": int(row["control_step"]),
            "source_episode_time_s": float(row["episode_time_s"]),
            "global_phase": output_index / max(len(cropped) - 1, 1),
            "source_command_vx": float(row["command.vx"]),
            "source_command_vy": float(row["command.vy"]),
            "source_command_wz": float(row["command.wz"]),
            "source_root_pos.x": float(row["root_pos.x"]),
            "source_root_pos.y": float(row["root_pos.y"]),
            "source_root_pos.z": float(row["root_pos.z"]),
            "source_root_quat.w": float(row["root_quat.w"]),
            "source_root_quat.x": float(row["root_quat.x"]),
            "source_root_quat.y": float(row["root_quat.y"]),
            "source_root_quat.z": float(row["root_quat.z"]),
            "source_root_lin_vel_b.x": float(row["root_lin_vel_b.x"]),
            "source_root_lin_vel_b.y": float(row["root_lin_vel_b.y"]),
            "source_root_lin_vel_b.z": float(row["root_lin_vel_b.z"]),
            "source_root_ang_vel_b.x": float(row["root_ang_vel_b.x"]),
            "source_root_ang_vel_b.y": float(row["root_ang_vel_b.y"]),
            "source_root_ang_vel_b.z": float(row["root_ang_vel_b.z"]),
            "root_pos_z": float(row["root_pos.z"]),
            "root_pos_y": float(row["root_pos.y"]),
        }
        for index, name in enumerate(joint_names):
            record[f"source_mapped_target.{name}"] = source_target[index]
            record[f"reference_target.{name}"] = reference[index]
            record[f"measured_joint_pos.{name}"] = measured[index]
            record[f"measured_joint_vel.{name}"] = float(row[f"joint_vel.{name}"])
        output.append(record)

    exported_violations = sum(
        not (lower[index] <= float(row[f"reference_target.{name}"]) <= upper[index])
        for row in output
        for index, name in enumerate(joint_names)
    )
    audit = {
        "source_frame_start": start,
        "source_frame_stop_exclusive": stop,
        "frame_count": len(output),
        "source_target_violation_frames": source_violation_frames,
        "source_target_violation_values": source_violation_values,
        "exported_target_violation_values": exported_violations,
        "maximum_transform_delta": maximum_projection,
        "per_joint_transform_delta_max": per_joint_projection_max,
    }
    if exported_violations:
        raise RuntimeError(f"bounded candidate still contains {exported_violations} limit violations")
    return output, audit


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_references(preregistration_path: Path, output_dir: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite reference output directory: {output_dir}")

    preregistration = load_json(preregistration_path)
    if preregistration.get("status") != "frozen_before_implementation":
        raise RuntimeError("preregistration is not frozen_before_implementation")
    if preregistration.get("workflow_id") != "highstep_fixed_condition_phase_residual_20260717":
        raise RuntimeError("unexpected workflow_id")

    branch_spec = preregistration["branch_spec"]
    teacher_path = verify_file(
        preregistration["unchanged_contract"]["teacher_checkpoint"],
        preregistration["unchanged_contract"]["teacher_sha256"],
        "Teacher checkpoint",
    )
    branch_spec_path = verify_file(branch_spec["path"], branch_spec["sha256"], "branch spec")
    source = preregistration["source_recording"]
    manifest_path = verify_file(source["manifest_path"], source["manifest_sha256"], "source manifest")
    trace_path = verify_file(source["trace_path"], source["trace_sha256"], "source trace")
    source_manifest = load_json(manifest_path)

    joint_names = list(preregistration["unchanged_contract"]["joint_order"])
    lower = [float(value) for value in preregistration["unchanged_contract"]["physical_lower"]]
    upper = [float(value) for value in preregistration["unchanged_contract"]["physical_upper"]]
    if not (len(joint_names) == len(lower) == len(upper) == 16):
        raise RuntimeError("invalid 16-joint physical contract")
    if source_manifest.get("joint_names_in_action_order") != joint_names:
        raise RuntimeError("source manifest joint order does not match preregistration")

    _, episodes = read_trace(trace_path)
    if len(episodes) != int(source["episode_count"]):
        raise RuntimeError(
            f"source episode count changed: expected={source['episode_count']}, actual={len(episodes)}"
        )
    medoid_episode, medoid_totals = choose_medoid(episodes, joint_names, lower, upper)

    requested: list[tuple[str, int]] = []
    for candidate in source["source_candidates"]:
        kind = candidate["kind"]
        if kind == "fixed_episode":
            episode_id = int(candidate["episode_id"])
            label = f"episode{episode_id}"
        elif kind == "computed_medoid":
            episode_id = medoid_episode
            label = "medoid"
        else:
            raise RuntimeError(f"unsupported candidate kind: {kind}")
        if episode_id not in episodes:
            raise RuntimeError(f"candidate episode is absent: {episode_id}")
        requested.append((label, episode_id))

    output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for label, episode_id in requested:
        for transform in ("physical_clamped_mapped_target", "measured_joint_position"):
            rows, audit = candidate_rows(episodes[episode_id], joint_names, lower, upper, transform)
            filename = f"{label}_{transform}.csv"
            path = output_dir / filename
            write_csv(path, rows)
            records.append(
                {
                    "label": label,
                    "source_episode_id": episode_id,
                    "transform": transform,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "audit": audit,
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "highstep_phase_residual_reference_candidates",
        "status": "reference_candidates_frozen_pending_isaac_replay",
        "workflow_id": preregistration["workflow_id"],
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": sha256_file(preregistration_path),
        "branch_spec_path": str(branch_spec_path),
        "branch_spec_sha256": sha256_file(branch_spec_path),
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": sha256_file(teacher_path),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_trace_path": str(trace_path),
        "source_trace_sha256": sha256_file(trace_path),
        "joint_names": joint_names,
        "physical_lower": lower,
        "physical_upper": upper,
        "crop_contract": {
            "active_command_threshold": 0.05,
            "pre_active_frames": PRE_ACTIVE_FRAMES,
            "post_active_frames": POST_ACTIVE_FRAMES,
        },
        "medoid_contract": {
            "resample_points": RESAMPLE_POINTS,
            "selected_episode_id": medoid_episode,
            "total_distance_by_episode": {str(key): value for key, value in sorted(medoid_totals.items())},
        },
        "candidates": records,
    }
    manifest_path_out = output_dir / "reference_candidates_manifest.json"
    write_bytes_atomic(manifest_path_out, canonical_json_bytes(manifest))
    manifest["manifest_path"] = str(manifest_path_out)
    manifest["manifest_sha256"] = sha256_file(manifest_path_out)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_references(args.preregistration, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
