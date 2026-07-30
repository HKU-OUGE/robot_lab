#!/usr/bin/env python3
"""Extract the frozen zero-command Teacher pre-roll for the phase-residual branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
LOWER = (-1.2217304764,) * 4 + (-1.5708,) * 4 + (-2.7750735107,) * 4 + (0.0,) * 4
UPPER = (1.2217304764,) * 4 + (3.4907,) * 4 + (-0.6457718232,) * 4 + (0.06,) * 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: expected={expected} actual={actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def build_preroll(
    preregistration_path: Path,
    preregistration_sha256: str,
    selected_manifest_path: Path,
    selected_manifest_sha256: str,
    timing_path: Path,
    timing_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite pre-roll output: {output_dir}")
    preregistration = _bound_json(preregistration_path, preregistration_sha256, "preregistration")
    selected = _bound_json(selected_manifest_path, selected_manifest_sha256, "selected reference")
    timing = _bound_json(timing_path, timing_sha256, "source timing amendment")
    if selected.get("preregistration", {}).get("sha256") != preregistration_sha256:
        raise RuntimeError("selected reference preregistration binding changed")
    if timing.get("selected_reference_manifest", {}).get("sha256") != selected_manifest_sha256:
        raise RuntimeError("timing amendment selected-reference binding changed")
    settle_steps = int(timing["corrected_timing_contract"]["settle_steps_before_dataset"])
    if settle_steps != 296:
        raise RuntimeError("pre-roll extraction requires the frozen 296-step source timing")

    source = preregistration["source_recording"]
    trace_path = Path(source["trace_path"]).expanduser().resolve()
    if sha256_file(trace_path) != source["trace_sha256"]:
        raise RuntimeError("source Teacher trace SHA mismatch")
    episode_id = int(selected["selected_reference"]["source_episode_id"])
    selected_reference_path = Path(selected["selected_reference"]["path"])
    with selected_reference_path.open("r", encoding="utf-8", newline="") as handle:
        reference_first = next(csv.DictReader(handle))
    active_source_step = int(reference_first["source_control_step"])
    if active_source_step != settle_steps:
        raise RuntimeError("selected reference no longer starts immediately after the frozen pre-roll")

    rows: list[dict[str, Any]] = []
    projection_values = 0
    projection_frames = 0
    projection_max = 0.0
    with trace_path.open("r", encoding="utf-8", newline="") as handle:
        for source_row in csv.DictReader(handle):
            if int(source_row["episode_id"]) != episode_id:
                continue
            control_step = int(source_row["control_step"])
            if control_step >= active_source_step:
                continue
            if control_step != len(rows):
                raise RuntimeError("source pre-roll control steps are not contiguous from zero")
            if max(
                abs(float(source_row["command.vx"])),
                abs(float(source_row["command.vy"])),
                abs(float(source_row["command.wz"])),
            ) > 1.0e-9:
                raise RuntimeError("source pre-roll contains a non-zero command")
            projected: list[float] = []
            frame_projected = False
            for index, name in enumerate(JOINT_NAMES):
                value = float(source_row[f"mapped_target.{name}"])
                safe = min(max(value, LOWER[index]), UPPER[index])
                delta = abs(safe - value)
                if delta > 0.0:
                    projection_values += 1
                    frame_projected = True
                    projection_max = max(projection_max, delta)
                projected.append(safe)
            projection_frames += int(frame_projected)
            record: dict[str, Any] = {
                "schema_version": 1,
                "preroll_step": control_step,
                "source_episode_id": episode_id,
                "source_control_step": control_step,
                "source_episode_time_s": float(source_row["episode_time_s"]),
            }
            record.update(
                {f"reference_target.{name}": projected[index] for index, name in enumerate(JOINT_NAMES)}
            )
            rows.append(record)
    if len(rows) != settle_steps:
        raise RuntimeError(f"source pre-roll has {len(rows)} rows; expected {settle_steps}")

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "teacher_zero_command_preroll.csv"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".preroll.", dir=output_dir, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "kind": "highstep_fixed_condition_teacher_zero_command_preroll",
        "status": "frozen_preroll_ready_before_behavior_retry",
        "workflow_id": "highstep_fixed_condition_phase_residual_20260717",
        "preregistration_path": str(preregistration_path.expanduser().resolve()),
        "preregistration_sha256": preregistration_sha256,
        "selected_reference_manifest_path": str(selected_manifest_path.expanduser().resolve()),
        "selected_reference_manifest_sha256": selected_manifest_sha256,
        "selected_reference_sha256": selected["selected_reference"]["sha256"],
        "source_timing_amendment_path": str(timing_path.expanduser().resolve()),
        "source_timing_amendment_sha256": timing_sha256,
        "source_trace_path": str(trace_path),
        "source_trace_sha256": source["trace_sha256"],
        "source_episode_id": episode_id,
        "frame_count": len(rows),
        "command": [0.0, 0.0, 0.0],
        "preroll_path": str(csv_path),
        "preroll_sha256": sha256_file(csv_path),
        "projection": {
            "projected_frames": projection_frames,
            "projected_values": projection_values,
            "maximum_abs_delta": projection_max,
            "exported_target_limit_violations": 0,
        },
        "transform": "per_joint_physical_clamp_of_source_post_prior_mapped_target",
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_path = output_dir / "preroll_manifest.json"
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--selected-manifest-sha256", required=True)
    parser.add_argument("--timing-amendment", type=Path, required=True)
    parser.add_argument("--timing-amendment-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_preroll(
        args.preregistration,
        args.preregistration_sha256,
        args.selected_manifest,
        args.selected_manifest_sha256,
        args.timing_amendment,
        args.timing_amendment_sha256,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
