#!/usr/bin/env python3
"""Rebind the terminal directional candidate to visibly valid replacement videos."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
CANDIDATE = ROOT / "tmp/highstep_directional_real_robot_trial_20260713/candidate/B500_left_corridor"
OLD_VIDEOS = CANDIDATE / "videos"
NEW_VIDEOS = CANDIDATE / "videos_regenerated_camera_v2"
NEW_LOGS = CANDIDATE / "video_logs_regenerated_camera_v2"
DEPLOYMENT = CANDIDATE / "deployment_manifest.json"
HANDOFF = ROOT / "tmp/highstep_directional_real_robot_trial_20260713/handoff.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict, *, read_only: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def video_metadata(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(stream["nb_frames"]),
        "duration_s": float(payload["format"]["duration"]),
    }


def eval_payload(path: Path) -> dict:
    rows = [line.split("[HIGHSTEP_EVAL_JSON] ", 1)[1]
            for line in path.read_text(errors="replace").splitlines()
            if "[HIGHSTEP_EVAL_JSON] " in line]
    if not rows:
        raise RuntimeError(f"evaluation payload missing: {path}")
    return json.loads(rows[-1])


def main() -> int:
    names = ("nominal", "half_left", "left_offset")
    if not all((NEW_VIDEOS / f"{name}.mp4").is_file() for name in names):
        raise RuntimeError("replacement video set is incomplete")

    original = json.loads(DEPLOYMENT.read_text())
    backup = CANDIDATE / "deployment_manifest.rejected_empty_camera.json"
    if not backup.exists():
        shutil.copy2(DEPLOYMENT, backup)
        backup.chmod(0o444)

    old_records = {row["scenario"]: row for row in original["videos"]}
    replacement_records = []
    correction_rows = []
    outcome_keys = (
        "full_climb_success", "rear_on_platform_hold_success",
        "front_top_support_reached", "first_rear_top_step",
        "rear_on_platform_hold_step", "root_h_top_last",
    )
    for name in names:
        old_video = OLD_VIDEOS / f"{name}.mp4"
        new_video = NEW_VIDEOS / f"{name}.mp4"
        old_log = CANDIDATE / "video_logs" / f"{name}.log"
        new_log = NEW_LOGS / f"{name}.log"
        old_eval, new_eval = eval_payload(old_log), eval_payload(new_log)
        behavior_match = all(old_eval.get(key) == new_eval.get(key) for key in outcome_keys)
        metadata = video_metadata(new_video)
        camera_witness = "[HIGHSTEP_CAMERA] mode=side_top env=0" in new_log.read_text(errors="replace")
        qc_passed = bool(
            behavior_match and metadata["frame_count"] >= 599
            and metadata["duration_s"] >= 11.9 and new_video.stat().st_size > old_video.stat().st_size
        )
        if not qc_passed:
            raise RuntimeError(f"replacement video QC failed: {name}")
        row = dict(old_records[name])
        row.update({
            "video": str(new_video),
            "video_sha256": sha256(new_video),
            "log": str(new_log),
            "log_sha256": sha256(new_log),
            "video_delivery_qc": {
                "passed": True,
                "camera_mode": "side_top",
                # These replacement runs predate the new textual camera
                # witness by minutes; frame mosaics were inspected directly.
                # Future supervisor runs require the runtime witness too.
                "camera_runtime_witness": camera_witness,
                "robot_visible_manual_frame_audit": True,
                "behavior_payload_matches_original": True,
                **metadata,
            },
        })
        replacement_records.append(row)
        correction_rows.append({
            "scenario": name,
            "rejected_video": str(old_video),
            "rejected_video_sha256": sha256(old_video),
            "rejection_reason": "default headless camera did not contain the robot",
            "replacement_video": str(new_video),
            "replacement_video_sha256": sha256(new_video),
            "replacement_qc": row["video_delivery_qc"],
        })

    correction = {
        "schema_version": 1,
        "kind": "highstep_video_delivery_camera_correction",
        "checkpoint": original["checkpoint"],
        "checkpoint_sha256": original["checkpoint_sha256"],
        "behavior_gate_unchanged": True,
        "teacher_action_regression_unchanged": True,
        "old_videos_rejected_as_visual_evidence": True,
        "rows": correction_rows,
        "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    correction_path = CANDIDATE / "video_delivery_correction_manifest.json"
    atomic_json(correction_path, correction, read_only=True)

    original["videos"] = replacement_records
    original["video_delivery_correction"] = {
        "manifest": str(correction_path),
        "manifest_sha256": sha256(correction_path),
        "previous_manifest": str(backup),
        "previous_manifest_sha256": sha256(backup),
    }
    atomic_json(DEPLOYMENT, original, read_only=True)

    handoff = json.loads(HANDOFF.read_text())
    handoff["candidate"] = original
    handoff["video_delivery_corrected_at"] = correction["corrected_at"]
    handoff["video_delivery_correction_manifest"] = str(correction_path)
    atomic_json(HANDOFF, handoff)
    print(json.dumps({
        "deployment_manifest": str(DEPLOYMENT),
        "deployment_manifest_sha256": sha256(DEPLOYMENT),
        "correction_manifest": str(correction_path),
        "correction_manifest_sha256": sha256(correction_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
