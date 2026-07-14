#!/usr/bin/env python3
"""Fail-closed visibility checks for delivered highstep videos."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any


def video_delivery_qc(video: Path, log: Path) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames:format=duration",
            "-of", "json", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(probe.stdout)
    streams = payload.get("streams", [])
    stream = streams[0] if streams else {}
    frame_count = int(stream.get("nb_frames", 0) or 0)
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    log_text = log.read_text(errors="replace")
    camera_witness = "[HIGHSTEP_CAMERA] mode=side_top env=0" in log_text
    result = {
        "passed": bool(
            camera_witness
            and int(stream.get("width", 0) or 0) >= 640
            and int(stream.get("height", 0) or 0) >= 360
            and frame_count >= 599
            and duration >= 11.9
        ),
        "camera_mode": "side_top",
        "camera_runtime_witness": camera_witness,
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "frame_count": frame_count,
        "duration_s": duration,
    }
    if not result["passed"]:
        raise RuntimeError(f"video delivery QC failed: {video}: {result}")
    return result
