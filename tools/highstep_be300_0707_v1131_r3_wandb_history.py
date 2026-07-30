#!/usr/bin/env python3
"""Export, mirror, and verify the finite scalar history for v1.13.1 r3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORK = ROOT / "tmp/highstep_be300_0707_distill_20260716"
ENTITY = "xinqili551-the-university-of-hong-kong"
PROJECT = "isaaclab"
GROUP = "highstep_be300_0707_distill_20260716"
SOURCE_RUN = "be3000707v1131"
TARGET_RUN = "be3000707v1131r2"
TARGET_NAME = "highstep_be300_0707_distill_E4000_recovery_r3_scalar_mirror"
EVENT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_"
    "student_no_prior_Student/2026-07-16_08-35-41_highstep_be300_0707_"
    "distill_long_E4000_20260716/events.out.tfevents.1784162151.lxq-MS-7E30.2898910.0"
)
EVENT_SHA = "367b7b827726d237be6a891dc9b53dfd331cc9d3160a79fd3272517d050e12b3"
EXPORT = WORK / "wandb_source_scalar_history_steps_0_999_v1131_r3.jsonl"
EXPORT_MANIFEST = WORK / "wandb_source_scalar_history_steps_0_999_v1131_r3_manifest.json"
MIRROR_GATE = WORK / "wandb_recovery_r2_scalar_mirror_remote_gate.json"
MAX_TB_ABS_DELTA = 5.0e-5
EXPECTED_TAGS = 287
EXPECTED_STEPS = 1000


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: str, *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if read_only:
            path.chmod(0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def numeric_payload(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in row.items():
        if key in {"_step", "_timestamp", "_runtime"} or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            result[key] = float(value)
    return result


def load_export(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        step = value.pop("_step")
        if isinstance(step, bool) or not isinstance(step, int) or step in rows:
            raise RuntimeError(f"invalid or duplicate export step: {step!r}")
        payload = numeric_payload(value)
        if len(payload) != EXPECTED_TAGS:
            raise RuntimeError(f"export step {step} has {len(payload)} scalar tags")
        rows[step] = payload
    if set(rows) != set(range(EXPECTED_STEPS)):
        raise RuntimeError("export does not cover exactly steps 0..999")
    keys = {frozenset(payload) for payload in rows.values()}
    if len(keys) != 1 or len(next(iter(keys))) != EXPECTED_TAGS:
        raise RuntimeError("export scalar tag set is not stable across all steps")
    return rows


def export_history() -> None:
    import wandb
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if sha256_file(EVENT) != EVENT_SHA:
        raise RuntimeError("TensorBoard event SHA mismatch")
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/{SOURCE_RUN}")
    if run.state != "crashed" or int(run.lastHistoryStep or -1) != 1078:
        raise RuntimeError("source W&B run state/last step changed")
    remote_rows = list(run.scan_history(min_step=0, max_step=1000, page_size=1000))
    remote = {int(row["_step"]): numeric_payload(row) for row in remote_rows}
    if set(remote) != set(range(EXPECTED_STEPS)):
        raise RuntimeError("source W&B history does not cover exactly steps 0..999")
    tag_sets = {frozenset(payload) for payload in remote.values()}
    if len(tag_sets) != 1 or len(next(iter(tag_sets))) != EXPECTED_TAGS:
        raise RuntimeError("source W&B scalar tag set is incomplete or unstable")

    accumulator = EventAccumulator(str(EVENT), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = list(accumulator.Tags().get("scalars", []))
    if len(tags) != EXPECTED_TAGS or set(tags) != set(next(iter(tag_sets))):
        raise RuntimeError("TensorBoard and W&B scalar tag sets disagree")
    tensorboard: dict[int, dict[str, float]] = {step: {} for step in range(EXPECTED_STEPS)}
    for tag in tags:
        for event in accumulator.Scalars(tag):
            if 0 <= event.step < EXPECTED_STEPS:
                tensorboard[event.step][tag] = float(event.value)
    deltas: list[float] = []
    for step in range(EXPECTED_STEPS):
        if set(tensorboard[step]) != set(remote[step]):
            raise RuntimeError(f"TensorBoard/W&B tag disagreement at step {step}")
        for tag, remote_value in remote[step].items():
            deltas.append(abs(remote_value - tensorboard[step][tag]))
    max_delta = max(deltas, default=0.0)
    if max_delta > MAX_TB_ABS_DELTA:
        raise RuntimeError(f"TensorBoard/W&B scalar mismatch: max_abs_delta={max_delta}")

    lines = []
    for step in range(EXPECTED_STEPS):
        lines.append(json.dumps({"_step": step, **remote[step]}, sort_keys=True, allow_nan=False))
    atomic_write(EXPORT, "\n".join(lines) + "\n", read_only=True)
    export_sha = sha256_file(EXPORT)
    manifest = {
        "schema_version": 1,
        "kind": "highstep_be300_0707_v1131_verified_source_scalar_history_export",
        "workflow_id": GROUP,
        "source_run_id": SOURCE_RUN,
        "source_run_state": run.state,
        "source_remote_last_history_step": int(run.lastHistoryStep),
        "exported_range": [0, 999],
        "safe_recovery_boundary": 1000,
        "rows": EXPECTED_STEPS,
        "scalar_tags": EXPECTED_TAGS,
        "scalar_pairs": EXPECTED_STEPS * EXPECTED_TAGS,
        "finite_numeric_scalars_only": True,
        "excluded": ["system_metrics", "videos", "artifacts", "checkpoints", "steps_1000_to_1080"],
        "tensorboard_event": str(EVENT),
        "tensorboard_event_sha256": EVENT_SHA,
        "tensorboard_step_range": [0, 1080],
        "tensorboard_cross_validation_max_abs_delta": max_delta,
        "tensorboard_cross_validation_tolerance": MAX_TB_ABS_DELTA,
        "export": str(EXPORT),
        "export_sha256": export_sha,
        "created_at": now(),
    }
    atomic_write(
        EXPORT_MANIFEST,
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        read_only=True,
    )
    print(json.dumps({**manifest, "manifest_sha256": sha256_file(EXPORT_MANIFEST)}, indent=2))


def mirror_history(config: Path, config_sha: str, export_sha: str, manifest_sha: str) -> None:
    import wandb

    if sha256_file(EXPORT) != export_sha or sha256_file(EXPORT_MANIFEST) != manifest_sha:
        raise RuntimeError("history export or manifest SHA mismatch")
    if sha256_file(config) != config_sha or config.stat().st_mode & 0o222:
        raise RuntimeError("immutable r3 W&B config mismatch")
    rows = load_export(EXPORT)
    try:
        wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{TARGET_RUN}")
    except Exception:
        pass
    else:
        raise RuntimeError("target W&B run already exists; refusing ambiguous mirror")

    for name in ("WANDB_FORK_FROM", "WANDB_RESUME_FROM"):
        os.environ.pop(name, None)
    os.environ.update(
        {
            "WANDB_MODE": "online",
            "WANDB_RUN_ID": TARGET_RUN,
            "WANDB_RESUME": "never",
            "WANDB_RUN_GROUP": GROUP,
            "WANDB_ENTITY": ENTITY,
            "WANDB_USERNAME": ENTITY,
        }
    )
    wandb.setup(settings=wandb.Settings(config_paths=[str(config)]))
    run = wandb.init(
        project=PROJECT,
        entity=ENTITY,
        id=TARGET_RUN,
        name=TARGET_NAME,
        group=GROUP,
        resume="never",
    )
    if run is None or run.id != TARGET_RUN:
        raise RuntimeError("W&B target run initialization failed")
    for step in range(EXPECTED_STEPS):
        run.log(rows[step], step=step, commit=True)
    run.finish()
    verify_mirror(export_sha=export_sha, manifest_sha=manifest_sha, config_sha=config_sha)


def verify_mirror(*, export_sha: str, manifest_sha: str, config_sha: str) -> None:
    import wandb

    if sha256_file(EXPORT) != export_sha or sha256_file(EXPORT_MANIFEST) != manifest_sha:
        raise RuntimeError("history evidence changed before remote verification")
    rows = load_export(EXPORT)
    deadline = time.monotonic() + 300
    last_error: Exception | None = None
    remote_run = None
    while time.monotonic() < deadline:
        try:
            remote_run = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{TARGET_RUN}")
            if int(remote_run.lastHistoryStep or -1) == 999:
                break
        except Exception as error:
            last_error = error
        time.sleep(5)
    if remote_run is None or int(remote_run.lastHistoryStep or -1) != 999:
        raise RuntimeError(f"mirrored W&B history did not finalize at step 999: {last_error}")
    if remote_run.group != GROUP:
        raise RuntimeError("mirrored W&B group mismatch")
    remote_rows = list(remote_run.scan_history(min_step=0, max_step=1000, page_size=1000))
    remote = {int(row["_step"]): numeric_payload(row) for row in remote_rows}
    if set(remote) != set(rows):
        raise RuntimeError("mirrored W&B step range differs from export")
    max_delta = 0.0
    for step, expected in rows.items():
        if set(remote[step]) != set(expected):
            raise RuntimeError(f"mirrored W&B scalar tags differ at step {step}")
        for tag, value in expected.items():
            max_delta = max(max_delta, abs(remote[step][tag] - value))
    if max_delta > 1.0e-12:
        raise RuntimeError(f"mirrored W&B values differ from export: max_abs_delta={max_delta}")
    critical = [
        key for key in next(iter(rows.values()))
        if key.startswith("Loss/") or key in {"Episode_Reward/front_legs_reach", "Train/mean_reward"}
    ]
    if not critical:
        raise RuntimeError("no critical loss/reward scalar keys found in mirror")
    gate = {
        "schema_version": 1,
        "kind": "highstep_be300_0707_v1131_r3_scalar_mirror_remote_gate",
        "workflow_id": GROUP,
        "passed": True,
        "run_id": TARGET_RUN,
        "url": remote_run.url,
        "group": remote_run.group,
        "remote_state": remote_run.state,
        "remote_last_history_step": int(remote_run.lastHistoryStep),
        "mirrored_range": [0, 999],
        "safe_recovery_boundary": 1000,
        "rows": len(remote),
        "scalar_tags": len(next(iter(rows.values()))),
        "remote_export_max_abs_delta": max_delta,
        "critical_loss_reward_keys_verified": sorted(critical),
        "history_export": str(EXPORT),
        "history_export_sha256": export_sha,
        "history_manifest": str(EXPORT_MANIFEST),
        "history_manifest_sha256": manifest_sha,
        "wandb_config_sha256": config_sha,
        "verified_at": now(),
    }
    atomic_write(MIRROR_GATE, json.dumps(gate, indent=2, sort_keys=True) + "\n", read_only=True)
    print(json.dumps({**gate, "gate_sha256": sha256_file(MIRROR_GATE)}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("export")
    for command in ("mirror", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--config-sha256", required=True)
        sub.add_argument("--export-sha256", required=True)
        sub.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_history()
    elif args.command == "mirror":
        mirror_history(args.config, args.config_sha256, args.export_sha256, args.manifest_sha256)
    else:
        verify_mirror(
            export_sha=args.export_sha256,
            manifest_sha=args.manifest_sha256,
            config_sha=args.config_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
