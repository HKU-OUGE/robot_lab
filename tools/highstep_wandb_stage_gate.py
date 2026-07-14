#!/usr/bin/env python3
"""Fail-closed W&B observability contract for future custom B/R/R5 stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence


ENTITY = "xinqili551-the-university-of-hong-kong"
PROJECT = "isaaclab"
ROUTES = {"B", "R", "R5", "student_environment_curriculum"}
REQUIRED_CONFIG = {
    "workflow_id", "route", "stage", "attempt", "spec_sha256",
    "preregistration_sha256", "start_checkpoint", "start_checkpoint_sha256",
    "teacher_checkpoint", "teacher_sha256", "frozen_tensors", "trainable_tensors",
    "optimizer", "budget", "training_task", "historical_sync",
}
REQUIRED_SUMMARY = {
    "output_checkpoint", "output_checkpoint_sha256", "effective_updates",
    "gate_metrics", "gate_conclusion", "manifest_path", "manifest_sha256",
}


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def offline_run_candidates(wandb_root: Path, run_id: str) -> list[Path]:
    """Return every local fragment for a W&B run in creation order."""
    return sorted(
        (path.resolve() for path in wandb_root.glob(f"offline-run-*-{run_id}") if path.is_dir()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def recorded_offline_runs(marker_path: Path, run_id: str) -> set[Path]:
    """Read both the legacy single-fragment marker and the complete marker."""
    if not marker_path.is_file():
        return set()
    payload = json.loads(marker_path.read_text())
    if str(payload.get("run_id")) != run_id:
        raise RuntimeError("W&B offline sync marker run_id changed")
    recorded: set[Path] = set()
    legacy = payload.get("offline_run")
    if legacy:
        recorded.add(Path(str(legacy)).resolve())
    for item in payload.get("offline_runs", []):
        value = item.get("offline_run") if isinstance(item, Mapping) else item
        if value:
            recorded.add(Path(str(value)).resolve())
    return recorded


def pending_offline_runs(wandb_root: Path, marker_path: Path, run_id: str) -> list[Path]:
    recorded = recorded_offline_runs(marker_path, run_id)
    return [path for path in offline_run_candidates(wandb_root, run_id) if path not in recorded]


def offline_sync_requires_append(marker_path: Path, run_id: str) -> bool:
    """A second fragment must append to the already-created remote run."""
    return bool(recorded_offline_runs(marker_path, run_id))


def offline_fragment_history_steps(offline_run: Path) -> set[int]:
    """Read exact history steps from an immutable local W&B transaction log."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    files = sorted(offline_run.resolve(strict=True).glob("*.wandb"))
    if not files:
        raise RuntimeError(f"offline W&B fragment has no .wandb file: {offline_run}")
    steps: set[int] = set()
    for path in files:
        store = DataStore()
        store.open_for_scan(str(path))
        while True:
            try:
                data = store.scan_data()
            except AssertionError as error:
                raise RuntimeError(f"offline W&B history is not fully readable: {path}") from error
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "history":
                steps.add(int(record.history.step.num))
    if not steps:
        raise RuntimeError(f"offline W&B fragment has no history rows: {offline_run}")
    return steps


def expected_offline_history_steps(marker_path: Path, run_id: str) -> set[int]:
    fragments = recorded_offline_runs(marker_path, run_id)
    if not fragments:
        raise RuntimeError("W&B offline sync marker contains no fragments")
    expected: set[int] = set()
    for fragment in fragments:
        expected.update(offline_fragment_history_steps(fragment))
    return expected


def record_offline_run_synced(
    marker_path: Path,
    *,
    run_id: str,
    offline_run: Path,
    sync_log: Path,
    synced_at: str,
    expected_offline_runs: Sequence[Path],
) -> dict[str, Any]:
    """Crash-safely extend the marker without losing legacy fragment evidence."""
    records: dict[str, dict[str, Any]] = {}
    if marker_path.is_file():
        old = json.loads(marker_path.read_text())
        if str(old.get("run_id")) != run_id:
            raise RuntimeError("W&B offline sync marker run_id changed")
        legacy = old.get("offline_run")
        if legacy:
            legacy_path = str(Path(str(legacy)).resolve())
            records[legacy_path] = {
                "offline_run": legacy_path,
                "offline_run_wandb_files": old.get("offline_run_wandb_files", []),
                "synced_at": old.get("synced_at"),
            }
        for item in old.get("offline_runs", []):
            if isinstance(item, Mapping) and item.get("offline_run"):
                records[str(Path(str(item["offline_run"])).resolve())] = dict(item)
    offline_run = offline_run.resolve(strict=True)
    records[str(offline_run)] = {
        "offline_run": str(offline_run),
        "synced_at": synced_at,
    }
    expected = {str(path.resolve(strict=True)) for path in expected_offline_runs}
    for key, record in records.items():
        path = Path(key).resolve(strict=True)
        files = sorted(path.glob("*.wandb"))
        if not files:
            raise RuntimeError(f"offline W&B fragment has no .wandb file: {path}")
        record["offline_run_wandb_files"] = [
            {"path": str(item.resolve()), "sha256": sha256_file(item), "size_bytes": item.stat().st_size}
            for item in files
        ]
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "offline_runs": [records[key] for key in sorted(records)],
        "sync_log": str(sync_log.resolve()),
        "sync_log_sha256": sha256_file(sync_log),
        "expected_offline_runs": sorted(expected),
        "coverage_complete_for_local_fragments": expected.issubset(records),
        "verified_at": synced_at,
    }
    atomic_json(marker_path, payload, read_only=True)
    return payload


def plain_value(value: Any) -> Any:
    """Materialize W&B SummarySubDict values for strict content comparison."""
    if isinstance(value, Mapping) or (hasattr(value, "keys") and hasattr(value, "__getitem__")):
        return {str(key): plain_value(value[key]) for key in value.keys()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value


def verified_remote_history_steps(remote: Any, expected_steps: set[int]) -> set[int]:
    """Read full rows because W&B may return no rows for a system-only key query."""
    remote_steps = {
        int(row["_step"])
        for row in remote.scan_history(page_size=1000)
        if isinstance(row.get("_step"), (int, float))
    }
    missing_steps = sorted(expected_steps - remote_steps)
    if missing_steps:
        raise RuntimeError(
            f"remote W&B history verification failed; missing {len(missing_steps)} steps"
        )
    return remote_steps


def validate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_CONFIG - set(config))
    if missing:
        raise RuntimeError(f"W&B stage config missing fields: {missing}")
    route = str(config["route"])
    if route not in ROUTES:
        raise RuntimeError(f"unsupported custom route: {route}")
    workflow = str(config["workflow_id"])
    stage = str(config["stage"])
    attempt = str(config["attempt"])
    name = str(config.get("run_name", ""))
    for token in (workflow, route, stage, attempt):
        if token not in name:
            raise RuntimeError(f"W&B run name does not contain {token!r}")
    if config.get("group") != workflow:
        raise RuntimeError("W&B group must equal workflow_id")
    for key in ("spec_sha256", "preregistration_sha256", "start_checkpoint_sha256", "teacher_sha256"):
        value = str(config[key])
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError(f"invalid SHA256 in W&B stage config: {key}")
    for path_key, sha_key in (("start_checkpoint", "start_checkpoint_sha256"), ("teacher_checkpoint", "teacher_sha256")):
        path = Path(str(config[path_key])).expanduser().resolve(strict=True)
        if sha256_file(path) != config[sha_key]:
            raise RuntimeError(f"W&B stage artifact binding changed: {path_key}")
    if config.get("historical_sync") is not False:
        raise RuntimeError("new custom training stages require historical_sync=false")
    if not isinstance(config["optimizer"], Mapping) or not isinstance(config["budget"], Mapping):
        raise RuntimeError("optimizer and budget must be structured mappings")
    return dict(config)


def assert_all_prior_synced(workflow_root: Path) -> None:
    pending = []
    for path in workflow_root.glob("wandb_stages/*/stage_manifest.json"):
        payload = json.loads(path.read_text())
        if payload.get("sync_status") != "synced":
            pending.append(str(path))
    if pending:
        raise RuntimeError(f"W&B sync gate blocks the next training stage: {pending}")


def prepare_stage(config_path: Path, workflow_root: Path) -> Path:
    assert_all_prior_synced(workflow_root)
    config_path = config_path.resolve(strict=True)
    config = validate_contract(json.loads(config_path.read_text()))
    stage_dir = workflow_root / "wandb_stages" / str(config["run_name"])
    manifest = stage_dir / "stage_manifest.json"
    if manifest.exists():
        raise RuntimeError(f"W&B stage run is not independent/unique: {manifest}")
    import wandb
    run_id = wandb.util.generate_id()
    payload = {
        "schema_version": 1, "kind": "highstep_custom_training_wandb_stage",
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "workflow_id": config["workflow_id"], "route": config["route"],
        "stage": config["stage"], "attempt": config["attempt"],
        "run_name": config["run_name"], "group": config["group"],
        "entity": ENTITY, "project": PROJECT, "run_id": run_id,
        "run_url": f"https://wandb.ai/{ENTITY}/{PROJECT}/runs/{run_id}",
        "sync_status": "prepared", "historical_sync": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(manifest, payload)
    atomic_json(stage_dir / "config.snapshot.json", config, read_only=True)
    return manifest


def process_environment(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((manifest_path.parent / "config.snapshot.json").read_text())
    validate_contract(config)
    return {
        "WANDB_ENTITY": manifest["entity"], "WANDB_PROJECT": manifest["project"],
        "WANDB_RUN_ID": manifest["run_id"], "WANDB_NAME": manifest["run_name"],
        "WANDB_RUN_GROUP": manifest["group"], "WANDB_RESUME": "allow",
        "HIGHSTEP_WANDB_STAGE_MANIFEST": str(manifest_path.resolve()),
    }


def finalize_stage(manifest_path: Path, summary_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    config = json.loads((manifest_path.parent / "config.snapshot.json").read_text())
    summary_path = summary_path.resolve(strict=True)
    summary = json.loads(summary_path.read_text())
    missing = sorted(REQUIRED_SUMMARY - set(summary))
    if missing:
        raise RuntimeError(f"W&B stage summary missing fields: {missing}")
    if sha256_file(summary["output_checkpoint"]) != summary["output_checkpoint_sha256"]:
        raise RuntimeError("W&B output checkpoint SHA binding changed")
    if sha256_file(summary["manifest_path"]) != summary["manifest_sha256"]:
        raise RuntimeError("W&B output manifest SHA binding changed")
    import wandb
    try:
        run = wandb.init(
            entity=manifest["entity"], project=manifest["project"], id=manifest["run_id"],
            name=manifest["run_name"], group=manifest["group"], resume="allow",
            config=config, settings=wandb.Settings(init_timeout=60),
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        run.config.update(config, allow_val_change=False)
        run.summary.update({**summary, "historical_sync": False})
        run.finish()
        remote = wandb.Api(timeout=60).run(f"{manifest['entity']}/{manifest['project']}/{manifest['run_id']}")
        remote_config = plain_value(dict(remote.config))
        if any(
            plain_value(remote_config.get(key)) != plain_value(value)
            for key, value in config.items()
        ):
            raise RuntimeError("remote W&B config verification failed")
        remote_summary = dict(remote.summary)
        if any(
            plain_value(remote_summary.get(key)) != plain_value(value)
            for key, value in {**summary, "historical_sync": False}.items()
        ):
            raise RuntimeError("remote W&B summary verification failed")
        marker_path = manifest_path.parent / "offline_training_run_synced.json"
        expected_steps = expected_offline_history_steps(marker_path, str(manifest["run_id"]))
        remote_steps = verified_remote_history_steps(remote, expected_steps)
        manifest.pop("sync_error", None)
        manifest.update({"sync_status": "synced", "summary_path": str(summary_path),
                         "summary_sha256": sha256_file(summary_path),
                         "remote_history_verification": {
                             "expected_unique_steps": len(expected_steps),
                             "remote_unique_steps": len(remote_steps),
                             "step_min": min(expected_steps), "step_max": max(expected_steps),
                             "missing_steps": 0,
                         },
                         "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    except Exception as error:
        manifest.update({"sync_status": "pending_sync", "sync_error": f"{type(error).__name__}: {error}",
                         "summary_path": str(summary_path), "summary_sha256": sha256_file(summary_path)})
    atomic_json(manifest_path, manifest)
    return manifest


def verify_historical_b500(output: Path) -> dict[str, Any]:
    import wandb
    run = wandb.Api(timeout=60).run(f"{ENTITY}/{PROJECT}/b500best")
    expected = {
        "name": "highstep_student_B500_behavior_best_20260713",
        "checkpoint_sha256": "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159",
        "historical_sync": True,
    }
    if run.name != expected["name"] or any(run.config.get(key) != value for key, value in expected.items() if key != "name"):
        raise RuntimeError("historical B500 W&B config verification failed")
    payload = {"schema_version": 1, "kind": "historical_wandb_sync_verification",
               "entity": ENTITY, "project": PROJECT, "run_id": run.id, "name": run.name,
               "url": run.url, **{key: value for key, value in expected.items() if key != "name"},
               "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(output, payload, read_only=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--config", type=Path, required=True); prepare.add_argument("--workflow-root", type=Path, required=True)
    finalize = sub.add_parser("finalize"); finalize.add_argument("--manifest", type=Path, required=True); finalize.add_argument("--summary", type=Path, required=True)
    verify = sub.add_parser("verify-historical-b500"); verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": print(prepare_stage(args.config, args.workflow_root))
    elif args.command == "finalize": print(json.dumps(finalize_stage(args.manifest, args.summary), sort_keys=True))
    else: print(json.dumps(verify_historical_b500(args.output), sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
