#!/usr/bin/env python3
"""Autonomous fail-closed supervisor for the preregistered directional trial route."""

from __future__ import annotations

import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path("/home/lxq/Softwares/robot_lab")
sys.path.insert(0, str(ROOT))
from tools import highstep_student_recovery_r2_supervisor as recovery
from tools import highstep_wandb_stage_gate as wandb_stage_gate

WORKFLOW_ID = "highstep_directional_real_robot_trial_20260713"
STATE_ROOT = ROOT / "tmp/highstep_directional_real_robot_trial_20260713"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
PREREG = STATE_ROOT / "preregistration.json"
SPEC_SHA256 = "1fbb17ab6c23a24005e8e34dadc11a6e4592a7be671a575206fd24819e5f3f1c"
CURRENT_SPEC_SHA256 = "4baed191f98f9b746eec9181b3f31bcdd16e3cc147726b676d9949d7e1fe4425"
WANDB_AMENDMENT_SPEC_SHA256 = "8e536923ee057e30e9eac7d203b20aa26a311fb6ca258bc54a1d88c74c0feac1"
PREREG_SHA256 = "6cfe31cf04e1321416673c4d13e795773f678b529fcc7f147201f46f1048223d"
REJECTED_STATE_SHA256 = "978831b49c023bab44f0d61efb8de409de99a13e2954ca3027d3e99175651fed"
REJECTED_HANDOFF_SHA256 = "5e846e85e6a23f626251750de763b807e647d0cb230aca9b3dd8ae2df902ce08"
REJECTION_AMENDMENT_SHA256 = "980965787b8e280ccda4029cb1c67e5d780bc537f0bf3101f02b2ca1bf07c4e3"
B500_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0"
TRACE_SCRIPT = ROOT / "scripts/rsl_rl/base/highstep_directional_same_state_audit_play.py"
R5_SCRIPT = ROOT / "tools/highstep_v13_r5_train.py"
R5_DATASET = ROOT / "tmp/highstep_student_recovery_v13_20260713/replay_dataset_manifest.json"
OLD_STATE = ROOT / "tmp/highstep_student_recovery_v13_20260713/state.json"
OLD_HANDOFF = ROOT / "tmp/highstep_student_recovery_v13_20260713/handoff.json"
SERVICE = ROOT / "scripts/systemd/highstep-directional-trial.service"
REPAIR_SERVICE = ROOT / "scripts/systemd/highstep-directional-trial-repair.service"
REPAIR_SCRIPT = ROOT / "tools/highstep_directional_failure_wake_codex.py"

def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def video_delivery_qc(video: Path, log: Path) -> dict[str, Any]:
    """Fail closed when a delivered video is not bound to env0's moving robot camera."""
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
    stream = payload.get("streams", [{}])[0]
    frame_count = int(stream.get("nb_frames", 0))
    duration = float(payload.get("format", {}).get("duration", 0.0))
    log_text = log.read_text(errors="replace")
    camera_witness = "[HIGHSTEP_CAMERA] mode=side_top env=0" in log_text
    passed = bool(
        camera_witness
        and int(stream.get("width", 0)) >= 640
        and int(stream.get("height", 0)) >= 360
        and frame_count >= 599
        and duration >= 11.9
    )
    result = {
        "passed": passed,
        "camera_mode": "side_top",
        "camera_runtime_witness": camera_witness,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_count": frame_count,
        "duration_s": duration,
    }
    if not passed:
        raise RuntimeError(f"video delivery QC failed: {video}: {result}")
    return result


def validate_directional_probe_payload(
    payload: Mapping[str, Any], checkpoint: Path
) -> dict[str, Any]:
    """Validate one payload against its real source-task alias without leakage."""

    previous_train, previous_eval = recovery.TRAIN_TASK, recovery.EVAL_TASK
    try:
        # The reusable validator defaults to R2's training-only task.  These
        # directional checkpoints and their immutable schedule sidecars use
        # the unchanged deployment-compatible task for both train and play.
        recovery.TRAIN_TASK = TASK
        recovery.EVAL_TASK = TASK
        return recovery.validate_probe_payload(payload, checkpoint)
    finally:
        recovery.TRAIN_TASK = previous_train
        recovery.EVAL_TASK = previous_eval


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def is_read_only(path: Path) -> bool:
    return path.is_file() and not bool(path.stat().st_mode & 0o222)


def critical_paths() -> tuple[Path, ...]:
    paths = (
        SPEC, PREREG, TRACE_SCRIPT, R5_SCRIPT, SERVICE, REPAIR_SERVICE, REPAIR_SCRIPT,
        Path(__file__).resolve(), ROOT / "scripts/rsl_rl/base/play.py",
        ROOT / "scripts/rsl_rl/base/highstep_same_state_audit_play.py",
        ROOT / "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py",
        ROOT / "tools/highstep_same_state_audit.py",
        ROOT / "tools/highstep_student_recovery_r2_supervisor.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py",
        ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/Arcdog_adjustable_leg/highstep_env_cfg.py",
        ROOT / "tests/test_highstep_directional_trial.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"directional critical files missing: {missing}")
    return paths


def active_jobs() -> list[dict[str, Any]]:
    suffixes = ("play.py", "train.py", "highstep_v13_r5_train.py", "highstep_directional_same_state_audit_play.py")
    result: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = [part.decode(errors="replace") for part in (entry / "cmdline").read_bytes().split(b"\0") if part]
        except OSError:
            continue
        tokens = [token.replace("\\", "/").rstrip("/") for token in argv if token and not any(c.isspace() for c in token)]
        if any(any(token == suffix or token.endswith("/" + suffix) for suffix in suffixes) for token in tokens):
            result.append({"pid": int(entry.name), "argv": " ".join(argv)})
    return result


def validate_user_rejected_terminal() -> dict[str, Any]:
    """Bind the historical route to the later user-rejection terminal.

    This check intentionally runs before the obsolete v1.4 preflight.  It may
    preserve the terminal, but it cannot resume evaluation, training, export,
    or deployment under the historical preregistration.
    """

    state_path = STATE_ROOT / "state.json"
    handoff_path = STATE_ROOT / "handoff.json"
    amendment_path = STATE_ROOT / "v15_user_rejection_amendment.json"
    for path, expected in (
        (SPEC, CURRENT_SPEC_SHA256),
        (PREREG, PREREG_SHA256),
        (state_path, REJECTED_STATE_SHA256),
        (handoff_path, REJECTED_HANDOFF_SHA256),
        (amendment_path, REJECTION_AMENDMENT_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"user-rejected terminal authority mismatch: {path}")
    if not is_read_only(PREREG):
        raise RuntimeError("historical directional preregistration is not read-only")

    prereg = json.loads(PREREG.read_text())
    state = json.loads(state_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    old_state = json.loads(OLD_STATE.read_text())
    old_handoff = json.loads(OLD_HANDOFF.read_text())
    checkpoint = Path(prereg["checkpoints"][0]["path"]).resolve(strict=True)
    if sha256_file(checkpoint) != B500_SHA256:
        raise RuntimeError("user-rejected B500 checkpoint SHA mismatch")
    if not (
        state.get("workflow_id") == WORKFLOW_ID
        and state.get("status") == "user_rejected_non_deployable"
        and state.get("phase") == "user_visual_rejection_recorded"
        and state.get("checkpoint") == str(checkpoint)
        and state.get("supervisor_pid") is None
        and handoff.get("workflow_id") == WORKFLOW_ID
        and handoff.get("status") == "user_rejected_non_deployable"
        and handoff.get("candidate", {}).get("state") == "user_rejected_non_deployable"
        and handoff.get("candidate", {}).get("checkpoint_sha256") == B500_SHA256
        and handoff.get("automatic_real_robot_deployment_performed") is False
        and amendment.get("kind") == "highstep_v15_user_rejection_amendment"
        and amendment.get("workflow_id") == WORKFLOW_ID
        and amendment.get("current_status") == "user_rejected_non_deployable"
        and amendment.get("checkpoint_sha256") == B500_SHA256
        and amendment.get("continued_training_forbidden") is True
        and amendment.get("deployment_forbidden") is True
        and old_state.get("status") == "stopped_by_gate"
        and old_handoff.get("status") == "stopped_by_gate"
    ):
        raise RuntimeError("user-rejected terminal contract is inconsistent")
    return {
        "schema_version": 1,
        "kind": "directional_user_rejected_terminal_guard",
        "workflow_id": WORKFLOW_ID,
        "status": "user_rejected_non_deployable",
        "current_spec_sha256": CURRENT_SPEC_SHA256,
        "historical_preregistration_sha256": PREREG_SHA256,
        "state_sha256": REJECTED_STATE_SHA256,
        "handoff_sha256": REJECTED_HANDOFF_SHA256,
        "rejection_amendment_sha256": REJECTION_AMENDMENT_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": B500_SHA256,
        "old_v13_status_preserved": "stopped_by_gate",
        "evaluation_started": False,
        "training_started": False,
        "export_started": False,
        "automatic_real_robot_deployment_performed": False,
    }


class Supervisor:
    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.state: dict[str, Any] = {"workflow_id": WORKFLOW_ID, "status": "initializing", "phase": "preflight"}
        if self.state_path.is_file():
            old = json.loads(self.state_path.read_text())
            if old.get("workflow_id") == WORKFLOW_ID:
                self.state.update(old)
        self.process: subprocess.Popen[bytes] | None = None
        self.hashes: dict[str, str] = {}

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state.update({"workflow_id": WORKFLOW_ID, "spec_sha256": SPEC_SHA256,
                           "preregistration_sha256": PREREG_SHA256, "updated_at": now(),
                           "supervisor_pid": os.getpid()})
        atomic_json(self.state_path, self.state)
        atomic_json(self.heartbeat_path, {"workflow_id": WORKFLOW_ID, "timestamp": now(),
                                         "pid": os.getpid(), "status": self.state.get("status"),
                                         "phase": self.state.get("phase"),
                                         "child_pid": self.process.pid if self.process else None})

    def stop_child(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.process = None

    def wait(self, process: subprocess.Popen[bytes], phase: str, stall: int = 1200) -> None:
        self.process = process
        last_progress = time.monotonic(); last_size = -1
        while process.poll() is None:
            self.update(status="running", phase=phase)
            size = sum(path.stat().st_size for path in STATE_ROOT.rglob("*.log") if path.is_file())
            if size != last_size:
                last_size = size; last_progress = time.monotonic()
            if time.monotonic() - last_progress > stall:
                self.stop_child(); raise RuntimeError(f"no observable progress for {stall}s in {phase}")
            time.sleep(5)
        code = process.returncode; self.process = None
        if code != 0:
            raise RuntimeError(f"{phase} returned {code}")

    def assert_unchanged(self) -> None:
        for path, expected in self.hashes.items():
            if sha256_file(path) != expected:
                raise RuntimeError(f"critical file changed after preflight: {path}")

    def preflight(self) -> dict[str, Any]:
        if active_jobs():
            raise RuntimeError(f"another train/eval/play job is active: {active_jobs()}")
        if sha256_file(SPEC) != SPEC_SHA256 or sha256_file(PREREG) != PREREG_SHA256 or not is_read_only(PREREG):
            raise RuntimeError("directional spec/preregistration authority mismatch")
        prereg = json.loads(PREREG.read_text())
        for row in prereg["checkpoints"]:
            if sha256_file(row["path"]) != row["sha256"]:
                raise RuntimeError(f"checkpoint SHA mismatch: {row['label']}")
        old_state = json.loads(OLD_STATE.read_text()); old_handoff = json.loads(OLD_HANDOFF.read_text())
        if old_state.get("status") != "stopped_by_gate" or old_handoff.get("status") != "stopped_by_gate":
            raise RuntimeError("old v1.3 stopped_by_gate fact is not preserved")
        self.hashes = {str(path.resolve()): sha256_file(path) for path in critical_paths()}
        manifest = {"schema_version": 1, "kind": "directional_preflight", "workflow_id": WORKFLOW_ID,
                    "created_at": now(), "spec_sha256": SPEC_SHA256, "preregistration_sha256": PREREG_SHA256,
                    "old_v13_status_preserved": "stopped_by_gate", "no_concurrent_job": True,
                    "training_initially_allowed": False, "critical_file_sha256": self.hashes}
        binding = hashlib.sha256(json.dumps(self.hashes, sort_keys=True).encode()).hexdigest()[:16]
        preflight_path = STATE_ROOT / f"preflight_{binding}.json"
        if preflight_path.exists():
            existing = json.loads(preflight_path.read_text())
            if not (
                is_read_only(preflight_path)
                and existing.get("workflow_id") == WORKFLOW_ID
                and existing.get("spec_sha256") == SPEC_SHA256
                and existing.get("preregistration_sha256") == PREREG_SHA256
                and existing.get("old_v13_status_preserved") == "stopped_by_gate"
                and existing.get("no_concurrent_job") is True
                and existing.get("critical_file_sha256") == self.hashes
            ):
                raise RuntimeError("existing directional preflight binding changed")
        else:
            atomic_json(preflight_path, manifest, read_only=True)
        self.update(status="preflight_passed", phase="B500_behavior")
        return prereg

    @staticmethod
    def matrix(prereg: Mapping[str, Any], corridor: str) -> list[dict[str, Any]]:
        return [{"seed": int(seed), **item} for seed in prereg["seeds"] for item in prereg["corridors"][corridor]]

    @staticmethod
    def gate(rows: Sequence[Mapping[str, Any]], half: str) -> tuple[bool, dict[str, int]]:
        counts = {key: sum(row.get(key) is True for row in rows) for key in
                  ("valid", "full_climb", "rear_hold", "front_top_support", "no_severe_inward")}
        half_rows = [row for row in rows if row["scenario"] == half]
        counts["half_full"] = sum(row.get("full_climb") is True for row in half_rows)
        counts["half_rear_hold"] = sum(row.get("rear_hold") is True for row in half_rows)
        passed = (len(rows) == 9 and counts["valid"] == 9 and counts["full_climb"] >= 8
                  and counts["rear_hold"] >= 8 and counts["front_top_support"] >= 8
                  and counts["no_severe_inward"] >= 8 and counts["half_full"] == 3
                  and counts["half_rear_hold"] == 3)
        return passed, counts

    def evaluate(self, checkpoint: Path, label: str, corridor: str, prereg: Mapping[str, Any]) -> dict[str, Any]:
        output = STATE_ROOT / "evaluations" / label
        terminal = output / "manifest.json"
        if terminal.is_file():
            payload = json.loads(terminal.read_text());
            if payload.get("checkpoint_sha256") != sha256_file(checkpoint) or not is_read_only(terminal):
                raise RuntimeError(f"stale directional evaluation: {label}")
            return payload
        output.mkdir(parents=True, exist_ok=True)
        plan = {"schema_version": 1, "workflow_id": WORKFLOW_ID, "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "corridor": corridor,
                "matrix": self.matrix(prereg, corridor), "gate": prereg["directional_gate"],
                "threshold_changes_after_results_allowed": False}
        atomic_json(output / "plan.json", plan, read_only=True)
        rows: list[dict[str, Any]] = []
        for item in plan["matrix"]:
            self.assert_unchanged()
            if active_jobs(): raise RuntimeError(f"concurrent job before eval: {active_jobs()}")
            run_id = f"seed{item['seed']}_{item['scenario']}"
            result_path = output / f"{run_id}.json"
            if result_path.is_file():
                row = json.loads(result_path.read_text()); rows.append(row); continue
            attempt = 1 + len(list(output.glob(f"{run_id}_attempt*.log")))
            log = output / f"{run_id}_attempt{attempt}.log"
            command = recovery.probe_play_command(checkpoint, seed=item["seed"],
                lateral=f"{item['lateral_offset_m']:.2f}", yaw=f"{item['yaw_offset_deg']:.1f}")
            atomic_json(output / f"{run_id}_attempt{attempt}.launch.json", {"command": command, **item,
                        "checkpoint_sha256": sha256_file(checkpoint)}, read_only=True)
            env = os.environ.copy(); env.pop("DISPLAY", None); env.pop("XAUTHORITY", None)
            with log.open("wb") as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.wait(proc, f"{label}_{run_id}")
            payload = recovery.R2Supervisor._single_eval_payload(log)
            if abs(float(payload.get("yaw_offset_deg", 999)) - float(item["yaw_offset_deg"])) > 1e-8:
                raise RuntimeError(f"yaw geometry mismatch: {run_id}")
            validated = validate_directional_probe_payload(payload, checkpoint)
            row = {**item, **validated, "log": str(log), "log_sha256": sha256_file(log)}
            atomic_json(result_path, row, read_only=True); rows.append(row)
        half = "half_left" if corridor.startswith("positive") else "half_right"
        passed, counts = self.gate(rows, half)
        terminal_payload = {"schema_version": 1, "kind": "directional_behavior_evaluation",
            "workflow_id": WORKFLOW_ID, "label": label, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint), "corridor": corridor, "rows": rows,
            "counts": counts, "passed": passed, "completed_at": now()}
        atomic_json(terminal, terminal_payload, read_only=True)
        self.update(status="running", phase=f"{label}_gate_complete", checkpoint=str(checkpoint), behavior=terminal_payload)
        return terminal_payload

    def trace_matrix(self, checkpoint: Path, label: str, corridor: str, prereg: Mapping[str, Any]) -> dict[str, Any]:
        root = STATE_ROOT / "teacher_action_regression" / label
        terminal = root / "manifest.json"
        if terminal.is_file():
            payload = json.loads(terminal.read_text());
            if payload.get("checkpoint_sha256") != sha256_file(checkpoint): raise RuntimeError("stale trace manifest")
            return payload
        records: list[dict[str, Any]] = []
        for item in self.matrix(prereg, corridor):
            run_id = f"seed{item['seed']}_{item['scenario']}"; stage = root / "stage" / f"{run_id}.json"
            if stage.is_file(): records.append(json.loads(stage.read_text())); continue
            attempt = root / "attempts" / run_id; attempt.mkdir(parents=True, exist_ok=True)
            log = attempt / "trace.log"
            command = [str(PYTHON), "-u", str(TRACE_SCRIPT), "--trajectory", run_id,
                       "--output-root", str(attempt), "--task", TASK, "--num_envs", "1",
                       "--checkpoint", str(checkpoint), "--headless"]
            env = os.environ.copy(); env.pop("DISPLAY", None); env.pop("XAUTHORITY", None)
            env.update({"LD_PRELOAD": "/lib/x86_64-linux-gnu/libstdc++.so.6",
                        "HIGHSTEP_DIRECTIONAL_CHECKPOINT": str(checkpoint),
                        "HIGHSTEP_DIRECTIONAL_CHECKPOINT_SHA256": sha256_file(checkpoint),
                        "HIGHSTEP_DIRECTIONAL_CORRIDOR": corridor})
            atomic_json(attempt / "launch.json", {"command": command, "environment_binding": {
                "checkpoint_sha256": sha256_file(checkpoint), "corridor": corridor}}, read_only=True)
            with log.open("wb") as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.wait(proc, f"trace_{label}_{run_id}")
            summary = attempt / run_id / "run_summary.json"; payload = json.loads(summary.read_text())
            frames = Path(payload["frames_path"]); errors: list[float] = []
            for line in frames.read_text().splitlines():
                record = json.loads(line); errors.extend(abs(float(value)) for value in record["action_error_student_minus_teacher"])
            row = {**item, "run_id": run_id, "frames": str(frames), "frames_sha256": sha256_file(frames),
                   "summary": str(summary), "summary_sha256": sha256_file(summary),
                   "teacher_action_post_prior_mae": sum(errors) / len(errors), "samples": len(errors) // 16,
                   "integrity_passed": payload.get("frame_count") == 600 and
                       float(payload.get("action_decomposition_residual_global_max_abs", 1)) <= 1e-6}
            if not row["integrity_passed"] or row["samples"] != 600: raise RuntimeError(f"trace integrity failed: {run_id}")
            frames.chmod(0o444); summary.chmod(0o444); atomic_json(stage, row, read_only=True); records.append(row)
        result = {"schema_version": 1, "kind": "directional_teacher_action_regression",
                  "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                  "corridor": corridor, "matrix_complete": len(records) == 9,
                  "total_frames": sum(row["samples"] for row in records),
                  "teacher_action_post_prior_mae": sum(row["teacher_action_post_prior_mae"] for row in records) / len(records),
                  "all_integrity_passed": all(row["integrity_passed"] for row in records), "rows": records}
        atomic_json(terminal, result, read_only=True); return result

    def action_gate(self, checkpoint: Path, label: str, corridor: str, prereg: Mapping[str, Any]) -> dict[str, Any]:
        candidate = self.trace_matrix(checkpoint, label, corridor, prereg)
        if checkpoint == Path(prereg["checkpoints"][0]["path"]).resolve():
            reference = candidate; passed = candidate["all_integrity_passed"] and candidate["total_frames"] == 5400
        else:
            b500 = Path(prereg["checkpoints"][0]["path"]).resolve()
            reference = self.trace_matrix(b500, f"B500_reference_{corridor}", corridor, prereg)
            passed = (candidate["all_integrity_passed"] and candidate["total_frames"] == 5400
                      and candidate["teacher_action_post_prior_mae"] <= 1.05 * reference["teacher_action_post_prior_mae"])
        result = {"passed": bool(passed), "candidate": candidate, "reference_checkpoint_sha256": reference["checkpoint_sha256"],
                  "reference_mae": reference["teacher_action_post_prior_mae"], "relative_limit": 1.05,
                  "lineage_and_schedule_review": "passed_by_behavior_payload_and_trace_checkpoint_binding"}
        atomic_json(STATE_ROOT / "teacher_action_regression" / f"{label}_gate.json", result, read_only=True)
        return result

    def train_r5(self, source: Path, label: str, additional: int, total: int) -> Path:
        result = STATE_ROOT / "r5_stage_results" / f"{label}.json"
        if result.is_file(): return Path(json.loads(result.read_text())["checkpoint"]).resolve(strict=True)
        output = STATE_ROOT / "r5_runs" / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}"; log = output.with_suffix(".log")
        prereg = json.loads((ROOT / "tmp/highstep_student_recovery_v13_20260713/r5_preregistration.json").read_text())
        teacher = Path(str(prereg["checkpoint_binding"]["teacher"])).resolve(strict=True)
        attempt = 1 + len(list((STATE_ROOT / "wandb_stages").glob(f"*_{label}_attempt*")))
        run_name = f"{WORKFLOW_ID}_R5_{label}_attempt{attempt}"
        wandb_config = STATE_ROOT / "wandb_stage_configs" / f"{run_name}.json"
        atomic_json(wandb_config, {
            "workflow_id": WORKFLOW_ID, "route": "R5", "stage": label,
            "attempt": f"attempt{attempt}", "run_name": run_name, "group": WORKFLOW_ID,
            "spec_sha256": WANDB_AMENDMENT_SPEC_SHA256,
            "preregistration_sha256": sha256_file(ROOT / "tmp/highstep_student_recovery_v13_20260713/r5_preregistration.json"),
            "start_checkpoint": str(source), "start_checkpoint_sha256": sha256_file(source),
            "teacher_checkpoint": str(teacher), "teacher_sha256": sha256_file(teacher),
            "frozen_tensors": "all except actor.4.weight/bias and actor.6.weight/bias",
            "trainable_tensors": ["actor.4.weight", "actor.4.bias", "actor.6.weight", "actor.6.bias"],
            "optimizer": {"class": "Adam", "lr": 1e-6, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
            "budget": {"additional_epochs": additional, "effective_epoch_target": total, "absolute_cap": 5},
            "training_task": "fixed_replay_highstep_student_R5", "historical_sync": False,
        }, read_only=True)
        wandb_manifest = wandb_stage_gate.prepare_stage(wandb_config, STATE_ROOT)
        command = [str(PYTHON), "-u", str(R5_SCRIPT), "--dataset-manifest", str(R5_DATASET),
                   "--checkpoint", str(source), "--load-mode", "full", "--additional-epochs", str(additional),
                   "--output-dir", str(output)]
        atomic_json(log.with_suffix(".launch.json"), {"command": command, "source_sha256": sha256_file(source),
                    "effective_epoch_target": total}, read_only=True)
        environment = os.environ.copy(); environment.update(wandb_stage_gate.process_environment(wandb_manifest))
        with log.open("wb") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            self.wait(proc, f"train_{label}", stall=900)
        manifest = json.loads((output / "candidate_manifest.json").read_text()); checkpoint = Path(manifest["checkpoint"]).resolve(strict=True)
        if int(manifest["effective_epochs"]) != total: raise RuntimeError("R5 epoch total mismatch")
        record = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "effective_epochs": total, "source": str(source), "source_sha256": sha256_file(source),
                  "log": str(log), "log_sha256": sha256_file(log),
                  "wandb_stage_manifest": str(wandb_manifest)}
        atomic_json(result, record, read_only=True); return checkpoint

    def finalize_r5_wandb(self, label: str, checkpoint: Path, behavior: Mapping[str, Any], conclusion: str) -> dict[str, Any]:
        stage_result = STATE_ROOT / "r5_stage_results" / f"{label}.json"
        record = json.loads(stage_result.read_text())
        evaluation_manifest = STATE_ROOT / "evaluations" / f"{label}_right_corridor" / "manifest.json"
        summary = STATE_ROOT / "wandb_stage_summaries" / f"{label}.json"
        atomic_json(summary, {
            "output_checkpoint": str(checkpoint), "output_checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": record["effective_epochs"], "gate_metrics": behavior["counts"],
            "gate_conclusion": conclusion, "manifest_path": str(evaluation_manifest),
            "manifest_sha256": sha256_file(evaluation_manifest),
        }, read_only=True)
        return wandb_stage_gate.finalize_stage(Path(record["wandb_stage_manifest"]), summary)

    def placement_card(self, candidate_dir: Path, corridor: str) -> Path:
        sign = 1 if corridor.startswith("positive") else -1
        card = candidate_dir / "placement_card.html"
        offsets = [("nominal", 0, 0), ("half-offset", .06 * sign, 2 * sign), ("outer-offset", .12 * sign, 4 * sign)]
        svg_rows = "".join(f'<g transform="translate({360 + y*900},{410-i*95}) rotate({yaw})"><rect x="-55" y="-25" width="110" height="50" rx="12" fill="#3b82f6"/><circle cx="0" cy="0" r="5" fill="white"/></g><text x="40" y="{415-i*95}">{html.escape(name)}: Δy={y:+.2f} m, yaw={yaw:+.0f}°</text>' for i,(name,y,yaw) in enumerate(offsets))
        content = f'''<!doctype html><meta charset="utf-8"><title>Directional placement card</title>
<style>body{{font:18px sans-serif;max-width:1000px;margin:30px auto}}svg{{border:1px solid #999;background:#fafafa}}.warn{{color:#b91c1c;font-weight:bold}}</style>
<h1>单侧标定工作域摆位卡（非通用鲁棒候选）</h1><p class="warn">必须有人保护；禁止自动真机部署。</p>
<svg viewBox="0 0 900 620"><rect x="620" y="40" width="240" height="540" fill="#d1d5db"/><line x1="620" x2="620" y1="40" y2="580" stroke="#111" stroke-width="6"/><line x1="740" x2="740" y1="40" y2="580" stroke="#ef4444" stroke-dasharray="10 8"/><text x="700" y="30">平台中心线</text><path d="M620 310 L540 310" stroke="#16a34a" stroke-width="5" marker-end="url(#a)"/><text x="440" y="290">平台边缘外法向</text><defs><marker id="a" markerWidth="10" markerHeight="10" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#16a34a"/></marker></defs>{svg_rows}<line x1="360" x2="620" y1="520" y2="520" stroke="#7c3aed"/><text x="390" y="550">机器人中心到平台起步距离 2.05 m；前缘净距 0.55 m</text><path d="M360 480 L520 480" stroke="#f97316" stroke-width="5" marker-end="url(#a)"/><text x="380" y="465">命令 v=(+0.45, 0, 0) m/s</text></svg>
<p>俯视坐标定义：+x 指向平台；+y 为图中向上。真实横移和偏航按图示有符号数值标定，不以“左/右”文字猜测。</p>'''
        card.write_text(content); return card

    def finalize(self, checkpoint: Path, label: str, corridor: str, behavior: Mapping[str, Any], action: Mapping[str, Any], prereg: Mapping[str, Any]) -> None:
        candidate = STATE_ROOT / "candidate" / label; videos = candidate / "videos"; logs = candidate / "video_logs"
        videos.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
        selected = [row for row in self.matrix(prereg, corridor) if row["seed"] == 11]
        records = []
        for index, item in enumerate(selected):
            before = {path: path.stat().st_mtime_ns for path in (checkpoint.parent / "videos/play").glob("*.mp4")}
            command = recovery.probe_play_command(checkpoint, seed=item["seed"], lateral=f"{item['lateral_offset_m']:.2f}", yaw=f"{item['yaw_offset_deg']:.1f}")
            command.extend([
                "--video", "--video_length", "600",
                "--highstep_gap_camera", "side_top",
            ])
            if index == 0: command.remove("--skip_policy_export")
            log = logs / f"{item['scenario']}.log"
            env = os.environ.copy(); env.pop("DISPLAY", None); env.pop("XAUTHORITY", None)
            with log.open("wb") as stream:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                self.wait(proc, f"video_{item['scenario']}")
            source_dir = checkpoint.parent / "videos/play"
            created = [path for path in source_dir.glob("*.mp4") if path.stat().st_mtime_ns > before.get(path, 0)]
            if not created: raise RuntimeError(f"video missing: {item['scenario']}")
            destination = videos / f"{item['scenario']}.mp4"; shutil.copy2(max(created, key=lambda p:p.stat().st_mtime_ns), destination)
            video_qc = video_delivery_qc(destination, log)
            records.append({**item, "video": str(destination), "video_sha256": sha256_file(destination),
                            "log_sha256": sha256_file(log), "video_delivery_qc": video_qc})
        exported = checkpoint.parent / "exported/policy_student.pt"
        if not exported.is_file(): raise RuntimeError("exact Student export missing")
        policy = candidate / "policy_student.pt"; shutil.copy2(exported, policy)
        card = self.placement_card(candidate, corridor)
        deploy = {"schema_version": 1, "state": "directional_real_robot_trial_candidate",
                  "not_a_general_robust_candidate": True, "automatic_real_robot_deployment_allowed": False,
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "policy": str(policy), "policy_sha256": sha256_file(policy), "corridor": corridor,
                  "behavior_gate": behavior["counts"], "teacher_action_regression": action,
                  "videos": records, "placement_card": str(card), "placement_card_sha256": sha256_file(card),
                  "real_robot_conditions": {"human_protection_required": True, "command": [0.45,0,0],
                      "start_distance_m": 2.05, "front_edge_clearance_m": .55}}
        atomic_json(candidate / "deployment_manifest.json", deploy, read_only=True)
        handoff = {"workflow_id": WORKFLOW_ID, "status": "directional_real_robot_trial_candidate",
                   "not_a_general_robust_candidate": True, "candidate": deploy, "completed_at": now(),
                   "user_action_required": True, "automatic_real_robot_deployment_performed": False}
        atomic_json(self.handoff_path, handoff)
        self.update(status="directional_real_robot_trial_candidate", phase="candidate_ready",
                    checkpoint=str(checkpoint), candidate_manifest=str(candidate / "deployment_manifest.json"))

    def terminal(self, status: str, reason: str) -> None:
        atomic_json(self.handoff_path, {"workflow_id": WORKFLOW_ID, "status": status, "reason": reason,
                    "not_a_general_robust_candidate": True, "completed_at": now()})
        self.update(status=status, phase="terminal", stop_reason=reason)

    def infrastructure_repair_authorized(self, handoff: Mapping[str, Any]) -> bool:
        repair_path = STATE_ROOT / "repair_authorization.json"
        if not is_read_only(repair_path):
            return False
        try:
            repair = json.loads(repair_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            handoff.get("status") == "failed_closed"
            and handoff.get("reason")
            == "RuntimeError: R2 evaluation schedule/task/lineage/runtime binding is invalid"
            and repair.get("workflow_id") == WORKFLOW_ID
            and repair.get("kind") == "directional_infrastructure_repair_authorization"
            and repair.get("failed_handoff_sha256") == sha256_file(self.handoff_path)
            and repair.get("repaired_supervisor_sha256") == sha256_file(Path(__file__).resolve())
            and repair.get("spec_sha256") == SPEC_SHA256
            and repair.get("preregistration_sha256") == PREREG_SHA256
            and repair.get("full_test_count") == 170
            and repair.get("full_tests_passed") is True
            and repair.get("recorded_payload_revalidated") is True
            and repair.get("retry_allowed") is True
        )

    def run(self) -> None:
        if self.handoff_path.is_file():
            handoff = json.loads(self.handoff_path.read_text())
            if handoff.get("status") == "user_rejected_non_deployable":
                if active_jobs():
                    raise RuntimeError(f"job active while preserving user-rejected terminal: {active_jobs()}")
                witness = validate_user_rejected_terminal()
                witness_path = STATE_ROOT / "terminal_guard_v161.json"
                if witness_path.exists():
                    existing = json.loads(witness_path.read_text())
                    if not is_read_only(witness_path) or existing != witness:
                        raise RuntimeError("user-rejected terminal guard witness changed")
                else:
                    atomic_json(witness_path, witness, read_only=True)
                atomic_json(self.heartbeat_path, {
                    "workflow_id": WORKFLOW_ID,
                    "timestamp": now(),
                    "pid": None,
                    "status": "user_rejected_non_deployable",
                    "phase": "terminal_handoff_preserved",
                    "child_pid": None,
                })
                print("[DIRECTIONAL_TERMINAL_GUARD] user_rejected_non_deployable preserved; no job started")
                return
            if handoff.get("status") in {"directional_real_robot_trial_candidate", "stopped_by_gate"}:
                self.update(status=handoff["status"], phase="terminal_handoff_preserved"); return
            if handoff.get("status") == "failed_closed":
                if not self.infrastructure_repair_authorized(handoff):
                    self.update(status="failed_closed", phase="terminal_handoff_preserved"); return
                self.update(status="infrastructure_retry_authorized", phase="preflight",
                            repaired_failure_handoff_sha256=sha256_file(self.handoff_path))
        prereg = self.preflight(); checkpoints = prereg["checkpoints"]
        candidates = [(Path(checkpoints[0]["path"]), "B500_left_corridor", "positive_lateral_positive_yaw")]
        for checkpoint, label, corridor in candidates:
            behavior = self.evaluate(checkpoint, label, corridor, prereg)
            if behavior["passed"]:
                action = self.action_gate(checkpoint, label, corridor, prereg)
                if action["passed"]: self.finalize(checkpoint, label, corridor, behavior, action, prereg); return
        r51 = Path(checkpoints[1]["path"]); corridor = "negative_lateral_negative_yaw"
        behavior = self.evaluate(r51, "R5_1_right_corridor", corridor, prereg)
        if behavior["passed"]:
            action = self.action_gate(r51, "R5_1_right_corridor", corridor, prereg)
            if action["passed"]: self.finalize(r51, "R5_1_right_corridor", corridor, behavior, action, prereg); return
        source = r51
        for total, additional in ((3, 2), (5, 2)):
            self.update(status="running", phase=f"R5_{total}_training_authorized_after_both_existing_failed")
            candidate = self.train_r5(source, f"R5_{total}", additional, total)
            behavior = self.evaluate(candidate, f"R5_{total}_right_corridor", corridor, prereg)
            if behavior["passed"]:
                action = self.action_gate(candidate, f"R5_{total}_right_corridor", corridor, prereg)
                conclusion = "directional_gate_passed" if action["passed"] else "teacher_action_regression_failed"
                upload = self.finalize_r5_wandb(f"R5_{total}", candidate, behavior, conclusion)
                if upload["sync_status"] != "synced":
                    raise RuntimeError(f"W&B sync pending before promotion/next stage: {upload.get('sync_error')}")
                if action["passed"]: self.finalize(candidate, f"R5_{total}_right_corridor", corridor, behavior, action, prereg); return
            else:
                upload = self.finalize_r5_wandb(f"R5_{total}", candidate, behavior, "directional_behavior_gate_failed")
                if upload["sync_status"] != "synced":
                    raise RuntimeError(f"W&B sync pending before next stage: {upload.get('sync_error')}")
            source = candidate
        self.terminal("stopped_by_gate", "both existing checkpoints and conditional R5_3/R5_5 failed fixed directional gate")


def main() -> int:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (STATE_ROOT / "supervisor.lock").open("a+")
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: return 2
    supervisor = Supervisor()
    def stop(signum: int, _frame: Any) -> None:
        supervisor.stop_child(); supervisor.terminal("interrupted", f"signal_{signum}"); raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try: supervisor.run(); return 0
    except KeyboardInterrupt: return 130
    except BaseException as error:
        supervisor.stop_child(); supervisor.terminal("failed_closed", f"{type(error).__name__}: {error}"); raise


if __name__ == "__main__": raise SystemExit(main())
