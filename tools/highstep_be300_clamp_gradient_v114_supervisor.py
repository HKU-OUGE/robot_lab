#!/usr/bin/env python3
"""Fail-closed supervisor for the approved v1.14 clamp-gradient repair route.

This file deliberately contains orchestration only.  The sole training-semantic
change (hard-clamp backward -> straight-through backward for the estimator
action-loss graph) lives in the preregistered task implementation.  This
supervisor never edits training code, the specification, or the
preregistration.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/home/lxq/Softwares/robot_lab")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import highstep_wandb_stage_gate as wandb_gate


PYTHON = Path("/home/lxq/miniconda3/envs/env_isaaclab/bin/python")
WORKFLOW_ID = "highstep_be300_clamp_gradient_repair_v114_20260718"
STATE_ROOT = ROOT / "tmp/highstep_be300_clamp_gradient_repair_v114_20260718"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "d3e69743993008a27bf1d30e5f78f634b8710a9dde1239b102cad4370ff84108"
PREREG = STATE_ROOT / "preregistration_v114_wandb_rebinding1.json"
PREREG_SHA_ENV = "HIGHSTEP_V114_PREREGISTRATION_SHA256"
PREREG_PATH_ENV = "HIGHSTEP_BE300_0707_PREREGISTRATION_PATH"
PREREG_RUNTIME_SHA_ENV = "HIGHSTEP_BE300_0707_PREREGISTRATION_SHA256"
PARENT_LINEAGE = STATE_ROOT / "teacher_parent_lineage_v114.json"
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_front_geometry_v1123_treatment_Teacher/"
    "2026-07-16_05-09-30_v1123_B-E300_20260716_050924/model_173499.pt"
)
TEACHER_SHA = "d97ad3ac886c419f59e31d1b30e69353d70ab294a1f890887358d9bc4efb5431"
TRAIN_TASK = (
    "RobotLab-Isaac-Velocity-HighstepFrontGeometryV114STEStudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
EXPERIMENT_ROOT = ROOT / (
    "logs/rsl_rl/"
    "arclab_arcdog_adjustable_leg_highstep_front_geometry_v114_ste_student_no_prior_Student"
)
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
PLAY = ROOT / "scripts/rsl_rl/base/play.py"
FROZEN_GATE = ROOT / "scripts/rsl_rl/base/highstep_v114_frozen_gate.py"
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
SERVICE = "highstep-be300-clamp-gradient-v114.service"
SUPERVISOR_PATH = Path(__file__).resolve()
INFRA_REPAIR = STATE_ROOT / "manifests/tensorboard_scalar_namespace_repair_rebinding2.json"
INFRA_REPAIR_SHA_ENV = "HIGHSTEP_V114_INFRASTRUCTURE_REPAIR_SHA256"
SAVE_POINTS = (100, 300, 700, 1400, 1800, 2500)
FORMAL_POINTS = (100, 300, 700, 1400, 1800)
BEHAVIOR_SEEDS = tuple(range(1101, 1110))
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"
FAIL_CLOSED_EXIT_CODE = 4
MU_OOB_LOSS_KEY = "Debug/Mu_Out_Of_Bounds_Ratio"
# RSL-RL's OnPolicyRunner writes every algorithm ``loss_dict`` item below
# ``Loss/``.  Keep the algorithm-facing key separate from its persisted
# TensorBoard tag so the logging boundary is explicit and testable.
MU_OOB_TENSORBOARD_TAG = f"Loss/{MU_OOB_LOSS_KEY}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def is_read_only(path: Path) -> bool:
    return not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def first_nested(value: Mapping[str, Any], candidates: Sequence[Sequence[str]]) -> Any:
    for candidate in candidates:
        found = nested(value, *candidate)
        if found is not None:
            return found
    return None


def require_sha(value: Any, label: str) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise RuntimeError(f"{label} must be a lowercase SHA256, got {value!r}")
    return text


def require_bound_file(path_value: Any, sha_value: Any, label: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve(strict=True)
    expected = require_sha(sha_value, f"{label} SHA256")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 changed: expected {expected}, got {actual}")
    return path


class Supervisor:
    def __init__(self) -> None:
        STATE_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_path = STATE_ROOT / "state.json"
        self.heartbeat_path = STATE_ROOT / "heartbeat.json"
        self.handoff_path = STATE_ROOT / "handoff.json"
        self.lock_path = STATE_ROOT / "supervisor.lock"
        self.lock_stream = self.lock_path.open("a+")
        fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.prereg_sha = require_sha(os.environ.get(PREREG_SHA_ENV), PREREG_SHA_ENV)
        self.infrastructure_repair_sha = require_sha(
            os.environ.get(INFRA_REPAIR_SHA_ENV), INFRA_REPAIR_SHA_ENV
        )
        existing: dict[str, Any] = {}
        if self.state_path.is_file():
            try:
                candidate = json.loads(self.state_path.read_text(encoding="utf-8"))
                if candidate.get("workflow_id") == WORKFLOW_ID:
                    existing = candidate
            except (OSError, json.JSONDecodeError):
                pass
        self.state: dict[str, Any] = {
            **existing,
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG.resolve()),
            "preregistration_sha256": self.prereg_sha,
            "infrastructure_repair_manifest": str(INFRA_REPAIR.resolve()),
            "infrastructure_repair_manifest_sha256": self.infrastructure_repair_sha,
            "supervisor_pid": os.getpid(),
            "status": "starting",
            "phase": "preflight",
            "active_pid": None,
            "requires_user_action": False,
        }
        self.active: subprocess.Popen[bytes] | None = None
        self.stop_requested = False
        self.prereg: dict[str, Any] = {}
        self.critical_hashes: dict[str, str] = {}
        self.runtime_code_overrides: dict[str, str] = {}
        self.dataset = Path()
        self.dataset_sha = ""
        self.e4000_report = Path()
        self.e4000_report_sha = ""
        signal.signal(signal.SIGTERM, self._signal)
        signal.signal(signal.SIGINT, self._signal)
        self.update()

    def _signal(self, *_: Any) -> None:
        self.stop_requested = True
        if self.active is not None:
            self.stop_group(self.active)

    def update(self, **values: Any) -> None:
        self.state.update(values)
        self.state["updated_at"] = now()
        self.state["last_supervisor_check_at"] = time.time()
        atomic_json(self.state_path, self.state)
        atomic_json(
            self.heartbeat_path,
            {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "timestamp": self.state["updated_at"],
                "unix_time": time.time(),
                "supervisor_pid": os.getpid(),
                "status": self.state.get("status"),
                "phase": self.state.get("phase"),
                "active_pid": self.state.get("active_pid"),
                "checkpoint": self.state.get("checkpoint"),
                "effective_updates": self.state.get("effective_updates"),
                "infrastructure_repair_manifest": str(INFRA_REPAIR.resolve()),
                "infrastructure_repair_manifest_sha256": self.infrastructure_repair_sha,
            },
        )

    @staticmethod
    def stop_group(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        for sent, grace in (
            (signal.SIGINT, 30),
            (signal.SIGTERM, 30),
            (signal.SIGKILL, 5),
        ):
            try:
                os.killpg(process.pid, sent)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + grace
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(1)
            if process.poll() is not None:
                return

    @staticmethod
    def progress(path: Path) -> tuple[int, int]:
        total = 0
        newest = 0
        if path.exists():
            for item in path.rglob("*"):
                if not item.is_file():
                    continue
                try:
                    info = item.stat()
                except OSError:
                    continue
                total += info.st_size
                newest = max(newest, info.st_mtime_ns)
        return total, newest

    def assert_code(self) -> None:
        changed = []
        for name, expected in self.critical_hashes.items():
            path = Path(name)
            effective_expected = self.runtime_code_overrides.get(name, expected)
            if not path.is_file() or sha256_file(path) != effective_expected:
                changed.append(name)
        if changed:
            raise RuntimeError(f"v1.14 critical code changed while active: {changed}")

    def _validate_infrastructure_repair(self) -> None:
        if sha256_file(INFRA_REPAIR) != self.infrastructure_repair_sha:
            raise RuntimeError("v1.14 infrastructure repair manifest SHA mismatch")
        if not is_read_only(INFRA_REPAIR):
            raise RuntimeError("v1.14 infrastructure repair manifest must be read-only")
        payload = json.loads(INFRA_REPAIR.read_text(encoding="utf-8"))
        supervisor_name = str(SUPERVISOR_PATH)
        original_supervisor_sha = self.critical_hashes.get(supervisor_name)
        repaired_supervisor_sha = sha256_file(SUPERVISOR_PATH)
        code_rebinding = payload.get("code_rebinding", {})
        fault = payload.get("fault", {})
        preservation = payload.get("preservation", {})
        if not (
            payload.get("kind")
            == "highstep_v114_tensorboard_scalar_namespace_repair_rebinding2"
            and payload.get("workflow_id") == WORKFLOW_ID
            and payload.get("authority_version") == "v1.14"
            and Path(str(payload.get("spec_path", ""))).resolve() == SPEC.resolve()
            and payload.get("spec_sha256") == SPEC_SHA
            and Path(str(payload.get("preregistration_path", ""))).resolve()
            == PREREG.resolve()
            and payload.get("preregistration_sha256") == self.prereg_sha
            and fault.get("producer_loss_key") == MU_OOB_LOSS_KEY
            and fault.get("persisted_tensorboard_tag") == MU_OOB_TENSORBOARD_TAG
            and fault.get("observed_scalar_count") == 100
            and fault.get("observed_last_step") == 99
            and preservation.get("effective_updates") == 100
            and preservation.get("training_semantics_changed") is False
            and preservation.get("optimizer_reset") is False
            and preservation.get("schedule_reset") is False
            and preservation.get("checkpoint_rewritten") is False
            and preservation.get("tensorboard_event_rewritten") is False
            and code_rebinding.get("changed_files") == [supervisor_name]
            and code_rebinding.get("original_supervisor_sha256")
            == original_supervisor_sha
            and code_rebinding.get("repaired_supervisor_sha256")
            == repaired_supervisor_sha
        ):
            raise RuntimeError("v1.14 infrastructure repair contract mismatch")

        artifacts = payload.get("artifact_bindings")
        if not isinstance(artifacts, list) or {
            item.get("label") for item in artifacts if isinstance(item, Mapping)
        } != {
            "E100 training result",
            "E100 checkpoint",
            "E100 TensorBoard event",
            "supervisor regression test",
        }:
            raise RuntimeError("v1.14 infrastructure repair artifact set mismatch")
        for item in artifacts:
            if not isinstance(item, Mapping):
                raise RuntimeError("v1.14 infrastructure repair artifact is malformed")
            require_bound_file(
                item.get("path"), item.get("sha256"), str(item.get("label"))
            )

        if original_supervisor_sha == repaired_supervisor_sha:
            raise RuntimeError("v1.14 infrastructure repair does not change the supervisor")
        self.runtime_code_overrides = {supervisor_name: repaired_supervisor_sha}

    def wait(self, process: subprocess.Popen[bytes], root: Path, phase: str, stall: int) -> None:
        self.active = process
        last_progress = self.progress(root)
        last_progress_at = time.monotonic()
        last_code_check = 0.0
        while process.poll() is None:
            if self.stop_requested:
                self.stop_group(process)
                raise RuntimeError("supervisor received stop request")
            moment = time.monotonic()
            if moment - last_code_check >= 60:
                self.assert_code()
                last_code_check = moment
            current = self.progress(root)
            if current != last_progress:
                last_progress = current
                last_progress_at = moment
            elif moment - last_progress_at > stall:
                self.stop_group(process)
                raise RuntimeError(f"{phase} made no observable progress for {stall}s")
            self.update(
                status="running",
                phase=phase,
                active_pid=process.pid,
                progress_bytes=current[0],
            )
            time.sleep(10)
        self.active = None
        self.update(active_pid=None)
        if process.returncode != 0:
            raise RuntimeError(f"{phase} exited with return code {process.returncode}")

    def run_command(
        self,
        command: Sequence[str],
        log: Path,
        root: Path,
        phase: str,
        env: Mapping[str, str] | None = None,
        *,
        stall: int = 1200,
    ) -> Path:
        if log.exists():
            attempt = 2
            while log.with_name(f"{log.stem}_infra_attempt{attempt}{log.suffix}").exists():
                attempt += 1
            log = log.with_name(f"{log.stem}_infra_attempt{attempt}{log.suffix}")
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("XAUTHORITY", None)
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "LD_PRELOAD": "/lib/x86_64-linux-gnu/libstdc++.so.6",
                "TERM": "xterm",
            }
        )
        if env:
            environment.update(env)
        with log.open("wb") as stream:
            process = subprocess.Popen(
                list(command),
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.wait(process, root, phase, stall)
        return log

    @staticmethod
    def active_jobs() -> list[dict[str, Any]]:
        jobs = []
        watched = {"train.py", "play.py"}
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                argv = [
                    part.decode(errors="replace")
                    for part in (proc / "cmdline").read_bytes().split(b"\0")
                    if part
                ]
            except OSError:
                continue
            if any(Path(token).name in watched for token in argv):
                jobs.append({"pid": int(proc.name), "argv": " ".join(argv)})
        return jobs

    def require_idle(self) -> None:
        jobs = self.active_jobs()
        if jobs:
            raise RuntimeError(f"another train/eval/play process is active: {jobs}")

    def _prereg_binding(
        self,
        label: str,
        path_candidates: Sequence[Sequence[str]],
        sha_candidates: Sequence[Sequence[str]],
    ) -> Path:
        return require_bound_file(
            first_nested(self.prereg, path_candidates),
            first_nested(self.prereg, sha_candidates),
            label,
        )

    def preflight(self) -> None:
        if sha256_file(SPEC) != SPEC_SHA:
            raise RuntimeError("v1.14 spec SHA mismatch")
        if sha256_file(PREREG) != self.prereg_sha:
            raise RuntimeError("v1.14 preregistration SHA mismatch")
        if not is_read_only(PREREG):
            raise RuntimeError("v1.14 preregistration must be read-only")
        self.prereg = json.loads(PREREG.read_text(encoding="utf-8"))
        authority = self.prereg.get("authority", {})
        if not (
            self.prereg.get("kind") == "highstep_be300_clamp_gradient_repair_preregistration"
            and self.prereg.get("workflow_id") == WORKFLOW_ID
            and authority.get("version") == "v1.14"
            and Path(str(authority.get("spec_path", ""))).resolve() == SPEC.resolve()
            and authority.get("spec_sha256") == SPEC_SHA
        ):
            raise RuntimeError("v1.14 preregistration authority mismatch")
        if sha256_file(TEACHER) != TEACHER_SHA:
            raise RuntimeError("protected B-E300 Teacher SHA mismatch")

        parent_sha = first_nested(
            self.prereg,
            (
                ("teacher", "parent_lineage_sha256"),
                ("teacher_parent_lineage", "sha256"),
                ("parent_teacher_manifest_sha256",),
            ),
        )
        require_bound_file(PARENT_LINEAGE, parent_sha, "v1.14 Teacher parent lineage")
        lineage = json.loads(PARENT_LINEAGE.read_text(encoding="utf-8"))
        if not (
            lineage.get("workflow_id") == WORKFLOW_ID
            and lineage.get("selected_checkpoint_sha256") == TEACHER_SHA
            and nested(lineage, "authority", "spec_sha256") == SPEC_SHA
        ):
            raise RuntimeError("v1.14 Teacher parent lineage content mismatch")

        self.dataset = self._prereg_binding(
            "v1.14 frozen comparison dataset",
            (
                ("frozen_comparison", "dataset_path"),
                ("frozen_gate", "dataset_path"),
                ("e4000_comparison", "dataset_path"),
            ),
            (
                ("frozen_comparison", "dataset_sha256"),
                ("frozen_gate", "dataset_sha256"),
                ("e4000_comparison", "dataset_sha256"),
            ),
        )
        self.dataset_sha = sha256_file(self.dataset)
        self.e4000_report = self._prereg_binding(
            "v1.14 E4000 frozen baseline report",
            (
                ("frozen_comparison", "baseline_report_path"),
                ("frozen_gate", "baseline_report_path"),
                ("e4000_comparison", "report_path"),
            ),
            (
                ("frozen_comparison", "baseline_report_sha256"),
                ("frozen_gate", "baseline_report_sha256"),
                ("e4000_comparison", "report_sha256"),
            ),
        )
        self.e4000_report_sha = sha256_file(self.e4000_report)
        baseline = json.loads(self.e4000_report.read_text(encoding="utf-8"))
        if not (
            baseline.get("dataset_sha256") == self.dataset_sha
            and baseline.get("kind") == "highstep_v114_frozen_comparison"
            and set(baseline.get("statistics", {}))
            == {"first_rear", "second_rear", "rear_hold"}
        ):
            raise RuntimeError("E4000 frozen baseline report contract mismatch")

        smoke = self._prereg_binding(
            "v1.14 throwaway smoke manifest",
            (("smoke", "manifest_path"), ("smoke_manifest", "path")),
            (("smoke", "manifest_sha256"), ("smoke_manifest", "sha256")),
        )
        smoke_payload = json.loads(smoke.read_text(encoding="utf-8"))
        self._validate_smoke(smoke_payload)

        code_sha = self.prereg.get("code_sha256")
        if not isinstance(code_sha, Mapping) or not code_sha:
            raise RuntimeError("v1.14 preregistration has no critical code SHA map")
        self.critical_hashes = {str(Path(name).resolve()): require_sha(value, name) for name, value in code_sha.items()}
        self._validate_infrastructure_repair()
        self.assert_code()

        gates = self.prereg.get("gates", {})
        if not (
            nested(gates, "e300", "max_mu_oob_absolute_increase") == 0.05
            and nested(gates, "e300", "max_mu_oob_ratio") == 1.5
            and nested(gates, "e700", "explicit_improvement_fraction") == 0.05
            and nested(gates, "e700", "max_other_metric_degradation_fraction") == 0.05
            and nested(gates, "e1400", "target_improvement_fraction") == 0.20
            and nested(gates, "e1800", "conditional_metric_improvement_fraction") == 0.05
        ):
            raise RuntimeError("v1.14 preregistered numeric gate thresholds mismatch")

        dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
        if not (
            dashboard.get("workflow_id") == WORKFLOW_ID
            and Path(str(dashboard.get("state_path", ""))).resolve() == self.state_path.resolve()
            and dashboard.get("spec_sha256") == SPEC_SHA
            and Path(str(dashboard.get("preregistration_path", ""))).resolve() == PREREG.resolve()
            and dashboard.get("preregistration_sha256") == self.prereg_sha
            and Path(str(dashboard.get("infrastructure_repair_manifest", ""))).resolve()
            == INFRA_REPAIR.resolve()
            and dashboard.get("infrastructure_repair_manifest_sha256")
            == self.infrastructure_repair_sha
        ):
            raise RuntimeError("dashboard has not been atomically rebound to v1.14")
        self.require_idle()
        self.update(
            status="preflight_passed",
            phase="ready_for_E100",
            teacher_checkpoint=str(TEACHER),
            teacher_checkpoint_sha256=TEACHER_SHA,
            frozen_dataset=str(self.dataset),
            frozen_dataset_sha256=self.dataset_sha,
            e4000_baseline_report=str(self.e4000_report),
            e4000_baseline_report_sha256=self.e4000_report_sha,
            code_sha256=self.critical_hashes,
        )

    @staticmethod
    def _truth(payload: Mapping[str, Any], names: Sequence[str]) -> bool:
        return any(payload.get(name) is True for name in names)

    def _validate_smoke(self, payload: Mapping[str, Any]) -> None:
        if not self._truth(payload, ("passed", "all_checks_passed")):
            raise RuntimeError("v1.14 throwaway smoke did not pass")
        requirements = (
            ("STE/hard-clamp forward parity", ("ste_matches_hard_clamp_forward", "ste_forward_equals_hard_clamp")),
            ("out-of-bounds action gradient", ("oob_action_gradient_nonzero", "oob_mu_action_gradient_nonzero")),
            ("estimator-only update", ("only_estimator_updated", "warmup_only_estimator_updated")),
            ("fresh Student", ("student_fresh", "student_not_inherited")),
            ("fresh optimizer", ("optimizer_fresh", "optimizer_not_inherited")),
            ("independent Teacher storage", ("teacher_student_storage_independent", "student_teacher_storage_independent")),
        )
        missing = [label for label, names in requirements if not self._truth(payload, names)]
        if missing:
            raise RuntimeError(f"v1.14 throwaway smoke missing proofs: {missing}")
        if payload.get("teacher_sha256") != TEACHER_SHA and payload.get("source_teacher_sha256") != TEACHER_SHA:
            raise RuntimeError("v1.14 smoke Teacher SHA mismatch")
        count = payload.get("effective_updates", payload.get("end_effective_updates"))
        if not isinstance(count, int) or not 1 <= count <= 5:
            raise RuntimeError("v1.14 smoke is not a 1-5 update throwaway run")

    @staticmethod
    def checkpoint_payload(path: Path) -> dict[str, Any]:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, dict):
            raise RuntimeError(f"checkpoint is not a mapping: {path}")
        return value

    @classmethod
    def checkpoint_count(cls, path: Path) -> int:
        payload = cls.checkpoint_payload(path)
        return int(payload["infos"][ALGORITHM_KEY]["student_distill_update_count"])

    def checkpoint_scope(self, path: Path, expected: int) -> dict[str, Any]:
        payload = self.checkpoint_payload(path)
        algorithm = payload.get("infos", {}).get(ALGORITHM_KEY)
        if not isinstance(algorithm, Mapping):
            raise RuntimeError("checkpoint has no versioned algorithm state")
        recovery = algorithm.get("student_recovery")
        if not isinstance(recovery, Mapping):
            raise RuntimeError("checkpoint has no versioned Student recovery state")
        binding = recovery.get("binding_manifest")
        if not isinstance(binding, Mapping):
            raise RuntimeError("checkpoint has no Student/Teacher binding manifest")
        count = int(algorithm.get("student_distill_update_count", -1))
        if not (
            count == expected
            and int(recovery.get("effective_update_count", -1)) == expected
            and recovery.get("preregistration_sha256") == self.prereg_sha
            and binding.get("preregistration_sha256") == self.prereg_sha
            and binding.get("teacher_sha256") == TEACHER_SHA
            and binding.get("initial_student_sha256") == TEACHER_SHA
            and binding.get("student_actor_latent_clamp_backward") == "straight_through"
            and binding.get("student_ppo_permanently_disabled") is True
            and binding.get("actor_body_frozen") is True
            and binding.get("critic_frozen") is True
            and binding.get("student_privileged_encoder_frozen") is True
            and binding.get("warmup_updates") == 1400
            and binding.get("warmup_main_action_target") == "teacher_pre_prior"
            and binding.get("warmup_prior_box_loss_coefficient") == 0.0
            and binding.get("phase_scale") == 2.0
            and binding.get("rear_box_scale") == 1.5
        ):
            raise RuntimeError(f"v1.14 checkpoint binding mismatch at E{expected}")
        if not isinstance(algorithm.get("vae_optimizer_state_dict"), Mapping):
            raise RuntimeError("v1.14 checkpoint has no full VAE optimizer state")
        self._assert_finite_tree(payload.get("model_state_dict"), "model_state_dict")
        self._assert_finite_tree(algorithm.get("vae_optimizer_state_dict"), "vae_optimizer_state_dict")
        return {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": sha256_file(path),
            "effective_updates": expected,
            "teacher_sha256": binding.get("teacher_sha256"),
            "initial_student_sha256": binding.get("initial_student_sha256"),
            "preregistration_sha256": recovery.get("preregistration_sha256"),
            "student_actor_latent_clamp_backward": binding.get("student_actor_latent_clamp_backward"),
            "optimizer_parameter_names": recovery.get("optimizer_parameter_names"),
            "optimizer_present": True,
            "all_checkpoint_tensors_finite": True,
        }

    @classmethod
    def _assert_finite_tree(cls, value: Any, label: str) -> None:
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite tensor in {label}")
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                cls._assert_finite_tree(item, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                cls._assert_finite_tree(item, f"{label}[{index}]")

    def find_checkpoint(self, run_dir: Path, expected: int) -> Path | None:
        matches = []
        for path in run_dir.glob("model_*.pt"):
            try:
                if self.checkpoint_count(path) == expected:
                    matches.append(path)
            except Exception:
                continue
        return max(matches, key=lambda item: item.stat().st_mtime_ns) if matches else None

    def find_completed_run(self, run_name: str, expected: int) -> tuple[Path, Path] | None:
        completed = []
        for run_dir in EXPERIMENT_ROOT.glob(f"*_{run_name}"):
            checkpoint = self.find_checkpoint(run_dir, expected)
            if checkpoint is not None:
                completed.append((run_dir, checkpoint))
        if len(completed) > 1:
            raise RuntimeError(f"multiple completed E{expected} runs share {run_name}: {completed}")
        return completed[0] if completed else None

    def wandb_files(self, stage: int, source: Path, additional: int) -> tuple[Path, Path, Path]:
        repaired_e100 = stage == 100 and bool(self.prereg.get("infrastructure_rebinding"))
        suffix = "_wandb_rebinding1" if repaired_e100 else ""
        attempt = "attempt2" if repaired_e100 else "attempt1"
        config_path = STATE_ROOT / "wandb_stage_configs" / f"E{stage}{suffix}.json"
        native_config_path = (
            STATE_ROOT / "wandb_stage_configs" / f"E{stage}{suffix}.wandb.yaml"
        )
        run_name = f"{WORKFLOW_ID}_R_E{stage}_{attempt}"
        expected_history_start = (
            0 if stage == 100 else int(self.checkpoint_payload(source).get("iter", -1))
        )
        if expected_history_start < 0:
            raise RuntimeError(f"cannot determine the E{stage} W&B history start step")
        trainable = ["estimator.*"] if stage <= 1400 else [
            "estimator.*",
            "actor.final.weight[12:16]",
            "actor.final.bias[12:16]",
        ]
        config = {
            "workflow_id": WORKFLOW_ID,
            "route": "R",
            "stage": f"E{stage}",
            "attempt": attempt,
            "run_name": run_name,
            "group": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_sha256": self.prereg_sha,
            "start_checkpoint": str(source.resolve()),
            "start_checkpoint_sha256": sha256_file(source),
            "teacher_checkpoint": str(TEACHER.resolve()),
            "teacher_sha256": TEACHER_SHA,
            "frozen_tensors": [
                "teacher_actor.*",
                "teacher_priv_encoder.*",
                "critic.*",
                "student_priv_encoder.*",
                "actor body and rows 0:12 permanently",
                "actor all rows through E1400",
            ],
            "trainable_tensors": trainable,
            "optimizer": {
                "class": "Adam",
                "estimator_lr": 1.0e-3,
                "box_rows_lr": 1.0e-5,
                "vae_epochs_per_update": 4,
                "ppo": "permanently_disabled",
                "fresh_at_E100": True,
                "preserve_after_E100": True,
            },
            "budget": {
                "additional_updates": additional,
                "effective_update_target": stage,
                "absolute_cap": 2500,
            },
            "training_task": TRAIN_TASK,
            "wandb_mode": "online",
            "expected_history_step_start": expected_history_start,
            "expected_history_step_end": expected_history_start + additional - 1,
            "expected_history_unique_steps": additional,
            "historical_sync": False,
            "single_training_semantic_change": (
                "student_actor_latent_clamp_backward_hard_to_straight_through"
            ),
        }
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config or not is_read_only(config_path):
                raise RuntimeError(f"stored W&B config changed for E{stage}")
        else:
            atomic_json(config_path, config, read_only=True)
        # W&B config_paths consumes its native config-defaults schema, while
        # the stage gate intentionally retains the plain JSON contract above.
        # Keep both immutable and derived from the same in-memory mapping.
        import yaml

        native_config: dict[str, Any] = {"wandb_version": 1}
        native_config.update({key: {"value": value} for key, value in config.items()})
        unwrapped = {
            key: value["value"]
            for key, value in native_config.items()
            if key != "wandb_version"
        }
        if unwrapped != config:
            raise RuntimeError(f"native W&B config round-trip failed for E{stage}")
        if native_config_path.exists():
            if (
                yaml.safe_load(native_config_path.read_text(encoding="utf-8")) != native_config
                or not is_read_only(native_config_path)
            ):
                raise RuntimeError(f"stored native W&B config changed for E{stage}")
        else:
            native_config_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = native_config_path.with_name(
                f".{native_config_path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                yaml.safe_dump(native_config, sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
            os.replace(temporary, native_config_path)
            native_config_path.chmod(0o444)
        manifest = STATE_ROOT / "wandb_stages" / run_name / "stage_manifest.json"
        if not manifest.exists():
            manifest = wandb_gate.prepare_stage(config_path, STATE_ROOT)
        return config_path, native_config_path, manifest

    def _launch_record(self, stage: int, source: Path, previous: int, manifest: Path) -> dict[str, Any]:
        stage_root = STATE_ROOT / "stages" / f"E{stage}"
        wandb_payload = json.loads(manifest.read_text(encoding="utf-8"))
        run_name = str(wandb_payload["run_name"])
        launch_path = stage_root / (
            "launch_wandb_rebinding1.json"
            if run_name.endswith("_attempt2")
            else "launch.json"
        )
        if launch_path.is_file():
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            if not is_read_only(launch_path):
                raise RuntimeError(f"E{stage} launch record is writable")
            return launch
        additional = stage - previous
        command = [
            str(PYTHON),
            "-u",
            str(TRAIN.relative_to(ROOT)),
            "--task",
            TRAIN_TASK,
            "--num_envs",
            "4096",
            "--max_iterations",
            str(additional),
            "--seed",
            "42",
            "--headless",
            "--logger",
            "wandb",
            "--log_project_name",
            "isaaclab",
            "--resume",
            "--checkpoint",
            str(source.resolve()),
            "--highstep_resume_mode",
            "refine",
            "--highstep_checkpoint_load_mode",
            "weights_only" if previous == 0 else "full",
            "--highstep_schedule_resume_mode",
            "preserve",
            "--highstep_parent_teacher_manifest",
            str(PARENT_LINEAGE.resolve()),
            "--run_name",
            run_name,
        ]
        launch = {
            "schema_version": 1,
            "stage": stage,
            "previous_effective_updates": previous,
            "additional_updates": additional,
            "source_checkpoint": str(source.resolve()),
            "source_checkpoint_sha256": sha256_file(source),
            "load_mode": "weights_only" if previous == 0 else "full",
            "schedule_resume_mode": "preserve",
            "run_name": run_name,
            "command": command,
            "wandb_stage_manifest": str(manifest.resolve()),
            "created_at": now(),
        }
        atomic_json(launch_path, launch, read_only=True)
        return launch

    def train_stage(self, stage: int, source: Path, previous: int) -> tuple[Path, Path, Path]:
        stage_root = STATE_ROOT / "stages" / f"E{stage}"
        result_path = stage_root / "training_result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            checkpoint = Path(result["checkpoint"]).resolve(strict=True)
            run_dir = Path(result["run_dir"]).resolve(strict=True)
            if result.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise RuntimeError(f"stored E{stage} checkpoint SHA changed")
            self.checkpoint_scope(checkpoint, stage)
            return run_dir, checkpoint, Path(result["wandb_stage_manifest"]).resolve(strict=True)

        additional = stage - previous
        config_path, native_config_path, wandb_manifest = self.wandb_files(
            stage, source, additional
        )
        wandb_payload = json.loads(wandb_manifest.read_text(encoding="utf-8"))
        self.update(
            wandb_run_id=wandb_payload["run_id"],
            wandb_run_url=wandb_payload["run_url"],
            wandb_mode="online",
            wandb_sync_status="online_recording",
        )
        launch = self._launch_record(stage, source, previous, wandb_manifest)
        if launch.get("source_checkpoint_sha256") != sha256_file(source):
            raise RuntimeError(f"E{stage} launch source checkpoint changed")
        completed = self.find_completed_run(str(launch["run_name"]), stage)
        if completed is None:
            self.require_idle()
            environment = wandb_gate.process_environment(wandb_manifest)
            environment.update(
                {
                    PREREG_PATH_ENV: str(PREREG.resolve()),
                    PREREG_RUNTIME_SHA_ENV: self.prereg_sha,
                    "WANDB_MODE": "online",
                    "WANDB_X_DISABLE_STATS": "true",
                    "WANDB_DISABLE_GIT": "true",
                    "ROBOT_LAB_WANDB_CONFIG_PATH": str(native_config_path.resolve()),
                    "ROBOT_LAB_WANDB_CONFIG_SHA256": sha256_file(native_config_path),
                }
            )
            log = self.run_command(
                list(launch["command"]),
                stage_root / "train.log",
                stage_root,
                f"training_E{stage}",
                environment,
                stall=1800,
            )
            completed = self.find_completed_run(str(launch["run_name"]), stage)
            if completed is None:
                raise RuntimeError(f"training E{stage} ended without a full target checkpoint")
        else:
            log = next(iter(sorted(stage_root.glob("train*.log"))), None)
        run_dir, checkpoint = completed
        scope = self.checkpoint_scope(checkpoint, stage)
        scope_path = stage_root / "checkpoint_scope.json"
        if scope_path.exists():
            if json.loads(scope_path.read_text(encoding="utf-8")) != scope:
                raise RuntimeError(f"stored E{stage} checkpoint scope changed")
        else:
            atomic_json(scope_path, scope, read_only=True)
        schedule = run_dir / "params/highstep_schedule_manifest.json"
        runtime = run_dir / "params/highstep_runtime_state.json"
        if not schedule.is_file() or not runtime.is_file():
            raise RuntimeError(f"E{stage} run lacks schedule/runtime state")
        result = {
            "schema_version": 1,
            "kind": "highstep_v114_training_result",
            "stage": stage,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": stage,
            "run_dir": str(run_dir.resolve()),
            "train_log": None if log is None else str(log.resolve()),
            "train_log_sha256": None if log is None else sha256_file(log),
            "checkpoint_scope": str(scope_path.resolve()),
            "checkpoint_scope_sha256": sha256_file(scope_path),
            "schedule_manifest": str(schedule.resolve()),
            "schedule_manifest_sha256": sha256_file(schedule),
            "runtime_state": str(runtime.resolve()),
            "runtime_state_sha256": sha256_file(runtime),
            "wandb_stage_manifest": str(wandb_manifest.resolve()),
            "completed_at": now(),
        }
        atomic_json(result_path, result, read_only=True)
        self.update(
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=sha256_file(checkpoint),
            effective_updates=stage,
            run_dir=str(run_dir.resolve()),
        )
        return run_dir, checkpoint, wandb_manifest

    @staticmethod
    def tensorboard_scalar(run_dir: Path, tag: str) -> float:
        from tensorboard.backend.event_processing import event_accumulator

        values = []
        for event in run_dir.rglob("events.out.tfevents.*"):
            accumulator = event_accumulator.EventAccumulator(str(event), size_guidance={"scalars": 0})
            accumulator.Reload()
            if tag in accumulator.Tags().get("scalars", []):
                values.extend(accumulator.Scalars(tag))
        if not values:
            raise RuntimeError(f"TensorBoard scalar {tag!r} is missing from {run_dir}")
        value = float(max(values, key=lambda item: (item.step, item.wall_time)).value)
        if not math.isfinite(value):
            raise RuntimeError(f"TensorBoard scalar {tag!r} is non-finite")
        return value

    def mechanism_check(
        self,
        stage: int,
        run_dir: Path,
        checkpoint: Path,
        e100_oob: float | None,
    ) -> dict[str, Any]:
        scope = self.checkpoint_scope(checkpoint, stage)
        oob = self.tensorboard_scalar(run_dir, MU_OOB_TENSORBOARD_TAG)
        result: dict[str, Any] = {
            "schema_version": 1,
            "kind": "highstep_v114_mechanism_check",
            "stage": stage,
            "checkpoint_sha256": sha256_file(checkpoint),
            "all_checkpoint_tensors_finite": scope["all_checkpoint_tensors_finite"],
            "student_actor_latent_clamp_backward": scope["student_actor_latent_clamp_backward"],
            "mu_oob_fraction": oob,
            "e100_mu_oob_fraction": e100_oob,
            "ste_gradient_proof_bound_to_unchanged_smoke_and_code": True,
            "passed": True,
            "stop_reasons": [],
        }
        if stage == 300:
            if e100_oob is None:
                raise RuntimeError("E300 mechanism gate requires the frozen E100 mu-OOB baseline")
            absolute_increase = oob - e100_oob
            # Keep the manifest strict-JSON even for the zero-baseline edge case.
            ratio = 1.0e300 if e100_oob == 0.0 and oob > 0.0 else (oob / e100_oob if e100_oob else 1.0)
            obvious_worsening = absolute_increase > 0.05 and ratio > 1.5
            result.update(
                {
                    "mu_oob_absolute_increase": absolute_increase,
                    "mu_oob_ratio_to_E100": ratio,
                    "obvious_mu_oob_worsening": obvious_worsening,
                    "obvious_worsening_requires_both_thresholds": True,
                }
            )
            if obvious_worsening:
                result["passed"] = False
                result["stop_reasons"].append("mu_oob_obviously_worsened")
        return result

    def frozen_gate_report(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        stage_root = STATE_ROOT / "stages" / f"E{stage}" / "frozen_gate"
        report = stage_root / "report.json"
        if report.is_file():
            result = json.loads(report.read_text(encoding="utf-8"))
        else:
            self.require_idle()
            command = [
                str(PYTHON),
                "-u",
                str(TRAIN.relative_to(ROOT)),
                "--task",
                TRAIN_TASK,
                "--num_envs",
                "4096",
                "--max_iterations",
                "1",
                "--seed",
                "42",
                "--headless",
                "--logger",
                "tensorboard",
                "--resume",
                "--checkpoint",
                str(checkpoint.resolve()),
                "--highstep_resume_mode",
                "refine",
                "--highstep_checkpoint_load_mode",
                "full",
                "--highstep_schedule_resume_mode",
                "preserve",
                "--highstep_parent_teacher_manifest",
                str(PARENT_LINEAGE.resolve()),
                "--v114_frozen_gate_dataset",
                str(self.dataset.resolve()),
                "--v114_frozen_gate_report",
                str(report.resolve()),
                "--v114_expected_frozen_gate_dataset_sha256",
                self.dataset_sha,
                "--run_name",
                f"v114_frozen_gate_E{stage}_{time.strftime('%Y%m%d_%H%M%S')}",
            ]
            environment = {
                PREREG_PATH_ENV: str(PREREG.resolve()),
                PREREG_RUNTIME_SHA_ENV: self.prereg_sha,
            }
            self.run_command(
                command,
                stage_root / "frozen_gate.log",
                stage_root,
                f"frozen_gate_E{stage}",
                environment,
                stall=1200,
            )
            result = json.loads(report.read_text(encoding="utf-8"))
        if not (
            result.get("kind") == "highstep_v114_frozen_comparison"
            and result.get("checkpoint_sha256") == sha256_file(checkpoint)
            and result.get("dataset_sha256") == self.dataset_sha
            and result.get("dataset_created_in_this_run") is False
            and result.get("protected_unchanged") is True
            and result.get("optimizer_step_calls") == 0
            and result.get("runner_learn_calls") == 0
            and set(result.get("statistics", {}))
            == {"first_rear", "second_rear", "rear_hold"}
        ):
            raise RuntimeError(f"invalid v1.14 frozen report at E{stage}")
        if sha256_file(self.dataset) != self.dataset_sha:
            raise RuntimeError("v1.14 frozen comparison dataset changed during gate")
        return result

    @staticmethod
    def metric_mean(report: Mapping[str, Any], metric: str) -> float:
        values = []
        for phase in ("first_rear", "second_rear", "rear_hold"):
            value = nested(report, "statistics", phase, metric)
            if not finite_number(value):
                raise RuntimeError(f"frozen report lacks finite {phase}.{metric}")
            values.append(float(value))
        return sum(values) / len(values)

    @classmethod
    def comparison_metrics(cls, report: Mapping[str, Any], *, require_box: bool) -> dict[str, float]:
        result = {
            "clamped_latent_mae": cls.metric_mean(report, "clamped_latent_mae"),
            "nonbox_pre_prior_action_mae": cls.metric_mean(
                report, "nonbox_pre_prior_action_mae"
            ),
        }
        box_names = (
            "post_prior_box_action_mae",
            "box_post_prior_action_mae",
            "post_prior_box_mae",
        )
        for name in box_names:
            try:
                result["post_prior_box_action_mae"] = cls.metric_mean(report, name)
                break
            except RuntimeError:
                continue
        if require_box and "post_prior_box_action_mae" not in result:
            raise RuntimeError("frozen report lacks preregistered post-prior box action MAE")
        return result

    @staticmethod
    def relative_improvement(baseline: float, candidate: float) -> float:
        if baseline <= 0.0:
            return 0.0 if candidate == baseline else (-1.0e300 if candidate > baseline else 1.0e300)
        return (baseline - candidate) / baseline

    def frozen_decision(
        self,
        stage: int,
        report: Mapping[str, Any],
        e1400_report: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        baseline = self.comparison_metrics(
            json.loads(self.e4000_report.read_text(encoding="utf-8")),
            require_box=False,
        )
        candidate = self.comparison_metrics(report, require_box=stage >= 1400)
        improvements = {
            name: self.relative_improvement(baseline[name], candidate[name])
            for name in ("clamped_latent_mae", "nonbox_pre_prior_action_mae")
        }
        decision: dict[str, Any] = {
            "stage": stage,
            "e4000_baseline": baseline,
            "candidate": candidate,
            "improvement_vs_E4000": improvements,
            "passed": True,
            "reason": "record_only",
        }
        if stage == 700:
            latent = improvements["clamped_latent_mae"]
            action = improvements["nonbox_pre_prior_action_mae"]
            passed = (latent >= 0.05 and action >= -0.05) or (action >= 0.05 and latent >= -0.05)
            decision.update(
                passed=passed,
                reason="E700_explicit_improvement" if passed else "E700_no_explicit_improvement",
            )
        elif stage == 1400:
            passed = all(value >= 0.20 for value in improvements.values())
            decision.update(
                passed=passed,
                reason="E1400_mechanism_gate_passed" if passed else "E1400_mechanism_gate_failed",
            )
        if stage >= 1800:
            if e1400_report is None:
                raise RuntimeError("E1800/E2500 comparison requires E1400 report")
            e1400 = self.comparison_metrics(e1400_report, require_box=True)
            decision["improvement_vs_E1400"] = {
                name: self.relative_improvement(e1400[name], candidate[name])
                for name in (
                    "clamped_latent_mae",
                    "nonbox_pre_prior_action_mae",
                    "post_prior_box_action_mae",
                )
            }
        return decision

    @staticmethod
    def parse_eval(log: Path, checkpoint: Path) -> dict[str, Any]:
        payloads = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("[HIGHSTEP_EVAL_JSON] "):
                payloads.append(json.loads(line[len("[HIGHSTEP_EVAL_JSON] ") :]))
        if len(payloads) != 1:
            raise RuntimeError(f"expected exactly one HIGHSTEP_EVAL_JSON payload: {log}")
        item = payloads[0]
        required_true = (
            "runtime_snapshot_checkpoint_sha256_verified",
            "schedule_valid",
            "schedule_runtime_match",
            "schedule_clock_runtime_match",
            "reset_valid",
            "initial_geometry_valid",
            "target_action_order_valid",
            "target_limit_contract_valid",
            "target_limit_action_input_valid",
            "eval_action_delay_runtime_match",
        )
        if not (
            int(item.get("schema_version", 0)) >= 8
            and item.get("checkpoint_sha256") == sha256_file(checkpoint)
            and all(item.get(name) is True for name in required_true)
            and item.get("target_limit_invalid_action_steps") == 0
            and item.get("eval_action_delay_requested") == 0
            and item.get("eval_action_delay_steps_runtime") == 0
            and item.get("keep_play_randomization") is False
            and item.get("fixed_velocity_command") == [0.45, 0.0, 0.0]
            and item.get("action_prior_enabled") is False
        ):
            raise RuntimeError(f"invalid v1.14 behavior runtime contract: {log}")
        return item

    @staticmethod
    def play_command(checkpoint: Path, seed: int) -> list[str]:
        return [
            str(PYTHON),
            "-u",
            str(PLAY.relative_to(ROOT)),
            "--task",
            TRAIN_TASK,
            "--num_envs",
            "1",
            "--headless",
            "--seed",
            str(seed),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--play_terrain_type",
            "box",
            "--play_terrain_level",
            "9",
            "--fixed_velocity_command",
            "0.45",
            "0.0",
            "0.0",
            "--reset_after_play_terrain_selection",
            "--front_step_eval_reset",
            "--front_step_eval_side",
            "x-",
            "--front_step_eval_edge_gap",
            "0.55",
            "--front_step_eval_lateral_offset",
            "0.0",
            "--front_step_eval_yaw_offset_deg",
            "0.0",
            "--print_rear_width_metrics",
            "--skip_policy_export",
            "--rear_width_metric_interval",
            "60",
            "--play_max_steps",
            "600",
            "--eval_action_delay_steps",
            "0",
        ]

    def behavior9(self, stage: int, checkpoint: Path) -> dict[str, Any]:
        root = STATE_ROOT / "stages" / f"E{stage}" / "behavior9_nominal"
        manifest = root / "behavior9_manifest.json"
        if manifest.is_file():
            result = json.loads(manifest.read_text(encoding="utf-8"))
            if result.get("checkpoint_sha256") != sha256_file(checkpoint):
                raise RuntimeError(f"stored E{stage} behavior checkpoint changed")
            return result
        rows = []
        environment = {
            PREREG_PATH_ENV: str(PREREG.resolve()),
            PREREG_RUNTIME_SHA_ENV: self.prereg_sha,
        }
        for seed in BEHAVIOR_SEEDS:
            valid_record = root / f"seed{seed}.json"
            if valid_record.is_file():
                row = json.loads(valid_record.read_text(encoding="utf-8"))
                if row.get("checkpoint_sha256") != sha256_file(checkpoint):
                    raise RuntimeError(f"stored E{stage} seed {seed} checkpoint changed")
                rows.append(row)
                continue
            attempt = 1
            while True:
                self.require_idle()
                log = root / f"seed{seed}_attempt{attempt}.log"
                try:
                    completed = self.run_command(
                        self.play_command(checkpoint, seed),
                        log,
                        root,
                        f"behavior_E{stage}_seed{seed}",
                        environment,
                        stall=1200,
                    )
                    item = self.parse_eval(completed, checkpoint)
                    row = {
                        "seed": seed,
                        "valid": True,
                        "full_climb": item.get("full_climb_success") is True,
                        "rear_hold": item.get("rear_on_platform_hold_success") is True,
                        "behavior_failure": not (
                            item.get("full_climb_success") is True
                            and item.get("rear_on_platform_hold_success") is True
                        ),
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "log": str(completed.resolve()),
                        "log_sha256": sha256_file(completed),
                        "eval": item,
                    }
                    atomic_json(valid_record, row, read_only=True)
                    rows.append(row)
                    break
                except RuntimeError as error:
                    infra = root / f"seed{seed}_attempt{attempt}_infrastructure_failure.json"
                    atomic_json(
                        infra,
                        {
                            "seed": seed,
                            "attempt": attempt,
                            "failure_class": "evaluation_infrastructure_failure",
                            "error": f"{type(error).__name__}: {error}",
                            "recorded_at": now(),
                        },
                        read_only=True,
                    )
                    if attempt >= 3:
                        raise
                    attempt += 1
        counts = {
            "valid": sum(row["valid"] is True for row in rows),
            "full_climb": sum(row["full_climb"] is True for row in rows),
            "rear_hold": sum(row["rear_hold"] is True for row in rows),
        }
        result = {
            "schema_version": 1,
            "kind": "highstep_v114_nominal_behavior9",
            "stage": stage,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "seeds": list(BEHAVIOR_SEEDS),
            "rows": rows,
            "counts": counts,
            "valid_complete": counts["valid"] == 9,
            "exploratory_behavior_gate_passed": (
                counts["valid"] == 9
                and counts["full_climb"] >= 3
                and counts["rear_hold"] >= 3
            ),
            "completed_at": now(),
        }
        if not result["valid_complete"]:
            raise RuntimeError(f"E{stage} behavior9 is not infrastructure-valid 9/9")
        atomic_json(manifest, result, read_only=True)
        return result

    def finalize_wandb(
        self,
        stage: int,
        checkpoint: Path,
        stage_result: Path,
        wandb_manifest: Path,
        gate_metrics: Mapping[str, Any],
        conclusion: str,
    ) -> None:
        summary = STATE_ROOT / "wandb_stage_summaries" / f"E{stage}.json"
        expected_summary = {
            "output_checkpoint": str(checkpoint.resolve()),
            "output_checkpoint_sha256": sha256_file(checkpoint),
            "effective_updates": stage,
            "gate_metrics": dict(gate_metrics),
            "gate_conclusion": conclusion,
            "manifest_path": str(stage_result.resolve()),
            "manifest_sha256": sha256_file(stage_result),
        }
        if summary.exists():
            if json.loads(summary.read_text(encoding="utf-8")) != expected_summary:
                raise RuntimeError(f"stored W&B summary changed for E{stage}")
        else:
            atomic_json(summary, expected_summary, read_only=True)

        manifest_payload = json.loads(wandb_manifest.read_text(encoding="utf-8"))
        config = json.loads((wandb_manifest.parent / "config.snapshot.json").read_text())
        expected_steps = set(
            range(
                int(config["expected_history_step_start"]),
                int(config["expected_history_step_end"]) + 1,
            )
        )
        while True:
            try:
                import wandb

                remote = wandb.Api(timeout=60).run(
                    f"{manifest_payload['entity']}/{manifest_payload['project']}/"
                    f"{manifest_payload['run_id']}"
                )
                remote_steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                remote_config = wandb_gate.plain_value(dict(remote.config))
                if any(
                    wandb_gate.plain_value(remote_config.get(key))
                    != wandb_gate.plain_value(value)
                    for key, value in config.items()
                ):
                    raise RuntimeError("remote W&B config verification failed")

                run = wandb.init(
                    entity=manifest_payload["entity"],
                    project=manifest_payload["project"],
                    id=manifest_payload["run_id"],
                    name=manifest_payload["run_name"],
                    group=manifest_payload["group"],
                    resume="must",
                    mode="online",
                    settings=wandb.Settings(init_timeout=60),
                )
                if run is None:
                    raise RuntimeError("wandb.init returned no run during online finalization")
                run.config.update(config, allow_val_change=False)
                run.summary.update({**expected_summary, "historical_sync": False})
                run.finish()

                remote = wandb.Api(timeout=60).run(
                    f"{manifest_payload['entity']}/{manifest_payload['project']}/"
                    f"{manifest_payload['run_id']}"
                )
                remote_summary = wandb_gate.plain_value(dict(remote.summary))
                if any(
                    remote_summary.get(key) != wandb_gate.plain_value(value)
                    for key, value in {**expected_summary, "historical_sync": False}.items()
                ):
                    raise RuntimeError("remote W&B summary verification failed")
                remote_steps = wandb_gate.verified_remote_history_steps(remote, expected_steps)
                manifest_payload.pop("sync_error", None)
                manifest_payload.update(
                    {
                        "sync_status": "synced",
                        "transport": "online",
                        "summary_path": str(summary.resolve()),
                        "summary_sha256": sha256_file(summary),
                        "remote_history_verification": {
                            "expected_unique_steps": len(expected_steps),
                            "remote_unique_steps": len(remote_steps),
                            "step_min": min(expected_steps),
                            "step_max": max(expected_steps),
                            "missing_steps": 0,
                        },
                        "verified_at": now(),
                    }
                )
                atomic_json(wandb_manifest, manifest_payload)
                self.update(
                    wandb_run_id=manifest_payload["run_id"],
                    wandb_run_url=manifest_payload["run_url"],
                    wandb_mode="online",
                    wandb_sync_status="synced",
                    last_error=None,
                )
                return
            except Exception as error:
                try:
                    if "wandb" in locals() and wandb.run is not None:
                        wandb.finish(exit_code=1)
                except Exception:
                    pass
                self.update(
                    status="pending_sync",
                    phase=f"wandb_sync_E{stage}",
                    last_error=f"{type(error).__name__}: {error}",
                    wandb_sync_status="pending_remote_verification",
                )
                self._heartbeat_sleep(60)

    def _heartbeat_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_requested:
                raise RuntimeError("supervisor received stop request")
            self.update()
            time.sleep(min(10.0, deadline - time.monotonic()))

    def stage_result(
        self,
        stage: int,
        checkpoint: Path,
        run_dir: Path,
        wandb_manifest: Path,
        *,
        mechanism: Mapping[str, Any] | None,
        frozen_report: Mapping[str, Any] | None,
        frozen_decision: Mapping[str, Any] | None,
        behavior: Mapping[str, Any] | None,
        conclusion: str,
        continue_training: bool,
    ) -> Path:
        path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_v114_stage_result",
            "stage": stage,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "run_dir": str(run_dir.resolve()),
            "mechanism_check": mechanism,
            "frozen_comparison": frozen_report,
            "frozen_decision": frozen_decision,
            "behavior9": behavior,
            "conclusion": conclusion,
            "continue_training": continue_training,
            "wandb_stage_manifest": str(wandb_manifest.resolve()),
            "completed_at": now(),
        }
        if path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            stable = {key: value for key, value in payload.items() if key != "completed_at"}
            old = {key: value for key, value in stored.items() if key != "completed_at"}
            if stable != old or not is_read_only(path):
                raise RuntimeError(f"stored E{stage} stage result changed")
        else:
            atomic_json(path, payload, read_only=True)
        return path

    @staticmethod
    def gate_metrics(
        mechanism: Mapping[str, Any] | None,
        decision: Mapping[str, Any] | None,
        behavior: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "mechanism": None if mechanism is None else dict(mechanism),
            "frozen_decision": None if decision is None else dict(decision),
            "behavior_counts": None if behavior is None else dict(behavior["counts"]),
        }

    def terminal(self, status: str, reason: str, checkpoint: Path | None) -> None:
        handoff = {
            "schema_version": 1,
            "workflow_id": WORKFLOW_ID,
            "spec_sha256": SPEC_SHA,
            "preregistration_path": str(PREREG.resolve()),
            "preregistration_sha256": self.prereg_sha,
            "infrastructure_repair_manifest": str(INFRA_REPAIR.resolve()),
            "infrastructure_repair_manifest_sha256": self.infrastructure_repair_sha,
            "status": status,
            "reason": reason,
            "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
            "checkpoint_sha256": None if checkpoint is None else sha256_file(checkpoint),
            "effective_updates": self.state.get("effective_updates"),
            "automatic_real_robot_deployment": False,
            "next_action": (
                "mujoco_exploratory_smoke"
                if status == "mujoco_exploratory_smoke_ready"
                else "none_under_v1.14"
            ),
            "written_at": now(),
        }
        atomic_json(self.handoff_path, handoff)
        self.update(
            status=status,
            phase=status,
            stop_reason=reason,
            active_pid=None,
            requires_user_action=False,
        )
        self.disable_service("terminal_gate")

    def disable_service(self, reason: str) -> None:
        completed = subprocess.run(
            ["systemctl", "--user", "disable", SERVICE],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        disable_record = STATE_ROOT / "service_disabled_terminal.json"
        atomic_json(
            disable_record,
            {
                "service": SERVICE,
                "reason": reason,
                "return_code": completed.returncode,
                "output": completed.stdout,
                "disabled_at": now(),
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to disable terminal service {SERVICE}: {completed.stdout}")

    def _existing_stage(self, stage: int) -> dict[str, Any] | None:
        path = STATE_ROOT / "stages" / f"E{stage}" / "stage_result.json"
        if not path.is_file():
            return None
        result = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoint"]).resolve(strict=True)
        if result.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError(f"stored E{stage} stage checkpoint changed")
        manifest = Path(result["wandb_stage_manifest"]).resolve(strict=True)
        if json.loads(manifest.read_text(encoding="utf-8")).get("sync_status") != "synced":
            self.finalize_wandb(
                stage,
                checkpoint,
                path,
                manifest,
                self.gate_metrics(
                    result.get("mechanism_check"),
                    result.get("frozen_decision"),
                    result.get("behavior9"),
                ),
                str(result["conclusion"]),
            )
        return result

    def run(self) -> None:
        self.preflight()
        source = TEACHER
        previous = 0
        e100_oob: float | None = None
        e1400_report: Mapping[str, Any] | None = None

        for stage in FORMAL_POINTS:
            self.assert_code()
            existing = self._existing_stage(stage)
            if existing is not None:
                checkpoint = Path(existing["checkpoint"]).resolve(strict=True)
                source, previous = checkpoint, stage
                mechanism = existing.get("mechanism_check")
                if stage == 100 and mechanism is not None:
                    e100_oob = float(mechanism["mu_oob_fraction"])
                if stage == 1400:
                    e1400_report = existing.get("frozen_comparison")
                if existing.get("continue_training") is not True:
                    status = (
                        "mujoco_exploratory_smoke_ready"
                        if existing.get("conclusion")
                        in {
                            "E1800_mujoco_exploratory_gate_passed",
                            "E2500_mujoco_exploratory_gate_passed",
                        }
                        else "stopped_by_gate"
                    )
                    self.terminal(status, str(existing["conclusion"]), checkpoint)
                    return
                continue

            run_dir, checkpoint, wandb_manifest = self.train_stage(stage, source, previous)
            mechanism: Mapping[str, Any] | None = None
            report: Mapping[str, Any] | None = None
            decision: Mapping[str, Any] | None = None
            behavior: Mapping[str, Any] | None = None
            conclusion = "continue"
            continue_training = True

            if stage in (100, 300):
                mechanism = self.mechanism_check(stage, run_dir, checkpoint, e100_oob)
                if stage == 100:
                    e100_oob = float(mechanism["mu_oob_fraction"])
                if mechanism.get("passed") is not True:
                    conclusion = "E300_mechanism_gate_failed"
                    continue_training = False
            if stage in (700, 1400, 1800):
                report = self.frozen_gate_report(stage, checkpoint)
                decision = self.frozen_decision(stage, report, e1400_report)
                if stage in (700, 1400) and decision.get("passed") is not True:
                    conclusion = str(decision["reason"])
                    continue_training = False
                if stage == 1400:
                    e1400_report = report
            if stage == 1400:
                behavior = self.behavior9(stage, checkpoint)
                # v1.14 explicitly records E1400 behavior but never gates on it.
            if stage == 1800:
                behavior = self.behavior9(stage, checkpoint)
                counts = behavior["counts"]
                score = min(int(counts["full_climb"]), int(counts["rear_hold"]))
                if behavior["exploratory_behavior_gate_passed"]:
                    conclusion = "E1800_mujoco_exploratory_gate_passed"
                    continue_training = False
                elif score == 0:
                    conclusion = "E1800_zero_of_nine_behavior_stop"
                    continue_training = False
                elif score in (1, 2):
                    improvements = decision.get("improvement_vs_E1400", {}) if decision else {}
                    if all(
                        finite_number(improvements.get(name))
                        and float(improvements[name]) >= 0.05
                        for name in (
                            "clamped_latent_mae",
                            "nonbox_pre_prior_action_mae",
                            "post_prior_box_action_mae",
                        )
                    ):
                        conclusion = "E1800_conditional_E2500_authorized"
                        continue_training = True
                    else:
                        conclusion = "E1800_metrics_plateau_stop"
                        continue_training = False
                else:
                    # score >=3 is already covered by the behavior gate; this is defensive.
                    raise RuntimeError("inconsistent E1800 behavior gate state")

            result_path = self.stage_result(
                stage,
                checkpoint,
                run_dir,
                wandb_manifest,
                mechanism=mechanism,
                frozen_report=report,
                frozen_decision=decision,
                behavior=behavior,
                conclusion=conclusion,
                continue_training=continue_training,
            )
            self.finalize_wandb(
                stage,
                checkpoint,
                result_path,
                wandb_manifest,
                self.gate_metrics(mechanism, decision, behavior),
                conclusion,
            )
            self.update(
                checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=sha256_file(checkpoint),
                effective_updates=stage,
                mechanism_check=mechanism,
                frozen_decision=decision,
                behavior9_counts=None if behavior is None else behavior["counts"],
                decision=conclusion,
            )
            if not continue_training:
                status = (
                    "mujoco_exploratory_smoke_ready"
                    if conclusion == "E1800_mujoco_exploratory_gate_passed"
                    else "stopped_by_gate"
                )
                self.terminal(status, conclusion, checkpoint)
                return
            source, previous = checkpoint, stage

        # Reaching here is only legal after the frozen E1800 conditional decision.
        e1800 = json.loads(
            (STATE_ROOT / "stages/E1800/stage_result.json").read_text(encoding="utf-8")
        )
        if e1800.get("conclusion") != "E1800_conditional_E2500_authorized":
            raise RuntimeError("E2500 reached without the immutable E1800 conditional gate")
        run_dir, checkpoint, wandb_manifest = self.train_stage(2500, source, previous)
        report = self.frozen_gate_report(2500, checkpoint)
        decision = self.frozen_decision(2500, report, e1400_report)
        behavior = self.behavior9(2500, checkpoint)
        conclusion = (
            "E2500_mujoco_exploratory_gate_passed"
            if behavior["exploratory_behavior_gate_passed"]
            else "E2500_absolute_cap_without_exploratory_gate"
        )
        result_path = self.stage_result(
            2500,
            checkpoint,
            run_dir,
            wandb_manifest,
            mechanism=None,
            frozen_report=report,
            frozen_decision=decision,
            behavior=behavior,
            conclusion=conclusion,
            continue_training=False,
        )
        self.finalize_wandb(
            2500,
            checkpoint,
            result_path,
            wandb_manifest,
            self.gate_metrics(None, decision, behavior),
            conclusion,
        )
        self.update(
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=sha256_file(checkpoint),
            effective_updates=2500,
            frozen_decision=decision,
            behavior9_counts=behavior["counts"],
            decision=conclusion,
        )
        self.terminal(
            "mujoco_exploratory_smoke_ready"
            if behavior["exploratory_behavior_gate_passed"]
            else "stopped_by_gate",
            conclusion,
            checkpoint,
        )


def main() -> int:
    supervisor: Supervisor | None = None
    try:
        supervisor = Supervisor()
        supervisor.run()
        return 0
    except BlockingIOError:
        print("v1.14 supervisor lock is already held", file=sys.stderr)
        return 3
    except Exception as error:
        traceback.print_exc()
        if supervisor is not None:
            payload = {
                "schema_version": 1,
                "workflow_id": WORKFLOW_ID,
                "status": "infrastructure_or_mechanism_error_requires_repair",
                "error": f"{type(error).__name__}: {error}",
                "checkpoint": supervisor.state.get("checkpoint"),
                "effective_updates": supervisor.state.get("effective_updates"),
                "infrastructure_repair_manifest": str(INFRA_REPAIR.resolve()),
                "infrastructure_repair_manifest_sha256": (
                    supervisor.infrastructure_repair_sha
                ),
                "written_at": now(),
                "next_action": "repair exact fault; do not advance a training stage",
            }
            atomic_json(supervisor.handoff_path, payload)
            supervisor.update(
                status=payload["status"],
                phase=payload["status"],
                last_error=payload["error"],
                active_pid=None,
            )
            # A fail-closed infrastructure/mechanism terminal must not become a
            # systemd restart loop that silently repeats a training segment.
            supervisor.disable_service("infrastructure_or_mechanism_error")
        else:
            subprocess.run(
                ["systemctl", "--user", "disable", SERVICE],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return FAIL_CLOSED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
