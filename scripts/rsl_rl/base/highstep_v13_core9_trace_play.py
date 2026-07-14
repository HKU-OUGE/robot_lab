#!/usr/bin/env python3
"""Collect one immutable B500 same-state trace from the v1.3 core9 matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
BASE_RUNTIME_PATH = Path(__file__).with_name("highstep_same_state_audit_play.py").resolve()
AUDIT_TOOL_PATH = ROOT / "tools/highstep_same_state_audit.py"
PREREGISTRATION = ROOT / "tmp/highstep_student_recovery_v13_20260713/r4_preregistration.json"
BASE_RUNTIME_SHA256 = "e63c2843d51d7b880166acb33b02d0c89abaa651e986ce7a07e0ece5b4ddad40"
AUDIT_TOOL_SHA256 = "08feea8121405677db3d29a4e1ca27a74d4611b8a214c1229ef35cbd9c7f6c30"
PREREGISTRATION_SHA256 = "8f43112e94fbe032fd4d640292151210b692c3f853d460a98a0456fb0ed7b45e"
B500 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_no_prior_Student/2026-07-12_23-55-22_student_recovery_v111_"
    "B500_20260712_235515/model_498.pt"
)
B500_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
MAX_STEPS = 600


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_authority() -> dict[str, Any]:
    expected = {
        BASE_RUNTIME_PATH: BASE_RUNTIME_SHA256,
        AUDIT_TOOL_PATH: AUDIT_TOOL_SHA256,
        PREREGISTRATION: PREREGISTRATION_SHA256,
        B500: B500_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path.resolve(strict=True)) != digest:
            raise RuntimeError(f"v1.3 trace authority changed: {path}")
    if PREREGISTRATION.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("v1.3 R4 preregistration is writable")
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if not (
        payload.get("kind") == "highstep_student_recovery_v13_r4_preregistration"
        and payload.get("workflow_id") == "highstep_student_recovery_v13_20260713"
        and payload.get("training_allowed") is False
    ):
        raise RuntimeError("v1.3 R4 preregistration fields changed")
    return payload


def _load_base_runtime() -> Any:
    module_name = "highstep_v13_reused_same_state_runtime"
    spec = importlib.util.spec_from_file_location(module_name, BASE_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen same-state runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_read_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def install_v13_matrix(base_runtime: Any, prereg: Mapping[str, Any]) -> None:
    sys.path.insert(0, str(ROOT))
    from tools import highstep_same_state_audit as audit

    trajectories = {
        f"seed{int(row['seed'])}_{row['scenario']}": base_runtime.Trajectory(
            f"seed{int(row['seed'])}_{row['scenario']}",
            int(row["seed"]),
            str(row["scenario"]),
            float(row["lateral_offset_m"]),
            float(row["yaw_offset_deg"]),
        )
        for row in prereg["core9_matrix"]
    }
    if len(trajectories) != 9:
        raise RuntimeError("v1.3 trace matrix is not exactly core9")
    base_runtime.TRAJECTORIES = trajectories
    original_hashes = base_runtime._runtime_code_hashes
    original_binding = base_runtime._ensure_runtime_binding
    base_plan = audit.base_suite_plan_payload()

    # ``AuditPreregistration`` intentionally validates base-suite identities
    # against this module variable.  The reused runtime used to update only its
    # CLI trajectory table, so the six v1.3 identities outside the historical
    # base3 were rejected before a trace could start.  Keep the historical
    # audit tool unchanged and install exactly the immutable, SHA-bound core9
    # matrix for this wrapper process only.
    audit.DEFAULT_AUDIT_TRAJECTORIES = tuple(
        {
            "run_id": name,
            "seed": value.seed,
            "scenario": value.scenario,
            "lateral_offset_m": value.lateral_offset_m,
            "yaw_offset_deg": value.yaw_offset_deg,
        }
        for name, value in sorted(trajectories.items())
    )

    def plan_payload() -> dict[str, Any]:
        payload = copy.deepcopy(base_plan)
        payload.update(
            {
                "schema_version": 2,
                "kind": "highstep_v13_b500_core9_same_state_plan",
                "workflow_id": "highstep_student_recovery_v13_20260713",
                "preregistration": str(PREREGISTRATION.resolve()),
                "preregistration_sha256": PREREGISTRATION_SHA256,
                "trajectories": [
                    {
                        "run_id": name,
                        "seed": value.seed,
                        "scenario": value.scenario,
                        "lateral_offset_m": value.lateral_offset_m,
                        "yaw_offset_deg": value.yaw_offset_deg,
                        "requested_play_max_steps": MAX_STEPS,
                        "termination_allowed": False,
                        "b500_role": next(
                            row["b500_role"]
                            for row in prereg["core9_matrix"]
                            if int(row["seed"]) == value.seed
                            and row["scenario"] == value.scenario
                        ),
                    }
                    for name, value in sorted(trajectories.items())
                ],
                "training_allowed": False,
            }
        )
        return payload

    def runtime_hashes() -> dict[str, str]:
        payload = dict(original_hashes())
        payload[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
        payload[str(PREREGISTRATION.resolve())] = PREREGISTRATION_SHA256
        return payload

    def runtime_binding(
        output_root: Path, suite_plan_path: Path, suite_plan_sha256: str
    ) -> tuple[Path, str]:
        path, _ = original_binding(output_root, suite_plan_path, suite_plan_sha256)
        base_payload = json.loads(path.read_text(encoding="utf-8"))
        v13_path = output_root.expanduser().resolve() / "v13_core9_runtime_binding.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_v13_b500_core9_runtime_binding",
            "workflow_id": "highstep_student_recovery_v13_20260713",
            "base_runtime_binding": str(path.resolve()),
            "base_runtime_binding_sha256": sha256_file(path),
            "suite_plan": str(suite_plan_path.resolve()),
            "suite_plan_sha256": suite_plan_sha256,
            "checkpoint": str(B500.resolve()),
            "checkpoint_sha256": B500_SHA256,
            "preregistration": str(PREREGISTRATION.resolve()),
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "runtime_code_sha256": runtime_hashes(),
            "base_binding_payload": base_payload,
        }
        if v13_path.exists():
            if json.loads(v13_path.read_text(encoding="utf-8")) != payload:
                raise RuntimeError("v1.3 core9 runtime binding changed")
            if v13_path.stat().st_mode & 0o222:
                raise RuntimeError("v1.3 core9 runtime binding is writable")
        else:
            _atomic_read_only_json(v13_path, payload)
        return v13_path, sha256_file(v13_path)

    audit.base_suite_plan_payload = plan_payload
    base_runtime._runtime_code_hashes = runtime_hashes
    base_runtime._ensure_runtime_binding = runtime_binding


def validate_completed_run(output_root: Path, trajectory: Any) -> dict[str, Any]:
    summary_path = output_root.resolve() / trajectory.run_id / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcome = summary.get("outcome") or {}
    frames = Path(str(summary.get("frames_path", ""))).resolve(strict=True)
    if not (
        summary.get("frame_count") == MAX_STEPS
        and outcome.get("loop_steps") == MAX_STEPS
        and outcome.get("terminated") is False
        and summary.get("runner_manual_action_allclose_all_frames") is True
        and summary.get("all_forward_state_digests_unchanged") is True
        and summary.get("all_observation_and_prior_inputs_unchanged") is True
        and float(summary.get("action_decomposition_residual_global_max_abs", 1.0)) <= 1.0e-6
        and summary.get("frames_sha256") == sha256_file(frames)
    ):
        raise RuntimeError(f"v1.3 trace failed integrity checks: {trajectory.run_id}")
    return {
        "run_id": trajectory.run_id,
        "frames": str(frames),
        "frames_sha256": sha256_file(frames),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "full_climb_success": outcome.get("full_climb_success"),
        "rear_hold_success": outcome.get("rear_on_platform_hold_success"),
        "front_top_support": outcome.get("front_top_support_reached"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    prereg = _assert_authority()
    base_runtime = _load_base_runtime()
    install_v13_matrix(base_runtime, prereg)

    from isaaclab.app import AppLauncher

    parser = base_runtime._build_parser()
    args = parser.parse_args(argv)
    trajectory = base_runtime._validate_fixed_cli(args)
    args.seed = trajectory.seed
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        result = base_runtime._run(args, trajectory, simulation_app)
        if result != 0:
            raise RuntimeError(f"same-state runtime returned {result}")
        validation = validate_completed_run(Path(args.output_root), trajectory)
        print(
            "[HIGHSTEP_V13_CORE9_TRACE_JSON] "
            + json.dumps(validation, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
