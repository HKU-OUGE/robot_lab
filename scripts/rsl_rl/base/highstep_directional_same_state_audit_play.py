#!/usr/bin/env python3
"""Collect one fail-closed same-state trace for the preregistered directional matrix."""

from __future__ import annotations

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
BASE_RUNTIME = Path(__file__).with_name("highstep_same_state_audit_play.py")
CANDIDATE_RUNTIME = Path(__file__).with_name("highstep_candidate_same_state_audit_play.py")
PREREG = ROOT / "tmp/highstep_directional_real_robot_trial_20260713/preregistration.json"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
B500 = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_no_prior_Student/2026-07-12_23-55-22_student_recovery_v111_"
    "B500_20260712_235515/model_498.pt"
)
PREREG_SHA256 = "6cfe31cf04e1321416673c4d13e795773f678b529fcc7f147201f46f1048223d"
SPEC_SHA256 = "1fbb17ab6c23a24005e8e34dadc11a6e4592a7be671a575206fd24819e5f3f1c"
B500_SHA256 = "f76c8ff2b772c1868b8a97b505febc09ad1f0ece7b5080a748f20e80101eb159"
MAX_STEPS = 600


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _atomic_read_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _authority() -> dict[str, Any]:
    for path, expected in ((PREREG, PREREG_SHA256), (SPEC, SPEC_SHA256), (B500, B500_SHA256)):
        actual = sha256_file(path.resolve(strict=True))
        if actual != expected:
            raise RuntimeError(f"directional authority changed: {path}: {actual} != {expected}")
    if PREREG.stat().st_mode & 0o222:
        raise RuntimeError("directional preregistration is writable")
    payload = json.loads(PREREG.read_text())
    if payload.get("candidate_state_name") != "directional_real_robot_trial_candidate":
        raise RuntimeError("directional preregistration state changed")
    return payload


def _install_matrix(base: Any, audit: Any, prereg: Mapping[str, Any], corridor: str) -> None:
    rows = [
        {"run_id": f"seed{seed}_{item['scenario']}", "seed": seed, **item}
        for seed in prereg["seeds"] for item in prereg["corridors"][corridor]
    ]
    if len(rows) != 9 or len({row["run_id"] for row in rows}) != 9:
        raise RuntimeError("directional matrix is not exactly 3x3")
    base.TRAJECTORIES = {
        row["run_id"]: base.Trajectory(
            row["run_id"], int(row["seed"]), str(row["scenario"]),
            float(row["lateral_offset_m"]), float(row["yaw_offset_deg"]),
        ) for row in rows
    }
    audit.DEFAULT_AUDIT_TRAJECTORIES = tuple(copy.deepcopy(rows))


def _install_plan(base: Any, audit: Any, prereg: Mapping[str, Any], corridor: str, checkpoint: Path) -> None:
    original_hashes = base._runtime_code_hashes
    base_plan = audit.base_suite_plan_payload()

    def plan() -> dict[str, Any]:
        payload = copy.deepcopy(base_plan)
        payload.update({
            "schema_version": 3,
            "kind": "highstep_directional_same_state_plan",
            "workflow_id": prereg["workflow_id"],
            "candidate_state_name": prereg["candidate_state_name"],
            "corridor": corridor,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "preregistration": str(PREREG.resolve()),
            "preregistration_sha256": PREREG_SHA256,
            "trajectories": [
                {**row, "requested_play_max_steps": MAX_STEPS, "termination_allowed": False,
                 "behavior_outcome": "observed_not_preregistered"}
                for row in audit.DEFAULT_AUDIT_TRAJECTORIES
            ],
            "training_allowed": False,
        })
        return payload

    def hashes() -> dict[str, str]:
        payload = dict(original_hashes())
        payload[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
        payload[str(PREREG.resolve())] = PREREG_SHA256
        payload[str(SPEC.resolve())] = SPEC_SHA256
        return payload

    audit.base_suite_plan_payload = plan
    base._runtime_code_hashes = hashes


def _validate(output_root: Path, trajectory: Any, checkpoint: Path) -> dict[str, Any]:
    summary_path = output_root.resolve() / trajectory.run_id / "run_summary.json"
    summary = json.loads(summary_path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    outcome = summary.get("outcome") or {}
    frames = Path(str(summary.get("frames_path", ""))).resolve(strict=True)
    binding = summary.get("checkpoint_binding") or {}
    if not (
        summary.get("frame_count") == MAX_STEPS
        and outcome.get("loop_steps") == MAX_STEPS
        and outcome.get("terminated") is False
        and summary.get("runner_manual_action_allclose_all_frames") is True
        and summary.get("all_forward_state_digests_unchanged") is True
        and summary.get("all_observation_and_prior_inputs_unchanged") is True
        and float(summary.get("action_decomposition_residual_global_max_abs", 1.0)) <= 1.0e-6
        and binding.get("student_checkpoint_sha256") == sha256_file(checkpoint)
        and summary.get("frames_sha256") == sha256_file(frames)
    ):
        raise RuntimeError(f"directional trace integrity failed: {trajectory.run_id}")
    return {
        "run_id": trajectory.run_id,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "frames": str(frames), "frames_sha256": sha256_file(frames),
        "summary": str(summary_path), "summary_sha256": sha256_file(summary_path),
        "frame_count": MAX_STEPS, "finite_parity_digest_decomposition_verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    prereg = _authority()
    sys.path.insert(0, str(ROOT))
    from tools import highstep_same_state_audit as audit
    audit.EXPECTED_SPEC_SHA256 = SPEC_SHA256

    checkpoint_text = os.environ.get("HIGHSTEP_DIRECTIONAL_CHECKPOINT", "")
    corridor = os.environ.get("HIGHSTEP_DIRECTIONAL_CORRIDOR", "")
    checkpoint = Path(checkpoint_text).expanduser().resolve(strict=True)
    allowed = {Path(row["path"]).resolve(): row["sha256"] for row in prereg["checkpoints"]}
    expected_sha = allowed.get(checkpoint)
    if expected_sha is None:
        continuation_root = (ROOT / "tmp/highstep_directional_real_robot_trial_20260713/r5_runs").resolve()
        if continuation_root not in checkpoint.parents:
            raise RuntimeError("directional checkpoint is outside the preregistered R5 continuation root")
        expected_sha = os.environ.get("HIGHSTEP_DIRECTIONAL_CHECKPOINT_SHA256", "")
    if len(expected_sha) != 64 or sha256_file(checkpoint) != expected_sha:
        raise RuntimeError("directional checkpoint SHA is not bound")
    if corridor not in prereg["corridors"]:
        raise RuntimeError("directional corridor is not preregistered")

    if checkpoint == B500.resolve():
        base = _load(BASE_RUNTIME, "highstep_directional_base_runtime")
        base.B500_CHECKPOINT = checkpoint
        base.B500_SHA256 = B500_SHA256
        _install_matrix(base, audit, prereg, corridor)
        _install_plan(base, audit, prereg, corridor, checkpoint)
    else:
        os.environ["HIGHSTEP_CANDIDATE_STAGE"] = "R5"
        os.environ["HIGHSTEP_R2_CANDIDATE_CHECKPOINT"] = str(checkpoint)
        os.environ["HIGHSTEP_R2_CANDIDATE_SHA256"] = expected_sha
        candidate = _load(CANDIDATE_RUNTIME, "highstep_directional_candidate_runtime")
        _install_matrix(type("holder", (), {"Trajectory": None})(), audit, prereg, corridor) if False else None
        # The candidate adapter builds its suite plan from the audit matrix, so install it first there.
        rows = [
            {"run_id": f"seed{seed}_{item['scenario']}", "seed": seed, **item}
            for seed in prereg["seeds"] for item in prereg["corridors"][corridor]
        ]
        audit.DEFAULT_AUDIT_TRAJECTORIES = tuple(copy.deepcopy(rows))
        binding = candidate.resolve_candidate_binding()
        base = candidate.install_candidate_adapter(binding)
        _install_matrix(base, audit, prereg, corridor)
        _install_plan(base, audit, prereg, corridor, checkpoint)

    from isaaclab.app import AppLauncher
    parser = base._build_parser()
    args = parser.parse_args(argv)
    trajectory = base._validate_fixed_cli(args)
    args.seed = trajectory.seed
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        if base._run(args, trajectory, app) != 0:
            raise RuntimeError("directional same-state runtime returned non-zero")
        result = _validate(Path(args.output_root), trajectory, checkpoint)
        print("[HIGHSTEP_DIRECTIONAL_TRACE_JSON] " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
