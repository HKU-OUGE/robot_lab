#!/usr/bin/env python3
"""Audit one v1.5 candidate on the frozen 0707 shared-state driver.

This launcher deliberately reuses the already validated historical reference
runtime without changing it.  The deployed 0707 Student drives the simulator;
the candidate and the protected v1.5 Teacher are read-only forward passes on
exactly those states.  Candidate paths and hashes are supplied explicitly by
the supervisor and are captured in every output manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_LAUNCHER = Path(__file__).with_name("highstep_v15_reference_audit_play.py")
TEACHER = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_Teacher/"
    "2026-07-11_11-22-23/model_172300.pt"
)
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
TASK = "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior-ArcdogAdjustableLeg-v0"
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"
SPEC_SHA = "c892c0d8bc811228b34bb3c439e2977d5f473ecf1374453f3032509e2776f63e"
HISTORICAL_V171_SPEC_SHA = "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917"
V152_WORKFLOW = ROOT / "tmp/highstep_student_recovery_v152_20260713"
V152_PREREG = V152_WORKFLOW / "preregistration_v1.json"
V152_PREREG_SHA = "473b8554d25d253886547fe4af72d49e30dad796d66fd9a3944e1d0ad8238ec2"
V152_REBINDING_AUDIT = V152_WORKFLOW / "rebinding_audit.json"
V152_REBINDING_AUDIT_SHA = "2d2826404676a04a5ff215dea891bc812516b55d579ccc0ba86e62ea4c269a30"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reference() -> Any:
    spec = importlib.util.spec_from_file_location("highstep_v15_candidate_reference_runtime", REFERENCE_LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference runtime {REFERENCE_LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    candidate = Path(os.environ["HIGHSTEP_V15_CANDIDATE_CHECKPOINT"]).resolve(strict=True)
    expected_sha = os.environ["HIGHSTEP_V15_CANDIDATE_SHA256"]
    output_root = Path(os.environ["HIGHSTEP_V15_CANDIDATE_AUDIT_ROOT"]).resolve()
    if sha256_file(candidate) != expected_sha:
        raise RuntimeError("candidate checkpoint SHA256 mismatch")
    if sha256_file(TEACHER) != TEACHER_SHA:
        raise RuntimeError("protected v1.5 Teacher SHA256 mismatch")
    payload = torch.load(candidate, map_location="cpu", weights_only=False)
    recovery = payload.get("infos", {}).get("robot_lab_algorithm_checkpoint_state", {}).get(
        "student_recovery", {}
    )
    binding = recovery.get("binding_manifest", {})
    exact_0707 = recovery.get("stage") == "0707_EXACT"
    historical_0707_exact = recovery.get("stage") == "HISTORICAL_0707_EXACT"
    authority_files = [
        (
            "formal spec",
            SPEC,
            HISTORICAL_V171_SPEC_SHA
            if historical_0707_exact else
            binding.get("recovery_spec_sha256")
            if exact_0707 else SPEC_SHA,
        ),
    ]
    if exact_0707 or historical_0707_exact:
        authority_files.append((
            "0707 route preregistration",
            Path(str(binding.get("preregistration_path", ""))),
            str(binding.get("preregistration_sha256", "")),
        ))
        expected = (
            (1400, 2.0, 1.5) if historical_0707_exact else (1200, 0.0, 0.0)
        )
        if not (
            binding.get("warmup_main_action_target") == "teacher_pre_prior"
            and binding.get("warmup_updates") == expected[0]
            and binding.get("phase_scale") == expected[1]
            and binding.get("rear_box_scale") == expected[2]
            and binding.get("warmup_prior_box_loss_coefficient") == 0.0
        ):
            raise RuntimeError("0707 route candidate binding semantics changed")
        if historical_0707_exact:
            current_prereg = Path(
                os.environ["HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_PATH"]
            )
            current_prereg_sha = os.environ[
                "HIGHSTEP_HISTORICAL_0707_EXACT_PREREGISTRATION_SHA256"
            ]
            rebinding = Path(os.environ["HIGHSTEP_HISTORICAL_0707_EXACT_REBINDING_PATH"])
            rebinding_sha = os.environ[
                "HIGHSTEP_HISTORICAL_0707_EXACT_REBINDING_SHA256"
            ]
            authority_files.extend((
                ("v1.7.1 preregistration", current_prereg, current_prereg_sha),
                ("v1.7.1 authority rebinding", rebinding, rebinding_sha),
            ))
    else:
        authority_files.extend((
            ("v1.5.2 preregistration", V152_PREREG, V152_PREREG_SHA),
            ("v1.5.2 rebinding audit", V152_REBINDING_AUDIT, V152_REBINDING_AUDIT_SHA),
        ))
    for label, path, expected in authority_files:
        if sha256_file(path.resolve(strict=True)) != expected:
            raise RuntimeError(f"{label} SHA256 mismatch")

    ref = _load_reference()
    teacher_agent = TEACHER.parent / "params/agent.yaml"
    teacher_env = TEACHER.parent / "params/env.yaml"
    candidate_agent = candidate.parent / "params/agent.yaml"
    candidate_env = candidate.parent / "params/env.yaml"
    for path in (teacher_agent, teacher_env, candidate_agent, candidate_env):
        path.resolve(strict=True)

    profile = {
        "checkpoint": candidate,
        "checkpoint_sha256": expected_sha,
        "agent": candidate_agent,
        "agent_sha256": sha256_file(candidate_agent),
        "env": candidate_env,
        "env_sha256": sha256_file(candidate_env),
    }
    ref.TEACHER = TEACHER
    ref.TEACHER_SHA = TEACHER_SHA
    ref.TEACHER_AGENT = teacher_agent
    ref.TEACHER_AGENT_SHA = sha256_file(teacher_agent)
    ref.TEACHER_ENV = teacher_env
    ref.TEACHER_ENV_SHA = sha256_file(teacher_env)
    ref.TASK = TASK
    ref.WORKFLOW = output_root
    ref.SPEC_SHA = (
        HISTORICAL_V171_SPEC_SHA if historical_0707_exact else
        binding.get("recovery_spec_sha256") if exact_0707 else SPEC_SHA
    )
    ref.PROFILES = {"candidate": profile}
    ref._profile = lambda: ("candidate", dict(profile))

    original_install = ref._install

    def install(base: Any, audit: Any, name: str, selected: Any) -> None:
        original_contract = audit.PostPriorContract.from_teacher_env_config.__func__
        # The frozen 0707 reference data remain bound to v1.5.  Only this
        # candidate-side runtime is rebound to the current v1.5.2 authority.
        audit.EXPECTED_SPEC_SHA256 = ref.SPEC_SHA
        original_install(base, audit, name, selected)

        def v15_contract(cls: Any, path: Path = teacher_env):
            return original_contract(
                cls, teacher_env, expected_sha256=ref.TEACHER_ENV_SHA, update_count=172300
            )

        audit.PostPriorContract.from_teacher_env_config = classmethod(v15_contract)
        runtime_hashes = base._runtime_code_hashes

        def hashes() -> dict[str, str]:
            payload = runtime_hashes()
            payload[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
            return payload

        base._runtime_code_hashes = hashes

    ref._install = install

    def install_schedule(profile_name: str, selected: Any) -> Path:
        from robot_lab.tasks.locomotion.velocity.mdp import highstep_schedule as schedule

        driver = Path(ref.DRIVER["checkpoint"])
        schedule_update = float(int(driver.stem.split("_")[-1]))
        manifest_path = output_root / "0707_reference" / "shared_driver_v2_candidate" / "historical_schedule_reconstruction.json"
        payload = {
            "schema_version": 1,
            "kind": "highstep_v15_candidate_shared_driver_schedule_reconstruction",
            "profile": profile_name,
            "shared_state_driver_checkpoint": str(driver.resolve()),
            "shared_state_driver_checkpoint_sha256": ref.DRIVER["checkpoint_sha256"],
            "candidate_checkpoint": str(candidate),
            "candidate_checkpoint_sha256": expected_sha,
            "schedule_update": schedule_update,
            "driver_agent_config_sha256": ref.DRIVER["agent_sha256"],
            "driver_env_config_sha256": ref.DRIVER["env_sha256"],
            "teacher_agent_config_sha256": ref.TEACHER_AGENT_SHA,
            "teacher_env_config_sha256": ref.TEACHER_ENV_SHA,
            "reason": "The frozen deployed 0707 Student drives all reference and candidate audit states; the candidate is a side-effect-free read-only forward.",
            "old_run_modified": False,
        }
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != payload:
                raise RuntimeError("candidate shared-driver schedule reconstruction changed")
        else:
            ref._atomic_read_only_json(manifest_path, payload)

        schedule.resolve_loaded_schedule_update = lambda *args, **kwargs: (
            schedule_update,
            {
                "method": "v15_candidate_frozen_0707_shared_driver",
                "source_manifest": str(manifest_path),
                "checkpoint_iteration": int(schedule_update),
                "legacy_fallback_allowed": False,
                "legacy_fallback_used": False,
            },
        )
        schedule.load_source_manifest = lambda *args, **kwargs: (manifest_path, payload)

        def compatible(source: Any, current: Any) -> None:
            driver_env = driver.parent / "params/env.yaml"
            if source != payload or sha256_file(driver) != ref.DRIVER["checkpoint_sha256"] or sha256_file(driver_env) != ref.DRIVER["env_sha256"]:
                raise RuntimeError("frozen 0707 shared-driver schedule/config authority changed")

        schedule.assert_schedule_definition_compatible = compatible
        return manifest_path

    ref._install_historical_schedule = install_schedule
    return int(ref.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
