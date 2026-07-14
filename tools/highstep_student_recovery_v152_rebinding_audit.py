#!/usr/bin/env python3
"""Read-only proof that v1.5 E300 is safe to resume under v1.5.2 authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

import torch


ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"
EXPECTED_E300_SHA = "79449ba461647781a769d9211bc357f1936238c408226b33a2e500a4921095aa"
EXPECTED_ROOT_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
EXPECTED_OLD_PREREG_SHA = "0be872b32062fe6d162a6b25435ba81c7b45fbc30a703eb0d7597a29a8ff0421"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)


def finite_tree(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--e300", type=Path, required=True)
    parser.add_argument("--old-prereg", type=Path, required=True)
    parser.add_argument("--old-stage-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [args.root, args.e300, args.old_prereg, args.old_stage_result]
    for path in paths:
        path.resolve(strict=True)
    checks = {
        "root_sha256": sha256_file(args.root),
        "e300_sha256": sha256_file(args.e300),
        "old_preregistration_sha256": sha256_file(args.old_prereg),
        "old_stage_result_sha256": sha256_file(args.old_stage_result),
    }
    if checks["root_sha256"] != EXPECTED_ROOT_SHA:
        raise RuntimeError("protected model_172300 root SHA mismatch")
    if checks["e300_sha256"] != EXPECTED_E300_SHA:
        raise RuntimeError("protected E300 SHA mismatch")
    if checks["old_preregistration_sha256"] != EXPECTED_OLD_PREREG_SHA:
        raise RuntimeError("old v1.5 preregistration SHA mismatch")

    root = torch.load(args.root, map_location="cpu", weights_only=False)
    e300 = torch.load(args.e300, map_location="cpu", weights_only=False)
    root_state = root["model_state_dict"]
    e300_state = e300["model_state_dict"]
    if set(root_state) != set(e300_state):
        raise RuntimeError("E300 model-state keys differ from the protected root")
    changed = sorted(
        name for name in root_state if not tensor_equal(root_state[name], e300_state[name])
    )
    expected_changed = sorted(name for name in root_state if name.startswith("estimator."))
    if changed != expected_changed or len(changed) != 16:
        raise RuntimeError(f"E300 changed forbidden tensors: {changed}")

    extra = e300["infos"][ALGORITHM_KEY]
    recovery = extra["student_recovery"]
    binding = recovery["binding_manifest"]
    count = int(extra["student_distill_update_count"])
    if count != 300 or int(recovery["effective_update_count"]) != 300:
        raise RuntimeError("E300 effective-update count is not exactly 300")
    if recovery["preregistration_sha256"] != EXPECTED_OLD_PREREG_SHA:
        raise RuntimeError("E300 is not bound to the frozen v1.5 preregistration")
    if not (
        binding.get("initial_student_sha256") == EXPECTED_ROOT_SHA
        and binding.get("teacher_sha256") == EXPECTED_ROOT_SHA
        and binding.get("student_teacher_storage_independent") is True
        and binding.get("student_ppo_permanently_disabled") is True
        and binding.get("actor_body_frozen") is True
        and binding.get("critic_frozen") is True
        and binding.get("student_privileged_encoder_frozen") is True
        and binding.get("warmup_updates") == 1400
        and binding.get("post_warmup_trainable_action_rows") == [12, 13, 14, 15]
    ):
        raise RuntimeError("E300 checkpoint binding/freeze contract mismatch")

    optimizer = extra["vae_optimizer_state_dict"]
    groups = optimizer.get("param_groups", [])
    if [float(group["lr"]) for group in groups] != [1.0e-3, 1.0e-5]:
        raise RuntimeError("E300 optimizer learning-rate groups changed")
    state = optimizer.get("state", {})
    if len(state) != 16 or not finite_tree(state):
        raise RuntimeError("E300 Adam state is incomplete or non-finite")
    steps = sorted({int(item["step"].item()) for item in state.values()})
    if steps != [4800]:
        raise RuntimeError(f"E300 Adam step is not 300x4x4: {steps}")
    if any(parameter_id in state for parameter_id in groups[1]["params"]):
        raise RuntimeError("box-head Adam state exists before the 1400-update warmup")

    old_result = json.loads(args.old_stage_result.read_text())
    if not (
        old_result.get("stage") == 300
        and old_result.get("checkpoint_sha256") == EXPECTED_E300_SHA
        and old_result.get("decision", {}).get("reason")
        == "two_consecutive_regressions_without_latent_improvement"
    ):
        raise RuntimeError("old stopped_by_gate evidence no longer matches E300")

    payload = {
        "schema_version": 1,
        "kind": "highstep_student_recovery_v152_rebinding_audit",
        "conclusion": "mechanism_and_optimizer_correct_exact_resume_allowed",
        "exact_resume_allowed": True,
        "source_checkpoint": str(args.e300.resolve()),
        "source_checkpoint_sha256": checks["e300_sha256"],
        "source_effective_updates": 300,
        "protected_root_checkpoint": str(args.root.resolve()),
        "protected_root_sha256": checks["root_sha256"],
        "old_preregistration": str(args.old_prereg.resolve()),
        "old_preregistration_sha256": checks["old_preregistration_sha256"],
        "old_stage_result": str(args.old_stage_result.resolve()),
        "old_stage_result_sha256": checks["old_stage_result_sha256"],
        "changed_model_tensors": changed,
        "changed_model_tensor_count": len(changed),
        "forbidden_model_tensor_changes": [],
        "optimizer_group_learning_rates": [1.0e-3, 1.0e-5],
        "adam_state_parameter_count": len(state),
        "adam_step": 4800,
        "effective_updates": count,
        "old_stop_classification": "diagnostic_auxiliary_gate_not_mechanism_failure",
        "preserved_old_terminal_fact": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    args.output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output), **payload}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
