#!/usr/bin/env python3
"""Audit one behavior-gated R2 candidate in the frozen base3 same-state runtime.

The candidate is selected only through two environment variables::

    HIGHSTEP_R2_CANDIDATE_CHECKPOINT=/absolute/path/model_N.pt
    HIGHSTEP_R2_CANDIDATE_SHA256=<64 lowercase hex characters>

This launcher is a thin adapter around ``highstep_same_state_audit_play.py``.
It reuses that launcher's environment, reset, tracker, 600-frame loop and the
unchanged ``tools.highstep_same_state_audit`` session/runtime adapter/shared
post-prior.  It replaces only the read-only Student checkpoint binding and the
suite plan: candidate behavior is observed, never forced to reproduce B500's
nominal/offset success or failure outcomes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[3]
BASE_RUNTIME_PATH = Path(__file__).with_name("highstep_same_state_audit_play.py").resolve()
AUDIT_TOOL_PATH = ROOT / "tools/highstep_same_state_audit.py"
R2_PREREGISTRATION_PATH = (
    ROOT / "tmp/highstep_student_recovery_v11_20260712/r2_preregistration.json"
)
R3_PREREGISTRATION_PATH = (
    ROOT / "tmp/highstep_student_recovery_v12_20260713/r3_preregistration.json"
)
R4_PREREGISTRATION_PATH = (
    ROOT / "tmp/highstep_student_recovery_v13_20260713/r4_preregistration.json"
)
R5_PREREGISTRATION_PATH = (
    ROOT / "tmp/highstep_student_recovery_v13_20260713/r5_preregistration.json"
)
R3_ANCHOR_CHECKPOINT = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_"
    "student_recovery_r2_Student/2026-07-13_03-37-24_student_recovery_r2_"
    "main_50_20260713_033718_2942949/model_49.pt"
)

BASE_RUNTIME_SHA256 = "83d5026dca671ff0a997a42adf61baccc8c0bdc55b5af369830d336a339dc62c"
AUDIT_TOOL_SHA256 = "fcd3069bffd0d2ef1ff608400a479409785b56380ad9cf0d58f25ad8b676b69c"
R2_PREREGISTRATION_SHA256 = (
    "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
)
R3_PREREGISTRATION_SHA256 = (
    "13c184e4b6e2c3b514f52466dac1e27ff18256433716fd0c6b5d0890d93945e6"
)
R4_PREREGISTRATION_SHA256 = (
    "8f43112e94fbe032fd4d640292151210b692c3f853d460a98a0456fb0ed7b45e"
)
R5_PREREGISTRATION_SHA256 = (
    "c222cbeafc5ca1dbd1cb94cdb93b588a25cc7b7d160af8385c4fdd8a14a48736"
)
R3_ANCHOR_SHA256 = (
    "e875424ed69eaaa474a9ac6849a6066dd264ce0eaf04ad63a418c3af1d66918c"
)

CANDIDATE_STAGE = os.environ.get("HIGHSTEP_CANDIDATE_STAGE", "R2").strip().upper()
if CANDIDATE_STAGE not in {"R2", "R3", "R4", "R5"}:
    raise RuntimeError(f"unsupported HIGHSTEP_CANDIDATE_STAGE={CANDIDATE_STAGE!r}")
CANDIDATE_PREREGISTRATION_PATH = {
    "R2": R2_PREREGISTRATION_PATH,
    "R3": R3_PREREGISTRATION_PATH,
    "R4": R4_PREREGISTRATION_PATH,
    "R5": R5_PREREGISTRATION_PATH,
}[CANDIDATE_STAGE]
CANDIDATE_PREREGISTRATION_SHA256 = {
    "R2": R2_PREREGISTRATION_SHA256,
    "R3": R3_PREREGISTRATION_SHA256,
    "R4": R4_PREREGISTRATION_SHA256,
    "R5": R5_PREREGISTRATION_SHA256,
}[CANDIDATE_STAGE]
CANDIDATE_ANCHOR_CHECKPOINT = (
    R3_ANCHOR_CHECKPOINT if CANDIDATE_STAGE == "R3" else None
)
CANDIDATE_ANCHOR_SHA256 = R3_ANCHOR_SHA256 if CANDIDATE_STAGE == "R3" else None

CANDIDATE_CHECKPOINT_ENV = "HIGHSTEP_R2_CANDIDATE_CHECKPOINT"
CANDIDATE_SHA256_ENV = "HIGHSTEP_R2_CANDIDATE_SHA256"
MAX_STEPS = 600

R2_OPTIMIZER_PARAMETER_NAMES = (
    "estimator.encoder.0.weight",
    "estimator.encoder.0.bias",
    "estimator.encoder.2.weight",
    "estimator.encoder.2.bias",
    "estimator.fc_mu.weight",
    "estimator.fc_mu.bias",
    "algorithm.r2_box_weight",
    "algorithm.r2_box_bias",
)
R2_ALLOWED_FULL_MODEL_TENSORS = frozenset(R2_OPTIMIZER_PARAMETER_NAMES[:6])
R2_ACTOR_BOX_ROW_TENSORS = frozenset({"actor.6.weight", "actor.6.bias"})
R3_OPTIMIZER_PARAMETER_NAMES = ("actor.6.weight", "actor.6.bias")
R5_OPTIMIZER_PARAMETER_NAMES = (
    "actor.4.weight",
    "actor.4.bias",
    "actor.6.weight",
    "actor.6.bias",
)

sys.path.insert(0, str(ROOT))
from tools import highstep_same_state_audit as audit  # noqa: E402


_BASE_BOUND_AUDIT_MODELS = audit.BoundAuditModels
_ACTIVE_BINDING: "CandidateBinding | None" = None
_ACTIVE_BASE_RUNTIME: Any | None = None


@dataclass(frozen=True)
class CandidateBinding:
    checkpoint: Path
    sha256: str


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return text


def _assert_frozen_sources() -> None:
    expected = {
        BASE_RUNTIME_PATH: BASE_RUNTIME_SHA256,
        AUDIT_TOOL_PATH: AUDIT_TOOL_SHA256,
        CANDIDATE_PREREGISTRATION_PATH: CANDIDATE_PREREGISTRATION_SHA256,
    }
    if CANDIDATE_STAGE == "R3":
        expected[R3_ANCHOR_CHECKPOINT] = R3_ANCHOR_SHA256
    mismatches = []
    for path, expected_sha in expected.items():
        actual_sha = sha256_file(path.resolve(strict=True))
        if actual_sha != expected_sha:
            mismatches.append(f"{path}: {actual_sha} != {expected_sha}")
    if mismatches:
        raise RuntimeError("candidate audit frozen source binding changed: " + "; ".join(mismatches))


def resolve_candidate_binding(
    environ: Mapping[str, str] | None = None,
) -> CandidateBinding:
    """Resolve and hash the environment-bound R2 candidate before Isaac starts."""

    source = os.environ if environ is None else environ
    raw_path = str(source.get(CANDIDATE_CHECKPOINT_ENV, "")).strip()
    expected_sha = _require_sha256(
        source.get(CANDIDATE_SHA256_ENV, ""), CANDIDATE_SHA256_ENV
    )
    if not raw_path:
        raise ValueError(f"{CANDIDATE_CHECKPOINT_ENV} is required")
    if not os.path.isabs(os.path.expanduser(raw_path)):
        raise ValueError(f"{CANDIDATE_CHECKPOINT_ENV} must be an absolute path")
    checkpoint = Path(raw_path).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError(f"candidate checkpoint is not a regular file: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"candidate checkpoint SHA256 mismatch: {actual_sha} != {expected_sha}"
        )
    reserved = {
        audit.CURRENT_STUDENT_CHECKPOINT.resolve(strict=True),
        audit.ROOT_STUDENT_CHECKPOINT.resolve(strict=True),
        audit.TEACHER_CHECKPOINT.resolve(strict=True),
    }
    if checkpoint in reserved:
        raise ValueError("candidate checkpoint must be distinct from B500, model_900 and Teacher")
    return CandidateBinding(checkpoint=checkpoint, sha256=actual_sha)


def validate_candidate_tensor_deltas(
    candidate_state: Mapping[str, torch.Tensor],
    b500_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove that the candidate changed only its preregistered live scope."""

    if set(candidate_state) != set(b500_state):
        missing = sorted(set(b500_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(b500_state))
        raise RuntimeError(f"candidate/B500 model-state keys differ: missing={missing}, extra={extra}")

    changed_full: list[str] = []
    changed_box_rows: list[str] = []
    allowed_full = (
        frozenset(R5_OPTIMIZER_PARAMETER_NAMES)
        if CANDIDATE_STAGE == "R5"
        else (
            frozenset(R3_OPTIMIZER_PARAMETER_NAMES)
            if CANDIDATE_STAGE in {"R3", "R4"}
            else R2_ALLOWED_FULL_MODEL_TENSORS
        )
    )
    actor_row_tensors = (
        frozenset()
        if CANDIDATE_STAGE in {"R3", "R4", "R5"}
        else R2_ACTOR_BOX_ROW_TENSORS
    )
    for key in sorted(candidate_state):
        candidate = candidate_state[key]
        anchor = b500_state[key]
        if not isinstance(candidate, torch.Tensor) or not isinstance(anchor, torch.Tensor):
            raise RuntimeError(f"candidate/B500 state contains a non-tensor value: {key}")
        if candidate.shape != anchor.shape or candidate.dtype != anchor.dtype:
            raise RuntimeError(
                f"candidate/B500 tensor contract changed at {key}: "
                f"{tuple(candidate.shape)}/{candidate.dtype} != {tuple(anchor.shape)}/{anchor.dtype}"
            )
        if not bool(torch.isfinite(candidate).all().item()):
            raise RuntimeError(f"candidate tensor is non-finite: {key}")

        if key in allowed_full:
            if not torch.equal(candidate, anchor):
                changed_full.append(key)
            continue

        if key in actor_row_tensors:
            expected_ndim = 2 if key.endswith("weight") else 1
            if candidate.ndim != expected_ndim or candidate.shape[0] != 16:
                raise RuntimeError(f"candidate actor output tensor has invalid shape at {key}")
            if not torch.equal(candidate[:12], anchor[:12]):
                raise RuntimeError(f"candidate has an unauthorized non-box actor-row delta: {key}")
            if not torch.equal(candidate[12:16], anchor[12:16]):
                changed_box_rows.append(key)
            continue

        if not torch.equal(candidate, anchor):
            raise RuntimeError(f"candidate has an unauthorized tensor delta: {key}")

    manifest = {
        "allowed_full_model_tensors": sorted(allowed_full),
        "allowed_actor_final_rows": (
            list(range(16)) if CANDIDATE_STAGE in {"R3", "R4", "R5"} else [12, 13, 14, 15]
        ),
        "changed_full_model_tensors": changed_full,
        "changed_actor_box_row_tensors": changed_box_rows,
        "all_other_tensors_byte_equal_to_candidate_anchor": True,
    }
    if CANDIDATE_STAGE == "R2":
        manifest["all_other_tensors_byte_equal_to_b500"] = True
    return manifest


def validate_candidate_checkpoint_extra(
    checkpoint: Mapping[str, Any],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate the versioned route state and all immutable ancestry bindings."""

    optimizer_names = (
        R3_OPTIMIZER_PARAMETER_NAMES
        if CANDIDATE_STAGE == "R3"
        else R2_OPTIMIZER_PARAMETER_NAMES
    )

    infos = checkpoint.get("infos")
    extra = infos.get("robot_lab_algorithm_checkpoint_state") if isinstance(infos, Mapping) else None
    if not isinstance(extra, Mapping):
        raise RuntimeError("candidate checkpoint lacks robot_lab_algorithm_checkpoint_state")
    if extra.get("schema_version") != 1 or extra.get("algorithm_class") != "VAEPPO":
        raise RuntimeError("candidate checkpoint algorithm extra schema/class mismatch")
    if extra.get("distill_stage") != 2:
        raise RuntimeError("candidate checkpoint is not a Student distillation checkpoint")

    if CANDIDATE_STAGE == "R4":
        r4 = infos.get("highstep_v13_r4")
        if not (
            isinstance(r4, Mapping)
            and r4.get("schema_version") == 1
            and r4.get("stage") == "R4"
            and r4.get("workflow_id") == "highstep_student_recovery_v13_20260713"
            and r4.get("preregistration_path", r4.get("preregistration"))
            == os.path.realpath(R4_PREREGISTRATION_PATH)
            and r4.get("preregistration_sha256") == R4_PREREGISTRATION_SHA256
            and r4.get("anchor_checkpoint") == os.path.realpath(
                audit.CURRENT_STUDENT_CHECKPOINT
            )
            and r4.get("anchor_checkpoint_sha256") == audit.CURRENT_STUDENT_SHA256
            and r4.get("trainable_tensors") == list(R3_OPTIMIZER_PARAMETER_NAMES)
            and isinstance(r4.get("ridge_lambda"), (int, float))
            and math.isfinite(float(r4["ridge_lambda"]))
            and float(r4["ridge_lambda"]) > 0.0
            and isinstance(r4.get("dataset_manifest"), str)
            and os.path.isfile(r4["dataset_manifest"])
            and sha256_file(r4["dataset_manifest"])
            == r4.get("dataset_manifest_sha256")
        ):
            raise RuntimeError("candidate R4 immutable binding mismatch")
        return {
            "stage": "R4",
            "preregistration_sha256": R4_PREREGISTRATION_SHA256,
            "effective_update_count": 0,
            "protected_root_bound": True,
            "b500_anchor_bound": True,
            "teacher_bound": True,
            "optimizer_parameter_names": list(R3_OPTIMIZER_PARAMETER_NAMES),
            "ridge_lambda": float(r4["ridge_lambda"]),
            "dataset_manifest_sha256": r4["dataset_manifest_sha256"],
        }

    if CANDIDATE_STAGE == "R5":
        r5 = infos.get("highstep_v13_r5")
        if not (
            isinstance(r5, Mapping)
            and r5.get("schema_version") == 1
            and r5.get("stage") == "R5"
            and r5.get("workflow_id") == "highstep_student_recovery_v13_20260713"
            and r5.get("preregistration") == os.path.realpath(R5_PREREGISTRATION_PATH)
            and r5.get("preregistration_sha256") == R5_PREREGISTRATION_SHA256
            and r5.get("anchor_checkpoint") == os.path.realpath(
                audit.CURRENT_STUDENT_CHECKPOINT
            )
            and r5.get("anchor_checkpoint_sha256") == audit.CURRENT_STUDENT_SHA256
            and r5.get("trainable_tensors") == list(R5_OPTIMIZER_PARAMETER_NAMES)
            and isinstance(r5.get("effective_epochs"), int)
            and not isinstance(r5.get("effective_epochs"), bool)
            and 0 < int(r5["effective_epochs"]) <= 5
            and isinstance(r5.get("dataset_manifest"), str)
            and os.path.isfile(r5["dataset_manifest"])
            and sha256_file(r5["dataset_manifest"])
            == r5.get("dataset_manifest_sha256")
            and isinstance(r5.get("optimizer_state_dict"), Mapping)
        ):
            raise RuntimeError("candidate R5 immutable binding mismatch")
        return {
            "stage": "R5",
            "preregistration_sha256": R5_PREREGISTRATION_SHA256,
            "effective_update_count": int(r5["effective_epochs"]),
            "protected_root_bound": True,
            "b500_anchor_bound": True,
            "teacher_bound": True,
            "optimizer_parameter_names": list(R5_OPTIMIZER_PARAMETER_NAMES),
            "dataset_manifest_sha256": r5["dataset_manifest_sha256"],
        }

    recovery = extra.get("student_recovery")
    if not isinstance(recovery, Mapping) or recovery.get("schema_version") != 1:
        raise RuntimeError("candidate checkpoint lacks versioned recovery state")
    if recovery.get("stage") != CANDIDATE_STAGE:
        if CANDIDATE_STAGE == "R2":
            raise RuntimeError("candidate checkpoint recovery stage is not R2")
        raise RuntimeError("candidate checkpoint recovery stage is not R3")
    if recovery.get("preregistration_sha256") != CANDIDATE_PREREGISTRATION_SHA256:
        raise RuntimeError("candidate checkpoint preregistration SHA256 mismatch")
    if tuple(recovery.get("optimizer_parameter_names") or ()) != optimizer_names:
        raise RuntimeError("candidate checkpoint optimizer tensor order changed")

    effective_count = recovery.get("effective_update_count")
    if isinstance(effective_count, bool) or not isinstance(effective_count, int):
        raise RuntimeError("candidate R2 effective_update_count is not an integer")
    maximum_count = 100 if CANDIDATE_STAGE == "R3" else 1000
    if effective_count <= 0 or effective_count > maximum_count:
        raise RuntimeError("candidate effective_update_count is outside the route limit")
    if extra.get("student_distill_update_count") != effective_count:
        raise RuntimeError("candidate R2 extra/recovery update counts disagree")

    binding = recovery.get("binding_manifest")
    if not isinstance(binding, Mapping) or binding.get("stage") != CANDIDATE_STAGE:
        raise RuntimeError("candidate checkpoint lacks the route binding manifest")
    expected_bindings = {
        "preregistration_path": os.path.realpath(CANDIDATE_PREREGISTRATION_PATH),
        "preregistration_sha256": CANDIDATE_PREREGISTRATION_SHA256,
        "protected_student_root": os.path.realpath(audit.ROOT_STUDENT_CHECKPOINT),
        "protected_student_root_sha256": audit.ROOT_STUDENT_SHA256,
        "initial_student_checkpoint": os.path.realpath(
            R3_ANCHOR_CHECKPOINT
            if CANDIDATE_STAGE == "R3"
            else audit.CURRENT_STUDENT_CHECKPOINT
        ),
        "initial_student_sha256": (
            R3_ANCHOR_SHA256 if CANDIDATE_STAGE == "R3" else audit.CURRENT_STUDENT_SHA256
        ),
        "teacher_checkpoint": os.path.realpath(audit.TEACHER_CHECKPOINT),
        "teacher_sha256": audit.TEACHER_SHA256,
        "teacher_env_yaml": os.path.realpath(audit.TEACHER_ENV_CONFIG),
        "teacher_env_yaml_sha256": audit.TEACHER_ENV_CONFIG_SHA256,
        "schedule_resume_mode": "preserve",
        "optimizer_parameter_tensor_count": len(optimizer_names),
    }
    if CANDIDATE_STAGE == "R3":
        expected_bindings.update(
            {
                "behavior_reference_b500": os.path.realpath(
                    audit.CURRENT_STUDENT_CHECKPOINT
                ),
                "behavior_reference_b500_sha256": audit.CURRENT_STUDENT_SHA256,
            }
        )
    mismatches = {
        key: {"actual": binding.get(key), "expected": expected}
        for key, expected in expected_bindings.items()
        if binding.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"candidate R2 immutable binding mismatch: {mismatches}")
    if binding.get("effective_update_count") != effective_count:
        raise RuntimeError("candidate R2 binding/effective update counts disagree")
    if tuple(binding.get("optimizer_parameter_names") or ()) != optimizer_names:
        raise RuntimeError("candidate binding optimizer tensor order changed")
    if CANDIDATE_STAGE == "R2":
        if binding.get("trainable_live_tensors") != list(R2_OPTIMIZER_PARAMETER_NAMES[:6]):
            raise RuntimeError("candidate R2 binding trainable live tensors changed")
        if binding.get("materialized_actor_rows") != [12, 13, 14, 15]:
            raise RuntimeError("candidate R2 binding actor box rows changed")
    if binding.get("checkpoint_load_mode") not in {"weights_only", "full"}:
        raise RuntimeError("candidate R2 binding checkpoint load mode is invalid")
    if not isinstance(binding.get("runtime_contract"), Mapping):
        raise RuntimeError("candidate R2 binding runtime contract is missing")

    if CANDIDATE_STAGE == "R2":
        box_weight = recovery.get("box_weight")
        box_bias = recovery.get("box_bias")
        actor_weight = candidate_state.get("actor.6.weight")
        actor_bias = candidate_state.get("actor.6.bias")
        if (
            not isinstance(box_weight, torch.Tensor)
            or not isinstance(box_bias, torch.Tensor)
            or not isinstance(actor_weight, torch.Tensor)
            or not isinstance(actor_bias, torch.Tensor)
            or tuple(box_weight.shape) != tuple(actor_weight[12:16].shape)
            or tuple(box_bias.shape) != tuple(actor_bias[12:16].shape)
            or not torch.equal(box_weight.detach().cpu(), actor_weight[12:16].detach().cpu())
            or not torch.equal(box_bias.detach().cpu(), actor_bias[12:16].detach().cpu())
        ):
            raise RuntimeError("candidate R2 materialized box rows disagree with model_state_dict")

    return {
        "stage": CANDIDATE_STAGE,
        "preregistration_sha256": CANDIDATE_PREREGISTRATION_SHA256,
        "effective_update_count": effective_count,
        "protected_root_bound": True,
        "b500_anchor_bound": True,
        "teacher_bound": True,
        "optimizer_parameter_names": list(optimizer_names),
    }


class CandidateBoundAuditModels(_BASE_BOUND_AUDIT_MODELS):
    """R2 candidate + model_900 read-only anchor + independent frozen Teacher."""

    def __init__(
        self,
        *,
        candidate_checkpoint: Path | None = None,
        candidate_sha256: str | None = None,
        device: str | torch.device = "cpu",
    ):
        if candidate_checkpoint is None or candidate_sha256 is None:
            candidate_binding = resolve_candidate_binding()
        else:
            candidate_binding = resolve_candidate_binding(
                {
                    CANDIDATE_CHECKPOINT_ENV: str(candidate_checkpoint),
                    CANDIDATE_SHA256_ENV: str(candidate_sha256),
                }
            )

        # This validates model_900, qualified B500, frozen Teacher and their
        # evidence before the B500 driver modules are replaced by the candidate.
        super().__init__(device=device)
        b500_binding_manifest = copy.deepcopy(self.binding_manifest)

        candidate, actual_candidate_sha = audit._load_checkpoint(
            candidate_binding.checkpoint, candidate_binding.sha256
        )
        candidate_anchor_path = (
            R3_ANCHOR_CHECKPOINT
            if CANDIDATE_STAGE == "R3"
            else audit.CURRENT_STUDENT_CHECKPOINT
        )
        candidate_anchor_sha = (
            R3_ANCHOR_SHA256
            if CANDIDATE_STAGE == "R3"
            else audit.CURRENT_STUDENT_SHA256
        )
        b500, actual_b500_sha = audit._load_checkpoint(
            candidate_anchor_path, candidate_anchor_sha
        )
        candidate_state = candidate.get("model_state_dict")
        b500_state = b500.get("model_state_dict")
        if not isinstance(candidate_state, Mapping) or not isinstance(b500_state, Mapping):
            raise RuntimeError("candidate or B500 checkpoint lacks model_state_dict")
        delta_manifest = validate_candidate_tensor_deltas(candidate_state, b500_state)
        extra_manifest = validate_candidate_checkpoint_extra(candidate, candidate_state)

        candidate_actor_state = audit._component_state(candidate_state, "actor.")
        candidate_estimator_state = audit._component_state(candidate_state, "estimator.")
        candidate_priv_state = audit._component_state(candidate_state, "priv_encoder.")
        self.student_actor = audit.CheckpointMLP(candidate_actor_state)
        self.student_estimator = audit.StudentEstimator(candidate_estimator_state)
        self.student_privileged_encoder = audit.CheckpointMLP(
            audit._component_state(candidate_state, "priv_encoder.net."), final_tanh=True
        )
        self.to(device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self._validate_dimensions()

        candidate_pointers = self._module_storage_pointers(
            self.student_actor, self.student_estimator, self.student_privileged_encoder
        )
        anchors_and_teacher_pointers = self._module_storage_pointers(
            self.root_student_actor,
            self.root_student_estimator,
            self.root_student_privileged_encoder,
            self.teacher_actor,
            self.teacher_privileged_encoder,
        )
        if candidate_pointers & anchors_and_teacher_pointers:
            raise RuntimeError("candidate unexpectedly shares tensor storage with an anchor or Teacher")

        candidate_actor_sha = audit._component_sha256(candidate_actor_state)
        candidate_estimator_sha = audit._component_sha256(candidate_estimator_state)
        candidate_priv_sha = audit._component_sha256(candidate_priv_state)
        self.binding_manifest = {
            "schema_version": 2,
            "student_checkpoint_role": f"behavior-gated {CANDIDATE_STAGE} candidate driver",
            "student_checkpoint": str(candidate_binding.checkpoint),
            "student_checkpoint_sha256": actual_candidate_sha,
            "student_checkpoint_iteration": int(candidate.get("iter", -1)),
            "student_effective_update_count": extra_manifest["effective_update_count"],
            "student_actor_component_sha256": candidate_actor_sha,
            "student_estimator_component_sha256": candidate_estimator_sha,
            "student_privileged_encoder_component_sha256": candidate_priv_sha,
            "candidate_r2_extra_binding": extra_manifest,
            "candidate_vs_b500_tensor_delta_audit": delta_manifest,
            "b500_anchor_checkpoint": str(candidate_anchor_path.resolve()),
            "b500_anchor_sha256": actual_b500_sha,
            "b500_anchor_binding_verified": True,
            "protected_root_checkpoint": str(audit.ROOT_STUDENT_CHECKPOINT.resolve()),
            "protected_root_sha256": audit.ROOT_STUDENT_SHA256,
            "protected_root_binding_verified": True,
            "teacher_checkpoint": str(audit.TEACHER_CHECKPOINT.resolve()),
            "teacher_checkpoint_sha256": audit.TEACHER_SHA256,
            "teacher_actor_component_sha256": audit.TEACHER_ACTOR_SHA256,
            "teacher_privileged_encoder_component_sha256": (
                audit.TEACHER_PRIVILEGED_ENCODER_SHA256
            ),
            "teacher_binding_verified": True,
            "r2_preregistration_path": str(CANDIDATE_PREREGISTRATION_PATH.resolve()),
            "r2_preregistration_sha256": CANDIDATE_PREREGISTRATION_SHA256,
            "student_teacher_paths_distinct": (
                candidate_binding.checkpoint.resolve() != audit.TEACHER_CHECKPOINT.resolve()
            ),
            "student_teacher_storage_independent": True,
            "student_anchor_storage_independent": True,
            "all_parameters_frozen": all(not parameter.requires_grad for parameter in self.parameters()),
            "student_policy_observation_dim": 570,
            "student_estimator_observation_dim": 570,
            "teacher_critic_observation_dim": 162,
            "teacher_privileged_input_dim": 159,
            "latent_dim": 64,
            "action_dim": 16,
            "empirical_normalization": False,
            "qualified_b500_reference_binding": b500_binding_manifest,
        }


def candidate_suite_plan_payload(
    binding: CandidateBinding,
    *,
    base_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a base3-identity plan without B500 behavior outcome assertions."""

    _assert_frozen_sources()
    source = copy.deepcopy(
        audit.base_suite_plan_payload() if base_payload is None else dict(base_payload)
    )
    trajectories = [
        {
            "run_id": trajectory["run_id"],
            "seed": trajectory["seed"],
            "scenario": trajectory["scenario"],
            "lateral_offset_m": trajectory["lateral_offset_m"],
            "yaw_offset_deg": trajectory["yaw_offset_deg"],
            "requested_play_max_steps": MAX_STEPS,
            "termination_allowed": False,
            "behavior_outcome": "observed_not_preregistered",
        }
        for trajectory in audit.DEFAULT_AUDIT_TRAJECTORIES
    ]
    return {
        "schema_version": 1,
        "kind": "highstep_candidate_same_state_base3_plan",
        "immutable_after_write": True,
        "candidate_checkpoint": str(binding.checkpoint.resolve()),
        "candidate_sha256": binding.sha256,
        "candidate_required_stage": CANDIDATE_STAGE,
        "candidate_required_preregistration_sha256": CANDIDATE_PREREGISTRATION_SHA256,
        "reference_b500_checkpoint": str(audit.CURRENT_STUDENT_CHECKPOINT.resolve()),
        "reference_b500_sha256": audit.CURRENT_STUDENT_SHA256,
        "protected_root_checkpoint": str(audit.ROOT_STUDENT_CHECKPOINT.resolve()),
        "protected_root_sha256": audit.ROOT_STUDENT_SHA256,
        "teacher_checkpoint": str(audit.TEACHER_CHECKPOINT.resolve()),
        "teacher_sha256": audit.TEACHER_SHA256,
        "runtime_contract": source["runtime_contract"],
        "trajectories": trajectories,
        "candidate_behavior_outcomes_are_not_preregistered": True,
        "required_evidence": {
            "frame_count_per_trajectory": MAX_STEPS,
            "termination_allowed": False,
            "finite_all_frames": True,
            "runner_manual_parity_all_frames": True,
            "physical_and_input_digest_unchanged_all_frames": True,
            "decomposition_residual_max_abs": 1.0e-6,
        },
        "base_runtime_path": str(BASE_RUNTIME_PATH),
        "base_runtime_sha256": BASE_RUNTIME_SHA256,
        "audit_tool_path": str(AUDIT_TOOL_PATH),
        "audit_tool_sha256": AUDIT_TOOL_SHA256,
        "training_allowed": False,
    }


def _atomic_read_only_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def ensure_candidate_suite_plan(
    output_root: Path,
    binding: CandidateBinding,
    *,
    expected_payload: Mapping[str, Any] | None = None,
) -> tuple[Path, str]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "candidate_base3_suite_plan.json"
    expected = dict(expected_payload or candidate_suite_plan_payload(binding))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"existing candidate suite plan differs from immutable plan: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"existing candidate suite plan is writable: {path}")
    else:
        _atomic_read_only_json(path, expected)
    return path, sha256_file(path)


def candidate_runtime_binding_payload(
    binding: CandidateBinding,
    *,
    suite_plan_path: Path,
    suite_plan_sha256: str,
    runtime_code_hashes: Mapping[str, str],
    play_contract_nodes: Sequence[str],
    play_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "highstep_candidate_same_state_base3_runtime_binding",
        "immutable_after_write": True,
        "suite_plan_path": str(suite_plan_path.resolve()),
        "suite_plan_sha256": suite_plan_sha256,
        "candidate_checkpoint": str(binding.checkpoint.resolve()),
        "candidate_sha256": binding.sha256,
        "candidate_required_stage": CANDIDATE_STAGE,
        "r2_preregistration_path": str(CANDIDATE_PREREGISTRATION_PATH.resolve()),
        "r2_preregistration_sha256": CANDIDATE_PREREGISTRATION_SHA256,
        "base_runtime_path": str(BASE_RUNTIME_PATH),
        "base_runtime_sha256": BASE_RUNTIME_SHA256,
        "audit_tool_path": str(AUDIT_TOOL_PATH),
        "audit_tool_sha256": AUDIT_TOOL_SHA256,
        "runtime_code_and_config_sha256": dict(runtime_code_hashes),
        "core9_play_contract_nodes": list(play_contract_nodes),
        "core9_play_sha256": play_sha256,
    }


def _load_base_runtime() -> Any:
    _assert_frozen_sources()
    module_name = "highstep_candidate_reused_base_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, BASE_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base same-state runtime: {BASE_RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_candidate_adapter(binding: CandidateBinding) -> Any:
    """Inject only candidate binding/plan hooks into the frozen base runtime."""

    global _ACTIVE_BINDING, _ACTIVE_BASE_RUNTIME
    if _ACTIVE_BINDING is not None:
        if _ACTIVE_BINDING != binding:
            raise RuntimeError("candidate adapter is already bound to a different checkpoint")
        return _ACTIVE_BASE_RUNTIME

    base_runtime = _load_base_runtime()
    base_plan_function = audit.base_suite_plan_payload
    candidate_plan = candidate_suite_plan_payload(
        binding, base_payload=base_plan_function()
    )
    original_runtime_code_hashes = base_runtime._runtime_code_hashes

    def candidate_plan_function() -> dict[str, Any]:
        return copy.deepcopy(candidate_plan)

    def runtime_code_hashes() -> dict[str, str]:
        result = dict(original_runtime_code_hashes())
        result[str(Path(__file__).resolve())] = sha256_file(Path(__file__).resolve())
        return result

    def ensure_plan(output_root: Path, audit_module: Any) -> tuple[Path, str]:
        if audit_module is not audit:
            raise RuntimeError("candidate adapter received an unexpected audit module")
        return ensure_candidate_suite_plan(
            output_root, binding, expected_payload=audit_module.base_suite_plan_payload()
        )

    def ensure_runtime_binding(
        output_root: Path,
        suite_plan_path: Path,
        suite_plan_sha256: str,
    ) -> tuple[Path, str]:
        path = output_root.expanduser().resolve() / "candidate_base3_runtime_binding.json"
        payload = candidate_runtime_binding_payload(
            binding,
            suite_plan_path=suite_plan_path,
            suite_plan_sha256=suite_plan_sha256,
            runtime_code_hashes=runtime_code_hashes(),
            play_contract_nodes=base_runtime.PLAY_CONTRACT_NODES,
            play_sha256=base_runtime.PLAY_SHA256,
        )
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise RuntimeError(f"candidate runtime binding changed between trajectories: {path}")
            if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise RuntimeError(f"candidate runtime binding is writable: {path}")
        else:
            _atomic_read_only_json(path, payload)
        return path, sha256_file(path)

    audit.base_suite_plan_payload = candidate_plan_function
    audit.BoundAuditModels = CandidateBoundAuditModels
    base_runtime.B500_CHECKPOINT = binding.checkpoint
    base_runtime.B500_SHA256 = binding.sha256
    base_runtime._ensure_base_suite_plan = ensure_plan
    base_runtime._runtime_code_hashes = runtime_code_hashes
    base_runtime._ensure_runtime_binding = ensure_runtime_binding

    _ACTIVE_BINDING = binding
    _ACTIVE_BASE_RUNTIME = base_runtime
    return base_runtime


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def validate_completed_candidate_run(
    output_root: Path,
    trajectory: Any,
    binding: CandidateBinding,
) -> dict[str, Any]:
    """Reject any trace that is short, terminated, non-finite or loses parity."""

    output_dir = output_root.expanduser().resolve() / trajectory.run_id
    summary_path = output_dir / "run_summary.json"
    summary = json.loads(
        summary_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
    )
    outcome = summary.get("outcome")
    checkpoint_binding = summary.get("checkpoint_binding")
    if not isinstance(outcome, Mapping) or not isinstance(checkpoint_binding, Mapping):
        raise RuntimeError("candidate run summary lacks outcome or checkpoint binding")
    failures = []
    if summary.get("frame_count") != MAX_STEPS:
        failures.append(f"frame_count={summary.get('frame_count')}")
    if outcome.get("loop_steps") != MAX_STEPS:
        failures.append(f"loop_steps={outcome.get('loop_steps')}")
    if outcome.get("requested_play_max_steps") != MAX_STEPS:
        failures.append(f"requested_play_max_steps={outcome.get('requested_play_max_steps')}")
    if outcome.get("terminated") is not False:
        failures.append(f"terminated={outcome.get('terminated')}")
    if summary.get("runner_manual_action_allclose_all_frames") is not True:
        failures.append("runner/manual parity")
    if summary.get("all_forward_state_digests_unchanged") is not True:
        failures.append("physical digest")
    if summary.get("all_observation_and_prior_inputs_unchanged") is not True:
        failures.append("observation/prior digest")
    if float(summary.get("action_decomposition_residual_global_max_abs", 1.0)) > 1.0e-6:
        failures.append("action decomposition")
    if checkpoint_binding.get("student_checkpoint_sha256") != binding.sha256:
        failures.append("candidate checkpoint binding")
    candidate_extra = checkpoint_binding.get("candidate_r2_extra_binding")
    if not isinstance(candidate_extra, Mapping) or candidate_extra.get("stage") != CANDIDATE_STAGE:
        failures.append("candidate route extra binding")

    frames_path = Path(summary.get("frames_path", ""))
    frame_count = 0
    with frames_path.resolve(strict=True).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line, parse_constant=_reject_json_constant)
            frame_count += 1
            if record.get("runner_manual_action_allclose") is not True:
                failures.append(f"frame {frame_count - 1} runner/manual parity")
            if record.get("physical_state_unchanged_by_forward") is not True:
                failures.append(f"frame {frame_count - 1} physical digest")
            if record.get("observation_and_prior_inputs_unchanged_by_forward") is not True:
                failures.append(f"frame {frame_count - 1} observation/prior digest")
            if float(record.get("action_decomposition_residual_max_abs", 1.0)) > 1.0e-6:
                failures.append(f"frame {frame_count - 1} action decomposition")
            if record.get("causal_replacement_applied") is not False:
                failures.append(f"frame {frame_count - 1} causal replacement")
    if frame_count != MAX_STEPS:
        failures.append(f"frames.jsonl lines={frame_count}")
    if failures:
        raise RuntimeError("candidate same-state trace failed: " + "; ".join(failures[:20]))
    return {
        "run_id": trajectory.run_id,
        "candidate_checkpoint": str(binding.checkpoint),
        "candidate_sha256": binding.sha256,
        "frame_count": frame_count,
        "terminated": False,
        "finite_parity_digest_decomposition_verified": True,
        "full_climb_success_observed": outcome.get("full_climb_success"),
        "rear_hold_success_observed": outcome.get("rear_on_platform_hold_success"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    binding = resolve_candidate_binding()
    base_runtime = install_candidate_adapter(binding)

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
            raise RuntimeError(f"base same-state runtime returned {result}")
        validation = validate_completed_candidate_run(args.output_root, trajectory, binding)
        print(
            "[HIGHSTEP_CANDIDATE_SAME_STATE_AUDIT_JSON] "
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
