# Copyright (c) 2024-2025 ArcLab
# SPDX-License-Identifier: Apache-2.0

"""Process-independent high-step schedule bookkeeping.

Isaac Lab's ``common_step_counter`` starts from zero in every process.  It is
therefore suitable for local periodic work, but it is not a checkpoint clock.
This module anchors that local counter to an explicit training/evaluation
update so action-prior and curriculum schedules survive process restarts.

The module intentionally has no Isaac Lab or torch imports.  This keeps the
clock/provenance logic testable with plain Python and lets launch scripts use
the same implementation as environment terms.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEDULE_MANIFEST_NAME = "highstep_schedule_manifest.json"
RUNTIME_STATE_NAME = "highstep_runtime_state.json"
_BASE_UPDATE_ATTR = "_highstep_schedule_base_update"
_ANCHOR_STEP_ATTR = "_highstep_schedule_anchor_step"
_SOURCE_ATTR = "_highstep_schedule_source"
_MODE_ATTR = "_highstep_schedule_resume_mode"
_RUNNER_ITERATION_ATTR = "_highstep_schedule_runner_iteration_at_anchor"
_ADVANCE_ATTR = "_highstep_schedule_advance_with_local_steps"


class ScheduleContinuityError(RuntimeError):
    """Base error for schedule provenance that cannot be trusted."""


class ScheduleManifestMissingError(ScheduleContinuityError):
    """Raised when a checkpoint-continuous schedule sidecar is required but absent."""


class ScheduleManifestInvalidError(ScheduleContinuityError):
    """Raised when a present schedule sidecar cannot be trusted."""


_ROBUST_TEACHER_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0"
)
_STANDARD_TEACHER_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0"
)
_ROBUST_STUDENT_TASKS = frozenset(
    {
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-ArcdogAdjustableLeg-v0",
        "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR2-ArcdogAdjustableLeg-v0",
    }
)
_CORE_WORKFLOW_ID = "highstep_real_climb_core_20260712"
_STUDENT_LINEAGE_KEYS = {
    "parent_teacher_manifest_path",
    "parent_teacher_manifest_sha256",
    "selected_teacher_checkpoint_path",
    "selected_teacher_checkpoint_sha256",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of one regular file."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ScheduleManifestInvalidError(f"SHA-256 source is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ScheduleManifestInvalidError(f"Cannot hash SHA-256 source {source}: {error}") from error
    return digest.hexdigest()


def _resolved_regular_file(path: str | os.PathLike[str], *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ScheduleManifestInvalidError(f"{label} is not a regular file: {resolved}")
    return resolved


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_parent_teacher_lineage(
    parent_manifest_path: str | os.PathLike[str], *, require_robust: bool = False
) -> dict[str, str]:
    """Validate a released Teacher manifest and pin its selected checkpoint.

    The current real-robot route requires Robust Student to descend from the
    isolated real-gain Robust-Teacher core9 gate.  The older 18-run Robust
    release remains accepted so historical Robust-Student lineages stay
    readable.  Standard-Student ancestry remains readable but is not a release
    route in the current automation policy.
    """
    parent_path = _resolved_regular_file(parent_manifest_path, label="Parent Teacher manifest")
    try:
        parent_bytes = parent_path.read_bytes()
        parent = json.loads(parent_bytes.decode("utf-8"))
        selected_path = _resolved_regular_file(
            str(parent["selected_checkpoint"]), label="Selected Teacher checkpoint"
        )
        summaries = parent["checkpoint_summaries"]
        selected_matches = [
            summary
            for summary in summaries
            if isinstance(summary, Mapping)
            and Path(str(summary.get("checkpoint"))).expanduser().resolve() == selected_path
        ]
        if len(selected_matches) != 1:
            raise ValueError("selected checkpoint must have exactly one summary")
        selected_summary = selected_matches[0]
    except (KeyError, OSError, UnicodeError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ScheduleManifestInvalidError(
            f"Cannot resolve parent Teacher lineage from {parent_path}: {error}"
        ) from error

    selected_sha = sha256_file(selected_path)
    summary_sha = selected_summary.get("checkpoint_sha256")
    common_valid = bool(
        isinstance(parent, Mapping)
        and parent.get("schema_version") == 4
        and isinstance(parent.get("evaluation_payload_schema_version"), int)
        and not isinstance(parent.get("evaluation_payload_schema_version"), bool)
        and int(parent["evaluation_payload_schema_version"]) >= 6
        and parent.get("evaluation_complete") is True
        and parent.get("matrix_complete") is True
        and _valid_sha256(summary_sha)
        and summary_sha == selected_sha
        and isinstance(selected_summary.get("valid_count"), (int, float))
        and not isinstance(selected_summary.get("valid_count"), bool)
        and isinstance(selected_summary.get("pass_rate"), (int, float))
        and not isinstance(selected_summary.get("pass_rate"), bool)
    )
    core_teacher_valid = bool(
        common_valid
        and parent.get("workflow_id") == _CORE_WORKFLOW_ID
        and parent.get("evaluation_profile") == "core9"
        and parent.get("role") == "teacher"
        and parent.get("task") == _STANDARD_TEACHER_TASK
        and parent.get("decision")
        in {"teacher_candidate_for_student", "teacher_candidate_requires_real_gain_validation"}
        and float(selected_summary["valid_count"]) == 9.0
        and float(selected_summary["pass_rate"]) >= 0.60
    )
    core_robust_valid = bool(
        common_valid
        and parent.get("workflow_id") == _CORE_WORKFLOW_ID
        and parent.get("evaluation_profile") == "core9"
        and parent.get("role") == "teacher_robust"
        and parent.get("task") == _ROBUST_TEACHER_TASK
        and parent.get("decision") == "robust_teacher_candidate_for_student"
        and float(selected_summary["valid_count"]) == 9.0
        and float(selected_summary["pass_rate"]) >= 0.60
    )
    historical_robust_valid = bool(
        common_valid
        and parent.get("role") == "teacher_robust"
        and parent.get("task") == _ROBUST_TEACHER_TASK
        and parent.get("decision") == "robust_teacher_candidate_for_student"
        and float(selected_summary["valid_count"]) == 18.0
        and float(selected_summary["pass_rate"]) >= 0.80
    )
    robust_valid = core_robust_valid or historical_robust_valid
    release_valid = robust_valid if require_robust else (core_teacher_valid or robust_valid)
    if not release_valid:
        raise ScheduleManifestInvalidError(
            (
                "Robust Student requires a current core9 or historical Robust-Teacher release: "
                if require_robust
                else "Parent Teacher manifest is not a recognized standard or Robust release: "
            )
            + str(parent_path)
        )
    return {
        "parent_teacher_manifest_path": str(parent_path),
        "parent_teacher_manifest_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "selected_teacher_checkpoint_path": str(selected_path),
        "selected_teacher_checkpoint_sha256": selected_sha,
    }


def _normalize_student_parent_lineage(
    lineage: Mapping[str, Any], *, require_robust: bool = False
) -> dict[str, str]:
    if set(lineage) != _STUDENT_LINEAGE_KEYS:
        raise ScheduleManifestInvalidError(
            "Student parent lineage must contain exactly " + ", ".join(sorted(_STUDENT_LINEAGE_KEYS))
        )
    parent_path = _resolved_regular_file(
        str(lineage["parent_teacher_manifest_path"]), label="Inherited parent Teacher manifest"
    )
    checkpoint_path = _resolved_regular_file(
        str(lineage["selected_teacher_checkpoint_path"]), label="Inherited selected Teacher checkpoint"
    )
    normalized = {
        "parent_teacher_manifest_path": str(parent_path),
        "parent_teacher_manifest_sha256": str(lineage["parent_teacher_manifest_sha256"]),
        "selected_teacher_checkpoint_path": str(checkpoint_path),
        "selected_teacher_checkpoint_sha256": str(lineage["selected_teacher_checkpoint_sha256"]),
    }
    if not all(
        _valid_sha256(normalized[key])
        for key in _STUDENT_LINEAGE_KEYS
        if key.endswith("sha256")
    ):
        raise ScheduleManifestInvalidError("Inherited Student lineage contains an invalid SHA-256")
    current = _load_parent_teacher_lineage(parent_path, require_robust=require_robust)
    if normalized != current:
        raise ScheduleContinuityError(
            "Student parent Teacher lineage changed since the source training manifest was written"
        )
    return normalized


def resolve_student_parent_lineage(
    *,
    task: str | None,
    checkpoint_path: str | os.PathLike[str] | None,
    parent_teacher_manifest_path: str | os.PathLike[str] | None,
    source_manifest: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Pin or inherit immutable Student->Teacher ancestry."""
    is_student = "student" in str(task or "").lower()
    require_robust = str(task or "") in _ROBUST_STUDENT_TASKS
    if not is_student:
        if parent_teacher_manifest_path:
            raise ScheduleContinuityError(
                "--highstep_parent_teacher_manifest is only valid for a Student task"
            )
        return None
    if not checkpoint_path:
        raise ScheduleContinuityError("Student training requires a checkpoint with pinned Teacher ancestry")

    inherited = source_manifest.get("student_parent_lineage") if isinstance(source_manifest, Mapping) else None
    source_is_student = "student" in str((source_manifest or {}).get("task") or "").lower()
    if source_is_student:
        if not isinstance(inherited, Mapping):
            raise ScheduleManifestInvalidError(
                "Source Student schedule manifest has no student_parent_lineage; ancestry cannot be reconstructed"
            )
        lineage = _normalize_student_parent_lineage(inherited, require_robust=require_robust)
        if parent_teacher_manifest_path:
            explicit = _load_parent_teacher_lineage(
                parent_teacher_manifest_path, require_robust=require_robust
            )
            if explicit != lineage:
                raise ScheduleContinuityError(
                    "Explicit parent Teacher manifest does not match inherited Student ancestry"
                )
        return lineage

    if not parent_teacher_manifest_path:
        raise ScheduleContinuityError(
            "The first Student training run requires --highstep_parent_teacher_manifest"
        )
    lineage = _load_parent_teacher_lineage(
        parent_teacher_manifest_path, require_robust=require_robust
    )
    loaded_checkpoint = _resolved_regular_file(checkpoint_path, label="Initial Student source checkpoint")
    loaded_sha = sha256_file(loaded_checkpoint)
    exact_parent_path = str(loaded_checkpoint) == lineage["selected_teacher_checkpoint_path"]
    reconstruction = (
        source_manifest.get("reconstruction", {})
        if isinstance(source_manifest, Mapping) else {}
    )
    verified_v15_mirror = bool(
        not exact_parent_path
        and (source_manifest or {}).get("kind")
        == "highstep_v15_reconstructed_train_source_schedule"
        and reconstruction.get("canonical_teacher_checkpoint")
        == lineage["selected_teacher_checkpoint_path"]
        and reconstruction.get("canonical_teacher_checkpoint_sha256")
        == lineage["selected_teacher_checkpoint_sha256"]
        and reconstruction.get("old_teacher_run_modified") is False
    )
    if (
        (not exact_parent_path and not verified_v15_mirror)
        or loaded_sha != lineage["selected_teacher_checkpoint_sha256"]
    ):
        raise ScheduleContinuityError(
            "Initial Student training checkpoint is not the parent manifest's selected Teacher checkpoint"
        )
    return lineage


def unwrap_env(env: Any) -> Any:
    """Return the innermost environment without depending on wrapper types."""
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            current = unwrapped
            continue
        nested = getattr(current, "env", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        break
    return current


def _finite_nonnegative(value: float | int, name: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return parsed


def install_global_update(
    env: Any,
    schedule_update: float | int,
    *,
    source: str,
    resume_mode: str,
    runner_iteration: int,
    advance_with_local_steps: bool = True,
) -> Any:
    """Anchor ``env.common_step_counter`` to an explicit global update.

    The anchor is installed after ``runner.load``.  Subsequent local control
    steps advance the schedule normally; resetting/restarting the process no
    longer rewinds it to update zero.
    """
    target = unwrap_env(env)
    base_update = _finite_nonnegative(schedule_update, "schedule_update")
    local_step = _finite_nonnegative(getattr(target, "common_step_counter", 0), "common_step_counter")
    setattr(target, _BASE_UPDATE_ATTR, base_update)
    setattr(target, _ANCHOR_STEP_ATTR, local_step)
    setattr(target, _SOURCE_ATTR, str(source))
    setattr(target, _MODE_ATTR, str(resume_mode))
    setattr(target, _RUNNER_ITERATION_ATTR, int(runner_iteration))
    setattr(target, _ADVANCE_ATTR, bool(advance_with_local_steps))
    return target


def global_update(env: Any, num_steps_per_update: int | float) -> float:
    """Return the effective checkpoint-continuous learning update."""
    steps_per_update = float(num_steps_per_update)
    if steps_per_update <= 0.0:
        raise ValueError("num_steps_per_update must be positive")
    target = unwrap_env(env)
    local_step = float(getattr(target, "common_step_counter", 0))
    if not hasattr(target, _BASE_UPDATE_ATTR):
        return local_step / steps_per_update
    base_update = float(getattr(target, _BASE_UPDATE_ATTR))
    if not bool(getattr(target, _ADVANCE_ATTR, True)):
        return base_update
    anchor_step = float(getattr(target, _ANCHOR_STEP_ATTR, 0.0))
    # A task reset should not normally rewind common_step_counter.  Clamp the
    # delta defensively so an unexpected wrapper reset cannot silently rewind
    # every high-step schedule.
    local_delta = max(0.0, local_step - anchor_step)
    return base_update + local_delta / steps_per_update


def global_step(env: Any, num_steps_per_update: int | float) -> float:
    """Return the effective global control step for update-based schedules."""
    return global_update(env, num_steps_per_update) * float(num_steps_per_update)


def schedule_clock_state(env: Any, num_steps_per_update: int | float) -> dict[str, Any]:
    """Return the live clock state, independently of a previously written manifest."""
    target = unwrap_env(env)
    return {
        "global_update": global_update(target, num_steps_per_update),
        "advance_with_local_steps": bool(getattr(target, _ADVANCE_ATTR, True)),
        "source": getattr(target, _SOURCE_ATTR, None),
        "resume_mode": getattr(target, _MODE_ATTR, None),
        "runner_iteration_at_anchor": getattr(target, _RUNNER_ITERATION_ATTR, None),
        "local_common_step": int(getattr(target, "common_step_counter", 0)),
    }


def linear_blend(update: float, start_update: float, ramp_updates: float) -> float:
    """Return a clipped [0, 1] linear schedule value."""
    ramp = max(float(ramp_updates), 1.0)
    return max(0.0, min(1.0, (float(update) - float(start_update)) / ramp))


def prior_scale(update: float, start_update: float, full_update: float) -> float:
    """Return the action-prior scale for start/full update endpoints."""
    return linear_blend(update, start_update, float(full_update) - float(start_update))


def stage_index(update: float, thresholds: Sequence[int | float] | None) -> int:
    """Return the number of crossed stage thresholds."""
    if thresholds is None:
        return 0
    return sum(float(update) >= float(threshold) for threshold in thresholds)


def allowed_max_level(
    update: float,
    thresholds: Sequence[int | float] | None,
    max_levels: Sequence[int] | None,
) -> int | None:
    """Return the curriculum ceiling selected by a staged terrain schedule."""
    if thresholds is None or max_levels is None:
        return None
    if len(max_levels) != len(thresholds) + 1:
        raise ValueError("stage_max_levels must contain one more value than stage_update_thresholds")
    return int(max_levels[min(stage_index(update, thresholds), len(max_levels) - 1)])


def resolve_schedule_resume_mode(
    *,
    checkpoint_load_mode: str,
    requested_mode: str,
    highstep_resume_kind: str | None,
) -> str:
    """Resolve an explicit preserve/reset decision for a loaded checkpoint.

    A weights-only load intentionally decouples optimizer/runner iteration from
    the policy weights.  Its schedule decision must therefore never be inferred.
    For a full load, same-task refinement preserves the schedule while a
    cross-task migration starts a new high-step schedule.
    """
    if checkpoint_load_mode not in {"full", "weights_only"}:
        raise ValueError(f"Unsupported checkpoint load mode: {checkpoint_load_mode!r}")
    if requested_mode not in {"auto", "preserve", "reset"}:
        raise ValueError(f"Unsupported schedule resume mode: {requested_mode!r}")
    if checkpoint_load_mode == "weights_only" and requested_mode == "auto":
        raise ValueError(
            "weights_only requires an explicit high-step schedule decision: "
            "use --highstep_schedule_resume_mode=preserve or =reset"
        )
    if requested_mode != "auto":
        return requested_mode
    return "preserve" if highstep_resume_kind == "refine" else "reset"


def checkpoint_iteration_from_mapping(checkpoint: Mapping[str, Any]) -> int:
    """Read the runner iteration from a loaded RSL-RL checkpoint mapping."""
    for key in ("iter", "iteration", "current_learning_iteration"):
        if key not in checkpoint:
            continue
        value = checkpoint[key]
        if hasattr(value, "item"):
            value = value.item()
        iteration = int(value)
        if iteration < 0:
            raise ValueError(f"Checkpoint iteration must be non-negative, got {iteration}")
        return iteration
    raise KeyError("Checkpoint does not contain iter/iteration/current_learning_iteration")


def _manifest_candidates(checkpoint_path: str | os.PathLike[str]) -> tuple[Path, ...]:
    run_dir = Path(checkpoint_path).expanduser().resolve().parent
    return (
        run_dir / "params" / SCHEDULE_MANIFEST_NAME,
        run_dir / SCHEDULE_MANIFEST_NAME,
    )


def _runtime_state_candidates(checkpoint_path: str | os.PathLike[str]) -> tuple[Path, ...]:
    run_dir = Path(checkpoint_path).expanduser().resolve().parent
    return (
        run_dir / "params" / RUNTIME_STATE_NAME,
        run_dir / RUNTIME_STATE_NAME,
    )


def _normalized_sequence(value: Any) -> list[Any] | None:
    if value is None:
        return None
    return list(value)


def checkpoint_sha256(checkpoint_path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of a checkpoint, failing closed on an invalid path."""
    return sha256_file(checkpoint_path)


def schedule_definition_from_configs(
    *,
    action_cfg: Any,
    terrain_term_cfg: Any,
    support_term_cfg: Any,
    command_term_cfg: Any = None,
    command_curriculum_enabled: bool = False,
    reward_stage_definition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the static schedule contract that must match on preserve."""
    action_enabled = all(hasattr(action_cfg, name) for name in ("prior_start_update", "prior_full_update"))
    terrain_params = _term_params(terrain_term_cfg)
    support_params = _term_params(support_term_cfg) or terrain_params
    command_params = _term_params(command_term_cfg)
    return {
        "action_prior": {
            "enabled": bool(action_enabled),
            "num_steps_per_update": int(getattr(action_cfg, "num_steps_per_update", 24) or 24),
            "start_update": int(getattr(action_cfg, "prior_start_update")) if action_enabled else None,
            "full_update": int(getattr(action_cfg, "prior_full_update")) if action_enabled else None,
        },
        "terrain_schedule": {
            "num_steps_per_update": int(terrain_params.get("num_steps_per_update", 24) or 24),
            "stage_update_thresholds": _normalized_sequence(terrain_params.get("stage_update_thresholds")),
            "stage_max_levels": _normalized_sequence(terrain_params.get("stage_max_levels")),
            "score_warmup_updates": terrain_params.get("score_warmup_updates"),
            "score_stage_update_thresholds": _normalized_sequence(
                terrain_params.get("score_stage_update_thresholds")
            ),
            "action_score_thresholds": _normalized_sequence(terrain_params.get("action_score_thresholds")),
            "support_score_thresholds": _normalized_sequence(terrain_params.get("support_score_thresholds")),
            "support_bottleneck_start_update": terrain_params.get("support_bottleneck_start_update"),
            "support_bottleneck_ramp_updates": terrain_params.get("support_bottleneck_ramp_updates"),
            "support_bottleneck_warmup_min_gate": terrain_params.get(
                "support_bottleneck_warmup_min_gate"
            ),
        },
        "support_bottleneck": {
            "metric_enabled": support_term_cfg is not None,
            "num_steps_per_update": int(support_params.get("num_steps_per_update", 24) or 24),
            "start_update": float(support_params.get("support_bottleneck_start_update", 0)),
            "ramp_updates": float(support_params.get("support_bottleneck_ramp_updates", 1)),
            "warmup_min_gate": float(support_params.get("support_bottleneck_warmup_min_gate", 0.0)),
        },
        "command_curriculum": {
            "enabled": bool(command_curriculum_enabled),
            "range_multiplier": _normalized_sequence(command_params.get("range_multiplier")),
            "gated_multiplier": command_params.get("gated_multiplier"),
            "terrain_gate_level": command_params.get("terrain_gate_level"),
            "delta": command_params.get("delta"),
            "reward_threshold": command_params.get("reward_threshold"),
        },
        "reward_stages": dict(reward_stage_definition or {}),
    }


def assert_schedule_definition_compatible(
    source_manifest: Mapping[str, Any],
    current_definition: Mapping[str, Any],
) -> None:
    """Fail closed when preserve would silently change schedule endpoints."""
    source_definition = source_manifest.get("schedule_definition")
    if not isinstance(source_definition, Mapping):
        raise ScheduleManifestInvalidError("Source manifest has no schedule_definition mapping")
    source_json = json.dumps(dict(source_definition), sort_keys=True, separators=(",", ":"))
    current_json = json.dumps(dict(current_definition), sort_keys=True, separators=(",", ":"))
    if source_json != current_json:
        raise ScheduleContinuityError(
            "High-step schedule definition changed across a preserve resume. "
            "Use an explicit migration/reset after reviewing the changed prior, terrain, support or command schedule. "
            f"source={source_json}; current={current_json}"
        )


def load_source_manifest(
    checkpoint_path: str | os.PathLike[str] | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load and validate a training schedule sidecar adjacent to a checkpoint.

    Absence is returned as ``(None, None)`` so a caller can make an explicit
    legacy decision.  A sidecar that exists but is unreadable, malformed or
    incomplete is never treated as absence: it raises and fails closed.
    """
    if not checkpoint_path:
        return None, None
    for candidate in _manifest_candidates(checkpoint_path):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if not candidate.is_file():
            raise ScheduleManifestInvalidError(f"Schedule sidecar is not a regular file: {candidate}")
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScheduleManifestInvalidError(f"Cannot read schedule sidecar {candidate}: {error}") from error
        if not isinstance(data, dict):
            raise ScheduleManifestInvalidError(f"Schedule sidecar root must be a JSON object: {candidate}")
        try:
            schema_version = int(data["schema_version"])
            context = str(data["context"])
            runner_anchor = int(data["runner_iteration_at_anchor"])
            schedule_anchor = _finite_nonnegative(
                data["schedule_update_at_anchor"], "schedule_update_at_anchor"
            )
            if schema_version < 3:
                raise ValueError("schema_version must be >= 3 for checkpoint-continuous runs")
            if context != "train":
                raise ValueError(f"context must be 'train', got {context!r}")
            if runner_anchor < 0:
                raise ValueError("runner_iteration_at_anchor must be non-negative")
            if not isinstance(data.get("schedule_definition"), Mapping):
                raise ValueError("schedule_definition must be a JSON object")
            if data.get("runtime_state_required_for_preserve") is not True:
                raise ValueError("runtime_state_required_for_preserve must be true")
        except (KeyError, TypeError, ValueError) as error:
            raise ScheduleManifestInvalidError(f"Invalid schedule sidecar {candidate}: {error}") from error
        # Store normalized values so downstream derivation cannot interpret a
        # bool/string differently from this validation pass.
        data["schema_version"] = schema_version
        data["runner_iteration_at_anchor"] = runner_anchor
        data["schedule_update_at_anchor"] = schedule_anchor
        return candidate, data
    return None, None


def _normalize_runtime_snapshot(
    snapshot: Mapping[str, Any],
    *,
    source: str,
    require_checkpoint_sha256: bool = False,
) -> dict[str, Any]:
    try:
        checkpoint_file = str(snapshot["checkpoint_file"])
        checkpoint_digest = snapshot.get("checkpoint_sha256")
        runner_iteration = int(snapshot["runner_iteration"])
        schedule_update = _finite_nonnegative(snapshot["schedule_update"], "schedule_update")
        command_state = snapshot["command_curriculum"]
        moving_best = snapshot["moving_best"]
        if not checkpoint_file:
            raise ValueError("checkpoint_file is empty")
        if checkpoint_digest is None:
            if require_checkpoint_sha256:
                raise ValueError("checkpoint_sha256 is required")
        else:
            if not isinstance(checkpoint_digest, str):
                raise ValueError("checkpoint_sha256 must be a string")
            checkpoint_digest = checkpoint_digest.lower()
            if len(checkpoint_digest) != 64 or any(
                character not in "0123456789abcdef" for character in checkpoint_digest
            ):
                raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
        if runner_iteration < 0:
            raise ValueError("runner_iteration must be non-negative")
        if not isinstance(command_state, Mapping):
            raise ValueError("command_curriculum must be an object")
        if not isinstance(moving_best, Mapping):
            raise ValueError("moving_best must be an object")
    except (KeyError, TypeError, ValueError) as error:
        raise ScheduleManifestInvalidError(f"Invalid runtime snapshot in {source}: {error}") from error
    normalized = dict(snapshot)
    normalized["checkpoint_file"] = checkpoint_file
    normalized["checkpoint_sha256"] = checkpoint_digest
    normalized["runner_iteration"] = runner_iteration
    normalized["schedule_update"] = schedule_update
    normalized["command_curriculum"] = dict(command_state)
    normalized["moving_best"] = dict(moving_best)
    return normalized


def load_runtime_state(
    checkpoint_path: str | os.PathLike[str],
) -> tuple[Path | None, dict[str, Any] | None]:
    """Load a runtime-state history; a present invalid file always raises."""
    for candidate in _runtime_state_candidates(checkpoint_path):
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if not candidate.is_file():
            raise ScheduleManifestInvalidError(f"Runtime-state sidecar is not a regular file: {candidate}")
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScheduleManifestInvalidError(f"Cannot read runtime-state sidecar {candidate}: {error}") from error
        try:
            if not isinstance(data, dict):
                raise ValueError("root must be a JSON object")
            schema_version = int(data["schema_version"])
            if schema_version not in (1, 2):
                raise ValueError("schema_version must equal 1 or 2")
            if str(data["context"]) != "train":
                raise ValueError("context must equal 'train'")
            snapshots_raw = data["snapshots"]
            if not isinstance(snapshots_raw, list):
                raise ValueError("snapshots must be a list")
            snapshots = [
                _normalize_runtime_snapshot(
                    snapshot,
                    source=str(candidate),
                    require_checkpoint_sha256=schema_version >= 2,
                )
                for snapshot in snapshots_raw
                if isinstance(snapshot, Mapping)
            ]
            if len(snapshots) != len(snapshots_raw):
                raise ValueError("every snapshots entry must be an object")
        except (KeyError, TypeError, ValueError) as error:
            raise ScheduleManifestInvalidError(f"Invalid runtime-state sidecar {candidate}: {error}") from error
        normalized = dict(data)
        normalized["schema_version"] = schema_version
        normalized["context"] = "train"
        normalized["snapshots"] = snapshots
        return candidate, normalized
    return None, None


def runtime_snapshot_for_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    checkpoint_iteration: int,
) -> tuple[Path, dict[str, Any]]:
    """Return the exact runtime state written with one checkpoint."""
    state_path, state = load_runtime_state(checkpoint_path)
    if state_path is None or state is None:
        raise ScheduleManifestMissingError(
            f"Missing required runtime-state sidecar beside patched checkpoint {checkpoint_path}"
        )
    checkpoint_name = Path(checkpoint_path).name
    matches = [
        snapshot
        for snapshot in state["snapshots"]
        if snapshot["checkpoint_file"] == checkpoint_name
        and int(snapshot["runner_iteration"]) == int(checkpoint_iteration)
    ]
    if len(matches) != 1:
        raise ScheduleManifestInvalidError(
            f"Runtime-state sidecar {state_path} contains {len(matches)} exact records for "
            f"{checkpoint_name} at runner iteration {checkpoint_iteration}; expected exactly one"
        )
    snapshot = matches[0]
    expected_digest = snapshot.get("checkpoint_sha256")
    if expected_digest is None:
        raise ScheduleManifestInvalidError(
            f"Runtime snapshot in {state_path} does not bind {checkpoint_name} to checkpoint_sha256. "
            "Preserve is unsafe; use an explicit migration/reset."
        )
    actual_digest = checkpoint_sha256(checkpoint_path)
    if expected_digest != actual_digest:
        raise ScheduleManifestInvalidError(
            f"Checkpoint SHA-256 does not match runtime snapshot for {checkpoint_name}: "
            f"snapshot={expected_digest}, actual={actual_digest}"
        )
    return state_path, snapshot


def append_runtime_snapshot(
    state_path: str | os.PathLike[str],
    snapshot: Mapping[str, Any],
) -> Path:
    """Atomically append/replace the state paired with a saved checkpoint."""
    target = Path(state_path)
    normalized_snapshot = _normalize_runtime_snapshot(
        snapshot,
        source=str(target),
        require_checkpoint_sha256=True,
    )
    state_schema_version = 2
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ScheduleManifestInvalidError(f"Cannot update runtime-state sidecar {target}: {error}") from error
        if (
            not isinstance(data, dict)
            or data.get("schema_version") not in (1, 2)
            or data.get("context") != "train"
        ):
            raise ScheduleManifestInvalidError(f"Cannot update invalid runtime-state sidecar {target}")
        state_schema_version = int(data["schema_version"])
        snapshots = data.get("snapshots")
        if not isinstance(snapshots, list):
            raise ScheduleManifestInvalidError(f"Runtime-state snapshots must be a list: {target}")
        normalized_existing = [
            _normalize_runtime_snapshot(
                item,
                source=str(target),
                require_checkpoint_sha256=state_schema_version >= 2,
            )
            for item in snapshots
            if isinstance(item, Mapping)
        ]
        if len(normalized_existing) != len(snapshots):
            raise ScheduleManifestInvalidError(f"Runtime-state snapshot entry is not an object: {target}")
    else:
        normalized_existing = []
    key = (normalized_snapshot["checkpoint_file"], normalized_snapshot["runner_iteration"])
    normalized_existing = [
        item
        for item in normalized_existing
        if (item["checkpoint_file"], item["runner_iteration"]) != key
    ]
    normalized_existing.append(normalized_snapshot)
    normalized_existing.sort(key=lambda item: (item["runner_iteration"], item["checkpoint_file"]))
    return write_manifest(
        target,
        {"schema_version": state_schema_version, "context": "train", "snapshots": normalized_existing},
    )


def schedule_update_for_checkpoint(
    checkpoint_path: str | os.PathLike[str] | None,
    checkpoint_iteration: int,
    *,
    allow_legacy_checkpoint_fallback: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Resolve a checkpoint's schedule update from its run sidecar.

    The sidecar stores both runner and schedule anchors.  Their difference is
    essential after a weights-only or cross-task reset: future checkpoints can
    keep advancing the intended schedule even when runner iteration uses a
    different origin.
    """
    checkpoint_iteration = int(checkpoint_iteration)
    if checkpoint_iteration < 0:
        raise ValueError("checkpoint_iteration must be non-negative")
    manifest_path, manifest = load_source_manifest(checkpoint_path)
    if manifest is None:
        runtime_path, runtime_state = load_runtime_state(checkpoint_path) if checkpoint_path else (None, None)
        if runtime_path is not None or runtime_state is not None:
            raise ScheduleManifestInvalidError(
                f"Runtime-state sidecar {runtime_path} exists but the required schedule manifest is missing"
            )
        if not allow_legacy_checkpoint_fallback:
            raise ScheduleManifestMissingError(
                "No high-step schedule sidecar exists beside checkpoint "
                f"{checkpoint_path}. New/patched runs and weights-only preserve must fail closed. "
                "For a verified legacy checkpoint only, explicitly enable "
                "allow_legacy_checkpoint_fallback."
            )
        return float(max(checkpoint_iteration, 0)), {
            "method": "explicit_legacy_checkpoint_iteration_fallback",
            "source_manifest": None,
            "checkpoint_iteration": checkpoint_iteration,
            "legacy_fallback_allowed": True,
            "legacy_fallback_used": True,
        }
    try:
        runner_anchor = int(manifest["runner_iteration_at_anchor"])
        schedule_anchor = _finite_nonnegative(manifest["schedule_update_at_anchor"], "schedule_update_at_anchor")
        if checkpoint_iteration < runner_anchor:
            raise ValueError(
                f"checkpoint iteration {checkpoint_iteration} predates run anchor {runner_anchor}"
            )
        anchor_derived = schedule_anchor + float(checkpoint_iteration - runner_anchor)
    except (KeyError, TypeError, ValueError) as error:
        raise ScheduleManifestInvalidError(
            f"Cannot derive checkpoint schedule from {manifest_path}: {error}"
        ) from error
    runtime_path, runtime_snapshot = runtime_snapshot_for_checkpoint(
        checkpoint_path, checkpoint_iteration
    )
    resolved = _finite_nonnegative(runtime_snapshot["schedule_update"], "runtime schedule_update")
    runtime_checkpoint_digest = str(runtime_snapshot["checkpoint_sha256"])
    return resolved, {
        "method": "runtime_snapshot_exact",
        "source_manifest": str(manifest_path),
        "source_runtime_state": str(runtime_path),
        "checkpoint_iteration": checkpoint_iteration,
        "source_runner_iteration_at_anchor": runner_anchor,
        "source_schedule_update_at_anchor": schedule_anchor,
        "anchor_derived_schedule_update": anchor_derived,
        "runtime_snapshot_schedule_update": resolved,
        "runtime_snapshot_checkpoint_sha256": runtime_checkpoint_digest,
        "runtime_snapshot_checkpoint_sha256_verified": True,
        "runtime_minus_anchor_derived_update": resolved - anchor_derived,
        "source_manifest_schema_version": int(manifest["schema_version"]),
        "legacy_fallback_allowed": bool(allow_legacy_checkpoint_fallback),
        "legacy_fallback_used": False,
        "runtime_snapshot": runtime_snapshot,
    }


def resolve_loaded_schedule_update(
    checkpoint_path: str | os.PathLike[str],
    checkpoint_iteration: int,
    *,
    checkpoint_load_mode: str,
    schedule_resume_mode: str,
    allow_legacy_checkpoint_fallback: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Resolve preserve/reset semantics with fail-closed provenance rules."""
    if checkpoint_load_mode not in {"full", "weights_only"}:
        raise ValueError(f"Unsupported checkpoint load mode: {checkpoint_load_mode!r}")
    if schedule_resume_mode not in {"preserve", "reset"}:
        raise ValueError(f"Schedule resume mode must be resolved first, got {schedule_resume_mode!r}")

    if schedule_resume_mode == "reset":
        # Absence is fine because no prior schedule is used, but a present bad
        # sidecar still indicates corrupted provenance and must never be hidden.
        manifest_path, manifest = load_source_manifest(checkpoint_path)
        runtime_path, runtime_state = load_runtime_state(checkpoint_path)
        if manifest is None and runtime_state is not None:
            raise ScheduleManifestInvalidError(
                f"Runtime-state sidecar {runtime_path} exists without its schedule manifest"
            )
        return 0.0, {
            "method": "explicit_schedule_reset",
            "source_manifest": str(manifest_path) if manifest_path is not None else None,
            "source_manifest_present": manifest is not None,
            "source_runtime_state": str(runtime_path) if runtime_path is not None else None,
            "source_runtime_state_present": runtime_state is not None,
            "checkpoint_iteration": int(checkpoint_iteration),
            "legacy_fallback_allowed": False,
            "legacy_fallback_used": False,
        }

    if checkpoint_load_mode == "weights_only" and allow_legacy_checkpoint_fallback:
        raise ScheduleContinuityError(
            "weights_only + preserve cannot use legacy checkpoint-iteration fallback; "
            "provide/reconstruct a valid source schedule sidecar or choose reset"
        )
    return schedule_update_for_checkpoint(
        checkpoint_path,
        checkpoint_iteration,
        allow_legacy_checkpoint_fallback=(
            bool(allow_legacy_checkpoint_fallback) if checkpoint_load_mode == "full" else False
        ),
    )


def _term_params(term_cfg: Any) -> dict[str, Any]:
    if term_cfg is None:
        return {}
    params = getattr(term_cfg, "params", None)
    if isinstance(params, Mapping):
        return dict(params)
    if isinstance(term_cfg, Mapping):
        return dict(term_cfg)
    return {}


def build_schedule_manifest(
    *,
    context: str,
    task: str | None,
    env: Any,
    action_cfg: Any,
    terrain_term_cfg: Any,
    support_term_cfg: Any,
    terrain_curriculum_enabled: bool,
    support_metric_enabled: bool,
    checkpoint_path: str | os.PathLike[str] | None,
    checkpoint_iteration: int | None,
    checkpoint_load_mode: str,
    requested_resume_mode: str,
    resolved_resume_mode: str,
    runner_iteration_at_anchor: int,
    schedule_update_at_anchor: float,
    schedule_source: Mapping[str, Any] | None = None,
    command_term_cfg: Any = None,
    command_curriculum_enabled: bool = False,
    runtime_state_path: str | os.PathLike[str] | None = None,
    command_state_restored: bool = False,
    moving_best_state_restored: bool = False,
    clean_ab_comparison_allowed: bool = False,
    reward_stage_definition: Mapping[str, Any] | None = None,
    student_parent_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready manifest of the schedule that is actually active."""
    target = unwrap_env(env)
    update = float(schedule_update_at_anchor)

    action_enabled = all(hasattr(action_cfg, name) for name in ("prior_start_update", "prior_full_update"))
    action_steps = int(getattr(action_cfg, "num_steps_per_update", 24) or 24)
    action_start = getattr(action_cfg, "prior_start_update", None)
    action_full = getattr(action_cfg, "prior_full_update", None)
    action_scale = (
        prior_scale(update, float(action_start), float(action_full)) if action_enabled else 0.0
    )

    terrain_params = _term_params(terrain_term_cfg)
    terrain_thresholds = terrain_params.get("stage_update_thresholds")
    terrain_levels = terrain_params.get("stage_max_levels")
    score_thresholds = terrain_params.get("score_stage_update_thresholds")

    support_params = _term_params(support_term_cfg)
    if not support_params:
        support_params = terrain_params
    support_start = float(support_params.get("support_bottleneck_start_update", 0))
    support_ramp = float(support_params.get("support_bottleneck_ramp_updates", 1))
    support_blend = linear_blend(update, support_start, support_ramp)
    schedule_definition = schedule_definition_from_configs(
        action_cfg=action_cfg,
        terrain_term_cfg=terrain_term_cfg,
        support_term_cfg=support_term_cfg,
        command_term_cfg=command_term_cfg,
        command_curriculum_enabled=command_curriculum_enabled,
        reward_stage_definition=reward_stage_definition,
    )
    resolved_checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
    )
    resolved_checkpoint_sha256 = (
        checkpoint_sha256(resolved_checkpoint_path) if resolved_checkpoint_path is not None else None
    )

    return {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": str(context),
        "task": task,
        "checkpoint_path": str(resolved_checkpoint_path) if resolved_checkpoint_path is not None else None,
        "checkpoint_sha256": resolved_checkpoint_sha256,
        "checkpoint_iteration": int(checkpoint_iteration) if checkpoint_iteration is not None else None,
        "checkpoint_load_mode": str(checkpoint_load_mode),
        "schedule_resume_mode_requested": str(requested_resume_mode),
        "schedule_resume_mode_resolved": str(resolved_resume_mode),
        "runner_iteration_at_anchor": int(runner_iteration_at_anchor),
        "schedule_update_at_anchor": update,
        "local_step_at_anchor": float(getattr(target, _ANCHOR_STEP_ATTR, 0.0)),
        "advance_with_local_steps": bool(getattr(target, _ADVANCE_ATTR, True)),
        "schedule_source": dict(schedule_source or {}),
        "student_parent_lineage": (
            dict(student_parent_lineage) if student_parent_lineage is not None else None
        ),
        "schedule_definition": schedule_definition,
        "runtime_state_required_for_preserve": str(context) == "train",
        "runtime_state_path": str(Path(runtime_state_path).resolve()) if runtime_state_path else None,
        "clean_ab_comparison_allowed": bool(clean_ab_comparison_allowed),
        "action_prior": {
            "enabled": bool(action_enabled),
            "num_steps_per_update": action_steps,
            "start_update": int(action_start) if action_start is not None else None,
            "full_update": int(action_full) if action_full is not None else None,
            "actual_prior_scale": action_scale,
        },
        "terrain_schedule": {
            "curriculum_enabled": bool(terrain_curriculum_enabled),
            "num_steps_per_update": schedule_definition["terrain_schedule"]["num_steps_per_update"],
            "stage_update_thresholds": list(terrain_thresholds) if terrain_thresholds is not None else None,
            "stage_max_levels": list(terrain_levels) if terrain_levels is not None else None,
            "configured_stage_index": stage_index(update, terrain_thresholds),
            "configured_allowed_max_level": allowed_max_level(update, terrain_thresholds, terrain_levels),
            "score_stage_update_thresholds": list(score_thresholds) if score_thresholds is not None else None,
            "configured_score_stage_index": stage_index(update, score_thresholds),
        },
        "support_bottleneck": {
            "metric_enabled": bool(support_metric_enabled),
            "num_steps_per_update": schedule_definition["support_bottleneck"]["num_steps_per_update"],
            "start_update": support_start,
            "ramp_updates": support_ramp,
            "warmup_min_gate": float(support_params.get("support_bottleneck_warmup_min_gate", 0.0)),
            "actual_blend": support_blend,
        },
        "command_curriculum": {
            **schedule_definition["command_curriculum"],
            "state_restored": bool(command_state_restored),
            "state_reset_at_process_start": bool(command_curriculum_enabled and not command_state_restored),
        },
        "moving_best_state": {
            "action_score_best_restored": bool(moving_best_state_restored),
            "support_score_best_restored": bool(moving_best_state_restored),
            "reset_at_process_start": not bool(moving_best_state_restored),
            "clean_ab_comparison_allowed": bool(clean_ab_comparison_allowed),
        },
    }


def eval_schedule_fields(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the flat schedule fields embedded in ``HIGHSTEP_EVAL_JSON``."""
    if manifest is None:
        return {
            "schedule_valid": False,
            "global_update": None,
            "schedule_source": None,
            "schedule_source_method": None,
            "runtime_snapshot_checkpoint_sha256": None,
            "runtime_snapshot_checkpoint_sha256_verified": None,
            "schedule_frozen_for_evaluation": None,
            "action_prior_enabled": None,
            "action_prior_scale": None,
            "terrain_curriculum_enabled": None,
            "terrain_stage_index": None,
            "terrain_allowed_max_level": None,
            "support_bottleneck_blend": None,
            "schedule_manifest_schema_version": None,
        }
    source = manifest.get("schedule_source") or {}
    return {
        "schedule_valid": True,
        "global_update": manifest["schedule_update_at_anchor"],
        "schedule_source": source,
        "schedule_source_method": source.get("method"),
        "runtime_snapshot_checkpoint_sha256": source.get("runtime_snapshot_checkpoint_sha256"),
        "runtime_snapshot_checkpoint_sha256_verified": source.get(
            "runtime_snapshot_checkpoint_sha256_verified"
        ),
        "schedule_frozen_for_evaluation": not manifest["advance_with_local_steps"],
        "action_prior_enabled": manifest["action_prior"]["enabled"],
        "action_prior_scale": manifest["action_prior"]["actual_prior_scale"],
        "terrain_curriculum_enabled": manifest["terrain_schedule"]["curriculum_enabled"],
        "terrain_stage_index": manifest["terrain_schedule"]["configured_stage_index"],
        "terrain_allowed_max_level": manifest["terrain_schedule"]["configured_allowed_max_level"],
        "support_bottleneck_blend": manifest["support_bottleneck"]["actual_blend"],
        "schedule_manifest_schema_version": manifest["schema_version"],
    }


def runtime_schedule_match(
    manifest: Mapping[str, Any],
    *,
    runtime_prior_scale: float | None,
    runtime_support_blend: float | None,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    """Validate that required runtime schedule signals were observed and match."""
    action_prior = manifest["action_prior"]
    support = manifest["support_bottleneck"]
    prior_required = bool(action_prior["enabled"])
    support_required = bool(support["metric_enabled"])
    expected_prior = float(action_prior["actual_prior_scale"])
    expected_support = float(support["actual_blend"])

    prior_observed = runtime_prior_scale is not None
    support_observed = runtime_support_blend is not None
    prior_match = (
        prior_observed and abs(float(runtime_prior_scale) - expected_prior) <= tolerance
        if prior_required
        else (not prior_observed or abs(float(runtime_prior_scale) - expected_prior) <= tolerance)
    )
    support_match = (
        support_observed and abs(float(runtime_support_blend) - expected_support) <= tolerance
        if support_required
        else (not support_observed or abs(float(runtime_support_blend) - expected_support) <= tolerance)
    )
    return {
        "schedule_runtime_match": bool(prior_match and support_match),
        "action_prior_runtime_required": prior_required,
        "action_prior_runtime_observed": prior_observed,
        "action_prior_runtime_match": bool(prior_match),
        "support_runtime_required": support_required,
        "support_runtime_observed": support_observed,
        "support_runtime_match": bool(support_match),
        "action_prior_scale_runtime": float(runtime_prior_scale) if prior_observed else None,
        "support_bottleneck_blend_runtime": float(runtime_support_blend) if support_observed else None,
    }


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """Atomically write a schedule manifest and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target
