"""Exact fail-closed source/evaluation task aliases for canonical highstep core9."""

from __future__ import annotations


ROBUST_STUDENT_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPrior-"
    "ArcdogAdjustableLeg-v0"
)
V152_SOURCE_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorV15-"
    "ArcdogAdjustableLeg-v0"
)
EXACT_0707_SOURCE_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPrior0707Exact-"
    "ArcdogAdjustableLeg-v0"
)
HISTORICAL_0707_EXACT_SOURCE_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreStudentNoPriorHistorical0707Exact-"
    "ArcdogAdjustableLeg-v0"
)
ROBUST_TEACHER_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobust-ArcdogAdjustableLeg-v0"
)
TEACHER_TASK = "RobotLab-Isaac-Velocity-HighstepActionScore-ArcdogAdjustableLeg-v0"

_STUDENT_ALIASES = frozenset({
    (
        "highstep_student_recovery_v152_20260713",
        "student",
        V152_SOURCE_TASK,
        ROBUST_STUDENT_TASK,
    ),
    (
        "highstep_0707_exact_new_teacher_20260713",
        "student",
        EXACT_0707_SOURCE_TASK,
        ROBUST_STUDENT_TASK,
    ),
    (
        "highstep_historical_0707_exact_new_teacher_20260714",
        "student",
        HISTORICAL_0707_EXACT_SOURCE_TASK,
        ROBUST_STUDENT_TASK,
    ),
})


def source_task_is_compatible(
    *, workflow_id: str, role: str, source_task: str, eval_task: str
) -> bool:
    """Allow only identity, two named Student aliases, or the existing Teacher alias."""
    return bool(
        source_task == eval_task
        or (workflow_id, role, source_task, eval_task) in _STUDENT_ALIASES
        or (
            role == "teacher_robust"
            and source_task == TEACHER_TASK
            and eval_task == ROBUST_TEACHER_TASK
        )
    )


def source_authority_is_valid(
    *,
    workflow_id: str,
    role: str,
    source_task: str,
    eval_task: str,
    checkpoint_sha_valid: bool,
    lineage_valid: bool,
) -> bool:
    """Task alias never bypasses checkpoint-byte or parent-lineage validation."""
    return bool(
        checkpoint_sha_valid
        and lineage_valid
        and source_task_is_compatible(
            workflow_id=workflow_id,
            role=role,
            source_task=source_task,
            eval_task=eval_task,
        )
    )
