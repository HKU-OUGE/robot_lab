#!/usr/bin/env python3
"""Stop at E1400 when 0707 behavior is not reproduced and isolate the cause."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path("/home/lxq/Softwares/robot_lab")
WORKFLOW_ROOT = ROOT / "tmp/highstep_0707_exact_new_teacher_20260713"
EXACT_SERVICE = "highstep-0707-exact-new-teacher.service"
PHASES = (
    "approach", "front_lift", "front_top_support",
    "first_rear_top", "second_rear_top", "rear_hold",
)
JOINT_NAMES = (
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
    "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
)
PATHS = {
    "actor_gap_student_teacher_latent_vs_teacher": (
        "student_action_with_teacher_latent_SzT", "teacher_action_pre_prior"
    ),
    "estimator_latent_effect_student_vs_student_teacher_latent": (
        "student_action", "student_action_with_teacher_latent_SzT"
    ),
    "student_vs_teacher_pre_prior": ("student_action", "teacher_action_pre_prior"),
    "student_vs_teacher_post_prior": (
        "student_action", "teacher_action_post_prior_raw_equivalent"
    ),
    "teacher_pre_vs_post_prior": (
        "teacher_action_pre_prior", "teacher_action_post_prior_raw_equivalent"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def behavior_comparison(stage_result: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    actual = stage_result.get("core9", {}).get("counts", {})
    reference = baseline.get("counts", {})
    required = {
        "valid": int(reference.get("valid", 9)),
        "full_climb": int(reference.get("full_climb", 7)),
        "rear_hold": int(reference.get("rear_hold", 7)),
    }
    reproduced = bool(
        int(actual.get("valid", -1)) == required["valid"]
        and int(actual.get("full_climb", -1)) >= required["full_climb"]
        and int(actual.get("rear_hold", -1)) >= required["rear_hold"]
    )
    return {
        "schema_version": 1,
        "kind": "highstep_e1400_vs_0707_behavior_comparison",
        "actual": dict(actual),
        "reference": dict(reference),
        "actual_behavior_score": int(actual.get("full_climb", 0)) + int(actual.get("rear_hold", 0)),
        "reference_behavior_score": int(reference.get("full_climb", 0)) + int(reference.get("rear_hold", 0)),
        "componentwise_reproduction_required": True,
        "reproduced_0707_behavior": reproduced,
        "root_cause_isolation_required": not reproduced,
        "thresholds_modified_after_results": False,
    }


def _load_runs(trace_root: Path, checkpoint_sha256: str) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    for seed in (11, 22, 33):
        directory = trace_root / f"seed{seed}_nominal"
        summary_path = directory / "run_summary.json"
        frames_path = directory / "frames.jsonl"
        summary = json.loads(summary_path.read_text())
        frames = [json.loads(line) for line in frames_path.read_text().splitlines() if line.strip()]
        if len(frames) != 600 or summary.get("frame_count") != 600:
            raise RuntimeError(f"same-state trace is incomplete: {directory}")
        binding = summary.get("checkpoint_binding", {})
        if binding.get("student_checkpoint_sha256") != checkpoint_sha256:
            raise RuntimeError(f"same-state Student checkpoint binding changed: {directory}")
        if not (
            binding.get("student_teacher_paths_distinct") is True
            and binding.get("student_teacher_storage_independent") is True
            and summary.get("all_forward_state_digests_unchanged") is True
            and summary.get("all_observation_and_prior_inputs_unchanged") is True
        ):
            raise RuntimeError(f"same-state independence contract failed: {directory}")
        if sha256_file(frames_path) != summary.get("frames_sha256"):
            raise RuntimeError(f"same-state trace SHA changed: {directory}")
        runs.append(frames)
        summaries.append(summary)
    return runs, summaries


def _joint_error_metrics(
    runs: Sequence[Sequence[Mapping[str, Any]]], phase: str, left: str, right: str
) -> dict[str, Any]:
    rows = [row for run in runs for row in run if row.get("phase") == phase]
    if not rows:
        raise RuntimeError(f"same-state phase has no frames: {phase}")
    errors = [[] for _ in JOINT_NAMES]
    sign_errors = [[] for _ in JOINT_NAMES]
    delta_errors = [[] for _ in JOINT_NAMES]
    for row in rows:
        lhs = row[left]; rhs = row[right]
        if len(lhs) != len(JOINT_NAMES) or len(rhs) != len(JOINT_NAMES):
            raise RuntimeError(f"invalid action dimension in phase {phase}")
        for index in range(len(JOINT_NAMES)):
            errors[index].append(abs(float(lhs[index]) - float(rhs[index])))
            if abs(float(rhs[index])) >= 0.05:
                sign_errors[index].append(float(lhs[index]) * float(rhs[index]) < 0.0)
    for run in runs:
        phase_rows = [row for row in run if row.get("phase") == phase]
        for previous, current in zip(phase_rows, phase_rows[1:]):
            for index in range(len(JOINT_NAMES)):
                left_delta = float(current[left][index]) - float(previous[left][index])
                right_delta = float(current[right][index]) - float(previous[right][index])
                delta_errors[index].append(abs(left_delta - right_delta))
    mae = [sum(values) / len(values) for values in errors]
    q95 = [quantile(values, 0.95) for values in errors]
    delta_mae = [sum(values) / len(values) if values else 0.0 for values in delta_errors]
    delta_q95 = [quantile(values, 0.95) for values in delta_errors]
    sign = [
        (sum(bool(value) for value in values) / len(values)) if values else None
        for values in sign_errors
    ]
    return {
        "samples": len(rows),
        "joint_mae": mae,
        "joint_q95_abs_error": q95,
        "joint_sign_mismatch_rate": sign,
        "action_first_difference_error_mae": delta_mae,
        "action_first_difference_error_q95": delta_q95,
        "global_mae": sum(mae) / len(mae),
        "global_q95": sum(q95) / len(q95),
    }


def _latent_metrics(runs: Sequence[Sequence[Mapping[str, Any]]], phase: str) -> dict[str, Any]:
    rows = [row for run in runs for row in run if row.get("phase") == phase]
    errors = [[abs(float(value)) for value in row["latent_error_student_minus_teacher"]] for row in rows]
    dimensions = len(errors[0]) if errors else 0
    by_dimension = [[row[index] for row in errors] for index in range(dimensions)]
    return {
        "samples": len(rows),
        "mae": sum(sum(row) for row in errors) / (len(errors) * dimensions),
        "q95_abs_error_by_dim": [quantile(values, 0.95) for values in by_dimension],
    }


def trace_metrics(runs: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in PHASES:
        result[phase] = {
            "latent": _latent_metrics(runs, phase),
            "action_paths": {
                name: _joint_error_metrics(runs, phase, left, right)
                for name, (left, right) in PATHS.items()
            },
        }
    return result


def _actor_tensor_audit(teacher_path: Path, e1200_path: Path, e1400_path: Path) -> dict[str, Any]:
    import torch

    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False)["model_state_dict"]
    before = torch.load(e1200_path, map_location="cpu", weights_only=False)["model_state_dict"]
    after = torch.load(e1400_path, map_location="cpu", weights_only=False)["model_state_dict"]
    actor_keys = sorted(key for key in teacher if key.startswith("actor."))
    exact_before = {key: bool(torch.equal(before[key], teacher[key])) for key in actor_keys}
    forbidden_differences: list[str] = []
    allowed_differences: dict[str, Any] = {}
    for key in actor_keys:
        difference = after[key] - teacher[key]
        if key == "actor.6.weight":
            forbidden = difference.clone(); forbidden[12:16] = 0
            if bool(torch.any(forbidden != 0).item()):
                forbidden_differences.append(key)
            allowed = difference[12:16]
        elif key == "actor.6.bias":
            forbidden = difference.clone(); forbidden[12:16] = 0
            if bool(torch.any(forbidden != 0).item()):
                forbidden_differences.append(key)
            allowed = difference[12:16]
        else:
            if bool(torch.any(difference != 0).item()):
                forbidden_differences.append(key)
            continue
        allowed_differences[key] = {
            "changed_elements": int(torch.count_nonzero(allowed).item()),
            "max_abs": float(torch.max(torch.abs(allowed)).item()),
        }
    return {
        "actor_keys": actor_keys,
        "e1200_student_actor_exactly_equals_teacher": all(exact_before.values()),
        "e1200_mismatched_actor_keys": [key for key, value in exact_before.items() if not value],
        "e1400_only_allowed_box_output_rows_changed": not forbidden_differences,
        "e1400_forbidden_actor_differences": forbidden_differences,
        "e1400_allowed_box_row_differences": allowed_differences,
        "loaded_tensor_storage_independent": all(
            teacher[key].untyped_storage().data_ptr() != before[key].untyped_storage().data_ptr()
            for key in actor_keys
        ),
    }


def build_root_cause_report(prereg: Mapping[str, Any], behavior: Mapping[str, Any]) -> dict[str, Any]:
    stage_results = {}
    stage_runs = {}
    stage_summaries = {}
    for stage in (1200, 1400):
        result_path = WORKFLOW_ROOT / f"stages/E{stage}/stage_result.json"
        result = json.loads(result_path.read_text())
        checkpoint = Path(result["checkpoint"]).resolve(strict=True)
        if sha256_file(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError(f"E{stage} checkpoint SHA changed")
        runs, summaries = _load_runs(
            WORKFLOW_ROOT / f"stages/E{stage}/imitation/traces", result["checkpoint_sha256"]
        )
        stage_results[stage] = result
        stage_runs[stage] = runs
        stage_summaries[stage] = summaries
    # Required causal order: prove initialization/binding and freeze scope
    # before using any action or latent statistics to assign a cause.
    tensor_audit = _actor_tensor_audit(
        Path(str(prereg["teacher_checkpoint"])),
        Path(stage_results[1200]["checkpoint"]),
        Path(stage_results[1400]["checkpoint"]),
    )
    reference_checkpoint = str(prereg["reference_deployed_checkpoint_sha256"])
    reference_runs, reference_summaries = _load_runs(
        Path(str(prereg["reference_trace_root"])), reference_checkpoint
    )
    # Only after the tensor/binding audit, quantify the three same-state paths
    # and their six-phase per-joint decomposition.
    stage_metrics = {stage: trace_metrics(stage_runs[stage]) for stage in (1200, 1400)}
    reference_metrics = trace_metrics(reference_runs)
    comparisons: dict[str, Any] = {}
    for phase in PHASES:
        comparisons[phase] = {}
        for path_name in PATHS:
            current = stage_metrics[1400][phase]["action_paths"][path_name]["global_mae"]
            reference = reference_metrics[phase]["action_paths"][path_name]["global_mae"]
            comparisons[phase][path_name] = {
                "e1400_global_mae": current,
                "0707_deployed_global_mae": reference,
                "ratio_to_0707": current / reference if reference > 1.0e-12 else None,
            }
    aggregate = {}
    for stage in (1200, 1400):
        aggregate[stage] = {
            name: sum(stage_metrics[stage][phase]["action_paths"][name]["global_mae"] for phase in PHASES) / len(PHASES)
            for name in PATHS
        }
    if not tensor_audit["e1200_student_actor_exactly_equals_teacher"]:
        classification = "student_actor_initialization_or_binding_failure"
    elif not tensor_audit["e1400_only_allowed_box_output_rows_changed"]:
        classification = "actor_freeze_scope_violation_after_warmup"
    elif aggregate[1200]["actor_gap_student_teacher_latent_vs_teacher"] > 1.0e-6:
        classification = "runtime_actor_or_teacher_binding_mismatch"
    elif aggregate[1400]["estimator_latent_effect_student_vs_student_teacher_latent"] > max(
        1.25 * aggregate[1400]["actor_gap_student_teacher_latent_vs_teacher"], 1.0e-6
    ):
        classification = "estimator_latent_path_primary"
    elif aggregate[1400]["actor_gap_student_teacher_latent_vs_teacher"] > max(
        1.25 * aggregate[1400]["estimator_latent_effect_student_vs_student_teacher_latent"], 1.0e-6
    ):
        classification = "post_warmup_box_head_actor_path_primary"
    else:
        classification = "mixed_estimator_and_post_warmup_box_head_paths"
    unique_contracts = sorted({
        str(row["post_prior_contract_sha256"])
        for runs in (*stage_runs.values(), reference_runs)
        for run in runs for row in run
    })
    saved_configs = [summary["checkpoint_binding"]["saved_configs"] for summary in stage_summaries[1400]]
    for group in saved_configs:
        for item in group.values():
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"saved config SHA changed: {path}")
    return {
        "schema_version": 1,
        "kind": "highstep_e1400_root_cause_isolation",
        "workflow_id": prereg["workflow_id"],
        "behavior_comparison": dict(behavior),
        "classification": classification,
        "evidence_closed": classification != "mixed_estimator_and_post_warmup_box_head_paths",
        "tensor_initialization_and_freeze_audit": tensor_audit,
        "checkpoint_lineage_and_contract": {
            "teacher_checkpoint": prereg["teacher_checkpoint"],
            "teacher_checkpoint_sha256": prereg["teacher_checkpoint_sha256"],
            "e1200_checkpoint": stage_results[1200]["checkpoint"],
            "e1200_checkpoint_sha256": stage_results[1200]["checkpoint_sha256"],
            "e1400_checkpoint": stage_results[1400]["checkpoint"],
            "e1400_checkpoint_sha256": stage_results[1400]["checkpoint_sha256"],
            "student_teacher_storage_independent_all_runs": True,
            "saved_config_sha_verified": True,
            "action_dimension": 16,
            "joint_names": list(JOINT_NAMES),
            "post_prior_contract_sha256_values": unique_contracts,
        },
        "three_path_aggregate_global_mae": aggregate,
        "phase_metrics": {
            "e1200_pre_box_adaptation": stage_metrics[1200],
            "e1400_post_box_adaptation": stage_metrics[1400],
            "0707_deployed_reference": reference_metrics,
        },
        "e1400_vs_0707_phase_comparison": comparisons,
        "global_average_is_not_an_acceptance_override": True,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _supervisor_pid() -> int:
    state = json.loads((WORKFLOW_ROOT / "state.json").read_text())
    pid = int(state.get("supervisor_pid") or 0)
    cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    if b"highstep_0707_exact_new_teacher_supervisor.py" not in cmdline:
        raise RuntimeError("exact supervisor PID binding is invalid")
    return pid


def _stop_exact_service_from_paused_pid(pid: int) -> None:
    subprocess.run(
        ["systemctl", "--user", "kill", "--signal=SIGTERM", EXACT_SERVICE], check=True
    )
    os.kill(pid, signal.SIGCONT)
    subprocess.run(["systemctl", "--user", "stop", EXACT_SERVICE], check=False, timeout=90)
    subprocess.run(["systemctl", "--user", "disable", EXACT_SERVICE], check=False, timeout=30)


def run_guard(prereg: Mapping[str, Any], poll_seconds: float) -> int:
    output_root = WORKFLOW_ROOT / "root_cause_isolation_e1400"
    stage_result_path = WORKFLOW_ROOT / "stages/E1400/stage_result.json"
    heartbeat = output_root / "heartbeat.json"
    while not stage_result_path.is_file():
        atomic_json(heartbeat, {
            "status": "waiting_for_E1400", "workflow_id": prereg["workflow_id"],
            "e1800_allowed_before_decision": False,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        time.sleep(poll_seconds)
    pid = _supervisor_pid()
    os.kill(pid, signal.SIGSTOP)
    try:
        result = json.loads(stage_result_path.read_text())
        if int(result.get("stage", -1)) != 1400:
            raise RuntimeError("E1400 stage result identity changed")
        checkpoint = Path(result["checkpoint"])
        if sha256_file(checkpoint) != result["checkpoint_sha256"]:
            raise RuntimeError("E1400 stage result checkpoint SHA changed")
        baseline_path = Path(str(prereg["behavior_baseline_manifest"])).resolve(strict=True)
        if sha256_file(baseline_path) != prereg["behavior_baseline_manifest_sha256"]:
            raise RuntimeError("0707 behavior baseline SHA changed")
        comparison = behavior_comparison(result, json.loads(baseline_path.read_text()))
        comparison.update({
            "e1400_stage_result": str(stage_result_path),
            "e1400_stage_result_sha256": sha256_file(stage_result_path),
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(output_root / "behavior_comparison.json", comparison, read_only=True)
        if comparison["reproduced_0707_behavior"]:
            atomic_json(output_root / "guard_result.json", {
                "status": "e1400_reproduced_0707_behavior",
                "e1800_blocked": False,
                "root_cause_isolation_required": False,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, read_only=True)
            os.kill(pid, signal.SIGCONT)
            return 0
        _stop_exact_service_from_paused_pid(pid)
        report = build_root_cause_report(prereg, comparison)
        report_path = output_root / "root_cause_report.json"
        atomic_json(report_path, report, read_only=True)
        state_path = WORKFLOW_ROOT / "state.json"
        state = json.loads(state_path.read_text())
        state.update({
            "status": "root_cause_isolation_complete" if report["evidence_closed"] else "root_cause_isolation_requires_followup",
            "phase": "E1400_root_cause_isolation",
            "active_pid": None,
            "stop_reason": "E1400_below_frozen_0707_behavior_baseline",
            "root_cause_report": str(report_path),
            "root_cause_report_sha256": sha256_file(report_path),
            "E1800_E2500_allowed": False,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
        atomic_json(state_path, state)
        atomic_json(output_root / "guard_result.json", {
            "status": state["status"],
            "e1800_blocked": True,
            "exact_service_disabled": True,
            "root_cause_report": str(report_path),
            "root_cause_report_sha256": sha256_file(report_path),
            "classification": report["classification"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, read_only=True)
        return 0
    except Exception:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.10)
    args = parser.parse_args()
    prereg_path = args.preregistration.resolve(strict=True)
    if sha256_file(prereg_path) != args.preregistration_sha256:
        raise RuntimeError("E1400 root-cause preregistration SHA changed")
    prereg = json.loads(prereg_path.read_text())
    if not (
        prereg.get("kind") == "highstep_e1400_root_cause_preregistration"
        and prereg.get("workflow_id") == "highstep_0707_exact_new_teacher_20260713"
        and prereg.get("e1800_e2500_blocked_until_evidence_closure") is True
        and prereg.get("training_semantics_changed_before_e1400") is False
        and sha256_file(Path(__file__).resolve()) == prereg.get("guard_script_sha256")
    ):
        raise RuntimeError("E1400 root-cause preregistration identity changed")
    for path_key, sha_key in (
        ("spec_path", "spec_sha256"),
        ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ("reference_manifest", "reference_manifest_sha256"),
        ("behavior_baseline_manifest", "behavior_baseline_manifest_sha256"),
    ):
        path = Path(str(prereg[path_key])).resolve(strict=True)
        if sha256_file(path) != prereg[sha_key]:
            raise RuntimeError(f"E1400 root-cause authority changed: {path_key}")
    return run_guard(prereg, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
