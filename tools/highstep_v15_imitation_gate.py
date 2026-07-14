#!/usr/bin/env python3
"""Evaluate a v1.5 candidate against the frozen independent 0707 envelope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import highstep_v15_reference_manifest as reference


WORKFLOW = ROOT / "tmp/highstep_student_recovery_v15_20260713"
REFERENCE = WORKFLOW / "0707_reference/0707_imitation_reference_manifest.json"
REFERENCE_SHA = "f910798c5b8fab23b37c7fc10ccd5ac9a734734f65e4e318d0dbdd3446527e9f"
SPEC_SHA = "2e385c15ef58db1c45b25ff910d7b5f9d57ab06e86333cb596dd68264496085a"
HISTORICAL_0707_SPEC_SHA = "eff246af70dbfca7b3a6661f0a29f71bb1aa3b769a944351431f8b589663437a"
HISTORICAL_0707_V171_SPEC_SHA = "140fd81d4f6877d25e72f3f1e05799cb6771dfdfc9775fe84a46ecf7d7a6a917"
PREREG_SHA = "0be872b32062fe6d162a6b25435ba81c7b45fbc30a703eb0d7597a29a8ff0421"
V152_WORKFLOW = ROOT / "tmp/highstep_student_recovery_v152_20260713"
V152_PREREG = V152_WORKFLOW / "preregistration_v1.json"
V152_PREREG_SHA = "473b8554d25d253886547fe4af72d49e30dad796d66fd9a3944e1d0ad8238ec2"
V152_REBINDING_AUDIT = V152_WORKFLOW / "rebinding_audit.json"
V152_REBINDING_AUDIT_SHA = "2d2826404676a04a5ff215dea891bc812516b55d579ccc0ba86e62ea4c269a30"
SOURCE = WORKFLOW / "source_model_172300_v3/model_172300.pt"
TEACHER_SHA = "dee40bff6b1c1e15b29012aaa19767a5e9d28ddaffd759b472564a5d5ef4eb35"
ALGORITHM_KEY = "robot_lab_algorithm_checkpoint_state"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frames(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path: Path, payload: Mapping[str, Any], read_only: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _le(name: str, actual: float, limit: float, failures: list[dict[str, Any]]) -> None:
    if not math.isfinite(float(actual)) or float(actual) > float(limit) + 1.0e-9:
        failures.append({"metric": name, "actual": actual, "upper": limit})


def _vector(name: str, actual: Sequence[float], limit: Sequence[float], failures: list[dict[str, Any]]) -> None:
    if len(actual) != len(limit):
        failures.append({"metric": name, "actual_width": len(actual), "expected_width": len(limit)})
        return
    for index, (value, upper) in enumerate(zip(actual, limit, strict=True)):
        _le(f"{name}[{index}]", float(value), float(upper), failures)


def checkpoint_scope(checkpoint: Path, expected_updates: int) -> dict[str, Any]:
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    candidate = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state0 = source["model_state_dict"]
    state = candidate["model_state_dict"]
    recovery = candidate["infos"][ALGORITHM_KEY]["student_recovery"]
    binding = recovery["binding_manifest"]
    count = int(candidate["infos"][ALGORITHM_KEY]["student_distill_update_count"])
    if count != expected_updates or int(recovery["effective_update_count"]) != expected_updates:
        raise RuntimeError(f"effective update mismatch: {count} != {expected_updates}")
    prereg_sha = binding.get("preregistration_sha256")
    exact_0707 = recovery.get("stage") == "0707_EXACT"
    historical_0707_exact = recovery.get("stage") == "HISTORICAL_0707_EXACT"
    exact_prereg_path = Path(str(binding.get("preregistration_path", "")))
    exact_authority = bool(
        exact_0707
        and binding.get("warmup_updates") == 1200
        and binding.get("warmup_main_action_target") == "teacher_pre_prior"
        and binding.get("phase_scale") == 0.0
        and binding.get("rear_box_scale") == 0.0
        and recovery.get("preregistration_sha256") == prereg_sha
        and exact_prereg_path.is_absolute()
        and sha256_file(exact_prereg_path) == prereg_sha
    )
    historical_authority = bool(
        historical_0707_exact
        and binding.get("recovery_spec_sha256") in {
            HISTORICAL_0707_SPEC_SHA,
            HISTORICAL_0707_V171_SPEC_SHA,
        }
        and binding.get("warmup_updates") == 1400
        and binding.get("warmup_main_action_target") == "teacher_pre_prior"
        and binding.get("phase_scale") == 2.0
        and binding.get("rear_box_scale") == 1.5
        and binding.get("warmup_prior_box_loss_coefficient") == 0.0
        and recovery.get("preregistration_sha256") == prereg_sha
        and exact_prereg_path.is_absolute()
        and sha256_file(exact_prereg_path) == prereg_sha
    )
    v15_authority = prereg_sha == PREREG_SHA
    v152_authority = bool(
        prereg_sha == V152_PREREG_SHA
        and recovery.get("preregistration_sha256") == V152_PREREG_SHA
        and binding.get("authority_rebound_from_preregistration_sha256") == PREREG_SHA
        and binding.get("authority_rebinding_audit")
        == {"path": str(V152_REBINDING_AUDIT), "sha256": V152_REBINDING_AUDIT_SHA}
        and sha256_file(V152_PREREG) == V152_PREREG_SHA
        and sha256_file(V152_REBINDING_AUDIT) == V152_REBINDING_AUDIT_SHA
    )
    if not (
        binding["teacher_sha256"] == TEACHER_SHA
        and binding["initial_student_sha256"] == TEACHER_SHA
        and recovery.get("preregistration_sha256") == prereg_sha
        and (v15_authority or v152_authority or exact_authority or historical_authority)
        and binding["student_ppo_permanently_disabled"] is True
        and binding["student_teacher_storage_independent"] is True
        and binding["warmup_updates"] == (1200 if exact_0707 else 1400)
    ):
        raise RuntimeError("v1.5 checkpoint binding changed")

    changed: list[str] = []
    forbidden: list[str] = []
    for name, before in state0.items():
        after = state.get(name)
        if not torch.is_tensor(before) or not torch.is_tensor(after) or torch.equal(before, after):
            continue
        changed.append(name)
        if name.startswith("estimator."):
            continue
        if expected_updates > binding["warmup_updates"] and name.endswith("actor.6.weight"):
            if torch.equal(before[:12], after[:12]):
                continue
        if expected_updates > binding["warmup_updates"] and name.endswith("actor.6.bias"):
            if torch.equal(before[:12], after[:12]):
                continue
        forbidden.append(name)
    if forbidden:
        raise RuntimeError(f"forbidden v1.5 parameter changes: {forbidden}")
    if expected_updates <= binding["warmup_updates"] and any(not name.startswith("estimator.") for name in changed):
        raise RuntimeError("actor changed during the fixed warmup freeze")
    groups = candidate["optimizer_state_dict"]["param_groups"]
    if len(groups) != 2 or len(groups[0]["params"]) != 16 or len(groups[1]["params"]) != 2:
        raise RuntimeError("v1.5 optimizer scope changed")
    return {
        "effective_updates": count,
        "changed_parameter_names": changed,
        "forbidden_parameter_names": forbidden,
        "actor_fully_frozen_through_warmup": expected_updates <= binding["warmup_updates"],
        "post_warmup_only_box_rows_outside_estimator": expected_updates > binding["warmup_updates"],
        "optimizer_group_sizes": [len(group["params"]) for group in groups],
        "authority_version": (
            "v1.7-historical-0707-exact" if historical_authority
            else "v1.6-zero-scale-ablation" if exact_authority
            else "v1.5.2" if v152_authority else "v1.5"
        ),
        "binding": binding,
    }


def evaluate(checkpoint: Path, audit_root: Path, expected_updates: int) -> dict[str, Any]:
    if sha256_file(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("frozen 0707 imitation reference SHA changed")
    frozen = json.loads(REFERENCE.read_text())
    envelope = frozen["acceptance_envelope"]
    runs: list[list[dict[str, Any]]] = []
    summaries = []
    for seed in (11, 22, 33):
        directory = audit_root / f"seed{seed}_nominal"
        summary_path = directory / "run_summary.json"
        frames_path = directory / "frames.jsonl"
        summary = json.loads(summary_path.read_text())
        frames = _read_frames(frames_path)
        if len(frames) != 600 or summary.get("frame_count") != 600:
            raise RuntimeError(f"incomplete candidate trace: {directory}")
        if not (
            summary.get("read_only_model_audit") is True
            and summary.get("runner_manual_action_allclose_all_frames") is False
            and summary.get("outcome", {}).get("same_state_runtime_contract_verified") is True
        ):
            raise RuntimeError(
                f"candidate trace did not preserve the frozen shared-state contract: {directory}"
            )
        binding = summary["checkpoint_binding"]
        if binding.get("student_checkpoint_sha256") != sha256_file(checkpoint):
            raise RuntimeError("candidate trace checkpoint binding changed")
        if binding.get("teacher_checkpoint_sha256") != TEACHER_SHA:
            raise RuntimeError("candidate trace Teacher binding changed")
        runs.append(frames)
        summaries.append(summary)

    failures: list[dict[str, Any]] = []
    phase_metrics: dict[str, Any] = {}
    for phase in reference.PHASES:
        records = [row for run in runs for row in run if row["phase"] == phase]
        limit = envelope["phase_metrics"][phase]
        if len(records) < int(limit["minimum_reference_samples"]):
            failures.append({"metric": f"{phase}.samples", "actual": len(records), "minimum": limit["minimum_reference_samples"]})
            continue
        actual = reference._phase_metrics(records)
        delta = reference._delta_metrics(runs, phase)
        actual["action_first_difference"] = delta
        phase_metrics[phase] = actual
        _vector(f"{phase}.joint_abs_bias", [abs(v) for v in actual["joint_bias_student_minus_teacher"]], limit["joint_abs_bias_upper"], failures)
        _vector(f"{phase}.joint_mae", actual["joint_mae"], limit["joint_mae_upper"], failures)
        _vector(f"{phase}.joint_q95", actual["joint_q95_abs_error"], limit["joint_q95_abs_error_upper"], failures)
        sign = [0.0 if v is None else float(v) for v in actual["joint_sign_mismatch_rate"]]
        _vector(f"{phase}.joint_sign_mismatch", sign, limit["joint_sign_mismatch_rate_upper"], failures)
        _vector(f"{phase}.student_clip", actual["student_target_clip_rate"], limit["student_target_clip_rate_upper"], failures)
        _le(f"{phase}.latent_mae", actual["latent_mae"], limit["latent_mae_upper"], failures)
        _vector(f"{phase}.latent_q95", actual["latent_q95_abs_error_by_dim"], limit["latent_q95_abs_error_by_dim_upper"], failures)
        _vector(f"{phase}.velocity_mean", actual["velocity_abs_error_mean_by_dim"], limit["velocity_abs_error_mean_by_dim_upper"], failures)
        _vector(f"{phase}.velocity_q95", actual["velocity_abs_error_q95_by_dim"], limit["velocity_abs_error_q95_by_dim_upper"], failures)
        for group, upper in limit["action_groups_upper"].items():
            _le(f"{phase}.{group}.mae", actual["action_groups"][group]["mae"], upper["mae"], failures)
            _le(f"{phase}.{group}.q95", actual["action_groups"][group]["q95"], upper["q95"], failures)
        _vector(f"{phase}.delta_mae", delta["student_abs_first_difference_mae_by_joint"], limit["student_abs_first_difference_mae_by_joint_upper"], failures)
        _vector(f"{phase}.delta_q95", delta["student_abs_first_difference_q95_by_joint"], limit["student_abs_first_difference_q95_by_joint_upper"], failures)

    event_rows = [reference._event_timing(run) for run in runs]
    event_timing: dict[str, Any] = {}
    for field, limit in envelope["event_timing"].items():
        values = [float(row[field]) for row in event_rows if row[field] is not None]
        event_timing[field] = values
        if len(values) != len(runs):
            failures.append({"metric": f"event.{field}", "observed_runs": len(values), "required_runs": len(runs)})
        for value in values:
            if value < float(limit["minimum"]) - 1.0e-9 or value > float(limit["maximum"]) + 1.0e-9:
                failures.append({"metric": f"event.{field}", "actual": value, **limit})

    return {
        "schema_version": 1,
        "kind": "highstep_v15_candidate_imitation_gate",
        "spec_sha256": SPEC_SHA,
        "reference_manifest": str(REFERENCE),
        "reference_manifest_sha256": REFERENCE_SHA,
        "candidate_checkpoint": str(checkpoint.resolve()),
        "candidate_checkpoint_sha256": sha256_file(checkpoint),
        "expected_effective_updates": expected_updates,
        "candidate_scope_audit": checkpoint_scope(checkpoint, expected_updates),
        "run_summaries": [str((audit_root / f"seed{seed}_nominal/run_summary.json").resolve()) for seed in (11, 22, 33)],
        "phase_metrics": phase_metrics,
        "event_timing": event_timing,
        "failed_elements": failures,
        "failed_element_count": len(failures),
        "passed": not failures,
        "global_average_can_override": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--effective-updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(args.checkpoint.resolve(strict=True), args.audit_root.resolve(strict=True), args.effective_updates)
    _write(args.output.resolve(), payload)
    print(json.dumps({"passed": payload["passed"], "failed_element_count": payload["failed_element_count"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
