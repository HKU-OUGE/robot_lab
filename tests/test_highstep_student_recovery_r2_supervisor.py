from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from tools import highstep_student_recovery_r2_supervisor as supervisor


def _summary(*, valid=9, full=4, hold=4, front=7, no_severe=7):
    return {
        "valid_count": valid,
        "full_count": full,
        "rear_hold_count": hold,
        "front_top_support_count": front,
        "no_severe_inward_count": no_severe,
    }


def _states():
    anchor = {
        name: torch.zeros((2, 2)) if name.endswith("weight") else torch.zeros(2)
        for name in supervisor.R2_OPTIMIZER_NAMES[:6]
    }
    anchor.update(
        {
            "actor.6.weight": torch.zeros(16, 2),
            "actor.6.bias": torch.zeros(16),
            "critic.0.weight": torch.zeros(2, 2),
        }
    )
    candidate = copy.deepcopy(anchor)
    candidate[supervisor.R2_OPTIMIZER_NAMES[0]][0, 0] = 0.25
    candidate["actor.6.weight"][12:16] = 0.1
    candidate["actor.6.bias"][12:16] = 0.2
    return anchor, candidate


def _optimizer(candidate_state, count):
    shapes = [tuple(candidate_state[name].shape) for name in supervisor.R2_OPTIMIZER_NAMES[:6]]
    shapes.extend(((4, 2), (4,)))
    state = {
        index: {
            "step": torch.tensor(float(count * 4)),
            "exp_avg": torch.zeros(shape),
            "exp_avg_sq": torch.zeros(shape),
        }
        for index, shape in enumerate(shapes)
    }
    common = {
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
    }
    return {
        "state": state,
        "param_groups": [
            {
                **common,
                "name": "estimator_encoder_and_mu",
                "lr": 1.0e-4,
                "params": list(range(6)),
            },
            {
                **common,
                "name": "box_output_rows",
                "lr": 1.0e-5,
                "params": [6, 7],
            },
        ],
    }


def _checkpoint_payload(count=4, load_mode="weights_only"):
    anchor_state, candidate_state = _states()
    optimizer = _optimizer(candidate_state, count)
    binding = {
        "stage": "R2",
        "preregistration_path": os.path.realpath(supervisor.PREREGISTRATION),
        "preregistration_sha256": supervisor.PREREGISTRATION_SHA256,
        "protected_student_root": os.path.realpath(supervisor.PROTECTED_ROOT),
        "protected_student_root_sha256": supervisor.PROTECTED_ROOT_SHA256,
        "initial_student_checkpoint": os.path.realpath(supervisor.B500),
        "initial_student_sha256": supervisor.B500_SHA256,
        "teacher_checkpoint": os.path.realpath(supervisor.TEACHER),
        "teacher_sha256": supervisor.TEACHER_SHA256,
        "teacher_env_yaml": os.path.realpath(supervisor.TEACHER_ENV),
        "teacher_env_yaml_sha256": supervisor.TEACHER_ENV_SHA256,
        "schedule_resume_mode": "preserve",
        "checkpoint_load_mode": load_mode,
        "effective_update_count": count,
        "optimizer_parameter_tensor_count": 8,
        "optimizer_parameter_names": list(supervisor.R2_OPTIMIZER_NAMES),
        "trainable_live_tensors": list(supervisor.R2_OPTIMIZER_NAMES[:6]),
        "materialized_actor_rows": [12, 13, 14, 15],
        "teacher_independent_storage": True,
        "anchor_independent_storage": True,
        "actor_prefix_equivalence_max_error": 0.0,
        "runtime_contract": {
            "joint_order": list(supervisor.JOINT_ORDER),
            "action_scale": list(supervisor.ACTION_SCALE),
            "action_offset": list(supervisor.ACTION_OFFSET),
            "joint_pos_clip": [[-60.0, 60.0] for _ in range(16)],
            "teacher_context_shape": [3],
            "teacher_context_order": ["height_delta", "command_x", "front_rear_delta"],
        },
    }
    recovery = {
        "schema_version": 1,
        "stage": "R2",
        "effective_update_count": count,
        "preregistration_sha256": supervisor.PREREGISTRATION_SHA256,
        "optimizer_parameter_names": list(supervisor.R2_OPTIMIZER_NAMES),
        "box_weight": candidate_state["actor.6.weight"][12:16].clone(),
        "box_bias": candidate_state["actor.6.bias"][12:16].clone(),
        "binding_manifest": binding,
    }
    extra = {
        "schema_version": 1,
        "algorithm_class": "VAEPPO",
        "distill_stage": 2,
        "student_distill_update_count": count,
        "vae_optimizer_state_dict": copy.deepcopy(optimizer),
        "student_recovery": recovery,
    }
    anchor_payload = {"model_state_dict": anchor_state}
    candidate_payload = {
        "model_state_dict": candidate_state,
        "optimizer_state_dict": optimizer,
        "infos": {supervisor.ALGORITHM_STATE_KEY: extra},
    }
    return anchor_payload, candidate_payload


def test_exact_preregistered_behavior_gates():
    assert supervisor.gate_100(_summary())[0]
    assert not supervisor.gate_100(_summary(front=6))[0]
    assert supervisor.gate_300(_summary(full=5, hold=3, no_severe=7))[0]
    assert not supervisor.gate_300(_summary(full=5, hold=5, no_severe=6))[0]
    assert supervisor.gate_500(_summary(full=6, hold=6))[0]
    assert not supervisor.gate_500(_summary(full=6, hold=5))[0]
    assert supervisor.behavior_final_gate(_summary(full=8, hold=8, no_severe=8))
    assert not supervisor.behavior_final_gate(_summary(full=8, hold=7, no_severe=9))


def test_probe3_and_strict_lexicographic_gate():
    rows = [
        {"seed": 11, "scenario": "nominal", "valid": True, "full_climb": True,
         "rear_hold": True, "front_top_support": True, "no_severe_inward": True},
        {"seed": 11, "scenario": "left_offset", "valid": True, "full_climb": False,
         "rear_hold": False, "front_top_support": True, "no_severe_inward": True},
        {"seed": 22, "scenario": "right_offset", "valid": True, "full_climb": False,
         "rear_hold": False, "front_top_support": False, "no_severe_inward": False},
    ]
    assert supervisor.gate_probe3(rows)[0]
    rows[0]["rear_hold"] = False
    assert not supervisor.gate_probe3(rows)[0]
    assert supervisor.gate_enter_1000(
        _summary(full=4, hold=4),
        _summary(full=5, hold=4),
        _summary(full=5, hold=5),
    )[0]
    assert not supervisor.gate_enter_1000(
        _summary(full=4, hold=4),
        _summary(full=5, hold=4),
        _summary(full=5, hold=4),
    )[0]


def test_scope_audit_rejects_any_frozen_or_revolute_row_delta():
    anchor, candidate = _states()
    result = supervisor.audit_model_scope(anchor, candidate)
    assert result["frozen_tensor_violation_count"] == 0
    escaped = copy.deepcopy(candidate)
    escaped["critic.0.weight"][0, 0] = 1.0
    with pytest.raises(RuntimeError, match="frozen"):
        supervisor.audit_model_scope(anchor, escaped)
    escaped = copy.deepcopy(candidate)
    escaped["actor.6.weight"][11, 0] = 1.0
    with pytest.raises(RuntimeError, match="actor row"):
        supervisor.audit_model_scope(anchor, escaped)


def test_checkpoint_audit_binds_scope_adam_count_and_full_resume():
    anchor, checkpoint4 = _checkpoint_payload(4, "weights_only")
    audit4 = supervisor.audit_r2_checkpoint_payload(
        anchor, checkpoint4, expected_count=4, expected_load_mode="weights_only"
    )
    assert audit4["adam"]["optimizer_parameter_tensor_count"] == 8
    anchor, checkpoint5 = _checkpoint_payload(5, "full")
    audit5 = supervisor.audit_r2_checkpoint_payload(
        anchor, checkpoint5, expected_count=5, expected_load_mode="full"
    )
    assert supervisor.audit_smoke_resume(audit4, audit5)["adam_step_continuity_verified"]

    tampered = copy.deepcopy(checkpoint5)
    tampered["optimizer_state_dict"]["state"][0]["exp_avg"][0, 0] = float("nan")
    tampered["infos"][supervisor.ALGORITHM_STATE_KEY]["vae_optimizer_state_dict"] = copy.deepcopy(
        tampered["optimizer_state_dict"]
    )
    with pytest.raises(RuntimeError, match="Adam"):
        supervisor.audit_r2_checkpoint_payload(
            anchor, tampered, expected_count=5, expected_load_mode="full"
        )
    malformed = copy.deepcopy(checkpoint5)
    for location in (
        malformed["optimizer_state_dict"],
        malformed["infos"][supervisor.ALGORITHM_STATE_KEY]["vae_optimizer_state_dict"],
    ):
        location["state"][0]["step"] = torch.tensor(20.5)
    with pytest.raises(RuntimeError, match="non-integral"):
        supervisor.audit_r2_checkpoint_payload(
            anchor, malformed, expected_count=5, expected_load_mode="full"
        )


def test_main_launch_latch_requires_read_only_fresh_bound_evidence(tmp_path, monkeypatch):
    static = tmp_path / "static.json"
    smoke = tmp_path / "smoke.json"
    monitor = tmp_path / "monitor.sh"
    static.write_text("{}\n")
    smoke.write_text("{}\n")
    monitor.write_text("#!/bin/sh\n")
    static.chmod(0o444)
    smoke.chmod(0o444)
    monitor.chmod(0o444)
    launch = tmp_path / "launch.json"
    critical = {"source": "digest"}
    payload = {
        "kind": "highstep_student_recovery_r2_main_launch_authorization",
        "workflow_id": supervisor.WORKFLOW_ID,
        "spec_sha256": supervisor.SPEC_SHA256,
        "preregistration_sha256": supervisor.PREREGISTRATION_SHA256,
        "preregistration_training_allowed_latch": False,
        "static_passed": True,
        "smoke_passed": True,
        "smoke_discarded": True,
        "main_training_authorized": True,
        "threshold_changes_allowed": False,
        "critical_file_sha256": critical,
        "static_evidence": str(static),
        "static_evidence_sha256": supervisor.sha256_file(static),
        "smoke_evidence": str(smoke),
        "smoke_evidence_sha256": supervisor.sha256_file(smoke),
        "r2_monitor_snapshot": str(monitor),
        "r2_monitor_snapshot_sha256": supervisor.sha256_file(monitor),
        "only_schedule_task_alias": {
            "source_train_task": supervisor.TRAIN_TASK,
            "eval_task": supervisor.EVAL_TASK,
        },
        "schema4_monitor_source_sha256": supervisor.sha256_file(supervisor.SCHEMA4_MONITOR),
        "pinned_b500_teacher_action_reference": supervisor.pinned_base3_reference_binding(),
    }
    launch.write_text(json.dumps(payload))
    launch.chmod(0o444)
    instance = object.__new__(supervisor.R2Supervisor)
    instance.main_launch_path = launch
    instance.critical_hashes = critical
    monkeypatch.setattr(instance, "assert_code_unchanged", lambda: None)
    assert instance.assert_main_training_authorized()["main_training_authorized"] is True
    launch.chmod(0o644)
    with pytest.raises(RuntimeError, match="absent or writable"):
        instance.assert_main_training_authorized()


def test_preregistration_is_an_immutable_pre_main_latch():
    # R2 has terminated and the formal spec has advanced to v1.2.  The old
    # preregistration remains immutable, and the SHA divergence is now the
    # fail-closed tombstone that forbids relaunching R2.
    assert supervisor.sha256_file(supervisor.SPEC) != supervisor.SPEC_SHA256
    assert supervisor.sha256_file(supervisor.PREREGISTRATION) == supervisor.PREREGISTRATION_SHA256
    prereg = json.loads(supervisor.PREREGISTRATION.read_text())
    assert prereg["training_allowed"] is False
    assert supervisor.is_read_only(supervisor.PREREGISTRATION)


def test_smoke_log_requires_exact_buffer_24_to_zero(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        "Debug/R2_Buffer_Step_Before_Clear 24.0\n"
        "Debug/R2_Buffer_Step_After_Clear 0.0\n"
    )
    evidence = supervisor.R2Supervisor._audit_smoke_log(log)
    assert evidence["buffer_step_before_clear"] == 24
    assert evidence["buffer_step_after_clear"] == 0
    log.write_text("Debug/R2_Buffer_Step_Before_Clear 23.0\n")
    with pytest.raises(RuntimeError, match="24 before update"):
        supervisor.R2Supervisor._audit_smoke_log(log)


def test_schema4_canonical_helper_contains_only_the_exact_approved_r2_task_pair():
    source = supervisor.SCHEMA4_MONITOR.read_text()
    rendered = supervisor.render_r2_monitor_source(source)
    assert rendered == source
    from tools.highstep_core9_task_compat import source_task_is_compatible
    assert source_task_is_compatible(
        workflow_id=supervisor.WORKFLOW_ID,
        role="student",
        source_task=supervisor.TRAIN_TASK,
        eval_task=supervisor.EVAL_TASK,
    )
    assert not source_task_is_compatible(
        workflow_id="wrong_workflow",
        role="student",
        source_task=supervisor.TRAIN_TASK,
        eval_task=supervisor.EVAL_TASK,
    )
    with pytest.raises(RuntimeError, match="compatibility block changed"):
        supervisor.render_r2_monitor_source(source.replace("source_authority_is_valid", "changed", 2))


def test_candidate_action_mae_uses_exact_three_by_600_matrix(tmp_path):
    paths = []
    frame = {
        "action_error_student_minus_teacher": [0.2] * 12 + [0.6] * 4,
        "runner_manual_action_allclose": True,
        "physical_state_unchanged_by_forward": True,
        "observation_and_prior_inputs_unchanged_by_forward": True,
        "causal_replacement_applied": False,
        "action_decomposition_residual_max_abs": 0.0,
    }
    for index in range(3):
        path = tmp_path / f"frames_{index}.jsonl"
        path.write_text("".join(json.dumps(frame) + "\n" for _ in range(600)))
        paths.append(path)
    result = supervisor.aggregate_action_mae(paths)
    assert result["frame_count"] == 1800
    assert result["all12_revolute_mae"] == pytest.approx(0.2)
    assert result["all4_box_mae"] == pytest.approx(0.6)
    paths[0].write_text(json.dumps(frame) + "\n")
    with pytest.raises(RuntimeError, match="expected 600"):
        supervisor.aggregate_action_mae(paths)


def test_probe_command_and_background_unit_bind_the_dedicated_tasks():
    command = supervisor.probe_play_command(
        Path("/tmp/model_49.pt"), seed=11, lateral="0.12", yaw="4.0"
    )
    assert command[command.index("--task") + 1] == supervisor.EVAL_TASK
    assert command[command.index("--front_step_eval_lateral_offset") + 1] == "0.12"
    assert command[command.index("--front_step_eval_yaw_offset_deg") + 1] == "4.0"
    unit = (
        supervisor.ROOT / "scripts/systemd/highstep-student-recovery-r2.service"
    ).read_text()
    assert "tools/highstep_student_recovery_r2_supervisor.py" in unit
    assert "KillMode=control-group" in unit
    assert "TimeoutStopSec=75" in unit


def test_completed_invalid_core9_can_never_be_classified_as_retryable(tmp_path):
    checkpoint = tmp_path / "model_99.pt"
    checkpoint.write_bytes(b"checkpoint")
    rows = [
        {
            "seed": seed,
            "scenario": scenario,
            "checkpoint": str(checkpoint),
        }
        for seed in (11, 22, 33)
        for scenario in ("nominal", "left_offset", "right_offset")
    ]
    (tmp_path / "eval_runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    assert supervisor.core9_matrix_was_fully_executed(tmp_path, checkpoint)
    (tmp_path / "eval_runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[:-1])
    )
    assert not supervisor.core9_matrix_was_fully_executed(tmp_path, checkpoint)


def test_gpu_job_detection_matches_entrypoint_tokens_not_shell_command_text():
    train = "/home/lxq/Softwares/robot_lab/scripts/rsl_rl/base/train.py"
    play = "scripts/rsl_rl/base/play.py"
    audit = "scripts/rsl_rl/base/highstep_candidate_same_state_audit_play.py"
    monitor = "/home/lxq/Softwares/robot_lab/tmp/highstep_centerline_guard_monitor_20260709.sh"

    assert supervisor.R2Supervisor._is_managed_gpu_job_argv(
        ["/usr/bin/python", "-u", train, "--headless"]
    )
    assert supervisor.R2Supervisor._is_managed_gpu_job_argv(
        ["/usr/bin/python", play, "--task", supervisor.EVAL_TASK]
    )
    assert supervisor.R2Supervisor._is_managed_gpu_job_argv(["/usr/bin/python", audit])
    assert supervisor.R2Supervisor._is_managed_gpu_job_argv(["/bin/bash", monitor])

    status_command = (
        "pgrep -af 'scripts/rsl_rl/base/train.py|"
        "highstep_candidate_same_state_audit_play.py' || true"
    )
    assert not supervisor.R2Supervisor._is_managed_gpu_job_argv(
        ["/bin/bash", "-lc", status_command]
    )
    assert not supervisor.R2Supervisor._is_managed_gpu_job_argv(
        ["rg", "train.py", "tools/highstep_student_recovery_r2_supervisor.py"]
    )


def test_legacy_schema_reader_imports_in_isolated_direct_script_context(tmp_path):
    legacy_path = supervisor.ROOT / "tools/highstep_student_recovery_supervisor.py"
    code = """
import importlib.util
from pathlib import Path
import sys

r2_path = Path(sys.argv[1])
legacy_path = Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location("isolated_r2_supervisor", r2_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
legacy = module.R2Supervisor._configure_legacy_schema4_reader()
assert Path(legacy.__file__).resolve() == legacy_path
print("legacy-reader-import-ok")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(Path(supervisor.__file__).resolve()),
            str(legacy_path),
        ],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "legacy-reader-import-ok"
