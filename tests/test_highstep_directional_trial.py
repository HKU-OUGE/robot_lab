from pathlib import Path
import importlib.util
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/robotlab_memory_zh/library/highstep_student_recovery_spec_20260712.md"

def load_supervisor():
    path = ROOT / "tools/highstep_directional_trial_supervisor.py"
    spec = importlib.util.spec_from_file_location("directional_supervisor_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_fixed_directional_gate():
    module = load_supervisor()
    rows=[]
    for seed in (11,22,33):
        for scenario in ("nominal","half_left","left_offset"):
            rows.append({"seed":seed,"scenario":scenario,"valid":True,"full_climb":True,
                         "rear_hold":True,"front_top_support":True,"no_severe_inward":True})
    passed, counts = module.Supervisor.gate(rows,"half_left")
    assert passed and counts["full_climb"] == 9 and counts["half_full"] == 3
    rows[0]["full_climb"] = False; rows[1]["full_climb"] = False
    assert module.Supervisor.gate(rows,"half_left")[0] is False

def test_half_offset_is_mandatory():
    module=load_supervisor(); rows=[]
    for seed in (11,22,33):
        for scenario in ("nominal","half_right","right_offset"):
            value=scenario != "half_right" or seed != 33
            rows.append({"seed":seed,"scenario":scenario,"valid":True,"full_climb":value,
                         "rear_hold":value,"front_top_support":True,"no_severe_inward":True})
    assert module.Supervisor.gate(rows,"half_right")[0] is False

def test_spec_preserves_old_result_and_directional_scope():
    text=SPEC.read_text()
    assert "stopped_by_gate" in text
    assert "directional_real_robot_trial_candidate" in text
    assert "不得称为通用鲁棒候选" in text
    assert "禁止自动真机部署" in text

def test_services_pin_model_and_repair_scope():
    repair=(ROOT/"tools/highstep_directional_failure_wake_codex.py").read_text()
    assert '"gpt-5.6-sol"' in repair and 'model_reasoning_effort="max"' in repair
    assert "ultra" in repair
    service=(ROOT/"scripts/systemd/highstep-directional-trial.service").read_text()
    assert "OnFailure=highstep-directional-trial-repair.service" in service
    assert "systemd-inhibit" not in service
    assert "ExecStart=/home/lxq/miniconda3/envs/env_isaaclab/bin/python -u" in service
    assert "Type=oneshot" in service and "RemainAfterExit=yes" in service

def test_directional_validator_does_not_leak_checkpoint_source_task_alias():
    module = load_supervisor()
    original = (module.recovery.TRAIN_TASK, module.recovery.EVAL_TASK)
    try:
        module.validate_directional_probe_payload({}, Path("/missing"))
    except (FileNotFoundError, RuntimeError):
        pass
    assert (module.recovery.TRAIN_TASK, module.recovery.EVAL_TASK) == original

def test_failed_closed_retry_requires_sha_bound_repair_authorization():
    source = (ROOT / "tools/highstep_directional_trial_supervisor.py").read_text()
    assert "failed_handoff_sha256" in source
    assert "repaired_supervisor_sha256" in source
    assert "recorded_payload_revalidated" in source
    assert "full_test_count" in source


def test_candidate_video_delivery_is_robot_camera_bound_and_fail_closed():
    supervisor = (ROOT / "tools/highstep_directional_trial_supervisor.py").read_text()
    play = (ROOT / "scripts/rsl_rl/base/play.py").read_text()
    assert '"--highstep_gap_camera", "side_top"' in supervisor
    assert "video_delivery_qc(destination, log)" in supervisor
    assert "[HIGHSTEP_CAMERA] mode=side_top env=0" in supervisor
    assert "[HIGHSTEP_CAMERA] " in play


def test_user_rejected_terminal_is_preserved_before_obsolete_route_preflight():
    module = load_supervisor()
    # v1.7 intentionally supersedes this authority.  The obsolete route must
    # remain fail-closed instead of rebinding itself to the current spec.
    with pytest.raises(RuntimeError, match="terminal authority mismatch"):
        module.validate_user_rejected_terminal()
    source = (ROOT / "tools/highstep_directional_trial_supervisor.py").read_text()
    terminal_guard = source.index('handoff.get("status") == "user_rejected_non_deployable"')
    obsolete_preflight = source.index("prereg = self.preflight()")
    assert terminal_guard < obsolete_preflight


def test_user_rejected_terminal_fails_closed_on_current_spec_mismatch(monkeypatch):
    module = load_supervisor()
    monkeypatch.setattr(module, "CURRENT_SPEC_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="terminal authority mismatch"):
        module.validate_user_rejected_terminal()
