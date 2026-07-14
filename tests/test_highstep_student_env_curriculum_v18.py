from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path("/home/lxq/Softwares/robot_lab")
TASKS = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/__init__.py"
)
ENV = TASKS.with_name("highstep_env_cfg.py")
TRAIN = ROOT / "scripts/rsl_rl/base/train.py"
SUPERVISOR = ROOT / "tools/highstep_student_env_curriculum_v18_supervisor.py"
BASE_SUPERVISOR = ROOT / "tools/highstep_student_recovery_v15_supervisor.py"
PROFILES = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714/profiles"
V18_ROOT = ROOT / "tmp/highstep_student_env_curriculum_v18_20260714"
E100_RUN = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student/"
    "2026-07-14_16-37-17_v18_stage_a_E100_20260714_163712"
)
E300_RUN = ROOT / (
    "logs/rsl_rl/arclab_arcdog_adjustable_leg_highstep_action_score_vae_student_"
    "environment_curriculum_v18_Student/"
    "2026-07-14_18-41-20_v18_stage_a_E300_20260714_184115"
)
V18_PREREG = V18_ROOT / "preregistration_v18_12.json"
V18_PREREG_SHA = "fd38dd7e3270fd257c211ca8c23d15d9719cbef3e817dd1281b73f122563fac1"
V18_REBINDING = V18_ROOT / "failure_recovery/e100_checkpoint_scope_rebinding_v8.json"
E300_REBINDING = V18_ROOT / "failure_recovery/e300_checkpoint_scope_rebinding_v2.json"
E300_RECOVERY = V18_ROOT / "failure_recovery/e300_checkpoint_recovery_manifest_v2.json"
VAE_PPO = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/agents/vae_ppo.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v18_python_files_parse() -> None:
    for path in (TASKS, ENV, TRAIN, SUPERVISOR):
        ast.parse(path.read_text())


def test_v18_exact_tasks_and_profile_binding_are_registered() -> None:
    tasks = TASKS.read_text()
    env = ENV.read_text()
    train = TRAIN.read_text()
    for token in (
        "HighstepActionScoreStudentNoPriorV18Bootstrap",
        "HighstepActionScoreStudentNoPriorV18Robust",
        "HighstepActionScoreTeacherV18Bootstrap",
    ):
        assert token in tasks
        assert token in env
    assert "source_snapshot_projection_verified_before_environment_creation" in train
    assert "v1.8 environment task/profile binding mismatch" in train
    assert "v18_environment_transition_effective_update_anchor" in train
    assert "highstep_student_environment_curriculum_v18_binding.json" in train


def test_v18_profiles_bind_exact_saved_yaml_sha() -> None:
    expected = {
        "stage_a_0707_student_bootstrap.json": "f83b0e8d2fd0a388201efe6628b383ca7a3dce1bce8f1b6d88cab9dcefb78636",
        "stage_b_current_robust.json": "61d70655405a49ad8fd72377aed3e193b0d8435898ca1fd39316d08f1989a729",
        "teacher_0707_bootstrap.json": "25ebac11c2bce467fc09ae00471d200b46bb30e5370b7a34aebf942e808da9e7",
    }
    for name, digest in expected.items():
        profile = PROFILES / name
        payload = json.loads(profile.read_text())
        assert payload["immutable"] is True
        assert payload["source_env_yaml_sha256"] == digest
        assert sha256(Path(payload["source_env_yaml"])) == digest
        assert profile.stat().st_mode & 0o222 == 0


def test_v18_supervisor_has_fixed_stage_gates_and_no_deployment() -> None:
    source = SUPERVISOR.read_text()
    assert "STAGE_A_POINTS = (100, 300, 500, 700, 900, 1400)" in source
    assert 'counts["full_climb"] >= 7' in source
    assert 'counts["rear_hold"] >= 7' in source
    assert 'value["counts"]["full_climb"], value["counts"]["rear_hold"]' in source
    assert "student_directional_candidate_pending_user_visual_review" in source
    assert "automatic_real_robot_deployment" not in source
    assert 'base.SCHEDULE_RESUME_MODE = "reset" if previous == 0 else "preserve"' in source
    assert 'base.V18_SCHEDULE_ANCHOR_EFFECTIVE_UPDATE = str(previous)' in source


def test_v18_completed_e100_uses_exact_narrow_checkpoint_scope() -> None:
    from tools.highstep_v18_checkpoint_scope import checkpoint_scope_v18

    scope = checkpoint_scope_v18(
        E100_RUN / "model_99.pt",
        100,
        preregistration=V18_PREREG,
        preregistration_sha256=V18_PREREG_SHA,
        run_dir=E100_RUN,
        authority_rebinding=V18_REBINDING,
    )
    assert scope["workflow_id"] == "highstep_student_env_curriculum_v18_20260714"
    assert scope["checkpoint_sha256"] == "a80247db645250e9d5569f4509061e529c02853534ef233dba77852a3c964e98"
    assert scope["effective_updates"] == 100
    assert scope["forbidden_parameter_names"] == []
    assert scope["optimizer_group_sizes"] == [16, 2]
    assert scope["optimizer_state_entries"] == 16
    assert scope["authority"]["mode"] == "infrastructure_rebinding"
    assert scope["old_v15_helper_authority_used"] is False


def test_v18_completed_e300_uses_effective_and_runtime_schedule_clocks() -> None:
    from tools.highstep_v18_checkpoint_scope import checkpoint_scope_v18

    scope = checkpoint_scope_v18(
        E300_RUN / "model_298.pt",
        300,
        preregistration=V18_PREREG,
        preregistration_sha256=V18_PREREG_SHA,
        run_dir=E300_RUN,
        authority_rebinding=E300_REBINDING,
    )
    assert scope["checkpoint_sha256"] == "79b2d103640153065b6d15d2481f2d97c234265802bfc84d61ed4aed7cfa724a"
    assert scope["effective_updates"] == 300
    assert scope["runner_iteration"] == 298
    assert scope["forbidden_parameter_names"] == []
    assert scope["optimizer_group_sizes"] == [16, 2]
    assert scope["optimizer_state_entries"] == 16
    assert scope["authority"]["mode"] == "infrastructure_rebinding"


def test_v18_e300_recovery_selects_only_a_completed_sha_bound_run() -> None:
    recovery = json.loads(E300_RECOVERY.read_text())
    selected = Path(recovery["selected_checkpoint"])
    interrupted = Path(recovery["interrupted_run_forbidden"])
    assert recovery["changed_training_semantics"] is False
    assert recovery["repeat_e300_training_forbidden"] is True
    assert recovery["effective_updates"] == 300
    assert recovery["selected_checkpoint_sha256"] == sha256(selected)
    assert selected.parent == E300_RUN
    assert interrupted != E300_RUN
    assert not (interrupted / "model_298.pt").exists()
    assert E300_RECOVERY.stat().st_mode & 0o222 == 0


def test_v18_posttrain_scope_hook_preserves_the_frozen_v15_default() -> None:
    base_source = BASE_SUPERVISOR.read_text()
    v18_source = SUPERVISOR.read_text()
    assert "scope = self.checkpoint_scope(checkpoint, stage, run_dir)" in base_source
    assert "return checkpoint_scope(checkpoint, stage)" in base_source
    assert "def checkpoint_scope(self, checkpoint: Path, stage: int, run_dir: Path)" in v18_source
    assert "return checkpoint_scope_v18(" in v18_source


@pytest.mark.parametrize("field,bad_value", [
    ("workflow_id", "wrong_workflow"),
    ("checkpoint_sha256", "0" * 64),
    ("runtime_preregistration_sha256", "1" * 64),
])
def test_v18_checkpoint_scope_rebinding_fails_closed(
    tmp_path: Path, field: str, bad_value: str
) -> None:
    from tools.highstep_v18_checkpoint_scope import checkpoint_scope_v18

    payload = json.loads(V18_REBINDING.read_text())
    payload[field] = bad_value
    bad = tmp_path / "bad_rebinding.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="rebinding changed"):
        checkpoint_scope_v18(
            E100_RUN / "model_99.pt",
            100,
            preregistration=V18_PREREG,
            preregistration_sha256=V18_PREREG_SHA,
            run_dir=E100_RUN,
            authority_rebinding=bad,
        )


def test_v15_checkpoint_scope_authority_remains_narrow_and_unchanged() -> None:
    from tools.highstep_v15_imitation_gate import checkpoint_scope

    helper = ROOT / "tools/highstep_v15_imitation_gate.py"
    assert sha256(helper) == "040b99bbfe11f4bdfccbecc9d79ee93c4740c7550f438a924f047afd42a9c498"
    with pytest.raises(RuntimeError, match="v1.5 checkpoint binding changed"):
        checkpoint_scope(E100_RUN / "model_99.pt", 100)


def _v18_algorithm_with_prereg(payload: dict):
    spec = importlib.util.spec_from_file_location("v18_resume_target", VAE_PPO)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    algorithm = object.__new__(module.VAEPPO)
    algorithm._v15_preregistration = payload
    return algorithm


def test_v18_full_checkpoint_resume_rebinding_accepts_only_selected_e300_for_e500() -> None:
    prereg = json.loads(V18_PREREG.read_text())
    algorithm = _v18_algorithm_with_prereg(prereg)
    result = algorithm._validate_v18_resume_rebinding(
        "38102f2fb963a94410888d533f7df07d15e2f953bf232580873f5ae876c8d84b",
        effective_updates=300,
        loaded_checkpoint=str((E300_RUN / "model_298.pt").resolve()),
    )
    assert result["preserve_optimizer"] is True
    assert result["training_contract_changed"] is False


@pytest.mark.parametrize("mutation,match", [
    (("workflow_id", "wrong_workflow"), "contract mismatch"),
    (("source_checkpoint_sha256", "0" * 64), "checkpoint SHA"),
])
def test_v18_full_checkpoint_resume_rebinding_fails_closed(
    mutation: tuple[str, str], match: str
) -> None:
    prereg = json.loads(V18_PREREG.read_text())
    prereg = copy.deepcopy(prereg)
    prereg["resume_rebinding"][mutation[0]] = mutation[1]
    algorithm = _v18_algorithm_with_prereg(prereg)
    with pytest.raises(ValueError, match=match):
        algorithm._validate_v18_resume_rebinding(
            "38102f2fb963a94410888d533f7df07d15e2f953bf232580873f5ae876c8d84b",
            effective_updates=300,
            loaded_checkpoint=str((E300_RUN / "model_298.pt").resolve()),
        )


def test_v18_full_checkpoint_resume_rebinding_rejects_wrong_source_authority() -> None:
    algorithm = _v18_algorithm_with_prereg(json.loads(V18_PREREG.read_text()))
    with pytest.raises(ValueError, match="contract mismatch"):
        algorithm._validate_v18_resume_rebinding(
            "0" * 64,
            effective_updates=300,
            loaded_checkpoint=str((E300_RUN / "model_298.pt").resolve()),
        )


def test_v18_supervisor_skips_completed_e100_training() -> None:
    source = SUPERVISOR.read_text()
    assert "E100_RECOVERY_MANIFEST" in source
    assert 'recovery.get("repeat_training_forbidden") is True' in source
    assert 'recovery.get("changed_training_semantics") is False' in source
    assert '"repeated_e100_training_skipped": True' in source


def test_v18_core9_monitor_adds_only_the_two_exact_tasks() -> None:
    from tools.highstep_v18_core9_monitor import (
        CANONICAL, NEEDLE, REPLACEMENT, ROBUST_PARENT_NEEDLE,
        ROBUST_PARENT_REPLACEMENT, REUSE_NEEDLE, REUSE_REPLACEMENT,
        STAGE_A_TASK, STAGE_B_TASK, expected_rendered,
    )

    source = CANONICAL.read_text()
    rendered = expected_rendered()
    assert rendered.count(STAGE_A_TASK) == 2
    assert rendered.count(STAGE_B_TASK) == 2
    assert rendered.replace(REPLACEMENT, NEEDLE).replace(
        ROBUST_PARENT_REPLACEMENT, ROBUST_PARENT_NEEDLE
    ).replace(REUSE_REPLACEMENT, REUSE_NEEDLE) == source


def test_v18_core9_monitor_fails_closed_if_canonical_anchor_changes() -> None:
    from tools.highstep_v18_core9_monitor import render_v18_monitor

    with pytest.raises(RuntimeError, match="allowlist anchor changed"):
        render_v18_monitor("case changed in")
