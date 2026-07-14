"""CPU-only tests for the R2 train-time authority and runtime binding gates."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts/rsl_rl/base/train.py"


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract_namespace() -> dict:
    """Execute only the pure R2 helpers; importing train.py would launch Isaac Sim."""
    syntax = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    constants = {
        "_R2_PREREGISTRATION_PATH",
        "_R2_PREREGISTRATION_SHA256",
        "_R2_SPEC_PATH",
        "_R2_SPEC_SHA256",
    }
    functions = {
        "_load_r2_preregistration_contract",
        "_validate_r2_load_request",
        "_assert_r2_action_config_contract",
        "_r2_tensor_rows",
        "_assert_r2_runtime_contract",
    }
    selected = []
    for node in syntax.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in constants for target in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    namespace = {
        "json": json,
        "os": os,
        "torch": torch,
        "checkpoint_sha256": _sha256,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return namespace


CONTRACT = _load_contract_namespace()
PREREGISTRATION_PATH = CONTRACT["_R2_PREREGISTRATION_PATH"]
PREREGISTRATION_SHA256 = CONTRACT["_R2_PREREGISTRATION_SHA256"]


class _TermCfg:
    def __init__(self, history_length: int) -> None:
        self.history_length = history_length
        self.flatten_history_dim = True


class _History:
    max_length = 10


class _ObservationManager:
    def __init__(self, num_envs: int) -> None:
        history_terms = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        critic_terms = [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
            "height_scan",
        ]
        self._group_obs_term_names = {
            "policy": list(history_terms),
            "estimator": list(history_terms),
            "critic": critic_terms,
            "teacher_context": ["prior_context"],
        }
        history_dims = [(30,), (30,), (30,), (160,), (160,), (160,)]
        self._group_obs_term_dim = {
            "policy": list(history_dims),
            "estimator": list(history_dims),
            "critic": [(3,), (3,), (3,), (3,), (16,), (16,), (16,), (102,)],
            "teacher_context": [(3,)],
        }
        self._group_obs_term_cfgs = {
            "policy": [_TermCfg(10) for _ in history_terms],
            "estimator": [_TermCfg(10) for _ in history_terms],
            "critic": [_TermCfg(0) for _ in critic_terms],
            "teacher_context": [_TermCfg(0)],
        }
        self._group_obs_dim = {
            "policy": (570,),
            "estimator": (570,),
            "critic": (162,),
            "teacher_context": (3,),
        }
        self._group_obs_concatenate = {key: True for key in self._group_obs_term_names}
        self._group_obs_term_history_buffer = {
            "policy": {name: _History() for name in history_terms},
            "estimator": {name: _History() for name in history_terms},
            "critic": {},
            "teacher_context": {},
        }
        self.context = torch.zeros(num_envs, 3)

    def compute_group(self, group_name: str, *, update_history: bool = False):
        if group_name != "teacher_context" or update_history:
            raise AssertionError("unexpected observation-manager call")
        return self.context.clone()


def _make_runtime(num_envs: int = 2):
    joint_order = [
        "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
        "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
        "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",
    ]
    scale = torch.tensor([[0.1] * 12 + [0.02] * 4]).repeat(num_envs, 1)
    offset = torch.tensor([[0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4]).repeat(num_envs, 1)
    clip = torch.tensor([[[-60.0, 60.0]] * 16]).repeat(num_envs, 1, 1)
    robot = SimpleNamespace(
        joint_names=joint_order,
        data=SimpleNamespace(default_joint_pos=offset.clone()),
    )
    action_term = SimpleNamespace(
        _joint_names=joint_order,
        _joint_ids=list(range(16)),
        action_dim=16,
        _asset=robot,
        _scale=scale,
        _offset=offset.clone(),
        _clip=clip,
    )
    observation_manager = _ObservationManager(num_envs)
    raw_env = SimpleNamespace(
        num_envs=num_envs,
        action_manager=SimpleNamespace(_terms={"joint_pos": action_term}, total_action_dim=16),
        observation_manager=observation_manager,
        scene={"robot": robot},
    )
    env = SimpleNamespace(unwrapped=raw_env)
    policy = SimpleNamespace(
        obs_groups={
            "policy": ["policy"],
            "estimator": ["estimator"],
            "critic": ["critic"],
            "teacher_context": ["teacher_context"],
        },
        policy_keys=["policy"],
        estimator_keys=["estimator"],
        critic_keys=["critic"],
        teacher_context_keys=["teacher_context"],
        policy_obs_dim=570,
    )
    return env, policy, action_term, observation_manager


def _policy_binding() -> SimpleNamespace:
    return SimpleNamespace(
        student_recovery_r2_preregistration_path=PREREGISTRATION_PATH,
        student_recovery_r2_preregistration_sha256=PREREGISTRATION_SHA256,
    )


class HighstepStudentRecoveryR2TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # R2 is a completed, terminal route.  Its immutable preregistration is
        # still useful to exercise the pure load/runtime gates, while the live
        # formal spec has legitimately advanced to v1.2 and must prevent any
        # new R2 launch.
        cls.preregistration = json.loads(Path(PREREGISTRATION_PATH).read_text())

    def test_immutable_preregistration_spec_and_teacher_yaml_are_bound(self) -> None:
        self.assertEqual(PREREGISTRATION_SHA256, _sha256(PREREGISTRATION_PATH))
        self.assertEqual(
            "7dbb47d76b5eafab0e425be7ac20b5a0e9487466c6512831600f22fb7509bbe5",
            self.preregistration["authority"]["spec_sha256"],
        )
        self.assertEqual(
            "f8aa66bbc417eb67a7889900e4a1781eb2852cb3d94432a1282963cc0909c009",
            self.preregistration["checkpoint_binding"]["frozen_teacher"]["env_yaml_sha256"],
        )
        self.assertEqual(0, os.stat(PREREGISTRATION_PATH).st_mode & 0o222)
        with self.assertRaisesRegex(RuntimeError, "R2 spec SHA256 mismatch"):
            CONTRACT["_load_r2_preregistration_contract"](_policy_binding())

    def test_wrong_policy_preregistration_sha_fails_closed(self) -> None:
        policy = _policy_binding()
        policy.student_recovery_r2_preregistration_sha256 = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "policy preregistration SHA256"):
            CONTRACT["_load_r2_preregistration_contract"](policy)

    def test_r2_load_modes_are_strict(self) -> None:
        binding = self.preregistration["checkpoint_binding"]
        b500 = binding["behavior_start_and_anchor"]["path"]
        validate = CONTRACT["_validate_r2_load_request"]
        validate(
            self.preregistration,
            loaded_checkpoint=b500,
            checkpoint_load_mode="weights_only",
            schedule_resume_mode="preserve",
            resume_enabled=True,
            legacy_student_count=None,
        )
        with self.assertRaisesRegex(RuntimeError, "preregistered B500"):
            validate(
                self.preregistration,
                loaded_checkpoint=binding["protected_student_root"]["path"],
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="preserve",
                resume_enabled=True,
                legacy_student_count=None,
            )
        with self.assertRaisesRegex(RuntimeError, "requires an R2 checkpoint"):
            validate(
                self.preregistration,
                loaded_checkpoint=b500,
                checkpoint_load_mode="full",
                schedule_resume_mode="preserve",
                resume_enabled=True,
                legacy_student_count=None,
            )
        with self.assertRaisesRegex(RuntimeError, "schedule preserve"):
            validate(
                self.preregistration,
                loaded_checkpoint=b500,
                checkpoint_load_mode="weights_only",
                schedule_resume_mode="reset",
                resume_enabled=True,
                legacy_student_count=None,
            )

    def test_initialized_runtime_contract_is_exact_and_json_compatible(self) -> None:
        env, policy, _, _ = _make_runtime()
        runtime = CONTRACT["_assert_r2_runtime_contract"](
            env, policy, self.preregistration
        )
        self.assertEqual([0.1] * 12 + [0.02] * 4, runtime["action_scale"])
        self.assertEqual([0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4, runtime["action_offset"])
        self.assertEqual([[-60.0, 60.0]] * 16, runtime["joint_pos_clip"])
        self.assertEqual([3], runtime["teacher_context_shape"])
        self.assertEqual(
            ["height_delta", "command_x", "front_rear_delta"],
            runtime["teacher_context_order"],
        )
        self.assertFalse(runtime["policy_estimator_critic_input"])
        json.dumps(runtime, sort_keys=True)

    def test_context_leak_nonfinite_context_and_offset_drift_fail_closed(self) -> None:
        check = CONTRACT["_assert_r2_runtime_contract"]

        env, policy, _, _ = _make_runtime()
        policy.obs_groups["policy"].append("teacher_context")
        with self.assertRaisesRegex(RuntimeError, "observation routing"):
            check(env, policy, self.preregistration)

        env, policy, _, observations = _make_runtime()
        observations.context[0, 0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            check(env, policy, self.preregistration)

        env, policy, action_term, _ = _make_runtime()
        action_term._offset[0, 0] = 0.1
        with self.assertRaisesRegex(RuntimeError, "action offset"):
            check(env, policy, self.preregistration)

    def test_r2_branch_calls_post_load_binding_without_changing_stage_b_branch(self) -> None:
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('if student_recovery_stage == "B":', source)
        self.assertIn('elif student_recovery_stage == "R2":', source)
        self.assertIn("runner.alg.bind_student_recovery_checkpoints(", source)
        self.assertIn("runner.alg.bind_student_recovery_r2_checkpoints(", source)
        self.assertIn("schedule_resume_mode=highstep_schedule_resolved_mode", source)
        self.assertIn("runtime_contract=r2_runtime_contract", source)
        self.assertIn("highstep_student_recovery_r2_binding.json", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
