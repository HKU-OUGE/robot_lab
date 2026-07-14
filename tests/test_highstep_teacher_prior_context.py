"""CPU-only contract tests for the R2 training-only Teacher context."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tools import highstep_same_state_audit as audit


ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS_PATH = (
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/observations.py"
)
HIGHSTEP_CFG_PATH = ROOT / (
    "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped/"
    "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
TASK_INIT_PATH = HIGHSTEP_CFG_PATH.parent / "__init__.py"
SCHEDULE_PATH = (
    ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/highstep_schedule.py"
)
ACTIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/locomotion/velocity/mdp/actions.py"
PREREG_PATH = ROOT / "tmp/highstep_student_recovery_v11_20260712/r2_preregistration.json"

R2_TASK = (
    "RobotLab-Isaac-Velocity-HighstepActionScoreRobustStudentNoPriorR2-"
    "ArcdogAdjustableLeg-v0"
)
R2_ENV_CFG = "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorR2EnvCfg"
R2_RUNNER_CFG = "ArclabArcdogAdjustableLegHighstepActionScoreStudentNoPriorR2PPORunnerCfg"
PREREG_SHA256 = "36d39316f8fbba407899a14d1d659873f75b33423464f01ea92000e588c56fc5"
ACTIONS_SHA256 = "7f1d340cf53382a7fb3b6281a144e1a75546510aa6fe3078a107d3236ebeb54a"


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env

    @property
    def num_envs(self) -> int:
        return self._env.num_envs


def _load_context_term():
    tree = ast.parse(OBSERVATIONS_PATH.read_text(encoding="utf-8"), filename=str(OBSERVATIONS_PATH))
    future = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )
    context_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HighstepTeacherPriorContext"
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, context_node], type_ignores=[]))
    namespace = {"ManagerTermBase": _ManagerTermBase, "torch": torch}
    exec(compile(module, str(OBSERVATIONS_PATH), "exec"), namespace)
    return namespace["HighstepTeacherPriorContext"]


CONTEXT_TERM = _load_context_term()


class _CommandManager:
    def __init__(self, command: torch.Tensor):
        self.command = command

    def get_command(self, name: str) -> torch.Tensor:
        if name != "base_velocity":
            raise KeyError(name)
        return self.command


def _fixture(num_envs: int = 2):
    ray_starts = torch.tensor(
        [
            [0.30, 0.00, 20.0],
            [0.40, 0.10, 20.0],
            [-0.25, 0.00, 20.0],
            [-0.30, 0.10, 20.0],
            [0.00, 0.40, 20.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    ray_hits_w = torch.zeros(num_envs, ray_starts.shape[1], 3, dtype=torch.float32)
    sensor_pos_w = torch.zeros(num_envs, 3, dtype=torch.float32)
    sensor_pos_w[:, 2] = 2.0
    sensor = SimpleNamespace(
        ray_starts=ray_starts,
        data=SimpleNamespace(ray_hits_w=ray_hits_w, pos_w=sensor_pos_w),
    )
    body_pos_w = torch.zeros(num_envs, 4, 3, dtype=torch.float32)
    asset = SimpleNamespace(data=SimpleNamespace(body_pos_w=body_pos_w))
    command = torch.zeros(num_envs, 3, dtype=torch.float32)
    env = SimpleNamespace(
        num_envs=num_envs,
        scene={"height_scanner": sensor, "robot": asset},
        command_manager=_CommandManager(command),
    )
    sensor_cfg = SimpleNamespace(name="height_scanner")
    foot_asset_cfg = SimpleNamespace(
        name="robot",
        body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        body_ids=[0, 1, 2, 3],
    )
    params = {
        "sensor_cfg": sensor_cfg,
        "foot_asset_cfg": foot_asset_cfg,
        "command_name": "base_velocity",
        "front_x_min": 0.25,
        "rear_x_max": -0.20,
        "max_abs_y": 0.30,
    }
    cfg = SimpleNamespace(params=params)
    return env, cfg


def _call(term, env, cfg):
    return term(env, **cfg.params)


class HighstepTeacherPriorContextTests(unittest.TestCase):
    def test_exact_three_column_order_and_live_sources(self) -> None:
        env, cfg = _fixture()
        ray_z = env.scene["height_scanner"].data.ray_hits_w[..., 2]
        ray_z[0] = torch.tensor([0.35, 0.33, 0.01, 0.03, 9.0])
        ray_z[1] = torch.tensor([float("inf"), 0.25, float("nan"), 0.05, 9.0])
        env.command_manager.command[:, 0] = torch.tensor([0.40, 0.20])
        body_z = env.scene["robot"].data.body_pos_w[..., 2]
        body_z[0] = torch.tensor([0.40, 0.44, 0.10, 0.14])
        body_z[1] = torch.tensor([0.30, 0.20, 0.15, 0.05])

        context = _call(CONTEXT_TERM(cfg, env), env, cfg)

        ray_starts = env.scene["height_scanner"].ray_starts[0]
        side = torch.abs(ray_starts[:, 1]) <= audit.PostPriorContract.max_abs_y
        front_mask = (ray_starts[:, 0] >= audit.PostPriorContract.front_x_min) & side
        rear_mask = (ray_starts[:, 0] <= audit.PostPriorContract.rear_x_max) & side
        sensor_z = env.scene["height_scanner"].data.pos_w[:, 2]
        audit_front = audit.HighstepRuntimeAdapter._masked_mean(ray_z, front_mask, sensor_z)
        audit_rear = audit.HighstepRuntimeAdapter._masked_mean(ray_z, rear_mask, audit_front)
        audit_context = torch.stack(
            (
                audit_front - audit_rear,
                env.command_manager.command[:, 0],
                body_z[:, :2].mean(dim=1) - body_z[:, 2:].mean(dim=1),
            ),
            dim=-1,
        )

        expected = torch.tensor(
            [
                [0.32, 0.40, 0.30],
                [0.20, 0.20, 0.15],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(tuple(context.shape), (2, 3))
        self.assertTrue(torch.equal(context, audit_context))
        torch.testing.assert_close(context, expected, rtol=0.0, atol=1.0e-7)

    def test_all_invalid_hits_keep_exact_teacher_fallback(self) -> None:
        env, cfg = _fixture(num_envs=1)
        ray_z = env.scene["height_scanner"].data.ray_hits_w[..., 2]
        ray_z[:] = float("inf")
        ray_z[:, 2:] = float("nan")
        env.scene["height_scanner"].data.pos_w[:, 2] = 1.75
        env.command_manager.command[:, 0] = 0.31
        env.scene["robot"].data.body_pos_w[0, :, 2] = torch.tensor([0.4, 0.2, 0.1, 0.1])

        context = _call(CONTEXT_TERM(cfg, env), env, cfg)

        torch.testing.assert_close(
            context,
            torch.tensor([[0.0, 0.31, 0.20]]),
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_structural_contract_errors_fail_closed_at_initialization(self) -> None:
        cases = []

        env, cfg = _fixture(num_envs=1)
        cfg.params["command_name"] = "wrong_command"
        cases.append((env, cfg, "command changed"))

        env, cfg = _fixture(num_envs=1)
        cfg.params["front_x_min"] = 0.24
        cases.append((env, cfg, "ray-mask contract changed"))

        env, cfg = _fixture(num_envs=1)
        cfg.params["foot_asset_cfg"].body_names = ["FR_foot", "FL_foot", "RL_foot", "RR_foot"]
        cases.append((env, cfg, "permanent order"))

        env, cfg = _fixture(num_envs=1)
        env.scene["height_scanner"].ray_starts[..., 0] = 0.0
        cases.append((env, cfg, "structurally empty"))

        env, cfg = _fixture(num_envs=1)
        del env.scene["height_scanner"]
        cases.append((env, cfg, "scene binding"))

        for env, cfg, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    CONTEXT_TERM(cfg, env)

    def test_malformed_or_nonfinite_live_data_fails_closed(self) -> None:
        env, cfg = _fixture(num_envs=1)
        term = CONTEXT_TERM(cfg, env)
        env.command_manager.command = torch.tensor([[float("nan"), 0.0, 0.0]])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _call(term, env, cfg)

        env, cfg = _fixture(num_envs=1)
        term = CONTEXT_TERM(cfg, env)
        env.scene["height_scanner"].data.ray_hits_w = torch.zeros(1, 4, 3)
        with self.assertRaisesRegex(RuntimeError, "ray-hit shape"):
            _call(term, env, cfg)

        env, cfg = _fixture(num_envs=1)
        term = CONTEXT_TERM(cfg, env)
        env.scene["height_scanner"].data.ray_hits_w[..., 2] = float("inf")
        env.scene["height_scanner"].data.pos_w[:, 2] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _call(term, env, cfg)

    def test_r2_config_is_isolated_and_matches_frozen_teacher_contract(self) -> None:
        tree = ast.parse(HIGHSTEP_CFG_PATH.read_text(encoding="utf-8"), filename=str(HIGHSTEP_CFG_PATH))
        names = {
            "ArcdogAdjustableLegHighstepR2TeacherContextCfg",
            "ArcdogAdjustableLegHighstepR2ObservationsCfg",
            R2_ENV_CFG,
        }
        selected = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names
        ]
        self.assertEqual({node.name for node in selected}, names)

        class _ObsTerm:
            def __init__(self, *, func, params, **kwargs):
                self.func = func
                self.params = params
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _SceneEntityCfg:
            def __init__(self, name, **kwargs):
                self.name = name
                for key, value in kwargs.items():
                    setattr(self, key, value)

        marker = object()
        namespace = {
            "configclass": lambda cls: cls,
            "ObsGroup": type("ObsGroup", (), {}),
            "ObsTerm": _ObsTerm,
            "SceneEntityCfg": _SceneEntityCfg,
            "ObservationsCfg": type("ObservationsCfg", (), {}),
            "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg": type(
                "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg",
                (),
                {},
            ),
            "mdp": SimpleNamespace(HighstepTeacherPriorContext=marker),
        }
        future = ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
        module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
        exec(compile(module, str(HIGHSTEP_CFG_PATH), "exec"), namespace)

        group = namespace["ArcdogAdjustableLegHighstepR2TeacherContextCfg"]()
        group.__post_init__()
        term = group.prior_context
        self.assertIs(term.func, marker)
        self.assertEqual(term.history_length, 0)
        self.assertFalse(group.enable_corruption)
        self.assertTrue(group.concatenate_terms)
        self.assertEqual(group.history_length, 0)
        self.assertEqual(term.params["sensor_cfg"].name, "height_scanner")
        self.assertEqual(term.params["command_name"], "base_velocity")
        self.assertEqual(term.params["front_x_min"], 0.25)
        self.assertEqual(term.params["rear_x_max"], -0.20)
        self.assertEqual(term.params["max_abs_y"], 0.30)
        self.assertEqual(
            term.params["foot_asset_cfg"].body_names,
            ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        )
        self.assertTrue(term.params["foot_asset_cfg"].preserve_order)

        old_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name
            == "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg"
        )
        self.assertNotIn("teacher_context", ast.get_source_segment(HIGHSTEP_CFG_PATH.read_text(), old_class))
        r2_class = next(node for node in selected if node.name == R2_ENV_CFG)
        self.assertIn(
            "ArclabArcdogAdjustableLegHighstepActionScoreRobustStudentNoPriorEnvCfg",
            [ast.unparse(base) for base in r2_class.bases],
        )

    def test_r2_task_registration_and_robust_lineage_are_explicit(self) -> None:
        tree = ast.parse(TASK_INIT_PATH.read_text(encoding="utf-8"), filename=str(TASK_INIT_PATH))
        registrations = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "gym":
                continue
            if node.func.attr != "register":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if ast.literal_eval(keywords["id"]) == R2_TASK:
                registrations.append(keywords)
        self.assertEqual(len(registrations), 1)
        registration_source = ast.unparse(registrations[0]["kwargs"])
        self.assertIn(R2_ENV_CFG, registration_source)
        self.assertIn(R2_RUNNER_CFG, registration_source)

        spec = importlib.util.spec_from_file_location("r2_context_schedule", SCHEDULE_PATH)
        assert spec is not None and spec.loader is not None
        schedule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schedule)
        self.assertIn(R2_TASK, schedule._ROBUST_STUDENT_TASKS)

        captured = []
        lineage = {"sentinel": "lineage"}

        def normalize(value, *, require_robust):
            captured.append(require_robust)
            return dict(value)

        with mock.patch.object(schedule, "_normalize_student_parent_lineage", side_effect=normalize):
            result = schedule.resolve_student_parent_lineage(
                task=R2_TASK,
                checkpoint_path="unused_student_checkpoint.pt",
                parent_teacher_manifest_path=None,
                source_manifest={"task": R2_TASK, "student_parent_lineage": lineage},
            )
        self.assertEqual(result, lineage)
        self.assertEqual(captured, [True])

    def test_immutable_prereg_and_shared_actions_bindings(self) -> None:
        self.assertEqual(hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(hashlib.sha256(ACTIONS_PATH.read_bytes()).hexdigest(), ACTIONS_SHA256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
