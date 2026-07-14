"""Static contract tests that do not launch Isaac Sim or a ROS controller."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
HIGHSTEP_CFG = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/locomotion/velocity/config/quadruped"
    / "Arcdog_adjustable_leg/highstep_env_cfg.py"
)
DEPLOYMENT_CFG = (
    Path("/home/lxq/colcon_ws/src/quadruped_control_ros2/robot_description")
    / "arcdog_adjustable_leg_description/config/rl_policy/config.yaml"
)
POLICY = ROOT / "tmp/highstep_automation_policy.json"

EXPECTED_DEPLOYMENT_ENVELOPE = {
    ".*_hip_joint": (-1.2217304763960306, 1.2217304763960306),
    ".*_thigh_joint": (-1.5708, 3.4907),
    ".*_calf_joint": (-2.775073510670984, -0.6457718232379019),
    ".*_box_joint": (0.0, 0.06),
}


class HighstepTargetContractTests(unittest.TestCase):
    def test_training_action_mapping_remains_checkpoint_compatible(self) -> None:
        source = HIGHSTEP_CFG.read_text(encoding="utf-8")
        self.assertNotIn("HIGHSTEP_DEPLOYMENT_TARGET_LIMITS", source)
        self.assertGreaterEqual(source.count('clip={".*": (-60.0, 60.0)}'), 4)
        self.assertGreaterEqual(source.count('".*_box_joint": 0.02'), 4)
        self.assertGreaterEqual(
            source.count('".*_(hip_joint|thigh_joint|calf_joint)$": 0.1'), 4
        )
        self.assertGreaterEqual(source.count("use_default_offset=True"), 4)

    def test_deployment_yaml_physical_envelope_transposes_to_same_controller_contract(self) -> None:
        config = yaml.safe_load(DEPLOYMENT_CFG.read_text(encoding="utf-8"))
        rows = int(config["rows"])
        cols = int(config["cols"])
        self.assertEqual((4, 4), (rows, cols))
        lower = config["physical_dof_pos_lower"]
        upper = config["physical_dof_pos_upper"]
        self.assertEqual(16, len(lower))
        self.assertEqual(16, len(upper))

        # YAML is leg-major; isaacsim deployment transposes it to joint-major.
        transposed_lower = [lower[row * cols + col] for col in range(cols) for row in range(rows)]
        transposed_upper = [upper[row * cols + col] for col in range(cols) for row in range(rows)]
        expected_types = ("hip", "thigh", "calf", "box")
        for joint_type_index, joint_type in enumerate(expected_types):
            expected = EXPECTED_DEPLOYMENT_ENVELOPE[f".*_{joint_type}_joint"]
            start = joint_type_index * 4
            self.assertEqual([expected[0]] * 4, transposed_lower[start : start + 4])
            self.assertEqual([expected[1]] * 4, transposed_upper[start : start + 4])

        action_scale = config["action_scale"]
        default_dof_pos = config["default_dof_pos"]
        transposed_scale = [action_scale[row * cols + col] for col in range(cols) for row in range(rows)]
        transposed_offset = [
            default_dof_pos[row * cols + col] for col in range(cols) for row in range(rows)
        ]
        self.assertEqual([0.1] * 12 + [0.02] * 4, transposed_scale)
        self.assertEqual([0.0] * 4 + [0.7] * 4 + [-1.3] * 4 + [0.03] * 4, transposed_offset)

    def test_action_order_and_delay_capability_are_explicit(self) -> None:
        source = HIGHSTEP_CFG.read_text(encoding="utf-8")
        expected_order = (
            '"FL_hip_joint", "FR_hip_joint", "RL_hip_joint",\n'
            '        "RR_hip_joint", "FL_thigh_joint", "FR_thigh_joint",\n'
            '        "RL_thigh_joint", "RR_thigh_joint", "FL_calf_joint",\n'
            '        "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",\n'
            '        "FL_box_joint", "FR_box_joint", "RL_box_joint", "RR_box_joint",'
        )
        self.assertIn(expected_order, source)
        # The two standard student tasks retain their pre-audit action term.
        self.assertGreaterEqual(source.count("mdp.JointPositionActionCfg("), 2)

    def test_standard_action_score_teacher_keeps_model172300_zero_delay(self) -> None:
        tree = ast.parse(HIGHSTEP_CFG.read_text(encoding="utf-8"), filename=str(HIGHSTEP_CFG))
        target_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "ArclabArcdogAdjustableLegHighstepActionScoreEnvCfg"
        )
        wanted = {
            "self.actions.joint_pos.min_action_delay_steps",
            "self.actions.joint_pos.max_action_delay_steps",
        }
        assignments = {}
        for node in ast.walk(target_class):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = ast.unparse(node.targets[0])
            if target in wanted:
                assignments[target] = ast.literal_eval(node.value)
        self.assertEqual(0, assignments["self.actions.joint_pos.min_action_delay_steps"])
        self.assertEqual(0, assignments["self.actions.joint_pos.max_action_delay_steps"])

    def test_target_limit_penalty_is_not_part_of_training(self) -> None:
        source = HIGHSTEP_CFG.read_text(encoding="utf-8")
        self.assertNotIn("joint_action_target_limit_penalty", source)

    def test_automation_policy_locks_action_contract_and_keeps_target_audit_informational(self) -> None:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        self.assertEqual("must_not_change", policy["locked_action_contract"]["action_scale"])
        self.assertEqual("must_not_change", policy["locked_action_contract"]["joint_pos_clip"])
        self.assertEqual("must_not_change", policy["locked_action_contract"]["default_dof_pos"])
        self.assertTrue(policy["evaluation"]["target_limit_audit_is_informational_only"])
        for role in ("teacher", "teacher_robust", "student"):
            thresholds = policy["thresholds"][role]
            self.assertNotIn("max_target_limit_violation_fraction_worst", thresholds)
            self.assertNotIn("max_target_limit_delta_worst", thresholds)
            self.assertEqual(1.0, thresholds["min_schedule_valid_rate"])


if __name__ == "__main__":
    unittest.main()
