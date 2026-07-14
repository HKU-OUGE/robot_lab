#!/usr/bin/env python3
"""Read-only v1.11 Teacher robustness rollout wrapper around canonical play."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import traceback
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
PLAY = Path(__file__).with_name("play.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, read_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    if read_only:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def tensor_sha(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_play() -> Any:
    spec = importlib.util.spec_from_file_location("highstep_v111_bound_play", PLAY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical play")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Controller:
    def __init__(self, runner: Any, play: Any):
        import torch
        from tools.highstep_teacher_robustness_ab import matrix_by_id

        self.torch = torch
        self.play = play
        self.runner = runner
        self.env = runner.env
        self.unwrapped = runner.env.unwrapped
        self.asset = self.unwrapped.scene["robot"]
        self.run_id = os.environ["HIGHSTEP_TEACHER_AB_RUN_ID"]
        self.teacher = os.environ["HIGHSTEP_TEACHER_AB_TEACHER"]
        self.snapshot_mode = os.environ["HIGHSTEP_TEACHER_AB_SNAPSHOT_MODE"]
        self.row = matrix_by_id()[self.run_id]
        self.root = Path(os.environ["HIGHSTEP_TEACHER_AB_ROOT"]).resolve()
        self.prereg = Path(os.environ["HIGHSTEP_TEACHER_AB_PREREGISTRATION"]).resolve(strict=True)
        expected_prereg = os.environ["HIGHSTEP_TEACHER_AB_PREREGISTRATION_SHA256"]
        if sha(self.prereg) != expected_prereg:
            raise RuntimeError("v1.11 preregistration SHA mismatch")
        payload = json.loads(self.prereg.read_text())
        checkpoint = Path(play.args_cli.checkpoint).resolve(strict=True)
        expected_teacher = payload["teachers"][self.teacher]
        if str(checkpoint) != expected_teacher["checkpoint"] or sha(checkpoint) != expected_teacher["checkpoint_sha256"]:
            raise RuntimeError("Teacher checkpoint binding mismatch")
        if self.snapshot_mode != expected_teacher["snapshot_mode"]:
            raise RuntimeError("snapshot mode binding mismatch")
        self.checkpoint = checkpoint
        self.checkpoint_before = sha(checkpoint)
        self.policy_module = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
        if self.policy_module is None:
            raise RuntimeError("cannot locate loaded Teacher policy")
        self.tensor_before = tensor_sha(self.policy_module)
        self.tracker = None
        self.action_step = 0
        self.impulse_applied = False
        self.impulse_step = None
        self.impulse_actual_delta = None
        self.learn_calls = 0
        self.backward_calls = 0
        self.optimizer_step_calls = 0
        self.recovery_streak = 0
        self.recovery_step = None
        self.centerline_crossed = False
        self.width_min = float("inf")
        self.min_abs_y_min = float("inf")
        self.roll_max = 0.0
        self.yaw_deviation_max = 0.0
        self.initial_yaw = None
        self.samples = 0
        self.frames_path = self.root / "evidence" / self.teacher / f"{self.run_id}.frames.jsonl"
        self.summary_path = self.root / "evidence" / self.teacher / f"{self.run_id}.runtime.json"
        if self.frames_path.exists() or self.summary_path.exists():
            raise FileExistsError(f"runtime evidence already exists for {self.teacher}/{self.run_id}")
        self.frames_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path = self.root / "snapshots" / f"{self.run_id}.json"
        self.snapshot_sha = self._prepare_snapshot()

    def _refresh(self) -> None:
        self.unwrapped.scene.write_data_to_sim()
        self.unwrapped.sim.forward()

    def _rear_y(self):
        torch = self.torch
        math_utils = self.play.math_utils
        rear_ids = self.asset.find_bodies(["RL_foot", "RR_foot"])[0]
        pos = self.asset.data.body_pos_w[0, rear_ids, :]
        base = self.asset.data.root_pos_w[0]
        heading = math_utils.yaw_quat(self.asset.data.root_quat_w[0].unsqueeze(0))[0]
        body = math_utils.quat_apply_inverse(heading.unsqueeze(0).repeat(2, 1), pos - base.unsqueeze(0))
        return body[:, 1].clone()

    def _write_full_state(self, root_pose, root_vel, joint_pos, joint_vel) -> None:
        torch = self.torch
        env_ids = torch.tensor([0], device=self.asset.device, dtype=torch.long)
        self.asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._refresh()

    def _solve_width(self, target: float, mode: str) -> None:
        torch = self.torch
        names = list(self.asset.joint_names)
        hip_ids = {"rl_only": names.index("RL_hip_joint"), "rr_only": names.index("RR_hip_joint")}
        y0 = self._rear_y()
        if mode == "symmetric":
            desired = torch.tensor([0.5 * target, -0.5 * target], device=y0.device, dtype=y0.dtype)
            movable = [(0, hip_ids["rl_only"]), (1, hip_ids["rr_only"])]
        elif mode == "rl_only":
            desired = torch.stack((y0[1] + target, y0[1]))
            movable = [(0, hip_ids["rl_only"])]
        elif mode == "rr_only":
            desired = torch.stack((y0[0], y0[0] - target))
            movable = [(1, hip_ids["rr_only"])]
        else:
            raise RuntimeError(f"bad inward mode {mode}")
        q = self.asset.data.joint_pos[[0]].clone()
        dq = torch.zeros_like(self.asset.data.joint_vel[[0]])
        limits = self.asset.data.soft_joint_pos_limits[0]
        for _ in range(16):
            current = self._rear_y()
            if max(abs(float(current[i] - desired[i])) for i, _ in movable) <= 5.0e-4:
                break
            for foot_index, joint_index in movable:
                base_q = float(q[0, joint_index])
                epsilon = 0.01
                q[0, joint_index] = base_q + epsilon
                self.asset.write_joint_state_to_sim(q, dq)
                self._refresh()
                shifted = self._rear_y()[foot_index]
                derivative = float((shifted - current[foot_index]) / epsilon)
                q[0, joint_index] = base_q
                if abs(derivative) < 1.0e-4:
                    raise RuntimeError("rear width inverse kinematics derivative vanished")
                correction = float((desired[foot_index] - current[foot_index]) / derivative)
                correction = max(-0.12, min(0.12, correction))
                q[0, joint_index] = max(
                    float(limits[joint_index, 0]),
                    min(float(limits[joint_index, 1]), base_q + correction),
                )
                self.asset.write_joint_state_to_sim(q, dq)
                self._refresh()
        actual = abs(float(self._rear_y()[0] - self._rear_y()[1]))
        if abs(actual - target) > 0.002:
            raise RuntimeError(f"initial rear width solve failed: target={target} actual={actual}")

    def _context_payload(self) -> dict[str, Any]:
        context = self.unwrapped._front_step_eval_context
        return {
            key: (value.detach().cpu().tolist() if hasattr(value, "detach") else value)
            for key, value in context.items()
            if key in {"side", "approach", "lateral", "origin_xy", "top_z", "low_z", "half_width"}
        }

    def _state_payload(self) -> dict[str, Any]:
        y = self._rear_y().detach().cpu().tolist()
        return {
            "schema_version": 1,
            "kind": "highstep_v111_initial_physical_snapshot",
            "run_id": self.run_id,
            "row": self.row,
            "root_pose_w": self.torch.cat((self.asset.data.root_pos_w[[0]], self.asset.data.root_quat_w[[0]]), dim=1).detach().cpu().tolist(),
            "root_velocity_w": self.torch.cat((self.asset.data.root_lin_vel_w[[0]], self.asset.data.root_ang_vel_w[[0]]), dim=1).detach().cpu().tolist(),
            "joint_position": self.asset.data.joint_pos[[0]].detach().cpu().tolist(),
            "joint_velocity": self.asset.data.joint_vel[[0]].detach().cpu().tolist(),
            "joint_names": list(self.asset.joint_names),
            "terrain_origin": self.unwrapped.scene.env_origins[[0]].detach().cpu().tolist(),
            "front_step_context": self._context_payload(),
            "initial_rear_y_body": y,
            "initial_rear_width_m": abs(float(y[0] - y[1])),
        }

    def _restore_payload(self, payload: Mapping[str, Any]) -> None:
        torch = self.torch
        device = self.asset.device
        dtype = self.asset.data.joint_pos.dtype
        if payload["run_id"] != self.run_id or payload["row"] != self.row:
            raise RuntimeError("physical snapshot identity mismatch")
        if payload["joint_names"] != list(self.asset.joint_names):
            raise RuntimeError("physical snapshot joint order mismatch")
        current_context = self._context_payload()
        if current_context != payload["front_step_context"]:
            raise RuntimeError("front-step reset context differs before snapshot replay")
        self._write_full_state(
            torch.tensor(payload["root_pose_w"], device=device, dtype=dtype),
            torch.tensor(payload["root_velocity_w"], device=device, dtype=dtype),
            torch.tensor(payload["joint_position"], device=device, dtype=dtype),
            torch.tensor(payload["joint_velocity"], device=device, dtype=dtype),
        )
        actual = self._state_payload()
        for key in ("root_pose_w", "root_velocity_w", "joint_position", "joint_velocity", "initial_rear_y_body"):
            lhs = torch.tensor(actual[key], dtype=torch.float64)
            rhs = torch.tensor(payload[key], dtype=torch.float64)
            if float(torch.max(torch.abs(lhs - rhs))) > 1.0e-6:
                raise RuntimeError(f"physical snapshot replay mismatch: {key}")

    def _prepare_snapshot(self) -> str:
        target = self.row["initial_rear_width_m"]
        if self.snapshot_path.exists():
            payload = json.loads(self.snapshot_path.read_text())
            self._restore_payload(payload)
            return sha(self.snapshot_path)
        if self.snapshot_mode != "capture":
            raise RuntimeError("Teacher B cannot run before Teacher A freezes snapshot")
        if target is not None:
            self._solve_width(float(target), str(self.row["inward_mode"]))
        root_pose = self.torch.cat((self.asset.data.root_pos_w[[0]], self.asset.data.root_quat_w[[0]]), dim=1).clone()
        root_vel = self.torch.zeros((1, 6), device=self.asset.device, dtype=self.asset.data.joint_pos.dtype)
        joint_pos = self.asset.data.joint_pos[[0]].clone()
        joint_vel = self.torch.zeros_like(self.asset.data.joint_vel[[0]])
        self._write_full_state(root_pose, root_vel, joint_pos, joint_vel)
        payload = self._state_payload()
        if target is not None and abs(float(payload["initial_rear_width_m"]) - float(target)) > 0.002:
            raise RuntimeError("captured width outside preregistered tolerance")
        atomic_json(self.snapshot_path, payload, read_only=True)
        return sha(self.snapshot_path)

    def attach(self, tracker: Any) -> None:
        self.tracker = tracker

    def _phase_ready(self) -> bool:
        phase = self.row["impulse_phase"]
        if phase == "none":
            return False
        if phase == "approach":
            return self.action_step >= 15
        if self.tracker is None:
            return False
        if phase == "front_support":
            return int(self.tracker.phase) >= 3
        if phase == "first_rear":
            return self.tracker.first_rear_top_step is not None
        raise RuntimeError(f"bad impulse phase {phase}")

    def maybe_impulse(self) -> None:
        if self.impulse_applied or float(self.row["impulse_delta_v_mps"]) <= 0 or not self._phase_ready():
            return
        math_utils = self.play.math_utils
        heading = math_utils.yaw_quat(self.asset.data.root_quat_w[0].unsqueeze(0))[0]
        local_y = self.torch.tensor([[0.0, 1.0, 0.0]], device=self.asset.device, dtype=self.asset.data.joint_pos.dtype)
        left_w = math_utils.quat_apply(heading.unsqueeze(0), local_y)[0]
        sign = 1.0 if self.row["impulse_direction"] == "left" else -1.0
        delta = sign * float(self.row["impulse_delta_v_mps"]) * left_w
        velocity = self.torch.cat((self.asset.data.root_lin_vel_w[[0]], self.asset.data.root_ang_vel_w[[0]]), dim=1).clone()
        velocity[0, :3] += delta
        env_ids = self.torch.tensor([0], device=self.asset.device, dtype=self.torch.long)
        self.asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)
        self.impulse_applied = True
        self.impulse_step = self.action_step
        self.impulse_actual_delta = delta.detach().cpu().tolist()

    def policy(self, base_policy: Any, observations: Any):
        actions = base_policy(observations)
        self.maybe_impulse()
        self.action_step += 1
        return actions

    def observe(self, step: int) -> None:
        y = self._rear_y()
        width = abs(float(y[0] - y[1]))
        min_abs = min(abs(float(y[0])), abs(float(y[1])))
        crossing = bool(float(y[0]) <= 0.0 or float(y[1]) >= 0.0)
        self.centerline_crossed = self.centerline_crossed or crossing
        self.width_min = min(self.width_min, width)
        self.min_abs_y_min = min(self.min_abs_y_min, min_abs)
        roll_metric = abs(float(self.asset.data.projected_gravity_b[0, 1]))
        lateral_speed = abs(float(self.asset.data.root_lin_vel_b[0, 1]))
        self.roll_max = max(self.roll_max, roll_metric)
        heading = self.play.math_utils.yaw_quat(self.asset.data.root_quat_w[0].unsqueeze(0))[0]
        forward = self.play.math_utils.quat_apply(
            heading.unsqueeze(0), self.torch.tensor([[1.0, 0.0, 0.0]], device=self.asset.device)
        )[0]
        yaw = float(self.torch.atan2(forward[1], forward[0]))
        if self.initial_yaw is None:
            self.initial_yaw = yaw
        yaw_delta = abs(float(self.torch.atan2(self.torch.sin(self.torch.tensor(yaw - self.initial_yaw)), self.torch.cos(self.torch.tensor(yaw - self.initial_yaw)))))
        self.yaw_deviation_max = max(self.yaw_deviation_max, yaw_delta)
        recovery_window_open = float(self.row["impulse_delta_v_mps"]) <= 0 or self.impulse_applied
        stable = bool(
            recovery_window_open and width >= 0.32 and min_abs >= 0.08 and not crossing
            and lateral_speed <= 0.08 and roll_metric <= 0.25
        )
        self.recovery_streak = self.recovery_streak + 1 if stable else 0
        if self.recovery_step is None and self.recovery_streak >= 10:
            self.recovery_step = int(step - 9)
        record = {
            "step": int(step), "rear_y_body": [float(y[0]), float(y[1])],
            "rear_width_m": width, "rear_min_abs_y_m": min_abs,
            "centerline_crossed": crossing, "root_lateral_speed_abs_mps": lateral_speed,
            "roll_metric_abs": roll_metric, "yaw_deviation_abs_rad": yaw_delta,
            "impulse_applied": self.impulse_applied, "impulse_step": self.impulse_step,
            "recovery_streak": self.recovery_streak,
        }
        with self.frames_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.samples += 1

    def finalize(self) -> dict[str, Any]:
        tracker_summary = self.tracker.summary() if self.tracker is not None else {}
        recovery_origin = self.impulse_step if self.impulse_step is not None else 0
        recovery_time = None if self.recovery_step is None else max(0, self.recovery_step - recovery_origin) * float(self.unwrapped.step_dt)
        payload = {
            "schema_version": 1,
            "kind": "highstep_v111_teacher_robustness_runtime",
            "teacher": self.teacher,
            "run_id": self.run_id,
            "row": self.row,
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256_before": self.checkpoint_before,
            "checkpoint_sha256_after": sha(self.checkpoint),
            "policy_tensor_sha256_before": self.tensor_before,
            "policy_tensor_sha256_after": tensor_sha(self.policy_module),
            "snapshot": str(self.snapshot_path),
            "snapshot_sha256": self.snapshot_sha,
            "snapshot_mode": self.snapshot_mode,
            "initial_rear_width_m": json.loads(self.snapshot_path.read_text())["initial_rear_width_m"],
            "impulse_applied": self.impulse_applied,
            "impulse_step": self.impulse_step,
            "impulse_actual_delta_v_w": self.impulse_actual_delta,
            "recovery": self.recovery_step is not None,
            "recovery_step": self.recovery_step,
            "recovery_time_s": recovery_time,
            "rear_width_min_m": self.width_min,
            "rear_min_abs_y_min_m": self.min_abs_y_min,
            "centerline_crossed": self.centerline_crossed,
            "roll_metric_abs_max": self.roll_max,
            "yaw_deviation_abs_max_rad": self.yaw_deviation_max,
            "fell": bool(tracker_summary.get("terminated_early", False)),
            "frame_count": self.samples,
            "frames": str(self.frames_path),
            "frames_sha256": sha(self.frames_path),
            "runner_learn_calls": self.learn_calls,
            "backward_calls": self.backward_calls,
            "optimizer_step_calls": self.optimizer_step_calls,
            "tracker": tracker_summary,
        }
        payload["read_only_checks_passed"] = bool(
            payload["checkpoint_sha256_before"] == payload["checkpoint_sha256_after"]
            and payload["policy_tensor_sha256_before"] == payload["policy_tensor_sha256_after"]
            and payload["runner_learn_calls"] == payload["backward_calls"] == payload["optimizer_step_calls"] == 0
            and payload["frame_count"] > 0
        )
        if not payload["read_only_checks_passed"]:
            raise RuntimeError("read-only rollout contract failed")
        atomic_json(self.summary_path, payload, read_only=True)
        return payload


def main() -> int:
    play = load_play()
    controllers: list[Controller] = []
    original_policy = play.OnPolicyRunner.get_inference_policy
    original_tracker = play._HighstepEvalTracker

    def patched_policy(runner, device=None):
        base = original_policy(runner, device=device)
        controller = Controller(runner, play)
        controllers.append(controller)
        return lambda observations: controller.policy(base, observations)

    class Tracker(original_tracker):
        def __init__(self, env):
            super().__init__(env)
            if len(controllers) != 1:
                raise RuntimeError("v1.11 expected one active controller")
            controllers[0].attach(self)

        def update(self, step):
            super().update(step)
            controllers[0].observe(step)

    play.OnPolicyRunner.get_inference_policy = patched_policy
    play._HighstepEvalTracker = Tracker
    try:
        play.main()
        if len(controllers) != 1:
            raise RuntimeError("v1.11 controller cardinality mismatch")
        result = controllers[0].finalize()
        print("[HIGHSTEP_TEACHER_AB_RUNTIME_JSON] " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        play.simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
