"""Small, testable helpers for deterministic manual high-step respawns."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def restore_default_joint_state(robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Restore every environment to the asset default joint pose with zero velocity."""

    device = robot.data.joint_pos.device
    env_ids = torch.arange(robot.data.joint_pos.shape[0], device=device, dtype=torch.long)
    joint_pos = robot.data.default_joint_pos[env_ids].detach().clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel[env_ids])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    return env_ids, joint_pos, joint_vel


@dataclass
class ManualHighstepStartGate:
    """Hold keyboard commands at zero until a respawn has physically settled.

    The two-second minimum is intentional: the inspected manual trace reached
    the repeatable zero-command attractor after roughly 1.4--2.1 seconds.  The
    following consecutive-state check prevents time alone from opening the
    gate while the robot is still moving.
    """

    min_hold_steps: int = 100
    stable_steps_required: int = 10
    max_joint_position_step: float = 0.01
    max_joint_speed: float = 0.30
    max_root_linear_speed: float = 0.03
    max_root_angular_speed: float = 0.12

    def __post_init__(self) -> None:
        if self.min_hold_steps < 1:
            raise ValueError("min_hold_steps must be positive")
        if self.stable_steps_required < 1:
            raise ValueError("stable_steps_required must be positive")
        self.reset()

    def reset(self) -> None:
        self.steps_since_reset = 0
        self.stable_streak = 0
        self.ready = False
        self._previous_joint_pos: torch.Tensor | None = None
        self.last_metrics: dict[str, float | int | bool] = {
            "steps_since_reset": 0,
            "stable_streak": 0,
            "joint_position_step": float("inf"),
            "joint_speed": float("inf"),
            "root_linear_speed": float("inf"),
            "root_angular_speed": float("inf"),
            "stable_now": False,
        }

    @property
    def blocked(self) -> bool:
        return not self.ready

    def filter_command(self, command: torch.Tensor) -> torch.Tensor:
        """Return a zero command while blocked without mutating the device input."""

        return command if self.ready else torch.zeros_like(command)

    def observe(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> bool:
        """Observe one simulator control step and return True only on gate opening."""

        if self.ready:
            return False

        current_joint_pos = joint_pos.detach().reshape(-1)
        joint_position_step = (
            float("inf")
            if self._previous_joint_pos is None
            else float(torch.max(torch.abs(current_joint_pos - self._previous_joint_pos)).item())
        )
        joint_speed = float(torch.max(torch.abs(joint_vel.detach())).item())
        root_linear_speed = float(torch.linalg.vector_norm(root_lin_vel.detach()).item())
        root_angular_speed = float(torch.linalg.vector_norm(root_ang_vel.detach()).item())
        stable_now = bool(
            joint_position_step <= self.max_joint_position_step
            and joint_speed <= self.max_joint_speed
            and root_linear_speed <= self.max_root_linear_speed
            and root_angular_speed <= self.max_root_angular_speed
        )

        self.steps_since_reset += 1
        self.stable_streak = self.stable_streak + 1 if stable_now else 0
        self._previous_joint_pos = current_joint_pos.clone()
        self.last_metrics = {
            "steps_since_reset": self.steps_since_reset,
            "stable_streak": self.stable_streak,
            "joint_position_step": joint_position_step,
            "joint_speed": joint_speed,
            "root_linear_speed": root_linear_speed,
            "root_angular_speed": root_angular_speed,
            "stable_now": stable_now,
        }

        became_ready = bool(
            self.steps_since_reset >= self.min_hold_steps
            and self.stable_streak >= self.stable_steps_required
        )
        if became_ready:
            self.ready = True
        return became_ready
