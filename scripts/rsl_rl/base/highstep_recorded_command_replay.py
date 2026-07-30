"""Hash-bound replay of one command episode recorded by ``PlayJointRecorder``.

This helper intentionally replays only the three-dimensional velocity command.
The live policy, action prior, physics, observations, and joint targets remain in
the normal Isaac Lab execution path.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path


class RecordedCommandReplay:
    """Expose a validated per-control-step command sequence from one episode."""

    REQUIRED_COLUMNS = (
        "control_step",
        "episode_id",
        "episode_step",
        "reset_before_step",
        "command.vx",
        "command.vy",
        "command.wz",
    )

    def __init__(self, path: str | Path, *, expected_sha256: str, episode_id: int):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"recorded command trace not found: {self.path}")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha256 != expected_sha256:
            raise RuntimeError(
                "recorded command trace SHA256 mismatch: "
                f"expected={expected_sha256} actual={self.sha256}"
            )

        self.episode_id = int(episode_id)
        all_rows: list[dict[str, str]] = []
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in self.REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
            if missing:
                raise RuntimeError(f"recorded command trace is missing columns: {missing}")
            all_rows = list(reader)
        matching_indices = [
            index
            for index, row in enumerate(all_rows)
            if int(row["episode_id"]) == self.episode_id
        ]
        rows = [all_rows[index] for index in matching_indices]
        if not rows:
            raise RuntimeError(f"recorded command episode is absent: {self.episode_id}")
        if matching_indices != list(range(matching_indices[0], matching_indices[-1] + 1)):
            raise RuntimeError("recorded command episode is split into non-contiguous blocks")
        if matching_indices[0] == 0:
            raise RuntimeError(
                "recorded command episode has no preceding row for exact reset-state restoration"
            )
        self._pre_reset_row = all_rows[matching_indices[0] - 1]
        if int(self._pre_reset_row["episode_id"]) == self.episode_id:
            raise RuntimeError("recorded command reset boundary does not change episode id")
        if int(self._pre_reset_row["control_step"]) + 1 != int(rows[0]["control_step"]):
            raise RuntimeError("recorded command reset boundary is not control-step contiguous")

        episode_steps = [int(row["episode_step"]) for row in rows]
        if episode_steps != list(range(len(rows))):
            raise RuntimeError("recorded command episode steps are not contiguous from zero")
        reset_flags = [int(row["reset_before_step"]) for row in rows]
        if reset_flags[0] != 1 or any(reset_flags[1:]):
            raise RuntimeError("recorded command episode has an invalid reset boundary")

        commands = []
        for row in rows:
            command = tuple(float(row[name]) for name in ("command.vx", "command.vy", "command.wz"))
            if not all(math.isfinite(value) for value in command):
                raise RuntimeError("recorded command episode contains a non-finite command")
            commands.append(command)
        self.commands = tuple(commands)
        self._step = 0

    @property
    def frame_count(self) -> int:
        return len(self.commands)

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def command(self) -> tuple[float, float, float]:
        if 0 <= self._step < self.frame_count:
            return self.commands[self._step]
        return (0.0, 0.0, 0.0)

    def pre_reset_joint_state(
        self, joint_names: list[str] | tuple[str, ...]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return q/dq carried across the manual R reset into this episode."""
        positions = []
        velocities = []
        for name in joint_names:
            position_key = f"joint_pos.{name}"
            velocity_key = f"joint_vel.{name}"
            if position_key not in self._pre_reset_row or velocity_key not in self._pre_reset_row:
                raise RuntimeError(
                    f"recorded reset state is missing joint fields for {name!r}"
                )
            position = float(self._pre_reset_row[position_key])
            velocity = float(self._pre_reset_row[velocity_key])
            if not math.isfinite(position) or not math.isfinite(velocity):
                raise RuntimeError("recorded reset state contains a non-finite joint value")
            positions.append(position)
            velocities.append(velocity)
        return tuple(positions), tuple(velocities)

    def summary(self) -> dict[str, object]:
        transitions: list[dict[str, object]] = []
        previous = None
        for step, command in enumerate(self.commands):
            if command != previous:
                transitions.append({"episode_step": step, "command": list(command)})
                previous = command
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "episode_id": self.episode_id,
            "frame_count": self.frame_count,
            "pre_reset_source_control_step": int(self._pre_reset_row["control_step"]),
            "pre_reset_source_episode_id": int(self._pre_reset_row["episode_id"]),
            "transitions": transitions,
        }
