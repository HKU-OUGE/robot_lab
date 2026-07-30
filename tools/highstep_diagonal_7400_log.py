#!/usr/bin/env python3
"""Read-only compact dashboard for the dashboard-declared B300 7400 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time


ROOT = Path("/home/lxq/Softwares/robot_lab")
DASHBOARD = ROOT / "tmp/highstep_dashboard_active_workflow.json"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


class C:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def paint(self, code: str, value: object) -> str:
        text = str(value)
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def green(self, value: object) -> str:
        return self.paint("32", value)

    def yellow(self, value: object) -> str:
        return self.paint("33", value)

    def red(self, value: object) -> str:
        return self.paint("31", value)

    def cyan(self, value: object) -> str:
        return self.paint("36", value)

    def bold(self, value: object) -> str:
        return self.paint("1", value)

    def dim(self, value: object) -> str:
        return self.paint("2", value)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def pid_alive(value: object) -> bool:
    try:
        os.kill(int(value), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return int(value) > 1


def resolve_train_log(state_path: Path, state: dict) -> Path:
    """Resolve the log bound to the active child, never by directory mtime.

    New attempts intentionally keep earlier ``train.log`` evidence.  The
    active child's stdout is therefore the strongest read-only binding.  A
    declared state field and the attempt encoded in the W&B run name are
    retained as deterministic fallbacks for completed runs.
    """
    candidates: list[Path] = []
    for key in ("train_log_path", "train_log", "log_path"):
        value = state.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))

    active_pid = state.get("active_pid")
    try:
        pid = int(active_pid)
    except (TypeError, ValueError):
        pid = 0
    if pid > 1:
        for descriptor in (1, 2):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{descriptor}")
            except OSError:
                continue
            if target.endswith(" (deleted)"):
                target = target[:-10]
            if target.startswith("/"):
                candidates.append(Path(target))

    run_name = str(state.get("wandb_run_name") or "")
    attempt = re.search(r"(?:^|_)attempt(\d+)(?:_|$)", run_name)
    if attempt:
        candidates.append(state_path.parent / f"train_attempt{attempt.group(1)}.log")
    candidates.append(state_path.parent / "train.log")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return candidates[-1]


def authority_matches(declaration: dict, state: dict, prereg: dict, prereg_path: Path) -> bool:
    workflow = declaration.get("workflow_id")
    workflow_ok = bool(workflow) and workflow == state.get("workflow_id") == prereg.get("workflow_id")
    spec_path = Path(str(declaration.get("spec_path", "")))
    spec_sha = declaration.get("spec_sha256")
    spec_ok = (
        bool(spec_sha)
        and sha256(spec_path) == spec_sha
        and state.get("spec_path") == str(spec_path)
        and state.get("spec_sha256") == spec_sha
        and prereg.get("authority", {}).get("spec_sha256") == spec_sha
    )
    prereg_sha = declaration.get("preregistration_sha256")
    prereg_ok = (
        bool(prereg_sha)
        and sha256(prereg_path) == prereg_sha
        and state.get("preregistration_path") == str(prereg_path)
        and state.get("preregistration_sha256") == prereg_sha
    )
    return bool(workflow_ok and spec_ok and prereg_ok)


def dashboard_title(workflow: object) -> str:
    name = str(workflow or "unknown workflow")
    if "critical_transition_balanced_diagonal" in name:
        return "B300 critical-transition A+B fresh 7400 — 只读实时日志"
    if "diagonal_imitation_fresh_7400" in name:
        return "B300 diagonal imitation fresh 7400 — 只读实时日志"
    return f"{name} — 只读实时日志"


def contract_lines(prereg: dict) -> list[str]:
    sampling = prereg.get("critical_transition_sampling", {})
    phase_loss = prereg.get("phase_diagonal_loss", {})
    if sampling and phase_loss:
        fractions = sampling.get("minibatch_source_fractions", {})
        ratio = phase_loss.get("gated_total_per_element_weight_ratio", "N/A")
        return [
            "Critical sampling="
            f"{fractions.get('front_transition', 'N/A')}/"
            f"{fractions.get('rear_transition', 'N/A')}/"
            f"{fractions.get('original_distribution', 'N/A')}"
            f"  window=±{sampling.get('window_radius_policy_steps', 'N/A')} steps",
            f"FR joints={phase_loss.get('front_indices', [])}  gate={phase_loss.get('front_gate_stages', [])}  weight={ratio}x",
            f"RL joints={phase_loss.get('rear_indices', [])}  gate={phase_loss.get('rear_gate_stages', [])}  weight={ratio}x",
        ]
    single = prereg.get("single_variable_control", {})
    return [
        f"Diagonal scale={single.get('treatment', 'N/A')}  "
        f"gate={single.get('gate', '-')}  joints={single.get('joint_indices', [])}"
    ]


def tail_text(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            raw = stream.read()
    except OSError:
        return ""
    return ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))


def number(text: str) -> float | None:
    found = NUMBER_RE.search(text)
    if not found:
        return None
    try:
        value = float(found.group(0))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def normalize_key(raw: str) -> str:
    key = raw.strip()
    if key.startswith("Mean ") and key not in {"Mean reward", "Mean episode length"}:
        key = key[5:]
        if key.endswith(" loss"):
            key = key[:-5]
    return key


def parse_blocks(text: str) -> list[dict]:
    matches = list(ITER_RE.finditer(text))
    output: list[dict] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start():end]
        if not re.search(r"^\s*ETA:\s*", segment, re.MULTILINE):
            continue
        row: dict[str, object] = {
            "absolute": int(match.group(1)),
            "absolute_total": int(match.group(2)),
        }
        for raw_line in segment.splitlines():
            line = raw_line.strip()
            if line.startswith("Computation:"):
                speed = re.search(r"(\d+)\s+steps/s", line)
                if speed:
                    row["steps_per_second"] = float(speed.group(1))
                continue
            key_text, separator, value_text = line.partition(":")
            if not separator:
                continue
            key = normalize_key(key_text)
            if key in {"Time elapsed", "ETA"}:
                row[key] = value_text.strip()
                continue
            value = number(value_text)
            if value is not None:
                row[key] = value
        output.append(row)
    return output


def age_seconds(path: Path) -> float:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return float("inf")


def latest_checkpoint(run_dir: object, absolute_origin: int) -> str:
    directory = Path(str(run_dir or ""))
    best: tuple[int, str] | None = None
    try:
        for path in directory.glob("model_*.pt"):
            match = re.fullmatch(r"model_(\d+)\.pt", path.name)
            if match:
                item = (int(match.group(1)), path.name)
                if best is None or item[0] > best[0]:
                    best = item
    except OSError:
        pass
    if best is None:
        return "尚未保存"
    effective = best[0] - absolute_origin + 1
    return f"{best[1]} (E{effective})"


def effective_clocks(state: dict, latest: dict) -> tuple[int, int]:
    """Return (state effective update, completed-log effective update).

    Fresh distillation runs historically stored the next effective update in
    ``relative_distill_update`` while their log metric was zero-based.  The
    E5700 continuation stores ``effective_updates`` and logs that same value
    directly.  Runner iteration is deliberately excluded: continuation
    sampling can skip optimizer updates, so runner and effective clocks are
    independent.
    """
    value = state.get("effective_updates")
    direct_clock = isinstance(value, (int, float))
    if direct_clock:
        state_effective = int(value)
    else:
        fallback = state.get("relative_distill_update")
        state_effective = int(fallback) if isinstance(fallback, (int, float)) else 0

    count = latest.get("Debug/Student_Distill_Update_Count")
    if not isinstance(count, (int, float)):
        return state_effective, 0
    log_effective = int(float(count)) + (0 if direct_clock else 1)
    return state_effective, log_effective


def target_effective_updates(state: dict, prereg: dict) -> int:
    value = state.get("target_effective_updates")
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    budget = prereg.get("training_budget", {})
    value = budget.get("target_effective_update") if isinstance(budget, dict) else None
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    value = state.get("max_effective_updates")
    return int(value) if isinstance(value, (int, float)) and int(value) > 0 else 7400


def duration_seconds(value: object) -> int | None:
    parts = str(value or "").strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def remaining_time_estimate(
    state: dict, latest: dict, current_effective: int, target_effective: int
) -> tuple[int, float] | None:
    """Estimate remaining seconds from this attempt's effective-update rate."""
    elapsed = duration_seconds(latest.get("Time elapsed"))
    start = state.get("start_effective_updates")
    if not isinstance(start, (int, float)):
        start = 0
    completed = current_effective - int(start)
    remaining = max(0, target_effective - current_effective)
    if elapsed is None or elapsed < 60 or completed <= 0:
        return None
    rate_per_second = completed / elapsed
    if not math.isfinite(rate_per_second) or rate_per_second <= 0:
        return None
    return int(math.ceil(remaining / rate_per_second)), rate_per_second * 60.0


def format_duration(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(done: int, total: int, width: int = 28) -> str:
    ratio = min(1.0, max(0.0, done / total)) if total else 0.0
    filled = round(width * ratio)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def fmt(value: object, digits: int = 4) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


def trend(blocks: list[dict], key: str) -> str:
    values = [float(row[key]) for row in blocks[-20:] if isinstance(row.get(key), (int, float))]
    if len(values) < 2:
        return "·"
    delta = values[-1] - values[0]
    scale = max(abs(values[0]), 1.0e-12)
    if abs(delta) / scale < 0.01:
        return "→"
    return "↓" if delta < 0 else "↑"


def cell(label: str, key: str, latest: dict, blocks: list[dict], color: C) -> str:
    value = latest.get(key)
    arrow = trend(blocks, key)
    return f"{label:<11} {fmt(value):>9} {color.cyan(arrow)}"


def render(use_color: bool) -> str:
    color = C(use_color)
    declaration = read_json(DASHBOARD)
    state_path = Path(str(declaration.get("state_path", "")))
    state = read_json(state_path)
    prereg_path = Path(str(declaration.get("preregistration_path", "")))
    prereg = read_json(prereg_path)
    heartbeat_path = Path(str(declaration.get("heartbeat_path", state_path.parent / "heartbeat.json")))
    train_log = resolve_train_log(state_path, state)
    blocks = parse_blocks(tail_text(train_log))
    latest = blocks[-1] if blocks else {}

    authority_ok = authority_matches(declaration, state, prereg, prereg_path)
    train_alive = pid_alive(state.get("active_pid"))
    launcher_alive = pid_alive(state.get("supervisor_pid") or state.get("launcher_pid"))
    heartbeat_age = age_seconds(heartbeat_path)
    heartbeat_ok = heartbeat_age <= 45.0

    state_relative, relative_from_count = effective_clocks(state, latest)
    total = target_effective_updates(state, prereg)
    absolute_origin = int(state.get("absolute_runner_step_origin") or 173499)
    log_absolute = int(latest.get("absolute") or 0)
    # Bind progress and all displayed metrics to the same completed log block.
    # Runner iteration is a separate clock: continuation sampling may consume
    # a rollout without performing an optimizer/effective update.
    relative = (
        relative_from_count
        if relative_from_count > 0
        else state_relative
    )
    warmup = int(latest.get("Debug/Student_Warmup_Updates") or 1400)
    stage = "Estimator warmup" if relative < warmup else "Estimator + 4 box rows"
    percent = 100.0 * relative / total if total else 0.0
    remaining_estimate = remaining_time_estimate(state, latest, relative, total)
    if remaining_estimate is None:
        remaining_text = "预计剩余: 暂无足够有效更新样本"
    else:
        remaining_seconds, effective_per_minute = remaining_estimate
        finish_at = time.strftime(
            "%m-%d %H:%M", time.localtime(time.time() + remaining_seconds)
        )
        remaining_text = (
            f"预计剩余: {format_duration(remaining_seconds)}  | "
            f"预计完成: {finish_at}  | 本次累计速度={effective_per_minute:.2f} E/min"
        )
    absolute = log_absolute or int(state.get("absolute_runner_step") or 0)
    state_lag = relative - state_relative
    state_absolute = int(state.get("absolute_runner_step") or 0)
    runner_lag = absolute - state_absolute if absolute and state_absolute else 0
    if not blocks:
        sync_text = color.red("没有完整日志块")
    elif relative_from_count <= 0:
        sync_text = color.red("完整日志块缺少有效更新计数")
    elif abs(state_lag) <= 5:
        sync_text = color.green(f"正常(state差{state_lag:+d})")
    elif abs(state_lag) <= 15:
        sync_text = color.yellow(f"state等待heartbeat({state_lag:+d})")
    else:
        sync_text = color.red(f"state明显不同步({state_lag:+d})")
    if not blocks or not absolute:
        runner_sync_text = color.red("没有runner日志时钟")
    elif not state_absolute:
        runner_sync_text = color.yellow("state尚无runner快照")
    elif abs(runner_lag) <= 5:
        runner_sync_text = color.green(f"正常(state差{runner_lag:+d})")
    elif abs(runner_lag) <= 15:
        runner_sync_text = color.yellow(f"state等待heartbeat({runner_lag:+d})")
    else:
        runner_sync_text = color.red(f"state明显不同步({runner_lag:+d})")

    authority_text = color.green("OK") if authority_ok else color.red("MISMATCH")
    train_text = color.green(f"alive({state.get('active_pid')})") if train_alive else color.red("NOT RUNNING")
    launcher_text = color.green(f"alive({state.get('supervisor_pid') or state.get('launcher_pid')})") if launcher_alive else color.red("NOT RUNNING")
    heartbeat_text = (
        color.green(f"{heartbeat_age:.0f}s") if heartbeat_ok
        else color.yellow(f"{heartbeat_age:.0f}s") if heartbeat_age <= 90
        else color.red(f"{heartbeat_age:.0f}s STALE")
    )
    wandb_text = color.green("online") if state.get("wandb_online_initialized") else color.red("not ready")

    distill = prereg.get("distillation_contract", {})
    lines = [
        color.bold(dashboard_title(declaration.get("workflow_id"))),
        f"Authority {authority_text}  Train {train_text}  Launcher {launcher_text}  Heartbeat {heartbeat_text}",
        f"W&B {wandb_text}  run={state.get('wandb_run_id', '-')}  auto-restart={'ON' if state.get('automatic_restart') else 'OFF'}",
        "",
        color.bold("进度"),
        f"{bar(relative, total)}  E{relative}/{total} ({percent:.2f}%)  runner_step={absolute}",
        f"有效更新来源: 最后完整日志块  | state快照=E{state_relative}  | 同步={sync_text}",
        f"Runner时钟（独立，不换算E）: log={absolute}  | state={state_absolute}  | 同步={runner_sync_text}",
        f"阶段: {stage}  warmup={min(relative, warmup)}/{warmup}  elapsed={latest.get('Time elapsed', 'N/A')}",
        remaining_text,
        f"最新checkpoint: {latest_checkpoint(state.get('output_directory'), absolute_origin)}",
        "",
        color.bold("冻结合同（不是在线评分）"),
        f"Teacher fresh weights-only  | optimizer fresh | target=pre-prior | VAE epochs={distill.get('student_vae_epochs', 4)}",
        *contract_lines(prereg),
        f"phase_scale={distill.get('phase_scale', 'N/A')}  rear_box_scale={distill.get('rear_box_scale', 'N/A')}  PPO=OFF",
        "",
        color.bold("核心蒸馏指标（当前值；箭头=最近20轮方向，不是门禁）"),
        "  ".join((
            cell("Teacher MSE", "Loss/Teacher_Action_MSE", latest, blocks, color),
            cell("Latent MSE", "Loss/Distill_Latent_MSE", latest, blocks, color),
            cell("Velocity", "Loss/VAE_Vel_MSE", latest, blocks, color),
        )),
        "  ".join((
            cell("Phase MSE", "Loss/Highstep_Phase_Teacher_Action_MSE", latest, blocks, color),
            cell("Rear-box", "Loss/Highstep_Rear_Box_Action_MSE", latest, blocks, color),
            cell("Mu OOB", "Debug/Mu_Out_Of_Bounds_Ratio", latest, blocks, color),
        )),
        "  ".join((
            cell("Recon", "Loss/VAE_Recon_MSE", latest, blocks, color),
            cell("KL", "Loss/VAE_KL", latest, blocks, color),
            cell("Prior-box", "Loss/Prior_Box_Loss", latest, blocks, color),
        )),
        "",
        color.bold("机制与运行健康"),
        f"Actor adapt={fmt(latest.get('Debug/Student_Actor_Adapt_Enabled'), 0)}  Box adapt={fmt(latest.get('Debug/Student_Box_Head_Adapt_Enabled'), 0)}  Actor grad={fmt(latest.get('Debug/Student_Actor_Grad_Norm'))}",
        f"Reward={fmt(latest.get('Mean reward'), 2)}  Episode len={fmt(latest.get('Mean episode length'), 1)}  Bad orient={fmt(latest.get('Episode_Termination/bad_orientation'))}",
        f"速度={fmt(latest.get('steps_per_second'), 0)} steps/s  单轮={fmt(latest.get('Iteration time'), 2)}s  趋势窗口可用样本={len(blocks)}",
        "",
        color.dim(f"日志: {train_log}"),
        color.dim("每3秒刷新；Ctrl+C只关闭本查看器，不会停止训练。原始日志可用：dtlog raw"),
    ]
    fatal = state.get("last_error")
    if fatal:
        lines.insert(3, color.red(f"ERROR: {fatal}"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("mode", nargs="?", choices=("dashboard", "raw"), default="dashboard")
    args = parser.parse_args()
    if args.mode == "raw":
        declaration = read_json(DASHBOARD)
        state_path = Path(str(declaration.get("state_path", "")))
        state = read_json(state_path)
        log_path = resolve_train_log(state_path, state)
        os.execvp("tail", ["tail", "-n", "120", "-F", str(log_path)])
    use_color = sys.stdout.isatty() and not args.no_color
    while True:
        if not args.once:
            print("\033[2J\033[H", end="")
        print(render(use_color), flush=True)
        if args.once:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
