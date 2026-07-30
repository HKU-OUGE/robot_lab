from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAY_PATH = ROOT / "scripts/rsl_rl/base/play.py"
HSPLAY_WRAPPER = Path("/home/lxq/.local/bin/hsplay")
HSPLAY_LAUNCHER = (
    ROOT
    / "tmp/highstep_be300_0707_distill_20260716/manual_play_completed_checkpoints.txt"
)
BPLAY = Path("/home/lxq/bplay")


def _array_body(source: str, name: str, *, start: int = 0) -> str:
    match = re.search(rf"{re.escape(name)}=\(\n(?P<body>.*?)\n\s*\)", source[start:], re.DOTALL)
    assert match is not None
    return match.group("body")


def test_play_training_distribution_allows_only_parallel_env_count_override():
    source = PLAY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_TRAINING_DISTRIBUTION_ALLOWED_CONFIG_DIFFS"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == (
        ("scene", "num_envs"),
        ("scene", "terrain", "num_envs"),
    )
    assert "env_cfg.from_dict(saved_config)" in source
    assert "env_cfg.scene.num_envs = requested_num_envs" in source
    assert "env_cfg.scene.terrain.num_envs = requested_num_envs" in source


def test_play_training_distribution_preserves_training_randomization_and_curricula():
    source = PLAY_PATH.read_text(encoding="utf-8")
    assert "if not args_cli.training_distribution:" in source
    assert "env_cfg.scene.terrain.max_init_terrain_level = None" in source
    assert "group.enable_corruption = False" in source
    assert "env_cfg.curriculum.terrain_levels = None" in source
    assert "env_cfg.curriculum.command_levels = None" in source
    assert "[TRAINING_DISTRIBUTION_INITIAL_JSON]" in source
    assert '"root_relative_x"' in source
    assert '"terrain_level_counts"' in source
    assert '"terrain_type_counts"' in source


def test_student_auto_mode_is_hash_bound_and_manual_mode_keeps_recording():
    wrapper = HSPLAY_WRAPPER.read_text(encoding="utf-8")
    assert 'exec bash "$launcher" "$@"' in wrapper
    source = HSPLAY_LAUNCHER.read_text(encoding="utf-8")
    auto_start = source.index('if [[ -n "$AUTO_ENVS" ]]')
    auto_body = _array_body(source, "MODE_ARGS", start=auto_start)
    manual_start = source.index("else", auto_start)
    manual_body = _array_body(source, "MODE_ARGS", start=manual_start)
    assert "--training_distribution" in auto_body
    assert "--training_distribution_env_snapshot" in auto_body
    assert "--training_distribution_env_sha256" in auto_body
    for forbidden in ("--keyboard", "--debug", "--record_joint_data", "--seed"):
        assert forbidden not in auto_body
    for required in ("--keyboard", "--debug", "--record_joint_data", "--seed"):
        assert required in manual_body
    # The launcher uses $RUN_DIR, so validate the declared file and digest directly.
    run_dir = Path(re.search(r"^RUN_DIR=(?P<path>.+)$", source, re.MULTILINE).group("path"))
    env_path = run_dir / "params/env.yaml"
    expected = re.search(
        r"^TRAINING_ENV_SHA256=(?P<sha>[0-9a-f]{64})$", source, re.MULTILINE
    ).group("sha")
    assert hashlib.sha256(env_path.read_bytes()).hexdigest() == expected


def test_teacher_auto_mode_is_hash_bound_and_rejects_manual_overrides():
    source = BPLAY.read_text(encoding="utf-8")
    auto_start = source.index('if [[ -n "$AUTO_ENVS" ]]', source.index("marker_tmp="))
    auto_body = _array_body(source, "MODE_ARGS", start=auto_start)
    manual_start = source.index("else", auto_start)
    manual_body = _array_body(source, "MODE_ARGS", start=manual_start)
    assert "--training_distribution" in auto_body
    for forbidden in ("--keyboard", "--debug", "--record_joint_data", "--seed"):
        assert forbidden not in auto_body
    for required in ("--keyboard", "--record_joint_data", "--seed"):
        assert required in manual_body
    assert "不能与 --debug、--seed、--terrain 或 --hard 混用" in source
    assert "--auto 只支持 b101/b300" in source
    env_text = re.search(r'^TRAINING_ENV="(?P<path>[^\"]+)"$', source, re.MULTILINE).group("path")
    env_path = Path(env_text.replace("$ROOT", str(ROOT)))
    expected = re.search(
        r'^TRAINING_ENV_SHA256="(?P<sha>[0-9a-f]{64})"$', source, re.MULTILINE
    ).group("sha")
    assert hashlib.sha256(env_path.read_bytes()).hexdigest() == expected


def test_play_launchers_have_valid_shell_syntax():
    for script in (HSPLAY_WRAPPER, HSPLAY_LAUNCHER, BPLAY):
        subprocess.run(["bash", "-n", str(script)], check=True)
