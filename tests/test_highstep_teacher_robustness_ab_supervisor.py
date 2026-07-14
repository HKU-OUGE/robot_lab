import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_executor_forbids_training_paths_and_binds_snapshot_replay():
    source = (ROOT / "scripts/rsl_rl/base/highstep_teacher_robustness_ab_play.py").read_text()
    assert "optimizer_step_calls" in source
    assert "backward_calls" in source
    assert "runner_learn_calls" in source
    assert "Teacher B cannot run before Teacher A freezes snapshot" in source
    assert "physical snapshot replay mismatch" in source
    ast.parse(source)


def test_supervisor_uses_exact_204_order_and_never_launches_train():
    source = (ROOT / "tools/highstep_teacher_robustness_ab_supervisor.py").read_text()
    assert 'for teacher in ("A", "B")' in source
    assert "highstep_teacher_robustness_ab_play.py" in source
    assert "scripts/rsl_rl/base/train.py" not in source
    assert '"--eval_action_delay_steps", "0"' in source
    assert '"--fixed_velocity_command", "0.45", "0.0", "0.0"' in source
    ast.parse(source)


def test_monitor_is_read_only_and_dashboard_selected():
    source = (ROOT / "tools/highstep_teacher_robustness_ab_monitor.py").read_text()
    assert "highstep_dashboard_active_workflow.json" in source
    assert "mtime" not in source
    assert "systemctl\", \"--user\", \"start" not in source
    assert "write_text" not in source
    ast.parse(source)


def test_executor_bootstraps_repo_root_when_directly_executed_from_project_root():
    script = ROOT / "scripts/rsl_rl/base/highstep_teacher_robustness_ab_play.py"
    code = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='v111_import_probe'); "
        "from tools.highstep_teacher_robustness_ab import matrix_by_id; "
        "assert len(matrix_by_id()) == 102"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
