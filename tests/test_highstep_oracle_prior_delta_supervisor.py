from __future__ import annotations

import json
from pathlib import Path

from tools import highstep_oracle_prior_delta_supervisor as supervisor


def test_supervisor_binds_exact_authority_and_disallows_training() -> None:
    prereg = json.loads(supervisor.PREREG.read_text())
    base = json.loads(supervisor.PREREG_BASE.read_text())
    assert supervisor.sha256_file(supervisor.SPEC) == supervisor.SPEC_SHA256
    assert supervisor.sha256_file(supervisor.PREREG) == supervisor.PREREG_SHA256
    assert supervisor.sha256_file(supervisor.CHECKPOINT) == supervisor.CHECKPOINT_SHA256
    assert supervisor.sha256_file(supervisor.TEACHER) == supervisor.TEACHER_SHA256
    assert prereg["training_allowed"] is False
    assert prereg["base_preregistration_sha256"] == supervisor.PREREG_BASE_SHA256
    assert base["residual_training_allowed_before_decision"] is False


def test_derived_monitor_uses_only_oracle_wrapper_as_play_launcher() -> None:
    text = supervisor.MONITOR.read_text()
    assert '"$PY" scripts/rsl_rl/base/highstep_oracle_prior_delta_play.py' in text
    assert '"$PY" scripts/rsl_rl/base/play.py\n' not in text
    assert supervisor.sha256_file(supervisor.MONITOR) == supervisor.MONITOR_SHA256


def test_supervisor_retains_exact_historical_v18_checkpoint_constructor_binding() -> None:
    source = Path(supervisor.__file__).read_text()
    assert '"HIGHSTEP_V18_PREREGISTRATION_PATH"' in source
    assert '"HIGHSTEP_V18_PREREGISTRATION_SHA256"' in source
    assert "fd38dd7e3270fd257c211ca8c23d15d9719cbef3e817dd1281b73f122563fac1" in source
