from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "highstep_phase_residual_preroll.py"


def test_preroll_tool_has_frozen_contiguous_zero_command_contract():
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'settle_steps != 296' in source
    assert 'control_step != len(rows)' in source
    assert 'source pre-roll contains a non-zero command' in source
    assert '"exported_target_limit_violations": 0' in source
    assert '"per_joint_physical_clamp_of_source_post_prior_mapped_target"' in source
