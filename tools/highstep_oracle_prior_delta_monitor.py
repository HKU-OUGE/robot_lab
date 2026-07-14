#!/usr/bin/env python3
"""Materialize the v1.9 oracle monitor as one exact v1.8 derivation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat

from tools.highstep_v18_core9_monitor import expected_rendered as v18_expected_rendered


NEEDLE = '    "$PY" scripts/rsl_rl/base/play.py\n'
REPLACEMENT = '    "$PY" scripts/rsl_rl/base/highstep_oracle_prior_delta_play.py\n'


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_rendered() -> str:
    source = v18_expected_rendered()
    if source.count(NEEDLE) != 1:
        raise RuntimeError("v1.8 monitor canonical play command anchor changed")
    rendered = source.replace(NEEDLE, REPLACEMENT)
    if rendered.replace(REPLACEMENT, NEEDLE) != source:
        raise RuntimeError("oracle monitor changed content outside the exact launcher path")
    return rendered


def materialize(output: Path) -> str:
    rendered = expected_rendered()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_text() != rendered:
        raise RuntimeError(f"stored oracle monitor differs from exact derivation: {output}")
    if not output.exists():
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered)
        temporary.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        os.replace(temporary, output)
    if output.stat().st_mode & 0o222:
        raise RuntimeError("oracle monitor must be read-only")
    return sha256_file(output)
