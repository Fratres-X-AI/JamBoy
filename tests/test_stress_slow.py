"""Slow stress tests — run with pytest --runslow --gpu."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.gpu
def test_monte_carlo_mini_batch():
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "monte_carlo_stress.py"),
            "--trajectories",
            "50",
            "--workers",
            "8",
            "--json-only",
        ],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    assert "by_condition" in r.stdout
