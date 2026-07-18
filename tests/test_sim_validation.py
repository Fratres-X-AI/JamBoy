"""Sim readiness validator tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_validate_sim_passes_after_profile_run():
    env = {**dict(__import__("os").environ), "PYTHONPATH": "src"}
    gen = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_dummy_data.py")],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert gen.returncode == 0, gen.stderr
    sim = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_simulation.py"), "--cpu", "--profile", "--json-only"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert sim.stdout.strip(), sim.stderr
    bench = json.loads(sim.stdout)
    assert bench.get("all_passed"), bench.get("passed")

    val = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_sim.py"), "--json-only"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert val.returncode == 0, val.stdout + val.stderr
    report = json.loads(val.stdout)
    assert report["ready_for_sim"] is True
