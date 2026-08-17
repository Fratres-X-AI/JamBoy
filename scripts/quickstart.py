#!/usr/bin/env python3
"""One-command CPU clone gate. No CUDA. No PYTHONPATH required."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(label: str, args: list[str]) -> None:
    print(f"==> {label}", flush=True)
    result = subprocess.run(args, cwd=ROOT, check=False)
    if result.returncode != 0:
        print(f"JAMBOY_SIM_FAIL: {label} (exit {result.returncode})", flush=True)
        raise SystemExit(result.returncode)


def _rasterio_ok() -> bool:
    probe = subprocess.run(
        [PY, "-c", "import rasterio; print(rasterio.__version__)"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def main() -> int:
    if not _rasterio_ok():
        print(
            "JAMBOY_SIM_BLOCKED: rasterio is not importable on this host "
            "(common on Windows under WDAC / missing GDAL).",
            flush=True,
        )
        print(
            "Full sim gate is Linux/CI. On this machine run: "
            "powershell -File scripts/check_laptop.ps1",
            flush=True,
        )
        return 2

    _run("generate dummy data", [PY, str(ROOT / "scripts" / "generate_dummy_data.py")])
    _run("pytest", [PY, "-m", "pytest", "-q", "--tb=short"])
    _run(
        "CPU simulation",
        [PY, str(ROOT / "scripts" / "run_simulation.py"), "--cpu", "--profile", "--json-only"],
    )
    _run("validate sim", [PY, str(ROOT / "scripts" / "validate_sim.py"), "--json-only"])
    print("JAMBOY_SIM_PASS", flush=True)
    print("See data/sim/output/sim_readiness.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
