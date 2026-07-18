#!/usr/bin/env python3
"""Post-simulation readiness gate — exit 0 only when confident to trust sim results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            timeout=3,
        ).strip()
    except Exception:
        return "unknown"


def load_gt(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def wgs84_to_local(lat: float, lon: float, o_lat: float, o_lon: float) -> tuple[float, float]:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    az, _, d = geod.inv(o_lon, o_lat, lon, lat)
    return d * np.sin(np.radians(az)), d * np.cos(np.radians(az))


def validate_nav_log(gt: list[dict], log_path: Path, o_lat: float, o_lon: float) -> dict:
    if not log_path.exists():
        return {"passed": False, "error": f"missing {log_path}"}

    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    errors = []
    aborts = 0
    for row in rows:
        if row.get("state") == "ABORT":
            aborts += 1
        fi = int(float(row.get("frame", -1)))
        if fi < 0 or fi >= len(gt):
            continue
        g = gt[fi]
        gx, gy = wgs84_to_local(float(g["lat"]), float(g["lon"]), o_lat, o_lon)
        errors.append(float(np.hypot(float(row["x"]) - gx, float(row["y"]) - gy)))

    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
    max_err = float(max(errors)) if errors else float("inf")
    return {
        "passed": rmse < 15.0 and aborts == 0 and len(rows) > 0,
        "frames_logged": len(rows),
        "nav_log_rmse_m": round(rmse, 3),
        "nav_log_max_err_m": round(max_err, 3),
        "abort_count": aborts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="JamBoy sim readiness validation")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--benchmarks", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = cfg.resolve(cfg.sim.get("output_dir", "data/sim/output"))
    bench_path = args.benchmarks or (out / "benchmarks.json")
    if not bench_path.exists():
        bench_path = out / "profile_report.json"
    if not bench_path.exists():
        bench_path = out / "final_profile.json"

    gt_path = cfg.resolve("data/sim/ground_truth.csv")
    log_path = out / "nav_log.csv"

    report: dict = {
        "git_revision": _git_revision(),
        "ready_for_sim": False,
        "checks": {},
        "failures": [],
    }

    if not bench_path.exists():
        report["failures"].append(f"benchmarks missing: {bench_path}")
    else:
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        report["benchmarks_file"] = str(bench_path)
        report["checks"]["benchmarks_all_passed"] = bool(bench.get("all_passed"))
        report["checks"]["rmse_m"] = bench.get("position_rmse_m")
        report["checks"]["geo_success_pct"] = bench.get("geo_match_success_pct")
        if not bench.get("all_passed"):
            report["failures"].append(f"benchmarks all_passed=false: {bench.get('passed')}")

        deg = bench.get("degraded_geo", {})
        if deg:
            for kind in ("smoke", "low_light", "snow", "textureless"):
                pct = deg.get(kind, {}).get("success_pct", 0)
                ok = pct >= 60.0
                report["checks"][f"degraded_{kind}_pct"] = pct
                if not ok:
                    report["failures"].append(f"degraded {kind} {pct}% < 60%")

    if gt_path.exists():
        gt = load_gt(gt_path)
        o_lat, o_lon = float(gt[0]["lat"]), float(gt[0]["lon"])
        nav_check = validate_nav_log(gt, log_path, o_lat, o_lon)
        report["checks"]["nav_log"] = nav_check
        if not nav_check.get("passed"):
            report["failures"].append(f"nav_log validation failed: {nav_check}")

    report["ready_for_sim"] = len(report["failures"]) == 0

    out_path = out / "sim_readiness.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not args.json_only:
        print(f"Wrote {out_path}", file=sys.stderr)
    return 0 if report["ready_for_sim"] else 1


if __name__ == "__main__":
    sys.exit(main())
