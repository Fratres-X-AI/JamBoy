#!/usr/bin/env python3
"""Monte-Carlo realism stress — parallel geo_match + nav latency across conditions."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.geo_match import CONDITION_SEASON_HINT, build_geo_matcher_for_condition
from jamboy.gpu_backend import resolve_device


CONDITIONS = ("clear", "smoke", "low_light", "snow", "textureless")


def _worker_eval(args: tuple) -> dict:
    seed, condition, frame_paths, truth_xy, origin_lat, origin_lon, device = args
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from jamboy.config import load_config
    from jamboy.geo_match import GeoMatcher
    from jamboy.map_loader import build_map_index
    from jamboy.gpu_backend import resolve_device

    cfg = load_config()
    geo = build_geo_matcher_for_condition(cfg, origin_lat, origin_lon, condition, device=device)
    season_hint = CONDITION_SEASON_HINT.get(condition)
    rng = np.random.default_rng(seed)
    ok, confs, lats, errs = 0, [], [], []
    alt = 100.0 + rng.uniform(-5, 5)
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        t0 = time.perf_counter()
        fix = geo.find_absolute_fix(img, altitude_m=alt, preferred_season=season_hint)
        lats.append((time.perf_counter() - t0) * 1000.0)
        if fix and fix.confidence > 0.15:
            ok += 1
            confs.append(fix.confidence)
            try:
                frame_idx = int(Path(fp).stem.split("_")[-1])
                gx, gy = truth_xy[frame_idx]
                errs.append(float(np.hypot(fix.x_local - gx, fix.y_local - gy)))
            except (IndexError, ValueError):
                pass
    n = max(len(frame_paths), 1)
    return {
        "seed": seed,
        "condition": condition,
        "success_pct": 100.0 * ok / n,
        "confidence_mean": float(np.mean(confs)) if confs else 0.0,
        "rmse_m": float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf"),
        "latency_ms_avg": float(np.mean(lats)) if lats else 0.0,
        "latency_ms_p95": float(np.percentile(lats, 95)) if lats else 0.0,
        "frames": n,
    }


def nvidia_snapshot() -> dict:
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]),
            "mem_util_pct": float(parts[1]),
            "vram_mb": float(parts[2]),
            "power_w": float(parts[3]),
            "temp_c": float(parts[4]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def load_truth_xy(path: Path) -> tuple[list[tuple[float, float]], float, float]:
    from pyproj import Geod

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return [], 0.0, 0.0

    origin_lat = float(rows[0]["lat"])
    origin_lon = float(rows[0]["lon"])
    geod = Geod(ellps="WGS84")
    out: list[tuple[float, float]] = []
    for row in rows:
        lat = float(row["lat"])
        lon = float(row["lon"])
        az12, _, dist = geod.inv(origin_lon, origin_lat, lon, lat)
        out.append((dist * np.sin(np.radians(az12)), dist * np.cos(np.radians(az12))))
    return out, origin_lat, origin_lon


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=int, default=500)
    parser.add_argument("--workers", type=int, default=31)
    parser.add_argument("--frames-per-traj", type=int, default=40)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    device = resolve_device(cfg) if args.device == "auto" else args.device
    degraded = cfg.resolve(cfg.sim.get("degraded_dir", "data/sim/degraded"))
    truth_xy, origin_lat, origin_lon = load_truth_xy(cfg.resolve("data/sim/ground_truth.csv"))

    tasks: list[tuple] = []
    per_cond = max(1, args.trajectories // len(CONDITIONS))
    for ci, cond in enumerate(CONDITIONS):
        frames = sorted((degraded / cond).glob("frame_*.png"))
        if not frames:
            continue
        for i in range(per_cond):
            seed = ci * 10_000 + i
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(frames), size=min(args.frames_per_traj, len(frames)), replace=True)
            paths = [frames[int(j)] for j in pick]
            # Round-robin device: most workers CPU to saturate cores; every 8th uses GPU
            dev = device if (i % 8 == 0 and device == "cuda") else "cpu"
            tasks.append((seed, cond, paths, truth_xy, origin_lat, origin_lon, dev))

    workers = min(args.workers, 31, max(1, mp.cpu_count() - 1), len(tasks))
    t0 = time.perf_counter()
    results: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker_eval, t) for t in tasks]
        for fut in as_completed(futs):
            results.append(fut.result())

    wall_s = time.perf_counter() - t0
    by_cond: dict[str, list] = {c: [] for c in CONDITIONS}
    for r in results:
        by_cond.setdefault(r["condition"], []).append(r)

    summary = {
        "trajectories": len(results),
        "workers": workers,
        "wall_s": wall_s,
        "throughput_traj_per_s": len(results) / max(wall_s, 1e-6),
        "cpu_count": mp.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / (1024**3),
        "nvidia_smi": nvidia_snapshot(),
        "by_condition": {},
    }
    all_success, all_lat = [], []
    for cond, rows in by_cond.items():
        if not rows:
            continue
        succ = [r["success_pct"] for r in rows]
        lats = [r["latency_ms_avg"] for r in rows]
        rmses = [r["rmse_m"] for r in rows if np.isfinite(r["rmse_m"])]
        all_success.extend(succ)
        all_lat.extend(lats)
        all_rmse = summary.setdefault("_all_rmse", [])
        all_rmse.extend(rmses)
        summary["by_condition"][cond] = {
            "n": len(rows),
            "success_pct_mean": statistics.mean(succ),
            "success_pct_std": statistics.pstdev(succ) if len(succ) > 1 else 0.0,
            "rmse_m_mean": statistics.mean(rmses) if rmses else float("inf"),
            "rmse_m_std": statistics.pstdev(rmses) if len(rmses) > 1 else 0.0,
            "latency_ms_avg": statistics.mean(lats),
            "latency_ms_p95_mean": statistics.mean([r["latency_ms_p95"] for r in rows]),
        }

    if all_success:
        all_rmse_values = summary.pop("_all_rmse", [])
        summary["overall"] = {
            "success_pct_mean": statistics.mean(all_success),
            "success_pct_std": statistics.pstdev(all_success) if len(all_success) > 1 else 0.0,
            "rmse_m_mean": statistics.mean(all_rmse_values) if all_rmse_values else float("inf"),
            "rmse_m_std": statistics.pstdev(all_rmse_values) if len(all_rmse_values) > 1 else 0.0,
            "latency_ms_avg": statistics.mean(all_lat),
        }

    out = cfg.resolve("data/sim/output/monte_carlo_stress.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.json_only:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Monte-Carlo: {len(results)} trajectories, {workers} workers, {wall_s:.1f}s")
        print(json.dumps(summary["by_condition"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
