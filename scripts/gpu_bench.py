#!/usr/bin/env python3
"""GPU-saturated benchmark — single process, batched CUDA ops, no CPU process spam."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.geo_match import GeoMatcher
from jamboy.gpu_backend import CudaTimer, batch_hamming_gpu, gpu_device_info, resolve_device
from jamboy.map_loader import build_map_index


def nvidia_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": float(parts[0]) if parts else 0,
            "mem_util_pct": float(parts[1]) if len(parts) > 1 else 0,
            "mem_used_mb": float(parts[2]) if len(parts) > 2 else 0,
            "mem_total_mb": float(parts[3]) if len(parts) > 3 else 0,
            "temp_c": float(parts[4]) if len(parts) > 4 else 0,
            "power_w": float(parts[5]) if len(parts) > 5 else 0,
        }
    except Exception as exc:
        return {"error": str(exc)}


def run(batch_size: int = 32, passes: int = 20, use_gpu: bool = True) -> dict:
    cfg = load_config()
    device = resolve_device(cfg, cli_gpu=use_gpu)
    frames_dir = cfg.resolve(cfg.sim.get("frames_dir", "data/sim/frames"))
    output_dir = cfg.resolve(cfg.sim.get("output_dir", "data/sim/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print("No frames. Run generate_dummy_data.py first.")
        sys.exit(1)

    from jamboy.gpu_backend import detect_gpu_backend, gpu_saturate

    print("=== GPU BENCH === backend:", detect_gpu_backend())
    print(json.dumps(gpu_device_info(), indent=2))
    print("nvidia-smi (before):", nvidia_snapshot())

    map_index = build_map_index(cfg)
    geo = GeoMatcher(cfg, map_index, origin_lat=50.01, origin_lon=30.01, device=device)

    # Warmup
    for p in frame_paths[: cfg.compute.get("warmup_frames", 5)]:
        f = cv2.imread(str(p))
        if f is not None:
            geo.find_absolute_fix(f, altitude_m=100.0)
    if device == "cuda":
        import cupy as cp
        cp.cuda.Stream.null.synchronize()

    latencies = []
    gpu_latencies = []
    successes = 0
    total_ops = 0

    t_wall = time.perf_counter()
    for pass_i in range(passes):
        batch_frames = []
        batch_desc = []
        orb = cv2.ORB_create(nfeatures=1000)

        for p in frame_paths:
            f = cv2.imread(str(p))
            if f is None:
                continue
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            batch_frames.append(f)

            t0 = time.perf_counter()
            with CudaTimer(device):
                fix = geo.find_absolute_fix(f, altitude_m=100.0)
            latencies.append((time.perf_counter() - t0) * 1000)
            if fix:
                successes += 1
                gpu_latencies.append(fix.gpu_ms)
            total_ops += 1

        # Extra GPU saturation: batch hamming on largest tile
        if device == "cuda" and map_index.get_all_tiles():
            tile = map_index.get_all_tiles()[0]
            if tile.desc is not None:
                descs = []
                for f in batch_frames[:batch_size]:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    _, d = orb.detectAndCompute(g, None)
                    if d is not None:
                        descs.append(d)
                if descs:
                    with CudaTimer("cuda") as ct:
                        batch_hamming_gpu(descs, tile.desc, device="cuda")
                    gpu_latencies.append(ct.elapsed_ms())

        if device == "cuda":
            burst = gpu_saturate(duration_sec=5.0)
            gpu_latencies.append(burst["wall_sec"] * 1000 / max(burst["matmul_ops"], 1))

    wall_sec = time.perf_counter() - t_wall
    snap_after = nvidia_snapshot()

    report = {
        "device": device,
        "gpu_info": gpu_device_info(),
        "passes": passes,
        "total_geo_ops": total_ops,
        "success_pct": round(100.0 * successes / max(total_ops, 1), 1),
        "wall_sec": round(wall_sec, 2),
        "throughput_ops_per_sec": round(total_ops / wall_sec, 1),
        "geo_latency_ms_avg": round(float(np.mean(latencies)), 2) if latencies else 0,
        "geo_latency_ms_p99": round(float(np.percentile(latencies, 99)), 2) if latencies else 0,
        "gpu_match_ms_avg": round(float(np.mean(gpu_latencies)), 2) if gpu_latencies else 0,
        "nvidia_smi_after": snap_after,
    }

    out = output_dir / "gpu_bench.json"
    with out.open("w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU baseline")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--passes", type=int, default=20)
    args = parser.parse_args()
    run(batch_size=args.batch, passes=args.passes, use_gpu=not args.cpu)
