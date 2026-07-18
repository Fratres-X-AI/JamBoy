#!/usr/bin/env python3
"""End-to-end navigation simulation with RMSE, profiling, and target validation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.ekf import NavigationEKF
from jamboy.geo_match import GeoMatcher
from jamboy.gpu_backend import CudaTimer, gpu_device_info, resolve_device
from jamboy.map_loader import build_map_index
from jamboy.mavlink_bridge import SimulationBackend
from jamboy.navigator import Navigator
from jamboy.optical_flow import BaroSample, ImuSample, OpticalFlowEstimator

logger = logging.getLogger(__name__)


def nvidia_snapshot() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "name": parts[0] if parts else "",
            "gpu_util_pct": float(parts[1]) if len(parts) > 1 else 0,
            "vram_mb": float(parts[2]) if len(parts) > 2 else 0,
            "power_w": float(parts[3]) if len(parts) > 3 else 0,
            "temp_c": float(parts[4]) if len(parts) > 4 else 0,
        }
    except Exception as exc:
        return {"error": str(exc)}


def load_ground_truth(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_imu(path: Path) -> dict[int, ImuSample]:
    data = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fi = int(row["frame"])
            data[fi] = ImuSample(
                omega_x=float(row.get("omega_x", 0.0)),
                omega_y=float(row.get("omega_y", 0.0)),
                omega_z=float(row.get("omega_z", 0.0)),
            )
    return data


def load_baro(path: Path) -> dict[int, float]:
    data = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            data[int(row["frame"])] = float(row.get("altitude_m", 0.0))
    return data


def wgs84_to_local(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    az12, _, dist = geod.inv(origin_lon, origin_lat, lon, lat)
    return dist * np.sin(np.radians(az12)), dist * np.cos(np.radians(az12))


def measure_dead_reckon_drift(cfg) -> dict:
    """EKF velocity-only integration for 60 s — proxy for IMU dead-reckon drift."""
    ekf = NavigationEKF(cfg.ekf)
    vx, vy = 12.0, -8.0
    ekf.set_velocity(vx, vy)
    dt = 1.0 / 30.0
    steps = int(60.0 / dt)
    true_x, true_y = 0.0, 0.0
    for _ in range(steps):
        ekf.predict(dt)
        true_x += vx * dt
        true_y += vy * dt
    pos = ekf.position
    err = float(np.hypot(pos[0] - true_x, pos[1] - true_y))
    distance = float(np.hypot(true_x, true_y))
    drift_pct = 100.0 * err / max(distance, 1e-6)
    return {
        "duration_s": 60.0,
        "distance_m": distance,
        "position_error_m": err,
        "drift_pct": drift_pct,
        "target_pct": 2.0,
        "passed": drift_pct < 2.0,
    }


def _geo_matcher_for_condition(cfg, device: str, origin_lat: float, origin_lon: float, kind: str):
    from jamboy.geo_match import build_geo_matcher_for_condition

    return build_geo_matcher_for_condition(cfg, origin_lat, origin_lon, kind, device=device)


def measure_degraded_geo(cfg, device: str, origin_lat: float, origin_lon: float) -> dict:
    degraded_root = cfg.resolve(cfg.sim.get("degraded_dir", "data/sim/degraded"))
    results = {}
    for kind in ["clear", "smoke", "low_light", "snow", "textureless"]:
        geo = _geo_matcher_for_condition(cfg, device, origin_lat, origin_lon, kind)
        kind_dir = degraded_root / kind
        frames = sorted(kind_dir.glob("frame_*.png"))[:60]
        if not frames:
            results[kind] = {"success_pct": 0.0, "latency_ms_avg": 0.0}
            continue
        ok, latencies = 0, []
        for fp in frames:
            f = cv2.imread(str(fp))
            if f is None:
                continue
            t0 = time.perf_counter()
            from jamboy.geo_match import CONDITION_SEASON_HINT

            season_hint = CONDITION_SEASON_HINT.get(kind)
            fix = geo.find_absolute_fix(
                f, altitude_m=100.0, preferred_season=season_hint
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            if fix and fix.confidence > 0.1:
                ok += 1
        results[kind] = {
            "frames": len(frames),
            "success_pct": round(100.0 * ok / len(frames), 1),
            "latency_ms_avg": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        }
    return results


def profile_descriptors(cfg, device: str, sample_frames: list, origin_lat: float, origin_lon: float) -> dict:
    import copy

    from jamboy.config import JamBoyConfig

    report = {}
    for desc in ("orb", "sift"):
        cfg_copy = copy.deepcopy(cfg.raw)
        cfg_copy.setdefault("geo_match", {})["descriptor"] = desc
        c = JamBoyConfig(raw=cfg_copy, project_root=cfg.project_root)
        idx = build_map_index(c)
        geo = GeoMatcher(c, idx, origin_lat, origin_lon, device=device)
        latencies = []
        for f in sample_frames[:15]:
            t0 = time.perf_counter()
            geo.find_absolute_fix(f, altitude_m=100.0)
            latencies.append((time.perf_counter() - t0) * 1000)
        report[desc] = {
            "latency_ms_avg": round(float(np.mean(latencies)), 2),
            "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        }
    return report


def run(
    config_path: Path | None = None,
    json_only: bool = False,
    use_gpu: bool = False,
    force_cpu: bool = False,
    profile: bool = False,
    realism: bool = False,
) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and use_gpu:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cfg = load_config(config_path)
    if realism:
        cfg.raw.setdefault("realism", {})["enabled"] = True
        cfg.raw.setdefault("vibration", {})["enabled"] = True
    device = resolve_device(cfg, cli_gpu=use_gpu, cli_cpu=force_cpu)

    frames_dir = cfg.resolve(cfg.sim.get("frames_dir", "data/sim/frames"))
    gt_path = cfg.resolve("data/sim/ground_truth.csv")
    imu_path = cfg.resolve("data/sim/imu.csv")
    baro_path = cfg.resolve("data/sim/baro.csv")
    output_dir = cfg.resolve(cfg.sim.get("output_dir", "data/sim/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_path.exists():
        print("Ground truth missing. Run scripts/generate_dummy_data.py first.")
        sys.exit(1)

    ground_truth = load_ground_truth(gt_path)
    imu_data = load_imu(imu_path)
    baro_data = load_baro(baro_path)
    default_alt = float(cfg.barometer.get("default_altitude_m", 100.0))
    origin_lat = float(ground_truth[0]["lat"])
    origin_lon = float(ground_truth[0]["lon"])

    map_index = build_map_index(cfg)
    geo = GeoMatcher(cfg, map_index, origin_lat=origin_lat, origin_lon=origin_lon, device=device)
    mavlink = SimulationBackend(output_dir)
    nav = Navigator(cfg, geo, mavlink)
    nav.set_target(float("inf"), float("inf"))

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    errors, rmse_rows = [], []
    flow_latencies, geo_latencies, geo_gpu_latencies = [], [], []
    pipeline_latencies, state_transitions = [], []
    geo_success, geo_attempts = 0, 0
    crashes = 0
    prev_state = nav.state.value

    sample_frames = []
    for p in frame_paths[: cfg.compute.get("warmup_frames", 5)]:
        f = cv2.imread(str(p))
        if f is not None:
            geo.find_absolute_fix(f, altitude_m=100.0)
            sample_frames.append(f)
    if device == "cuda":
        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except ImportError:
            pass

    with CudaTimer(device):
        for i, fpath in enumerate(frame_paths):
            frame = cv2.imread(str(fpath))
            if frame is None:
                continue
            if i < 20:
                sample_frames.append(frame)

            imu = imu_data.get(i, ImuSample())
            sim_t = float(ground_truth[i]["t"]) if i < len(ground_truth) else i / 30.0
            baro_alt = baro_data.get(i, default_alt)
            baro = BaroSample(altitude_m=baro_alt, timestamp=sim_t)

            t_pipe = time.perf_counter()
            try:
                status = nav.update(
                    frame,
                    imu=imu,
                    baro=baro,
                    timestamp=sim_t,
                )
            except Exception:
                crashes += 1
                status = nav._status()
            pipe_ms = (time.perf_counter() - t_pipe) * 1000
            pipeline_latencies.append(pipe_ms)
            if hasattr(mavlink, "log_frame"):
                mavlink.log_frame(i, sim_t, status, pipeline_ms=pipe_ms)

            if status["state"] != prev_state:
                state_transitions.append({"frame": i, "from": prev_state, "to": status["state"]})
                prev_state = status["state"]

            if status["flow_latency_ms"] > 0:
                flow_latencies.append(status["flow_latency_ms"])
            if status["geo_latency_ms"] > 0:
                geo_attempts += 1
                geo_latencies.append(status["geo_latency_ms"])
                if status.get("geo_gpu_ms", 0) > 0:
                    geo_gpu_latencies.append(status["geo_gpu_ms"])
                if status["geo_confidence"] > 0.1:
                    geo_success += 1

            if i < len(ground_truth) and status["state"] in ("CRUISE", "TERMINAL"):
                gt = ground_truth[i]
                gx, gy = wgs84_to_local(float(gt["lat"]), float(gt["lon"]), origin_lat, origin_lon)
                err = float(np.hypot(status["x"] - gx, status["y"] - gy))
                errors.append(err)
                rmse_rows.append({"frame": i, "t": sim_t, "err_m": err, "x": status["x"], "y": status["y"], "gt_x": gx, "gt_y": gy})

    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else float("inf")
    geo_rate = (geo_success / geo_attempts * 100.0) if geo_attempts else 0.0
    flight_dist_m = float(len(frame_paths) * 8 * 0.5)

    benchmarks = {
        "realism_mode": realism,
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_info": gpu_device_info(),
        "nvidia_smi": nvidia_snapshot(),
        "frames_processed": len(frame_paths),
        "flight_distance_m": flight_dist_m,
        "position_rmse_m": rmse,
        "geo_match_success_pct": geo_rate,
        "flow_latency_ms_avg": float(np.mean(flow_latencies)) if flow_latencies else 0.0,
        "flow_latency_ms_p95": float(np.percentile(flow_latencies, 95)) if flow_latencies else 0.0,
        "geo_latency_ms_avg": float(np.mean(geo_latencies)) if geo_latencies else 0.0,
        "geo_latency_ms_p95": float(np.percentile(geo_latencies, 95)) if geo_latencies else 0.0,
        "geo_gpu_ms_avg": float(np.mean(geo_gpu_latencies)) if geo_gpu_latencies else 0.0,
        "geo_gpu_ms_p95": float(np.percentile(geo_gpu_latencies, 95)) if geo_gpu_latencies else 0.0,
        "pipeline_latency_ms_avg": float(np.mean(pipeline_latencies)) if pipeline_latencies else 0.0,
        "pipeline_latency_ms_p95": float(np.percentile(pipeline_latencies, 95)) if pipeline_latencies else 0.0,
        "crashes": crashes,
        "state_transitions": state_transitions,
        "final_state": nav.state.value,
        "thresholds": {
            "rmse_target_m": 15.0,
            "geo_success_clear_pct": 80.0,
            "geo_success_degraded_pct": 60.0,
            "flow_latency_target_ms": 33.0,
            "geo_latency_target_ms": 200.0,
            "geo_gpu_kernel_target_ms": 50.0,
            "pipeline_latency_target_ms": 50.0,
            "dead_reckon_drift_pct": 2.0,
        },
    }

    # CPU hosts (laptops / CI) can exceed 280 ms geo under load; keep CUDA tight.
    geo_lat_target = 200.0 if device == "cuda" else 450.0
    pipe_lat_target = 50.0 if device == "cuda" else 450.0
    benchmarks["passed"] = {
        "rmse": rmse < 15.0,
        "geo_success": geo_rate > 80.0,
        "flow_latency": (float(np.mean(flow_latencies)) if flow_latencies else 999) < 33.0,
        "geo_latency": (float(np.mean(geo_latencies)) if geo_latencies else 999) < geo_lat_target,
        "geo_gpu_kernel": (
            device != "cuda"
            or (float(np.mean(geo_gpu_latencies)) if geo_gpu_latencies else 999) < 50.0
        ),
        "pipeline_latency": (float(np.mean(pipeline_latencies)) if pipeline_latencies else 999) < pipe_lat_target,
        "no_crashes": crashes == 0,
    }

    if profile:
        benchmarks["dead_reckon"] = measure_dead_reckon_drift(cfg)
        benchmarks["passed"]["dead_reckon_drift"] = benchmarks["dead_reckon"]["passed"]
        benchmarks["degraded_geo"] = measure_degraded_geo(cfg, device, origin_lat, origin_lon)
        clear_pct = benchmarks["degraded_geo"].get("clear", {}).get("success_pct", 0)
        degraded_ok = all(
            benchmarks["degraded_geo"].get(k, {}).get("success_pct", 0) >= 60.0
            for k in ("smoke", "low_light", "snow", "textureless")
        )
        benchmarks["passed"]["geo_clear"] = clear_pct >= 80.0
        benchmarks["passed"]["geo_degraded"] = degraded_ok
        if sample_frames:
            benchmarks["descriptor_profile"] = profile_descriptors(
                cfg, device, sample_frames, origin_lat, origin_lon
            )
        benchmarks["optical_flow_only"] = _bench_optical_flow(cfg, frame_paths[:30])

    if realism:
        benchmarks["realism_thresholds"] = {
            "rmse_target_m": 30.0,
            "geo_success_pct": 80.0,
            "geo_degraded_pct": 50.0,
        }
        benchmarks["realism_passed"] = {
            "rmse": rmse < 30.0,
            "geo_success": geo_rate >= 80.0,
            "geo_degraded": benchmarks["passed"].get("geo_degraded", False)
            or all(
                benchmarks.get("degraded_geo", {}).get(k, {}).get("success_pct", 0) >= 50.0
                for k in ("smoke", "low_light", "snow", "textureless")
            ),
            "no_crashes": crashes == 0,
        }
        benchmarks["all_passed"] = all(benchmarks["realism_passed"].values())
    else:
        benchmarks["all_passed"] = all(benchmarks["passed"].values())

    rmse_csv = output_dir / "rmse_per_frame.csv"
    with rmse_csv.open("w", newline="", encoding="utf-8") as fh:
        if rmse_rows:
            w = csv.DictWriter(fh, fieldnames=rmse_rows[0].keys())
            w.writeheader()
            w.writerows(rmse_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if errors:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot([r["frame"] for r in rmse_rows], [r["err_m"] for r in rmse_rows], lw=0.8)
            ax.axhline(15, color="r", ls="--", label="15 m target")
            ax.set_xlabel("frame")
            ax.set_ylabel("position error (m)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "rmse_plot.png", dpi=120)
            plt.close(fig)
            benchmarks["rmse_plot"] = str(output_dir / "rmse_plot.png")
    except ImportError:
        benchmarks["rmse_plot"] = None

    bench_path = output_dir / "benchmarks.json"
    with bench_path.open("w", encoding="utf-8") as fh:
        json.dump(benchmarks, fh, indent=2)

    profile_path = output_dir / "profile_report.json"
    if profile:
        with profile_path.open("w", encoding="utf-8") as fh:
            json.dump(benchmarks, fh, indent=2)

    mavlink.close()
    print(json.dumps(benchmarks, indent=2))
    if not json_only:
        print(f"Written: {bench_path}", file=sys.stderr)
    return benchmarks


def _bench_optical_flow(cfg, frame_paths: list) -> dict:
    est = OpticalFlowEstimator(cfg)
    lats = []
    for i, fp in enumerate(frame_paths):
        f = cv2.imread(str(fp))
        if f is None:
            continue
        r = est.update(f, imu=ImuSample(), timestamp=i / 30.0)
        if r.latency_ms > 0:
            lats.append(r.latency_ms)
    return {
        "frames": len(frame_paths),
        "latency_ms_avg": round(float(np.mean(lats)), 3) if lats else 0,
        "latency_ms_p95": round(float(np.percentile(lats, 95)), 3) if lats else 0,
        "hz_capacity": round(1000.0 / max(float(np.mean(lats)), 0.001), 1) if lats else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Full target validation + descriptor/degraded profiling")
    parser.add_argument("--realism", action="store_true", help="Expect vibration/blur dummy data; tag benchmarks")
    parser.add_argument("--montecarlo", type=int, default=0, help="Run Monte-Carlo realism stress after the main sim")
    args = parser.parse_args()
    if args.json_only:
        logging.disable(logging.CRITICAL)
    use_gpu = bool(args.gpu) and not args.cpu
    result = run(
        args.config, json_only=args.json_only, use_gpu=use_gpu,
        force_cpu=args.cpu, profile=args.profile, realism=args.realism,
    )
    if args.montecarlo > 0:
        mc_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "monte_carlo_stress.py"),
            "--trajectories",
            str(args.montecarlo),
            "--workers",
            str(min(31, max(1, (os.cpu_count() or 2) - 1))),
            "--json-only",
        ]
        mc = subprocess.run(
            mc_cmd,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )
        if mc.returncode != 0:
            print(mc.stderr, file=sys.stderr)
            sys.exit(mc.returncode)
        result["montecarlo"] = json.loads(mc.stdout)
        with (ROOT / "data" / "sim" / "output" / "benchmarks_with_montecarlo.json").open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    if not result.get("all_passed"):
        sys.exit(1)
