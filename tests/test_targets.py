"""Phase 1 performance target validation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.ekf import NavigationEKF
from jamboy.geo_match import GeoMatcher
from jamboy.map_loader import build_map_index
from jamboy.navigator import Navigator
from jamboy.mavlink_bridge import SimulationBackend
from jamboy.optical_flow import ImuSample, OpticalFlowEstimator


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_optical_flow_under_33ms(cfg):
    est = OpticalFlowEstimator(cfg)
    rng = np.random.default_rng(1)
    lats = []
    for i in range(20):
        img = rng.integers(0, 255, (480, 640), dtype=np.uint8)
        r = est.update(img, imu=ImuSample(), timestamp=i / 30.0)
        if r.latency_ms > 0:
            lats.append(r.latency_ms)
    assert lats, "no flow measurements"
    assert np.mean(lats) < 33.0, f"flow avg {np.mean(lats):.2f} ms"


def test_ekf_mahalanobis_gate(cfg):
    ekf = NavigationEKF(cfg.ekf)
    ekf.set_position(0, 0)
    assert ekf.update_position(1000.0, 1000.0, confidence=0.01) is False


def test_dead_reckon_drift_under_2pct(cfg):
    ekf = NavigationEKF(cfg.ekf)
    vx, vy = 10.0, 5.0
    ekf.set_velocity(vx, vy)
    dt = 1.0 / 30.0
    true_x = true_y = 0.0
    for _ in range(int(60 / dt)):
        ekf.predict(dt)
        true_x += vx * dt
        true_y += vy * dt
    err = np.hypot(ekf.position[0] - true_x, ekf.position[1] - true_y)
    dist = np.hypot(true_x, true_y)
    drift_pct = 100.0 * err / dist
    assert drift_pct < 2.0, f"drift {drift_pct:.2f}%"


def test_navigator_handles_bad_frames(cfg, tmp_path):
    map_path = tmp_path / "m.tif"
    import rasterio
    img = np.random.default_rng(2).integers(20, 200, (512, 512), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(30.0, 50.0, 30.02, 50.02, 512, 512)
    with rasterio.open(
        map_path, "w", driver="GTiff", height=512, width=512, count=1,
        dtype=img.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(img, 1)
    from jamboy.map_loader import MapIndex
    idx = MapIndex(cfg.geo_match)
    idx.load_geotiff(map_path)
    geo = GeoMatcher(cfg, idx, 50.01, 30.01)
    nav = Navigator(cfg, geo, SimulationBackend(tmp_path / "out"))
    good = cv2.cvtColor(img[100:420, 100:420], cv2.COLOR_GRAY2BGR)
    for bad in [None, np.array([]), np.zeros((10, 10, 3), np.uint8)]:
        try:
            if bad is None:
                continue
            nav.update(bad, imu=ImuSample(), altitude_m=100.0, timestamp=0.0)
        except Exception as exc:
            pytest.fail(f"crashed on bad frame: {exc}")
    nav.update(good, imu=ImuSample(), altitude_m=100.0, timestamp=0.1)
    mavlink = nav.mavlink
    if hasattr(mavlink, "close"):
        mavlink.close()


@pytest.mark.gpu
def test_gpu_geo_latency_under_200ms(cfg, use_gpu):
    if not use_gpu:
        pytest.skip("pass --gpu on RunPod")
    from jamboy.gpu_backend import resolve_device
    device = resolve_device(cfg, cli_gpu=True)
    if device != "cuda":
        pytest.skip("CUDA unavailable")
    frames_dir = cfg.resolve("data/sim/frames")
    frames = sorted(frames_dir.glob("frame_*.png"))[:20]
    if not frames:
        pytest.skip("run generate_dummy_data.py first")
    idx = build_map_index(cfg)
    geo = GeoMatcher(cfg, idx, 50.01, 30.01, device=device)
    import time
    lats = []
    for fp in frames:
        f = cv2.imread(str(fp))
        t0 = time.perf_counter()
        geo.find_absolute_fix(f, altitude_m=100.0)
        lats.append((time.perf_counter() - t0) * 1000)
    assert np.mean(lats) < 200.0, f"geo avg {np.mean(lats):.1f} ms"


@pytest.mark.gpu
def test_full_pipeline_profile_passes(use_gpu):
    if not use_gpu:
        pytest.skip("pass --gpu on RunPod")
    import os
    env = {**os.environ, "PYTHONPATH": "src"}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_simulation.py"), "--gpu", "--profile", "--json-only"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip(), proc.stderr
    data = json.loads(proc.stdout)
    required = [
        "rmse", "geo_success", "flow_latency", "geo_latency", "geo_gpu_kernel",
        "pipeline_latency", "no_crashes", "dead_reckon_drift", "geo_clear",
    ]
    failed = [k for k in required if not data.get("passed", {}).get(k)]
    assert not failed, json.dumps(data.get("passed"), indent=2)
    for kind in ("smoke", "low_light", "snow"):
        pct = data.get("degraded_geo", {}).get(kind, {}).get("success_pct", 0)
        assert pct >= 60.0, f"{kind} success {pct}% < 60%"
