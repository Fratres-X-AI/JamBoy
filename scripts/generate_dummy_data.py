#!/usr/bin/env python3
"""Generate synthetic overflight frames, degraded sequences, and ground truth."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.nir_speckle import apply_nir_speckle, speckle_from_config
from jamboy.realism import apply_realism_pipeline


def create_synthetic_map(path: Path, size: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    base = rng.integers(40, 180, (size, size), dtype=np.uint8)

    for _ in range(80):
        cx, cy = rng.integers(0, size, 2)
        axes = (rng.integers(20, 120), rng.integers(10, 80))
        angle = int(rng.integers(0, 180))
        color = int(rng.integers(20, 220))
        cv2.ellipse(base, (cx, cy), axes, angle, 0, 360, color, -1)

    for _ in range(30):
        x1, y1 = rng.integers(0, size, 2)
        x2, y2 = rng.integers(0, size, 2)
        cv2.line(base, (x1, y1), (x2, y2), int(rng.integers(0, 80)), 3)

    for i in range(0, size, 48):
        cv2.line(base, (i, 0), (i, size), 95, 1)
        cv2.line(base, (0, i), (size, i), 95, 1)

    transform = rasterio.transform.from_bounds(30.0, 50.0, 30.05, 50.05, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype=base.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(base, 1)
    print(f"Created synthetic map: {path}")


def create_winter_map_variant(base_path: Path, winter_path: Path) -> None:
    """Snow-blanketed corridor map with preserved roads/structure for multi-season matching."""
    winter_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(base_path) as src:
        base = src.read(1).astype(np.uint8)
        profile = src.profile.copy()

    snow = np.clip(base.astype(np.float32) * 0.32 + 168.0, 0, 255).astype(np.uint8)
    structure = cv2.Canny(base, 35, 110)
    snow[structure > 0] = np.clip(base[structure > 0] * 0.55 + 40, 0, 255).astype(np.uint8)

    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    roads = (cv2.magnitude(gx, gy) > 18).astype(np.uint8) * 255
    snow[roads > 0] = np.clip(base[roads > 0] * 0.45, 0, 120).astype(np.uint8)

    with rasterio.open(winter_path, "w", **profile) as dst:
        dst.write(snow, 1)
    print(f"Created winter map variant: {winter_path}")


def apply_degradation(bgr: np.ndarray, kind: str) -> np.ndarray:
    out = bgr.copy()
    if kind == "clear":
        return out
    if kind == "smoke":
        fog = np.full_like(out, 180)
        alpha = 0.45
        return cv2.addWeighted(out, 1 - alpha, fog, alpha, 0)
    if kind == "low_light":
        return cv2.convertScaleAbs(out, alpha=0.35, beta=5)
    if kind == "snow":
        noise = np.random.default_rng(7).integers(200, 255, out.shape, dtype=np.uint8)
        return cv2.addWeighted(out, 0.55, noise, 0.45, 0)
    if kind == "textureless":
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (31, 31), 0)
        g = (g // 48) * 48
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return out


def generate(config_path: Path | None = None, vibration: bool = False) -> None:
    cfg = load_config(config_path)
    if vibration:
        cfg.raw.setdefault("vibration", {})["enabled"] = True
        cfg.raw.setdefault("realism", {})["enabled"] = True
    maps_dir = cfg.resolve(cfg.maps.get("directory", "data/maps"))
    map_path = maps_dir / cfg.maps.get("default_map", "test_tile.tif")
    if not map_path.exists():
        create_synthetic_map(map_path)
    winter_path = maps_dir / "test_tile_winter.tif"
    if not winter_path.exists():
        create_winter_map_variant(map_path, winter_path)

    crop = int(cfg.camera.get("crop_size", 320))
    frames_dir = cfg.resolve(cfg.sim.get("frames_dir", "data/sim/frames"))
    degraded_root = cfg.resolve(cfg.sim.get("degraded_dir", "data/sim/degraded"))
    frames_dir.mkdir(parents=True, exist_ok=True)
    degraded_root.mkdir(parents=True, exist_ok=True)

    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    gt_path = cfg.resolve("data/sim/ground_truth.csv")
    imu_path = cfg.resolve("data/sim/imu.csv")
    baro_path = cfg.resolve("data/sim/baro.csv")
    manifest_path = cfg.resolve("data/sim/degraded_manifest.csv")
    base_alt = float(cfg.barometer.get("default_altitude_m", cfg.optical_flow.get("altitude_m", 100.0)))

    with rasterio.open(map_path) as src:
        image = src.read(1)
        h, w = image.shape
        step = 8
        rows = list(range(crop, h - crop, step))

    cam_cfg = cfg.camera
    speckle = speckle_from_config(cam_cfg, h, w)
    speckle_intensity = float(cam_cfg.get("speckle_intensity", 0.35))
    use_speckle = speckle is not None

    gt_rows = []
    imu_rows = []
    baro_rows = []
    baro_rng = np.random.default_rng(99)
    realism_rng = np.random.default_rng(77)
    alt_var = float(cfg.realism.get("altitude_variation_m", 0.0))
    manifest_rows = []
    frame_idx = 0
    kinds = ["clear", "smoke", "low_light", "snow", "textureless"]
    realism_on = bool(cfg.realism.get("enabled", False))

    for kind in kinds:
        kind_dir = degraded_root / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        for old in kind_dir.glob("frame_*.png"):
            old.unlink()

    for row in rows:
        col = crop + (row - crop) % (w - 2 * crop)
        patch = image[row - crop // 2 : row + crop // 2, col - crop // 2 : col + crop // 2]
        if patch.shape[0] != crop or patch.shape[1] != crop:
            continue

        patch_gray = patch.astype(np.uint8)
        if use_speckle:
            row0 = row - crop // 2
            col0 = col - crop // 2
            patch_gray = apply_nir_speckle(
                patch_gray, speckle, row0, col0, intensity=speckle_intensity
            )
        bgr_clear = cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2BGR)
        t = frame_idx / 30.0
        vx_est = 4.0
        bgr_flight = bgr_clear
        if realism_on:
            bgr_flight = apply_realism_pipeline(
                bgr_clear, t, cfg.raw, realism_rng, velocity=(vx_est, 0.0),
            )
        cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.png"), bgr_flight)

        row0 = row - crop // 2
        col0 = col - crop // 2
        for kind in kinds:
            if kind == "textureless":
                g = patch.astype(np.uint8)
                g = cv2.GaussianBlur(g, (31, 31), 0)
                g = (g // 48) * 48
                if use_speckle:
                    g = apply_nir_speckle(g, speckle, row0, col0, intensity=speckle_intensity)
                deg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            elif kind == "low_light":
                deg = cv2.convertScaleAbs(bgr_clear, alpha=0.4, beta=8)
            else:
                deg = apply_degradation(bgr_clear, kind)
            cv2.imwrite(str(degraded_root / kind / f"frame_{frame_idx:05d}.png"), deg)
            manifest_rows.append({"frame": frame_idx, "condition": kind, "path": f"degraded/{kind}/frame_{frame_idx:05d}.png"})

        with rasterio.open(map_path) as src:
            lon, lat = rasterio.transform.xy(src.transform, row, col)

        gt_rows.append({"frame": frame_idx, "t": t, "lat": lat, "lon": lon, "row": row, "col": col})
        pitch_event = 0.12 if realism_on else (0.1 if 50 < frame_idx < 55 else 0.0)
        if realism_on:
            import math
            pitch_event += math.radians(float(cfg.vibration.get("pitch_deg", 3.0))) * math.sin(18.0 * t)
        imu_rows.append({
            "frame": frame_idx,
            "t": t,
            "omega_x": round(float(baro_rng.normal(0, 0.02)), 4),
            "omega_y": round(pitch_event, 4),
            "omega_z": round(float(baro_rng.normal(0, 0.01)), 4),
        })
        alt_noise = alt_var if realism_on else 0.3
        baro_rows.append({
            "frame": frame_idx,
            "t": t,
            "altitude_m": round(base_alt + float(baro_rng.normal(0, alt_noise * 0.1)), 3),
        })
        frame_idx += 1

    with gt_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "t", "lat", "lon", "row", "col"])
        w.writeheader()
        w.writerows(gt_rows)

    with imu_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "t", "omega_x", "omega_y", "omega_z"])
        w.writeheader()
        w.writerows(imu_rows)

    with baro_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "t", "altitude_m"])
        w.writeheader()
        w.writerows(baro_rows)

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "condition", "path"])
        w.writeheader()
        w.writerows(manifest_rows)

    dist_m = frame_idx * step * 0.5
    tag = " [realism/vibration]" if realism_on else ""
    print(f"Generated {frame_idx} clear frames (~{dist_m:.0f} m path){tag} in {frames_dir}")
    print(f"Degraded sets: {', '.join(kinds)} under {degraded_root}")
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--vibration", action="store_true", help="Enable FPV vibration + realism in dummy frames")
    parser.add_argument("--realism", action="store_true", help="Alias for --vibration (FPV realism pipeline)")
    args = parser.parse_args()
    generate(args.config, vibration=args.vibration or args.realism)
