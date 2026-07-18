"""Realism and hardware-config smoke tests."""

from __future__ import annotations

import numpy as np

from jamboy.config import load_config
from jamboy.optical_flow import ImuSample, OpticalFlowEstimator
from jamboy.realism import apply_prop_wash, apply_realism_pipeline, apply_vibration_warp, fpv_vibration_offset_px


def test_vibration_warp_changes_image():
    cfg = load_config()
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[40:80, 60:100] = 200
    cfg.raw["vibration"] = {"enabled": True, "freq_hz": [15.0], "amplitude_px": [3.0]}
    out = apply_vibration_warp(img, 0.5, cfg.raw["vibration"])
    assert not np.array_equal(img, out)


def test_rolling_shutter_correction_path():
    cfg = load_config()
    cfg.raw["camera"]["global_shutter"] = False
    est = OpticalFlowEstimator(cfg)
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (240, 320), dtype=np.uint8)
    r1 = est.update(img, imu=ImuSample(omega_x=0.1, omega_y=0.05), timestamp=0.0)
    r2 = est.update(img, imu=ImuSample(omega_x=0.1, omega_y=0.05), timestamp=1 / 30.0)
    assert r2.valid or not r1.valid


def test_fpv_prop_wash_changes_bottom_band():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:, :] = (100, 100, 100)
    cfg = {"prop_wash_enabled": True, "prop_wash_band_fraction": 0.2, "prop_wash_strength": 0.6}
    out = apply_prop_wash(img, 0.25, cfg)
    band = int(120 * 0.2)
    assert not np.array_equal(img[-band:, :], out[-band:, :])


def test_fpv_vibration_includes_prop_harmonics():
    cfg = {"freq_hz": [12.0], "amplitude_px": [1.0], "prop_wash_freq_hz": [48.0], "prop_wash_amplitude_px": [2.0]}
    a = fpv_vibration_offset_px(0.0, cfg)
    b = fpv_vibration_offset_px(0.01, cfg)
    assert a != b


def test_realism_pipeline_runs():
    cfg = load_config()
    cfg.raw.setdefault("realism", {})["enabled"] = True
    cfg.raw.setdefault("vibration", {})["enabled"] = True
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    out = apply_realism_pipeline(img, 1.0, cfg.raw)
    assert out.shape == img.shape
