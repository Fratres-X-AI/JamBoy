"""Sim-to-real effects: FPV vibration, prop-wash, motion blur, lighting, occlusion."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def fpv_vibration_offset_px(t: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """
    Multi-harmonic FPV vibration + prop harmonics (typical 80–220 Hz motor bleed
    folded into effective pixel motion at 30 Hz sample rate).
    """
    freqs = list(cfg.get("freq_hz", [12.0, 18.0, 25.0]))
    amps = list(cfg.get("amplitude_px", [2.5, 1.5, 0.8]))
    prop_freqs = cfg.get("prop_wash_freq_hz", [48.0, 96.0])
    prop_amps = cfg.get("prop_wash_amplitude_px", [1.2, 0.6])
    for f, a in zip(prop_freqs, prop_amps):
        freqs.append(float(f))
        amps.append(float(a))

    dx = dy = 0.0
    for f, a in zip(freqs, amps):
        phase = f * 2.0 * math.pi * t
        dx += a * math.sin(phase)
        dy += a * math.cos(phase * 1.07)

    roll = math.radians(float(cfg.get("roll_deg", 2.0))) * math.sin(18.0 * t)
    pitch = math.radians(float(cfg.get("pitch_deg", 3.0))) * math.sin(22.0 * t + 0.4)
    return dx, dy, roll + pitch * 0.5


def vibration_offset_px(t: float, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Backward-compatible alias."""
    return fpv_vibration_offset_px(t, cfg)


def apply_prop_wash(bgr: np.ndarray, t: float, cfg: dict[str, Any]) -> np.ndarray:
    """Bottom-band directional smear simulating rotor wash on downward FPV."""
    if not cfg.get("prop_wash_enabled", True):
        return bgr
    h, w = bgr.shape[:2]
    band_frac = float(cfg.get("prop_wash_band_fraction", 0.14))
    band_h = max(8, int(h * band_frac))
    y0 = h - band_h
    out = bgr.copy()
    band = out[y0:h, :].astype(np.float32)
    strength = float(cfg.get("prop_wash_strength", 0.45))
    shift = int(strength * 4 * math.sin(40.0 * t))
    if shift != 0:
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        band = cv2.warpAffine(band.astype(np.uint8), M, (w, band_h), borderMode=cv2.BORDER_REPLICATE)
    blur_k = max(3, int(strength * 7) | 1)
    band = cv2.GaussianBlur(band.astype(np.uint8), (blur_k, blur_k), 0)
    ripple = (8 * np.sin(np.linspace(0, 6 * math.pi, w))).astype(np.int16)
    band = np.clip(band.astype(np.int16) + ripple[np.newaxis, :, np.newaxis], 0, 255).astype(np.uint8)
    out[y0:h, :] = band
    return out


def apply_vibration_warp(
    bgr: np.ndarray,
    t: float,
    vib_cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not vib_cfg.get("enabled", False):
        return bgr
    dx, dy, roll = fpv_vibration_offset_px(t, vib_cfg)
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(roll), 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    out = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    out = apply_prop_wash(out, t, vib_cfg)
    if rng is not None and float(vib_cfg.get("jitter_px", 0.0)) > 0:
        j = float(vib_cfg["jitter_px"])
        out = cv2.warpAffine(
            out,
            np.float32([[1, 0, rng.uniform(-j, j)], [0, 1, rng.uniform(-j, j)]]),
            (w, h),
            borderMode=cv2.BORDER_REPLICATE,
        )
    return out


def apply_motion_blur(bgr: np.ndarray, blur_px: float, angle_deg: float = 0.0) -> np.ndarray:
    if blur_px <= 0.05:
        return bgr
    k = max(3, int(blur_px * 2) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    kernel /= kernel.sum()
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(bgr, -1, kernel)


def motion_blur_from_velocity(vx: float, vy: float, altitude_m: float, focal_px: float, max_px: float) -> tuple[float, float]:
    if focal_px <= 0 or altitude_m <= 0:
        return 0.0, 0.0
    px_per_m = focal_px / altitude_m
    blur = min(max_px, math.hypot(vx, vy) * px_per_m * 0.033)
    angle = math.degrees(math.atan2(vy, vx)) if (vx or vy) else 0.0
    return blur, angle


def apply_lighting_variation(bgr: np.ndarray, factor: float) -> np.ndarray:
    return cv2.convertScaleAbs(bgr, alpha=factor, beta=int((factor - 1.0) * 20))


def apply_partial_occlusion(bgr: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
    if prob <= 0 or rng.random() > prob:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    bw = rng.integers(w // 8, w // 3)
    bh = rng.integers(h // 12, h // 4)
    x0 = rng.integers(0, max(1, w - bw))
    y0 = rng.integers(0, max(1, h - bh))
    color = int(rng.integers(20, 60))
    cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (color, color, color), -1)
    return out


def apply_realism_pipeline(
    bgr: np.ndarray,
    t: float,
    cfg: dict[str, Any],
    rng: np.random.Generator | None = None,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    realism = cfg.get("realism", {})
    if not realism.get("enabled", False):
        return bgr

    rng = rng or np.random.default_rng(int(t * 1000) % 2**31)
    out = bgr
    vib = cfg.get("vibration", {})
    if realism.get("vibration", True):
        out = apply_vibration_warp(out, t, vib, rng)

    if realism.get("motion_blur", True):
        cam = cfg.get("camera", {})
        mb = cam.get("motion_blur", {})
        max_px = float(mb.get("max_blur_px", 2.0))
        focal = float(cam.get("intrinsics", {}).get("fx", cam.get("focal_length_px", 500.0)))
        alt = float(cfg.get("barometer", {}).get("default_altitude_m", 100.0))
        blur, ang = motion_blur_from_velocity(velocity[0], velocity[1], alt, focal, max_px)
        if blur > 0.1:
            out = apply_motion_blur(out, blur, ang)

    if realism.get("variable_lighting", True):
        base = float(realism.get("lighting_base", 1.0))
        swing = float(realism.get("lighting_swing", 0.15))
        factor = base + swing * math.sin(0.4 * t) + swing * 0.3 * rng.standard_normal()
        out = apply_lighting_variation(out, float(np.clip(factor, 0.5, 1.3)))

    prob = float(realism.get("partial_occlusion_prob", 0.05))
    if realism.get("partial_occlusion", True):
        out = apply_partial_occlusion(out, rng, prob)
    return out
