"""Active NIR speckle projector model for textureless surface feature recovery.

Hardware: 850/940 nm IR LED + diffuser (or laser speckle) co-mounted with global-
shutter camera. Map tiles must be collected with the same projector ON (mission
mapping pass) so ORB sees consistent pseudo-texture in live + offline map.
"""

from __future__ import annotations

import numpy as np
import cv2


def build_speckle_field(
    height: int,
    width: int,
    seed: int = 4242,
    blur_ksize: int = 5,
) -> np.ndarray:
    """Fixed pseudo-random speckle field in map/world pixel space."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(30, 225, (height, width), dtype=np.uint8)
    if blur_ksize > 1:
        raw = cv2.GaussianBlur(raw, (blur_ksize, blur_ksize), 1.0)
    return raw


def apply_nir_speckle(
    gray: np.ndarray,
    pattern: np.ndarray,
    row_offset: int = 0,
    col_offset: int = 0,
    intensity: float = 0.35,
) -> np.ndarray:
    """Blend speckle into grayscale image at map-aligned offset."""
    h, w = gray.shape[:2]
    ph, pw = pattern.shape[:2]
    ro = max(0, min(row_offset, ph - h))
    co = max(0, min(col_offset, pw - w))
    patch = pattern[ro : ro + h, co : co + w]
    if patch.shape != gray.shape:
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    blended = (1.0 - intensity) * gray.astype(np.float32) + intensity * patch.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def speckle_from_config(camera_cfg: dict, map_h: int, map_w: int) -> np.ndarray | None:
    if not camera_cfg.get("active_ir_speckle", False):
        return None
    seed = int(camera_cfg.get("speckle_seed", 4242))
    return build_speckle_field(map_h, map_w, seed=seed)
