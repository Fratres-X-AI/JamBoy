"""Terminal-phase visual tracker with range-from-bbox estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TerminalTrack:
    """Bearing offset (px) and slant range estimate from apparent target size."""

    offset_x_px: float
    offset_y_px: float
    range_m: float
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


class _TemplateTracker:
    """Fallback when contrib trackers are unavailable (headless opencv-python)."""

    def __init__(self) -> None:
        self._template: np.ndarray | None = None
        self._box: tuple[float, float, float, float] = (0, 0, 0, 0)

    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x, y, bw, bh = bbox
        self._box = (float(x), float(y), float(bw), float(bh))
        self._template = frame[int(y) : int(y + bh), int(x) : int(x + bw)].copy()

    def update(self, frame: np.ndarray) -> tuple[bool, tuple[float, float, float, float]]:
        if self._template is None:
            return False, self._box
        x, y, bw, bh = self._box
        h, w = frame.shape[:2]
        pad = 20
        x0 = max(0, int(x) - pad)
        y0 = max(0, int(y) - pad)
        x1 = min(w, int(x + bw) + pad)
        y1 = min(h, int(y + bh) + pad)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0 or self._template.size == 0:
            return False, self._box
        th, tw = self._template.shape[:2]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return True, self._box
        res = cv2.matchTemplate(roi, self._template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        nx, ny = x0 + max_loc[0], y0 + max_loc[1]
        self._box = (float(nx), float(ny), bw, bh)
        return True, self._box


def _create_tracker(kind: str):
    kind = kind.upper()
    factories = {
        "CSRT": getattr(cv2, "TrackerCSRT_create", None),
        "KCF": getattr(cv2, "TrackerKCF_create", None),
        "MOSSE": getattr(cv2, "legacy_TrackerMOSSE_create", getattr(cv2, "TrackerMOSSE_create", None)),
        "MIL": getattr(cv2, "TrackerMIL_create", None),
    }
    for name in (kind, "CSRT", "KCF", "MIL", "MOSSE"):
        factory = factories.get(name)
        if factory is None:
            continue
        try:
            return factory()
        except cv2.error:
            continue
    logger.info("OpenCV contrib tracker unavailable — using template fallback")
    return _TemplateTracker()


class TerminalObjectTracker:
    """
    OpenCV correlation tracker for TERMINAL guidance.
    Range estimated from known target width and pinhole geometry.
    """

    def __init__(self, config: dict[str, Any] | Any) -> None:
        if hasattr(config, "navigator"):
            nav_cfg = config.navigator
        else:
            nav_cfg = config
        term = nav_cfg.get("terminal", {}) if isinstance(nav_cfg, dict) else {}

        self.tracker_type = str(term.get("tracker_type", "CSRT"))
        self.target_width_m = float(term.get("target_width_m", 2.0))
        self.init_bbox_fraction = float(term.get("init_bbox_fraction", 0.15))
        self.min_confidence = float(term.get("min_confidence", 0.35))
        self.focal_px = float(term.get("focal_px", 500.0))

        self._tracker: Any = None
        self.active = False
        self.last_track: TerminalTrack | None = None

    def _default_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        side = max(24, int(min(h, w) * self.init_bbox_fraction))
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return x0, y0, side, side

    def initialize(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> bool:
        tracker = _create_tracker(self.tracker_type)
        if tracker is None:
            return False
        box = bbox or self._default_bbox(frame)
        try:
            if hasattr(tracker, "init"):
                tracker.init(frame, box)
            else:
                return False
        except cv2.error as exc:
            logger.warning("Tracker init failed: %s", exc)
            return False
        self._tracker = tracker
        self.active = True
        self.last_track = None
        return True

    def _range_from_bbox(self, bw: float, altitude_m: float | None) -> float:
        if bw <= 1.0 or self.focal_px <= 0:
            return float("inf")
        ground_width_m = self.target_width_m
        if altitude_m is not None and altitude_m > 0:
            # slant range from similar triangles: bw/f = W/R
            return max(5.0, ground_width_m * self.focal_px / bw)
        return max(5.0, 100.0 * self.target_width_m / bw)

    def update(
        self,
        frame: np.ndarray,
        altitude_m: float | None = None,
    ) -> TerminalTrack | None:
        if not self.active or self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(frame)
        except cv2.error as exc:
            logger.debug("Tracker update failed: %s", exc)
            self.active = False
            return None
        if not ok:
            self.active = False
            return None

        x, y, bw, bh = (float(v) for v in box)
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        tcx, tcy = x + bw / 2.0, y + bh / 2.0
        area_ratio = (bw * bh) / max(w * h, 1.0)
        conf = float(np.clip(area_ratio * 50.0, 0.0, 1.0))
        if conf < self.min_confidence:
            return None

        track = TerminalTrack(
            offset_x_px=tcx - cx,
            offset_y_px=tcy - cy,
            range_m=self._range_from_bbox(bw, altitude_m),
            confidence=conf,
            bbox=(int(x), int(y), int(bw), int(bh)),
        )
        self.last_track = track
        return track

    def reset(self) -> None:
        self._tracker = None
        self.active = False
        self.last_track = None
