"""Lucas-Kanade optical flow with gyro de-rotation for GPS-denied navigation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jamboy.realism import apply_motion_blur, motion_blur_from_velocity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowResult:
    vx: float
    vy: float
    valid: bool
    timestamp: float
    latency_ms: float = 0.0
    tracked_features: int = 0
    rolling_shutter_corrected: bool = False
    reject_reason: str = ""


@dataclass
class ImuSample:
    omega_x: float = 0.0
    omega_y: float = 0.0
    omega_z: float = 0.0
    timestamp: float = 0.0


@dataclass
class BaroSample:
    altitude_m: float = 0.0
    timestamp: float = 0.0


class OpticalFlowTracker:
    """
    Tracks ground texture with Lucas-Kanade, removes gyro-induced rotation,
    and emits metric ground velocity for EKF fusion.
    """

    def __init__(self, config: Any) -> None:
        flow_cfg = config.optical_flow if hasattr(config, "optical_flow") else config
        camera = config.camera if hasattr(config, "camera") else {}

        self._flow_cfg = flow_cfg
        self._camera = camera

        intrinsics = camera.get("intrinsics", {})
        self.focal_length_px = float(intrinsics.get("fx", camera.get("focal_length_px", 500.0)))
        self.fy_px = float(intrinsics.get("fy", self.focal_length_px))
        self.cx = float(intrinsics.get("cx", camera.get("frame_width", 640) / 2.0))
        self.cy = float(intrinsics.get("cy", camera.get("frame_height", 480) / 2.0))

        self.global_shutter = bool(camera.get("global_shutter", True))
        self.rolling_fallback = bool(camera.get("rolling_shutter_fallback", True))
        self.readout_time_ms = float(camera.get("readout_time_ms", 32.0))
        self.rs_correction = bool(flow_cfg.get("rolling_shutter_correction", True))

        mb = camera.get("motion_blur", {})
        self.motion_blur_enabled = bool(mb.get("enabled", False))
        self.motion_blur_max_px = float(mb.get("max_blur_px", 2.0))

        self.max_corners = int(flow_cfg.get("max_corners", 100))
        self.quality_level = float(flow_cfg.get("quality_level", 0.01))
        self.min_distance = int(flow_cfg.get("min_distance", 10))
        self.min_tracked_features = int(flow_cfg.get("min_tracked_features", 4))
        self.altitude_m = float(flow_cfg.get("altitude_m", 100.0))

        self.prev_gray: np.ndarray | None = None
        self.prev_pts: np.ndarray | None = None
        self.prev_time: float | None = None
        self._last_vx = 0.0
        self._last_vy = 0.0

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def set_altitude(self, altitude_m: float) -> None:
        if altitude_m > 0.0:
            self.altitude_m = float(altitude_m)

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.prev_time = None

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray | None:
        try:
            if frame is None or frame.size == 0:
                return None
            if frame.ndim == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        except cv2.error as exc:
            logger.error("Grayscale conversion failed: %s", exc)
            return None

    def _detect_corners(self, gray: np.ndarray) -> np.ndarray | None:
        try:
            return cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
            )
        except cv2.error as exc:
            logger.error("Corner detection failed: %s", exc)
            return None

    def _gyro_derotate_pixels(
        self,
        dx: float,
        dy: float,
        imu: ImuSample | None,
        dt: float,
    ) -> tuple[float, float, bool]:
        """
        Strip rotational pixel motion from flow using body rates.
        Pitch rate induces horizontal flow; roll induces vertical flow.
        """
        if imu is None or dt <= 0.0:
            return dx, dy, False

        try:
            corrected_x = dx - (self.focal_length_px * imu.omega_y * dt)
            corrected_y = dy - (self.fy_px * imu.omega_x * dt)
            rs_corrected = False

            if not self.global_shutter and self.rolling_fallback and self.rs_correction:
                readout_s = self.readout_time_ms * 1e-3
                h = float(self._camera.get("frame_height", 480))
                line_factor = (h / 2.0) * readout_s / max(dt, 1e-4)
                corrected_x -= self.focal_length_px * imu.omega_x * line_factor
                rs_corrected = True

            return corrected_x, corrected_y, rs_corrected
        except (TypeError, ValueError) as exc:
            logger.warning("Gyro de-rotation failed: %s", exc)
            return dx, dy, False

    def _pixels_to_velocity(self, dx: float, dy: float, dt: float) -> tuple[float, float]:
        if dt <= 0.0 or self.focal_length_px <= 0.0 or self.altitude_m <= 0.0:
            return 0.0, 0.0
        scale = self.altitude_m / self.focal_length_px
        vx = -(dx / dt) * scale
        vy = -(dy / dt) * scale
        return float(vx), float(vy)

    def _prepare_frame(self, gray: np.ndarray) -> np.ndarray:
        if not self.motion_blur_enabled:
            return gray
        try:
            blur, ang = motion_blur_from_velocity(
                self._last_vx,
                self._last_vy,
                self.altitude_m,
                self.focal_length_px,
                self.motion_blur_max_px,
            )
            if blur <= 0.1:
                return gray
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            bgr = apply_motion_blur(bgr, blur, ang)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        except cv2.error as exc:
            logger.warning("Motion blur prep failed: %s", exc)
            return gray

    def track(
        self,
        frame: np.ndarray,
        imu: ImuSample | None = None,
        timestamp: float | None = None,
        altitude_m: float | None = None,
    ) -> FlowResult:
        """Process one frame and return metric ground velocity."""
        t0 = time.perf_counter()
        now = float(timestamp if timestamp is not None else time.time())

        if altitude_m is not None:
            self.set_altitude(altitude_m)

        gray = self._to_grayscale(frame)
        if gray is None:
            return FlowResult(0.0, 0.0, False, now, reject_reason="empty_frame")

        gray = self._prepare_frame(gray)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_corners(gray)
            self.prev_time = now
            return FlowResult(0.0, 0.0, False, now, reject_reason="bootstrap")

        if self.prev_pts is None or len(self.prev_pts) < self.min_tracked_features:
            self.prev_pts = self._detect_corners(self.prev_gray)

        if self.prev_pts is None or len(self.prev_pts) < self.min_tracked_features:
            self.prev_gray = gray
            self.prev_time = now
            return FlowResult(0.0, 0.0, False, now, reject_reason="insufficient_features")

        try:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params,
            )
        except cv2.error as exc:
            logger.error("LK optical flow failed: %s", exc)
            self.prev_gray = gray
            self.prev_pts = self._detect_corners(gray)
            self.prev_time = now
            return FlowResult(0.0, 0.0, False, now, reject_reason="lk_failure")

        dt = now - (self.prev_time or now)
        if dt <= 0.0 or next_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_corners(gray)
            self.prev_time = now
            return FlowResult(0.0, 0.0, False, now, reject_reason="invalid_dt")

        mask = status.ravel() == 1
        tracked = int(mask.sum())
        if tracked < self.min_tracked_features:
            self.prev_gray = gray
            self.prev_pts = self._detect_corners(gray)
            self.prev_time = now
            return FlowResult(
                0.0, 0.0, False, now,
                tracked_features=tracked,
                reject_reason="feature_dropout",
            )

        good_prev = self.prev_pts[mask]
        good_next = next_pts[mask]
        try:
            flow = (good_next - good_prev).reshape(-1, 2)
            mean_dx, mean_dy = float(flow[:, 0].mean()), float(flow[:, 1].mean())
        except (ValueError, IndexError) as exc:
            logger.error("Flow aggregation failed: %s", exc)
            return FlowResult(0.0, 0.0, False, now, reject_reason="flow_aggregate")

        corrected_dx, corrected_dy, rs_corrected = self._gyro_derotate_pixels(
            mean_dx, mean_dy, imu, dt,
        )
        vx, vy = self._pixels_to_velocity(corrected_dx, corrected_dy, dt)
        self._last_vx, self._last_vy = vx, vy

        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)
        self.prev_time = now

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return FlowResult(
            vx=vx,
            vy=vy,
            valid=True,
            timestamp=now,
            latency_ms=latency_ms,
            tracked_features=tracked,
            rolling_shutter_corrected=rs_corrected,
        )

    def update(
        self,
        frame: np.ndarray,
        imu: ImuSample | None = None,
        timestamp: float | None = None,
        altitude_m: float | None = None,
    ) -> FlowResult:
        """Backward-compatible alias for track()."""
        return self.track(frame, imu=imu, timestamp=timestamp, altitude_m=altitude_m)


OpticalFlowEstimator = OpticalFlowTracker


class DummyOpticalFlowBackend:
    """Replay frames from disk for simulation."""

    def __init__(self, frames_dir: Path) -> None:
        self.frames_dir = Path(frames_dir)
        self.paths = sorted(self.frames_dir.glob("frame_*.png"))
        self.index = 0
        self.estimator: OpticalFlowTracker | None = None

    def read_frame(self) -> np.ndarray | None:
        if self.index >= len(self.paths):
            return None
        frame = cv2.imread(str(self.paths[self.index]))
        self.index += 1
        return frame
