"""Navigation state machine with EKF fusion."""

from __future__ import annotations

import enum
import logging
import time

import numpy as np

from jamboy.ekf import NavigationEKF
from jamboy.geo_match import GeoFix, GeoMatcher
from jamboy.mavlink_bridge import MavlinkBackend, NavHealth
from jamboy.optical_flow import BaroSample, FlowResult, ImuSample, OpticalFlowEstimator
from jamboy.terminal_tracker import TerminalObjectTracker

logger = logging.getLogger(__name__)


class NavState(enum.Enum):
    INIT = "INIT"
    CRUISE = "CRUISE"
    DEAD_RECKON = "DEAD_RECKON"
    TERMINAL = "TERMINAL"
    ABORT = "ABORT"


class Navigator:
    def __init__(
        self,
        config,
        geo_matcher: GeoMatcher,
        mavlink: MavlinkBackend,
    ):
        self.config = config
        nav_cfg = config.navigator
        self.state = NavState.INIT
        self.flow = OpticalFlowEstimator(config)
        self.ekf = NavigationEKF(config.ekf, baro_config=config.barometer)
        self.geo = geo_matcher
        self.mavlink = mavlink
        self.tracker = TerminalObjectTracker(config)
        self._terminal_entered = False

        self.dead_reckon_max = float(nav_cfg.get("dead_reckon_max_sec", 60.0))
        self.terminal_range = float(nav_cfg.get("terminal_range_m", 100.0))
        self.max_uncertainty = float(nav_cfg.get("max_position_uncertainty_m", 50.0))

        self.target_x = float("inf")
        self.target_y = float("inf")
        self.last_geo_time = -1.0
        self.geo_interval = 1.0 / float(config.geo_match.get("match_hz", 5))
        self.dead_reckon_start: float | None = None
        self.last_time: float | None = None
        self._sim_clock = 0.0
        self.origin_lat = geo_matcher.origin_lat
        self.origin_lon = geo_matcher.origin_lon
        baro_cfg = config.barometer
        self.baro_enabled = bool(baro_cfg.get("enabled", True))
        self.default_altitude_m = float(baro_cfg.get("default_altitude_m", 100.0))

    def set_target(self, x: float, y: float) -> None:
        self.target_x = x
        self.target_y = y

    def initialize(self, geo_fix: GeoFix) -> None:
        self.ekf.set_position(geo_fix.x_local, geo_fix.y_local)
        self.state = NavState.CRUISE
        logger.info("Navigator initialized at (%.1f, %.1f)", geo_fix.x_local, geo_fix.y_local)

    def _range_to_target(self) -> float:
        pos = self.ekf.position
        return float(np.hypot(self.target_x - pos[0], self.target_y - pos[1]))

    def _transition(self, new_state: NavState) -> None:
        if self.state != new_state:
            logger.info("State %s -> %s", self.state.value, new_state.value)
            if self.state == NavState.TERMINAL and new_state != NavState.TERMINAL:
                self.tracker.reset()
                self._terminal_entered = False
            self.state = new_state

    def update(
        self,
        frame: np.ndarray,
        imu: ImuSample | None = None,
        altitude_m: float | None = None,
        baro: BaroSample | None = None,
        timestamp: float | None = None,
    ) -> dict:
        if timestamp is not None:
            now = timestamp
        else:
            self._sim_clock += 1.0 / 30.0
            now = self._sim_clock

        dt = (now - self.last_time) if self.last_time is not None else 0.033
        self.last_time = now

        if self.state == NavState.ABORT:
            return self._status()

        self.ekf.predict(dt)

        baro_alt = self._resolve_baro_altitude(baro, altitude_m)
        if self.baro_enabled and baro_alt is not None:
            self.ekf.update_altitude(baro_alt)

        flow: FlowResult = self.flow.update(
            frame, imu=imu, timestamp=now, altitude_m=baro_alt,
        )
        geo_fix: GeoFix | None = None

        if (
            self.state in (NavState.CRUISE, NavState.INIT, NavState.DEAD_RECKON)
            and now - self.last_geo_time >= self.geo_interval - 1e-6
        ):
            current_lat = None
            current_lon = None
            if self.state != NavState.INIT:
                pos = self.ekf.position
                current_lat = self.origin_lat + pos[1] / 111_320.0
                current_lon = self.origin_lon + pos[0] / (111_320.0 * np.cos(np.radians(self.origin_lat)))

            geo_fix = self.geo.find_absolute_fix(
                frame,
                current_lat=current_lat,
                current_lon=current_lon,
                altitude_m=baro_alt,
                baro_altitude_m=baro_alt,
            )
            self.last_geo_time = now

            if geo_fix and geo_fix.confidence > 0.1:
                self.ekf.set_position(geo_fix.x_local, geo_fix.y_local)
                self.ekf.zero_velocity()
                if self.state == NavState.INIT:
                    self._transition(NavState.CRUISE)
                elif self.state == NavState.DEAD_RECKON:
                    self.dead_reckon_start = None
                    self._transition(NavState.CRUISE)

        if flow.valid and self.state == NavState.DEAD_RECKON:
            self.ekf.update_velocity(flow.vx, flow.vy)

        if self.state == NavState.INIT and geo_fix is None:
            return self._status(flow=flow)

        vision_lost = not flow.valid and geo_fix is None
        if self.state == NavState.CRUISE and vision_lost:
            if self.dead_reckon_start is None:
                self.dead_reckon_start = now
            self._transition(NavState.DEAD_RECKON)
        elif self.state == NavState.DEAD_RECKON and not vision_lost:
            self.dead_reckon_start = None
            self._transition(NavState.CRUISE)
        elif self.state == NavState.DEAD_RECKON:
            if self.dead_reckon_start and (now - self.dead_reckon_start) > self.dead_reckon_max:
                self.mavlink.abort()
                self._transition(NavState.ABORT)

        if (
            self.state == NavState.CRUISE
            and np.isfinite(self.target_x)
            and self._range_to_target() < self.terminal_range
        ):
            self._transition(NavState.TERMINAL)

        terminal_track = None
        if self.state == NavState.TERMINAL:
            if not self._terminal_entered:
                self._terminal_entered = self.tracker.initialize(frame)
            if self._terminal_entered:
                terminal_track = self.tracker.update(frame, altitude_m=baro_alt)

        if self.ekf.covariance_trace > self.max_uncertainty ** 2:
            self.mavlink.abort()
            self._transition(NavState.ABORT)

        pos = self.ekf.position
        status = self._status(flow=flow, geo_fix=geo_fix, baro_alt=baro_alt, terminal_track=terminal_track)
        health = NavHealth(
            flow_valid=status["flow_valid"],
            geo_confidence=status["geo_confidence"],
            baro_ok=baro_alt is not None,
            state=status["state"],
            covariance_trace=self.ekf.covariance_trace,
            healthy=(
                status["state"] in ("CRUISE", "TERMINAL")
                and (status["flow_valid"] or status["geo_confidence"] > 0.15)
            ),
        )
        self.mavlink.send_position(
            float(pos[0]), float(pos[1]), float(pos[2]),
            health=health,
        )
        status["nav_healthy"] = health.healthy
        return status

    def _resolve_baro_altitude(
        self,
        baro: BaroSample | None,
        altitude_m: float | None,
    ) -> float | None:
        if baro is not None:
            return float(baro.altitude_m)
        if altitude_m is not None:
            return float(altitude_m)
        if self.baro_enabled:
            return self.default_altitude_m
        return None

    def _status(
        self,
        flow: FlowResult | None = None,
        geo_fix: GeoFix | None = None,
        baro_alt: float | None = None,
        terminal_track=None,
    ) -> dict:
        pos = self.ekf.position
        vel = self.ekf.velocity
        status = {
            "state": self.state.value,
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "vx": float(vel[0]),
            "vy": float(vel[1]),
            "flow_valid": flow.valid if flow else False,
            "flow_latency_ms": flow.latency_ms if flow else 0.0,
            "geo_confidence": geo_fix.confidence if geo_fix else 0.0,
            "geo_latency_ms": geo_fix.latency_ms if geo_fix else 0.0,
            "geo_gpu_ms": geo_fix.gpu_ms if geo_fix else 0.0,
            "baro_altitude_m": float(baro_alt) if baro_alt is not None else 0.0,
            "confidence_pct": round(self.ekf.confidence_score, 1),
            "terminal_range_m": 0.0,
            "terminal_confidence": 0.0,
            "terminal_offset_x_px": 0.0,
            "terminal_offset_y_px": 0.0,
        }
        if terminal_track is not None:
            status["terminal_range_m"] = float(terminal_track.range_m)
            status["terminal_confidence"] = float(terminal_track.confidence)
            status["terminal_offset_x_px"] = float(terminal_track.offset_x_px)
            status["terminal_offset_y_px"] = float(terminal_track.offset_y_px)
        return status
