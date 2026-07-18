"""MAVLink output bridge with simulation backend and hardware handoff."""

from __future__ import annotations

import csv
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class NavHealth:
    flow_valid: bool = False
    geo_confidence: float = 0.0
    baro_ok: bool = True
    state: str = "INIT"
    covariance_trace: float = 0.0
    healthy: bool = False

    def as_flags(self) -> dict:
        return {
            "flow_valid": self.flow_valid,
            "geo_confidence": round(self.geo_confidence, 3),
            "baro_ok": self.baro_ok,
            "state": self.state,
            "healthy": self.healthy,
        }


class MavlinkBackend(ABC):
    @abstractmethod
    def send_position(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        health: NavHealth | None = None,
    ) -> None:
        ...

    @abstractmethod
    def abort(self) -> None:
        ...


class SimulationBackend(MavlinkBackend):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "nav_log.csv"
        self._file = self.log_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._fields = [
            "sim_t", "frame", "x", "y", "z", "vx", "vy",
            "state", "geo_confidence", "baro_altitude_m",
            "flow_valid", "pipeline_ms", "nav_healthy",
        ]
        self._writer.writerow(self._fields)

    def send_position(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        health: NavHealth | None = None,
    ) -> None:
        pass

    def log_frame(self, frame: int, sim_t: float, status: dict, pipeline_ms: float = 0.0) -> None:
        self._writer.writerow([
            sim_t,
            frame,
            status.get("x", 0.0),
            status.get("y", 0.0),
            status.get("z", 0.0),
            status.get("vx", 0.0),
            status.get("vy", 0.0),
            status.get("state", ""),
            status.get("geo_confidence", 0.0),
            status.get("baro_altitude_m", 0.0),
            status.get("flow_valid", False),
            round(pipeline_ms, 3),
            status.get("nav_healthy", False),
        ])
        self._file.flush()

    def abort(self) -> None:
        self._writer.writerow([time.time(), 0, 0, 0, 0, 0, 0, "ABORT", 0, 0, False, 0, False])
        self._file.flush()
        logger.warning("Mission ABORT signaled")

    def close(self) -> None:
        self._file.close()


class SITLBackend(MavlinkBackend):
    """Hardware FC bridge: VISION_POSITION_ESTIMATE primary, DO_REPOSITION fallback."""

    def __init__(self, config: dict):
        self.config = config
        self.connection = config.get("connection", "udp:127.0.0.1:14550")
        self.command_hz = float(config.get("command_hz", 5.0))
        self.prefer_vision = bool(config.get("prefer_vision_position", True))
        self.fallback_reposition = bool(config.get("fallback_do_reposition", True))
        self.master = None
        self._last_cmd = 0.0
        self._last_health_report = 0.0
        self._vision_failures = 0
        self._connect()

    def _connect(self) -> None:
        try:
            from pymavlink import mavutil

            self.master = mavutil.mavlink_connection(self.connection)
            self.master.wait_heartbeat(timeout=5)
            logger.info("MAVLink connected: %s", self.connection)
        except Exception as exc:
            logger.error("MAVLink connection failed: %s", exc)
            self.master = None

    def _rate_ok(self) -> bool:
        now = time.time()
        if now - self._last_cmd < 1.0 / max(self.command_hz, 1.0):
            return False
        self._last_cmd = now
        return True

    def send_position(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
        health: NavHealth | None = None,
    ) -> None:
        if self.master is None or not self._rate_ok():
            return

        health = health or NavHealth()
        now_us = int(time.time() * 1e6)
        sent = False

        if self.prefer_vision and health.healthy:
            try:
                self.master.mav.vision_position_estimate_send(
                    now_us, x, y, z, 0.0, 0.0, yaw,
                )
                sent = True
                self._vision_failures = 0
            except Exception as exc:
                self._vision_failures += 1
                logger.error("VISION_POSITION_ESTIMATE failed: %s", exc)

        if not sent and self.fallback_reposition and health.geo_confidence > 0.2:
            try:
                self.master.mav.command_int_send(
                    self.master.target_system,
                    self.master.target_component,
                    0,
                    0,
                    192,
                    0,
                    0,
                    0,
                    0,
                    int(x * 1e7) if abs(x) < 90 else 0,
                    int(y * 1e7) if abs(y) < 180 else 0,
                    z,
                )
                sent = True
            except Exception as exc:
                logger.debug("DO_REPOSITION fallback skipped: %s", exc)

        hr_hz = float(self.config.get("health_report_hz", 1.0))
        if time.time() - self._last_health_report > 1.0 / max(hr_hz, 0.1):
            self._last_health_report = time.time()
            try:
                self.master.mav.sys_status_send(
                    self.master.target_system,
                    0,
                    0,
                    0,
                    500 if health.healthy else 200,
                    0,
                    0,
                    0,
                )
            except Exception:
                pass

    def abort(self) -> None:
        if self.master is None:
            return
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                20,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        except Exception as exc:
            logger.error("RTL command failed: %s", exc)


def create_backend(config) -> MavlinkBackend:
    mavlink_cfg = config.mavlink
    if mavlink_cfg.get("sim_mode", True):
        out = config.resolve(config.sim.get("output_dir", "data/sim/output"))
        return SimulationBackend(out)
    return SITLBackend(mavlink_cfg)
