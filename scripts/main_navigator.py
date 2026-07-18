#!/usr/bin/env python3
"""Hardware entry point — downward camera + MAVLink out. No extra deps beyond requirements.txt."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.config import load_config
from jamboy.map_loader import build_map_index
from jamboy.geo_match import GeoMatcher
from jamboy.mavlink_bridge import create_backend
from jamboy.navigator import Navigator
from jamboy.optical_flow import BaroSample, ImuSample

logger = logging.getLogger(__name__)


def open_camera(hw: dict) -> cv2.VideoCapture:
    source = hw.get("camera_source", "v4l2")
    device = hw.get("camera_device", "/dev/video0")
    if source == "file":
        return cv2.VideoCapture(str(device))
    idx = int(device.replace("/dev/video", "")) if device.startswith("/dev/video") else 0
    cap = cv2.VideoCapture(idx)
    fps = float(hw.get("frame_rate_hz", 30))
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def read_baro_mavlink(backend) -> BaroSample | None:
    if not hasattr(backend, "master") or backend.master is None:
        return None
    msg = backend.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
    if msg is None:
        return None
    alt_m = msg.relative_alt / 1000.0
    return BaroSample(altitude_m=alt_m, timestamp=time.time())


def read_imu_mavlink(backend) -> ImuSample | None:
    if not hasattr(backend, "master") or backend.master is None:
        return None
    msg = backend.master.recv_match(type="ATTITUDE", blocking=False)
    if msg is None:
        return None
    return ImuSample(omega_x=msg.rollspeed, omega_y=msg.pitchspeed, omega_z=msg.yawspeed, timestamp=time.time())


def main() -> int:
    parser = argparse.ArgumentParser(description="JamBoy hardware navigator")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print status only, no MAVLink required")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    hw = cfg.hardware

    map_index = build_map_index(cfg)
    origin_lat = origin_lon = 0.0
    tiles = map_index.get_all_tiles()
    if tiles:
        origin_lat = tiles[0].center_lat
        origin_lon = tiles[0].center_lon

    geo = GeoMatcher(cfg, map_index, origin_lat=origin_lat, origin_lon=origin_lon, device="cpu")
    if args.dry_run:
        cfg.raw.setdefault("mavlink", {})["sim_mode"] = True
    mavlink = create_backend(cfg)
    nav = Navigator(cfg, geo, mavlink)

    cap = open_camera(hw)
    if not cap.isOpened():
        logger.error("Camera not available: %s", hw.get("camera_device"))
        return 1

    logger.info(
        "JamBoy running — camera=%s map=%s mavlink_sim=%s",
        hw.get("camera_device"),
        cfg.maps.get("default_map"),
        cfg.mavlink.get("sim_mode", True),
    )

    frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            imu = read_imu_mavlink(mavlink) if cfg.hardware.get("imu_source") == "mavlink" else None
            baro = read_baro_mavlink(mavlink) if cfg.hardware.get("baro_source") == "mavlink" else None
            status = nav.update(frame, imu=imu, baro=baro, timestamp=time.time())
            if hasattr(mavlink, "log_frame"):
                mavlink.log_frame(frame_i, time.time(), status)
            if frame_i % 30 == 0:
                logger.info(
                    "state=%s healthy=%s geo=%.2f flow=%s",
                    status["state"],
                    status.get("nav_healthy"),
                    status["geo_confidence"],
                    status["flow_valid"],
                )
            frame_i += 1
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        cap.release()
        if hasattr(mavlink, "close"):
            mavlink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
