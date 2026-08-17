# JamBoy

[![CI](https://github.com/Fratres-X-AI/JamBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/Fratres-X-AI/JamBoy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

GPS-denied navigation layer for small drones: optical flow + offline map match + EKF + baro.

**Sim-validated. Not flight-certified. Not jam-proof.** The name is GPS **jam** + **Game Boy** — a pun, not a claim.

JamBoy is a **navigation layer**, not a targeting or weapons system. See [`docs/NON_CLAIMS.md`](docs/NON_CLAIMS.md).

## Clone gate (one path)

Requires Python 3.10+. Linux is the reliable full gate (`rasterio` / GDAL). Windows can run it when those imports work; otherwise use the laptop subset below.

```bash
git clone https://github.com/Fratres-X-AI/JamBoy.git
cd JamBoy
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/quickstart.py
```

Expected last line: `JAMBOY_SIM_PASS`

That command generates dummy data, runs tests, runs the CPU sim, and checks readiness. No `PYTHONPATH`. No CUDA.

Windows if `rasterio` is blocked (WDAC / GDAL):

```powershell
powershell -File scripts/check_laptop.ps1
```

That is a non-geo subset, not the public sim gate. CI on `main` is the proof.

## What it does

Given a downward camera, pre-loaded GeoTIFF maps, IMU rates, and barometer altitude, JamBoy estimates local position and velocity without GPS:

1. Lucas–Kanade optical flow with gyro de-rotation → ground velocity
2. ORB (optional SIFT) + RANSAC homography against tiled maps → absolute geo fix
3. 6-state EKF fusion with baro altitude
4. State machine: Cruise → DeadReckon → Terminal (stub) → Abort
5. MAVLink out: `VISION_POSITION_ESTIMATE` (preferred) or reposition fallback

Global shutter is strongly preferred; rolling shutter is a degraded fallback.

Pass criteria: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md) · [`docs/SIM_TEST_GUIDE.md`](docs/SIM_TEST_GUIDE.md)

## Optional extras

CPU-only install without the editable extra (same deps as CI):

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
```

GPU (CUDA host, not the laptop default):

```bash
python -m pip install -e ".[gpu,dev]"
bash scripts/run_sim_confident.sh
```

## Hardware (unvalidated)

| Path | Doc |
|------|-----|
| Drop-in checklist (any FC + camera) | [`docs/HARDWARE_DEPLOY.md`](docs/HARDWARE_DEPLOY.md) |
| ~$250 COTS BOM (Pi 5 + IMX296 + Pixhawk) | [`docs/COTS_PROTOTYPE.md`](docs/COTS_PROTOTYPE.md) |
| Hardware trade study | [`docs/HARDWARE_TRADE_STUDY.md`](docs/HARDWARE_TRADE_STUDY.md) |

```bash
python scripts/calibrate_camera_imu.py --camera 0
# Set mavlink.sim_mode: false for a real FC
python scripts/main_navigator.py --config config/pi5_pixhawk.yaml
```

This path has not passed a flight gate.

## Architecture

| Module | Role |
|--------|------|
| [`src/jamboy/optical_flow.py`](src/jamboy/optical_flow.py) | LK flow + gyro de-rotation |
| [`src/jamboy/geo_match.py`](src/jamboy/geo_match.py) | ORB/SIFT + FLANN/RANSAC map match |
| [`src/jamboy/ekf.py`](src/jamboy/ekf.py) | 6-state EKF |
| [`src/jamboy/navigator.py`](src/jamboy/navigator.py) | Fusion + state machine |
| [`src/jamboy/mavlink_bridge.py`](src/jamboy/mavlink_bridge.py) | Sim log or MAVLink SITL/hardware |
| [`src/jamboy/gpu_backend.py`](src/jamboy/gpu_backend.py) | Optional CUDA (CuPy) matcher |

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md) · Capabilities: [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md)

## References

Methods and standards this stack builds on (not an exhaustive survey):

1. **B. D. Lucas and T. Kanade**, “An Iterative Image Registration Technique with an Application to Stereo Vision,” *IJCAI*, 1981. — Optical flow (`optical_flow.py`).
2. **J.-Y. Bouguet**, “Pyramidal Implementation of the Lucas Kanade Feature Tracker,” Intel / OpenCV technical report, 2000. — Pyramid LK used in practice via OpenCV.
3. **E. Rublee, V. Rabaud, K. Konolige, and G. Bradski**, “ORB: An Efficient Alternative to SIFT or SURF,” *ICCV*, 2011. — Primary geo-match descriptors (`geo_match.py`).
4. **D. G. Lowe**, “Distinctive Image Features from Scale-Invariant Keypoints,” *IJCV*, 2004. — Optional SIFT path + Lowe ratio test for match filtering.
5. **M. A. Fischler and R. C. Bolles**, “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,” *CACM*, 1981. — Homography estimation / outlier rejection (RANSAC).
6. **R. E. Kalman**, “A New Approach to Linear Filtering and Prediction Problems,” *Journal of Basic Engineering*, 1960. — Recursive estimation backbone for the EKF.
7. **Y. Bar-Shalom, X. R. Li, and T. Kirubarajan**, *Estimation with Applications to Tracking and Navigation*. Wiley, 2001. — Multi-sensor fusion, innovation / Mahalanobis gating practices used in `ekf.py`.
8. **G. Conte and P. Doherty**, “An Integrated UAV Navigation System Based on Aerial Image Matching,” *IEEE Aerospace Conference*, 2008. — Aerial image / map registration for GPS-denied flight (geo-match motivation).
9. **A. I. Mourikis and S. I. Roumeliotis**, “A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation,” *ICRA*, 2007. — Broader vision–inertial navigation context (JamBoy uses a simpler 6-state EKF, not full MSCKF).
10. **A. Lukežič, T. Vojíř, L. Čehovin Zajc, J. Matas, and M. Kristan**, “Discriminative Correlation Filter with Channel and Spatial Reliability,” *CVPR*, 2017. — CSRT option in the TERMINAL visual-cue stub (`terminal_tracker.py`).
11. **MAVLink Developer Guide** — `VISION_POSITION_ESTIMATE`, `ATTITUDE`, `GLOBAL_POSITION_INT`, mission/command messages. https://mavlink.io/en/
12. **PX4 Autopilot** — external vision / EKF2 vision fusion (`EKF2_EV_CTRL`, height modes). https://docs.px4.io/

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Security reports: [`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE) — Copyright (c) 2026 Fratres-X AI
