# JamBoy

[![CI](https://github.com/Fratres-X-AI/JamBoy/actions/workflows/ci.yml/badge.svg)](https://github.com/Fratres-X-AI/JamBoy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Jam-proof GPS-denied navigation** for attritable drones: optical flow + offline map geo-registration + EKF fusion + barometric altitude.

Name note: **JamBoy** = GPS **jam** + **Game Boy**-era pun. Nothing to do with historical golf caddie slang.

JamBoy is a **navigation layer**, not a targeting or weapons system. It is sim-validated on synthetic data; it is **not flight-certified**.

## What it does

Given a downward camera, pre-loaded GeoTIFF maps, IMU rates, and barometer altitude, JamBoy estimates local position and velocity without GPS:

1. Lucas–Kanade optical flow with gyro de-rotation → ground velocity  
2. ORB (optional SIFT) + RANSAC homography against tiled maps → absolute geo fix  
3. 6-state EKF fusion with baro altitude  
4. State machine: Cruise → DeadReckon → Terminal (stub) → Abort  
5. MAVLink out: `VISION_POSITION_ESTIMATE` (preferred) or reposition fallback  

Global shutter is strongly preferred; rolling shutter is supported with fallback modes.

## Install (local / CI)

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

CPU runtime only (no CUDA wheels):

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

Optional GPU acceleration (CuPy / PyTorch):

```bash
pip install -r requirements-gpu.txt
# or: pip install -e ".[gpu,dev]"
```

## Quick sim gate

```bash
export PYTHONPATH=src   # Windows: $env:PYTHONPATH = "src"
python scripts/generate_dummy_data.py
pytest -q
python scripts/run_simulation.py --cpu --profile
python scripts/validate_sim.py
```

One-shot helper (CUDA if available, else CPU):

```bash
bash scripts/run_sim_confident.sh
```

Realism / vibration data:

```bash
python scripts/generate_dummy_data.py --realism
python scripts/run_simulation.py --cpu --profile --realism
```

Pass criteria and metrics: [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md), [`docs/SIM_TEST_GUIDE.md`](docs/SIM_TEST_GUIDE.md).

## Hardware

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

## Architecture

| Module | Role |
|--------|------|
| [`src/jamboy/optical_flow.py`](src/jamboy/optical_flow.py) | LK flow + gyro de-rotation |
| [`src/jamboy/geo_match.py`](src/jamboy/geo_match.py) | ORB/SIFT + FLANN/RANSAC map match |
| [`src/jamboy/ekf.py`](src/jamboy/ekf.py) | 6-state EKF |
| [`src/jamboy/navigator.py`](src/jamboy/navigator.py) | Fusion + state machine |
| [`src/jamboy/mavlink_bridge.py`](src/jamboy/mavlink_bridge.py) | Sim log or MAVLink SITL/hardware |
| [`src/jamboy/gpu_backend.py`](src/jamboy/gpu_backend.py) | Optional CUDA (CuPy) matcher |

Details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md)

## Optional cloud GPU

Any CUDA host works. Use your own SSH endpoint from your cloud provider dashboard — **do not commit hostnames, ports, or keys**.

```bash
ssh root@<POD_IP> -p <PORT> -i ~/.ssh/<your_key>
cd /workspace/JamBoy   # or your clone path
source .venv/bin/activate
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
pip install -e ".[gpu,dev]"
bash scripts/run_sim_confident.sh
```

## Disclaimer

JamBoy is research / simulation software for GPS-denied navigation experiments. It does not provide targeting, ROE, or kinetic autonomy. Validate on your own hardware before flight. Use in accordance with applicable law and export controls.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Security reports: [`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE) — Copyright (c) 2026 Fratres-X AI
