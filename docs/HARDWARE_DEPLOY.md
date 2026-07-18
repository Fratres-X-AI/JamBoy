# JamBoy Hardware Deploy Guide

Drop-in steps when you have **any cheap FC + downward camera** — no new purchases required.

**Easiest COTS path (~$250):** see [`COTS_PROTOTYPE.md`](COTS_PROTOTYPE.md) (Pi 5 + IMX296 + FTDI → Pixhawk).

## Minimum Viable Hardware

| Item | Minimum | Strongly preferred |
|------|---------|-------------------|
| Flight controller | Any MAVLink FC with baro (Pixhawk, Matek, iNav, Betaflight + MSP bridge) | Same + spare UART for companion |
| Camera | USB UVC downward cam (IMX219 rolling shutter OK with fallback) | Global shutter (IMX296 / OV9281 class) |
| Compute | Same board as FC if USB cam + Python fits (RPi 4/5, old laptop) | Jetson Orin Nano class if geo GPU needed |
| Map | GeoTIFF from mapping pass or satellite tile | Mission map flown with **NIR speckle ON** |
| Mount | Downward-facing, rigid, ~5 cm above belly | Vibration-damped bracket |

**Not required:** LiDAR, RTK GPS, new ESCs, dedicated GPU on aircraft.

## Exact Hardware Interfaces

### Camera input
| Source | Interface | Config key |
|--------|-----------|------------|
| USB webcam | V4L2 `/dev/video0` | `hardware.camera_source: v4l2` |
| File replay | Path to device node or file | `hardware.camera_source: file` |
| Gazebo (SITL) | ROS topic → v4l2 loopback | `gazebo/jamboy_downward_camera.sdf` |

**Frame format:** BGR `uint8`, 640×480 default, 30 Hz.

### IMU
| Message | Rate | Fields used |
|---------|------|-------------|
| MAVLink `ATTITUDE` | ≥50 Hz (200 preferred) | `rollspeed`, `pitchspeed`, `yawspeed` |

Config: `hardware.imu_source: mavlink`

### Barometer / altitude
| Message | Rate | Fields used |
|---------|------|-------------|
| MAVLink `GLOBAL_POSITION_INT` | ≥10 Hz | `relative_alt` (mm → m) |

Config: `hardware.baro_source: mavlink`, `barometer.enabled: true`

### Navigation output (to FC)
| Priority | MAVLink message | When |
|----------|-----------------|------|
| 1 | `VISION_POSITION_ESTIMATE` | `nav_healthy` and CRUISE/TERMINAL |
| 2 | `COMMAND_INT` DO_REPOSITION | Geo confidence > 0.2, vision fails |
| Abort | `COMMAND_LONG` RTL (20) | Dead-reckon timeout or covariance blow-up |

Config: `mavlink.sim_mode: false`, `mavlink.connection: udp:127.0.0.1:14550`

**PX4 params to set:** `EKF2_EV_CTRL=15`, `EKF2_HGT_MODE=3` (vision height), `COM_ARM_WO_GPS=1` for bench.

## Drop-in Checklist

### Before first power-on
- [ ] Software gate on any CUDA host or CI (see Phase 0 below)
- [ ] Pass: `all_passed: true`, `position_rmse_m` within targets in [`REQUIREMENTS.md`](REQUIREMENTS.md)
- [ ] `sim_readiness.json` → `ready_for_sim: true`
- [ ] Calibrate camera: `python scripts/calibrate_camera_imu.py --camera 0`
- [ ] Merge `config/calibration.yaml` intrinsics into your active config (auto-merged if present)
- [ ] Set `camera.global_shutter: false` if using IMX219 / rolling shutter
- [ ] Set `geo_match.reference_altitude_m` to planned cruise AGL
- [ ] Place mission GeoTIFF in `data/maps/` and set `maps.default_map`
- [ ] If textureless terrain: enable `camera.active_ir_speckle: true` and fly map with projector ON

### Bench test (props off)
- [ ] FC connected via USB or radio → MAVLink heartbeat visible
- [ ] `mavlink.sim_mode: false` in config
- [ ] `python scripts/main_navigator.py --config config/default.yaml`
- [ ] QGroundControl shows vision position moving when sliding map under camera
- [ ] `nav_log.csv` or console: `state=CRUISE`, `healthy=True` within ~2 s of first geo fix

---

## First-Flight Test Plan (tethered → short hop → mission)

Execute in order. **Do not skip steps.** Log everything to `data/sim/output/nav_log.csv` or QGC `.tlog`.

### Phase 0 — Pre-flight software gate (ground, no props)

Run on **any machine with the required deps** (CPU for a smoke gate; CUDA host recommended for full realism + Monte Carlo).

| Step | Action | Pass criteria |
|------|--------|---------------|
| 0.1 | `python scripts/generate_dummy_data.py --realism` | Frames + winter map + degraded sets |
| 0.2 | `python scripts/run_simulation.py --device cuda --profile --realism` (or `--cpu`) | `all_passed: true` (realism RMSE < 30 m) |
| 0.3 | `python -m pytest tests/ -q` | All non-skipped tests pass |
| 0.4 | `python scripts/validate_sim.py --json-only` | `ready_for_sim: true` |
| 0.5 | Confirm `geo_match.reference_altitude_m` = planned AGL | Matches baro cruise altitude ±5 m |

Optional stress (CUDA host): `python scripts/run_simulation.py --device cuda --profile --realism --montecarlo 500`

### Phase 1 — Static bench (props off, aircraft secured)
| Step | Action | Pass criteria |
|------|--------|---------------|
| 1.1 | Power companion + FC, verify MAVLink heartbeat | QGC connected, no prearm errors |
| 1.2 | Run `main_navigator.py`, hold map tile under camera | `state=CRUISE` within 3 s, `geo_confidence > 0.3` |
| 1.3 | Slide map 2 m in X and Y | Vision position in QGC tracks motion, lag < 0.5 s |
| 1.4 | Cover camera 10 s | `state=DEAD_RECKON`; uncover → `CRUISE` within 2 geo cycles |
| 1.5 | Block camera 65 s | `state=ABORT`, RTL command sent |

**Abort if:** geo confidence stuck at 0, vision position diverges >10 m on bench, or CPU cycle_ms > 200 ms sustained.

### Phase 2 — Tethered hover (props on, 1 m leash, open area)
| Step | Action | Pass criteria |
|------|--------|---------------|
| 2.1 | Arm in Altitude mode, hover at 1–2 m AGL | Baro altitude stable ±0.5 m |
| 2.2 | Enable vision fusion (`EKF2_EV_CTRL` set) | QGC shows vision aiding active |
| 2.3 | Hover 60 s over mapped area | Horizontal drift < 5 m, `healthy=True` > 80% of log |
| 2.4 | Light tap on airframe (vibration) | Flow recovers within 1 s; no false geo teleport (`max_geo_jump_m` guard) |

**Abort if:** uncommanded lateral translation > 3 m, oscillation, or `state=ABORT`.

### Phase 3 — Slow transit (3–5 m AGL, 5 m path)
| Step | Action | Pass criteria |
|------|--------|---------------|
| 3.1 | Fly 5 m straight line over map at 2 m/s | RMSE vs GPS/baseline < 15 m (post-flight CSV) |
| 3.2 | Turn 90° at corner | Geo re-lock within 2 s, no ABORT |
| 3.3 | Brief shade / motion blur (cloud pass) | Dead-reckon < 3 s before geo reacquire |

### Phase 4 — Terminal guidance (optional, if target set)
| Step | Action | Pass criteria |
|------|--------|---------------|
| 4.1 | Set target in config / mission planner | `range_to_target_m` decreases in log |
| 4.2 | Enter `state=TERMINAL` inside `terminal_range_m` | CSRT tracker initializes (`terminal_confidence > 0.35`) |
| 4.3 | Approach target marker | `terminal_range_m` within 2× truth; offset_px trends to center |

**Abort if:** tracker confidence drops to 0 for > 5 s in TERMINAL.

### Phase 5 — Post-flight analysis
```bash
python scripts/validate_sim.py --json-only   # if replaying logged frames
# Inspect nav_log.csv: state transitions, geo_confidence, cycle_ms, terminal_range_m
```

| Metric | Target |
|--------|--------|
| Geo match success | > 80% frames in CRUISE |
| Mean pipeline `cycle_ms` | < 100 ms (RPi), < 50 ms (GPU) |
| ABORT count | 0 |
| Max geo confidence drop event | Recovers < 5 s |

---

## Quick start commands

```bash
cd /path/to/JamBoy
source .venv/bin/activate   # Windows: .venv\Scripts\activate
export PYTHONPATH=src

# 1. Calibrate (chessboard)
python scripts/calibrate_camera_imu.py --camera 0

# 2. Load maps
ls data/maps/*.tif

# 3. Sim gate (realism)
python scripts/generate_dummy_data.py --realism
python scripts/run_simulation.py --cpu --profile --realism

# 4. Run navigator (hardware)
# Edit config: mavlink.sim_mode: false
python scripts/main_navigator.py
```

## What will likely break on first real flight

| Risk | Cause | Mitigation |
|------|-------|------------|
| Flow invalid under vibration | Rolling shutter + prop wash | Global shutter cam OR soft-mount + `rolling_shutter_fallback: true` |
| Geo false fix | Wrong tile / no map overlap | Pre-load correct GeoTIFF; check geo confidence >0.3 before arming |
| Scale rejection | Baro altitude wrong | Set `geo_match.reference_altitude_m` to cruise AGL |
| Geo teleport | Bad homography on repetitive texture | `max_geo_jump_m` + altitude scale gate; fly map with NIR speckle |
| MAVLink ignored | PX4 EKF not configured for vision | Set `EKF2_EV_CTRL` params |
| Latency spike on RPi | CPU geo-match | Lower `geo_match.nfeatures`, use ORB only, reduce resolution |
| Textureless field failure | No NIR speckle on map | Fly mapping pass with projector ON |
| TERMINAL tracker loss | Low texture target / motion blur | Use high-contrast target; reduce approach speed |

## Dependencies

Install from `requirements.txt` (CPU) or `requirements-gpu.txt` (CUDA). OpenCV, pymavlink, rasterio, pyproj, filterpy, numpy, pyyaml. No ROS required for bare-metal deploy (Gazebo stub optional for SITL — see [`ROADMAP.md`](ROADMAP.md)).
