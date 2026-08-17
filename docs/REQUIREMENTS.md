# JamBoy Requirements Specification

## Hardware (Phase 1 — minimum viable)

| Component | Requirement |
|-----------|-------------|
| FC | Any MAVLink flight controller with baro (MS5611/BMP280 class) |
| Camera | Downward-facing UVC camera; **global shutter strongly preferred** (IMX296/OV9281). Rolling shutter (IMX219) supported via `rolling_shutter_fallback` — expect degraded flow under vibration |
| Active NIR | Optional $5–15 IR LED + diffuser; **required** for textureless geo without prior speckle map |
| Barometer | FC baro → EKF z-state + geo homography scale gate (`GLOBAL_POSITION_INT.relative_alt`) |
| IMU | FC attitude rates → flow de-rotation (`ATTITUDE` ≥50 Hz) |
| Compute | RPi 4/5, laptop, or Jetson — no GPU required on aircraft for ORB geo |

### Interface summary

| Signal | MAVLink / topic | Rate |
|--------|-----------------|------|
| IMU rates | `ATTITUDE` | 50–200 Hz |
| Baro AGL | `GLOBAL_POSITION_INT` | 10–50 Hz |
| Nav out | `VISION_POSITION_ESTIMATE` | 5 Hz (configurable) |
| Fallback | `COMMAND_INT` DO_REPOSITION | On vision failure |
| Abort | `COMMAND_LONG` RTL | On integrity fail |

## Phase 1 Targets — Clean sim (synthetic; RTX PRO 6000, 2026-06-08)

These numbers are a dated **synthetic** run. They are not a field CEP and not a jam-resistance claim.

Command: `python scripts/run_simulation.py --gpu --profile`

| Metric | Result | Target | Pass |
|--------|--------|--------|------|
| Fused position RMSE | **~4.4 m** | < 15 m | **yes** |
| Geo success (fused run) | **100%** | > 80% | **yes** |
| All degraded geo sets | **≥93%** | > 60% | **yes** |
| Pipeline latency avg | **~8 ms** | < 50 ms | **yes** |
| Dead-reckon drift (60 s) | **0%** | < 2% | **yes** |
| **all_passed** | **true** | — | **yes** |

## Phase 1 Targets — Realism sim (vibration + blur)

Command:
```bash
python scripts/generate_dummy_data.py --vibration
python scripts/run_simulation.py --gpu --profile --realism
```

| Metric | Measured (CPU, 2026-06-08) | Realism target | Pass |
|--------|---------------------------|----------------|------|
| Fused RMSE | **28.7 m** | < 30 m | **yes** |
| Geo success (fused) | **95.3%** | > 80% | **yes** |
| Pipeline latency avg | **11.6 ms** | < 50 ms | **yes** |
| Degraded geo (no extra vibration) | clear 100%, smoke 100%, low_light 95%, snow 90%, textureless 100% | ≥50% each | **yes** |
| **realism all_passed** | **true** | — | **yes** |

GPU re-measure when pod is on: `bash scripts/pod_run_all.sh`

**Blunt gap vs real world:** Sim vibration is 2D warp + harmonics — not true rolling-shutter line exposure. First flight on IMX219 without damping **will** see flow dropouts. Global shutter or soft-mount is non-optional for production.

GPU backend: **CuPy** on Blackwell sm_120.

## Operational notes

- Textureless/low-light geo requires **active IR speckle** and **mission maps** built with the same projector geometry.
- Satellite orthophotos without speckle overlay will not match on uniform terrain — fly a mapping pass first.
- Set `geo_match.reference_altitude_m` to your cruise AGL before hardware deploy.
- Run `docs/HARDWARE_DEPLOY.md` drop-in checklist before arming.
