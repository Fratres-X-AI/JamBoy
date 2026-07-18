# JamBoy Architecture

## Pipeline

```
Camera (V4L2 / file / Gazebo)
    + IMU (MAVLink ATTITUDE) + Baro (GLOBAL_POSITION_INT)
    → optical_flow.py (LK velocity, gyro + rolling-shutter correction)
    → geo_match.py (ORB map match, partial-match + altitude scale gate)
    → ekf.py (6-state fusion)
    → navigator.py (state machine + health flags)
    → mavlink_bridge.py (VISION_POSITION_ESTIMATE / DO_REPOSITION fallback)
```

## State Vector

`[x, y, z, vx, vy, vz]` in local NED meters.

## State Machine

| State | Entry | Behavior |
|-------|-------|----------|
| INIT | Boot | Await first geo fix |
| CRUISE | Geo init OK | Fuse flow + geo-match |
| DEAD_RECKON | Vision loss | Flow velocity only, max 60 s |
| TERMINAL | < 100 m to target | Object tracker stub |
| ABORT | Drift/integrity fail | RTL via MAVLink |

## Hardware interfaces

### Input
- **Camera:** `hardware.camera_device` (default `/dev/video0`), 30 Hz BGR frames
- **IMU:** `ATTITUDE` roll/pitch/yaw rates at `hardware.imu_rate_hz` (200 nominal)
- **Baro:** `GLOBAL_POSITION_INT.relative_alt` at `hardware.baro_rate_hz` (50 nominal)

### Output
- **Primary:** `VISION_POSITION_ESTIMATE` at `mavlink.command_hz` (5 Hz default), gated by `NavHealth.healthy`
- **Fallback:** `DO_REPOSITION` when geo confidence > 0.2 and vision send fails
- **Health:** `SYS_STATUS` onboard control sensors field (best-effort)

## Sim realism layer

`jamboy/realism.py` injects FPV vibration (10–30 Hz harmonics), motion blur from velocity, lighting swing, and partial occlusion into `generate_dummy_data.py --vibration`.

## Measurement Models

- **Optical flow** → velocity observation; altitude from baro scales pixels→meters
- **Geo-match** → position observation; partial matches accepted at reduced confidence
- **Baro** → z-state update
- Outlier gating via Mahalanobis distance

## Config layers

| File | Purpose |
|------|---------|
| `config/default.yaml` | Full stack defaults |
| `config/calibration.yaml` | Camera intrinsics from chessboard script |

## Entry points

| Script | Use |
|--------|-----|
| `scripts/run_simulation.py` | Synthetic benchmark |
| `scripts/main_navigator.py` | **Hardware** main loop |
| `scripts/calibrate_camera_imu.py` | Intrinsics calibration |

## Gazebo / PX4 stub

`gazebo/jamboy_downward_camera.sdf` + `gazebo/px4_sitl_launch.sh` — camera plugin pointing down, MAVLink to `udp:127.0.0.1:14550`.
