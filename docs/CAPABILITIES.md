# JamBoy Capability Matrix

Status key: **DONE** | **PARTIAL** | **PLANNED** | **N/A**

## 1. Camera Input and Image Processing

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Grab frames ≥60 FPS global shutter | **PARTIAL** | `main_navigator.py` V4L2 @30 Hz default; 60 Hz config-ready |
| Lens distortion correction | **PARTIAL** | Intrinsics + distortion in `config/default.yaml`; undistort in calibrate script output — not yet in live pipeline |
| Grayscale conversion | **DONE** | `optical_flow.py`, `geo_match.py` |
| Low-light / sensor noise filter | **PARTIAL** | Realism sim + degraded sets; no dedicated denoise on live path |

## 2. Optical Flow and Motion Tracking

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Lucas-Kanade feature tracking | **DONE** | `optical_flow.py` |
| Gyro de-rotation | **DONE** | `ImuSample` + `_gyro_correct_flow` |
| Ground speed vectors | **DONE** | Pixels→meters via baro altitude |
| Textureless failure detection | **PARTIAL** | `flow.valid` flag; no explicit water/snow classifier |

## 3. Map Matching (Geo-Registration)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Load GeoTIFF tiles | **DONE** | `map_loader.py` |
| ORB/SIFT landmarks | **DONE** | `geo_match.py` |
| FLANN/BF matcher + GPU Hamming | **DONE** | `geo_match.py`, `gpu_backend.py` |
| Absolute lat/lon position | **DONE** | Homography + `pixel_to_wgs84` |
| Reject bad matches | **DONE** | RANSAC, scale gate, Mahalanobis at EKF layer |

## 4. Sensor Fusion (EKF)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Barometer altitude | **DONE** | `JamBoyEKF.update_barometer` |
| Continuous predict with IMU accel | **DONE** | `JamBoyEKF.predict(dt, imu_accel)` |
| Flow + geo updates | **DONE** | `update_optical_flow`, `update_geo_match` |
| Error margins / gating | **DONE** | Mahalanobis χ² gate (default 9.0) |
| Clean X,Y,Z output | **DONE** | `get_state()`, MAVLink bridge |

## 5. Communication and System Logic

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Listen MAVLink from FC | **PARTIAL** | `main_navigator.py` reads ATTITUDE + GLOBAL_POSITION_INT |
| Send position to FC | **DONE** | `VISION_POSITION_ESTIMATE` + DO_REPOSITION fallback |
| Safety state machine | **DONE** | INIT/CRUISE/DEAD_RECKON/ABORT in `navigator.py` |
| Diagnostic logging | **DONE** | `nav_log.csv`, `SimulationBackend` |

---

## Advanced Engineering (Roadmap)

### Electronic Warfare / Anti-Jamming

| Requirement | Status |
|-------------|--------|
| GPS spoof vs optical speed check | **PLANNED** |
| Laser/blinding saturation detect | **PLANNED** |
| NIR smoke/fog spectrum | **PARTIAL** — `nir_speckle.py`, active IR in config |
| Repeating pattern rejection (crop rows) | **PLANNED** |

### Extreme Fault Tolerance

| Requirement | Status |
|-------------|--------|
| Dead reckoning gyro+baro only | **PARTIAL** — DEAD_RECKON state, flow-only; full IMU propagation **PLANNED** |
| Dynamic AGL via parallax | **PLANNED** |
| Sub-10 ms emergency loop (skip geo) | **PLANNED** |
| Lens smudge detection | **PLANNED** |

### Smarter Map Matching

| Requirement | Status |
|-------------|--------|
| Seasonal change (summer map / winter flight) | **PARTIAL** — snow degraded eval only |
| Sun shadow compensation | **PLANNED** |
| Live map update | **PLANNED** |
| Path-ahead quality prediction | **PLANNED** |

### Hard Real-Time

| Requirement | Status |
|-------------|--------|
| Drop late frames | **PLANNED** |
| Manual memory management | **N/A** — Python; Jetson port may use C++ subset |
| Thread-isolated EKF | **PLANNED** |
| Sub-1 s auto-restart | **PLANNED** |

### Military-Grade Comms

| Requirement | Status |
|-------------|--------|
| Signed MAVLink | **PLANNED** |
| Continuous confidence 0–100% | **PARTIAL** — `JamBoyEKF.confidence_score`, `NavHealth` |
| 60 s blackbox video | **PLANNED** |

---

## Production handoff priorities (next code changes)

1. Undistort in `main_navigator.py` before flow/geo
2. Use `update_geo_match` in CRUISE (not hard position reset)
3. GPS spoof check: compare FC GPS velocity vs optical flow
4. Frame deadline dropper in hardware loop
