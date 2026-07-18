# COTS Prototype — Pi 5 + IMX296 + Pixhawk

Recommended **partner-demo** stack using commercial off-the-shelf parts. No custom PCBs. Total ~**$227–277**.

## Bill of Materials

| Part | Purpose | Est. cost | Notes |
|------|---------|-----------|-------|
| **Raspberry Pi 5 (8GB)** | Companion computer — EKF, flow, geo-match | $125–175 | DigiKey, CanaKit |
| **Arducam IMX296** | 1.58MP mono **global shutter** downward camera | ~$65 | Arducam, Waveshare |
| **Pi 5 camera ribbon** | 22-pin → 15-pin FPC to CAM1 | ~$5 | Contacts face PCB |
| **FTDI USB-TTL** | Serial MAVLink to FC TELEM2 | ~$12 | TX↔RX cross, GND only — **no 5V** |
| **Pi Active Cooler** | Thermal headroom for OpenCV | ~$10 | Fan → Pi fan header |
| **microSD 32GB Class 10** | OS + maps + config | ~$10 | Pi OS Lite 64-bit |
| **Existing Pixhawk-class FC** | Baro + IMU + motor control | (owned) | TELEM2 port |

**You already need:** downward mount, 5V BEC for Pi, mission GeoTIFF.

## Wiring

```
  Arducam IMX296 (face-down)
           |
    22→15 FPC ribbon
           v
  Raspberry Pi 5  ----USB---->  FTDI USB-TTL
  (JamBoy brain)                    |
                                    | serial (57600)
                                    v
                            Pixhawk TELEM2
                            (TX↔RX, GND)
```

| FTDI wire | FC TELEM2 |
|-----------|-----------|
| TX | RX |
| RX | TX |
| GND | GND |
| 5V | *(leave disconnected)* |

## PX4 / ArduPilot prep

On the flight controller (QGroundControl):

- `SERIAL2` (TELEM2): MAVLink2, baud **57600**
- `EKF2_EV_CTRL = 15` (vision XY + yaw + height)
- `EKF2_HGT_MODE = 3` (vision height) — or baro primary with vision aiding
- `COM_ARM_WO_GPS = 1` for bench / GPS-denied test only

## Software setup (Pi 5)

### 1. Flash OS

[Raspberry Pi Imager](https://www.raspberrypi.com/software/) → **Raspberry Pi OS Lite (64-bit)**. Enable SSH and Wi-Fi in imager advanced options.

### 2. Enable camera

```bash
sudo raspi-config
# Interface Options → Camera → Enable
sudo reboot
```

Verify: `libcamera-hello --list-cameras` or `v4l2-ctl --list-devices`

### 3. Clone JamBoy

```bash
git clone https://github.com/Fratres-X-AI/JamBoy.git
cd JamBoy
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Pi 5 has **no CUDA** — geo-match runs CPU ORB (expect ~10–30 ms/frame at 320px crop).

### 4. Sanity check (no airframe)

```bash
python scripts/generate_dummy_data.py
python scripts/run_simulation.py --cpu --profile
python -m pytest tests/ -q
```

Exit 0 = software stack healthy on Pi.

### 5. Calibrate IMX296

```bash
python scripts/calibrate_camera_imu.py --camera 0 --samples 12
# Merge intrinsics from config/calibration.yaml into your active config
```

### 6. Load mission map

```bash
mkdir -p data/maps
# Copy mission GeoTIFF
cp /path/to/mission_tile.tif data/maps/
```

### 7. Run on hardware

```bash
cp config/pi5_pixhawk.yaml config/active.yaml   # or pass --config
# Edit mavlink.connection if FTDI is not /dev/ttyUSB0
python scripts/main_navigator.py --config config/pi5_pixhawk.yaml
```

Set `mavlink.sim_mode: false` in config for live FC.

## Config profile

Use [`config/pi5_pixhawk.yaml`](../config/pi5_pixhawk.yaml):

- `camera.global_shutter: true`, IMX296 resolution hints
- `hardware.camera_device: /dev/video0`
- `mavlink.connection: serial:/dev/ttyUSB0:57600`
- `compute.device: cpu`

## Partner demo checklist

- [ ] Pi boots, SSH works
- [ ] Camera shows downward scene (`libcamera-vid` or OpenCV grab)
- [ ] QGC shows heartbeat from Pi MAVLink route (optional UDP sniffer)
- [ ] Slide map under camera → QGC position moves
- [ ] `nav_log.csv` or console: `flight_mode=FULL_PRECISION`, `confidence_pct > 50`
- [ ] Tethered hover before free flight

## Limits (be honest in demos)

| Risk | Mitigation |
|------|------------|
| Pi CPU thermal throttle | Active cooler, 320px crop, ORB not SIFT |
| Serial latency | 57600 ok for 5 Hz vision; use 921600 if FC supports |
| No GPU | Degraded geo on large maps — pre-crop tiles |
| Textureless field | NIR speckle mapping pass required |

See also: [`HARDWARE_DEPLOY.md`](HARDWARE_DEPLOY.md), [`CAPABILITIES.md`](CAPABILITIES.md).
