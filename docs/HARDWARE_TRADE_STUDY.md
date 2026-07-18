# Hardware Trade Study

## Recommended COTS prototype (~$250)

**Partner-demo stack** — full write-up: [`COTS_PROTOTYPE.md`](COTS_PROTOTYPE.md)

| Part | Role |
|------|------|
| Raspberry Pi 5 8GB | Companion computer |
| Arducam IMX296 | Global shutter downward camera |
| FTDI USB-TTL → Pixhawk TELEM2 | MAVLink serial bridge |
| Active cooler + 32GB SD | Reliability |

Config: `config/pi5_pixhawk.yaml` — CPU ORB, serial MAVLink @ 57600.

## Minimum viable (use what you have)

| Part | Cheapest option | Notes |
|------|-----------------|-------|
| FC | Any used Pixhawk / Matek F405 + baro | MAVLink out; iNav/Betaflight need bridge |
| Camera | Raspberry Pi Camera v2 (IMX219) | Rolling shutter — enable `rolling_shutter_fallback` |
| Better camera | Arducam OV9281 / IMX296 global shutter | ~$30–80 if already owned |
| NIR speckle | 850 nm LED + diffuser | Optional; required for textureless without map pass |
| Compute | RPi 4 4GB or laptop on bench | Geo ORB @320px ~10–30 ms/frame CPU |
| Mount | 3D-printed downward bracket | Soft-mount rubber for vibration |

**No new purchases assumed** — stack degrades gracefully on rolling shutter.

## Tier comparison (when buying later)

| Tier | Example | Est. Power | Mass | 7" FPV | Long-Range |
|------|---------|------------|------|--------|------------|
| A — Heavy | Jetson Orin Nano 8GB | 5–15 W | ~120 g+ | Poor | Good |
| B — Mid | Coral USB / Hailo-8 | 2–5 W | ~30–50 g | Marginal | Good |
| C — Light | RPi 5 / FC-only USB cam | 1–3 W | ~20–40 g | Best | Limited |

**Camera:** Global-shutter (IMX296/OV9281 class) eliminates flow/geo failure under prop wash. IMX219 works on bench; risky in hover.

**Active NIR speckle (850 nm):** IR LED + diffuser co-mounted. Mission map flown with projector ON. ~$5–15 BOM.

## Interface by tier

| Tier | Camera | IMU/Baro | Nav out |
|------|--------|----------|---------|
| C — Light | USB `/dev/video0` | MAVLink from FC USB | UDP to FC `14550` |
| B — Mid | CSI/USB on companion | MAVLink relay | Same |
| A — Heavy | Multiple cam + GPU geo | High-rate IMU sync | Same + lower latency |

## Phase 1 default

Develop on a workstation or cloud GPU host; deploy to **Tier C** first with `scripts/main_navigator.py`.

## Drop-in checklist reference

See [`HARDWARE_DEPLOY.md`](HARDWARE_DEPLOY.md) for step-by-step when hardware arrives.
