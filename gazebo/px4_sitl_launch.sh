#!/usr/bin/env bash
# PX4 SITL + Gazebo stub launcher. Requires PX4-Autopilot + Gazebo Classic installed locally.
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
if [[ ! -d "$PX4_DIR" ]]; then
  echo "Set PX4_DIR to your PX4-Autopilot clone."
  exit 1
fi
echo "1. Start PX4 SITL (separate terminal):"
echo "   cd $PX4_DIR && make px4_sitl gazebo-classic_iris"
echo "2. Set mavlink.sim_mode: false in config/default.yaml"
echo "3. Run JamBoy:"
echo "   PYTHONPATH=src python scripts/main_navigator.py --config config/default.yaml"
echo "4. Camera topic: subscribe /jamboy/down/image_raw or use v4l2 loopback from Gazebo"
