#!/usr/bin/env bash
# Confident sim test — no GPU burn loops.
set -euo pipefail
exec "$(dirname "$0")/run_sim_confident.sh"
