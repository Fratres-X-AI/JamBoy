#!/usr/bin/env bash
# One command: generate data → test → full profile sim → readiness gate.
# Uses CUDA when available; otherwise CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT="data/sim/output"
mkdir -p "$OUT"

DEVICE="cpu"
if python -c "from jamboy.gpu_backend import cuda_available; exit(0 if cuda_available() else 1)" 2>/dev/null; then
  DEVICE="cuda"
fi

REALISM_FLAG=""
if [[ "${REALISM:-0}" == "1" ]]; then
  REALISM_FLAG="--realism"
  python scripts/generate_dummy_data.py --realism
else
  python scripts/generate_dummy_data.py
fi

echo "=== JamBoy confident sim test (device: $DEVICE realism: ${REALISM:-0}) ==="
python -m pytest tests/ -q --tb=line
# Prefer --device when available; fall back to --gpu/--cpu for older checkouts
if python scripts/run_simulation.py --help 2>&1 | grep -q -- '--device'; then
  python scripts/run_simulation.py --device "$DEVICE" --profile $REALISM_FLAG --json-only | tee "$OUT/benchmarks.json"
else
  FLAG="--cpu"
  [[ "$DEVICE" == "cuda" ]] && FLAG="--gpu"
  python scripts/run_simulation.py $FLAG --profile $REALISM_FLAG --json-only | tee "$OUT/benchmarks.json"
fi
python scripts/validate_sim.py --json-only
if [[ "$DEVICE" == "cuda" ]]; then
  python -m pytest tests/ -q --gpu --tb=line || true
fi
echo "=== SIM READY — see $OUT/sim_readiness.json ==="
