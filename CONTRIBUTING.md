# Contributing to JamBoy

Thanks for helping improve GPS-denied navigation tooling.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export PYTHONPATH=src
python scripts/generate_dummy_data.py
pytest -q
```

GPU extras (optional): `pip install -e ".[gpu,dev]"` or `pip install -r requirements-gpu.txt`.

## Pull requests

1. Keep changes focused; prefer repair/docs over speculative features.
2. Do not commit secrets, SSH endpoints, API keys, or `.env` files.
3. Run the CPU gate before opening a PR:
   - `python scripts/generate_dummy_data.py`
   - `pytest -q`
   - `python scripts/run_simulation.py --cpu --profile`
   - `python scripts/validate_sim.py`
4. Update docs when behavior or install steps change.

## Code style

Match existing Python style in `src/jamboy/`. Prefer clear names and small diffs.
Do not add heavy GPU dependencies to the default (CPU) install path.

## Dual-use note

JamBoy is a navigation layer for research and simulation. Do not contribute
targeting, strike, or ROE automation.
