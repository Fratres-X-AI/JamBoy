# Contributing to JamBoy

Thanks for helping improve GPS-denied navigation tooling.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/quickstart.py
```

Expected: `JAMBOY_SIM_PASS`. No `PYTHONPATH`. CPU only.

GPU extras (optional, not the default gate): `python -m pip install -e ".[gpu,dev]"`.

Windows if `rasterio` is blocked: `powershell -File scripts/check_laptop.ps1` (non-geo subset).

## Pull requests

1. Keep changes focused; prefer repair/docs over speculative features.
2. Do not commit secrets, SSH endpoints, API keys, or `.env` files.
3. Run `python scripts/quickstart.py` before opening a PR (Linux/CI if Windows is WDAC-blocked).
4. Update docs when behavior or install steps change.
5. Do not add "jam-proof", "flight-proven", or certified language. See [`docs/NON_CLAIMS.md`](docs/NON_CLAIMS.md).

## Code style

Match existing Python style in `src/jamboy/`. Prefer clear names and small diffs.
Do not add heavy GPU dependencies to the default (CPU) install path.

## Dual-use note

JamBoy is a navigation layer for research and simulation. Do not contribute
targeting, strike, or ROE automation.
