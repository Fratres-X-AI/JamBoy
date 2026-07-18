# Confident Simulation Test Guide

Run this **before** trusting any navigation result or moving to PX4 SITL.

## One command (local or any CUDA host)

```bash
cd /path/to/JamBoy
source .venv/bin/activate   # Windows: .venv\Scripts\activate
export PYTHONPATH=src
bash scripts/run_sim_confident.sh
```

Exit code **0** means:
1. Synthetic mission data generated (frames + GT + baro + degraded sets)
2. All unit tests pass
3. Full pipeline profile passes (`all_passed: true` in `benchmarks.json`)
4. `validate_sim.py` confirms nav log matches ground truth (RMSE < 15 m, no ABORT)

## Artifacts to inspect

| File | What it proves |
|------|----------------|
| `data/sim/output/benchmarks.json` | Latency + RMSE + degraded geo targets |
| `data/sim/output/sim_readiness.json` | Final go/no-go checklist |
| `data/sim/output/nav_log.csv` | Per-frame fused state vs sim clock |
| `data/sim/output/rmse_plot.png` | Position error over flight |
| `data/sim/output/rmse_per_frame.csv` | Frame-level errors |

## Requirements for valid sim

1. **Global shutter** camera assumed (`camera.global_shutter: true`)
2. **Active NIR speckle** ON for textureless/low-light (`camera.active_ir_speckle: true`)
3. **Mission map** built with same IR projector geometry (see `nir_speckle.py`)
4. **Baro** CSV present (`data/sim/baro.csv`) for altitude fusion
5. **Multi-season maps** for snow/winter corridors (`maps.season_maps` in `config/default.yaml`)

## Snow / textureless limitations

ORB (and SIFT) depend on distinctive local texture. Uniform snow, water, heavy shadow, or
battle-damaged ground can still fail even with:

- **Multi-season GeoTIFFs** (`test_tile.tif` + `test_tile_winter.tif`) — runtime picks highest-confidence season
- **Hybrid edge matching** (`geo_match.edge_mode: hybrid`) — Sobel/Canny structure on roads, rivers, ridgelines
- **Baro altitude constraint** — rejects homographies with implausible scale vs `reference_altitude_m`
- **Template + edge template fallback** — last resort for speckle-assisted textureless scenes

When all geo fixes fail, the navigator **coasts on dead-reckoning** (`navigator.dead_reckon_max_sec`)
with inflated uncertainty. Prepare winter/summer map variants per route before field trials.

### Map prep

```bash
python scripts/generate_dummy_data.py          # creates test_tile.tif + test_tile_winter.tif
# For real missions: add GeoTIFFs named *winter* / *snow* / *summer* under data/maps/
```

## GPU vs CPU

- **CUDA host**: auto-detects CUDA → `--gpu` / `--device cuda` (CuPy Hamming path when installed)
- **GitHub CI / laptop**: CPU path still must pass all targets (slower geo, same logic)

## Next step after green sim

1. Replace `data/maps/test_tile.tif` with real mission GeoTIFF
2. Re-run mapping pass with IR projector ON
3. Connect `mavlink.sim_mode: false` to PX4 SITL (`udp:127.0.0.1:14550`)
