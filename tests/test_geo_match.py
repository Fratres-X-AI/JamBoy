import numpy as np
import cv2
import rasterio
from pathlib import Path

from jamboy.config import load_config
from jamboy.geo_match import GeoMatcher
from jamboy.map_loader import MapIndex


def _make_test_map(path: Path, size: int = 1024) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(30, 200, (size, size), dtype=np.uint8)
    for i in range(20):
        cv2.circle(img, (rng.integers(0, size), rng.integers(0, size)), 40, int(rng.integers(0, 255)), -1)
    for i in range(0, size, 48):
        cv2.line(img, (i, 0), (i, size), 95, 1)
        cv2.line(img, (0, i), (size, i), 95, 1)
    transform = rasterio.transform.from_bounds(30.0, 50.0, 30.02, 50.02, size, size)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size,
        count=1, dtype=img.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(img, 1)


def test_geo_match_finds_fix_on_shifted_patch(tmp_path):
    cfg = load_config()
    map_path = tmp_path / "test.tif"
    _make_test_map(map_path)

    index = MapIndex(cfg.geo_match, descriptor="orb")
    index.load_geotiff(map_path)

    with rasterio.open(map_path) as src:
        full = src.read(1)

    crop = 256
    row, col = 400, 400
    patch = full[row : row + crop, col : col + crop]
    frame = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    matcher = GeoMatcher(cfg, index, origin_lat=50.01, origin_lon=30.01)
    fix = matcher.find_absolute_fix(frame, altitude_m=100.0)

    assert fix is not None
    assert fix.confidence > 0.2


def test_multi_season_index_loads(tmp_path):
    cfg = load_config()
    summer = tmp_path / "route_summer.tif"
    winter = tmp_path / "route_winter.tif"
    _make_test_map(summer)
    _make_test_map(winter)

    index = MapIndex(cfg.geo_match, descriptor="orb")
    index.load_geotiff(summer)
    index.load_geotiff(winter)
    assert "summer" in index.seasons or "default" in index.seasons
    assert "snow" in index.seasons
    assert len(index.tiles) >= 2


def test_edge_hybrid_finds_fix(tmp_path):
    cfg = load_config()
    raw = cfg.raw.copy()
    raw.setdefault("geo_match", {})["edge_mode"] = "hybrid"
    from jamboy.config import JamBoyConfig

    c = JamBoyConfig(raw=raw, project_root=cfg.project_root)
    map_path = tmp_path / "test.tif"
    _make_test_map(map_path)
    index = MapIndex(c.geo_match, descriptor="orb")
    index.load_geotiff(map_path)

    with rasterio.open(map_path) as src:
        full = src.read(1)
    patch = full[400:656, 400:656]
    frame = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    matcher = GeoMatcher(c, index, origin_lat=50.01, origin_lon=30.01)
    fix = matcher.find_absolute_fix(frame, altitude_m=100.0)
    assert fix is not None
    assert fix.confidence > 0.15
    assert fix.match_mode in ("intensity", "edge", "template", "template_edge")


def test_bad_frame_returns_none(tmp_path):
    cfg = load_config()
    map_path = tmp_path / "test.tif"
    _make_test_map(map_path)
    index = MapIndex(cfg.geo_match)
    index.load_geotiff(map_path)
    matcher = GeoMatcher(cfg, index)

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert matcher.find_absolute_fix(blank) is None
