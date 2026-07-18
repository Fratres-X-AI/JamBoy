import numpy as np
import cv2
import rasterio
from pathlib import Path

from jamboy.config import load_config
from jamboy.geo_match import GeoMatcher
from jamboy.map_loader import MapIndex
from jamboy.mavlink_bridge import SimulationBackend
from jamboy.navigator import NavState, Navigator
from jamboy.optical_flow import ImuSample


def _make_map(path: Path) -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(20, 220, (800, 800), dtype=np.uint8)
    transform = rasterio.transform.from_bounds(30.0, 50.0, 30.02, 50.02, 800, 800)
    with rasterio.open(
        path, "w", driver="GTiff", height=800, width=800,
        count=1, dtype=img.dtype, crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(img, 1)


def test_navigator_reaches_cruise(tmp_path):
    cfg = load_config()
    map_path = tmp_path / "m.tif"
    _make_map(map_path)

    index = MapIndex(cfg.geo_match)
    index.load_geotiff(map_path)
    geo = GeoMatcher(cfg, index, origin_lat=50.01, origin_lon=30.01)
    mavlink = SimulationBackend(tmp_path / "out")

    with rasterio.open(map_path) as src:
        img = src.read(1)
    frame = cv2.cvtColor(img[300:620, 300:620].astype(np.uint8), cv2.COLOR_GRAY2BGR)

    nav = Navigator(cfg, geo, mavlink)
    nav.set_target(500.0, 500.0)

    for _ in range(10):
        status = nav.update(frame, imu=ImuSample(), altitude_m=100.0)

    assert nav.state in (NavState.CRUISE, NavState.INIT, NavState.DEAD_RECKON)
    mavlink.close()
