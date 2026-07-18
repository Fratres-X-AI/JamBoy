"""GeoTIFF tiling, spatial index, and coordinate transforms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio

from jamboy.nir_speckle import apply_nir_speckle, speckle_from_config
from rasterio.windows import Window
from rasterio.transform import rowcol

logger = logging.getLogger(__name__)


@dataclass
class MapTile:
    key: str
    image: np.ndarray
    transform: Any
    crs: Any
    row_off: int
    col_off: int
    kp: list | None = None
    desc: np.ndarray | None = None
    center_lat: float = 0.0
    center_lon: float = 0.0
    season: str = "default"
    source_stem: str = ""


class MapIndex:
    """Grid-indexed map tiles with precomputed ORB/SIFT descriptors."""

    def __init__(
        self,
        config: dict[str, Any],
        descriptor: str = "orb",
        camera_cfg: dict[str, Any] | None = None,
    ):
        self.config = config
        self.camera_cfg = camera_cfg or {}
        self._speckle_pattern: np.ndarray | None = None
        self._speckle_intensity = float(self.camera_cfg.get("speckle_intensity", 0.35))
        self.tile_size = int(config.get("tile_size", 512))
        edge_cfg = config.get("edge_mode", False)
        self.edge_mode = edge_cfg is True or edge_cfg == "true"
        self.descriptor_name = descriptor.lower()
        self.tiles: dict[str, MapTile] = {}
        self._grid: dict[tuple[int, int], list[str]] = {}
        self.seasons: set[str] = set()

        nfeat = int(config.get("nfeatures", 500))
        if self.descriptor_name == "sift":
            self._detector = cv2.SIFT_create(nfeatures=nfeat)
        else:
            self._detector = cv2.ORB_create(nfeatures=nfeat)

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        if not self.edge_mode:
            return gray
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if bool(self.config.get("edge_canny", True)):
            edges = cv2.Canny(gray, 50, 150)
            mag = cv2.max(mag, edges)
        return mag

    def _flatten_textureless_base(self, gray: np.ndarray) -> np.ndarray:
        g = cv2.GaussianBlur(gray, (31, 31), 0)
        return (g // 48) * 48

    def _darken_low_light_base(self, gray: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(gray, alpha=0.4, beta=8)

    def _apply_speckle_tile(self, gray: np.ndarray, row_off: int, col_off: int) -> np.ndarray:
        if self.camera_cfg.get("textureless_map_flatten", False):
            gray = self._flatten_textureless_base(gray)
        if self._speckle_pattern is None:
            out = gray
        else:
            out = apply_nir_speckle(
                gray,
                self._speckle_pattern,
                row_offset=row_off,
                col_offset=col_off,
                intensity=self._speckle_intensity,
            )
        if self.camera_cfg.get("low_light_map_darken", False):
            out = self._darken_low_light_base(out)
        return out

    def _compute_descriptors(
        self,
        gray: np.ndarray,
        row_off: int = 0,
        col_off: int = 0,
    ) -> tuple[list, np.ndarray | None]:
        gray = self._apply_speckle_tile(gray, row_off, col_off)
        processed = self._preprocess(gray)
        kp, desc = self._detector.detectAndCompute(processed, None)
        return kp or [], desc

    @staticmethod
    def infer_season(path: Path) -> str:
        stem = path.stem.lower()
        for tag in ("winter", "snow", "summer", "autumn", "fall", "clear", "spring"):
            if tag in stem:
                return "snow" if tag in ("winter", "snow") else tag
        return "default"

    def load_geotiff(self, path: Path, season: str | None = None) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Map not found: {path}")
        season = season or self.infer_season(path)
        self.seasons.add(season)
        tiles_before = len(self.tiles)

        with rasterio.open(path) as src:
            width, height = src.width, src.height
            if self._speckle_pattern is None:
                self._speckle_pattern = speckle_from_config(self.camera_cfg, height, width)
            for row_off in range(0, height, self.tile_size):
                for col_off in range(0, width, self.tile_size):
                    w = min(self.tile_size, width - col_off)
                    h = min(self.tile_size, height - row_off)
                    window = Window(col_off, row_off, w, h)
                    data = src.read(1, window=window)
                    transform = rasterio.windows.transform(window, src.transform)

                    key = f"{path.stem}_{row_off}_{col_off}"
                    kp, desc = self._compute_descriptors(
                        data.astype(np.uint8), row_off=row_off, col_off=col_off
                    )

                    center_row = row_off + h // 2
                    center_col = col_off + w // 2
                    center_lon, center_lat = rasterio.transform.xy(
                        src.transform, center_row, center_col
                    )

                    tile = MapTile(
                        key=key,
                        image=data.astype(np.uint8),
                        transform=transform,
                        crs=src.crs,
                        row_off=row_off,
                        col_off=col_off,
                        kp=kp,
                        desc=desc,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        season=season,
                        source_stem=path.stem,
                    )
                    self.tiles[key] = tile
                    grid_key = (row_off // self.tile_size, col_off // self.tile_size)
                    self._grid.setdefault(grid_key, []).append(key)

        logger.info(
            "Loaded %d tiles from %s (season=%s)",
            len(self.tiles) - tiles_before,
            path.name,
            season,
        )

    def load_directory(self, directory: Path, patterns: tuple[str, ...] | None = None) -> int:
        directory = Path(directory)
        count = 0
        globs = patterns or ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
        seen: set[Path] = set()
        for pattern in globs:
            for path in sorted(directory.glob(pattern)):
                if path in seen:
                    continue
                seen.add(path)
                before = len(self.tiles)
                self.load_geotiff(path)
                if len(self.tiles) > before:
                    count += 1
        return count

    def get_all_tiles(self) -> list[MapTile]:
        return list(self.tiles.values())

    def get_candidate_tiles(
        self,
        lat: float | None = None,
        lon: float | None = None,
        max_tiles: int | None = None,
    ) -> list[MapTile]:
        if not self.tiles:
            return []
        if max_tiles is None:
            base = int(self.config.get("max_candidate_tiles", 2))
            max_tiles = base * max(1, len(self.seasons))

        def dist(t: MapTile) -> float:
            if lat is None or lon is None:
                return 0.0
            return (t.center_lat - lat) ** 2 + (t.center_lon - lon) ** 2

        ranked = sorted(self.tiles.values(), key=dist)
        if lat is None or lon is None:
            return ranked[: max(1, len(self.seasons))]
        return ranked[:max_tiles]

    def pixel_to_wgs84(self, tile: MapTile, px: float, py: float) -> tuple[float, float]:
        lon, lat = rasterio.transform.xy(tile.transform, py, px)
        return lat, lon

    def wgs84_to_local_m(
        self,
        tile: MapTile,
        lat: float,
        lon: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        az12, _, dist = geod.inv(origin_lon, origin_lat, lon, lat)
        x = dist * np.sin(np.radians(az12))
        y = dist * np.cos(np.radians(az12))
        return x, y


def build_map_index(
    config_module: Any,
    map_path: Path | None = None,
    season_maps: list[str] | None = None,
) -> MapIndex:
    gm_cfg = config_module.geo_match
    maps_cfg = config_module.maps
    index = MapIndex(
        gm_cfg,
        descriptor=gm_cfg.get("descriptor", "orb"),
        camera_cfg=config_module.camera,
    )

    if map_path is not None:
        index.load_geotiff(Path(map_path))
        return index

    maps_dir = config_module.resolve(maps_cfg.get("directory", "data/maps"))
    explicit = season_maps or maps_cfg.get("season_maps")
    if explicit:
        for name in explicit:
            p = Path(name)
            target = p if p.is_absolute() else maps_dir / name
            if target.exists():
                index.load_geotiff(target)
    elif maps_cfg.get("load_all_tifs", True):
        index.load_directory(maps_dir)
    else:
        default_name = maps_cfg.get("default_map", "test_tile.tif")
        candidate = maps_dir / default_name
        if candidate.exists():
            index.load_geotiff(candidate)
        else:
            index.load_directory(maps_dir)

    if not index.tiles:
        index.load_directory(maps_dir)

    return index
