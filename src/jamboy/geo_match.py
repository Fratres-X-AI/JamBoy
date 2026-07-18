"""Absolute localization via feature matching against offline map tiles.

Requires global shutter sensor (e.g. IMX296 class) to eliminate rolling shutter
distortion that breaks homography estimation under vibration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from jamboy.gpu_backend import CudaTimer, GpuOrbMatcher, preprocess_gray_gpu
from jamboy.map_loader import MapIndex, MapTile

logger = logging.getLogger(__name__)

CONDITION_SEASON_HINT = {
    "snow": "snow",
    "winter": "snow",
    "clear": "default",
    "smoke": "default",
    "low_light": "default",
    "textureless": "default",
}


@dataclass
class GeoFix:
    lat: float
    lon: float
    confidence: float
    x_local: float = 0.0
    y_local: float = 0.0
    latency_ms: float = 0.0
    gpu_ms: float = 0.0
    map_season: str = "default"
    match_mode: str = "intensity"


class GeoMatcher:
    def __init__(
        self,
        config,
        map_index: MapIndex,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        device: str = "cpu",
    ):
        self.config = config.geo_match
        self.map_index = map_index
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.device = device

        self.descriptor_name = self.config.get("descriptor", "orb").lower()
        nfeat = int(self.config.get("nfeatures", 500))
        if self.descriptor_name == "sift":
            self.detector = cv2.SIFT_create(nfeatures=nfeat)
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
            self.gpu_matcher = None
        else:
            self.detector = cv2.ORB_create(nfeatures=nfeat)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.gpu_matcher = GpuOrbMatcher(device) if device == "cuda" else None
            if self.gpu_matcher:
                for tile in map_index.get_all_tiles():
                    if tile.desc is not None:
                        self.gpu_matcher.cache_tile(tile.key, tile.desc, tile.kp or [])

        self.lowe_ratio = float(self.config.get("lowe_ratio", 0.7))
        self.min_good_matches = int(self.config.get("min_good_matches", 15))
        self.partial_match_ratio = float(self.config.get("partial_match_ratio", 0.5))
        self.reference_altitude_m = float(self.config.get("reference_altitude_m", 100.0))
        self.altitude_scale_tolerance = float(self.config.get("altitude_scale_tolerance", 0.5))
        self.ransac_threshold = float(self.config.get("ransac_threshold", 5.0))
        edge_cfg = self.config.get("edge_mode", False)
        self.edge_only = edge_cfg is True or edge_cfg == "true"
        self.edge_hybrid = edge_cfg in ("hybrid", "both")
        self.edge_canny = bool(self.config.get("edge_canny", True))
        self.max_scale_error = float(self.config.get("max_scale_error", 0.35))
        self.min_confidence = float(self.config.get("min_confidence", 0.15))
        self.min_inlier_ratio = float(self.config.get("min_inlier_ratio", 0.25))
        self.template_fallback = bool(self.config.get("template_fallback", True))
        self.template_min_score = float(self.config.get("template_min_score", 0.42))
        self._last_tile_key: str | None = None
        self.max_candidate_tiles = int(self.config.get("max_candidate_tiles", 2))
        self._edge_tile_cache: dict[str, tuple[list, np.ndarray]] = {}

    def _to_edges(self, gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if self.edge_canny:
            edges = cv2.Canny(gray, 50, 150)
            mag = cv2.max(mag, edges)
        if self.device == "cuda":
            gpu_edge = preprocess_gray_gpu(mag, True, self.device)
            if gpu_edge.shape == mag.shape:
                return gpu_edge
        return mag

    def _preprocess(self, gray: np.ndarray) -> np.ndarray:
        if self.edge_only:
            return self._to_edges(gray)
        return gray

    def _tile_edge_features(self, tile: MapTile) -> tuple[list, np.ndarray | None]:
        cached = self._edge_tile_cache.get(tile.key)
        if cached is not None:
            return cached
        edge_img = self._to_edges(tile.image)
        kp, desc = self.detector.detectAndCompute(edge_img, None)
        if desc is None:
            self._edge_tile_cache[tile.key] = ([], None)
            return [], None
        self._edge_tile_cache[tile.key] = (kp, desc)
        return kp, desc

    def _match_descriptors(
        self,
        desc_live,
        tile: MapTile,
        tile_kp: list | None = None,
        tile_desc: np.ndarray | None = None,
    ) -> list:
        kp = tile_kp if tile_kp is not None else tile.kp
        desc = tile_desc if tile_desc is not None else tile.desc
        if desc is None:
            return []
        if self.gpu_matcher and self.descriptor_name == "orb" and tile_kp is None and tile_desc is None:
            return self.gpu_matcher.knn_match_lowe(
                desc_live,
                tile.key,
                tile.desc,
                tile.kp or [],
                lowe_ratio=self.lowe_ratio,
                k=2,
            )
        try:
            matches = self.matcher.knnMatch(desc_live, desc, k=2)
        except cv2.error:
            return []
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        return good

    def _score_confidence(
        self,
        inlier_ratio: float,
        inliers: int,
        good_count: int,
        is_partial: bool,
    ) -> float:
        match_quality = min(1.0, good_count / max(self.min_good_matches, 1))
        inlier_quality = min(1.0, inliers / 40.0)
        conf = float(np.clip(inlier_ratio * inlier_quality * match_quality, 0.0, 1.0))
        if is_partial:
            conf *= good_count / max(self.min_good_matches, 1)
        return conf

    def _altitude_scale_ok(self, M: np.ndarray, altitude_m: float | None) -> bool:
        if altitude_m is None or altitude_m <= 0 or self.reference_altitude_m <= 0:
            return True
        scale = np.sqrt(abs(np.linalg.det(M[:2, :2])))
        expected_scale = self.reference_altitude_m / altitude_m
        abs_tol = max(self.max_scale_error * expected_scale, 0.08)
        rel_tol = self.altitude_scale_tolerance
        abs_err = abs(scale - expected_scale)
        rel_err = abs_err / max(expected_scale, 1e-3)
        return abs_err <= abs_tol or rel_err <= rel_tol

    def _homography_fix(
        self,
        live_gray: np.ndarray,
        tile: MapTile,
        kp_live: list,
        good: list,
        altitude_m: float | None,
        match_mode: str,
        gpu_ms: float = 0.0,
    ) -> GeoFix | None:
        partial_min = max(4, int(self.min_good_matches * self.partial_match_ratio))
        if len(good) < partial_min:
            return None
        is_partial = len(good) < self.min_good_matches

        src_pts = np.float32([kp_live[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        tile_kp = tile.kp if match_mode == "intensity" else self._tile_edge_features(tile)[0]
        dst_pts = np.float32([tile_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        if M is None or mask is None:
            return None

        inliers = int(mask.ravel().sum())
        inlier_ratio = inliers / max(len(good), 1)
        if inlier_ratio < self.min_inlier_ratio:
            return None
        if not self._altitude_scale_ok(M, altitude_m):
            logger.debug(
                "Rejected homography scale at alt=%.1f m (mode=%s, season=%s)",
                altitude_m or -1.0,
                match_mode,
                tile.season,
            )
            return None

        h, w = live_gray.shape
        center = np.float32([[[w / 2, h / 2]]])
        map_pixel = cv2.perspectiveTransform(center, M)
        px, py = map_pixel[0][0]

        lat, lon = self.map_index.pixel_to_wgs84(tile, float(px), float(py))
        x_local, y_local = self.map_index.wgs84_to_local_m(
            tile, lat, lon, self.origin_lat, self.origin_lon
        )
        confidence = self._score_confidence(inlier_ratio, inliers, len(good), is_partial)
        if confidence < self.min_confidence:
            return None

        return GeoFix(
            lat=lat,
            lon=lon,
            confidence=confidence,
            x_local=x_local,
            y_local=y_local,
            gpu_ms=gpu_ms,
            map_season=tile.season,
            match_mode=match_mode,
        )

    def _feature_match_tile(
        self,
        processed: np.ndarray,
        live_gray: np.ndarray,
        tile: MapTile,
        altitude_m: float | None,
        match_mode: str,
        tile_kp: list | None = None,
        tile_desc: np.ndarray | None = None,
    ) -> GeoFix | None:
        kp_live, desc_live = self.detector.detectAndCompute(processed, None)
        desc = tile_desc if tile_desc is not None else tile.desc
        if desc_live is None or desc is None or len(kp_live) < 4:
            return None

        good = self._match_descriptors(desc_live, tile, tile_kp=tile_kp, tile_desc=desc)
        gpu_ms = self.gpu_matcher.last_gpu_ms if self.gpu_matcher else 0.0
        return self._homography_fix(
            live_gray, tile, kp_live, good, altitude_m, match_mode, gpu_ms=gpu_ms
        )

    def _match_tile(
        self,
        live_gray: np.ndarray,
        tile: MapTile,
        altitude_m: float | None = None,
    ) -> GeoFix | None:
        candidates: list[GeoFix | None] = []

        if not self.edge_only:
            candidates.append(
                self._feature_match_tile(
                    self._preprocess(live_gray),
                    live_gray,
                    tile,
                    altitude_m,
                    "intensity",
                )
            )

        if self.edge_only or self.edge_hybrid:
            edge_kp, edge_desc = self._tile_edge_features(tile)
            edge_live = self._to_edges(live_gray)
            candidates.append(
                self._feature_match_tile(
                    edge_live,
                    live_gray,
                    tile,
                    altitude_m,
                    "edge",
                    tile_kp=edge_kp,
                    tile_desc=edge_desc,
                )
            )

        best = None
        for fix in candidates:
            if fix and (best is None or fix.confidence > best.confidence):
                best = fix

        if best is not None:
            return best
        return self._template_match_tile(live_gray, tile)

    def _template_match_tile(self, live_gray: np.ndarray, tile: MapTile) -> GeoFix | None:
        """Fallback for low-feature frames, especially active-IR textureless scenes."""
        if not self.template_fallback:
            return None
        th, tw = tile.image.shape[:2]
        lh, lw = live_gray.shape[:2]
        if lh > th or lw > tw or lh < 16 or lw < 16:
            return None

        try:
            live = cv2.equalizeHist(live_gray.astype(np.uint8))
            base = cv2.equalizeHist(tile.image.astype(np.uint8))
            edge_live = self._to_edges(live)
            edge_base = self._to_edges(base)
            res_int = cv2.matchTemplate(base, live, cv2.TM_CCOEFF_NORMED)
            res_edge = cv2.matchTemplate(edge_base, edge_live, cv2.TM_CCOEFF_NORMED)
            _, max_int, _, loc_int = cv2.minMaxLoc(res_int)
            _, max_edge, _, loc_edge = cv2.minMaxLoc(res_edge)
            if max_edge >= max_int:
                score, max_loc, mode = float(max_edge), loc_edge, "template_edge"
            else:
                score, max_loc, mode = float(max_int), loc_int, "template"
        except cv2.error as exc:
            logger.debug("Template fallback failed: %s", exc)
            return None

        if score < self.template_min_score:
            return None

        px = float(max_loc[0] + lw / 2.0)
        py = float(max_loc[1] + lh / 2.0)
        lat, lon = self.map_index.pixel_to_wgs84(tile, px, py)
        x_local, y_local = self.map_index.wgs84_to_local_m(
            tile, lat, lon, self.origin_lat, self.origin_lon
        )
        return GeoFix(
            lat=lat,
            lon=lon,
            confidence=float(np.clip(score, 0.0, 1.0)),
            x_local=x_local,
            y_local=y_local,
            map_season=tile.season,
            match_mode=mode,
        )

    def _candidate_tiles(
        self,
        current_lat: float | None,
        current_lon: float | None,
        preferred_season: str | None = None,
    ) -> list[MapTile]:
        season_count = max(1, len(self.map_index.seasons))
        max_tiles = self.max_candidate_tiles * season_count
        candidates = self.map_index.get_candidate_tiles(
            current_lat, current_lon, max_tiles=max_tiles
        )
        if preferred_season:
            candidates.sort(
                key=lambda t: (0 if t.season == preferred_season else 1, t.key)
            )
        if self._last_tile_key:
            last = self.map_index.tiles.get(self._last_tile_key)
            if last and last not in candidates:
                candidates = [last] + candidates
        return candidates

    def find_absolute_fix(
        self,
        live_frame: np.ndarray,
        current_lat: float | None = None,
        current_lon: float | None = None,
        altitude_m: float | None = None,
        baro_altitude_m: float | None = None,
        preferred_season: str | None = None,
    ) -> GeoFix | None:
        alt = baro_altitude_m if baro_altitude_m is not None else altitude_m
        wall_t0 = time.perf_counter()
        if live_frame is None or live_frame.size == 0:
            return None

        if live_frame.ndim == 3:
            live_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        else:
            live_gray = live_frame
        if float(live_gray.std()) < 6.0:
            return None

        timer_device = self.device if self.device == "cuda" else "cpu"
        with CudaTimer(timer_device):
            candidates = self._candidate_tiles(
                current_lat, current_lon, preferred_season=preferred_season
            )

            best: GeoFix | None = None
            best_tile_key: str | None = None
            for tile in candidates:
                fix = self._match_tile(live_gray, tile, altitude_m=alt)
                if fix and (best is None or fix.confidence > best.confidence):
                    best = fix
                    best_tile_key = tile.key

            if best is None:
                for tile in self.map_index.get_all_tiles():
                    if tile.key == best_tile_key:
                        continue
                    fix = self._match_tile(live_gray, tile, altitude_m=alt)
                    if fix and (best is None or fix.confidence > best.confidence):
                        best = fix
                        best_tile_key = tile.key

        if best:
            best.latency_ms = (time.perf_counter() - wall_t0) * 1000.0
            if best_tile_key:
                self._last_tile_key = best_tile_key
            logger.info(
                "Geo fix conf=%.2f season=%s mode=%s (%.1f ms, gpu=%.1f ms)",
                best.confidence,
                best.map_season,
                best.match_mode,
                best.latency_ms,
                best.gpu_ms,
            )
        else:
            logger.warning("Geo localization failed (%.1f ms)", (time.perf_counter() - wall_t0) * 1000.0)

        return best


def build_geo_matcher_for_condition(
    cfg,
    origin_lat: float,
    origin_lon: float,
    condition: str,
    device: str = "cpu",
) -> GeoMatcher:
    """Build a GeoMatcher with map/camera tweaks for degraded sim conditions."""
    import copy

    from jamboy.config import JamBoyConfig
    from jamboy.map_loader import build_map_index

    use_cfg = cfg
    if condition in ("textureless", "low_light") and cfg.camera.get("active_ir_speckle", False):
        raw = copy.deepcopy(cfg.raw)
        cam = raw.setdefault("camera", {})
        if condition == "textureless":
            cam["textureless_map_flatten"] = True
        if condition == "low_light":
            cam["low_light_map_darken"] = True
        use_cfg = JamBoyConfig(raw=raw, project_root=cfg.project_root)

    idx = build_map_index(use_cfg)
    return GeoMatcher(use_cfg, idx, origin_lat, origin_lon, device=device)
