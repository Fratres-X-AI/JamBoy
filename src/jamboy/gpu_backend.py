"""CUDA via CuPy (RTX 5090 / sm_120) with torch fallback on older GPUs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_cupy = None
_torch = None
_backend: str | None = None
_popcount_lut = None


def _load_cupy():
    global _cupy
    if _cupy is None:
        import cupy as cp

        _cupy = cp
    return _cupy


def _load_torch():
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _torch_cuda_works() -> bool:
    try:
        t = _load_torch()
        if not t.cuda.is_available():
            return False
        x = t.zeros(1, device="cuda")
        t.cuda.synchronize()
        del x
        return True
    except Exception:
        return False


def detect_gpu_backend() -> str:
    """Prefer CuPy (Blackwell sm_120); fall back to torch; else cpu."""
    global _backend
    if _backend:
        return _backend
    try:
        cp = _load_cupy()
        cp.cuda.runtime.getDeviceCount()
        _backend = "cupy"
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        logger.info("GPU backend: CuPy on %s", name)
        return _backend
    except Exception as exc:
        logger.debug("CuPy unavailable: %s", exc)
    if _torch_cuda_works():
        _backend = "torch"
        logger.info("GPU backend: PyTorch CUDA")
        return _backend
    _backend = "cpu"
    return _backend


def cuda_available() -> bool:
    return detect_gpu_backend() != "cpu"


def resolve_device(config, cli_gpu: bool = False, cli_cpu: bool = False) -> str:
    if cli_cpu:
        return "cpu"
    want = "cuda" if cli_gpu else config.raw.get("compute", {}).get("device", "cpu")
    if want == "cuda" and cuda_available():
        return "cuda"
    if want == "cuda":
        logger.warning("CUDA requested but unavailable — CPU fallback")
    return "cpu"


class CudaTimer:
    """GPU/CPU wall timer."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.gpu_ms = 0.0
        self.cpu_ms = 0.0
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        if self.device == "cuda":
            backend = detect_gpu_backend()
            if backend == "cupy":
                cp = _load_cupy()
                self._start = cp.cuda.Event()
                self._end = cp.cuda.Event()
                self._start.record()
            elif backend == "torch":
                t = _load_torch()
                self._start = t.cuda.Event(enable_timing=True)
                self._end = t.cuda.Event(enable_timing=True)
                self._start.record()
        return self

    def __exit__(self, *args):
        if self.device == "cuda":
            backend = detect_gpu_backend()
            if backend == "cupy":
                cp = _load_cupy()
                self._end.record()
                self._end.synchronize()
                self.gpu_ms = cp.cuda.get_elapsed_time(self._start, self._end)
            elif backend == "torch":
                t = _load_torch()
                self._end.record()
                t.cuda.synchronize()
                self.gpu_ms = float(self._start.elapsed_time(self._end))
        else:
            self.gpu_ms = 0.0
        self.cpu_ms = (time.perf_counter() - self._t0) * 1000.0

    def elapsed_ms(self) -> float:
        return self.gpu_ms if self.device == "cuda" and self.gpu_ms > 0 else self.cpu_ms


class GpuOrbMatcher:
    """GPU ORB Hamming matcher — CuPy xor + popcount LUT."""

    def __init__(self, device: str = "cuda"):
        self.device_str = device
        self._tile_cache: dict[str, tuple] = {}
        self._last_gpu_ms = 0.0
        self._lut = None

    def _get_lut(self, cp):
        global _popcount_lut
        if _popcount_lut is None:
            lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
            _popcount_lut = cp.asarray(lut)
        return _popcount_lut

    def cache_tile(self, key: str, desc: np.ndarray, kp: list) -> None:
        if self.device_str != "cuda" or desc is None or len(desc) == 0:
            return
        cp = _load_cupy()
        self._tile_cache[key] = (cp.asarray(desc), kp)

    def knn_match_lowe(
        self,
        query_desc: np.ndarray,
        train_key: str,
        train_desc: np.ndarray | None,
        train_kp: list,
        lowe_ratio: float = 0.7,
        k: int = 2,
    ) -> list:
        import cv2

        if query_desc is None or train_desc is None:
            return []

        t0 = time.perf_counter()
        dist = _hamming_distance_matrix(query_desc, train_desc)
        self._last_gpu_ms = (time.perf_counter() - t0) * 1000.0

        k_eff = min(k, train_desc.shape[0])
        if k_eff < 2:
            return []

        good = []
        for qi in range(query_desc.shape[0]):
            row = dist[qi]
            idx = np.argpartition(row, k_eff - 1)[:k_eff]
            vals = row[idx]
            order = np.argsort(vals)
            d1, d2 = float(vals[order[0]]), float(vals[order[1]])
            if d1 < lowe_ratio * d2:
                m = cv2.DMatch()
                m.queryIdx = qi
                m.trainIdx = int(idx[order[0]])
                m.distance = d1
                good.append(m)
        return good

    @property
    def last_gpu_ms(self) -> float:
        return self._last_gpu_ms


def _desc_to_bits(desc: np.ndarray) -> np.ndarray:
    """ORB [N,32] uint8 -> bit vectors [N,256] float32."""
    return np.unpackbits(desc, axis=1).astype(np.float32)


def _hamming_distance_matrix(query_desc: np.ndarray, train_desc: np.ndarray) -> np.ndarray:
    """ORB Hamming via GPU matmul (RTX 5090 sm_120) or NumPy xor fallback."""
    if detect_gpu_backend() == "cupy":
        try:
            cp = _load_cupy()
            q = cp.asarray(_desc_to_bits(query_desc))
            tr = cp.asarray(_desc_to_bits(train_desc))
            # hamming(bit_a, bit_b) = sum(a) + sum(b) - 2 * dot(a,b)
            qsum = q.sum(axis=1, keepdims=True)
            tsum = tr.sum(axis=1, keepdims=True)
            dist = qsum + tsum.T - 2.0 * (q @ tr.T)
            cp.cuda.Stream.null.synchronize()
            return cp.asnumpy(dist)
        except Exception as exc:
            logger.warning("GPU matmul hamming failed, NumPy fallback: %s", exc)

    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
    xor = query_desc[:, np.newaxis, :] ^ train_desc[np.newaxis, :, :]
    return lut[xor].sum(axis=2)


def _cpu_knn_match(query_desc, train_desc, lowe_ratio, k):
    import cv2

    if query_desc is None or train_desc is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        pairs = matcher.knnMatch(query_desc, train_desc, k=k)
    except cv2.error:
        return []
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    return good


def preprocess_gray_gpu(gray: np.ndarray, edge_mode: bool, device: str) -> np.ndarray:
    if device != "cuda" or not edge_mode:
        return gray
    cp = _load_cupy()
    t = cp.asarray(gray, dtype=cp.float32)
    gx = t[:, 1:] - t[:, :-1]
    gy = t[1:, :] - t[:-1, :]
    edge = cp.clip(cp.abs(gx[:-1, :]) + cp.abs(gy[:, :-1]), 0, 255)
    return cp.asnumpy(edge.astype(cp.uint8))


def batch_hamming_gpu(query_batch: list[np.ndarray], train_desc: np.ndarray, device: str = "cuda") -> list:
    if device != "cuda":
        return []
    cp = _load_cupy()
    tr = cp.asarray(train_desc)
    lut = _get_lut_cupy(cp)
    results = []
    for qd in query_batch:
        if qd is None or len(qd) == 0:
            results.append(np.array([]))
            continue
        q = cp.asarray(qd)
        xor = q[:, cp.newaxis, :] ^ tr[cp.newaxis, :, :]
        results.append(cp.asnumpy(lut[xor].sum(axis=2)))
    cp.cuda.Stream.null.synchronize()
    return results


def _get_lut_cupy(cp):
    global _popcount_lut
    if _popcount_lut is None:
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
        _popcount_lut = cp.asarray(lut)
    return _popcount_lut


def gpu_saturate(duration_sec: float = 10.0) -> dict:
    """Sustained GPU load for utilization testing."""
    cp = _load_cupy()
    t0 = time.perf_counter()
    ops = 0
    a = cp.random.standard_normal((8192, 8192), dtype=cp.float32)
    b = cp.random.standard_normal((8192, 8192), dtype=cp.float32)
    while time.perf_counter() - t0 < duration_sec:
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        ops += 1
    return {"matmul_ops": ops, "wall_sec": duration_sec, "backend": "cupy"}


def gpu_device_info() -> dict:
    if not cuda_available():
        return {"available": False}
    backend = detect_gpu_backend()
    if backend == "cupy":
        cp = _load_cupy()
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {
            "available": True,
            "backend": "cupy",
            "name": props["name"].decode(),
            "vram_gb": round(props["totalGlobalMem"] / 1e9, 1),
        }
    t = _load_torch()
    props = t.cuda.get_device_properties(0)
    return {
        "available": True,
        "backend": "torch",
        "name": t.cuda.get_device_name(0),
        "vram_gb": round(props.total_memory / 1e9, 1),
    }
