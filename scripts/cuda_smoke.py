#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jamboy.gpu_backend import cuda_available, detect_gpu_backend, gpu_device_info, gpu_saturate

print("backend:", detect_gpu_backend(), "available:", cuda_available())
print("gpu_info:", gpu_device_info())
if cuda_available():
    print("saturate:", gpu_saturate(3.0))
