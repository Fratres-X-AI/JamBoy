#!/usr/bin/env python3
import cupy as cp

print("cupy device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
a = cp.random.randn(4096, 4096, dtype=cp.float32)
b = a @ a
cp.cuda.Stream.null.synchronize()
print("matmul_ok", b.shape)
