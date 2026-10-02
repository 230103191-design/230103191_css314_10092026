import time
import numpy as np
from numba import njit, prange
import numba

@njit(parallel=True)
def heat_step(u, u_next, alpha=0.20):
    rows, cols = u.shape
    for i in prange(1, rows - 1):
        for j in range(1, cols - 1):
            u_next[i, j] = u[i, j] + alpha * (
                u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - 4.0 * u[i, j]
            )

GRID_SIZE = 1500
STEPS = 300

numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

# float64
u = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
u_next = np.zeros_like(u)
u[0, :] = 100.0
u[:, 0] = 100.0
u_next[0, :] = 100.0
u_next[:, 0] = 100.0

heat_step(u, u_next)  # warmup

start = time.perf_counter()
for step in range(STEPS):
    heat_step(u, u_next)
    u, u_next = u_next, u
elapsed = time.perf_counter() - start

cells_per_sec = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed / 1e6
print(f"Float64 Heat Diffusion: {elapsed:.3f} s, Throughput: {cells_per_sec:.2f} Megacells/sec")

# float32
u32 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
u_next32 = np.zeros_like(u32)
u32[0, :] = 100.0
u32[:, 0] = 100.0
u_next32[0, :] = 100.0
u_next32[:, 0] = 100.0

heat_step(u32, u_next32)  # warmup

start = time.perf_counter()
for step in range(STEPS):
    heat_step(u32, u_next32)
    u32, u_next32 = u_next32, u32
elapsed32 = time.perf_counter() - start

cells_per_sec32 = (GRID_SIZE * GRID_SIZE * STEPS) / elapsed32 / 1e6
print(f"Float32 Heat Diffusion: {elapsed32:.3f} s, Throughput: {cells_per_sec32:.2f} Megacells/sec")
print(f"Runtime reduction factor: {elapsed / elapsed32:.2f}x")