import time
import numpy as np
from numba import njit, prange
import numba

@njit(parallel=True)
def render_mandelbrot_rows(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for r in prange(h):
        cy = -1.2 + (r / h) * 2.4
        for c in range(w):
            cx = -2.0 + (c / w) * 2.5
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

@njit(parallel=True)
def render_mandelbrot_cols(h, w, max_iter):
    img = np.zeros((h, w), dtype=np.int32)
    for c in prange(w):
        cx = -2.0 + (c / w) * 2.5
        for r in range(h):
            cy = -1.2 + (r / h) * 2.4
            z_real, z_imag = 0.0, 0.0
            it = 0
            while (z_real * z_real + z_imag * z_imag <= 4.0) and (it < max_iter):
                next_real = z_real * z_real - z_imag * z_imag + cx
                z_imag = 2.0 * z_real * z_imag + cy
                z_real = next_real
                it += 1
            img[r, c] = it
    return img

H, W = 2500, 2500
MAX_ITER = 1000

# Warmup
render_mandelbrot_rows(100, 100, 100)
render_mandelbrot_cols(100, 100, 100)

numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

t0 = time.perf_counter()
img_rows = render_mandelbrot_rows(H, W, MAX_ITER)
t1 = time.perf_counter()
print(f"Mandelbrot Rows: {t1 - t0:.4f} s")

t0 = time.perf_counter()
img_cols = render_mandelbrot_cols(H, W, MAX_ITER)
t1 = time.perf_counter()
print(f"Mandelbrot Cols: {t1 - t0:.4f} s")