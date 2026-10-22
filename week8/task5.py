import os
import time
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np

STUDENT_ID = 230103191
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
A = 2.5
B = 1.0
REPEATS = 5


def get_cpu_model():
    if platform.system() == "Windows":
        try:
            out = subprocess.check_output(["wmic", "cpu", "get", "name"],
                                          text=True, stderr=subprocess.DEVNULL)
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) >= 2:
                return lines[1]
        except Exception:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def fma_chunk(args):
    x, a, b = args
    return a * x + b


def serial_fma(x):
    return A * x + B


def parallel_fma(x, workers):
    n = x.size
    size = (n + workers - 1) // workers
    chunks = [x[i * size: (i + 1) * size] for i in range(workers)]
    chunks = [c for c in chunks if c.size > 0]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(fma_chunk, [(c, A, B) for c in chunks]))
    return np.concatenate(parts)


def time_serial(x, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = serial_fma(x)
        best = min(best, time.perf_counter() - t0)
    return best


def time_parallel(x, workers, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = parallel_fma(x, workers)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    cores = os.cpu_count()
    print("=" * 74)
    print("TASK 5 — The Overhead Crossover Point")
    print("=" * 74)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", cores)
    print("Operation     : y[i] = a*x[i] + b   (a=%.2f, b=%.2f)" % (A, B))
    print("Repeats       :", REPEATS, "(min taken)")
    print("=" * 74)

    rng = np.random.default_rng(STUDENT_ID)
    warm = rng.standard_normal(10_000)
    serial_fma(warm)
    parallel_fma(warm, cores)

    rows = []
    print()
    print(f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>14} | {'Speedup':>9} | {'Winner':>8}")
    print("-" * 74)
    for n in SIZES:
        x = rng.standard_normal(n).astype(np.float64)
        ts = time_serial(x, REPEATS) * 1000.0
        tp = time_parallel(x, cores, REPEATS) * 1000.0
        speedup = ts / tp if tp > 0 else 0.0
        winner = "serial" if ts < tp else "parallel"
        rows.append((n, ts, tp, speedup, winner))
        print(f"{n:>12,} | {ts:>12.3f} | {tp:>14.3f} | {speedup:>8.2f}x | {winner:>8}")
    print("-" * 74)

    crossover = None
    for i in range(1, len(rows)):
        prev_winner = rows[i - 1][4]
        cur_winner = rows[i][4]
        if prev_winner == "serial" and cur_winner == "parallel":
            crossover = (rows[i - 1][0], rows[i][0])
            break

    print()
    if crossover:
        lo, hi = crossover
        print(f"Crossover detected: parallel becomes faster between "
              f"N = {lo:,} and N = {hi:,}")
        print(f"Exact tipping point lies in the range [{lo:,}, {hi:,}] on this CPU.")
    elif rows[0][4] == "parallel":
        print("Parallel is already faster at the smallest size — overhead is negligible.")
    else:
        print("Parallel never caught up to serial in this sweep — overhead dominates at all tested sizes.")
    print("=" * 74)


if __name__ == "__main__":
    main()