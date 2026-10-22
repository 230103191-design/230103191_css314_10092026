import os
import time
import platform
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor

STUDENT_ID = 230103191
N_ELEMENTS = 20_000_000
P = 0.95


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


def sum_chunk(chunk):
    total = 0.0
    for x in chunk:
        total += x
    return total


def parallel_sum(data, workers):
    n = data.size
    size = (n + workers - 1) // workers
    chunks = [data[i * size: min((i + 1) * size, n)] for i in range(workers)]
    chunks = [c for c in chunks if c.size > 0]

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(sum_chunk, chunks))
    return sum(results)


def amdahl(p, n):
    return 1.0 / ((1 - p) + p / n)


def main():
    print("=" * 70)
    print("TASK 2 — Multi-Core Scaling vs. Amdahl's Law")
    print("=" * 70)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    cores = os.cpu_count()
    print("Logical cores :", cores)
    print("=" * 70)

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.standard_normal(N_ELEMENTS, dtype=np.float64)

    worker_counts = sorted(set([1, 2, 4, cores, cores * 2]))

    parallel_sum(data[:100_000], 1)  # warm-up

    times = {}
    for w in worker_counts:
        t0 = time.perf_counter()
        _ = parallel_sum(data, w)
        times[w] = time.perf_counter() - t0

    t1 = times[1]

    print()
    print(f"{'Cores':<7} | {'Wall Time':>11} | {'Speedup':>8} | {'Amdahl Bound':>13}")
    print("-" * 50)
    for w in worker_counts:
        s = t1 / times[w]
        a = amdahl(P, w)
        print(f"{w:<7} | {times[w]:>9.3f} s | {s:>7.2f}x | {a:>12.2f}x")
    print("-" * 50)
    print(f"(P = {P:.2f}, base T(1) = {t1:.3f} s)")


if __name__ == "__main__":
    main()