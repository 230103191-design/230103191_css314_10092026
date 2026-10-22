import os
import time
import platform
import subprocess

import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103191
N = 30_000_000
A_CONST = 1.0
B_CONST = 0.5
REPEATS = 3


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


def kernel_copy(a, b, start, end):
    b[start:end] = a[start:end]


def kernel_compute(a, b, start, end):
    x = a[start:end]
    b[start:end] = np.sin(x) * np.cos(x) + np.tan(x)


def run_serial(a, b, kernel):
    kernel(a, b, 0, a.size)


def run_parallel(a, b, kernel, workers):
    n = a.size
    size = (n + workers - 1) // workers
    ranges = [(i * size, min((i + 1) * size, n)) for i in range(workers)]
    ranges = [(s, e) for s, e in ranges if s < e]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(kernel, a, b, s, e) for s, e in ranges]
        for f in futures:
            f.result()


def best_time(a, b, kernel, workers, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        if workers == 1:
            run_serial(a, b, kernel)
        else:
            run_parallel(a, b, kernel, workers)
        best = min(best, time.perf_counter() - t0)
    return best


def benchmark_kernel(name, a, b, kernel, worker_counts):
    print(f"\n[Kernel: {name}]")
    print(f"{'Cores':<7} | {'Time':>10} | {'Speedup':>8} | {'Efficiency':>11} | {'Bandwidth':>12}")
    print("-" * 62)
    times = {}
    for w in worker_counts:
        t = best_time(a, b, kernel, w, REPEATS)
        times[w] = t
        speedup = times[1] / t if 1 in times else 1.0
        eff = speedup / w
        if "copy" in name.lower():
            bytes_moved = N * 8 * 2
        else:
            bytes_moved = N * 8 * 3
        bw = bytes_moved / t / 1e9
        print(f"{w:<7} | {t:>8.3f} s | {speedup:>7.2f}x | {eff:>10.2f} | {bw:>9.1f} GB/s")
    print("-" * 62)
    return times


def main():
    cores = os.cpu_count()
    worker_counts = sorted(set([1, 2, cores]))

    print("=" * 74)
    print("TASK 7 — Arithmetic Intensity vs. Memory Bottlenecks")
    print("=" * 74)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", cores)
    print("Array size    :", f"{N:,} float64 elements ({N * 8 / 1e9:.2f} GB)")
    print("Repeats       :", REPEATS, "(min taken)")
    print("=" * 74)

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.standard_normal(N).astype(np.float64)
    b = np.empty_like(a)

    run_serial(a, b, kernel_copy)
    run_parallel(a, b, kernel_compute, cores)

    t_copy = benchmark_kernel("A: Memory-Bound (B = A)", a, b, kernel_copy, worker_counts)
    t_comp = benchmark_kernel("B: Compute-Bound (sin*cos + tan)", a, b, kernel_compute, worker_counts)

    print("\n" + "=" * 74)
    print("SCALING ANALYSIS")
    print("=" * 74)

    for label, times in [("Memory-Bound (copy)", t_copy), ("Compute-Bound (sin*cos+tan)", t_comp)]:
        w_max = max(times)
        speedup_max = times[1] / times[w_max]
        eff_max = speedup_max / w_max
        print(f"{label:<28} : max speedup {speedup_max:.2f}x on {w_max} cores -> efficiency {eff_max:.2f}")

    copy_max_eff = (t_copy[1] / t_copy[max(t_copy)]) / max(t_copy)
    comp_max_eff = (t_comp[1] / t_comp[max(t_comp)]) / max(t_comp)

    print("\nConclusion:")
    if copy_max_eff < comp_max_eff:
        print("  Kernel A (memory copy) scaled POORLY because it is bandwidth-bound:")
        print("  as soon as 2+ cores run, they compete for the same DRAM bus/channels")
        print("  and saturate the memory controller. Adding more cores cannot help")
        print(f"  because RAM bandwidth is already the bottleneck (efficiency {copy_max_eff:.2f}).")
        print("  Kernel B (sin*cos+tan) scaled BETTER because the transcendental")
        print("  math dominates runtime, so cores spend most of their time in ALU,")
        print(f"  not waiting on memory (efficiency {comp_max_eff:.2f}).")
    else:
        print("  Result reversed from typical hardware — inspect ratios above.")
    print("=" * 74)


if __name__ == "__main__":
    main()