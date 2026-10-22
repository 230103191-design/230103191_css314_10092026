import os
import time
import platform
import subprocess
import threading

STUDENT_ID = 230103191
N_THREADS = 4
INCREMENTS = 2_000_000
EXPECTED = N_THREADS * INCREMENTS


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


def run_unsynchronized(n_threads, iters):
    counter = [0]

    def worker():
        for _ in range(iters):
            counter[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, counter[0]


def run_locked(n_threads, iters):
    counter = [0]
    lock = threading.Lock()

    def worker():
        for _ in range(iters):
            with lock:
                counter[0] += 1

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, counter[0]


def run_correct_counter(n_threads, iters):
    """Correct without lock: accumulate locally, merge at end."""
    results = [0] * n_threads

    def worker(idx):
        local = 0
        for _ in range(iters):
            local += 1
        results[idx] = local

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, sum(results)


def main():
    print("=" * 70)
    print("TASK 6 — Synchronization Cost & Data Loss")
    print("=" * 70)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", os.cpu_count())
    print("Threads       :", N_THREADS)
    print("Increments    :", f"{INCREMENTS:,} per thread")
    print("Expected total:", f"{EXPECTED:,}")
    print("=" * 70)

    print("\n[1] Unsynchronized shared counter ...")
    t_unsync, actual_unsync = run_unsynchronized(N_THREADS, INCREMENTS)
    lost = EXPECTED - actual_unsync
    loss_pct = 100.0 * lost / EXPECTED
    print(f"    Wall time     : {t_unsync:.3f} s")
    print(f"    Actual value  : {actual_unsync:,}")
    print(f"    Lost updates  : {lost:,} ({loss_pct:.2f}%)")

    print("\n[2] Lock-protected shared counter ...")
    t_locked, actual_locked = run_locked(N_THREADS, INCREMENTS)
    correct_locked = (actual_locked == EXPECTED)
    print(f"    Wall time     : {t_locked:.3f} s")
    print(f"    Actual value  : {actual_locked:,}")
    print(f"    Correct       : {correct_locked}")

    print("\n[3] Lock-free correct counter (local accumulators) ...")
    t_local, actual_local = run_correct_counter(N_THREADS, INCREMENTS)
    correct_local = (actual_local == EXPECTED)
    print(f"    Wall time     : {t_local:.3f} s")
    print(f"    Actual value  : {actual_local:,}")
    print(f"    Correct       : {correct_local}")

    print("\n" + "-" * 70)
    print(f"{'Version':<28} | {'Time':>10} | {'Final':>12} | {'Correct':>7}")
    print("-" * 70)
    print(f"{'Unsynchronized':<28} | {t_unsync:>8.3f} s | {actual_unsync:>12,} | {'no':>7}")
    print(f"{'Locked':<28} | {t_locked:>8.3f} s | {actual_locked:>12,} | {str(correct_locked):>7}")
    print(f"{'Lock-free (local)':<28} | {t_local:>8.3f} s | {actual_local:>12,} | {str(correct_local):>7}")
    print("-" * 70)

    if t_unsync > 0:
        print(f"Locking time penalty vs. unsynchronized : {t_locked / t_unsync:.2f}x")
    if t_local > 0:
        print(f"Local-accumulator penalty vs. unsafe    : {t_local / t_unsync:.2f}x")
    print(f"Data loss without lock                  : {loss_pct:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()