import os
import time
import platform
import subprocess
import threading
import queue

STUDENT_ID = 230103191
N_ITEMS = 1000
N_WORKERS = 4
CHUNK_SIZE = 5


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


def workload(i):
    cost = 0.0001 * (i % 20)
    end = time.perf_counter() + cost
    acc = 0
    while time.perf_counter() < end:
        acc += 1
    return acc


def strategy_static(n_items, n_workers):
    block = (n_items + n_workers - 1) // n_workers
    finish_times = [0.0] * n_workers
    start_barrier = threading.Barrier(n_workers)

    def worker(wid):
        start = wid * block
        end = min((wid + 1) * block, n_items)
        start_barrier.wait()
        for i in range(start, end):
            workload(i)
        finish_times[wid] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    delta = max(finish_times) - min(finish_times)
    return total, delta, finish_times


def strategy_dynamic(n_items, n_workers, chunk_size):
    work_q = queue.Queue()
    for i in range(0, n_items, chunk_size):
        work_q.put(list(range(i, min(i + chunk_size, n_items))))

    finish_times = [0.0] * n_workers
    start_barrier = threading.Barrier(n_workers)

    def worker(wid):
        start_barrier.wait()
        while True:
            try:
                chunk = work_q.get_nowait()
            except queue.Empty:
                break
            for i in chunk:
                workload(i)
            work_q.task_done()
        finish_times[wid] = time.perf_counter()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(n_workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.perf_counter() - t0
    delta = max(finish_times) - min(finish_times)
    return total, delta, finish_times


def main():
    print("=" * 70)
    print("TASK 4 — Work Stealing & Scheduling Under Skew")
    print("=" * 70)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", os.cpu_count())
    print("Workers       :", N_WORKERS)
    print("Items         :", N_ITEMS)
    print("Chunk size    :", CHUNK_SIZE)
    print("=" * 70)

    print("\n[Strategy A] Static equal-block distribution ...")
    t_a, d_a, f_a = strategy_static(N_ITEMS, N_WORKERS)
    print(f"    Wall time         : {t_a:.3f} s")
    print(f"    Finish delta      : {d_a * 1000:.1f} ms")
    print(f"    Per-thread finish : {[round(f, 3) for f in f_a]}")

    print(f"\n[Strategy B] Dynamic queue (chunk = {CHUNK_SIZE}) ...")
    t_b, d_b, f_b = strategy_dynamic(N_ITEMS, N_WORKERS, CHUNK_SIZE)
    print(f"    Wall time         : {t_b:.3f} s")
    print(f"    Finish delta      : {d_b * 1000:.1f} ms")
    print(f"    Per-thread finish : {[round(f, 3) for f in f_b]}")

    print("\n" + "-" * 70)
    print(f"{'Strategy':<28} | {'Wall Time':>10} | {'Finish Delta':>14}")
    print("-" * 70)
    print(f"{'A: Static':<28} | {t_a:>8.3f} s | {d_a * 1000:>11.1f} ms")
    print(f"{'B: Dynamic (chunk=' + str(CHUNK_SIZE) + ')':<28} | {t_b:>8.3f} s | {d_b * 1000:>11.1f} ms")
    print("-" * 70)
    print(f"Dynamic speedup : {t_a / t_b:.2f}x")
    print(f"Load-balance gain: delta reduced by {d_a / max(d_b, 1e-9):.1f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()