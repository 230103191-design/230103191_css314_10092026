import os
import time
import platform
import subprocess
import threading

STUDENT_ID = 230103191
ITERATIONS = 50_000_000
PHYSICAL_CORES = 4


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


def run_unpadded(n_threads, iters):
    shared_counts = [0] * n_threads

    def worker(idx):
        for _ in range(iters):
            shared_counts[idx] += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, sum(shared_counts)


def run_padded(n_threads, iters):
    PAD = 64 // 8
    shared_counts = [0] * (n_threads * PAD)

    def worker(idx):
        local = 0
        for _ in range(iters):
            local += 1
        shared_counts[idx * PAD] = local

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = sum(shared_counts[i * PAD] for i in range(n_threads))
    return time.perf_counter() - t0, total


def main():
    print("=" * 70)
    print("TASK 3 — Cache Line Contention (False Sharing)")
    print("=" * 70)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", os.cpu_count())
    print("Threads used  :", PHYSICAL_CORES)
    print("Iterations    :", f"{ITERATIONS:,} per thread")
    print("=" * 70)

    print("\n[1] Unpadded shared array (false sharing) ...")
    t_unpadded, total_u = run_unpadded(PHYSICAL_CORES, ITERATIONS)
    print(f"    Time   : {t_unpadded:.3f} s")
    print(f"    Total  : {total_u:,}")

    print("\n[2] Padded / local accumulation (no false sharing) ...")
    t_padded, total_p = run_padded(PHYSICAL_CORES, ITERATIONS)
    print(f"    Time   : {t_padded:.3f} s")
    print(f"    Total  : {total_p:,}")

    print("\n" + "-" * 70)
    print(f"{'Version':<28} | {'Time':>10} | {'Ratio':>10}")
    print("-" * 70)
    print(f"{'Unpadded Shared':<28} | {t_unpadded:>8.3f} s | {1.0:>9.2f}x")
    print(f"{'Padded / Local':<28} | {t_padded:>8.3f} s | {t_unpadded / t_padded:>9.2f}x")
    print("-" * 70)
    print(f"False-sharing penalty: {t_unpadded / t_padded:.2f}x slower unpadded")
    print("=" * 70)


if __name__ == "__main__":
    main()