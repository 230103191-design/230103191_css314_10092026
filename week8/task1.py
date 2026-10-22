import os
import sys
import time
import platform
import subprocess
import numpy as np

STUDENT_ID = 230103191
N_ELEMENTS = 20_000_000


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


def main():
    print("=" * 60)
    print("Student ID    :", STUDENT_ID)
    print("OS            :", platform.system(), platform.release())
    print("CPU model     :", get_cpu_model())
    print("Logical cores :", os.cpu_count())
    print("=" * 60)

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.standard_normal(N_ELEMENTS, dtype=np.float64)
    print(f"Generated {data.size:,} float64 elements (seed={STUDENT_ID})")

    t0 = time.perf_counter()

    total = 0.0
    for x in data:
        total += x
    mean = total / data.size

    sq = 0.0
    for x in data:
        d = x - mean
        sq += d * d
    std = (sq / data.size) ** 0.5

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Sum           : {total:.6f}")
    print(f"Std (pop.)    : {std:.6f}")
    print(f"Time          : {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()