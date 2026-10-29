import numpy as np
import time
import json

data = np.ones(16_777_216, dtype=np.float64)
strides = [1, 2, 4, 8, 16, 32, 64, 128]
results = []

for s in strides:
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        total = 0.0
        for i in range(0, len(data), s):
            total += data[i]
        times.append((time.perf_counter() - t0) * 1e9)
    
    times.sort()
    median_ns = times[len(times)//2]
    latency_per_element = median_ns / (len(data) / s)
    results.append(f"{s}\t{median_ns:.2f}\t{latency_per_element:.2f}")

with open("results/stride_results.tsv", "w") as f:
    f.write("Stride\tTotal_Time_ns\tLatency_per_element_ns\n")
    f.write("\n".join(results))
print("Stride bench done.")