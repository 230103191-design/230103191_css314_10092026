import numpy as np
import time
import multiprocessing as mp
import json
import os
import sys
from numba import njit
sys.path.append('src')

@njit(nogil=True)
def run_stencil_heavy(chunk_rows, N, iterations=200):
    arr = np.ones((chunk_rows, N))
    for _ in range(iterations):
        arr[1:-1, 1:-1] = (arr[2:, 1:-1] + arr[:-2, 1:-1] + arr[1:-1, 2:] + arr[1:-1, :-2]) / 4.0
    return arr.sum()

if __name__ == "__main__":
    with open("results/salted_params.json", "r") as f:
        params = json.load(f)
    N = params['N']
    max_cores = min(4, os.cpu_count()) 
    P_values = [1, 2, 4][:max_cores]

    with open("results/scaling_metrics.tsv", "w") as f:
        f.write("P\tStrong_T\tWeak_T\n")
        
        t1_strong = 0
        for P in P_values:
            with mp.Pool(P) as pool:
                t0 = time.perf_counter()
                chunk = N // P
                pool.starmap(run_stencil_heavy, [(chunk, N, 200) for _ in range(P)])
                t_strong = time.perf_counter() - t0
            if P == 1: t1_strong = t_strong
            
            N_weak = N * P
            with mp.Pool(P) as pool:
                t0 = time.perf_counter()
                pool.starmap(run_stencil_heavy, [(N_weak // P, N_weak, 200) for _ in range(P)])
                t_weak = time.perf_counter() - t0
                
            f.write(f"{P}\t{t_strong:.4f}\t{t_weak:.4f}\n")
            print(f"P={P} | Strong: {t_strong:.2f}s | Weak: {t_weak:.2f}s")

    roofline = {
        "Arithmetic_Intensity": 5 / 48,
        "Bound_Type": "Memory-Bandwidth Bound"
    }
    with open("results/roofline_summary.json", "w") as f:
        json.dump(roofline, f, indent=4)
    print("Saved scaling and roofline results.")