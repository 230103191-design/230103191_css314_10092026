import numpy as np
import time
import json
import sys
from numba import njit

@njit
def run_naive(A, B, M):
    C = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit
def run_interchanged(A, B, M):
    C = np.zeros((M, M))
    for i in range(M):
        for k in range(M):
            aik = A[i, k]
            for j in range(M):
                C[i, j] += aik * B[k, j]
    return C

with open("results/salted_params.json", "r") as f:
    params = json.load(f)
N = params['N']
M = N // 2  
print(f"Matrix size M = {M}. Pre-compiling Numba functions...")
A = np.random.rand(M, M)
B = np.random.rand(M, M)

run_naive(np.ones((2,2)), np.ones((2,2)), 2)
run_interchanged(np.ones((2,2)), np.ones((2,2)), 2)

print("Running Naive i-j-k...")
t0 = time.perf_counter()
C_naive = run_naive(A, B, M)
t_naive = time.perf_counter() - t0

print("Running Interchanged i-k-j...")
t0 = time.perf_counter()
C_interchanged = run_interchanged(A, B, M)
t_interchanged = time.perf_counter() - t0

speedup = t_naive / t_interchanged
print(f"Naive: {t_naive:.4f}s | Interchanged: {t_interchanged:.4f}s | Speedup: {speedup:.2f}x")

with open("results/loop_interchange.json", "w") as f:
    json.dump({"t_naive": t_naive, "t_interchanged": t_interchanged, "speedup": speedup}, f, indent=4)
print("Saved to results/loop_interchange.json")
