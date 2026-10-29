import numpy as np
import time
import json
import sys
sys.path.append('src')
from salt import get_student_params

params = get_student_params("230103191")
N = params['N']
M = N // 2  

A = np.random.rand(M, M)
B = np.random.rand(M, M)
C = np.zeros((M, M))

# Naive i-j-k
t0 = time.perf_counter()
for i in range(M):
    for j in range(M):
        for k in range(M):
            C[i, j] += A[i, k] * B[k, j]
t_naive = time.perf_counter() - t0

# Interchanged i-k-j
C = np.zeros((M, M))
t0 = time.perf_counter()
for i in range(M):
    for k in range(M):
        aik = A[i, k]
        for j in range(M):
            C[i, j] += aik * B[k, j]
t_interchanged = time.perf_counter() - t0

speedup = t_naive / t_interchanged
with open("results/loop_interchange.json", "w") as f:
    json.dump({"t_naive": t_naive, "t_interchanged": t_interchanged, "speedup": speedup}, f, indent=4)
print(f"Loop interchange speedup: {speedup:.2f}x")