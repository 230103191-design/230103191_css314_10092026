import multiprocessing as mp
import time
import numpy as np
from numba import njit

@njit(nogil=True)
def run_count(counter_array, worker_id, iterations, padded):
    idx = worker_id * 8 if padded else worker_id
    for _ in range(iterations):
        counter_array[idx] += 1

@njit(nogil=True)
def run_private(counter_array, worker_id, iterations):
    idx = worker_id * 8
    local_sum = 0
    for _ in range(iterations):
        local_sum += 1
    counter_array[idx] += local_sum

def worker_task(counter_array, worker_id, iterations, mode):
    if mode == "unpadded":
        run_count(counter_array, worker_id, iterations, False)
    elif mode == "padded":
        run_count(counter_array, worker_id, iterations, True)
    elif mode == "private":
        run_private(counter_array, worker_id, iterations)

if __name__ == "__main__":
    K = 5_000_000
    P_values = [1, 2, 4, 8]
    
    with open("results/false_sharing_karp_flatt.tsv", "w") as f:
        f.write("P\tUnpadded_s\tPadded_s\tPrivate_s\tKarpFlatt_Unpadded\n")
        
        t1_unpadded = 0
        for P in P_values:
            
            shared_arr = mp.Array('q', 8, lock=False) 
            procs = [mp.Process(target=worker_task, args=(shared_arr, p, K, "unpadded")) for p in range(P)]
            t0 = time.perf_counter()
            for p in procs: p.start()
            for p in procs: p.join()
            t_unpadded = time.perf_counter() - t0
            if P == 1: t1_unpadded = t_unpadded
            
            shared_arr = mp.Array('q', P * 8, lock=False)
            procs = [mp.Process(target=worker_task, args=(shared_arr, p, K, "padded")) for p in range(P)]
            t0 = time.perf_counter()
            for p in procs: p.start()
            for p in procs: p.join()
            t_padded = time.perf_counter() - t0

            shared_arr = mp.Array('q', P * 8, lock=False)
            procs = [mp.Process(target=worker_task, args=(shared_arr, p, K, "private")) for p in range(P)]
            t0 = time.perf_counter()
            for p in procs: p.start()
            for p in procs: p.join()
            t_private = time.perf_counter() - t0

            S = t1_unpadded / t_unpadded
            e_P = ((1/S) - (1/P)) / (1 - (1/P)) if P > 1 else 0.0
            
            f.write(f"{P}\t{t_unpadded:.4f}\t{t_padded:.4f}\t{t_private:.4f}\t{e_P:.4f}\n")
            print(f"P={P} | Unpadded: {t_unpadded:.2f}s | Padded: {t_padded:.2f}s | Private: {t_private:.2f}s")