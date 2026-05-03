from os.path import join
import sys
import time
from multiprocessing import Pool

import numpy as np
from numba import jit


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


@jit(nopython=True, cache=True)
def jacobi_numba(u, interior_mask, max_iter, atol):
    rows, cols = u.shape
    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, rows - 1):        # outer: rows (C-contiguous)
            for j in range(1, cols - 1):    # inner: columns
                if interior_mask[i - 1, j - 1]:
                    old_val = u[i, j]
                    new_val = 0.25 * (u[i-1, j] + u[i+1, j] +
                                      u[i, j-1] + u[i, j+1])
                    u[i, j] = new_val
                    diff = abs(new_val - old_val)
                    if diff > delta:
                        delta = diff
        if delta < atol:
            break
    return u



def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp':    u_interior.mean(),
        'std_temp':     u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }


# Worker initializer: compile Numba ONCE per worker process 
def worker_init(load_dir):
    global _LOAD_DIR
    _LOAD_DIR = load_dir

    u_warm = np.zeros((514, 514))
    m_warm = np.zeros((512, 512), dtype=bool)
    m_warm[0, 0] = True
    jacobi_numba(u_warm, m_warm, 1, 1e-4)   # compile-only warmup


def process_building(args):
    bid, max_iter, abs_tol = args
    u0, interior_mask = load_data(_LOAD_DIR, bid)
    u = jacobi_numba(u0, interior_mask, max_iter, abs_tol)
    stats = summary_stats(u, interior_mask)
    return bid, stats


if __name__ == '__main__':

    t0 = time.perf_counter()

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N         = int(sys.argv[1]) if len(sys.argv) >= 2 else 1
    n_workers = int(sys.argv[2]) if len(sys.argv) >= 3 else 1

    building_ids = building_ids[:N]

    MAX_ITER = 20_000
    ABS_TOL  = 1e-4

    task_args = [(bid, MAX_ITER, ABS_TOL) for bid in building_ids]

    # LOAD + RUN JACOBI: imap_unordered → dynamic scheduling
    with Pool(processes=n_workers, initializer=worker_init, initargs=(LOAD_DIR,)) as pool:
        results = list(pool.imap_unordered(process_building, task_args, chunksize=1))

    results.sort(key=lambda x: x[0])  # ordenar por building_id

    end_time = time.perf_counter() - t0

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
    
    print(f"\nTotal computation time for {N} floors: {end_time:.4f} s")
    print(f"Average computation per floor: {end_time/N:.4f} s")