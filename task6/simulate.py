from os.path import join
import sys
import time
from multiprocessing import Pool

import numpy as np


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'pct_above_18': pct_above_18,
        'pct_below_15': pct_below_15,
    }

# Function for processing one building
def process_building(args):
    bid, load_dir, max_iter, abs_tol = args

    u0, interior_mask = load_data(load_dir, bid)
    u = jacobi(u0, interior_mask, max_iter, abs_tol)
    stats = summary_stats(u, interior_mask)

    return bid, stats

if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    if len(sys.argv) < 3:
        n_workers = 1
    else:
        n_workers = int(sys.argv[2])

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    building_ids = building_ids[:N]

    building_args = []
    for bid in building_ids:
        building_args.append((bid, LOAD_DIR, MAX_ITER, ABS_TOL))

    t0 = time.perf_counter()

    with Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(process_building, building_args, chunksize=1))

    t1 = time.perf_counter()
    elapsed = t1 - t0

    results.sort(key=lambda x: x[0])

    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']

    print(f'# workers={n_workers}, N={N}, elapsed_seconds={elapsed:.6f}')
    print('building_id,' + ','.join(stat_keys))
    for bid, stats in results:
        print(f"{bid}," + ",".join(str(stats[k]) for k in stat_keys))
