from os.path import join
import sys
import numpy as np
import cupy as cp
from time import perf_counter as time

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

def jacobi_cupy(u, interior_mask, max_iter, atol=1e-6):
    # Transfer the data to the GPU
    u_gpu = cp.asarray(u)
    mask_gpu = cp.asarray(interior_mask)

    for i in range(max_iter):
        u_new = 0.25 * (u_gpu[1:-1, :-2] + u_gpu[1:-1, 2:] + u_gpu[:-2, 1:-1] + u_gpu[2:, 1:-1])
        u_new_interior = u_new[mask_gpu]
        delta = cp.abs(u_gpu[1:-1, 1:-1][mask_gpu] - u_new_interior).max()
        u_gpu[1:-1, 1:-1][mask_gpu] = u_new_interior

        if float(delta) < atol:  
            break

    return cp.asnumpy(u_gpu)  

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

if __name__ == '__main__':

    start_time = time()

    MAX_ITER = 20_000   
    ABS_TOL = 1e-4   

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 1

    # Warm-up de CuPy 
    u_warm, mask_warm = load_data(LOAD_DIR, building_ids[0])
    jacobi_cupy(u_warm, mask_warm, max_iter=1)   

    building_ids = building_ids[:N]   

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cupy(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    end_time = time() - start_time

    print(f"\n Total Computation Time ({N} floors): {end_time:.4f} s")
    print(f"Estimation for 4571 floors: {end_time / N * 4571 / 3600:.2f} h")