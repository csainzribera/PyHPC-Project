import sys
from os.path import join
import numpy as np
import cupy as cp


def load_data_gpu(load_dir, bid):
    SIZE = 512
    u = cp.zeros((SIZE + 2, SIZE + 2), dtype=cp.float32)

    domain = np.load(join(load_dir, f"{bid}_domain.npy"))
    u[1:-1, 1:-1] = cp.asarray(domain, dtype=cp.float32)

    interior_mask = cp.asarray(
        np.load(join(load_dir, f"{bid}_interior.npy")),
        dtype=bool
    )

    return u, interior_mask


def jacobi_gpu(u, interior_mask, max_iter, atol=1e-4, check_every=100):
    u = u.copy()
    mask = interior_mask

    u_new = cp.empty_like(u)

    for i in range(max_iter):

        # 0) save old interior values for convergence check
        u_old = u[1:-1, 1:-1].copy()

        # 1) compute into preallocated array (no new allocation)
        u_new[1:-1, 1:-1] = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )

        # 2) apply mask using boolean indexing (avoids cp.where compilation)
        u[1:-1, 1:-1][mask] = u_new[1:-1, 1:-1][mask]

        # 3) convergence check on interior points only (reduced sync)
        if (i % check_every) == 0:
            delta = float(cp.max(cp.abs((u_new[1:-1, 1:-1] - u_old)[mask])).get())
            if delta < atol:
                break

    return u


def summary_stats_gpu(u, interior_mask):
    u_cpu = cp.asnumpy(u[1:-1, 1:-1][interior_mask])
    return {
        'mean_temp': float(u_cpu.mean()),
        'std_temp': float(u_cpu.std()),
        'pct_above_18': float((u_cpu > 18).sum() / u_cpu.size * 100),
        'pct_below_15': float((u_cpu < 15).sum() / u_cpu.size * 100),
    }


if __name__ == '__main__':
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt')) as f:
        building_ids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = building_ids[:N]

    all_u0 = []
    all_masks = []

    for bid in building_ids:
        u0, mask = load_data_gpu(LOAD_DIR, bid)
        all_u0.append(u0)
        all_masks.append(mask)

    MAX_ITER = 20000
    ABS_TOL = 1e-4

    results = []

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()

    for u0, mask in zip(all_u0, all_masks):
        u_final = jacobi_gpu(u0, mask, MAX_ITER, atol=ABS_TOL)
        results.append((u_final, mask))

    end.record()
    end.synchronize()

    print(f"GPU time: {cp.cuda.get_elapsed_time(start, end)/1000:.3f}s for {N}")

    print("building_id,mean,std,%>18,%<15")
    for bid, (u, mask) in zip(building_ids, results):
        stats = summary_stats_gpu(u, mask)
        print(f"{bid},{stats['mean_temp']:.4f},{stats['std_temp']:.4f},"
              f"{stats['pct_above_18']:.1f},{stats['pct_below_15']:.1f}")