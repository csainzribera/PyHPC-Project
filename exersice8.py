from os.path import join
import sys
import numpy as np
from time import perf_counter as time
from numba import cuda
import matplotlib.pyplot as plt


def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


# -----------------------------
# CUDA KERNEL (1 iteration)
# -----------------------------
@cuda.jit
def jacobi_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)

    rows, cols = u.shape

    if 1 <= i < rows - 1 and 1 <= j < cols - 1:
        if interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (
                u[i - 1, j] +
                u[i + 1, j] +
                u[i, j - 1] +
                u[i, j + 1]
            )
        else:
            u_new[i, j] = u[i, j]


# -----------------------------
# HELPER FUNCTION (loop iterations)
# -----------------------------
def jacobi_cuda(u, interior_mask, max_iter):
    u_device = cuda.to_device(u)
    u_new_device = cuda.device_array_like(u)
    mask_device = cuda.to_device(interior_mask)

    threads_per_block = (16, 16)

    blocks_x = (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    tolerance = 1e-4  # convergence threshold

    for iteration in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](
            u_device, u_new_device, mask_device
        )

        # Copy to host ONLY for convergence check
        u_host = u_device.copy_to_host()
        u_new_host = u_new_device.copy_to_host()

        diff = np.max(np.abs(u_new_host - u_host))

        if diff < tolerance:
            print(f"Converged at iteration {iteration}, diff={diff:.6f}")
            break

        # Swap arrays
        u_device, u_new_device = u_new_device, u_device

    return u_device.copy_to_host()


# -----------------------------
# SUMMARY STATS
# -----------------------------
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

    MAX_ITER = 20000
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    # Load building IDs
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    # Number of floorplans
    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])

    building_ids = building_ids[:N]

    # -----------------------------
    # LOAD ALL DATA
    # -----------------------------
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')

    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # -----------------------------
    # RUN CUDA SOLVER
    # -----------------------------
    all_u = np.empty_like(all_u0)

    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_cuda(u0, interior_mask, MAX_ITER)
        all_u[i] = u

    end_time = time() - start_time

    print(f"\nTotal computation time for {N} floors: {end_time:.4f} seconds")
    print(f"Average computation per floor: {end_time/N:.4f} seconds")

    stats = summary_stats(all_u[0], all_interior_mask[0])
    print("\nSample statistics (first floor):")
    print(stats)

    plt.imshow(all_u[0], cmap='inferno', origin='lower')
    plt.colorbar(label="Temperature")
    plt.title("Heat Distribution (Flipped)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("output.png")