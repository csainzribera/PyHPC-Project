from os.path import join
import sys
import numpy as np
import pandas as pd  # Added Pandas
from time import perf_counter as time
from numba import cuda
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

# -----------------------------
# CUDA KERNEL (Jacobi Stencil)
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
# CUDA RUNNER
# -----------------------------
def jacobi_cuda(u, interior_mask, max_iter):
    u_device = cuda.to_device(u)
    u_new_device = cuda.device_array_like(u)
    mask_device = cuda.to_device(interior_mask)

    threads_per_block = (16, 16)
    blocks_x = (u.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (u.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    for _ in range(max_iter):
        jacobi_kernel[blocks_per_grid, threads_per_block](
            u_device, u_new_device, mask_device
        )
        u_device, u_new_device = u_new_device, u_device

    return u_device.copy_to_host()

# -----------------------------
# SUMMARY STATS
# -----------------------------
def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    return {
        'mean_temp': u_interior.mean(),
        'std_temp': u_interior.std(),
        'pct_above_18': np.sum(u_interior > 18) / u_interior.size * 100,
        'pct_below_15': np.sum(u_interior < 15) / u_interior.size * 100,
    }

# -----------------------------
# MAIN PROGRAM
# -----------------------------
if __name__ == '__main__':
    MAX_ITER = 20000
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'

    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        all_bids = f.read().splitlines()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    building_ids = all_bids[:N]

    results_list = [] # List to store stats for Pandas
    start_total = time()
    
    print(f"Processing {N} buildings...")

    for bid in building_ids:
        # Load and Solve
        u0, mask = load_data(LOAD_DIR, bid)
        u_final = jacobi_cuda(u0, mask, MAX_ITER)
        
        # Calculate Stats
        stats = summary_stats(u_final, mask)
        stats['building_id'] = bid
        results_list.append(stats)

        # Print progress to console
        print(f"Done: {bid} | Mean: {stats['mean_temp']:.2f}")

    # --- PANDAS STORAGE ---
    df = pd.DataFrame(results_list)
    
    # Reorder columns to put building_id first
    cols = ['building_id', 'mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    df = df[cols]

    # Save to results.csv
    df.to_csv('results.csv', index=False)
    print("\nResults successfully saved to 'results.csv'")

    # --- FINAL ANALYSIS (Using Pandas) ---
    avg_mean = df['mean_temp'].mean()
    avg_std = df['std_temp'].mean()
    p18_count = (df['pct_above_18'] >= 50).sum()
    p15_count = (df['pct_below_15'] >= 50).sum()

    print("\n" + "="*40)
    print(f"RESULTS FOR {N} BUILDINGS")
    print("="*40)
    print(f"Average mean temperature: {avg_mean:.4f} °C")
    print(f"Average std deviation:   {avg_std:.4f}")
    print(f"Buildings >= 50% area > 18°C: {p18_count}")
    print(f"Buildings >= 50% area < 15°C: {p15_count}")
    print(f"Total time: {time() - start_total:.2f} seconds")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.hist(df['mean_temp'], bins=30, color='royalblue', edgecolor='black', alpha=0.7)
    plt.title(f'Mean Temperature Distribution (N={N})')
    plt.xlabel('Mean Temperature (°C)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.6) 
    plt.savefig('temperature_histogram.png')