from os.path import join
import sys

import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()

    if len(sys.argv) < 2:
        N = 1
    else:
        N = int(sys.argv[1])
    building_ids = building_ids[:N]

    # Load floor plans
    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype='bool')
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    # Run jacobi iterations for each floor plan
    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    # Visualize results for the first few floorplans
    num_to_visualize = min(3, N)
    fig, axes = plt.subplots(num_to_visualize, 2, figsize=(10, 5*num_to_visualize))
    if num_to_visualize == 1:
        axes = [axes]  # Make it 2D

    for i in range(num_to_visualize):
        bid = building_ids[i]
        u0 = all_u0[i]
        u = all_u[i]
        interior_mask = all_interior_mask[i]

        # Prepare data for plotting: set exterior to NaN for better visualization
        temp_initial = np.full((512, 512), np.nan)
        temp_initial[interior_mask] = u0[1:-1, 1:-1][interior_mask]

        temp_final = np.full((512, 512), np.nan)
        temp_final[interior_mask] = u[1:-1, 1:-1][interior_mask]

        # Plot initial temperature
        im1 = axes[i][0].imshow(temp_initial, cmap='coolwarm', vmin=10, vmax=25)
        axes[i][0].set_title(f'Building {bid} - Initial Temperature')
        axes[i][0].axis('off')
        plt.colorbar(im1, ax=axes[i][0], shrink=0.8)

        # Plot final temperature
        im2 = axes[i][1].imshow(temp_final, cmap='coolwarm', vmin=10, vmax=25)
        axes[i][1].set_title(f'Building {bid} - Final Temperature')
        axes[i][1].axis('off')
        plt.colorbar(im2, ax=axes[i][1], shrink=0.8)

    plt.tight_layout()
    plt.savefig('floorplan_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
