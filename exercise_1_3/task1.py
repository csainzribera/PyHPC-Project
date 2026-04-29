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

def save_visual_comparison(all_u_init, all_u_final, all_masks, building_ids, N, folder='results'):
    fig, axes = plt.subplots(2, N, figsize=(3 * N, 6), squeeze=False)
    vmin, vmax = 0, 25

    for i in range(N):
        u_init = all_u_init[i]
        u_final = all_u_final[i]
        interior_mask = all_masks[i]
        bid = building_ids[i]

        init_full = u_init[1:-1, 1:-1]
        final_interior = u_final[1:-1, 1:-1]

        im = axes[0, i].imshow(init_full, cmap='inferno', origin='lower',
                                vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"ID: {bid}")

        axes[1, i].imshow(final_interior, cmap='inferno', origin='lower',
                           vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"ID: {bid}")

        if i != 0:
            axes[0, i].set_yticks([])
            axes[1, i].set_yticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.set_label('Temperature (°C)')
    plt.show()
    plt.savefig("results/task1_3.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    # Load data
    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    processed_data = {}

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
        bid = building_ids[i]
        processed_data[bid] = {'u_init': u0, 'u_final': u, 'mask': interior_mask}


    # Print summary statistics in CSV format
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(building_ids, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))
    
    #all_u_init = np.array([processed_data[bid]['u_init'] for bid in building_ids])
    #all_u_final = np.array([processed_data[bid]['u_final'] for bid in building_ids])
    #all_masks = np.array([processed_data[bid]['mask'] for bid in building_ids])

    print(f"\nPlotting results for {N} buildings...")
    save_visual_comparison(all_u0, all_u, all_interior_mask, building_ids, N)