from os.path import join
import sys
import numpy as np
from time import perf_counter as time
from numba import jit

def load_data(load_dir, bid):
    SIZE = 512
    u = np.zeros((SIZE + 2, SIZE + 2))
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask

@jit(nopython=True)
def jacobi_numba(u, interior_mask, max_iter, atol):
    rows, cols = u.shape
    # Bucle de iteraciones
    for n in range(max_iter):
        delta = 0.0
        # Bucle de FILAS (i) - Acceso contiguo en memoria
        for i in range(1, rows - 1):
            # Bucle de COLUMNAS (j)
            for j in range(1, cols - 1):
                # Solo calculamos si es un nodo interior
                if interior_mask[i-1, j-1]:
                    old_val = u[i, j]
                    # Media de los 4 vecinos
                    new_val = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    u[i, j] = new_val
                    
                    # Calcular el error máximo para convergencia
                    diff = abs(new_val - old_val)
                    if diff > delta:
                        delta = diff
        
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

    start_time = time() 

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    LOAD_DIR = '/dtu/projects/02613_2025/data/modified_swiss_dwellings/'
    with open(join(LOAD_DIR, 'building_ids.txt'), 'r') as f:
        building_ids = f.read().splitlines()
    
    # First iteration for the compilation
    u_warm, mask_warm = load_data(LOAD_DIR, building_ids[0])  
    jacobi_numba(u_warm, mask_warm, 1, ABS_TOL)  

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

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi_numba(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    end_time = time() - start_time  # End of time counting 
    
    print(f"\nTotal computation time for {N} floors: {end_time:.4f} s")
    print(f"Average computation per floor: {end_time/N:.4f} s")