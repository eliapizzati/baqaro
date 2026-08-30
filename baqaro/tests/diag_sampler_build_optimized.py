"""Diagnostic: incremental build of the sum-of-lognormals table.

Exercises the optimised (incremental) construction of the 3-D inverse-CDF table
on a reduced grid, so its behaviour can be checked without the multi-hour
production build.

Script-style: executes on import and plots. Run it directly, never under pytest.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

# --- 1. PASTE THE OPTIMIZED BUILDER HERE ---
# (Using the incremental logic we validated previously)
def build_3d_sampler_arrays_optimized(n_min=10, n_max=100, n_steps=10, 
                                      sigma_bounds=(0.5, 1.5), sigma_steps=5, 
                                      z_res=10000):
    print(f"--- Building Optimized Sampler (Testing Mode) ---")
    
    # Grids
    n_grid_log = np.linspace(np.log10(n_min), np.log10(n_max), n_steps)
    n_grid_vals = np.unique(np.round(10**n_grid_log).astype(int))
    sigma_grid = np.linspace(sigma_bounds[0], sigma_bounds[1], sigma_steps)
    
    # Physics Grid (High Res with Deep Stitching)
    stitch_cut = 0.99995
    p_low = np.linspace(0.0, 0.90, 500)
    p_trans = 1.0 - np.logspace(np.log10(0.1), np.log10(1.0 - stitch_cut), 2000)
    p_deep_tail = 1.0 - np.logspace(np.log10(1.0 - stitch_cut), -10, 500)
    p_calc = np.unique(np.concatenate([p_low, p_trans, p_deep_tail]))
    
    # Z-Grid
    z_max = 23.0
    z_grid = np.linspace(0, z_max, z_res)
    p_equivalent = 1.0 - np.exp(-z_grid)

    master_table = np.zeros((len(n_grid_vals), len(sigma_grid), len(z_grid)), dtype=np.float32)

    # --- COMPUTE LOOP (OPTIMIZED) ---
    for j, sig in enumerate(sigma_grid):
        # Accumulators
        S_naive = np.zeros(2_000_000) # Reduced for test speed
        prev_n_naive = 0
        
        n_ak = 20000
        S_ak = np.zeros(n_ak)
        M_ak = np.zeros(n_ak)
        prev_n_ak = 0
        
        for i, n_val in enumerate(n_grid_vals):
            # 1. Update Naive
            delta_n = n_val - prev_n_naive
            if delta_n > 0:
                X_new = np.random.lognormal(0, sig, (len(S_naive), delta_n))
                S_naive += np.sum(X_new, axis=1)
                prev_n_naive = n_val
            
            mask_body = p_calc <= stitch_cut
            body_vals = np.percentile(S_naive, p_calc[mask_body] * 100)
            
            # 2. Update AK
            target_n_bg = n_val - 1
            delta_ak = target_n_bg - prev_n_ak
            if delta_ak > 0:
                X_ak_new = np.random.lognormal(0, sig, (n_ak, delta_ak))
                S_ak += np.sum(X_ak_new, axis=1)
                M_new = np.max(X_ak_new, axis=1)
                M_ak = np.maximum(M_ak, M_new)
                prev_n_ak = target_n_bg
            
            # 3. Compute Tail
            mask_tail = ~mask_body
            if np.any(mask_tail):
                s_start = body_vals[-1]
                scan_thresh = np.logspace(np.log10(s_start), np.log10(s_start*100), 100)
                
                scan_probs = []
                for t in scan_thresh:
                    req = np.maximum(t - S_ak, M_ak)
                    pt = np.mean(stats.lognorm.sf(req, s=sig, scale=1.0) * n_val)
                    scan_probs.append(1.0 - pt)
                
                tail_vals = np.interp(p_calc[mask_tail], scan_probs, scan_thresh)
            else:
                tail_vals = np.array([])
                
            full_curve = np.concatenate([body_vals, tail_vals])
            master_table[i, j, :] = np.interp(p_equivalent, p_calc, np.log(full_curve))
            
    return master_table, n_grid_vals, sigma_grid, z_grid

# --- 2. EXECUTE THE TEST ---

# A. Build the Sampler covering a RANGE of N and Sigma
# N: 10 to 100
# Sigma: 0.5 to 1.5
table, n_grid, s_grid, z_grid = build_3d_sampler_arrays_optimized(
    n_min=10, n_max=100, n_steps=10, 
    sigma_bounds=(0.5, 1.5), sigma_steps=5
)

# Helper function to extract a specific curve from the table
def get_sampler_for_case(target_n, target_sigma):
    # Find nearest indices
    n_idx = np.abs(n_grid - target_n).argmin()
    s_idx = np.abs(s_grid - target_sigma).argmin()
    
    real_n = n_grid[n_idx]
    real_s = s_grid[s_idx]
    
    curve_log_sums = table[n_idx, s_idx, :]
    z_step = z_grid[1] - z_grid[0]
    
    def sampler(n_samples):
        u = np.random.uniform(0, 1, n_samples)
        z = -np.log(1.0 - u)
        idx = np.clip((z / z_step).astype(int), 0, len(z_grid) - 1)
        return np.exp(curve_log_sums[idx])
        
    return sampler, real_n, real_s

# --- 3. DEFINE TEST CASES ---
# We will test 4 corners of the parameter space
test_cases = [
    (10, 0.5),   # Low N, Low Sigma
    (10, 1.5),   # Low N, High Sigma
    (100, 0.5),  # High N, Low Sigma
    (100, 1.5)   # High N, High Sigma
]

# --- 4. PLOT GRID ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (test_n, test_sigma) in enumerate(test_cases):
    ax = axes[i]
    
    # A. Get Optimized Sampler
    my_sampler, used_n, used_s = get_sampler_for_case(test_n, test_sigma)
    S_opt = my_sampler(100_000)
    
    # B. Get Ground Truth
    print(f"Testing Case {i+1}: N={used_n}, Sigma={used_s:.2f}...")
    X_true = np.random.lognormal(0, used_s, (100_000, used_n))
    S_true = np.sum(X_true, axis=1)
    
    # C. Plot Survival Function
    def get_sf(data):
        sorted_data = np.sort(data)
        y = 1.0 - np.linspace(0, 1, len(data))
        return sorted_data, y

    x_true, y_true = get_sf(S_true)
    x_opt, y_opt = get_sf(S_opt)
    
    ax.plot(x_true, y_true, 'k-', lw=4, alpha=0.3, label='Brute Force (Truth)')
    ax.plot(x_opt, y_opt, 'r--', lw=2, label='Optimized Sampler')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1e-5, 1.0)
    ax.set_title(rf"Case {i+1}: N={used_n}, $\sigma$={used_s:.2f}")
    ax.set_ylabel("Survival P(Sum > x)")
    ax.grid(True, which='both', alpha=0.3)
    if i == 0: ax.legend()

plt.tight_layout()
plt.show()