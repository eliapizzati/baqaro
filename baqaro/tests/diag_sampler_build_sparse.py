"""Diagnostic: interpolation accuracy of a sparsely-sampled sampler table.

Builds the table on a deliberately sparse N grid (skipping intermediate values)
and checks what the N-interpolation reconstructs, which bounds the error the
production table's grid spacing introduces.

Script-style: executes on import and plots. Run it directly, never under pytest.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. BUILDER (Same Optimized Incremental Builder) ---
def build_sparse_sampler(sigma_bounds=(1.0, 1.0), sigma_steps=1):
    print(f"--- Building SPARSE Sampler (N = 10, 50, 100) ---")
    
    # DEFINE A SPARSE N GRID
    # We purposefully skip 20, 30, 40 to test interpolation later
    n_grid_vals = np.array([10, 50, 100])
    
    sigma_grid = np.linspace(sigma_bounds[0], sigma_bounds[1], sigma_steps)
    
    # Physics Grid
    stitch_cut = 0.99995
    p_low = np.linspace(0.0, 0.90, 500)
    p_trans = 1.0 - np.logspace(np.log10(0.1), np.log10(1.0 - stitch_cut), 2000)
    p_deep_tail = 1.0 - np.logspace(np.log10(1.0 - stitch_cut), -10, 500)
    p_calc = np.unique(np.concatenate([p_low, p_trans, p_deep_tail]))
    
    # Z-Grid
    z_res = 10000
    z_max = 23.0
    z_grid = np.linspace(0, z_max, z_res)
    p_equivalent = 1.0 - np.exp(-z_grid)

    master_table = np.zeros((len(n_grid_vals), len(sigma_grid), len(z_grid)), dtype=np.float32)

    # COMPUTE LOOP
    for j, sig in enumerate(sigma_grid):
        S_naive = np.zeros(2_000_000)
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

# --- 2. UPGRADED SAMPLER (Linear Interpolation on N) ---
def get_interpolating_sampler(table, n_grid, z_grid, target_n):
    """
    Returns a sampler for 'target_n' by interpolating between the two nearest N-slices.
    """
    # Find neighbors
    # We assume log-interpolation is best for N
    target_log = np.log10(target_n)
    n_log_grid = np.log10(n_grid)
    
    # Find index to the left (i) such that n[i] <= target < n[i+1]
    idx = np.searchsorted(n_log_grid, target_log, side='right') - 1
    idx = np.clip(idx, 0, len(n_grid)-2)
    
    n_left = n_grid[idx]
    n_right = n_grid[idx+1]
    
    # Calculate weight (0 to 1)
    w = (target_log - np.log10(n_left)) / (np.log10(n_right) - np.log10(n_left))
    
    print(f" Interpolating N={target_n}: Blending N={n_left} (w={1-w:.2f}) and N={n_right} (w={w:.2f})")
    
    z_step = z_grid[1] - z_grid[0]
    
    # Pre-fetch the two curves (Log Sums)
    curve_left = table[idx, 0, :]   # Sigma index 0
    curve_right = table[idx+1, 0, :]
    
    def sampler(n_samples):
        u = np.random.uniform(0, 1, n_samples)
        z = -np.log(1.0 - u)
        z_idx = np.clip((z / z_step).astype(int), 0, len(z_grid) - 1)
        
        # INTERPOLATE VALUES
        # value = (1-w)*left + w*right
        log_val = (1-w) * curve_left[z_idx] + w * curve_right[z_idx]
        
        return np.exp(log_val)
        
    return sampler

# --- 3. RUN TEST ---
# A. Build Sparse Table
table, n_grid, s_grid, z_grid = build_sparse_sampler()

# B. Target N=25 (Absent from grid [10, 50, 100])
TEST_N = 25
TEST_SIGMA = 1.0

# C. Get Interpolated Sampler
my_sampler = get_interpolating_sampler(table, n_grid, z_grid, TEST_N)
S_interp = my_sampler(100_000)

# D. Ground Truth
print(f"Generating Ground Truth for N={TEST_N}...")
X_true = np.random.lognormal(0, TEST_SIGMA, (100_000, TEST_N))
S_true = np.sum(X_true, axis=1)

# --- 4. PLOT ---
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

def get_sf(data):
    sorted_data = np.sort(data)
    y = 1.0 - np.linspace(0, 1, len(data))
    return sorted_data, y

x_true, y_true = get_sf(S_true)
x_int, y_int = get_sf(S_interp)

ax.plot(x_true, y_true, 'k-', lw=5, alpha=0.3, label=f'Brute Force (N={TEST_N})')
ax.plot(x_int, y_int, 'r--', lw=2, label=f'Interpolated Sampler (from N=10, 50)')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-5, 1.0)
ax.set_title(f"Validation: Interpolating Missing N (Target={TEST_N})")
ax.set_ylabel("Survival P(Sum > x)")
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Means -> Truth: {np.mean(S_true):.2f}, Interp: {np.mean(S_interp):.2f}")