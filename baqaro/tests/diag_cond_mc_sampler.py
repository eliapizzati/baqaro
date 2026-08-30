"""Diagnostic: build a sum-of-lognormals sampler by direct Monte Carlo.

An independent, deliberately simple construction of the inverse-CDF table that
``core_functions/build_lognormal_sampler.py`` produces -- used to check the
production table rather than to generate it.

Script-style: defines its builder at import. Run it directly, never under pytest.
"""

import numpy as np
import scipy.stats as stats

def build_lognormal_sum_sampler(n_vars=100, 
                                sigma_bounds=(0.1, 3.0), 
                                n_sigma_points=35,
                                stitch_prob=0.99995): # CHANGED: Deeper stitch (10^-5)
    
    print(f"--- Building Universal Sampler for Sum of {n_vars} Lognormals ---")
    
    # --- 1. SETUP HIGH-PRECISION GRIDS ---
    # Sigma Grid
    sigma_grid = np.linspace(sigma_bounds[0], sigma_bounds[1], n_sigma_points)
    
    # Probability Grid (The "Map Coordinates")
    # CHANGED: We use Log-Spacing for the transition to avoid interpolation bias
    # 1. Linear Body (0 to 0.90)
    p_low = np.linspace(0.0, 0.90, 1000)
    # 2. Log-Spaced Transition (0.90 to Stitch)
    p_trans = 1.0 - np.logspace(np.log10(0.1), np.log10(1.0 - stitch_prob), 2000)
    # 3. Log-Spaced Tail (Stitch to 10^-10)
    p_tail = 1.0 - np.logspace(np.log10(1.0 - stitch_prob), -10, 500)
    
    # Combine
    p_grid = np.unique(np.concatenate([p_low, p_trans, p_tail]))
    
    # Surface Storage
    surface_values = np.zeros((len(sigma_grid), len(p_grid)))
    
    # --- 2. POPULATE SURFACE ---
    print(f"Computing probability surface ({len(sigma_grid)} sigma slices)...")
    
    for i, sig in enumerate(sigma_grid):
        # A. NAIVE MC (Body)
        # CHANGED: Increased to 5 Million to stabilize the stitch point
        n_naive = 5_000_000 
        X = np.random.lognormal(0, sig, (n_naive, n_vars))
        S = np.sum(X, axis=1)
        
        mask_body = p_grid <= stitch_prob
        body_vals = np.percentile(S, p_grid[mask_body] * 100)
        
        # B. ASMUSSEN-KROESE (Tail)
        mask_tail = ~mask_body
        if np.any(mask_tail):
            s_start = body_vals[-1]
            s_end = s_start * 100 # Scanned range
            
            scan_thresholds = np.logspace(np.log10(s_start), np.log10(s_end), 100)
            
            # Re-use noise (increased to 10k for smoothness)
            X_bg = np.random.lognormal(0, sig, (10000, n_vars-1))
            S_bg = np.sum(X_bg, axis=1)
            M_bg = np.max(X_bg, axis=1)
            
            scan_probs = []
            for t in scan_thresholds:
                req = np.maximum(t - S_bg, M_bg)
                # AK Formula
                p_tail_event = np.mean(stats.lognorm.sf(req, s=sig, scale=1.0) * n_vars)
                scan_probs.append(1.0 - p_tail_event)
            
            tail_vals = np.interp(p_grid[mask_tail], scan_probs, scan_thresholds)
        else:
            tail_vals = np.array([])

        full_curve = np.concatenate([body_vals, tail_vals])
        surface_values[i, :] = np.log(full_curve)
        
    print("Surface complete. Building fast sampler...")

    # --- 3. PREPARE DATA FOR FAST LOOKUP ---
    sigma_grid = np.ascontiguousarray(sigma_grid)
    p_grid = np.ascontiguousarray(p_grid)
    surface_values = np.ascontiguousarray(surface_values)
    
    sig_min = sigma_grid[0]
    sig_step = sigma_grid[1] - sigma_grid[0]
    n_sigma = len(sigma_grid)
    n_p = len(p_grid)

    # --- 4. DEFINE THE FAST WRAPPER FUNCTION ---
    def sampler(mu_arr, sigma_arr, u_arr=None):
        mu_arr = np.asarray(mu_arr)
        sigma_arr = np.asarray(sigma_arr)
        n = len(mu_arr)
        
        if u_arr is None:
            u_arr = np.random.uniform(0, 1, n)
            
        # A. SIGMA INDEXING
        s_float_idx = (sigma_arr - sig_min) / sig_step
        i = np.clip(s_float_idx.astype(int), 0, n_sigma - 2)
        w_s = np.clip(s_float_idx - i, 0.0, 1.0)

        # B. PROBABILITY INDEXING
        j = np.searchsorted(p_grid, u_arr, side='right') - 1
        j = np.clip(j, 0, n_p - 2)
        
        p_left = p_grid[j]
        p_right = p_grid[j+1]
        
        # Division safety
        p_denom = p_right - p_left
        # If p_denom is 0 (shouldn't happen with unique), default to 0
        w_p = np.divide(u_arr - p_left, p_denom, out=np.zeros_like(u_arr), where=p_denom!=0)
        
        # C. INTERPOLATE
        v00 = surface_values[i, j]
        v10 = surface_values[i+1, j]
        v01 = surface_values[i, j+1]
        v11 = surface_values[i+1, j+1]
        
        val_left  = v00 + w_s * (v10 - v00)
        val_right = v01 + w_s * (v11 - v01)
        
        log_base_sums = val_left + w_p * (val_right - val_left)
        
        return np.exp(log_base_sums) * np.exp(mu_arr)

    return sampler