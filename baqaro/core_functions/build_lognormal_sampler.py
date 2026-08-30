"""3D Sampler for Sums of Lognormal Random Variables.

This module provides tools to efficiently sample from the distribution of
S = sum_{k=1}^{N} X_k, where each X_k ~ Lognormal(mu, sigma).

Direct Monte Carlo sampling of such sums is expensive when N is large or when
millions of samples are needed. This module precomputes the inverse CDF of S
over a 3D grid of (N, sigma, z) and stores it in a lookup table. At runtime,
samples are drawn via fast trilinear interpolation.

Workflow:
    1. build_3d_sampler_arrays() - Precompute the lookup table (one-time cost)
    2. save_3d_sampler() - Persist the table to disk
    3. load_3d_sampler() - Load table and return a fast sampler function
    4. sampler(n, mu, sigma, rng) - Draw samples via interpolation

The z-grid is defined via z = -ln(1 - p), where p is the CDF probability.
This transformation maps [0, 1) to [0, inf), providing high resolution in
both the bulk and the tail of the distribution.
"""

import numpy as np
import scipy.stats as stats
import os
import time
from numba import njit, prange

from baqaro.utils.my_dir import get_output_path

# Module-level configuration. The site (and hence where the table is written)
# comes from the environment, as for every other entry point; set
# BAQARO_OUTPUT_DIR to place the table without editing my_dir.py.
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")


# ==============================================================================
# 1. THE BUILDER (Optimized CumSum Strategy)
# ==============================================================================
def build_3d_sampler_arrays(n_min=5, n_max=1000, n_steps=200, 
                                    sigma_dex_bounds=(0.1, 3), # Input range in DEX
                                      sigma_steps=50, 
                                      z_res=10000,
                                      n_naive_samples=5_000_000,
                                      seed=42):
    """
    Build the 3D lookup table for lognormal sum sampling.
    
    This function precomputes log(S) values on a 3D grid of (N, sigma, z),
    where S is the sum of N lognormal variables with scatter sigma (in dex).
    The z-axis encodes the CDF via z = -ln(1 - p).
    
    The algorithm uses two strategies:
        - Bulk (p < 0.99995): Direct Monte Carlo percentiles from 5M samples
        - Tail (p >= 0.99995): Asmussen-Kroese importance sampling for accuracy
    
    Incremental accumulation is used: when stepping from N to N+1, only one
    new lognormal variable is added per sample, avoiding redundant computation.
    
    Parameters
    ----------
    n_min : int
        Minimum number of lognormal variables to sum (default: 5).
    n_max : int
        Maximum number of lognormal variables to sum (default: 1000).
    n_steps : int
        Number of log-spaced grid points for N (default: 200).
    sigma_dex_bounds : tuple of float
        (min, max) scatter in dex for the lognormal distribution (default: (0.1, 3)).
    sigma_steps : int
        Number of linearly-spaced grid points for sigma (default: 50).
    z_res : int
        Number of grid points for the z-axis / CDF resolution (default: 10000).
    n_naive_samples : int
        Number of Monte Carlo samples for bulk CDF estimation (default: 5,000,000).
        Reduce for memory-constrained systems.
    seed : int
        Random seed for reproducibility (default: 42).
    
    Returns
    -------
    master_table : ndarray, shape (n_len, sigma_len, z_len), dtype float32
        Precomputed log(S) values. Access as table[n_idx, sigma_idx, z_idx].
    n_grid_vals : ndarray of int
        The N values corresponding to axis 0 of master_table.
    sigma_grid_dex : ndarray of float
        The sigma values (in dex) corresponding to axis 1.
    z_grid : ndarray of float
        The z values corresponding to axis 2, where z = -ln(1 - p).
    
    Notes
    -----
    - Memory usage: ~(n_len * sigma_len * z_len * 4) bytes for float32 storage.
    - Build time: O(sigma_steps * n_max) due to incremental accumulation.
    - The table stores log(S) to avoid numerical issues with large sums.
    """
    print(f"--- Building 3D Master Table (Seed: {seed}) ---")
    print(f"    N Range: {n_min}-{n_max} | Sigma Range: {sigma_dex_bounds}")
    
    # Create a local, deterministic generator
    rng = np.random.default_rng(seed)
    
    # Constants
    LN_10 = np.log(10.0)

    # --- A. SETUP GRIDS ---
    # 1. N-Grid (Log Spaced)
    n_grid_log = np.linspace(np.log10(n_min), np.log10(n_max), n_steps)
    n_grid_vals = np.unique(np.round(10**n_grid_log).astype(int))
    
    # 2. Sigma Grid (Linear in DEX)
    # The saved grid will be [0.1, 0.15, ..., 1.3]
    sigma_grid_dex = np.linspace(sigma_dex_bounds[0], sigma_dex_bounds[1], sigma_steps)

    # 3. Physics Grid (High Precision Stitching)
    stitch_cut = 0.99995
    p_low = np.linspace(0.0, 0.90, 500)
    p_trans = 1.0 - np.logspace(np.log10(0.1), np.log10(1.0 - stitch_cut), 2000)
    p_deep_tail = 1.0 - np.logspace(np.log10(1.0 - stitch_cut), -10, 500)
    p_calc = np.unique(np.concatenate([p_low, p_trans, p_deep_tail]))
    
    # 4. Z-Grid (Linear for O(1) Lookup)
    z_max = 23.0
    z_grid = np.linspace(0, z_max, z_res)
    p_equivalent = 1.0 - np.exp(-z_grid)

    # Storage
    master_table = np.zeros((len(n_grid_vals), len(sigma_grid_dex), len(z_grid)), dtype=np.float32)

    # --- B. COMPUTE LOOP (Loop Sigma -> Then N) ---
    for j, sig_dex in enumerate(sigma_grid_dex):
        print(f" Processing Sigma {j+1}/{len(sigma_grid_dex)}: {sig_dex:.2f} dex")
        t0 = time.time()
        
        # *** CRITICAL: Convert Dex -> Natural for Physics ***
        sig_natural = sig_dex * LN_10

        # Memory Safety: n_naive_samples can be reduced for memory-constrained systems
        S_naive = np.zeros(n_naive_samples) 
        prev_n_naive = 0

        # Initialize AK Accumulators (20k samples for tail)
        n_ak = 20000
        S_ak = np.zeros(n_ak)
        M_ak = np.zeros(n_ak)
        prev_n_ak = 0
        
        for i, n_val in enumerate(n_grid_vals):
            print(f"   > N {i+1}/{len(n_grid_vals)}: {n_val}", end="\r")
            # 1. Increment Naive Sum
            delta_n = n_val - prev_n_naive
            if delta_n > 0:
                # Add only the new variables
                X_new = rng.lognormal(0, sig_natural, (len(S_naive), delta_n))
                S_naive += np.sum(X_new, axis=1)
                prev_n_naive = n_val
            
            mask_body = p_calc <= stitch_cut
            body_vals = np.percentile(S_naive, p_calc[mask_body] * 100)
            
            # 2. Increment AK State
            # AK uses sum of (N-1) variables
            target_n_bg = n_val - 1
            delta_ak = target_n_bg - prev_n_ak
            if delta_ak > 0:
                X_ak_new = rng.lognormal(0, sig_natural, (n_ak, delta_ak))
                S_ak += np.sum(X_ak_new, axis=1)
                # Update Max efficiently
                M_new = np.max(X_ak_new, axis=1)
                M_ak = np.maximum(M_ak, M_new)
                prev_n_ak = target_n_bg
            
            # 3. Compute Tail
            mask_tail = ~mask_body
            if np.any(mask_tail):
                # IMPROVEMENT: Align tail scan with actual body end to prevent gaps
                s_start = body_vals[-1] 
                scan_thresh = np.logspace(np.log10(s_start), np.log10(s_start*100), 100)
                
                scan_probs = []
                for t in scan_thresh:
                    req = np.maximum(t - S_ak, M_ak)
                    pt = np.mean(stats.lognorm.sf(req, s=sig_natural, scale=1.0) * n_val)
                    scan_probs.append(1.0 - pt)
                
                # Ensure monotonicity for interp
                scan_probs = np.array(scan_probs)
                scan_thresh = np.array(scan_thresh)
                
                # Sort just in case MC noise broke order (rare but possible)
                sort_idx = np.argsort(scan_probs)
                tail_vals = np.interp(p_calc[mask_tail], scan_probs[sort_idx], scan_thresh[sort_idx])            
            else:
                tail_vals = np.array([])
                
            # 4. Store Result
            full_curve = np.concatenate([body_vals, tail_vals])
            log_curve = np.log(full_curve)
            master_table[i, j, :] = np.interp(p_equivalent, p_calc, log_curve)
            
            # Validate: check for NaN/Inf
            if np.any(np.isnan(master_table[i, j, :])):
                raise ValueError(f"NaN in table at (i={i}, j={j}, sigma={sig_dex:.2f})")
            if np.any(np.isinf(master_table[i, j, :])):
                raise ValueError(f"Inf in table at (i={i}, j={j}, sigma={sig_dex:.2f})")
            
            # Validate: check z-monotonicity (should be increasing)
            z_diff = np.diff(master_table[i, j, :])
            if np.any(z_diff < -1e-6):  # Allow tiny numerical noise
                non_mono_count = np.sum(z_diff < -1e-6)
                print(f"  ⚠ Warning: {non_mono_count} non-monotonic entries at (i={i}, j={j})")
        
        
        print(f"  > Done in {time.time()-t0:.1f}s")    
    
    print("Build Complete.")
    
    return master_table, n_grid_vals, sigma_grid_dex, z_grid


# ==============================================================================
# 2. THE SAVER
# ==============================================================================

def save_3d_sampler(filename, table, n_vals, s_grid, z_grid, source_dir=None):
    """
    Save the precomputed sampler table and grids to a compressed .npz file.
    
    Parameters
    ----------
    filename : str
        Name of the output file (e.g., 'sampler.npz'). Will be saved in
        the 'transfer_functions' subdirectory of the output path.
    table : ndarray
        The 3D master table from build_3d_sampler_arrays().
    n_vals : ndarray
        The N grid values (axis 0 of table).
    s_grid : ndarray
        The sigma grid values in dex (axis 1 of table).
    z_grid : ndarray
        The z grid values (axis 2 of table).
    source_dir : str, optional
        Source directory identifier (default: DEFAULT_SOURCE_DIR).
    """
    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    path_out = get_output_path(source=source_dir)
    if not os.path.exists(os.path.join(path_out, "transfer_functions")):
        os.makedirs(os.path.join(path_out, "transfer_functions"))
    
    path_file = os.path.join(path_out, "transfer_functions", filename)
    np.savez_compressed(path_file, table=table, n_vals=n_vals, s_grid=s_grid, z_grid=z_grid)
    print(f"Saved to {path_file}")



# ==============================================================================
# 3. THE LOADER
# ==============================================================================
def load_3d_sampler(filename, source_dir=None):
    """
    Load a precomputed sampler table and return a fast sampling function.
    
    This function loads the 3D lookup table from disk, precomputes grid
    constants for O(1) index lookups, and returns a closure that performs
    trilinear interpolation to sample from the lognormal sum distribution.
    
    Parameters
    ----------
    filename : str
        Name of the .npz file to load (e.g., 'sampler.npz').
    
    Returns
    -------
    sampler : callable
        A function with signature:
            sampler(n_arr, mu_dex_arr, sigma_dex_arr, rng) -> ndarray
        
        Parameters of the returned sampler:
            n_arr : int or array-like
                Number of lognormal variables to sum. If array, all elements
                must be the same value (optimized for batched calls).
            mu_dex_arr : array-like
                Mean of each lognormal variable in dex (log10 scale).
                The final sum is scaled by 10**mu_dex.
            sigma_dex_arr : array-like
                Scatter of each lognormal variable in dex.
            rng : numpy.random.Generator
                Random number generator for reproducibility.
        
        Returns:
            ndarray: Samples of S = (sum of N lognormals) * 10**mu_dex.
    
    Notes
    -----
    The sampler uses trilinear interpolation over (N, sigma, z) where
    z = -ln(U) and U ~ Uniform(0,1). This avoids expensive per-sample
    Monte Carlo and enables vectorized sampling of millions of values.
    """
    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    path_out = get_output_path(source=source_dir)
    path_file = os.path.join(path_out, "transfer_functions", filename)

    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File {filename} not found.")
        
    print(f"Loading sampler from {filename}...")
    data = np.load(path_file)
    table = data['table']      
    n_vals = data['n_vals']    
    s_grid = data['s_grid']
    z_grid = data['z_grid']
    
    # --- PRE-COMPUTE GRID CONSTANTS ---
    # 1. N Grid (Log-spaced, irregular)
    n_log_vals = np.log10(n_vals)
    n_len = len(n_vals)
    
    # 2. Sigma Grid (Linear) - Pre-calculate step for arithmetic lookup
    s_min = s_grid[0]
    s_step = s_grid[1] - s_grid[0]
    s_inv_step = 1.0 / s_step
    s_len = len(s_grid)
    
    # 3. Z Grid (Linear)
    z_min = z_grid[0]
    z_step = z_grid[1] - z_grid[0]
    z_inv_step = 1.0 / z_step
    z_len = len(z_grid)

    def sampler(n_arr, mu_dex_arr, sigma_dex_arr, rng, check_bounds=False):
        """
        Sample from the lognormal sum distribution via table interpolation.
        
        Draws samples of S = (sum_{k=1}^{N} X_k) * 10**mu_dex, where each
        X_k ~ Lognormal(0, sigma_nat) with sigma_nat = sigma_dex * ln(10).
        
        The algorithm:
            1. Draw U ~ Uniform(0,1), compute z = -ln(U)
            2. Look up log(S) in the precomputed table via trilinear interpolation
            3. Return exp(log(S)) * 10**mu_dex
        
        Parameters
        ----------
        n_arr : int or array-like
            Number of lognormals to sum. Must be a single value (scalar or
            array with identical elements) for the current optimized implementation.
        mu_dex_arr : array-like, shape (n_samples,)
            Mean offset in dex. The final result is scaled by 10**mu_dex.
        sigma_dex_arr : array-like, shape (n_samples,)
            Scatter in dex for each sample.
        rng : numpy.random.Generator
            Random number generator.
        check_bounds : bool, optional
            If True, warn when inputs are outside the precomputed grid.
            Disabled by default for performance on large arrays.
        
        Returns
        -------
        samples : ndarray, shape (n_samples,)
            Sampled values of the lognormal sum.
        """
        # Ensure inputs are arrays
        mu_dex_arr = np.asarray(mu_dex_arr)
        sigma_dex_arr = np.asarray(sigma_dex_arr)
        
        # --- OPTIONAL BOUNDS CHECKING ---
        # Disabled by default for performance (np.any is O(N) on large arrays)
        if check_bounds:
            import warnings
            sigma_min, sigma_max = s_grid[0], s_grid[-1]
            if np.any(sigma_dex_arr < sigma_min) or np.any(sigma_dex_arr > sigma_max):
                warnings.warn(
                    f"sigma_dex values outside grid [{sigma_min}, {sigma_max}]. "
                    "Results will be extrapolated and may be inaccurate.",
                    RuntimeWarning
                )
            n_min_grid, n_max_grid = n_vals[0], n_vals[-1]
            n_scalar = np.atleast_1d(n_arr)[0]
            if n_scalar < n_min_grid or n_scalar > n_max_grid:
                warnings.warn(
                    f"n={n_scalar} outside grid [{n_min_grid}, {n_max_grid}]. "
                    "Results will be extrapolated and may be inaccurate.",
                    RuntimeWarning
                )
        
        # --- OPTIMIZATION: SCALAR N CHECK ---
        # In process_evolution, n_arr is passed as [N] or scalar N.
        # We assume all BHs in this call share the same Age (N).
        # This allows us to slice the table ONCE, avoiding 3D advanced indexing.
        
        n_input_scalar = np.atleast_1d(n_arr)[0]
        
        # 1. FIND N INDICES (Once per chunk)
        #    Map log10(N) to the grid indices
        n_val_log = np.log10(n_input_scalar)
        
        # Binary search for N (fast since n_vals is small, <1000)
        n_idx = np.searchsorted(n_log_vals, n_val_log) - 1
        n_idx = np.clip(n_idx, 0, n_len - 2)
        
        # Calculate N weight
        n_left_log = n_log_vals[n_idx]
        n_right_log = n_log_vals[n_idx + 1]
        n_w = (n_val_log - n_left_log) / (n_right_log - n_left_log)
        n_w = np.clip(n_w, 0.0, 1.0)
        
        # 2. SLICE THE TABLE (The Big Speedup)
        #    We extract the two relevant 2D planes (Sigma vs Z)
        #    Now we only do lookups on these smaller 2D arrays.
        plane_low = table[n_idx]      # Shape (s_len, z_len)
        plane_high = table[n_idx + 1] # Shape (s_len, z_len)
        
        # --- VECTORIZED 2D INTERPOLATION ---
        
        # 3. Calculate Sigma Indices (Arithmetic is faster than np.interp)
        #    s_idx = floor((val - min) / step)
        s_float = (sigma_dex_arr - s_min) * s_inv_step
        s_idx = np.floor(s_float).astype(np.int32)
        s_idx = np.clip(s_idx, 0, s_len - 2)
        s_w = s_float - s_idx
        np.clip(s_w, 0.0, 1.0, out=s_w)
        
        # 4. Calculate Z Indices
        #    Generate Z from uniform U
        u = rng.uniform(0, 1, len(mu_dex_arr))
        np.maximum(u, 1e-10, out=u) # Avoid log(0)
        
        #    z = -ln(1-u). Note: u ~ U[0,1] => (1-u) ~ U[0,1]
        z_vals = -np.log(u) # Simplified math, equivalent distribution
        
        z_float = (z_vals - z_min) * z_inv_step
        z_idx = np.floor(z_float).astype(np.int32)
        z_idx = np.clip(z_idx, 0, z_len - 2)
        z_w = z_float - z_idx
        np.clip(z_w, 0.0, 1.0, out=z_w)
        
        # 5. Advanced Indexing on 2D Planes
        #    This is much faster than 3D indexing.
        #    We need values at (s, z), (s, z+1), (s+1, z), (s+1, z+1)
        
        # Pre-calculate index arrays for the 4 corners
        r0 = s_idx
        r1 = s_idx + 1
        c0 = z_idx
        c1 = z_idx + 1
        
        # Fetch corners from Plane Low (N)
        v00 = plane_low[r0, c0]
        v01 = plane_low[r0, c1]
        v10 = plane_low[r1, c0]
        v11 = plane_low[r1, c1]
        
        # Bilinear Interpolate Plane Low
        # i1 = lerp(v00, v01, z_w)
        i1 = v00 + z_w * (v01 - v00)
        i2 = v10 + z_w * (v11 - v10)
        val_low = i1 + s_w * (i2 - i1)
        
        # Fetch corners from Plane High (N+1)
        v00_h = plane_high[r0, c0]
        v01_h = plane_high[r0, c1]
        v10_h = plane_high[r1, c0]
        v11_h = plane_high[r1, c1]
        
        # Bilinear Interpolate Plane High
        i1_h = v00_h + z_w * (v01_h - v00_h)
        i2_h = v10_h + z_w * (v11_h - v10_h)
        val_high = i1_h + s_w * (i2_h - i1_h)
        
        # 6. Final Mix (N interpolation)
        log_sums = val_low + n_w * (val_high - val_low)
        
        return np.exp(log_sums) * (10.0 ** mu_dex_arr)
    
    return sampler



def verify_sampler(sampler, n_test, sig_dex, mu_dex, n_samples=100_000, seed=42):
    """
    Verify sampler accuracy by comparing to direct Monte Carlo.
    
    Generates samples from both the interpolated sampler and ground-truth
    direct summation, then compares their distributions visually and
    prints summary statistics (mean ratio).
    
    Parameters
    ----------
    sampler : callable
        The sampler function returned by load_3d_sampler().
    n_test : int
        Number of lognormal variables to sum.
    sig_dex : float
        Scatter in dex.
    mu_dex : float
        Mean offset in dex.
    n_samples : int
        Number of samples to draw for comparison (default: 100,000).
    seed : int
        Random seed for reproducibility (default: 42).
    """
    import matplotlib.pyplot as plt
    
    print(f"--- Verifying N={n_test}, Sigma={sig_dex} dex ---")
    
    # 1. Ground Truth (Hand Calculation)
    # Note: numpy.lognormal takes natural log mean/sigma
    sig_nat = sig_dex * np.log(10)
    mu_nat = mu_dex * np.log(10) # effectively mean of log(X)
    
    # Ground truth: Sum of N lognormals
    # Use separate RNG to ensure independent samples
    rng_truth = np.random.default_rng(seed)
    X_hand = rng_truth.lognormal(mean=0, sigma=sig_nat, size=(n_samples, n_test))
    S_hand = np.sum(X_hand, axis=1) * (10**mu_dex)
    
    # 2. Sampler (Interpolated) - use separate RNG
    rng_sampler = np.random.default_rng(seed + 1)
    S_samp = sampler(
        n_arr=np.full(n_samples, n_test), 
        mu_dex_arr=np.full(n_samples, mu_dex), 
        sigma_dex_arr=np.full(n_samples, sig_dex), 
        rng=rng_sampler
    )

    # 3. Compare Statistics
    print(f"Mean (Hand): {np.mean(S_hand):.4e}")
    print(f"Mean (Samp): {np.mean(S_samp):.4e}")
    print(f"Ratio:       {np.mean(S_samp)/np.mean(S_hand):.4f}")
    
    # 4. Plot
    plt.figure(figsize=(10, 6))
    bins = np.logspace(np.log10(np.min(S_hand)), np.log10(np.max(S_hand)), 100)
    
    plt.hist(S_hand, bins=bins, alpha=0.5, density=True, label='Hand Calc (Truth)', color='black')
    plt.hist(S_samp, bins=bins, alpha=0.5, density=True, label='Sampler (Interp)', color='orange')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Comparison: N={n_test}, $sigma$={sig_dex} dex")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.show()


# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    
    path_out = get_output_path(source=DEFAULT_SOURCE_DIR)

    SAMPLER_FILE = "Universal_Lognormal_Sampler_final.npz"
    path_file = os.path.join(path_out, "transfer_functions", SAMPLER_FILE)
    
    # PHASE 1: BUILD (If missing)
    if not os.path.exists(path_file):
        print("Initializing build...")
        table, n_grid, s_grid, z_grid = build_3d_sampler_arrays(
            n_min=5, 
            n_max=1000, 
            n_steps=200,           # Log-spaced grid
            sigma_dex_bounds=(0.1, 3.0), 
            sigma_steps=50,       
            z_res=50000,     
            n_naive_samples=5_000_000,
            seed=12345     
        )
        save_3d_sampler(SAMPLER_FILE, table, n_grid, s_grid, z_grid)

    # PHASE 2: RUN
    my_sampler = load_3d_sampler(SAMPLER_FILE)
    


    # TESTING AND PLOTTING
    # Test on a 10^7 dataset

    N_test = 25
    SIGMA_test_dex = 0.3
    MU_test_dex = 2.0
    print(f"Sampling 10^7 values for N={N_test}, Sigma={SIGMA_test_dex} dex, Mu={MU_test_dex} dex...")
    S_test = my_sampler(
        n_arr = np.full(10_000_000, N_test),
        mu_dex_arr = np.full(10_000_000, MU_test_dex),
        sigma_dex_arr = np.full(10_000_000, SIGMA_test_dex),
        rng = np.random.default_rng(42)
    )
    print("Sample complete.")

    # Verify accuracy against direct Monte Carlo in a plot
    verify_sampler(my_sampler, n_test=15, sig_dex=0.3, mu_dex=2.0)