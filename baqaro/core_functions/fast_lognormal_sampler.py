"""
Fast sampler for sums of lognormal random variables.
=====================================================

Drop-in replacement for ``build_lognormal_sampler.load_3d_sampler()``.

Background
----------
We need to draw samples from S = sum_{k=1}^{N} X_k, where each X_k is
Lognormal(mu, sigma). The distribution of S has no closed form, so we use
a precomputed 3D inverse-CDF lookup table indexed by (N, sigma, z), where
z = -ln(1-p) is the CDF percentile in a transformed coordinate.

This module replaces the original NumPy-based sampler with Numba JIT loops
that eliminate temporary array allocations.

Table layout
------------
The .npz file contains:
  - table[n_idx, s_idx, z_idx]  : log(S/mu) on a 3D grid, float32
  - n_vals   : N values (number of summands)
  - s_grid   : sigma_dex values (sigma of individual lognormals, in dex)
  - z_grid   : z = -ln(1-p) values (transformed CDF percentile)

For a given (N, sigma, z), trilinear interpolation gives log(S), then
out = exp(log_S) * 10^mu_dex.

Note: bh_accretion_fast.py does NOT call the sampler closure. It only
reads ``sampler._table_data`` and does its own fused table interpolation
inline within its Numba kernels. The closure is used by the original
(non-fused) code path in bh_accretion.py.

Fixed-sigma optimisation
------------------------
When sigma is the same for all samples (scalar or constant array), the
sampler pre-interpolates the sigma dimension once, collapsing the 2D
(sigma, z) bilinear lookup (8 table reads per sample) into a 1D z-only
linear lookup (2 reads per sample).
"""

import numpy as np
from numba import njit
import os
import time

from baqaro.utils.my_dir import get_output_path

# Site-driven, following the BAQARO_SOURCE_DIR convention used across the
# pipeline (e.g. emulation/main_training.py), so every caller that omits
# source_dir resolves the sampler table under the same site as the rest of
# the run. Defaults to machine_igm when unset.
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")


# ==========================================================================
#  2D KERNEL -- variable sigma per sample
# ==========================================================================

@njit(nogil=True)
def _sample_kernel_2d(plane_low, plane_high, n_w,
                      sigma_dex_arr, u_arr, mu_dex_arr,
                      s_min, s_inv_step, s_max_idx,
                      z_min, z_inv_step, z_max_idx,
                      out):
    """2D table kernel (serial, releases GIL).

    Bilinear interpolation in (sigma, z) on two bracketing N-planes,
    then linear interpolation between N-planes.
    8 table reads + 6 lerps + 1 exp per sample.
    """
    n = len(mu_dex_arr)

    for i in range(n):
        # --- Sigma grid index ---
        s_f = (sigma_dex_arr[i] - s_min) * s_inv_step
        s_idx = int(s_f)
        if s_idx < 0:
            s_idx = 0
        if s_idx > s_max_idx:
            s_idx = s_max_idx
        s_w = s_f - s_idx
        if s_w < 0.0:
            s_w = 0.0
        if s_w > 1.0:
            s_w = 1.0

        # --- Z grid index from uniform random ---
        u = u_arr[i]
        if u < 1e-10:
            u = 1e-10
        z = -np.log(u)

        z_f = (z - z_min) * z_inv_step
        z_idx = int(z_f)
        if z_idx < 0:
            z_idx = 0
        if z_idx > z_max_idx:
            z_idx = z_max_idx
        z_w = z_f - z_idx
        if z_w < 0.0:
            z_w = 0.0
        if z_w > 1.0:
            z_w = 1.0

        # --- Bilinear interpolation on N-low plane ---
        v00 = plane_low[s_idx,     z_idx]
        v01 = plane_low[s_idx,     z_idx + 1]
        v10 = plane_low[s_idx + 1, z_idx]
        v11 = plane_low[s_idx + 1, z_idx + 1]
        i1 = v00 + z_w * (v01 - v00)
        i2 = v10 + z_w * (v11 - v10)
        val_low = i1 + s_w * (i2 - i1)

        # --- Bilinear interpolation on N-high plane ---
        v00h = plane_high[s_idx,     z_idx]
        v01h = plane_high[s_idx,     z_idx + 1]
        v10h = plane_high[s_idx + 1, z_idx]
        v11h = plane_high[s_idx + 1, z_idx + 1]
        i1h = v00h + z_w * (v01h - v00h)
        i2h = v10h + z_w * (v11h - v10h)
        val_high = i1h + s_w * (i2h - i1h)

        # --- Interpolate between N-planes and convert ---
        log_sum = val_low + n_w * (val_high - val_low)
        out[i] = np.exp(log_sum) * (10.0 ** mu_dex_arr[i])


# ==========================================================================
#  1D KERNEL -- fixed sigma, pre-interpolated row
# ==========================================================================

@njit(nogil=True)
def _sample_kernel_1d(row_interp, u_arr, mu_dex_arr,
                      z_min, z_inv_step, z_max_idx,
                      out):
    """1D table kernel for fixed sigma (serial, releases GIL).

    ``row_interp`` is a 1D float64 array of length z_len, pre-interpolated
    across the sigma and N dimensions. Just 2 reads + 1 lerp per sample.
    """
    n = len(mu_dex_arr)

    for i in range(n):
        u = u_arr[i]
        if u < 1e-10:
            u = 1e-10
        z = -np.log(u)

        z_f = (z - z_min) * z_inv_step
        z_idx = int(z_f)
        if z_idx < 0:
            z_idx = 0
        if z_idx > z_max_idx:
            z_idx = z_max_idx
        z_w = z_f - z_idx
        if z_w < 0.0:
            z_w = 0.0
        if z_w > 1.0:
            z_w = 1.0

        log_sum = row_interp[z_idx] + z_w * (row_interp[z_idx + 1] - row_interp[z_idx])
        out[i] = np.exp(log_sum) * (10.0 ** mu_dex_arr[i])


# ==========================================================================
#  PRE-INTERPOLATION HELPER (for fixed-sigma 1D path)
# ==========================================================================

def _interpolate_sigma_row(plane_low, plane_high, n_w, sigma_dex,
                           s_min, s_inv_step, s_len):
    """Collapse the sigma and N dimensions into a single 1D z-row.

    Given two N-planes of shape (s_len, z_len) and a single sigma_dex value:
      1. Linearly interpolate between adjacent sigma rows in each N-plane
      2. Linearly interpolate between the two N-planes

    Returns a float64 array of shape (z_len,) ready for ``_sample_kernel_1d``.
    """
    s_f = (sigma_dex - s_min) * s_inv_step
    s_idx = int(s_f)
    s_idx = max(0, min(s_idx, s_len - 2))
    s_w = np.clip(s_f - s_idx, 0.0, 1.0)

    row_low = (plane_low[s_idx].astype(np.float64)
               + s_w * (plane_low[s_idx + 1].astype(np.float64)
                        - plane_low[s_idx].astype(np.float64)))
    row_high = (plane_high[s_idx].astype(np.float64)
                + s_w * (plane_high[s_idx + 1].astype(np.float64)
                         - plane_high[s_idx].astype(np.float64)))

    return row_low + n_w * (row_high - row_low)


# ==========================================================================
#  JIT WARMUP
# ==========================================================================

def _warmup_kernels():
    """Compile kernels with tiny dummy arrays."""
    t0 = time.time()
    _d_plane = np.zeros((2, 2), dtype=np.float32)
    _d_arr = np.zeros(1, dtype=np.float64)
    _d_out = np.zeros(1, dtype=np.float64)
    _d_row = np.zeros(2, dtype=np.float64)
    _sample_kernel_2d(_d_plane, _d_plane, 0.5,
                      _d_arr, np.array([0.5]), _d_arr,
                      0.0, 1.0, 0, 0.0, 1.0, 0, _d_out)
    _sample_kernel_1d(_d_row, np.array([0.5]), _d_arr,
                      0.0, 1.0, 0, _d_out)
    print(f"  Sampler kernels JIT-compiled in {time.time() - t0:.2f}s")

_warmup_kernels()


# ==========================================================================
#  PUBLIC API: load_3d_sampler()
# ==========================================================================

def load_3d_sampler(filename, source_dir=None, **kwargs):
    """Load a precomputed table and return a Numba-accelerated sampler closure.

    Parameters
    ----------
    filename : str
        Name of the ``.npz`` file (same format as ``build_lognormal_sampler``).
    source_dir : str, optional
        Source directory identifier (default: ``DEFAULT_SOURCE_DIR``).
    **kwargs
        Accepted for backward compatibility (``parallel``, ``fw_p_threshold``)
        but ignored — all sampling uses serial nogil kernels with pure table
        lookup. Parallelization is handled by the caller (bh_accretion_fast
        uses its own fused prange kernels).

    Returns
    -------
    sampler : callable
        ``sampler(n_arr, mu_dex_arr, sigma_dex_arr, rng) -> ndarray``
    """
    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    path_out = get_output_path(source=source_dir)
    path_file = os.path.join(path_out, "transfer_functions", filename)

    if not os.path.exists(path_file):
        raise FileNotFoundError(f"File {filename} not found at {path_file}")

    print(f"Loading fast sampler from {filename}...")
    t0 = time.time()

    data = np.load(path_file)
    table = data['table']    # shape (n_len, s_len, z_len), float32
    n_vals = data['n_vals']  # shape (n_len,)
    s_grid = data['s_grid']  # shape (s_len,)
    z_grid = data['z_grid']  # shape (z_len,)

    # Precompute grid constants for index arithmetic
    n_log_vals = np.log10(n_vals).astype(np.float64)
    n_len = len(n_vals)

    s_min = float(s_grid[0])
    s_inv_step = 1.0 / float(s_grid[1] - s_grid[0])
    s_len = len(s_grid)

    z_min = float(z_grid[0])
    z_inv_step = 1.0 / float(z_grid[1] - z_grid[0])
    z_len = len(z_grid)

    print(f"  Ready in {time.time() - t0:.2f}s")

    # Cache for pre-interpolated 1D rows (fixed-sigma fast path)
    _row_cache = {}

    def sampler(n_arr, mu_dex_arr, sigma_dex_arr, rng, check_bounds=False):
        """Draw samples from the sum-of-lognormals distribution.

        Automatically selects the fastest kernel:
          - Fixed sigma  -> 1D kernel (2 reads/sample)
          - Variable sigma -> 2D table kernel (8 reads/sample)
        """
        mu_dex_arr = np.asarray(mu_dex_arr, dtype=np.float64)
        sigma_dex_arr = np.asarray(sigma_dex_arr, dtype=np.float64)
        n_samples = len(mu_dex_arr)

        # N index (scalar, same for all samples in one call)
        n_scalar = int(np.atleast_1d(n_arr)[0])
        n_val_log = np.log10(n_scalar)

        n_idx = int(np.searchsorted(n_log_vals, n_val_log) - 1)
        n_idx = max(0, min(n_idx, n_len - 2))

        n_left_log = n_log_vals[n_idx]
        n_right_log = n_log_vals[n_idx + 1]
        n_w = float(np.clip(
            (n_val_log - n_left_log) / (n_right_log - n_left_log),
            0.0, 1.0))

        plane_low  = table[n_idx]
        plane_high = table[n_idx + 1]

        u = rng.uniform(0.0, 1.0, n_samples)
        out = np.empty(n_samples, dtype=np.float64)

        # Detect fixed sigma -> use fast 1D path
        sigma_is_fixed = (sigma_dex_arr.ndim == 0
                          or len(sigma_dex_arr) == 1
                          or (sigma_dex_arr[0] == sigma_dex_arr[-1]
                              and np.all(sigma_dex_arr == sigma_dex_arr[0])))

        if sigma_is_fixed:
            sigma_val = float(sigma_dex_arr.flat[0])
            cache_key = (n_idx, round(n_w, 10), round(sigma_val, 10))

            if cache_key in _row_cache:
                row_interp = _row_cache[cache_key]
            else:
                row_interp = _interpolate_sigma_row(
                    plane_low, plane_high, n_w, sigma_val,
                    s_min, s_inv_step, s_len)
                _row_cache[cache_key] = row_interp

            _sample_kernel_1d(row_interp, u, mu_dex_arr,
                              z_min, z_inv_step, z_len - 2,
                              out)
        else:
            _sample_kernel_2d(plane_low, plane_high, n_w,
                              sigma_dex_arr, u, mu_dex_arr,
                              s_min, s_inv_step, s_len - 2,
                              z_min, z_inv_step, z_len - 2,
                              out)

        return out

    # Expose table metadata for fused kernels in bh_accretion_fast.py
    sampler._table_data = {
        'table': table,
        'n_log_vals': n_log_vals,
        'n_len': n_len,
        's_grid': s_grid,
        's_min': s_min,
        's_inv_step': s_inv_step,
        's_len': s_len,
        'z_min': z_min,
        'z_inv_step': z_inv_step,
        'z_len': z_len,
    }

    return sampler
