"""
Fast fused BH accretion evolution — Numba drop-in for `bh_accretion.py`.
=============================================================================

Usage::

    from baqaro.core_functions.bh_accretion_fast import (
        evolve_BHs_fast as evolve_BHs)

Physical model
--------------
Each BH grows over ``n_steps`` sub-timesteps within one snapshot interval.
At each sub-step *j*, an Eddington ratio ``eta_j`` is drawn from a
lognormal ERDF:  ``log10(eta_j) ~ Normal(mu_i, sigma_i)``, where mu_i and
sigma_i depend on the halo's cold gas specific accretion rate.

The cumulative growth over n_steps is:

    M_new = M_old * exp( sum_j [ eta_j * f(eps_j) * growth_const ] )

where ``growth_const = (1-eps)/eps / c^2 * 10^log_csi * (ls/ms) * dt_substep``
and ``f(eps_j)`` accounts for the Madau+ spin-dependent radiative efficiency
when ``rad_efficiency_model="madau+"``. The ``ls/ms`` factor converts the
Eddington factor ``10^log_csi`` (Lsun/Msun, same units as L_bol) into a cgs
mass-growth rate; it keeps the mass build-up consistent with L_bol so an
Eddington (eta=1) accretor e-folds on the correct Salpeter time (~45 Myr).

The final bolometric luminosity uses only the *last* sub-step's eta:

    L_bol = M_new * eta_last * (eps_last / eps_base) * 10^log_csi

Two strategies compute the growth sum:

**Branch A** (n_steps < 6 or no transfer function):
    Direct sampling — draw all n_steps standard normals per halo, loop
    over sub-steps and accumulate growth. Simple but requires a 2D random
    array of shape (n_halos, n_steps).

**Branch B** (n_steps >= 6 with transfer function):
    Transfer function path — the sum of (n_steps-1) lognormals is drawn
    in one shot from a precomputed 3D inverse-CDF table indexed by
    (N, sigma_dex, z=-ln(u)). Only the *last* sub-step is sampled
    directly (needed for L_bol). This avoids the O(n_steps) inner loop.
    The table floor is N=5 summands, so Branch B requires
    n_steps - 1 >= 5; dispatching n_steps == 5 here would clamp the
    N=4 lookup to the N=5 plane and over-grow the mean by 6/5.

    Within Branch B, sigma determines the lookup strategy:

    **B-1D** (fixed sigma — all halos share one sigma value):
        The sigma and N dimensions are pre-interpolated into a single 1D
        row of shape (z_len,). Each halo needs just 2 table reads + 1 lerp.
        This is the fast path for ``log_normal_evol_halo_mass`` ERDF model.

    **B-2D** (variable sigma — sigma differs per halo):
        Full bilinear interpolation in (sigma, z) on two N-bracketing
        planes. 8 table reads + 6 lerps per halo.
        Used by ``log_normal_evol_halo_mass_dd`` ERDF model.

    **Hybrid FW approx** (fw_p_threshold > 0, B-2D only):
        For the bulk of the distribution (bottom 70% by default), the
        sum-of-lognormals is approximated via Fenton-Wilkinson moment
        matching as a single lognormal with precomputed (mu_S, sigma_S)
        grids. Cost: 4 L1-cached reads + 1 exp. The tail (top 30%) uses
        the exact table for QLF bright-end accuracy. Controlled by
        ``fw_p_threshold`` (0 = pure table, 0.7 = 70% approx).

Kernel variants
---------------
Each branch (A, B-2D, B-1D) has two kernel variants:
- ``_serial`` (nogil=True): releases the GIL, used by joblib threading workers
- ``_parallel`` (parallel=True): uses Numba prange, for single-process mode

The caller selects via ``parallelize`` (joblib) vs ``use_parallel_kernel``
(prange). Both achieve multi-core execution; prange has lower overhead.

Pre-allocated buffers
---------------------
The caller (main_evolution.py) can pass pre-allocated buffers to avoid
per-snapshot allocation:
- ``L_bols_out``: float64 array for luminosity output
- ``rng_buffers``: dict with 'z_last' (float32), 'u' (float64),
  'z_approx' (float32) — filled in-place via NumPy RNG ``out=`` parameter

Safety clamps
-------------
- ``growth > growth_max`` clamped to ``growth_max`` (default 50 ⇒ exp(50) ~
  5e21, finite; finite-but-astronomical guard, the standard production /
  fiducial setting). Lower it via the ``growth_max`` parameter of
  ``evolve_BHs_fast`` (or the ``BAQARO_GROWTH_SUM_MAX`` env var on the
  ``main_evolution`` entry point) to bound the per-snapshot mass-growth
  factor at ``exp(growth_max)`` — a numerical guardrail mode that trims the
  worst stochastic chains while leaving normal super-Eddington physics
  untouched. See ``main_evolution.py`` for the env-knob workflow.
- ``m_new > 1e15`` clamped to 1e15 M_sun
- ``lbol > 1e20`` clamped to 1e20
- Radiative efficiency ``eps`` clamped to [0, 0.999]
"""

import os
import time
import numpy as np
from scipy.special import ndtri          # inverse standard-normal CDF
from numba import njit, prange
from joblib import Parallel, delayed
from multiprocessing import cpu_count

import qhtools.utils.natconst as nc
from .madau_feff import feff_per_halo

# --- Madau effective-efficiency correction (the f_eff correction) ---
# Branch B applies eps once at the median eta; under madau+ that under-grows the
# bright end. When enabled, mu_scale is multiplied by f_eff(mu,sigma) to restore
# the per-sub-step mean (see madau_feff.py). OFF by
# default so existing runs are bit-identical; toggle with the env var or the
# evolve_BHs_fast(madau_feff_correction=...) argument.
_MADAU_FEFF_ENV = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION", "0").strip().lower()
                   not in ("0", "", "false", "no", "off"))


# ======================================================================
# Physical constants (precomputed at import time)
# ======================================================================

# Eddington luminosity factor: L_Edd = 10^log_csi * M_BH (in code units)
_FACTOR_CSI = np.float64(10.0 ** nc.log_csi)

# Madau+ spin-dependent radiative efficiency fit (Madau et al. 2014, Eq. 2).
# eps(eta) = eps_base * _MA / eta * (0.985 / (1.6/eta + _MB) + 0.015 / (1.6/eta + _MC))
# These coefficients are precomputed from the fitting formula.
_MA = np.float64(1.8260922439282448)
_MB = np.float64(0.7586284639465684)
_MC = np.float64(0.01611606486191978)


# ======================================================================
# Branch A: direct sub-step sampling (n_steps < 6 or no transfer function)
# ======================================================================
# Used when n_steps is small enough that drawing all standard normals
# explicitly is cheaper than the transfer function table lookup — and
# REQUIRED for n_steps <= 5: the transfer table's floor is N=5 summands,
# so Branch B's history lookup (N = n_steps-1) is only valid for
# n_steps >= 6.
#
# Input Z has shape (n_halos, n_steps). For each halo i, the kernel loops
# over sub-steps j=0..n_steps-1, draws eta_j = 10^(mu_i + sigma_i * Z[i,j]),
# accumulates growth, then applies mass/luminosity clamps.

@njit(nogil=True)
def _fused_branch_a_serial(
    Z, mus, sigmas, M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    L_bols, growth_max,
):
    """Branch A kernel — serial (nogil), used by joblib threading workers.

    ``growth_max`` is the upper clamp on the dimensionless integrated growth
    exponent (``growth_sum``) per snapshot. Fiducial = 50.0 (=> exp(50) ~ 5e21,
    pure numerical overflow guard, no physical effect). Smaller values impose
    a per-snapshot growth ceiling: 4.6 -> max 100x per snap, 7 -> max ~1100x,
    etc. See ``evolve_BHs_fast`` for the env knob.
    """
    LN10 = 2.302585092994046  # ln(10), local for Numba efficiency
    n = Z.shape[0]
    n_steps = Z.shape[1]

    for i in range(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])
        growth = 0.0
        eta_last = 0.0
        eps_last = rad_eff

        # Loop over sub-steps, accumulating dimensionless growth exponent
        for j in range(n_steps):
            # Draw Eddington ratio: eta = 10^(mu + sigma * z)
            eta = np.exp((mu_i + sig_i * float(Z[i, j])) * LN10)
            if use_madau:
                # Madau+ spin-dependent radiative efficiency
                inv_eta = 1.0 / max(eta, 1e-20)
                t1 = 0.985 / (1.6 * inv_eta + mB)
                t2 = 0.015 / (1.6 * inv_eta + mC)
                eps = rad_eff * mA * inv_eta * (t1 + t2)
                if eps < 0.0:
                    eps = 0.0
                if eps > 0.999:
                    eps = 0.999
                growth += eta * (1.0 - eps) * inv_1_minus_rad_eff * growth_const
            else:
                growth += eta * growth_const
            # Save last sub-step's eta and eps for luminosity calculation
            if j == n_steps - 1:
                eta_last = eta
                if use_madau:
                    eps_last = eps

        # Apply safety clamps and compute final mass + luminosity
        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new
        # L_bol uses last sub-step's eta (instantaneous luminosity)
        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


@njit(parallel=True)
def _fused_branch_a_parallel(
    Z, mus, sigmas, M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    L_bols, growth_max,
):
    """Branch A kernel — parallel (prange), used in single-process mode."""
    LN10 = 2.302585092994046
    n = Z.shape[0]
    n_steps = Z.shape[1]

    for i in prange(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])
        growth = 0.0
        eta_last = 0.0
        eps_last = rad_eff

        for j in range(n_steps):
            eta = np.exp((mu_i + sig_i * float(Z[i, j])) * LN10)
            if use_madau:
                inv_eta = 1.0 / max(eta, 1e-20)
                t1 = 0.985 / (1.6 * inv_eta + mB)
                t2 = 0.015 / (1.6 * inv_eta + mC)
                eps = rad_eff * mA * inv_eta * (t1 + t2)
                if eps < 0.0:
                    eps = 0.0
                if eps > 0.999:
                    eps = 0.999
                growth += eta * (1.0 - eps) * inv_1_minus_rad_eff * growth_const
            else:
                growth += eta * growth_const
            if j == n_steps - 1:
                eta_last = eta
                if use_madau:
                    eps_last = eps

        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new
        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


# ======================================================================
# Branch C: analytical mean growth (n_steps == 0, independent draws)
# ======================================================================
# When coherence time is zero, all draws are independent. The proper
# tau->0 analytical limit of the lognormal sub-step engine accumulates
# the ARITHMETIC mean E[eta] (not the geometric mean 10^mu), because
#   sum_i c * eta_i  -->  N * c * E[eta]   as N -> infinity.
# For eta = 10^X with X ~ N(mu, sigma^2):
#   E[eta] = 10^mu * exp(0.5 * (ln10)^2 * sigma^2)
# (using the median 10^mu instead would undershoot the limit by
# exp(0.5 * (ln10)^2 * sigma^2)). Hence:
#   constant eps : growth = E[eta] * growth_const_total
#   madau+       : growth = E[eta(1-eps(eta))] * inv_1_minus_re * growth_const_total
#                       = (1 - eps(10^mu)) * E[eta] * f_eff * inv_1_minus_re * gc
# where f_eff is the same (mu,sigma) lookup used by Branch B's
# f_eff correction. f_eff -> 1 in the constant-eps
# limit, so the constant-eps formula is unchanged structurally.
#
# Luminosity is unchanged: a single random eta draw (one standard normal
# per halo) drives Lbol — represents the instantaneous accretion state.
#
# Legacy (geometric-mean) behaviour is retained for bit-identical
# reproduction of tau=0 runs made with that convention via
# BAQARO_BRANCH_C_LEGACY=1 (env read by evolve_BHs_fast).

@njit(nogil=True)
def _fused_branch_c_serial(
    Z_lbol, mus, sigmas, feff_arr, M_BHs,
    growth_const_total, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    L_bols, growth_max, use_arithmetic_mean,
):
    """Branch C kernel — serial (nogil). Analytical mean growth + single eta draw for Lbol.

    ``growth_max`` clamps the per-snapshot growth exponent (see Branch A doc).
    ``use_arithmetic_mean`` toggles the lognormal-mean correction (True =
    arithmetic mean E[eta] = 10^mu * exp(0.5*(ln10)^2*sigma^2); False = legacy
    geometric mean 10^mu). ``feff_arr`` is the per-halo madau f_eff (1.0 in
    constant-eps mode or when madau_feff_correction is off → no-op).
    """
    LN10 = 2.302585092994046
    LN10_SQ_HALF = 2.6516504294495533     # 0.5 * (ln 10)^2 — lognormal mean
    n = len(M_BHs)

    for i in range(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])

        # Median eta — always used for the (1-eps(10^mu)) factor under madau+
        # so the f_eff definition stays consistent: f_eff multiplies that exact
        # (1-eps(median)) prefactor.
        eta_med = np.exp(mu_i * LN10)

        # Growth amplitude eta: arithmetic mean (corrected) or median (legacy)
        if use_arithmetic_mean:
            eta_growth = np.exp(mu_i * LN10 + LN10_SQ_HALF * sig_i * sig_i)
        else:
            eta_growth = eta_med

        if use_madau:
            inv_eta = 1.0 / max(eta_med, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_med = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            feff_i = float(feff_arr[i])
            growth = eta_growth * (1.0 - eps_med) * inv_1_minus_rad_eff * feff_i * growth_const_total
        else:
            growth = eta_growth * growth_const_total

        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        # --- Lbol from a single random eta draw (unchanged) ---
        eta_lbol = np.exp((mu_i + sig_i * float(Z_lbol[i])) * LN10)
        eps_lbol = rad_eff
        if use_madau:
            inv_eta_l = 1.0 / max(eta_lbol, 1e-20)
            t1_l = 0.985 / (1.6 * inv_eta_l + mB)
            t2_l = 0.015 / (1.6 * inv_eta_l + mC)
            eps_lbol = rad_eff * mA * inv_eta_l * (t1_l + t2_l)
            if eps_lbol < 0.0:
                eps_lbol = 0.0
            if eps_lbol > 0.999:
                eps_lbol = 0.999
        lbol = m_new * eta_lbol * (eps_lbol / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


@njit(parallel=True)
def _fused_branch_c_parallel(
    Z_lbol, mus, sigmas, feff_arr, M_BHs,
    growth_const_total, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    L_bols, growth_max, use_arithmetic_mean,
):
    """Branch C kernel — parallel (prange). Analytical mean growth + single eta draw for Lbol."""
    LN10 = 2.302585092994046
    LN10_SQ_HALF = 2.6516504294495533
    n = len(M_BHs)

    for i in prange(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])

        eta_med = np.exp(mu_i * LN10)
        if use_arithmetic_mean:
            eta_growth = np.exp(mu_i * LN10 + LN10_SQ_HALF * sig_i * sig_i)
        else:
            eta_growth = eta_med

        if use_madau:
            inv_eta = 1.0 / max(eta_med, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_med = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            feff_i = float(feff_arr[i])
            growth = eta_growth * (1.0 - eps_med) * inv_1_minus_rad_eff * feff_i * growth_const_total
        else:
            growth = eta_growth * growth_const_total

        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        eta_lbol = np.exp((mu_i + sig_i * float(Z_lbol[i])) * LN10)
        eps_lbol = rad_eff
        if use_madau:
            inv_eta_l = 1.0 / max(eta_lbol, 1e-20)
            t1_l = 0.985 / (1.6 * inv_eta_l + mB)
            t2_l = 0.015 / (1.6 * inv_eta_l + mC)
            eps_lbol = rad_eff * mA * inv_eta_l * (t1_l + t2_l)
            if eps_lbol < 0.0:
                eps_lbol = 0.0
            if eps_lbol > 0.999:
                eps_lbol = 0.999
        lbol = m_new * eta_lbol * (eps_lbol / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


# ======================================================================
# Branch B-2D: transfer function path with variable sigma (n_steps >= 6)
# ======================================================================
# The sum of (n_steps - 1) lognormals is drawn from a 3D inverse-CDF
# table[N, sigma_dex, z] where z = -ln(u) and u ~ Uniform(0,1).
# Only the last sub-step is sampled directly (needed for L_bol).
#
# For each halo:
#   1. Sample eta_last from Normal(mu_i, sigma_i) — the final sub-step
#   2. Compute mu_scale = median(eta) * efficiency_correction — the
#      expected value of one sub-step (used to rescale the table output)
#   3. Look up the sum of (n_steps-1) iid lognormals from the table:
#      - Bilinear interpolation in (sigma, z) on two N-bracketing planes
#      - Linear interpolation between the two N planes
#      Result: history_val = exp(table_interp) * mu_scale
#   4. Total growth = (eta_last_contribution + history_val) * growth_const
#
# The FW hybrid path (fw_z_thresh > 0) replaces the table lookup with
# a cheap lognormal approximation for the bulk of the distribution
# (z < fw_z_thresh), using precomputed mu_S and sigma_S grids.
# When fw_z_thresh <= 0, the table is always used (zero overhead).

@njit(nogil=True)
def _fused_branch_b_serial(
    z_last_arr, u_arr,
    z_approx_arr,            # float32[N]: standard normal for approx path
    mus, sigmas,
    M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    plane_low, plane_high, n_w,
    s_min, s_inv_step, s_max_idx,
    z_min, z_inv_step, z_max_idx,
    mu_S_grid, sigma_S_grid, # float64[s_len]: precomputed FW params
    fw_z_thresh,              # float64: z threshold (0 = pure table)
    L_bols, growth_max,
    feff_arr,                 # float64[N]: madau f_eff per halo (1.0 = correction off)
):
    """Branch B-2D kernel — serial (nogil), used by joblib threading workers.

    Bilinear interpolation in (sigma, z) on two N-planes.
    8 table reads + 6 lerps + 1 exp per halo for the table path.

    ``growth_max`` clamps the per-snapshot growth exponent (see Branch A doc).
    """
    LN10 = 2.302585092994046
    n = len(M_BHs)

    for i in range(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])

        # === STEP 1: Sample the LAST sub-step directly ===
        # This is needed for L_bol (instantaneous luminosity at snapshot end).
        eta_last = np.exp((mu_i + sig_i * float(z_last_arr[i])) * LN10)

        if use_madau:
            inv_eta = 1.0 / max(eta_last, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_last = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_last < 0.0:
                eps_last = 0.0
            if eps_last > 0.999:
                eps_last = 0.999
            eff_eta_last = eta_last * (1.0 - eps_last) * inv_1_minus_rad_eff
        else:
            eps_last = rad_eff
            eff_eta_last = eta_last

        # Last step's contribution to total growth
        growth = eff_eta_last * growth_const

        # === STEP 2: Compute mu_scale for the history (n_steps-1) sum ===
        # mu_scale = median(eta) * efficiency_correction. The table stores
        # log(S/mu) where S = sum of lognormals with mu=0. We multiply by
        # mu_scale to shift to the correct mean.
        if use_madau:
            median_eta = np.exp(mu_i * LN10)
            inv_med = 1.0 / max(median_eta, 1e-20)
            t1m = 0.985 / (1.6 * inv_med + mB)
            t2m = 0.015 / (1.6 * inv_med + mC)
            eps_med = rad_eff * mA * inv_med * (t1m + t2m)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            mu_scale = median_eta * (1.0 - eps_med) * inv_1_minus_rad_eff
        else:
            mu_scale = np.exp(mu_i * LN10)
        # f_eff correction: restore the per-sub-step mean (feff_arr is
        # all-1.0 when the correction is off -> bit-identical to before).
        mu_scale *= feff_arr[i]

        # === STEP 3: Look up the sum of (n_steps-1) lognormals ===
        # Sigma grid index (reused by both FW approx and table paths)
        s_f = (sig_i - s_min) * s_inv_step
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

        # --- Transform uniform to CDF coordinate: z = -ln(u) ---
        u = u_arr[i]
        if u < 1e-10:
            u = 1e-10
        z = -np.log(u)

        if fw_z_thresh > 0.0 and z < fw_z_thresh:
            # --- FW APPROX path (bulk of distribution) ---
            # 4 L1-cached reads + 1 exp. Interpolate precomputed mu_S,
            # sigma_S from 1D grid, then draw from approximating lognormal.
            mu_S = mu_S_grid[s_idx] + s_w * (mu_S_grid[s_idx + 1] - mu_S_grid[s_idx])
            sig_S = sigma_S_grid[s_idx] + s_w * (sigma_S_grid[s_idx + 1] - sigma_S_grid[s_idx])
            history_val = np.exp(mu_S + sig_S * float(z_approx_arr[i])) * mu_scale
        else:
            # --- EXACT TABLE path (tail of distribution or pure-table mode) ---
            # 8 table reads + 6 lerps + 1 exp. Bilinear in (sigma, z)
            # on two N-planes, then linear between N-planes.
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

            v00 = plane_low[s_idx,     z_idx]
            v01 = plane_low[s_idx,     z_idx + 1]
            v10 = plane_low[s_idx + 1, z_idx]
            v11 = plane_low[s_idx + 1, z_idx + 1]
            i1 = v00 + z_w * (v01 - v00)
            i2 = v10 + z_w * (v11 - v10)
            val_low = i1 + s_w * (i2 - i1)

            v00h = plane_high[s_idx,     z_idx]
            v01h = plane_high[s_idx,     z_idx + 1]
            v10h = plane_high[s_idx + 1, z_idx]
            v11h = plane_high[s_idx + 1, z_idx + 1]
            i1h = v00h + z_w * (v01h - v00h)
            i2h = v10h + z_w * (v11h - v10h)
            val_high = i1h + s_w * (i2h - i1h)

            log_sum = val_low + n_w * (val_high - val_low)
            history_val = np.exp(log_sum) * mu_scale

        # Add history contribution to growth
        growth += history_val * growth_const

        # === STEP 4: Apply clamps, compute final mass and luminosity ===
        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


@njit(parallel=True)
def _fused_branch_b_parallel(
    z_last_arr, u_arr,
    z_approx_arr,
    mus, sigmas,
    M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    plane_low, plane_high, n_w,
    s_min, s_inv_step, s_max_idx,
    z_min, z_inv_step, z_max_idx,
    mu_S_grid, sigma_S_grid,
    fw_z_thresh,
    L_bols, growth_max,
    feff_arr,                 # float64[N]: madau f_eff per halo (1.0 = correction off)
):
    """Branch B-2D kernel — parallel (prange), used in single-process mode.

    ``growth_max`` clamps the per-snapshot growth exponent (see Branch A doc).
    """
    LN10 = 2.302585092994046
    n = len(M_BHs)

    for i in prange(n):
        mu_i = float(mus[i])
        sig_i = float(sigmas[i])

        eta_last = np.exp((mu_i + sig_i * float(z_last_arr[i])) * LN10)
        if use_madau:
            inv_eta = 1.0 / max(eta_last, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_last = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_last < 0.0:
                eps_last = 0.0
            if eps_last > 0.999:
                eps_last = 0.999
            eff_eta_last = eta_last * (1.0 - eps_last) * inv_1_minus_rad_eff
        else:
            eps_last = rad_eff
            eff_eta_last = eta_last

        growth = eff_eta_last * growth_const

        if use_madau:
            median_eta = np.exp(mu_i * LN10)
            inv_med = 1.0 / max(median_eta, 1e-20)
            t1m = 0.985 / (1.6 * inv_med + mB)
            t2m = 0.015 / (1.6 * inv_med + mC)
            eps_med = rad_eff * mA * inv_med * (t1m + t2m)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            mu_scale = median_eta * (1.0 - eps_med) * inv_1_minus_rad_eff
        else:
            mu_scale = np.exp(mu_i * LN10)
        # f_eff correction: restore the per-sub-step mean (feff_arr is
        # all-1.0 when the correction is off -> bit-identical to before).
        mu_scale *= feff_arr[i]

        s_f = (sig_i - s_min) * s_inv_step
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

        u = u_arr[i]
        if u < 1e-10:
            u = 1e-10
        z = -np.log(u)

        if fw_z_thresh > 0.0 and z < fw_z_thresh:
            mu_S = mu_S_grid[s_idx] + s_w * (mu_S_grid[s_idx + 1] - mu_S_grid[s_idx])
            sig_S = sigma_S_grid[s_idx] + s_w * (sigma_S_grid[s_idx + 1] - sigma_S_grid[s_idx])
            history_val = np.exp(mu_S + sig_S * float(z_approx_arr[i])) * mu_scale
        else:
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

            v00 = plane_low[s_idx,     z_idx]
            v01 = plane_low[s_idx,     z_idx + 1]
            v10 = plane_low[s_idx + 1, z_idx]
            v11 = plane_low[s_idx + 1, z_idx + 1]
            i1 = v00 + z_w * (v01 - v00)
            i2 = v10 + z_w * (v11 - v10)
            val_low = i1 + s_w * (i2 - i1)

            v00h = plane_high[s_idx,     z_idx]
            v01h = plane_high[s_idx,     z_idx + 1]
            v10h = plane_high[s_idx + 1, z_idx]
            v11h = plane_high[s_idx + 1, z_idx + 1]
            i1h = v00h + z_w * (v01h - v00h)
            i2h = v10h + z_w * (v11h - v10h)
            val_high = i1h + s_w * (i2h - i1h)

            log_sum = val_low + n_w * (val_high - val_low)
            history_val = np.exp(log_sum) * mu_scale

        growth += history_val * growth_const

        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


# ======================================================================
# Branch B-1D: fixed sigma — pre-interpolated row (2 reads + 1 lerp)
# ======================================================================
# When all halos share the same sigma (e.g. log_normal_evol_halo_mass
# model where sigma is a scalar), the sigma and N dimensions can be
# pre-interpolated once into a single 1D row of shape (z_len,).
# This collapses the (sigma, z) bilinear lookup into a simple 1D
# linear interpolation: just 2 reads + 1 lerp per halo, vs 8+6 for B-2D.
#
# The pre-interpolation is done by _preinterp_sigma_row() before the
# kernel call. The FW hybrid path is NOT used here because the 1D lookup
# is already very cheap.

@njit(nogil=True)
def _fused_branch_b_1d_serial(
    z_last_arr, u_arr,
    mus, sigma_fixed,
    M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    preinterp_row,            # float64[z_len]: pre-interpolated log-sum row
    z_min, z_inv_step, z_max_idx,
    L_bols, growth_max,
    feff_arr,                 # float64[N]: madau f_eff per halo (1.0 = correction off)
):
    """Branch B-1D kernel — serial (nogil), used by joblib threading workers.

    2 table reads + 1 lerp per halo (sigma and N already collapsed).

    ``growth_max`` clamps the per-snapshot growth exponent (see Branch A doc).
    """
    LN10 = 2.302585092994046
    n = len(M_BHs)
    sig_f = float(sigma_fixed)

    for i in range(n):
        mu_i = float(mus[i])

        # === LAST STEP ===
        eta_last = np.exp((mu_i + sig_f * float(z_last_arr[i])) * LN10)

        if use_madau:
            inv_eta = 1.0 / max(eta_last, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_last = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_last < 0.0:
                eps_last = 0.0
            if eps_last > 0.999:
                eps_last = 0.999
            eff_eta_last = eta_last * (1.0 - eps_last) * inv_1_minus_rad_eff
        else:
            eps_last = rad_eff
            eff_eta_last = eta_last

        growth = eff_eta_last * growth_const

        # === HISTORY: compute mu_scale ===
        if use_madau:
            median_eta = np.exp(mu_i * LN10)
            inv_med = 1.0 / max(median_eta, 1e-20)
            t1m = 0.985 / (1.6 * inv_med + mB)
            t2m = 0.015 / (1.6 * inv_med + mC)
            eps_med = rad_eff * mA * inv_med * (t1m + t2m)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            mu_scale = median_eta * (1.0 - eps_med) * inv_1_minus_rad_eff
        else:
            mu_scale = np.exp(mu_i * LN10)
        # f_eff correction: restore the per-sub-step mean (feff_arr is
        # all-1.0 when the correction is off -> bit-identical to before).
        mu_scale *= feff_arr[i]

        # --- 1D table lookup: 2 reads + 1 lerp ---
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

        log_sum = preinterp_row[z_idx] + z_w * (preinterp_row[z_idx + 1] - preinterp_row[z_idx])
        history_val = np.exp(log_sum) * mu_scale

        growth += history_val * growth_const

        # === MASS AND LUMINOSITY ===
        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


@njit(parallel=True)
def _fused_branch_b_1d_parallel(
    z_last_arr, u_arr,
    mus, sigma_fixed,
    M_BHs,
    growth_const, rad_eff, inv_1_minus_rad_eff, factor_csi,
    use_madau, mA, mB, mC,
    preinterp_row,
    z_min, z_inv_step, z_max_idx,
    L_bols, growth_max,
    feff_arr,                 # float64[N]: madau f_eff per halo (1.0 = correction off)
):
    """Branch B-1D kernel — parallel (prange), used in single-process mode.

    ``growth_max`` clamps the per-snapshot growth exponent (see Branch A doc).
    """
    LN10 = 2.302585092994046
    n = len(M_BHs)
    sig_f = float(sigma_fixed)

    for i in prange(n):
        mu_i = float(mus[i])

        eta_last = np.exp((mu_i + sig_f * float(z_last_arr[i])) * LN10)
        if use_madau:
            inv_eta = 1.0 / max(eta_last, 1e-20)
            t1 = 0.985 / (1.6 * inv_eta + mB)
            t2 = 0.015 / (1.6 * inv_eta + mC)
            eps_last = rad_eff * mA * inv_eta * (t1 + t2)
            if eps_last < 0.0:
                eps_last = 0.0
            if eps_last > 0.999:
                eps_last = 0.999
            eff_eta_last = eta_last * (1.0 - eps_last) * inv_1_minus_rad_eff
        else:
            eps_last = rad_eff
            eff_eta_last = eta_last

        growth = eff_eta_last * growth_const

        if use_madau:
            median_eta = np.exp(mu_i * LN10)
            inv_med = 1.0 / max(median_eta, 1e-20)
            t1m = 0.985 / (1.6 * inv_med + mB)
            t2m = 0.015 / (1.6 * inv_med + mC)
            eps_med = rad_eff * mA * inv_med * (t1m + t2m)
            if eps_med < 0.0:
                eps_med = 0.0
            if eps_med > 0.999:
                eps_med = 0.999
            mu_scale = median_eta * (1.0 - eps_med) * inv_1_minus_rad_eff
        else:
            mu_scale = np.exp(mu_i * LN10)
        # f_eff correction: restore the per-sub-step mean (feff_arr is
        # all-1.0 when the correction is off -> bit-identical to before).
        mu_scale *= feff_arr[i]

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

        log_sum = preinterp_row[z_idx] + z_w * (preinterp_row[z_idx + 1] - preinterp_row[z_idx])
        history_val = np.exp(log_sum) * mu_scale

        growth += history_val * growth_const

        if growth > growth_max:
            growth = growth_max
        m_new = float(M_BHs[i]) * np.exp(growth)
        if m_new > 1e15:
            m_new = 1e15
        M_BHs[i] = m_new

        lbol = m_new * eta_last * (eps_last / rad_eff) * factor_csi
        if lbol > 1e20:
            lbol = 1e20
        L_bols[i] = lbol


# ======================================================================
# JIT warmup (serial only — parallel kernels skipped for fork safety)
# ======================================================================
# Called at import time. Compiles serial kernels with tiny dummy data so
# the first real call doesn't pay the JIT cost. Parallel kernels are NOT
# warmed up here because they use prange, which spawns threads that are
# unsafe to create before a fork() in the training pipeline.

def _warmup_serial_kernels():
    """Compile serial kernels with tiny dummy arrays (called at import)."""
    t0 = time.time()
    _d1 = np.zeros((1, 1), dtype=np.float32)
    _d = np.zeros(1, dtype=np.float32)
    _df = np.zeros(1, dtype=np.float64)
    _out = np.zeros(1, dtype=np.float64)
    _plane = np.zeros((2, 2), dtype=np.float32)
    _grid = np.zeros(2, dtype=np.float64)
    _feff = np.ones(1, dtype=np.float64)   # madau f_eff dummy (1.0 = off)

    _fused_branch_a_serial(
        _d1, _d, _d, _d,
        1.0, 0.1, 1.0 / 0.9, 1e4,
        False, 1.0, 1.0, 1.0, _out, 50.0)
    _fused_branch_b_serial(
        _d, _df, _d,
        _d, _d, _d,
        1.0, 0.1, 1.0 / 0.9, 1e4,
        False, 1.0, 1.0, 1.0,
        _plane, _plane, 0.5,
        0.0, 1.0, 0, 0.0, 1.0, 0,
        _grid, _grid, 0.0,
        _out, 50.0, _feff)
    _fused_branch_b_1d_serial(
        _d, _df,
        _d, 0.5,
        _d,
        1.0, 0.1, 1.0 / 0.9, 1e4,
        False, 1.0, 1.0, 1.0,
        _grid, 0.0, 1.0, 0,
        _out, 50.0, _feff)
    print(f"  Fused kernels JIT-compiled in {time.time() - t0:.2f}s")

_warmup_serial_kernels()


# ======================================================================
# Precompute FW (Fenton-Wilkinson) 1D approximation tables
# ======================================================================
# The FW method approximates a sum of N iid Lognormal(0, sigma) variables
# as a single Lognormal(mu_S, sigma_S) by matching the first two moments.
# This is much cheaper than the full 3D table lookup (4 reads + 1 exp vs
# 8 reads + 6 lerps + 1 exp) and accurate for the bulk of the distribution.
# Only used in B-2D path when fw_p_threshold > 0.

def _precompute_fw_grids(s_grid, N):
    """Precompute FW moment-matched (mu_S, sigma_S) for each sigma_dex.

    Parameters
    ----------
    s_grid : ndarray
        sigma_dex grid values from the sampler table.
    N : int
        Number of summands (= n_steps - 1, the "history" sub-steps).

    Returns
    -------
    mu_S_grid, sigma_S_grid : ndarray (float64)
        FW parameters indexed by sigma_dex, same length as s_grid.
    """
    LN10 = np.log(10.0)
    sigma_nat = s_grid * LN10
    sigma_nat2 = sigma_nat ** 2
    exp_sn2 = np.exp(sigma_nat2)

    # FW variance of the approximating lognormal
    sigma_S2 = np.log(1.0 + (exp_sn2 - 1.0) / N)

    # FW mean (in log-space, relative to mu=0)
    mu_S = np.log(float(N)) + 0.5 * sigma_nat2 - 0.5 * sigma_S2

    sigma_S = np.sqrt(sigma_S2)

    return mu_S.astype(np.float64), sigma_S.astype(np.float64)


def _preinterp_sigma_row(plane_low, plane_high, n_w, sigma_fixed, td):
    """Pre-interpolate sigma and N dims into a 1D z-only row for B-1D.

    Called once per snapshot when sigma is fixed. For each z grid point,
    linearly interpolates the sigma dimension in both N-planes, then
    blends between the two N-planes.

    Returns float64 array of shape (z_len,) — ready for the B-1D kernel.
    """
    s_f = (sigma_fixed - td['s_min']) * td['s_inv_step']
    s_idx = int(s_f)
    s_idx = max(0, min(s_idx, td['s_len'] - 2))
    s_w = np.clip(s_f - s_idx, 0.0, 1.0)

    # Interpolate sigma dimension for both N-planes: shape (z_len,)
    row_low = plane_low[s_idx] + s_w * (plane_low[s_idx + 1] - plane_low[s_idx])
    row_high = plane_high[s_idx] + s_w * (plane_high[s_idx + 1] - plane_high[s_idx])

    # Blend N-planes
    row = row_low + n_w * (row_high - row_low)

    return np.ascontiguousarray(row, dtype=np.float64)


# ======================================================================
# Top-level API: evolve_BHs_fast
# ======================================================================
# This is the public entry point — drop-in replacement for
# bh_accretion.evolve_BHs. Called from main_evolution.py.
#
# Handles:
#   1. growth_const computation from physical parameters
#   2. ERDF param extraction (mu, sigma per halo)
#   3. RNG generation (with optional pre-allocated buffer reuse)
#   4. Transfer function table setup (N-index, plane selection)
#   5. Branch/kernel selection and dispatch
#   6. Optional joblib parallelization (slice-view, no array_split)

def evolve_BHs_fast(
    transfer_function, M_BHs, delta_t, erdf,
    rad_efficiency=0.1, n_steps=100,
    log_halo_rates_array=None, rad_efficiency_model="constant",
    parallelize=True, num_workers=None, backend="threading",
    # fw_p_threshold DEFAULT is 0.0 (exact table path). Every production
    # caller passes 0.0 explicitly (main_evolution, main_evolution_chunked,
    # run_single_model via fw_approx=False); the FW hybrid path is an
    # approximation a caller opts INTO, never inherits.
    use_parallel_kernel=False, fw_p_threshold=0.0,
    print_info=False, L_bols_out=None, rng_buffers=None,
    growth_max=50.0, madau_feff_correction=None,
):
    """Evolve BH masses over one snapshot interval. M_BHs modified in-place.

    Parameters
    ----------
    transfer_function : sampler object (from fast_lognormal_sampler)
        Must have ``._table_data`` dict with the 3D inverse-CDF table.
        Pass None to force Branch A (direct sampling).
    M_BHs : ndarray, float32
        BH masses at previous snapshot. Modified in-place to new masses.
    delta_t : float
        Snapshot time interval in Gyr.
    erdf : Erdf instance
        Provides ``.get_mu_sigma_arrays()`` and ``.rng``.
    rad_efficiency : float
        Base radiative efficiency epsilon (default 0.1).
    n_steps : int
        Number of sub-timesteps within this snapshot interval.
    log_halo_rates_array : ndarray or None
        log10(specific cold accretion rate) per halo. Determines ERDF mu.
    rad_efficiency_model : str
        "constant" or "madau+" (spin-dependent).
    parallelize : bool
        If True, use joblib threading with serial (nogil) kernels.
    num_workers : int or None
        Number of joblib workers (default: cpu_count).
    backend : str
        joblib backend ("threading", "loky", "multiprocessing").
    use_parallel_kernel : bool
        If True, use Numba prange kernels instead of joblib. Preferred
        for single-process runs (lower overhead than joblib).
    fw_p_threshold : float
        FW hybrid approximation fraction (0.0 = pure table, 0.7 = 70%
        approx). Only affects B-2D path.
    L_bols_out : ndarray (float64) or None
        Pre-allocated buffer for luminosity output. If None, allocated here.
    rng_buffers : dict or None
        Pre-allocated RNG buffers. Keys: 'z_last' (float32), 'u' (float64),
        'z_approx' (float32). Sliced to size, filled via RNG ``out=`` param.
        Avoids allocation + page faults after first snapshot.

    Returns
    -------
    M_BHs : ndarray
        Updated BH masses (same array as input, modified in-place).
    L_bols : ndarray (float64)
        Bolometric luminosities at the last sub-step.
    """
    if rad_efficiency_model not in ("constant", "madau+"):
        raise ValueError(f"Unknown rad_efficiency_model: {rad_efficiency_model}")

    if log_halo_rates_array is not None and len(log_halo_rates_array) != len(M_BHs):
        raise ValueError("log_halo_rates_array length must match M_BHs length")

    if len(M_BHs) == 0:
        return np.array([]), np.array([])

    size = len(M_BHs)

    # --- ERDF params -> per-halo mu, sigma (once, before parallel) ---
    mus, sigmas = erdf.get_mu_sigma_arrays(log_halo_rates_array, dtype=np.float32)
    sigma_is_fixed = (np.ndim(sigmas) == 0)
    if sigma_is_fixed:
        # NB: the fixed (scalar) sigma is deliberately NOT clipped to [0.1, 3.0],
        # unlike the per-halo array path below. The per-halo clip guards against
        # numerical zeros the rate-driven ERDF can emit; a user-chosen fixed
        # std_0 is exact, and clipping the lower edge would break the
        # deterministic std_0=0 Eddington-e-fold=Salpeter invariant (Branch C
        # arithmetic mean; tests/test_growth_const_units.py). Production std_0
        # prior [0.2, 1.0] is already inside the range, so this is moot there.
        sigma_fixed = float(sigmas)
    else:
        np.clip(sigmas, np.float32(0.1), np.float32(3.0), out=sigmas)

    use_madau = (rad_efficiency_model == "madau+")
    re = float(rad_efficiency)
    inv_1_minus_re = 1.0 / (1.0 - re)

    # --- Madau f_eff correction (Branch B only; ones => bit-identical no-op) ---
    # Resolve the toggle: explicit arg wins, else the BAQARO_MADAU_FEFF_CORRECTION
    # env default. Only meaningful under madau+ (f_eff == 1 for constant eps).
    feff_on = _MADAU_FEFF_ENV if madau_feff_correction is None else bool(madau_feff_correction)
    if feff_on and use_madau:
        feff_arr = feff_per_halo(mus, sigma_fixed if sigma_is_fixed else sigmas, re)
    else:
        feff_arr = np.ones(size, dtype=np.float64)

    if print_info:
        print(f"[bh_accretion_fast] n_steps={n_steps}, size={size}, "
              f"fw={fw_p_threshold}, model={rad_efficiency_model}, "
              f"sigma_fixed={sigma_is_fixed}, madau_feff={feff_on and use_madau}")

    time_here = time.time()

    # --- Precompute everything that needs the GIL (RNG, table setup) ---
    L_bols = L_bols_out if L_bols_out is not None else np.empty(size, dtype=np.float64)

    if n_steps == 0:
        # =============================================================
        # BRANCH C: analytical mean growth (independent draws)
        # =============================================================
        # All draws independent — no coherence. Growth uses the geometric
        # mean eta = 10^mu over the full delta_t analytically.
        # Lbol is computed from a single random eta draw.
        # nc.ls/nc.ms converts the Eddington factor 10^log_csi (in Lsun/Msun, the
        # same units as L_bol) into a cgs mass-growth rate: Lsun->erg/s (*ls),
        # g->Msun (/ms). Must stay consistent with the L_bol formula
        # `m_new * eta * (eps/eps_base) * 10^log_csi` (Lsun).
        growth_const_total = float(
            (1 - rad_efficiency) / rad_efficiency
            / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms
            * delta_t * nc.year * 1e9
        )

        t_rng = time.time()
        if rng_buffers is not None:
            Z_lbol = rng_buffers['z_last'][:size]
            erdf.rng.standard_normal(size, dtype=np.float32, out=Z_lbol)
        else:
            Z_lbol = erdf.rng.standard_normal(size).astype(np.float32)
        if sigma_is_fixed:
            sigmas_arr = np.full(size, sigma_fixed, dtype=np.float32)
        else:
            sigmas_arr = sigmas
        dt_rng = time.time() - t_rng

        # Branch C arithmetic-mean correction.
        # Default: ON (use E[eta] = 10^mu * exp(0.5*(ln10)^2*sigma^2)).
        # BAQARO_BRANCH_C_LEGACY=1 reverts to the legacy geometric-mean (10^mu)
        # path for bit-identical reproduction of runs made with that convention.
        _use_arith_mean = os.environ.get("BAQARO_BRANCH_C_LEGACY", "0") != "1"
        if print_info:
            print(f"  [C] arithmetic_mean={_use_arith_mean}  (f_eff applied={feff_on and use_madau})")

        if parallelize:
            num_workers = cpu_count() if num_workers is None else num_workers
            num_workers = min(num_workers, size)
            boundaries = np.linspace(0, size, num_workers + 1, dtype=np.int64)

            def _worker_c(w):
                s, e = int(boundaries[w]), int(boundaries[w + 1])
                if s == e:
                    return
                _fused_branch_c_serial(
                    Z_lbol[s:e], mus[s:e], sigmas_arr[s:e], feff_arr[s:e], M_BHs[s:e],
                    growth_const_total, re, inv_1_minus_re, _FACTOR_CSI,
                    use_madau, _MA, _MB, _MC,
                    L_bols[s:e], growth_max, _use_arith_mean)

            Parallel(n_jobs=num_workers, backend=backend)(
                delayed(_worker_c)(w) for w in range(num_workers))
        else:
            kernel = (_fused_branch_c_parallel if use_parallel_kernel
                      else _fused_branch_c_serial)
            kernel(Z_lbol, mus, sigmas_arr, feff_arr, M_BHs,
                   growth_const_total, re, inv_1_minus_re, _FACTOR_CSI,
                   use_madau, _MA, _MB, _MC,
                   L_bols, growth_max, _use_arith_mean)

        if print_info:
            print(f"  [C] rng={dt_rng:.4f}s, total={time.time() - time_here:.4f}s")

    elif n_steps < 6 or transfer_function is None:
        # =============================================================
        # BRANCH A: direct sub-step sampling
        # =============================================================
        # n_steps <= 5 MUST come here: the transfer table's floor is N=5
        # summands, so Branch B's history lookup (N = n_steps-1) is only
        # exact for n_steps >= 6 (an N=4 lookup would clamp to the N=5
        # plane and over-grow the mean by 6/5).
        # Draw all (size, n_steps) standard normals up front, then dispatch.
        # No buffer reuse: Z is 2D and n_steps varies between snapshots,
        # so pre-allocating max size would waste ~16 GB at 200M halos.
        timestep = delta_t / n_steps
        # * nc.ls / nc.ms: Lsun->erg/s, g->Msun unit conversion (see Branch C note).
        gc = float((1 - rad_efficiency) / rad_efficiency
                   / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms
                   * timestep * nc.year * 1e9)

        t_rng = time.time()
        Z = erdf.rng.standard_normal((size, n_steps)).astype(np.float32)
        if sigma_is_fixed:
            sigmas_arr = np.full(size, sigma_fixed, dtype=np.float32)
        else:
            sigmas_arr = sigmas
        dt_rng = time.time() - t_rng

        if parallelize:
            num_workers = cpu_count() if num_workers is None else num_workers
            num_workers = min(num_workers, size)
            boundaries = np.linspace(0, size, num_workers + 1, dtype=np.int64)

            def _worker_a(w):
                s, e = int(boundaries[w]), int(boundaries[w + 1])
                if s == e:
                    return
                _fused_branch_a_serial(
                    Z[s:e], mus[s:e], sigmas_arr[s:e], M_BHs[s:e],
                    gc, re, inv_1_minus_re, _FACTOR_CSI,
                    use_madau, _MA, _MB, _MC,
                    L_bols[s:e], growth_max)

            Parallel(n_jobs=num_workers, backend=backend)(
                delayed(_worker_a)(w) for w in range(num_workers))
        else:
            kernel = (_fused_branch_a_parallel if use_parallel_kernel
                      else _fused_branch_a_serial)
            kernel(Z, mus, sigmas_arr, M_BHs,
                   gc, re, inv_1_minus_re, _FACTOR_CSI,
                   use_madau, _MA, _MB, _MC,
                   L_bols, growth_max)

        if print_info:
            print(f"  [A] rng={dt_rng:.4f}s, total={time.time() - time_here:.4f}s")

    else:
        # =============================================================
        # BRANCH B: transfer function path (n_steps >= 6)
        # =============================================================
        # Draw z_last (last sub-step) and u (CDF percentile for history)
        # up front, then look up the sum of (n_steps-1) lognormals from
        # the precomputed 3D table.
        # --- Consistent sub-step cap ---
        # The transfer-function table spans at most n_max_table summands. For
        # n_steps-1 beyond that, the history-sum lookup SATURATES at n_max_table
        # terms; if growth_const still divides by the TRUE n_steps the mean
        # growth is suppressed by ~n_max_table/n_steps (the looked-up sum has
        # fewer terms than the per-step normalization assumes). We therefore cap
        # the EFFECTIVE sub-step count for BOTH the table lookup AND growth_const
        # so the mean growth stays n_steps-independent (the physical invariant:
        # finer sub-stepping cannot change the time-averaged accretion). Capping
        # at ~1000 sub-steps is a fine time-average; only the (already tiny)
        # growth SCATTER is mildly over-estimated past the cap. Regression test:
        # tests/test_substep_cap_growth.py.
        td = transfer_function._table_data
        n_max_table = int(round(10.0 ** float(td['n_log_vals'][-1])))
        n_eff = n_steps if n_steps <= n_max_table + 1 else n_max_table + 1

        timestep = delta_t / n_eff
        # * nc.ls / nc.ms: Lsun->erg/s, g->Msun unit conversion (see Branch C note).
        gc = float((1 - rad_efficiency) / rad_efficiency
                   / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms
                   * timestep * nc.year * 1e9)

        t_rng = time.time()
        if rng_buffers is not None:
            z_last = rng_buffers['z_last'][:size]
            u = rng_buffers['u'][:size]
            erdf.rng.standard_normal(size, dtype=np.float32, out=z_last)
            erdf.rng.random(size, out=u)
        else:
            z_last = erdf.rng.standard_normal(size).astype(np.float32)
            u = erdf.rng.uniform(0.0, 1.0, size)
        dt_rng = time.time() - t_rng

        n_scalar = n_eff - 1
        n_val_log = np.log10(float(n_scalar))
        n_idx = int(np.searchsorted(td['n_log_vals'], n_val_log) - 1)
        n_idx = max(0, min(n_idx, td['n_len'] - 2))
        n_left = td['n_log_vals'][n_idx]
        n_right = td['n_log_vals'][n_idx + 1]
        n_w = float(np.clip((n_val_log - n_left) / (n_right - n_left),
                            0.0, 1.0))
        plane_low = td['table'][n_idx]
        plane_high = td['table'][n_idx + 1]
        z_max_idx = td['z_len'] - 2

        if sigma_is_fixed:
            # --- B-1D: fixed sigma, pre-interpolate to 1D row ---
            t_pre = time.time()
            preinterp_row = _preinterp_sigma_row(
                plane_low, plane_high, n_w, sigma_fixed, td)
            dt_pre = time.time() - t_pre

            if parallelize:
                num_workers = cpu_count() if num_workers is None else num_workers
                num_workers = min(num_workers, size)
                boundaries = np.linspace(0, size, num_workers + 1, dtype=np.int64)

                def _worker_b1d(w):
                    s, e = int(boundaries[w]), int(boundaries[w + 1])
                    if s == e:
                        return
                    _fused_branch_b_1d_serial(
                        z_last[s:e], u[s:e],
                        mus[s:e], sigma_fixed,
                        M_BHs[s:e],
                        gc, re, inv_1_minus_re, _FACTOR_CSI,
                        use_madau, _MA, _MB, _MC,
                        preinterp_row,
                        td['z_min'], td['z_inv_step'], z_max_idx,
                        L_bols[s:e], growth_max, feff_arr[s:e])

                t_kern = time.time()
                Parallel(n_jobs=num_workers, backend=backend)(
                    delayed(_worker_b1d)(w) for w in range(num_workers))
                dt_kern = time.time() - t_kern
            else:
                kernel = (_fused_branch_b_1d_parallel if use_parallel_kernel
                          else _fused_branch_b_1d_serial)
                t_kern = time.time()
                kernel(z_last, u,
                       mus, sigma_fixed,
                       M_BHs,
                       gc, re, inv_1_minus_re, _FACTOR_CSI,
                       use_madau, _MA, _MB, _MC,
                       preinterp_row,
                       td['z_min'], td['z_inv_step'], z_max_idx,
                       L_bols, growth_max, feff_arr)
                dt_kern = time.time() - t_kern

            if print_info:
                print(f"  [B-1D] sigma={sigma_fixed:.4f}, row_len={len(preinterp_row)}, "
                      f"rng={dt_rng:.4f}s, preinterp={dt_pre:.4f}s, "
                      f"kernel={dt_kern:.4f}s, total={time.time() - time_here:.4f}s")

        else:
            # --- B-2D: variable sigma, full bilinear table lookup ---
            s_max_idx = td['s_len'] - 2

            t_fw = time.time()
            if fw_p_threshold > 0.0:
                fw_z_thresh = float(-np.log(1.0 - fw_p_threshold))
                mu_S_grid, sigma_S_grid = _precompute_fw_grids(
                    td['s_grid'], n_scalar)
                # The FW deviate MUST be the SAME quantile as the uniform `u`
                # that selected this branch — i.e. the FW lognormal evaluated at
                # CDF probability p = 1 - u, hence ndtri(1-u). A fresh,
                # INDEPENDENT standard normal here would make the bulk branch
                # sample from the UNCONDITIONAL FW lognormal instead of FW
                # *conditioned* on p < fw_p_threshold, i.e. a mixture of the
                # full FW law and the exact top tail, biased high in mean and
                # median and compounding over the snapshots.
                #
                # Deriving it from `u` also makes the hybrid path a strict
                # quantile-coupled approximation of the exact table: same u ->
                # same quantile, only the interpolation of the quantile function
                # differs between the two branches. No extra RNG is consumed.
                z_approx = ndtri(1.0 - np.asarray(u[:size], dtype=np.float64))
                np.clip(z_approx, -8.0, 8.0, out=z_approx)   # guard u -> {0,1}
                z_approx = z_approx.astype(np.float32)
            else:
                fw_z_thresh = 0.0
                mu_S_grid = np.empty(1, dtype=np.float64)
                sigma_S_grid = np.empty(1, dtype=np.float64)
                z_approx = np.empty(1, dtype=np.float32)
            dt_fw = time.time() - t_fw

            if parallelize:
                num_workers = cpu_count() if num_workers is None else num_workers
                num_workers = min(num_workers, size)
                boundaries = np.linspace(0, size, num_workers + 1, dtype=np.int64)

                def _worker_b2d(w):
                    s, e = int(boundaries[w]), int(boundaries[w + 1])
                    if s == e:
                        return
                    _fused_branch_b_serial(
                        z_last[s:e], u[s:e],
                        z_approx[s:e] if fw_p_threshold > 0.0 else z_approx,
                        mus[s:e], sigmas[s:e], M_BHs[s:e],
                        gc, re, inv_1_minus_re, _FACTOR_CSI,
                        use_madau, _MA, _MB, _MC,
                        plane_low, plane_high, n_w,
                        td['s_min'], td['s_inv_step'], s_max_idx,
                        td['z_min'], td['z_inv_step'], z_max_idx,
                        mu_S_grid, sigma_S_grid, fw_z_thresh,
                        L_bols[s:e], growth_max, feff_arr[s:e])

                t_kern = time.time()
                Parallel(n_jobs=num_workers, backend=backend)(
                    delayed(_worker_b2d)(w) for w in range(num_workers))
                dt_kern = time.time() - t_kern
            else:
                kernel = (_fused_branch_b_parallel if use_parallel_kernel
                          else _fused_branch_b_serial)
                t_kern = time.time()
                kernel(z_last, u, z_approx,
                       mus, sigmas, M_BHs,
                       gc, re, inv_1_minus_re, _FACTOR_CSI,
                       use_madau, _MA, _MB, _MC,
                       plane_low, plane_high, n_w,
                       td['s_min'], td['s_inv_step'], s_max_idx,
                       td['z_min'], td['z_inv_step'], z_max_idx,
                       mu_S_grid, sigma_S_grid, fw_z_thresh,
                       L_bols, growth_max, feff_arr)
                dt_kern = time.time() - t_kern

            if print_info:
                print(f"  [B-2D] fw={fw_p_threshold}, n_sigmas={len(sigmas)}, "
                      f"rng={dt_rng:.4f}s, fw_setup={dt_fw:.4f}s, "
                      f"kernel={dt_kern:.4f}s, total={time.time() - time_here:.4f}s")

    if print_info:
        print(f"TIME TO EVOLVE BHs: {time.time() - time_here:.2f}s")

    return M_BHs, L_bols
