"""
Shared helpers for the madau-sampler-bias investigation suite.
=============================================================

Background (see ``core_functions/madau_feff.py``):
the fast accretion path (``bh_accretion_fast.evolve_BHs_fast``, Branch B) draws
the SUM of (n_steps-1) lognormal Eddington ratios in one shot from the transfer
table. That is EXACT when the radiative efficiency eps is constant (growth is
linear in eta_j). Under the production ``madau+`` efficiency each sub-step grows
by ``eta_j * (1 - eps(eta_j))`` which is NONLINEAR in eta_j; Branch B approximates
it by evaluating ``(1 - eps)`` ONCE at the median eta and applying that single
factor to the whole history sum (``mu_scale``). This module collects everything
the layered diagnostics need so each layer matches the production kernel exactly:

* ``madau_eps`` / ``one_minus_eps``  - the exact Madau eps(eta) used in the kernel.
* ``per_substep_growth_mean``        - E[eta(1-eps(eta))] by quadrature (LAYER 1).
* ``eff_factor_exact`` / ``_approx`` - the per-term effective factors the two
                                       paths use; their ratio is the f_eff fix.
* ``make_fixed_erdf``                - an Erdf whose mu, sigma are CONSTANT across
                                       halos (control over (mu, sigma) directly).
* ``run_kernel_growth``              - call the real ``evolve_BHs_fast`` Branch B
                                       (table) vs Branch A (direct) and return the
                                       per-halo growth + L_bol (LAYER 2/3).
* ``load_sampler`` / ``load_cadence``/``load_rate_cache`` - production inputs.
* ``pdiff_dex`` / ``summarize``       - percentile reporting in dex.

All constants (_MA/_MB/_MC, ls/ms, log_csi) are imported from the same sources the
kernel uses so there is a single source of truth.
"""
import os
import glob
import numpy as np

import qhtools.utils.natconst as nc
from baqaro.core_functions.erdf_core_functions import Erdf

LN10 = np.log(10.0)

# Madau+ (2014) spin-dependent efficiency constants -- identical to the inline
# numba kernel (_MA/_MB/_MC) and the reference get_madau_efficiency_epsilon.
_MA = 1.8260922439282448
_MB = 0.7586284639465684
_MC = 0.01611606486191978

# Eddington / unit factors (the same ones evolve_BHs_fast builds growth_const from)
_LS_OVER_MS = nc.ls / nc.ms            # ~1.9359  (Lsun->erg/s, g->Msun)
_CSI = 10.0 ** nc.log_csi             # Eddington L per Msun (Lsun/Msun)


# ----------------------------------------------------------------------
# Efficiency model (numpy, matches kernel bit-for-bit in float64)
# ----------------------------------------------------------------------
def madau_eps(eta, rad_eff=0.1):
    """eps(eta) Madau+ fit, clamped to [0, 0.999] -- matches the kernel."""
    eta = np.asarray(eta, dtype=np.float64)
    inv = 1.0 / np.maximum(eta, 1e-20)
    t1 = 0.985 / (1.6 * inv + _MB)
    t2 = 0.015 / (1.6 * inv + _MC)
    eps = (rad_eff * _MA) * inv * (t1 + t2)
    return np.clip(eps, 0.0, 0.999)


def one_minus_eps(eta, rad_eff=0.1):
    return 1.0 - madau_eps(eta, rad_eff)


# ----------------------------------------------------------------------
# LAYER 1 analytical pieces: per-sub-step effective-growth means
# ----------------------------------------------------------------------
# A sub-step draws eta = 10**(mu + sigma*z), z ~ N(0,1). The dimensionless
# growth contribution (madau) is  eta * (1 - eps(eta)) / (1 - rad_eff).
# We integrate over z with a dense Gauss-Hermite-style trapezoid -- exact to
# ~1e-12 for the MEAN (tails handled separately by the kernel layer).
_ZG = np.linspace(-9.0, 9.0, 6001)
_PHI = np.exp(-0.5 * _ZG**2) / np.sqrt(2.0 * np.pi)
_PHI = _PHI / np.trapezoid(_PHI, _ZG)    # renormalize the truncated gaussian


def _E_over_z(f_of_eta, mu, sigma):
    eta = 10.0 ** (mu + sigma * _ZG)
    return np.trapezoid(f_of_eta(eta) * _PHI, _ZG)


def E_eta(mu, sigma):
    """E[eta] for eta lognormal with median 10**mu, scatter sigma (dex)."""
    return _E_over_z(lambda e: e, mu, sigma)


def per_substep_growth_mean_exact(mu, sigma, rad_eff=0.1):
    """E[eta*(1-eps(eta))] -- the per-sub-step mean Branch A actually realizes."""
    return _E_over_z(lambda e: e * one_minus_eps(e, rad_eff), mu, sigma)


def per_substep_growth_mean_approx(mu, sigma, rad_eff=0.1):
    """E[eta]*(1-eps(median)) -- the per-sub-step mean Branch B (mu_scale) realizes.

    Branch B factors the median out of the table sum (table has median 1, mean
    exp(sigma_nat^2/2)) and multiplies by mu_scale = median*(1-eps(median)).
    The per-term mean is therefore E[eta] * (1 - eps(10**mu)).
    """
    median = 10.0 ** mu
    return E_eta(mu, sigma) * one_minus_eps(median, rad_eff)


def f_eff_fix(mu, sigma, rad_eff=0.1):
    """Mean-correction factor for mu_scale: exact_mean / approx_mean.

    Multiplying mu_scale by f_eff makes the Branch-B history MEAN match Branch A
    exactly (the f_eff correction). Returns 1.0 in the constant-eps limit.
    """
    approx = per_substep_growth_mean_approx(mu, sigma, rad_eff)
    exact = per_substep_growth_mean_exact(mu, sigma, rad_eff)
    return exact / approx


_FEFF_GRID_CACHE = {}


def f_eff_fix_vec(mu_array, sigma, rad_eff=0.1, mu_lo=-30.0, mu_hi=3.0, step=0.05):
    """Vectorized f_eff over an array of mu at FIXED sigma, via grid+interp.

    Mirrors a deployable fix: precompute f_eff on a 1D mu grid for the run's
    (fixed) sigma, then interpolate per halo. Grid cached per (sigma, rad_eff).
    """
    key = (round(float(sigma), 5), round(float(rad_eff), 5), mu_lo, mu_hi, step)
    cached = _FEFF_GRID_CACHE.get(key)
    if cached is None:
        grid = np.arange(mu_lo, mu_hi + step / 2, step)
        vals = np.array([f_eff_fix(m, sigma, rad_eff) for m in grid])
        _FEFF_GRID_CACHE[key] = cached = (grid, vals)
    grid, vals = cached
    return np.interp(np.clip(mu_array, grid[0], grid[-1]), grid, vals)


# ----------------------------------------------------------------------
# ERDF with constant mu, sigma (control variable for kernel tests)
# ----------------------------------------------------------------------
def make_fixed_erdf(mu, sigma, seed=12345):
    """Erdf whose mu == `mu` and sigma == `sigma` regardless of halo rate.

    Uses log_normal_evol_halo_mass with log_eta_mean_evol=0 so mu(r)=log_eta_mean_0
    for any r; std_0 fixes sigma. Pass log_halo_rates = zeros to evolve_BHs_fast.
    """
    return Erdf(
        "log_normal_evol_halo_mass",
        {"log_eta_mean_0": float(mu), "log_eta_mean_evol": 0.0, "std_0": float(sigma)},
        rng=np.random.default_rng(seed),
    )


# ----------------------------------------------------------------------
# Production-input loaders
# ----------------------------------------------------------------------
def load_sampler(filename="Universal_Lognormal_Sampler_final.npz"):
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    return load_3d_sampler(filename)


def load_cadence(sim="L2800N10080", max_snap=144):
    """Return (redshifts, delta_t_gyr) per snapshot, matching main_evolution."""
    from qhtools.utils.cosmology import cosmo
    from baqaro.utils.my_dir import get_input_path_HBT_data
    snaps = np.arange(max_snap + 1)
    path_sim = get_input_path_HBT_data(source="machine_igm")
    zfile = os.path.join(path_sim, f"{sim}/output_list.txt")
    redshifts = np.asarray(np.loadtxt(zfile))[snaps]
    ages = cosmo.age(redshifts)
    dt = np.diff(ages, prepend=0.01)
    return redshifts, dt


def find_rate_cache(pattern="*root71_flatN20000*specific_cold_accretion_rates.npy"):
    from baqaro.utils.my_dir import get_output_path
    base = os.path.join(get_output_path("machine_igm"), "halo_subsets")
    hits = sorted(glob.glob(os.path.join(base, pattern)))
    return hits[0] if hits else None


# ----------------------------------------------------------------------
# Kernel driver: Branch B (table) vs Branch A (direct), one snapshot
# ----------------------------------------------------------------------
def run_kernel_growth(tf, mu, sigma, n_steps, delta_t, size=200_000,
                      rad_model="madau+", seed=12345, M0=1.0e6):
    """Evolve `size` BHs over ONE snapshot at fixed (mu, sigma); return ln-growth.

    tf=None  -> Branch A (exact direct sub-step sampling).
    tf=table -> Branch B (transfer-table history sum + mu_scale approx).
    Same seed/erdf so the only difference is the engine path. Returns
    (ln_growth[size], L_bol[size]).
    """
    from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
    erdf = make_fixed_erdf(mu, sigma, seed=seed)
    M = np.full(size, M0, dtype=np.float32)
    rates = np.zeros(size, dtype=np.float32)   # -> mu(r)=log_eta_mean_0 exactly
    M_out, L = evolve_BHs_fast(
        tf, M, float(delta_t), erdf,
        rad_efficiency=0.1, n_steps=int(n_steps),
        log_halo_rates_array=rates, rad_efficiency_model=rad_model,
        parallelize=False, use_parallel_kernel=False, growth_max=50.0,
    )
    ln_growth = np.log(np.asarray(M_out, dtype=np.float64) / M0)
    return ln_growth, np.asarray(L, dtype=np.float64)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
DEFAULT_PCTS = (16, 50, 84, 95, 99, 99.9)


def pdiff_dex(table_vals, direct_vals, pcts=DEFAULT_PCTS):
    """Per-percentile (table - direct) difference in dex of the two samples.

    Positive => table predicts a HIGHER value at that percentile.
    Operates on log10 of the values (e.g. M_BH or L_bol).
    """
    out = {}
    lt = np.log10(np.maximum(table_vals, 1e-300))
    ld = np.log10(np.maximum(direct_vals, 1e-300))
    for p in pcts:
        out[p] = float(np.percentile(lt, p) - np.percentile(ld, p))
    return out


def summarize(name, d, pcts=DEFAULT_PCTS):
    cells = "  ".join(f"p{p}={d[p]:+.4f}" for p in pcts)
    return f"{name:<22s} {cells}"
