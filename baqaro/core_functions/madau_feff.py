"""
Madau effective-efficiency correction factor f_eff(mu, sigma).
==============================================================

Branch B of ``bh_accretion_fast.evolve_BHs_fast`` draws the sum of (n_steps-1)
lognormal Eddington ratios from the transfer table and applies the radiative
efficiency ONCE at the median eta (``mu_scale``). Under the ``madau+`` model
eps(eta) is nonlinear in eta, so each sub-step should grow by ``eta*(1-eps(eta))``
and the median-evaluated factor under-counts the high-eta tail (eps DECREASES
with eta, so (1-eps) is larger there). The result is a systematic BHMF/QLF
under-growth at the bright end.

The correction (the f_eff correction) restores the per-sub-step MEAN exactly
by multiplying ``mu_scale`` by

                E[ eta * (1 - eps(eta)) ]
    f_eff(mu,sigma) = ---------------------------------
                (1 - eps(10**mu)) * E[eta]

where eta ~ lognormal(median=10**mu, scatter=sigma dex). f_eff >= 1 and -> 1 in
the constant-eps limit. It is a smooth function of (mu, sigma) only, so it is
precomputed on a grid and interpolated per halo -- a handful of flops per BH.

This module is the single source of f_eff for the kernels; it reuses the SAME
``get_madau_efficiency_epsilon`` the engines use so the correction is consistent
with the efficiency it corrects. Validated in
``tests/madau_sampler_bias/`` (closes the compounded BHMF bias to <0.01 dex out
to p99.99 at the fiducial sigma). OFF by default; enabled via
``BAQARO_MADAU_FEFF_CORRECTION`` / the ``madau_feff_correction`` argument so older
runs stay bit-identical.
"""
import numpy as np

from .bh_accretion import get_madau_efficiency_epsilon

# Gaussian quadrature grid over z (eta = 10**(mu + sigma*z)). 2001 points to
# +-9 sigma resolves the mean and the relevant tail of the per-term integrand.
_ZG = np.linspace(-9.0, 9.0, 2001)
_PHI = np.exp(-0.5 * _ZG**2) / np.sqrt(2.0 * np.pi)
_PHI = _PHI / np.trapezoid(_PHI, _ZG)        # renormalize the truncated gaussian

# mu grid spans well below any accreting-halo median (down to -30 where eps is
# ~constant so f_eff -> 1) up to the super-Eddington regime.
_MU_LO, _MU_HI, _MU_STEP = -30.0, 4.0, 0.05
_SIG_LO, _SIG_HI, _SIG_STEP = 0.05, 3.0, 0.05

_GRID_1D_CACHE = {}     # (round(sigma,5), round(rad_eff,5)) -> (mu_grid, feff)
_GRID_2D_CACHE = {}     # round(rad_eff,5) -> (mu_grid, sig_grid, feff_2d)


def _feff_on_mu_grid(mu_grid, sigma, rad_eff):
    """f_eff at each mu in mu_grid for a single sigma (vectorized quadrature)."""
    eta = 10.0 ** (mu_grid[:, None] + sigma * _ZG[None, :])         # (G, Z)
    one_minus = 1.0 - get_madau_efficiency_epsilon(eta, rad_eff)
    E_eta = (eta * _PHI[None, :]).sum(axis=1)
    E_eta_om = (eta * one_minus * _PHI[None, :]).sum(axis=1)
    median = 10.0 ** mu_grid
    one_minus_med = 1.0 - get_madau_efficiency_epsilon(median, rad_eff)
    feff = E_eta_om / np.maximum(one_minus_med * E_eta, 1e-300)
    return feff


def _grid_1d(sigma, rad_eff):
    key = (round(float(sigma), 5), round(float(rad_eff), 5))
    c = _GRID_1D_CACHE.get(key)
    if c is None:
        mu_grid = np.arange(_MU_LO, _MU_HI + _MU_STEP / 2, _MU_STEP)
        _GRID_1D_CACHE[key] = c = (mu_grid, _feff_on_mu_grid(mu_grid, float(sigma), rad_eff))
    return c


def _grid_2d(rad_eff):
    key = round(float(rad_eff), 5)
    c = _GRID_2D_CACHE.get(key)
    if c is None:
        mu_grid = np.arange(_MU_LO, _MU_HI + _MU_STEP / 2, _MU_STEP)
        sig_grid = np.arange(_SIG_LO, _SIG_HI + _SIG_STEP / 2, _SIG_STEP)
        feff_2d = np.empty((sig_grid.size, mu_grid.size))
        for j, s in enumerate(sig_grid):
            feff_2d[j] = _feff_on_mu_grid(mu_grid, float(s), rad_eff)
        _GRID_2D_CACHE[key] = c = (mu_grid, sig_grid, feff_2d)
    return c


def feff_per_halo(mus, sigma, rad_eff=0.1):
    """Per-halo madau f_eff correction factor (float64 array, same length as mus).

    Parameters
    ----------
    mus : ndarray
        Per-halo median log10(eta) (ERDF mu).
    sigma : float or ndarray
        Fixed scalar sigma (B-1D path) -> 1D mu interpolation; or per-halo array
        (B-2D path) -> bilinear (mu, sigma) interpolation.
    rad_eff : float
        Base radiative efficiency (eps_base), default 0.1.
    """
    mus = np.asarray(mus, dtype=np.float64)
    if np.ndim(sigma) == 0:
        mu_grid, feff = _grid_1d(sigma, rad_eff)
        return np.interp(np.clip(mus, mu_grid[0], mu_grid[-1]), mu_grid, feff)
    # variable sigma: bilinear in (mu, sigma)
    sig = np.asarray(sigma, dtype=np.float64)
    mu_grid, sig_grid, feff_2d = _grid_2d(rad_eff)
    mu_c = np.clip(mus, mu_grid[0], mu_grid[-1])
    sig_c = np.clip(sig, sig_grid[0], sig_grid[-1])
    mu_f = (mu_c - mu_grid[0]) / _MU_STEP
    sg_f = (sig_c - sig_grid[0]) / _SIG_STEP
    mi = np.clip(mu_f.astype(np.int64), 0, mu_grid.size - 2)
    sj = np.clip(sg_f.astype(np.int64), 0, sig_grid.size - 2)
    mw = mu_f - mi
    sw = sg_f - sj
    f00 = feff_2d[sj, mi]; f01 = feff_2d[sj, mi + 1]
    f10 = feff_2d[sj + 1, mi]; f11 = feff_2d[sj + 1, mi + 1]
    return ((f00 * (1 - mw) + f01 * mw) * (1 - sw)
            + (f10 * (1 - mw) + f11 * mw) * sw)
