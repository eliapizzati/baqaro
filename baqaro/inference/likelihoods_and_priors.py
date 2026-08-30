"""
Likelihood functions, priors, and precomputation helpers for MCMC inference.

This module provides the statistical machinery for fitting black hole population
model parameters to observational data.  Three independent observables are
supported, each with its own likelihood:

1. **Quasar Luminosity Function (QLF)** — number density of quasars as a
   function of luminosity in redshift bins.  Two likelihood variants:
   - *Gaussian*: standard chi-squared in log-space with an ad-hoc bright-end
     penalty that discourages the model from predicting too many extremely
     luminous quasars beyond what is observed.
   - *Hybrid Poisson/Gaussian* (Cash 1979): uses Gaussian chi-squared for
     high-count bins, Poisson (Cash C-statistic) for low-count bins, and a
     Poisson non-detection penalty (N_obs=0) for luminosity bins beyond the
     brightest data point.

2. **Conditional Eddington Ratio Distribution Function (CERDF)** — the
   probability distribution of Eddington ratios for quasars in a given
   luminosity and redshift range.  Evaluated as a per-QSO likelihood:
   L(theta) = prod_i P(log_eta_i | L_bol_i, z_i, theta).

3. **Correlation function** — spatial clustering of quasars (auto-correlation
   wp/rp, or quasar-galaxy cross-correlation volume-averaged xi), computed
   from the quasar halo mass function (QHMF) via the halo model.

Architecture
------------
Each likelihood follows a **precompute + evaluate** pattern:

- ``precompute_*_inputs(...)`` does expensive one-time setup (index lookups,
  grid interpolation, triangle/HMF computation) and returns a config dict.
- ``log_likelihood_*(params, ..., cfg)`` evaluates the likelihood cheaply,
  calling the GP emulator and using the precomputed config.
- ``log_probability_*(params, ...)`` wraps the likelihood with a uniform
  prior check (parameter bounds from the emulator) and returns a
  ``(log_prob, blob)`` tuple for ``emcee`` compatibility.  The blob is
  always 0.0 (unused placeholder).

All functions operate in **log-space** (log10 for luminosities, number
densities, and Eddington ratios; natural log for likelihoods).
"""

import os
import numpy as np
from scipy.special import gammaln
from scipy.ndimage import gaussian_filter1d
# Correlation-function machinery migrated to qhtools (faster, JIT'd, and the
# projection functions use exact piecewise integration instead of a grid sum).
# get_projected_wp / get_volume_averaged_xi take the radial xi bins as their 3rd
# arg and auto-detect centers vs edges by length, so we pass get_corr_from_triangle's
# bin centers (10 ** log_rbins) directly.
from qhtools.utils.chi2 import get_chi2
from qhtools.utils import my_utils
from qhtools.clustering.qhmf_to_corr import (
    get_corr_from_triangle,
    get_corr_from_triangle_cross,
)
from qhtools.clustering.projected_correlation_functions import (
    get_projected_wp,
    get_volume_averaged_xi,
)




def setup_initial_positions(best_fit, ranges, n_walkers, rng, jitter=1e-3):
    """
    Initialize emcee walkers in a tiny Gaussian ball around the best-fit point.

    Each walker is perturbed from ``best_fit`` by Gaussian noise scaled to
    ``jitter * (high - low)`` per dimension.  Walkers are clipped to stay
    strictly inside the prior bounds (with a tiny 1e-5 margin) so that no
    walker starts with -inf log-prior.

    Parameters
    ----------
    best_fit : array-like, shape (ndim,)
        Centre of the initialization ball (typically from a prior optimizer).
    ranges : list of (float, float)
        Prior bounds ``(low, high)`` for each parameter.
    n_walkers : int
        Number of emcee walkers.
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    jitter : float
        Fraction of the parameter range used as the Gaussian sigma.
        Default 1e-3 keeps the ball very tight so burn-in is short.

    Returns
    -------
    pos : ndarray, shape (n_walkers, ndim)
        Initial walker positions.
    """
    ndim = len(best_fit)
    pos = []
    for _ in range(n_walkers):
        walker = np.zeros(ndim)
        for i in range(ndim):
            low, high = ranges[i]
            width = high - low
            proposal = best_fit[i] + jitter * width * rng.standard_normal()
            # Keep walkers strictly inside bounds to avoid -inf prior
            walker[i] = np.clip(proposal, low + 1e-5 * width, high - 1e-5 * width)
        pos.append(walker)
    return np.array(pos)


# ==============================================================================
# QLF LIKELIHOOD
# ==============================================================================

def precompute_qlf_inputs(datas_qlf, qlf_emulator, penalty_limit=-8.5,
                          penalty_strength=100.0, weights=None):
    """
    Precompute everything the QLF likelihood needs.

    Parameters
    ----------
    datas_qlf : list
        List of data objects (one per z-bin), or None for skipped bins.
    qlf_emulator : GeneralEmulatorGP
        Trained QLF emulator.
    penalty_limit : float
        Floor value (log-space) for the bright-end penalty.
    penalty_strength : float
        Multiplier for the penalty chi2 term.
    weights : list or None
        Per-redshift-bin weight on the chi2. Default 1.0 for every bin.

    Returns
    -------
    qlf_cfg : dict
        'log_bins'          : 1-D array, emulator luminosity grid
        'penalty_masks'     : list of bool arrays (or None) per z-bin
        'penalty_limits'    : list of float (or None) per z-bin
        'penalty_strength'  : float
        'weights'           : 1-D array of per-bin weights
    """
    log_bins = qlf_emulator.axis_data["log_bins"]
    n_z = len(datas_qlf)

    penalty_masks = []
    penalty_limits = []

    for data_qlf in datas_qlf:
        if data_qlf is None:
            penalty_masks.append(None)
            penalty_limits.append(None)
            continue

        max_data_L = np.max(data_qlf.x)
        # Penalty region starts 0.1 dex above the brightest observed data point,
        # i.e. in luminosity bins with no observational constraints.  The small
        # margin avoids penalising the last data bin itself.
        penalty_masks.append(log_bins > max_data_L + 0.1)
        penalty_limits.append(penalty_limit)

    if weights is None:
        weights = np.ones(n_z)
    else:
        weights = np.asarray(weights, dtype=float)

    return {
        'log_bins': log_bins,
        'penalty_masks': penalty_masks,
        'penalty_limits': penalty_limits,
        'penalty_strength': penalty_strength,
        'weights': weights,
    }


def log_likelihood_qlf(params, datas_qlf, qlf_emulator, qlf_cfg):
    """
    Gaussian QLF log-likelihood with optional bright-end penalty.

    For each redshift bin, the emulated log10(Phi) is interpolated onto the
    observed luminosity bins and compared via chi-squared (using ``get_chi2``,
    which supports both diagonal errors and full covariance matrices).

    An additional *bright-end penalty* is applied in emulator bins beyond
    the brightest observed data point: any model prediction that exceeds
    ``penalty_limit`` (in log10 Phi) is penalised quadratically.  This
    prevents the model from producing an unphysical uptick at the bright end
    where there is no data to constrain it.

    Parameters
    ----------
    params : array-like
        Emulator parameter vector.
    datas_qlf : list
        Observational data objects (one per z-bin, or None for skipped bins).
        Each object must expose ``.x``, ``.data``, ``.err``, ``.covariance``.
    qlf_emulator : GeneralEmulatorGP
    qlf_cfg : dict
        Output of ``precompute_qlf_inputs``.

    Returns
    -------
    float
        Log-likelihood value (-0.5 * chi2_total).
    """
    chi2 = 0.0

    # Single emulator call for all redshift bins at once
    log_qlf_emul = qlf_emulator.predict_mean_only(np.atleast_2d(params))[0]

    log_bins = qlf_cfg['log_bins']
    penalty_masks = qlf_cfg['penalty_masks']
    penalty_limits = qlf_cfg['penalty_limits']
    penalty_strength = qlf_cfg['penalty_strength']
    weights = qlf_cfg['weights']

    for i_z, data_qlf in enumerate(datas_qlf):
        if data_qlf is None:
            continue

        w = weights[i_z]

        # Standard chi2 (get_chi2 interpolates model onto data bins internally)
        chi2 += w * get_chi2(data_qlf, log_bins, log_qlf_emul[i_z, :])

        # Bright-end penalty: quadratic cost for model exceeding the floor
        # in luminosity bins beyond the data.  Only bins where the model
        # is *above* the penalty_limit contribute (diff > 0).
        mask = penalty_masks[i_z]
        limit = penalty_limits[i_z]
        if mask is not None:
            diff = log_qlf_emul[i_z, mask] - limit
            violators = diff[diff > 0]
            if violators.size > 0:
                chi2 += np.sum(violators ** 2) * penalty_strength

    return -0.5 * chi2


def log_likelihood_qlf_from_prediction(log_qlf_emul, datas_qlf, qlf_cfg):
    """
    Gaussian QLF log-likelihood from a pre-computed emulator prediction.

    Identical to ``log_likelihood_qlf`` but skips the emulator call, which
    is useful when the prediction is already available (e.g. the reference
    implementation evaluates the emulator once and reuses the output for both
    plotting and likelihood display).

    Parameters
    ----------
    log_qlf_emul : ndarray, shape (n_z, n_bins)
        Pre-computed emulator output (log10 Phi per redshift bin).
    datas_qlf : list
        Observational data objects (one per z-bin, or None).
    qlf_cfg : dict
        Output of ``precompute_qlf_inputs``.

    Returns
    -------
    float
        Log-likelihood value.
    """
    chi2 = 0.0
    log_bins = qlf_cfg['log_bins']
    penalty_masks = qlf_cfg['penalty_masks']
    penalty_limits = qlf_cfg['penalty_limits']
    penalty_strength = qlf_cfg['penalty_strength']
    weights = qlf_cfg['weights']

    for i_z, data_qlf in enumerate(datas_qlf):
        if data_qlf is None:
            continue
        w = weights[i_z]
        chi2 += w * get_chi2(data_qlf, log_bins, log_qlf_emul[i_z, :])
        mask = penalty_masks[i_z]
        limit = penalty_limits[i_z]
        if mask is not None:
            diff = log_qlf_emul[i_z, mask] - limit
            violators = diff[diff > 0]
            if violators.size > 0:
                chi2 += np.sum(violators ** 2) * penalty_strength

    return -0.5 * chi2


# ==============================================================================
# QLF LIKELIHOOD — POISSON/GAUSSIAN HYBRID  (Cash 1979)
# ==============================================================================

def precompute_qlf_inputs_poisson(datas_qlf, qlf_emulator, dlogL_data=0.5,
                                  N_threshold=20, weights=None,
                                  datas_qlf_stat=None,
                                  nondet_max_dlogL=1.0):
    # Env-var override of the bright-end non-detection penalty cap:
    #   BAQARO_QLF_NONDET_MAX_DLOGL=0    -> disable the penalty entirely
    #                                      (nondet_mask is forced empty since
    #                                      no log_bin satisfies > max+0.1 AND
    #                                      <= max+0; useful for testing the
    #                                      bright-end constraint's weight)
    #   BAQARO_QLF_NONDET_MAX_DLOGL=0.5  -> halve the band
    #   BAQARO_QLF_NONDET_MAX_DLOGL=2.0  -> extend it
    """
    Precompute inputs for the hybrid Poisson/Gaussian QLF likelihood.

    This implements the statistical framework of Cash (1979) extended with
    a systematic-error degradation.  The key idea: bins with many detections
    are well-described by Gaussian errors in log-space, while bins with few
    counts (the bright end) require Poisson statistics to avoid bias from
    the Gaussian approximation.

    **Effective count estimation** (from statistical errors only):

    For Poisson-dominated counting, the fractional error on the number
    density Phi is 1/sqrt(N).  Propagating to log10:

        sigma_{log Phi, stat} = 1 / (ln10 * sqrt(N))
        => N_eff = 1 / (ln10 * sigma_stat)^2

    This inversion is exact for single-survey bins and remains valid for
    inverse-variance-weighted combinations of multiple surveys, because
    N_eff is additive under that weighting scheme.

    **Systematic-error degradation of V_eff**:

    The raw V_eff = N_eff / (Phi * dlogL) reflects the survey volume implied
    by statistical errors alone.  When systematic uncertainty inflates the
    total error, the information content per bin decreases.  We degrade
    V_eff by the factor (sigma_stat / sigma_full)^2, which equals 1.0 when
    there is no systematic and approaches 0 for systematic-dominated bins.
    This prevents Poisson bins from being overconfident when the error budget
    is dominated by systematics (e.g. incompleteness corrections).

    Parameters
    ----------
    datas_qlf : list
        Data objects per z-bin with FULL errors (stat + systematic).
        Used for the Gaussian chi2 in high-count bins.
    qlf_emulator : GeneralEmulatorGP
    dlogL_data : float
        Bin width of the observational data in dex (default 0.5).
    N_threshold : int
        Below this effective count, switch from Gaussian to Poisson.
    weights : list or None
        Per-redshift weight. Default 1.0.
    datas_qlf_stat : list or None
        Data objects per z-bin with PURE STATISTICAL errors (no systematic
        floor).  Used to estimate N_eff and V_eff.  If None, falls back
        to ``datas_qlf`` (works if no systematic was added).
    nondet_max_dlogL : float
        Maximum luminosity range (in dex) beyond the brightest data point
        for non-detection bins.  Emulator bins further out are ignored.
        Default 1.0 dex.  Set to ``None`` to use all emulator bins.

    Returns
    -------
    dict
        'log_bins'    : emulator luminosity grid
        'per_z'       : list of per-redshift dicts (or None), each with
                        N_obs, N_eff, V_eff, is_poisson, nondet_mask, etc.
        'weights'     : per-redshift weight array
        'N_threshold' : threshold for Gaussian/Poisson switching
    """
    _env_cap = os.environ.get("BAQARO_QLF_NONDET_MAX_DLOGL", "").strip()
    if _env_cap:
        nondet_max_dlogL = float(_env_cap)
        print(f"[qlf-likelihood] BAQARO_QLF_NONDET_MAX_DLOGL={nondet_max_dlogL} "
              f"(non-det band cap; 0 = penalty OFF)")

    # Opt-in; default OFF = bit-identical. See the block
    # near `V_eff_bright` below for the derivation.
    _QLF_NONDET_TRUE_WIDTH = (
        os.environ.get("BAQARO_QLF_NONDET_TRUE_WIDTH", "0").strip().lower()
        not in ("0", "", "false", "no", "off")
    )
    # Max allowed boost of V_eff_bright, guarding against a degenerate top-bin
    # sliver (see the clamp below). Override with BAQARO_QLF_NONDET_MAX_BOOST.
    _QLF_NONDET_MAX_BOOST = float(os.environ.get("BAQARO_QLF_NONDET_MAX_BOOST", "10.0"))
    if _QLF_NONDET_TRUE_WIDTH:
        print("[qlf-likelihood] BAQARO_QLF_NONDET_TRUE_WIDTH=1 -> V_eff_bright "
              "rescaled to the brightest bin's TRUE width (strengthens the "
              "bright-end non-detection penalty). CHANGES the likelihood. "
              f"Boost capped at {_QLF_NONDET_MAX_BOOST:.0f}x.")
    if datas_qlf_stat is None:
        datas_qlf_stat = datas_qlf

    log_bins = qlf_emulator.axis_data["log_bins"]
    n_z = len(datas_qlf)
    LN10 = np.log(10.0)

    # Emulator bin spacing (for non-detection bins)
    if len(log_bins) > 1:
        dlogL_emul = np.abs(log_bins[1] - log_bins[0])
    else:
        dlogL_emul = dlogL_data

    per_z = []
    for data_qlf, data_stat in zip(datas_qlf, datas_qlf_stat):
        if data_qlf is None:
            per_z.append(None)
            continue

        # Positions and values from the main (systematic-inflated) data
        x_data = np.asarray(data_qlf.x)
        logPhi_data = np.asarray(data_qlf.data)
        sigma_logPhi_full = np.asarray(data_qlf.err)  # stat + sys

        # Pure statistical errors for N_eff estimation
        sigma_logPhi_stat = np.asarray(data_stat.err)

        # Invert the Poisson error formula: sigma_logPhi = 1/(ln10*sqrt(N))
        # => N_eff = 1/(ln10*sigma)^2.  Uses stat-only errors so that
        # systematic inflation does not artificially inflate N_eff.
        N_eff = 1.0 / (LN10 * sigma_logPhi_stat) ** 2
        N_eff = np.maximum(N_eff, 0.5)  # floor to avoid div-by-zero in V_eff

        # Effective volume: V_eff = N / (Phi_linear * dlogL)
        Phi_linear = 10.0 ** logPhi_data
        V_eff = N_eff / (Phi_linear * dlogL_data)

        # Degrade V_eff to account for systematic uncertainty.
        # In the Gaussian case, sigma_full^2 = sigma_stat^2 + sigma_sys^2,
        # so the information content scales as (sigma_stat/sigma_full)^2.
        # Apply the same degradation to V_eff so Poisson bins are not
        # overconfident when the systematic floor is significant.
        degradation = (sigma_logPhi_stat / sigma_logPhi_full) ** 2
        V_eff_degraded = V_eff * degradation

        # Recompute N_obs from degraded volume.  This is the integer count
        # the survey would have detected if V_eff were reduced by the
        # systematic degradation factor.  These are the counts fed to the
        # Poisson term: N*ln(lam) - lam - ln(N!).
        N_obs_degraded = Phi_linear * V_eff_degraded * dlogL_data
        N_obs = np.round(N_obs_degraded).astype(int)
        N_obs = np.maximum(N_obs, 0)

        # Poisson for all low-count bins (no stat_dominates gate)
        is_poisson = N_eff < N_threshold

        # --- Non-detection (upper limit) bins ---
        # Emulator bins beyond the brightest observed data point had zero
        # detections.  We treat these as Poisson N=0 observations, which
        # contribute ll = -lambda (penalising models that predict quasars
        # where none were seen).  nondet_max_dlogL caps how far beyond the
        # data this penalty reaches, because the emulator is unreliable at
        # extreme extrapolation.
        max_data_L = np.max(x_data)
        nondet_mask = log_bins > max_data_L + 0.1
        if nondet_max_dlogL is not None:
            nondet_mask &= log_bins <= max_data_L + nondet_max_dlogL

        # Use V_eff from the brightest detected bin as a conservative
        # estimate for the non-detection bins (the survey volume was at
        # least this large at those luminosities).
        brightest_idx = np.argmax(x_data)
        V_eff_bright = V_eff_degraded[brightest_idx]

        # --- True-width V_eff_bright (opt-in) --------------------------------
        # V_eff above was built with the NOMINAL dlogL_data (0.5) for EVERY bin.
        # For the fitted bins that is harmless: the Poisson term
        #   lam = Phi_model * V_eff * dlogL_data = Phi_model * N_eff / Phi_data
        # cancels the width exactly. But the NON-DETECTION term is
        #   lam_nondet = Phi_model * V_eff_bright * dlogL_emul
        # where dlogL_data does NOT cancel (dlogL_emul is the emulator's width).
        # Because the brightest bin's upper edge is anchored at max(x) it can be
        # far narrower than 0.5 dex, so V_eff_bright is mis-scaled by the ratio.
        # Rescaling to the bin's TRUE width w_b is exact:
        #   V_eff_true = N_eff*degr/(Phi*w_b) = V_eff_bright * (dlogL_data / w_b)
        # OFF by default => bit-identical to every existing chain. Enable with
        # BAQARO_QLF_NONDET_TRUE_WIDTH=1; it CHANGES the likelihood, so tag chains
        # via BAQARO_MCMC_NOTES and re-DE.
        #
        # SAFETY CLAMP — a DEGENERATE top sliver would blow this up. When max(x)
        # sits a hair above a bin_width grid point, rebin_data appends a
        # near-zero-width top bin (measured: 3.98e-4 dex at z=1!), and the exact
        # rescale would then boost the penalty by dlogL_data/w_b ≈ 1255x, which
        # is a binning artifact, not survey information. Cap the boost and shout.
        # Enable BAQARO_QLF_MERGE_TOP_SLIVER=1 (obs_data) to remove the sliver at
        # the source, after which the surviving corrections are a sane 1.3-4x.
        if _QLF_NONDET_TRUE_WIDTH:
            w_bins = getattr(data_qlf, "dlogL", None)
            if w_bins is not None and len(w_bins) == len(x_data):
                w_b = float(w_bins[brightest_idx])
                if w_b > 0:
                    boost = dlogL_data / w_b
                    if boost > _QLF_NONDET_MAX_BOOST:
                        print(f"[qlf-likelihood] ⚠ degenerate top bin "
                              f"(width={w_b:.2e} dex) would boost the bright-end "
                              f"penalty {boost:.0f}x; CLAMPING to "
                              f"{_QLF_NONDET_MAX_BOOST:.0f}x. Set "
                              f"BAQARO_QLF_MERGE_TOP_SLIVER=1 to fix the binning.")
                        boost = _QLF_NONDET_MAX_BOOST
                    V_eff_bright = V_eff_bright * boost

        per_z.append({
            'x_data': x_data,
            'logPhi_data': logPhi_data,
            'sigma_logPhi': sigma_logPhi_full,  # full errors for Gaussian bins
            'N_obs': N_obs,
            'N_eff': N_eff,
            'V_eff': V_eff_degraded,            # degraded for systematic
            'V_eff_raw': V_eff,                 # raw (stat-only) for diagnostics
            'degradation': degradation,
            'dlogL_data': dlogL_data,
            'is_poisson': is_poisson,
            'nondet_mask': nondet_mask,
            'V_eff_bright': V_eff_bright,
            'dlogL_emul': dlogL_emul,
        })

    if weights is None:
        weights = np.ones(n_z)
    else:
        weights = np.asarray(weights, dtype=float)

    return {
        'log_bins': log_bins,
        'per_z': per_z,
        'weights': weights,
        'N_threshold': N_threshold,
    }


def log_likelihood_qlf_poisson(params, datas_qlf, qlf_emulator, qlf_cfg):
    """
    Hybrid Poisson/Gaussian QLF log-likelihood (Cash 1979).

    Three regimes per luminosity bin:

    - **High-count** (N_eff >= N_threshold): Gaussian chi-squared in
      log-space using the full (stat + systematic) errors.
    - **Low-count** (N_eff < N_threshold): Poisson Cash C-statistic,
      ``ll = N*ln(lam) - lam - ln(N!)``, using degraded V_eff.
    - **Non-detection** (beyond brightest data): Poisson with N_obs=0,
      giving ``ll = -lambda``.

    Parameters
    ----------
    params : array-like
        Emulator parameter vector.
    datas_qlf : list
        Observational data objects (unused here; kept for API consistency).
    qlf_emulator : GeneralEmulatorGP
    qlf_cfg : dict
        Output of ``precompute_qlf_inputs_poisson``.

    Returns
    -------
    float
        Log-likelihood value.
    """
    log_qlf_emul = qlf_emulator.predict_mean_only(np.atleast_2d(params))[0]
    return _log_likelihood_qlf_poisson_from_prediction(log_qlf_emul, qlf_cfg)


def log_likelihood_qlf_poisson_from_prediction(log_qlf_emul, datas_qlf, qlf_cfg):
    """
    Hybrid Poisson/Gaussian QLF likelihood from a pre-computed prediction.

    Identical to ``log_likelihood_qlf_poisson`` but skips the emulator call.
    Used by the reference implementation where the prediction is already available.
    """
    return _log_likelihood_qlf_poisson_from_prediction(log_qlf_emul, qlf_cfg)


def _log_likelihood_qlf_poisson_from_prediction(log_qlf_emul, qlf_cfg):
    """
    Core implementation of the hybrid Poisson/Gaussian QLF likelihood.

    Called by both ``log_likelihood_qlf_poisson`` (which runs the emulator)
    and ``log_likelihood_qlf_poisson_from_prediction`` (which receives an
    already-computed prediction).
    """
    log_bins = qlf_cfg['log_bins']
    weights = qlf_cfg['weights']
    ll_total = 0.0

    for i_z, z_cfg in enumerate(qlf_cfg['per_z']):
        if z_cfg is None:
            continue

        w = weights[i_z]
        ll_z = 0.0

        x_data = z_cfg['x_data']
        logPhi_data = z_cfg['logPhi_data']
        sigma_logPhi = z_cfg['sigma_logPhi']
        N_obs = z_cfg['N_obs']
        V_eff = z_cfg['V_eff']
        dlogL_data = z_cfg['dlogL_data']
        is_poisson = z_cfg['is_poisson']

        # Interpolate emulator prediction (on its own grid) onto the
        # observational luminosity bins (which may differ in spacing/range)
        logPhi_model = np.interp(x_data, log_bins, log_qlf_emul[i_z, :])

        # --- Gaussian bins (high count) ---
        gauss_mask = ~is_poisson
        if gauss_mask.any():
            residuals = (logPhi_model[gauss_mask] - logPhi_data[gauss_mask]) / sigma_logPhi[gauss_mask]
            ll_z -= 0.5 * np.sum(residuals ** 2)

        # --- Poisson bins (low count, Cash C-statistic) ---
        poisson_mask = is_poisson
        if poisson_mask.any():
            # Expected count from the model: lambda = Phi_model * V_eff * dlogL
            # V_eff is the degraded effective volume (accounts for systematics)
            Phi_model_lin = 10.0 ** logPhi_model[poisson_mask]
            lam = Phi_model_lin * V_eff[poisson_mask] * dlogL_data
            lam = np.maximum(lam, 1e-30)  # avoid log(0)
            N = N_obs[poisson_mask]

            # Cash C-statistic: ln P(N|lam) = N*ln(lam) - lam - ln(N!)
            # gammaln(N+1) = ln(N!) via the gamma function identity
            ll_z += np.sum(N * np.log(lam) - lam - gammaln(N + 1))

        # --- Non-detection bins (N=0, Poisson upper limits) ---
        # These are emulator bins beyond the brightest data point where
        # the survey saw zero quasars.  The Poisson likelihood for N=0 is
        # P(0|lam) = exp(-lam), so ln P = -lam.  This penalises models
        # that predict quasars in bins where none were observed.
        nondet_mask = z_cfg['nondet_mask']
        if nondet_mask.any():
            V_eff_bright = z_cfg['V_eff_bright']
            dlogL_emul = z_cfg['dlogL_emul']
            Phi_model_nondet = 10.0 ** log_qlf_emul[i_z, nondet_mask]
            lam_nondet = Phi_model_nondet * V_eff_bright * dlogL_emul
            ll_z -= np.sum(lam_nondet)

        ll_total += w * ll_z

    return ll_total


# ==============================================================================
# CONDITIONAL EDDINGTON RATIO DISTRIBUTION LIKELIHOOD
# ==============================================================================

def _convolve_pdf(pdf, log_bins, scatter_dex):
    """
    Convolve a 1-D PDF on a uniform log-space grid with a Gaussian kernel.

    This models the broadening of the Eddington ratio distribution due to
    observational uncertainty in black hole mass estimates (~0.5 dex).
    The convolution is performed via ``gaussian_filter1d`` (FFT-free, fast
    for 1-D arrays) and the result is renormalised to account for edge
    effects where the kernel extends beyond the grid.
    """
    if scatter_dex <= 0:
        return pdf
    bin_width = np.abs(np.diff(log_bins[:2])[0])
    sigma_bins = scatter_dex / bin_width
    pdf_conv = gaussian_filter1d(pdf, sigma_bins, mode='constant', cval=0.0)
    # Re-normalise after convolution (edge effects can leak area)
    integral = np.trapezoid(pdf_conv, log_bins)
    if integral > 0:
        pdf_conv /= integral
    return pdf_conv


def get_cerdf_pdf(params, cerdf_emulator, redshift_idx, lmin_idx, lmax_idx,
                  scatter_dex=0.3):
    """
    Compute P(log_lambda_Edd | Lmin < L < Lmax) from the CERDF emulator.

    The CERDF emulator predicts cumulative number densities above a luminosity
    threshold.  The differential distribution in [Lmin, Lmax] is obtained by
    subtracting CERDF(>Lmax) from CERDF(>Lmin), then normalising to a PDF.
    The result is convolved with a Gaussian of width ``scatter_dex`` to model
    observational scatter in BH mass estimates.

    Parameters
    ----------
    params : array-like, shape (n_params,)
        Emulator parameter vector.
    cerdf_emulator : GeneralEmulatorGP
        Trained CERDF emulator with ``axis_data`` containing
        ``"log_bins"`` (eddington-ratio bin centres) and
        ``"log_L_threshold"`` (luminosity thresholds).
    redshift_idx : int
        Index into the emulator's redshift axis.
    lmin_idx : int
        Index into ``log_L_threshold`` for the lower luminosity bound.
    lmax_idx : int
        Index into ``log_L_threshold`` for the upper luminosity bound.
    scatter_dex : float
        Gaussian scatter (sigma) in dex to convolve with the PDF.
        Default 0.3 dex (typical BH mass uncertainty).

    Returns
    -------
    log_bins : ndarray, shape (n_bins,)
        Log-lambda_Edd bin centres (same as emulator grid).
    pdf : ndarray, shape (n_bins,)
        Normalised probability density.  All zeros if the integral vanishes.
    """
    log_bins = cerdf_emulator.axis_data["log_bins"]

    log_cerdfs = cerdf_emulator.predict_mean_only(np.atleast_2d(params))[0]

    cerdf_lmin = 10 ** log_cerdfs[redshift_idx, lmin_idx, :]
    cerdf_lmax = 10 ** log_cerdfs[redshift_idx, lmax_idx, :]
    cerdf_diff = np.clip(cerdf_lmin - cerdf_lmax, 0.0, None)

    integral = np.trapezoid(cerdf_diff, log_bins)
    if integral > 0:
        pdf = cerdf_diff / integral
    else:
        pdf = np.zeros_like(cerdf_diff)

    # Convolve with Gaussian scatter
    pdf = _convolve_pdf(pdf, log_bins, scatter_dex)

    return log_bins, pdf


def precompute_cerdf_inputs(redshifts_data, logL_Bols_data, log_etas_data,
                            cerdf_emulator):
    """
    Precompute per-data-point index lookups for the CERDF likelihood.

    For each observed QSO, finds the closest emulator redshift and the two
    adjacent luminosity thresholds that bracket its logL_Bol.

    Parameters
    ----------
    redshifts_data : array, shape (N,)
        Observed redshifts.
    logL_Bols_data : array, shape (N,)
        Observed log bolometric luminosities (erg/s).
    log_etas_data : array, shape (N,)
        Observed log eddington ratios.
    cerdf_emulator : GeneralEmulatorGP
        Trained CERDF emulator.

    Returns
    -------
    cerdf_cfg : dict
        'redshift_idx'  : int array (N,) -- closest emulator redshift index
        'lmin_idx'      : int array (N,) -- lower bracketing L threshold index
        'lmax_idx'      : int array (N,) -- upper bracketing L threshold index
        'log_etas'      : float array (N,) -- observed log eddington ratios
    """
    emul_redshifts = cerdf_emulator.axis_data["redshift"]
    log_L_thresholds = cerdf_emulator.axis_data["log_L_threshold"]
    log_bins = np.asarray(cerdf_emulator.axis_data["log_bins"])

    redshifts_data = np.asarray(redshifts_data)
    logL_Bols_data = np.asarray(logL_Bols_data)
    log_etas_data = np.asarray(log_etas_data)

    # Each QSO is snapped to the NEAREST emulator redshift slice.  The cells are
    # therefore the Voronoi cells of a NON-UNIFORM grid, so their widths are very
    # uneven (dz = 0.12 at z=0.5 but 1.10 at z=5.02).  The model PDF is evaluated
    # at the single slice redshift for every object in the cell.
    redshift_idx = np.array([int(np.abs(emul_redshifts - z).argmin())
                             for z in redshifts_data])

    # --- Restrict which emulator slices may serve as cells -------------------
    # BAQARO_CERDF_Z_SLICE_MIN (default 1.0): stop using the low-z emulator slices
    # (0.0, 0.26, 0.5, 0.74) as CERDF cells.  Their Voronoi cells are narrow and
    # sparsely populated, with almost no constraining power.
    #
    # The QSO SAMPLE keeps its own z floor (CERDF_Z_MIN, default 0.5): objects
    # below the lowest allowed slice are RE-SNAPPED onto it, not discarded.  The
    # data cut and the model-grid cut are deliberately separate knobs.
    #
    #   BAQARO_CERDF_Z_SLICE_MIN=0  -> use every slice (legacy behaviour)
    #   BAQARO_CERDF_Z_SLICE_DROP=1 -> discard the re-snapped objects instead of
    #                                 keeping them (alternative, off by default)
    #
    # The re-snapped objects (a z=0.5 quasar compared against the z=1.0 model
    # PDF) can be removed instead with BAQARO_CERDF_Z_SLICE_DROP=1.
    z_slice_min = float(os.environ.get("BAQARO_CERDF_Z_SLICE_MIN", "1.0") or 1.0)
    z_slice_drop = os.environ.get(
        "BAQARO_CERDF_Z_SLICE_DROP", "").strip().lower() in ("1", "true", "yes", "on")

    emul_z = np.asarray(emul_redshifts)
    allowed = np.where(emul_z >= z_slice_min)[0]
    if len(allowed) == 0:
        raise ValueError(f"BAQARO_CERDF_Z_SLICE_MIN={z_slice_min} leaves no emulator "
                         f"redshift slices (grid: {np.round(emul_z, 2).tolist()})")

    below = emul_z[redshift_idx] < z_slice_min
    n_below = int(below.sum())
    if n_below:
        if z_slice_drop:
            keep = ~below
            print(f"[cerdf-env] BAQARO_CERDF_Z_SLICE_MIN={z_slice_min} + _DROP=1: "
                  f"{n_below:,} QSOs below the lowest allowed slice REMOVED "
                  f"({int(keep.sum()):,} kept)")
            redshift_idx = redshift_idx[keep]
            redshifts_data = redshifts_data[keep]
            logL_Bols_data = logL_Bols_data[keep]
            log_etas_data = log_etas_data[keep]
        else:
            # Re-snap onto the nearest ALLOWED slice (the data keeps its own
            # z floor; only the model grid is restricted).
            resnap = allowed[np.abs(
                emul_z[allowed][None, :] - redshifts_data[below][:, None]
            ).argmin(axis=1)]
            redshift_idx = redshift_idx.copy()
            redshift_idx[below] = resnap
            z_lo, z_hi = redshifts_data[below].min(), redshifts_data[below].max()
            print(f"[cerdf-env] BAQARO_CERDF_Z_SLICE_MIN={z_slice_min}: emulator "
                  f"slices below {z_slice_min} disabled as cells; {n_below:,} QSOs "
                  f"(z in [{z_lo:.2f}, {z_hi:.2f}]) RE-SNAPPED onto z="
                  f"{emul_z[allowed].min():.2f} (sample z floor unchanged)")

    # For each QSO, bracket its logL_Bol between two adjacent emulator
    # luminosity thresholds.  The differential CERDF in this narrow L bin
    # gives P(log_eta | Lmin < L < Lmax), which approximates
    # P(log_eta | L_bol) when the threshold spacing is fine enough.
    insert_idx = np.searchsorted(log_L_thresholds, logL_Bols_data)

    # Lower bracket: the threshold just below logL (clamped to valid range)
    lmin_idx = np.clip(insert_idx - 1, 0, len(log_L_thresholds) - 2).astype(int)
    # Upper bracket: one threshold above the lower
    lmax_idx = (lmin_idx + 1).astype(int)

    # Many QSOs share the same (z, Lmin, Lmax) cell.  Deduplicating here
    # means the likelihood function only computes ~140 unique PDFs instead
    # of ~293k (one per QSO).  The inverse array maps each QSO back to
    # its unique group.
    keys = np.column_stack([redshift_idx, lmin_idx, lmax_idx])
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    # --- Flexibility knobs for the per-object (unbinned) estimator ----------
    # All three default to the historical behaviour, so an unset environment
    # reproduces the original likelihood bit-for-bit.
    #
    #   BAQARO_CERDF_SCATTER_DEX  (default 0.3)
    #       Gaussian width convolved into the model PDF, representing the
    #       observational error on log_eta (dominated by the virial BH-mass
    #       uncertainty, ~0.3-0.5 dex).  This is the knob that sets how SHARP
    #       the model PDF is, and hence how brittle the per-object product is.
    #
    #   BAQARO_CERDF_OUTLIER_FRAC  (default 0.0 = off)
    #       Robust mixture:  p_i -> (1-f) * p_model,i  +  f * p_uniform
    #       with p_uniform = 1/(eta_max - eta_min) flat over the emulator's
    #       log_eta grid.  This is the principled replacement for `pdf_floor`:
    #       it states "a fraction f of the catalogue is not described by the
    #       model" (bad virial mass, misclassification, a population we don't
    #       simulate) instead of the current hard cliff, where a single object
    #       in a zero-probability region costs ln(1e-30) ~ -69 and can outvote
    #       the entire QLF.  f = 0.01-0.05 is the usual range.
    #
    #   BAQARO_CERDF_TEMP  is applied in main_mcmc.py via LIKELIHOOD_WEIGHTS
    #       (ll_cerdf / T), not here -- see the note in that file.
    scatter_dex = float(os.environ.get("BAQARO_CERDF_SCATTER_DEX", "0.3") or 0.3)
    outlier_frac = float(os.environ.get("BAQARO_CERDF_OUTLIER_FRAC", "0.0") or 0.0)
    if not (0.0 <= outlier_frac < 1.0):
        raise ValueError(f"BAQARO_CERDF_OUTLIER_FRAC must be in [0, 1); got {outlier_frac}")
    if scatter_dex != 0.3:
        print(f"[cerdf-env] BAQARO_CERDF_SCATTER_DEX={scatter_dex}")
    if outlier_frac > 0:
        print(f"[cerdf-env] BAQARO_CERDF_OUTLIER_FRAC={outlier_frac}: robust mixture "
              f"p -> (1-f)*p_model + f*uniform over log_eta in "
              f"[{log_bins[0]:.2f}, {log_bins[-1]:.2f}] (pdf_floor cliff bypassed)")

    return {
        'redshift_idx': redshift_idx,
        'lmin_idx': lmin_idx,
        'lmax_idx': lmax_idx,
        'log_etas': np.asarray(log_etas_data),
        'unique_keys': unique_keys,
        'inverse': inverse,
        # log_bins is REQUIRED by log_likelihood_cerdf_from_prediction (the
        # vectorized MCMC path).
        'log_bins': log_bins,
        'scatter_dex': scatter_dex,
        'outlier_frac': outlier_frac,
    }


def log_likelihood_cerdf(params, cerdf_emulator, cerdf_cfg, scatter_dex=0.3,
                         pdf_floor=1e-30):
    """
    CERDF log-likelihood: product of emulated P(log_eta_i | L_i) over all
    observed QSOs.

    For each data point, the emulated PDF is computed from the differential
    CERDF between the two L thresholds bracketing L_bol, normalised,
    convolved with a Gaussian of width ``scatter_dex``, and evaluated
    (interpolated) at the observed log_eta.

    log L = sum_i  log P(log_eta_i | L_bol_i, z_i, theta)

    Parameters
    ----------
    params : array-like
        Emulator parameter vector.
    cerdf_emulator : GeneralEmulatorGP
    cerdf_cfg : dict
        Output of ``precompute_cerdf_inputs``.
    scatter_dex : float
        Gaussian scatter (sigma) in dex to convolve with the PDF.
        Default 0.3 dex (typical BH mass uncertainty).
    pdf_floor : float
        Minimum PDF value to avoid log(0).  Default 1e-30.  Superseded by the
        robust mixture when ``cerdf_cfg['outlier_frac'] > 0``.
    """
    # Single emulator call: shape (n_z, n_L_thresholds, n_bins)
    log_cerdfs = cerdf_emulator.predict_mean_only(np.atleast_2d(params))[0]
    return log_likelihood_cerdf_from_prediction(
        log_cerdfs, cerdf_cfg, scatter_dex=scatter_dex, pdf_floor=pdf_floor)


def log_likelihood_cerdf_from_prediction(log_cerdfs, cerdf_cfg, scatter_dex=0.3,
                                          pdf_floor=1e-30):
    """
    Per-QSO CERDF log-likelihood from a pre-computed emulator prediction.

    This is the core of the unbinned estimator; ``log_likelihood_cerdf``
    just prepends the emulator call.  ``log_cerdfs`` has shape
    ``(n_z, n_L_thresholds, n_bins)`` — a single sample from
    ``predict_mean_only(...)[0]``.

    ``scatter_dex`` and the robust-mixture fraction are taken from
    ``cerdf_cfg`` when present (set there from BAQARO_CERDF_SCATTER_DEX /
    BAQARO_CERDF_OUTLIER_FRAC), so the vectorized MCMC path — which cannot
    pass keyword arguments through the dispatcher — honours them too.  The
    ``scatter_dex`` argument is the fallback for direct callers.
    """
    log_bins = cerdf_cfg['log_bins']
    n_bins = len(log_bins)

    log_etas = cerdf_cfg['log_etas']
    unique_keys = cerdf_cfg['unique_keys']
    inverse = cerdf_cfg['inverse']

    scatter_dex = cerdf_cfg.get('scatter_dex', scatter_dex)
    outlier_frac = cerdf_cfg.get('outlier_frac', 0.0)

    # --- Vectorized differential CERDF for all unique (z, Lmin, Lmax) groups ---
    # The CERDF emulator predicts cumulative distributions: N(>L | log_eta).
    # The differential distribution in the luminosity bin [Lmin, Lmax] is
    # CERDF(>Lmin) - CERDF(>Lmax), analogous to CDF subtraction.
    # Typically ~62 unique groups out of ~9000 QSOs, so this vectorised
    # approach avoids redundant computation.
    ridx_arr = unique_keys[:, 0].astype(int)
    lmin_arr = unique_keys[:, 1].astype(int)
    lmax_arr = unique_keys[:, 2].astype(int)

    # Shape: (n_groups, n_bins) — convert from log to linear for subtraction
    cerdf_lo = 10 ** log_cerdfs[ridx_arr, lmin_arr, :]
    cerdf_hi = 10 ** log_cerdfs[ridx_arr, lmax_arr, :]
    cerdf_diff = np.clip(cerdf_lo - cerdf_hi, 0.0, None)

    # Integrate each group over log_eta to get normalisation
    integrals = np.trapezoid(cerdf_diff, log_bins, axis=1)
    if np.any(integrals <= 0):
        # A vanishing integral means the model predicts zero quasars in this
        # (z, L) cell — incompatible with the observation.  Return -inf.
        return -np.inf

    # Normalise to proper PDFs: integral P(log_eta) d(log_eta) = 1
    pdfs = cerdf_diff / integrals[:, np.newaxis]

    # --- Batch convolution: model BH mass uncertainty ---
    # Convolve all group PDFs at once along the log_eta axis (axis=1).
    # This is equivalent to marginalising over a Gaussian BH mass error
    # of width scatter_dex, since log_eta = log(L_bol) - log(M_BH) - const
    # and the L_bol error is subdominant.
    if scatter_dex > 0:
        bin_width = np.abs(log_bins[1] - log_bins[0])
        sigma_bins = scatter_dex / bin_width
        pdfs = gaussian_filter1d(pdfs, sigma_bins, axis=1,
                                 mode='constant', cval=0.0)
        # Re-normalise: convolution with zero-padded edges leaks area
        integrals_conv = np.trapezoid(pdfs, log_bins, axis=1)
        good = integrals_conv > 0
        pdfs[good] /= integrals_conv[good, np.newaxis]

    # --- Evaluate PDF at each observed log_eta via linear interpolation ---
    # Instead of calling np.interp per data point, we do manual vectorised
    # interpolation: find the fractional bin index, then lerp between the
    # two bracketing bins.
    bin_start = log_bins[0]
    bin_width = log_bins[1] - log_bins[0]
    frac_idx = (log_etas - bin_start) / bin_width

    frac_idx = np.clip(frac_idx, 0, n_bins - 1 - 1e-10)
    idx_lo = frac_idx.astype(int)
    idx_hi = idx_lo + 1
    w = frac_idx - idx_lo  # interpolation weight

    # Map each data point to its unique group via the inverse index
    group_idx = inverse
    pdf_lo = pdfs[group_idx, idx_lo]
    pdf_hi = pdfs[group_idx, np.minimum(idx_hi, n_bins - 1)]
    pdf_at_data = pdf_lo * (1.0 - w) + pdf_hi * w

    if outlier_frac > 0:
        # Robust mixture: a fraction `outlier_frac` of the catalogue is assumed
        # NOT to be described by the model (bad virial mass, misclassification,
        # a population we don't simulate) and is drawn from a flat log_eta
        # distribution.  This caps the per-object penalty at
        #   ln(f / (eta_max - eta_min))   ~= -6 for f=0.02 over 4 dex
        # instead of the pdf_floor cliff at ln(1e-30) = -69, so a handful of
        # tail objects can no longer outvote every other likelihood component.
        p_bg = 1.0 / (log_bins[-1] - log_bins[0])
        pdf_at_data = (1.0 - outlier_frac) * pdf_at_data + outlier_frac * p_bg

    # Floor prevents log(0) for data points in the tails of the PDF
    pdf_at_data = np.clip(pdf_at_data, pdf_floor, None)
    return np.sum(np.log(pdf_at_data))


# ==============================================================================
# CONDITIONAL EDDINGTON RATIO DISTRIBUTION LIKELIHOOD — BINNED (chi2)
# ==============================================================================

def precompute_cerdf_inputs_binned(redshifts_data, logL_Bols_data, log_etas_data,
                                   cerdf_emulator, logL_bins=None,
                                   log_eta_bins=None, min_count=5,
                                   scatter_dex=0.3, z_centers_use=None):
    """
    Precompute inputs for the binned CERDF likelihood.

    Instead of evaluating P(log_eta) per QSO (~9000 terms), this bins the
    observed QSOs into a (redshift x luminosity) grid, builds a normalised
    histogram of log_eta in each cell, and stores the data needed for a
    chi-squared comparison against the model PDF.

    This makes the effective number of data points comparable to the QLF
    (~100--400 non-empty histogram bins), enabling fair joint inference
    without arbitrary likelihood weighting.

    The data binning is delegated to
    ``obs_data.qso_obs_data_setup.bin_qso_eddington_ratios``.  This function
    adds the emulator-specific index lookups (L threshold indices) and
    convolution parameters needed by the likelihood evaluator.

    Parameters
    ----------
    redshifts_data : array, shape (N,)
        Observed QSO redshifts.
    logL_Bols_data : array, shape (N,)
        Observed log bolometric luminosities (erg/s).
    log_etas_data : array, shape (N,)
        Observed log Eddington ratios.
    cerdf_emulator : GeneralEmulatorGP
        Trained CERDF emulator.
    logL_bins : array or None
        Luminosity bin edges (in log10 erg/s).  Default: [45.5, 46.0, 46.5, 47.0, 47.5].
    log_eta_bins : array or None
        Eddington ratio bin edges for the histogram.  Default: 18 bins
        from -3.0 to 1.5 (width 0.25 dex).
    min_count : int
        Minimum QSO count in a (z, L) cell to include it.
    scatter_dex : float
        Gaussian scatter (sigma) in dex for BH mass uncertainty convolution.  Default: 0.3 dex.             

    Returns
    -------
    cerdf_cfg : dict
        'cells'         : list of per-cell dicts with obs histogram and emulator indices
        'log_eta_centers': array, histogram bin centres
        'log_bins_emul' : array, emulator log_eta grid
        'sigma_bins_emul': float, convolution kernel width in emulator bins
        'scatter_dex'   : float
        'n_data'        : int, total number of non-empty histogram bins
    """
    from baqaro.obs_data.qso_obs_data_setup import bin_qso_eddington_ratios

    # Env-driven overrides for the per-bin error model (sensitivity tests):
    #   BAQARO_CERDF_SYS_FRAC=0.20    # fractional sys floor (default 0.15)
    #   BAQARO_CERDF_SYS_ABS=0.05     # absolute sys floor in PDF density (default 0.20)
    _sys_frac = os.environ.get("BAQARO_CERDF_SYS_FRAC", "").strip()
    _sys_abs  = os.environ.get("BAQARO_CERDF_SYS_ABS",  "").strip()
    _bin_kw = {}
    if _sys_frac:
        _bin_kw["sys_frac"] = float(_sys_frac)
        print(f"[cerdf-env] BAQARO_CERDF_SYS_FRAC={_bin_kw['sys_frac']}")
    if _sys_abs:
        _bin_kw["sys_abs"] = float(_sys_abs)
        print(f"[cerdf-env] BAQARO_CERDF_SYS_ABS={_bin_kw['sys_abs']}")

    emul_redshifts = cerdf_emulator.axis_data["redshift"]
    log_L_thresholds = cerdf_emulator.axis_data["log_L_threshold"]
    log_bins_emul = cerdf_emulator.axis_data["log_bins"]

    # --- Step 1: Bin observed data (pure data preparation) ---
    cells_raw, log_eta_bins_out, log_eta_centers, d_eta = bin_qso_eddington_ratios(
        redshifts_data, logL_Bols_data, log_etas_data,
        z_centers=emul_redshifts,
        logL_bins=logL_bins,
        log_eta_bins=log_eta_bins,
        min_count=min_count,
        **_bin_kw,
    )

    # --- Step 2: Add emulator-specific index lookups ---
    # For each cell, find the closest emulator L threshold indices for the
    # luminosity bin edges.  These are used in the likelihood to compute the
    # differential CERDF: CERDF(>Lmin) - CERDF(>Lmax).
    cells = []
    for cell in cells_raw:
        lmin_idx = int(np.abs(log_L_thresholds - cell['lmin_val']).argmin())
        lmax_idx = int(np.abs(log_L_thresholds - cell['lmax_val']).argmin())
        cell['lmin_idx'] = lmin_idx
        cell['lmax_idx'] = lmax_idx
        cells.append(cell)

    # Convolution kernel width for the emulator grid
    bin_width_emul = np.abs(log_bins_emul[1] - log_bins_emul[0])
    sigma_bins_emul = scatter_dex / bin_width_emul if scatter_dex > 0 else 0.0

    # ---- Cell z-center filter (post-build) ----
    # Resolution order (highest priority first):
    #   1. BAQARO_CERDF_Z_USE env var (per-run override). Use the literal value
    #      "all" or "*" to DISABLE the default filter (recover legacy behaviour
    #      of using every emulator z slice as a cell center).
    #   2. The `z_centers_use` argument passed by the caller.
    #   3. PRODUCTION DEFAULT: the 6 cell centers
    #      [1.0, 2.0, 3.0, 3.94, 5.02, 6.14] — integer z plus the closest
    #      emulator slices at z=4, 5, 6. Dropping the BASE binning's
    #      additional z={0.26, 0.5, 0.74, 1.5, 2.5, 7.31} cells eliminates
    #      QSO double-counting in adjacent ±0.5 windows (BASE had every cell
    #      ≤0.5 apart from another, so each z<3 QSO was binned into 2-3
    #      cells), without losing any z range covered by the SDSS sample.
    _env_zuse = os.environ.get("BAQARO_CERDF_Z_USE", "").strip()
    if _env_zuse:
        if _env_zuse.lower() in ("all", "*"):
            print(f"[cerdf-env] BAQARO_CERDF_Z_USE={_env_zuse} -> z filter DISABLED "
                  f"(keeping all {len(cells)} cells on the emulator grid)")
            z_centers_use = None   # explicit disable
        else:
            z_centers_use = [float(s) for s in _env_zuse.split(",") if s.strip()]
            print(f"[cerdf-env] BAQARO_CERDF_Z_USE={z_centers_use}")
    elif z_centers_use is None:
        # Production default policy.
        z_centers_use = [1.0, 2.0, 3.0, 3.94, 5.02, 6.14]

    if z_centers_use is not None:
        before = len(cells)
        cells = [c for c in cells if any(abs(c['z_center'] - z) < 0.3 for z in z_centers_use)]
        print(f"[cerdf] z_centers_use={z_centers_use} -> {len(cells)}/{before} cells kept")

    # ---- Optional hybrid Poisson/Gaussian low-count treatment ----
    # SEPARATE, opt-in pipeline (default OFF -> bit-identical to the legacy
    # pure-Gaussian chi^2). When ON, each histogram bin with an observed count
    # BELOW `poisson_threshold` (including empty bins) is scored with the
    # Poisson Cash log-likelihood on counts instead of the Gaussian-on-PDF term,
    # mirroring the QLF hybrid (Cash 1979). Empty bins then contribute a real
    # -lambda penalty (model predicting QSOs where none are observed) rather
    # than being silently dropped. See `log_likelihood_cerdf_binned`.
    poisson_lown = os.environ.get(
        "BAQARO_CERDF_POISSON_LOWN", "").strip().lower() in ("1", "true", "yes", "on")
    poisson_threshold = int(os.environ.get("BAQARO_CERDF_POISSON_THRESHOLD", "5"))

    # ---- Optional bin-integrated model measure ----
    # OPT-IN, default OFF -> bit-identical to the legacy point-sampled model.
    # The data side is a bin-AVERAGED histogram density normalized as
    # sum(p_i * dEta_i) = 1 over the histogram bins; the legacy model side is
    # POINT-sampled at bin centers and trapezoid-renormalized over the
    # CENTERS' span (one bin width narrower). With the production 1.0-dex
    # bigeta bins the per-bin mismatch is 15-40% for realistic lognormal
    # shapes — comparable to the sys_abs=0.20 floor. When ON, the model PDF
    # is bin-averaged over each histogram bin and normalized with the data's
    # own measure. CHANGES THE LIKELIHOOD -> tag chains via BAQARO_MCMC_NOTES.
    bin_integrated = os.environ.get(
        "BAQARO_CERDF_BIN_INTEGRATED", "").strip().lower() in ("1", "true", "yes", "on")
    if bin_integrated:
        print("[cerdf-env] BAQARO_CERDF_BIN_INTEGRATED=1: model PDF bin-AVERAGED "
              "over histogram bins with sum(p*dEta)=1 normalization. "
              "CHANGES the likelihood - tag chains via BAQARO_MCMC_NOTES.")

    n_data = sum(int(np.sum(c['valid_mask'])) for c in cells)
    n_cells = len(cells)
    n_qsos = sum(c['N_total'] for c in cells)

    if poisson_lown:
        n_lown = sum(int(np.sum((c['counts'] > 0) & (c['counts'] < poisson_threshold)))
                     for c in cells)
        n_empty = sum(int(np.sum(c['counts'] == 0)) for c in cells)
        print(f"[cerdf-env] BAQARO_CERDF_POISSON_LOWN=1 threshold={poisson_threshold}: "
              f"{n_lown} low-count bins + {n_empty} empty bins -> Poisson/Cash "
              f"(rest Gaussian)")

    print(f"CERDF binned: {n_cells} cells, {n_data} non-empty histogram bins, "
          f"{n_qsos} QSOs total")
    for c in cells:
        n_valid = int(np.sum(c['valid_mask']))
        print(f"  z={c['z_center']:.1f}, logL=[{c['lmin_val']:.1f},{c['lmax_val']:.1f}]: "
              f"{c['N_total']} QSOs, {n_valid} non-empty bins")

    return {
        'cells': cells,
        'log_eta_bins': log_eta_bins_out,
        'log_eta_centers': log_eta_centers,
        'd_eta': d_eta,
        'log_bins_emul': log_bins_emul,
        'sigma_bins_emul': sigma_bins_emul,
        'scatter_dex': scatter_dex,
        'n_data': n_data,
        'poisson_lown': poisson_lown,
        'poisson_threshold': poisson_threshold,
        'bin_integrated': bin_integrated,
        # Precomputed bin-mass projection (integrated mode only): the
        # grid/edges pair is fixed for the whole run, so the extraction is
        # a single mat-vec in the MCMC hot loop.
        'bin_proj': (_bin_mass_projection(log_bins_emul, log_eta_bins_out)
                     if bin_integrated else None),
    }


def _bin_mass_projection(log_bins_emul, log_eta_bins):
    """Fixed linear operator P with ``bin_masses = P @ pdf_model``.

    Encodes the cumulative-trapezoid bin-mass extraction of
    ``_model_pdf_on_obs_bins`` (integrated mode) for a FIXED (emulator grid,
    histogram edges) pair, built by probing the exact same arithmetic with
    unit vectors — so ``P @ pdf`` is identical (to float roundoff) to the
    direct computation, but a single small mat-vec at runtime (the grid and
    edges never change during MCMC). Precomputed once in
    ``precompute_cerdf_inputs_binned``.
    """
    n_grid = len(log_bins_emul)
    n_bins = len(log_eta_bins) - 1
    P = np.empty((n_bins, n_grid), dtype=np.float64)
    unit = np.zeros(n_grid, dtype=np.float64)
    for j in range(n_grid):
        unit[j] = 1.0
        F = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (unit[1:] + unit[:-1]) * np.diff(log_bins_emul)),
        ))
        P[:, j] = np.diff(np.interp(log_eta_bins, log_bins_emul, F))
        unit[j] = 0.0
    return P


def _model_pdf_on_obs_bins(pdf_model, log_bins_emul, log_eta_centers,
                           log_eta_bins, bin_integrated, bin_proj=None):
    """Project the model PDF (on the emulator eta grid) onto the obs bins.

    Legacy mode (``bin_integrated=False``, DEFAULT — bit-identical to the
    original inline code): point-sample the PDF at the histogram bin
    CENTERS and re-normalize with a trapezoid over the centers' span. This
    compares point values against the data's bin-AVERAGED density and
    normalizes over a domain one bin narrower than the histogram.

    Bin-integrated mode (``BAQARO_CERDF_BIN_INTEGRATED=1``): bin-AVERAGE the
    model PDF over each histogram bin (cumulative-trapezoid mass / width)
    and normalize with the data's own measure, sum(p_i * dEta_i) = 1 over
    the histogram bins — the same density definition as
    ``obs_pdf = counts / (N_total * d_eta)``. Handles non-uniform bin
    widths; bins outside the emulator grid get zero mass.
    """
    if not bin_integrated:
        model_at_centers = np.interp(log_eta_centers, log_bins_emul, pdf_model)
        model_integral = np.trapezoid(model_at_centers, log_eta_centers)
        if model_integral > 0:
            model_at_centers /= model_integral
        return model_at_centers

    if bin_proj is not None:
        # Fast path (MCMC hot loop): the bin-mass extraction is linear in
        # pdf_model, so it is a single precomputed mat-vec.
        masses = bin_proj @ pdf_model
    else:
        # Direct path (standalone callers / tests): cumulative mass F(eta)
        # on the emulator grid (trapezoid), then per-bin mass =
        # F(edge_hi) - F(edge_lo); np.interp clamps outside the grid so
        # out-of-grid bins get zero mass.
        F = np.concatenate((
            [0.0],
            np.cumsum(0.5 * (pdf_model[1:] + pdf_model[:-1]) * np.diff(log_bins_emul)),
        ))
        masses = np.diff(np.interp(log_eta_bins, log_bins_emul, F))
    widths = np.diff(log_eta_bins)
    model_avg = masses / widths
    total = float(masses.sum())
    if total > 0:
        model_avg = model_avg / total   # => sum(model_avg * widths) == 1
    return model_avg


def _cerdf_cell_loglike(cell, model_at_centers, poisson_lown, poisson_threshold):
    """Per-cell CERDF log-likelihood contribution.

    ``model_at_centers`` is the model PDF (density) evaluated at the histogram
    bin centres, already normalised over the histogram domain.

    Default (``poisson_lown=False``): Gaussian chi-squared on the normalised
    PDF over non-empty bins, returned as ``-0.5 * chi2``. Bit-identical to the
    legacy path.

    Hybrid (``poisson_lown=True``): bins with observed count
    ``>= poisson_threshold`` keep the Gaussian + systematic-floor treatment;
    bins with count ``< poisson_threshold`` (INCLUDING empty bins) use the
    Poisson Cash log-likelihood on counts, ``n*ln(lam) - lam - ln(n!)``, with
    ``lam_i = N_total * p_i`` (``p_i`` the normalised model fraction per bin).
    The empty-bin ``-lam`` term penalises model density where no QSOs are
    observed (the pure-Gaussian path drops those bins entirely). Same
    convention as the QLF hybrid (`_log_likelihood_qlf_poisson_from_prediction`).
    """
    obs_pdf = cell['obs_pdf']
    sigma_pdf = cell['sigma_pdf']
    valid = cell['valid_mask']

    if not poisson_lown:
        resid = (obs_pdf[valid] - model_at_centers[valid]) / sigma_pdf[valid]
        return -0.5 * np.sum(resid ** 2)

    counts = cell['counts']
    N_total = cell['N_total']

    # Expected counts per bin: lam_i = N_total * p_i, p_i = model fraction.
    model_sum = model_at_centers.sum()
    p = model_at_centers / model_sum if model_sum > 0 else model_at_centers
    lam = np.maximum(N_total * p, 1e-30)

    lown = counts < poisson_threshold      # includes empty bins (count 0)
    high = valid & ~lown                    # well-populated -> Gaussian + floor

    ll = 0.0
    if high.any():
        resid = (obs_pdf[high] - model_at_centers[high]) / sigma_pdf[high]
        ll -= 0.5 * np.sum(resid ** 2)
    if lown.any():
        n = counts[lown].astype(float)
        ll += np.sum(n * np.log(lam[lown]) - lam[lown] - gammaln(n + 1))
    return ll


def log_likelihood_cerdf_binned(params, cerdf_emulator, cerdf_cfg,
                                verbose=False):
    """
    Binned CERDF log-likelihood: chi-squared on observed vs model histograms.

    For each (redshift, luminosity) cell, the model PDF is obtained from the
    differential CERDF, convolved with BH mass scatter, interpolated onto the
    histogram bin centres, and compared to the observed histogram via chi2.

    Parameters
    ----------
    params : array-like
        Emulator parameter vector.
    cerdf_emulator : GeneralEmulatorGP
    cerdf_cfg : dict
        Output of ``precompute_cerdf_inputs_binned``.
    verbose : bool
        If True, print per-cell diagnostics.

    Returns
    -------
    float
        Log-likelihood value (-0.5 * chi2_total), or -inf if any cell has
        a vanishing model integral (model predicts zero QSOs in that cell).
    """
    log_cerdfs = cerdf_emulator.predict_mean_only(np.atleast_2d(params))[0]

    log_bins_emul = cerdf_cfg['log_bins_emul']
    sigma_bins_emul = cerdf_cfg['sigma_bins_emul']
    log_eta_centers = cerdf_cfg['log_eta_centers']
    poisson_lown = cerdf_cfg.get('poisson_lown', False)
    poisson_threshold = cerdf_cfg.get('poisson_threshold', 5)

    ll_total = 0.0

    for cell in cerdf_cfg['cells']:
        i_z = cell['i_z']
        lmin_idx = cell['lmin_idx']
        lmax_idx = cell['lmax_idx']

        # Differential CERDF: CERDF(>Lmin) - CERDF(>Lmax)
        cerdf_lmin = 10.0 ** log_cerdfs[i_z, lmin_idx, :]
        cerdf_lmax = 10.0 ** log_cerdfs[i_z, lmax_idx, :]
        cerdf_diff = np.clip(cerdf_lmin - cerdf_lmax, 0.0, None)

        integral = np.trapezoid(cerdf_diff, log_bins_emul)
        if integral <= 0:
            if verbose:
                print(f"  z={cell['z_center']:.1f} L=[{cell['lmin_val']:.1f},{cell['lmax_val']:.1f}]: "
                      f"integral=0 -> -inf")
            return -np.inf

        pdf_model = cerdf_diff / integral

        # Convolve with BH mass uncertainty
        if sigma_bins_emul > 0:
            pdf_model = gaussian_filter1d(pdf_model, sigma_bins_emul,
                                          mode='constant', cval=0.0)
            integral_conv = np.trapezoid(pdf_model, log_bins_emul)
            if integral_conv > 0:
                pdf_model /= integral_conv

        # Project the model PDF onto the obs histogram bins. Default: legacy
        # point-sample-at-centers + trapezoid renorm; opt-in
        # BAQARO_CERDF_BIN_INTEGRATED=1 bin-averages with the data's own
        # sum(p*dEta)=1 measure — see _model_pdf_on_obs_bins.
        model_at_centers = _model_pdf_on_obs_bins(
            pdf_model, log_bins_emul, log_eta_centers,
            cerdf_cfg['log_eta_bins'], cerdf_cfg.get('bin_integrated', False),
            bin_proj=cerdf_cfg.get('bin_proj'))

        cell_ll = _cerdf_cell_loglike(cell, model_at_centers,
                                      poisson_lown, poisson_threshold)
        ll_total += cell_ll

        if verbose:
            n_valid = int(np.sum(cell['valid_mask']))
            print(f"  z={cell['z_center']:.1f} L=[{cell['lmin_val']:.1f},{cell['lmax_val']:.1f}]: "
                  f"N={cell['N_total']}, n_bins={n_valid}, cell_ll={cell_ll:.2f}, "
                  f"-2ll/nbin={-2*cell_ll/max(n_valid,1):.2f}")

    if verbose:
        n_data = cerdf_cfg['n_data']
        print(f"  TOTAL: -2ll={-2*ll_total:.1f}, n_data={n_data}, "
              f"-2ll/n={-2*ll_total/max(n_data,1):.2f}, ll={ll_total:.1f}")

    return ll_total


def log_likelihood_cerdf_binned_from_prediction(log_cerdfs, cerdf_cfg):
    """
    Binned CERDF log-likelihood from a pre-computed emulator prediction.

    Identical to ``log_likelihood_cerdf_binned`` but skips the emulator call.
    ``log_cerdfs`` has shape ``(n_z, n_L_thresholds, n_bins)``.
    """
    log_bins_emul = cerdf_cfg['log_bins_emul']
    sigma_bins_emul = cerdf_cfg['sigma_bins_emul']
    log_eta_centers = cerdf_cfg['log_eta_centers']
    poisson_lown = cerdf_cfg.get('poisson_lown', False)
    poisson_threshold = cerdf_cfg.get('poisson_threshold', 5)

    ll_total = 0.0

    for cell in cerdf_cfg['cells']:
        i_z = cell['i_z']
        lmin_idx = cell['lmin_idx']
        lmax_idx = cell['lmax_idx']

        cerdf_lmin = 10.0 ** log_cerdfs[i_z, lmin_idx, :]
        cerdf_lmax = 10.0 ** log_cerdfs[i_z, lmax_idx, :]
        cerdf_diff = np.clip(cerdf_lmin - cerdf_lmax, 0.0, None)

        integral = np.trapezoid(cerdf_diff, log_bins_emul)
        if integral <= 0:
            return -np.inf

        pdf_model = cerdf_diff / integral

        if sigma_bins_emul > 0:
            pdf_model = gaussian_filter1d(pdf_model, sigma_bins_emul,
                                          mode='constant', cval=0.0)
            integral_conv = np.trapezoid(pdf_model, log_bins_emul)
            if integral_conv > 0:
                pdf_model /= integral_conv

        model_at_centers = _model_pdf_on_obs_bins(
            pdf_model, log_bins_emul, log_eta_centers,
            cerdf_cfg['log_eta_bins'], cerdf_cfg.get('bin_integrated', False),
            bin_proj=cerdf_cfg.get('bin_proj'))

        ll_total += _cerdf_cell_loglike(cell, model_at_centers,
                                        poisson_lown, poisson_threshold)

    return ll_total


def log_probability_cerdf_binned(params, cerdf_emulator, cerdf_cfg):
    """
    Prior + binned CERDF likelihood.

    Drop-in replacement for ``log_probability_cerdf`` when using the
    binned likelihood mode.
    """
    param_ranges = cerdf_emulator.param_ranges

    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll = log_likelihood_cerdf_binned(params, cerdf_emulator, cerdf_cfg)
    return ll, 0.0


# ==============================================================================
# CORRELATION FUNCTION LIKELIHOOD
# ==============================================================================

def precompute_corr_inputs(datas_corr, len_rbins=101, len_mbins=51):
    """
    Precompute halo-model ingredients for the correlation function likelihood.

    For each redshift bin, this computes the two-halo "triangle" matrix
    (xi_hh(r, M1, M2) integrated over one mass axis to give xi(r, M))
    and the halo mass function (HMF).  These are expensive (~seconds each)
    but depend only on cosmology and redshift, not on the BH model
    parameters, so they are computed once before MCMC.

    At each likelihood evaluation, only the *quasar* halo mass function
    (QHMF, which depends on the BH model) needs to be recomputed and
    convolved with the precomputed triangle.

    Parameters
    ----------
    datas_corr : list
        List of data objects (one per redshift bin). Each must have:
          .redshift : float
          .corr_type : "auto" or "cross"
          .logM_min, .logM_max : float (halo mass range for the grid)
    len_rbins : int
        Number of radial bins for the triangle grid (default 101).
    len_mbins : int
        Number of halo mass bins for the triangle grid (default 51).

    Returns
    -------
    precomputed_inputs : list
        List of dicts (same length as datas_corr), each containing:
          'log_m_axis', 'rbins', 'rpbins', 'mf_fit', 'triangle_fit',
          'corr_type', 'log_L_threshold'
        Entry is None where data_corr is None.
    """
    # Cached, portable triangle/HMF inputs (precomputed by
    # obs_data/build_corr_inputs.py; loaded via qhtools). Replaces the
    # swift_qso_model fitting path — no raw sim data needed at runtime.
    from baqaro.obs_data.corr_inputs_loader import (
        get_input_quantities_auto,
        get_input_quantities_cross,
    )

    precomputed_inputs = []

    for data_corr in datas_corr:
        
        if data_corr is None:
            precomputed_inputs.append(None)
            continue
        
        corr_type = data_corr.corr_type
        redshift = data_corr.redshift
        logM_min = data_corr.logM_min
        logM_max = data_corr.logM_max

        if corr_type == "cross":
            get_input_fn = get_input_quantities_cross
        else:
            get_input_fn = get_input_quantities_auto

        log_m_axis, rbins, rpbins, mf_fit, triangle_fit = get_input_fn(
            redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
            log_M_min=logM_min, log_M_max=logM_max,
        )

        precomputed_inputs.append({
            'log_m_axis': log_m_axis,
            'rbins': rbins,
            'rpbins': rpbins,
            'mf_fit': mf_fit,
            'triangle_fit': triangle_fit,
            'corr_type': corr_type,
            'log_L_threshold': my_utils.to_ergs(data_corr.log_L_threshold),
        })

    return precomputed_inputs


def log_likelihood_corr(params, datas_corr, qhmf_emulator, precomputed_inputs,
                        verbose=False):
    """
    Correlation function log-likelihood using the emulated QHMF.

    The model correlation function is computed from the halo model:
    the precomputed halo-halo correlation "triangle" is weighted by the
    quasar halo mass function (QHMF, from the emulator) to produce the
    quasar auto-correlation (wp/rp) or quasar-galaxy cross-correlation
    (volume-averaged xi).

    For auto-correlation, the pipeline is:
      QHMF -> triangle weighting -> 3D xi(r) -> projected wp(rp) -> wp/rp

    For cross-correlation, a simple galaxy HMF proxy is used:
      qhmf_gal = mf_fit / 5.0 with a hard cutoff at log M_halo < 10.7.
      The galaxy and quasar QHMFs weight opposite axes of the triangle.

    Parameters
    ----------
    params : array-like
        Emulator parameter vector.
    datas_corr : list
        List of data objects (one per redshift bin). Each must have:
          .x, .data, .err, .pimax, .log_L_threshold
          .corr_type : "auto" or "cross"
    qhmf_emulator : GeneralEmulatorGP
        Trained QHMF emulator with axis_data["log_bins"] and
        axis_data["log_L_threshold"].
    precomputed_inputs : list
        Output of ``precompute_corr_inputs``.
    verbose : bool
        If True, print per-redshift and per-bin diagnostics.

    Returns
    -------
    float
        Log-likelihood value (-0.5 * chi2_total), or -inf if the QHMF
        is entirely at floor values (model produces no quasars).
    """
    chi2 = 0.0

    # 1. Predict QHMF once
    log_qhmf_emul = qhmf_emulator.predict_mean_only(np.atleast_2d(params))[0]

    log_L_thresholds = qhmf_emulator.axis_data["log_L_threshold"]
    log_mbins_qhmf = qhmf_emulator.axis_data["log_bins"]

    # 2. Loop over redshift bins
    for i_z, data_corr in enumerate(datas_corr):
        if data_corr is None:
            continue

        inputs = precomputed_inputs[i_z]
        log_m_axis = inputs['log_m_axis']
        rbins = inputs['rbins']
        rpbins = inputs['rpbins']
        mf_fit = inputs['mf_fit']
        triangle_fit = inputs['triangle_fit']
        corr_type = inputs['corr_type']
        log_L_threshold_data = inputs['log_L_threshold']

        # Find threshold index — nearest neighbour on the emulator's DISCRETE
        # log_L_threshold grid (no interpolation between thresholds).
        #
        # Guard the snap: `argmin` ALWAYS returns
        # something, so a dataset whose threshold falls outside the emulator grid
        # would silently CLAMP to the nearest edge and be fitted at the wrong
        # luminosity cut, with no warning. The current three corr datasets all land
        # within 0.05 dex of a grid point, so this never fires today — but a new
        # dataset (or a retrained emulator with a different grid) would hit it
        # silently. Half the grid spacing is the largest snap that can ever be
        # legitimate.
        threshold_idx = int(np.abs(log_L_thresholds - log_L_threshold_data).argmin())
        _snap_gap = abs(log_L_thresholds[threshold_idx] - log_L_threshold_data)
        _grid_step = (np.max(np.diff(log_L_thresholds))
                      if len(log_L_thresholds) > 1 else np.inf)
        if _snap_gap > 0.5 * _grid_step:
            raise ValueError(
                f"corr dataset {getattr(data_corr, 'label', '?')!r} has "
                f"log_L_threshold={log_L_threshold_data:.3f}, which is "
                f"{_snap_gap:.3f} dex from the nearest emulator threshold "
                f"{log_L_thresholds[threshold_idx]:.3f} — more than half the grid "
                f"step ({_grid_step:.3f}). It would be silently clamped to the edge "
                f"and fitted at the WRONG luminosity cut. Emulator grid: "
                f"[{log_L_thresholds[0]:.2f}, {log_L_thresholds[-1]:.2f}].")
        # print(f"Redshift {data_corr.redshift}: Using log_L_threshold={log_L_thresholds[threshold_idx]:.2f} (data threshold={log_L_threshold_data:.2f})")

        # Interpolate the emulator QHMF (on its own mass grid) onto the
        # triangle's mass grid.  Out-of-range masses get -10.0 (effectively
        # zero in linear space), which is safe for the triangle weighting.
        log_qhmf_slice = log_qhmf_emul[i_z, threshold_idx, :]
        log_qhmf_interp = np.interp(log_m_axis, log_mbins_qhmf, log_qhmf_slice, left=-10.0, right=-10.0)

        # If the QHMF is entirely at floor values, the model predicts zero
        # quasars above this luminosity threshold — no correlation signal
        # is possible, so reject immediately.
        if np.all(log_qhmf_interp <= -9.5):
            return -np.inf


        if corr_type == "auto":
            log_rbins, xi = get_corr_from_triangle(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis,
                qhmf=10 ** log_qhmf_interp,
            )
            wp = get_projected_wp(
                rpbins, xi, 10 ** log_rbins, pimax=data_corr.pimax,
            )
            wp_rp = wp / rpbins
            cell_chi2 = get_chi2(data_corr, rpbins, wp_rp)
            chi2 += cell_chi2

            if verbose:
                # Interpolate model onto data rp bins for per-bin comparison
                model_at_data = np.interp(data_corr.x, rpbins, wp_rp)
                n_bins = len(data_corr.x)
                resid = (data_corr.data - model_at_data) / data_corr.err
                worst = np.argmax(np.abs(resid))
                print(f"  z={data_corr.redshift:.1f} (auto): n_bins={n_bins}, "
                      f"chi2={cell_chi2:.1f}, chi2/nbin={cell_chi2/max(n_bins,1):.2f}")
                print(f"    worst bin: rp={data_corr.x[worst]:.2f}, "
                      f"obs={data_corr.data[worst]:.4f}, mod={model_at_data[worst]:.4f}, "
                      f"err={data_corr.err[worst]:.4f}, resid={resid[worst]:.2f}")

        elif corr_type == "cross":
            qhmf_gal = np.copy(mf_fit) / 5.0
            # Hard galaxy halo-mass cut (the only physically-meaningful part of
            # `qhmf_gal` since `_compute_mass_weights` normalises away any scalar
            # multiplier). Default 10.7 dex; env override BAQARO_CORR_GAL_LOGM_MIN.
            _gal_logM_min = float(os.environ.get("BAQARO_CORR_GAL_LOGM_MIN", "10.7"))
            qhmf_gal[log_m_axis < _gal_logM_min] = 0.0

            log_rbins, xi_cross = get_corr_from_triangle_cross(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis,
                qhmf1=10 ** log_qhmf_interp, qhmf2=qhmf_gal,
            )
            xi_vol = get_volume_averaged_xi(
                rpbins, xi_cross, 10 ** log_rbins, pimax=data_corr.pimax,
            )
            # x-axis: geometric (log) centers of the rpbins output edges —
            # the convention of the original clustering code's
            # get_volume_averaged_xi first output.
            rcross_bins = 10 ** (0.5 * (np.log10(rpbins[:-1]) + np.log10(rpbins[1:])))
            cell_chi2 = get_chi2(data_corr, rcross_bins, xi_vol)
            chi2 += cell_chi2

            if verbose:
                model_at_data = np.interp(data_corr.x, rcross_bins, xi_vol)
                n_bins = len(data_corr.x)
                resid = (data_corr.data - model_at_data) / data_corr.err
                worst = np.argmax(np.abs(resid))
                print(f"  z={data_corr.redshift:.1f} (cross): n_bins={n_bins}, "
                      f"chi2={cell_chi2:.1f}, chi2/nbin={cell_chi2/max(n_bins,1):.2f}")
                print(f"    worst bin: rp={data_corr.x[worst]:.2f}, "
                      f"obs={data_corr.data[worst]:.4f}, mod={model_at_data[worst]:.4f}, "
                      f"err={data_corr.err[worst]:.4f}, resid={resid[worst]:.2f}")

    if verbose:
        print(f"  TOTAL: chi2={chi2:.1f}, ll={-0.5*chi2:.1f}")

    return -0.5 * chi2


def log_likelihood_corr_from_prediction(log_qhmf_emul, datas_corr,
                                         qhmf_emulator, precomputed_inputs):
    """
    Correlation log-likelihood from a pre-computed emulator prediction.

    Identical to ``log_likelihood_corr`` but skips the emulator call.
    ``log_qhmf_emul`` has shape ``(n_z, n_L_thresholds, n_m_bins)``.
    """
    chi2 = 0.0

    log_L_thresholds = qhmf_emulator.axis_data["log_L_threshold"]
    log_mbins_qhmf = qhmf_emulator.axis_data["log_bins"]

    for i_z, data_corr in enumerate(datas_corr):
        if data_corr is None:
            continue

        inputs = precomputed_inputs[i_z]
        log_m_axis = inputs['log_m_axis']
        rbins = inputs['rbins']
        rpbins = inputs['rpbins']
        mf_fit = inputs['mf_fit']
        triangle_fit = inputs['triangle_fit']
        corr_type = inputs['corr_type']
        log_L_threshold_data = inputs['log_L_threshold']

        threshold_idx = np.abs(log_L_thresholds - log_L_threshold_data).argmin()
        log_qhmf_slice = log_qhmf_emul[i_z, threshold_idx, :]
        log_qhmf_interp = np.interp(log_m_axis, log_mbins_qhmf, log_qhmf_slice,
                                     left=-10.0, right=-10.0)

        if np.all(log_qhmf_interp <= -9.5):
            return -np.inf

        if corr_type == "auto":
            log_rbins, xi = get_corr_from_triangle(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis,
                qhmf=10 ** log_qhmf_interp,
            )
            wp = get_projected_wp(
                rpbins, xi, 10 ** log_rbins, pimax=data_corr.pimax,
            )
            wp_rp = wp / rpbins
            chi2 += get_chi2(data_corr, rpbins, wp_rp)

        elif corr_type == "cross":
            qhmf_gal = np.copy(mf_fit) / 5.0
            # Hard galaxy halo-mass cut (the only physically-meaningful part of
            # `qhmf_gal` since `_compute_mass_weights` normalises away any scalar
            # multiplier). Default 10.7 dex; env override BAQARO_CORR_GAL_LOGM_MIN.
            _gal_logM_min = float(os.environ.get("BAQARO_CORR_GAL_LOGM_MIN", "10.7"))
            qhmf_gal[log_m_axis < _gal_logM_min] = 0.0

            log_rbins, xi_cross = get_corr_from_triangle_cross(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis,
                qhmf1=10 ** log_qhmf_interp, qhmf2=qhmf_gal,
            )
            xi_vol = get_volume_averaged_xi(
                rpbins, xi_cross, 10 ** log_rbins, pimax=data_corr.pimax,
            )
            rcross_bins = 10 ** (0.5 * (np.log10(rpbins[:-1]) + np.log10(rpbins[1:])))
            chi2 += get_chi2(data_corr, rcross_bins, xi_vol)

    return -0.5 * chi2


# ==============================================================================
# PRIOR + COMBINED PROBABILITY
# ==============================================================================

def log_prior_uniform(params, param_ranges):
    """
    Flat (uniform) prior: returns 0 if all params are inside their bounds
    (INCLUSIVE of the edges), -inf otherwise.

    The bounds ARE the emulator's training box, so a point sitting exactly on an
    edge is inside its support — not an extrapolation. Inclusive bounds also
    give DE's `polish=True` and L-BFGS-B — which legitimately evaluate AT their
    bounds — a finite value there; a strict inequality would exclude only a
    measure-zero set, so no posterior volume moves.

    Parameters
    ----------
    params : array-like, shape (ndim,)
        Current parameter vector.
    param_ranges : list of (float, float)
        Prior bounds ``(low, high)`` for each parameter.

    Returns
    -------
    float
        0.0 if in bounds, -np.inf otherwise.
    """
    # INCLUSIVE at the bounds (`<=`), not strict `<`: `param_ranges` IS the
    # emulator's training box, so a point sitting exactly ON an edge is inside
    # the emulator's support and must not be -inf (DE `polish=True` and
    # L-BFGS-B legitimately evaluate AT their bounds).
    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):
            return -np.inf
    return 0.0


# --- emcee-compatible wrappers ---
# Each ``log_probability_*`` function wraps a likelihood with a prior check
# and returns a ``(log_prob, blob)`` tuple as required by ``emcee``.  The
# blob is always 0.0 (placeholder, unused).  The prior check is done first
# (fast integer comparisons) to avoid the expensive GP emulator call when
# the proposal is out of bounds.


def log_probability_qlf(params, datas_qlf, qlf_emulator, qlf_cfg):
    """
    Prior + Gaussian QLF likelihood.

    Used as the ``log_prob`` callable for ``emcee.EnsembleSampler`` when
    ``QLF_LIKELIHOOD_MODE == "gaussian"``.

    Returns
    -------
    tuple (float, float)
        (log_posterior, blob).  blob is always 0.0.
    """
    param_ranges = qlf_emulator.param_ranges

    # EARLY EXIT: bounds check before expensive GP prediction
    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll = log_likelihood_qlf(params, datas_qlf, qlf_emulator, qlf_cfg)
    return ll, 0.0


def log_probability_qlf_poisson(params, datas_qlf, qlf_emulator, qlf_cfg):
    """
    Prior + hybrid Poisson/Gaussian QLF likelihood.

    Drop-in replacement for ``log_probability_qlf`` when
    ``QLF_LIKELIHOOD_MODE == "poisson"``.
    """
    param_ranges = qlf_emulator.param_ranges

    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll = log_likelihood_qlf_poisson(params, datas_qlf, qlf_emulator, qlf_cfg)
    return ll, 0.0


def log_probability_cerdf(params, cerdf_emulator, cerdf_cfg):
    """
    Prior + CERDF likelihood.

    Used when running CERDF-only inference.  The prior is read from the
    emulator's stored ``param_ranges``.
    """
    param_ranges = cerdf_emulator.param_ranges

    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll = log_likelihood_cerdf(params, cerdf_emulator, cerdf_cfg)
    return ll, 0.0


def log_probability_qlf_cerdf(params, datas_qlf, qlf_emulator, qlf_cfg,
                              cerdf_emulator, cerdf_cfg):
    """
    Joint QLF + CERDF likelihood, each weighted by 1/N_datapoints.

    This is a **legacy** two-component wrapper predating the unified
    multi-component dispatcher in ``main_mcmc.py``.  It divides each
    log-likelihood by the number of data points to balance the two
    components (QLF ~50 bins vs CERDF ~9000 individual QSOs).

    Note: ``n_cerdf`` uses ``//2`` as a crude downweighting of the CERDF
    contribution, which empirically prevents the much larger CERDF dataset
    from dominating the joint fit.

    Parameters
    ----------
    params : array-like
        Emulator parameter vector (shared between both emulators).
    datas_qlf : list
        QLF observational data objects.
    qlf_emulator, cerdf_emulator : GeneralEmulatorGP
    qlf_cfg : dict
        Output of ``precompute_qlf_inputs``.
    cerdf_cfg : dict
        Output of ``precompute_cerdf_inputs``.
    """
    param_ranges = qlf_emulator.param_ranges

    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    n_qlf = sum(len(d.x) for d in datas_qlf if d is not None)
    n_cerdf = len(cerdf_cfg['log_etas']) // 2

    ll_qlf = log_likelihood_qlf(params, datas_qlf, qlf_emulator, qlf_cfg)
    ll_cerdf = log_likelihood_cerdf(params, cerdf_emulator, cerdf_cfg)

    if ll_cerdf == -np.inf:
        return -np.inf, -np.inf

    # Per-data-point normalization so both components contribute equally
    ll = ll_qlf / n_qlf + ll_cerdf / n_cerdf
    return ll, 0.0


def log_probability_corr(params, datas_corr, qhmf_emulator, precomputed_inputs):
    """
    Prior + correlation function likelihood.

    Used when running correlation-only inference.
    """
    param_ranges = qhmf_emulator.param_ranges

    # EARLY EXIT: bounds check before expensive halo-model computation
    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll = log_likelihood_corr(params, datas_corr, qhmf_emulator, precomputed_inputs)
    return ll, 0.0