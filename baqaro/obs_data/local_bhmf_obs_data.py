"""Local (z ~ 0) black-hole mass function of the TOTAL population.

Unlike the active BHMFs in :mod:`abhmf_obs_data`, these count all black holes,
active or not, and so constrain the end point of the model's mass assembly
rather than its instantaneous duty cycle.

Entries live in ``data_local_bhmf`` as :class:`LocalBHMF_data`, each optionally
carrying a ``reliable_range`` so plotters can clip the extrapolated tails that
several of these determinations extend into.

Sources: Marconi+2004, Shankar+2009, Shankar+2013, Vika+2009, Liepold & Ma 2024.

Volume-unit caution: entries published in h70^3 Mpc^-3 are converted here to
the h-free Mpc^-3 used everywhere else in the codebase. Getting that factor
wrong is a ~0.04 dex shift at the FLAMINGO cosmology -- small, but systematic.
"""

import numpy as np

from qhtools.utils.cosmology import cosmo   # for the Vika+09 h70^3 volume conversion


class LocalBHMF_data:
    """One published local (total-population) BHMF measurement.

    Same shape as :class:`~..abhmf_obs_data.BHMF_data`, plus an optional
    ``reliable_range`` marking where the determination is physically
    trustworthy -- several of these are published well beyond the mass range
    their data actually constrain, and plotters clip to it.
    """

    def __init__(self, x_data, y_data, errs, label, reliable_range=None):
        """Local-BHMF measurement.

        Parameters
        ----------
        x_data, y_data, errs, label
            log10 M_BH, log10 phi, [err_down, err_up], display label.
        reliable_range : tuple (logM_min, logM_max) or None
            Optional mass range where the entry is physically trustworthy.
            Downstream plotters honour this to clip extrapolated regions.
            ``None`` means the full ``x_data`` range is reliable.
        """
        self.x = x_data
        self.data = y_data
        self.errs = errs
        self.err_down = errs[0]
        self.err_up = errs[1]
        err_down_finite = np.where(np.isfinite(self.err_down), self.err_down, 0)
        err_up_finite = np.where(np.isfinite(self.err_up), self.err_up, 0)
        self.err = (err_down_finite + err_up_finite) / 2.
        self.label = label
        self.reliable_range = reliable_range

    def reliable_mask(self):
        """Boolean mask True where ``self.x`` is inside ``reliable_range``.

        Returns an all-True mask if ``reliable_range`` is ``None``.
        """
        if self.reliable_range is None:
            return np.ones_like(self.x, dtype=bool)
        lo, hi = self.reliable_range
        return (self.x >= lo) & (self.x <= hi)

    def reliable_arrays(self):
        """Return ``(x, data, err_down, err_up)`` clipped to ``reliable_range``.

        Plotting scripts should call this to suppress mass ranges where
        the BHMF entry is known to be an extrapolation. For entries with
        ``reliable_range=None`` (the default), the returned arrays cover
        the full data range and are equivalent to ``(x, data, err_down, err_up)``.
        """
        m = self.reliable_mask()
        return self.x[m], self.data[m], self.err_down[m], self.err_up[m]


data_local_bhmf = {}

# --------------------------------------------------------------------------
# Marconi+2004 local BHMF (z~0) — no error bars provided
# --------------------------------------------------------------------------
_logM_m04 = np.array([
    7.10039, 7.27799, 7.47104, 7.65637, 7.83398, 8.03475, 8.18919,
    8.35907, 8.50193, 8.65637, 8.80309, 8.95367, 9.10425, 9.25483,
    9.37066, 9.47104, 9.56371, 9.65251, 9.72587, 9.80695, 9.87259,
    9.95367,
])
_logPhi_m04 = np.array([
    -2.21092, -2.27872, -2.33898, -2.42185, -2.54991, -2.68550, -2.80603,
    -2.94915, -3.09228, -3.29567, -3.48399, -3.72505, -4.04143, -4.31262,
    -4.59887, -4.88512, -5.19397, -5.44256, -5.69868, -5.93974, -6.22599,
    -6.51977,
])
_zeros_m04 = np.zeros_like(_logPhi_m04)

data_local_bhmf['Marconi04'] = LocalBHMF_data(
    x_data=_logM_m04, y_data=_logPhi_m04,
    errs=[_zeros_m04+0.1, _zeros_m04+0.1],  # Add small arbitrary error bars for visualization
    label="Marconi+2004"
)



# --------------------------------------------------------------------------
# Shankar+2009 local BHMF — data given as log(M) and log(Phi) bounds
# Columns: logM, lo, up  (lo and up are log10(Phi) confidence bounds)
# --------------------------------------------------------------------------
# Columns: z (ignored), logM, then 5 estimates of logPhi [Mpc^-3 dex^-1]
# Take mean as central value, min/max as error bounds
_shankar09_raw = np.array([
    [5.000, -1.263914, -1.398109, -1.161824, -0.928193, -0.952486],
    [5.200, -1.393714, -1.527906, -1.291592, -1.060015, -1.081840],
    [5.400, -1.519155, -1.653348, -1.417025, -1.190526, -1.209768],
    [5.600, -1.640046, -1.774211, -1.537929, -1.319667, -1.336193],
    [5.800, -1.756081, -1.890216, -1.653939, -1.447366, -1.461016],
    [6.000, -1.866583, -2.000693, -1.764388, -1.573533, -1.584118],
    [6.200, -1.970193, -2.104331, -1.867995, -1.698091, -1.705393],
    [6.400, -2.064800, -2.199106, -1.962558, -1.820937, -1.824722],
    [6.600, -2.147210, -2.282573, -2.044318, -1.941952, -1.942019],
    [6.800, -2.213713, -2.352388, -2.108829, -2.061024, -2.057273],
    [7.000, -2.266814, -2.408659, -2.160181, -2.178049, -2.170564],
    [7.200, -2.316486, -2.457227, -2.210563, -2.293059, -2.282098],
    [7.400, -2.371406, -2.508028, -2.267692, -2.406165, -2.392010],
    [7.600, -2.441676, -2.573755, -2.340455, -2.517543, -2.500275],
    [7.800, -2.537390, -2.665261, -2.438696, -2.627306, -2.607056],
    [8.000, -2.664529, -2.788563, -2.568216, -2.735395, -2.714626],
    [8.200, -2.822907, -2.943322, -2.728815, -2.842105, -2.832045],
    [8.400, -3.003822, -3.120621, -2.911950, -2.950282, -2.979541],
    [8.600, -3.195992, -3.309027, -3.106451, -3.070084, -3.183221],
    [8.800, -3.412806, -3.522263, -3.325501, -3.222933, -3.456882],
    [9.000, -3.683950, -3.790228, -3.598675, -3.435153, -3.791734],
    [9.200, -4.002519, -4.105700, -3.919256, -3.717911, -4.166062],
    [9.400, -4.340411, -4.440422, -4.259225, -4.058855, -4.560632],
    [9.600, -4.683076, -4.779878, -4.604011, -4.436392, -4.963504],
])

_logM_s09 = _shankar09_raw[:, 0]
_phi_estimates = _shankar09_raw[:, 1:]  # 5 columns of logPhi
_logPhi_s09 = np.mean(_phi_estimates, axis=1)
_lo_s09 = np.min(_phi_estimates, axis=1)
_up_s09 = np.max(_phi_estimates, axis=1)
_err_down_s09 = _logPhi_s09 - _lo_s09  # positive
_err_up_s09 = _up_s09 - _logPhi_s09    # positive

data_local_bhmf['Shankar09'] = LocalBHMF_data(
    x_data=_logM_s09, y_data=_logPhi_s09,
    errs=[_err_down_s09, _err_up_s09],
    label="Shankar+2009"
)

# --------------------------------------------------------------------------
# Shankar+2013 local BHMF — 5 model variants
# Columns: logM, then 5 estimates of logPhi [Mpc^-3 dex^-1]
# Take mean as central value, min/max as error bounds
# --------------------------------------------------------------------------
_shankar13_raw = np.array([
    [6.0, -2.0291, -2.4174, -2.0527, -2.1924, -1.4914],
    [6.25, -2.1452, -2.5370, -2.3312, -2.3110, -1.6372],
    [6.5, -2.2465, -2.6547, -2.4211, -2.4235, -1.7866],
    [6.75, -2.3227, -2.7629, -2.5266, -2.5225, -1.9296],
    [7.0, -2.3720, -2.8542, -2.6074, -2.6005, -2.0559],
    [7.25, -2.4043, -2.9203, -2.6514, -2.6529, -2.1602],
    [7.5, -2.4526, -2.9510, -2.6969, -2.6871, -2.2561],
    [7.75, -2.5471, -2.9401, -2.7377, -2.7158, -2.3630],
    [8.0, -2.6840, -2.9176, -2.8136, -2.7621, -2.5001],
    [8.25, -2.8443, -2.9539, -2.9423, -2.8760, -2.6849],
    [8.5, -3.0261, -3.1026, -3.1097, -3.1186, -2.9193],
    [8.75, -3.2494, -3.3570, -3.3274, -3.4131, -3.1964],
    [9.0, -3.5340, -3.6796, -3.5762, -3.6810, -3.5092],
    [9.25, -3.8839, -4.0557, -3.8651, -4.1092, -3.8504],
    [9.5, -4.2811, -4.4617, -4.1823, -4.5055, -4.2131],
    [9.75, -4.7058, -4.8853, -4.5281, -4.9311, -4.5902],
])
_logM_s13 = _shankar13_raw[:, 0]
_phi_estimates_s13 = _shankar13_raw[:, 1:]
_logPhi_s13 = np.mean(_phi_estimates_s13, axis=1)
_lo_s13 = np.min(_phi_estimates_s13, axis=1)
_up_s13 = np.max(_phi_estimates_s13, axis=1)
_err_down_s13 = _logPhi_s13 - _lo_s13
_err_up_s13 = _up_s13 - _logPhi_s13

data_local_bhmf['Shankar13'] = LocalBHMF_data(
    x_data=_logM_s13, y_data=_logPhi_s13,
    errs=[_err_down_s13, _err_up_s13],
    label="Shankar+2013"
)

# --------------------------------------------------------------------------
# Vika+2009 local BHMF
# Columns: logM, Phi (1e-4 h70^3 Mpc^-3 dex^-1), uperr, loerr
# --------------------------------------------------------------------------
_logM_v09 = np.array([7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50])

# Volume units: the table is in h70^3 Mpc^-3 dex^-1, i.e. it assumes H0 = 70 km/s/Mpc.
# Converting to the h-free Mpc^-3 dex^-1 used everywhere else in this module needs a
# factor h70^3 = (h / 0.7)^3. At the FLAMINGO cosmology (h = 0.681) that is 0.921,
# i.e. a -0.036 dex shift. Small, but it propagates into any chi^2 built on this
# dataset, not just the plots.
_H70_V09 = (cosmo.h / 0.7) ** 3

_Phi_v09 = np.array([34.1, 38.5, 34.0, 22.8, 11.1, 3.63, 0.78, 0.10]) * 1e-4 * _H70_V09
_uperr_v09 = np.array([4.97, 4.84, 4.92, 4.40, 3.51, 2.09, 0.77, 0.19]) * 1e-4 * _H70_V09
_loerr_v09 = np.array([4.53, 4.17, 4.17, 4.05, 3.20, 1.58, 0.47, 0.07]) * 1e-4 * _H70_V09

_logPhi_v09 = np.log10(_Phi_v09)
_err_up_v09 = np.log10(_Phi_v09 + _uperr_v09) - _logPhi_v09
_err_down_v09 = _logPhi_v09 - np.log10(_Phi_v09 - _loerr_v09)

data_local_bhmf['Vika09'] = LocalBHMF_data(
    x_data=_logM_v09, y_data=_logPhi_v09,
    errs=[_err_down_v09, _err_up_v09],
    label="Vika+2009"
)





# --------------------------------------------------------------------------
# Liepold & Ma 2024 local BHMF (z~0)
# Generated DIRECTLY from their Eq. 3 (page 7-8 of arXiv:2407.14595):
#
#     dn / d(ln M_BH) = phi * (M_BH / M_s)^(alpha+1) * exp(-(M_BH / M_s)^beta)
#
# with their quoted central best-fit and 1-sigma marginal errors:
#     alpha             = -1.27 ± 0.02
#     beta              =  0.45 ± 0.02
#     log10(phi/Mpc^-3) = -2.00 ± 0.07
#     log10(M_s/Msun)   =  8.09 ± 0.09
#
# y_data here is the central best-fit log10(dn/dlog10M); the 90% CI band
# is propagated by Monte-Carlo sampling the four parameter marginals as
# independent Gaussians (LM's posterior covariance is not published; this
# is the cleanest approximation available).
#
# IMPORTANT CAVEAT - LOW-MASS BEHAVIOUR:
# LM's paper says Eq. 3 "well approximates" their actual MCMC BHMF, but
# the analytic stretched-Schechter overshoots the violet band of their
# Fig. 4 at log10 M_BH < 9 by a factor 1.9x at log M = 8.5, 3.5x at
# log M = 8, and 5.8x at log M = 7. (The underlying GSMF was calibrated
# at M_* > 10^11.3 M_sun and extrapolated below; the analytic form
# doesn't capture the low-mass roll-over.) Consequence: rho_BH from Eq. 3
# OVERESTIMATES LM's published value (integrating gives 2.4 x 10^6
# Msun/Mpc^3 vs LM's quoted (1.8 +0.8/-0.5) x 10^6), because the rho_BH
# integrand peaks at log M ~ 8.6, well inside the overshoot region. Do NOT
# quote rho_BH from this fit -- use LM's published value instead.
# --------------------------------------------------------------------------
_LM24_alpha_central, _LM24_alpha_sigma = -1.27, 0.02
_LM24_beta_central,  _LM24_beta_sigma  =  0.45, 0.02
_LM24_logphi_central, _LM24_logphi_sigma = -2.00, 0.07
_LM24_logMs_central,  _LM24_logMs_sigma  =  8.09, 0.09


def _lm24_phi_per_dex(log10M, alpha, beta, log10_phi, log10_Ms):
    """LM 2024 Eq. 3 evaluated as dn/dlog10M (Mpc^-3 dex^-1)."""
    x = 10 ** (log10M - log10_Ms)
    dn_dlnM = (10 ** log10_phi) * x ** (alpha + 1) * np.exp(-x ** beta)
    return np.log(10) * dn_dlnM


_logM_lm24 = np.arange(5.0, 11.05, 0.1)
_logPhi_lm24 = np.log10(_lm24_phi_per_dex(
    _logM_lm24,
    _LM24_alpha_central, _LM24_beta_central,
    _LM24_logphi_central, _LM24_logMs_central,
))

# 90% CI via MC (independent Gaussians on the four marginals).
_lm24_rng = np.random.default_rng(2024)
_lm24_n_mc = 20000
_lm24_alphas = _lm24_rng.normal(_LM24_alpha_central, _LM24_alpha_sigma, _lm24_n_mc)
_lm24_betas  = _lm24_rng.normal(_LM24_beta_central,  _LM24_beta_sigma,  _lm24_n_mc)
_lm24_logphi = _lm24_rng.normal(_LM24_logphi_central, _LM24_logphi_sigma, _lm24_n_mc)
_lm24_logMs  = _lm24_rng.normal(_LM24_logMs_central,  _LM24_logMs_sigma,  _lm24_n_mc)
_lm24_band = np.zeros((_lm24_n_mc, _logM_lm24.size))
for _i in range(_lm24_n_mc):
    _lm24_band[_i] = np.log10(_lm24_phi_per_dex(
        _logM_lm24, _lm24_alphas[_i], _lm24_betas[_i],
        _lm24_logphi[_i], _lm24_logMs[_i],
    ))
_logPhi_lm24_p5  = np.percentile(_lm24_band, 5.0,  axis=0)
_logPhi_lm24_p95 = np.percentile(_lm24_band, 95.0, axis=0)
_err_down_lm24 = _logPhi_lm24 - _logPhi_lm24_p5   # positive
_err_up_lm24   = _logPhi_lm24_p95 - _logPhi_lm24  # positive

data_local_bhmf['LiepoldMa24'] = LocalBHMF_data(
    x_data=_logM_lm24, y_data=_logPhi_lm24,
    errs=[_err_down_lm24, _err_up_lm24],
    label="Liepold&Ma2024",
    # The Eq. 3 fit overshoots LM's MCMC posterior for log10 M_BH < 9
    # (factor 2-6x at log10 M_BH = 8.5 down to 7) and the very-high-mass
    # tail at log10 M_BH > 10.5 is poorly constrained; restrict to where
    # the fit matches Fig. 4 violet band to <~ factor 2.
    reliable_range=(9.0, 10.5),
)





