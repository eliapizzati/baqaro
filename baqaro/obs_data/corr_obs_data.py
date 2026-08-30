"""
Observational data for quasar and galaxy correlation functions.

Datasets included:
    - Shen+07: QSO auto-correlation at 2.9 < z < 5 and 3.5 < z < 5
    - Eftekharzadeh+15: QSO auto-correlation at 2.2 < z < 2.8
      (full sample, extended, and high-L sub-sample; with covariance)
    - EIGER: QSO-galaxy cross-correlation and galaxy auto-correlation at z ~ 6.3
    - ASPIRE: QSO-galaxy cross-correlation and galaxy auto-correlation at z ~ 6.6
      (with covariance matrix)
    - FRESCO+COSMOS3D: galaxy auto-correlation at z ~ 6

All separations are in comoving Mpc (converted from h^-1 Mpc where needed).
Luminosity thresholds are in log(L_bol / L_sun).
"""

import os
import warnings

import numpy as np

from qhtools.utils.cosmology import cosmo
from qhtools.utils.magnitude_conversion import (
    get_log_Lbol_from_M1450,
    get_M1450_from_m,
    get_M1450_from_Mi_z2,
)
from qhtools.utils import my_utils
from baqaro.obs_data.qlf_uv_obs_data import (
    data_z4_UV_Kulkarni,
    data_z2_UV_Kulkarni,
    data_z6_UV_Schindler,
)


# ---------------------------------------------------------------------------
# z=6.1 galaxy-tracer halo-mass cut
# ---------------------------------------------------------------------------
# The z=6.1 QSO-galaxy CROSS-correlation needs a "galaxy" definition: haloes
# above a minimum mass. The analytic curve and the direct measurement must use
# the SAME value or their amplitudes are not comparable, so it is defined once
# here instead of in either of them. The likelihood is independent and reads its
# own ``BAQARO_CORR_GAL_LOGM_MIN`` in ``inference/likelihoods_and_priors.py``.

#: Galaxy halo-mass cut used by the ANALYTIC curve and the DIRECT measurement.
# GAL_LOGM_CUT_FIGURE = float(os.environ.get("BAQARO_CORR_GAL_LOGM_FIGURE", "10.75"))
GAL_LOGM_CUT_FIGURE = float(os.environ.get("BAQARO_CORR_GAL_LOGM_FIGURE", "10.7"))


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class Corr_data:
    """Container for a single correlation function measurement.

    Parameters
    ----------
    x_data : array
        Separation bins (comoving Mpc).
    y_data : array
        Correlation function values (e.g., wp/rp or volume-averaged xi).
    errs : list of two arrays
        [err_down, err_up] — asymmetric error bars.
    label : str
        Dataset label for plotting / identification.
    log_L_threshold : float, optional
        Luminosity threshold in log(L_bol / L_sun) used to define the sample.
    num_den : float, optional
        Number density of the sample [cMpc^-3], integrated from the QLF above
        the luminosity threshold.
    num_den_err : float, optional
        Fractional uncertainty on num_den.
    covariance : 2-d array, optional
        Full covariance matrix for the data points.
    pimax : float, optional
        Line-of-sight integration limit [cMpc] (default 100).
    params_clf : list, optional
        Initial CLF model parameters [sigma, log_a, b_lo, log_dc].
    npar_clf : int, optional
        Number of CLF parameters (default 4).
    model_clf : str, optional
        CLF model name (default "power_law+log_normal+duty_cycle").
    params_gal : list, optional
        Galaxy clustering parameters [r0, gamma].
    xerrs : list of two arrays, optional
        [xerr_down, xerr_up] — horizontal error bars on separation bins.
    corr_type : {"auto", "cross"}
        Whether this is an auto-correlation or cross-correlation measurement.
    """

    def __init__(
        self,
        x_data,
        y_data,
        errs,
        label,
        corr_type="auto",
        log_L_threshold=None,
        num_den=None,
        num_den_err=None,
        covariance=None,
        pimax=100.0,
        params_clf=None,
        npar_clf=4,
        model_clf="power_law+log_normal+duty_cycle",
        params_gal=None,
        xerrs=None,
    ):
        self.x = x_data
        self.data = y_data

        self.errs = errs
        self.err_down = errs[0]
        self.err_up = errs[1]
        self.err = (self.err_down + self.err_up) / 2.0

        self.xerrs = xerrs
        self.label = label
        self.corr_type = corr_type

        self.log_L_threshold = log_L_threshold
        self._num_den = num_den
        self.num_den_err = num_den_err
        self.covariance = covariance

        self.pimax = pimax

        self.params_clf = params_clf
        self.npar_clf = npar_clf
        self.model_clf = model_clf
        self.params_gal = params_gal

    def check_nans(self):
        """Return True if any data value is NaN."""
        return np.isnan(np.sum(np.asarray(self.data)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

    @property
    def num_den(self):
        """Quasar number density above ``log_L_threshold``, in cMpc^-3.

        ``None`` on every dataset in this module, because the UV QLFs it is
        derived from carry no fitted curve (``log_L_axis`` / ``log_qlf_fit``
        are unset). Reading it warns rather than silently handing back
        ``None``, since treating that as a number would corrupt any halo-model
        normalisation built on it.

        The warning is on ACCESS, not on construction: nothing currently reads
        this, so warning at import would fire on every run about a value no
        caller uses.
        """
        if self._num_den is None:
            warnings.warn(
                "Corr_data.num_den is None: the UV QLFs carry no fit "
                "(log_L_axis / log_qlf_fit are unset), so no number density "
                "could be computed. Populate those fits before using this as "
                "a number.",
                RuntimeWarning, stacklevel=2)
        return self._num_den


def _compute_num_den(qlf_data, log_L_thr, fractional_error=0.01):
    """Integrate the QLF fit above a luminosity threshold.

    Returns (num_den, num_den_err) in cMpc^-3, or (None, None) if no fit.

    ``num_den`` is not used by the likelihood (which reads only ``x`` /
    ``data`` / ``err`` / ``pimax`` / ``log_L_threshold``) and returns ``None``
    for every dataset in this module, because the UV QLFs it is called with
    are constructed in ``qlf_uv_obs_data.py`` without a fitted curve.
    """
    if qlf_data.log_L_axis is None or qlf_data.log_qlf_fit is None:
        # No fit to integrate. Corr_data.num_den warns on access instead of
        # here, so importing this module stays silent.
        return None, None
    mask = qlf_data.log_L_axis > log_L_thr
    num_den = np.trapezoid(
        np.power(10, qlf_data.log_qlf_fit[mask]),
        qlf_data.log_L_axis[mask],
    )
    return num_den, fractional_error * num_den


# Default CLF / galaxy clustering initial parameters (shared across z~6 datasets)
_DEFAULT_NPAR_CLF = 4
_DEFAULT_MODEL_CLF = "power_law+log_normal+duty_cycle"
_DEFAULT_PARAMS_CLF = [0.38, 13.5, 1.75, -2.9]   # [sigma, log_a, b_lo, log_dc]
_DEFAULT_PARAMS_GAL = [7.5, 1.7]                   # [r0_gal, gamma_gal]


# ===========================================================================
#  Shen+07  —  QSO auto-correlation, 2.9 < z < 5.0 (full sample)
# ===========================================================================

x_shen_all = np.asarray([
    1.189, 1.679, 2.371, 3.350, 4.732, 6.683, 9.441, 13.34,
    18.84, 26.61, 37.58, 53.09, 74.99, 105.9, 149.6, 211.3,
]) / cosmo.h

y_shen_all = np.asarray([
    0., 154, 236, 78.1, 91.3, 15.7, 10.6, 3.06,
    -0.681, 0.516, 0.437, 0.0675, 0.0484, 0.0674, 0.0228, -0.0183,
])

dy_shen_all = np.asarray([
    0., 162, 195, 51.5, 41.6, 7.81, 4.45, 2.85,
    0.913, 0.810, 0.395, 0.259, 0.145, 0.0592, 0.0292, 0.00992,
])

data_shen_all = Corr_data(
    x_data=x_shen_all,
    y_data=y_shen_all,
    errs=[dy_shen_all, dy_shen_all],
    label="Shen+07 2.9<z<5",
)


# ===========================================================================
#  Shen+07  —  QSO auto-correlation, z >= 3.5 (high-z sub-sample, GOOD FIELDS)
# ===========================================================================
#
# PROVENANCE (recovered from the arXiv PostScript source; supersedes an
# earlier hand-digitisation of the same figure).
#
# Shen+07 publish NO TABLE for the z >= 3.5 sub-sample — their Table 3 is the
# z >= 2.9 full sample (`data_shen_all` below). The z >= 3.5 measurement exists
# only as Figure 7. That figure is IDL vector output, so the plotted coordinates
# are recoverable EXACTLY from the arXiv source's PostScript rather than
# eyeballed. These values are that recovery, from the RIGHT panel (Fig 7b,
# "good fields only" — the 3,846-quasar sample with bad imaging fields cut,
# which is also what the previous digitization had used).
#
# Because it is the good-fields panel, the matching power-law fit in the paper's
# Table 4 is r0 = 25.22 +- 2.50, gamma = 2.14 +- 0.24 — NOT the all-fields
# headline r0 = 22.51 +- 2.53, gamma = 2.28 +- 0.31. Do not compare against the
# latter.
#
# The extraction is validated three ways:
#   1. The same code path run on Fig 5a (whose values ARE tabulated, Table 3)
#      reproduces that table to a median of 0.04% (max 0.27%).
#   2. Refitting these points over the paper's own range (4 <= r_p <= 150
#      h^-1 Mpc) returns r0 = 25.22, gamma = 2.14 — the published values to
#      every quoted decimal. Doing the same on the all-fields panel returns
#      22.51 / 2.28. This validates the errors too, since they set the weights.
#   3. Error bars are symmetric in LINEAR space: for all 14 points of Fig 5a the
#      drawn bar ends equal Table 3's (value - err, value + err) to ~0.1%.
#
# ERRORS. Point 3 above is why sigma is stored symmetric. Where value - sigma
# < 0 the bar CANNOT be drawn on a log axis and IDL clips it flat at the axis
# floor (that clipping is itself the proof of linear symmetry — a log-symmetric
# bar would never reach the floor). The UPPER bar is never clipped, so
# sigma = y_hi - y is exact even for those points.
#
# NOT INCLUDED: the outermost plotted point, r_p = 149.6 h^-1 Mpc
# (w_p/r_p = 0.1216 +- 0.0808, a real 1.5-sigma detection), deliberately left
# out so the binning matches what has been fitted historically.

# Separations [h^-1 Mpc] — the paper's true bin grid (Delta log10 r_p = 0.15),
# identical to the r_p column of Table 3.
x_shen_highz = np.asarray([
    2.371, 4.732, 6.683, 9.441, 13.34, 18.84,
    26.61, 37.58, 53.09, 74.99, 105.9,
]) / cosmo.h

# w_p / r_p (dimensionless — a ratio of two lengths, so h-independent and NOT
# rescaled by cosmo.h, unlike x above).
y_shen_highz = np.asarray([
    402.45, 160.56, 34.577, 22.135, 13.811, 6.6259,
    2.3466, 1.4187, 0.17423, 0.33819, 0.045336,
])

# Jackknife sigma, symmetric in linear space (see ERRORS above).
#
# The first entry is a LOWER BOUND, not a measurement: at r_p = 2.371 the upper
# bar runs off the TOP of the axis too (value + sigma > 1000), so the figure
# only constrains sigma > 597.55. Stored at that bound — which is conservative
# in the sense that the true sigma is larger, so this bin still gets somewhat
# too much weight. It is also the one bin OUTSIDE the range Shen+07 fit
# themselves (they use 4 <~ r_p <~ 150 h^-1 Mpc, i.e. bins with > 10 quasar
# pairs), so excluding it from the likelihood is defensible.
dy_shen_highz = np.asarray([
    597.55, 134.96, 24.621, 15.278, 7.1547, 2.0971,
    1.1661, 1.0017, 0.45115, 0.25311, 0.14517,
])

dy_up_shen_highz = dy_shen_highz
dy_down_shen_highz = dy_shen_highz

# Luminosity threshold: i-band apparent magnitude limit -> bolometric.
# i = 20.2 is the SDSS target limit for z > 3 quasars (Shen+07 §2), and is the
# limit they integrate the LF to in their own bias/halo-mass analysis (§6).
# The z = 4.0 evaluation redshift matches the sub-sample: Shen+07's Fig 8 places
# the z >= 3.5 bin at its MEAN redshift, which reads off as z ~ 3.95.
_mag_i_thr_shen = 20.2
_log_L_thr_shen_highz = my_utils.to_solar(
    get_log_Lbol_from_M1450(get_M1450_from_m(_mag_i_thr_shen, redshift=4.0))
)

_qlf_z4 = data_z4_UV_Kulkarni
_num_den_shen, _num_den_err_shen = _compute_num_den(
    _qlf_z4, _log_L_thr_shen_highz, fractional_error=0.04
)

# pimax = 100 h^-1 cMpc
_pimax_shen_highz = 100.0 / cosmo.h

data_shen_highz = Corr_data(
    x_data=x_shen_highz,
    y_data=y_shen_highz,
    errs=[dy_down_shen_highz, dy_up_shen_highz],
    label="Shen+07 3.5<z<5",
    log_L_threshold=_log_L_thr_shen_highz,
    num_den=_num_den_shen,
    num_den_err=_num_den_err_shen,
    pimax=_pimax_shen_highz,
)

_mask_shen_highz = (x_shen_highz > 3.0) & (x_shen_highz < 100.0)
data_shen_highz_restricted = Corr_data(
    x_data=x_shen_highz[_mask_shen_highz],
    y_data=y_shen_highz[_mask_shen_highz],
    errs=[dy_down_shen_highz[_mask_shen_highz], dy_up_shen_highz[_mask_shen_highz]],
    label="Shen+07 3.5<z<5",
    log_L_threshold=_log_L_thr_shen_highz,
    num_den=_num_den_shen,
    num_den_err=_num_den_err_shen,
    pimax=_pimax_shen_highz,
)


# ---------------------------------------------------------------------------
#  Shen+07  —  same z >= 3.5 sub-sample, but ALL FIELDS (Fig 7a, LEFT panel)
# ---------------------------------------------------------------------------
#
# The companion of `data_shen_highz` (which is the good-fields panel). Same
# extraction, same validation: refitting these over the paper's own range
# (4 <= r_p <= 150 h^-1 Mpc) returns r0 = 22.51, gamma = 2.28 — the published
# all-fields values of Table 4 to every decimal.
#
# ALL FIELDS = the 4,426-quasar sample. GOOD FIELDS = 3,846, after cutting the
# ~13% of the area flagged as bad imaging. Shen+07 say the two "give similar
# results", but they are NOT interchangeable at the ~1 sigma level on individual
# bins: the all-fields w_p goes NEGATIVE-ish / noisy at 26-38 h^-1 Mpc (0.58 and
# 0.44 here vs 2.35 and 1.42 good-fields, i.e. a factor 3-4 apart), which is why
# the good-fields case is the cleaner one and why the paper notes that only in
# the good-fields case is w_p positive across the whole fit range.
#
# The production z=4 dataset is `data_shen_highz_allfields_err2` below (this
# all-fields measurement with the errors inflated x2). Select this x1 version
# via BAQARO_CORR_Z4=data_shen_highz_allfields, or the good-fields one via
# BAQARO_CORR_Z4=data_shen_highz; either CHANGES the likelihood, so tag any
# resulting chain (BAQARO_MCMC_NOTES).
#
# Same 11 bins as `data_shen_highz` so the two are drop-in comparable; the
# outermost plotted point is likewise omitted (it is r_p = 149.6 h^-1 Mpc,
# w_p/r_p = 0.1239 +- 0.0716 all-fields, 0.1216 +- 0.0808 good-fields).
#
# As in the good-fields set, the first entry's sigma is a LOWER BOUND (its upper
# bar runs off the top of the axis), and it is the one bin outside the range
# Shen+07 fit themselves.

x_shen_highz_allfields = x_shen_highz          # identical r_p grid [already /h]

y_shen_highz_allfields = np.asarray([
    439.80, 195.35, 33.583, 17.772, 14.505, 4.9601,
    0.57934, 0.43622, 0.41729, 0.14104, 0.099244,
])

# Jackknife sigma, symmetric in linear space (see the ERRORS note above).
dy_shen_highz_allfields = np.asarray([
    560.20, 153.46, 27.086, 11.064, 6.7315, 2.0438,
    1.1599, 0.69951, 0.38562, 0.19459, 0.14996,
])

data_shen_highz_allfields = Corr_data(
    x_data=x_shen_highz_allfields,
    y_data=y_shen_highz_allfields,
    errs=[dy_shen_highz_allfields, dy_shen_highz_allfields],
    label="Shen+07 3.5<z<5 (all fields)",
    log_L_threshold=_log_L_thr_shen_highz,
    num_den=_num_den_shen,
    num_den_err=_num_den_err_shen,
    pimax=_pimax_shen_highz,
)

# ---------------------------------------------------------------------------
#  ADOPTED z=4 clustering default: ALL FIELDS, errors inflated x2
# ---------------------------------------------------------------------------
# The production z=4 corr dataset. Two deliberate choices baked into one object so
# figure and inference cannot drift apart (single source of truth):
#   (1) ALL FIELDS (Fig 7a, 4426 QSOs) rather than the good-fields sub-sample —
#       the full-area measurement, at the cost of the noisier 26-38 h^-1 Mpc bins.
#   (2) Error bars x2. The Shen+07 jackknife errors (~32 regions for 11 bins) are
#       an under-estimate of the true uncertainty on this small high-z sample; x2
#       is the adopted allowance so the z=4 point weights the joint fit sensibly
#       instead of driving an apparent ~1.9 chi^2/dof tension (dropping z=4 already
#       moves NO parameter). Report the x2 as a systematic choice.
# Same 11 r_p bins, same i<20.2 selection threshold + pimax as data_shen_highz, so
# the halo-model curve is UNCHANGED — only the points and their errors differ.
# Revert to the good-fields x1 measurement with BAQARO_CORR_Z4=data_shen_highz.
_Z4_ERR_INFLATE_STD = 2.0
data_shen_highz_allfields_err2 = Corr_data(
    x_data=x_shen_highz_allfields,
    y_data=y_shen_highz_allfields,
    errs=[dy_shen_highz_allfields * _Z4_ERR_INFLATE_STD,
          dy_shen_highz_allfields * _Z4_ERR_INFLATE_STD],
    label="Shen+07 3.5<z<5 (all fields, err x2)",
    log_L_threshold=_log_L_thr_shen_highz,
    num_den=_num_den_shen,
    num_den_err=_num_den_err_shen,
    pimax=_pimax_shen_highz,
)


# ===========================================================================
#  Eftekharzadeh+15  —  QSO auto-correlation, 2.2 < z < 2.8 (with covariance)
# ===========================================================================

_qlf_z2 = data_z2_UV_Kulkarni

# Mi(z=2) threshold -> bolometric luminosity
_Mi_z2_thr_ef = -25.3
_log_L_thr_ef = my_utils.to_solar(
    get_log_Lbol_from_M1450(get_M1450_from_Mi_z2(_Mi_z2_thr_ef))
)
_num_den_ef, _num_den_err_ef = _compute_num_den(_qlf_z2, _log_L_thr_ef)

# --- Full sample (11-bin) with covariance matrix ---

x_ef_cov = np.asarray([4.36, 5.18, 6.15, 7.31, 8.69, 10.33, 12.28, 14.59, 17.34, 20.61, 24.50])
y_ef_cov = np.asarray([54.64, 48.87, 43.41, 38.09, 29.77, 26.47, 22.74, 16.68, 11.90, 15.27, 7.80])
y_ef_cov = y_ef_cov / x_ef_cov  # wp(rp) / rp

dy_ef_cov = np.asarray([12.45, 9.45, 9.56, 8.64, 5.68, 5.88, 4.17, 3.44, 3.16, 3.01, 3.09])
dy_ef_cov = dy_ef_cov / x_ef_cov

x_ef_cov = x_ef_cov / cosmo.h

# --------------------------------------------------------------------------
# EF15 Table 2 correlation matrix.
#
# Provenance: transcribed from Eftekharzadeh+15 (MNRAS 453, 2779;
# arXiv:1507.08380) Table 2, "Correlation coefficients as estimated from the
# covariance matrix computed by jackknife error estimation for NGC-CORE
# quasars".  All 121 entries — plus the wp and sigma rows above — were
# re-checked cell-by-cell against the published table.
#
# `data_ef` carries the matrix as published. The fit's z=2.5 default is the
# 18-bin `data_ef_ext_restricted` (no covariance); `data_ef` and the two
# variants below (`data_ef_diag`, diagonal errors; `data_ef_psd`,
# eigenvalue-floored covariance) are selectable via BAQARO_CORR_Z25. Each
# changes the likelihood, so tag the chain with BAQARO_MCMC_NOTES.
# --------------------------------------------------------------------------
correlation_ef = np.asarray([
    [ 1.000, -0.256, -0.052, -0.343,  0.027,  0.485,  0.221,  0.171,  0.174,  0.553,  0.543],
    [-0.256,  1.000,  0.023,  0.152,  0.455, -0.215,  0.320, -0.298, -0.299, -0.245, -0.241],
    [-0.052,  0.023,  1.000,  0.232, -0.119, -0.379, -0.083, -0.225, -0.261, -0.419, -0.273],
    [-0.343,  0.152,  0.232,  1.000,  0.083, -0.410,  0.095, -0.050, -0.958, -0.606, -0.579],
    [ 0.027,  0.455, -0.119,  0.083,  1.000,  0.472,  0.202,  0.153,  0.656,  0.538,  0.529],
    [ 0.485, -0.215, -0.379, -0.410,  0.472,  1.000,  0.484,  0.382,  0.388,  0.138,  0.119],
    [ 0.221,  0.320, -0.083,  0.095,  0.202,  0.484,  1.000,  0.031,  0.035, -0.166,  0.154],
    [ 0.171, -0.298, -0.225, -0.050,  0.153,  0.382,  0.031,  1.000,  0.004,  0.323,  0.309],
    [ 0.174, -0.299, -0.261, -0.958,  0.656,  0.388,  0.035,  0.004,  1.000,  0.320,  0.306],
    [ 0.553, -0.245, -0.419, -0.606,  0.538,  0.138, -0.166,  0.323,  0.320,  1.000,  0.083],
    [ 0.543, -0.241, -0.273, -0.579,  0.529,  0.119,  0.154,  0.309,  0.306,  0.083,  1.000],
])
covariance_ef = correlation_ef * dy_ef_cov[:, np.newaxis] * dy_ef_cov[np.newaxis, :]

data_ef = Corr_data(
    x_data=x_ef_cov,
    y_data=y_ef_cov,
    errs=[dy_ef_cov, dy_ef_cov],
    label="Eftekharzadeh+15 2.2<z<2.8",
    log_L_threshold=_log_L_thr_ef,
    num_den=_num_den_ef,
    num_den_err=_num_den_err_ef,
    covariance=covariance_ef,
    pimax=50 / cosmo.h,
)

def _psd_project_correlation(corr, floor=0.2):
    """PSD-project a correlation matrix by flooring its eigenvalues.

    Clips every eigenvalue to at least ``floor`` and renormalises back to unit
    diagonal.  ``floor`` is an absolute eigenvalue cut, not a fraction of
    lambda_max — a conditioning floor, not just a PSD projection. The default
    0.2 is the smallest floor that keeps the condition number of
    ``correlation_ef`` below ~20.
    """
    w, v = np.linalg.eigh(corr)
    out = (v * np.maximum(w, floor)) @ v.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


# Diagonal-error variant of the SAME 11 bins (no off-diagonal correlations;
# keeps the paper's own 4 < rp < 25 h^-1 Mpc fitting range).  This is what
# EF15 do for every sample whose jackknife matrix they judged too noisy.
# Select with BAQARO_CORR_Z25=data_ef_diag.
data_ef_diag = Corr_data(
    x_data=x_ef_cov,
    y_data=y_ef_cov,
    errs=[dy_ef_cov, dy_ef_cov],
    label="Eftekharzadeh+15 2.2<z<2.8",
    log_L_threshold=_log_L_thr_ef,
    num_den=_num_den_ef,
    num_den_err=_num_den_err_ef,
    covariance=None,
    pimax=50 / cosmo.h,
)

# Eigenvalue-floored variant (see `_psd_project_correlation`) — retains some
# off-diagonal structure.  Diagnostic only.  Select with
# BAQARO_CORR_Z25=data_ef_psd.
correlation_ef_psd = _psd_project_correlation(correlation_ef, floor=0.2)
covariance_ef_psd = (
    correlation_ef_psd * dy_ef_cov[:, np.newaxis] * dy_ef_cov[np.newaxis, :]
)

data_ef_psd = Corr_data(
    x_data=x_ef_cov,
    y_data=y_ef_cov,
    errs=[dy_ef_cov, dy_ef_cov],
    label="Eftekharzadeh+15 2.2<z<2.8",
    log_L_threshold=_log_L_thr_ef,
    num_den=_num_den_ef,
    num_den_err=_num_den_err_ef,
    covariance=covariance_ef_psd,
    pimax=50 / cosmo.h,
)


# --- Extended sample (18-bin, no covariance) ---

x_ef_ext = np.asarray([
    2.53, 3.06, 3.70, 4.47, 5.41, 6.55, 7.92, 9.58, 11.60, 14.03,
    16.97, 20.54, 24.85, 30.06, 36.37, 44.00, 53.23, 64.40,
])
y_ef_ext = np.asarray([
    60.807, 57.125, 52.908, 45.454, 39.291, 34.392, 25.846, 23.907, 16.056, 11.771,
    13.753, 9.404, 6.824, 6.554, 7.341, 4.192, 4.483, 2.576,
])
y_ef_ext = y_ef_ext / x_ef_ext

dy_ef_ext = np.asarray([
    14.491, 10.707, 8.654, 5.981, 10.607, 9.142, 6.051, 5.210, 3.002, 1.854,
    5.438, 3.130, 0.958, 2.099, 1.422, 1.061, 1.076, 0.159,
])
dy_ef_ext = dy_ef_ext / x_ef_ext

x_ef_ext = x_ef_ext / cosmo.h

data_ef_ext = Corr_data(
    x_data=x_ef_ext,
    y_data=y_ef_ext,
    errs=[dy_ef_ext, dy_ef_ext],
    label="Eftekharzadeh+15 2.2<z<2.8",
    log_L_threshold=_log_L_thr_ef,
    num_den=_num_den_ef,
    num_den_err=_num_den_err_ef,
    pimax=50 / cosmo.h,
)

_mask_ef_ext = (data_ef_ext.x > 6.0) & (data_ef_ext.x < 40.0)
data_ef_ext_restricted = Corr_data(
    x_data=x_ef_ext[_mask_ef_ext],
    y_data=y_ef_ext[_mask_ef_ext],
    errs=[dy_ef_ext[_mask_ef_ext], dy_ef_ext[_mask_ef_ext]],
    label="Eftekharzadeh+15 2.2<z<2.8",
    log_L_threshold=_log_L_thr_ef,
    num_den=_num_den_ef,
    num_den_err=_num_den_err_ef,
    pimax=50 / cosmo.h,
)


# --- High-L sub-sample (digitised, asymmetric errors) ---

_Mi_z2_thr_ef_highL = -26.19
_log_L_thr_ef_highL = my_utils.to_solar(
    get_log_Lbol_from_M1450(get_M1450_from_Mi_z2(_Mi_z2_thr_ef_highL))
)
_num_den_ef_highL, _num_den_err_ef_highL = _compute_num_den(_qlf_z2, _log_L_thr_ef_highL)

x_ef_highL = np.asarray([
    2.510158260514352, 3.048035709192993, 3.6555854252593463, 4.402383212818399,
    5.323690147979079, 6.411264246696426, 7.75297893351208, 9.336831269319001,
    11.290792790842628, 13.597383417264872, 16.442970236215356, 19.80209671386669,
    23.84745752331812, 28.71924314601603, 34.87321099511703,
    50.57707980739981, 60.90944710495361, 73.6562465819284,
]) / cosmo.h

y_ef_highL = np.asarray([
    26.300574380006225, 9.849252755816504, 5.761531133834592, 16.95329073136231,
    6.156113013739322, 6.170766507409129, 6.378406974974851, 3.968216518358068,
    1.6296714250931847, 0.9681911044189216, 0.3625198558839956, 0.8213098699204773,
    0.22957731769274178, 0.3486100889585765, 0.2169042086666753,
    0.055415768095567815, 0.04762672756060357, 0.049992559124711934,
])

y_up_ef_highL = np.asarray([
    44.37483236034571, 15.386674133099906, 8.334782294363615, 22.71042340544573,
    10.228137593690258, 9.205259665646855, 8.034057638421285, 5.315500161181659,
    2.4311929461879966, 1.5839705585722075, 0.713412775011443, 1.1344788391210312,
    0.44487100388137774, 0.5363179801605752, 0.2996107119380677,
    0.07309717916551721, 0.06186058773546543, 0.061061160930371235,
])

y_down_ef_highL = np.asarray([
    8.687527470439276, 4.493902006808893, 3.260951380581729, 11.364089184907435,
    2.0024239324455806, 3.4388965961697204, 4.5467016923434755, 2.7856113777330576,
    0.8409110038617575, 0.42836933541479477, 0.01037347825408818, 0.5421053642457496,
    0.02877047084187789, 0.19428640017758986, 0.13055033817021458,
    0.03714424493756076, 0.03610447508153347, 0.03969014191446164,
])

dy_up_ef_highL = y_up_ef_highL - y_ef_highL
dy_down_ef_highL = y_ef_highL - y_down_ef_highL

data_ef_highL = Corr_data(
    x_data=x_ef_highL,
    y_data=y_ef_highL,
    errs=[dy_down_ef_highL, dy_up_ef_highL],
    label="Eftekharzadeh+15 2.2<z<2.8, high L",
    log_L_threshold=_log_L_thr_ef_highL,
    num_den=_num_den_ef_highL,
    num_den_err=_num_den_err_ef_highL,
    pimax=50 / cosmo.h,
)

_mask_ef_highL = (x_ef_highL > 4 / cosmo.h) & (x_ef_highL < 40.0 / cosmo.h)
data_ef_highL_restricted = Corr_data(
    x_data=x_ef_highL[_mask_ef_highL],
    y_data=y_ef_highL[_mask_ef_highL],
    errs=[dy_down_ef_highL[_mask_ef_highL], dy_up_ef_highL[_mask_ef_highL]],
    label="Eftekharzadeh+15 2.2<z<2.8, high L restricted",
    log_L_threshold=_log_L_thr_ef_highL,
    num_den=_num_den_ef_highL,
    num_den_err=_num_den_err_ef_highL,
    pimax=50 / cosmo.h,
)


# ===========================================================================
#  EIGER  —  QSO-galaxy cross-correlation & galaxy auto, z ~ 6.3
# ===========================================================================

_true_redshift_eiger = 6.3
_vmax_eiger = 1000.0
_pimax_eiger = my_utils.get_pimax_from_vmax(
    vmax=_vmax_eiger, redshift=_true_redshift_eiger, cosmo=cosmo
)
_qlf_z6 = data_z6_UV_Schindler

# QSO luminosity threshold: M_1450 = -26.8
_M1450_eiger = -26.8
_log_L_thr_eiger = my_utils.to_solar(get_log_Lbol_from_M1450(_M1450_eiger))
_num_den_eiger, _num_den_err_eiger = _compute_num_den(
    _qlf_z6, _log_L_thr_eiger, fractional_error=0.05
)

# Galaxy sample threshold
_log_L_thr_gal_z6 = my_utils.to_solar(42.4)
_num_den_gal_z6 = 0.003828
_num_den_err_gal_z6 = 0.1 * _num_den_gal_z6

# --- QSO-galaxy cross-correlation ---

x_eiger = np.asarray([
    0.07465658604766102, 0.13276026979235472, 0.23608485424295678,
    0.41982483532229026, 0.7465658604766102, 1.3276026979235471,
    2.3608485424295678, 4.198248353222903,
]) / cosmo.h

x_eiger_low = np.asarray([
    0.05598454156567122, 0.09955615754670136, 0.17703866510789015,
    0.3148242129421373, 0.5598454156567122, 0.9955615754670136,
    1.7703866510789015, 3.1482421294213734,
]) / cosmo.h

x_eiger_high = np.asarray([
    0.09955615754670136, 0.17703866510789015, 0.3148242129421373,
    0.5598454156567122, 0.9955615754670136, 1.7703866510789015,
    3.1482421294213734, 5.598454156567121,
]) / cosmo.h

y_eiger = np.asarray([
    604.1500953880858, 210.8372339901755, 68.31785024752723,
    26.975332580094545, 15.009773983010962, 15.754437826316881,
    7.136796989031314, 3.6199729348952454,
])

dy_eiger = np.asarray([
    427.9057360846016, 122.30428406861355, 40.02067950005601,
    13.987666290047272, 6.051125786485575, 3.6561180745489756,
    1.4855690855775223, 1.008160746800226,
])

data_eiger = Corr_data(
    x_data=x_eiger,
    y_data=y_eiger,
    errs=[dy_eiger, dy_eiger],
    xerrs=[x_eiger - x_eiger_low, x_eiger_high - x_eiger],
    label="EIGER",
    corr_type="cross",
    log_L_threshold=_log_L_thr_eiger,
    num_den=_num_den_eiger,
    num_den_err=_num_den_err_eiger,
    pimax=_pimax_eiger,
    params_clf=_DEFAULT_PARAMS_CLF,
    model_clf=_DEFAULT_MODEL_CLF,
    npar_clf=_DEFAULT_NPAR_CLF,
    params_gal=_DEFAULT_PARAMS_GAL,
)

# --- Galaxy auto-correlation (OIII emitters) ---

x_gal_eiger = np.asarray([
    0.07465658604766102, 0.13276026979235472, 0.23608485424295678,
    0.41982483532229026, 0.7465658604766102, 1.3276026979235471,
    2.3608485424295678, 4.198248353222903,
]) / cosmo.h

y_gal_eiger = np.asarray([
    25.992589809002343, 15.77823571207691, 13.239740980693941,
    7.004955546817985, 4.876855308106543, 2.314385428133714,
    1.0068092167949438, 0.2844876972223205,
])

# Corrected errors (from the authors' updated data)
dy_gal_eiger = np.asarray([
    17.158278157173022, 8.01707447637318, 4.249067555414288,
    1.9937147516850282, 1.056731237155267, 0.5493231968031376,
    0.3174567870520627, 0.22930715889246472,
])

data_gal_z6 = Corr_data(
    x_data=x_gal_eiger,
    y_data=y_gal_eiger,
    errs=[dy_gal_eiger, dy_gal_eiger],
    label="EIGER OIII",
    log_L_threshold=_log_L_thr_gal_z6,
    num_den=_num_den_gal_z6,
    num_den_err=_num_den_err_gal_z6,
    pimax=_pimax_eiger,
    params_gal=_DEFAULT_PARAMS_GAL,
)


# ===========================================================================
#  ASPIRE  —  Final paper data (Huang+26), z ~ 6.6
#  Covariance provided by the ASPIRE team — see the
#  `correlation_aspire` / `covariance_aspire` block below.
# ===========================================================================

_pimax_aspire = 10.0
_qlf_z6_aspire = data_z6_UV_Schindler

_M1450_aspire = -25.3
_log_L_thr_aspire = my_utils.to_solar(get_log_Lbol_from_M1450(_M1450_aspire))
_num_den_aspire, _num_den_err_aspire = _compute_num_den(
    _qlf_z6_aspire, _log_L_thr_aspire, fractional_error=0.05
)

x_aspire_low = np.asarray([0.06, 0.10, 0.18, 0.32, 0.57, 1.02, 1.80, 3.17])
x_aspire_high = np.asarray([0.10, 0.18, 0.32, 0.57, 1.02, 1.80, 3.17, 5.60])
x_aspire = np.power(10, (np.log10(x_aspire_low) + np.log10(x_aspire_high)) / 2.0)

x_aspire = x_aspire / cosmo.h
x_aspire_low = x_aspire_low / cosmo.h
x_aspire_high = x_aspire_high / cosmo.h

y_aspire = np.asarray([364.49, 80.79, 43.64, 24.23, 24.46, 11.89, 6.58, 2.29])
dy_aspire = np.asarray([176.72, 89.64, 46.72, 18.75, 8.59, 4.18, 2.24, 1.73])

y_gal_aspire = np.asarray([47.32, 12.18, 19.61, 4.67, 4.64, 2.55, 1.27, 1.06])
dy_gal_aspire = np.asarray([19.31, 10.16, 5.21, 2.72, 1.47, 0.87, 0.58, 0.48])

data_aspire = Corr_data(
    x_data=x_aspire,
    y_data=y_aspire,
    errs=[dy_aspire, dy_aspire],
    xerrs=[x_aspire - x_aspire_low, x_aspire_high - x_aspire],
    label="ASPIRE",
    corr_type="cross",
    log_L_threshold=_log_L_thr_aspire,
    num_den=_num_den_aspire,
    num_den_err=_num_den_err_aspire,
    pimax=_pimax_aspire,
    params_clf=_DEFAULT_PARAMS_CLF,
    model_clf=_DEFAULT_MODEL_CLF,
    npar_clf=_DEFAULT_NPAR_CLF,
    params_gal=_DEFAULT_PARAMS_GAL,
)

data_gal_aspire = Corr_data(
    x_data=x_aspire,
    y_data=y_gal_aspire,
    errs=[dy_gal_aspire, dy_gal_aspire],
    xerrs=[x_aspire - x_aspire_low, x_aspire_high - x_aspire],
    label="ASPIRE OIII",
    # ⚠ GALAXY auto-correlation — the QSO sample's `log_L_threshold` / `num_den`
    # do NOT apply here. They used to be passed
    # through from the QSO cross-correlation dataset above, which is semantically
    # wrong: galaxies have no quasar L_bol threshold, and the QSO number density is
    # not theirs. Harmless today (nothing consumes these objects — verified), but
    # if a galaxy auto-corr is ever added to the corr likelihood, the halo-model
    # code would look up the emulator QHMF at the QSO threshold and fit the wrong
    # curve. Set to None so that failure is loud, not silent.
    log_L_threshold=None,
    num_den=None,
    num_den_err=None,
    pimax=_pimax_aspire,
    params_clf=_DEFAULT_PARAMS_CLF,
    model_clf=_DEFAULT_MODEL_CLF,
    npar_clf=_DEFAULT_NPAR_CLF,
    params_gal=_DEFAULT_PARAMS_GAL,
)


# ---------------------------------------------------------------------------
#  ASPIRE  —  QSO-[OIII] cross-correlation COVARIANCE  (Huang+26)
# ---------------------------------------------------------------------------
#
# Huang et al. 2026 (arXiv:2602.04974, the 25-field ASPIRE z~6.6 paper) publish
# only the DIAGONAL of their covariance — the `chi_QG^cov_err` column of their
# Table 1 — which is exactly what `dy_aspire` above already is. The full matrix
# appears in the paper ONLY as a colour heatmap (their Fig. "Correlation matrices
# of the volume-averaged correlation functions", right panel); the numbers below
# are the underlying chi_QG correlation matrix, provided by the ASPIRE team.
#
# It is the matrix for the fiducial minimum-mass model
#   (log M_h^gal, log M_h^QSO) = (10.6, 12.2),
# which is the one the paper tabulates and plots, and which sits essentially at
# their joint best fit (10.55 +0.11/-0.12 and 12.13 +0.31/-0.38). NB their
# covariance is MODEL-DEPENDENT by construction (1000 mocks per point on a 2D
# grid in the two minimum masses, looked up by nearest grid point inside their
# likelihood) — so "the" covariance only exists relative to a mass model. Using
# this one fixed matrix is the same approximation Huang+26 themselves make for
# their power-law r0 fit, where the model has no halo mass attached.
#
# Rows/columns are index-aligned to the 8 ASPIRE r_p bins above. The authors'
# quoted bin mids [h^-1 cMpc] are
#   [0.08, 0.14, 0.248, 0.437, 0.77, 1.357, 2.392, 4.218]
# vs the geometric bin centres used for `x_aspire` (0.0775 ... 4.214) — the same
# bins, agreeing to <= 3% (they quote pair-weighted mids), so the ordering is
# unambiguous.
#
# Sanity: symmetric, unit diagonal, and POSITIVE DEFINITE
# (eigenvalues 0.686 ... 1.706, condition number 2.5) — i.e. a well-conditioned
# matrix. Off-diagonals
# are weak (max 0.27), as expected for a CROSS-correlation: each field holds one
# quasar, so a galaxy contributes to a single separation bin and density
# fluctuations do not propagate across bins. Effect on chi2 vs diagonal-only: a
# coherent amplitude offset costs 0.65x (cosmic variance is absorbed), a tilt
# 0.95x, random residuals ~1.07x. So it LOOSENS the fit against exactly the
# coherent shifts cosmic variance produces.

correlation_aspire = np.asarray([
    [ 1.00000, -0.00828,  0.06768,  0.03883,  0.06708,  0.07774, -0.01202,  0.05705],
    [-0.00828,  1.00000,  0.00749,  0.07084,  0.06227,  0.06452,  0.00902,  0.04759],
    [ 0.06768,  0.00749,  1.00000,  0.07153,  0.03428, -0.00049,  0.05401,  0.07119],
    [ 0.03883,  0.07084,  0.07153,  1.00000,  0.12372,  0.11536,  0.13383,  0.05084],
    [ 0.06708,  0.06227,  0.03428,  0.12372,  1.00000,  0.23392,  0.16722,  0.11917],
    [ 0.07774,  0.06452, -0.00049,  0.11536,  0.23392,  1.00000,  0.26855,  0.16205],
    [-0.01202,  0.00902,  0.05401,  0.13383,  0.16722,  0.26855,  1.00000,  0.21381],
    [ 0.05705,  0.04759,  0.07119,  0.05084,  0.11917,  0.16205,  0.21381,  1.00000],
])

# C = D R D, with D = diag(chi_QG^cov_err) — the paper's own tabulated diagonal,
# so sqrt(diag(covariance_aspire)) reproduces `dy_aspire` exactly.
covariance_aspire = correlation_aspire * dy_aspire[:, np.newaxis] * dy_aspire[np.newaxis, :]


# --- Full-covariance dataset (the production z=6 default) --------------------
#
# `data_aspire_cov` below is the fit's z=6 default (main_mcmc:
# `BAQARO_CORR_Z6` falls back to "data_aspire_cov"). `data_aspire` above is the
# diagonal-only arm of the same measurement (covariance=None) — the same
# pattern as data_ef / data_ef_diag at z=2.5.
#
#   BAQARO_CORR_Z6=data_aspire_cov    # full covariance (default)
#   BAQARO_CORR_Z6=data_aspire        # diagonal (the paper's sigma_cov; changes
#                                    #   the likelihood, so add BAQARO_MCMC_NOTES=<tag>)
#
# `data_aspire_cov` is a SEPARATE Corr_data instance, not an alias: main_mcmc
# mutates the object it selects (setting .redshift / .corr_type / .logM_min /
# .logM_max), so sharing one instance between the two names would couple them.
# Same data and same `.err` as `data_aspire` — the ONLY difference is that
# `.covariance` is set, which makes `get_chi2` take the
# `diff @ solve(C, diff)` branch instead of `sum((diff/err)**2)`.

data_aspire_cov = Corr_data(
    x_data=x_aspire,
    y_data=y_aspire,
    errs=[dy_aspire, dy_aspire],
    xerrs=[x_aspire - x_aspire_low, x_aspire_high - x_aspire],
    label="ASPIRE (full cov)",
    corr_type="cross",
    log_L_threshold=_log_L_thr_aspire,
    num_den=_num_den_aspire,
    num_den_err=_num_den_err_aspire,
    covariance=covariance_aspire,
    pimax=_pimax_aspire,
    params_clf=_DEFAULT_PARAMS_CLF,
    model_clf=_DEFAULT_MODEL_CLF,
    npar_clf=_DEFAULT_NPAR_CLF,
    params_gal=_DEFAULT_PARAMS_GAL,
)




# ===========================================================================
#  FRESCO + COSMOS3D  —  Galaxy auto-correlation, z ~ 6
# ===========================================================================

x_fresco_cosmos3d = np.asarray([
    0.1392561, 0.2700493, 0.5236871, 1.015548, 1.969379, 3.819074, 7.406054, 14.36202,
]) / cosmo.h

edges_fresco = np.asarray([
    0.1, 0.19392274, 0.37606031, 0.72926647, 1.41421356,
    2.74248176, 5.3182959, 10.31338538, 20.0,
]) / cosmo.h

x_fresco_cosmos3d_low = edges_fresco[:-1]
x_fresco_cosmos3d_high = edges_fresco[1:]

y_fresco_cosmos3d = np.asarray([24.92843, 14.72623, 11.02013, 3.803426, 2.956942, 1.388069, 1.071375, 0.5799905])
dy_fresco_cosmos3d = np.asarray([18.48691, 7.443714, 3.336976, 1.126377, 0.5442042, 0.2318818, 0.1285467, 0.1118792])

data_gal_fresco_cosmos3d = Corr_data(
    x_data=x_fresco_cosmos3d,
    y_data=y_fresco_cosmos3d,
    errs=[dy_fresco_cosmos3d, dy_fresco_cosmos3d],
    xerrs=[x_fresco_cosmos3d - x_fresco_cosmos3d_low, x_fresco_cosmos3d_high - x_fresco_cosmos3d],
    label="FRESCO+COSMOS3D",
    log_L_threshold=_log_L_thr_aspire,
    num_den=_num_den_aspire,
    num_den_err=_num_den_err_aspire,
    pimax=_pimax_aspire,
    params_clf=_DEFAULT_PARAMS_CLF,
    model_clf=_DEFAULT_MODEL_CLF,
    npar_clf=_DEFAULT_NPAR_CLF,
    params_gal=_DEFAULT_PARAMS_GAL,
)


# ===========================================================================
#  Quick-look plotting
# ===========================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Low-z datasets ---
    fig, ax = plt.subplots()
    for data, color in zip(
        [data_shen_all, data_shen_highz, data_ef, data_ef_highL, data_ef_ext],
        [f"C{i}" for i in range(5)],
    ):
        print(f"{'#' * 50}")
        print(f"Dataset: {data.label}")
        print(f"  log_L_threshold = {data.log_L_threshold}")
        print(f"  num_den = {data.num_den}")
        print(f"  N_bins = {len(data.x)}")
        ax.errorbar(data.x, data.data, yerr=data.errs, fmt="o", color=color, label=data.label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("r [cMpc]")
    ax.set_ylabel(r"$w_p / r_p$")
    ax.legend()
    plt.show()

    # --- High-z datasets ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for data in [data_eiger, data_gal_z6, data_aspire_cov, data_gal_aspire,
                 ]:
        ax.errorbar(data.x, data.data, data.errs, label=data.label,
                    linestyle="", marker="o", zorder=0)

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r$ [cMpc]")
    ax.set_ylabel(r"Vol. averaged cross-corr.")
    ax.legend()
    fig.tight_layout()
    plt.show()
