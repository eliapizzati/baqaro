"""Bolometric quasar luminosity function — the primary likelihood dataset.

The QLF is the observable the model is principally fitted to. This module owns
its full preparation: loading the per-redshift measurements from the bundled
``data/qlf/all_data_{z}.csv`` tables -- the bolometric QLF compilation of
Shen et al. (2020, MNRAS 495, 3252; X-ray, UV/optical and IR surveys with
their obscuration corrections), one file per redshift slice, columns
``Lbol`` (log10 erg/s), ``Phi`` (log10 dex^-1 cMpc^-3) and their errors --
applying the faint-end and censoring
cuts, rebinning onto a common 0.5-dex grid, and attaching the error model.

Two views of each redshift are exposed:

``data_qlf_global_raw``
    The individual published points, after the Phi-censoring mask.
``data_qlf_global_binned``
    The 0.5-dex rebinned points that actually enter the likelihood, together
    with per-bin true widths and the systematic error floor.

Faint-end threshold
-------------------
Which faint bins are fitted is a likelihood *definition*, not a detail: the
per-redshift floors in ``min_Lbol_binned_per_z`` decide how much of the
poorly-constrained faint end the fit sees. Several environment variables adjust
them (``BAQARO_QLF_MIN_LBOL_PER_Z``, ``_FLOOR``, ``_SHIFT``,
``BAQARO_QLF_PRECISE_FAINT_CUT``, ``BAQARO_QLF_PHI_MIN``). All are read at
import, so they must be set before this module is first imported, and any of
them changes the likelihood — tag affected chains via ``BAQARO_MCMC_NOTES``.

Note that a non-empty ``BAQARO_QLF_MIN_LBOL_PER_Z`` (which has a non-trivial
default) also implies the precise 0.5-dex cut, so the per-z floors are honoured
exactly rather than floored to the nearest integer dex.
"""

import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from qhtools.utils.luminosity_function import create_qlf_Lbol_from_fit
from qhtools.utils import my_utils
# Single source of truth for the Shen+20 "polished" DPL fit parameters (see z_params below).
from baqaro.obs_data.qlf_shen_model import (
    dic_phi_star_polished,
    dic_L_star_polished,
    dic_gamma_1_polished,
    dic_gamma_2_polished,
)

# Bundled QLF measurements, resolved relative to this package so they are found
# both from a source checkout and from an installed copy.
_INPUT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "qlf")


class Qlf_data:
    """One quasar luminosity function, raw or rebinned.

    Parameters
    ----------
    x_data, y_data : array_like
        log10(L_bol / Lsun) and log10(phi / dex^-1 cMpc^-3).
    errs : (array_like, array_like)
        ``[err_down, err_up]`` in dex.
    label : str
        Display label.
    systematic_err : float, optional
        Systematic floor added in quadrature to ALL THREE error views
        (``.err``, ``.err_down``, ``.err_up``), so no consumer can pick up a
        purely statistical error from an object that is meant to carry the
        floor.
    log_L_axis, log_qlf_fit : array_like, optional
        A fitted curve to draw alongside the points.
    covariance : array_like, optional
        Full covariance, when the published errors are correlated.
    dlogL : array_like, optional
        TRUE per-bin widths. These are all ``BIN_WIDTH`` except the topmost,
        which is anchored at ``max(x)`` and can be much narrower. The fitted
        Poisson term is width-independent, but the bright-end non-detection
        penalty is not.
    """
    def __init__(self, x_data, y_data, errs, label, systematic_err=None, log_L_axis=None, log_qlf_fit=None,
                 covariance=None, dlogL=None):
        self.x = np.array(x_data)
        self.data = np.array(y_data)
        self.errs = errs
        # Per-bin TRUE widths (binned datasets only; None for raw/unbinned).
        # The topmost bin can be narrower than BIN_WIDTH after the top-edge fix.
        # Consumed by the QLF Poisson precompute when
        # BAQARO_QLF_NONDET_TRUE_WIDTH=1.
        self.dlogL = np.array(dlogL) if dlogL is not None else None

        # Calculate average error
        self.err_down = np.asarray(errs[0], dtype=float)
        self.err_up = np.asarray(errs[1], dtype=float)
        self.err = (self.err_down + self.err_up) / 2.

        self.label = label

        if systematic_err is not None:
            self.sys_err = systematic_err
            # The systematic floor is added in quadrature to ALL THREE error
            # views, not just `.err`.
            self.err = np.sqrt(self.err ** 2 + self.sys_err ** 2)
            self.err_down = np.sqrt(self.err_down ** 2 + self.sys_err ** 2)
            self.err_up = np.sqrt(self.err_up ** 2 + self.sys_err ** 2)
            self.errs = [self.err_down, self.err_up]
        else:
            self.sys_err = None

        self.log_L_axis = log_L_axis
        self.log_qlf_fit = log_qlf_fit

        self.covariance = covariance


def rebin_data(x, y, y_err, min_Lbol, max_Lbol, bin_width=0.5, return_widths=False):
    """
    Groups data into bins and calculates weighted average and PURE statistical error.
    Does NOT apply clipping, as that is handled by the Qlf_data class.

    ``return_widths=True`` additionally returns the TRUE width of each surviving
    bin. These are all ``bin_width`` except the topmost, whose upper edge is
    anchored at ``max(x)`` and which can therefore be much narrower (e.g.
    0.04 dex at z=3). The fitted-bin Poisson likelihood is width-independent
    (the width cancels), but the bright-end NON-DETECTION penalty is not — see
    BAQARO_QLF_NONDET_TRUE_WIDTH.
    """
    if globals().get("_QLF_PRECISE_CUT", False):
        # An env-var faint threshold is active: snap the bin grid start to the
        # bin_width grid so the requested min_Lbol is honored at 0.5-dex
        # granularity (not rounded down to the nearest integer).
        start = np.floor(min_Lbol / bin_width) * bin_width
    else:
        # Legacy default: round the faint cut DOWN to the nearest integer.
        start = np.floor(min_Lbol)
    # TOP EDGE: anchor the topmost detection-bin edge to the brightest observed
    # data point, so the brightest bin never extends past any actual data.
    # Standard 0.5-dex grid up to `floor(max(x)/bw)*bw`,
    # then a final edge AT max(x) — so the last bin has upper edge exactly at
    # the brightest data, with its centre at the data's mid-range. The Poisson
    # likelihood is bin-width-independent (Lambda = Phi_model * V_eff * dlogL
    # = Phi_model * N_eff / Phi_data, dlogL cancels) so V_eff plumbing
    # (BIN_WIDTH=0.5) does not need changing.
    end_data = float(np.max(x))
    if end_data <= start:
        if return_widths:
            return (np.array([]), np.array([]), np.array([]), np.array([]))
        return np.array([]), np.array([]), np.array([])
    # Full 0.5-dex edges from start up to (but not exceeding) end_data:
    full_edges = np.arange(start, end_data, bin_width)
    # DEGENERATE TOP SLIVER:
    # when max(x) lands just ABOVE a bin_width grid point, appending a final
    # edge at max(x) creates a near-zero-width top bin holding a single point.
    # Measured: z=1 max(x)=48.000398 -> top bin width 3.98e-4 dex (!); z=4 ->
    # 0.021; z=3 -> 0.042. Harmless for the FITTED Poisson bins (the width
    # cancels), but it makes any true-width V_eff explode (V_eff ∝ 1/w), and a
    # 1-point sliver bin is not a meaningful measurement anyway.
    # BAQARO_QLF_MERGE_TOP_SLIVER=1 merges such a sliver into the previous bin
    # (replace the last full edge with max(x) instead of appending), giving a
    # top bin of width ~bin_width.
    # Default ON (the adopted fiducial, `qcc_ck22final_v1`, was fitted with the
    # merge): the top partial-width sliver bin is merged into the previous full
    # bin. Override with BAQARO_QLF_MERGE_TOP_SLIVER=0; that changes the data
    # vector, so tag the chain.
    _merge_sliver = (os.environ.get("BAQARO_QLF_MERGE_TOP_SLIVER", "1").strip().lower()
                     not in ("0", "", "false", "no", "off"))
    _sliver_frac = float(os.environ.get("BAQARO_QLF_SLIVER_FRAC", "0.1"))
    if full_edges.size == 0 or full_edges[-1] < end_data - 1e-12:
        residual = end_data - full_edges[-1] if full_edges.size else np.inf
        if _merge_sliver and full_edges.size >= 2 and residual < _sliver_frac * bin_width:
            # Merge: extend the last FULL bin up to max(x) rather than adding a
            # sliver bin on top of it.
            bins = np.concatenate([full_edges[:-1], [end_data]])
        else:
            bins = np.concatenate([full_edges, [end_data]])
    else:
        bins = full_edges
    
    x_binned = []
    y_binned = []
    y_err_binned = []
    dlogL_binned = []   # TRUE width of each surviving bin (the top one can be
                        # narrower than bin_width after the top-edge clipping)

    indices = np.digitize(x, bins)

    for i in range(1, len(bins)):
        mask = indices == i
        # Include data at the top edge (x == bins[-1] = max(x)) in the LAST
        # bin. Without this, np.digitize(right=False) puts x == max(x) at
        # index len(bins) (past the last interior bin) and the brightest
        # observed point is silently dropped — visible at z=6 where the raw
        # max=47.622 fell exactly on the new top edge.
        if i == len(bins) - 1:
            mask = mask | (x >= bins[-1])
        if np.sum(mask) > 0:
            current_x = x[mask]
            current_y = y[mask]
            current_err = y_err[mask]
            
            # 1. Weighted Average of Y (Phi)
            weights = 1.0 / (current_err**2)
            sum_weights = np.sum(weights)
            
            if sum_weights > 0:
                weighted_mean_y = np.sum(current_y * weights) / sum_weights
                # Standard Error of the Weighted Mean = 1 / sqrt(Sum of Weights)
                stat_err = np.sqrt(1.0 / sum_weights)
            else:
                weighted_mean_y = np.mean(current_y)
                stat_err = np.mean(current_err)

            # 2. X position: bin CENTER (equally-spaced points at the 0.5-dex
            # grid). Previously the data barycenter (np.mean(current_x)), which
            # made the inference-ready points unevenly spaced and bunched at the
            # bright end (the steep QLF pulls each bin's barycenter to its faint
            # edge). The bin center pairs naturally with the bin-averaged Phi.
            x_binned.append(0.5 * (bins[i - 1] + bins[i]))
            y_binned.append(weighted_mean_y)
            y_err_binned.append(stat_err)
            dlogL_binned.append(bins[i] - bins[i - 1])

    if return_widths:
        return (np.array(x_binned), np.array(y_binned), np.array(y_err_binned),
                np.array(dlogL_binned))
    return np.array(x_binned), np.array(y_binned), np.array(y_err_binned)


# --- CONFIG ---

min_Lbol_raw = 43.5
max_Lbol_raw = 48.5
max_Lbol_binned = 48.5
# Redshift-dependent minimum Lbol for binning
min_Lbol_binned_per_z = {
    '0.2': 45.0,
    '1.0': 45.0,
    '2.0': 45.5,
    '3.0': 45.5,
    '4.0': 46.0,
    '5.0': 46.0,
    '6.0': 46.0,
    '7.0': 46.0,
}
min_Lbol_binned_default = 46.0

# --- Optional env-var control of the QLF faint-end fit threshold ---
# Lets you raise the minimum L_bol that enters the QLF likelihood (i.e. fit
# only brighter QSOs) for threshold sweeps / re-optimization, without editing
# the per-z dict above. Applied at import, so any consumer (main_mcmc,
# plotting) that imports this module with the var set sees the shifted floors.
#   BAQARO_QLF_MIN_LBOL_SHIFT  -- uniform additive dex shift on every per-z floor
#                                AND the default. e.g. "0.5" -> 45.5/46.0/46.5.
#   BAQARO_QLF_MIN_LBOL_FLOOR  -- absolute uniform floor that REPLACES the per-z
#                                values entirely (takes precedence). e.g. "46.0".
#   BAQARO_QLF_MIN_LBOL_PER_Z  -- per-redshift floor overrides, comma-separated
#                                "z:logL" pairs, e.g. "2.0:45.0,3.0:45.0,4.0:45.5,
#                                5.0:45.5". Applied LAST (overrides FLOOR/SHIFT for
#                                the listed z only); keys must match the dict keys
#                                above. Implies the precise 0.5-dex cut, so a 45.5
#                                override is honored exactly (see below).
_qlf_min_floor = os.environ.get("BAQARO_QLF_MIN_LBOL_FLOOR", "").strip()
_qlf_min_shift = float(os.environ.get("BAQARO_QLF_MIN_LBOL_SHIFT", "0.0"))
# Default: the per-z faint floors the adopted fiducial (`qcc_ck22final_v1`) was
# fitted with — z=2/3 at 45.0, z=4/5 at 45.5, z=6/7 at the dict's 46.0 (41
# fitted QLF points). Override with an explicit string, or
# BAQARO_QLF_MIN_LBOL_PER_Z= (any empty value) to fall back to the dict above.
# Applied LAST (overrides FLOOR/SHIFT for the listed z).
_qlf_min_per_z = os.environ.get(
    "BAQARO_QLF_MIN_LBOL_PER_Z",
    "2.0:45.0,3.0:45.0,4.0:45.5,5.0:45.5").strip()
# When a threshold env var is active, rebin_data honors min_Lbol at 0.5-dex
# granularity (see rebin_data). With no env var, behavior is byte-identical to
# before (integer floor) so prior baseline runs stay reproducible.
#
# BAQARO_QLF_PRECISE_FAINT_CUT=1 (opt-in, default OFF):
# activates the SAME precise 0.5-dex grid WITHOUT changing the per-z floor
# values, so a half-dex floor in the dict is honoured exactly instead of being
# rounded down to the integer. CHANGES THE LIKELIHOOD when a floor is not an
# integer -> tag chains via BAQARO_MCMC_NOTES.
_qlf_precise_flag = (os.environ.get("BAQARO_QLF_PRECISE_FAINT_CUT", "")
                     .strip().lower() in ("1", "true", "yes", "on"))
_QLF_PRECISE_CUT = (bool(_qlf_min_floor) or (_qlf_min_shift != 0.0)
                    or bool(_qlf_min_per_z) or _qlf_precise_flag)
if _qlf_precise_flag:
    print("[qlf_obs_data] BAQARO_QLF_PRECISE_FAINT_CUT=1: faint cut honored at "
          "0.5-dex granularity (45.5 stays 45.5 at z=2/3). CHANGES the "
          "likelihood - tag chains via BAQARO_MCMC_NOTES.")
if _qlf_min_floor:
    _floor_val = float(_qlf_min_floor)
    min_Lbol_binned_per_z = {k: _floor_val for k in min_Lbol_binned_per_z}
    min_Lbol_binned_default = _floor_val
    print(f"[qlf_obs_data] BAQARO_QLF_MIN_LBOL_FLOOR={_floor_val}: uniform QLF faint-end cut at log L_bol={_floor_val}")
elif _qlf_min_shift != 0.0:
    min_Lbol_binned_per_z = {k: v + _qlf_min_shift for k, v in min_Lbol_binned_per_z.items()}
    min_Lbol_binned_default += _qlf_min_shift
    print(f"[qlf_obs_data] BAQARO_QLF_MIN_LBOL_SHIFT=+{_qlf_min_shift} dex applied -> {min_Lbol_binned_per_z}")
if _qlf_min_per_z:
    for _pair in _qlf_min_per_z.split(","):
        if not _pair.strip():
            continue
        _zk, _lv = _pair.split(":")
        _zk = _zk.strip()
        if _zk not in min_Lbol_binned_per_z:
            raise ValueError(
                f"BAQARO_QLF_MIN_LBOL_PER_Z: unknown redshift key '{_zk}'. "
                f"Valid keys: {sorted(min_Lbol_binned_per_z)}")
        min_Lbol_binned_per_z[_zk] = float(_lv)
    print(f"[qlf_obs_data] BAQARO_QLF_MIN_LBOL_PER_Z applied -> {min_Lbol_binned_per_z}")

BIN_WIDTH = 0.5
# Minimum systematic error added in quadrature to each QLF bin (in dex). Feeds
# `data_qlf_global_sys_err_binned` (Gaussian error inflation + the Poisson
# degradation factor) and the error clipping below. Env-tunable for the
# QLF-systematic robustness scan; default 0.3 = byte-identical. Changing it
# alters the likelihood -> tag chains via BAQARO_MCMC_NOTES.
SYS_ERR_FLOOR = float(os.environ.get("BAQARO_QLF_SYS_ERR_FLOOR", "0.3"))
if SYS_ERR_FLOOR != 0.3:
    print(f"[qlf_obs_data] BAQARO_QLF_SYS_ERR_FLOOR={SYS_ERR_FLOOR} (default 0.3)")
N_POISSON_THRESHOLD = 15  # Bins with N_eff < this are flagged as Poisson-dominated

# --- Optional control of the raw-data Phi censoring floor ---
# The raw-data mask drops points with log Phi <= -9, which removes the
# brightest, sparsest survey points at every redshift. (Upper limits are
# already removed separately by the dy > 0 cut.)
#   BAQARO_QLF_PHI_MIN=<float>   -- move the floor (default "-9" = byte-identical
#                                  legacy behavior).
#   BAQARO_QLF_PHI_MIN=none/off  -- disable the censoring entirely (keep every
#                                  detection regardless of Phi).
# CHANGES THE LIKELIHOOD when non-default -> tag chains via BAQARO_MCMC_NOTES.
_qlf_phi_min_env = os.environ.get("BAQARO_QLF_PHI_MIN", "").strip().lower()
if _qlf_phi_min_env in ("none", "off", "disable", "disabled"):
    QLF_PHI_MIN = -np.inf
    print("[qlf_obs_data] BAQARO_QLF_PHI_MIN=none: log Phi censoring DISABLED "
          "(brightest survey points retained). CHANGES the likelihood "
          "- tag chains via BAQARO_MCMC_NOTES.")
elif _qlf_phi_min_env:
    QLF_PHI_MIN = float(_qlf_phi_min_env)
    print(f"[qlf_obs_data] BAQARO_QLF_PHI_MIN={QLF_PHI_MIN}: non-default Phi "
          "censoring floor. CHANGES the likelihood - tag chains via "
          "BAQARO_MCMC_NOTES.")
else:
    QLF_PHI_MIN = -9.0   # legacy default, byte-identical

data_qlf_global_raw = {}
data_qlf_global_binned = {}
data_qlf_global_sys_err_binned = {}

# QLF double-power-law fit params per redshift: [log10(phi_star), log10(L_star)[erg/s], gamma_1, gamma_2].
# Built DIRECTLY from qlf_shen_model's `*_polished` dicts (the single source of truth) rather than
# re-transcribed here. Only the plotted DPL curve uses them (`log_qlf_fit`); the likelihood uses
# the binned data, not the fit.
#
# z=0.2 and z=1.0 stay PLACEHOLDERS by choice: qlf_shen_model has no z=1 entry at all, and its z=0.2
# polished fit is deliberately not adopted here (doing so would change the plotted z=0.2 curve).
_SHEN_FIT_Z = ('2.0', '3.0', '4.0', '5.0', '6.0')
z_params = {
    '0.2': [-5.0, my_utils.to_ergs(12.0), 0.0, 1.0],          # placeholder (see above)
    '1.0': [-5.0, my_utils.to_ergs(12.0), 0.0, 1.0],          # placeholder (no Shen fit at z=1)
}
for _zk in _SHEN_FIT_Z:
    z_params[_zk] = [
        dic_phi_star_polished[_zk],
        my_utils.to_ergs(dic_L_star_polished[_zk]),
        dic_gamma_1_polished[_zk],
        dic_gamma_2_polished[_zk],
    ]

# z=7 data: no Shen+20 CSV, hardcoded from Matsuoka+2023
_z7_x   = np.array([47.23480608, 46.96544608, 46.74340608])
_z7_y   = np.array([-9.38791512, -9.00901749, -8.64781748]) + 0.9  # +0.9 dex shift to roughly account for obscuration (not applied to lower-z Shen+20 data, which is already corrected)
_z7_err = np.array([0.14061851, 0.09378318, 0.17542861])
_z7_fit_params = [-5.452, my_utils.to_ergs(11.978), 1.509, 1.509]

# --- MAIN LOOP ---
for key, params in z_params.items():
    file_path = os.path.join(_INPUT_DATA_DIR, f'all_data_{key}.csv')

    # Deliberately no try/except: a missing input file must fail loudly rather
    # than silently drop a redshift out of the likelihood.
    data = pd.read_csv(file_path)

    x_obs = data['Lbol'].values
    y_obs = data['Phi'].values
    dy_obs = data['Error_Phi'].values

    # 2. FILTER DATA
    # QLF_PHI_MIN: env-controllable Phi censoring floor (default -9.0, legacy;
    # see the BAQARO_QLF_PHI_MIN block above).
    mask_data_raw = (x_obs > min_Lbol_raw) & (x_obs < max_Lbol_raw) & (y_obs > QLF_PHI_MIN) & (dy_obs > 0.)
    
    x_obs = x_obs[mask_data_raw]
    y_obs = y_obs[mask_data_raw]
    dy_obs = dy_obs[mask_data_raw]
    
    if len(x_obs) == 0:
        warnings.warn(
            f"QLF z={key}: every point was masked out; check the min/max "
            f"log L_bol range and the Phi censoring floor. This redshift will "
            f"be absent from the likelihood.",
            RuntimeWarning, stacklevel=2)
        continue

    data_qlf_global_raw[key] = Qlf_data(
        x_data=x_obs, 
        y_data=y_obs, 
        errs=[dy_obs, dy_obs], 
        label=f"Shen+20 z={key}",
        systematic_err=None
    )

    # 3. REBIN DATA
    min_Lbol_binned = min_Lbol_binned_per_z.get(key, min_Lbol_binned_default)
    x_binned, y_binned, dy_binned, dlogL_binned = rebin_data(
        x_obs, y_obs, dy_obs, min_Lbol_binned, max_Lbol_binned, BIN_WIDTH, return_widths=True)
    
    
    # 4. STORE
    log_L_axis, log_qlf_fit = create_qlf_Lbol_from_fit(
        params, lowest_logL=44., highest_logL=49., n_points=100
    )

    label_str = f"Shen+20 z={key}"
    
    # Store in dict
    data_qlf_global_binned[key] = Qlf_data(
        x_data=x_binned, 
        y_data=y_binned, 
        errs=[dy_binned, dy_binned], 
        label=label_str,
        log_L_axis=log_L_axis, 
        log_qlf_fit=log_qlf_fit,
        dlogL=dlogL_binned,
    )

    data_qlf_global_sys_err_binned[key] = Qlf_data(
        x_data=x_binned,
        y_data=y_binned,
        errs=[dy_binned, dy_binned],
        label=label_str,
        systematic_err=SYS_ERR_FLOOR,
        log_L_axis=log_L_axis,
        log_qlf_fit=log_qlf_fit,
        dlogL=dlogL_binned,
    )



# --- z=7 (Matsuoka+2023, hardcoded — see _z7_x/y/err above) ---
# This block is loaded; main_mcmc masks z=7 from the QLF likelihood by default
# (mask_redshifts_qlf=[7.0]). Set BAQARO_QLF_INCLUDE_Z7=1 to include z=7 in the fit.
_z7_key = '7.0'

data_qlf_global_raw[_z7_key] = Qlf_data(
    x_data=_z7_x, y_data=_z7_y, errs=[_z7_err, _z7_err],
    label="Matsuoka+23 z=7",
)

_z7_min_Lbol = min_Lbol_binned_per_z.get(_z7_key, min_Lbol_binned_default)
_z7_xb, _z7_yb, _z7_eb, _z7_wb = rebin_data(
    _z7_x, _z7_y, _z7_err, _z7_min_Lbol, max_Lbol_binned, BIN_WIDTH, return_widths=True)

_z7_L_axis, _z7_qlf_fit = create_qlf_Lbol_from_fit(
    _z7_fit_params, lowest_logL=44., highest_logL=49., n_points=100
)
data_qlf_global_binned[_z7_key] = Qlf_data(
    x_data=_z7_xb, y_data=_z7_yb, errs=[_z7_eb, _z7_eb],
    label="Matsuoka+23 z=7", log_L_axis=_z7_L_axis, log_qlf_fit=_z7_qlf_fit,
    dlogL=_z7_wb,
)
data_qlf_global_sys_err_binned[_z7_key] = Qlf_data(
    x_data=_z7_xb, y_data=_z7_yb, errs=[_z7_eb, _z7_eb],
    label="Matsuoka+23 z=7", systematic_err=SYS_ERR_FLOOR,
    log_L_axis=_z7_L_axis, log_qlf_fit=_z7_qlf_fit,
    dlogL=_z7_wb,
)




# --- N_eff / V_eff DIAGNOSTICS ---
# Compute effective counts and survey volumes from the binned statistical errors.
# This runs at import time so the data is always available for inspection.

LN10 = np.log(10.0)
data_qlf_diagnostics = {}

for key in data_qlf_global_binned:
    d_stat = data_qlf_global_binned[key]      # pure statistical errors
    d_full = data_qlf_global_sys_err_binned[key]  # stat + systematic

    sigma_stat = np.asarray(d_stat.err)
    sigma_full = np.asarray(d_full.err)
    logPhi = np.asarray(d_stat.data)

    N_eff = 1.0 / (LN10 * sigma_stat) ** 2
    N_eff = np.maximum(N_eff, 0.5)

    Phi_linear = 10.0 ** logPhi
    V_eff_raw = N_eff / (Phi_linear * BIN_WIDTH)

    # Degrade V_eff by (sigma_stat/sigma_full)^2 to account for systematic
    degradation = (sigma_stat / sigma_full) ** 2
    V_eff = V_eff_raw * degradation

    # Effective observed count after degradation
    N_obs = np.round(Phi_linear * V_eff * BIN_WIDTH).astype(int)
    N_obs = np.maximum(N_obs, 0)

    is_poisson = N_eff < N_POISSON_THRESHOLD

    data_qlf_diagnostics[key] = {
        'x': np.asarray(d_stat.x),
        'logPhi': logPhi,
        'sigma_stat': sigma_stat,
        'sigma_full': sigma_full,
        'N_eff': N_eff,
        'N_obs': N_obs,
        'V_eff_raw': V_eff_raw,
        'V_eff': V_eff,
        'degradation': degradation,
        'is_poisson': is_poisson,
    }


# --- PLOTTING ---
if __name__ == "__main__":
    plot_all_qlfs = True
    plot_binning_debug = True
    plot_neff_diagnostics = True

    if plot_all_qlfs:
        print("\n--- Plotting ---")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.set_xlabel(r"$\log_{10} L_{bol}$", fontsize=14)
        ax.set_ylabel(r"$\log_{10} \Phi$", fontsize=14)

        count = 0
        for key, qlf in data_qlf_global_sys_err_binned.items():
            if len(qlf.x) > 0:
                print(f"Plotting {key}: {len(qlf.x)} points")
                ax.errorbar(qlf.x, qlf.data, yerr=qlf.err, fmt='o', label=qlf.label,
                            markersize=6, capsize=4, elinewidth=1.5, alpha=0.8)
                
                # Optional: fit line
                # ax.plot(qlf.log_L_axis, qlf.log_qlf_fit, lw=2, alpha=0.3)
                count += 1
            else:
                print(f"Skipping {key}: Empty data in Qlf object")

        if count == 0:
            print("CRITICAL: Nothing was plotted.")
        else:
            ax.legend()
            plt.tight_layout()
            plt.show()

    if plot_binning_debug:
        # --- DEBUG CONFIG ---
        target_z = '1.0'  # Change this to check different redshifts
        min_Lbol_raw = 43.5
        max_Lbol_raw = 48.5
        min_Lbol_binned = 45.5
        max_Lbol_binned = 48.5
        bin_width = BIN_WIDTH
        min_err_clip = SYS_ERR_FLOOR  # Minimum error to add in quadrature for clipping

        # --- LOAD DATA ---
        file_path = os.path.join(_INPUT_DATA_DIR, f'all_data_{target_z}.csv')
        print(f"Loading data for z={target_z}...")

        data = pd.read_csv(file_path)
        
        x_raw = data['Lbol'].values
        y_raw = data['Phi'].values
        dy_raw = data['Error_Phi'].values

        # Masking
        mask = (x_raw > min_Lbol_raw) & (x_raw < max_Lbol_raw) & (y_raw > -9.) & (dy_raw > 0.)
        x_raw = x_raw[mask]
        y_raw = y_raw[mask]
        dy_raw = dy_raw[mask]

        print(f"Raw data points: {len(x_raw)}")

        # Binning
        x_bin, y_bin, dy_bin = rebin_data(x_raw, y_raw, dy_raw, min_Lbol_binned, max_Lbol_binned, bin_width=bin_width)
        
        print(f"Binned data points: {len(x_bin)}")

        # --- PLOT ---
        fig, ax = plt.subplots(figsize=(8, 6))

        # 1. Plot Raw Data (faint background)
        ax.errorbar(x_raw, y_raw, yerr=dy_raw, fmt='.', color='gray', alpha=0.3, 
                    label=f'Raw Data (N={len(x_raw)})', zorder=1)

        # 2. Plot Binned Data (prominent foreground)
        ax.errorbar(x_bin, y_bin, yerr=dy_bin, fmt='o', color='red', capsize=5, elinewidth=2, 
                    label=f'Binned (N={len(x_bin)})\nWidth={bin_width}', zorder=2)

        dy_bin_sys = np.sqrt(dy_bin**2 + min_err_clip**2)
        ax.errorbar(x_bin, y_bin, yerr=dy_bin_sys, fmt='o', color='blue', capsize=5, elinewidth=2, alpha=0.7,
                    label=f'Binned + SysErr (N={len(x_bin)})\nSysErr={min_err_clip}', zorder=3)

        ax.set_xlabel(r"$\log_{10} L_{bol}$", fontsize=14)
        ax.set_ylabel(r"$\log_{10} \Phi$", fontsize=14)
        ax.set_title(f"Re-binning Check: z={target_z}", fontsize=15)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.show()

    if plot_neff_diagnostics:
        # --- N_eff / V_eff diagnostic table and plots ---
        print("\n" + "=" * 90)
        print("N_eff / V_eff DIAGNOSTICS (from binned statistical errors)")
        print("=" * 90)

        for key in sorted(data_qlf_diagnostics.keys(), key=float):
            diag = data_qlf_diagnostics[key]
            print(f"\n  z = {key}")
            print(f"  {'logL':>6s}  {'logPhi':>7s}  {'sig_stat':>8s}  {'sig_full':>8s}  "
                  f"{'N_eff':>7s}  {'degrad':>6s}  {'N_obs':>5s}  {'log10(V)':>8s}  {'log10(Vdeg)':>11s}  {'mode':>8s}")
            print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*8}  "
                  f"{'-'*7}  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*11}  {'-'*8}")
            for j in range(len(diag['x'])):
                mode = "POISSON" if diag['is_poisson'][j] else "GAUSS"
                log_vraw = np.log10(diag['V_eff_raw'][j]) if diag['V_eff_raw'][j] > 0 else -np.inf
                log_vdeg = np.log10(diag['V_eff'][j]) if diag['V_eff'][j] > 0 else -np.inf
                print(f"  {diag['x'][j]:6.2f}  {diag['logPhi'][j]:7.3f}  "
                      f"{diag['sigma_stat'][j]:8.4f}  {diag['sigma_full'][j]:8.4f}  "
                      f"{diag['N_eff'][j]:7.1f}  {diag['degradation'][j]:6.4f}  "
                      f"{diag['N_obs'][j]:5d}  {log_vraw:8.2f}  {log_vdeg:11.2f}  {mode:>8s}")

        # --- Plot: N_eff vs logL for all redshifts ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        try:
            from baqaro.plotting_common.plot_config import z_scalar_mapper as _zsm
        except ImportError:
            _zsm = None

        for key in sorted(data_qlf_diagnostics.keys(), key=float):
            diag = data_qlf_diagnostics[key]
            color = _zsm.to_rgba(float(key)) if _zsm is not None else None
            ax1.semilogy(diag['x'], diag['N_eff'], 'o-', label=f"z={key}", color=color)
            ax2.semilogy(diag['x'], diag['V_eff'], 'o-', label=f"z={key}", color=color)

            # Mark Poisson bins
            poisson = diag['is_poisson']
            if poisson.any():
                ax1.scatter(diag['x'][poisson], diag['N_eff'][poisson],
                           marker='x', s=100, color='red', zorder=5)
                ax2.scatter(diag['x'][poisson], diag['V_eff'][poisson],
                           marker='x', s=100, color='red', zorder=5)

        ax1.axhline(N_POISSON_THRESHOLD, color='gray', ls='--', lw=1, label=f'N_threshold={N_POISSON_THRESHOLD}')
        ax1.set_xlabel(r"$\log_{10} L_{\mathrm{bol}}$ [erg/s]")
        ax1.set_ylabel(r"$N_{\mathrm{eff}}$")
        ax1.set_title("Effective counts per bin")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)

        ax2.set_xlabel(r"$\log_{10} L_{\mathrm{bol}}$ [erg/s]")
        ax2.set_ylabel(r"$V_{\mathrm{eff}}$ [Mpc$^3$]")
        ax2.set_title("Effective volume per bin")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)

        fig.suptitle(f"Red X = Poisson bins (N_eff < {N_POISSON_THRESHOLD} & stat error dominates)", fontsize=11)
        plt.tight_layout()
        plt.show()
