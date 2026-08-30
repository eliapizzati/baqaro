"""Shared helpers for the local black-hole scaling relations.

The M_BH-sigma and M_BH-M_star relations are drawn both as working figures and
as one panel of the paper's ``local_relations`` figure. The pieces both need
live here so neither has to import from the other:

* selection floors and the random-subsample size, so the two renderings sample
  the population identically;
* ``binned_median_envelope``, the weight-aware median-and-percentile band --
  it takes the Horvitz-Thompson weights, so it is correct on a subsampled run
  where a plain median would not be;
* ``overlay_mmstar_data`` and the fit evaluators, which draw the published
  relations from :mod:`~baqaro.obs_data.m_sigma_m_star_obs_data`.

Import the published fits themselves (``RV15_MMSTAR``, ``KH13_MBULGE``,
``KH13_MSIGMA``) from ``obs_data`` directly rather than through here.
"""

import numpy as np
import matplotlib.pyplot as plt

from baqaro.obs_data.m_sigma_m_star_obs_data import (
    gs2023_mmstar,
    gs2023_binned_husko26,
    bin_gs2023_scatter,
    KH13_MBULGE,
    RV15_MMSTAR,
)
from baqaro.plotting_common.load_data_to_plot import weighted_percentile
from baqaro.utils.sim_config import subsample_log_M_lo as _SIM_LOG_M_LO

#: Marker style per Graham & Sahu 2023 morphological class. Lives here rather
#: than in a figure script because both the paper and the working figures draw
#: the same sample and must not drift apart.
_GS_TYPE_STYLE = {
    "E":     dict(marker="o", color="firebrick",  label="G&S23 E"),
    "ES/S0": dict(marker="s", color="darkorange", label="G&S23 ES/S0"),
    "S":     dict(marker="^", color="steelblue",  label="G&S23 S"),
}

REDSHIFTS_TO_PLOT = [0.0, 2.0, 4.0, 6.0, 8.0]
LOG_MBH_MIN = 5.0
# Sim-aware halo-mass floor: sim's resolution + 0.5 dex safety margin.
# L2800N5040 (subsample_log_M_lo=11.0) -> 11.5; L2800N10080 (10.0) -> 10.5.
# Override here for a fixed value if you're cross-sim-comparing.
LOG_MHALO_RES_MARGIN = 0.5
LOG_MHALO_MIN = _SIM_LOG_M_LO + LOG_MHALO_RES_MARGIN
MIN_COUNT_PER_BIN = 20
N_HIST_BINS = 70

RANDOM_SUBSAMPLE_PER_SNAP = 1_000_000
RANDOM_SUBSAMPLE_SEED = 42


def _eval_fit_mstar(log10_Mstar, fit):
    """Power-law log M_BH = alpha + beta (log M_* - log mstar0)."""
    return fit["alpha"] + fit["beta"] * (log10_Mstar - np.log10(fit["mstar0"]))


def plot_fit_with_scatter(ax, xs, ys, scatter, color, label):
    """Draw the fit line plus +/-1sigma intrinsic-scatter band."""
    ax.plot(xs, ys, color=color, ls="--", lw=1.4, alpha=0.9, label=label,
            zorder=3)
    ax.fill_between(xs, ys - scatter, ys + scatter,
                    color=color, alpha=0.10, zorder=1,
                    label=f"{label} $\\pm$1$\\sigma$ ({scatter:.2f} dex)")

def binned_median_envelope(x, y, weights, x_edges, min_count=MIN_COUNT_PER_BIN):
    """Weighted 16/50/84th percentiles of ``y`` in bins of ``x``.

    Bins holding fewer than ``min_count`` points are left as NaN rather than
    reported, so a sparsely populated tail does not masquerade as a measured
    envelope.

    Returns
    -------
    centers, p16, p50, p84 : ndarray
        Bin centres and the three percentile curves, each of length
        ``len(x_edges) - 1``, with NaN wherever the bin was under-populated.
    """
    centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    p16 = np.full(centers.size, np.nan)
    p50 = np.full(centers.size, np.nan)
    p84 = np.full(centers.size, np.nan)
    for i in range(centers.size):
        m = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if m.sum() < min_count:
            continue
        p16[i] = weighted_percentile(y[m], weights[m], 0.16)
        p50[i] = weighted_percentile(y[m], weights[m], 0.50)
        p84[i] = weighted_percentile(y[m], weights[m], 0.84)
    return centers, p16, p50, p84

def overlay_mmstar_data(ax, x_range=(8.0, 12.5), show_data_scatter=True,
                        show_fits=True):
    """Scatter the Graham&Sahu 2023 M-M_*,gal sample (with 2024 erratum
    correction already in the loader) split by morphological type,
    plus the Huško+26 Table G1 morphology-weighted binned median.

    If `show_data_scatter`, overlay the true 16-50-84 percentile envelope
    of the G&S 2023 data per stellar-mass bin (the actual intrinsic
    scatter -- distinct from the Huško error bars, which are the
    uncertainty on the median).

    If `show_fits`, add Reines&Volonteri 2015 and K&H 2013 M-M_bulge
    reference power-law fits with their +/-1sigma intrinsic-scatter
    bands.
    """
    d = gs2023_mmstar
    for gt, sty in _GS_TYPE_STYLE.items():
        sel = d["gtype"] == gt
        ax.errorbar(d["log_Mstar_gal"][sel], d["log_MBH"][sel],
                    xerr=d["log_Mstar_gal_err"][sel],
                    yerr=d["log_MBH_err"][sel],
                    fmt=sty["marker"], color=sty["color"],
                    ms=4, lw=0, elinewidth=0.7, capsize=0,
                    alpha=0.85, zorder=4, label=sty["label"])

    b = gs2023_binned_husko26
    ax.errorbar(b["log_Mstar_gal"], b["log_MBH"],
                xerr=b["log_Mstar_gal_err"], yerr=b["log_MBH_err"],
                fmt="D", color="k", ms=8, mfc="gold", mec="k",
                lw=0, elinewidth=1.4, capsize=3, zorder=5,
                label="Huško+26 binned median\n(err = median uncertainty)")

    if show_data_scatter:
        scat_edges = np.linspace(9.5, 12.0, 8)
        c_b, p16_b, p50_b, p84_b, n_b = bin_gs2023_scatter(scat_edges)
        valid = np.isfinite(p50_b) & (n_b >= 4)
        if valid.any():
            ax.fill_between(c_b[valid], p16_b[valid], p84_b[valid],
                            color="gold", alpha=0.25, zorder=2,
                            label="G&S23 data 16-84% (per bin)")

    if show_fits:
        xs = np.linspace(x_range[0], x_range[1], 100)
        ys_rv = _eval_fit_mstar(xs, RV15_MMSTAR)
        plot_fit_with_scatter(ax, xs, ys_rv, RV15_MMSTAR["scatter_dex"],
                              color="navy", label="R&V 2015")
        ys_kh = _eval_fit_mstar(xs, KH13_MBULGE)
        plot_fit_with_scatter(ax, xs, ys_kh, KH13_MBULGE["scatter_dex"],
                              color="dimgray", label="K&H 2013 (bulge)")
