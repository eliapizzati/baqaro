"""Local-universe M_BH-sigma and M_BH-M_* compilations.

Two datasets:

* `sahu19_msigma`  -- Sahu, Graham & Davis 2019 (ApJ 887, 10; arXiv:1908.06838),
  Table 1.  M_BH vs central stellar velocity dispersion sigma for 143 local
  galaxies (early- + late-type).  Errors are on log M_BH only; sigma error
  bars are not tabulated in their paper.

* `gs2023_mmstar` -- Graham & Sahu 2023 (MNRAS 518, 2177; arXiv:2209.14526),
  Table 1, with the 2024 erratum correction (-0.15 dex on log M_*).  M_BH vs
  spheroid and total-galaxy stellar mass for 104 local galaxies.
  Morphological type lives in `gtype` ('E', 'ES/S0', 'S').

`gs2023_binned_husko26` -- Huško et al 2026 (arXiv:2509.05179) Table G1: the
morphology-weighted *intrinsic* binned median of the G&S 2023 sample,
suitable for direct comparison against simulations.  Weights use Moffett+16
type fractions.  Bins are in log M_*,gal.

CSVs sit in obs_data/data/m_sigma_m_star/ and were transcribed from the
arXiv source LaTeX of the respective papers.  All masses are
log10(M / Msun); sigma is log10(sigma / km s^-1).
"""

import os
import csv
import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "m_sigma_m_star")

# Graham & Sahu 2024 erratum: log M_* is too high by 0.15 dex in the 2023
# table.  Apply at load time.
GS2024_ERRATUM_LOG_DEX = -0.15


def _read_csv(path):
    """Read a CSV with '#'-prefixed comment lines. Returns dict-of-columns."""
    cols = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            row = next(csv.reader([line]))
            if cols is None:
                cols = row
            else:
                rows.append(row)
    out = {c: [] for c in cols}
    for r in rows:
        for c, v in zip(cols, r):
            out[c].append(v)
    return out


def _load_msigma():
    raw = _read_csv(os.path.join(_DATA_DIR, "sahu_graham_davis_2019_msigma.csv"))
    return dict(
        name=np.asarray(raw["name"]),
        gtype=np.asarray(raw["gtype"]),
        log_sigma=np.asarray(raw["log_sigma_kms"], dtype=float),
        log_MBH=np.asarray(raw["log_MBH"], dtype=float),
        log_MBH_err=np.asarray(raw["log_MBH_err"], dtype=float),
        reference="Sahu, Graham & Davis 2019, ApJ 887, 10",
    )


def _load_mmstar():
    raw = _read_csv(os.path.join(_DATA_DIR, "graham_sahu_2023_mmstar.csv"))
    return dict(
        name=np.asarray(raw["name"]),
        gtype=np.asarray(raw["gtype"]),
        log_Mstar_sph=np.asarray(raw["log_Mstar_sph"], dtype=float)
                       + GS2024_ERRATUM_LOG_DEX,
        log_Mstar_sph_err=np.asarray(raw["log_Mstar_sph_err"], dtype=float),
        log_Mstar_gal=np.asarray(raw["log_Mstar_gal"], dtype=float)
                       + GS2024_ERRATUM_LOG_DEX,
        log_Mstar_gal_err=np.asarray(raw["log_Mstar_gal_err"], dtype=float),
        log_MBH=np.asarray(raw["log_MBH"], dtype=float),
        log_MBH_err=np.asarray(raw["log_MBH_err"], dtype=float),
        reference="Graham & Sahu 2023, MNRAS 518, 2177 "
                  "(stellar masses corrected by Graham & Sahu 2024)",
    )


sahu19_msigma = _load_msigma()
gs2023_mmstar = _load_mmstar()

# Huško et al. 2026 (arXiv:2509.05179) Table G1: morphology-weighted binned
# medians of the G&S 2023 data using Moffett+16 type fractions, with the
# 2024 erratum stellar-mass correction already applied.  Bin width in log
# M_*,gal is set by hand to keep ~30-40 galaxies per bin.
#
# IMPORTANT: the *_err entries are the uncertainty on the WEIGHTED MEDIAN
# (Husko+26 Sec. 4.2.3: "constrain the median BH masses to within ~0.1 dex
# per stellar mass bin").  They are NOT the population scatter -- the
# intrinsic galaxy-to-galaxy spread in the G&S 2023 sample is much larger
# (~0.5-0.6 dex in log M_BH).  To visualise the data scatter, compute the
# 16-84 percentiles of the raw G&S 2023 data per stellar-mass bin (see
# `bin_gs2023_scatter` below).
gs2023_binned_husko26 = dict(
    log_Mstar_gal=np.array([10.50, 10.85, 11.37, 11.70]),
    log_Mstar_gal_err=np.array([0.03, 0.02, 0.02, 0.03]),
    log_MBH=np.array([7.44, 7.95, 8.85, 9.40]),
    log_MBH_err=np.array([0.14, 0.10, 0.10, 0.13]),
    reference="Huško et al. 2026 Table G1 (binned G&S 2023, Moffett+16 weights)",
    note="*_err entries are uncertainties on the median, not the scatter.",
)


def bin_gs2023_scatter(x_edges, mass_key="log_Mstar_gal"):
    """16-50-84 percentile of log M_BH per bin of G&S 2023 data.
    Use this for visualising the *intrinsic* scatter; the Huško table
    above gives the uncertainty on the median instead."""
    x = gs2023_mmstar[mass_key]
    y = gs2023_mmstar["log_MBH"]
    centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    p16 = np.full(centers.size, np.nan)
    p50 = np.full(centers.size, np.nan)
    p84 = np.full(centers.size, np.nan)
    counts = np.zeros(centers.size, dtype=int)
    for i in range(centers.size):
        sel = (x >= x_edges[i]) & (x < x_edges[i + 1])
        if sel.sum() >= 4:
            p16[i], p50[i], p84[i] = np.percentile(y[sel], [16, 50, 84])
            counts[i] = sel.sum()
    return centers, p16, p50, p84, counts


# -----------------------------------------------------------------------------
# Power-law reference fits with published intrinsic scatter, for guide lines.

# Kormendy & Ho 2013, ARA&A 51, 511 -- ellipticals + classical bulges only.
# Eq. 7 in their paper:  log10(M_BH/Msun) = 8.49 + 4.38 log10(sigma/200 km/s)
# Their reported intrinsic scatter is ~0.29 dex in log M_BH.
KH13_MSIGMA = dict(alpha=8.49, beta=4.38, sigma0=200.0,
                   scatter_dex=0.29,
                   reference="Kormendy & Ho 2013, ARA&A 51, 511 (Eq. 7)")

# K&H 2013 Eq. 10: log10(M_BH) = 8.69 + 1.16 log10(M_bulge/1e11)
# Intrinsic scatter ~0.29 dex.  This is strictly M-M_bulge, not M-M_*,gal,
# so plotting it against galaxy stellar mass overestimates M_BH for
# disc-dominated systems.
KH13_MBULGE = dict(alpha=8.69, beta=1.16, mstar0=1e11,
                   scatter_dex=0.29,
                   reference="Kormendy & Ho 2013, ARA&A 51, 511 (Eq. 10), "
                             "M_BH vs M_bulge (NOT M_*,gal)")

# Reines & Volonteri 2015, ApJ 813, 82 -- their Eq. 5:
# log10(M_BH) = 7.45 + 1.05 (log10 M_* - 11)
# Intrinsic scatter ~0.55 dex.  Sample is broad-line AGN hosts, M_* is
# total galaxy stellar mass.
RV15_MMSTAR = dict(alpha=7.45, beta=1.05, mstar0=1e11,
                   scatter_dex=0.55,
                   reference="Reines & Volonteri 2015, ApJ 813, 82 (Eq. 5)")
