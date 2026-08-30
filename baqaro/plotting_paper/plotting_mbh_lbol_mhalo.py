"""
PAPER FIGURE: black-hole / quasar scaling with host-halo mass — single 2-panel.

  * LEFT  (M_BH - M_halo): the weighted median log M_BH in bins of log M_halo,
    with the 16-84% band, one curve per redshift (colour = z). Dashed reference
    M_BH = M_halo / 1e5.
  * RIGHT (M_halo - L_bol): the weighted median log M_halo in bins of log L_bol
    (so the relation reads "what halo hosts a quasar of this luminosity"), with
    the 16-84% band, one curve per redshift.

**Each panel puts its conditioning variable on the x-axis** (left bins in
M_halo, right bins in L_bol), so every curve is honestly ⟨Y⟩ at fixed X — they
do NOT share an axis on purpose (⟨M_BH | M_halo⟩ and ⟨M_halo | L_bol⟩ are
different conditionings; at fixed M_halo the duty cycle makes ⟨L_bol⟩ ill-posed,
so the quasar question must condition on L_bol). Colour encodes redshift via the
shared ``z_scalar_mapper`` (the fixed vmax=7 population-statistics scale).

Reads the evolved BH population from the main paper fiducial evolution HDF5 (via
``fiducial_data``). Both are **population** median/percentile relations →
**weight-correct on the subsample**: ``weighted_percentile`` with the per-halo
Horvitz-Thompson ``loader.weights`` (None -> unweighted in full-sim mode).

Merges the working M_BH-M_halo figure (`_allz` overlay) +
the working L_bol-M_halo figure (`_allz` overlay) into one paper
figure ``mbh_lbol_mhalo``.
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt

# Pinned fiducial run — imported FIRST so BAQARO_* is force-set before
# load_data_to_plot / sim_config resolve below.
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    snapshot_index_for_redshift,
)
# load_data_to_plot is already imported (with the pinned env) by fiducial_data;
# this just pulls the shared weighted-percentile helper from the cached module.
from baqaro.plotting_common.load_data_to_plot import weighted_percentile
from baqaro.utils.sim_config import (
    subsample_log_M_lo as _sim_log_M_lo,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "mbh_lbol_mhalo"
LOG10_LSUN = np.log10(3.826e33)  # L_sun -> erg/s

redshift_keys = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
colors = z_scalar_mapper.to_rgba(redshift_keys)

# Sim-aware halo-mass axis (floor = resolution + 0.5 dex margin).
LOG_MHALO_RES_MARGIN = 0.5
_logm_lo = _sim_log_M_lo + LOG_MHALO_RES_MARGIN
_logm_hi = 15.5
_n_bins = int(round((_logm_hi - _logm_lo) / 0.25))
logm_bins = np.linspace(_logm_lo, _logm_hi, _n_bins + 1)
mh_centers = 0.5 * (logm_bins[1:] + logm_bins[:-1])

# L_bol bins (right panel), log10 erg/s.
log_Lbol_bins = np.linspace(44.0, 48.0, 17)
lb_centers = 0.5 * (log_Lbol_bins[1:] + log_Lbol_bins[:-1])


# ==============================================================================
# FIGURE
# ==============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

with h5py.File(path_file, "r", rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    redshifts = data["redshifts"]
    snapshots = data["snapshots"]
    loader = data["loader"]

    for redshift, color in zip(redshift_keys, colors):
        snap = snapshot_index_for_redshift(
            redshifts, redshift, snapshots, label="mbh_lbol_mhalo")
        print(f"Working on z={redshift} (snap {snap})", flush=True)

        M_BH = loader.get_BH_mass(snap)
        M_h = loader.get_Halo_mass(snap)
        L = loader.get_Lbol(snap)
        w_all = loader.weights

        # ---- LEFT: median log M_BH in bins of log M_halo ----
        base_L = (M_BH > 0) & (M_h > 10 ** _logm_lo)
        med = np.full(len(mh_centers), np.nan)
        lo = np.full(len(mh_centers), np.nan)
        hi = np.full(len(mh_centers), np.nan)
        for j in range(len(mh_centers)):
            m = base_L & (M_h > 10 ** logm_bins[j]) & (M_h < 10 ** logm_bins[j + 1])
            if not np.any(m):
                continue
            vals = np.log10(M_BH[m])
            ww = w_all[m] if w_all is not None else None
            med[j] = weighted_percentile(vals, ww, 0.50)
            lo[j] = weighted_percentile(vals, ww, 0.16)
            hi[j] = weighted_percentile(vals, ww, 0.84)
        v = ~np.isnan(med)
        axL.plot(mh_centers[v], med[v], lw=2.4, color=color, zorder=10)
        axL.fill_between(mh_centers[v], lo[v], hi[v], alpha=0.15, color=color, zorder=5)

        # ---- RIGHT: median log M_halo in bins of log L_bol ----
        with np.errstate(divide="ignore"):
            logL = np.where(L > 0, np.log10(L) + LOG10_LSUN, np.nan)
        base_R = (logL >= 44.0) & (M_h > 0)
        medh = np.full(len(lb_centers), np.nan)
        loh = np.full(len(lb_centers), np.nan)
        hih = np.full(len(lb_centers), np.nan)
        for j in range(len(lb_centers)):
            m = base_R & (logL >= log_Lbol_bins[j]) & (logL < log_Lbol_bins[j + 1])
            if not np.any(m):
                continue
            vals = np.log10(M_h[m])
            ww = w_all[m] if w_all is not None else None
            medh[j] = weighted_percentile(vals, ww, 0.50)
            loh[j] = weighted_percentile(vals, ww, 0.16)
            hih[j] = weighted_percentile(vals, ww, 0.84)
        v = ~np.isnan(medh)
        axR.plot(lb_centers[v], medh[v], lw=2.4, color=color, zorder=10)
        axR.fill_between(lb_centers[v], loh[v], hih[v], alpha=0.15, color=color, zorder=5)

# ---- LEFT formatting ----
_xx = np.linspace(_logm_lo - 0.2, _logm_hi + 0.2, 50)
axL.plot(_xx, _xx - 5.0, lw=1.5, color="black", ls="--", label=r"$M_{\rm BH} = M_{\rm halo}/10^5$")
axL.set_xlim(_logm_lo, _logm_hi + 0.2)
axL.set_ylim(5.5, 10.5)
axL.set_xlabel(r"$\log_{10}\, M_{\rm halo}$ [$M_\odot$]", labelpad=0)
axL.set_ylabel(r"$\log_{10}\, M_{\rm BH}$ [$M_\odot$]", labelpad=0)
axL.minorticks_on()
axL.legend(loc="lower right",
           fontsize=12, handlelength=1.6, labelspacing=0.25,
           borderpad=0.4, framealpha=0.85)

# ---- RIGHT formatting ----
axR.set_xlim(44.0, 48.0)
axR.set_ylim(11.0, 13.5)
axR.set_xlabel(r"$\log_{10}\, L_{\rm bol}$ [erg s$^{-1}$]", labelpad=0)
axR.set_ylabel(r"$\log_{10}\, M_{\rm halo}$ [$M_\odot$]", labelpad=0)
axR.minorticks_on()

cbar = fig.colorbar(z_scalar_mapper, ax=[axL, axR], pad=0.012, fraction=0.045)
cbar.set_label("Redshift")
cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7])
cbar.ax.minorticks_on()

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
