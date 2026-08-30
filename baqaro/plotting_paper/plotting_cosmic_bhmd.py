"""
PAPER FIGURE: cosmic black-hole mass density and accretion-rate density.

One 2-panel figure ``cosmic_bhmd`` (z=0 companion layout to ``local_relations``):

  * LEFT  — cosmic BH mass density rho_BH(z): the total ("This work", black) plus
    the per-log-M_BH-bin breakdown (Purples, inset colourbar), over the obs/model
    compilation (Marconi04 / MerloniHeinz08 / Trinity23 / Shankar13 / Shen20).
  * RIGHT — cosmic BH accretion-rate density rho_dot_BH(z) [M_sun/yr/cMpc^3]:
    the Soltan / luminosity-based estimate (mass-growth (1-eps)*Mdot_acc, black)
    with the direct d(rho_BH)/dt as a cross-check (grey dashed). Model-only (no
    BHAR obs dataset in the repo yet).

Both quantities use a **logM_BH in [6,10] window**, matching the mass range of
the observational compilations; on a subsampled run it also keeps the sums from
being dominated by a few rare, high-weight objects.

BHAR formula (Soltan, per BH): Mdot_BH = (1-eps)/eps * L_bol/c^2, eps from the
Madau+(2014) fit at base 0.1 (the run's rad_efficiency_0, madau+ model),
lambda_Edd = L_bol/(M_BH*10^log_csi). Conversion K = ls*yr/(c^2*ms).

Reads the evolved BH population from the main paper fiducial evolution HDF5 (via
``fiducial_data``). rho_BH and the luminosity-weighted BHAR are population sums →
**weight-correct on the subsample** (per-BH Horvitz-Thompson ``loader.weights``,
None -> ones in full-sim mode).
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt

import qhtools.utils.natconst as nc
from qhtools.utils.cosmology import cosmo

from baqaro.obs_data.cbhmd_obs_data import data_cbhmd
from baqaro.obs_data.cbhad_obs_data import data_cbhad
from baqaro.core_functions.bh_accretion import get_madau_efficiency_epsilon

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    snapshot_index_for_redshift,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "cosmic_bhmd"

RAD_EFF_BASE = 0.1  # matches main_evolution.rad_efficiency_0 (madau+ model)
# L_bol[Lsun] -> Mdot[Msun/yr] for the L/c^2 term (before the (1-eps)/eps factor).
K_LBOL_TO_MDOT = nc.ls * nc.year / (nc.cc**2 * nc.ms)
V = boxsize**3  # cMpc^3

# Mass window (tames subsample shot noise; see module docstring).
M_LO, M_HI = 1e6, 1e10
log_mbins = np.linspace(6, 10, 5)             # 4 bins: [6,7],[7,8],[8,9],[9,10]
mbin_centers = 0.5 * (log_mbins[:-1] + log_mbins[1:])
mbin_norm = plt.Normalize(vmin=6, vmax=10)
mbin_cmap = plt.cm.Purples
colors_bins = mbin_cmap(0.30 + 0.6 * (mbin_centers - 6) / 4)

# Fine z grid (smooth model curves); left panel xlim 0-8.
Z_GRID = np.arange(0.0, 8.001, 0.25)


# ==============================================================================
# COMPUTE rho_BH(z) (total + per mass bin) and Soltan rho_dot_BH(z)
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    redshifts = data["redshifts"]
    snapshots = data["snapshots"]
    loader = data["loader"]
    w_all = loader.weights  # None (full-sim) or per-halo HT weights

    snaps = []
    for zt in Z_GRID:
        i = snapshot_index_for_redshift(
            redshifts, zt, snapshots, label="cosmic_bhmd")
        if i not in snaps:
            snaps.append(i)
    snaps = sorted(snaps)

    z_arr = []
    rho_total, yerr_total = [], []
    rho_bins = [[] for _ in range(len(log_mbins) - 1)]
    rhodot_soltan = []
    for i in snaps:
        M = loader.get_BH_mass(i)
        L = loader.get_Lbol(i)
        cut = (M >= M_LO) & (M <= M_HI)
        if not np.any(cut):
            continue
        w = w_all if w_all is not None else np.ones_like(M)
        logM = np.log10(M[cut])
        wc = w[cut]
        Mc = M[cut]

        # rho_BH per mass bin + total (mass-weighted HT sum).
        mass_sum, _ = np.histogram(logM, bins=log_mbins, weights=wc * Mc)
        counts, _ = np.histogram(logM, bins=log_mbins, weights=wc)
        w2, _ = np.histogram(logM, bins=log_mbins, weights=wc**2)
        for j in range(len(mass_sum)):
            rho_bins[j].append(mass_sum[j] / V)
        tot = float(np.sum(wc * Mc)) / V
        rho_total.append(tot)
        # 1sigma on log10(rho_BH) for the MASS-WEIGHTED Horvitz-Thompson sum
        # S = sum(w_i * M_i). Its HT variance is sum(w_i^2 * M_i^2), so
        #     sigma_log10(S) = 0.434 * sqrt(sum(w^2 M^2)) / sum(w M).
        #
        # This is the variance of a mass-weighted SUM, not the count-based Kish
        # n_eff = (sum w)^2 / sum(w^2): rho_BH is dominated by rare massive
        # high-weight objects, so the two differ most exactly where it matters.
        wM = wc * Mc
        s1 = float(np.sum(wM)); s2 = float(np.sum(wM ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            yerr_total.append(0.434 * np.sqrt(s2) / s1 if s1 > 0 else np.nan)

        # Soltan rho_dot_BH (mass-growth (1-eps)*Mdot_acc), same [6,10] window.
        lum = cut & (L > 0)
        lam = L[lum] / (M[lum] * 10 ** nc.log_csi)
        eps = get_madau_efficiency_epsilon(lam, RAD_EFF_BASE)
        mdot = (1.0 - eps) / eps * K_LBOL_TO_MDOT * L[lum]  # Msun/yr per BH
        wl = w[lum]
        rhodot_soltan.append(float(np.sum(mdot * wl)) / V)

        z_arr.append(redshifts[i])

z_arr = np.asarray(z_arr)
rho_total = np.asarray(rho_total)
yerr_total = np.asarray(yerr_total)
rho_bins = [np.asarray(r) for r in rho_bins]
rhodot_soltan = np.asarray(rhodot_soltan)

# d(rho_BH)/dt cross-check (cosmo.age in Gyr -> yr).
order = np.argsort(z_arr)
t_yr = cosmo.age(z_arr[order]) * 1e9
rhodot_deriv = np.abs(np.gradient(rho_total[order], t_yr))


# ==============================================================================
# OBSERVATIONAL OVERLAY (rho_BH only) — lines only, colour + linestyle vary
# per entry (left and right panels use disjoint qualitative palettes drawn
# from matplotlib's tab10 so the two panels read as "same family, different
# entries").
# ==============================================================================
# Palette: Paul Tol "bright" (CB-safe, distinct from matplotlib tab10).
# Right panel uses Tol "vibrant" — same family, different individual colours.
_obs_rho_style = {
    "Marconi04":      dict(color="#4477AA", ls="-"),
    "MerloniHeinz08": dict(color="#EE6677", ls="--"),
    "Trinity23":      dict(color="#228833", ls="-."),
    "Shankar13":      dict(color="#CCBB44", ls=":"),
    "Shen20":         dict(color="#66CCEE", ls=(0, (4, 1, 1, 1))),
}
obs_to_plot = {"Marconi04": True, "MerloniHeinz08": True, "Trinity23": True,
               "Shankar13": True, "Shen20": True}


def _plot_obs(ax):
    handles = []
    for obs_name, obs in data_cbhmd.items():
        if not obs_to_plot.get(obs_name, False):
            continue
        st = _obs_rho_style.get(obs_name, dict(color="gray", ls="-"))
        h, = ax.plot(obs.z, obs.data, lw=2.4, color=st["color"], ls=st["ls"],
                     alpha=0.9, label=obs.label)
        handles.append(h)
    return handles


# ==============================================================================
# FIGURE (2 panels)
# ==============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

# ----------------------------------------------------------- LEFT: rho_BH(z)
# Deliberately simplified: no per-mass-bin Purples lines or inset
# colourbar — they were visually confusing with no extra information vs the
# total. Keep the total ρ_BH curve and the obs compilation.
mt = rho_total > 0
zt, yt, et = z_arr[mt], np.log10(rho_total[mt]), yerr_total[mt]
h_tot, = axL.plot(zt, yt, lw=2.6, color="black", zorder=12, label="BAQARO (total)")
axL.fill_between(zt, yt - np.nan_to_num(et), yt + np.nan_to_num(et),
                 color="black", alpha=0.15, zorder=3)
obs_handles = _plot_obs(axL)

# Local BH mass density band + label.
axL.axhspan(np.log10(3.2e5), np.log10(8.5e5), color="gray", alpha=0.15, zorder=0)
axL.text(6.0, np.log10(5e5), "Local $\\rho_{\\rm BH}$", color="gray",
         fontsize=17, ha="center", va="center")

axL.set_xlim(0, 8); axL.set_ylim(2.0, 6.6)
axL.set_xlabel(r"Redshift $z$", labelpad=0)
axL.set_ylabel(r"$\log_{10}\,\rho_{\rm BH}$ [$M_\odot$ cMpc$^{-3}$]", labelpad=0)
axL.minorticks_on()
axL.legend(handles=[h_tot] + obs_handles, loc="lower left", markerfirst=True,
           fontsize=12, handlelength=1.6, labelspacing=0.25,
           borderpad=0.4, framealpha=0.85)

# ------------------------------------------------- RIGHT: BHAD vs z
# Obs compilation: lines only (Yang23 stays binned points). Colours are a
# DIFFERENT subset of tab10 from the left panel, with varying linestyles
# mirroring the left-panel convention.
# Palette: Paul Tol "vibrant" — disjoint from the left panel's "bright".
_obs_style = {
    "Shen20":   dict(color="#EE7733", ls="-",  lw=2.4),
    "Aird15":   dict(color="#0077BB", ls="--", lw=2.2),
    "Ananna19": dict(color="#33BBEE", ls="-.", lw=2.2),
    "Ueda14":   dict(color="#CC3311", ls=":",  lw=2.2),
    "Yang23":   dict(color="#EE3377", marker="o", ms=5),
}
_h_bhad_obs = []
for _name, _obs in data_cbhad.items():
    st = _obs_style.get(_name, dict(color="gray", ls=":", lw=1.4))
    # Binned entries with z-error bars (e.g. Yang23) are drawn as points with
    # asymmetric x/y error bars; smooth/line entries stay clean lines.
    if getattr(_obs, "z_err_down", None) is not None:
        h_ = axR.errorbar(_obs.z, _obs.data,
                          xerr=[_obs.z_err_down, _obs.z_err_up],
                          yerr=[_obs.err_down, _obs.err_up],
                          color=st["color"], marker=st.get("marker", "o"),
                          ms=st.get("ms", 5), ls="none", lw=1.3, capsize=2.5,
                          elinewidth=1.0, alpha=0.9, zorder=4, label=_obs.label)
    else:
        h_, = axR.plot(_obs.z, _obs.data, color=st["color"], ls=st["ls"],
                       lw=st["lw"], alpha=0.9, zorder=2, label=_obs.label)
    _h_bhad_obs.append(h_)

# Model: smoothed Soltan curve only (drop the d(rho_BH)/dt cross-check).
# Restricted to z>0.25, the range over which the curve is well sampled.
from scipy.ndimage import gaussian_filter1d
ms = (rhodot_soltan > 0) & (z_arr > 0.25)
_z_s = z_arr[ms]
_y_s = np.log10(rhodot_soltan[ms])
_order_s = np.argsort(_z_s)
_z_s = _z_s[_order_s]; _y_s = _y_s[_order_s]
_y_s_smooth = gaussian_filter1d(_y_s, sigma=1.2, mode="nearest")
h_bhad_model, = axR.plot(_z_s, _y_s_smooth, color="black", lw=2.4, zorder=10,
                         label="BAQARO (total)")

axR.set_xlim(0, 8)
# User-set upper cap; lower edge keeps the auto-scale.
_yrlo, _yrhi = axR.get_ylim()
axR.set_ylim(_yrlo, -3.7)
axR.set_xlabel(r"Redshift $z$", labelpad=0)
axR.set_ylabel(r"$\log_{10}\,\mathrm{BHAD}$ [$M_\odot$ yr$^{-1}$ cMpc$^{-3}$]", labelpad=0)
axR.minorticks_on()
# BAQARO first, then the obs entries in their plotting order.
axR.legend(handles=[h_bhad_model] + _h_bhad_obs, loc="lower left",
           fontsize=12, handlelength=1.6, labelspacing=0.25,
           borderpad=0.4, framealpha=0.85)

# Layout handled by constrained_layout=True at figure creation.
save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)

maybe_show()
