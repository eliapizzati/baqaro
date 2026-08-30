"""
PAPER FIGURE: local (z=0) black-hole scaling relations — single 2-panel figure.

  * LEFT  (``local BHMF``): the model BHMF for all SMBHs at z=0 (solid + error
    band) and, if present on disk, the no-mergers variant (dashed), against the
    local-BHMF compilation (Marconi04 / Shankar09 / Shankar13 / Vika09 /
    LiepoldMa24) as shaded bands.
  * RIGHT (``local M_BH-M_*``): model halo -> M_*,gal via UniverseMachine
    (Behroozi+18) with M_BH from the run, as a weighted 2D density + median /
    16-84% envelope, over Graham&Sahu23 (+ Huško26 binned median + RV15 / KH13
    fits).

Saved as one figure ``local_relations`` — the z=0 companion to the cosmic
``cosmic_bhmd`` 2-panel (same side-by-side layout, shared model accent colour).

Reads the z=0 snapshot of the main paper fiducial evolution HDF5 (via
``fiducial_data``; the current fiducial IS a z=0 run, max_snap=144). Both panels
are population estimates → **weight-correct on the subsample**: the BHMF via
``mass_function_auto`` + ``loader.weights``; the 2D density + envelope via
``hist2d(weights=)`` + the weighted ``binned_median_envelope``.

Merges ``plotting_local_bhmf.py`` + ``plotting_mbh_mstar_z0.py`` (which it
supersedes as paper figures). The shared M_BH-M_* helpers
(``binned_median_envelope``, ``overlay_mmstar_data``, the selection constants,
the UniverseMachine conversion) are imported unchanged from their
``plotting_analysis`` home.
"""

import os
import re

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from baqaro.obs_data.local_bhmf_obs_data import data_local_bhmf

# Pinned fiducial run — imported FIRST so BAQARO_* is force-set before the helper
# modules below pull in load_data_to_plot / sim_config.
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    path_out,
    boxsize,
    name_file,
    load_simulation_metadata,
    mass_function_auto,
    snapshot_index_for_redshift,
)

from baqaro.plotting_common.halo_to_observable import (
    mstar_universemachine_b18,
)
from baqaro.plotting_common.local_relation_helpers import (
    binned_median_envelope,
    overlay_mmstar_data,
    LOG_MBH_MIN, LOG_MHALO_MIN,
    RANDOM_SUBSAMPLE_PER_SNAP, RANDOM_SUBSAMPLE_SEED,
    _eval_fit_mstar,
)
from baqaro.obs_data.m_sigma_m_star_obs_data import (
    RV15_MMSTAR, KH13_MBULGE,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "local_relations"
MODEL_COLOR = "rebeccapurple"   # shared model accent across both panels

# BHMF panel.
min_mass_BH, max_mass_BH = 1e6, 1e11
BHMF_XLIM = (7.0, 11.0)
BHMF_YLIM = (-7.6, -1.5)
# Variant overlay runs: same fiducial bestfit + plumbing, but with the variant
# token inserted into the NOTES_FILE position (the v3-bigeta convention since
# Variants are run with BAQARO_NOTES_FILE=<notes>_{variant} keeping
# the same BAQARO_BESTFIT_NAME). Drawn as dashed/dotted overlays only if the
# files are on disk.
def _variant_path(suffix_tag, legacy_tag=None):
    """Locate a physics-variation run, trying BOTH filename conventions.

    CURRENT (utils/sim_config.physics_toggle_suffix): the toggle token is
    APPENDED at the very end, after `_g{cap}[_feffcorr]` —
        ..._bestfit_{name}_sub_{tag}_g6.21_feffcorr_nomerge.hdf5
    and the tau-zero token is `_tau0`.

    LEGACY (older runs, BAQARO_NOTES_FILE=<notes>_{variant}): the token sat in
    the notes_file slot, just before `_bestfit_`, and tau-zero was spelled
    `tauzero`.

    Try the current form first, then the legacy one.
    """
    nf_new = f"{name_file}_{suffix_tag}"
    p_new = os.path.join(path_out, "evolution", f"bh_evolution_{nf_new}.hdf5")
    if os.path.exists(p_new):
        return nf_new, p_new, True
    tag = legacy_tag or suffix_tag
    nf_old = re.sub(r"_bestfit_", f"_{tag}_bestfit_", name_file, count=1)
    p_old = os.path.join(path_out, "evolution", f"bh_evolution_{nf_old}.hdf5")
    return nf_old, p_old, (nf_old != name_file) and os.path.exists(p_old)

name_file_no_mergers, path_file_no_mergers, _has_no_mergers_init = _variant_path("nomerge")
name_file_tauc0,      path_file_tauc0,      _has_tauc0_init      = _variant_path("tau0", legacy_tag="tauzero")

# M_BH-M_* panel.
N_HIST_BINS = 80
MM_XLIM = (8.5, 12.5)
MM_YLIM = (5.0, 10.5)


# ==============================================================================
# LOAD z=0 + COMPUTE BOTH PANELS' MODEL QUANTITIES
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    redshifts = data["redshifts"]
    snapshots = data["snapshots"]
    loader = data["loader"]

    snap_idx = snapshot_index_for_redshift(
        redshifts, 0.0, snapshots, label="local_relations")
    print(f"z=0 -> snapshot {snap_idx} (z={redshifts[snap_idx]:.3f})", flush=True)

    M_BH = np.asarray(loader.get_BH_mass(snap_idx))
    M_h = np.asarray(loader.get_Halo_mass(snap_idx))
    if loader.is_subsample and loader.weights is not None:
        w_all = np.asarray(loader.weights, dtype=float)
    else:
        w_all = np.ones_like(M_BH, dtype=float)

    # --- BHMF (all active SMBHs) ---
    active = M_BH > 0
    mbins_all, mf_all, error_all = mass_function_auto(
        M_BH[active], boxsize**3, weights=w_all[active],
        lowest_mass=min_mass_BH, highest_mass=max_mass_BH, n_bins=51, minimum_in_bin=1,
    )

    # --- M_BH-M_* selection + UniverseMachine M_* ---
    # The uniform random cap (RANDOM_SUBSAMPLE_PER_SNAP) is representative ONLY
    # for the mass-stratified subsample: it is roughly flat in halo mass, so a
    # uniform draw spans the whole relation and the HT weights fill the
    # high-mass pixels. For a FULL-CATALOGUE read (weights == 1, steeply
    # mass-declining) the same uniform draw is swamped by the billions of
    # low-mass halos and STARVES the high-M_BH wing (few points land there, and
    # weight 1 gives no upweighting to compensate) -> the density collapses to
    # the low-mass corner and the relation looks under-populated. In that case
    # histogram ALL masked halos: the weighted 2D density is then exact and a
    # full hist2d over the masked set is cheap (it is pixels, not scatter).
    rng = np.random.default_rng(RANDOM_SUBSAMPLE_SEED)
    weighted_run = loader.weights is not None and not np.all(w_all == 1.0)
    mm_mask = (M_BH > 10.0 ** LOG_MBH_MIN) & (M_h > 10.0 ** LOG_MHALO_MIN)
    if (weighted_run and RANDOM_SUBSAMPLE_PER_SNAP is not None
            and mm_mask.sum() > RANDOM_SUBSAMPLE_PER_SNAP):
        idxs = np.flatnonzero(mm_mask)
        chosen = rng.choice(idxs, size=RANDOM_SUBSAMPLE_PER_SNAP, replace=False)
        mm_mask = np.zeros_like(mm_mask)
        mm_mask[chosen] = True
    print(f"  {mm_mask.sum()} halos for M_BH-M_* density "
          f"({'uniform-capped' if weighted_run else 'full, uncapped'})", flush=True)
    log_Mh = np.log10(M_h[mm_mask])
    log_MBH = np.log10(M_BH[mm_mask])
    ww = w_all[mm_mask]
    log_Mstar = mstar_universemachine_b18(log_Mh, float(redshifts[snap_idx]))


# ==============================================================================
# LOAD VARIANT BHMFs (optional overlays)
# ==============================================================================
def _load_variant_bhmf(label, path, exists_flag):
    if not exists_flag:
        print(f"{label} file not found (skipping): {path}", flush=True)
        return False, None, None, None
    print(f"Loading {label}: {path}", flush=True)
    with h5py.File(path, "r", rdcc_nbytes=128 * 1024 * 1024, rdcc_nslots=10007) as fv:
        dv = load_simulation_metadata(fv)
        loader_v = dv["loader"]
        M_BH_v = loader_v.get_BH_mass(snap_idx)
        active_v = M_BH_v > 0
        w_v = loader_v.weights[active_v] if loader_v.weights is not None else None
        mb_v, mf_v, err_v = mass_function_auto(
            M_BH_v[active_v], boxsize**3, weights=w_v,
            lowest_mass=min_mass_BH, highest_mass=max_mass_BH, n_bins=51, minimum_in_bin=1,
        )
    return True, mb_v, mf_v, err_v

has_no_mergers, mbins_nm, mf_nm, error_nm = _load_variant_bhmf(
    "no-mergers run", path_file_no_mergers, _has_no_mergers_init,
)
# NB the tau_coh=0 (Branch C) variant is deliberately NOT loaded here. It was
# read and then never plotted (the overlay lives exclusively in
# plotting_bhmf_erdf_highz), which cost a pointless ~425 GB file read per
# render. Since removed. `_variant_path("tau0", ...)` above still resolves
# the path if a future panel wants it.


# ==============================================================================
# FIGURE (2 panels)
# ==============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

# ------------------------------------------------------------------ LEFT: BHMF
x_all = np.log10(mbins_all)
axL.plot(x_all, np.log10(mf_all), lw=2.5, color=MODEL_COLOR, ls="-", label="BAQARO (with mergers)")
axL.fill_between(x_all, np.log10(mf_all - error_all), np.log10(mf_all + error_all),
                 alpha=0.2, color=MODEL_COLOR)
if has_no_mergers:
    x_nm = np.log10(mbins_nm)
    axL.plot(x_nm, np.log10(mf_nm), lw=2.5, color=MODEL_COLOR, ls="--", label="BAQARO (no mergers)")
    axL.fill_between(x_nm, np.log10(mf_nm - error_nm), np.log10(mf_nm + error_nm),
                     alpha=0.2, color=MODEL_COLOR)
# τ_coh=0 overlay intentionally NOT shown on the local-BHMF panel (the comparison
# is exclusively in plotting_bhmf_highz now — keeps this panel uncluttered).

obs_colors = {"Marconi04": "saddlebrown", "Shankar09": "darkseagreen",
              "Shankar13": "cadetblue", "Vika09": "olive", "LiepoldMa24": "darkslategray"}
# Per-source upper M_BH cap (don't draw obs estimates above their reliable end).
# LiepoldMa24 is fitted out to log M_BH ≈ 10.25; truncate to avoid extrapolation.
OBS_LOGMBH_HI = {"LiepoldMa24": 10.25}
for obs_name, obs in data_local_bhmf.items():
    c = obs_colors.get(obs_name, "black")
    x, y, ed, eu = obs.reliable_arrays()
    cap = OBS_LOGMBH_HI.get(obs_name)
    if cap is not None:
        m = x <= cap
        x, y, ed, eu = x[m], y[m], ed[m], eu[m]
    if np.any(obs.err > 0):
        axL.fill_between(x, y - ed, y + eu, alpha=0.25, color=c, label=obs.label)
    else:
        axL.plot(x, y, lw=1.5, color=c, ls="-", label=obs.label)

axL.set_xlim(*BHMF_XLIM)
axL.set_ylim(*BHMF_YLIM)
axL.set_xlabel(r"$\log_{10}\, M_{\rm BH}$ [$M_\odot$]", labelpad=0)
axL.set_ylabel(r"$\mathrm{d}n / \mathrm{d}\log_{10} M_{\rm BH}$ [dex$^{-1}$ cMpc$^{-3}$]", labelpad=0)
axL.minorticks_on()
axL.legend(loc="lower left", markerfirst=True,
           fontsize=12, handlelength=1.6, labelspacing=0.25,
           borderpad=0.4, framealpha=0.85)

# ------------------------------------------------------- RIGHT: M_BH - M_*,gal
mstar_edges = np.linspace(MM_XLIM[0], MM_XLIM[1], N_HIST_BINS + 1)
mbh_edges = np.linspace(MM_YLIM[0], MM_YLIM[1], N_HIST_BINS + 1)

# Weighted-count 2D density — transparent so the model band and the
# observational relations remain visible through it.
h = axR.hist2d(log_Mstar, log_MBH, bins=[mstar_edges, mbh_edges],
               weights=ww, cmin=1, norm=LogNorm(), cmap="Purples", alpha=0.45)
# Inset colorbar at the BOTTOM-RIGHT corner — label + ticks on TOP, larger.
cax = axR.inset_axes([0.56, 0.06, 0.38, 0.035])
cb = fig.colorbar(h[3], cax=cax, orientation="horizontal")
cb.ax.xaxis.set_ticks_position("top")
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=9)
cb.set_label("weighted count", fontsize=10, labelpad=2)

from matplotlib.legend_handler import HandlerTuple

# Model median + 16-84% band — combined into ONE legend entry (Line+Patch tuple).
x_c, p16, p50, p84 = binned_median_envelope(log_Mstar, log_MBH, ww, mstar_edges)
valid = np.isfinite(p50)
_h_model_line, = axR.plot(x_c[valid], p50[valid], color=MODEL_COLOR, lw=2.4, zorder=6)
# Stronger band alpha so the ±1σ region reads clearly through the count map.
_h_model_band = axR.fill_between(x_c[valid], p16[valid], p84[valid],
                                 color=MODEL_COLOR, alpha=0.30, zorder=2)

# Obs data + binned median markers (incl. Huško+26 / G&S23 median), but NOT
# the R&V/K&H fits — we add those explicitly below so we can capture their
# (line, band) handles cleanly.
overlay_mmstar_data(axR, x_range=MM_XLIM, show_data_scatter=False, show_fits=False)

# K&H 2013 (gray, kept first) and R&V 2015 (recoloured to a distinct warm hue
# so it doesn't read as the same family as the purple model). Stronger band
# alpha so the ±1σ region is visible.
_xs = np.linspace(MM_XLIM[0], MM_XLIM[1], 100)
_rv_y = _eval_fit_mstar(_xs, RV15_MMSTAR);  _rv_s = RV15_MMSTAR["scatter_dex"]
_kh_y = _eval_fit_mstar(_xs, KH13_MBULGE);  _kh_s = KH13_MBULGE["scatter_dex"]
_KH_COLOR = "0.25"      # dark grey
_RV_COLOR = "0.6"       # lighter grey
_h_kh_line, = axR.plot(_xs, _kh_y, color=_KH_COLOR, ls="--", lw=1.4, alpha=0.9, zorder=3)
_h_kh_band  = axR.fill_between(_xs, _kh_y - _kh_s, _kh_y + _kh_s,
                               color=_KH_COLOR, alpha=0.22, zorder=1)
_h_rv_line, = axR.plot(_xs, _rv_y, color=_RV_COLOR, ls="--", lw=1.4, alpha=0.9, zorder=3)
_h_rv_band  = axR.fill_between(_xs, _rv_y - _rv_s, _rv_y + _rv_s,
                               color=_RV_COLOR, alpha=0.22, zorder=1)

axR.set_xlim(*MM_XLIM)
axR.set_ylim(*MM_YLIM)
axR.set_xlabel(r"$\log_{10}(M_{*,\rm gal} / M_\odot)$", labelpad=0)
axR.set_ylabel(r"$\log_{10}(M_{\rm BH} / M_\odot)$", labelpad=0)
axR.minorticks_on()

# ---- Slim legend: pick out the auto handles (G&S23 morph types + the
# G&S+Huško binned median). With show_data_scatter=False and show_fits=False,
# overlay_mmstar_data adds ONLY marker entries — no bands to filter out — so
# we just clean up the wordy labels. (Note: the Huško label has a multi-line
# "...uncertainty" suffix; keep the skip list narrow so it is not matched.)
_handles, _labels = axR.get_legend_handles_labels()
_seen = set()
_h_other, _l_other = [], []
for h_, l_ in zip(_handles, _labels):
    lab = l_.split("\n")[0].strip()
    if "binned median" in lab or "Huško" in lab or lab.startswith("Hu"):
        lab = "G&S23 median"
    if lab in _seen:
        continue
    _seen.add(lab)
    _h_other.append(h_); _l_other.append(lab)

# Legend order: Model → K&H → R&V → all G&S/Huško auto-handles.
_h_keep = [(_h_model_line, _h_model_band),
           (_h_kh_line, _h_kh_band),
           (_h_rv_line, _h_rv_band)] + _h_other
_l_keep = [r"BAQARO ($M_{\star,\rm gal}$ from UM; $\pm 1\sigma$)",
           r"K&H13 bulge ($\pm 1\sigma$)",
           r"R&V15 ($\pm 1\sigma$)"] + _l_other

axR.legend(_h_keep, _l_keep, loc="upper left", ncol=1,
           fontsize=11, handlelength=1.8, labelspacing=0.2,
           borderpad=0.3, framealpha=0.85,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

# Layout handled by constrained_layout=True at figure creation.
save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)

maybe_show()
