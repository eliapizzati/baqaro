"""
TEST/COMPARISON figure: DIRECT vs QHMF (halo-model) quasar clustering.

Overlays, per panel (z=2.5, 4.0 auto w_p/r_p; z=6.1 cross chi_V), on the SAME
observations:
  * SOLID  — analytic QHMF -> halo-model "triangle" prediction (the method in
             plotting_clustering.py): the run's quasar host-halo mass function
             fed through the precomputed triangle, then projected.
  * DASHED — DIRECTLY measured real-space xi(r) from the full-catalogue 3D
             positions (clustering_direct/measure_clustering_direct.py), projected
             with the SAME qhtools integrators.

Both use identical quasar selections, projection grids, and pimax, so the panel
isolates "measured xi vs halo-model xi". Opens the run (via load_data_to_plot,
env-driven) for the QHMF and reads the cached direct .npz for the measurement.

Run (point at the multinode full-cat run + its direct .npz). The env must match
the run the CACHES were measured on — `cache_provenance.verify_cache` prints the
source and refuses a subsample source. For the adopted ck22final fiducial the
caches on disk are `qcc_ck22final_v1` / `multinode_root144_v4` / full-catalogue,
and the growth-cap and f_eff tokens are already the zero-env defaults:

  env BAQARO_USE_SUBSAMPLE=0 BAQARO_SUBSET_TAG=multinode_root144_v4 \\
      BAQARO_BESTFIT_NAME=qcc_ck22final_v1 BAQARO_NOTES_FILE= \\
      BAQARO_SAVE_FIGS=1 BAQARO_HEADLESS=1 \\
      python -m baqaro.clustering_direct.plot_compare_direct_vs_qhmf

"""

import os
import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from qhtools.utils import my_utils
from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle, get_corr_from_triangle_cross
from qhtools.clustering.projected_correlation_functions import get_projected_wp, get_volume_averaged_xi

from baqaro.obs_data.corr_obs_data import (
    data_ef_ext, data_shen_highz, data_aspire, data_eiger,
)
from baqaro.obs_data.corr_inputs_loader import (
    get_input_quantities_auto, get_input_quantities_cross,
)
# env-driven run handles (point these at the multinode full-cat run)
from baqaro.plotting_common.load_data_to_plot import (
    path_file, boxsize, load_simulation_metadata, mass_function_auto,
    snapshot_index_for_redshift,
)
from baqaro.utils.my_dir import get_output_path
from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show

name_fig = "correlation_plot_compare"
len_rbins, len_mbins = 101, 51
min_mass, max_mass = 1e10, 1e15


def fill_empty_bins(r, xi):
    """Interpolate zero-pair (xi == -1) bins in log(1+xi) vs log(r) from the
    bins that have pairs. Matches plotting_clustering_direct.fill_empty_bins."""
    xi = np.asarray(xi, dtype=float)
    good = xi > -1.0
    if good.all() or good.sum() < 2:
        return xi
    lr = np.log10(r)
    yi = np.interp(lr, lr[good], np.log10(1.0 + xi[good]))
    return np.power(10.0, yi) - 1.0

source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
clust_outdir = os.environ.get("BAQARO_CLUST_OUTDIR") or os.path.join(
    get_output_path(source_dir), "clustering_direct")

# per panel: key, obs object, kind, logM range, L_bol threshold [Lsun]
PANELS = [
    ("2.5", data_ef_ext,     "auto",  11.5, 14.5, data_ef_ext.log_L_threshold),
    ("4.0", data_shen_highz, "auto",  11.5, 14.5, data_shen_highz.log_L_threshold),
    ("6.1", data_aspire,     "cross", 10.5, 14.0, my_utils.to_solar(46.5)),
]
colors = z_scalar_mapper.to_rgba(np.asarray([float(p[0]) for p in PANELS]))

fig, axs = plt.subplots(1, 3, figsize=(10, 2.9), sharex=True, sharey=True)
axs = axs.flatten()

with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as f:
    meta = load_simulation_metadata(f)
    redshifts = meta["redshifts"]
    snapshots = meta["snapshots"]
    loader = meta["loader"]

    for i, ((key, data, kind, logM_min, logM_max, log_L_thr), ax, color) in enumerate(
            zip(PANELS, axs, colors)):
        redshift = float(key)
        snap_idx = snapshot_index_for_redshift(
            redshifts, redshift, snapshots, label="compare-direct-vs-qhmf")
        print(f"z={key}: run col={snap_idx} (z={redshifts[snap_idx]:.3f})", flush=True)

        # ---------- analytic QHMF -> halo-model triangle (SOLID) ----------
        if kind == "auto":
            log_m_axis, rbins, rp_arr, mf_fit, triangle_fit = get_input_quantities_auto(
                redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
                log_M_min=logM_min, log_M_max=logM_max)
        else:
            log_m_axis, rbins, rvol_bins, mf_fit, triangle_fit = get_input_quantities_cross(
                redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
                log_M_min=logM_min, log_M_max=logM_max)

        Lbol = loader.get_Lbol(snap_idx)
        Mh = loader.get_Halo_mass(snap_idx)
        qmask = Lbol > 10 ** log_L_thr
        w = loader.weights[qmask] if loader.weights is not None else None
        mbins_q, mf_q, _ = mass_function_auto(
            Mh[qmask], boxsize ** 3, weights=w,
            lowest_mass=min_mass, highest_mass=max_mass, n_bins=51, minimum_in_bin=1)
        qhmf_here = np.interp(log_m_axis, np.log10(mbins_q), mf_q, left=0., right=0.)

        if kind == "auto":
            log_rbins, xi_a = get_corr_from_triangle(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis, qhmf=qhmf_here)
            wp_a = get_projected_wp(rp_arr, xi_a, 10 ** log_rbins, pimax=data.pimax)
            ax.plot(rp_arr, wp_a / rp_arr, color=color, lw=2.2, ls="-", zorder=9)
        else:
            qhmf_gal = np.copy(mf_fit); qhmf_gal /= 5.; qhmf_gal[log_m_axis < 10.75] = 0.
            log_rbins, xi_a = get_corr_from_triangle_cross(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis, qhmf1=qhmf_here, qhmf2=qhmf_gal)
            xivol_a = get_volume_averaged_xi(rvol_bins, xi_a, 10 ** log_rbins, pimax=data.pimax)
            rlo, rhi = rvol_bins[:-1], rvol_bins[1:]
            rvol_c = np.power(10, (np.log10(rlo) + np.log10(rhi)) / 2.)
            ax.plot(rvol_c, xivol_a, color=color, lw=2.2, ls="-", zorder=9)

        # ---------- directly measured xi(r) (DASHED) ----------
        npz = os.path.join(clust_outdir, f"clustering_direct_z{key}.npz")
        if os.path.exists(npz):
            d = np.load(npz)
            r_c, pimax = d["r_centers"], float(d["pimax"])
            xi_d = fill_empty_bins(r_c, d["xi"])   # interpolate zero-pair bins
            if kind == "auto":
                wp_d = get_projected_wp(rp_arr, xi_d, r_c, pimax=pimax)
                ax.plot(rp_arr, wp_d / rp_arr, color=color, lw=2.2, ls="--", zorder=10)
            else:
                xivol_d = get_volume_averaged_xi(rvol_bins, xi_d, r_c, pimax=pimax)
                ax.plot(rvol_c, xivol_d, color=color, lw=2.2, ls="--", zorder=10)
        else:
            print(f"  [no direct npz for z={key}] {npz}")

        # ---------- observations ----------
        ax.errorbar(data.x, data.data, yerr=data.err, fmt="o", markersize=4, capsize=2,
                    elinewidth=0.8, alpha=0.5, color=color, mec=color)
        if key == "6.1":
            ax.errorbar(data_eiger.x, data_eiger.data, yerr=data_eiger.err, fmt="s",
                        markersize=4, capsize=2, elinewidth=0.8, alpha=0.5, color=color,
                        mec=color, mfc="none")

        if i == 0:
            ax.set_ylabel(r"$w_p(r_p)/r_p,\, \chi_{\mathrm{V}}$", labelpad=-1)
        else:
            ax.tick_params(labelleft=False)
        # h-FREE comoving Mpc (separations are /cosmo.h at load) — the
        # "h^-1 Mpc" label was wrong.
        ax.set_xlabel(r"$r_p$ [cMpc]", labelpad=-3)
        pos = ("right", 0.95) if key == "6.1" else ("left", 0.05)
        ax.text(pos[1], 0.95, f"$z = {key}$, {kind}", transform=ax.transAxes,
                fontweight="bold", color=color, ha=pos[0], va="top", fontsize=9)

# method legend (only once)
axs[0].legend(handles=[Line2D([], [], color="k", ls="-", label="QHMF (halo model)"),
                       Line2D([], [], color="k", ls="--", label="direct (positions)")],
              fontsize=8, loc="lower left", frameon=False)
for a in axs:
    a.set_xlim(0.09, 95.); a.set_ylim(1.1e-1, 1e3)
    a.set_xscale("log"); a.set_yscale("log"); a.minorticks_on()
fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.19, hspace=0.05, wspace=0.05)
save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
