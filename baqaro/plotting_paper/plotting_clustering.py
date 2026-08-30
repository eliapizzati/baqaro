"""
PAPER FIGURE: quasar clustering (projected/volume-averaged correlation).

Three panels at z = 2.5, 4.0 (auto-correlation, projected w_p(r_p)/r_p) and
z = 6.1 (quasar–galaxy cross, volume-averaged xi). Observations are overlaid
(EF/extended, Shen high-z, ASPIRE + EIGER at z~6). Each panel carries TWO
model curves for "this work":

  * SOLID  — the analytic halo-model prediction: the quasar host-halo mass
             function (QHMF) of the fiducial run pushed through the halo-model
             "triangle" and projected.
  * DASHED — the DIRECTLY-MEASURED real-space xi(r) from the full-catalogue 3D
             positions (Corrfunc, analytic-RR natural estimator; see
             ``clustering_direct/measure_clustering_direct.py``), projected with
             the SAME qhtools integrators as the analytic curve — apples to
             apples, measured-vs-halo-model. Drawn only where the cached
             ``clustering_direct_z{key}.npz`` exists (else the panel just shows
             the analytic curve); the cache is read, NOT the multi-hundred-GB run.
             (Merged from the former standalone ``plotting_clustering_direct.py``.)

Reads the evolved BH population from the main paper fiducial evolution HDF5
(via ``fiducial_data``). The QHMF is a population mass function, so it is
**weight-correct on the subsampled fiducial**: ``mass_function_auto`` applies
the per-host Horvitz–Thompson ``loader.weights`` (a no-op in full-sim mode).

The analytic content is unchanged from the working version of this
figure; only the data handles come from the pinned ``fiducial_data`` and the
save routes to the git-tracked ``figures_paper/`` (PDF by default).

Env:
  BAQARO_CLUST_OUTDIR  dir holding the direct-measurement clustering_direct_z*.npz
                      (default {output}/clustering_direct)
  BAQARO_CLUST_Z4_ERR_INFLATE
                      OPTIONAL factor (default 1.0 = fiducial, no change) that
                      scales ONLY the z=4 observed error bars (data_shen_highz) at
                      plot time — a what-if on how much the z=4 clustering point
                      constrains the fit. Leaves every MODEL curve (QHMF solid /
                      direct dashed) and the on-disk data untouched; when != 1.0 the
                      output figure name gets a ``_z4err{factor}`` token so it can
                      never overwrite the fiducial ``correlation_plot``.
"""

import os

import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from qhtools.utils import my_utils
from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle, get_corr_from_triangle_cross
from qhtools.clustering.projected_correlation_functions import get_projected_wp, get_volume_averaged_xi

from baqaro.obs_data.corr_obs_data import GAL_LOGM_CUT_FIGURE
from baqaro.obs_data import corr_obs_data as _corr_mod
from baqaro.obs_data.corr_obs_data import (
    data_ef_ext_restricted, data_ef_ext, data_aspire, data_eiger,
)
from baqaro.utils.my_dir import get_output_path
from baqaro.clustering_direct.cache_provenance import verify_cache, source_bestfit

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    mass_function_auto,
    snapshot_index_for_redshift,
    RESOLVED as _PAPER_RESOLVED,
)

# The fiducial's model identity — the direct (dashed) overlay must match it, else
# it is skipped (the direct measurement only exists for a run with a multinode
# full-cat version; a repoint to a model without one shows the halo-model curve only).
_FID_BESTFIT = _PAPER_RESOLVED.get("BAQARO_BESTFIT_NAME") or None

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# Directory holding the cached direct-measurement .npz (one per panel).
_direct_outdir = os.environ.get("BAQARO_CLUST_OUTDIR") or os.path.join(
    get_output_path(os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")),
    "clustering_direct")


def _fill_empty_bins(r, xi):
    """Empty (zero-pair) bins from the natural estimator read xi == -1 (full
    anti-correlation), which drags the projected w_p/chi_V low. Treat them as
    no-data: linearly interpolate log(1+xi) vs log(r) from the bins that DO have
    pairs (so 1+xi stays >0), holding the end values at the edges. Bins with
    pairs are untouched. (Ported verbatim from plotting_clustering_direct.py.)"""
    xi = np.asarray(xi, dtype=float)
    good = xi > -1.0
    if good.all() or good.sum() < 2:
        return xi
    lr = np.log10(r)
    yi = np.interp(lr, lr[good], np.log10(1.0 + xi[good]))
    return np.power(10.0, yi) - 1.0


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "correlation_plot"

# z=4 clustering observation. STANDARD (default) = ALL-FIELDS Shen+07 z>=3.5 with
# x2-inflated errors (`data_shen_highz_allfields_err2`), matching the inference
# default BAQARO_CORR_Z4 — one dataset object, so figure and fit cannot drift apart.
#   * BAQARO_CLUST_Z4_DATA=<key>  : swap the z=4 dataset (e.g. `data_shen_highz` for
#                                  the good-fields x1 measurement).
#   * BAQARO_CLUST_Z4_ERR_INFLATE : EXTRA error multiplier stacked on top (def 1.0).
# The MODEL curves are unchanged by either (same i<20.2 selection threshold + pimax
# across the Shen datasets). Any non-default choice renames the output so the
# standard figure is never overwritten.
Z4_DATA_KEY = (os.environ.get("BAQARO_CLUST_Z4_DATA", "data_shen_highz_allfields_err2").strip()
               or "data_shen_highz_allfields_err2")
Z4_DATA = getattr(_corr_mod, Z4_DATA_KEY)
Z4_ERR_INFLATE = float(os.environ.get("BAQARO_CLUST_Z4_ERR_INFLATE", "1.0"))
_tok = []
if Z4_DATA_KEY != "data_shen_highz_allfields_err2":
    _tok.append(Z4_DATA_KEY)
if Z4_ERR_INFLATE != 1.0:
    _tok.append(f"z4err{Z4_ERR_INFLATE:g}")
if _tok:
    name_fig = "correlation_plot_" + "_".join(_tok)

keys = ["2.5", "4.0", "6.1"]
redshift_keys = np.asarray([2.5, 4.0, 6.1])
colors = z_scalar_mapper.to_rgba(redshift_keys)

min_mass, max_mass = 1e10, 1e15
len_rbins, len_mbins = 101, 51


# ==============================================================================
# MAIN
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    meta = load_simulation_metadata(file)
    redshifts = meta["redshifts"]
    snapshots = meta["snapshots"]
    loader = meta["loader"]

    fig, axs = plt.subplots(1, 3, figsize=(10, 2.9), sharex=True, sharey=True)
    axs = axs.flatten()
    nrows, ncols = 1, 3
    drew_direct = False   # set True once any panel overlays a direct measurement

    for i, (key, redshift, ax, color) in enumerate(zip(keys, redshift_keys, axs, colors)):
        print("RUNNING WITH KEY =", key)

        if key == "2.5":
            # data_ef_ext (Eftekharzadeh+2015, full sample, 18 bins, diagonal
            # errors). Paired with the same dataset in comparison_corr so the
            # two clustering figures use the same z=2.5 observation.
            data = data_ef_ext
            logM_min, logM_max = 11.5, 14.5
            from baqaro.obs_data.corr_inputs_loader import get_input_quantities_auto as get_input_quantities
            log_m_axis, rbins, rp_arr, mf_fit, triangle_fit = get_input_quantities(
                redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
                log_M_min=logM_min, log_M_max=logM_max)
        elif key == "4.0":
            data = Z4_DATA          # standard = all-fields x2 (see BAQARO_CLUST_Z4_DATA)
            logM_min, logM_max = 11.5, 14.5
            from baqaro.obs_data.corr_inputs_loader import get_input_quantities_auto as get_input_quantities
            log_m_axis, rbins, rp_arr, mf_fit, triangle_fit = get_input_quantities(
                redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
                log_M_min=logM_min, log_M_max=logM_max)
        elif key == "6.1":
            data = data_aspire
            data.log_L_threshold = my_utils.to_solar(46.5)
            logM_min, logM_max = 10.5, 14.
            from baqaro.obs_data.corr_inputs_loader import get_input_quantities_cross as get_input_quantities
            log_m_axis, rbins, rvol_bins, mf_fit, triangle_fit = get_input_quantities(
                redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
                log_M_min=logM_min, log_M_max=logM_max)

        print("Working on redshift", redshift, "creating corr")
        snapshot_index = snapshot_index_for_redshift(
            redshifts, redshift, snapshots, label="correlation_plot")

        Lbols_snapshot = loader.get_Lbol(snapshot_index)
        halo_masses_snapshot = loader.get_Halo_mass(snapshot_index)

        mask_qhmf = Lbols_snapshot > 10**data.log_L_threshold
        mass_halos_qhmf = halo_masses_snapshot[mask_qhmf]
        # Subsample-aware per-halo HT weights; None in full-sim mode.
        weights_qhmf = (
            loader.weights[mask_qhmf] if loader.weights is not None else None
        )

        mbins_all_qhmf, mf_all_qhmf, error_all_qhmf = mass_function_auto(
            mass_halos_qhmf, boxsize**3, weights=weights_qhmf,
            lowest_mass=min_mass, highest_mass=max_mass, n_bins=51, minimum_in_bin=1)

        qhmf_here = np.interp(log_m_axis, np.log10(mbins_all_qhmf), mf_all_qhmf, left=0., right=0.)

        if key == "2.5" or key == "4.0":
            log_rbins, xi_init = get_corr_from_triangle(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis, qhmf=qhmf_here)
            wp_init = get_projected_wp(rp_arr, xi_init, 10**log_rbins, pimax=data.pimax)
            wp_rp_init = wp_init / rp_arr
            ax.plot(rp_arr, wp_rp_init, color=color, lw=2.2, zorder=10)
        elif key == "6.1":
            qhmf_gal = np.copy(mf_fit)
            qhmf_gal /= 5.   # scalar: cancels in _compute_mass_weights (no-op, kept for git-archaeology)
            # Galaxy-proxy halo-mass cut = 10.75, from the shared constant, so the
            # analytic curve here and the DIRECT Corrfunc measurement overlaid on the
            # same panel can never drift apart (see
            # obs_data/corr_obs_data.py for the full rationale).
            _gal_logM_min = GAL_LOGM_CUT_FIGURE
            qhmf_gal[log_m_axis < _gal_logM_min] = 0.
            log_rbins, xi_cross_init = get_corr_from_triangle_cross(
                log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                triangle=triangle_fit, log_m_axis=log_m_axis,
                qhmf1=qhmf_here, qhmf2=qhmf_gal)
            xi_vol_init = get_volume_averaged_xi(rvol_bins, xi_cross_init, 10**log_rbins, pimax=data.pimax)
            rvol_bins_lo, rvol_bins_hi = rvol_bins[:-1], rvol_bins[1:]
            rvol_bins_log_centers = np.power(10, (np.log10(rvol_bins_lo) + np.log10(rvol_bins_hi)) / 2.)
            ax.plot(rvol_bins_log_centers, xi_vol_init, color=color, lw=2.2, zorder=10)

        # --- direct-measured overlay (dashed), if the cached measurement exists.
        # Same qhtools projector as the analytic curve, on the SAME rp / rvol grid
        # (get_projected_wp for auto, get_volume_averaged_xi for cross), so the
        # only difference vs the solid curve is measured-vs-halo-model xi(r).
        _npz = os.path.join(_direct_outdir, f"clustering_direct_z{key}.npz")
        _dcache = None
        if os.path.exists(_npz):
            _dcache = np.load(_npz)
            # Guard 1: the dashed overlay is a DIRECT pair-count measurement, only
            # valid on the full-catalogue (multinode) run — refuse a subsample-
            # sourced cache; warn on an unstamped legacy one.
            verify_cache(_dcache, label=f"direct z={key}", expect_full_cat=True)
            # Guard 2: the direct measurement must be the SAME model as the QHMF
            # solid (the pinned fiducial). If not (a fiducial with no multinode
            # run of its own), skip the dashed overlay rather than borrow another
            # model's measurement.
            _dbf = source_bestfit(_dcache)
            if _FID_BESTFIT and _dbf and _dbf != _FID_BESTFIT:
                print(f"  [direct z={key}] SKIP: source model '{_dbf}' != fiducial "
                      f"'{_FID_BESTFIT}' (no matching-model multinode run).")
                _dcache = None
            # Guard 3: for the z=6.1 CROSS, the direct measurement and the analytic
            # curve must use the SAME galaxy-tracer halo-mass cut, or the dashed and
            # solid curves differ by pure bookkeeping and it reads as a halo-model
            # failure (~9% in xi_cross for a 0.05 dex offset). Figure and direct
            # measurement therefore both take GAL_LOGM_CUT_FIGURE from
            # obs_data/corr_obs_data.py.
            if _dcache is not None and key == "6.1" and "logMh_gal_cut" in _dcache.files:
                _cached_cut = float(_dcache["logMh_gal_cut"])
                _want_cut = GAL_LOGM_CUT_FIGURE
                if abs(_cached_cut - _want_cut) > 1e-6:
                    print(f"  [direct z={key}] SKIP: cached galaxy cut "
                          f"logMh>{_cached_cut} != this figure's cut "
                          f"logMh>{_want_cut}. Solid and dashed would differ by "
                          f"bookkeeping, not physics. Re-measure with:\n"
                          f"      env BAQARO_USE_SUBSAMPLE=0 "
                          f"BAQARO_SUBSET_TAG=multinode_root144_v4 \\\n"
                          f"          BAQARO_NOTES_FILE= "
                          f"BAQARO_BESTFIT_NAME=qcc_ck22final_v1 \\\n"
                          f"          BAQARO_GROWTH_SUM_MAX=6.21 BAQARO_CLUST_Z=6.1 python -m "
                          f"baqaro.clustering_direct.measure_clustering_direct")
                    _dcache = None
        if _dcache is not None:
            _d = _dcache
            _r_c = _d["r_centers"]
            _xi = _fill_empty_bins(_r_c, _d["xi"])
            _pimax = float(_d["pimax"])
            if key in ("2.5", "4.0"):
                _wp = get_projected_wp(rp_arr, _xi, _r_c, pimax=_pimax)
                ax.plot(rp_arr, _wp / rp_arr, color=color, lw=1.8, ls="--", zorder=9)
            else:
                _xi_vol = get_volume_averaged_xi(rvol_bins, _xi, _r_c, pimax=_pimax)
                ax.plot(rvol_bins_log_centers, _xi_vol, color=color, lw=1.8, ls="--", zorder=9)
            drew_direct = True

        # z=4 uses the standard all-fields x2 dataset (Z4_DATA); its .err is already
        # x2. An OPTIONAL extra multiplier (BAQARO_CLUST_Z4_ERR_INFLATE) stacks on top
        # for what-if plots (local copy; on-disk data + model curves untouched).
        if key == "4.0":
            print(f"  [z=4] dataset = {Z4_DATA_KEY} (label='{getattr(data,'label','?')}')"
                  + (f", extra err x{Z4_ERR_INFLATE:g}" if Z4_ERR_INFLATE != 1.0 else ""))
        _yerr = data.err
        if key == "4.0" and Z4_ERR_INFLATE != 1.0:
            _yerr = np.asarray(data.err, dtype=float) * Z4_ERR_INFLATE
        ax.errorbar(data.x, data.data, yerr=_yerr, fmt="o",
                    markersize=4, capsize=2, elinewidth=0.8, alpha=0.5, color=color, mec=color)

        # z~6 cross panel: overlay BOTH measurements (ASPIRE 'o', EIGER 's').
        if key == "6.1":
            ax.errorbar(data_eiger.x, data_eiger.data, yerr=data_eiger.err, fmt="s",
                        markersize=4, capsize=2, elinewidth=0.8, alpha=0.5, color=color,
                        mec=color, mfc="none")
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color=color, marker="o", ls="", label="ASPIRE"),
                               Line2D([], [], color=color, marker="s", mfc="none", ls="", label="EIGER")],
                      fontsize=9, loc="lower right", markerfirst=False,
                      handletextpad=0.5, handlelength=0.9, borderpad=0.4)

        is_left = (i % ncols == 0)
        if is_left:
            ax.set_ylabel(r"$w_p(r_p)/r_p,\, \chi_{\mathrm{V}}$", labelpad=-1)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        # Units are COMOVING Mpc, h-free: every corr dataset's separations are
        # divided by cosmo.h at load (corr_obs_data.py), and the model r-grid is
        # h-free too.
        ax.set_xlabel(r"$r_p$ [cMpc]", labelpad=-3)

        if key == "6.1":
            ax.text(0.95, 0.95, f"$z = {key}$, cross", transform=ax.transAxes,
                    fontweight="bold", color=color, ha="right", va="top")
        else:
            ax.text(0.05, 0.95, f"$z = {key}$, auto", transform=ax.transAxes,
                    fontweight="bold", color=color, ha="left", va="top")

    for a in axs:
        a.set_xlim(xmin=0.09, xmax=95.)
        a.set_ylim(ymin=1.1e-1, ymax=1e3)
        a.set_xscale("log")
        a.set_yscale("log")
        a.minorticks_on()

    # Solid = halo-model, dashed = direct measurement — only advertise the
    # dashed convention if a direct curve was actually drawn on some panel.
    if drew_direct:
        axs[0].legend(
            handles=[Line2D([], [], color="0.3", ls="-", label="QHMF-based clustering"),
                     Line2D([], [], color="0.3", ls="--", label="direct quasar clustering")],
            fontsize=10, loc="lower left", markerfirst=True,
            handletextpad=0.5, handlelength=1.6, borderpad=0.4)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.19, hspace=0.05, wspace=0.05)
    save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)

maybe_show()
