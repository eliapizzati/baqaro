"""
PAPER FIGURE: quasar luminosity function (QLF).

Two saved figures:
  * ``qlf_individual`` — a 2×4 panel grid (z = 0.2, 1, 2, 3, 4, 5, 6, 7), each
    panel showing the model QLF for all SMBHs (solid) and M_BH>1e7 (dashed)
    over the observations: the bolometric global compilation, the per-z UV
    points (Kulkarni/Niida/Schindler/Wang/Matsuoka), the JWST LRDs, and
    Bulichi+26 mid-IR AGN.
  * ``qlf_evolution`` — the binned number density vs redshift in three bright
    L_bol bins, against the Shen+20 global model bands.

The **z=7 panel** draws the model at the nearest snapshot on the *emulator's*
redshift ladder (``training_snapshots_default``) rather than the nearest stored
one, so it and its twin in ``plotting_results_comparison``'s ``comparison_qlf``
are the same snapshot. Every other panel keeps the snapshot closest to its
nominal redshift — see the ``LADDER_Z_TARGETS`` block below.

Reads the evolved BH population from the main paper fiducial evolution HDF5
(via ``fiducial_data``). This is a population number-density estimate, so it is
**weight-correct on the subsampled fiducial**: ``luminosity_function_auto`` and
the binned-density helper apply the Horvitz–Thompson ``loader.weights`` (a no-op
in full-sim mode).

Adapted for the paper from the working version of this figure (which
already had the LRD + Bulichi overlays and the bright-bin number-density
panel). Changes: the data handles come from the pinned
``fiducial_data``, the save routes to the git-tracked ``figures_paper/`` (PDF by
default), and the built-but-never-saved combined-overlay panel is dropped (the
grid already shows each redshift).
"""

import os

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from qhtools.utils import my_utils

# Observational data.
from baqaro.obs_data.qlf_uv_obs_data import (
    data_z4_UV_Kulkarni_sys_err,
    data_z2_UV_Kulkarni_sys_err,
    data_z5_UV_Niida_hsc,
    data_z5_UV_Niida_sdss,
    data_z6_UV_Schindler,
    data_z7_UV_Wang,
    data_z7_UV_Matsuoka,
)
from baqaro.obs_data.lrds_agn_obs_data import lrd_qlf_data, bulichi_qlf_data
from baqaro.obs_data.qlf_obs_data import data_qlf_global_raw as data_qlf_global

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    luminosity_function_auto,
    snapshot_index_for_redshift,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show
from baqaro.utils.sim_config import training_snapshots_default


# ==============================================================================
# CONFIG
# ==============================================================================
plot_qlf_base = os.environ.get("BAQARO_QLF_PLOT_BASE", "1") == "1"      # qlf_individual
plot_qlf_global_model = os.environ.get("BAQARO_QLF_PLOT_GLOBAL", "1") == "1"  # qlf_evolution
# The z=7 bolometric compilation point (Matsuoka+23 bolometric, obscuration-
# corrected) overlaps the new Matsuoka+23 UV points, so it's hidden by default
# in the z=7 panel — BAQARO_QLF_SHOW_Z7_BOL=1 to restore it.
show_z7_bol = os.environ.get("BAQARO_QLF_SHOW_Z7_BOL", "0") == "1"

# ------------------------------------------------------------------------------
# EMULATOR-LADDER PANELS (default: z=7 only).
#
# The forward run stores 41 snapshots, the emulator only 17
# (`training_snapshots_default`), and the two grids disagree at three of the
# eight panel targets. Listing a target here makes its model curve use the
# nearest snapshot ON THE EMULATOR LADDER instead of the nearest stored one, so
# the panel matches its twin in the emulator-based `comparison_qlf`.
#
# Only z=7 is listed, deliberately:
#   z=7   run has snap 36 (z=7.005), emulator's nearest rung is snap 37
#         (z=6.708) -> dz=0.30, and the two "z=7" curves differ by ~0.29 dex
#         over log L = 45-46.8, of which ~0.21 dex is the redshift gap alone
#         (not model or emulator error: at a MATCHED snapshot the emulator
#         reproduces the direct run to 0.05-0.10 dex). Worth aligning.
#   z=6   run picks snap 40 (z=5.879), emulator picks snap 39 (z=6.145) — the
#   z=0.2 run picks snap 122 (z=0.200), emulator snap 118 (z=0.261).
#         Both are one-snapshot, near-tie disagreements (at z=6 the two
#         candidates sit 0.121 vs 0.145 from the nominal z), so these panels
#         KEEP the snapshot closest to their nominal redshift. The residual
#         offset against `comparison_qlf` is +0.18 dex at z=6 and small at
#         z=0.2 — a known, accepted difference.
#
# `BAQARO_QLF_LADDER_Z` overrides the list (comma-separated; empty = none).
_ladder_env = os.environ.get("BAQARO_QLF_LADDER_Z", "7.0").strip()
LADDER_Z_TARGETS = [float(t) for t in _ladder_env.split(",") if t.strip()]


def _restrict(redshift):
    """The emulator ladder for a listed panel target, else None (nearest snap)."""
    on_ladder = any(abs(redshift - t) < 1e-6 for t in LADDER_Z_TARGETS)
    return training_snapshots_default if on_ladder else None


# Panel labels stay the OBS key ("z = 7.0") — that is what the plotted DATA is,
# and it keeps this figure's labels identical to `comparison_qlf`'s. The model
# snapshot each panel actually used is printed to stdout at run time.


# ==============================================================================
# MAIN
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    snapshots = data["snapshots"]
    redshifts = data["redshifts"]
    loader = data["loader"]
    erdf = data["erdf"]

    # --------------------------------------------------------------------------
    # FIGURE 1 (qlf_individual): QLF panel grid across redshift.
    # --------------------------------------------------------------------------
    if plot_qlf_base:
        keys = ["0.2", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]
        redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        colors = z_scalar_mapper.to_rgba(redshift_keys)

        nrows, ncols = 2, 4
        # The bottom strip of the canvas is the external obs legend, so the
        # panels are a touch shorter than the bare 2x4 grid would be.
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5.15), sharex=True, sharey=True)
        axs = axs.flatten()

        # ----------------------------------------------------------------------
        # Obs-dataset registry. The UV sets all share the open square (they are
        # one measurement family, and get a single legend entry); the remaining
        # families each get their own marker so the gray legend proxies stay
        # distinguishable once the per-z colour is gone.
        # ----------------------------------------------------------------------
        # Per-redshift UV QLF datasets: {key: [(Qlf_data, marker), ...]}.
        _uv_by_z = {
            "2.0": [(data_z2_UV_Kulkarni_sys_err, "s")],
            "4.0": [(data_z4_UV_Kulkarni_sys_err, "s")],
            # Niida+20 is one measurement in two samples: HSC faint end + SDSS
            # bright end. Both share the UV open square (same paper, same family).
            "5.0": [(data_z5_UV_Niida_hsc, "s"), (data_z5_UV_Niida_sdss, "s")],
            "6.0": [(data_z6_UV_Schindler, "s")],
            "7.0": [(data_z7_UV_Wang, "s"), (data_z7_UV_Matsuoka, "s")],
        }
        # JWST broad-line AGN: marker keyed by survey (was a positional cycle,
        # which made the marker depend on how many sets happen to live at that z).
        _lrd_markers = {"Greene+24": "p", "Matthee+24": "h"}

        # (marker, style, label) for the bottom data legend, in draw order.
        _OPEN = dict(mfc="none", mec="0.35", mew=1.0)
        _FILL = dict(mfc="0.7", mec="0.35", mew=0.8)
        _WHITE = dict(mfc="white", mec="0.35", mew=1.5)
        obs_legend_spec = [
            ("o", _FILL, "Shen+20 (bolometric)"),
            ("s", _OPEN, "Kulkarni+19, Niida+20, Schindler+23, Wang+19, Matsuoka+23 (UV)"),
            ("X", _WHITE, "Bulichi+26 (JWST mid-IR)"),
            ("p", _WHITE, "Greene+24 (JWST broad lines)"),
            ("h", _WHITE, "Matthee+24 (JWST broad lines)"),
        ]

        for i, (key, redshift, ax, color) in enumerate(zip(keys, redshift_keys, axs, colors)):
            print(f"Working on redshift {redshift}")
            snapshot_index = snapshot_index_for_redshift(
                redshifts, redshift, snapshots, label="qlf_individual",
                restrict_to=_restrict(redshift))
            z_model = float(redshifts[snapshot_index])
            print(f"  -> snap {int(snapshots[snapshot_index])} (z={z_model:.3f}, "
                  f"dz={z_model - redshift:+.3f})")

            Lbols_snapshot = loader.get_Lbol(snapshot_index)
            active_mask = Lbols_snapshot > 0
            Lbols_active = Lbols_snapshot[active_mask]
            # In subsample mode each kept halo carries an inverse-probability
            # weight; in full-sim mode loader.weights is None and the dispatcher
            # falls back to unweighted counting.
            weights_active = (
                loader.weights[active_mask] if loader.weights is not None else None
            )
            lbins_all, lf_all, error_lf_all = luminosity_function_auto(
                Lbols_active, box_volume=boxsize**3, weights=weights_active,
                input_type="L_bol",
                lowest_lim=1e9, highest_lim=1e15, n_bins=51, minimum_in_bin=1,
            )
            x_plot = my_utils.to_ergs(np.log10(lbins_all))
            y_plot = np.log10(lf_all)
            ax.plot(x_plot, y_plot, color=color, lw=2.2, zorder=10, ls="-", alpha=1)

            # M_BH > 1e7 cut.
            mass_snapshot = loader.get_BH_mass(snapshot_index)
            combined_mask = active_mask & (mass_snapshot > 1e7)
            Lbols_masked = Lbols_snapshot[combined_mask]
            weights_masked = (
                loader.weights[combined_mask] if loader.weights is not None else None
            )
            lbins_mask, lf_mask, _ = luminosity_function_auto(
                Lbols_masked, box_volume=boxsize**3, weights=weights_masked,
                input_type="L_bol",
                lowest_lim=1e9, highest_lim=1e15, n_bins=51, minimum_in_bin=1,
            )
            x_plot_mask = my_utils.to_ergs(np.log10(lbins_mask))
            y_plot_mask = np.log10(lf_mask)
            ax.plot(x_plot_mask, y_plot_mask, color=color, lw=2.2, zorder=10, alpha=1, ls="--")

            # Bolometric global compilation (z=7 hidden by default — see
            # show_z7_bol above; it duplicates the Matsuoka+23 UV points).
            if key == "7.0" and not show_z7_bol:
                pass
            else:
                try:
                    obs_entry = data_qlf_global[key]
                    ax.errorbar(obs_entry.x, obs_entry.data, yerr=obs_entry.err, fmt="o",
                                label=obs_entry.label, markersize=4, capsize=2, elinewidth=0.8,
                                alpha=0.3, color=color, mec=color)
                except KeyError:
                    print(f"No bolometric data for z={key}")

            # Per-z UV points (each entry: (Qlf_data, marker)). z=7 shows both
            # Wang+19 (bright end) and Matsuoka+23 (extends to the faint-end
            # flattening); same UV-QLF convention.
            for obs_d, uv_mk in _uv_by_z.get(key, []):
                ax.errorbar(my_utils.to_ergs(obs_d.x), obs_d.data, obs_d.err,
                            c=color, mec=color, marker=uv_mk, alpha=0.3, capsize=2,
                            elinewidth=0.8, fillstyle="none", linestyle="none", markersize=4)

            # JWST LRDs (matched by redshift key) — drawn as UPPER LIMITS in
            # L_bol (leftward arrows): LRD bolometric luminosities are upper
            # bounds on the intrinsic AGN L_bol (uncertain bolometric
            # corrections / possible host contamination), so their true L_bol
            # lies to the left of the plotted point. The arrow is a clean
            # fixed-length annotate (0.5 dex) rather than a matplotlib caret.
            if key in lrd_qlf_data:
                for lrd in lrd_qlf_data[key]:
                    mk = next((m for surv, m in _lrd_markers.items()
                               if lrd.label.startswith(surv)), "o")
                    ax.errorbar(lrd.x, lrd.data, yerr=[lrd.err_down, lrd.err_up],
                                fmt=mk, label=lrd.label, markersize=6, capsize=2,
                                elinewidth=1.0, ecolor=color, alpha=0.8, color="white",
                                mec=color, markeredgewidth=1.5, linestyle="none", zorder=5)
                    # Arrow tail sits at the marker (shrinkA=0) so it reads as
                    # emerging from the point, not detached.
                    for xi, yi in zip(np.atleast_1d(lrd.x), np.atleast_1d(lrd.data)):
                        ax.annotate("", xy=(xi - 0.5, yi), xytext=(xi, yi), zorder=4,
                                    arrowprops=dict(arrowstyle="->", color=color,
                                                    lw=1.2, alpha=0.85,
                                                    mutation_scale=11,
                                                    shrinkA=0, shrinkB=0))

            # Bulichi+26 mid-IR AGN (matched by redshift key).
            if key in bulichi_qlf_data:
                bul = bulichi_qlf_data[key]
                ax.errorbar(bul.x, bul.data, yerr=[bul.err_down, bul.err_up],
                            fmt="X", label=bul.label, markersize=7, capsize=2,
                            elinewidth=1.0, alpha=0.8, color="white", mec=color,
                            markeredgewidth=1.5, linestyle="none", zorder=5)

            ax.text(0.05, 0.05, f"$z = {key}$", transform=ax.transAxes,
                    fontweight="bold", color=color, ha="left", va="bottom")

            is_left = (i % ncols == 0)
            is_bottom = (i // ncols == nrows - 1)
            if is_left:
                ax.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]", labelpad=-1)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if is_bottom:
                ax.set_xlabel(r"$\log_{10}\, L_{\rm bol}$ [erg s$^{-1}$]", labelpad=-3)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        for a in axs:
            a.set_xlim(xmin=43.5, xmax=48.35)
            a.set_ylim(ymin=-9.8, ymax=-2.5)
            a.minorticks_on()

        # Model-curve legend (gray proxies) on the last panel.
        solid_proxy = matplotlib.lines.Line2D([0], [0], color="gray", lw=2.2, linestyle="-", label=r"All SMBHs")
        dashed_proxy = matplotlib.lines.Line2D([0], [0], color="gray", lw=2.2, linestyle="--", label=r"$M_{\rm BH}>10^7\,M_\odot$")
        axs[-1].legend(handles=[solid_proxy, dashed_proxy], loc="upper right", markerfirst=False,
                       handlelength=1.8, labelspacing=0.25, fontsize=11, borderpad=0.5)

        # Observations: external legend under the grid (gray proxies, one per
        # dataset — same pattern as the quasar-bias figure).
        #
        # Two rows, grouped by facility: pre-JWST (bolometric + UV) on top, the
        # three JWST sets below. A single legend can only fill column-major, so
        # the rows are two separate fig.legend calls stacked by bbox_to_anchor.
        # (One row would force ~7 pt to fit the 10-in width, and an overflowing
        # legend makes the tight-bbox save widen the canvas, shrinking the
        # panels against the text column.)
        def _handles(labels):
            return [matplotlib.lines.Line2D([], [], ls="", marker=mk, ms=7, label=lbl, **st)
                    for mk, st, lbl in obs_legend_spec if lbl in labels]

        _jwst = [lbl for _, _, lbl in obs_legend_spec if "JWST" in lbl]
        _pre_jwst = [lbl for _, _, lbl in obs_legend_spec if "JWST" not in lbl]
        # borderpad=0 matters: with frameon=False the padding is invisible but
        # still occupies space, and the tight-bbox save keeps it as white margin.
        _legend_kw = dict(loc="lower center", frameon=False, fontsize=11,
                          handletextpad=0.35, columnspacing=1.6,
                          borderaxespad=0.0, borderpad=0.0)
        fig.legend(handles=_handles(_pre_jwst), bbox_to_anchor=(0.5, 0.044),
                   ncol=len(_pre_jwst), **_legend_kw)
        fig.legend(handles=_handles(_jwst), bbox_to_anchor=(0.5, 0.0),
                   ncol=len(_jwst), **_legend_kw)

        fig.subplots_adjust(left=0.08, right=0.98, top=0.985, bottom=0.165, hspace=0.05, wspace=0.05)
        save_fig(fig, plot_config.FIGURES_DIR, "qlf_individual", force_dir=True)

    # --------------------------------------------------------------------------
    # FIGURE 2 (qlf_evolution): number density vs redshift in bright L_bol bins.
    # --------------------------------------------------------------------------
    if plot_qlf_global_model:
        from baqaro.obs_data.qlf_shen_model import plot_phi_obs_A, plot_phi_obs_B

        fig_glob, ax_glob = plt.subplots(1, 1, figsize=(5.5, 4.2))
        log_Lbins = np.linspace(45.5, 48.5, 4)
        colors_Lbin = plt.cm.cividis(np.linspace(0.15, 0.85, len(log_Lbins) - 1))

        # Shen+20 global-model bands.
        for j in range(len(log_Lbins) - 1):
            Phi_obs_A, Error_obs_A, Z_A = plot_phi_obs_A(Lmin=log_Lbins[j], Lmax=log_Lbins[j + 1])
            Phi_obs_B, Error_obs_B, Z_B = plot_phi_obs_B(Lmin=log_Lbins[j], Lmax=log_Lbins[j + 1])
            min_obs = np.minimum(Phi_obs_A - Error_obs_A, Phi_obs_B - Error_obs_B)
            max_obs = np.maximum(Phi_obs_A + Error_obs_A, Phi_obs_B + Error_obs_B)
            lbl = rf"$\log_{{10}}\,L_{{\rm bol}}/\mathrm{{erg\,s^{{-1}}}} = {log_Lbins[j]:.1f}$–${log_Lbins[j + 1]:.1f}$"
            ax_glob.fill_between(Z_A, max_obs, min_obs, color=colors_Lbin[j], alpha=0.35,
                                 edgecolor=colors_Lbin[j], linewidth=1.2, label=lbl)

        redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        for redshift in redshift_keys:
            print(f"Working on redshift {redshift}")
            # Same ladder rule as the grid above (the points are drawn at their
            # TRUE redshift, so this only changes which snapshots appear).
            snapshot_index = snapshot_index_for_redshift(
                redshifts, redshift, snapshots, label="qlf_evolution",
                restrict_to=_restrict(redshift))

            lb_snap = loader.get_Lbol(snapshot_index)
            # Subsample HT weights restore the true number density; without them
            # this counts only the KEPT halos (undercount). None in full-sim mode.
            w_all = loader.weights
            Lbins_solar = my_utils.to_solar(log_Lbins)

            def _weighted_density(active_mask):
                """Weighted number density + 1sigma error per L-bin, IN DEX.

                The HT error on the linear density is sqrt(sum w^2)/V, so the
                RELATIVE error is sqrt(sum w^2)/sum(w). It is plotted against
                log10(nd), so it must be converted to dex with the
                d(log10 x)/dx = 1/(x ln10) Jacobian, i.e. x 0.434.
                """
                logL = np.log10(lb_snap[active_mask])
                w = w_all[active_mask] if w_all is not None else None
                c, _ = np.histogram(logL, bins=Lbins_solar, weights=w)
                sumw2 = c if w is None else np.histogram(logL, bins=Lbins_solar, weights=w**2)[0]
                nd = c / (boxsize**3)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel_err = (np.sqrt(sumw2) / (boxsize**3)) / nd
                    yerr = 0.4342944819032518 * rel_err      # log10(e): linear -> dex
                return nd, yerr

            active_all = lb_snap > 0
            ndens, yerr = _weighted_density(active_all)
            for j in range(len(ndens)):
                if ndens[j] > 0:
                    ax_glob.errorbar(redshifts[snapshot_index], np.log10(ndens[j]), yerr=yerr[j],
                                     linestyle="", marker="o", zorder=10, color=colors_Lbin[j],
                                     alpha=0.9, markersize=7, capsize=2, elinewidth=1.0)

        # Legend: the three L_bol bands plus the all-SMBH model-marker key.
        all_proxy = matplotlib.lines.Line2D([], [], color="gray", marker="o", linestyle="none",
                                            markersize=7, label="All SMBHs")
        band_handles, _ = ax_glob.get_legend_handles_labels()
        ax_glob.legend(handles=band_handles + [all_proxy], loc="upper right",
                       markerfirst=False, handlelength=1.6, fontsize=10, labelspacing=0.3,
                       borderpad=0.5)
        ax_glob.set_ylim(ymin=-9.8, ymax=-3.2)
        ax_glob.set_xlabel(r"Redshift $z$", labelpad=4)
        ax_glob.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]", labelpad=-1)
        ax_glob.minorticks_on()
        fig_glob.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.15)
        save_fig(fig_glob, plot_config.FIGURES_DIR, "qlf_evolution", force_dir=True)

maybe_show()
