"""
PAPER FIGURE: quasar host-halo mass function (QHMF).

One saved figure:
  * ``qhmf_individual`` — a 2×4 panel grid (z = 0.2…7); each panel shows the
    QHMF (hosts of QSOs above log L_bol = 46.5, solid + error band) against the
    total halo mass function (dashed), with the integrated median + 16–84% band.

Reads the evolved BH population from the main paper fiducial evolution HDF5
(via ``fiducial_data``). The QHMF and total HMF are population mass functions, so
they are **weight-correct on the subsampled fiducial**: ``mass_function_auto``
applies the per-host Horvitz–Thompson ``loader.weights`` (a no-op in full-sim).

The **z=7 panel** draws the model at the nearest snapshot on the *emulator's*
redshift ladder (``training_snapshots_default``) rather than the nearest stored
one, so it and its twin in ``plotting_results_comparison``'s ``comparison_qhmf``
are the same snapshot. Every other panel keeps the snapshot closest to its
nominal redshift — see the ``LADDER_Z_TARGETS`` block below.

Adapted for the paper from the working version of this figure; the
figure content is unchanged — only the data handles come from the pinned
``fiducial_data`` and the save routes to the git-tracked ``figures_paper/``
(PDF by default).
"""

import os

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt

from qhtools.utils import my_utils

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    mass_function_auto,
    snapshot_index_for_redshift,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show
from baqaro.utils.sim_config import training_snapshots_default


# ==============================================================================
# CONFIG
# ==============================================================================
log_Lbol_cut = 46.5
min_mass, max_mass = 1e10, 1e15

# ------------------------------------------------------------------------------
# EMULATOR-LADDER PANELS (default: z=7 only) — same rule and rationale as
# `plotting_qlf`, kept in sync with it.
#
# At z=7 the forward run offers snap 36 (z=7.005) while the emulator's nearest
# rung is snap 37 (z=6.708): a 0.30-in-z gap that shows up as a spurious offset
# against `comparison_qhmf`. The z=6 and z=0.2 panels also sit one snapshot off
# the ladder (40/z=5.879 vs 39/z=6.145; 122/z=0.200 vs 118/z=0.261), but those
# are near-tie disagreements and those panels deliberately KEEP the snapshot
# closest to their nominal redshift, accepting the small residual offset.
#
# `BAQARO_QHMF_LADDER_Z` overrides the list (comma-separated; empty = none).
_ladder_env = os.environ.get("BAQARO_QHMF_LADDER_Z", "7.0").strip()
LADDER_Z_TARGETS = [float(t) for t in _ladder_env.split(",") if t.strip()]


def _restrict(redshift):
    """The emulator ladder for a listed panel target, else None (nearest snap)."""
    on_ladder = any(abs(redshift - t) < 1e-6 for t in LADDER_Z_TARGETS)
    return training_snapshots_default if on_ladder else None


# Panel labels stay the nominal z ("z = 7.0"), matching `comparison_qhmf`. The
# model snapshot each panel actually used is printed to stdout at run time.


# ==============================================================================
# MAIN
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    snapshots = data["snapshots"]
    redshifts = data["redshifts"]
    loader = data["loader"]

    keys = ["0.2", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]
    redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    colors = z_scalar_mapper.to_rgba(redshift_keys)

    nrows, ncols = 2, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, (key, redshift, ax, color) in enumerate(zip(keys, redshift_keys, axs, colors)):
        print(f"Working on redshift {redshift}")
        snapshot_index = snapshot_index_for_redshift(
            redshifts, redshift, snapshots, label="qhmf_individual",
            restrict_to=_restrict(redshift))
        z_model = float(redshifts[snapshot_index])
        print(f"  -> snap {int(snapshots[snapshot_index])} (z={z_model:.3f}, "
              f"dz={z_model - redshift:+.3f})")

        Lbols_snapshot = loader.get_Lbol(snapshot_index)
        halo_masses_snapshot = loader.get_Halo_mass(snapshot_index)

        # Quasar host halos (above the L_bol cut).
        mask_qhmf = Lbols_snapshot > 10 ** my_utils.to_solar(log_Lbol_cut)
        halo_masses_qhmf = halo_masses_snapshot[mask_qhmf]
        weights_qhmf = (
            loader.weights[mask_qhmf] if loader.weights is not None else None
        )

        # Total HMF (all halos).
        mbins_all_hmf, mf_all_hmf, _ = mass_function_auto(
            halo_masses_snapshot, boxsize**3, weights=loader.weights,
            lowest_mass=min_mass, highest_mass=max_mass, n_bins=51, minimum_in_bin=1,
        )
        # QHMF.
        mbins_all_qhmf, mf_all_qhmf, error_all_qhmf = mass_function_auto(
            halo_masses_qhmf, boxsize**3, weights=weights_qhmf,
            lowest_mass=min_mass, highest_mass=max_mass, n_bins=51, minimum_in_bin=1,
        )

        x_plot = np.log10(mbins_all_qhmf)
        y_plot = np.log10(mf_all_qhmf)
        # Floor the lower Poisson edge so single-object bins (where error == density,
        # giving density - error == 0) do not send log10 -> -inf and break the shaded
        # band in the sparse edge-redshift panels (z=0.2, z=7.0). The floor sits just
        # below the panel ymin, so those bins read as "consistent with zero" (band open
        # to the bottom) rather than as a gap.
        y_err_lower = np.log10(np.clip(mf_all_qhmf - error_all_qhmf, 1e-10, None))
        y_err_upper = np.log10(mf_all_qhmf + error_all_qhmf)

        ax.plot(x_plot, y_plot, color=color, lw=2.2, zorder=10, ls="-", alpha=1)
        ax.fill_between(x_plot, y_err_lower, y_err_upper, alpha=0.3, color=color)
        ax.plot(np.log10(mbins_all_hmf), np.log10(mf_all_hmf), lw=2.2, color=color, ls="--")

        # Integrated median + 16–84% band (robust vs the one-bin-noisy peak).
        lower, median, upper = my_utils.get_percentiles(
            mf_all_qhmf, np.log10(mbins_all_qhmf), percentiles=[0.16, 0.5, 0.84])
        ax.axvline(x=median, color=color, lw=1.5, ls="--")
        ax.axvspan(xmin=lower, xmax=upper, color=color, lw=1.5, ls=":", alpha=0.2)

        ax.text(0.95, 0.95, f"$z = {key}$", transform=ax.transAxes,
                fontweight="bold", color=color, ha="right", va="top")

        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1)
        if is_left:
            ax.set_ylabel(r"dn/dlog M$_\mathrm{h}$ [cMpc$^{-3}$ dex$^{-1}$]", labelpad=-1)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\log_{10}$ M$_\mathrm{halo}$ [M$_\odot$]", labelpad=-1)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

    for a in axs:
        a.set_xlim(xmin=11.5, xmax=13.9)
        a.set_ylim(ymin=-9.8, ymax=-2.5)
        a.minorticks_on()

    # L_bol threshold (log L_bol > 46.5) is stated in the caption, not the legend
    # (keeps the legend short).
    solid_proxy = matplotlib.lines.Line2D(
        [0], [0], color="gray", lw=2.2, linestyle="-", label="QHMF")
    dashed_proxy = matplotlib.lines.Line2D(
        [0], [0], color="gray", lw=2.2, linestyle="--", label="Total HMF")
    axs[-1].legend(handles=[solid_proxy, dashed_proxy], loc="upper right",
                   markerfirst=False, handlelength=1.3, labelspacing=0.4,
                   fontsize=10, borderpad=0.4, bbox_to_anchor=(1.0, 0.9))

    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, hspace=0.05, wspace=0.05)
    save_fig(fig, plot_config.FIGURES_DIR, "qhmf_individual", force_dir=True)

maybe_show()
