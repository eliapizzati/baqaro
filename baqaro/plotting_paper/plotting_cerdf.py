"""
PAPER FIGURE: conditional Eddington-ratio distribution (CERDF).

A grid of P(log lambda_Edd) panels — rows = bolometric-luminosity bins, columns
= redshift (z = 1…6). Each panel overlays the observed Eddington-ratio
histogram (from the QSO M_BH / L_bol sample) against the model distribution at
the same z and L_bol bin, with dotted/dashed vertical medians.

Reads the evolved BH population from the main paper fiducial evolution HDF5
(via ``fiducial_data``). The model CERDF is a conditional population PDF, so it
is **weight-correct on the subsampled fiducial**: the per-object Horvitz–Thompson
``loader.weights`` go into both the ``np.histogram(..., density=True)`` (the
per-bin weight cancels in the normalisation) and the ``weighted_median``.

Adapted for the paper from the working version of this figure; the
figure content is unchanged — only the data handles come from the pinned
``fiducial_data`` and the save routes to the git-tracked ``figures_paper/``
(PDF by default).
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

from qhtools.utils import my_utils
import qhtools.utils.natconst as nc

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    boxsize,
    load_simulation_metadata,
    weighted_median,
    snapshot_index_for_redshift,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show, figure_rng

# Seeded: the measurement-error convolution below must be reproducible.
# Set BAQARO_FIG_SEED to inspect realisation-to-realisation scatter.
_FIG_RNG = figure_rng()

from baqaro.obs_data.qso_obs_data_setup import (
    redshifts_data, logL_Bols_data, logM_BHs_data,
)


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "cerdf_distribution"

# Grid: rows = luminosity bins, columns = redshifts.
redshift_keys = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
keys = ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0"]

logL_bins = np.linspace(45.5, 47.5, 5)   # 4 rows: [45.5, 46), [46, 46.5), [46.5, 47), [47, 47.5)
logL_bin_centers = 0.5 * (logL_bins[1:] + logL_bins[:-1])
n_L_bins = len(logL_bin_centers)
n_z = len(redshift_keys)

z_colors = z_scalar_mapper.to_rgba(redshift_keys)


# ==============================================================================
# MAIN
# ==============================================================================
with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    snapshots = data["snapshots"]
    redshifts = data["redshifts"]
    loader = data["loader"]

    fig, axes = plt.subplots(n_L_bins, n_z, figsize=(10, 6.5), sharex=True, sharey=True)

    for i_z, (key, redshift) in enumerate(zip(keys, redshift_keys)):
        mask_z_data = (redshifts_data < redshift + 0.5) & (redshifts_data > redshift - 0.5)

        print(f"Working on redshift {redshift}")
        snapshot_index = snapshot_index_for_redshift(
            redshifts, redshift, snapshots, label="cerdf_distribution")

        Lbols_snapshot = loader.get_Lbol(snapshot_index)
        active_mask = Lbols_snapshot > 0
        Lbols_active = Lbols_snapshot[active_mask]

        black_hole_masses_snapshot = loader.get_BH_mass(snapshot_index)
        black_hole_masses_active = black_hole_masses_snapshot[active_mask]
        # HT weights for subsample mode; CERDF is a PDF (density=True).
        weights_active = (
            loader.weights[active_mask] if loader.weights is not None else None
        )

        # Model Eddington ratios: log_eta = log(Lbol/Lsun) - log(MBH/Msun) - log_csi
        log_eta_model_all = np.log10(Lbols_active) - np.log10(black_hole_masses_active) - nc.log_csi
        # Observed Eddington ratios.
        log_etas_data = my_utils.to_solar(logL_Bols_data) - logM_BHs_data - nc.log_csi

        for i_L in range(n_L_bins):
            ax = axes[i_L, i_z]
            line_color = z_colors[i_z]
            lmin_val, lmax_val = logL_bins[i_L], logL_bins[i_L + 1]

            # --- Observations ---
            mask_L_data = (logL_Bols_data > lmin_val) & (logL_Bols_data < lmax_val)
            mask_data = mask_z_data & mask_L_data
            if np.sum(mask_data) > 3:
                log_eta_obs = log_etas_data[mask_data]
                bins_obs = np.linspace(-3, 1.5, 25)
                ax.hist(log_eta_obs, bins_obs, density=True, histtype="step", lw=2.5,
                        alpha=0.5, linestyle="-", color=line_color, label="Observations")
                ax.axvline(np.median(log_eta_obs), lw=2.5, ls=":", alpha=0.5, color=line_color)

            # --- Model ---
            mask_L_model = ((Lbols_active > 10 ** my_utils.to_solar(lmin_val))
                            & (Lbols_active < 10 ** my_utils.to_solar(lmax_val)))
            log_eta_model = log_eta_model_all[mask_L_model]
            log_eta_model_scatter = log_eta_model + _FIG_RNG.normal(0, 0.3, size=log_eta_model.shape)
            weights_L_model = (
                weights_active[mask_L_model] if weights_active is not None else None
            )

            bins_model = np.linspace(-3, 1.5, 25)  # ~0.19 dex, matching obs binning
            bin_centers = 0.5 * (bins_model[1:] + bins_model[:-1])

            if len(log_eta_model_scatter) > 0:
                counts_scatter, _ = np.histogram(
                    log_eta_model_scatter, bins=bins_model,
                    weights=weights_L_model, density=True,
                )
                ax.plot(bin_centers, counts_scatter, lw=1.8, ls="-", color=line_color,
                        alpha=0.9, zorder=10, label="Model")
                ax.axvline(weighted_median(log_eta_model, weights_L_model),
                           lw=1.8, ls="--", alpha=0.9, color=line_color, zorder=10)

            # --- Labels ---
            if i_L == 0:
                ax.set_title(f"$z = {key}$", fontsize=13)
            if i_z == 0:
                lmin_str = f"{lmin_val:.0f}" if lmin_val == int(lmin_val) else f"{lmin_val:.1f}"
                lmax_str = f"{lmax_val:.0f}" if lmax_val == int(lmax_val) else f"{lmax_val:.1f}"
                ax.set_ylabel(rf"$L_{{bol}} \in [{lmin_str}, {lmax_str}]$"
                              "\n" + r"$P(\log \lambda_{\rm Edd})$", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            if i_L == n_L_bins - 1:
                ax.set_xlabel(r"$\log_{10}\,\lambda_{\rm Edd}$", labelpad=-1)
            else:
                ax.tick_params(labelbottom=False)
            if i_z == 0 and i_L == 0:
                # Top-left panel; slightly shorter marker→text spacing so the
                # labels sit a bit closer to their lines without cramping.
                ax.legend(loc="upper left", fontsize=10, handlelength=1.4,
                          handletextpad=0.4, labelspacing=0.3)

    for a in axes.flatten():
        a.set_xlim(-2.5, 1.5)
        a.set_ylim(0., 1.9)
        a.minorticks_on()

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.12, hspace=0.05, wspace=0.05)
    save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)

maybe_show()
