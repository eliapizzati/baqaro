"""
PAPER FIGURE: ERDF model visualisation.

Two figures, selected by the module-level flags below:

  * ``erdfs_theory_z``  (``plot_model_redshift``):
    the model Eddington-ratio distribution — the histogram of log10
    lambda_Edd over the evolved BH population — at several redshifts,
    colour-coded by z.

  * ``erdfs_theory_Mrate``  (``plot_model_halo_rate``):
    the ERDF binned by the cold specific halo accretion rate.  In each rate
    bin it compares the realized Eddington-ratio distribution (lambda_Edd,
    solid histogram) against the analytic ERDF model curve (dotted), so one
    can read off how the lognormal model maps a halo-rate bin onto the BH
    accretion-rate distribution.  Needs the on-disk specific-cold-accretion
    arrays and is **full-sim only** (it index-aligns the full halo catalogue to
    the BH loader, which breaks under a subsample) — to use it, override the
    fiducial to a full-sim run with a low ``BAQARO_PAPER_MAX_SNAP`` (see below).

Both read the evolved BH population from the pinned paper fiducial evolution
HDF5 (via ``fiducial_data``). The redshift ERDF is a marginal PDF over the BH
population, so on the subsampled fiducial its histogram **and its per-z median**
apply the Horvitz–Thompson ``loader.weights`` to stay unbiased (a no-op in
full-sim mode, where ``loader.weights is None``).

Adapted for the paper from the working ERDF-model figure with the
correctness fixes carried by every paper halo-array reader (cf.
``plotting_halo_rate.py``):
  * snapshot rows are indexed ``arr[i]`` — the on-disk layout is
    ``(n_snapshots, n_halos)`` C-order; the old halo-rate branch used the
    stale F-order ``arr[:, i]``.
  * the specific-cold-accretion filename is the canonical ``name_file_halos``
    (``{simulation_name}_`` prefix + optional ``_foldmass`` /
    ``_{merger_delay_mode}`` tokens) imported from ``load_data_to_plot``.

Output goes to the git-tracked ``figures_paper/`` via the shared paper config.
"""

import os

import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import qhtools.utils.natconst as nc

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
# This import must precede anything that reads BAQARO_*; it force-pins the env.
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    path_out,
    max_snap,
    name_file_halos,
    subset_tag,
    load_simulation_metadata,
    weighted_median,
    weighted_percentile,
    snapshot_index_for_redshift,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import cmap_z, cmap_M
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
name_fig_z = "erdfs_theory_z"
name_fig_M = "erdfs_theory_Mrate"
name_fig_z_summary = "erdfs_theory_z_summary"

plot_model_redshift = True    # ERDF of the BH population vs redshift
plot_model_halo_rate = False  # ERDF binned by cold specific halo accretion rate

# log10 lambda_Edd above which a BH counts as "actively accreting".
# log10(0.1) = -1 -> the active fraction is f(lambda_Edd > 0.1).
LOG_LAMBDA_ACTIVE = -1.0

# Redshift-driven (sim-independent): for each target z the script picks the
# nearest snapshot in the loaded sim's z grid.
REDSHIFT_TARGETS = [8.7, 7.3, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]


# ------------------------------------------------------------------------------
# LOAD
# ------------------------------------------------------------------------------
with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata(file)
    snapshots = data["snapshots"]
    redshifts = data["redshifts"]
    loader = data["loader"]
    erdf = data["erdf"]

    # --------------------------------------------------------------------------
    # FIGURE 1: ERDF of the evolved BH population vs redshift.
    # --------------------------------------------------------------------------
    if plot_model_redshift:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        # Snapshots nearest each target redshift (de-duplicated, in range).
        snapshots_to_plot = []
        for _tz in REDSHIFT_TARGETS:
            _i = snapshot_index_for_redshift(
                redshifts, _tz, snapshots, label="erdf_model")
            if _i <= max_snap and _i not in snapshots_to_plot:
                snapshots_to_plot.append(_i)
        redshifts_plot = redshifts[snapshots_to_plot]
        z_lo, z_hi = redshifts_plot.min(), redshifts_plot.max()

        def _z_color(z):
            return cmap_z((z - z_lo) / (z_hi - z_lo)) if z_hi > z_lo else cmap_z(0.5)

        # Per-redshift quantitative summary of each ERDF: weighted 1sigma
        # (16/84) and 2sigma (2.3/97.7) percentiles of log10 lambda_Edd, the
        # weighted median, and the active fraction f(lambda_Edd > 0.1). All
        # apply the HT loader.weights so they are unbiased on the subsample.
        summary_rows = []

        for i in snapshots_to_plot:
            redshift = redshifts[i]
            print("Working on snapshot", i, f"z={redshift:.2f}", "creating ERDF")

            Lbols_snapshot = loader.get_Lbol(i)
            black_hole_masses_snapshot = loader.get_BH_mass(i)

            with np.errstate(divide="ignore", invalid="ignore"):
                etas = Lbols_snapshot / black_hole_masses_snapshot / 10**nc.log_csi
                log_etas = np.log10(etas)
            finite_mask = np.isfinite(log_etas)
            # HT inverse-probability weights unbias this marginal-ERDF PDF on the
            # subsampled fiducial; None in full-sim mode -> identical to unweighted.
            weights = loader.weights[finite_mask] if loader.weights is not None else None
            ax.hist(log_etas[finite_mask], bins=100, weights=weights, density=True,
                    histtype="step", lw=2, log=True, color=_z_color(redshift), ls="-")

            # Dotted vertical at the (weighted) median log lambda_Edd, matching the
            # per-z median verticals in plotting_halo_rate.py. weighted_median falls
            # back to the unweighted median when weights is None (full-sim mode).
            log_etas_finite = log_etas[finite_mask]
            med = weighted_median(log_etas_finite, weights)
            ax.axvline(med, color=_z_color(redshift), lw=1.5, ls=":")

            # Quantitative summary: weighted percentiles (2sigma/1sigma/median)
            # and the weighted active fraction f(lambda_Edd > 0.1).
            p02 = weighted_percentile(log_etas_finite, weights, 0.0227)
            p16 = weighted_percentile(log_etas_finite, weights, 0.1587)
            p84 = weighted_percentile(log_etas_finite, weights, 0.8413)
            p98 = weighted_percentile(log_etas_finite, weights, 0.9773)
            active = log_etas_finite > LOG_LAMBDA_ACTIVE
            if weights is not None:
                w_tot = float(np.sum(weights))
                f_active = float(np.sum(weights[active]) / w_tot) if w_tot > 0 else float("nan")
            else:
                f_active = float(np.mean(active)) if log_etas_finite.size else float("nan")
            summary_rows.append(
                dict(z=redshift, p02=p02, p16=p16, med=med, p84=p84, p98=p98,
                     f_active=f_active)
            )

        ax.axvline(np.log10(1.0), color="black", lw=1.5, ls="--", zorder=-10)

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_z, norm=plt.Normalize(vmin=z_lo, vmax=z_hi)),
            ax=ax,
        )
        cbar.set_label("Redshift")
        cbar.set_ticks(np.arange(z_lo, z_hi, 0.6))

        ax.set_xlabel(r"Eddington ratio, $\log_{10}\,\lambda_\mathrm{Edd}$", labelpad=-1)
        ax.set_ylabel(r"Probability distribution")
        ax.set_xlim(-4, 1.4)
        ax.set_ylim(1e-4, 8e0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)

        save_fig(fig, plot_config.FIGURES_DIR, name_fig_z, force_dir=True)

        # ----------------------------------------------------------------------
        # Quantitative summary: print a per-redshift table of the ERDF
        # percentiles + active fraction, then a companion vs-z figure.
        # ----------------------------------------------------------------------
        # Sort high-z -> low-z for readability (snapshots_to_plot already is).
        print("\nERDF quantitative summary (log10 lambda_Edd, weighted)")
        print("  active fraction = f(lambda_Edd > 0.1)  [log10 lambda_Edd > -1]")
        header = (f"{'z':>6} | {'-2sig':>7} {'-1sig':>7} {'median':>7} "
                  f"{'+1sig':>7} {'+2sig':>7} | {'f_active':>8}")
        print("  " + header)
        print("  " + "-" * len(header))
        for r in summary_rows:
            print(f"  {r['z']:6.2f} | {r['p02']:7.2f} {r['p16']:7.2f} {r['med']:7.2f} "
                  f"{r['p84']:7.2f} {r['p98']:7.2f} | {r['f_active']:8.3f}")

        z_arr = np.array([r["z"] for r in summary_rows])
        med_arr = np.array([r["med"] for r in summary_rows])
        p16_arr = np.array([r["p16"] for r in summary_rows])
        p84_arr = np.array([r["p84"] for r in summary_rows])
        p02_arr = np.array([r["p02"] for r in summary_rows])
        p98_arr = np.array([r["p98"] for r in summary_rows])
        f_active_arr = np.array([r["f_active"] for r in summary_rows])

        fig2, (axA, axB) = plt.subplots(
            2, 1, figsize=(6, 6.5), sharex=True,
            gridspec_kw=dict(height_ratios=[2, 1]),
        )

        # Top: median log lambda_Edd with 1sigma + 2sigma percentile bands.
        axA.fill_between(z_arr, p02_arr, p98_arr, color="C0", alpha=0.18,
                         lw=0, label=r"$2\sigma$ (2.3-97.7%)")
        axA.fill_between(z_arr, p16_arr, p84_arr, color="C0", alpha=0.35,
                         lw=0, label=r"$1\sigma$ (16-84%)")
        axA.plot(z_arr, med_arr, color="C0", lw=2, marker="o", ms=4,
                 label="median")
        axA.axhline(LOG_LAMBDA_ACTIVE, color="gray", lw=1.2, ls=":",
                    zorder=-10, label=r"$\lambda_\mathrm{Edd}=0.1$")
        axA.axhline(0.0, color="black", lw=1.2, ls="--", zorder=-10)
        axA.set_ylabel(r"$\log_{10}\,\lambda_\mathrm{Edd}$")
        axA.legend(loc="best", fontsize=8, ncol=2)
        axA.yaxis.set_minor_locator(AutoMinorLocator())

        # Bottom: active fraction f(lambda_Edd > 0.1) vs z.
        axB.plot(z_arr, f_active_arr, color="C3", lw=2, marker="s", ms=4)
        axB.set_ylabel(r"$f(\lambda_\mathrm{Edd}>0.1)$")
        axB.set_xlabel("Redshift")
        axB.set_ylim(0, 1)
        axB.yaxis.set_minor_locator(AutoMinorLocator())
        axB.xaxis.set_minor_locator(AutoMinorLocator())

        fig2.subplots_adjust(left=0.13, right=0.96, top=0.97, bottom=0.1, hspace=0.07)

        save_fig(fig2, plot_config.FIGURES_DIR, name_fig_z_summary, force_dir=True)

    # --------------------------------------------------------------------------
    # FIGURE 2: ERDF binned by the cold specific halo accretion rate.
    # --------------------------------------------------------------------------
    if plot_model_halo_rate:
        # This branch reads the FULL halo catalogue and index-aligns it row-for-row
        # to loader.get_Lbol(i). That alignment only holds in full-sim mode — under
        # a subsample (the current paper fiducial) the full array is ~3.24B halos
        # while the loader is the ~386M subset, so the masks would be meaningless
        # (and the full maxsnap=144 specific-cold array may not even be on disk).
        if subset_tag:
            raise RuntimeError(
                "plot_model_halo_rate is incompatible with a subsampled run "
                f"(subset_tag={subset_tag!r}): the full halo catalogue does not "
                "align with the subset BH loader. Run it against a full-sim fiducial, "
                "e.g. BAQARO_PAPER_SIM=L2800N5040 BAQARO_PAPER_MAX_SNAP=78 "
                "BAQARO_PAPER_BESTFIT_NAME= BAQARO_PAPER_SUBSET_TAG= (and a low max_snap)."
            )
        # Canonical specific-cold-accretion filename (sim prefix + tokens) — the
        # same name_file_halos load_data_to_plot uses for halo_masses_*.npy.
        path_file_specific_cold_accretion = os.path.join(
            path_out, "halo_histories",
            f"specific_cold_accretion_rates_{name_file_halos}.npy",
        )
        print("Loading specific cold accretion rates from", path_file_specific_cold_accretion)
        # mmap, (n_snapshots, n_halos) C-order: arr[i] is a contiguous snapshot row.
        halo_specific_accretion_rates_all = np.load(
            path_file_specific_cold_accretion, mmap_mode="r"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        log_eta_array = np.linspace(-4, 2.0, 100)
        log_rate_bins = np.linspace(-2, 2, 21)
        bin_centers = (log_rate_bins[:-1] + log_rate_bins[1:]) / 2
        # Analytic ERDF curve per rate bin.
        #
        # The live model is parameterized by the specific cold accretion RATE —
        # exactly what this figure bins by — so the curve is just the lognormal
        # N(mu(rate), sigma(rate)) on log_eta_array.
        _mus, _sigmas = erdf.get_mu_sigma_arrays(bin_centers, dtype=np.float64)
        _mus = np.atleast_1d(_mus)
        _sigmas = np.broadcast_to(np.atleast_1d(_sigmas), _mus.shape)
        log_eta_distribution = (
            (1.0 / (_sigmas[:, None] * np.sqrt(2.0 * np.pi)))
            * np.exp(-0.5 * ((log_eta_array[None, :] - _mus[:, None]) / _sigmas[:, None]) ** 2)
        )   # shape (n_rate_bins, n_eta)

        for i in range(len(bin_centers)):
            if i % 2 == 0:
                continue
            print("Working on log specific accretion rate bin", bin_centers[i], "creating ERDF")

            log_etas_binned = np.array([])
            for index_snapshot in range(max_snap + 1):
                # Snapshot-major row (NOT arr[:, i] — the array is (n_snap, n_halo)).
                rate_row = halo_specific_accretion_rates_all[index_snapshot]
                with np.errstate(divide="ignore"):
                    log_rate_row = np.log10(rate_row)
                mask_rates = (log_rate_row >= log_rate_bins[i]) & (log_rate_row < log_rate_bins[i + 1])

                if np.sum(mask_rates) > 10:
                    Lbols_all = loader.get_Lbol(index_snapshot)
                    black_hole_masses_all = loader.get_BH_mass(index_snapshot)

                    etas = Lbols_all[mask_rates] / black_hole_masses_all[mask_rates] / 10**nc.log_csi
                    with np.errstate(divide="ignore"):
                        log_etas = np.log10(etas)
                    finite_mask = np.isfinite(log_etas)
                    log_etas_binned = np.concatenate((log_etas_binned, log_etas[finite_mask]))

            color = cmap_M((bin_centers[i] + 2) / 4)
            ax.hist(log_etas_binned, bins=100, density=True, histtype="step",
                    lw=2, log=True, color=color, ls="-")
            ax.plot(log_eta_array, log_eta_distribution[i, :], lw=2, color=color, ls=":")

        ax.axvline(np.log10(1.0), color="black", lw=1.5, ls="--", zorder=-10)

        # Legend proxies for the two line styles.
        ax.plot([], [], color="gray", lw=2, ls=":", label=r"sBHAR, $\eta_\mathrm{acc}$")
        ax.plot([], [], color="gray", lw=2, ls="-", label=r"Edd. ratio, $\lambda_\mathrm{Edd}$")
        ax.legend(loc="upper left")

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_M, norm=plt.Normalize(vmin=-2, vmax=2)),
            ax=ax,
        )
        cbar.set_label(
            r"$\log_{10}$ specific cold acc. rate, $s\dot{M}_\mathrm{cold,acc}$ [Gyr$^{-1}$]",
            labelpad=-1,
        )
        cbar.set_ticks(np.arange(-2, 2.0, 0.5))

        ax.set_xlabel(r"$\log_{10}\,\eta_\mathrm{acc}$, $\lambda_\mathrm{Edd}$")
        ax.set_ylabel(r"Probability distribution")
        ax.set_xlim(-4, 1.4)
        ax.set_ylim(1e-4, 8e0)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        fig.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.12)

        save_fig(fig, plot_config.FIGURES_DIR, name_fig_M, force_dir=True)

    maybe_show()
