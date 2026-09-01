"""
PAPER FIGURE: ERDF model visualisation.

Two figures, selected by the module-level flags below:

  * ``erdfs_theory_z``  (``plot_model_redshift``):
    Eddington-ratio structure of the evolved BH population with the
    NON-ACCRETING population included in the accounting (reworked
    2026-09-01; the old version — per-z conditional PDFs of the accreting
    subset plus a separate ``erdfs_theory_z_summary`` figure — silently
    dropped the lambda_Edd = 0 majority and is recoverable via git).
    Top panel: cumulative occupation fraction F(>lambda_Edd) over all
    established BHs at 9 redshifts — the gap between each curve's
    low-lambda plateau and the F=1 line IS the non-accreting population
    (77% of BHs at z=0). Bottom panel: stacked-area partition of the BH
    population by activity class (lambda>0.1 / 0<lambda<0.1 /
    non-accreting) vs redshift, densely sampled.

    Accounting choices (all measured on the K22 fiducial, 2026-09-01):
      - DENOMINATOR = "established" BHs: alive (M_BH > 0) and past their
        seeding snapshot. A BH seeded at snapshot i only enters the
        accretion step at i+1 (``main_evolution.py`` STEP 1 vs STEP 2), so
        at birth L_bol = 0 BY CONSTRUCTION (f_off|newborn == 1.000
        exactly). Newborns are 33% of BHs at z=8.7 but <1% at z<1;
        counting them fakes an activity downturn at z>6.
      - lambda_Edd = 0 is exact for plateaued (rate==0) halos — the
        non-accreting default of the forward model.
      - Snapshot targets avoid ``ANOMALY_SNAPS`` (FLAMINGO round-z
        inserts): their rate-window lookback differs from neighbours,
        biasing the zero-rate fraction by 0.03-0.08.
      - The stacked-band boundaries get a short rolling mean
        (``BAQARO_ERDF_THEORY_SMOOTH``, default 3; 1 disables) to iron out
        the residual lookback-quantization jitter between natural snaps.

    The single data pass (~45 snapshots, ~10-15 min on the K22 subsample)
    is cached per-run at ``{path_out}/erdf_theory/erdf_theory_cache_
    {name_file}.npz``; ``BAQARO_ERDF_THEORY_RECOMPUTE=1`` rebuilds.
    NEEDS A RUN WITH ALL SNAPSHOTS STORED (snap i-1 reads identify
    newborns), i.e. the K22 subsample fiducial — render with
    ``BAQARO_PAPER_SUBSET_TAG=root144_flatN500000_K22_logM10.0to15.5_seed42_v3``
    (no notes override; the on-disk K22 run carries no notes token).

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
HDF5 (via ``fiducial_data``). All population fractions and histograms apply
the Horvitz–Thompson ``loader.weights`` to stay unbiased on the subsampled
fiducial (a no-op in full-sim mode, where ``loader.weights is None``).

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
    name_file,
    name_file_halos,
    subset_tag,
    load_simulation_metadata,
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

plot_model_redshift = True    # occupation-fraction figure (top+bottom panel)
plot_model_halo_rate = False  # ERDF binned by cold specific halo accretion rate

# One cumulative F(>lambda) curve per target redshift (nearest natural snap).
CURVE_REDSHIFT_TARGETS = [8.7, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0]

# Bottom panel: dense sampling — every 3rd natural snapshot below this z.
Z_DENSE_MAX = 9.2

# FLAMINGO round-z insert snaps + small-dt companions (L2800N10080; root
# CLAUDE.md "anomaly snaps"). Their accretion-rate window walks back a
# different number of snapshots than their neighbours', which biases the
# zero-rate (non-accreting) fraction by 0.03-0.08 — skip them.
ANOMALY_SNAPS = {59, 65, 71, 72, 79, 82, 91, 100, 108, 116, 123, 133, 138, 139}

# log10 lambda_Edd histogram grid for the cumulative curves (0.05 dex bins).
BIN_EDGES = np.linspace(-6.0, 1.5, 151)

# Rolling-mean window for the stacked-band boundaries (odd; 1 = no smoothing).
SMOOTH_WINDOW = max(1, int(os.environ.get("BAQARO_ERDF_THEORY_SMOOTH", "3")))
RECOMPUTE_CACHE = os.environ.get("BAQARO_ERDF_THEORY_RECOMPUTE", "0") == "1"


def _build_erdf_theory_cache(loader, redshifts, snapshots, cache_path):
    """One pass over the run: per-snapshot activity fractions (dense z grid)
    + log10 lambda_Edd histograms (curve snaps), HT-weighted, established-BH
    denominator. Saved as an .npz keyed by the run's name_file."""
    w = loader.weights

    def natural_snap(tz):
        i = snapshot_index_for_redshift(redshifts, tz, snapshots, label="erdf_model")
        if i in ANOMALY_SNAPS:
            cands = [j for j in (i - 2, i - 1, i + 1, i + 2)
                     if 1 <= j <= max_snap and j not in ANOMALY_SNAPS]
            if cands:
                i = min(cands, key=lambda j: abs(redshifts[j] - tz))
        return i

    curve_snaps = []
    for tz in CURVE_REDSHIFT_TARGETS:
        i = natural_snap(tz)
        if i >= 1 and i not in curve_snaps:
            curve_snaps.append(i)
    dense_snaps = [i for i in range(3, max_snap + 1, 3)
                   if i not in ANOMALY_SNAPS and redshifts[i] <= Z_DENSE_MAX]
    all_snaps = sorted(set(dense_snaps) | set(curve_snaps))
    print(f"[erdf_theory] cache miss -> single pass over {len(all_snaps)} "
          f"snapshots (curves at {curve_snaps})", flush=True)

    recs = {k: [] for k in ["snap", "z", "n_est", "n_acc", "n001", "n01"]}
    hist_snaps, hists, unders, overs = [], [], [], []

    for i in all_snaps:
        print(f"[erdf_theory] snapshot {i}  z={redshifts[i]:.3f}", flush=True)
        L = loader.get_Lbol(i)
        M = loader.get_BH_mass(i)
        try:
            M_prev = loader.get_BH_mass(i - 1)
        except Exception as e:
            raise RuntimeError(
                f"erdfs_theory_z needs snapshot {i - 1} stored to identify "
                "newborn BHs (established-BH denominator). Use an "
                "all-snapshot run — the K22 subsample fiducial: "
                "BAQARO_PAPER_SUBSET_TAG=root144_flatN500000_K22_"
                "logM10.0to15.5_seed42_v3") from e

        # Established BHs: alive AND past the seeding snapshot (newborns have
        # L_bol = 0 by construction, so they never enter the accreting set).
        est = (M > 0) & (M_prev > 0)
        acc = est & (L > 0)
        lam_acc = L[acc] / M[acc] / 10**nc.log_csi
        w_acc = w[acc] if w is not None else None

        def wsum(mask, wa):
            return float(np.sum(wa[mask])) if wa is not None else float(np.count_nonzero(mask))

        recs["snap"].append(i)
        recs["z"].append(float(redshifts[i]))
        recs["n_est"].append(float(np.sum(w[est])) if w is not None
                             else float(np.count_nonzero(est)))
        recs["n_acc"].append(wsum(np.ones(lam_acc.shape, dtype=bool), w_acc)
                             if lam_acc.size else 0.0)
        recs["n001"].append(wsum(lam_acc > 0.01, w_acc))
        recs["n01"].append(wsum(lam_acc > 0.1, w_acc))

        if i in curve_snaps:
            with np.errstate(divide="ignore"):
                log_lam = np.log10(lam_acc)
            hist, _ = np.histogram(log_lam, bins=BIN_EDGES, weights=w_acc)
            hist_snaps.append(i)
            hists.append(hist)
            unders.append(wsum(lam_acc < 10.0**BIN_EDGES[0], w_acc))
            overs.append(wsum(lam_acc >= 10.0**BIN_EDGES[-1], w_acc))
        del L, M, M_prev, est, acc, lam_acc, w_acc

    np.savez(
        cache_path,
        bin_edges=BIN_EDGES,
        **{k: np.array(v) for k, v in recs.items()},
        hist_snaps=np.array(hist_snaps),
        hist=np.array(hists),
        under=np.array(unders),
        over=np.array(overs),
    )
    print(f"[erdf_theory] cached -> {cache_path}", flush=True)


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
    # FIGURE 1: occupation fraction F(>lambda_Edd) + activity partition vs z.
    # --------------------------------------------------------------------------
    if plot_model_redshift:
        cache_dir = os.path.join(path_out, "erdf_theory")
        cache_path = os.path.join(cache_dir, f"erdf_theory_cache_{name_file}.npz")
        if RECOMPUTE_CACHE or not os.path.exists(cache_path):
            os.makedirs(cache_dir, exist_ok=True)
            _build_erdf_theory_cache(loader, redshifts, snapshots, cache_path)
        else:
            print(f"[erdf_theory] using cache {cache_path}")
        c = np.load(cache_path)

        edges = c["bin_edges"]
        snap_arr = c["snap"]
        z_arr = c["z"]
        n_est = c["n_est"]
        f_acc = c["n_acc"] / n_est
        f01 = c["n01"] / n_est
        hist_snaps = c["hist_snaps"]
        hist = c["hist"]
        over = c["over"]

        order = np.argsort(z_arr)  # ascending z for the stacked bands
        zs, f_accs, f01s = z_arr[order], f_acc[order], f01[order]

        curve_z = np.array([z_arr[snap_arr == s][0] for s in hist_snaps])
        curve_f_acc = np.array([f_acc[snap_arr == s][0] for s in hist_snaps])
        z_lo, z_hi = 0.0, curve_z.max()

        def _z_color(z):
            return cmap_z((z - z_lo) / (z_hi - z_lo)) if z_hi > z_lo else cmap_z(0.5)

        # Per-curve quantitative log (established-BH denominator).
        print("\nBH activity summary (established BHs, HT-weighted)")
        header = f"{'z':>6} | {'N_est(w)':>11} | {'f_off':>6} {'f>0.01':>7} {'f>0.1':>6}"
        print("  " + header)
        print("  " + "-" * len(header))
        for s in hist_snaps:
            k = snap_arr == s
            print(f"  {z_arr[k][0]:6.2f} | {n_est[k][0]:11.4e} | "
                  f"{1 - f_acc[k][0]:6.3f} {c['n001'][k][0] / n_est[k][0]:7.3f} "
                  f"{f01[k][0]:6.3f}")

        X_LO, X_HI = -4.0, 1.4

        fig = plt.figure(figsize=(6, 4.9))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 0.7], hspace=0.27,
                              left=0.10, right=0.85, top=0.98, bottom=0.105)
        ax = fig.add_subplot(gs[0])
        axb = fig.add_subplot(gs[1])

        # ---------------- top panel: cumulative occupation fraction ----------
        draw = np.argsort(curve_z)[::-1]  # high z first, low z on top
        for k in draw:
            n_k = n_est[snap_arr == hist_snaps[k]][0]
            cum = (np.cumsum(hist[k][::-1])[::-1] + over[k]) / n_k
            m = edges[:-1] >= X_LO
            ax.plot(edges[:-1][m], cum[m], lw=2, color=_z_color(curve_z[k]))
            # left-edge marker at the curve's asymptote = the accreting fraction
            ax.plot(X_LO, curve_f_acc[k], marker="<", ms=6,
                    color=_z_color(curve_z[k]), clip_on=False, zorder=5)

        # reference lines, styled as in the other paper figures: full-height
        # black dashed at lambda_Edd=1, gray dotted at 0.1; thin line at F=1.
        ax.axhline(1.0, color="gray", lw=0.9, alpha=0.7, zorder=-10)
        ax.text(X_HI - 0.08, 0.90, "all BHs", ha="right", va="top", fontsize=10,
                color="gray")
        ax.axvline(0.0, color="black", lw=1.5, ls="--", zorder=-10)
        ax.axvline(-1.0, color="gray", lw=1.2, ls=":", zorder=-10)

        ax.set_yscale("log")
        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(1.5e-3, 1.3)
        ax.set_xlabel(r"$\log_{10}\,\lambda_\mathrm{Edd}$", labelpad=-0.5)
        ax.set_ylabel(r"$F(>\lambda_\mathrm{Edd})$  (fraction of all BHs)",
                      labelpad=-0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        # Redshift colorbar, identical convention to the other paper figures.
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_z,
                                  norm=plt.Normalize(vmin=z_lo, vmax=z_hi)),
            ax=ax,
        )
        cbar.set_label("Redshift")
        cbar.set_ticks(np.arange(z_lo, z_hi, 0.6))

        # save_fig crops to the ink bbox (bbox_inches="tight"), so extra
        # breathing room on the right edge needs an invisible glyph past the
        # colorbar label to stretch the crop box.
        fig.text(0.87, 0.5, ".", color="white", fontsize=6)
        fig.text(0.5, 0.03, ".", color="white", fontsize=6)

        # ---------------- bottom panel: stacked activity partition -----------
        COL_HI = "#08519c"    # lambda > 0.1
        COL_MID = "#6baed6"   # 0 < lambda < 0.1
        COL_OFF = "#cfcfcf"   # lambda = 0 (non-accreting)

        def _smooth(y):
            # short rolling mean, edge-preserving: irons out the +-0.03-0.08
            # snapshot-to-snapshot jitter from the discrete accretion-rate
            # window (lookback quantization) without touching the trend.
            if SMOOTH_WINDOW <= 1:
                return y
            pad = SMOOTH_WINDOW // 2
            yp = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
            return np.convolve(yp, np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW,
                               mode="valid")

        f01p, f_accp = _smooth(f01s), _smooth(f_accs)

        axb.fill_between(zs, 0, f01p, color=COL_HI, lw=0)
        axb.fill_between(zs, f01p, f_accp, color=COL_MID, lw=0)
        axb.fill_between(zs, f_accp, 1, color=COL_OFF, lw=0)
        for y in (f01p, f_accp):
            axb.plot(zs, y, color="white", lw=0.9)

        axb.text(3.6, 0.77, r"non-accreting ($\lambda_\mathrm{Edd}=0$)",
                 ha="center", va="center", fontsize=11, color="0.25")
        axb.text(6.5, 0.575, r"$0<\lambda_\mathrm{Edd}<0.1$",
                 ha="center", va="center", fontsize=10, color="white")
        axb.text(6.8, 0.17, r"$\lambda_\mathrm{Edd}>0.1$",
                 ha="center", va="center", fontsize=11, color="white")

        # tie-in markers: the redshifts of the curves in the top panel
        for zv in curve_z:
            axb.plot(zv, 1.0, marker="v", ms=4.5, color=_z_color(zv),
                     clip_on=False, zorder=5)

        axb.set_xlim(0, 8.75)
        axb.set_ylim(0, 1)
        axb.set_xlabel("Redshift", labelpad=-0.5)
        axb.set_ylabel("Fraction of BHs", labelpad=0)
        axb.set_yticks([0, 0.5, 1.0])
        axb.set_yticklabels(["0", "0.5", "1"])
        axb.yaxis.set_minor_locator(AutoMinorLocator(2))
        axb.xaxis.set_minor_locator(AutoMinorLocator())

        save_fig(fig, plot_config.FIGURES_DIR, name_fig_z, force_dir=True)

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
