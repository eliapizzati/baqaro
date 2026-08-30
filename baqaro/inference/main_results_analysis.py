"""
Unified MCMC results analysis script.

Produces corner plots and posterior predictive checks for the quasar luminosity
function (QLF), conditional Eddington ratio distribution function (CERDF), and
quasar clustering (auto- and cross-correlation functions).

Workflow
--------
1. Load GP emulators for whichever statistics are requested by PLOT_FLAGS.
2. Load a single emcee HDF5 chain whose filename is determined by LIKELIHOOD_FLAGS
   (i.e., which likelihoods were active when the chain was run).
3. Estimate burn-in and thinning from the chain's autocorrelation time.
4. Draw ``num_samples_to_plot`` posterior samples and push them through the
   emulators to get predicted statistics.
5. Plot each requested diagnostic:
   - **Corner**: marginal and joint posteriors with 16/50/84 quantiles.
   - **QLF**: per-redshift luminosity function with raw + binned observations.
   - **CERDF**: grid of (redshift x luminosity bin) panels comparing observed
     Eddington ratio histograms against model PDFs.
   - **Correlation**: projected auto-correlation (wp/rp) or volume-averaged
     cross-correlation, computed from the emulated quasar halo mass function
     (QHMF) via precomputed halo-model triangles.

Key design decisions
--------------------
* **PLOT_FLAGS vs LIKELIHOOD_FLAGS** — These are intentionally independent.
  PLOT_FLAGS controls which emulators to load and which figures to produce;
  LIKELIHOOD_FLAGS only affects the MCMC output filename so the correct chain
  is read.  This lets you, e.g., plot QLF predictions from a chain that was
  fit to QLF+CERDF jointly.

* **Autocorrelation-based burn-in** — burn_in = 2 * max(tau), thin = 0.5 *
  min(tau).  Falls back to the manual ``CONFIG["burn_in"]`` value if the chain
  is too short for a reliable autocorrelation estimate.

* **Emulator predict_mean_only** — We only need the GP posterior mean (not
  variance) for plotting, which is faster than full prediction.

Toggle which posterior predictive checks to display via PLOT_FLAGS.
Mirrors the naming convention from main_mcmc.py.
"""

# This module is a run script, not a library: its body loads catalogues and
# writes products at import time. Refuse a plain ``import`` so nobody starts a
# multi-hour job by accident (``python -m baqaro.inference.main_results_analysis`` sets __name__ to
# "__main__"; a multiprocessing "spawn" child re-imports it as "__mp_main__").
if __name__ not in ("__main__", "__mp_main__"):
    raise RuntimeError(
        "baqaro.inference.main_results_analysis is an entry-point script; run it with "
        "'python -m baqaro.inference.main_results_analysis' instead of importing it."
    )


import os
import emcee
import corner
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# --- PROJECT IMPORTS ---
from baqaro.emulation.loading_helpers import load_emulators
from baqaro.utils.my_dir import get_output_path
from baqaro.plotting_common.plot_config import z_scalar_mapper, fig_ext


# ==========================================
# CONFIGURATION
# ==========================================

# What to plot (independent of what likelihoods were used in the MCMC run).
# Each flag controls whether the corresponding emulator is loaded and whether
# the figure is produced.
# PLOT_FLAGS: what figures to produce. Override via BAQARO_PLOT_FLAGS, e.g.
#   BAQARO_PLOT_FLAGS=corner,qlf      # corner + QLF only
#   BAQARO_PLOT_FLAGS=corner          # just the corner plot
# Default keeps all four enabled.
_plot_env = os.environ.get("BAQARO_PLOT_FLAGS", "").strip()
if _plot_env:
    _enabled = {k.strip() for k in _plot_env.split(",") if k.strip()}
    PLOT_FLAGS = {k: (k in _enabled) for k in ("corner", "qlf", "cerdf", "corr")}
else:
    PLOT_FLAGS = {"corner": True, "qlf": True, "cerdf": True, "corr": True}

#: figure number -> short descriptive name, used to build the saved filename.
#: Populated as each figure is created (see the save loop at the bottom).
_FIG_NAMES = {}

# Which likelihoods were active in the MCMC run. Only used to reconstruct the
# chain filename — does NOT affect which plots are made. Override via
# BAQARO_LIKELIHOODS env var:
#   BAQARO_LIKELIHOODS=qlf                 # look up mcmc_..._qlf.h5
#   BAQARO_LIKELIHOODS=qlf,cerdf,corr      # look up mcmc_..._qlf+cerdf+corr.h5
_likelihoods_env = os.environ.get("BAQARO_LIKELIHOODS", "").strip()
if _likelihoods_env:
    _enabled = {k.strip() for k in _likelihoods_env.split(",") if k.strip()}
    LIKELIHOOD_FLAGS = {k: (k in _enabled) for k in ("qlf", "cerdf", "corr")}
else:
    LIKELIHOOD_FLAGS = {"qlf": True, "cerdf": True, "corr": True}

CONFIG = {
    "mcmc_filename_notes": None,        # optional extra tag (or None)
    "burn_in": 0,
    "num_samples_to_plot": 100,
}

# --- Model / emulator identification (must match main_mcmc.py) ---
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # site from the environment
source_dir = DEFAULT_SOURCE_DIR

from baqaro.utils.sim_config import FIDUCIAL_NOTES_EMULATION
notes_file_emulation = os.environ.get("BAQARO_NOTES_FILE_EMULATION") or FIDUCIAL_NOTES_EMULATION
min_snap = 0
erdf_model = "log_normal_evol_halo_mass"

from baqaro.utils.sim_config import boxsize, N_particles_per_side, simulation_name, max_snap, fold_subhalo_mass, merger_delay_mode


# --- Halo-data variant tokens (must match the emulator's training run) ---
# Subset tag (only when training used BAQARO_USE_SUBSAMPLE=1). Default below
# matches the L2800N10080 maxsnap=60 fid3_10k training run.
# Default to the production subsampled run's tag (auto-derived from
# sim_config geometry + subsample env defaults) so callers needn't set
# BAQARO_SUBSET_TAG. Env presence wins: BAQARO_SUBSET_TAG="" => full-sim.
if "BAQARO_SUBSET_TAG" in os.environ:
    subset_tag = os.environ["BAQARO_SUBSET_TAG"]
else:
    from baqaro.core_functions.tree_subsample import resolve_subset_tag
    subset_tag = resolve_subset_tag(max_snap)

name_file = "{}_erdf_{}_maxsnap_{}".format(simulation_name, erdf_model, max_snap)
if fold_subhalo_mass:
    name_file += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file += "_{}".format(merger_delay_mode)
if notes_file_emulation is not None:
    name_file += "_{}".format(notes_file_emulation)
if subset_tag:
    name_file += "_sub_{}".format(subset_tag)
# Growth-cap + madau-f_eff tokens (after _sub_, matching main_training /
# main_emulation / main_mcmc). Required so the emulator lookup here resolves
# the SAME _g{cap}/_feffcorr emulators the chain was run against — otherwise
# this analysis would load the uncapped / uncorrected emulators instead.
from baqaro.utils.sim_config import growth_feff_suffix
name_file += growth_feff_suffix()
# --- MCMC filename (derived from LIKELIHOOD_FLAGS, matches main_mcmc.py convention) ---
# The chain file is named after which likelihoods were toggled on during the
# MCMC run, joined by "+".  E.g. "mcmc_..._qlf+corr.h5".
# Notes suffix: env BAQARO_MCMC_NOTES wins (matches comparison_config.py so a
# re-fit set written with e.g. BAQARO_MCMC_NOTES=minL05 is found here), else
# the in-source CONFIG value.
mcmc_filename_notes = os.environ.get("BAQARO_MCMC_NOTES") or CONFIG["mcmc_filename_notes"]
active_likelihoods = "+".join(k for k in ("qlf", "cerdf", "corr") if LIKELIHOOD_FLAGS.get(k, False))
mcmc_filename = f"mcmc_{name_file}_{active_likelihoods}"
if mcmc_filename_notes is not None:
    mcmc_filename += f"_{mcmc_filename_notes}"


# ==========================================
# LOAD EMULATORS
# ==========================================

path_out = Path(get_output_path(source=source_dir))

# Map PLOT_FLAGS to the emulator keys expected by load_emulators.
# The correlation plot needs the QHMF emulator (not a "corr" emulator) because
# clustering is computed analytically from the quasar halo mass function via
# the halo-model triangle integral.
emulator_flags = {
    "qlf": PLOT_FLAGS["qlf"],
    "bhmf": False,
    "cerdf": PLOT_FLAGS["cerdf"],
    "qhmf": PLOT_FLAGS["corr"],
}

emulators = load_emulators(path_out, name_file, emulator_flags)

# All emulators share the same parameter names and ranges (they were trained
# on the same Latin hypercube), so grab them from whichever one loaded.
_any_emulator = next(v for v in emulators.values() if v is not None)
param_names = _any_emulator.param_names
param_ranges = _any_emulator.param_ranges
print(f"\nParameters ({len(param_names)}D): {param_names}")
print(f"Ranges: {param_ranges}")


# ==========================================
# LOAD MCMC CHAIN
# ==========================================

mcmc_path = path_out / "mcmc" / f"{mcmc_filename}.h5"
print(f"\nReading MCMC backend: {mcmc_path}")
if not mcmc_path.exists():
    raise FileNotFoundError(f"Could not find MCMC file at {mcmc_path}")

reader = emcee.backends.HDFBackend(str(mcmc_path), read_only=True)

# Autocorrelation-based burn-in: discard the first 2*max(tau) steps so all
# parameters have had time to decorrelate from their initial positions.
# Thin by 0.5*min(tau) to get approximately independent samples.
n_steps_chain = int(reader.iteration)
try:
    tau = reader.get_autocorr_time(quiet=True)
    if not np.all(np.isfinite(tau)):
        raise ValueError(f"non-finite autocorrelation time: {tau}")
    burnin = int(2 * np.max(tau))
    thin = max(1, int(0.5 * np.min(tau)))
    print(f"Autocorrelation time: {tau}")
    print(f"Using burn-in: {burnin}, thinning: {thin}")
except Exception as e:
    # Chain too short / tau non-finite — fall back to the manual value, but say
    # so LOUDLY: the chain is very likely unconverged and everything below
    # (corner, posterior predictives) is then being read off a burn-in-free,
    # unconverged sample.
    print("=" * 70)
    print(f"⚠ WARNING: autocorrelation estimate unusable ({e}).")
    print(f"⚠ Falling back to manual burn_in={CONFIG['burn_in']}, thin=1.")
    print("⚠ The chain is likely UNCONVERGED — treat all results below as "
          "provisional (run more steps).")
    print("=" * 70)
    burnin = CONFIG["burn_in"]
    thin = 1

# Guard the discard against the chain length: burnin >= n_steps would return an
# EMPTY chain and crash every downstream consumer with an opaque error.
if burnin >= n_steps_chain:
    _clamped = max(0, n_steps_chain // 2)
    print(f"⚠ WARNING: burn-in {burnin} >= chain length {n_steps_chain}; "
          f"clamping to {_clamped}. The chain is too short for this tau — "
          "results are provisional.")
    burnin = _clamped

flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)
if flat_samples.size == 0:
    raise RuntimeError(
        f"No posterior samples after burn-in={burnin}, thin={thin} on a chain of "
        f"{n_steps_chain} steps ({mcmc_path}). Run the chain longer, or lower "
        "CONFIG['burn_in']."
    )
print(f"Posterior shape: {flat_samples.shape}")


# ==========================================
# CORNER PLOT
# ==========================================

if PLOT_FLAGS["corner"]:
    print("\n--- Corner Plot ---")
    fig_corner = corner.corner(
        flat_samples,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        smooth=1.0,
        smooth1d=1.0,
        color="#0072C1",
    )
    _FIG_NAMES[fig_corner.number] = "corner"


# ==========================================
# QLF POSTERIOR PREDICTIVE CHECK
# ==========================================

if PLOT_FLAGS["qlf"]:
    from baqaro.obs_data.qlf_obs_data import (
        data_qlf_global_raw, data_qlf_global_sys_err_binned,
    )

    print("\n--- QLF Posterior Predictive Check ---")
    qlf_emulator = emulators["qlf"]
    num_samples = CONFIG["num_samples_to_plot"]

    # Draw random posterior samples and predict QLFs for all of them at once.
    # pred_qlfs shape: (num_samples, n_redshifts, n_bins)
    indices = np.random.randint(len(flat_samples), size=num_samples)
    subset_params = flat_samples[indices]
    pred_qlfs = qlf_emulator.predict_mean_only(subset_params)

    # Redshifts we want to show panels for.  Not all may exist in the emulator
    # grid, so we match with a 0.5 tolerance below.
    redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    emulator_z = np.asarray(qlf_emulator.axis_data["redshift"], dtype=float)

    # Match each desired redshift to the closest emulator redshift.
    # If no emulator redshift is within 0.5, mark as None (panel will show
    # data only, no model curves).
    matched_emul_idx = []
    for z in redshift_keys:
        idx = int(np.argmin(np.abs(emulator_z - z)))
        matched_emul_idx.append(idx if abs(z - emulator_z[idx]) < 0.5 else None)

    n_z = len(redshift_keys)
    nrows, ncols = 2, int(np.ceil(n_z / 2))
    colors = z_scalar_mapper.to_rgba(redshift_keys)

    fig_qlf, axes_qlf = plt.subplots(nrows, ncols, figsize=(3.75 * ncols, 3.4 * nrows),
                                      sharex=True, sharey=True)
    _FIG_NAMES[fig_qlf.number] = "qlf"
    axes_qlf = axes_qlf.flatten()

    for i, (z_key, color, emul_idx) in enumerate(zip(redshift_keys, colors, matched_emul_idx)):
        ax = axes_qlf[i]
        z_str = f"{z_key:.1f}"

        # --- Observational data (comparison layout) ---
        # Both sets are BLACK and separated by MARKER, not colour: small faded
        # circles = the individual survey measurements, larger solid squares =
        # the rebinned points the likelihood actually fits. The previous
        # black-vs-RED same-marker styling read as two conflicting datasets
        # rather than as one dataset and its binning.
        if z_str in data_qlf_global_raw:
            obs = data_qlf_global_raw[z_str]
            ax.errorbar(obs.x, obs.data, yerr=obs.err, fmt='o', color='black', ms=4,
                        capsize=2, elinewidth=0.8, alpha=0.25, zorder=0,
                        label="Surveys")
        if z_str in data_qlf_global_sys_err_binned:
            obs_b = data_qlf_global_sys_err_binned[z_str]
            ax.errorbar(obs_b.x, obs_b.data, yerr=obs_b.err, fmt='s', color='black', ms=6,
                        capsize=3, elinewidth=1.5, alpha=0.85, zorder=11,
                        label="Binned (fitted)")

        # --- Model posterior predictive ---
        if emul_idx is not None:
            log_bins = qlf_emulator.axis_data['log_bins']
            # Individual posterior draws (transparent spaghetti)
            for j in range(num_samples):
                ax.plot(log_bins, pred_qlfs[j, emul_idx, :], color=color, alpha=0.05, lw=1.5, zorder=1)

            # Summary statistics: median and 16-84 percentile band
            median_pred = np.median(pred_qlfs[:, emul_idx, :], axis=0)
            low_pred = np.percentile(pred_qlfs[:, emul_idx, :], 16, axis=0)
            high_pred = np.percentile(pred_qlfs[:, emul_idx, :], 84, axis=0)

            ax.plot(log_bins, median_pred, color=color, lw=2.5, zorder=5,
                    label=f"Model (z={z_key:.1f})")
            ax.fill_between(log_bins, low_pred, high_pred, color=color, alpha=0.2, zorder=2)

        ax.text(0.95, 0.95, f"z = {z_key:.1f}", transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

        # Only label outer edges to avoid clutter in the shared-axis grid
        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1)
        if is_left:
            ax.set_ylabel(r"$\log_{10} \Phi$ [dex$^{-1}$ cMpc$^{-3}$]")
        else:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\log_{10}$ L$_\mathrm{bol}$ [erg/s]")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_ylim(-9.8, -2.5)
        ax.set_xlim(43.5, 48.5)
        if i == 0:
            ax.legend(loc='lower left', fontsize=9)

    # Hide any unused subplots (when n_z is not a multiple of ncols)
    for j in range(i + 1, len(axes_qlf)):
        axes_qlf[j].axis('off')
    fig_qlf.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.12, hspace=0.05, wspace=0.05)


# ==========================================
# CERDF POSTERIOR PREDICTIVE CHECK
# ==========================================

if PLOT_FLAGS["cerdf"]:
    from qhtools.utils import my_utils
    import qhtools.utils.natconst as nc
    from baqaro.obs_data.qso_obs_data_setup import (
        redshifts_data, logL_Bols_data, logM_BHs_data,
    )

    print("\n--- CERDF Posterior Predictive Check ---")
    cerdf_emulator = emulators["cerdf"]
    num_samples = CONFIG["num_samples_to_plot"]

    indices = np.random.randint(len(flat_samples), size=num_samples)
    subset_params = flat_samples[indices]
    # pred_cerdfs shape: (num_samples, n_redshifts, n_L_thresholds, n_bins)
    pred_cerdfs = cerdf_emulator.predict_mean_only(subset_params)

    log_bins_cerdf = cerdf_emulator.axis_data["log_bins"]
    log_L_thresholds = cerdf_emulator.axis_data["log_L_threshold"]
    emul_z_cerdf = np.asarray(cerdf_emulator.axis_data["redshift"], dtype=float)

    redshift_keys_cerdf = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    # Luminosity bins for slicing the observed QSO sample.  Each panel shows
    # the ERDF for QSOs in a narrow L_bol range, enabling comparison of the
    # Eddington ratio distribution at fixed luminosity.
    logL_bins = np.linspace(45.5, 47.5, 5)
    logL_bin_centers = 0.5 * (logL_bins[1:] + logL_bins[:-1])
    n_L_bins = len(logL_bin_centers)

    # Convert observed (L_bol, M_BH) to Eddington ratio:
    # log(lambda_Edd) = log(L_bol/L_sun) - log(M_BH/M_sun) - log(c_Edd)
    # where c_Edd = L_Edd / M_BH (in solar units).
    log_etas_data = my_utils.to_solar(logL_Bols_data) - logM_BHs_data - nc.log_csi

    n_z_cerdf = len(redshift_keys_cerdf)
    # Color-code panels by luminosity bin using a discrete colormap
    cmap_L = matplotlib.cm.viridis
    norm_L = matplotlib.colors.BoundaryNorm(boundaries=logL_bins, ncolors=cmap_L.N)

    fig_cerdf, axes_cerdf = plt.subplots(
        n_z_cerdf, n_L_bins,
        figsize=(3.75 * n_L_bins, 3.5 * n_z_cerdf),
        sharex=True, sharey=True,
    )
    _FIG_NAMES[fig_cerdf.number] = "cerdf"
    if n_z_cerdf == 1:
        axes_cerdf = axes_cerdf[np.newaxis, :]
    if n_L_bins == 1:
        axes_cerdf = axes_cerdf[:, np.newaxis]

    # Bins for the observed Eddington ratio histogram
    bins_data = np.linspace(-3, 1.5, 25)
    bins_data_centers = 0.5 * (bins_data[1:] + bins_data[:-1])

    # Gaussian smoothing kernel width in bin units.  The 0.3 dex scatter
    # represents observational BH mass uncertainty that broadens the intrinsic
    # ERDF.  We convolve the model PDF with this kernel so the comparison to
    # observed histograms is apples-to-apples.
    bin_width_cerdf = np.abs(np.diff(log_bins_cerdf[:2])[0])
    sigma_bins = 0.3 / bin_width_cerdf

    for i_z, redshift in enumerate(redshift_keys_cerdf):
        emul_z_idx = int(np.abs(emul_z_cerdf - redshift).argmin())

        for i_L in range(n_L_bins):
            ax = axes_cerdf[i_z, i_L]
            line_color = cmap_L(norm_L(logL_bin_centers[i_L]))

            # Find the emulator luminosity threshold indices bracketing this
            # L_bol bin.  The CERDF is cumulative (N(>L) vs lambda), so the
            # differential ERDF in [Lmin, Lmax] = CERDF(Lmin) - CERDF(Lmax).
            lmin_val, lmax_val = logL_bins[i_L], logL_bins[i_L + 1]
            lmin_idx = int(np.abs(log_L_thresholds - lmin_val).argmin())
            lmax_idx = int(np.abs(log_L_thresholds - lmax_val).argmin())

            # --- Observational histogram ---
            # Select QSOs in this (redshift, luminosity) cell
            mask_z = (redshifts_data < redshift + 0.5) & (redshifts_data > redshift - 0.5)
            mask_L = (logL_Bols_data > lmin_val) & (logL_Bols_data < lmax_val)
            mask = mask_z & mask_L
            if np.sum(mask) > 3:  # need at least a few QSOs for a meaningful histogram
                log_eta_obs = log_etas_data[mask]
                counts_obs, _ = np.histogram(log_eta_obs, bins=bins_data, density=True)
                ax.step(bins_data_centers, counts_obs, where='mid', lw=2.5,
                        alpha=0.7, color=line_color, zorder=10, label="Obs data")
                # Dotted vertical: observed median Eddington ratio
                ax.axvline(np.median(log_eta_obs), lw=2, ls=':', alpha=0.7, color=line_color, zorder=10)

            # --- Posterior predictive model PDFs ---
            all_pdfs = []
            for j in range(num_samples):
                # Differential CERDF: number of QSOs with L > Lmin minus those
                # with L > Lmax, as a function of Eddington ratio.  This gives
                # the ERDF for the luminosity slice [Lmin, Lmax].
                cerdf_lmin = 10 ** pred_cerdfs[j, emul_z_idx, lmin_idx, :]
                cerdf_lmax = 10 ** pred_cerdfs[j, emul_z_idx, lmax_idx, :]
                cerdf_diff = np.clip(cerdf_lmin - cerdf_lmax, 0.0, None)
                integral = np.trapezoid(cerdf_diff, log_bins_cerdf)
                if integral > 0:
                    # Normalize to a PDF, then convolve with BH mass uncertainty
                    pdf = cerdf_diff / integral
                    pdf = gaussian_filter1d(pdf, sigma_bins, mode='constant', cval=0.0)
                    pdf /= np.trapezoid(pdf, log_bins_cerdf)  # re-normalize after convolution
                else:
                    pdf = np.zeros_like(cerdf_diff)
                all_pdfs.append(pdf)
                ax.plot(log_bins_cerdf, pdf, color=line_color, alpha=0.03, lw=0.8, zorder=1)

            all_pdfs = np.array(all_pdfs)
            valid = np.any(all_pdfs > 0, axis=1)  # exclude all-zero samples
            if np.sum(valid) > 0:
                median_pdf = np.median(all_pdfs[valid], axis=0)
                low_pdf = np.percentile(all_pdfs[valid], 16, axis=0)
                high_pdf = np.percentile(all_pdfs[valid], 84, axis=0)
                ax.plot(log_bins_cerdf, median_pdf, color=line_color, lw=2.5, zorder=5, label="Model")
                ax.fill_between(log_bins_cerdf, low_pdf, high_pdf, color=line_color, alpha=0.15, zorder=2)

                # Dashed vertical: model median Eddington ratio, obtained by
                # inverting the CDF of the median PDF at the 0.5 quantile.
                cdf = np.cumsum(median_pdf[:-1] * np.diff(log_bins_cerdf))
                if cdf[-1] > 0:
                    cdf_norm = np.concatenate(([0.0], cdf / cdf[-1]))
                    median_emul = np.interp(0.5, cdf_norm, log_bins_cerdf)
                    ax.axvline(median_emul, lw=2, ls='--', alpha=0.8, color=line_color, zorder=5)

            if i_z == 0:
                ax.set_title(rf"$\log L = [{lmin_val:.1f}, {lmax_val:.1f}]$", fontsize=10)
            ax.text(0.95, 0.95, f"z = {redshift:.1f}", transform=ax.transAxes, fontsize=10,
                    fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))

            if i_L == 0:
                ax.set_ylabel(r"Probability density")
            else:
                ax.tick_params(labelleft=False)
            if i_z == n_z_cerdf - 1:
                ax.set_xlabel(r"$\log_{10}$ $\lambda_{\rm Edd}$")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xlim(-3, 1.5)
            ax.set_ylim(0, 2.0)
            if i_z == 0 and i_L == 0:
                ax.legend(loc='upper left', fontsize=8)

    fig_cerdf.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, hspace=0.05, wspace=0.05)


# ==========================================
# CORRELATION POSTERIOR PREDICTIVE CHECK
# ==========================================

if PLOT_FLAGS["corr"]:
    from qhtools.clustering.qhmf_to_corr import (
        get_corr_from_triangle, get_corr_from_triangle_cross,
    )
    from qhtools.clustering.projected_correlation_functions import (
        get_projected_wp, get_volume_averaged_xi,
    )
    # The corr datasets MUST mirror main_mcmc.py's selection, or the
    # posterior-predictive panel overlays different observations than the ones the
    # chain was actually fit to.
    #
    # They are read from the same env vars with the same defaults.
    from baqaro.obs_data import corr_obs_data as _corr_mod
    from baqaro.obs_data.corr_obs_data import (
        data_ef, data_ef_ext_restricted, data_shen_highz, data_aspire, data_eiger,
    )
    _z25_key = (os.environ.get("BAQARO_CORR_Z25", "data_ef_ext_restricted").strip()
                or "data_ef_ext_restricted")
    # z=4 default = ALL-FIELDS Shen+07 with x2-inflated errors (matching
    # main_mcmc). Revert with BAQARO_CORR_Z4=data_shen_highz.
    _z4_key = (os.environ.get("BAQARO_CORR_Z4", "data_shen_highz_allfields_err2").strip()
               or "data_shen_highz_allfields_err2")
    # z=6 default is data_aspire_cov (same 8 points and same .err
    # as data_aspire — only .covariance differs — so the PLOT is identical; this
    # just keeps the provenance honest).
    _z6_key = os.environ.get("BAQARO_CORR_Z6", "data_aspire_cov").strip() or "data_aspire_cov"
    _corr_z25 = getattr(_corr_mod, _z25_key)
    _corr_z4 = getattr(_corr_mod, _z4_key)
    _corr_z6 = getattr(_corr_mod, _z6_key)

    # --- DISPLAY vs FITTED at z=2.5 -----------------------------------------
    # The z=2.5 LIKELIHOOD uses `data_ef_ext_restricted` — EF15 Table 3 masked to
    # EF15's own fit range, 4 < rp < 25 h^-1 Mpc (10 of the 18 bins). But a figure
    # should show the DATA, not just the part of it we chose to fit: plot the full
    # `data_ef_ext` (18 bins, rp = 3.7-94.6) so the reader can see how the model
    # does outside the fitted window too. This also extends the posterior-predictive
    # model curve across the full rp range, since the curve is evaluated on the
    # displayed dataset's rp grid.
    #
    # Safe because the two share pimax (73.42) and log_L_threshold (12.498) — same
    # sample, the restricted one is a pure row-mask of the other — so the halo-model
    # inputs are identical and only the rp grid grows.
    #
    # NB no chi2 is computed in this block, so widening the displayed set cannot
    # silently change any reported goodness-of-fit. Override with BAQARO_CORR_Z25_PLOT.
    _z25_plot_key = os.environ.get("BAQARO_CORR_Z25_PLOT", "").strip() or (
        "data_ef_ext" if _z25_key == "data_ef_ext_restricted" else _z25_key)
    _corr_z25_plot = getattr(_corr_mod, _z25_plot_key)
    print(f"[corr] posterior-predictive datasets: z=2.5={_z25_key}  "
          f"z=4={_z4_key}  z=6={_z6_key}")
    if _z25_plot_key != _z25_key:
        print(f"[corr] z=2.5 PLOTTED as {_z25_plot_key} ({len(_corr_z25_plot.x)} bins, "
              f"rp {_corr_z25_plot.x.min():.1f}-{_corr_z25_plot.x.max():.1f}); "
              f"the FIT used {_z25_key} ({len(_corr_z25.x)} bins, "
              f"rp {_corr_z25.x.min():.1f}-{_corr_z25.x.max():.1f})")
    from baqaro.inference.likelihoods_and_priors import precompute_corr_inputs

    print("\n--- Correlation Posterior Predictive Check ---")
    qhmf_emulator = emulators["qhmf"]
    num_samples = CONFIG["num_samples_to_plot"]

    # Monkey-patch correlation metadata onto data objects.  The imported data
    # objects carry (x, data, err, pimax) but not the redshift / correlation
    # type / halo mass range needed by the halo-model machinery.  These are the
    # SELECTED objects, so they match whatever the chain was fit to.
    _corr_z25_plot.redshift = 2.5
    _corr_z25_plot.corr_type = "auto"
    _corr_z25_plot.logM_min = 11.5
    _corr_z25_plot.logM_max = 14.5

    _corr_z4.redshift = 4.0
    _corr_z4.corr_type = "auto"
    _corr_z4.logM_min = 11.5
    _corr_z4.logM_max = 14.5

    # z~6 cross: the selected ASPIRE variant drives the model (cached z=6.1
    # halo-model inputs; emulator's nearest slice is z=6.14); EIGER overlaid in
    # the same panel as a second measurement. Both span to lower halo masses to
    # include the galaxy population.
    for _d in (_corr_z6, data_eiger):
        _d.redshift = 6.1
        _d.corr_type = "cross"
        _d.logM_min = 10.5
        _d.logM_max = 14.0

    # z=2.5 entry is the DISPLAY dataset (full rp range); see the note above.
    corr_datasets = {2.5: _corr_z25_plot, 4.0: _corr_z4, 6.1: _corr_z6}
    extra_obs = {id(_corr_z6): [data_eiger]}
    # rp window actually fitted at z=2.5, so the panel can shade it.
    fitted_rp_range = {id(_corr_z25_plot): (float(_corr_z25.x.min()),
                                            float(_corr_z25.x.max()))} \
        if _z25_plot_key != _z25_key else {}

    # Match emulator redshifts to available correlation datasets (tolerance 0.5)
    target_z_corr = qhmf_emulator.axis_data["redshift"]
    datas_corr = []
    target_z_corr = np.asarray(target_z_corr, dtype=float)
    for _i, tz in enumerate(target_z_corr):
        best_key = min(corr_datasets, key=lambda z: abs(z - tz))
        # Uniqueness guard (back-ported from plotting_paper/plotting_results_comparison):
        # a dataset binds to exactly ONE panel -- the emulator slice CLOSEST to it.
        # Otherwise a dataset with two nearby slices (z=4 sits between emulator
        # z=3.937, gap 0.063, and z=3.534, gap 0.466) is drawn twice AND the panel
        # cap silently drops a real one (z=6.1).
        _closest = int(np.argmin(np.abs(target_z_corr - best_key)))
        if abs(best_key - tz) < 0.5 and _closest == _i:
            datas_corr.append(corr_datasets[best_key])
            print(f"  z={tz:.2f} -> corr at z={best_key}")
        else:
            datas_corr.append(None)
            print(f"  z={tz:.2f} -> no corr data")

    # Precompute halo-model triangle grids, halo mass functions, and radial
    # bin arrays for each redshift.  This is expensive (~seconds) so done once
    # outside the sample loop.
    print("Precomputing correlation inputs...")
    precomputed_inputs = precompute_corr_inputs(datas_corr)

    log_L_thresholds_corr = qhmf_emulator.axis_data["log_L_threshold"]
    log_mbins_qhmf = qhmf_emulator.axis_data["log_bins"]

    indices = np.random.randint(len(flat_samples), size=num_samples)
    subset_params = flat_samples[indices]
    # pred_qhmfs shape: (num_samples, n_redshifts, n_L_thresholds, n_bins)
    pred_qhmfs = qhmf_emulator.predict_mean_only(subset_params)

    # Only plot panels for redshifts that have matching correlation data
    active_indices = [i for i, d in enumerate(datas_corr) if d is not None]
    # Emulator z axis is DESCENDING; the paper figures order panels low-z -> high-z.
    active_indices.sort(key=lambda i: datas_corr[i].redshift)
    n_panels = len(active_indices)

    if n_panels > 0:
        ncols_corr = min(n_panels, 3)
        nrows_corr = int(np.ceil(n_panels / ncols_corr))

        fig_corr, axes_corr = plt.subplots(
            nrows_corr, ncols_corr,
            figsize=(3.75 * ncols_corr, 3.5 * nrows_corr),
            sharex=True, sharey=True, squeeze=False,
        )
        _FIG_NAMES[fig_corr.number] = "corr"
        axes_corr = axes_corr.flatten()

        for panel_idx, i_z in enumerate(active_indices):
            ax = axes_corr[panel_idx]
            data_corr = datas_corr[i_z]
            inputs = precomputed_inputs[i_z]
            z_val = data_corr.redshift
            corr_type = getattr(data_corr, "corr_type", "auto")
            color = z_scalar_mapper.to_rgba(z_val)

            # Shade the rp window that was actually FITTED, when the panel shows
            # more data than the likelihood used (z=2.5: EF15's own 4 < rp < 25
            # h^-1 Mpc fit range). Without this the figure implies every plotted
            # point constrained the model, which is not true.
            _fit_lo_hi = fitted_rp_range.get(id(data_corr))
            if _fit_lo_hi is not None:
                ax.axvspan(_fit_lo_hi[0], _fit_lo_hi[1], color="grey", alpha=0.12,
                           zorder=0, label="fitted range")

            # Plot observed correlation function
            ax.errorbar(data_corr.x, data_corr.data, yerr=data_corr.err,
                        fmt="o", color="black", capsize=3, markersize=6,
                        elinewidth=1.5, alpha=0.8, zorder=10,
                        label=getattr(data_corr, "label", "Data"))
            # Extra measurements overlaid in the same panel (e.g. ASPIRE next to
            # EIGER): distinct open markers, data only.
            for _k, _ex in enumerate(extra_obs.get(id(data_corr), [])):
                ax.errorbar(_ex.x, _ex.data, yerr=_ex.err, fmt=["s", "D", "^"][_k % 3],
                            color="black", mfc="none", capsize=3, markersize=6,
                            elinewidth=1.2, alpha=0.85, zorder=10,
                            label=getattr(_ex, "label", "Data2"))

            log_m_axis = inputs["log_m_axis"]
            rbins = inputs["rbins"]
            rpbins = inputs["rpbins"]
            mf_fit = inputs["mf_fit"]
            triangle_fit = inputs["triangle_fit"]
            # Find which luminosity threshold in the emulator grid matches
            # the one used for this correlation dataset
            threshold_idx = np.abs(log_L_thresholds_corr - inputs["log_L_threshold"]).argmin()

            all_y = []
            all_x = []
            for j in range(num_samples):
                log_qhmf_slice = pred_qhmfs[j, i_z, threshold_idx, :]
                # Skip samples where the QHMF is entirely floored (no QSOs
                # predicted above the luminosity threshold)
                if np.all(log_qhmf_slice <= -9.5):
                    continue
                # Interpolate emulator QHMF onto the halo-model mass axis
                log_qhmf_interp = np.interp(log_m_axis, log_mbins_qhmf, log_qhmf_slice,
                                            left=-10.0, right=-10.0)

                if corr_type == "auto":
                    # Auto-correlation: QHMF -> 3D xi(r) via triangle integral
                    # -> projected wp(rp) via line-of-sight integration
                    log_rbins, xi = get_corr_from_triangle(
                        log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                        triangle=triangle_fit, log_m_axis=log_m_axis, qhmf=10 ** log_qhmf_interp)
                    wp = get_projected_wp(
                        rpbins, xi, 10 ** log_rbins, pimax=data_corr.pimax)
                    x_plot, y_plot = rpbins, wp / rpbins
                elif corr_type == "cross":
                    # Cross-correlation (QSO x galaxy).
                    # Galaxy QHMF: use the halo mass function divided by 5
                    # (crude HOD) with a low-mass cutoff at log M = 10.7.
                    qhmf_gal = np.copy(mf_fit) / 5.0
                    qhmf_gal[log_m_axis < 10.7] = 0.0
                    log_rbins, xi_cross = get_corr_from_triangle_cross(
                        log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                        triangle=triangle_fit, log_m_axis=log_m_axis,
                        qhmf1=10 ** log_qhmf_interp, qhmf2=qhmf_gal)
                    # Volume-averaged correlation function (appropriate
                    # statistic for cross-correlations). qhtools returns just
                    # xi_vol; reconstruct the (log) bin centers for the x-axis.
                    xi_vol = get_volume_averaged_xi(
                        rpbins, xi_cross, 10 ** log_rbins, pimax=data_corr.pimax)
                    rcross_bins = 10 ** (0.5 * (np.log10(rpbins[:-1]) + np.log10(rpbins[1:])))
                    x_plot, y_plot = rcross_bins, xi_vol
                else:
                    continue

                ax.plot(x_plot, y_plot, color=color, alpha=0.05, lw=1.0, zorder=1)
                all_x.append(x_plot)
                all_y.append(y_plot)

            if len(all_y) > 0:
                all_y = np.array(all_y)
                x_ref = all_x[0]  # all samples share the same x-axis
                ax.plot(x_ref, np.median(all_y, axis=0), color=color, lw=2.5, zorder=5,
                        label=f"Model (z={z_val:.1f})")
                ax.fill_between(x_ref, np.percentile(all_y, 16, axis=0),
                                np.percentile(all_y, 84, axis=0), color=color, alpha=0.2, zorder=2)

            corr_label = "auto" if corr_type == "auto" else "cross"
            ax.text(0.95, 0.95, f"z = {z_val:.1f} ({corr_label})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.09, 95.0)
            ax.set_ylim(1.1e-1, 1e3)

            is_left = (panel_idx % ncols_corr == 0)
            is_bottom = (panel_idx // ncols_corr == nrows_corr - 1)
            if is_left:
                ax.set_ylabel(r"$w_p(r_p) / r_p$")
            else:
                ax.tick_params(labelleft=False)
            if is_bottom:
                ax.set_xlabel(r"$r_p$ [Mpc/h]")
            else:
                ax.tick_params(labelbottom=False)
            if panel_idx == 0:
                ax.legend(loc="lower left", fontsize=9)

        # Hide unused subplot slots
        for j in range(n_panels, len(axes_corr)):
            axes_corr[j].axis("off")
        fig_corr.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.12, hspace=0.05, wspace=0.05)
    else:
        print("No correlation data matched. Skipping corr plot.")


# Save all open figures when headless (BAQARO_HEADLESS=1) or when
# BAQARO_SAVE_FIGS=1, so we can run from a non-interactive shell without
# blocking on plt.show().
if os.environ.get("BAQARO_HEADLESS", "0") == "1" or os.environ.get("BAQARO_SAVE_FIGS", "0") == "1":
    # Repo-local plots/ by default; BAQARO_PLOTS_ON_DATA3=1 redirects to the
    # production tree. Shared resolver so this cannot drift from the rest.
    from baqaro.plotting_common.plot_config import resolve_plots_dir
    _out_dir = resolve_plots_dir(subdir="mcmc_results_analysis", source=source_dir)
    # Short, DESCRIPTIVE filenames: "analysis_<what>_<short model tag>".
    # The old scheme used the full 200-char mcmc_filename plus matplotlib's
    # window title, which is "Figure 1" under Agg -- so headless runs produced
    # 227-char names that did not say what the figure was. `_FIG_NAMES` is set
    # where each figure is built; anything unregistered keeps a numbered
    # fallback so a new figure can never be silently dropped.
    from baqaro.inference.comparison_config import short_name as _short
    _notes = f"_{mcmc_filename_notes}" if mcmc_filename_notes else ""
    _active = "".join(c if (c.isalnum() or c in "._-") else "_" for c in active_likelihoods)
    for i, fnum in enumerate(plt.get_fignums()):
        fig = plt.figure(fnum)
        what = _FIG_NAMES.get(fnum, f"fig{i:02d}")
        path = os.path.join(
            _out_dir, f"analysis_{what}_{_short}_{_active}{_notes}.{fig_ext()}")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved: {path}")
if os.environ.get("BAQARO_HEADLESS", "0") != "1":
    plt.show()
