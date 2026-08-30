"""Plot emulator accuracy against the training data it was fitted on.

Compares emulator predictions with the true simulated summary statistics, both
as a function of a single varied parameter and against the luminosity threshold,
to show where in parameter space the surrogate can be trusted.

This is the qualitative, by-eye counterpart to
:mod:`cross_validate_emulators`, which produces held-out error statistics.
Reads a training set and its emulators through the shared run identity, so it
needs no arguments beyond the environment that produced them.
"""

import lzma
import os
import pickle
import time
import h5py
import numpy as np
from matplotlib import gridspec, pyplot as plt

#: figure number -> short descriptive name, used to build the saved filename.
#: Without it the save loop falls back to matplotlib's window title, which is
#: "Figure 1" under Agg -- producing 185-char names that say nothing.
_FIG_NAMES = {}
import matplotlib.colors as mcolors
from baqaro.utils.my_dir import get_output_path


from baqaro.emulation.loading_helpers import load_training_data, load_emulators




# ==============================================================================
# 1. PLOTTING HELPERS
# ==============================================================================

def _setup_plot_canvas(title, ylabel, xlabel, xlims, ylims, ylims_small=(-1.5, 1.5)):
    """
    Internal helper to set up the figure, gridspec, and common axis properties.
    Now accepts 'xlabel' to customize the x-axis unit.
    """
    fig = plt.figure(figsize=(8., 6.5))
    spec = gridspec.GridSpec(ncols=1, nrows=2,
                             top=0.95, bottom=0.1,
                             left=0.1, right=0.82,
                             wspace=0.2, hspace=0.2, height_ratios=[3, 1])

    ax = fig.add_subplot(spec[0, 0])
    ax_small = fig.add_subplot(spec[1, 0], sharex=ax)

    # Main Axis Styling
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.setp(ax.get_xticklabels(), visible=False) # Hide x-ticks for top plot

    # Residual Axis Styling
    ax_small.set_xlabel(xlabel)  # <--- Dynamic Label Here
    ax_small.set_ylabel("log10 ratio")
    ax_small.set_ylim(ylims_small)
    ax_small.axhline(0., color="gray", linestyle="--")
    ax_small.axhspan(-0.2, 0.2, alpha=0.2, color="gray")

    return fig, ax, ax_small

def _add_colorbar(fig, cmap, norm, label):
    """Internal helper to add a standardized colorbar."""
    ax_cb = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label(label)
    return cbar

# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_emulator_performance_single_param(emulator, log_bins, log_sample,
                              index_param, index_redshift, index_log_L_threshold,
                              params, param_ranges, param_names_str,
                              xlabel, physical_floor=-9.5,
                              plot_uncertainties=True, plot_emulator=True):
    """
    Plots emulator performance colored by a specific parameter value.
    Added 'xlabel' argument.
    """

    # 1. Setup Canvas
    title = f"Emulator Performance - z bin {index_redshift}"
    if index_log_L_threshold is not None:
        title += f" (L_thr bin {index_log_L_threshold})"

    fig, ax, ax_small = _setup_plot_canvas(
        title=title,
        ylabel=r"$\log_{10} \Phi$ [Mpc$^{-3}$ dex$^{-1}$]",
        xlabel=xlabel,  # <--- Passed here
        xlims=(log_bins[0], log_bins[-1]),
        ylims=(-10.5, -2)
    )

    # 2. Setup Color Mapping
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=param_ranges[index_param][0], vmax=param_ranges[index_param][1])

    # 3. Emulator X-axis (High res)
    if plot_emulator and emulator is not None:
        if plot_uncertainties:
            pred_mu, pred_var = emulator.predict(params)
        else:
            pred_mu = emulator.predict_mean_only(params)

    # 4. Loop over samples

    for i in range(len(log_sample)):
        param_vals = params[i]
        color = cmap(norm(param_vals[index_param]))
        # Plot Data
        ax.plot(log_bins, log_sample[i, :],
                    color=color, alpha=0.8, marker=".", markersize=3, ls="None")

        # Plot Emulator
        if plot_emulator and emulator is not None:

            if index_log_L_threshold is not None:
                pred_mu_here = pred_mu[i, index_redshift, index_log_L_threshold, :]
                if plot_uncertainties:
                    pred_var_here = pred_var[i, index_redshift, index_log_L_threshold, :]
            else:
                pred_mu_here = pred_mu[i, index_redshift, :]
                if plot_uncertainties:
                    pred_var_here = pred_var[i, index_redshift, :]

            if plot_uncertainties:
                pred_std = np.sqrt(pred_var_here)
                ax.fill_between(log_bins, pred_mu_here - pred_std, pred_mu_here + pred_std,
                                alpha=0.2, color=color, lw=0)
            ax.plot(log_bins, pred_mu_here, color=color, alpha=0.8, ls="-")



            # Residuals: clamp both to physical_floor (matching cross_validate_physical)
            true_clamped = np.maximum(log_sample[i, :], physical_floor)
            pred_clamped = np.maximum(pred_mu_here, physical_floor)
            resid = true_clamped - pred_clamped
            ax_small.plot(log_bins, resid, color=color, alpha=0.8, ls="-")

    # 5. Add Colorbar
    _add_colorbar(fig, cmap, norm, param_names_str[index_param])

    return fig, ax, ax_small


def plot_emulator_performance_log_Lthr(emulator, log_bins, log_sample,
                              index_run, index_redshift,
                              params, param_ranges, param_names_str,
                              xlabel, log_L_thresholds, physical_floor=-9.5,
                              plot_uncertainties=True, plot_emulator=True):
    """
    Plots emulator performance for a single run, colored by log Luminosity Threshold.

    Parameters
    ----------
    emulator : GeneralEmulatorGP
        The trained emulator object.
    log_bins : array
        Bin centers for the x-axis (e.g., log M_BH).
    log_sample : array
        Shape (n_L_thresholds, n_bins). The training data for this run/redshift.
    index_run : int
        Index of the run to plot.
    index_redshift : int
        Index of the redshift slice.
    params : array
        Shape (n_samples, n_params). All parameter combinations.
    param_ranges : array
        Shape (n_params, 2). Min/max for each parameter (unused here; kept for API symmetry).
    param_names_str : list of str
        Names of parameters (unused here; kept for API symmetry).
    xlabel : str
        Label for x-axis.
    log_L_thresholds : array
        Array of log luminosity thresholds.
    plot_uncertainties : bool
        Whether to plot uncertainty bands.
    plot_emulator : bool
        Whether to plot emulator predictions.
    """
    log_bins = np.asarray(log_bins)
    if log_bins.size == (log_sample.shape[-1] + 1):
        # Handle edge-defined bins by converting to centers
        log_bins = 0.5 * (log_bins[:-1] + log_bins[1:])

    # 1. Setup Canvas
    title = f"Emulator Performance - z bin {index_redshift} (Run {index_run})"

    fig, ax, ax_small = _setup_plot_canvas(
        title=title,
        ylabel=r"$\log_{10} \Phi$ [Mpc$^{-3}$ dex$^{-1}$]",
        xlabel=xlabel,
        xlims=(log_bins[0], log_bins[-1]),
        ylims=(-10.5, -2)
    )

    # 2. Setup Color Mapping
    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=np.min(log_L_thresholds), vmax=np.max(log_L_thresholds))

    # 3. Get emulator predictions for all params at once (if plotting emulator)
    if plot_emulator and emulator is not None:
        if plot_uncertainties:
            pred_mu, pred_var = emulator.predict(params)
        else:
            pred_mu = emulator.predict_mean_only(params)

    # 4. Loop over thresholds
    for i, l_thr in enumerate(log_L_thresholds):
        color = cmap(norm(l_thr))

        # Plot Data
        ax.plot(log_bins, log_sample[i, :],
                color=color, alpha=0.8, marker=".", markersize=3, ls="None")

        # Plot Emulator
        if plot_emulator and emulator is not None:
            # Extract prediction for this run, redshift, and L_threshold
            pred_mu_here = pred_mu[index_run, index_redshift, i, :]
            if plot_uncertainties:
                pred_var_here = pred_var[index_run, index_redshift, i, :]
                pred_std = np.sqrt(pred_var_here)
                ax.fill_between(log_bins, pred_mu_here - pred_std, pred_mu_here + pred_std,
                                alpha=0.2, color=color, lw=0)

            ax.plot(log_bins, pred_mu_here, color=color, alpha=0.8, ls="-")

            # Residuals: clamp both to physical_floor (matching cross_validate_physical)
            true_clamped = np.maximum(log_sample[i, :], physical_floor)
            pred_clamped = np.maximum(pred_mu_here, physical_floor)
            resid = true_clamped - pred_clamped
            ax_small.plot(log_bins, resid, color=color, alpha=0.8, ls="-")

    # 5. Add Colorbar
    _add_colorbar(fig, cmap, norm, r"$\log_{10} L_{\mathrm{thr}}$")

    return fig, ax, ax_small



# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":


    # ==========================================
    # MAIN SCRIPT
    # ==========================================

    # Module-level configuration
    DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # site from the environment
    source_dir = DEFAULT_SOURCE_DIR


    # --- Run Identification ---
    # Override via env vars so the same script works for L2800N5040 and L2800N10080.
    # Falls back to the fiducial notes tokens, matching main_emulation.py and
    # cross_validate_emulators.py, so a bare run resolves the fiducial products
    # rather than a notes-less filename that was never written.
    from baqaro.utils.sim_config import (
        FIDUCIAL_NOTES_TRAINING, FIDUCIAL_NOTES_EMULATION)
    notes_file_training  = (os.environ.get("BAQARO_NOTES_FILE_TRAINING")
                            or FIDUCIAL_NOTES_TRAINING)
    notes_file_emulation = (os.environ.get("BAQARO_NOTES_FILE_EMULATION")
                            or os.environ.get("BAQARO_NOTES_FILE_TRAINING")
                            or FIDUCIAL_NOTES_EMULATION)

    # --- Floor values ---
    physical_floor = -9.5  # Threshold below which bins are considered empty

    # --- FILTERING TRAINING DATA ---
    filter_training_data = True  # Whether to filter out "bad" simulations

    # --- Snapshot Range ---
    min_snap = 0
    from baqaro.utils.sim_config import max_snap

    # --- Halo-data variant tokens (must match the training run filename) ---
    from baqaro.utils.sim_config import fold_subhalo_mass, merger_delay_mode

    # --- Subset tag (only meaningful when the training run used BAQARO_USE_SUBSAMPLE=1) ---
    # For the L2800N10080 maxsnap=60 fid3_10k run, the default below matches
    # what main_training.py wrote. Override with BAQARO_SUBSET_TAG="" to
    # load a full-sim (un-subsampled) training file.
    # Default to the production subsampled run's tag (auto-derived from
    # sim_config geometry + subsample env defaults) so callers needn't set
    # BAQARO_SUBSET_TAG. Env presence wins: BAQARO_SUBSET_TAG="" => full-sim.
    if "BAQARO_SUBSET_TAG" in os.environ:
        subset_tag = os.environ["BAQARO_SUBSET_TAG"]
    else:
        from baqaro.core_functions.tree_subsample import resolve_subset_tag
        subset_tag = resolve_subset_tag(max_snap)


    # --- ERDF (Eddington Ratio Distribution Function) ---
    # Controls the distribution of accretion rates relative to Eddington
    erdf_model = "log_normal_evol_halo_mass"

    # --- Simulation Parameters ---
    from baqaro.utils.sim_config import boxsize, N_particles_per_side, simulation_name


    # LOAD DATA
    # Mirror the exact order that main_training.py uses to compose the
    # output filename: maxsnap -> foldmass -> merger_delay_mode -> notes -> subset_tag.
    name_file = "{}_erdf_{}_maxsnap_{}".format(
            simulation_name, erdf_model, max_snap
            )
    if fold_subhalo_mass:
        name_file += "_foldmass"
    if merger_delay_mode != "instant_old":
        name_file += "_{}".format(merger_delay_mode)

    if notes_file_training is not None:
        name_file_training = name_file + "_{}".format(notes_file_training)
    else:
        name_file_training = name_file

    if notes_file_emulation is not None:
        name_file_emulation = name_file + "_{}".format(notes_file_emulation)
    else:
        name_file_emulation = name_file

    if subset_tag:
        name_file_training  += "_sub_{}".format(subset_tag)
        name_file_emulation += "_sub_{}".format(subset_tag)

    # Growth-cap + madau-f_eff tokens (after _sub_, matching main_training /
    # main_emulation). Required so these sanity plots load the SAME
    # _g{cap}/_feffcorr training + emulator products the fiducial pipeline
    # writes — otherwise they would validate the uncapped / uncorrected
    # files instead.
    from baqaro.utils.sim_config import growth_feff_suffix
    _gf_suffix = growth_feff_suffix()
    name_file_training  += _gf_suffix
    name_file_emulation += _gf_suffix

    path_out = get_output_path(source=source_dir)
    path_file_training = os.path.join(path_out, "training", f"training_data_emulation_{name_file_training}.hdf5")
    print(f"Loading training data from: {path_file_training}")


    # --- CONFIGURATION ---

    # Emulators ON by default: the whole point of this script is to overlay the
    # surrogate on the training data it was fitted to, so a bare run must show
    # both. Set BAQARO_LOAD_EMULATORS=0 to plot the raw training outputs alone
    # (useful when sanity-checking a training set before any emulator exists).
    _load_emus = os.environ.get("BAQARO_LOAD_EMULATORS", "1") != "0"
    emulator_flags = {
        "qlf":   _load_emus,
        "cerdf": _load_emus,
        "bhmf":  _load_emus,
        "qhmf":  _load_emus,
    }

    # index_z: which entry of snapshots_to_save to plot. For L2800N10080
    # maxsnap=60 the first ~4 saved snaps are at z > 17, before BHs exist
    # (all-NaN rows). Default to 8 → z~3 (the last saved snap, richest data).
    # Override via BAQARO_INDEX_Z env var for other redshifts.
    plot_config = {
        "uncertainties": _load_emus,
        "emulator": _load_emus,
        "index_param": 0,
        "index_z": int(os.environ.get("BAQARO_INDEX_Z", "8")),
        "index_L_thr": 2,
        "index_run": 2,
        "max_samples": 50,  # Max training samples to plot (None = all)
    }

    # --- LOAD DATA ---
    data = load_training_data(path_file_training)
    emulators = load_emulators(path_out, name_file_emulation, emulator_flags)

    # --- SUBSAMPLE TRAINING DATA (for performance plots only, NOT corner plot) ---
    n_total = len(data["params"])
    max_samples = plot_config["max_samples"]
    if max_samples is not None and n_total > max_samples:
        rng = np.random.default_rng(42)
        sub_idx = np.sort(rng.choice(n_total, size=max_samples, replace=False))
        # Keep full data for corner plot
        data_full = {
            "params": data["params"],
            "designs": data.get("designs", np.array([""] * n_total)),
        }
        # Subsample for performance plots
        data["params"] = data["params"][sub_idx]
        for key in ["log_qlfs", "log_cerdfs", "log_bhmfs", "log_qhmfs"]:
            if key in data:
                data[key] = data[key][sub_idx]
        if "designs" in data:
            data["designs"] = data["designs"][sub_idx]
        print(f"Subsampled {n_total} -> {max_samples} training points (corner plot uses full sample)")
    else:
        data_full = {
            "params": data["params"],
            "designs": data.get("designs", np.array([""] * n_total)),
        }
        print(f"Plotting all {n_total} training points")

    # --- EXTRACT PLOTTING SUBSETS ---
    idx_z = plot_config["index_z"]
    idx_l = plot_config["index_L_thr"]
    idx_p = plot_config["index_param"]
    idx_run = plot_config["index_run"]

    # Sanity-clip idx_z to the actual data's n_z (training files generated
    # with different snapshots_to_save schedules have different n_z; the
    # default index_z=8 would IndexError on the L2800N10080 maxsnap=60
    # fid3_10k file which only has n_z=5).
    _nz_avail = len(data["redshift_keys"])
    if idx_z >= _nz_avail:
        print(f"WARNING: BAQARO_INDEX_Z={idx_z} >= n_z={_nz_avail} in this training file; "
              f"clipping to idx_z={_nz_avail - 1} (z={data['redshift_keys'][_nz_avail - 1]:.2f})")
        idx_z = _nz_avail - 1

    redshift = data["redshift_keys"][idx_z]
    log_L_thr = data["log_L_thresholds"][idx_l]


    #
    # DEFINING LABELS
    label_Lbol = r"$\log_{10} L_{\mathrm{bol}}$ [erg/s]"
    label_Mbh  = r"$\log_{10} M_{\mathrm{BH}}$ [$M_{\odot}$]"
    label_Edd  = r"$\log_{10} \lambda_{\mathrm{Edd}}$"
    label_Mhalo= r"$\log_{10} M_{\mathrm{halo}}$ [$M_{\odot}$]"

    print(f"Plotting for z={redshift}, L_thr={log_L_thr}, Param Index={idx_p}")

    # --- PLOT 0: Corner plot of training parameters (uses FULL sample) ---
    print("--- Plotting parameter corner distribution ---")
    params_corner = data_full["params"]
    designs_corner = data_full["designs"]
    n_params = params_corner.shape[1]

    # Separate global vs local_box points
    mask_global = designs_corner != "local_box"
    mask_local = designs_corner == "local_box"
    has_local = mask_local.any()

    # Infer local_box bounds from the local_box points themselves
    if has_local:
        local_bounds_inferred = np.array([
            [params_corner[mask_local, i].min(), params_corner[mask_local, i].max()]
            for i in range(n_params)
        ])

    fig_corner, axes_corner = plt.subplots(n_params, n_params, figsize=(2.5 * n_params, 2.5 * n_params))
    _FIG_NAMES[fig_corner.number] = "param_corner"
    for i in range(n_params):
        for j in range(n_params):
            ax = axes_corner[i, j]
            if j > i:
                ax.set_visible(False)
            elif i == j:
                ax.hist(params_corner[mask_global, i], bins=15,
                        color="steelblue", edgecolor="white", alpha=0.7, label="global")
                if has_local:
                    ax.hist(params_corner[mask_local, i], bins=15,
                            color="orangered", edgecolor="white", alpha=0.7, label="local_box")
                    ax.axvline(local_bounds_inferred[i, 0], color="orangered", ls="--", lw=1)
                    ax.axvline(local_bounds_inferred[i, 1], color="orangered", ls="--", lw=1)
            else:
                ax.scatter(params_corner[mask_global, j], params_corner[mask_global, i],
                           s=10, alpha=0.6, color="steelblue", label="global")
                if has_local:
                    ax.scatter(params_corner[mask_local, j], params_corner[mask_local, i],
                               s=14, alpha=0.8, color="orangered", zorder=5, label="local_box")
                    # Draw local_box rectangle
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (local_bounds_inferred[j, 0], local_bounds_inferred[i, 0]),
                        local_bounds_inferred[j, 1] - local_bounds_inferred[j, 0],
                        local_bounds_inferred[i, 1] - local_bounds_inferred[i, 0],
                        linewidth=1.5, edgecolor="orangered", facecolor="orangered",
                        alpha=0.1, linestyle="--", zorder=4
                    )
                    ax.add_patch(rect)
                ax.set_xlim(data["param_ranges"][j])
                ax.set_ylim(data["param_ranges"][i])
            if i == n_params - 1:
                ax.set_xlabel(data["param_names"][j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(data["param_names"][i], fontsize=9)
            elif j != 0:
                ax.set_yticklabels([])
    if has_local:
        axes_corner[0, 0].legend(fontsize=7)
    fig_corner.suptitle("Training Parameter Distribution", y=1.0)
    fig_corner.tight_layout()


    # --- PLOT 1: QLF ---
    print("--- Plotting QLF ---")
    plot_emulator_performance_single_param(
        emulators["qlf"], data["log_lbins"],
        data["log_qlfs"][:, idx_z],
        idx_p,
        idx_z,
        None,
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Lbol,
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "qlf"


    # --- PLOT 2: CERDF ---
    print("--- Plotting CERDF ---")
    plot_emulator_performance_single_param(
        emulators["cerdf"], data["log_bins_cerdf"],
        data["log_cerdfs"][:, idx_z, idx_l],
        idx_p,
        idx_z,
        idx_l,
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Edd,
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "cerdf"

    # L-thr plot for CERDF
    plot_emulator_performance_log_Lthr(
        emulators["cerdf"], data["log_bins_cerdf"],
        data["log_cerdfs"][idx_run, idx_z, :, :],
        idx_run, idx_z,
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Edd,
        log_L_thresholds=data["log_L_thresholds"],
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "cerdf_vs_lthr"


    # --- PLOT 3: BHMF (all BHs, no luminosity cut — no L_threshold axis) ---
    print("--- Plotting BHMF ---")
    plot_emulator_performance_single_param(
        emulators["bhmf"], data["log_mbins_bhmf"],
        data["log_bhmfs"][:, idx_z],   # shape (n_runs, n_mbins) — no idx_l
        idx_p,
        idx_z,
        None,                           # no L_threshold axis
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Mbh,
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "bhmf"


    # --- PLOT 4: QHMF ---
    print("--- Plotting QHMF ---")
    plot_emulator_performance_single_param(
        emulators["qhmf"], data["log_mbins_qhmf"],
        data["log_qhmfs"][:, idx_z, idx_l],
        idx_p,
        idx_z,
        idx_l,
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Mhalo,
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "qhmf"

    # L-thr plot for QHMF
    plot_emulator_performance_log_Lthr(
        emulators["qhmf"], data["log_mbins_qhmf"],
        data["log_qhmfs"][idx_run, idx_z, :, :],
        idx_run, idx_z,
        data["params"], data["param_ranges"], data["param_names"],
        xlabel=label_Mhalo,
        log_L_thresholds=data["log_L_thresholds"],
        physical_floor=physical_floor,
        plot_uncertainties=plot_config["uncertainties"],
        plot_emulator=plot_config["emulator"]
    )
    _FIG_NAMES[plt.gcf().number] = "qhmf_vs_lthr"

    # Save all open figures when BAQARO_SAVE_FIGS=1; suppress the GUI
    # show() when BAQARO_HEADLESS=1 (lets us run from a non-interactive
    # SSH/Bash session without blocking forever).
    if os.environ.get("BAQARO_SAVE_FIGS", "0") == "1":
        # Repo-local plots/ by default; BAQARO_PLOTS_ON_DATA3=1 redirects to the
        # production tree. Shared resolver so this cannot drift from the rest.
        from baqaro.plotting_common.plot_config import resolve_plots_dir
        out_dir = resolve_plots_dir(subdir="training_diagnostics", source=source_dir)
        # Short, DESCRIPTIVE names: "training_<what>_<short model tag>", matching
        # the scheme used by the inference/emulation figures.
        from baqaro.plotting_common.plot_config import fig_ext
        _short = "{}_snap{}".format(simulation_name, max_snap)
        if notes_file_training:
            _short += "_{}".format(notes_file_training)
        for i, fnum in enumerate(plt.get_fignums()):
            fig = plt.figure(fnum)
            what = _FIG_NAMES.get(fnum, f"fig{i:02d}")
            path = os.path.join(out_dir, f"training_{what}_{_short}.{fig_ext()}")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  saved: {path}")

    if os.environ.get("BAQARO_HEADLESS", "0") != "1":
        plt.show()
