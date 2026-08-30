"""
PAPER FIGURE: distribution of the specific halo accretion rate.

Top panel  : per-redshift histograms of log10 of the specific halo accretion
             rate, Mdot_halo / M_halo  [Gyr^-1], coloured by redshift;
             dotted vertical lines mark the medians. (This reads the TOTAL
             specific halo accretion rate Mdot_halo / M_halo — i.e. NOT
             multiplied by the cold-accretion fraction. Swap the input file
             back to specific_cold_accretion_rates_*.npy for the cold variant.)
Bottom panel: redshift evolution of the median (line) and 16-84% band.

Reads the precomputed halo-history arrays produced by
``core_functions/halo_mass_histories_saver.py`` directly (the full catalogue,
no subsample), so no BH-evolution run is needed.

.. note::
   Needs the TOTAL-rate array ``specific_accretion_rates_*_maxsnap144_*.npy``
   for the z=0 fiducial. ``BAQARO_HALO_RATE_COLD=1`` renders the COLD variant
   instead, which is DIFFERENT PHYSICS from the published figure -- it is not a
   substitute for it.

Adapted for the paper from the working version of this figure. The
on-disk arrays are (n_snapshots, n_halos) C-order, so a snapshot row is
``arr[i]``; the filename carries the ``{simulation_name}_`` prefix and the
optional ``_foldmass`` / ``_{merger_delay_mode}`` tokens, matching the
canonical naming in ``plotting_common/load_data_to_plot.py`` and the files
actually on disk.

Run (defaults to L2800N5040 maxsnap=78, legacy halo files):
    env BAQARO_HEADLESS=1 python -m baqaro.plotting_paper.plotting_halo_rate
Switch to the foldmass + instant_new halo files:
    env BAQARO_FOLD_SUBHALO_MASS=1 BAQARO_MERGER_DELAY_MODE=instant_new ... python -m ...
"""

import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Pin the paper fiducial run BEFORE sim_config / the os.environ reads below see
# the environment (it force-sets BAQARO_SIM / BAQARO_MAX_SNAP / BAQARO_FOLD_SUBHALO_MASS
# / BAQARO_MERGER_DELAY_MODE). Side-effecting import — must precede sim_config.
# BAQARO_SOURCE_DIR is intentionally left ambient (machine choice, not run identity).
from baqaro.plotting_paper import fiducial_data  # noqa: F401

from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
from baqaro.utils.sim_config import (
    simulation_name, tdyn_fraction_default,
    max_snap, fold_subhalo_mass, merger_delay_mode,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import cmap_z
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "halo_accretion_rate"

# Redshift-driven (sim-independent): for each target z the script picks the
# nearest snapshot in the loaded sim's z grid.
REDSHIFT_TARGETS = [8.7, 7.3, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]

# Halo-history file parameters — must match the saver run that produced them.
# max_snap / fold_subhalo_mass / merger_delay_mode come from sim_config
# (env-overridable; the side-effecting fiducial_data import above force-sets
# the corresponding BAQARO_* env vars before sim_config reads them).
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
nbound_threshold = 40
halo_filtering_mode = "global"
min_snap = 0


# ==============================================================================
# PATHS
# ==============================================================================
path_sim = get_input_path_HBT_data(source=source_dir)
path_out = get_output_path(source=source_dir)
redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")

# Canonical halo-history filename (matches load_data_to_plot.name_file_halos).
name_file_halos = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
    simulation_name, max_snap, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
)
if fold_subhalo_mass:
    name_file_halos += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file_halos += "_{}".format(merger_delay_mode)

path_file_halo_masses = os.path.join(path_out, "halo_histories", f"halo_masses_{name_file_halos}.npy")
# TOTAL rate by default; BAQARO_HALO_RATE_COLD=1 selects the cold-accretion
# variant. The two arrays are generated separately by halo_mass_histories_saver,
# and they are NOT both present for every max_snap -- at the z=0 fiducial
# (max_snap=144) only the cold one exists -- so the choice is surfaced here
# instead of being a manual edit of the filename, and a missing file is reported
# with the alternatives rather than as a bare numpy FileNotFoundError.
_rate_cold = os.environ.get("BAQARO_HALO_RATE_COLD", "0") == "1"
_rate_stem = "specific_cold_accretion_rates" if _rate_cold else "specific_accretion_rates"
path_file_specific = os.path.join(path_out, "halo_histories", f"{_rate_stem}_{name_file_halos}.npy")
if not os.path.exists(path_file_specific):
    import glob as _glob
    _alt_stem = "specific_accretion_rates" if _rate_cold else "specific_cold_accretion_rates"
    _alt = os.path.join(path_out, "halo_histories", f"{_alt_stem}_{name_file_halos}.npy")
    _have = sorted(os.path.basename(f).split("_maxsnap")[1].split("_")[0]
                   for f in _glob.glob(os.path.join(path_out, "halo_histories",
                                                    f"{_rate_stem}_{simulation_name}_maxsnap*.npy")))
    raise SystemExit(
        f"\n[halo-rate] Missing the {'COLD' if _rate_cold else 'TOTAL'} specific "
        f"accretion-rate array for max_snap={max_snap}:\n    {path_file_specific}\n"
        f"  - max_snap values that DO have it: {_have or 'none'}\n"
        f"  - the other variant is {'present' if os.path.exists(_alt) else 'also absent'}"
        f"{' -> rerun with BAQARO_HALO_RATE_COLD=' + ('0' if _rate_cold else '1') if os.path.exists(_alt) else ''}\n"
        f"  - or regenerate it with core_functions/halo_mass_histories_saver.py\n")


# ==============================================================================
# LOAD
# ==============================================================================
snapshots = np.arange(min_snap, max_snap + 1)
redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]

print("Loading halo masses from", path_file_halo_masses)
# mmap, read-only. Layout is (n_snapshots, n_halos) C-order: arr[i] is a
# contiguous, zero-copy row for snapshot i.
halo_masses_all = np.load(path_file_halo_masses, mmap_mode="r")
print("Loading specific accretion rates from", path_file_specific)
halo_specific_accretion_rates_all = np.load(path_file_specific, mmap_mode="r")
print("  array shape:", halo_masses_all.shape)


# ==============================================================================
# FIGURE
# ==============================================================================
fig = plt.figure(figsize=(6, 7))
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])
ax1 = fig.add_subplot(gs[0])   # top: per-z histograms
ax2 = fig.add_subplot(gs[1])   # bottom: median vs redshift

# Snapshots nearest each target redshift (de-duplicated, in range).
snapshots_to_plot = []
for _tz in REDSHIFT_TARGETS:
    _i = int(np.argmin(np.abs(redshifts - _tz)))
    if _i <= max_snap and _i not in snapshots_to_plot:
        snapshots_to_plot.append(_i)
redshifts_plot = redshifts[snapshots_to_plot]
z_lo, z_hi = redshifts_plot.min(), redshifts_plot.max()


def _z_color(z):
    return cmap_z((z - z_lo) / (z_hi - z_lo)) if z_hi > z_lo else cmap_z(0.5)


percentiles_plot = []
for i in snapshots_to_plot:
    redshift = redshifts[i]
    prev = max(i - 1, 0)  # mass at the previous snapshot (halo must exist)
    print(f"Working on z={redshift:.2f} (snap {i})")

    # Snapshot-major rows (NOT arr[:, i] — the array is (n_snap, n_halo)).
    mass_prev = halo_masses_all[prev]
    rate_now = halo_specific_accretion_rates_all[i]
    mask = (mass_prev > 0) & (rate_now > 0)
    n = int(np.count_nonzero(mask))
    if n == 0:
        print(f"  no positive accretion rates at snap {i}")
        continue

    # Clip on the masked slice to avoid materialising the full row twice.
    specific_slice = np.clip(rate_now[mask], 1e-8, 1e3)  # Gyr^-1
    log_specific = np.log10(specific_slice)
    pcts = np.nanpercentile(log_specific, [2.3, 16, 50, 84, 97.7])
    print(f"  median={pcts[2]:.2f}, 16-84%=[{pcts[1]:.2f}, {pcts[3]:.2f}], N={n}")

    color = _z_color(redshift)
    ax1.hist(log_specific, bins=30, histtype="step", density=True, lw=2, color=color)
    ax1.axvline(pcts[2], ls=":", lw=1.5, color=color)
    percentiles_plot.append((redshift, pcts))

# Bottom panel: median + 16-84% band vs redshift (sorted by z).
percentiles_plot.sort(key=lambda t: t[0])
z_arr = np.array([t[0] for t in percentiles_plot])
pct_arr = np.array([t[1] for t in percentiles_plot])  # (N, 5)
p2p3, p16, p50, p84, p97p7 = (
    pct_arr[:, 0], pct_arr[:, 1], pct_arr[:, 2], pct_arr[:, 3], pct_arr[:, 4]
)

# 2-sigma band (2.3-97.7%), lighter alpha than the 1-sigma band.
ax2.fill_between(z_arr, p2p3, p97p7, alpha=0.08, color="darkgray", label="2.3-97.7%")
ax2.fill_between(z_arr, p16, p84, alpha=0.2, color="darkgray", label="16-84%")
ax2.plot(z_arr, p50, color="darkgray", lw=2, label="median")

# Redshift colorbar on the top panel.
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap_z, norm=plt.Normalize(vmin=z_lo, vmax=z_hi)),
    ax=ax1,
)
cbar.set_label("Redshift")
cbar.set_ticks(np.arange(z_lo, z_hi, 0.6))

# Labels: top x-axis spells out "specific halo accretion rate" alongside
# the math; bottom y-axis uses just the math (axis is narrow). This reads the
# TOTAL specific halo accretion rate (no cold-fraction factor — see file
# docstring); the labels are generic so they hold for either variant.
ax1.set_xlabel(
    r"log$_{10}$ specific halo accretion rate, "
    r"s$\dot{M}_\mathrm{acc}$=$\dot{M}_{\rm halo}/M_{\rm halo}$ [Gyr$^{-1}$]"
)
_axis_quantity = r"$\log_{10}\, \dot{M}_{\rm halo}/M_{\rm halo}$ [Gyr$^{-1}$]"
ax1.set_ylabel("PDF")
ax1.set_yscale("log")
ax1.set_ylim(1e-5, 1e1)
ax1.set_xlim(-4.2, 2.1)
ax1.xaxis.set_minor_locator(AutoMinorLocator())

ax2.set_xlabel("Redshift")
ax2.set_ylabel(_axis_quantity)
ax2.set_yticks(np.arange(-2.5, 1.5, 0.5))
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

fig.subplots_adjust(left=0.16, right=0.965, top=0.965, bottom=0.08, hspace=0.25, wspace=0.0)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
