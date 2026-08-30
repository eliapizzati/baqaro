"""ANALYSIS FIGURE: halo accretion-rate distributions from the simulation.

Specific halo accretion rate against halo mass and redshift, taken straight from
the preprocessed halo histories -- no black-hole model involved. This is the
input the ERDF maps onto Eddington ratios, so its shape sets what the model can
produce before any parameter is fitted.
"""

import matplotlib
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot as plt

import numpy as np
import os

from baqaro.utils import tol_colors

from qhtools.utils.cosmology import cosmo

from baqaro.utils.my_dir import get_input_path_HBT_data, get_plots_path, get_output_path
# Redshift-driven plot (sim-independent): the script picks the nearest snapshot
# in the loaded sim's z grid for each target z. Same ladder L2800N5040 used to
# get via [4, 6, 8, 10, 12, 14, 16, 18, 28, 38, 48, 58, 68, 78].
REDSHIFT_TARGETS = [8.7, 7.3, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]

from baqaro.plotting_common.plot_config import save_fig, maybe_show

cmap = tol_colors.tol_cmap(colormap='rainbow_PuRd')

# TOTAL vs COLD specific accretion rate. Same variable as the paper module
# (`plotting_paper/plotting_halo_rate.py`), so one setting drives both.
# NOTE the two arrays are DIFFERENT PHYSICS -- cold is the total multiplied by
# the cold-accretion fraction -- not interchangeable presentations of one thing.
plot_cold_accretion = os.environ.get("BAQARO_HALO_RATE_COLD", "1") == "1"
name_fig = "halo_accretion_rate"  # sim/snap suffix appended below

source_dir = "machine_igm"  # Options: 'local', 'machine_cosma', 'machine_igm'
path_plots = get_plots_path(source=source_dir)
path_out = get_output_path(source=source_dir)

# Parameters matching those used in halo_mass_histories_saver

min_snap = 0
# max_snap comes from sim_config (env-overridable via BAQARO_MAX_SNAP, default =
# active sim's z=0 snapshot). For L2800N10080 down to z=4 use 50; for z=6 use 39.
from baqaro.utils.sim_config import max_snap


nbound_threshold = 40
halo_filtering_mode = "global"


# Simulation parameters
path_sim = get_input_path_HBT_data(source=source_dir)
from baqaro.utils.sim_config import (
    simulation_name, tdyn_fraction_default,
    halo_histories_name,
)


redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")


# defining basic redshift and time arrays
snapshots = np.arange(min_snap, max_snap + 1)
redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]
ages_of_the_universe = cosmo.age(redshifts)
delta_times_snapshots = np.diff(ages_of_the_universe, prepend=0.1)  # in Gyr
print("redshifts: {}".format(redshifts))
print("ages of the Universe (Gyr): {}".format(ages_of_the_universe))
print("delta times (Gyr): {}".format(delta_times_snapshots))



# load precomputed halo masses and accretion rates from files

print("Loading precomputed halo masses and accretion rates...")

# file i/o handling

# Use the shared builder: an inline name without the `_foldmass` /
# `_{merger_delay_mode}` tokens would load the legacy (no-fold, instant_old)
# arrays instead (both variants are on disk).
name_file_halos = halo_histories_name(
    max_snap_=max_snap,
    nbound_threshold=nbound_threshold,
    halo_filtering_mode=halo_filtering_mode,
    tdyn_fraction=tdyn_fraction_default,
)

name_file_specific_accretion = "specific_accretion_rates_{}".format(name_file_halos)
name_file_specific_cold_accretion = "specific_cold_accretion_rates_{}".format(name_file_halos)
name_file_halo_masses = "halo_masses_{}".format(name_file_halos)


path_file_specific_accretion = os.path.join(path_out, "halo_histories", f"{name_file_specific_accretion}.npy")
path_file_specific_cold_accretion = os.path.join(path_out, "halo_histories", f"{name_file_specific_cold_accretion}.npy")
path_file_halo_masses = os.path.join(path_out, "halo_histories", f"{name_file_halo_masses}.npy")


_rate_path = (path_file_specific_cold_accretion if plot_cold_accretion
              else path_file_specific_accretion)
if not os.path.exists(_rate_path):
    import glob as _glob
    _other = (path_file_specific_accretion if plot_cold_accretion
              else path_file_specific_cold_accretion)
    _stem = os.path.basename(_rate_path).split("_L2800")[0]
    _have = sorted({os.path.basename(f).split("_maxsnap")[1].split("_")[0]
                    for f in _glob.glob(os.path.join(path_out, "halo_histories",
                                                     f"{_stem}_{simulation_name}_maxsnap*.npy"))})
    raise SystemExit(
        f"\n[halo-rate] Missing the {'COLD' if plot_cold_accretion else 'TOTAL'} "
        f"specific accretion-rate array for max_snap={max_snap}:\n    {_rate_path}\n"
        f"  - max_snap values that DO have it: {_have or 'none'}\n"
        f"  - the other variant is "
        f"{'present -> rerun with BAQARO_HALO_RATE_COLD=' + ('0' if plot_cold_accretion else '1') if os.path.exists(_other) else 'also absent'}\n"
        f"  - or regenerate it with core_functions/halo_mass_histories_saver.py\n")


print("Loading halo masses from", path_file_halo_masses)
# All arrays are (n_snapshots, n_halos) C-order; snapshot access via arr[i] is contiguous.
halo_masses_all = np.load(path_file_halo_masses, mmap_mode='r')


if plot_cold_accretion:
    print("Loading specific cold accretion rates from", path_file_specific_cold_accretion)
    halo_specific_accretion_rates_all = np.load(path_file_specific_cold_accretion, mmap_mode='r')
else:
    print("Loading specific accretion rates from", path_file_specific_accretion)
    halo_specific_accretion_rates_all = np.load(path_file_specific_accretion, mmap_mode='r')

fig = plt.figure(figsize=(6, 7))
gs = matplotlib.gridspec.GridSpec(
    2, 1,
    height_ratios=[3, 1],
)

ax1 = fig.add_subplot(gs[0])  # larger (top) panel
ax2 = fig.add_subplot(gs[1])  # smaller (bottom) panel

# plot the halo accretion rate.
# Resolve the redshift ladder to actual snapshot indices in THIS sim's z grid.
# Each target z maps to argmin(|redshifts - tz|); duplicates are de-duplicated
# while preserving order, and snaps beyond max_snap are dropped.
snapshots_to_plot = []
for _tz in REDSHIFT_TARGETS:
    _i = int(np.argmin(np.abs(redshifts - _tz)))
    if _i <= max_snap and _i not in snapshots_to_plot:
        snapshots_to_plot.append(_i)

# --- Pass 1: scan snapshots, collect rates and percentiles for those that
# have data. We hold log-rate arrays in memory only long enough to build
# the histograms in pass 2 — but compute the redshift range first so the
# top-panel histogram colors and the colorbar share the *same* normalisation.
processed_snaps = []
log_rates_by_snap = {}
percentiles_plot = []

for i in snapshots_to_plot:
    mask = (halo_masses_all[i-1] > 0) & (halo_specific_accretion_rates_all[i] > 0)
    if np.sum(mask) == 0:
        print(f"No positive halo accretion rates at snapshot {snapshots[i]} (z={redshifts[i]:.2f}); skipping")
        continue
    # Clip on the slice to avoid materialising the full memmap.
    specific_slice = np.clip(halo_specific_accretion_rates_all[i, mask], 1e-8, 1e3)
    log_specific_accretion = np.log10(specific_slice)
    log_rates_by_snap[i] = log_specific_accretion
    percentiles = np.nanpercentile(log_specific_accretion, [2.3, 16, 50, 84, 97.7])
    percentiles_plot.append(percentiles)
    processed_snaps.append(i)
    print(f"Snapshot {snapshots[i]} (z={redshifts[i]:.2f}): "
          f"mean log sHAR = {np.mean(log_specific_accretion):.2f}, "
          f"percentiles [2.3,16,50,84,97.7] = {percentiles}")

if not processed_snaps:
    raise SystemExit("No snapshots produced data; can't plot.")

percentiles_plot_arr = np.array(percentiles_plot)  # shape (N, 5)

# Order processed_snaps by redshift (ascending z) so the bottom-panel line
# is monotonic in z, then use the SAME range for histogram colors AND the
# colorbar. This fixes the previous mismatch where colorbar used the
# processed range but histograms used the attempted range.
redshifts_processed = np.array([redshifts[i] for i in processed_snaps])
order = np.argsort(redshifts_processed)
redshifts_plot = redshifts_processed[order]
percentiles_plot_arr = percentiles_plot_arr[order]

z_lo = float(redshifts_plot.min())
z_hi = float(redshifts_plot.max())

# --- Pass 2: plot histograms using the consistent z range ---
for i in processed_snaps:
    log_specific_accretion = log_rates_by_snap[i]
    redshift = redshifts[i]
    color = cmap((redshift - z_lo) / (z_hi - z_lo)) if z_hi > z_lo else cmap(0.5)
    median = np.nanmedian(log_specific_accretion)
    ax1.hist(
        log_specific_accretion, bins=30, histtype='step',
        label=f"z={redshift:.2f}",
        color=color, density=True, lw=2,
    )
    ax1.axvline(median, linestyle=':', color=color, lw=1.5)
p3 = percentiles_plot_arr[:, 0]
p16 = percentiles_plot_arr[:, 1]
p50 = percentiles_plot_arr[:, 2]
p84 = percentiles_plot_arr[:, 3]
p97 = percentiles_plot_arr[:, 4]

ax2.plot(redshifts_plot, p50, label="median", color="darkgray", lw=2)
ax2.fill_between(redshifts_plot, p16, p84, alpha=0.2, label="16–84%", color="darkgray")
# ax2.fill_between(redshifts_plot, p3, p97, alpha=0.1, label="2.3–97.7%", color="darkgray")


cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=z_lo, vmax=z_hi)),
    ax=ax1,
)
cbar.set_label("Redshift")
# 5-6 evenly-spaced ticks across the actual z range (works for any sim).
cbar.set_ticks(np.linspace(z_lo, z_hi, 6))
cbar.ax.set_yticklabels([f"{z:.1f}" for z in np.linspace(z_lo, z_hi, 6)])



ax1.set_xlabel(r"log$_{10}$ specific halo accretion rate (sHAR) [Gyr$^{-1}$]", labelpad=-0.5)
ax1.set_ylabel("Probability distribution")
ax1.set_yscale('log')
ax1.set_ylim(1e-5, 1e1)
ax1.xaxis.set_minor_locator(AutoMinorLocator())


ax2.set_xlabel("Redshift")
ax2.set_ylabel(r"log$_{10}$ sHAR [Gyr$^{-1}$]")
ax2.set_yticks(np.arange(-1.5, 1.5, 0.5))
ax2.set_yticklabels(np.arange(-1.5, 1.5, 0.5))
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
# Invert x-axis so high-z (early universe) is on the left, matching the
# cosmic-time convention and the colorbar progression in the top panel.
ax2.invert_xaxis()

# ax1.legend()
# ax2.legend()

fig.subplots_adjust(left=0.16, right=0.965, top=0.965, bottom=0.08, hspace=0.25, wspace=0.0)



# Sim-aware filename so 5k and 10k outputs coexist.
tag = f"{simulation_name}_maxsnap{max_snap}"
save_fig(fig, path_plots, f"{name_fig}_{tag}")


maybe_show()