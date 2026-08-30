
"""
Full-history black hole evolution script.
=========================================

Variant of ``main_evolution.py`` that records **per-sub-step** BH mass,
luminosity, and Eddington ratio for a *subset* of halos.  The subset is
chosen to form "closed" merger trees (target halos + all progenitors)
so that mergers remain self-consistent even after subsetting.

Key differences from ``main_evolution.py``:

* Uses ``evolve_BHs_full_history`` (numpy, not Numba) which returns full
  ``(n_halos, n_steps)`` arrays per snapshot instead of just the final state.
* Adds a **halo subset selection** stage before the main loop — modes:
  ``"mass_threshold"``, ``"random"``, ``"track_ids"``, or ``None`` (all halos).
* Full-history arrays are **h-stacked** across snapshots and saved transposed
  as ``(total_steps, n_halos)`` in the ``full_history/`` HDF5 group.
* Halo masses *are* included in the output HDF5 (unlike the production script).

Output file: ``bh_evolution_full_history_{name_file}.hdf5``
"""

# This module is a run script, not a library: its body loads catalogues and
# writes products at import time. Refuse a plain ``import`` so nobody starts a
# multi-hour job by accident (``python -m baqaro.core_functions.main_evolution_full_history`` sets __name__ to
# "__main__"; a multiprocessing "spawn" child re-imports it as "__mp_main__").
if __name__ not in ("__main__", "__mp_main__"):
    raise RuntimeError(
        "baqaro.core_functions.main_evolution_full_history is an entry-point script; run it with "
        "'python -m baqaro.core_functions.main_evolution_full_history' instead of importing it."
    )


import os
# Set Numba thread count before any numba import (via fast_lognormal_sampler)
os.environ["NUMBA_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))

import numpy as np
import h5py
import time


from numpy.random import default_rng

from qhtools.utils.cosmology import cosmo

from baqaro.core_functions.bh_seeding import spawn_BHs
from baqaro.core_functions.erdf_core_functions import Erdf
from baqaro.core_functions.bh_accretion import evolve_BHs_full_history
from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
from baqaro.core_functions.select_merger_branches import (
    build_reverse_index, get_merger_branch_mask
)

from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler


from baqaro.utils.logging import  set_logger
from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
from baqaro.utils.local_utils import get_mass_resolution_simulation
from baqaro.utils.my_units import mass_units, halo_mass_units_hbt
from baqaro.utils.provenance import write_run_provenance



# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Module-level configuration
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # site selected by env var

# --- Debug and Reproducibility ---
print_debug_info = True  # Enable verbose mass conservation checks
rng = default_rng(12345)  # Fixed seed for reproducibility
source_dir = DEFAULT_SOURCE_DIR


# --- Run Identification ---
# Free-form run-TYPE label (e.g. "z6_massthr11p5", "firstborn100_z0"), appended
# to the output filename as `_{notes_file}`. Unlike main_evolution, full-history
# notes encode the SELECTION / run type — not a disposable tag — so they are set
# explicitly per run. Env presence wins; "" => no token; unset => None.
# Deliberately NOT inherited from the bestfit registry's `notes` field (that
# records the *main_evolution* notes_file and is irrelevant to a full-history run).
notes_file = (os.environ["BAQARO_NOTES_FILE"] or None) if "BAQARO_NOTES_FILE" in os.environ else None

# --- Snapshot Range ---
min_snap = 0
# Sim-aware default from sim_config (honours BAQARO_MAX_SNAP), consistent with
# main_evolution: 144 (z=0) for the L2800N10080 default. Full-history runs are
# expensive and ALWAYS set this explicitly (e.g. BAQARO_MAX_SNAP=39 for z=6, 71
# for z=2) alongside a subset selection — see "Halo Subset Selection" below.
from baqaro.utils.sim_config import (
    max_snap, env_bool,
    FIDUCIAL_GROWTH_SUM_MAX, FIDUCIAL_MADAU_FEFF_CORRECTION,
)
# Halo-data file selection: which precomputed maxsnap halo file to load.
# Defaults to max_snap (the matching file); override via BAQARO_MAX_SNAP_HALOS
# to reuse a larger precomputed file that already covers the evolution range.
# Precomputed maxsnap on disk: L2800N5040 {14,18,38,58,78};
# L2800N10080 {39,50,60,71,144}.
max_snap_halos = int(os.environ.get("BAQARO_MAX_SNAP_HALOS", str(max_snap)))

# --- ERDF + seeding + time-resolution parameter resolution ----------------
# Three-tier precedence (highest wins), mirroring main_evolution.py:
#   3. Individual env vars: BAQARO_LOG_ETA_MEAN_0, BAQARO_LOG_ETA_MEAN_EVOL,
#      BAQARO_STD_0, BAQARO_LOGFSEED, BAQARO_SIGMASEED, BAQARO_LOGTCOHERENCE
#   2. Registry entry from BESTFIT_REGISTRY (bestfit_registry.py), looked
#      up by BAQARO_BESTFIT_NAME if it matches a registered key.
#   1. Hardcoded defaults below.
# BAQARO_BESTFIT_NAME is ALSO appended to the output filename as _bestfit_<name>
# (regardless of registry membership) so the plotting loader finds the file.
erdf_model = "log_normal_evol_halo_mass"
_DEFAULTS = {
    "log_eta_mean_0":   -0.866391,
    "log_eta_mean_evol": 0.812765,
    "std_0":             0.444504,
    "logfseed":         -5.669447,
    "sigmaseed":         0.461684,
    "logtcoherence":     6.060304,
}

bestfit_name = os.environ.get("BAQARO_BESTFIT_NAME", None)
_registry_entry = None
if bestfit_name:
    from baqaro.core_functions.bestfit_registry import BESTFIT_REGISTRY
    if bestfit_name in BESTFIT_REGISTRY:
        _registry_entry = BESTFIT_REGISTRY[bestfit_name]
        print(f"Loaded BESTFIT_REGISTRY[{bestfit_name!r}] ({_registry_entry.get('label','')})")
    else:
        print(f"NOTE: BAQARO_BESTFIT_NAME={bestfit_name!r} is not in BESTFIT_REGISTRY — "
              "treating as a free-form filename tag (using hardcoded defaults unless "
              "individual BAQARO_<PARAM> env vars are also set).")

def _resolve(param, env_var):
    """Tier 3 (env var) > Tier 2 (registry) > Tier 1 (default)."""
    if env_var in os.environ:
        return float(os.environ[env_var])
    if _registry_entry is not None and param in _registry_entry:
        return float(_registry_entry[param])
    return float(_DEFAULTS[param])

erdf_params_dict = {
    "log_eta_mean_0":    _resolve("log_eta_mean_0",    "BAQARO_LOG_ETA_MEAN_0"),
    "log_eta_mean_evol": _resolve("log_eta_mean_evol", "BAQARO_LOG_ETA_MEAN_EVOL"),
    "std_0":             _resolve("std_0",             "BAQARO_STD_0"),
    # "std_evol": 0.0           # Redshift evolution of scatter
}
logfseed      = _resolve("logfseed",      "BAQARO_LOGFSEED")
sigmaseed     = _resolve("sigmaseed",     "BAQARO_SIGMASEED")
logtcoherence = _resolve("logtcoherence", "BAQARO_LOGTCOHERENCE")
time_step_for_accretion = 10**(logtcoherence - 6)  # Sub-stepping timestep in Myr
print("Using logfseed = {}, sigmaseed = {}".format(logfseed, sigmaseed))
print(f"Using time step for accretion: {time_step_for_accretion} Myr (logtcoherence={logtcoherence})")

# Per-snapshot integrated-growth cap (matches main_evolution.py). Default 50.0
# leaves the legacy behaviour intact (exp(50) ~ 5e21 is a finite-but-astronomical
# guard); lowering via BAQARO_GROWTH_SUM_MAX (e.g. =4.6 -> max 100x per snap)
# progressively freezes the per-sub-step lightcurve once the snapshot's
# cumulative growth hits the cap. L_bol still tracks eta_j on the plateau.
growth_sum_max = float(os.environ.get("BAQARO_GROWTH_SUM_MAX", str(FIDUCIAL_GROWTH_SUM_MAX)))
if growth_sum_max != FIDUCIAL_GROWTH_SUM_MAX:
    print(f"[GROWTH-CAP MODE] BAQARO_GROWTH_SUM_MAX={growth_sum_max} -> "
          f"per-snapshot M_BH growth bounded at exp({growth_sum_max}) "
          f"~ {np.exp(growth_sum_max):.2e}x. "
          "Lightcurves plateau once cumulative growth hits the cap. "
          f"NOT the fiducial cap ({FIDUCIAL_GROWTH_SUM_MAX}).")

# Madau f_eff correction toggle (the f_eff correction). The full-history
# engine (bh_accretion.process_evolution_full_history) samples eps(eta) per
# sub-step and is ALREADY EXACT — it never carried the transfer-table median-eps
# bias — so this flag is a PHYSICAL NO-OP here. It is honoured only so that a
# full-history run launched alongside a corrected (_feffcorr) fast-path run gets
# the matching filename token + provenance, grouping the corrected-pipeline
# products. FH output equals the corrected (feff-on) fast-path model in both
# states. See core_functions/madau_feff.py.
madau_feff_correction = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION",
                                        "1" if FIDUCIAL_MADAU_FEFF_CORRECTION else "0").strip().lower()
                         not in ("0", "", "false", "no", "off"))
if madau_feff_correction:
    print("[MADAU-FEFF FIX] BAQARO_MADAU_FEFF_CORRECTION set: full-history is "
          "exact per-sub-step (already == corrected model) -> physics unchanged; "
          "tagging output filename _feffcorr for pipeline grouping.")
if bestfit_name:
    print(f"  → tagging output filename: _bestfit_{bestfit_name}")
    for _k in ("log_eta_mean_0", "log_eta_mean_evol", "std_0"):
        print(f"     {_k} = {erdf_params_dict[_k]}")


# --- Physics Toggles ---
accretion_on = True   # Enable black hole accretion
merger_on = True       # Enable black hole mergers

# --- Fine sub-step history storage (env-gated; default ON for back-compat) ------
# The fine per-sub-step arrays (n_substeps_total, n_halos) scale as 1/tau_coh and
# can reach ~1 TB (and are ACCUMULATED in RAM before the vstack write) at short
# tau_coh -- a hard RAM blocker, not just disk. They are ONLY needed to draw the
# blocky IID *reference* lightcurve in the methods figure; nothing else reads them.
# Consumers that need only the snapshot-BOUNDARY accretion-only masses are
# served instead by a cheap
# (n_snap, n_halos) `black_hole_masses_acc_only_all` array (= the last sub-step of
# the evolving BHs, 0 for newly-seeded/dying -- exactly the old
# `full_history/black_hole_masses_full_history[last_sub]` slice). Set
# BAQARO_FH_STORE_FINE=0 for the boundaries-only mode: skips the fine arrays
# entirely (file + accumulated RAM become tau-INDEPENDENT, ~GB), writes
# `black_hole_masses_acc_only_all` instead of the `full_history/` group.
# NB: the direct engine `evolve_BHs_full_history` still RETURNS the per-snapshot
# fine arrays, so a transient ~n_steps*n_evolving spike remains (e.g. ~60 GB at
# the z=6 snapshot for ~2M BHs * 2644 sub-steps at tau x0.1) -- freed each snap,
# fine on a big-RAM node. Only the cross-snapshot ACCUMULATION (the 1 TB) is killed.
# Consumers that need only snapshot-boundary masses read the coarse array.
STORE_FINE_HISTORY = env_bool("BAQARO_FH_STORE_FINE", True)
if not STORE_FINE_HISTORY:
    print("[FH] boundaries-only mode (BAQARO_FH_STORE_FINE=0): skipping fine "
          "per-sub-step arrays; writing snapshot-resolution "
          "black_hole_masses_acc_only_all instead.", flush=True)

# --- Plateaued-halo accretion (env-gated; default NON-ACCRETING) ---------------
# Halos whose LastMaxMass has plateaued have specific cold accretion rate == 0.
# The clip floor would otherwise give them a spurious eta pile-up that crashes
# the 16th percentile of the eta band at low z. We instead treat them as
# strictly NON-ACCRETING: M_BH frozen, L_bol = 0, eta = 0 (-> masked from the
# band). QLF-invariant (verified to 0.000 dex). Set BAQARO_NONACCRETING_ZERO_RATE=0
# to restore the legacy clip-floor behaviour (output tagged _clipfloor).
NONACCRETING_ZERO_RATE = env_bool("BAQARO_NONACCRETING_ZERO_RATE", True)
if not NONACCRETING_ZERO_RATE:
    notes_file = (notes_file + "_clipfloor") if notes_file else "clipfloor"
    print(f"[NONACC OFF] legacy clip-floor accretion for plateaued halos; output tag -> '{notes_file}'.")

# --- Parallelization ---
parallelize = True
num_workers = 64
backend = "threading"  # Options: "loky", "threading", "multiprocessing"

# --- Radiative Efficiency ---
rad_efficiency_0 = 0.1          # Base radiative efficiency (epsilon)
rad_efficiency_model = "madau+"  # Options: "constant", "madau+" (spin-dependent)

# --- Halo Selection ---
nbound_threshold = 40       # Minimum bound particles for a resolved halo
halo_filtering_mode = "global"  # Options: "local", "global"

# Must match the saver run that produced the halo files we load below.
# See get_merger_trees() for the merger modes.
# Override via BAQARO_FOLD_SUBHALO_MASS / BAQARO_MERGER_DELAY_MODE. The
# L2800N10080 halo cache carries the foldmass+instant_new variant at
# maxsnap {39,50,60,71} with tdynfraction 0.25, so fiducial 10k runs use the
# defaults (fold on, instant_new).
#
# tdyn_fraction_default is sim-dependent (sim_config.py). 0.20 at
# L2800N5040, 0.25 at L2800N10080.
from baqaro.utils.sim_config import (
    tdyn_fraction_default, fold_subhalo_mass, merger_delay_mode,
)

# --- Halo Subset Selection ---
# Full-history recording is expensive, so we evolve only a subset of halos.
# The subset forms "closed" merger trees: every target halo and ALL of its
# progenitors are included so that BH mergers remain self-consistent.
# Set subset_selection_mode = None to evolve all halos (very slow / large output).
# Override per run:
#   BAQARO_SUBSET_SELECTION_MODE = "mass_threshold" | "random" | "first_born" | "track_ids" | "none"
#   BAQARO_N_RANDOM = integer (for "random" mode)
#   BAQARO_MASS_THRESHOLD = float (Msun) OR BAQARO_LOG_MASS_THRESHOLD = log10(M/Msun)
#       (for "mass_threshold" mode; the log form is natural for "halos above 10^11.5")
#   BAQARO_N_FIRST_BORN = integer (for "first_born" mode: the N halos with the
#       earliest birth snapshot = highest formation redshift)
_mode_env = os.environ.get("BAQARO_SUBSET_SELECTION_MODE", "mass_threshold").lower()
subset_selection_mode = None if _mode_env == "none" else _mode_env
# Mass threshold accepted in log10(M/Msun) (preferred) or linear Msun.
if "BAQARO_LOG_MASS_THRESHOLD" in os.environ:
    _mass_threshold = 10.0 ** float(os.environ["BAQARO_LOG_MASS_THRESHOLD"])
else:
    _mass_threshold = float(os.environ.get("BAQARO_MASS_THRESHOLD", "1.0"))
subset_params = {
    "mass_threshold": _mass_threshold,
    "n_random": int(os.environ.get("BAQARO_N_RANDOM", "1000")),
    "n_first_born": int(os.environ.get("BAQARO_N_FIRST_BORN", "1000")),
    "track_ids": np.arange(0, 1000, 1),            # For "track_ids": explicit list/array of target track IDs
    "random_seed": 42,            # Seed for random selection
}

# OPTIONAL subsample intersection.  When True the target list is intersected
# with the production tree-subsample mask BEFORE closure (closure can still
# walk into non-subsample progenitors so the merger tree remains complete).
# This caps the target count at ~12% of the full-sim selection (for the root144
# subsample at L2800N10080) while preserving topology. The filename token grows
# a ``_sub`` suffix so the output never collides with the unsubsampled run.
# Set ``BAQARO_FH_INTERSECT_SUBSAMPLE=1`` to enable; uses ``BAQARO_SUBSET_TAG`` if
# present, else auto-resolves the production subset tag from sim_config.
intersect_subsample = env_bool("BAQARO_FH_INTERSECT_SUBSAMPLE", False)
# When True the subset is closed under the merger tree (target halos +
# all progenitors). Required when merger_on=True so that BH masses
# delivered into descendants come from self-consistently evolved
# progenitors. With merger_on=False, dead progenitors are pure overhead
# (their BHs are discarded at merger), so this can be set False.
close_trees = False if not merger_on else True


def _make_selection_tag(mode, params, closed):
    """Compact, self-describing filename token for the halo selection, so
    different full-history versions never collide. The redshift target is
    already carried by the ``maxsnap_{snap}`` token, so it is NOT repeated here.

      mass_threshold -> massthr{log10(M/Msun):.1f}   (e.g. massthr11.5)
      first_born     -> firstborn{N}
      random         -> rand{N}seed{seed}
      track_ids      -> trackids{N}
      None (all)     -> allhalos

    A trailing ``_open`` marks selections NOT closed under mergers
    (close_trees=False) — a physically different run from the closed default.
    """
    if mode is None:
        tag = "allhalos"
    elif mode == "mass_threshold":
        thr = params["mass_threshold"]
        tag = "massthr{:.1f}".format(np.log10(thr) if thr > 0 else 0.0)
    elif mode == "first_born":
        tag = "firstborn{}".format(int(params["n_first_born"]))
    elif mode == "random":
        tag = "rand{}seed{}".format(int(params["n_random"]), int(params["random_seed"]))
    elif mode == "track_ids":
        tag = "trackids{}".format(len(np.asarray(params["track_ids"])))
    else:
        tag = str(mode)
    if not closed:
        tag += "_open"
    return tag


selection_tag = _make_selection_tag(subset_selection_mode, subset_params, close_trees)
if intersect_subsample and subset_selection_mode is not None:
    selection_tag += "_sub"
print(f"Selection tag (filename): {selection_tag}")




# ==============================================================================
# INITIALIZATION
# ==============================================================================

# --- Initialize ERDF Model ---
erdf = Erdf(erdf_model, erdf_params_dict, rng=rng)

# --- Set Paths ---
path_sim = get_input_path_HBT_data(source=source_dir)
path_out = get_output_path(source=source_dir)

# --- Simulation Parameters ---
from baqaro.utils.sim_config import boxsize, N_particles_per_side, simulation_name

mass_resolution = get_mass_resolution_simulation(boxsize, N_particles_per_side)

folder_hbt = os.path.join(path_sim, f"{simulation_name}/HBT_compressed")
redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")
path_in = os.path.join(folder_hbt, "OrderedSubSnap_{snap_nr:03d}.hdf5")

# --- Output File Naming ---
# Matches main_evolution.py convention: {sim}_erdf_{model}_maxsnap_{snap}_6d_{notes}
# NB: the historical `_6d` token ("6 free parameters") was dropped —
# it was constant in every name, never varied, and the emulator/chain names had
# already dropped it. All on-disk products were renamed to match.
name_file = "{}_erdf_{}_maxsnap_{}".format(
    simulation_name, erdf_model, max_snap
)
if fold_subhalo_mass:
    name_file += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file += "_{}".format(merger_delay_mode)
# Auto selection tag — the primary differentiator between full-history
# versions (mode + params). notes_file (optional) is an extra free-form suffix.
name_file += "_{}".format(selection_tag)
if notes_file is not None:
    name_file += "_{}".format(notes_file)
# Match main_evolution.py / load_data_to_plot.py ordering: _bestfit_<name>
# AFTER notes_file so the plotting loader resolves the same filename.
if bestfit_name:
    name_file += "_bestfit_{}".format(bestfit_name)

# Growth-cap mode appends _g<value> so capped runs don't clobber the fiducial.
# Default 50.0 → no token.
if growth_sum_max != 50.0:
    if float(growth_sum_max).is_integer():
        _g_tok = f"g{int(growth_sum_max)}"
    else:
        _g_tok = f"g{growth_sum_max:g}"
    name_file += f"_{_g_tok}"

# _feffcorr token (the f_eff correction). Physics here is exact regardless;
# the token groups this FH run with corrected-pipeline (_feffcorr) fast-path runs.
if madau_feff_correction:
    name_file += "_feffcorr"

path_file = os.path.join(path_out, "evolution", f"bh_evolution_full_history_{name_file}.hdf5")
path_log = os.path.join(path_out, "logs", f"bh_evolution_full_history_{name_file}.log")

# SYMLINK GUARD (mirrors main_evolution.py). The `_fiducial_` alias layer
# (scripts/make_fiducial_aliases.py) points symlinks at the settled full-history
# products. h5py "w" FOLLOWS a symlink, so writing here would truncate the file the
# alias points at. Refuse before any data is loaded.
if os.path.islink(path_file):
    raise SystemExit(
        "\n[SYMLINK GUARD] The resolved output is a SYMLINK:\n"
        f"    {path_file}\n    -> {os.readlink(path_file)}\n"
        "Writing here would follow the link and DESTROY the file it points at "
        "(settled paper product; see evolution/README_FIDUCIAL.txt).\n"
        "Fix: pick a real filename — `fiducial` is an alias token, not a notes label.\n")

logger = set_logger(path_log)


# ==============================================================================
# LOAD PRECOMPUTED HALO DATA
# ==============================================================================
start_time_global = time.time()

# --- Redshift and Time Arrays ---
snapshots = np.arange(min_snap, max_snap + 1)
redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]
ages_of_the_universe = cosmo.age(redshifts)  # Gyr
delta_times_snapshots = np.diff(ages_of_the_universe, prepend=0.01)  # Gyr

if print_debug_info:
    print("redshifts:", redshifts)
    print("ages of the Universe (Gyr):", ages_of_the_universe)
    print("delta times (Gyr):", delta_times_snapshots)

# --- Build File Paths for Precomputed Data ---

name_file_halos = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
    simulation_name, max_snap_halos, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
)
if fold_subhalo_mass:
    name_file_halos += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file_halos += "_{}".format(merger_delay_mode)

name_file_accretion = "accretion_rates_{}".format(name_file_halos)
name_file_specific_cold_accretion = "specific_cold_accretion_rates_{}".format(name_file_halos)
name_file_halo_masses = "halo_masses_{}".format(name_file_halos)
name_folder_merger_trees = "merger_trees_{}".format(name_file_halos)

path_file_accretion = os.path.join(path_out, "halo_histories", f"{name_file_accretion}.npy")
path_file_specific_cold_accretion = os.path.join(path_out, "halo_histories", f"{name_file_specific_cold_accretion}.npy")
path_file_halo_masses = os.path.join(path_out, "halo_histories", f"{name_file_halo_masses}.npy")
path_file_trees = os.path.join(path_out, "halo_histories", f"{name_folder_merger_trees}")
path_file_reverse_index = os.path.join(path_out, "halo_histories", f"reverse_index_{name_file_halos}.npz")

# --- Load Halo Masses (Memory-Mapped) ---
# Layout: (n_snapshots, n_halos), C-order
print("Loading halo masses from", path_file_halo_masses)
halo_masses_all = np.load(path_file_halo_masses, mmap_mode='r')

# --- Load Merger Trees ---
print("Loading merger trees from", path_file_trees)
merger_trees = MergerTreeLoader(path_file_trees)

# --- Load Cold Gas Accretion Rates (Memory-Mapped) ---
# Layout: (n_snapshots, n_halos), C-order
print("Loading accretion rates from", path_file_accretion)
halo_specific_cold_accretion_rates_all = np.load(path_file_specific_cold_accretion, mmap_mode='r')

# --- Prefetch 1D Merger Tree Arrays into RAM ---
print("Prefetching 1D merger tree arrays into RAM...")
mt_birth_index = np.array(merger_trees.snapshot_indexes_of_birth)
mt_death_index = np.array(merger_trees.snapshot_indexes_of_death)
mt_track_ids = np.array(merger_trees.track_ids)
mt_merger_ids = np.array(merger_trees.merger_track_ids)
print("Prefetch complete.")

# --- Precompute Constant Masks ---
mask_merged = mt_merger_ids != -1
mask_disrupted = mt_merger_ids == -1


# ==============================================================================
# HALO SUBSET SELECTION (Closed Merger Trees)
# ==============================================================================
# Select a subset of halos that form "closed" merger trees:
# - All target halos + all their progenitors
# - This ensures mergers are self-consistent

if subset_selection_mode is not None:
    print("\n" + "="*60)
    print("HALO SUBSET SELECTION")
    print("="*60)

    # 1. Determine target track IDs based on selection mode
    if subset_selection_mode == "mass_threshold":
        # Select halos above mass threshold (Msun) at final snapshot.
        # halo_masses_all is in INTERNAL units (= Msun / mass_units), so the
        # user's Msun input divides by ``mass_units`` only — NOT also by
        # ``halo_mass_units_hbt`` (1e3) which would shift the threshold 1000x.
        final_masses = halo_masses_all[max_snap]
        mass_thresh = subset_params["mass_threshold"] / mass_units
        target_mask = final_masses > mass_thresh
        target_track_ids = mt_track_ids[target_mask]
        print(f"Selection mode: mass_threshold > {subset_params['mass_threshold']:.2e} Msun")
        print(f"Found {len(target_track_ids)} target halos above threshold")

    elif subset_selection_mode == "random":
        final_masses = halo_masses_all[max_snap]
        alive_mask = final_masses > 0
        alive_indices = np.where(alive_mask)[0]
        rng_subset = np.random.default_rng(subset_params["random_seed"])
        n_select = min(subset_params["n_random"], len(alive_indices))
        target_indices = rng_subset.choice(alive_indices, size=n_select, replace=False)
        target_track_ids = mt_track_ids[target_indices]
        print(f"Selection mode: random {n_select} halos from {len(alive_indices)} alive at z=0")

    elif subset_selection_mode == "first_born":
        # Select the N halos with the EARLIEST birth snapshot (= highest
        # formation redshift). Only halos with birth_index >= 0 are eligible:
        # birth == -1 means "present from the first tracked snapshot" — these
        # are NEVER seeded (seeding fires on birth_index == i, i in [0, max])
        # and never accrete, so picking them yields zero BHs. We sort only the
        # eligible halos (no sentinel — a large int sentinel would wrap to -1
        # under int32 + NEP-50 casting and sort to the front). Closure under
        # mergers then pulls in their (few) progenitors. Good pool for the
        # lightcurve / M_BH-M_halo trajectory plots tracing the first BHs to z_end.
        n_fb = subset_params["n_first_born"]
        eligible = np.flatnonzero(mt_birth_index >= 0)
        order = eligible[np.argsort(mt_birth_index[eligible], kind="stable")]
        n_select = min(n_fb, order.size)
        target_indices = order[:n_select]
        target_track_ids = mt_track_ids[target_indices]
        _bsnaps = mt_birth_index[target_indices]
        print(f"Selection mode: first_born {n_select} earliest-born halos "
              f"(birth snaps {_bsnaps.min()}..{_bsnaps.max()}, "
              f"z {redshifts[_bsnaps.min()]:.2f}..{redshifts[_bsnaps.max()]:.2f})")

    elif subset_selection_mode == "track_ids":
        target_track_ids = np.asarray(subset_params["track_ids"], dtype=np.int64)
        print(f"Selection mode: explicit track_ids, {len(target_track_ids)} targets")

    else:
        raise ValueError(f"Unknown subset_selection_mode: {subset_selection_mode}")

    # 1b. Optional intersection with the production tree-subsample mask.
    # Loads the existing subsample .npz, restricts target_track_ids to halos
    # IN the subsample. Closure below still walks the FULL tree so progenitors
    # outside the subsample come along (needed for merger-tree topology).
    if intersect_subsample:
        from baqaro.core_functions.tree_subsample import resolve_subset_tag, load_subset
        _sub_tag = os.environ.get("BAQARO_SUBSET_TAG") or resolve_subset_tag(max_snap_halos)
        _sub_path = os.path.join(
            path_out, "halo_subsets",
            f"subset_{name_file_halos}_{_sub_tag}.npz",
        )
        print(f"Intersecting targets with subsample mask from {_sub_path}")
        _sub_data = load_subset(_sub_path)
        _sub_mask = _sub_data["subset_mask"]   # bool, shape (n_halos,)
        # target_track_ids[i] == i (track_ids double as array indices)
        _n_before = len(target_track_ids)
        target_track_ids = target_track_ids[_sub_mask[target_track_ids]]
        _n_after = len(target_track_ids)
        print(f"  Subsample intersection: {_n_before:,} -> {_n_after:,} target halos "
              f"({100.0 * _n_after / max(1, _n_before):.2f}% retained)")
        if _n_after == 0:
            raise RuntimeError("Empty target set after subsample intersection — check thresholds and subset cache.")

    # 2. Load or build reverse index (cached to disk) -- only needed for closed trees
    if close_trees:
        if os.path.exists(path_file_reverse_index):
            print(f"Loading cached reverse index from {path_file_reverse_index}...")
            t0 = time.time()
            data = np.load(path_file_reverse_index)
            reverse_index = (data['indptr'], data['indices'])
            print(f"  Reverse index loaded in {time.time() - t0:.2f}s")
        else:
            print("Building reverse merger index...")
            t0 = time.time()
            reverse_index = build_reverse_index(mt_merger_ids)
            print(f"  Reverse index built in {time.time() - t0:.2f}s")
            print("  Saving reverse index for next time...")
            np.savez(path_file_reverse_index, indptr=reverse_index[0], indices=reverse_index[1])
            print(f"  Saved to {path_file_reverse_index}")

    # 3. Get subset mask -- closed merger tree (all progenitors) or target-only
    if close_trees:
        print("Finding closed merger tree (all progenitors)...")
        t0 = time.time()
        subset_mask = get_merger_branch_mask(
            target_track_ids, mt_track_ids, mt_merger_ids,
            include_targets=True, reverse_index=reverse_index
        )
        n_selected = np.count_nonzero(subset_mask)
        print(f"  Selected {n_selected:,} halos ({100*n_selected/len(subset_mask):.2f}% of total)")
        print(f"  Completed in {time.time() - t0:.2f}s")
    else:
        print("Skipping closed-tree expansion (merger_on=False); using target halos only.")
        subset_mask = np.zeros(len(mt_track_ids), dtype=bool)
        # target_track_ids double as array indices (track_ids[i] == i)
        subset_mask[target_track_ids] = True
        n_selected = np.count_nonzero(subset_mask)
        print(f"  Selected {n_selected:,} halos ({100*n_selected/len(subset_mask):.2f}% of total)")

    # 4. Build remapping arrays: old_index -> new_index
    print("Building index remapping...")
    old_to_new = np.full(len(mt_track_ids), -1, dtype=np.int64)
    new_indices = np.where(subset_mask)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))

    # 5. Remap merger_track_ids to new indices
    mt_merger_ids_remapped = np.full(n_selected, -1, dtype=np.int64)
    for new_idx, old_idx in enumerate(new_indices):
        old_dest = mt_merger_ids[old_idx]
        if old_dest >= 0:
            mt_merger_ids_remapped[new_idx] = old_to_new[old_dest]

    # 6. Subset all arrays using integer indexing
    # On-disk layout is (n_snapshots, n_halos) — subset along halo axis.
    #
    # Speed note: the naive `arr[:, cols]` does column fancy
    # indexing on a memory-mapped (n_snaps, n_halos) row-major file, which
    # generates per-element page faults across the ENTIRE file (gigabytes of
    # random I/O). At L2800N10080 with 386 M halos / 145 snaps and a 26 k-halo
    # closure, this took hours.
    #
    # Fix: read each snapshot row (contiguous on disk) sequentially, then
    # slice in RAM. Mathematically identical, ~100-1000× faster.
    def _subset_rows(arr, cols):
        out = np.empty((arr.shape[0], len(cols)), dtype=arr.dtype)
        for i in range(arr.shape[0]):
            out[i] = arr[i][cols]   # arr[i] is a contiguous row → 1 sequential read
        return out
    print("Subsetting arrays...")
    print(f"  Reading {len(new_indices)} halos from disk (row-by-row)...")
    halo_masses_all = _subset_rows(halo_masses_all, new_indices)
    halo_specific_cold_accretion_rates_all = _subset_rows(halo_specific_cold_accretion_rates_all, new_indices)

    mt_birth_index = mt_birth_index[new_indices]
    mt_death_index = mt_death_index[new_indices]
    mt_track_ids_original = mt_track_ids[new_indices]
    mt_track_ids = np.arange(n_selected)
    mt_merger_ids = mt_merger_ids_remapped

    # Recompute masks with new arrays
    mask_merged = mt_merger_ids != -1
    mask_disrupted = mt_merger_ids == -1

    # Store mapping for output
    subset_old_to_new = old_to_new
    subset_new_indices = new_indices

    # Compute indices of original targets in the subset
    print("  Finding original targets in subset...")
    target_set = set(target_track_ids)
    original_targets_new_indices = np.array([i for i, tid in enumerate(mt_track_ids_original) if tid in target_set], dtype=np.int64)

    print(f"\nSubset complete: {n_selected:,} halos selected for evolution")
    print(f"Found {len(original_targets_new_indices)} original targets in subset")
    print("="*60 + "\n")

else:
    # No subsetting - use all halos
    subset_mask = None
    mt_track_ids_original = mt_track_ids


# ==============================================================================
# LOAD LOGNORMAL SUM SAMPLER (Transfer Function)
# ==============================================================================
time_here = time.time()
logger.info("LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
sampler_file = "Universal_Lognormal_Sampler_final.npz"
transfer_function = load_3d_sampler(sampler_file)
logger.info("FINISHED LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
logger.info("Total time elapsed {} seconds".format(time.time() - time_here))


# ==============================================================================
# BLACK HOLE EVOLUTION LOOP
# ==============================================================================
logger.info("STARTING BH EVOLUTION ROUTINE")

# --- Allocate Output Arrays ---
# Layout: (n_snapshots, n_halos) — matches on-disk convention
total_number_of_objects = halo_masses_all.shape[1]
black_hole_masses_all = np.zeros((len(snapshots), total_number_of_objects), dtype=np.float32)
Lbols_all = np.zeros((len(snapshots), total_number_of_objects), dtype=np.float32)

# Boundaries-only product: accretion-only BH mass at each snapshot boundary
# (= last sub-step of the evolving BHs, BEFORE merger deliveries). Only allocated
# in boundaries-only mode; replaces the fine full_history slice the export reads.
black_hole_masses_acc_only_all = (
    np.zeros((len(snapshots), total_number_of_objects), dtype=np.float32)
    if not STORE_FINE_HISTORY else None
)

# Full history lists: one (n_steps_i, n_halos) array per snapshot, v-stacked after the loop.
# n_steps_i varies per snapshot because delta_t varies. (Unused in boundaries-only mode.)
Lbols_full_history = []
black_hole_masses_full_history = []
etas_full_history = []
times_full_history = []

start_time_bh = time.time()

# --- Counters for Summary Statistics ---
lost_black_holes_total = 0
new_black_holes_total = 0
merged_black_holes_total = 0

# --- Initialize Variables for Debug Block ---
mass_lost_BHs = 0.0
mass_from_aborted_mergers = 0.0

# --- Memory safeguard ---
# Full-history runs grow RSS monotonically as the per-substep history arrays
# extend with each snapshot. If a too-generous selection (mass_thr too low /
# subsample retention too high) is asked for, an unprotected run can OOM the
# whole machine. This in-loop guard reads
# /proc/meminfo each snap and aborts cleanly when MemAvailable drops below
# ``BAQARO_FH_MIN_AVAIL_GB`` (default 200 GB) — a clear error message + a chance
# for the OS to release pages before the kernel OOM-killer fires. Also prints
# RSS+available every snap so the trajectory is visible in real time.
import resource as _resource

def _read_mem_available_gb():
    try:
        with open("/proc/meminfo") as _f:
            for _ln in _f:
                if _ln.startswith("MemAvailable:"):
                    return int(_ln.split()[1]) / 1024.0 / 1024.0   # kB -> GB
    except Exception:
        return None
    return None

def _self_rss_gb():
    # ru_maxrss is the high-water mark in KB on Linux.
    return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0

_min_avail_gb = float(os.environ.get("BAQARO_FH_MIN_AVAIL_GB", "200"))
print(f"[mem-guard] BAQARO_FH_MIN_AVAIL_GB = {_min_avail_gb} GB", flush=True)

# --- Main Loop Over Snapshots ---
for i in snapshots:
    # ------------------------------------------------------------------
    # MEMORY GUARD (per-snap, fail fast before OOM)
    # ------------------------------------------------------------------
    _avail_gb = _read_mem_available_gb()
    _rss_gb = _self_rss_gb()
    if _avail_gb is not None:
        print(f"[mem-guard] snap {i} z={redshifts[i]:.2f}  RSS={_rss_gb:.1f} GB  "
              f"MemAvailable={_avail_gb:.1f} GB", flush=True)
        if _avail_gb < _min_avail_gb:
            raise MemoryError(
                f"[mem-guard] MemAvailable={_avail_gb:.1f} GB fell below "
                f"BAQARO_FH_MIN_AVAIL_GB={_min_avail_gb} GB at snap {i} (z={redshifts[i]:.2f}). "
                f"Current RSS={_rss_gb:.1f} GB. Aborting before the OOM-killer fires. "
                f"Reduce the selection (raise BAQARO_LOG_MASS_THRESHOLD, drop "
                f"BAQARO_FH_INTERSECT_SUBSAMPLE, or restrict max_snap) and retry."
            )

    # ------------------------------------------------------------------
    # SNAPSHOT SETUP
    # ------------------------------------------------------------------
    logger.info("#" * 50)
    logger.info("Snapshot {}, redshift {:.2f}".format(i, redshifts[i]))
    logger.info("Time since last snapshot: {:.2f} Myr".format(delta_times_snapshots[i] * 1e3))

    # Calculate number of sub-steps for accretion.
    # Clamp to >=1: when tau_coh > dt, fall through to n_steps=1 (single
    # coherent draw covering the whole snapshot) rather than int()-truncating
    # to 0, which would decouple L_bol from mass build (and also break the
    # division on the next line / np.linspace below).
    n_steps = max(1, int(delta_times_snapshots[i] * 1e3 / time_step_for_accretion))
    logger.info("Number of sub-steps: {}".format(n_steps))
    logger.info("Actual timestep: {:.2f} Myr (target: {} Myr)".format(
        delta_times_snapshots[i] * 1e3 / n_steps, time_step_for_accretion))

    # Build the sub-step time axis (Gyr) for this snapshot interval.
    # Used to assign physical times to each sub-step in the full history.
    # Each sub-step k advances the mass by Δt/n_steps, so the mass AFTER
    # sub-step k lands at t0 + k·Δt/n_steps for k=1..n_steps. Use
    # linspace(t0, t1, n_steps+1)[1:] to get exactly those n_steps end-of-step
    # times (spacing Δt/n_steps); the old linspace(t0, t1, n_steps) gave
    # spacing Δt/(n_steps-1), mis-stamped the first sub-step at the interval
    # start, and duplicated the boundary timestamp across snapshots.
    if i == 0:
        times_Gyr_steps = np.linspace(0.01, cosmo.age(redshifts[i]), n_steps + 1)[1:]
    else:
        times_Gyr_steps = np.linspace(cosmo.age(redshifts[i-1]), cosmo.age(redshifts[i]), n_steps + 1)[1:]

    times_full_history_local = times_Gyr_steps

    # ------------------------------------------------------------------
    # LOAD CURRENT SNAPSHOT DATA
    # ------------------------------------------------------------------
    current_halo_masses = halo_masses_all[i]
    if accretion_on:
        current_rates = halo_specific_cold_accretion_rates_all[i]

    # ------------------------------------------------------------------
    # COMPUTE MASKS FOR THIS SNAPSHOT
    # ------------------------------------------------------------------
    mask_spawning_now = mt_birth_index == i
    mask_spawned_before = (mt_birth_index < i) & (mt_birth_index != -1)
    mask_not_dead_yet = (mt_death_index > i) | (mt_death_index == -1)
    mask_dying_now = mt_death_index == i

    # ----------------------------------------------------------
    # FULL HISTORY ARRAYS FOR THIS SNAPSHOT
    # ----------------------------------------------------------
    # Per-sub-step arrays, shape (n_steps, n_halos) — consistent with the
    # global (n_snapshots, n_halos) layout.  Filled during the accretion
    # step; non-evolving halos stay at zero.  v-stacked after the main loop.
    if STORE_FINE_HISTORY:
        Lbols_full_history_local = np.zeros((n_steps, total_number_of_objects), dtype=np.float32)
        black_hole_masses_full_history_local = np.zeros((n_steps, total_number_of_objects), dtype=np.float32)
        etas_full_history_local = np.zeros((n_steps, total_number_of_objects), dtype=np.float32)

    # ------------------------------------------------------------------
    # STEP 1: SEED NEW BLACK HOLES
    # ------------------------------------------------------------------
    start_time_here = time.time()
    mask_new_BHs = mask_spawning_now
    halo_slice = current_halo_masses[mask_new_BHs]
    spawned_now = spawn_BHs(halo_slice, logfseed, sigmaseed=sigmaseed, rng=rng)
    black_hole_masses_all[i, mask_new_BHs] = spawned_now

    spawned_count = np.count_nonzero(mask_new_BHs)
    logger.info("New black holes {}".format(spawned_count))
    new_black_holes_total += spawned_count
    mass_new_BHs = np.sum(spawned_now)
    logger.info("Initialization done in {:.2f}s".format( time.time() - start_time_here))


    # STEP 2: ACCRETION
    # ------------------------------------------------------------------
    # Evolve existing black holes via gas accretion.
    # Uses evolve_BHs_full_history (numpy path) which returns per-sub-step
    # arrays of shape (n_evolving, n_steps) — needed for lightcurve analysis.
    start_time_here = time.time()
    mask_evolving_BHs = mask_spawned_before & mask_not_dead_yet

    if accretion_on:
        idx_evolving = np.flatnonzero(mask_evolving_BHs)
        n_evolving = len(idx_evolving)

        if n_evolving > 0:
            bh_prev = black_hole_masses_all[i - 1, idx_evolving].copy()
            # Clip bounds match main_evolution.py. Non-accreting default: floor=0
            # lets a zero rate (plateaued LastMaxMass) flow to log10(0) = -inf ->
            # eta = 0 in the ERDF -> M_BH frozen, L_bol = 0, eta = 0 (masked from
            # the band). Verified NaN-clean for the madau+ model. Legacy clip-floor
            # (BAQARO_NONACCRETING_ZERO_RATE=0) uses 1e-8.  [1e5 upper -> 1e3 to
            # match production was a separate consistency fix.]
            _rate_floor = 0.0 if NONACCRETING_ZERO_RATE else 1e-8
            clipped_rates = np.clip(current_rates[idx_evolving], _rate_floor, 1e3)
            with np.errstate(divide="ignore"):
                log_halo_rates = np.log10(clipped_rates)

            # Returns (n_evolving, n_steps) arrays for mass, Lbol, and eta
            black_hole_masses_evolution, Lbols_next_evolution, etas_evolution = evolve_BHs_full_history(
                transfer_function,
                bh_prev,
                delta_times_snapshots[i],
                erdf,
                n_steps=n_steps,
                log_halo_rates_array=log_halo_rates,
                rad_efficiency=rad_efficiency_0,
                rad_efficiency_model=rad_efficiency_model,
                parallelize=parallelize,
                num_workers=num_workers,
                backend=backend,
                growth_max=growth_sum_max,
                madau_feff_correction=madau_feff_correction,
            )

            # Final sub-step → snapshot-level arrays (same as main_evolution.py)
            black_hole_masses_all[i, idx_evolving] = black_hole_masses_evolution[:, -1]
            Lbols_all[i, idx_evolving] = Lbols_next_evolution[:, -1]

            if STORE_FINE_HISTORY:
                # Full sub-step history → local arrays for this snapshot
                # evolve_BHs_full_history returns (n_evolving, n_steps), transpose
                # to write into (n_steps, n_halos) layout.
                Lbols_full_history_local[:, idx_evolving] = Lbols_next_evolution.T
                black_hole_masses_full_history_local[:, idx_evolving] = black_hole_masses_evolution.T
                etas_full_history_local[:, idx_evolving] = etas_evolution.T
            else:
                # Boundaries-only: record the accretion-only boundary mass (last
                # sub-step), 0 for newly-seeded/dying — exactly the export's
                # `full_history/...[last_sub]` slice (bh_acc_end).
                black_hole_masses_acc_only_all[i, idx_evolving] = black_hole_masses_evolution[:, -1]
        else:
            idx_evolving = np.array([], dtype=np.intp)

    else:
        # No accretion: just copy masses from previous snapshot
        idx_evolving = np.flatnonzero(mask_evolving_BHs)
        black_hole_masses_all[i, idx_evolving] = black_hole_masses_all[i - 1, idx_evolving]
        if not STORE_FINE_HISTORY:
            black_hole_masses_acc_only_all[i, idx_evolving] = black_hole_masses_all[i - 1, idx_evolving]

    evolving_count = len(idx_evolving)
    logger.info("Evolving black holes: {}".format(evolving_count))
    logger.info("Accretion done in {:.2f}s".format(time.time() - start_time_here))

    # ------------------------------------------------------------------
    # STEP 3: MERGERS AND DISRUPTIONS
    # ------------------------------------------------------------------
    # Handle halos that die this snapshot:
    #   - Merged halos: add their BH mass to the merger target's BH
    #   - Disrupted halos: BH mass is lost (set to zero)
    if merger_on:
        start_time_here = time.time()
        mask_merging_BHs = mask_spawned_before & mask_dying_now & mask_merged

        src_ids = mt_track_ids[mask_merging_BHs]
        dest_ids = mt_merger_ids[mask_merging_BHs]

        # Filter out mergers where the target halo isn't resolved yet OR is
        # already dead: halo mass is frozen LastMaxMass
        # ("ever resolved"), so a long-dead target would swallow the source
        # BH into a row that is never carried forward. Dying-at-this-snap
        # targets are kept (they are sources; chain resolution forwards).
        dest_death = mt_death_index[dest_ids]
        mask_halos_are_alive = (current_halo_masses[dest_ids] > 0.0) & (
            (dest_death == -1) | (dest_death >= i)
        )
        src_ids_not_merged = src_ids[~mask_halos_are_alive]
        mass_from_aborted_mergers = np.sum(
            black_hole_masses_all[i - 1, src_ids_not_merged], dtype=np.float64
        )

        src_ids = src_ids[mask_halos_are_alive]
        dest_ids = dest_ids[mask_halos_are_alive]

        if print_debug_info:
            print("Mass pre mergers", np.sum(black_hole_masses_all[i, np.unique(dest_ids)]))
            copy_debug = np.copy(black_hole_masses_all[i, np.unique(dest_ids)])
        else:
            copy_debug = None

        # --- Resolve Merger Chains ---
        if len(src_ids) > 0:
            mapping = np.full(black_hole_masses_all.shape[1], -1, dtype=dest_ids.dtype)
            mapping[src_ids] = dest_ids
            final_dest_ids = dest_ids.copy()
            for _ in range(len(src_ids)):
                next_dest_ids = mapping[final_dest_ids]
                mask_chain = next_dest_ids != -1
                if not np.any(mask_chain):
                    break
                final_dest_ids[mask_chain] = next_dest_ids[mask_chain]
            dest_ids = final_dest_ids

        # --- Accumulate Merged Masses ---
        np.add.at(black_hole_masses_all[i], dest_ids, black_hole_masses_all[i - 1, src_ids])

        if print_debug_info:
            print("Amount of mass merged", np.sum(black_hole_masses_all[i - 1, src_ids]))
            print("Mass post mergers", np.sum(black_hole_masses_all[i, np.unique(dest_ids)]),
                  " (should be equal to: {})".format(
                      np.sum(black_hole_masses_all[i - 1, src_ids]) + np.sum(copy_debug)))

        merged_black_holes_total += src_ids.size
        logger.info("Merged black holes: {}".format(src_ids.size))

        # --- Handle Disrupted Halos ---
        mask_lost_BHs = mask_spawned_before & mask_dying_now & mask_disrupted
        lost_count = np.count_nonzero(mask_lost_BHs)
        lost_black_holes_total += lost_count
        mass_lost_BHs = np.sum(black_hole_masses_all[i - 1, mask_lost_BHs], dtype=np.float64)

        logger.info("Disrupted black holes: {}".format(lost_count))
        logger.info("Mergers + disruptions done in {:.2f}s".format(time.time() - start_time_here))

    # ------------------------------------------------------------------
    # DEBUG: MASS CONSERVATION CHECK
    # ------------------------------------------------------------------
    if print_debug_info:
        total_mass_BHs = np.sum(black_hole_masses_all[i], dtype=np.float64)
        print("#" * 50)
        print("Total mass in black holes:", total_mass_BHs)

        mass_before = np.sum(black_hole_masses_all[i - 1], dtype=np.float64)
        mass_after = np.sum(black_hole_masses_all[i], dtype=np.float64)
        spawned = np.sum(mass_new_BHs, dtype=np.float64) if np.ndim(mass_new_BHs) else float(mass_new_BHs)
        lost_disrupted = mass_lost_BHs

        effective_lost = lost_disrupted + mass_from_aborted_mergers

        accreted    = (mass_after - mass_before) - (spawned - effective_lost)

        print(f"Snapshot {i}: mass_before={mass_before:.6g}, mass_after={mass_after:.6g}")
        print(f"Snapshot {i}: spawned_mass={spawned:.6g}, lost_disrupted_mass(i-1)={lost_disrupted:.6g}, mass_from_aborted_mergers(i-1)={mass_from_aborted_mergers:.6g}")
        print(f"Snapshot {i}: effective_lost_mass(i-1)={effective_lost:.6g}")
        print(f"Snapshot {i}: calculated 'accreted' mass = {accreted:.6g}")

        tol = max(1e-8 * mass_after, 1e-3)
        if accretion_on is False and abs(accreted) > tol:
            print(f"Warning: unexpected mass change at snapshot {i}: delta_M_total={mass_after-mass_before:.6g}, spawned_M - effective_lost_M={spawned-effective_lost:.6g}, diff(accreted)={accreted:.6g}, tol={tol:.3g}")

    # ------------------------------------------------------------------
    # STORE FULL HISTORY ARRAYS FOR THIS SNAPSHOT
    # ------------------------------------------------------------------
    # Append per-snapshot (n_steps_i, n_halos) arrays to lists.
    # These are v-stacked after the main loop into (total_steps, n_halos).
    # Boundaries-only mode skips this entirely (the ~1 TB accumulation) — only
    # the per-snapshot accretion-only boundary masses are kept.
    if STORE_FINE_HISTORY:
        Lbols_full_history.append(Lbols_full_history_local)
        black_hole_masses_full_history.append(black_hole_masses_full_history_local)
        etas_full_history.append(etas_full_history_local)
        times_full_history.append(times_full_history_local)

logger.info("END OF BH EVOLUTION ROUTINE")
logger.info("TOTAL TIME ELAPSED: {:.1f} seconds".format(time.time() - start_time_bh))

if print_debug_info:
    logger.info("--- FINAL SUMMARY ---")
    logger.info("Total mass in black holes: {:.4e}".format(np.sum(black_hole_masses_all[-1])))
    logger.info("BHs spawned: {}".format(new_black_holes_total))
    logger.info("BHs merged: {}".format(merged_black_holes_total))
    logger.info("BHs disrupted: {}".format(lost_black_holes_total))
    logger.info("Net BHs (spawned - disrupted): {}".format(new_black_holes_total - lost_black_holes_total))


# ==============================================================================
# UNIT CONVERSION
# ==============================================================================
logger.info("Rescaling to physical units")

mass_units_f32 = np.float32(mass_units)
black_hole_masses_all *= mass_units_f32
Lbols_all *= mass_units_f32

# Halo masses: after subsetting these are in-RAM arrays, not mmap
halo_masses_all = halo_masses_all * mass_units_f32

# Concatenate full history across snapshots:
#   Per-snapshot arrays are (n_steps_i, n_halos) with varying n_steps_i.
#   vstack joins them along axis=0 → (total_steps, n_halos).
#   times is 1-D → plain concatenate.
# Boundaries-only mode skips this (no fine arrays accumulated); it instead
# converts the snapshot-resolution accretion-only boundary masses to solar units.
if STORE_FINE_HISTORY:
    Lbols_global_evolution = np.vstack(Lbols_full_history)
    black_hole_masses_global_evolution = np.vstack(black_hole_masses_full_history)
    etas_global_evolution = np.vstack(etas_full_history)
    times_global_evolution = np.concatenate(times_full_history)

    # Convert full history arrays from code units to solar units
    Lbols_global_evolution *= mass_units_f32
    black_hole_masses_global_evolution *= mass_units_f32
else:
    black_hole_masses_acc_only_all *= mass_units_f32


# ==============================================================================
# SAVE OUTPUT
# ==============================================================================
logger.info("Black hole mass array size: {:.2f} GB".format(black_hole_masses_all.nbytes / 1e9))
logger.info("Luminosity array size: {:.2f} GB".format(Lbols_all.nbytes / 1e9))

time_here = time.time()
logger.info("Saving the arrays to an HDF5 file")
with h5py.File(path_file, "w") as file:
    file.create_dataset("snapshots", data=snapshots)
    file.create_dataset("redshifts", data=redshifts)
    file.create_dataset("ages_of_the_universe", data=ages_of_the_universe)
    file.create_dataset("delta_times_snapshots", data=delta_times_snapshots)

    # Main arrays: already in (n_snapshots, n_halos) layout
    file.create_dataset("halo_masses_all", data=halo_masses_all)
    file.create_dataset("black_hole_masses_all", data=black_hole_masses_all)
    file.create_dataset("Lbols_all", data=Lbols_all)

    file.create_group("merger_trees")
    file.create_dataset("merger_trees/track_ids", data=mt_track_ids)
    file.create_dataset("merger_trees/track_ids_original", data=mt_track_ids_original)
    file.create_dataset("merger_trees/merger_track_ids", data=mt_merger_ids)
    file.create_dataset("merger_trees/snapshot_indexes_of_birth", data=mt_birth_index)
    file.create_dataset("merger_trees/snapshot_indexes_of_death", data=mt_death_index)

    # Save subset selection info if applicable
    if subset_mask is not None:
        grp = file.create_group("subset_selection")
        grp.attrs["mode"] = subset_selection_mode
        grp.attrs["tag"] = selection_tag
        grp.attrs["close_trees"] = bool(close_trees)
        grp.attrs["mass_threshold_Msun"] = float(subset_params["mass_threshold"])
        grp.attrs["n_first_born"] = int(subset_params["n_first_born"])
        grp.attrs["n_random"] = int(subset_params["n_random"])
        grp.attrs["random_seed"] = int(subset_params["random_seed"])
        file.create_dataset("subset_selection/mode", data=subset_selection_mode)
        file.create_dataset("subset_selection/original_indices", data=subset_new_indices)
        file.create_dataset("subset_selection/n_targets", data=len(target_track_ids))
        file.create_dataset("subset_selection/n_selected", data=n_selected)
        file.create_dataset("subset_selection/original_targets_new_indices", data=original_targets_new_indices)

    if STORE_FINE_HISTORY:
        # Full history arrays: already in (total_steps, n_halos) layout — consistent
        # with the global (n_snapshots, n_halos) convention. No transpose needed.
        # times_full_history is 1-D (total_steps,) — shared across all halos.
        file.create_dataset("full_history/Lbols_full_history", data=Lbols_global_evolution)
        file.create_dataset("full_history/black_hole_masses_full_history", data=black_hole_masses_global_evolution)
        file.create_dataset("full_history/etas_full_history", data=etas_global_evolution)
        file.create_dataset("full_history/times_full_history", data=times_global_evolution)
    else:
        # Boundaries-only mode: snapshot-resolution accretion-only boundary mass
        # (n_snap, n_halos) — the fine-history-derived quantity downstream
        # consumers need (bh_acc_end → merger/seed injection). They read this
        # if present, else fall back to full_history/...[last_sub].
        file.create_dataset("black_hole_masses_acc_only_all", data=black_hole_masses_acc_only_all)
        file.attrs["fh_store_fine"] = False

    file.create_group("parameters")
    file.create_dataset("parameters/erdf_model", data=erdf_model)
    erdf_grp = file.require_group("parameters/erdf_params")
    for key, val in erdf_params_dict.items():
        erdf_grp.attrs[key] = val
    file.create_dataset("parameters/logfseed", data=logfseed)
    if sigmaseed is not None:
        file.create_dataset("parameters/sigmaseed", data=sigmaseed)
    file.create_dataset("parameters/time_step_for_accretion", data=time_step_for_accretion)
    file.create_dataset("parameters/growth_sum_max", data=growth_sum_max)
    file.create_dataset("parameters/rad_efficiency", data=rad_efficiency_0)
    file.create_dataset("parameters/rad_efficiency_model", data=rad_efficiency_model)
    file.create_dataset("parameters/nbound_threshold", data=nbound_threshold)
    file.create_dataset("parameters/halo_filtering_mode", data=halo_filtering_mode)

    file.create_group("metadata")
    file.create_dataset("metadata/boxsize", data=boxsize)
    file.create_dataset("metadata/N_particles_per_side", data=N_particles_per_side)
    file.create_dataset("metadata/mass_resolution", data=mass_resolution)
    file.create_dataset("metadata/mass_units", data=mass_units)
    file.create_dataset("metadata/halo_mass_units_hbt", data=halo_mass_units_hbt)
    file.create_dataset("metadata/rng_seed", data=rng.bit_generator._seed_seq.entropy)

    # --- Self-describing run provenance (see utils/provenance.py) ---
    # Read f["provenance"].attrs (or f.attrs) later to recover exactly how this
    # full-history run was produced: identity tokens + selection + the 6 params
    # + git revision + the full BAQARO_* env snapshot.
    write_run_provenance(file, {
        "engine": "main_evolution_full_history",
        "simulation_name": simulation_name,
        "max_snap": int(max_snap),
        "max_snap_halos": int(max_snap_halos),
        "min_snap": int(min_snap),
        "erdf_model": erdf_model,
        "fold_subhalo_mass": bool(fold_subhalo_mass),
        "merger_delay_mode": merger_delay_mode,
        "tdyn_fraction_default": float(tdyn_fraction_default),
        "nbound_threshold": int(nbound_threshold),
        "halo_filtering_mode": halo_filtering_mode,
        "notes_file": notes_file,
        "bestfit_name": bestfit_name,
        "name_file": name_file,
        "selection_tag": selection_tag,
        "subset_selection_mode": str(subset_selection_mode),
        "mass_threshold_Msun": float(subset_params["mass_threshold"]),
        "n_first_born": int(subset_params["n_first_born"]),
        "n_random": int(subset_params["n_random"]),
        "subsample_random_seed": int(subset_params["random_seed"]),
        "close_trees": bool(close_trees),
        "merger_on": bool(merger_on),
        "accretion_on": bool(accretion_on),
        "rad_efficiency_model": rad_efficiency_model,
        "rad_efficiency_0": float(rad_efficiency_0),
        "time_step_for_accretion_Myr": float(time_step_for_accretion),
        "logtcoherence": float(logtcoherence),
        "log_eta_mean_0": float(erdf_params_dict["log_eta_mean_0"]),
        "log_eta_mean_evol": float(erdf_params_dict["log_eta_mean_evol"]),
        "std_0": float(erdf_params_dict["std_0"]),
        "logfseed": float(logfseed),
        "sigmaseed": float(sigmaseed) if sigmaseed is not None else None,
    })


logger.info("Time to save the arrays {} seconds".format( time.time() - time_here))
logger.info("Total time elapsed for the run {} seconds".format( time.time() - start_time_global))
