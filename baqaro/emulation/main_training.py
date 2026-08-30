




"""
Emulator Training Data Generation
==================================

This script generates training data for Gaussian Process emulation of black hole
population statistics by running the BH evolution model across a Latin Hypercube
Sampling (LHS) grid of ERDF parameters.

Pipeline Overview
-----------------
1. Load precomputed halo mass histories and merger trees from disk
2. Generate LHS parameter samples for ERDF model parameters
3. For each parameter combination:
   - Run BH evolution (seeding, accretion, mergers) across all snapshots
   - Compute summary statistics at selected redshifts:
     * QLF: Quasar Luminosity Function
     * BHMF: Black Hole Mass Function (all BHs, no luminosity cut)
     * CERDF: Conditional Eddington Ratio Distribution Function (L > threshold)
     * QHMF: Quasar Host Halo Mass Function (L > threshold)
4. Save all statistics and parameter samples to HDF5 for emulator training

Output Array Shapes
-------------------
- log_qlfs:   (n_redshifts, n_simulations, n_Lbins)
- log_bhmfs:  (n_redshifts, n_simulations, n_Mbins)
- log_cerdfs: (n_redshifts, n_L_thresholds, n_simulations, n_eta_bins)
- log_qhmfs:  (n_redshifts, n_L_thresholds, n_simulations, n_Mhalo_bins)

Notes
-----
- Arrays are reused across LHS runs to minimize memory allocation
- Histogram bins and normalizations are precomputed once for efficiency
- Halo data loaded as memory-mapped arrays to reduce RAM footprint

Z0 variant differences from main pipeline
------------------------------------------
- BHMF (all BHs) replaces QBHMF (luminosity-threshold-dependent):
  shape ``(n_z, n_mbins)`` instead of ``(n_z, n_Lthr, n_mbins)``.
- No Numba parallelization config (prange doesn't survive fork).
- No ``os.environ`` thread-limiting variables.
- Preallocated statistics buffers in ``run_single_model.py``
  to reduce GC pressure (~3.2 GB/snapshot × 13 snapshots = ~40 GB saved).
- Extra snapshots (63, 73), 40 mass bins, BHMF range extended to 10^11.
"""

# This module is a run script, not a library: its body loads catalogues and
# writes products at import time. Refuse a plain ``import`` so nobody starts a
# multi-hour job by accident (``python -m baqaro.emulation.main_training`` sets __name__ to
# "__main__"; a multiprocessing "spawn" child re-imports it as "__mp_main__").
if __name__ not in ("__main__", "__mp_main__"):
    raise RuntimeError(
        "baqaro.emulation.main_training is an entry-point script; run it with "
        "'python -m baqaro.emulation.main_training' instead of importing it."
    )


import os

import time
import h5py
import numpy as np

from qhtools.utils.cosmology import cosmo
from qhtools.utils import natconst as nc, my_utils

from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader

from baqaro.emulation.run_training_set import run_training_batch, run_training_batch_parallel
from baqaro.emulation.writing_helpers import init_training_file, write_training_metadata
from baqaro.utils.logging import  set_logger
from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
from baqaro.utils.local_utils import get_mass_resolution_simulation
from baqaro.utils.my_units import mass_units, halo_mass_units_hbt
from baqaro.utils.provenance import write_run_provenance

from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler




from numpy.random import default_rng


# Module-level configuration
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # site from the environment


rng_training = default_rng(847594)  # for the emulation parameters, testing run
rng_training_local = default_rng(235423)  # for the local box sampling, testing run
source_dir = DEFAULT_SOURCE_DIR




# SETTING THE BASIC PARAMETERS FOR THE RUNS

# --- Run Identification ---
num_simulations = int(os.environ.get("BAQARO_NUM_SIMULATIONS", "256"))
num_parameters = 6

param_definitions_emulation = {
    "log_eta_mean_0": [-2.5, -0.5],  # Mean log(Eddington ratio) at cold sSAR=0
    "log_eta_mean_evol": [0.2, 2.0],  # Dependence of mean log(Eddington ratio) on the specific cold accretion rate (sSAR)
    "std_0": [0.2, 1.0],   # Scatter at cold sSAR=0 (dex)
    # "std_evol": [-0.3, 0.3],   # Redshift evolution of scatter
    "logtcoherence": [3.5, 7.3],  # log10(coherence time in yr)
    "logfseed": [-7.5, -3.5],     # log10(M_BH / M_halo) at seeding
    "sigmaseed": [0.2, 1.5],      # dex scatter on logfseed (0 = deterministic seeding)
}

# Optional env-var override of prior bounds (per-param lo/hi). Set
# BAQARO_PRIOR_<PARAM>_LO / _HI in upper-case to narrow the Sobol box
# without editing this file. Useful for narrowed-prior retrains.
_PRIOR_ENV_KEYS = {
    "log_eta_mean_0":    ("BAQARO_PRIOR_LOG_ETA_MEAN_0_LO",    "BAQARO_PRIOR_LOG_ETA_MEAN_0_HI"),
    "log_eta_mean_evol": ("BAQARO_PRIOR_LOG_ETA_MEAN_EVOL_LO", "BAQARO_PRIOR_LOG_ETA_MEAN_EVOL_HI"),
    "std_0":             ("BAQARO_PRIOR_STD_0_LO",             "BAQARO_PRIOR_STD_0_HI"),
    "logtcoherence":     ("BAQARO_PRIOR_LOGTCOHERENCE_LO",     "BAQARO_PRIOR_LOGTCOHERENCE_HI"),
    "logfseed":          ("BAQARO_PRIOR_LOGFSEED_LO",          "BAQARO_PRIOR_LOGFSEED_HI"),
    "sigmaseed":         ("BAQARO_PRIOR_SIGMASEED_LO",         "BAQARO_PRIOR_SIGMASEED_HI"),
}
_prior_changed = False
for _pk, (_envlo, _envhi) in _PRIOR_ENV_KEYS.items():
    _lo = os.environ.get(_envlo)
    _hi = os.environ.get(_envhi)
    if _lo is not None:
        param_definitions_emulation[_pk][0] = float(_lo)
        _prior_changed = True
    if _hi is not None:
        param_definitions_emulation[_pk][1] = float(_hi)
        _prior_changed = True
if _prior_changed:
    print("Prior bounds OVERRIDDEN via BAQARO_PRIOR_* env vars:")
    for _pk, _rng in param_definitions_emulation.items():
        print(f"  {_pk:<22} [{_rng[0]:>+7.3f}, {_rng[1]:>+7.3f}]")

# Tag appended to the training-file name.
from baqaro.utils.sim_config import FIDUCIAL_NOTES_TRAINING
#
# WRITER/READER ENV-VAR SPLIT. Reading only `BAQARO_NOTES_FILE` here would be a
# trap, because EVERY downstream reader of the training file
# (main_emulation, cross_validate_emulators, testing_performance, emulator_vs_real,
# inference/comparison_config, inference/main_mcmc) reads `BAQARO_NOTES_FILE_TRAINING`
# — two different names for the same token, so training with one and reading with
# the other resolved DIFFERENT files.
#
# Now `BAQARO_NOTES_FILE_TRAINING` is accepted here too and takes precedence, so the
# SAME variable works end-to-end. `BAQARO_NOTES_FILE` still works (legacy, and it is
# what the batch scripts set) but is now the fallback.
#
# NB the fallback is deliberately one-directional: readers must NOT fall back to
# `BAQARO_NOTES_FILE`, because in `core_functions/main_evolution.py` that same var
# means the *forward-run* notes — a different file entirely.
# Bare runs fall back to the FIDUCIAL notes token, so that a no-env invocation
# resolves to the same file every reader resolves to (main_emulation and friends
# already default this way). Because that makes a bare run TARGET A REAL
# PRODUCTION FILE, it is paired with the guard further below.
_notes_training_env = os.environ.get("BAQARO_NOTES_FILE_TRAINING")
_notes_legacy_env = os.environ.get("BAQARO_NOTES_FILE")
_notes_from_env = (_notes_training_env if _notes_training_env is not None
                   else _notes_legacy_env) or None
notes_file = _notes_from_env or FIDUCIAL_NOTES_TRAINING or None

if _notes_training_env is None and _notes_legacy_env:
    print(f"[main_training] NOTE: notes token '{notes_file}' came from the LEGACY "
          f"BAQARO_NOTES_FILE. Downstream readers use BAQARO_NOTES_FILE_TRAINING — "
          f"run them with BAQARO_NOTES_FILE_TRAINING={notes_file} (or just use "
          f"BAQARO_NOTES_FILE_TRAINING here too; it now takes precedence).")
elif (_notes_training_env is not None and _notes_legacy_env
        and _notes_training_env != _notes_legacy_env):
    print(f"[main_training] ⚠ BAQARO_NOTES_FILE_TRAINING='{_notes_training_env}' and "
          f"BAQARO_NOTES_FILE='{_notes_legacy_env}' DISAGREE. Using "
          f"'{_notes_training_env}' (TRAINING wins).")

# --- Snapshot Range ---
min_snap = 0
from baqaro.utils.sim_config import (
    max_snap, env_bool,
    FIDUCIAL_GROWTH_SUM_MAX, FIDUCIAL_MADAU_FEFF_CORRECTION,
)


# --- ERDF (Eddington Ratio Distribution Function) ---
# Controls the distribution of accretion rates relative to Eddington
erdf_model = "log_normal_evol_halo_mass"

# --- Seeding ---
# Default values used as config fallback; overridden per-run in 6D mode
logfseed = -4.2  # log10(M_BH / M_halo) at seeding
sigmaseed = 0.3  # dex scatter on logfseed (None -> deterministic seeding)

# --- Time Resolution ---
# Default value; overridden per-run by sampled logtcoherence in 6D mode
time_step_for_accretion = 7  # Sub-stepping timestep in Myr


# --- Physics Toggles ---
accretion_on = True   # Enable black hole accretion
merger_on = True      # Enable black hole mergers


# --- Radiative Efficiency ---
rad_efficiency_0 = 0.1          # Base radiative efficiency (epsilon)
rad_efficiency_model = "madau+"  # Options: "constant", "madau+" (spin-dependent)
# Madau f_eff correction. OFF by default so existing
# training files are bit-identical; enable with BAQARO_MADAU_FEFF_CORRECTION. When
# on, the training filename gets a _feffcorr token (corrected emulators must not
# mix with uncorrected ones). See core_functions/madau_feff.py.
madau_feff_correction = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION",
                                        "1" if FIDUCIAL_MADAU_FEFF_CORRECTION else "0").strip().lower()
                         not in ("0", "", "false", "no", "off"))
if madau_feff_correction != FIDUCIAL_MADAU_FEFF_CORRECTION:
    print(f"[MADAU-FEFF] training with BAQARO_MADAU_FEFF_CORRECTION="
          f"{'on' if madau_feff_correction else 'off'} -> NOT the fiducial "
          f"setting ({'on' if FIDUCIAL_MADAU_FEFF_CORRECTION else 'off'}); "
          "must match the forward run.")

# Per-snapshot growth-sum cap (numerical guardrail mode), mirrored from
# core_functions/main_evolution.py so a capped emulator can be trained
# consistently with a capped forward run. Default 50.0 effectively never bites
# (exp(50) ~ 5e21); lowering it bounds the per-snapshot mass-growth factor at
# exp(growth_max) (e.g. 4.6 -> 100x, 3.91 -> 50x, 2.3 -> 10x). When != 50.0 the
# training filename gets a _g{cap} token so capped products never overwrite the
# fiducial. IMPORTANT: any value other than FIDUCIAL_GROWTH_SUM_MAX CHANGES the
# model and must match the forward run's BAQARO_GROWTH_SUM_MAX; it requires
# re-DE + re-MCMC.
growth_sum_max = float(os.environ.get("BAQARO_GROWTH_SUM_MAX", str(FIDUCIAL_GROWTH_SUM_MAX)))
if growth_sum_max != FIDUCIAL_GROWTH_SUM_MAX:
    print(f"[GROWTH-CAP MODE] training with BAQARO_GROWTH_SUM_MAX={growth_sum_max} "
          f"-> per-snapshot M_BH growth bounded at exp({growth_sum_max}) "
          f"~ {np.exp(growth_sum_max):.2e}x. "
          f"NOT the fiducial cap ({FIDUCIAL_GROWTH_SUM_MAX}); "
          "must match the forward run's cap.")

# --- Halo Selection ---
nbound_threshold = 40       # Minimum bound particles for a resolved halo
halo_filtering_mode = "global"  # Options: "local", "global"

# Must match the saver run that produced the halo files we load below.
# See get_merger_trees() for the merger modes.
from baqaro.utils.sim_config import fold_subhalo_mass, merger_delay_mode

# tdyn_fraction_default is sim-dependent (sim_config.py). 0.20 at
# L2800N5040, 0.25 at L2800N10080.
from baqaro.utils.sim_config import tdyn_fraction_default


# --- Subsampling (off by default; preserves production behaviour unless enabled) ---
# Enable via ``BAQARO_USE_SUBSAMPLE=1``. When on, every training run
# evolves only the stratified subset of merger tree roots (+ closure)
# instead of the full halo catalogue. Per-halo weights propagated
# through histograms so saved summary stats are unbiased in expectation.
# The output HDF5 filename auto-includes a subset tag.
#
# ON by default, matching core_functions/main_evolution.py, so a bare run
# reproduces the fiducial training set. The full catalogue at z=0 needs ~2.6 TB
# of private RAM and is not runnable on the production nodes, so defaulting this
# off meant a bare invocation started an infeasible run AND resolved a filename
# with no `_sub_` token. Set BAQARO_USE_SUBSAMPLE=0 deliberately for full-N.
use_subsample = env_bool("BAQARO_USE_SUBSAMPLE", True)
subsample_root_snap = max_snap

# --- Chunked multi-node training (additive; default = single-node) ---
# Split the subsample's roots into BAQARO_N_CHUNKS disjoint groups (one per
# job-array task) and run only BAQARO_CHUNK_ID here. n_chunks=1 (default)
# is the exact single-node path: no partition applied, no filename change.
# The per-chunk training HDF5s are recombined afterwards by
# emulation/pool_chunks.py.
n_chunks = int(os.environ.get("BAQARO_N_CHUNKS", "1"))
chunk_id = int(os.environ.get("BAQARO_CHUNK_ID", "0"))
if n_chunks < 1:
    raise ValueError(f"BAQARO_N_CHUNKS={n_chunks} must be >= 1")
if not (0 <= chunk_id < n_chunks):
    raise ValueError(
        f"BAQARO_CHUNK_ID={chunk_id} out of range for BAQARO_N_CHUNKS={n_chunks}"
    )
if n_chunks > 1 and not use_subsample:
    raise ValueError(
        "Chunked training (BAQARO_N_CHUNKS>1) requires BAQARO_USE_SUBSAMPLE=1 "
        "in Phase 1 — the full-catalogue chunk path is not implemented yet."
    )
if n_chunks > 1:
    print(f"CHUNKED training: chunk {chunk_id} of {n_chunks}")

# Default N_b=500_000 uniform with keep_above=None — empirically best after
# the 4-way bench: ~27% less RAM than the previous (5k + keep_above=13)
# config, matches the full-sim bright-end QLF/BHMF within statistical noise
# everywhere, and crushes the z=3 logM=10.5 BHMF outlier of the old default.
# See `core_functions/tree_subsample.py` for the subsampling policy and
# table that drove this choice. Override either with the env vars below.
# Subsample geometry + per-bin N_b + keep-above, resolved via the single
# canonical parser resolve_subsample_params() in tree_subsample (same env vars,
# byte-identical to the former inline block). One source of truth so the subset
# TAG (resolve_subset_tag) and the actual BUILD params can never drift apart.
from baqaro.core_functions.tree_subsample import resolve_subsample_params
_ss = resolve_subsample_params()
subsample_N_b_per_bin = _ss["N_b"]
subsample_log_M_lo = _ss["log_M_lo"]
subsample_log_M_hi = _ss["log_M_hi"]
subsample_n_bins = _ss["n_bins"]
subsample_rng_seed = _ss["seed"]
subsample_keep_all_above_log_M = _ss["keep_all_above_log_M"]
if not np.isscalar(subsample_N_b_per_bin):
    _bin_lo = np.linspace(subsample_log_M_lo, subsample_log_M_hi, subsample_n_bins + 1)[:-1]
    print("Variable N_b schedule per bin (log_M_lo -> N_b):")
    for _lo, _nb in zip(_bin_lo, subsample_N_b_per_bin):
        print(f"  {_lo:.2f} -> {int(_nb):>10,}")


# ==============================================================================
# INITIALIZATION
# ==============================================================================

# --- Initialize ERDF Model ---
# these are just placeholder values, will be updated in the loop over the emulation parameters
erdf_params_dict = {
    "log_eta_mean_0": -1.2,    # Mean log(Eddington ratio) at cold sSAR=0
    "log_eta_mean_evol": 0.9,  # Redshift evolution of mean
    "std_0": 0.52,              # Scatter at cold sSAR=0 (dex)
    # "std_evol": -0.08           # Redshift evolution of scatter
}

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
name_file = "{}_erdf_{}_maxsnap_{}".format(
    simulation_name, erdf_model, max_snap
    )

if fold_subhalo_mass:
    name_file += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file += "_{}".format(merger_delay_mode)

if notes_file is not None:
    name_file += "_{}".format(notes_file)

# Append subset tag so subsampled training sets land in different files
# from production full-sim runs.
if use_subsample:
    from baqaro.core_functions.tree_subsample import make_subset_tag
    subset_tag = make_subset_tag(
        root_snap=subsample_root_snap,
        N_b=subsample_N_b_per_bin,
        n_bins=subsample_n_bins,
        log_M_lo=subsample_log_M_lo,
        log_M_hi=subsample_log_M_hi,
        seed=subsample_rng_seed,
        keep_all_above_log_M=subsample_keep_all_above_log_M,
        chunk_id=chunk_id,
        n_chunks=n_chunks,
    )
    name_file += f"_sub_{subset_tag}"

# Tag the output filename when running in growth-cap mode so capped training
# products do not silently overwrite the fiducial. Default 50.0 -> no token.
# Same compact format + token order (g before feffcorr) as main_evolution.py.
if growth_sum_max != 50.0:
    if float(growth_sum_max).is_integer():
        _g_tok = f"g{int(growth_sum_max)}"
    else:
        _g_tok = f"g{growth_sum_max:g}"
    name_file += f"_{_g_tok}"

if madau_feff_correction:
    name_file += "_feffcorr"

path_file = os.path.join(path_out, "training", f"training_data_emulation_{name_file}.hdf5")
path_log = os.path.join(path_out, "logs", f"training_data_emulation_{name_file}.log")

# ==============================================================================
# APPEND GUARD — protects an existing training set from an accidental bare run
# ==============================================================================
# `init_training_file` APPENDS into an existing compatible file (that is the
# resume path, and it is how the production sets were built). Combined with the
# fiducial notes fallback above, a bare `python -m ...main_training` would
# therefore start writing new models straight into the PRODUCTION training set.
#
# So: if the notes token came from the FIDUCIAL DEFAULT rather than from the
# environment, and the target already exists, refuse. Naming the file explicitly
# (BAQARO_NOTES_FILE_TRAINING=...) is treated as "yes, I mean that file" and
# resumes exactly as before, so no existing workflow changes.
if (_notes_from_env is None
        and os.path.exists(path_file)
        and not env_bool("BAQARO_ALLOW_TRAINING_APPEND", False)):
    _sz = os.path.getsize(path_file) / 1e6
    raise SystemExit(
        f"\n[APPEND GUARD] The fiducial training set already exists "
        f"({_sz:.0f} MB):\n    {path_file}\n"
        "This run resolved its notes token from the FIDUCIAL DEFAULT, not from "
        "the environment, so it would APPEND new models into that production "
        "file.\n"
        f"  - To extend it deliberately: BAQARO_NOTES_FILE_TRAINING={notes_file}\n"
        "  - To build a separate set:   BAQARO_NOTES_FILE_TRAINING=<new_tag>\n"
        "  - To bypass this guard:      BAQARO_ALLOW_TRAINING_APPEND=1\n"
    )

# Archiving an existing file is destructive-adjacent: say so loudly.
if env_bool("BAQARO_TRAINING_OVERWRITE", False) and os.path.exists(path_file):
    print(f"[main_training] \u26a0 BAQARO_TRAINING_OVERWRITE=1 and {os.path.basename(path_file)} "
          f"exists: on a SCHEMA MISMATCH the existing file will be ARCHIVED (renamed) "
          f"and a fresh one started. Check you are not shadowing a production set.")

logger = set_logger(path_log, print_to_console=False)


# ==============================================================================
# LOAD PRECOMPUTED HALO DATA
# ==============================================================================
start_time_global = time.time()

# --- Redshift and Time Arrays ---
snapshots = np.arange(min_snap, max_snap + 1)
redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]
ages_of_the_universe = cosmo.age(redshifts)  # Gyr
delta_times_snapshots = np.diff(ages_of_the_universe, prepend=0.01)  # Gyr

# --- Build File Paths for Precomputed Data ---

name_file_halos = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
    simulation_name, max_snap, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
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

# --- Load Halo Masses (Memory-Mapped) ---
# mmap_mode='r' keeps data on disk; forked workers share the OS page cache.
# Thread oversubscription (not mmap) was the cause of I/O contention.
print("Loading halo masses from", path_file_halo_masses)
halo_masses_all = np.load(path_file_halo_masses, mmap_mode='r')

# --- Load Merger Trees ---
print("Loading merger trees from", path_file_trees)
merger_trees = MergerTreeLoader(path_file_trees)

# --- Load Cold Gas Accretion Rates (Memory-Mapped) ---
print("Loading accretion rates from", path_file_accretion)
halo_specific_cold_accretion_rates_all = np.load(path_file_specific_cold_accretion, mmap_mode='r')

# Do not clip in-place: memmap is read-only. Clip per-snapshot on slices instead.

# --- Prefetch 1D Merger Tree Arrays into RAM ---
# These are accessed every iteration; loading once avoids disk seeks
print("Prefetching 1D merger tree arrays into RAM...")
mt_birth_index = np.array(merger_trees.snapshot_indexes_of_birth)  # Snapshot of halo birth
mt_death_index = np.array(merger_trees.snapshot_indexes_of_death)  # Snapshot of halo death (-1 = alive)
mt_track_ids = np.array(merger_trees.track_ids)                    # Unique halo track ID
mt_merger_ids = np.array(merger_trees.merger_track_ids)            # Track ID of merger target (-1 = disrupted)
print("Prefetch complete.")

# --- Precompute Constant Masks ---
# These depend only on mt_merger_ids which never changes
mask_merged = mt_merger_ids != -1    # Halos that merge into another
mask_disrupted = mt_merger_ids == -1  # Halos that are tidally disrupted


# ==============================================================================
# LOAD OR BUILD SUBSAMPLE SUBSET
# ==============================================================================
# v3 BFS + eager-load. See core_functions/tree_subsample.py (
# entry) for the design and the load-bearing gotcha about HBT's
# monotonic LastMaxMass.
if use_subsample:
    from baqaro.core_functions.tree_subsample import (
        build_subsampled_subset, save_subset, load_subset,
        extract_subset_slice_cached, precompute_per_snapshot_indices,
    )

    log_mass_bins_sub = np.linspace(
        subsample_log_M_lo, subsample_log_M_hi, subsample_n_bins + 1
    )
    subsets_dir = os.path.join(path_out, "halo_subsets")
    subset_path = os.path.join(
        subsets_dir, f"subset_{name_file_halos}_{subset_tag}.npz"
    )

    if os.path.exists(subset_path):
        print(f"Loading cached subset from {subset_path}")
        subset = load_subset(subset_path)
    else:
        print(f"Building subset (not cached at {subset_path})")
        rev_path = os.path.join(
            path_out, "halo_histories", f"reverse_index_{name_file_halos}.npz"
        )
        if os.path.exists(rev_path):
            print(f"  Loading reverse index from {rev_path}")
            rev_data = np.load(rev_path)
            reverse_index = (rev_data["indptr"], rev_data["indices"])
        else:
            from baqaro.core_functions.select_merger_branches import (
                build_reverse_index,
            )
            print("  Building reverse index (slow)...")
            reverse_index = build_reverse_index(mt_merger_ids)

        masses_at_root = (
            np.array(halo_masses_all[subsample_root_snap]).astype(np.float64)
            * mass_units
        )
        # v3 alive_at_root: HBT's true liveness flag AND positive recorded mass.
        # See the closure-criterion notes in core_functions/tree_subsample.py.
        alive_at_root_mask = (mt_death_index == -1) & (masses_at_root > 0)
        subset = build_subsampled_subset(
            track_ids=mt_track_ids,
            merger_track_ids=mt_merger_ids,
            halo_masses_at_root=masses_at_root,
            alive_at_root=alive_at_root_mask,
            log_mass_bins=log_mass_bins_sub,
            N_b_per_bin=subsample_N_b_per_bin,
            keep_all_above_log_M=subsample_keep_all_above_log_M,
            rng_seed=subsample_rng_seed,
            chunk_id=chunk_id,
            n_chunks=n_chunks,
            reverse_index=reverse_index,
        )
        os.makedirs(subsets_dir, exist_ok=True)
        save_subset(subset_path, subset)
        print(f"  Saved subset to {subset_path}")

    subset_mask = subset["subset_mask"]
    subset_weights = subset["weights"]
    subset_indices = np.flatnonzero(subset_mask)
    n_subset = int(len(subset_indices))
    n_full_halos = int(len(subset_mask))
    print(
        f"Subsampling ON: keeping {n_subset:,} / {n_full_halos:,} halos "
        f"({100*n_subset/n_full_halos:.4f}%, reduction {n_full_halos/max(n_subset,1):.0f}x)"
    )

    n_storage = n_subset
    # Storage positions only run 0..n_subset-1 with a -1 sentinel.
    # int32 suffices as long as n_subset < 2**31, saving ~4 bytes per
    # full-N slot in this shared-memory array (e.g. 13.6 GB at
    # L2800N10080) and per worker via downstream `_merger_mapping`.
    assert n_subset < 2**31, (
        f"n_subset={n_subset} exceeds int32 range; bump global_to_storage "
        "and run_single_model._merger_mapping dtypes back to int64."
    )
    global_to_storage = np.full(n_full_halos, -1, dtype=np.int32)
    global_to_storage[subset_indices] = np.arange(n_subset, dtype=np.int32)

    # Eager-load subset slice of halo_masses + cold_accretion_rates into
    # RAM, replacing the mmap'd full-N arrays. Multi-node-friendly:
    # per-node halo-data footprint drops 4x (66 GB vs 270 GB cache).
    # Per-snap halo accesses use _to_storage on the subset-sized rows.
    #
    # The first invocation extracts subset columns from the full mmap'd
    # source row-by-row and writes a small .npy beside the subset cache;
    # subsequent invocations read the small file sequentially (~30 s
    # instead of ~11 min on L2800N10080 maxsnap=60).
    print("Eager-loading subset halo data into RAM (multi-node friendly)...")
    _t_eager = time.time()
    _eager_cache_masses = os.path.join(
        subsets_dir, f"subset_{name_file_halos}_{subset_tag}__halo_masses.npy"
    )
    _eager_cache_rates = os.path.join(
        subsets_dir,
        f"subset_{name_file_halos}_{subset_tag}__specific_cold_accretion_rates.npy",
    )
    halo_masses_all = extract_subset_slice_cached(
        full_path=path_file_halo_masses,
        subset_indices=subset_indices,
        cache_path=_eager_cache_masses,
    )
    halo_specific_cold_accretion_rates_all = extract_subset_slice_cached(
        full_path=path_file_specific_cold_accretion,
        subset_indices=subset_indices,
        cache_path=_eager_cache_rates,
    )
    _eager_gb = (halo_masses_all.nbytes + halo_specific_cold_accretion_rates_all.nbytes) / 1e9
    print(
        f"  Eager-loaded subset halo data in {time.time() - _t_eager:.1f}s "
        f"({_eager_gb:.1f} GB in RAM)"
    )

    subset_weights_compact = subset_weights[subset_indices].astype(np.float64)
else:
    subset = None
    subset_mask = None
    subset_weights = None
    subset_indices = None
    subset_weights_compact = None
    n_storage = halo_masses_all.shape[1]
    global_to_storage = None


# --- Precompute per-snapshot index arrays (shared across all workers) ---
# These depend only on merger tree arrays (birth/death/merged), not on
# ERDF parameters, so they are identical for every run. Restricted to
# subset_mask when subsampling is on.
print("Precomputing per-snapshot index arrays...")
# When subsampling, the helper subset-restricts the merger-tree arrays
# ONCE and operates on (n_subset,)-sized views, then maps results back
# to GLOBAL indices via subset_indices. ~50x fewer per-snap bool ops on
# L2800N10080. Output identity vs the prior full-N flatnonzero is
# unit-tested in tests/test_precomp_speedups.py.
if subset_mask is not None:
    from baqaro.core_functions.tree_subsample import (
        precompute_per_snapshot_indices,
    )
    _precomp = precompute_per_snapshot_indices(
        min_snap=min_snap,
        max_snap=max_snap,
        mt_birth_index=mt_birth_index,
        mt_death_index=mt_death_index,
        mask_merged=mask_merged,
        # Disrupted (dying, no merger target) halos must be retired in the
        # worker: their BHs are zeroed at the death snapshot so
        # the training BHMF matches main_evolution's live-only accounting.
        mask_disrupted=~mask_merged,
        subset_indices=subset_indices,
    )
    precomp_spawning = _precomp["spawning"]
    precomp_evolving = _precomp["evolving"]
    precomp_merging = _precomp["merging"]
    precomp_lost = _precomp["lost"]
else:
    _t_pre = time.time()
    _death_eff = mt_death_index.copy()
    _death_eff[_death_eff == -1] = np.iinfo(_death_eff.dtype).max

    precomp_spawning = {}   # snap -> int32 index array of halos born at this snap
    precomp_evolving = {}   # snap -> int32 index array of halos born before & alive
    precomp_merging = {}    # snap -> int32 index array of halos dying by merger
    precomp_lost = {}       # snap -> int32 index array of halos dying disrupted

    for _s in range(min_snap, max_snap + 1):
        _spawn = (mt_birth_index == _s)
        _alive = (mt_birth_index < _s) & (mt_birth_index != -1) & (_death_eff > _s)
        _dying_base = (mt_birth_index < _s) & (mt_birth_index != -1) & (mt_death_index == _s)
        precomp_spawning[_s] = np.flatnonzero(_spawn)
        precomp_evolving[_s] = np.flatnonzero(_alive)
        precomp_merging[_s] = np.flatnonzero(_dying_base & mask_merged)
        precomp_lost[_s] = np.flatnonzero(_dying_base & ~mask_merged)

    del _death_eff
    print(f"Precomputed index arrays for {max_snap - min_snap + 1} snapshots in {time.time() - _t_pre:.2f}s")


# ==============================================================================
# LOAD LOGNORMAL SUM SAMPLER (Transfer Function)
# ==============================================================================
# This precomputed table accelerates sampling from sums of lognormal variables,
# used to draw total accretion over multiple sub-steps efficiently.
time_here = time.time()
logger.info("LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
sampler_file = "Universal_Lognormal_Sampler_final.npz"
transfer_function = load_3d_sampler(sampler_file)
logger.info("FINISHED LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
logger.info("Total time elapsed {} seconds".format(time.time() - time_here))



# ==============================================================================
# BLACK HOLE EVOLUTION LOOP
# ==============================================================================

# --- build snapshots_to_save + mapping (do this once) ---
# Sim-aware default (lives in utils.sim_config). The two FLAMINGO boxes
# have different snapshot schedules, so the same snap NUMBER maps to
# different REDSHIFTS — keep the z ladder fixed per sim instead of the
# snap-index ladder. See SimSpec.snapshots_to_save_default.
from baqaro.utils.sim_config import snapshots_to_save_default
snapshots_to_save = [s for s in snapshots_to_save_default if s <= max_snap]
snap_to_zindex = {int(s): iz for iz, s in enumerate(snapshots_to_save)}
print(f"snapshots_to_save (filtered to max_snap={max_snap}): {snapshots_to_save}")
print(f"  corresponding z values: "
      f"{[float(redshifts[s]) if s < len(redshifts) else None for s in snapshots_to_save]}")


num_lbins = 40
num_mbins = 40
log_L_thresholds = np.linspace(45.5, 47.5, 21)  # for CERDF, QHMF

# shapes from your script
n_z = len(snapshots_to_save)
n_lbins = num_lbins
n_Lthr = len(log_L_thresholds)
n_mbins = num_mbins

output_datasets = {
    "log_qlfs":   (n_z, n_lbins),
    "log_bhmfs":  (n_z, n_mbins),
    "log_cerdfs": (n_z, n_Lthr, n_mbins),
    "log_qhmfs":  (n_z, n_Lthr, n_mbins),
}

print("Initializing training file at", path_file)
# Default behavior: REFUSE to write into an existing HDF5 with a different
# per-row shape (e.g. different n_z because snapshots_to_save changed).
# Set BAQARO_TRAINING_OVERWRITE=1 to auto-archive the old file and start fresh.
_on_mismatch = "archive_and_recreate" if env_bool("BAQARO_TRAINING_OVERWRITE", False) else "raise"
init_training_file(
    path_file=path_file,
    param_dict=param_definitions_emulation,
    n_z=n_z,
    n_lbins=n_lbins,
    n_Lthr=n_Lthr,
    n_mbins=n_mbins,
    output_datasets=output_datasets,
    on_schema_mismatch=_on_mismatch,
)


# --- precompute histogram bins + normalizations (do this once) ---
print("Precomputing histogram bins and normalizations...")
# QLF
qlf_bins = np.logspace(my_utils.to_solar(44), my_utils.to_solar(48.5), num_lbins + 1)
qlf_bin_width = np.log10(qlf_bins[1]) - np.log10(qlf_bins[0])
qlf_normalization = 1.0 / (qlf_bin_width * boxsize**3)

# BHMF (all BHs, no luminosity cut)
bhmf_bins = np.logspace(6.5, 11., num_mbins + 1)
bhmf_bin_width = np.log10(bhmf_bins[1]) - np.log10(bhmf_bins[0])
bhmf_normalization = 1.0 / (bhmf_bin_width * boxsize**3)

# CERDF
cerdf_bins = np.logspace(-3, 2, num_mbins + 1)
cerdf_bin_width = np.log10(cerdf_bins[1]) - np.log10(cerdf_bins[0])
cerdf_normalization = 1.0 / (cerdf_bin_width * boxsize**3)

# QHMF
# Sim-aware halo-mass range (see SimSpec). L2800N5040: 11.5–14.5;
# L2800N10080: 10.5–14.5 (its 0.9 dex lower mass floor lets it resolve
# smaller halos). Override per-run via BAQARO_LOG_M_HALO_QHMF_LO / _HI
# if you need to extend into the resolution floor for testing.
from baqaro.utils.sim_config import (
    log_M_halo_qhmf_lo as _qhmf_lo_default,
    log_M_halo_qhmf_hi as _qhmf_hi_default,
)
qhmf_lo = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_LO", _qhmf_lo_default))
qhmf_hi = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_HI", _qhmf_hi_default))
print(f"QHMF halo-mass bin range (log10 Msun): [{qhmf_lo}, {qhmf_hi}]")
qhmf_bins = np.logspace(qhmf_lo, qhmf_hi, num_mbins + 1)
qhmf_bin_width = np.log10(qhmf_bins[1]) - np.log10(qhmf_bins[0])
qhmf_normalization = 1.0 / (qhmf_bin_width * boxsize**3)

# compute bin centers for later use (avoid doing this repeatedly in the loop)
# For log-spaced bins, use geometric mean centers
qlf_bin_centers = np.sqrt(qlf_bins[1:] * qlf_bins[:-1])
bhmf_bin_centers = np.sqrt(bhmf_bins[1:] * bhmf_bins[:-1])
cerdf_bin_centers = np.sqrt(cerdf_bins[1:] * cerdf_bins[:-1])
qhmf_bin_centers = np.sqrt(qhmf_bins[1:] * qhmf_bins[:-1])

# --- upgraded config (minimal extra stuff, no duplication of big arrays) ---
config = {
    # parameters
    "param_names": list(param_definitions_emulation.keys()),
    "erdf_params_template": erdf_params_dict,
    "erdf_model": erdf_model,

    # seeding / physics / numerics
    "logfseed": logfseed,
    "sigmaseed": sigmaseed,
    "time_step_for_accretion": time_step_for_accretion,
    "rad_efficiency_0": rad_efficiency_0,
    "rad_efficiency_model": rad_efficiency_model,
    "madau_feff_correction": madau_feff_correction,
    "growth_sum_max": growth_sum_max,
    "accretion_on": accretion_on,
    "merger_on": merger_on,

    "fw_approx": False,            # Fenton-Wilkinson approximation for lognormal sum

    # sim bookkeeping
    "max_snap": max_snap,
    "nbound_threshold": nbound_threshold,
    "halo_filtering_mode": halo_filtering_mode,
    "tdyn_fraction_default": tdyn_fraction_default,

    # shapes
    "n_z": len(snapshots_to_save),
    "n_lbins": num_lbins,
    "n_Lthr": len(log_L_thresholds),
    "n_mbins": num_mbins,

    # snapshots/time arrays
    "snapshots": snapshots,  # np.arange(min_snap, max_snap+1)
    "snapshots_to_save": snapshots_to_save,
    "snap_to_zindex": snap_to_zindex,
    "redshifts": redshifts,  # array indexed by snapshot number
    "delta_times_snapshots": delta_times_snapshots,  # array indexed by snapshot number

    # bins + normalizations
    "qlf_bins": qlf_bins,
    "qlf_normalization": qlf_normalization,
    "log_lbins": my_utils.to_ergs(np.log10(qlf_bin_centers)),
    "bhmf_bins": bhmf_bins,
    "bhmf_normalization": bhmf_normalization,
    "log_mbins_bhmf": np.log10(bhmf_bin_centers),
    "cerdf_bins": cerdf_bins,
    "cerdf_normalization": cerdf_normalization,
    "log_bins_cerdf": np.log10(cerdf_bin_centers),
    "qhmf_bins": qhmf_bins,
    "qhmf_normalization": qhmf_normalization,
    "log_mbins_qhmf": np.log10(qhmf_bin_centers),
    "log_L_thresholds": log_L_thresholds,

    # units / constants
    "mass_units": mass_units,
    "rng_seed_bh": 12345,     # keep fixed to reduce noise across runs

    # big inputs (pass references; no extra memory)
    "halo_masses_all": halo_masses_all,  # memmap
    "halo_specific_cold_accretion_rates_all": halo_specific_cold_accretion_rates_all,  # memmap
    "transfer_function": transfer_function,

    # merger tree arrays (already prefetched into RAM in your script)
    "mt_birth_index": mt_birth_index,
    "mt_death_index": mt_death_index,
    "mt_track_ids": mt_track_ids,
    "mt_merger_ids": mt_merger_ids,
    "mask_merged": mask_merged,

    # precomputed per-snapshot index arrays (shared across all workers via fork COW)
    "precomp_spawning": precomp_spawning,
    "precomp_evolving": precomp_evolving,
    "precomp_merging": precomp_merging,
    "precomp_lost": precomp_lost,

    # subset bookkeeping (None when use_subsample=False)
    "use_subsample": use_subsample,
    "n_storage": n_storage,
    "global_to_storage": global_to_storage,
    "subset_indices": subset_indices,
    "subset_weights": subset_weights,
    "subset_weights_compact": subset_weights_compact,

    # output dataset schema (used by init_training_file in run_training_set)
    "output_datasets": output_datasets,

    # optional
    "logger": logger,
}

print("Writing training metadata to", path_file)
write_training_metadata(path_file, config)




# --- Design mode: switch global_sobol vs local_box via env var ---
# "global_sobol" : sample full param_definitions_emulation box (default).
# "local_box"    : sample tighter local_bounds defined below — useful for
#                  refining the emulator around a known good region.
# Override via BAQARO_TRAINING_DESIGN.
training_design = os.environ.get("BAQARO_TRAINING_DESIGN", "global_sobol")
if training_design not in ("global_sobol", "local_box"):
    raise ValueError(
        f"BAQARO_TRAINING_DESIGN={training_design!r} not in "
        "{'global_sobol', 'local_box'}"
    )

# Tighter param box used only when training_design == "local_box": the
# refinement box that brackets the adopted fiducial's (`qcc_ck22final_v1`) DE
# optima across all four likelihood arms. It is a valid sub-box of the global
# param_definitions_emulation. Per-param BAQARO_LOCAL_<PARAM>_LO/_HI override it.
local_bounds = {
    "log_eta_mean_0": [-1.50, -0.85],  # Mean log(Eddington ratio) at cold sSAR=0
    "log_eta_mean_evol": [0.65, 1.05],  # Redshift evolution of mean
    "std_0": [0.20, 0.65],   # Scatter at cold sSAR=0 (dex)
    # "std_evol": [-0.2, 0.05],   # Redshift evolution of scatter
    "logtcoherence": [4.50, 6.50],  # log10(coherence time in yr)
    "logfseed": [-7.00, -5.00],     # log10(M_BH / M_halo) at seeding
    "sigmaseed": [0.20, 1.00],      # dex scatter on logfseed (0 = deterministic seeding)
}

# Optional env-var override of the local_box bounds (per-param lo/hi), mirroring
# the BAQARO_PRIOR_* override of the global box above. Set BAQARO_LOCAL_<PARAM>_LO
# / _HI to widen/shift the refinement box for an APPEND local_box batch without
# editing this file — the Sobol n_skip continuation (per design) then space-fills
# the NEW box with fresh points, so the appended batch and the original coexist
# as two overlapping local boxes (both design-tagged "local_box"). Any local
# bound MUST stay inside the global param_definitions_emulation box (sample_in_box
# assumes the local box is a sub-box) — asserted below.
_LOCAL_ENV_KEYS = {
    "log_eta_mean_0":    ("BAQARO_LOCAL_LOG_ETA_MEAN_0_LO",    "BAQARO_LOCAL_LOG_ETA_MEAN_0_HI"),
    "log_eta_mean_evol": ("BAQARO_LOCAL_LOG_ETA_MEAN_EVOL_LO", "BAQARO_LOCAL_LOG_ETA_MEAN_EVOL_HI"),
    "std_0":             ("BAQARO_LOCAL_STD_0_LO",             "BAQARO_LOCAL_STD_0_HI"),
    "logtcoherence":     ("BAQARO_LOCAL_LOGTCOHERENCE_LO",     "BAQARO_LOCAL_LOGTCOHERENCE_HI"),
    "logfseed":          ("BAQARO_LOCAL_LOGFSEED_LO",          "BAQARO_LOCAL_LOGFSEED_HI"),
    "sigmaseed":         ("BAQARO_LOCAL_SIGMASEED_LO",         "BAQARO_LOCAL_SIGMASEED_HI"),
}
# Only the params actually overridden here are validated against the global box;
# the un-touched hardcoded defaults are left as-is (they form a valid sub-box).
_local_overridden = []
for _lk, (_envlo, _envhi) in _LOCAL_ENV_KEYS.items():
    _lo = os.environ.get(_envlo)
    _hi = os.environ.get(_envhi)
    if _lo is not None:
        local_bounds[_lk][0] = float(_lo)
        _local_overridden.append(_lk)
    if _hi is not None:
        local_bounds[_lk][1] = float(_hi)
        _local_overridden.append(_lk)
if _local_overridden:
    print("Local_box bounds OVERRIDDEN via BAQARO_LOCAL_* env vars:")
    for _lk in dict.fromkeys(_local_overridden):  # de-dup, keep order
        _rng = local_bounds[_lk]
        _glo, _ghi = param_definitions_emulation[_lk]
        if _rng[0] < _glo or _rng[1] > _ghi or _rng[0] >= _rng[1]:
            raise ValueError(
                f"BAQARO_LOCAL override for {_lk!r} -> [{_rng[0]}, {_rng[1]}] is not a "
                f"valid sub-box of the global bounds [{_glo}, {_ghi}] (need lo<hi and "
                f"lo>=global_lo and hi<=global_hi)."
            )
        print(f"  {_lk:<22} [{_rng[0]:>+7.3f}, {_rng[1]:>+7.3f}]  (was overridden)")

# Worker count is env-driven: a 1 TB node fits ~15 (RAM-bound at the
# 386M z=0 subset; per-worker ~28 GB peak + ~650 GB fixed eager/base). Larger
# memory nodes (2 TB) set this higher via BAQARO_N_WORKERS.
n_workers_cfg = int(os.environ.get("BAQARO_N_WORKERS", "15"))
print(f"\n=== Launching training: design={training_design}, n_new={num_simulations}, n_workers={n_workers_cfg} ===")
if training_design == "global_sobol":
    run_training_batch_parallel(
        path_file=path_file,
        num_parameters=num_parameters,
        param_dict=param_definitions_emulation,
        config=config,
        n_new=num_simulations,
        design="global_sobol",
        rng=rng_training,
        local_bounds=None,
        append=True,
        overwrite=False,
        n_workers=n_workers_cfg,
        max_in_flight=n_workers_cfg,
        memory_floor_gb=150,
    )
else:  # local_box
    run_training_batch_parallel(
        path_file=path_file,
        num_parameters=num_parameters,
        param_dict=param_definitions_emulation,
        config=config,
        n_new=num_simulations,
        design="local_box",
        rng=rng_training_local,
        local_bounds=local_bounds,
        append=True,
        overwrite=False,
        n_workers=n_workers_cfg,
        max_in_flight=n_workers_cfg,
        memory_floor_gb=150,
    )


# --- Self-describing run provenance (see utils/provenance.py) ---
# Stamp how this training set was generated into the file. Read later with
# h5py.File(path)["provenance"].attrs : design, N_simulations, bin grids, the
# 6-param sampling box, geometry (subset_tag), git revision + BAQARO_* env. The
# per-row metadata still lives in the existing `config` / `schema` groups.
import json as _json
with h5py.File(path_file, "a") as _prov_f:
    write_run_provenance(_prov_f, {
        "engine": "main_training",
        "simulation_name": simulation_name,
        "max_snap": int(max_snap),
        "min_snap": int(min_snap),
        "erdf_model": erdf_model,
        "fold_subhalo_mass": bool(fold_subhalo_mass),
        "merger_delay_mode": merger_delay_mode,
        "tdyn_fraction_default": float(tdyn_fraction_default),
        "nbound_threshold": int(nbound_threshold),
        "halo_filtering_mode": halo_filtering_mode,
        "notes_file": notes_file,
        "name_file": name_file,
        "use_subsample": bool(use_subsample),
        "subset_tag": subset_tag if use_subsample else "",
        "training_design": training_design,
        "num_simulations_requested": int(num_simulations),
        "num_parameters": int(num_parameters),
        "num_lbins": int(num_lbins),
        "num_mbins": int(num_mbins),
        "n_z": int(len(snapshots_to_save)),
        "snapshots_to_save": ",".join(str(s) for s in snapshots_to_save),
        "param_ranges_json": _json.dumps(param_definitions_emulation),
    })
print("Wrote run provenance to", path_file)