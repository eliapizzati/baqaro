"""Forward model: evolve the black-hole population over a halo merger tree.

The production entry point. Reads the preprocessed halo mass and cold-accretion
histories, seeds black holes into newly resolved haloes, grows them snapshot by
snapshot with ERDF-drawn Eddington ratios, applies mergers along the tree, and
writes per-snapshot masses and luminosities to HDF5.

Runs on a stratified subsample of merger-tree roots by default (closed under
mergers, with inverse-probability weights) so that a z=0 run fits in memory; see
:mod:`~baqaro.core_functions.tree_subsample`. Set
``BAQARO_USE_SUBSAMPLE=0`` for the full catalogue, or use
``main_evolution_chunked`` to partition it across nodes.

Configuration is entirely by environment variable -- a bare invocation
reproduces the adopted fiducial:

    python -m baqaro.core_functions.main_evolution

The six free parameters resolve in three tiers: an individual
``BAQARO_LOG_ETA_MEAN_0``-style variable wins over an entry named by
``BAQARO_BESTFIT_NAME`` in
:mod:`~baqaro.core_functions.bestfit_registry`, which wins over
the defaults below.

Every choice that changes the physics or the data a run consumes is encoded in
the output filename, so runs with different settings can never collide on disk,
and the file additionally carries a ``provenance/`` group recording the git
revision and the full environment. Two silent-clobber routes are hard errors
before any data is read: an individual parameter override that would not be
reflected in the filename, and an output path that already exists. Set
``BAQARO_ALLOW_OVERWRITE=1`` for a deliberate re-run.

Cost is roughly an hour and a few hundred GB of output at the production
configuration, so it is normally submitted as a batch job.
"""

# This module is a run script, not a library: its body loads catalogues and
# writes products at import time. Refuse a plain ``import`` so nobody starts a
# multi-hour job by accident (``python -m baqaro.core_functions.main_evolution`` sets __name__ to
# "__main__"; a multiprocessing "spawn" child re-imports it as "__mp_main__").
if __name__ not in ("__main__", "__mp_main__"):
    raise RuntimeError(
        "baqaro.core_functions.main_evolution is an entry-point script; run it with "
        "'python -m baqaro.core_functions.main_evolution' instead of importing it."
    )

import os
# Set Numba thread count before any numba import (via bh_accretion_fast)
os.environ["NUMBA_NUM_THREADS"] = str(max(1, os.cpu_count() // 2))

import numpy as np
import h5py
import time


from numpy.random import default_rng

from qhtools.utils.cosmology import cosmo

from baqaro.core_functions.bh_seeding import spawn_BHs
from baqaro.core_functions.erdf_core_functions import Erdf
from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast as evolve_BHs
# from baqaro.core_functions.bh_accretion import evolve_BHs
from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
from baqaro.core_functions.merger_catalog import MergerCatalogBuilder

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
print_debug_info = False  # Enable verbose mass conservation checks (slow on subsampled runs)
RNG_SEED = int(os.environ.get("BAQARO_RNG_SEED", "12345"))  # Fixed seed for reproducibility
rng = default_rng(RNG_SEED)  # keep the resolved int around: it is what gets persisted
source_dir = DEFAULT_SOURCE_DIR


# --- Run Identification ---
# Free-form run label, appended to the output filename as `_{notes_file}`.
# Resolved three-tier (mirrors the param resolution below):
#   3. BAQARO_NOTES_FILE env var  (highest — explicit per-run override)
#   2. the registry entry's "notes" field (when BAQARO_BESTFIT_NAME is a
#      registered key; older entries pin a fixed notes token so re-running a
#      registered bestfit reproduces its exact on-disk filename)
#   1. None  (no notes token — the bestfit/subset tags carry the run identity)
# Tier 3 captured here; tier 2 applied just after the registry lookup below.
# Env *presence* wins (matching the param resolver): BAQARO_NOTES_FILE="" is an
# explicit "no token" that does NOT fall through to the registry default.
_notes_env_set = "BAQARO_NOTES_FILE" in os.environ
notes_file = (os.environ["BAQARO_NOTES_FILE"] or None) if _notes_env_set else None

# --- Snapshot Range ---
min_snap = 0
# Sim-aware default from sim_config (honours BAQARO_MAX_SNAP). 144 for
# L2800N10080 z=0, 78 for L2800N5040 z=0.
from baqaro.utils.sim_config import (
    max_snap, env_bool,
    FIDUCIAL_GROWTH_SUM_MAX, FIDUCIAL_MADAU_FEFF_CORRECTION,
)


# --- ERDF + seeding + time-resolution parameter resolution ----------------
# Three-tier precedence (highest wins):
#   3.  Individual env vars:  BAQARO_LOG_ETA_MEAN_0, BAQARO_LOG_ETA_MEAN_EVOL,
#                             BAQARO_STD_0, BAQARO_LOGFSEED, BAQARO_SIGMASEED,
#                             BAQARO_LOGTCOHERENCE
#   2.  Registry entry from BESTFIT_REGISTRY (bestfit_registry.py),
#       looked up by BAQARO_BESTFIT_NAME if the key matches a registered set.
#   1.  Hardcoded defaults below.
#
# Filename: BAQARO_BESTFIT_NAME is also appended as `_bestfit_<name>` to the
# output filename regardless of whether it matches a registry key — so
# free-form ad-hoc tags work for one-off runs that haven't been promoted
# to the registry yet.

erdf_model = "log_normal_evol_halo_mass"
_DEFAULTS = {
    "log_eta_mean_0":   -0.866391,
    "log_eta_mean_evol": 0.812765,
    "std_0":             0.444504,
    "logfseed":         -5.669447,
    "sigmaseed":         0.461684,
    "logtcoherence":     6.060304,
}

# Tier 2: registry lookup (if BAQARO_BESTFIT_NAME is a known key)
bestfit_name = os.environ.get("BAQARO_BESTFIT_NAME", None)
_registry_entry = None
if bestfit_name:
    from baqaro.core_functions.bestfit_registry import BESTFIT_REGISTRY
    if bestfit_name in BESTFIT_REGISTRY:
        _registry_entry = BESTFIT_REGISTRY[bestfit_name]
        print(f"Loaded BESTFIT_REGISTRY[{bestfit_name!r}] "
              f"({_registry_entry.get('label','')})")
        # Cross-check the entry's fit context against the ACTIVE run config.
        # A fit derived at (sim, max_snap) evolved under a different config runs
        # silently otherwise — e.g. selecting a max_snap=71 (z~2) fit and
        # running it under the z=0 default (max_snap=144). Warn loudly; don't block
        # (deliberate cross-config runs are legitimate for diagnostics).
        from baqaro.utils.sim_config import simulation_name as _active_sim
        _reg_sim = _registry_entry.get("sim")
        _reg_snap = _registry_entry.get("max_snap")
        if _reg_sim is not None and _reg_sim != _active_sim:
            print(f"  ⚠ WARNING: registry entry sim={_reg_sim!r} != active "
                  f"BAQARO_SIM={_active_sim!r}. Params were fit on a different simulation.")
        if _reg_snap is not None and int(_reg_snap) != int(max_snap):
            print(f"  ⚠ WARNING: registry entry max_snap={_reg_snap} != active "
                  f"max_snap={max_snap}. Params were fit at a different redshift/snapshot.")
    else:
        print(f"NOTE: BAQARO_BESTFIT_NAME={bestfit_name!r} is not in BESTFIT_REGISTRY — "
              "treating as a free-form filename tag (using hardcoded defaults unless "
              "individual BAQARO_<PARAM> env vars are also set).")

# notes_file tier 2: inherit the registry entry's "notes" when BAQARO_NOTES_FILE
# was not set at all. Lets a registered bestfit reproduce the exact filename its
# run originally produced (older entries carry a fixed notes token).
if not _notes_env_set and _registry_entry is not None:
    notes_file = _registry_entry.get("notes") or None
    if notes_file:
        print(f"  → notes_file from registry: {notes_file!r} (override with BAQARO_NOTES_FILE)")

# --- Filename tokens for PARAMETER overrides ----------------------------------
# A parameter env var CHANGES THE PHYSICS, so it must be reflected in the output
# filename. Otherwise
#     BAQARO_BESTFIT_NAME=<registered>  BAQARO_LOGTCOHERENCE=7.0
# would resolve tau to 10 Myr and then write to the REGISTERED BESTFIT'S OWN
# PATH, silently overwriting it. (The physics TOGGLES are guarded the same way,
# by the filename tokens in physics_toggle_suffix.)
#
# So: if BAQARO_BESTFIT_NAME names a REGISTRY ENTRY and an individual param env
# var moves that param AWAY from the registered value, a token is appended at the
# very end of name_file (after the physics toggles), e.g. `_tau7`.
#
# Deliberately NOT emitted when:
#   * the env value EQUALS the registry value — the run reproduces the registered
#     bestfit exactly, so its filename must not change (this keeps every existing
#     on-disk product, e.g. the runs launched with all six params spelled out
#     alongside their registry name, addressable at the same path);
#   * BAQARO_BESTFIT_NAME is a free-form tag with no registry entry — the tag is
#     already the differentiator.
#
# The token is appended (not inserted), so `_variant_path("tau7")`-style lookups
# in the plotting layer find it exactly like `_tau0` / `_nomerge` / `_radeff_*`.
_PARAM_TOKENS = {
    "log_eta_mean_0":    "eta0",
    "log_eta_mean_evol": "etaevol",
    "std_0":             "std0",
    "logtcoherence":     "tau",
    "logfseed":          "fseed",
    "sigmaseed":         "sseed",
}
_param_overrides = []      # (token, value) for params deviating from the registry


def _resolve(param, env_var):
    """Tier 3 (env var) > Tier 2 (registry) > Tier 1 (default)."""
    if env_var in os.environ:
        val = float(os.environ[env_var])
        if _registry_entry is not None and param in _registry_entry:
            reg = float(_registry_entry[param])
            if abs(val - reg) > 1e-9:
                _param_overrides.append((_PARAM_TOKENS[param], val))
                print(f"  ⚠ {env_var}={val:g} OVERRIDES registry "
                      f"{param}={reg:g} → filename token "
                      f"_{_PARAM_TOKENS[param]}{val:g} (so the registered run "
                      f"cannot be overwritten)")
        return val
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
time_step_for_accretion = 10**(logtcoherence - 6)
# BAQARO_TAU_COH_ZERO=1 forces Branch C (zero coherence time: analytical mean
# growth, L_bol decoupled from mass build). Default keeps the logtcoherence
# sub-step. See bh_accretion_fast Branch C for the zero-tau limit.
if env_bool("BAQARO_TAU_COH_ZERO", False):
    time_step_for_accretion = 0.0
    print("[BRANCH C] BAQARO_TAU_COH_ZERO=1 -> time_step_for_accretion=0 (analytical mean growth)")

print(f"Using logfseed = {logfseed}, sigmaseed = {sigmaseed}")
print(f"Using time step for accretion: {time_step_for_accretion} Myr (logtcoherence={logtcoherence})")

# Per-snapshot growth-sum cap (numerical guardrail mode). Default 50.0 is the
# legacy safety clamp that effectively never bites (exp(50) ~ 5e21); lowering it
# bounds the per-snapshot mass-growth factor at exp(growth_max). Used for the
# "Option 3" experiment to trim stochastic-chain runaway BHs without touching
# the ERDF / sampler. Set, e.g., 4.6 -> max 100x per snap; 2.3 -> max 10x.
# 50.0 disables the cap (exp(50) is unreachable) and drops the `_g` token.
# IMPORTANT: the fiducial cap is FIDUCIAL_GROWTH_SUM_MAX; ANY other value
# changes the model and requires re-DE + re-MCMC before being used downstream.
growth_sum_max = float(os.environ.get("BAQARO_GROWTH_SUM_MAX", str(FIDUCIAL_GROWTH_SUM_MAX)))
if growth_sum_max != FIDUCIAL_GROWTH_SUM_MAX:
    print(f"[GROWTH-CAP MODE] BAQARO_GROWTH_SUM_MAX={growth_sum_max} -> "
          f"per-snapshot M_BH growth bounded at exp({growth_sum_max}) "
          f"~ {np.exp(growth_sum_max):.2e}x. "
          f"This is NOT the fiducial cap ({FIDUCIAL_GROWTH_SUM_MAX}).")

# Madau effective-efficiency correction (the f_eff correction). ON in the
# fiducial configuration. When enabled, Branch B multiplies
# mu_scale by f_eff(mu,sigma), removing the bright-end under-growth of the
# transfer-table sampler under madau+ (see core_functions/madau_feff.py).
# IMPORTANT: toggling it CHANGES the model (BHMF/QLF move by ~0.15-0.2 dex at
# fiducial sigma) and requires re-DE + re-MCMC before downstream use, so
# corrected and uncorrected products must never be mixed.
madau_feff_correction = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION",
                                        "1" if FIDUCIAL_MADAU_FEFF_CORRECTION else "0").strip().lower()
                         not in ("0", "", "false", "no", "off"))
if madau_feff_correction != FIDUCIAL_MADAU_FEFF_CORRECTION:
    print(f"[MADAU-FEFF] BAQARO_MADAU_FEFF_CORRECTION="
          f"{'on' if madau_feff_correction else 'off'} -> NOT the fiducial "
          f"setting ({'on' if FIDUCIAL_MADAU_FEFF_CORRECTION else 'off'}). "
          "This changes the model; do not mix with fiducial products.")
if bestfit_name:
    print(f"  → tagging output filename: _bestfit_{bestfit_name}")
    print(f"     log_eta_mean_0    = {erdf_params_dict['log_eta_mean_0']}")
    print(f"     log_eta_mean_evol = {erdf_params_dict['log_eta_mean_evol']}")
    print(f"     std_0             = {erdf_params_dict['std_0']}")


# --- Physics Toggles ---
accretion_on = True   # Enable black hole accretion
merger_on = env_bool("BAQARO_MERGER_ON", True)  # BH-BH mergers; BAQARO_MERGER_ON=0 disables
fw_approx = False     # Use FW lognormal approximation for bulk (70%) of transfer function lookups

# --- Plateaued-halo accretion (env-gated; default NON-ACCRETING) ---------------
# Halos whose LastMaxMass has plateaued have specific cold accretion rate == 0.
# The fast ERDF path would clip that to a 1e-8 floor -> eta ~ 1e-8 -> negligible
# (but non-zero) growth + L_bol ~ 1e38 erg/s. We instead treat them as strictly
# NON-ACCRETING (M_BH frozen, L_bol = 0). This is QLF-invariant (verified to
# 0.000 dex: floored halos are ~5 dex below the faintest QLF bin) and removes a
# spurious pile-up in the eta distribution. Set BAQARO_NONACCRETING_ZERO_RATE=0
# to restore the legacy clip-floor behaviour (output tagged _clipfloor so it
# never clobbers the default).
NONACCRETING_ZERO_RATE = env_bool("BAQARO_NONACCRETING_ZERO_RATE", True)
if not NONACCRETING_ZERO_RATE:
    notes_file = (notes_file + "_clipfloor") if notes_file else "clipfloor"
    print(f"[NONACC OFF] legacy clip-floor accretion for plateaued halos; output tag -> '{notes_file}'.")

# --- Merger catalogue ---
# When True, every BH-BH merger event is recorded (z, M1, M2, progenitor IDs,
# descendant ID, halo ID) and saved to a separate HDF5 file
# {path_out}/evolution/merger_catalog_{name_file}.hdf5 (schema documented in
# ``core_functions/merger_catalog.py``).
# Adds <~5% to the merger-block runtime; <~1 GB peak RAM for typical runs.
# Survivor data (track IDs and BH masses of halos alive at z=0) is saved alongside.
record_merger_catalog = False

# --- Parallelization ---
# Option 1: Numba prange (preferred — no Python overhead, NUMBA_NUM_THREADS set at top of file)
use_parallel_kernel = True
# Option 2: joblib threading with serial (nogil) kernels
parallelize = False
num_workers = 64
backend = "threading"  # Options: "loky", "threading", "multiprocessing"

# --- Radiative Efficiency ---
rad_efficiency_0 = 0.1          # Base radiative efficiency (epsilon)
rad_efficiency_model = os.environ.get("BAQARO_RAD_EFFICIENCY_MODEL", "madau+")  # Options: "constant", "madau+" (spin-dependent); env-overridable

# --- Halo Selection ---
nbound_threshold = 40       # Minimum bound particles for a resolved halo
halo_filtering_mode = "global"  # Options: "local", "global"

# Must match the saver run that produced the halo files we load below.
# `fold_subhalo_mass=True`  -> reads halo_masses_..._foldmass.npy
# `merger_delay_mode != "instant_old"` -> reads ..._{mode}.npy
# Both tokens are also appended to the output filename so old z=0
# production runs do NOT get overwritten. To rerun the legacy pipeline,
# set BAQARO_FOLD_SUBHALO_MASS=0 and BAQARO_MERGER_DELAY_MODE=instant_old.
# See get_merger_trees() for the merger modes.
#
# tdyn_fraction_default is sim-dependent (sim_config.py). 0.20 at
# L2800N5040, 0.25 at L2800N10080 (the latter has finer Δt/t_dyn cadence).
from baqaro.utils.sim_config import (
    tdyn_fraction_default, fold_subhalo_mass, merger_delay_mode,
)


# --- Subsampling ---
# When True, the run evolves only a stratified-by-mass subset of merger tree
# roots (+ their full progenitor closure) instead of the full halo catalogue.
# Output is saved sliced to the kept halos, with per-halo inverse-probability
# weights so any downstream summary statistic stays unbiased in expectation.
# Output filename auto-includes a subset tag — different configs (or
# use_subsample=False) never overwrite each other.
# Override via BAQARO_USE_SUBSAMPLE=0 (off) or =1 (on).
use_subsample = env_bool("BAQARO_USE_SUBSAMPLE", True)
subsample_root_snap = max_snap   # snapshot defining roots (top of the tree)
# Halos kept per log-mass bin. Override via BAQARO_SUBSAMPLE_NB env var
# (uniform across bins). Bumping N_b cuts the Horvitz-Thompson variance
# per output bin as 1/N_b.
# Default 500_000 with keep_above=None: benchmarked to use ~27% less RAM than
# (N_b=5000 + keep_above=13.0) while matching the full-sim bright-end
# statistics within statistical noise.
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

if notes_file is not None:
    name_file += "_{}".format(notes_file)

# Best-fit tag (see BAQARO_BESTFIT_NAME above). Placed AFTER notes_file but
# BEFORE the subset suffix so all bestfit_X variants of the same notes
# config sort together on disk.
if bestfit_name:
    name_file += "_bestfit_{}".format(bestfit_name)

# Append subset tag so subsampled runs land in different files than full runs
# (and different subsample configs land in different files than each other).
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
    )
    name_file += f"_sub_{subset_tag}"

# Tag the output filename when running in growth-cap mode so capped runs
# do not silently overwrite the fiducial. Default 50.0 → no token.
if growth_sum_max != 50.0:
    # Format: 4.6 -> "g4.6", 7.0 -> "g7", 12.0 -> "g12" (compact).
    if float(growth_sum_max).is_integer():
        _g_tok = f"g{int(growth_sum_max)}"
    else:
        _g_tok = f"g{growth_sum_max:g}"
    name_file += f"_{_g_tok}"

# Tag the output filename when the madau f_eff correction is on, so corrected
# and uncorrected products coexist on disk. Default off → no token.
if madau_feff_correction:
    name_file += "_feffcorr"

# Tag the output filename for the model-changing physics toggles
# (BAQARO_MERGER_ON / BAQARO_RAD_EFFICIENCY_MODEL / BAQARO_TAU_COH_ZERO /
# BAQARO_FORCE_NO_SAMPLER). These change the physics but previously produced
# no token, so a diagnostic run under a registered bestfit name would
# overwrite the fiducial product. Production defaults →
# empty string → fiducial filename unchanged.
from baqaro.utils.sim_config import physics_toggle_suffix
_phys_tok = physics_toggle_suffix()
if _phys_tok:
    name_file += _phys_tok
    print(f"[PHYSICS-TOGGLE MODE] non-default physics toggle(s) active; "
          f"tagging output filename '{_phys_tok}'. This is NOT the production setting.")

# PARAMETER-override tokens, appended LAST (see _resolve above). This is what
# stops e.g. `BAQARO_BESTFIT_NAME=<registered> BAQARO_LOGTCOHERENCE=7.0` from
# silently overwriting the registered run, and it is the token the plotting
# layer's `_variant_path("tau7")` looks for.
if _param_overrides:
    _par_tok = "".join(f"_{tok}{val:g}" for tok, val in _param_overrides)
    name_file += _par_tok
    print(f"[PARAM-OVERRIDE MODE] {len(_param_overrides)} param(s) moved off the "
          f"registered bestfit; tagging output filename '{_par_tok}'.")

path_file = os.path.join(path_out, "evolution", f"bh_evolution_{name_file}.hdf5")
path_log = os.path.join(path_out, "logs", f"bh_evolution_{name_file}.log")

# ==============================================================================
# OVERWRITE GUARDS
# ==============================================================================
# These runs cost ~1 h and ~425 GB. Two distinct ways a variant run could
# silently clobber a production product, both now blocked:
#
#  (1) UNTOKENISED PARAM OVERRIDE. `_resolve()` only emits a `_tau7`-style token
#      when the bestfit is a REGISTRY entry (it needs the registered value to
#      diff against). With a free-form `BAQARO_BESTFIT_NAME`, setting e.g.
#      BAQARO_LOGTCOHERENCE changes the physics but produces NO token — the run
#      lands on the un-overridden filename. Abort instead.
#  (2) PLAIN CLOBBER. The resolved output already exists. Never silently
#      overwrite; require an explicit opt-in.
#
# Escape hatch for both: BAQARO_ALLOW_OVERWRITE=1 (deliberate re-runs).
_allow_overwrite = env_bool("BAQARO_ALLOW_OVERWRITE", False)

_param_env_set = [e for p, e in (
    ("log_eta_mean_0", "BAQARO_LOG_ETA_MEAN_0"),
    ("log_eta_mean_evol", "BAQARO_LOG_ETA_MEAN_EVOL"),
    ("std_0", "BAQARO_STD_0"),
    ("logtcoherence", "BAQARO_LOGTCOHERENCE"),
    ("logfseed", "BAQARO_LOGFSEED"),
    ("sigmaseed", "BAQARO_SIGMASEED"),
) if e in os.environ]
if _param_env_set and _registry_entry is None and not _allow_overwrite:
    raise SystemExit(
        "\n[OVERWRITE GUARD] Individual param override(s) "
        f"{_param_env_set} are set, but BAQARO_BESTFIT_NAME="
        f"{os.environ.get('BAQARO_BESTFIT_NAME', '<unset>')!r} is NOT a registry "
        "entry, so NO `_tau7`-style filename token can be emitted and this run "
        "would write to the un-overridden filename:\n"
        f"    {path_file}\n"
        "Fix: use a REGISTERED bestfit name (see bestfit_registry.py) so the "
        "override is tagged, or set a distinct BAQARO_NOTES_FILE, or "
        "BAQARO_ALLOW_OVERWRITE=1 if you really mean it.\n")

# (3) SYMLINK CLOBBER. The `_fiducial_` alias layer (scripts/make_fiducial_aliases.py)
#     puts symlinks in evolution/ pointing at the settled paper products. h5py opens
#     a symlink in "w" mode by FOLLOWING it, so writing here would truncate the
#     425-725 GB product the alias points at — verified empirically. This is NOT what
#     BAQARO_ALLOW_OVERWRITE means ("replace THIS run"), so the escape hatch does not
#     apply: refuse unconditionally and make the user name a real path.
if os.path.islink(path_file):
    raise SystemExit(
        "\n[SYMLINK GUARD] The resolved output is a SYMLINK:\n"
        f"    {path_file}\n    -> {os.readlink(path_file)}\n"
        "Writing here would follow the link and DESTROY the file it points at "
        "(these are the settled paper products; see evolution/README_FIDUCIAL.txt).\n"
        "Fix: pick a real filename — `fiducial` is an alias token, not a notes label. "
        "BAQARO_ALLOW_OVERWRITE does NOT bypass this guard.\n")

if os.path.exists(path_file) and not _allow_overwrite:
    _sz = os.path.getsize(path_file) / 1e9
    raise SystemExit(
        "\n[OVERWRITE GUARD] The resolved output already exists "
        f"({_sz:.1f} GB):\n    {path_file}\n"
        "Refusing to overwrite. If this is an intentional re-run, set "
        "BAQARO_ALLOW_OVERWRITE=1. If you expected a DIFFERENT filename, your "
        "variant token is missing — check BAQARO_BESTFIT_NAME is registered "
        "and the physics/param toggles you set actually emit a token.\n")

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

# --- Load Halo Masses + Cold-Accretion Rates (Memory-Mapped) ---
# mmap_mode='r' keeps data on disk, reads columns on demand.
#
# In subsample mode the full-N arrays are only needed to (a) BUILD the
# subset (reads masses at the root snapshot, ~line 400) and (b) EXTRACT
# the eager subset slice on the first run. Once the subset .npz and the
# per-array ``__*.npy`` slice caches exist on disk, the full-N files are
# never touched again -- so we mmap them only if present in subsample
# mode, letting a run proceed on a machine that holds just the (much
# smaller) subset caches (e.g. after a slice-only transfer).
# ``extract_subset_slice_cached`` re-opens via the path string, not this
# object, so a None here is fine as long as the slice cache exists.
# Full-sim mode still requires the full files (they back every
# per-snapshot access in the evolution loop).
def _mmap_full_array(path, label, *, required):
    if os.path.exists(path):
        print(f"Loading {label} from {path}")
        return np.load(path, mmap_mode="r")
    if required:
        raise FileNotFoundError(
            f"{label} not found at {path} (required in full-sim mode)."
        )
    print(
        f"  [subsample] full {label} absent at {path}; relying on the "
        f"cached subset slice (building/extracting it would need the "
        f"full file)."
    )
    return None

halo_masses_all = _mmap_full_array(
    path_file_halo_masses, "halo masses", required=not use_subsample
)

# --- Load Merger Trees ---
print("Loading merger trees from", path_file_trees)
merger_trees = MergerTreeLoader(path_file_trees)

# --- Load Cold Gas Accretion Rates (Memory-Mapped) ---
halo_specific_cold_accretion_rates_all = _mmap_full_array(
    path_file_specific_cold_accretion, "cold accretion rates",
    required=not use_subsample,
)

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
# Stratified mass-binned subsampling of merger tree roots, closed under
# mergers, weighted by inverse selection probability — see
# core_functions/tree_subsample.py for the math.
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

        # Halo masses at root snapshot in physical Msun (h-free, matches
        # run_single_model.py / stats convention: just `* mass_units`).
        if halo_masses_all is None:
            raise FileNotFoundError(
                f"Building the subset needs the full halo_masses file at "
                f"{path_file_halo_masses}, but it is absent. Provide it, or "
                f"copy a prebuilt subset cache to {subset_path}."
            )
        masses_at_root = (
            np.array(halo_masses_all[subsample_root_snap]).astype(np.float64)
            * mass_units
        )
        # ``alive_at_root`` controls both root eligibility and the closure-skip
        # in tree_subsample. CRITICAL: HBT preserves LastMaxMass monotonically
        # after death (it freezes at peak), so ``mass > 0 at z_root`` is True
        # for ALL halos that ever crossed the resolution threshold — including
        # ~118M dead "true progenitors" we DO want to walk via closure. The
        # correct "alive at z_root" test is ``death_index == -1`` (HBT's own
        # liveness flag), combined with ``mass > 0`` to drop the ~123M halos
        # that have death=-1 but no recorded peak (HBT artifacts; contribute
        # nothing to any BHMF/QLF estimator anyway).
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

    # Storage-array layout: BH/Lbol arrays will be sized to n_subset rather
    # than the full halo count. ``global_to_storage`` maps each global track
    # ID (i.e. mt_track_ids[i] == i) to its position in the compact storage
    # arrays; non-subset halos map to -1 and must never be indexed.
    n_storage = n_subset
    global_to_storage = np.full(n_full_halos, -1, dtype=np.int64)
    global_to_storage[subset_indices] = np.arange(n_subset, dtype=np.int64)

    # Eager-load the subset slice of halo_masses and cold-accretion
    # rates into RAM, replacing the mmap'd full-N arrays. This is the
    # multi-node-friendly path: per-node halo-data footprint drops 4×
    # (66 GB private vs 270 GB OS cache). Single-run cost: ~3 min
    # upfront, slightly negative impact on wall time vs mmap; for the
    # training pipeline this is done once in the parent and amortized
    # over all forked workers.
    #
    # The per-snap access patterns in this script use _to_storage(idx)
    # for halo data — which only resolves to the right indices when
    # the halo arrays are subset-sized. So eager-load is mandatory in
    # subsample mode (not opt-in) to keep the access patterns valid.
    #
    # The first invocation extracts subset columns from the 370 GB mmap
    # row-by-row and writes a ~15 GB .npy beside the subset cache;
    # subsequent invocations skip the gather entirely and read the
    # small file sequentially (~30 s instead of ~11 min on the 10k box).
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

    # Pre-compute compact subset weights (size n_subset) so per-snap
    # statistics don't have to fancy-index the full-N subset_weights
    # array each time.
    subset_weights_compact = subset_weights[subset_indices].astype(np.float64)
else:
    subset = None
    subset_mask = None
    subset_weights = None
    subset_indices = None
    subset_weights_compact = None
    n_storage = halo_masses_all.shape[1]   # full halo count
    global_to_storage = None


def _to_storage(x):
    """Convert global indices/bool masks to storage-array positions.

    In subsample mode, ``global_to_storage[x]`` returns the compact-array
    position(s); in full-sim mode it's a pass-through. All masks coming in
    must already be subset-restricted (closure under mergers is the
    responsibility of the caller / mask plumbing).
    """
    if global_to_storage is None:
        return x
    return global_to_storage[x]


# ==============================================================================
# PRECOMPUTE PER-SNAPSHOT INDEX ARRAYS
# ==============================================================================
# Mirrors the optimisation in main_training.py: instead of rebuilding the
# full-N (mt_birth_index == i) / death / merger masks every iteration of
# the snapshot loop, we walk the snapshot range *once* up front and store
# the resulting integer index arrays. Each snapshot's lookup then becomes
# a dict access of a small int array instead of multiple 426M-element
# bool operations. Saves several seconds per snapshot, which is the
# dominant per-snap cost for subsampled runs (where the per-halo kernel
# work is small but the full-N mask machinery is unchanged).
#
# All arrays are GLOBAL indices, already restricted to ``subset_mask`` if
# subsampling is on. Storage-array positions are obtained at use site via
# ``_to_storage(...)`` — cheap because the int arrays are tiny.
print("Precomputing per-snapshot index arrays...")
# When subsampling, the helper subset-restricts the merger-tree arrays
# ONCE and operates on (n_subset,)-sized views, then maps results back to
# GLOBAL indices through subset_indices. ~50× fewer bool-ops per snap on
# L2800N10080 (~30 M vs ~1.5 B). Output identity vs the prior full-N
# flatnonzero pattern is unit-tested in tests/test_precomp_speedups.py.
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
        mask_disrupted=mask_disrupted,
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

    precomp_spawning = {}   # halos born at snap i
    precomp_evolving = {}   # halos alive at snap i (born before, not yet dead)
    precomp_merging = {}    # halos dying at snap i and merging into a target
    precomp_lost = {}       # halos dying at snap i with no merger target (disrupted)

    for _s in range(min_snap, max_snap + 1):
        _spawn = (mt_birth_index == _s)
        _alive = (mt_birth_index < _s) & (mt_birth_index != -1) & (_death_eff > _s)
        _dying = (mt_birth_index < _s) & (mt_birth_index != -1) & (mt_death_index == _s)
        precomp_spawning[_s] = np.flatnonzero(_spawn)
        precomp_evolving[_s] = np.flatnonzero(_alive)
        precomp_merging[_s] = np.flatnonzero(_dying & mask_merged)
        precomp_lost[_s] = np.flatnonzero(_dying & mask_disrupted)

    del _death_eff
    print(f"  Precomputed index arrays for {max_snap - min_snap + 1} snapshots in {time.time() - _t_pre:.2f}s")


# ==============================================================================
# LOAD LOGNORMAL SUM SAMPLER (Transfer Function)
# ==============================================================================
# This precomputed table accelerates sampling from sums of lognormal variables,
# used to draw total accretion over multiple sub-steps efficiently.
# Not needed when time_step_for_accretion == 0 (analytical mean growth).
if time_step_for_accretion == 0:
    transfer_function = None
    logger.info("SKIPPING transfer function load (time_step_for_accretion=0, analytical growth)")
else:
    time_here = time.time()
    logger.info("LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
    sampler_file = "Universal_Lognormal_Sampler_final.npz"
    transfer_function = load_3d_sampler(sampler_file)
    logger.info("FINISHED LOADING UNIVERSAL LOGNORMAL SUM SAMPLER")
    logger.info("Total time elapsed {} seconds".format(time.time() - time_here))

# Optional: force Branch A direct sub-step sampling (no transfer table) while
# KEEPING time_step_for_accretion / n_steps unchanged. Used to benchmark the fast
# sampler against brute-force direct sampling on a REAL run (appendix sampler
# figure). Distinct from BAQARO_TAU_COH_ZERO, which sets n_steps=0 -> Branch C
# (different physics). Branch A allocates an (n_evolving, n_steps) array, so this
# is only feasible on a SMALL subsample (use a small BAQARO_SUBSAMPLE_NB).
if env_bool("BAQARO_FORCE_NO_SAMPLER", False):
    transfer_function = None
    logger.info("BAQARO_FORCE_NO_SAMPLER=1 -> transfer_function=None "
                "(Branch A direct sampling, n_steps unchanged)")


# ==============================================================================
# BLACK HOLE EVOLUTION LOOP
# ==============================================================================
logger.info("STARTING BH EVOLUTION ROUTINE")

# --- Allocate Output Arrays ---
# float32 for memory efficiency (~0.01% precision loss, saves 50% RAM)
# Storage arrays are sized to ``n_storage`` — equal to the full halo count
# when subsampling is off, equal to the closed subset size when on. Index
# into them with ``_to_storage(global_idx_or_mask)``.
total_number_of_objects = halo_masses_all.shape[1]
black_hole_masses_all = np.zeros((len(snapshots), n_storage), dtype=np.float32)
Lbols_all = np.zeros((len(snapshots), n_storage), dtype=np.float32)

# --- Pre-allocate Reusable Buffers ---
# Avoids repeated allocation of large arrays inside the snapshot loop.
# Sized to ``n_storage`` (upper bound on n_evolving any snapshot can produce);
# sliced to actual size each iteration.
_buf_bh_prev = np.empty(n_storage, dtype=np.float32)
_buf_log_rates = np.empty(n_storage, dtype=np.float32)
_buf_Lbols_out = np.empty(n_storage, dtype=np.float64)
_rng_buffers = {
    'z_last': np.empty(n_storage, dtype=np.float32),
    'u': np.empty(n_storage, dtype=np.float64),
    'z_approx': np.empty(n_storage, dtype=np.float32),
}

start_time_bh = time.time()

# --- Counters for Summary Statistics ---
lost_black_holes_total = 0
new_black_holes_total = 0
merged_black_holes_total = 0

# --- Initialize Variables for Debug Block ---
# These may not be set if merger_on=False, but debug block references them
mass_lost_BHs = 0.0
mass_from_aborted_mergers = 0.0

# --- Merger catalogue accumulator ---
catalog_builder = MergerCatalogBuilder() if (merger_on and record_merger_catalog) else None

# --- Main Loop Over Snapshots ---
for i in snapshots:
    # ------------------------------------------------------------------
    # SNAPSHOT SETUP
    # ------------------------------------------------------------------
    logger.info("#" * 50)
    logger.info("Snapshot {}, redshift {:.2f}".format(i, redshifts[i]))
    logger.info("Time since last snapshot: {:.2f} Myr".format(delta_times_snapshots[i] * 1e3))

    # Calculate number of sub-steps for accretion
    # time_step_for_accretion == 0 means fully independent draws (Branch C)
    if time_step_for_accretion == 0:
        n_steps = 0
        logger.info("Number of sub-steps: 0 (independent draws, analytical mean growth)")
    else:
        # Clamp to >=1: when tau_coh > dt, fall through to n_steps=1 (single
        # coherent draw covering the whole snapshot) rather than int()-truncating
        # to 0, which would fall into Branch C (the tau_coh -> 0 limit) and
        # decouple L_bol from mass build.
        n_steps = max(1, int(delta_times_snapshots[i] * 1e3 / time_step_for_accretion))
        logger.info("Number of sub-steps: {}".format(n_steps))
        logger.info("Actual timestep: {:.2f} Myr (target: {} Myr)".format(
            delta_times_snapshots[i] * 1e3 / n_steps, time_step_for_accretion))

    # ------------------------------------------------------------------
    # LOAD CURRENT SNAPSHOT DATA
    # ------------------------------------------------------------------
    # Read entire column at once (sequential disk access, much faster than random)
    current_halo_masses = halo_masses_all[i]
    if accretion_on:
        current_rates = halo_specific_cold_accretion_rates_all[i]

    # ------------------------------------------------------------------
    # PER-SNAPSHOT INDEX ARRAYS (precomputed once outside the loop)
    # ------------------------------------------------------------------
    # GLOBAL halo indices for mmap access; storage positions via
    # ``_to_storage(...)`` at each bh/Lbol array operation. Already
    # restricted to ``subset_mask`` if subsampling is on.
    idx_spawning = precomp_spawning[i]
    idx_evolving = precomp_evolving[i]
    idx_merging = precomp_merging[i]
    idx_lost = precomp_lost[i]

    # ------------------------------------------------------------------
    # STEP 1: SEED NEW BLACK HOLES
    # ------------------------------------------------------------------
    start_time_here = time.time()
    if idx_spawning.size > 0:
        # In subsample mode current_halo_masses is the eager-loaded
        # subset row (shape n_subset), so we index by storage positions.
        # In full-sim mode it's the full mmap row and _to_storage is
        # identity — same line works in both modes.
        idx_spawning_storage = _to_storage(idx_spawning)
        halo_slice = current_halo_masses[idx_spawning_storage]
        spawned_now = spawn_BHs(halo_slice, logfseed, sigmaseed=sigmaseed, rng=rng)
        black_hole_masses_all[i, idx_spawning_storage] = spawned_now
        mass_new_BHs = np.sum(spawned_now)
    else:
        spawned_now = np.empty(0, dtype=np.float32)
        mass_new_BHs = 0.0

    spawned_count = int(idx_spawning.size)
    logger.info("New black holes {}".format(spawned_count))
    new_black_holes_total += spawned_count
    logger.info("Initialization done in {:.2f}s".format(time.time() - start_time_here))


    # ------------------------------------------------------------------
    # STEP 2: ACCRETION
    # ------------------------------------------------------------------
    # Evolve existing black holes via gas accretion. Accretion rates are
    # drawn from the ERDF, scaled by halo cold gas rate.
    start_time_here = time.time()
    n_evolving = int(idx_evolving.size)
    idx_evolving_storage = _to_storage(idx_evolving)

    if accretion_on and n_evolving > 0:
        bh_prev = _buf_bh_prev[:n_evolving]
        bh_prev[:] = black_hole_masses_all[i - 1, idx_evolving_storage]

        log_halo_rates = _buf_log_rates[:n_evolving]
        # current_rates is eager-loaded subset row in subsample mode,
        # full mmap row otherwise — _to_storage matches the array shape.
        np.take(current_rates, idx_evolving_storage, out=log_halo_rates)
        # Non-accreting default: floor=0 lets a zero rate (plateaued LastMaxMass)
        # flow to log10(0) = -inf -> eta = exp(-inf) = 0 in the ERDF/kernel ->
        # M_BH frozen, L_bol = 0 (verified NaN-clean). No post-process mask
        # needed. Legacy clip-floor (BAQARO_NONACCRETING_ZERO_RATE=0) uses 1e-8.
        _rate_floor = 0.0 if NONACCRETING_ZERO_RATE else 1e-8
        np.clip(log_halo_rates, _rate_floor, 1e3, out=log_halo_rates)
        with np.errstate(divide="ignore"):
            np.log10(log_halo_rates, out=log_halo_rates)

        bh_next, Lbol_next = evolve_BHs(
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
            use_parallel_kernel=use_parallel_kernel,
            fw_p_threshold=0.7 if fw_approx else 0.0,
            print_info=print_debug_info,
            L_bols_out=_buf_Lbols_out[:n_evolving],
            rng_buffers=_rng_buffers,
            growth_max=growth_sum_max,
            madau_feff_correction=madau_feff_correction,
        )

        black_hole_masses_all[i, idx_evolving_storage] = bh_next
        Lbols_all[i, idx_evolving_storage] = Lbol_next
    elif (not accretion_on) and n_evolving > 0:
        # No accretion: just copy masses from previous snapshot
        black_hole_masses_all[i, idx_evolving_storage] = black_hole_masses_all[i - 1, idx_evolving_storage]

    logger.info("Evolving black holes: {}".format(n_evolving))
    logger.info("Accretion done in {:.2f}s".format(time.time() - start_time_here))

    # ------------------------------------------------------------------
    # STEP 3: MERGERS AND DISRUPTIONS
    # ------------------------------------------------------------------
    # Handle halos that die this snapshot:
    #   - Merged halos: add their BH mass to the merger target's BH
    #   - Disrupted halos: BH mass is lost (set to zero)
    if merger_on:
        start_time_here = time.time()

        # NOTE: Mergers assumed to happen before accretion in dying snapshot.
        # This is acceptable for small timesteps.

        # idx_merging: GLOBAL indices from precomp (halos dying & merged
        # at this snap, subset-restricted). src_ids / dest_ids: GLOBAL
        # track IDs (mmap-indexable). Storage positions are computed via
        # _to_storage where bh/Lbol access happens.
        src_ids = mt_track_ids[idx_merging]
        dest_ids = mt_merger_ids[idx_merging]

        # Targets must be alive at the merger snapshot: a merger onto a dead
        # or unresolved target is aborted and the source mass is lost. Halo
        # mass is frozen LastMaxMass, so ``> 0`` alone means "ever resolved",
        # not "alive", hence the death-index test. A target dying AT this
        # snapshot is kept:
        # it is itself a merger source and chain resolution forwards the
        # deposit. current_halo_masses is the eager-loaded subset row in
        # subsample mode; _to_storage maps dest_ids (global) to subset
        # positions. In full-sim mode _to_storage is identity.
        dest_death = mt_death_index[dest_ids]
        # _to_storage returns -1 for a target outside the subset; NumPy would
        # read that as the LAST storage slot, so such mergers must be aborted
        # explicitly rather than tested through the mass array.
        dest_pos_all = _to_storage(dest_ids)
        dest_in_storage = dest_pos_all >= 0
        mask_halos_are_alive = dest_in_storage & (
            current_halo_masses[np.where(dest_in_storage, dest_pos_all, 0)] > 0.0
        ) & ((dest_death == -1) | (dest_death >= i))
        if print_debug_info:
            src_ids_not_merged = src_ids[~mask_halos_are_alive]
            mass_from_aborted_mergers = np.sum(
                black_hole_masses_all[i - 1, _to_storage(src_ids_not_merged)], dtype=np.float64
            )

        src_ids = src_ids[mask_halos_are_alive]
        dest_ids = dest_ids[mask_halos_are_alive]
        # Storage positions for the surviving (src, dest) pairs. Pre-computed
        # once; reused for catalog, debug, chain resolution, and the np.add.at
        # accumulation.
        src_pos = _to_storage(src_ids)
        dest_pos = _to_storage(dest_ids)

        # --- Record the merger catalogue (before chain resolution) ---
        # Captures binary-event structure: chains like A->B->C and
        # multi-source-into-same-dest (A->D, B->D) are decomposed into
        # individual binary events (A,B), (B,C), (A,D), (B,D) with their
        # running M2 values. The dest masses passed here are post-accretion
        # but pre-merger (np.add.at runs below). src_indices / dest_indices
        # passed to the catalog stay as GLOBAL track IDs (the catalog HDF5
        # schema uses global IDs).
        if catalog_builder is not None and len(src_ids) > 0:
            catalog_builder.record_snapshot_mergers(
                z_snap=float(redshifts[i]),
                src_indices=src_ids,
                dest_indices=dest_ids,                 # immediate, NOT chain-resolved
                src_masses=black_hole_masses_all[i - 1, src_pos],
                dest_masses_initial=black_hole_masses_all[i, dest_pos],
                track_ids=mt_track_ids,
            )

        if print_debug_info:
            dest_pos_unique = np.unique(dest_pos)
            print("Mass pre mergers", np.sum(black_hole_masses_all[i, dest_pos_unique]))
            copy_debug = np.copy(black_hole_masses_all[i, dest_pos_unique])
        else:
            copy_debug = None

        # --- Resolve Merger Chains ---
        # Handle chains like A→B→C by iteratively following the mapping
        # until we find the final destination for each source. Operates
        # entirely in STORAGE-position space — ``mapping`` is sized to
        # n_storage rather than the full halo count, which is the bulk of
        # the per-snapshot memory savings under subsampling.
        if len(src_ids) > 0:
            mapping = np.full(n_storage, -1, dtype=dest_pos.dtype)
            mapping[src_pos] = dest_pos
            final_dest_pos = dest_pos.copy()
            for _ in range(len(src_pos)):  # Max iterations = chain length
                next_dest_pos = mapping[final_dest_pos]
                mask_chain = next_dest_pos != -1
                if not np.any(mask_chain):
                    break  # No more chains to resolve
                final_dest_pos[mask_chain] = next_dest_pos[mask_chain]
            dest_pos = final_dest_pos

        # --- Accumulate Merged Masses ---
        # Use np.add.at for safe accumulation when multiple sources merge into same dest
        np.add.at(black_hole_masses_all[i], dest_pos, black_hole_masses_all[i - 1, src_pos])

        if print_debug_info:
            print("Amount of mass merged", np.sum(black_hole_masses_all[i - 1, src_pos]))
            print("Mass post mergers", np.sum(black_hole_masses_all[i, np.unique(dest_pos)]),
                  " (should be equal to: {})".format(
                      np.sum(black_hole_masses_all[i - 1, src_pos]) + np.sum(copy_debug)))

        merged_black_holes_total += src_ids.size
        logger.info("Merged black holes: {}".format(src_ids.size))

        # --- Handle Disrupted Halos ---
        # These halos are destroyed; their BH mass is lost
        # (Mass already zero by default, just count them). idx_lost comes
        # from precomp (dying & disrupted at this snap, subset-restricted).
        lost_count = int(idx_lost.size)
        lost_black_holes_total += lost_count
        if print_debug_info:
            mass_lost_BHs = np.sum(
                black_hole_masses_all[i - 1, _to_storage(idx_lost)], dtype=np.float64
            )

        logger.info("Disrupted black holes: {}".format(lost_count))
        logger.info("Mergers + disruptions done in {:.2f}s".format(time.time() - start_time_here))

    # ------------------------------------------------------------------
    # DEBUG: MASS CONSERVATION CHECK
    # ------------------------------------------------------------------
    # Verify: mass_after = mass_before + spawned - lost + accreted
    if print_debug_info:
        # Use float64 to avoid float32 rounding errors in sums
        total_mass_BHs = np.sum(black_hole_masses_all[i], dtype=np.float64)
        print("#" * 50)
        print("Total mass in black holes:", total_mass_BHs)

        mass_before = np.sum(black_hole_masses_all[i - 1], dtype=np.float64)
        mass_after = np.sum(black_hole_masses_all[i], dtype=np.float64)
        spawned = np.sum(mass_new_BHs, dtype=np.float64) if np.ndim(mass_new_BHs) else float(mass_new_BHs)
        # Original 'lost' term (disrupted BHs)
        lost_disrupted = mass_lost_BHs # Sum of M(i-1) for BHs in mask_lost_BHs
        
        # Adjusted lost term including aborted mergers
        effective_lost = lost_disrupted + mass_from_aborted_mergers
        
        accreted    = (mass_after - mass_before) - (spawned - effective_lost)
        
        print(f"Snapshot {i}: mass_before={mass_before:.6g}, mass_after={mass_after:.6g}")
        print(f"Snapshot {i}: spawned_mass={spawned:.6g}, lost_disrupted_mass(i-1)={lost_disrupted:.6g}, mass_from_aborted_mergers(i-1)={mass_from_aborted_mergers:.6g}")
        print(f"Snapshot {i}: effective_lost_mass(i-1)={effective_lost:.6g}")
        print(f"Snapshot {i}: calculated 'accreted' mass = {accreted:.6g}")
        
        # Relative/absolute tolerance: allow tiny float32 drift
        tol = max(1e-8 * mass_after, 1e-3)
        if accretion_on is False and abs(accreted) > tol:
            print(f"Warning: unexpected mass change at snapshot {i}: delta_M_total={mass_after-mass_before:.6g}, spawned_M - effective_lost_M={spawned-effective_lost:.6g}, diff(accreted)={accreted:.6g}, tol={tol:.3g}")



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

# Cast to float32 to prevent type promotion (avoids temporary 2x memory allocation)
mass_units_f32 = np.float32(mass_units)
black_hole_masses_all *= mass_units_f32  # In-place multiplication
Lbols_all *= mass_units_f32

# Note: halo_masses_all is mmap'd read-only, so we must create a copy
# However, halo masses are not needed in the output file, so we can skip this step entirely to save time and memory.
# halo_masses_all = halo_masses_all * mass_units_f32


# ==============================================================================
# SAVE OUTPUT
# ==============================================================================
logger.info("Black hole mass array size: {:.2f} GB".format(black_hole_masses_all.nbytes / 1e9))
logger.info("Luminosity array size: {:.2f} GB".format(Lbols_all.nbytes / 1e9))




time_here = time.time()
logger.info("Saving the arrays to an HDF5 file")
# save the arrays to an HDF5 file (no compression, no chunking - maximum write speed)
with h5py.File(path_file, "w") as file:
    file.create_dataset("snapshots", data=snapshots)
    file.create_dataset("redshifts", data=redshifts)
    file.create_dataset("ages_of_the_universe", data=ages_of_the_universe)
    file.create_dataset("delta_times_snapshots", data=delta_times_snapshots)

    # saving the main arrays in the [snapshot, halo] order for better access patterns later
    # All arrays already in [snapshot, halo] order — no transpose needed
    # file.create_dataset("halo_masses_all", data=halo_masses_all)
    # bh/Lbol arrays are already sized to n_storage — write them out directly.
    file.create_dataset("black_hole_masses_all", data=black_hole_masses_all)
    file.create_dataset("Lbols_all", data=Lbols_all)

    if subset_mask is not None:
        # Subset metadata so downstream tools can:
        # (a) map storage positions back to global track IDs via subset_indices
        # (b) compute weighted summary statistics using per-halo weights
        subset_grp = file.create_group("subset")
        subset_grp.create_dataset("subset_indices", data=subset_indices)
        subset_grp.create_dataset("weights", data=subset_weights[subset_indices])
        subset_grp.create_dataset("root_mask", data=subset["root_mask"][subset_indices])
        subset_grp.create_dataset("root_bin", data=subset["root_bin"][subset_indices])
        subset_grp.create_dataset("log_mass_bins", data=subset["log_mass_bins"])
        subset_grp.create_dataset("n_per_bin_total", data=subset["n_per_bin_total"])
        subset_grp.create_dataset("n_per_bin_kept", data=subset["n_per_bin_kept"])
        subset_grp.create_dataset("weights_per_bin", data=subset["weights_per_bin"])
        subset_grp.attrs["root_snap"] = int(subsample_root_snap)
        subset_grp.attrs["N_b_per_bin"] = int(subset["meta"]["N_b_per_bin"])
        # ALSO persist the full per-bin N_b array.
        # `N_b_per_bin` is the `-1` SENTINEL in schedule mode (non-uniform N_b), so a
        # schedule-mode evolution file separated from its subset .npz could not
        # reconstruct its own sampling design — the provenance group claimed a
        # meaningless -1. `tree_subsample` already carries `N_b_array` in meta; we
        # simply never wrote it. Provenance-only; no numerical effect.
        _nb_arr = subset["meta"].get("N_b_array")
        if _nb_arr is not None:
            subset_grp.create_dataset(
                "N_b_array", data=np.asarray(_nb_arr, dtype=np.int64))
        subset_grp.attrs["rng_seed"] = int(subset["meta"]["rng_seed"])
        subset_grp.attrs["n_roots"] = int(subset["meta"]["n_roots"])
        subset_grp.attrs["n_halos_full"] = int(subset["meta"]["n_halos_full"])
        subset_grp.attrs["tag"] = subset_tag
    
    # file.create_group("merger_trees")
    # file.create_dataset("merger_trees/track_ids", data=merger_trees.track_ids)
    # file.create_dataset("merger_trees/merger_track_ids", data=merger_trees.merger_track_ids)
    # file.create_dataset("merger_trees/snapshot_indexes_of_birth", data=merger_trees.snapshot_indexes_of_birth)
    # file.create_dataset("merger_trees/snapshot_indexes_of_death", data=merger_trees.snapshot_indexes_of_death)


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
    # Save the original seed used for reproducibility.
    # Use the RESOLVED int we seeded with, NOT
    # `rng.bit_generator._seed_seq.entropy`: that is PRIVATE NumPy API, and it
    # would be touched HERE, at save time, i.e. AFTER the multi-hour snapshot loop. A NumPy rename would turn
    # the whole run into an AttributeError at the finish line and leave a truncated
    # HDF5. `RNG_SEED` is the same value, read from BAQARO_RNG_SEED at startup.
    file.create_dataset("metadata/rng_seed", data=RNG_SEED)

    # --- Self-describing run provenance (see utils/provenance.py) ---
    # Open the file later and read f["provenance"].attrs (or f.attrs) to recover
    # exactly how this run was produced: identity tokens + the 6 params + git
    # revision + the full BAQARO_* env snapshot.
    write_run_provenance(file, {
        "engine": "main_evolution",
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
        "bestfit_name": bestfit_name,
        "name_file": name_file,
        "use_subsample": bool(use_subsample),
        "subset_tag": subset_tag if use_subsample else "",
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


# ==============================================================================
# Save the merger catalogue (separate HDF5 file)
# ==============================================================================
if catalog_builder is not None:
    catalog_save_start = time.time()
    catalog_data = catalog_builder.finalize()
    n_events = catalog_data["z"].size
    logger.info(
        "Saving merger catalog: %d binary events across %d snapshots",
        n_events,
        len(catalog_builder.z_per_event),
    )

    # Survivors at z=0 (last snapshot in `snapshots`).
    # survivor_storage_pos are positions in the storage array (== global IDs
    # in full-sim mode, == subset positions in subsample mode). The catalog
    # records survivors by their GLOBAL track ID, so map back via
    # subset_indices when subsampling is on.
    last_snap = int(snapshots[-1])
    survivor_mask_z0 = black_hole_masses_all[last_snap, :] > 0.0
    survivor_storage_pos = np.flatnonzero(survivor_mask_z0)
    if subset_indices is not None:
        survivor_global = subset_indices[survivor_storage_pos]
    else:
        survivor_global = survivor_storage_pos
    survivor_track_ids = mt_track_ids[survivor_global].astype(np.int64)
    survivor_M_z0 = black_hole_masses_all[last_snap, survivor_storage_pos].astype(np.float32)
    logger.info(
        "Survivors at z=%.3f: %d halos with M_BH > 0",
        float(redshifts[last_snap]),
        survivor_track_ids.size,
    )

    # V_sim convention: boxsize=2800 Mpc
    # (no h factor) -> V_sim = 2800^3 Mpc^3 ~ 22 Gpc^3. If you re-interpret
    # boxsize as Mpc/h instead, multiply V_sim by 1/h^3 ~ 3.2 (h~0.681).
    V_sim_Mpc3 = float(boxsize) ** 3

    catalog_path = os.path.join(
        path_out, "evolution", f"merger_catalog_{name_file}.hdf5",
    )
    # All masses below are in Msun, all volumes in Mpc^3 (no astropy units in
    # the file itself -- matches the convention used by bh_evolution_*.hdf5
    # and the halo-history .npy files; downstream consumers attach units on
    # read).
    #
    # IMPORTANT: catalog_builder records masses DURING the snapshot loop, when
    # black_hole_masses_all is still in CODE UNITS (1 code unit = mass_units
    # = 1e7 Msun). The `*= mass_units_f32` conversion happens above (just
    # before the bh_evolution save) and applies to the array; but the
    # catalog buffers were already populated in code units. Convert to Msun
    # here, just before writing.
    M1_Msun = catalog_data["M1_Msun"].astype(np.float64) * mass_units
    M2_Msun = catalog_data["M2_Msun"].astype(np.float64) * mass_units
    with h5py.File(catalog_path, "w") as f:
        f.create_dataset("z", data=catalog_data["z"])
        f.create_dataset("M1", data=M1_Msun.astype(np.float32))
        f.create_dataset("M2", data=M2_Msun.astype(np.float32))
        f.create_dataset("progenitor1_id", data=catalog_data["progenitor1_id"])
        f.create_dataset("progenitor2_id", data=catalog_data["progenitor2_id"])
        f.create_dataset("descendant_id",  data=catalog_data["descendant_id"])
        f.create_dataset("halo_id",        data=catalog_data["halo_id"])
        f.create_dataset("survivor_id",    data=survivor_track_ids)
        f.create_dataset("survivor_M_z0",  data=survivor_M_z0)

        # Root attrs: V_sim and pass-through metadata.
        f.attrs["V_sim_Mpc3"] = V_sim_Mpc3
        f.attrs["boxsize"] = float(boxsize)
        f.attrs["boxsize_units_assumed"] = "Mpc (no h factor)"
        f.attrs["simulation_name"] = simulation_name
        f.attrs["max_snap"] = int(max_snap)
        f.attrs["min_snap"] = int(min_snap)
        f.attrs["notes_file"] = notes_file or ""
        f.attrs["erdf_model"] = erdf_model
        f.attrs["nbound_threshold"] = int(nbound_threshold)
        f.attrs["halo_filtering_mode"] = halo_filtering_mode
        f.attrs["tdyn_fraction_default"] = float(tdyn_fraction_default)
        f.attrs["catalog_schema_version"] = "1.0"

    logger.info(
        "Catalog saved to %s in %.2fs",
        catalog_path,
        time.time() - catalog_save_start,
    )

logger.info("Total time elapsed for the run {} seconds".format( time.time() - start_time_global))




