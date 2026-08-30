"""
Unified MCMC inference script for black hole population statistics.

This script constrains a 6-parameter BH evolution model by comparing
GP-emulated summary statistics against observational data.  It supports
three likelihood components that can be toggled independently:

  - **QLF** (Quasar Luminosity Function):  Number density of quasars as a
    function of luminosity and redshift.  Two likelihood modes are available:
      * ``"gaussian"`` -- standard chi-squared on log(Phi).
      * ``"poisson"``  -- hybrid Cash (1979) / Gaussian statistic that treats
        low-count bins with the Poisson likelihood and high-count bins with
        a Gaussian approximation, plus a non-detection penalty for emulator
        bins brighter than the brightest observed data point.

  - **CERDF** (Conditional Eddington Ratio Distribution Function):  Per-QSO
    likelihood P(log_eta | L_bol, z, theta) evaluated for individual objects,
    filtered by bolometric luminosity range and emulator redshift coverage.

  - **Correlation function** (auto / cross):  Projected auto-correlation
    (wp/rp) or volume-averaged cross-correlation (xi) computed from the
    quasar halo mass function (QHMF) emulator.

Workflow
--------
1. Load GP emulators for the active likelihood components.
2. Load and precompute observational data for each active component.
3. Build a combined log-probability function (single-component: no weighting
   overhead; multi-component: ``ll_total = sum(ll_i / weight_i)``).
4. Run differential evolution (DE) optimization with optional seed-point
   injection and local refinement (Nelder-Mead + L-BFGS-B).
5. If ``optimize_only`` is ``False``, launch an ``emcee`` ensemble MCMC
   sampler initialized in a tight ball around the DE optimum.
6. Print diagnostics (acceptance fraction, autocorrelation time) and produce
   a corner plot.

Configuration
-------------
All user-facing knobs are at the top of the file:

- ``LIKELIHOOD_FLAGS``: dict of bool -- which components to enable.
- ``QLF_LIKELIHOOD_MODE``: ``"gaussian"`` or ``"poisson"``.
- ``LIKELIHOOD_WEIGHTS``: relative weight per component (1.0 = raw log-likelihood).
- ``CONFIG_MCMC``: MCMC hyper-parameters (walkers, steps, burn-in, optimizer
  settings, seed point, etc.).
- ``mask_redshifts_qlf`` / ``mask_redshifts_corr``: redshift exclusion lists
  (matched with 0.3 dex tolerance).
- ``cerdf_Lbol_range``: bolometric luminosity filter for CERDF data.

Output
------
An HDF5 file written by ``emcee.backends.HDFBackend`` into
``{output_path}/mcmc/mcmc_{name_file}_{active_likelihoods}[_{notes}].h5``.
The active-likelihood tag is constructed by joining the enabled flag names
with ``+`` (e.g. ``qlf+cerdf``).
"""

# This module is a run script, not a library: its body loads catalogues and
# writes products at import time. Refuse a plain ``import`` so nobody starts a
# multi-hour job by accident (``python -m baqaro.inference.main_mcmc`` sets __name__ to
# "__main__"; a multiprocessing "spawn" child re-imports it as "__mp_main__").
if __name__ not in ("__main__", "__mp_main__"):
    raise RuntimeError(
        "baqaro.inference.main_mcmc is an entry-point script; run it with "
        "'python -m baqaro.inference.main_mcmc' instead of importing it."
    )


import os
import time
import numpy as np
import scipy.optimize
import emcee
import corner
import matplotlib.pyplot as plt
from pathlib import Path

# Project-internal modules
from qhtools.utils import my_utils          # unit conversions (e.g. to_solar)
import qhtools.utils.natconst as nc          # natural constants (log_csi = log10(L_Edd / M_BH))
from baqaro.emulation.loading_helpers import load_emulators
from baqaro.inference import likelihoods_and_priors
from baqaro.utils.my_dir import get_output_path


# ==========================================================================
# CONFIGURATION
# ==========================================================================

# Which likelihood components to include in the joint inference.
# Any combination is supported.  The output filename encodes the active set.
# Default is QLF + corr. Override per-run with BAQARO_LIKELIHOODS env var:
#   BAQARO_LIKELIHOODS=qlf            # QLF-only smoke test
#   BAQARO_LIKELIHOODS=qlf,cerdf      # both
#   BAQARO_LIKELIHOODS=qlf,cerdf,corr # full joint
# Useful when not all emulators are trained yet.
# Default: the adopted fiducial (`qcc_ck22final_v1`) is the joint
# qlf+cerdf+corr fit. Override per run with BAQARO_LIKELIHOODS=qlf,corr etc.
_LIKELIHOOD_DEFAULT = {"qlf": True, "cerdf": True, "corr": True}
_likelihoods_env = os.environ.get("BAQARO_LIKELIHOODS", "").strip()
if _likelihoods_env:
    _enabled = {k.strip() for k in _likelihoods_env.split(",") if k.strip()}
    LIKELIHOOD_FLAGS = {k: (k in _enabled) for k in ("qlf", "cerdf", "corr")}
else:
    LIKELIHOOD_FLAGS = _LIKELIHOOD_DEFAULT

# QLF likelihood mode:
#   "gaussian" -- standard chi-squared on log(Phi) with observational errors.
#   "poisson"  -- hybrid Cash (1979) / Gaussian: uses Poisson likelihood for
#                 low-count bins (N_eff < threshold) and Gaussian for well-
#                 sampled bins, plus a non-detection penalty term beyond the
#                 brightest observed data point.
QLF_LIKELIHOOD_MODE = "poisson"

# CERDF likelihood mode:
#   "per_object" -- per-QSO product P(log_eta_i | L_i, z_i, theta), ~9000 terms.
#   "binned"     -- chi-squared on observed vs model log_eta histograms in
#                   (redshift x luminosity) cells, ~100-400 effective data points.
#                   Better suited for joint inference with QLF (comparable n_data).
#                   Production default.
# Override with BAQARO_CERDF_MODE=per_object (see the per-object knobs
# BAQARO_CERDF_TEMP / _SCATTER_DEX / _OUTLIER_FRAC below).
# Default: the adopted fiducial (`qcc_ck22final_v1`) uses the UNBINNED
# (per-object) cERDF. Override with BAQARO_CERDF_MODE=binned for the binned estimator.
CERDF_LIKELIHOOD_MODE = (os.environ.get("BAQARO_CERDF_MODE", "per_object").strip()
                         or "per_object")
if CERDF_LIKELIHOOD_MODE not in ("binned", "per_object"):
    raise ValueError(f"BAQARO_CERDF_MODE must be 'binned' or 'per_object'; "
                     f"got '{CERDF_LIKELIHOOD_MODE}'")
if CERDF_LIKELIHOOD_MODE != "binned":
    print(f"[cerdf-env] BAQARO_CERDF_MODE={CERDF_LIKELIHOOD_MODE}")

# Luminosity and Eddington ratio bins for the binned CERDF mode.
# Only used when CERDF_LIKELIHOOD_MODE == "binned".
CERDF_BINNED_LOGL_BINS = [45.5, 46.0, 46.5, 47.0, 47.5]
# Production default: 4 bins × 1.0 dex, matching the SDSS sample's effective
# resolution. Override per run with BAQARO_CERDF_LOG_ETA_BINS.
CERDF_BINNED_LOG_ETA_BINS = np.linspace(-2.5, 1.5, 5)  # 4 bins × 1.0 dex
CERDF_BINNED_MIN_COUNT = 5        # min QSOs per (z, L) cell

# Env overrides (for binning-sensitivity experiments):
#   BAQARO_CERDF_LOGL_BINS=45.5,46.5,47.5        # comma-sep L_bol bin edges
#   BAQARO_CERDF_LOG_ETA_BINS=-2.5,-1.5,-0.5,0.5,1.5  # comma-sep log_eta bin edges
#   BAQARO_CERDF_Z_MIN=1.0  BAQARO_CERDF_Z_MAX=6.0      # restrict QSO redshift sample
_env_logL = os.environ.get("BAQARO_CERDF_LOGL_BINS", "").strip()
if _env_logL:
    CERDF_BINNED_LOGL_BINS = [float(s) for s in _env_logL.split(",") if s.strip()]
    print(f"[cerdf-env] BAQARO_CERDF_LOGL_BINS={CERDF_BINNED_LOGL_BINS}")
_env_eta = os.environ.get("BAQARO_CERDF_LOG_ETA_BINS", "").strip()
if _env_eta:
    CERDF_BINNED_LOG_ETA_BINS = np.array([float(s) for s in _env_eta.split(",") if s.strip()])
    print(f"[cerdf-env] BAQARO_CERDF_LOG_ETA_BINS={CERDF_BINNED_LOG_ETA_BINS.tolist()}")
_env_minc = os.environ.get("BAQARO_CERDF_MIN_COUNT", "").strip()
if _env_minc:
    CERDF_BINNED_MIN_COUNT = int(_env_minc)
    print(f"[cerdf-env] BAQARO_CERDF_MIN_COUNT={CERDF_BINNED_MIN_COUNT}")
# Production default: the QSO sample window for the CERDF fit is
# [0.5, 6.5]. Paired with the default z_centers_use = {1, 2, 3, 3.94, 5.02, 6.14}
# (set in `precompute_cerdf_inputs_binned`), this gives each
# integer-z cell its FULL ±0.5 dex z window without any cell extending below
# Z_MIN or above Z_MAX:
#   z=1.0 cell window [0.5, 1.5)  <- lower edge AT Z_MIN; no truncation
#   z=2.0, 3.0 cells fully interior
#   z=6.14 cell window [5.64, 6.64) -> truncated at 6.5 (last 0.14 dex of the
#         emulator's z=6.14 window is dropped, since the SDSS sample's upper
#         data limit is anyway near z~5.5, this is harmless)
# To recover the legacy unrestricted behaviour: BAQARO_CERDF_Z_MIN=0
# BAQARO_CERDF_Z_MAX=10 BAQARO_CERDF_Z_USE=all
CERDF_Z_MIN = float(os.environ.get("BAQARO_CERDF_Z_MIN", "0.5") or 0.5)
CERDF_Z_MAX = float(os.environ.get("BAQARO_CERDF_Z_MAX", "6.5") or 6.5)
if (CERDF_Z_MIN, CERDF_Z_MAX) != (0.5, 6.5):
    print(f"[cerdf-env] CERDF QSO z window: {CERDF_Z_MIN} <= z <= {CERDF_Z_MAX}")

# Relative weight for each likelihood component in the joint log-probability.
# When multiple components are active: ll_total = sum(ll_i / weight_i).
# weight=1.0 means raw log-likelihood; setting weight=n_data normalizes
# per data point, useful when component scales differ (e.g. QLF ~50 points
# vs CERDF ~9000 individual QSOs).
LIKELIHOOD_WEIGHTS = {
    "qlf": 1.0,
    "cerdf": 1.0,
    "corr": 1.0,
}

# --- CERDF likelihood tempering (BAQARO_CERDF_TEMP) --------------------------
# The joint log-probability is  sum_i ll_i / weight_i,  so setting the CERDF
# weight to T is exactly likelihood tempering:  L_cerdf -> L_cerdf^(1/T).
#
# This matters almost entirely for per-object mode: the unbinned estimator has
# far more terms than the QLF and clustering likelihoods, so T sets its
# effective weight in the joint fit and is reported as a systematic.
#
#   BAQARO_CERDF_TEMP=<float>   divide the CERDF log-likelihood by T
#   BAQARO_CERDF_TEMP=auto      T = N_QSO / n_cells_binned, i.e. give the
#                              unbinned estimator the SAME total weight as the
#                              binned likelihood it replaces.  This is the one
#                              choice with a concrete interpretation, and it
#                              makes the run directly comparable to every
#                              binned chain we already have.
# Default: the unbinned cERDF is tempered at T=5000 (the adopted fiducial's
# value; quoted as a systematic). BAQARO_CERDF_TEMP=1 -> weights unchanged.
# T only makes sense WITH clustering; a cerdf-only run should set BAQARO_CERDF_TEMP=1.
CERDF_TEMP = (os.environ.get("BAQARO_CERDF_TEMP", "").strip() or "5000")
if CERDF_TEMP != "auto":
    _t = float(CERDF_TEMP)
    if _t <= 0:
        raise ValueError(f"BAQARO_CERDF_TEMP must be > 0 or 'auto'; got {_t}")
    LIKELIHOOD_WEIGHTS["cerdf"] = _t
    if _t != 1.0:
        print(f"[cerdf-env] BAQARO_CERDF_TEMP={_t} -> ll_cerdf / {_t}")

CONFIG_MCMC = {
    "rng_seed": 423897,           # Reproducibility seed for RNG + DE optimizer
    "n_walkers": int(os.environ.get("BAQARO_MCMC_NWALKERS") or 50),
    # n_steps default 10k; BAQARO_MCMC_NSTEPS overrides (e.g. =100000 for 1e5 long chains).
    "n_steps": int(os.environ.get("BAQARO_MCMC_NSTEPS") or 10000),
    "mcmc_filename_notes": (os.environ.get("BAQARO_MCMC_NOTES") or None),  # Custom tag appended to output filename (None = omit). Env: BAQARO_MCMC_NOTES (e.g. "minL05" so a re-fit doesn't clobber the baseline chain).
    "vectorize": True,            # Use vectorized log_probability (batch GP calls)
    "discard_burnin": 100,        # Steps to discard as burn-in for corner plot
    "optimize_only": os.environ.get("BAQARO_OPTIMIZE_ONLY", "1") == "1",  # BAQARO_OPTIMIZE_ONLY=0 -> run full MCMC
    # DE restarts; env BAQARO_N_OPT_RESTARTS. Default 4: a single restart is
    # not reliably converged.
    "n_opt_restarts": int(os.environ.get("BAQARO_N_OPT_RESTARTS") or 4),
    "opt_popsize": 50,            # DE population size (multiplied by ndim internally)
    "opt_maxiter": 2000,          # Maximum DE iterations per restart
    # Seed point for optimizer: injected into DE initial population (row 0) and
    # used as starting point for Nelder-Mead + L-BFGS-B local refinement.
    # Set to None to skip seeding and rely on DE alone.
    "opt_x0": (
        [float(s) for s in os.environ["BAQARO_OPT_X0"].split(",")]
        if os.environ.get("BAQARO_OPT_X0", "").strip()
        else [-0.930035, 0.781136, 0.425527, 6.472524, -5.455216, 0.422219]
    ),  # env BAQARO_OPT_X0="e0,evol,std0,logtcoh,logfseed,sseed" seeds DE at a known optimum
}


# ==========================================================================
# Likelihood-definition knob guard
# ==========================================================================
# These env vars all CHANGE THE POSTERIOR but do NOT appear in the chain
# filename — which is `mcmc_{name_file}_{likelihoods}[_{notes}].h5`. So a
# sensitivity run (swap a corr dataset, coarsen the CERDF bins, tighten a
# prior, enable one of the likelihood-definition opt-ins) writes to the SAME
# path as the baseline, so an untagged sensitivity run would be
# indistinguishable from the baseline after the fact.
#
# Deliberately NOT auto-appended to the filename (that would rename every
# non-default chain and orphan the ones already on disk). Instead: if any knob
# is active and the run is untagged, say so loudly.
_LIKELIHOOD_KNOB_PREFIXES = (
    "BAQARO_QLF_",       # PRECISE_FAINT_CUT, PHI_MIN, NONDET_*, MIN_LBOL_*, SG_WINDOW, Z_MATCH_TOL, INCLUDE_Z7, MERGE_TOP_SLIVER
    "BAQARO_CERDF_",     # Z_MIN/Z_MAX/Z_USE, LOGL_BINS, LOG_ETA_BINS, SYS_FRAC/SYS_ABS, BIN_INTEGRATED
    "BAQARO_CORR_",      # Z25 / Z4 / Z6 dataset swaps, GAL_LOGM_MIN
    "BAQARO_PRIOR_",     # prior-box tightening
    "BAQARO_EXCLUDE_CORR_Z",
)
_active_knobs = sorted(
    f"{k}={v}" for k, v in os.environ.items()
    if k.startswith(_LIKELIHOOD_KNOB_PREFIXES) and str(v).strip() != ""
)
if _active_knobs:
    _tag = CONFIG_MCMC["mcmc_filename_notes"]
    print("\n" + "=" * 78)
    print("LIKELIHOOD-DEFINITION KNOBS ACTIVE — this run's posterior is NOT the baseline:")
    for _k in _active_knobs:
        print(f"    {_k}")
    if _tag:
        print(f"\n  Tagged as BAQARO_MCMC_NOTES={_tag} -> writes to its own chain file. Good.")
    else:
        print("\n  ⚠⚠ BAQARO_MCMC_NOTES IS NOT SET.")
        print("  These knobs do NOT appear in the chain filename, so this run will write to")
        print("  the SAME path as the untagged baseline. Set BAQARO_MCMC_NOTES=<tag> to keep")
        print("  the two apart. (The overwrite guard further down will stop you from")
        print("  destroying an existing chain, but an absent baseline would be silently")
        print("  replaced by this differently-defined posterior.)")
    print("=" * 78 + "\n")


# --- Model / emulator identification ---
# These settings uniquely identify which emulator files to load from disk.
# The combination (simulation_name, erdf_model, max_snap, notes_file_emulation)
# maps to a directory tree under the output path.
DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # site from the environment
source_dir = DEFAULT_SOURCE_DIR

from baqaro.utils.sim_config import FIDUCIAL_NOTES_EMULATION
notes_file_emulation = os.environ.get("BAQARO_NOTES_FILE_EMULATION") or FIDUCIAL_NOTES_EMULATION
min_snap = 0
erdf_model = "log_normal_evol_halo_mass"  # 3 ERDF params: log_eta_mean_0, log_eta_mean_evol, std_0
from baqaro.utils.sim_config import boxsize, N_particles_per_side, simulation_name, max_snap, fold_subhalo_mass, merger_delay_mode

# Canonical filename stem used to locate emulator pickle files on disk.

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
# main_emulation). Required so DE/MCMC locate the _g{cap}/_feffcorr emulators
# and write chains under a distinct name.
from baqaro.utils.sim_config import growth_feff_suffix
name_file += growth_feff_suffix()
# --- QLF-specific ---
# Redshifts to exclude from the QLF likelihood.  Matching uses a 0.1 dex
# tolerance because emulator redshifts (from simulation snapshots, e.g. 7.26)
# do not coincide exactly with round observational redshift labels (e.g. 7.0).
mask_redshifts_qlf = [7.0]
# Env override: BAQARO_QLF_INCLUDE_Z7=1 unmasks z=7.0 so the Matsuoka+2023 z~7
# data point (in qlf_obs_data.py) is included in the QLF likelihood.
# Requires the z=7 block in qlf_obs_data.py to be uncommented (it is).
if os.environ.get("BAQARO_QLF_INCLUDE_Z7", "0") == "1":
    mask_redshifts_qlf = [z for z in mask_redshifts_qlf if z != 7.0]

# --- CERDF-specific ---
# Only QSOs with log10(L_bol / [erg/s]) in this range are included.
# The lower bound avoids incompleteness; the upper bound avoids extrapolation
# beyond the emulator's luminosity grid.
cerdf_Lbol_range = (46.0, 48.5)

# --- Correlation-specific ---
mask_redshifts_corr = []


# ==========================================================================
# 1. LOAD EMULATORS
# ==========================================================================

path_out = Path(get_output_path(source=source_dir))

# Map likelihood flags to emulator flags.  The correlation likelihood uses
# the QHMF (quasar halo mass function) emulator, not a dedicated "corr"
# emulator -- so "corr" in LIKELIHOOD_FLAGS maps to "qhmf" here.
# BHMF is never needed for inference (only for plotting); _z0 loader
# uses the "bhmf" key (all BHs, no luminosity cut).
emulator_flags = {
    "qlf": LIKELIHOOD_FLAGS["qlf"],
    "cerdf": LIKELIHOOD_FLAGS["cerdf"],
    "bhmf": False,
    "qhmf": LIKELIHOOD_FLAGS["corr"],
}

emulators = load_emulators(path_out, name_file, emulator_flags)

# Optional Savitzky-Golay denoising of the QLF emulator OUTPUT (mode "1b"
# — tunable per-run, no retraining). Targets the subsample bright-end shot
# noise (see emulation/sg_smoothing.py). Enable with BAQARO_QLF_SG_WINDOW
# (odd int >= 5); default off so legacy behaviour is unchanged.
from baqaro.emulation.sg_smoothing import sg_params_from_env, wrap_emulator_predict
_sg_win, _sg_poly = sg_params_from_env()
if _sg_win > 0 and emulators.get("qlf") is not None:
    wrap_emulator_predict(emulators["qlf"], _sg_win, _sg_poly)
    print(f"QLF emulator output SG-smoothed: window={_sg_win}, poly={_sg_poly}")

# Pre-extract GP alpha vectors for fast prediction (K_star @ alpha instead
# of full gp.predict with Cholesky solve).  No retraining needed.
print("Precomputing GP alpha vectors...")
for emu_name, emu in emulators.items():
    if emu is not None:
        emu.precompute_alpha()
        print(f"  {emu_name}: {len(emu.gps)} components cached")



# All emulators share the same parameter space (same training set), so we
# extract names / ranges from whichever one loaded successfully.
_ref_emulator = next(e for e in emulators.values() if e is not None)
param_names = _ref_emulator.param_names
param_ranges = [list(r) for r in _ref_emulator.param_ranges]  # mutable copy for prior overrides
ndim = len(param_names)

# Inference-time prior-bound overrides via env vars: BAQARO_PRIOR_<NAME>_LO /
# BAQARO_PRIOR_<NAME>_HI (NAME = upper-cased param name, e.g.
# BAQARO_PRIOR_LOGTCOHERENCE_LO=5.0). Must stay WITHIN the emulator's training
# box (the emulator can't extrapolate); a tightened sub-box is fine. Used to
# probe whether a railed optimum has a competing interior minimum.
for _i, _nm in enumerate(param_names):
    _lo = os.environ.get(f"BAQARO_PRIOR_{_nm.upper()}_LO")
    _hi = os.environ.get(f"BAQARO_PRIOR_{_nm.upper()}_HI")
    if _lo is not None:
        param_ranges[_i][0] = max(param_ranges[_i][0], float(_lo))
        print(f"  [prior override] {_nm} LO -> {param_ranges[_i][0]}")
    if _hi is not None:
        param_ranges[_i][1] = min(param_ranges[_i][1], float(_hi))
        print(f"  [prior override] {_nm} HI -> {param_ranges[_i][1]}")
param_ranges = [tuple(r) for r in param_ranges]

print(f"\nParameters ({ndim}D): {param_names}")
print(f"Ranges: {param_ranges}")
print(f"Active likelihoods: {[k for k, v in LIKELIHOOD_FLAGS.items() if v]}")


# ==========================================================================
# 2. LOAD OBSERVATIONAL DATA & PRECOMPUTE
# ==========================================================================

# Registry of active likelihood components.  Each entry stores the
# log_prob function, its precomputed args, the number of data points
# (for diagnostics), and the weighting factor.
components = {}  # name -> {log_prob_func, args, n_data, weight}


# --- QLF ---
if LIKELIHOOD_FLAGS["qlf"]:
    # Two data dictionaries:
    #   data_qlf_global_sys_err_binned -- includes systematic error floor
    #       (used for the Gaussian sigma in log-space).
    #   data_qlf_global_binned -- pure statistical errors only
    #       (used to estimate N_eff and V_eff for the Poisson branch).
    from baqaro.obs_data.qlf_obs_data import data_qlf_global_sys_err_binned, data_qlf_global_binned

    qlf_emulator = emulators["qlf"]
    assert qlf_emulator is not None, "QLF emulator failed to load"
    print(f"\nQLF redshifts: {qlf_emulator.axis_data['redshift']}")

    # --- Match emulator redshifts to observational data redshifts ---
    # Emulator redshifts come from simulation snapshots (e.g. 0.95, 2.01, 4.97,
    # 7.26) and do not exactly match the round values used to label observational
    # data bins (e.g. 1.0, 2.0, 5.0, 7.0).  We find the closest data key for
    # each emulator redshift, accepting the match only if within 0.1 dex.
    target_z_floats = qlf_emulator.axis_data["redshift"]
    available_data_keys = list(data_qlf_global_sys_err_binned.keys())
    available_z_floats = [float(k) for k in available_data_keys]

    matched_keys = []
    target_z_arr = np.asarray(target_z_floats)
    avail_z_arr = np.asarray(available_z_floats)
    for i, target_z in enumerate(target_z_floats):
        closest_idx = int(np.argmin(np.abs(avail_z_arr - target_z)))
        closest_key = available_data_keys[closest_idx]
        closest_z = available_z_floats[closest_idx]
        # 0.4 dex tolerance: reject if no observational data is close enough.
        # Loosened from the original 0.1 (then 0.2) because L2800N10080 snapshot
        # redshifts don't land on integer z. At 0.2 the gap from emulator z=7.31
        # to obs z=7.0 (0.31) was rejected, silently dropping Matsuoka+2023 z=7
        # from the QLF fit even when BAQARO_QLF_INCLUDE_Z7=1 was set. 0.4 binds
        # z=7.31<->z=7.0; spurious low-z double-binding (e.g. emul z=0.0 also
        # winning obs z=0.2) is still prevented by the uniqueness check below
        # (which requires THIS emul slice to be the obs's closest emul too).
        # BAQARO_QLF_Z_MATCH_TOL overrides if needed for tighter sensitivity.
        _z_match_tol = float(os.environ.get("BAQARO_QLF_Z_MATCH_TOL", "0.4"))
        if abs(closest_z - target_z) > _z_match_tol:
            matched_keys.append(None)
            continue
        # Uniqueness: also require THIS emulator z to be the closest one to the
        # matched obs (so the same dataset isn't bound to two slices). E.g.
        # without this, both emulator z=0.26 (gap 0.06) and z=0.00 (gap 0.20)
        # match obs z=0.2 because both pass the 0.2-dex window. With this guard
        # only z=0.26 wins; z=0.00 falls through to None.
        closest_emul_to_obs = int(np.argmin(np.abs(target_z_arr - closest_z)))
        if closest_emul_to_obs != i:
            matched_keys.append(None)
            continue
        matched_keys.append(closest_key)

    def _key_masked(k, mask_list, tol=0.1):
        """Check if the MATCHED OBS KEY ``k`` is a masked redshift.

        Masks by the obs key's redshift (``float(k)``), NOT the emulator slice
        redshift, so it stays correct when the emulator↔obs match tolerance
        (``BAQARO_QLF_Z_MATCH_TOL``, default 0.4) exceeds this 0.1 window. E.g.
        emulator z=7.31 binds obs key '7.0' (gap 0.31); testing the emulator z
        against a 0.1 window would return False and silently defeat
        ``mask_redshifts_qlf=[7.0]`` (and make ``BAQARO_QLF_INCLUDE_Z7`` a no-op).
        Testing the key's 7.0 against [7.0] correctly masks it.
        """
        if k is None:
            return False
        return any(abs(float(k) - mz) <= tol for mz in mask_list)

    # Build per-redshift data lists aligned to the emulator's redshift grid.
    # Entries are None where no data is available or the matched key is masked.
    datas_qlf = [
        data_qlf_global_sys_err_binned[k] if (k is not None and not _key_masked(k, mask_redshifts_qlf)) else None
        for k in matched_keys
    ]
    # Pure statistical errors (no systematic floor) -- needed to estimate
    # N_eff = 1 / (ln10 * sigma_stat)^2 and the effective survey volume V_eff.
    datas_qlf_stat = [
        data_qlf_global_binned[k] if (k is not None and not _key_masked(k, mask_redshifts_qlf)) else None
        for k in matched_keys
    ]
    print("QLF data matching:")
    for z, k in zip(target_z_floats, matched_keys):
        masked = _key_masked(k, mask_redshifts_qlf)
        status = " [MASKED]" if masked else (" [no data]" if k is None else "")
        print(f"  z={z:.2f} -> '{k}'{status}")

    n_qlf = sum(len(d.x) for d in datas_qlf if d is not None)

    # Precompute QLF inputs: heavy work (N_eff estimation, V_eff degradation,
    # non-detection mask construction) is done once here and cached in qlf_cfg.
    if QLF_LIKELIHOOD_MODE == "poisson":
        # Poisson mode needs both stat+sys errors (datas_qlf) and stat-only
        # errors (datas_qlf_stat).  The stat-only errors determine:
        #   N_eff = 1 / (ln10 * sigma_stat)^2  -- effective observed counts
        #   V_eff = N_obs / (Phi * dlogL)       -- effective survey volume
        # V_eff is then degraded: V_eff *= (sigma_stat / sigma_full)^2 to
        # prevent Poisson overconfidence when a systematic floor inflates sigma.
        qlf_cfg = likelihoods_and_priors.precompute_qlf_inputs_poisson(
            datas_qlf, qlf_emulator, datas_qlf_stat=datas_qlf_stat
        )
        log_prob_qlf = likelihoods_and_priors.log_probability_qlf_poisson
        print(f"QLF: {n_qlf} data points (hybrid Poisson/Gaussian likelihood)")
    else:
        qlf_cfg = likelihoods_and_priors.precompute_qlf_inputs(datas_qlf, qlf_emulator)
        log_prob_qlf = likelihoods_and_priors.log_probability_qlf
        print(f"QLF: {n_qlf} data points (Gaussian likelihood)")

    components["qlf"] = {
        "log_prob_func": log_prob_qlf,
        "args": (datas_qlf, qlf_emulator, qlf_cfg),
        "n_data": n_qlf,
        "weight": LIKELIHOOD_WEIGHTS["qlf"],
    }

    # ------------------------------------------------------------------
    # DEBUG: Poisson likelihood diagnostics at parameter-space midpoint.
    # Prints per-redshift bin-by-bin breakdown of Gaussian / Poisson /
    # non-detection contributions to the log-likelihood.  Useful for
    # verifying the precomputed N_eff, V_eff, and degradation values
    # before launching the MCMC.
    # ------------------------------------------------------------------
    if QLF_LIKELIHOOD_MODE == "poisson":
        from scipy.special import gammaln as _gammaln

        print("\n" + "=" * 70)
        print("POISSON LIKELIHOOD DIAGNOSTICS")
        print("=" * 70)

        emul_z = qlf_emulator.axis_data["redshift"]
        for i_z, z_cfg in enumerate(qlf_cfg["per_z"]):
            if z_cfg is None:
                continue
            z_val = emul_z[i_z]
            n_gauss = int(np.sum(~z_cfg["is_poisson"]))
            n_poiss = int(np.sum(z_cfg["is_poisson"]))
            n_nondet = int(np.sum(z_cfg["nondet_mask"]))
            print(f"\n  z={z_val:.1f}: {n_gauss} Gaussian bins, {n_poiss} Poisson bins, "
                  f"{n_nondet} non-detection bins")
            for j in range(len(z_cfg["x_data"])):
                mode = "POISS" if z_cfg["is_poisson"][j] else "GAUSS"
                print(f"    logL={z_cfg['x_data'][j]:.1f}  logPhi={z_cfg['logPhi_data'][j]:.2f}  "
                      f"N_eff={z_cfg['N_eff'][j]:.1f}  N_obs={z_cfg['N_obs'][j]}  "
                      f"V_eff={z_cfg['V_eff'][j]:.2e}  degr={z_cfg['degradation'][j]:.3f}  "
                      f"[{mode}]")
            if n_nondet > 0:
                nondet_bins = qlf_cfg["log_bins"][z_cfg["nondet_mask"]]
                print(f"    Non-det bins: logL = {nondet_bins}")
                print(f"    V_eff_bright = {z_cfg['V_eff_bright']:.2e}")

        # --- Test evaluation at parameter-space midpoint ---
        print("\n" + "-" * 70)
        print("TEST EVALUATION: parameter-space midpoint")
        print("-" * 70)
        mid_params = np.array([(lo + hi) / 2.0 for lo, hi in param_ranges])
        print(f"  params = {dict(zip(param_names, mid_params))}")

        log_qlf_emul_test = qlf_emulator.predict_mean_only(np.atleast_2d(mid_params))[0]
        ll_mid = likelihoods_and_priors._log_likelihood_qlf_poisson_from_prediction(
            log_qlf_emul_test, qlf_cfg
        )
        print(f"  Total ll = {ll_mid:.4f}")

        _log_bins = qlf_cfg["log_bins"]
        for i_z, z_cfg in enumerate(qlf_cfg["per_z"]):
            if z_cfg is None:
                continue
            z_val = emul_z[i_z]
            x_data = z_cfg["x_data"]
            logPhi_data = z_cfg["logPhi_data"]
            sigma = z_cfg["sigma_logPhi"]
            N_obs = z_cfg["N_obs"]
            V_eff = z_cfg["V_eff"]
            is_poisson = z_cfg["is_poisson"]
            dlogL_data = z_cfg["dlogL_data"]

            logPhi_model = np.interp(x_data, _log_bins, log_qlf_emul_test[i_z, :])

            # Gaussian contribution
            ll_gauss = 0.0
            if (~is_poisson).any():
                resid = (logPhi_model[~is_poisson] - logPhi_data[~is_poisson]) / sigma[~is_poisson]
                ll_gauss = -0.5 * np.sum(resid ** 2)

            # Poisson contribution
            ll_poisson = 0.0
            if is_poisson.any():
                Phi_mod = 10.0 ** logPhi_model[is_poisson]
                lam = np.maximum(Phi_mod * V_eff[is_poisson] * dlogL_data, 1e-30)
                N = N_obs[is_poisson]
                ll_poisson = np.sum(N * np.log(lam) - lam - _gammaln(N + 1))

            # Non-detection contribution
            nondet_mask = z_cfg["nondet_mask"]
            ll_nondet = 0.0
            if nondet_mask.any():
                Phi_nondet = 10.0 ** log_qlf_emul_test[i_z, nondet_mask]
                lam_nd = Phi_nondet * z_cfg["V_eff_bright"] * z_cfg["dlogL_emul"]
                ll_nondet = -np.sum(lam_nd)

            print(f"\n  z={z_val:.1f}: ll_gauss={ll_gauss:.2f}, ll_poisson={ll_poisson:.2f}, "
                  f"ll_nondet={ll_nondet:.2f}, total_z={ll_gauss + ll_poisson + ll_nondet:.2f}")

            if nondet_mask.any():
                nd_bins = _log_bins[nondet_mask]
                lam_total = np.sum(10.0 ** log_qlf_emul_test[i_z, nondet_mask]
                                   * z_cfg["V_eff_bright"] * z_cfg["dlogL_emul"])
                print(f"    nondet: {len(nd_bins)} bins, logL=[{nd_bins[0]:.1f},{nd_bins[-1]:.1f}], "
                      f"lambda_total={lam_total:.3f}, penalty={-lam_total:.3f}")

        print("=" * 70 + "\n")


# --- CERDF ---
# The CERDF likelihood evaluates P(log_eta | L_bol, z, theta) for each
# individual QSO, where log_eta is the Eddington ratio.  This per-object
# approach is more constraining than binned statistics for the ERDF shape.
if LIKELIHOOD_FLAGS["cerdf"]:
    # Individual QSO catalog: redshifts, bolometric luminosities, BH masses
    from baqaro.obs_data.qso_obs_data_setup import redshifts_data, logL_Bols_data, logM_BHs_data

    cerdf_emulator = emulators["cerdf"]
    assert cerdf_emulator is not None, "CERDF emulator failed to load"
    print(f"\nCERDF redshifts: {cerdf_emulator.axis_data['redshift']}")
    print(f"CERDF L thresholds: {cerdf_emulator.axis_data['log_L_threshold']}")

    # Compute observed Eddington ratios: log_eta = log(L/L_Edd)
    # to_solar converts from erg/s to solar luminosities; nc.log_csi = log10(L_Edd/M_BH)
    log_etas_data = my_utils.to_solar(logL_Bols_data) - logM_BHs_data - nc.log_csi

    log_L_thresholds = cerdf_emulator.axis_data["log_L_threshold"]
    emul_redshifts = cerdf_emulator.axis_data["redshift"]

    # Filter QSOs to those within the emulator's valid coverage.
    # mask_valid: luminosity must fall between emulator's L threshold grid
    #   endpoints, and redshift must be within 0.5 of emulator's z range.
    # mask_Lbol: additional user-specified luminosity cut to focus inference.
    # --- Bright end: keep the QSOs above the emulator's top L threshold --------
    # The emulator's L-threshold grid stops at 47.5, but `cerdf_Lbol_range`
    # advertises an upper bound of 48.5. Objects above the top luminosity
    # threshold are bracketed into the top cell [47.4, 47.5]
    # (BAQARO_CERDF_KEEP_BRIGHT, default 1): `precompute_cerdf_inputs` clips
    # `lmin_idx` to `len(thresholds)-2`, so an over-range QSO lands in the top
    # bracket automatically; the same clip protects the binned path. A logL=48.0
    # quasar is thus compared against the model's P(log_eta | 47.4 < L < 47.5)
    # rather than its own luminosity slice — the same top-bin lumping the QLF
    # likelihood does via BAQARO_QLF_MERGE_TOP_SLIVER.
    #
    # Set BAQARO_CERDF_KEEP_BRIGHT=0 to drop them instead.
    _keep_bright = os.environ.get("BAQARO_CERDF_KEEP_BRIGHT", "1").strip() != "0"
    _n_bright = int(((logL_Bols_data > log_L_thresholds[-1]) &
                     (logL_Bols_data <= cerdf_Lbol_range[1]) &
                     (redshifts_data >= CERDF_Z_MIN) &
                     (redshifts_data <= CERDF_Z_MAX)).sum())
    mask_valid = (
        (logL_Bols_data >= log_L_thresholds[0]) &
        (redshifts_data >= emul_redshifts.min() - 0.5) &
        (redshifts_data <= emul_redshifts.max() + 0.5)
    )
    if not _keep_bright:
        mask_valid &= (logL_Bols_data <= log_L_thresholds[-1])
        print(f"[cerdf-env] BAQARO_CERDF_KEEP_BRIGHT=0: DROPPING {_n_bright:,} QSOs "
              f"above logL={log_L_thresholds[-1]:.1f} (legacy behaviour)")
    elif _n_bright:
        print(f"[cerdf] keeping {_n_bright:,} QSOs above the emulator's top L "
              f"threshold ({log_L_thresholds[-1]:.1f}); they fall into the top cell "
              f"[{log_L_thresholds[-2]:.1f}, {log_L_thresholds[-1]:.1f}]")
    mask_Lbol = (
        (logL_Bols_data >= cerdf_Lbol_range[0]) &
        (logL_Bols_data <= cerdf_Lbol_range[1])
    )
    # Env-driven z restriction for CERDF sensitivity tests (BAQARO_CERDF_Z_MIN/MAX).
    mask_zcut = (redshifts_data >= CERDF_Z_MIN) & (redshifts_data <= CERDF_Z_MAX)
    mask_cerdf = mask_valid & mask_Lbol & mask_zcut

    redshifts_filtered = redshifts_data[mask_cerdf]
    logL_Bols_filtered = logL_Bols_data[mask_cerdf]
    log_etas_filtered = log_etas_data[mask_cerdf]

    print(f"CERDF: {mask_cerdf.sum()} / {len(mask_cerdf)} QSOs retained")

    if CERDF_LIKELIHOOD_MODE == "binned":
        # Binned mode: histogram comparison, ~100-400 effective data points.
        cerdf_cfg = likelihoods_and_priors.precompute_cerdf_inputs_binned(
            redshifts_filtered, logL_Bols_filtered, log_etas_filtered,
            cerdf_emulator,
            logL_bins=CERDF_BINNED_LOGL_BINS,
            log_eta_bins=CERDF_BINNED_LOG_ETA_BINS,
            min_count=CERDF_BINNED_MIN_COUNT,
        )
        log_prob_cerdf = likelihoods_and_priors.log_probability_cerdf_binned
        n_cerdf = cerdf_cfg['n_data']
        print(f"CERDF: {n_cerdf} histogram bins (binned chi2 likelihood)")
    else:
        # Per-object mode: ~9000 terms (one per QSO).
        cerdf_cfg = likelihoods_and_priors.precompute_cerdf_inputs(
            redshifts_filtered, logL_Bols_filtered, log_etas_filtered,
            cerdf_emulator
        )
        log_prob_cerdf = likelihoods_and_priors.log_probability_cerdf
        n_cerdf = len(cerdf_cfg['log_etas'])
        print(f"CERDF: {n_cerdf} individual QSOs (per-object likelihood)")

    # Resolve BAQARO_CERDF_TEMP=auto now that n_cerdf is known: give the
    # unbinned estimator the same total weight as the binned likelihood it
    # replaces, T = N_QSO / n_cells_binned.  Requires building the binned
    # config too (cheap -- it is just a histogram of the same catalogue).
    if CERDF_TEMP == "auto":
        if CERDF_LIKELIHOOD_MODE == "binned":
            _t_auto = 1.0   # already the reference; nothing to rescale against
        else:
            _binned_ref = likelihoods_and_priors.precompute_cerdf_inputs_binned(
                redshifts_filtered, logL_Bols_filtered, log_etas_filtered,
                cerdf_emulator,
                logL_bins=CERDF_BINNED_LOGL_BINS,
                log_eta_bins=CERDF_BINNED_LOG_ETA_BINS,
                min_count=CERDF_BINNED_MIN_COUNT,
            )
            _t_auto = n_cerdf / float(_binned_ref['n_data'])
        LIKELIHOOD_WEIGHTS["cerdf"] = _t_auto
        print(f"[cerdf-env] BAQARO_CERDF_TEMP=auto -> T = {n_cerdf} QSOs / "
              f"{_binned_ref['n_data'] if CERDF_LIKELIHOOD_MODE != 'binned' else n_cerdf} "
              f"binned cells = {_t_auto:.2f}")

    components["cerdf"] = {
        "log_prob_func": log_prob_cerdf,
        "args": (cerdf_emulator, cerdf_cfg),
        "n_data": n_cerdf,
        "weight": LIKELIHOOD_WEIGHTS["cerdf"],
    }


# --- Correlation ---
# The correlation likelihood compares projected auto-correlation wp(rp) or
# volume-averaged cross-correlation xi against clustering measurements.
# It uses the QHMF emulator to predict the quasar halo mass function, from
# which the correlation function is computed via halo-model calculations.
if LIKELIHOOD_FLAGS["corr"]:
    from baqaro.obs_data import corr_obs_data as _corr_mod
    # Default datasets (production); env overrides allow swapping in alternatives
    # for sensitivity sweeps. The chosen z=2.5 / z=4 / z=6 datasets must expose
    # the same attribute interface (x, data, err, pimax, log_L_threshold, etc.).
    # Production defaults:
    #   z=2.5  →  data_ef_ext_restricted — EF15 Table 3 wp(rp), 10 bins, the
    #                                paper's OWN quoted errors, masked to EF15's
    #                                own fit range 4 < rp < 25 h^-1 Mpc.
    #   z=4    →  data_shen_highz_allfields_err2 — Shen+07 3.5<z<5 hi-z auto wp/rp,
    #                                ALL FIELDS (Fig 7a) with errors x2 (the x2 is a
    #                                systematic allowance on the Shen+07 jackknife,
    #                                ~32 regions for 11 bins). Revert to the
    #                                good-fields x1 measurement with
    #                                BAQARO_CORR_Z4=data_shen_highz.
    #   z=6    →  data_aspire_cov — ASPIRE final-paper 8 bins WITH the full 8x8
    #                                covariance (Huang+26, matrix provided by the
    #                                ASPIRE team; the paper publishes only its
    #                                diagonal). `data_aspire` is the same 8 bins
    #                                with diagonal errors only.
    #                                The matrix is POSITIVE DEFINITE and well-conditioned
    #                                (eigenvalues 0.686-1.706, cond 2.5), with weak
    #                                off-diagonals (max 0.27) — as expected for a
    #                                CROSS-correlation, where each field holds one
    #                                quasar so a galaxy lands in a single separation
    #                                bin and density fluctuations do not propagate
    #                                across bins.
    #                                NB the matrix is MODEL-DEPENDENT by construction
    #                                (built from mocks on a grid in the two minimum
    #                                halo masses); we use the one matrix Huang+26
    #                                themselves tabulate, at their fiducial
    #                                (logM_gal, logM_QSO) = (10.6, 12.2).
    # Fallbacks via env:
    #   BAQARO_CORR_Z25=data_ef                  # 11b EF Table 2 + its published cov
    #   BAQARO_CORR_Z25=data_ef_diag             # same 11 bins, diagonal errors
    #   BAQARO_CORR_Z25=data_ef_psd              # eigenvalue-floored cov (diagnostic)
    #   BAQARO_CORR_Z6=data_aspire               # ASPIRE 8-bin DIAGONAL
    #   BAQARO_CORR_Z6=data_eiger                # EIGER (legacy z=6 default)
    _z25_key = (os.environ.get("BAQARO_CORR_Z25", "data_ef_ext_restricted").strip()
                or "data_ef_ext_restricted")
    # z=4 default = ALL-FIELDS Shen+07 with x2-inflated errors (see
    # data_shen_highz_allfields_err2 in corr_obs_data). Revert to the good-fields
    # x1 measurement with BAQARO_CORR_Z4=data_shen_highz.
    _z4_key  = (os.environ.get("BAQARO_CORR_Z4",  "data_shen_highz_allfields_err2").strip()
                or "data_shen_highz_allfields_err2")
    _z6_key  = os.environ.get("BAQARO_CORR_Z6",  "data_aspire_cov").strip() or "data_aspire_cov"
    data_ef_ext_restricted = getattr(_corr_mod, _z25_key)
    data_shen_highz        = getattr(_corr_mod, _z4_key)
    data_eiger             = getattr(_corr_mod, _z6_key)
    if (_z25_key, _z4_key, _z6_key) != ("data_ef_ext_restricted", "data_shen_highz_allfields_err2", "data_aspire_cov"):
        print(f"[corr-env] dataset swap: z=2.5={_z25_key}  z=4={_z4_key}  z=6={_z6_key}")

    qhmf_emulator = emulators["qhmf"]
    assert qhmf_emulator is not None, "QHMF emulator failed to load"
    print(f"\nCorr redshifts: {qhmf_emulator.axis_data['redshift']}")

    # Monkey-patch observational data objects with metadata needed by the
    # correlation likelihood.  These attributes (.redshift, .corr_type,
    # .logM_min, .logM_max) are not stored in the original data classes
    # but are needed by the halo-model correlation calculator.
    data_ef_ext_restricted.redshift = 2.5
    data_ef_ext_restricted.corr_type = "auto"    # Projected auto-correlation wp/rp
    data_ef_ext_restricted.logM_min = 11.5        # Halo mass range for HOD integral
    data_ef_ext_restricted.logM_max = 14.5

    data_shen_highz.redshift = 4.0
    data_shen_highz.corr_type = "auto"
    data_shen_highz.logM_min = 11.5
    data_shen_highz.logM_max = 14.5

    # OPTIONAL likelihood-side inflation of the z=4 clustering ERRORS (default
    # 1.0 = byte-identical). Scales this dataset's err/errs used by get_chi2 —
    # a what-if on how much the z=4 point constrains the fit. Distinct from the
    # PLOT-only BAQARO_CLUST_Z4_ERR_INFLATE. Prints the real data + before/after
    # so the swap and inflation are verifiable. Tag chains via BAQARO_MCMC_NOTES.
    _z4_inflate = float(os.environ.get("BAQARO_CORR_Z4_ERR_INFLATE", "1.0"))
    print(f"[corr-env] z=4 dataset = '{_z4_key}' (label='{getattr(data_shen_highz,'label','?')}'), "
          f"err_inflate={_z4_inflate}")
    print(f"  z=4 r_p  = {np.round(np.asarray(data_shen_highz.x), 3)}")
    print(f"  z=4 w_p/rp = {np.round(np.asarray(data_shen_highz.data), 4)}")
    print(f"  z=4 err (pre)  = {np.round(np.asarray(data_shen_highz.err), 4)}")
    if _z4_inflate != 1.0:
        for _attr in ("err", "err_down", "err_up"):
            if hasattr(data_shen_highz, _attr) and getattr(data_shen_highz, _attr) is not None:
                setattr(data_shen_highz, _attr,
                        np.asarray(getattr(data_shen_highz, _attr)) * _z4_inflate)
        if getattr(data_shen_highz, "errs", None) is not None:
            data_shen_highz.errs = np.asarray(data_shen_highz.errs) * _z4_inflate
        print(f"  z=4 err (x{_z4_inflate}) = {np.round(np.asarray(data_shen_highz.err), 4)}")

    data_eiger.redshift = 6.1
    data_eiger.corr_type = "cross"               # Volume-averaged cross-correlation
    data_eiger.logM_min = 10.5                    # Lower mass limit for EIGER (deeper survey)
    data_eiger.logM_max = 14.0

    corr_datasets = {
        2.5: data_ef_ext_restricted,
        4.0: data_shen_highz,
        6.1: data_eiger,
    }

    # Drop specific clustering measurements by dataset key (the measurement
    # redshift): BAQARO_EXCLUDE_CORR_Z=4.0 removes the z=4 auto-correlation from
    # the fit entirely. Comma-sep; matched to the nearest key within 0.1.
    for _z in (float(s) for s in os.environ.get("BAQARO_EXCLUDE_CORR_Z", "").split(",") if s.strip()):
        _k = min(corr_datasets, key=lambda z: abs(z - _z)) if corr_datasets else None
        if _k is not None and abs(_k - _z) < 0.1:
            del corr_datasets[_k]
            print(f"  [corr] EXCLUDING z={_k} dataset (BAQARO_EXCLUDE_CORR_Z={_z})")

    # Excluding EVERY corr dataset while the corr likelihood is on is a config
    # error, not a valid run: the corr term would contribute a constant 0 and the
    # chain would be a QLF(+CERDF) fit wearing a `+corr` filename — silently
    # mislabelled forever. (The bare `min({})` crash this used to produce is
    # already guarded above; this catches the real problem.)
    if not corr_datasets:
        raise SystemExit(
            "corr likelihood is ENABLED but BAQARO_EXCLUDE_CORR_Z removed every "
            "dataset — the chain would be named `..._corr...` while containing no "
            "clustering information at all. Either drop `corr` from "
            "BAQARO_LIKELIHOODS, or exclude fewer redshifts."
        )

    # Match emulator redshifts to correlation datasets (0.5 dex tolerance).
    target_z_floats_corr = qhmf_emulator.axis_data["redshift"]
    datas_corr = []
    for target_z in target_z_floats_corr:
        if not corr_datasets:
            datas_corr.append(None)
            continue
        best_key = min(corr_datasets, key=lambda z: abs(z - target_z))
        best_dist = abs(best_key - target_z)

        # Uniqueness guard (mirrors the QLF double-binding guard):
        # require THIS emulator slice to be the closest one to `best_key`, else
        # another slice also binds the same dataset and it gets counted twice.
        closest_emul = min(target_z_floats_corr, key=lambda tz: abs(tz - best_key))
        is_unique = bool(np.isclose(closest_emul, target_z))

        # Mask by the MATCHED OBS KEY, not the emulator redshift.
        # `target_z not in mask_redshifts_corr` was exact float equality against
        # emulator z, so the natural entry [4.0] masked nothing (emulator z=3.94).
        # Matching on the dataset key with a tolerance makes the mask actually fire.
        is_masked = any(abs(best_key - m) <= 0.1 for m in mask_redshifts_corr)

        if best_dist < 0.5 and is_unique and not is_masked:
            datas_corr.append(corr_datasets[best_key])
            print(f"  z={target_z:.2f} -> corr at z={best_key}")
        else:
            datas_corr.append(None)
            _why = ("MASKED" if is_masked else
                    "not closest slice" if (best_dist < 0.5 and not is_unique) else
                    "no match")
            print(f"  z={target_z:.2f} -> no corr data ({_why})")

    # Precompute halo model ingredients (triangle arrays, HMF grids) once.
    print("Precomputing correlation inputs...")
    precomputed_corr = likelihoods_and_priors.precompute_corr_inputs(datas_corr)
    n_corr = sum(len(d.x) for d in datas_corr if d is not None)
    print(f"Corr: {n_corr} data points")

    components["corr"] = {
        "log_prob_func": likelihoods_and_priors.log_probability_corr,
        "args": (datas_corr, qhmf_emulator, precomputed_corr),
        "n_data": n_corr,
        "weight": LIKELIHOOD_WEIGHTS["corr"],
    }


# ==========================================================================
# 3. BUILD COMBINED LOG-PROBABILITY (VECTORIZED)
# ==========================================================================
#
# Instead of evaluating each walker independently (with separate GP calls),
# we batch all walker positions into a single predict_mean_only call per
# emulator.  The cheap post-prediction work (chi2, PDF eval) is then looped
# over walkers.  This eliminates:
#   - multiprocessing.Pool pickle/unpickle overhead
#   - redundant GP kernel computations across walkers
#   - process creation/teardown costs
#
# emcee's vectorize=True passes all walker positions as (n_walkers, ndim)
# and expects back an array of shape (n_walkers,).

n_components = len(components)
assert n_components > 0, "No likelihoods enabled!"

# Convert param_ranges to arrays for vectorized bounds checking
_param_lows = np.array([lo for lo, hi in param_ranges])
_param_highs = np.array([hi for lo, hi in param_ranges])


def _scalar_log_probability(params):
    """Single-walker log_probability (used by optimizer)."""
    for i, param in enumerate(params):
        low, high = param_ranges[i]
        if not (low <= param <= high):   # inclusive at bounds
            return -np.inf, -np.inf

    ll_total = 0.0
    for comp in components.values():
        ll_i = comp["log_prob_func"](params, *comp["args"])[0]
        if not np.isfinite(ll_i):
            return -np.inf, -np.inf
        ll_total += ll_i / comp["weight"]
    return ll_total, 0.0


# Build the vectorized version depending on which likelihoods are active.
# The key optimization: call predict_mean_only ONCE for all walkers per
# emulator, then loop over walkers for the cheap post-prediction part.

# Collect emulator references and their _from_prediction likelihood functions
_vec_components = []

if "qlf" in components:
    _qlf_emul = emulators["qlf"]
    if QLF_LIKELIHOOD_MODE == "poisson":
        _qlf_ll_func = likelihoods_and_priors._log_likelihood_qlf_poisson_from_prediction
        _qlf_ll_args = lambda pred: (pred, qlf_cfg)
    else:
        _qlf_ll_func = likelihoods_and_priors.log_likelihood_qlf_from_prediction
        _qlf_ll_args = lambda pred: (pred, datas_qlf, qlf_cfg)
    _vec_components.append({
        "name": "qlf",
        "emulator": _qlf_emul,
        "ll_func": _qlf_ll_func,
        "ll_args": _qlf_ll_args,
        "weight": components["qlf"]["weight"],
    })

if "cerdf" in components:
    _cerdf_emul = emulators["cerdf"]
    if CERDF_LIKELIHOOD_MODE == "binned":
        _cerdf_ll_func = likelihoods_and_priors.log_likelihood_cerdf_binned_from_prediction
        _cerdf_ll_args = lambda pred: (pred, cerdf_cfg)
    else:
        _cerdf_ll_func = likelihoods_and_priors.log_likelihood_cerdf_from_prediction
        _cerdf_ll_args = lambda pred: (pred, cerdf_cfg)
    _vec_components.append({
        "name": "cerdf",
        "emulator": _cerdf_emul,
        "ll_func": _cerdf_ll_func,
        "ll_args": _cerdf_ll_args,
        "weight": components["cerdf"]["weight"],
    })

if "corr" in components:
    _qhmf_emul = emulators["qhmf"]
    _corr_ll_func = likelihoods_and_priors.log_likelihood_corr_from_prediction
    _corr_ll_args = lambda pred: (pred, datas_corr, _qhmf_emul, precomputed_corr)
    _vec_components.append({
        "name": "corr",
        "emulator": _qhmf_emul,
        "ll_func": _corr_ll_func,
        "ll_args": _corr_ll_args,
        "weight": components["corr"]["weight"],
    })


def log_probability_vectorized(all_params):
    """
    Vectorized log-probability for emcee's vectorize=True mode.

    Parameters
    ----------
    all_params : ndarray, shape (n_walkers, ndim)
        Parameter vectors for all walkers in this half-step.

    Returns
    -------
    log_probs : ndarray, shape (n_walkers,)
        Log-posterior for each walker.
    """
    n_walkers = all_params.shape[0]
    log_probs = np.full(n_walkers, -np.inf)

    # Vectorized bounds check — INCLUSIVE at the bounds, matching
    # likelihoods_and_priors.log_prior_uniform. The
    # prior box IS the emulator's training box, so a point exactly on an edge is
    # inside its support.
    in_bounds = np.all(
        (all_params >= _param_lows[np.newaxis, :]) &
        (all_params <= _param_highs[np.newaxis, :]),
        axis=1,
    )
    valid_mask = in_bounds
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return log_probs

    valid_params = all_params[valid_indices]

    # Accumulate log-likelihood for valid walkers
    ll_accum = np.zeros(len(valid_indices))

    for vc in _vec_components:
        emulator = vc["emulator"]
        ll_func = vc["ll_func"]
        ll_args_builder = vc["ll_args"]
        weight = vc["weight"]

        # BATCH emulator call: predict for ALL valid walkers at once
        # Shape: (n_valid, n_z, n_bins) or (n_valid, n_z, n_L, n_bins)
        batch_predictions = emulator.predict_mean_only(valid_params)

        # Post-prediction: loop over walkers (cheap part)
        for j in range(len(valid_indices)):
            if ll_accum[j] == -np.inf:
                continue
            pred_j = batch_predictions[j]
            # Guard against NaN/Inf from GP extrapolation far from training data
            if not np.all(np.isfinite(pred_j)):
                ll_accum[j] = -np.inf
                continue
            ll_j = ll_func(*ll_args_builder(pred_j))
            if not np.isfinite(ll_j):
                ll_accum[j] = -np.inf
            else:
                ll_accum[j] += ll_j / weight

    log_probs[valid_indices] = ll_accum
    return log_probs


# For the optimizer (single-point evaluation), use the scalar version
def log_probability(params, *args):
    """Scalar wrapper used by optimizer and single-component diagnostics."""
    return _scalar_log_probability(params)

mcmc_args = ()


def diagnose_walker(params):
    """Print per-component likelihood for a single parameter vector."""
    params = np.atleast_2d(params)
    for vc in _vec_components:
        pred = vc["emulator"].predict_mean_only(params)
        pred_0 = pred[0]
        has_nan = not np.all(np.isfinite(pred_0))
        if has_nan:
            print(f"  {vc['name']}: emulator prediction has NaN/Inf")
            continue
        ll = vc["ll_func"](*vc["ll_args"](pred_0))
        print(f"  {vc['name']}: ll = {ll}")


# ==========================================================================
# SANITY CHECK: compare vectorized vs original at opt_x0
# ==========================================================================

_test_x0 = CONFIG_MCMC.get("opt_x0")
if _test_x0 is not None:
    _test_x0 = np.array(_test_x0)
    print("\n" + "=" * 70)
    print("SANITY CHECK: comparing evaluation paths at opt_x0")
    print("=" * 70)

    # 1. Combined scalar path (all active likelihoods)
    ll_scalar = _scalar_log_probability(_test_x0)[0]
    print(f"  Scalar (all components): ll = {ll_scalar:.8f}")

    # 2. Combined vectorized path
    ll_vec = log_probability_vectorized(np.atleast_2d(_test_x0))[0]
    print(f"  Vectorized path:         ll = {ll_vec:.8f}")

    # 3. Alpha vs no-alpha: disable alpha, check combined ll matches
    _saved_alpha = {}
    for emu_name, emu in emulators.items():
        if emu is not None:
            _saved_alpha[emu_name] = emu._alpha_vectors
            emu._alpha_vectors = None

    ll_no_alpha = _scalar_log_probability(_test_x0)[0]
    print(f"  Scalar (no alpha):       ll = {ll_no_alpha:.8f}")

    # 4. Per-emulator prediction diff (alpha vs no-alpha)
    for emu_name, emu in emulators.items():
        if emu is not None:
            pred_no_alpha = emu.predict_mean_only(np.atleast_2d(_test_x0))[0]
            emu._alpha_vectors = _saved_alpha[emu_name]
            pred_alpha = emu.predict_mean_only(np.atleast_2d(_test_x0))[0]
            diff = np.max(np.abs(pred_alpha - pred_no_alpha))
            print(f"  {emu_name} pred diff (alpha vs no-alpha): {diff:.2e}")

    # Summary
    match = np.isclose(ll_scalar, ll_vec) and np.isclose(ll_scalar, ll_no_alpha)
    print(f"  ALL MATCH: {match}")
    print("=" * 70 + "\n")


# ==========================================================================
# 4. OPTIMIZATION
# ==========================================================================

print("\n--- Running Differential Evolution Optimization ---")

# Convert log-probability to chi-squared for minimization:  chi2 = -2 * ll.
# The [0] index extracts the log-likelihood from the (ll, blob) tuple.
chi2_func = lambda x: -2.0 * log_probability(x, *mcmc_args)[0]

n_restarts = CONFIG_MCMC["n_opt_restarts"]
opt_popsize = CONFIG_MCMC["opt_popsize"]
opt_maxiter = CONFIG_MCMC["opt_maxiter"]

best_result = None

# --- Local refinement from seed point (if provided) ---
# When a good initial guess (opt_x0) is available from a prior run, we
# refine it with two complementary local optimizers before running the
# global DE search.  This often finds the local minimum faster than DE alone.
opt_x0 = CONFIG_MCMC.get("opt_x0")
if opt_x0 is not None:
    opt_x0 = np.array(opt_x0, dtype=float)

    # --- Validate the seed point ---
    # A wrong-length BAQARO_OPT_X0 used to fail much later, deep inside
    # `scaler.transform`, with an error that named neither the env var nor the
    # length.
    if opt_x0.shape != (ndim,):
        raise SystemExit(
            f"BAQARO_OPT_X0 has {opt_x0.size} value(s) but the model has {ndim} "
            f"parameters {list(param_names)}. Give exactly {ndim} comma-separated "
            f"floats in that order.")

    # A seed OUTSIDE an env-tightened prior box would be np.clip()ed ONTO the
    # bound by scipy's `init_population_array`, so guard it explicitly.
    _lo = np.asarray([r[0] for r in param_ranges], dtype=float)
    _hi = np.asarray([r[1] for r in param_ranges], dtype=float)
    _outside = (opt_x0 <= _lo) | (opt_x0 >= _hi)
    if _outside.any():
        print("\n  ⚠ BAQARO_OPT_X0 / opt_x0 lies ON or OUTSIDE the prior box — the "
              "strict prior would return -inf and the seed would be WASTED:")
        for _i in np.flatnonzero(_outside):
            print(f"      {param_names[_i]}: x0={opt_x0[_i]:+.6f} not strictly "
                  f"inside ({_lo[_i]:+.6f}, {_hi[_i]:+.6f})")
        # Nudge strictly inside rather than discard the seed: 1e-6 of the box width.
        _eps = 1e-6 * (_hi - _lo)
        opt_x0 = np.clip(opt_x0, _lo + _eps, _hi - _eps)
        print(f"  -> nudged strictly inside the box: {np.round(opt_x0, 6).tolist()}")
        print("     (If this is unintended, widen BAQARO_PRIOR_*_LO/HI or fix opt_x0.)\n")

    chi2_x0 = chi2_func(opt_x0)
    print(f"\n  Seed point chi2 = {chi2_x0:.4f}")
    # Nelder-Mead: gradient-free, robust in non-smooth landscapes (emulator
    # predictions can have small discontinuities near PCA component boundaries).
    print("  Running Nelder-Mead refinement from seed...")
    t0 = time.time()
    result_nm = scipy.optimize.minimize(
        chi2_func, opt_x0, method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6, "adaptive": True},
    )
    dt = time.time() - t0
    print(f"  -> chi2={result_nm.fun:.4f}, success={result_nm.success}, time={dt:.1f}s")
    # L-BFGS-B: gradient-based with box constraints, exploits the smoothness
    # of the GP emulator where Nelder-Mead may be inefficient.
    print("  Running L-BFGS-B refinement from seed...")
    t0 = time.time()
    result_lb = scipy.optimize.minimize(
        chi2_func, opt_x0, method="L-BFGS-B", bounds=param_ranges,
    )
    dt = time.time() - t0
    print(f"  -> chi2={result_lb.fun:.4f}, success={result_lb.success}, time={dt:.1f}s")
    # Keep the best of the two local refinements
    for r in [result_nm, result_lb]:
        if best_result is None or r.fun < best_result.fun:
            best_result = r
            print(f"  ** New best from local refinement: chi2={r.fun:.4f}")

# --- DE restarts (seeded with x0 in initial population) ---
# Multiple restarts with different random seeds help escape local minima
# in the 6D parameter space.  When opt_x0 is available, it is injected
# as the first member of the DE population to bias the search toward
# the known good region while still exploring globally.
for i_restart in range(n_restarts):
    seed_i = CONFIG_MCMC["rng_seed"] + i_restart
    print(f"\n  DE Restart {i_restart+1}/{n_restarts} (seed={seed_i})")

    # Build custom initial population with the seed point injected at row 0.
    # Without opt_x0, fall back to scipy's default Latin hypercube initialization.
    init_pop = None
    if opt_x0 is not None:
        rng_pop = np.random.default_rng(seed_i)
        pop_size_total = opt_popsize * ndim
        init_pop = rng_pop.uniform(size=(pop_size_total, ndim))
        for j in range(ndim):
            lo, hi = param_ranges[j]
            init_pop[:, j] = lo + init_pop[:, j] * (hi - lo)
        init_pop[0] = opt_x0  # Inject seed point

    t0 = time.time()
    result_i = scipy.optimize.differential_evolution(
        chi2_func,
        bounds=param_ranges,
        popsize=opt_popsize,
        maxiter=opt_maxiter,
        recombination=0.7,
        disp=True,
        polish=True,       # L-BFGS-B polish after DE converges
        seed=seed_i,
        init=init_pop if init_pop is not None else "latinhypercube",
    )
    dt = time.time() - t0
    print(f"  -> chi2={result_i.fun:.4f}, success={result_i.success}, time={dt:.1f}s")
    if best_result is None or result_i.fun < best_result.fun:
        best_result = result_i
        print(f"  ** New best!")

result_opt = best_result

print("-" * 50)
print(f"Optimization Success: {result_opt.success}")
print(f"Best -2*logL (combined): {result_opt.fun:.4f}")
print(f"Best Params:")
for name, val in zip(param_names, result_opt.x):
    print(f"  {name}: {val:.6f}")

# Per-component chi2 breakdown: diagnose which component dominates the fit.
# Only printed for multi-component runs where relative contributions matter.
if n_components > 1:
    for cname, comp in components.items():
        ll_i = comp["log_prob_func"](result_opt.x, *comp["args"])[0]
        print(f"  chi2_{cname} = {-2*ll_i:.2f}  (n_data={comp['n_data']}, weight={comp['weight']:.1f})")
print("-" * 50)

# ------------------------------------------------------------------
# DEBUG: Per-redshift, per-bin breakdown at the best-fit point.
# Decomposes the log-likelihood into Gaussian / Poisson / non-detection
# contributions at each redshift.  Essential for diagnosing whether the
# fit is driven by a handful of Poisson bins or dominated by the
# well-sampled Gaussian regime.
# ------------------------------------------------------------------
if LIKELIHOOD_FLAGS["qlf"] and QLF_LIKELIHOOD_MODE == "poisson":
    from scipy.special import gammaln as _gammaln

    print("\n" + "=" * 70)
    print("BEST-FIT POISSON BREAKDOWN")
    print("=" * 70)

    _qlf_emul = emulators["qlf"]
    _log_qlf_best = _qlf_emul.predict_mean_only(np.atleast_2d(result_opt.x))[0]
    _log_bins = qlf_cfg["log_bins"]
    _emul_z = _qlf_emul.axis_data["redshift"]

    ll_total_check = 0.0
    for i_z, z_cfg in enumerate(qlf_cfg["per_z"]):
        if z_cfg is None:
            continue
        z_val = _emul_z[i_z]
        x_data = z_cfg["x_data"]
        logPhi_data = z_cfg["logPhi_data"]
        sigma = z_cfg["sigma_logPhi"]
        N_obs = z_cfg["N_obs"]
        V_eff = z_cfg["V_eff"]
        is_poisson = z_cfg["is_poisson"]
        dlogL_data = z_cfg["dlogL_data"]

        # Interpolate emulator prediction onto data luminosity bins
        logPhi_model = np.interp(x_data, _log_bins, _log_qlf_best[i_z, :])

        # Gaussian contribution: chi2 for well-sampled bins
        ll_gauss = 0.0
        if (~is_poisson).any():
            resid = (logPhi_model[~is_poisson] - logPhi_data[~is_poisson]) / sigma[~is_poisson]
            ll_gauss = -0.5 * np.sum(resid ** 2)

        # Poisson (Cash) contribution: N*ln(lam) - lam - ln(N!)
        ll_poisson = 0.0
        if is_poisson.any():
            Phi_mod = 10.0 ** logPhi_model[is_poisson]
            lam = np.maximum(Phi_mod * V_eff[is_poisson] * dlogL_data, 1e-30)
            N = N_obs[is_poisson]
            ll_poisson = np.sum(N * np.log(lam) - lam - _gammaln(N + 1))

        # Non-detection: penalty for model overprediction (ll = -lambda for N=0)
        nondet_mask = z_cfg["nondet_mask"]
        ll_nondet = 0.0
        if nondet_mask.any():
            Phi_nondet = 10.0 ** _log_qlf_best[i_z, nondet_mask]
            lam_nd = Phi_nondet * z_cfg["V_eff_bright"] * z_cfg["dlogL_emul"]
            ll_nondet = -np.sum(lam_nd)

        ll_z = ll_gauss + ll_poisson + ll_nondet
        ll_total_check += ll_z

        print(f"\n  z={z_val:.1f}: ll_gauss={ll_gauss:.2f}, ll_poisson={ll_poisson:.2f}, "
              f"ll_nondet={ll_nondet:.2f}, total_z={ll_z:.2f}")

        # Per-bin detail
        for j in range(len(x_data)):
            mode = "POISS" if is_poisson[j] else "GAUSS"
            resid_j = (logPhi_model[j] - logPhi_data[j]) / sigma[j]
            if is_poisson[j]:
                Phi_m = 10.0 ** logPhi_model[j]
                lam_j = max(Phi_m * V_eff[j] * dlogL_data, 1e-30)
                print(f"    logL={x_data[j]:.1f}  model={logPhi_model[j]:.2f}  data={logPhi_data[j]:.2f}  "
                      f"N_obs={N_obs[j]}  lam={lam_j:.2f}  [{mode}]")
            else:
                print(f"    logL={x_data[j]:.1f}  model={logPhi_model[j]:.2f}  data={logPhi_data[j]:.2f}  "
                      f"resid={resid_j:.2f}  [{mode}]")

        if nondet_mask.any():
            nd_bins = _log_bins[nondet_mask]
            lam_total = np.sum(10.0 ** _log_qlf_best[i_z, nondet_mask]
                               * z_cfg["V_eff_bright"] * z_cfg["dlogL_emul"])
            print(f"    nondet: {len(nd_bins)} bins, logL=[{nd_bins[0]:.1f},{nd_bins[-1]:.1f}], "
                  f"lambda_total={lam_total:.3f}, penalty={-lam_total:.3f}")

    print(f"\n  SUM ll = {ll_total_check:.4f}  (chi2 = {-2*ll_total_check:.4f})")
    print("=" * 70 + "\n")


# ------------------------------------------------------------------
# DEBUG: Per-cell breakdown for binned CERDF at the best-fit point.
# ------------------------------------------------------------------
if LIKELIHOOD_FLAGS["cerdf"] and CERDF_LIKELIHOOD_MODE == "binned":
    print("\n" + "=" * 70)
    print("BEST-FIT BINNED CERDF BREAKDOWN")
    print("=" * 70)
    likelihoods_and_priors.log_likelihood_cerdf_binned(
        result_opt.x, emulators["cerdf"], cerdf_cfg, verbose=True)
    print("=" * 70 + "\n")

if LIKELIHOOD_FLAGS["corr"]:
    print("\n" + "=" * 70)
    print("BEST-FIT CORRELATION BREAKDOWN")
    print("=" * 70)
    likelihoods_and_priors.log_likelihood_corr(
        result_opt.x, datas_corr, emulators["qhmf"], precomputed_corr,
        verbose=True)
    print("=" * 70 + "\n")

if CONFIG_MCMC["optimize_only"]:
    print("\n--- optimize_only=True: skipping MCMC ---")
    plt.show()
    raise SystemExit(0)


# ==========================================================================
# 5. MCMC
# ==========================================================================

print("\n--- Setting up MCMC ---")

# --- Output file naming ---
# Encode the active likelihood combination in the filename so that runs
# with different component sets produce distinct files (e.g. "qlf+cerdf").
mcmc_dir = path_out / "mcmc"
mcmc_dir.mkdir(parents=True, exist_ok=True)
active_names = "+".join(k for k, v in LIKELIHOOD_FLAGS.items() if v)
mcmc_filename = f"mcmc_{name_file}_{active_names}"
if CONFIG_MCMC["mcmc_filename_notes"] is not None:
    mcmc_filename += f"_{CONFIG_MCMC['mcmc_filename_notes']}"
backend_path = mcmc_dir / f"{mcmc_filename}.h5"

nwalkers = CONFIG_MCMC["n_walkers"]
nsteps = CONFIG_MCMC["n_steps"]

# Initialize walkers in a tight Gaussian ball around the DE optimum.
# This is the standard emcee recipe to avoid starting walkers in low-
# probability regions, which would cause long burn-in or stuck walkers.
rng = np.random.default_rng(CONFIG_MCMC["rng_seed"])
initial_pos = likelihoods_and_priors.setup_initial_positions(
    result_opt.x, param_ranges, nwalkers, rng, jitter=0.05
)

# --- Guard: never destroy an existing chain without being asked ---
# `backend.reset()` below WIPES the file, so destroying a chain requires an
# explicit act.
if backend_path.exists():
    try:
        _existing = emcee.backends.HDFBackend(str(backend_path), read_only=True)
        _n_done = int(_existing.iteration)
    except Exception:
        _n_done = -1
    if os.environ.get("BAQARO_MCMC_OVERWRITE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        raise SystemExit(
            "\n" + "=" * 78 + "\n"
            f"REFUSING to overwrite an existing chain:\n"
            f"  {backend_path}\n"
            f"  it already holds {_n_done} steps x {nwalkers} walkers.\n\n"
            "`backend.reset()` would DESTROY it. Choose one:\n"
            "  * tag this run differently:  BAQARO_MCMC_NOTES=<tag>\n"
            "  * deliberately overwrite:    BAQARO_MCMC_OVERWRITE=1\n"
            "  * analyse the existing one:  python -m ...inference.main_results_analysis\n"
            + "=" * 78)
    print(f"⚠ BAQARO_MCMC_OVERWRITE=1: DESTROYING the existing {_n_done}-step chain at "
          f"{backend_path.name}")

backend = emcee.backends.HDFBackend(str(backend_path))
backend.reset(nwalkers, ndim)

print(f"Running MCMC: {nwalkers} walkers, {nsteps} steps (vectorized)")
print(f"Output: {backend_path}")

# Vectorized mode: emcee passes all walker positions as (n_walkers, ndim)
# to log_probability_vectorized in a single call.  No multiprocessing needed.
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability_vectorized,
    backend=backend, vectorize=True,
)
# Verify all walkers start with finite log-probability
init_lp = log_probability_vectorized(initial_pos)
n_bad = np.sum(~np.isfinite(init_lp))
if n_bad > 0:
    print(f"WARNING: {n_bad}/{nwalkers} walkers start with -inf log-prob!")
    for k in range(nwalkers):
        if not np.isfinite(init_lp[k]):
            print(f"  walker {k}: params={initial_pos[k]}, lp={init_lp[k]}")
            diagnose_walker(initial_pos[k])
else:
    print(f"All {nwalkers} walkers start with finite log-prob (range: {init_lp.min():.2f} to {init_lp.max():.2f})")

sampler.run_mcmc(initial_pos, nsteps, progress=True)


# ==========================================================================
# 6. DIAGNOSTICS
# ==========================================================================

print("\n--- MCMC Finished ---")

# --- Quick convergence diagnostics ---
# Acceptance fraction: emcee recommends 0.2--0.5 for the stretch-move sampler.
# Values below 0.2 suggest the proposal scale is too large or the posterior
# has problematic geometry (strong degeneracies, multimodality).
acc_frac = np.mean(sampler.acceptance_fraction)
print(f"Mean Acceptance Fraction: {acc_frac:.3f}")
if acc_frac < 0.2:
    print("WARNING: Acceptance fraction is low (< 0.2).")

# Autocorrelation time: the chain should be >> tau steps long for reliable
# posterior estimates.  Short chains raise AutocorrError.
try:
    tau = sampler.get_autocorr_time()
    print(f"Mean Autocorrelation Time: {np.mean(tau):.3f} steps")
except emcee.autocorr.AutocorrError:
    print("Warning: Chain too short to estimate autocorrelation time reliably.")

# Flatten the chain after discarding burn-in for the corner plot.
# No thinning applied here; the full analysis script (main_results_analysis.py)
# uses autocorrelation-based thinning for publication-quality results.
flat_samples = sampler.get_chain(discard=CONFIG_MCMC["discard_burnin"], flat=True, thin=1)
print(f"Corner plot: {len(flat_samples)} samples (burn-in {CONFIG_MCMC['discard_burnin']} discarded)")

fig = corner.corner(
    flat_samples,
    labels=param_names,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    smooth=1.0,
    smooth1d=1.0,
    color="#0072C1",
)

plt.show()
