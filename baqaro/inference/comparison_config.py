"""Shared configuration for the multi-run MCMC comparison scripts.

Single source of truth for:
  * the ``RUNS`` list (which likelihood combinations to compare, their display
    labels and colours) — used by BOTH ``main_results_corner_comparison.py``
    (corner overlays) and ``main_results_comparison.py`` (posterior-predictive
    overlays), so the colours match across every comparison figure;
  * the emulator / model identifier env block (sim, max_snap, foldmass,
    merger mode, notes, subset tag) → ``name_file``;
  * helpers to resolve each run's MCMC chain path and load its flat samples.

All emulator-identification env vars mirror ``main_mcmc.py`` so a comparison
reads exactly the chains a given run wrote. ``BAQARO_MCMC_NOTES`` (e.g. ``minL05``)
is applied uniformly to every run, so a re-fit set is compared instead of the
baseline, and the output figures are tagged with it.
"""
import os
from pathlib import Path

import numpy as np
import emcee

from baqaro.utils.my_dir import get_output_path, get_plots_path
from baqaro.utils.sim_config import simulation_name, max_snap, fold_subhalo_mass, merger_delay_mode, growth_feff_suffix, FIDUCIAL_NOTES_EMULATION


# ==========================================
# RUNS — likelihood combinations to compare
# ==========================================
# Colours are colour-blind-safe (Wong palette) and shared across the corner and
# predictive comparison figures.
RUNS = [
    # Colours match plotting_paper/plotting_mcmc_corner.py's RUNS (Wong CB-safe
    # palette) so the corner and the comparison figures share one colour key.
    {"likelihood_flags": {"qlf": True,  "cerdf": False, "corr": False}, "label": "QLF only",           "color": "#0072B2", "notes": None},
    {"likelihood_flags": {"qlf": True,  "cerdf": True,  "corr": False}, "label": "QLF + cERDF",        "color": "#D55E00", "notes": None},
    {"likelihood_flags": {"qlf": True,  "cerdf": False, "corr": True},  "label": "QLF + Corr",         "color": "#CC79A7", "notes": None},
    {"likelihood_flags": {"qlf": True,  "cerdf": True,  "corr": True},  "label": "QLF + Corr + cERDF", "color": "#009E73", "notes": None},
]

# Uniform notes suffix via env (e.g. BAQARO_MCMC_NOTES=minL05 to compare the
# +0.5-dex-QLF-cut chains). Falls back to each run's own "notes" when unset.
MCMC_NOTES_ENV = os.environ.get("BAQARO_MCMC_NOTES") or None
if MCMC_NOTES_ENV:
    for _r in RUNS:
        _r["notes"] = MCMC_NOTES_ENV


# ==========================================
# Model / emulator identifier (mirrors main_mcmc.py)
# ==========================================
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
notes_file_emulation = os.environ.get("BAQARO_NOTES_FILE_EMULATION") or FIDUCIAL_NOTES_EMULATION
erdf_model = "log_normal_evol_halo_mass"
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
# main_emulation) so chain + emulator lookup resolves the capped/_feffcorr run.
name_file += growth_feff_suffix()

# Short, human-readable tag for FIGURE filenames only. Drops the verbose
# `_erdf_<model>` and `_sub_<subset_tag>` tokens that make `name_file`
# unreadable (~150 chars). `name_file` stays full for chain/emulator lookup;
# `short_name` is used purely for the comparison figure names.
short_name = "{}_snap{}".format(simulation_name, max_snap)
if notes_file_emulation is not None:
    short_name += "_{}".format(notes_file_emulation)

path_out = Path(get_output_path(source=source_dir))


def active_likelihoods(run):
    """``qlf+cerdf+corr``-style active-likelihood string for a run."""
    f = run["likelihood_flags"]
    return "+".join(k for k in ("qlf", "cerdf", "corr") if f.get(k, False))


def mcmc_path(run):
    """Resolve a run's emcee HDF5 chain path (mirrors main_mcmc.py naming)."""
    fn = f"mcmc_{name_file}_{active_likelihoods(run)}"
    if run.get("notes"):
        fn += f"_{run['notes']}"
    return path_out / "mcmc" / f"{fn}.h5"


def notes_suffix():
    """Filename suffix for comparison outputs (so re-fit sets don't clobber)."""
    return f"_{MCMC_NOTES_ENV}" if MCMC_NOTES_ENV else ""


def load_flat_samples(run, manual_burnin=0, rng=None, n_draw=None):
    """Load a run's flat posterior samples (auto burn-in/thinning).

    Returns ``None`` if the chain file does not exist (so callers can skip a
    run whose chain is still running). If ``n_draw`` is given, returns a random
    subset of that many rows (for posterior-predictive draws).
    """
    p = mcmc_path(run)
    if not p.exists():
        return None
    reader = emcee.backends.HDFBackend(str(p), read_only=True)
    n_steps_chain = int(reader.iteration)
    try:
        tau = reader.get_autocorr_time(quiet=True)
        if not np.all(np.isfinite(tau)):
            raise ValueError(f"non-finite tau: {tau}")
        burnin = int(2 * np.nanmax(tau))
        thin = max(1, int(0.5 * np.nanmin(tau)))
    except Exception as e:
        # Loud, not silent: this run's chain is likely unconverged and is being
        # overlaid alongside converged ones.
        print(f"  ⚠ {p.name}: autocorr unusable ({e}); manual burn_in="
              f"{manual_burnin}, thin=1 — chain likely UNCONVERGED.")
        burnin, thin = manual_burnin, 1

    # Clamp: burnin >= chain length yields an EMPTY array that crashes the
    # downstream plotters with an opaque error.
    if burnin >= n_steps_chain:
        _clamped = max(0, n_steps_chain // 2)
        print(f"  ⚠ {p.name}: burn-in {burnin} >= chain length {n_steps_chain}; "
              f"clamping to {_clamped} (chain too short — provisional).")
        burnin = _clamped

    flat = reader.get_chain(discard=burnin, flat=True, thin=thin)
    if flat.size == 0:
        print(f"  ⚠ {p.name}: no samples after burn-in={burnin}, thin={thin} on "
              f"{n_steps_chain} steps — skipping this run.")
        return None
    if n_draw is not None and len(flat) > 0:
        rng = rng or np.random.default_rng()
        idx = rng.integers(len(flat), size=n_draw)
        return flat[idx]
    return flat


def comparison_plots_dir(subdir="mcmc_comparison"):
    """Output dir for comparison figures.

    Standard (matches ``plot_config.save_fig``): the repo-local ``plots/`` dir
    (``plot_config.LOCAL_PLOTS_DIR``), so bare runs drop figures locally
    (gitignored). Set ``BAQARO_PLOTS_ON_DATA3=1`` to write to the
    production plots tree (``get_plots_path``) instead.
    """
    from baqaro.plotting_common.plot_config import resolve_plots_dir
    return resolve_plots_dir(subdir=subdir, source=source_dir)
