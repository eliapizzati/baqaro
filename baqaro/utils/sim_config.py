"""
Centralized simulation configuration.

All entry points should import sim parameters from this module rather
than redefining them locally. To switch the pipeline between simulations,
set ``BAQARO_SIM`` (default ``L2800N10080``) — every consumer picks up the
new sim through the module-level aliases (``boxsize``,
``N_particles_per_side``, ``simulation_name``, ``max_snap``,
``fold_subhalo_mass``, ``merger_delay_mode``).

Per-run overrides
-----------------
``max_snap`` (``BAQARO_MAX_SNAP``), ``fold_subhalo_mass``
(``BAQARO_FOLD_SUBHALO_MASS``) and ``merger_delay_mode``
(``BAQARO_MERGER_DELAY_MODE``) are exposed as env-overridable module-level
defaults. Individual scripts may still shadow them locally for testing
(e.g. ``max_snap = 14``) without affecting other consumers.
"""

import os
from dataclasses import dataclass


_ENV_TRUE = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "", "false", "no", "off")


def env_bool(name, default=False):
    """Tolerantly parse a ``BAQARO_*`` boolean environment variable.

    Truthy: ``1`` / ``true`` / ``yes`` / ``on``; falsy: ``0`` / ``""`` /
    ``false`` / ``no`` / ``off`` (case-insensitive, whitespace-stripped).
    Unset or unrecognised → ``default``.

    Single canonical home for model/data-affecting toggle parsing, so
    ``BAQARO_MERGER_ON=true`` can never silently read as False the way a strict
    ``== "1"`` comparison would. Cosmetic
    toggles (headless / save-figs / plot flags) deliberately keep their local
    ``== "1"`` checks — the tolerant parser is reserved for toggles that change
    the physics or the on-disk data.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    v = val.strip().lower()
    if v in _ENV_TRUE:
        return True
    if v in _ENV_FALSE:
        return False
    return default


@dataclass(frozen=True)
class SimSpec:
    """Specification of a SWIFT/HBT simulation."""

    # Comoving box side length in **Mpc**, h-FREE. FLAMINGO names its boxes in
    # comoving Mpc (L2800 = 2.8 cGpc), and every consumer uses ``boxsize**3``
    # directly as a Mpc^3 volume for the h-free Mpc^-3 mass functions.
    # Do NOT "fix" this by dividing by h.
    boxsize: int
    N_particles_per_side: int   # DM particles per side
    max_snap: int               # default production max snapshot index

    # Mass-bin geometry for the stratified subsample, in log10(M_halo / Msun).
    # Resolution-dependent, so it lives per-simulation rather than as a global.
    #
    #   log_M_lo : should sit just below the lowest populated bin at this
    #              simulation's resolution (nbound_threshold = 40 particles).
    #              Alive-at-z_root halos BELOW it are dropped from the subset
    #              entirely — closure only recovers them if they later merge
    #              into a heavier descendant — so a floor set too high
    #              undercounts the faint end.
    #   log_M_hi : the mirror image. Alive halos AT OR ABOVE it fall outside
    #              the digitize range, are never selected as roots, and being
    #              alive cannot enter via closure either, so they are silently
    #              DROPPED. At 15.5 this bites only below z ~ 0.16, where the
    #              2800 Mpc/h boxes first produce haloes above 10^15.5 (14 of
    #              them by z=0, the largest 10^15.63).
    #   n_bins   : chosen to give ~0.25 dex bins across [log_M_lo, log_M_hi].
    #
    # Consumers derive the on-disk subset tag from this geometry (see
    # ``tree_subsample.resolve_subset_tag``), so changing any of the three
    # repoints every bare run at a different subset. Override per run with
    # BAQARO_SUBSAMPLE_LOG_M_LO / _HI / _NBINS.
    subsample_log_M_lo: float = 11.0
    subsample_log_M_hi: float = 15.5
    subsample_n_bins: int = 18

    # Time-window averaging fraction for accretion-rate computation
    # (see calculate_dynamical_time_windows). Larger value ⇒ wider
    # lookback when Δt[i] < tdyn_fraction × t_dyn — useful for sims
    # whose Δt/t_dyn is small enough that single-snap rates are noisy.
    # 0.20 is the legacy L2800N5040 default; 0.25 is a slight bump for
    # L2800N10080 whose Δt/t_dyn cadence is finer in some regimes.
    tdyn_fraction_default: float = 0.2

    # Histogram bin ranges for the training pipeline's saved functions.
    # These MUST be sim-aware because L2800N10080's 0.9 dex lower halo
    # resolution floor lets it populate halo-mass bins down to log M ~ 10.5
    # that L2800N5040 cannot resolve. Default values below are the
    # L2800N5040 production ranges; override in the L2800N10080 SimSpec.
    log_M_halo_qhmf_lo: float = 11.5   # QHMF lowest log10(M_halo / Msun) edge
    log_M_halo_qhmf_hi: float = 14.5   # QHMF highest log10(M_halo / Msun) edge

    # ---------------------------------------------------------------- z grids
    # TWO grids, deliberately separate. Sharing one constant is a trap: the
    # multinode forward run and the training pipeline would read the same
    # tuple, so widening the grid for a forward run would silently re-index
    # every training HDF5 and desync the emulators built on them.
    #
    # Both MUST be sim-specific because the snap → z mapping differs between
    # boxes (snap 28 = z=2.5 on L2800N5040 but z=9.8 on L2800N10080, which has
    # 145 snaps vs 79). Each consumer filters to ``[s for s in <grid> if s <=
    # max_snap]``, so short-range runs get a truncated but z-correct subset.

    # (1) TRAINING grid — emulation/main_training.py. The emulators' z axis.
    # The L2800N5040 default samples z ≈ 7.3, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0,
    # 1.5, 1.0, 0.75, 0.5, 0.25, 0.0 — keep its content stable for regression
    # compatibility with existing training HDF5s. The L2800N10080 list mirrors
    # that z grid as closely as the available snapshot schedule allows.
    # ⚠ Changing this invalidates existing training data / emulators.
    training_snapshots_default: tuple = (6, 8, 10, 14, 18, 28, 38, 48, 58, 63, 68, 73, 78)

    # (2) MULTINODE EVOLUTION grid — core_functions/main_evolution_chunked.py.
    # The z grid of the full-box forward run's summary stats, and the default
    # set of per-halo catalogue anchors. Free to be much denser than the
    # training grid: stats are ~KB, and only the CATALOGUE dump is expensive
    # (~n_union × 4 B × 2 ≈ 18 GB per anchor at the full box), which
    # BAQARO_CATALOG_ANCHOR_SNAPS can restrict to a subset independently.
    # ``None`` ⇒ fall back to the training grid.
    multinode_evolution_snapshots_default: tuple | None = None

    # NOTE: plotting scripts are redshift-driven (declare their own
    # REDSHIFT_TARGETS list and resolve via argmin against the loaded sim's
    # redshifts array). No sim-specific snapshot-to-plot table needed.

    @property
    def name(self) -> str:
        return "L{:04d}N{:04d}".format(self.boxsize, self.N_particles_per_side)


# Registry of known simulations.
#
# ``max_snap`` is the *production default* for each sim (highest snapshot
# index currently usable). Scripts may override locally for testing.
#
# Subsample geometry: ``log_M_lo`` is tuned to the lowest populated bin
# at the sim's resolution floor (nbound_threshold=40 particles ≡ ~0.9 dex
# lower at L2800N10080 because each particle is 8× lighter). Bumping
# log_M_lo down captures the faint-end halo population that would
# otherwise miss the subsample (an alive-at-z_root halo below log_M_lo
# is dropped even by closure). See ``core_functions/tree_subsample.py``.
SIMS = {
    "L2800N5040": SimSpec(
        boxsize=2800,
        N_particles_per_side=5040,
        max_snap=78,
        subsample_log_M_lo=11.0,
        subsample_log_M_hi=15.5,
        subsample_n_bins=18,
    ),
    "L2800N10080": SimSpec(
        boxsize=2800,
        N_particles_per_side=10080,
        max_snap=144,  # z=0 production target (snap 144). Set BAQARO_MAX_SNAP
                       # lower for short-range (higher-z) runs.
        # 8× more particles → 8× lighter m_p → halo floor drops ~0.9 dex.
        # ⚠ The geometry below ("K22": log_M_hi=15.5 over 22 bins) is what the
        # adopted fiducial was built on — its training set, emulators, chains
        # and forward run all carry
        # `subset/tag = root144_flatN500000_K22_logM10.0to15.5_seed42_v3`.
        # A bare run has to reproduce the fiducial, so the default follows it.
        # A "K24" variant (log_M_hi=16.0, n_bins=24) also exists on disk and
        # loads with explicit
        # BAQARO_SUBSAMPLE_LOG_M_HI=16.0 BAQARO_SUBSAMPLE_NBINS=24.
        subsample_log_M_lo=10.0,
        subsample_log_M_hi=15.5,
        subsample_n_bins=22,  # ~0.25 dex bin width over 10.0-15.5
        tdyn_fraction_default=0.25,
        # The L2800N5040 z ladder remapped onto the 145-snap L2800N10080
        # schedule (nearest snapshot to each target z), plus half-integer rungs
        # at z = 6.5, 5.5, 4.5, 3.5 that densify the range where the QLF and
        # BHMF data live. Short-range runs are auto-filtered by ``max_snap``,
        # so the post-z=1.5 entries are simply ignored until reached.
        # target z → snap (achieved z) — verified against output_list.txt:
        # 7.26→35 (7.315), 6.50→37 (6.708), 6.04→39 (6.145), 5.50→42 (5.377),
        # 5.00→44 (5.024), 4.50→47 (4.532), 4.00→51 (3.937), 3.50→54 (3.534),
        # 3.00→59, 2.50→65, 2.00→71, 1.50→78, 1.00→90, 0.75→98 (0.741),
        # 0.50→107, 0.25→118 (0.261), 0.00→144
        # (z=6.5 takes snap 37 rather than the nearest snap 38 (z=6.421), which
        # sits only 0.28 in z from the z=6.04 rung.)
        training_snapshots_default=(35, 37, 39, 42, 44, 47, 51, 54, 59, 65, 71,
                                    78, 90, 98, 107, 118, 144),
        # Multinode forward-run grid: a strict SUPERSET of the 17 training
        # rungs above, so it reproduces the training z ladder exactly and
        # refines between its rungs. 41 snapshots, z = 8.68 → 0, median Δz 0.21
        # against 0.50 for training. All 41 carry real HBT data — the
        # catalogue is complete over 0..144, with no nearest-snap fallbacks.
        multinode_evolution_snapshots_default=(
            31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49,
            51, 52, 54, 56, 59, 62, 65, 68, 71, 74, 75, 77, 78, 80, 83, 85,
            90, 96, 98, 103, 107, 111, 118, 122, 144),
        # 1 dex lower halo-mass floor than L2800N5040 (matches the
        # resolution gain from 8× more particles).
        log_M_halo_qhmf_lo=10.5,
        log_M_halo_qhmf_hi=14.5,
    ),
}

# === Production default ===
# The L2800N10080 z=0 (max_snap=144) configuration is the production
# baseline; the full-range halo arrays are available on disk. Set
# BAQARO_MAX_SNAP lower for short-range (higher-z) runs.


# === Active simulation ===
# Set the ``BAQARO_SIM`` env var to switch — e.g.
#   BAQARO_SIM=L2800N5040 python -m ...
# This is preferred over editing the default below, because it lets parallel
# jobs target different sims without race-y file edits.
ACTIVE_SIM = os.environ.get("BAQARO_SIM", "L2800N10080")


SIM = SIMS[ACTIVE_SIM]

# Module-level aliases resolved from the active simulation. Import these
# rather than hardcoding values:
#     from baqaro.utils.sim_config import boxsize, max_snap
boxsize = SIM.boxsize
N_particles_per_side = SIM.N_particles_per_side
simulation_name = SIM.name

# ``max_snap`` honours the per-run ``BAQARO_MAX_SNAP`` override and falls back
# to the active simulation's production default (144 for L2800N10080 at z=0,
# 78 for L2800N5040). Import it from here rather than re-reading the env var
# with a hardcoded fallback, which would not track the active simulation.
max_snap = int(os.environ.get("BAQARO_MAX_SNAP", str(SIM.max_snap)))

# --- Halo-data variant selection (single source of truth) ------------------
# Both flags select which on-disk halo products / merger classification a run
# consumes, and are appended to output filenames downstream. Centralised here
# so every entry point shares one default; override per run with the env vars.
#
#   fold_subhalo_mass  — load the mass-conserving ``_foldmass`` halo
#                        files (BAQARO_FOLD_SUBHALO_MASS=1/0, default on).
#   merger_delay_mode  — halo merger-timing classification
#                        (BAQARO_MERGER_DELAY_MODE, default "instant_new").
fold_subhalo_mass = env_bool("BAQARO_FOLD_SUBHALO_MASS", True)
merger_delay_mode = os.environ.get("BAQARO_MERGER_DELAY_MODE", "instant_new")

# Subsample geometry aliases (resolution-dependent; see SimSpec docstring).
subsample_log_M_lo = SIM.subsample_log_M_lo
subsample_log_M_hi = SIM.subsample_log_M_hi
subsample_n_bins = SIM.subsample_n_bins

# Time-window averaging fraction (sim-dependent — finer Δt/t_dyn at
# L2800N10080 motivates a slightly larger value than L2800N5040).
tdyn_fraction_default = SIM.tdyn_fraction_default

# ---------------------------------------------------------------------------
# Snapshot (z) grids — see SimSpec. TWO grids with separate owners:
#   training_snapshots_default            -> emulation/main_training.py
#   multinode_evolution_snapshots_default -> core_functions/main_evolution_chunked.py
# Keep them distinct: widening the forward-run grid must NOT re-index the
# training HDF5s / emulators.
# ---------------------------------------------------------------------------
def _parse_snap_list(env_name, fallback):
    """Comma-separated snapshot list from the env, else ``fallback``.

    Sorted + de-duplicated. Raises on a non-integer entry rather than silently
    dropping it (a typo'd grid would otherwise produce a valid-looking run on
    the wrong redshifts).
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(fallback)
    try:
        snaps = sorted({int(s) for s in raw.split(",") if s.strip()})
    except ValueError as exc:
        raise ValueError(f"{env_name}={raw!r}: expected comma-separated ints") from exc
    if not snaps:
        raise ValueError(f"{env_name}={raw!r}: parsed to an empty snapshot list")
    return snaps


training_snapshots_default = _parse_snap_list(
    "BAQARO_TRAINING_SNAPSHOTS", SIM.training_snapshots_default)

# ``None`` in the SimSpec ⇒ the multinode run mirrors the training grid.
multinode_evolution_snapshots_default = _parse_snap_list(
    "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS",
    SIM.multinode_evolution_snapshots_default
    if SIM.multinode_evolution_snapshots_default is not None
    else training_snapshots_default)

# Back-compat alias. The old name meant "the training grid" everywhere it was
# read, so it keeps pointing there. Prefer the explicit names above.
snapshots_to_save_default = training_snapshots_default

# Training-output halo-mass histogram range (per-sim).
log_M_halo_qhmf_lo = SIM.log_M_halo_qhmf_lo
log_M_halo_qhmf_hi = SIM.log_M_halo_qhmf_hi


# ---------------------------------------------------------------------------
# FIDUCIAL model-identity defaults.
# ---------------------------------------------------------------------------
# These make a BARE invocation of the pipeline (no BAQARO_* env vars) reproduce
# the adopted fiducial end-to-end, so no per-command env block is needed. Each
# is still individually overridable by its ``BAQARO_*`` var (the constants below
# are only the *fallback* the readers pass to ``os.environ.get``). The physics
# tokens (``_g6.21``/``_feffcorr``) are emitted by ``growth_feff_suffix`` below;
# the notes tokens name the training/emulator products of THIS batch on disk.
# With the K22 subset geometry above, a bare run resolves to
# `..._clean_z0_g6.21_K22_final_sub_...K22...g6.21_feffcorr...`.
#
# ⚠ Changing any of these silently repoints every bare reader to a different
# on-disk product. Do it only alongside a matching regeneration, and update
# the documentation in the same commit.
# Note that `plotting_paper/fiducial_data.py` does NOT read these constants: it
# pins its own product names explicitly, so the figures and a bare pipeline run
# can be moved to a new batch independently.
# The fiducial itself lives in utils/fiducial.py -- ONE place, plain literals,
# no env reads -- so it can also be imported by plotting_paper/fiducial_data.py
# BEFORE that module pins os.environ. These names are the historical aliases the
# entry points already import; they are re-exports, not a second definition.
from baqaro.utils import fiducial as _fid

FIDUCIAL_GROWTH_SUM_MAX = _fid.GROWTH_SUM_MAX            # ~498x M_BH growth/snap
FIDUCIAL_MADAU_FEFF_CORRECTION = _fid.MADAU_FEFF_CORRECTION  # -> `_feffcorr` token
FIDUCIAL_SMOOTH_TRAINING = _fid.SMOOTH_TRAINING          # SG pre-smoothing of training curves
FIDUCIAL_NOTES_TRAINING = _fid.NOTES_TRAINING            # training HDF5 notes token
FIDUCIAL_NOTES_EMULATION = _fid.NOTES_EMULATION          # emulator .xz notes token
#: Registry key of the ADOPTED fiducial parameter set. Forward-run outputs carry
#: it as the `_bestfit_<name>` filename token, so any reader that resolves a run
#: path needs it to reproduce the fiducial on a bare invocation. Kept here rather
#: than in plotting_paper/fiducial_data.py so plotting_common does not have to
#: depend on plotting_paper to know what "the fiducial" is.
FIDUCIAL_BESTFIT_NAME = _fid.BESTFIT_NAME


def halo_histories_name(max_snap_=None, nbound_threshold=40,
                        halo_filtering_mode="global", tdyn_fraction=None):
    """Canonical `name_file_halos` for the halo-history .npy products.

    Returns
    ``{sim}_maxsnap{N}_nboundthresh{T}_halofilter_{mode}_tdynfraction_{f}[_foldmass][_{merger_delay_mode}]``
    — i.e. the SAME string `halo_mass_histories_saver.py` writes and
    `plotting_common/load_data_to_plot.py` reads.

    ⚠ The trailing `_foldmass` / `_{merger_delay_mode}` tokens are NOT optional
    decoration — both variants coexist on disk, e.g. `..._tdynfraction_0.2.npy`
    alongside `..._tdynfraction_0.2_foldmass_instant_new.npy`. A token-less name
    silently loads the arrays built WITHOUT the sub-threshold mass fold, which
    have a different halo-mass budget, and with the legacy `instant_old` merger
    classification. Always build the name through this helper rather than
    formatting it inline, so no caller has to remember the tokens.
    """
    ms = max_snap if max_snap_ is None else max_snap_
    tf = tdyn_fraction_default if tdyn_fraction is None else tdyn_fraction
    name = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
        simulation_name, ms, nbound_threshold, halo_filtering_mode, tf)
    if fold_subhalo_mass:
        name += "_foldmass"
    if merger_delay_mode != "instant_old":
        name += "_{}".format(merger_delay_mode)
    return name


def growth_feff_suffix():
    """Filename suffix for the growth-cap + madau-f_eff model variants.

    Returns the ``_g{cap}[_feffcorr]`` tokens (in that order) that
    ``main_training.py`` appends to a run's ``name_file`` AFTER the
    ``_sub_{subset_tag}`` token, read from the env at call time:

    - ``BAQARO_GROWTH_SUM_MAX`` (default ``FIDUCIAL_GROWTH_SUM_MAX`` = 6.21 ⇒
      ``_g6.21``; integer caps render as ``_g4``, fractional as ``_g4.6``; set
      ``=50`` to disable the cap and drop the token)
    - ``BAQARO_MADAU_FEFF_CORRECTION`` (default ON ⇒ ``_feffcorr``; set ``=0`` to
      drop it)

    A bare reader therefore resolves to the ``_g6.21_feffcorr`` products;
    products built under a different cap need an explicit
    ``BAQARO_GROWTH_SUM_MAX`` override to be found.

    This is the single canonical home for the token logic — every consumer
    (training, pooling, emulation, inference, CV, emulator-vs-real,
    comparison plots) must call this rather than re-deriving it inline, so
    a capped/_feffcorr run resolves to the same filenames everywhere.
    """
    suffix = ""
    g = float(os.environ.get("BAQARO_GROWTH_SUM_MAX", str(FIDUCIAL_GROWTH_SUM_MAX)))
    if g != 50.0:
        suffix += f"_g{int(g)}" if float(g).is_integer() else f"_g{g:g}"
    # feff keeps its existing tolerant idiom (bit-identical to the runtime feff
    # parses in main_evolution / main_training / the engines) rather than
    # env_bool, so the _feffcorr token can never diverge from the physics for
    # any input. env_bool here is reserved for the strict-`== "1"` toggles.
    feff = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION",
                           "1" if FIDUCIAL_MADAU_FEFF_CORRECTION else "0")
            .strip().lower() not in ("0", "", "false", "no", "off"))
    if feff:
        suffix += "_feffcorr"
    return suffix


def physics_toggle_suffix():
    """Filename suffix for the model-changing physics toggles of main_evolution.

    Each of these env vars *changes the physics* of a forward run, so it must
    be reflected in ``name_file`` (like ``_g{cap}`` / ``_feffcorr`` /
    ``_clipfloor``); otherwise a diagnostic run with e.g.
    ``BAQARO_MERGER_ON=0 BAQARO_BESTFIT_NAME=qcc_ck22final_v1`` would land on
    the SAME path as the fiducial run. This helper emits a token for each
    toggle that is off its production default, read from the env at call time
    (mirrors ``growth_feff_suffix``):

    - ``BAQARO_MERGER_ON``          (default on)      → ``_nomerge`` when off
    - ``BAQARO_RAD_EFFICIENCY_MODEL`` (default ``madau+``) → ``_radeff_{model}``
    - ``BAQARO_TAU_COH_ZERO``       (default off)     → ``_tau0`` (Branch C)
    - ``BAQARO_FORCE_NO_SAMPLER``   (default off)     → ``_nosampler`` (Branch A)

    Appended AFTER the ``_g{cap}[_feffcorr]`` tokens in both
    ``main_evolution.py`` and ``main_evolution_chunked.py``. All production
    defaults ⇒ empty string ⇒ fiducial filenames unchanged.

    The boolean predicates go through the shared tolerant ``env_bool`` — the
    SAME parser the drivers now use for these toggles — so the token can never
    disagree with the physics actually run (e.g. ``BAQARO_MERGER_ON=true``
    disables mergers AND emits ``_nomerge``).
    """
    suffix = ""
    if not env_bool("BAQARO_MERGER_ON", True):
        suffix += "_nomerge"
    rad_model = os.environ.get("BAQARO_RAD_EFFICIENCY_MODEL", "madau+")
    if rad_model != "madau+":
        suffix += f"_radeff_{rad_model}"
    if env_bool("BAQARO_TAU_COH_ZERO", False):
        suffix += "_tau0"
    if env_bool("BAQARO_FORCE_NO_SAMPLER", False):
        suffix += "_nosampler"
    return suffix
