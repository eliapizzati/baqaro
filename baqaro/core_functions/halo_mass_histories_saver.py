"""
Halo Mass Histories and Accretion Rate Computation
===================================================

This module loads halo mass histories and merger tree data from HBT-HERONS
catalogues (Forouhar-Moreno et al. 2025), applies resolution filters, computes
various accretion rate metrics, and persists results to disk in a memory-
efficient format.

Deliberate modelling choices: per-halo mass is HBT's ``LastMaxMass`` (peak
mass), merger pointers follow the Sink -> Descendant -> NestedParent fallback
order, and no central/satellite distinction is made (see the notes at the end
of this docstring).

Pipeline Overview
-----------------
1. Load halo masses and merger trees from HBT-HERONS snapshot files
2. Apply resolution filter (global or local mode) to remove unresolved halos
3. Flush filtered data to disk as memory-mapped arrays ((n_snapshots, n_halos) C-order for fast row reads)
4. Compute dynamical-time-weighted accretion rates:
   - Absolute rates: dM/dt [internal units / Gyr]
   - Specific rates: (dM/dt) / M_prev [Gyr^-1]
   - Cold specific rates: specific rates × f_cold(M, z) [Gyr^-1]

Key Data Structures
-------------------
- halo_masses: (n_snapshots, n_halos) float32 array, C-ordered on disk [internal units]
- merger_trees: dict with keys 'track_ids', 'merger_track_ids',
  'snapshot_indexes_of_birth', 'snapshot_indexes_of_death'
- MergerTreeLoader: Lazy loader class for memory-mapped access to persisted trees

Physics References
------------------
- Cold accretion fraction: Based on Correa et al. (2018) fitting formulae
- Dynamical time: t_dyn = t_H(z) × sqrt(2/Delta_vir) × fraction
- HBT-HERONS subhalo finder: Forouhar-Moreno et al. (2025), MNRAS 543, 1339

Notes
-----
- All 2D arrays use (n_snapshots, n_halos) C-order for efficient snapshot-wise row access
- HDF5 reads use enlarged chunk cache (128MB) for faster bulk loading
- Per-halo mass is HBT's ``LastMaxMass`` (peak Mbound up to that snapshot), not
  instantaneous Mbound. This is monotonic along a track, so tidal stripping is
  invisible and accretion rates are non-negative by construction.
- HostHaloId / Rank / Depth are NOT loaded: every subhalo is treated identically
  for BH seeding/accretion, with no central/satellite split.
"""

import gc
import os
import time
import uuid
import h5py
import numpy as np
from tqdm import tqdm


def _unique_tmp_npy(pathfile):
    """Sibling ``.npy`` tmp path, unique per process (pid + uuid).

    Used for atomic publication of the big halo-history arrays: write the
    memmap to this tmp name, fill it, flush, then ``os.replace`` onto the final
    path. A job killed mid-fill leaves only the tmp (never the final), so
    downstream mmap loads never silently consume a zero-tailed partial file.
    The pid+uuid keeps two concurrent builders on a shared FS from colliding.
    """
    base = pathfile[:-4] if pathfile.endswith(".npy") else pathfile
    return f"{base}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npy"

from qhtools.utils.cosmology import cosmo

from baqaro.utils.local_utils import get_mass_resolution_simulation
from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
from baqaro.utils.my_units import mass_units, halo_mass_units_hbt


# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

_MERGER_DELAY_MODES = (
    "instant_old",
    "instant_new",
    "simha_cole_default",
    "simha_cole_calibrated",
    "no_merge",
)


def get_merger_trees(paths, max_snap, *, merger_delay_mode="instant_new",
                       df_delays_cache_dir=None, source=None, sim=None):
    """
    Read merger tree arrays from the final snapshot and resolve merger IDs.

    Parameters
    - paths: str
        Format string for the HBT files, e.g. ".../OrderedSubSnap_{snap_nr:03d}.hdf5".
    - max_snap: int
        Snapshot number to read (typically the last available snapshot).
    - merger_delay_mode: str, default "instant_new"
        How merger events are timed for orphan halos.

        The default is ``"instant_new"`` here and in ``utils/sim_config.py``;
        ``get_halo_mass_histories`` below defaults to ``"instant_old"``. Every
        live caller passes the mode explicitly (the saver's ``__main__`` passes
        ``merger_delay_mode=<sim_config>``).

        Modes:

          * ``"instant_old"`` -- legacy production behaviour (Sink -> Descendant
            -> NestedParent fallback chain at the FINAL snap; merger fires at
            ``SnapshotIndexOfDeath``). Kept for regression compatibility with
            earlier training/inference runs.
          * ``"instant_new"`` -- NEW-scheme classification: Sink -> Descendant ->
            NestedParent at SnapshotIndexOfDeath. Sunk orphans fire at
            ``SnapshotIndexOfSink``; everything else at ``SnapshotIndexOfDeath``.
          * ``"simha_cole_default"`` / ``"simha_cole_calibrated"`` --
            reserved modes, not used for any published product: NEW
            classification plus a delayed merger time on fall-through
            orphans, read from an external precomputed delay cache. Orphans
            whose delayed merger time falls past z=0 are dropped
            (``merger_track_ids`` becomes -1).
          * ``"no_merge"`` -- NEW classification; ALL fall-through orphans
            dropped (merger_track_ids = -1). Sunk orphans still merge at
            their sink snap.
    - df_delays_cache_dir: str, optional
        Directory holding the precomputed ``predicted_delays_max_snap_{N}.npz``
        used by the reserved ``simha_cole_*`` modes.
    - source, sim: str | None, optional
        Which (source, sim) that delay cache belongs to, when its location is
        resolved from the environment (BAQARO_SOURCE_DIR / BAQARO_SIM). Set
        these (the saver does) so another simulation's delays are never used.

    Returns
    - dict with keys 'track_ids', 'merger_track_ids',
        'snapshot_indexes_of_birth', 'snapshot_indexes_of_death'.

    Notes
    - The default mode "instant_old" reproduces the legacy fallback order
      (Sink -> Descendant -> NestedParent at the FINAL snap), which is the
      inverse of the HBT-HERONS docs' recommendation.
    - The "instant_new" / "simha_cole_*" / "no_merge" modes share a single
      classification (see ``orphan_classification.classify_orphans``):
      NestedParentTrackId is read at SnapshotIndexOfDeath, and sunk orphans
      fire at SnapshotIndexOfSink.
    """
    if merger_delay_mode not in _MERGER_DELAY_MODES:
        raise ValueError(f"merger_delay_mode={merger_delay_mode!r} not one of "
                         f"{_MERGER_DELAY_MODES}")

    # 1. READ FINAL-SNAP FIELDS (always; some modes also need SnapshotIndexOfSink).
    file_path = paths.format(snap_nr=max_snap)
    print(f"Loading merger trees from {file_path} (mode={merger_delay_mode!r})...")
    with h5py.File(file_path, 'r', rdcc_nbytes=128*1024*1024, rdcc_nslots=10007) as file:
        track_ids               = np.asarray(file['Subhalos/TrackId'])
        sink_track_ids          = np.asarray(file['Subhalos/SinkTrackId'])
        descendant_track_ids    = np.asarray(file['Subhalos/DescendantTrackId'])
        nestedparent_track_ids  = np.asarray(file['Subhalos/NestedParentTrackId'])
        snapshot_indexes_of_birth = np.asarray(file['Subhalos/SnapshotIndexOfBirth'])
        snapshot_indexes_of_death = np.asarray(file['Subhalos/SnapshotIndexOfDeath'])
        if merger_delay_mode != "instant_old":
            snapshot_indexes_of_sink = np.asarray(file['Subhalos/SnapshotIndexOfSink'])

    # 2. LEGACY FALLBACK CHAIN (default mode)
    if merger_delay_mode == "instant_old":
        # Sink -> Descendant -> NestedParent at FINAL snap.
        # Kept verbatim for backwards compatibility with existing runs.
        print("  Resolving merger history logic (legacy fallback)...")
        merger_track_ids = sink_track_ids.copy()
        mask_missing = (merger_track_ids == -1)
        if np.any(mask_missing):
            merger_track_ids[mask_missing] = descendant_track_ids[mask_missing]
        mask_still_missing = (merger_track_ids == -1)
        if np.any(mask_still_missing):
            merger_track_ids[mask_still_missing] = nestedparent_track_ids[mask_still_missing]
        print("  Done.")
        return {
            'track_ids': track_ids,
            'merger_track_ids': merger_track_ids,
            'snapshot_indexes_of_birth': snapshot_indexes_of_birth,
            'snapshot_indexes_of_death': snapshot_indexes_of_death,
        }

    # 3. NEW-SCHEME CLASSIFICATION (per-snap pass to read NestedParent at death).
    # Reused across instant_new / simha_cole_* / no_merge.
    from baqaro.core_functions.orphan_classification import classify_orphans
    print("  Classifying orphans under NEW scheme (this triggers a per-snap "
          "NestedParent-at-death pass)...")
    cls = classify_orphans(
        "new", sink_track_ids, descendant_track_ids, nestedparent_track_ids,
        snapshot_indexes_of_death, snapshot_indexes_of_sink,
        template=paths,
    )
    target_id_new   = cls['target_id']      # full-N int64
    merger_snap_new = cls['merger_snap']    # full-N int32

    if merger_delay_mode == "instant_new":
        print("  Done.")
        return {
            'track_ids': track_ids,
            'merger_track_ids': target_id_new.astype(track_ids.dtype, copy=False),
            'snapshot_indexes_of_birth': snapshot_indexes_of_birth,
            'snapshot_indexes_of_death': merger_snap_new,
        }

    # 4. DF-DELAY MODES: load predicted_delays cache and override fall-through.
    #    The cache is aligned with eval_idx of the orbit_state cache (i.e.
    #    NEW-scheme orphans with a valid target). Fall-through orphans are
    #    those with channel == 2 (desc) or channel == 3 (nest).
    target_id_out  = target_id_new.copy()
    death_snap_out = merger_snap_new.copy()

    # Identify fall-through orphans via the NEW classification masks.
    # Orphans not in fall-through (sunk or lost) keep their NEW-scheme target/snap.
    fall_mask_full = cls['desc_mask'] | cls['nest_mask']

    if merger_delay_mode == "no_merge":
        # Drop all fall-through orphans -- treat as disrupted.
        target_id_out[fall_mask_full] = -1
        print("  no_merge: dropped fall-through orphans (merger_track_ids = -1).")
        return {
            'track_ids': track_ids,
            'merger_track_ids': target_id_out.astype(track_ids.dtype, copy=False),
            'snapshot_indexes_of_birth': snapshot_indexes_of_birth,
            'snapshot_indexes_of_death': death_snap_out,
        }

    # simha_cole_default | simha_cole_calibrated
    if df_delays_cache_dir is None:
        try:
            from baqaro.df_delays._common import get_df_delays_output_dir
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise ImportError(
                f"merger_delay_mode={merger_delay_mode!r} is experimental and "
                "needs a precomputed delay cache that this distribution does "
                "not build. Either pass an explicit `df_delays_cache_dir=` "
                "pointing at a prebuilt predicted_delays_max_snap_*.npz, or use "
                "one of the self-contained modes: 'instant_new' (default), "
                "'instant_old', 'no_merge'."
            ) from exc
        # Resolve the delay-cache dir for THIS run's (source, sim, max_snap).
        # When the caller passes source/sim (the saver's __main__ does, from
        # BAQARO_SOURCE_DIR / sim_config.simulation_name) they authoritatively
        # select the cache; otherwise get_df_delays_output_dir falls back to its
        # env-driven defaults. Passing them matters: without it a hardcoded
        # default can silently load a DIFFERENT simulation's predicted delays,
        # i.e. track IDs from another catalogue used as indices here.
        _dd_kw = {}
        if source is not None:
            _dd_kw["source"] = source
        if sim is not None:
            _dd_kw["sim"] = sim
        df_delays_cache_dir = get_df_delays_output_dir(max_snap=max_snap, **_dd_kw)
    pred_path = os.path.join(df_delays_cache_dir,
                              f"predicted_delays_max_snap_{max_snap}.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"merger_delay_mode={merger_delay_mode!r} requires the Phase 5 "
            f"cache at {pred_path!r} (not built by this distribution)."
        )
    print(f"  loading {pred_path}")
    pred = np.load(pred_path)
    pred_track_id = pred['track_id'].astype(np.int64)
    pred_channel  = pred['channel']
    snap_merger_pred = pred[f"snap_merger__{merger_delay_mode}"]

    # Restrict pred to fall-through (channel 2 or 3).
    fall_pred = np.isin(pred_channel, (2, 3))
    pred_tids_fall = pred_track_id[fall_pred]
    pred_snap_fall = snap_merger_pred[fall_pred]
    # snap_merger == -1 means "out of horizon" -- mark as disrupted.
    out_of_horizon = (pred_snap_fall == -1)
    in_horizon     = ~out_of_horizon
    target_id_out[pred_tids_fall[out_of_horizon]] = -1
    death_snap_out[pred_tids_fall[in_horizon]]    = pred_snap_fall[in_horizon].astype(
        death_snap_out.dtype, copy=False
    )

    n_horizon_drop = int(out_of_horizon.sum())
    n_fall_total   = int(fall_pred.sum())
    print(f"  fall-through orphans     : {n_fall_total:,d}")
    print(f"  dropped (out of horizon) : {n_horizon_drop:,d} "
          f"({n_horizon_drop/max(1,n_fall_total):.2%})")

    return {
        'track_ids': track_ids,
        'merger_track_ids': target_id_out.astype(track_ids.dtype, copy=False),
        'snapshot_indexes_of_birth': snapshot_indexes_of_birth,
        'snapshot_indexes_of_death': death_snap_out,
    }





def get_halo_mass_histories(path, snapshots, min_snap=0, max_snap=None,
                              merger_delay_mode="instant_old", source=None, sim=None):
    """
    Load per-snapshot halo masses and accompanying merger tree metadata.

    Parameters
    - path: str
        Format string for HBT files, e.g. ".../OrderedSubSnap_{snap_nr:03d}.hdf5".
    - snapshots: list[int] | np.ndarray
        Snapshot numbers to process; columns in the returned mass array follow this order.
    - min_snap, max_snap: int
        **Snapshot numbers**, NOT positional indices — and this path ASSUMES
        `snapshots` is the contiguous 0-based range `min_snap..max_snap`
        (the production invariant: `snapshots = np.arange(min_snap, max_snap+1)`),
        so snapshot-number and list-index coincide. `max_snap` is used BOTH as
        the HBT snapshot number of the final tree (`get_merger_trees`) AND as
        the inclusive upper bound of the `snapshots[min_snap:max_snap+1]` row
        loop. Default `max_snap = max(snapshots)`. A non-contiguous or
        non-0-based `snapshots` (e.g. only round-z anchors) is NOT supported
        here — the slice would silently clamp rather than select by number.
    - merger_delay_mode: str, default "instant_old"
        Forwarded to `get_merger_trees`; controls how orphan mergers are
        classified and timed. `"instant_old"` is the legacy behaviour
        (Sink → Descendant → NestedParent at max_snap, death at
        SnapshotIndexOfDeath). See `get_merger_trees` for the other modes.

        `get_merger_trees` and `utils/sim_config.merger_delay_mode` default to
        "instant_new"; the only live caller (this module's `__main__`) always
        passes the mode explicitly.
    - source, sim: str | None
        Forwarded to `get_merger_trees` so the `simha_cole_*` delay cache is
        resolved for the correct (source, sim). None → env-driven defaults.

    Returns
    - halo_masses_all: np.ndarray
        Shape (n_snapshots, n_halos). Masses scaled by `halo_mass_units_hbt`, in internal units.
    - merger_trees: dict
        Dictionary returned by `get_merger_trees()` for the final snapshot.

    Notes
    - When `snapshots` is non-contiguous, rows are written according to
        their position within `snapshots`, not the raw snapshot number.
    """

    if max_snap is None:
        max_snap = max(snapshots)

    # Loading basic quantities from the last snapshot. source/sim are forwarded
    # so the simha_cole_* delay cache resolves for THIS run's simulation.
    merger_trees = get_merger_trees(path, max_snap=max_snap,
                                     merger_delay_mode=merger_delay_mode,
                                     source=source, sim=sim)
    total_number_of_objects = merger_trees['track_ids'].shape[0]

    # Initializing evolution arrays: (n_snapshots, n_halos) C-order
    halo_masses_all = np.zeros((len(snapshots), total_number_of_objects), dtype=np.float32)

    # Halo mass history
    # NOTE: We deliberately use ``LastMaxMass`` (the peak Mbound up to and
    # including this snapshot) rather than the instantaneous ``Mbound`` or
    # ``BoundM200Crit``. Consequences:
    #   - Mass is monotonic non-decreasing along a track ⇒ tidal stripping of
    #     satellites is invisible to the pipeline.
    #   - Accretion rates dM/dt are non-negative by construction.
    #   - LastMaxMass is effectively M_peak, the standard quantity used in
    #     subhalo abundance matching / BH-halo connection studies.
    for snap_idx, snap in enumerate(snapshots[min_snap:max_snap+1]):
        # Optimize cache for sequential reads of large arrays
        with h5py.File(path.format(snap_nr=snap), rdcc_nbytes=128*1024*1024, rdcc_nslots=10007) as file:
            # High-z snapshots (z>~15) can have an empty Subhalos group when no
            # halos have collapsed yet — relevant for L2800N10080 where snap 0
            # sits at z~30. Leave the row at zeros (already initialized).
            if "Subhalos/LastMaxMass" not in file:
                continue
            last_max_masses = file['Subhalos/LastMaxMass'][:] * halo_mass_units_hbt
            # Slice up to len(last_max_masses) because new halos spawn in later snapshots:
            # earlier snapshots have fewer halos than the final total_number_of_objects.
            halo_masses_all[snap_idx + min_snap, :len(last_max_masses)] = last_max_masses

    return halo_masses_all, merger_trees



# =============================================================================
# SECTION 2: RESOLUTION FILTERING
# =============================================================================

def _fold_below_threshold_mass_into_ancestors(halo_masses_all, merger_trees, global_mask):
    """
    Mass-conservation pass: before below-threshold halos are zeroed, fold
    their stored LastMaxMass into the resolved parent they merge into.
    Mutates `halo_masses_all` in place.

    A below-threshold candidate is folded iff its IMMEDIATE
    `merger_track_id` points to a resolved halo. Single-hop only, on
    purpose: HBT's LastMaxMass accumulates merger contributions
    (when A merges into B, B's Mbound jumps to include A's particles),
    so walking multi-hop chains would double-count A — once via its own
    deposit and again via the parent's LastMaxMass which already includes
    it. Mass from chained below-threshold lineages propagates naturally
    through the LastMaxMass of below-threshold intermediates, and lands
    in the resolved tree via the LAST below-threshold link's single-hop
    fold.

    Halos whose immediate parent is itself below threshold (or -1) are
    NOT folded in this pass. If their lineage eventually merges into a
    resolved halo, their mass is carried by the LastMaxMass of the
    below-threshold halo that DOES touch the resolved tree. If the
    lineage never reaches a resolved halo, the mass is lost — as without
    the fold, which is appropriate since no resolved halo could have
    absorbed it.
    """
    merger_track_ids = merger_trees['merger_track_ids']
    death_indexes = merger_trees['snapshot_indexes_of_death']

    n_snaps, n_halos = halo_masses_all.shape

    candidates_mask = (
        (~global_mask)
        & (merger_track_ids != -1)
        & (death_indexes != -1)
    )
    n_candidates = int(candidates_mask.sum())
    if n_candidates == 0:
        print("  Mass-conservation pass: no below-threshold halos with merger parents -- skipping.")
        return

    cand_idx = np.flatnonzero(candidates_mask)
    parents = merger_track_ids[cand_idx]
    death_snaps = death_indexes[cand_idx]

    survivors = global_mask[parents]
    n_survived = int(survivors.sum())
    n_dropped = n_candidates - n_survived

    print(f"  Mass-conservation pass: {n_candidates} below-threshold halos with merger parents.")
    print(f"    Folded (immediate parent resolved):              {n_survived}")
    print(f"    Dropped (immediate parent also below threshold): {n_dropped}")

    if n_survived == 0:
        return

    sub_idx = cand_idx[survivors]
    parent_idx = parents[survivors]
    deposit_snaps = death_snaps[survivors]

    # LastMaxMass at the sub's own death snap (HBT preserves this
    # monotonically thereafter, so reading at death_snap == reading
    # the peak). Use the sub's death snap, not the parent's, because
    # the parent could later die too -- but the deposit lands when
    # the merger event happens, which is the sub's death.
    lost_masses = halo_masses_all[deposit_snaps, sub_idx].copy()

    valid = lost_masses > 0
    n_zero = int((~valid).sum())
    if n_zero:
        print(f"    Note: {n_zero} survivors had zero recorded mass at death (skipped).")
        sub_idx = sub_idx[valid]
        parent_idx = parent_idx[valid]
        deposit_snaps = deposit_snaps[valid]
        lost_masses = lost_masses[valid]
    if lost_masses.size == 0:
        return

    print(f"    Total mass folded: {float(lost_masses.sum()):.3e} (internal units)")

    # --- Optional double-deposit diagnostic ----------------------------------
    # The fold deposits each sub's LastMaxMass onto its resolved parent. But
    # HBT's LastMaxMass is monotone and accumulates merger contributions: when
    # the sub sank into a GROWING central at its Mbound peak, the parent's own
    # LastMaxMass already absorbed the sub's particles at the merger snap — so
    # re-adding the sub double-counts (bounded by <40 particles per candidate,
    # but a systematic dM/dt inflation at deposit snaps). This read-only block
    # quantifies it without changing the fold result. Enable with
    # BAQARO_FOLD_DIAGNOSTIC=1 (default off → zero overhead).
    from baqaro.utils.sim_config import env_bool as _env_bool
    if _env_bool("BAQARO_FOLD_DIAGNOSTIC", False):
        # Parent RAW mass just before vs at the deposit snap. A positive jump
        # means the parent grew across the merger (its LastMaxMass likely
        # already contains the sub → double-count); a flat/zero jump means the
        # parent was plateaued/past-peak and the fold adds genuinely new mass.
        _before_snaps = np.maximum(deposit_snaps - 1, 0)
        _pm_before = halo_masses_all[_before_snaps, parent_idx]
        _pm_at = halo_masses_all[deposit_snaps, parent_idx]
        _parent_jump = (_pm_at - _pm_before).astype(np.float64)
        _growing = _parent_jump > 0
        _n_grow = int(_growing.sum())
        # Double-counted mass estimate: bounded by BOTH the sub's deposited mass
        # and the parent's jump across that snap (can't have double-counted more
        # than either).
        _dbl = np.minimum(lost_masses[_growing].astype(np.float64),
                          _parent_jump[_growing]).sum() if _n_grow else 0.0
        _tot = float(lost_masses.sum())
        print("    [FOLD-DIAGNOSTIC] double-deposit estimate:")
        print(f"      folded onto a GROWING parent (jump>0): {_n_grow:,} / {lost_masses.size:,} "
              f"({_n_grow/max(1,lost_masses.size):.1%})")
        print(f"      est. double-counted mass:              {_dbl:.3e} "
              f"({_dbl/max(1e-300,_tot):.2%} of total folded)")
        # Per-snapshot dM/dt inflation: double-counted mass deposited each snap.
        _grow_snaps = deposit_snaps[_growing]
        _grow_dbl = np.minimum(lost_masses[_growing].astype(np.float64),
                               _parent_jump[_growing])
        _by_snap = np.zeros(n_snaps, dtype=np.float64)
        np.add.at(_by_snap, _grow_snaps, _grow_dbl)
        _worst = np.argsort(_by_snap)[::-1][:5]
        print("      top-5 snaps by double-deposited mass (snap: mass):")
        for _s in _worst:
            if _by_snap[_s] > 0:
                print(f"        {int(_s):3d}: {_by_snap[_s]:.3e}")

    # Per-snapshot running boost: sort by deposit_snap so writes are
    # snapshot-sequential and the C-ordered (n_snaps, n_halos) layout
    # gets contiguous row writes.
    order = np.argsort(deposit_snaps, kind='stable')
    deposit_snaps_s = deposit_snaps[order]
    parents_s = parent_idx[order]
    masses_s = lost_masses[order].astype(halo_masses_all.dtype)

    boundaries = np.searchsorted(deposit_snaps_s, np.arange(n_snaps + 1))
    running_boost = np.zeros(n_halos, dtype=halo_masses_all.dtype)

    for snap in range(n_snaps):
        lo, hi = boundaries[snap], boundaries[snap + 1]
        if lo < hi:
            np.add.at(running_boost, parents_s[lo:hi], masses_s[lo:hi])
        nz = np.flatnonzero(running_boost)
        if nz.size:
            halo_masses_all[snap, nz] += running_boost[nz]


def apply_resolution_filter_combined(halo_masses_all, merger_trees, halo_mass_threshold, mode="global", fold_subhalo_mass=True):
    """
    Combined fast filtering: apply resolution filter to merger trees AND halo masses in a single pass.
    Computes the binary mask once and reuses it, avoiding redundant operations.

    halo_masses_all : np.ndarray
        Array of halo masses at each snapshot, shape (n_snapshots, n_halos). In internal units.
    merger_trees : dict
        Dictionary containing the merger tree information, including:
        - 'track_ids': Array of track IDs for each halo.
        - 'merger_track_ids': Array of track IDs for the mergers.
        - 'snapshot_indexes_of_birth': Array of snapshot indexes where each halo was born.
        - 'snapshot_indexes_of_death': Array of snapshot indexes where each halo died.
    halo_mass_threshold : float
        Threshold for the halo mass to pass the resolution filter, in internal units.
    mode : str, optional
        Mode of the resolution filter, either "global" or "local". Default is "global".
        If "global", halos must exceed the threshold in at least one snapshot to be retained.
        If "local", halos exceed the threshold in the snapshot of their birth and all subsequent snapshots.
    fold_subhalo_mass : bool, optional
        If True (default), fold the LastMaxMass of below-threshold halos that have
        a resolved merger ancestor into that ancestor's mass track (smooth-accretion
        treatment). Halos with no resolved ancestor are still dropped. Active for
        mode="global" only; ignored with a warning for mode="local".

    Returns
    -------
    halo_masses_all : np.ndarray
        Filtered array of halo masses at each snapshot, shape (n_snapshots, n_halos).
        Halos that do not pass the resolution filter are set to zero.
    merger_trees : dict
        Updated merger tree dictionary with modified snapshot indexes of birth and death based on the resolution filter.
    """

    if mode not in ["global", "local"]:
        raise ValueError("Invalid mode. Use 'global' or 'local'.")

    # NOTE: HBT-HERONS can re-resolve orphans -- a subhalo that died (orphaned)
    # can re-appear as a resolved track if it later becomes central in a new FoF
    # group, with SnapshotIndexOfDeath/Sink reset to -1. In "local" mode below,
    # we recompute birth as the first snapshot above threshold, which collapses
    # any orphaned gap; in "global" mode we don't track gaps at all. Either way
    # the pipeline treats a re-resolved track as a single continuous object,
    # which is consistent with our LastMaxMass-based mass model.
    #
    # Mass conservation: below-threshold halos with a valid merger_track_id
    # would otherwise have their mass silently zeroed. They are instead
    # folded into the first resolved ancestor (see
    # `_fold_below_threshold_mass_into_ancestors`). Halos with no resolved
    # ancestor remain dropped, as before.
    binary_mask = halo_masses_all > halo_mass_threshold
    global_mask = np.any(binary_mask, axis=0)

    if fold_subhalo_mass:
        if mode == "global":
            _fold_below_threshold_mass_into_ancestors(halo_masses_all, merger_trees, global_mask)
        else:
            print("  fold_subhalo_mass=True is not supported for mode='local'; ignored.")

    # Update merger trees for halos that never exceed threshold
    merger_trees['snapshot_indexes_of_birth'][~global_mask] = -1
    merger_trees['snapshot_indexes_of_death'][~global_mask] = -1

    if mode == "local":
        # For each retained halo, find the first snapshot where it exceeds threshold
        first_snap = np.argmax(binary_mask[:, global_mask], axis=0)
        merger_trees['snapshot_indexes_of_birth'][global_mask] = first_snap

        # Validate: ensure death >= birth
        invalid_indices = (merger_trees['snapshot_indexes_of_death'] <= merger_trees['snapshot_indexes_of_birth']) & (merger_trees['snapshot_indexes_of_death'] != -1)
        merger_trees['snapshot_indexes_of_birth'][invalid_indices] = -1
        merger_trees['snapshot_indexes_of_death'][invalid_indices] = -1

    # Apply mass filter in one operation
    if mode == "global":
        halo_masses_all *= global_mask[np.newaxis, :]
    else:  # local
        halo_masses_all *= binary_mask

    return halo_masses_all, merger_trees


# =============================================================================
# SECTION 3: DYNAMICAL TIME WINDOWS
# =============================================================================

def calculate_dynamical_time_windows(delta_times_snapshots, redshifts,
                                    Delta_vir=200.0, Om0=0.307, h=0.677,
                                    tdyn_fraction=0.2):
    """
    Compute snapshot-wise lookback indices and time windows based on halo dynamical time.

    NOTE (cosmology): the ``Om0=0.307, h=0.677`` defaults match the on-disk
    rate files; relative to the FLAMINGO values (``Om0=0.3046, h=0.681``) the
    t_dyn shift is <1%.

    Parameters
    - delta_times_snapshots: np.ndarray
        Length n_snapshots. Time since previous snapshot [Gyr].
    - redshifts: np.ndarray
        Length n_snapshots. Redshift at each snapshot.
    - Delta_vir: float
        Virial overdensity, used to scale dynamical time (default 200).
    - Om0: float
        Present-day matter density parameter.
    - h: float
        Hubble parameter normalization.
    - tdyn_fraction: float
        Fraction of the dynamical time to use for the effective window (e.g., 0.2).

    Returns
    - source_indices: np.ndarray (int32)
        Length n_snapshots. For each snapshot i, index j of the previous snapshot such that
        sum(delta_t[j:i]) >= target_dt_gyr(i). If no previous snapshot, -1.
    - effective_dts: np.ndarray (float64)
        Length n_snapshots. Effective window durations [Gyr] corresponding to `source_indices`.
    """
    
    n_snapshots = len(delta_times_snapshots)
    
    # 1. Hubble Factor E(z)
    E_z = np.sqrt(Om0 * (1 + redshifts)**3 + (1 - Om0))
    
    # 2. Dynamical Time t_dyn(z) [Gyr]
    # t_Hubble = 9.778 / h / E_z
    t_Hubble_gyr = (9.778 / h) / E_z
    
    # Factor ~0.1 for Delta=200
    dynamical_factor = np.sqrt(2.0 / Delta_vir)
    
    # Apply the user's scaling fraction
    target_dt_gyr = t_Hubble_gyr * dynamical_factor * tdyn_fraction
    
    # 3. Sliding Window Logic
    source_indices = np.full(n_snapshots, -1, dtype=np.int32)
    effective_dts = np.zeros(n_snapshots, dtype=np.float64)

    for i in range(n_snapshots):
        target = target_dt_gyr[i]
        current_dt = delta_times_snapshots[i]
        prev_idx = i - 1
        
        while current_dt < target and prev_idx >= 0:
            current_dt += delta_times_snapshots[prev_idx]
            prev_idx -= 1
            
        source_indices[i] = prev_idx
        effective_dts[i] = current_dt
        
    return source_indices, effective_dts


# =============================================================================
# SECTION 4: ACCRETION RATE COMPUTATION
# =============================================================================

def get_accretion_rates(halo_masses_all, source_indices, effective_dts, pathfile):
    """
    Compute absolute accretion rates dM/dt and persist to a (n_snapshots, n_halos) C-ordered memmap.

    Parameters
    - halo_masses_all: np.ndarray or memmap
        Shape (n_snapshots, n_halos). Halo masses in internal units.
    - source_indices: np.ndarray
        Length n_snapshots. Lookback indices per snapshot; -1 indicates start-of-simulation.
    - effective_dts: np.ndarray
        Length n_snapshots. Effective time windows [Gyr].
    - pathfile: str
        Output .npy file path.

    Returns
    - np.memmap: Read-write memmap pointing to on-disk dM/dt array (float32, C-order).
    """
    directory = os.path.dirname(pathfile)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    n_snapshots, n_halos = halo_masses_all.shape

    print(f"Allocating C-ordered file: {pathfile} ...")
    tmp_path = _unique_tmp_npy(pathfile)
    rates_disk = np.lib.format.open_memmap(tmp_path, mode='w+', dtype='float32',
                                           shape=(n_snapshots, n_halos), fortran_order=False)

    print(f"Computing dM/dt for {n_halos} halos across {n_snapshots} snapshots...")

    for i in tqdm(range(n_snapshots), desc="Computing dM/dt"):
        src_idx = source_indices[i]
        dt = effective_dts[i]

        if src_idx == -1 or dt <= 0:
            rates_disk[i] = 0.0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                rate = (np.asarray(halo_masses_all[i], dtype=np.float32)
                        - np.asarray(halo_masses_all[src_idx], dtype=np.float32)) / dt
                np.nan_to_num(rate, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            rates_disk[i] = rate

    rates_disk.flush()
    del rates_disk  # release fd before renaming
    os.replace(tmp_path, pathfile)
    print("Done. Saved absolute accretion rates to disk.")
    return np.load(pathfile, mmap_mode='r+')





def get_specific_accretion_rates(halo_masses_all, source_indices, effective_dts,
                                 pathfile, clipping=False):
    """
    Compute specific accretion rates (sSAR) and persist to a (n_snapshots, n_halos) C-ordered memmap.

    sSAR = (dM/dt) / M_prev.

    Parameters
    - halo_masses_all: np.ndarray or memmap
        Shape (n_snapshots, n_halos). Halo masses in internal units.
    - source_indices: np.ndarray
        Length n_snapshots. Previous snapshot index per snapshot; -1 = start of simulation.
    - effective_dts: np.ndarray
        Length n_snapshots. Effective time windows (Gyr).
    - pathfile: str
        Output .npy file path.
    - clipping: bool
        If True, clips sSAR to [1e-3, 1e3] for numerical stability.

    Returns
    - np.memmap: Read-write memmap pointing to the on-disk sSAR array (float32, C-order).
    """
    directory = os.path.dirname(pathfile)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    n_snapshots, n_halos = halo_masses_all.shape

    print(f"Allocating C-ordered file: {pathfile} ...")
    tmp_path = _unique_tmp_npy(pathfile)
    ssar_disk = np.lib.format.open_memmap(tmp_path, mode='w+', dtype='float32',
                                          shape=(n_snapshots, n_halos), fortran_order=False)

    print(f"Processing {n_halos} halos across {n_snapshots} snapshots...")

    for i in tqdm(range(n_snapshots), desc="Computing sSAR"):
        src_idx = source_indices[i]
        dt = effective_dts[i]

        if src_idx == -1 or dt <= 0:
            ssar_disk[i] = 0.0
        else:
            M_curr = np.asarray(halo_masses_all[i], dtype=np.float32)
            M_prev = np.asarray(halo_masses_all[src_idx], dtype=np.float32)

            with np.errstate(divide='ignore', invalid='ignore'):
                valid = M_prev > 0
                ssar = np.zeros_like(M_curr)
                diff = M_curr - M_prev
                np.divide(diff, dt, out=ssar, where=valid)
                np.divide(ssar, M_prev, out=ssar, where=valid)
                ssar[~valid] = 0.0

            if clipping:
                np.clip(ssar, 1e-3, 1e3, out=ssar)

            ssar_disk[i] = ssar

    ssar_disk.flush()
    del ssar_disk  # release fd before renaming
    os.replace(tmp_path, pathfile)
    print("Done. Saved sSAR to disk.")
    return np.load(pathfile, mmap_mode='r+')


# =============================================================================
# SECTION 5: COLD ACCRETION PHYSICS
# =============================================================================

def get_cold_accretion_fraction(M200, z, mass_units=1.0):
    """
    Calculates the cold accretion fraction for a given mass and redshift.

    Based on the fitting formulae from Correa et al. (2018, MNRAS 473, 538)
    for the hot-mode accretion suppression in massive halos.

    The cold fraction is computed as:
        f_cold = 1 - f_hot
        f_hot = 1 / (1 + (M200 / M_half)^a(z))

    where:
        - a(z): redshift-dependent power-law index (piecewise in z)
        - M_half(z): characteristic mass where f_hot = 0.5 (redshift-dependent)

    Parameters
    ----------
    M200 : np.ndarray or float
        Halo mass (in user units, e.g. 1e10 M_sun).
    z : float or np.ndarray
        Redshift. Can be a scalar if M200 is an array (efficient broadcasting).
    mass_units : float
        Conversion factor to Solar Masses. M_sun = M200 * mass_units.

    Returns
    -------
    f_cold : np.ndarray or float
        The cold accretion fraction (0.0 to 1.0).

    References
    ----------
    Correa et al. (2018), MNRAS 473, 538: "The formation of hot gaseous haloes
    around galaxies"
    """
    # Force z to at least 1D for consistent masking, 
    # but keep it scalar-like if input was scalar for efficiency later
    z_in = np.asanyarray(z)
    is_scalar_z = z_in.ndim == 0
    if is_scalar_z:
        z_in = z_in[None] # View as 1D array for masking logic
        
    z_tilde = np.log10(1.0 + z_in)
    
    # --- Calculate a(z) ---
    a_z = np.zeros_like(z_in, dtype=np.float32)
    
    # Masks
    mask1 = (z_in < 2.0)
    mask2 = (z_in >= 2.0) & (z_in < 4.0)
    mask3 = (z_in >= 4.0)
    
    # 0 <= z < 2
    if np.any(mask1):
        exponent1 = -1.26 * z_tilde[mask1] + 1.29 * (z_tilde[mask1]**2)
        a_z[mask1] = -1.86 * (10.0**exponent1)
    
    # 2 <= z < 4
    if np.any(mask2):
        exponent2 = 0.81 * z_tilde[mask2] - 0.42 * (z_tilde[mask2]**2)
        a_z[mask2] = -0.46 * (10.0**exponent2)
        
    # z >= 4
    a_z[mask3] = -1.07
    
    # --- Calculate M_1/2(z) log factor ---
    log_factor = np.zeros_like(z_in, dtype=np.float32)
    
    if np.any(mask1):
        log_factor[mask1] = -0.15 + 0.22 * z_in[mask1] + 0.07 * (z_in[mask1]**2)
    if np.any(mask2):
        log_factor[mask2] = -0.25 + 0.53 * z_in[mask2] - 0.07 * (z_in[mask2]**2)
    log_factor[mask3] = 0.72 + 0.01 * z_in[mask3]
    
    # M_1/2 in Solar Masses
    M_half_solar = 10.0**(12.0 + log_factor)
    
    # Convert M_1/2 to internal units so we can divide M200 / M_half
    M_half_internal = M_half_solar / mass_units
    
    # If z was scalar, extract the scalar values back out to enable broadcasting
    if is_scalar_z:
        a_z = a_z[0]
        M_half_internal = M_half_internal[0]
        
    # --- Calculate Fraction ---
    # f_hot = 1 / (1 + (M / M_half)^a)
    
    # We use M200 directly here. If M200 is array and M_half is scalar, this is fast.
    ratio = M200 / M_half_internal
    
    # Power is the most expensive op; broadcasting scalar exponent is faster
    ratio **= a_z 
    
    f_hot = 1.0 / (1.0 + ratio)
    
    return 1.0 - f_hot


def get_specific_cold_accretion_rates(halo_masses_all, source_indices, effective_dts,
                                      redshifts, mass_units,
                                      pathfile, clipping=False):
    """
    Compute redshift-dependent specific cold accretion rates (sSAR_cold) and write to disk.

    Applies the cold accretion fraction ``f_cold(M200, z)`` per snapshot to the
    specific accretion rates computed from halo mass histories.

    Parameters
    - halo_masses_all: np.ndarray or memmap
        Shape (n_snapshots, n_halos). Halo masses in internal units.
    - source_indices: np.ndarray
        Length n_snapshots. Previous snapshot indices (-1 indicates start of simulation).
    - effective_dts: np.ndarray
        Length n_snapshots. Effective time windows (Gyr).
    - redshifts: np.ndarray
        Length n_snapshots. Redshift per snapshot.
    - mass_units: float
        Conversion factor from user mass units to Solar Masses used inside ``f_cold``.
    - pathfile: str
        Output .npy file path.
    - clipping: bool
        If True, clips sSAR_cold to [1e-3, 1e3] Gyr^-1.

    Returns
    - np.memmap: Read-write memmap (float32, C-order, shape (n_snapshots, n_halos)).
    """
    directory = os.path.dirname(pathfile)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    n_snapshots, n_halos = halo_masses_all.shape

    print(f"Allocating C-ordered file: {pathfile} ...")
    tmp_path = _unique_tmp_npy(pathfile)
    ssar_disk = np.lib.format.open_memmap(tmp_path, mode='w+', dtype='float32',
                                          shape=(n_snapshots, n_halos), fortran_order=False)

    print(f"Processing {n_halos} halos across {n_snapshots} snapshots...")

    for i in tqdm(range(n_snapshots), desc="Computing cold sSAR"):
        src_idx = source_indices[i]
        dt = effective_dts[i]

        if src_idx == -1 or dt <= 0:
            ssar_disk[i] = 0.0
        else:
            M_curr = np.asarray(halo_masses_all[i], dtype=np.float32)
            M_prev = np.asarray(halo_masses_all[src_idx], dtype=np.float32)

            with np.errstate(divide='ignore', invalid='ignore'):
                valid = M_prev > 0
                ssar = np.zeros_like(M_curr)
                diff = M_curr - M_prev
                np.divide(diff, dt, out=ssar, where=valid)
                np.divide(ssar, M_prev, out=ssar, where=valid)
                ssar[~valid] = 0.0

            # Apply cold accretion fraction
            f_cold = get_cold_accretion_fraction(M_curr, redshifts[i], mass_units)
            ssar *= f_cold

            if clipping:
                np.clip(ssar, 1e-3, 1e3, out=ssar)

            ssar_disk[i] = ssar

    ssar_disk.flush()
    del ssar_disk  # release fd before renaming
    os.replace(tmp_path, pathfile)
    print("Done. Saved cold sSAR to disk.")
    return np.load(pathfile, mmap_mode='r+')







# =============================================================================
# SECTION 6: DISK I/O AND LAZY LOADING
# =============================================================================

class MergerTreeLoader:
    """
    Read-only lazy loader for merger tree arrays saved as individual .npy files.

    Parameters
    - folder_path: str
        Directory containing `<key>.npy` files (e.g., track_ids.npy).

    Usage
    - Instantiate with the folder path, then access arrays as attributes:
        `trees = MergerTreeLoader("path/to/folder")`
        `ids = trees.track_ids`  → Loads track_ids.npy as a memmap (mmap_mode='r').

    Methods
    - keys(): list[str]
        Returns the available array names based on files in the folder.
    """
    def __init__(self, folder_path):
        self.folder = folder_path
        self._cache = {}

    def __getattr__(self, name):
        # Return cached mmap if available
        if name in self._cache:
            return self._cache[name]
        
        # Look for the file
        filepath = os.path.join(self.folder, f"{name}.npy")
        if os.path.exists(filepath):
            # Load as memory map (0 RAM usage)
            # mmap_mode='r' automatically detects if it's F-order or C-order
            print(f"  [LazyLoader] Mounting {name} from disk...")
            arr = np.load(filepath, mmap_mode='r')
            self._cache[name] = arr
            return arr
        else:
            raise AttributeError(f"Array '{name}' not found in {self.folder}")
    
    def keys(self):
        """List available arrays in the folder."""
        return [f.replace('.npy', '') for f in os.listdir(self.folder) if f.endswith('.npy')]


# --- 2. Helper: Flush Single Array (Masses) ---
def flush_halo_masses_to_disk(halo_masses, output_filename):
    """
    Save a large halo mass array to disk as a (n_snapshots, n_halos) C-ordered .npy memmap.

    Parameters
    - halo_masses: np.ndarray
        Shape (n_snapshots, n_halos). Already in snapshot-major layout.
    - output_filename: str
        Destination .npy file path.

    Returns
    - str: The path to the created file.

    Notes
    - Output is (n_snapshots, n_halos) C-order for fast snapshot access via ``data[i]``.
    """
    directory = os.path.dirname(output_filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    n_snapshots, n_halos = halo_masses.shape
    dtype = halo_masses.dtype

    print(f"Flushing Halo Masses to {output_filename}...")
    print(f"  Shape: ({n_snapshots}, {n_halos}) | Dtype: {dtype} | C-Order")

    # Allocate as (n_snapshots, n_halos) C-order, on a tmp path, then rename
    # atomically onto output_filename (see _unique_tmp_npy).
    tmp_path = _unique_tmp_npy(output_filename)
    disk_arr = np.lib.format.open_memmap(tmp_path, mode='w+', dtype=dtype,
                                         shape=(n_snapshots, n_halos), fortran_order=False)

    # Write row-by-row: each row = one snapshot (contiguous in C-order)
    for i in tqdm(range(n_snapshots), desc="Writing Masses"):
        disk_arr[i] = halo_masses[i]

    disk_arr.flush()
    del disk_arr  # release the fd before renaming
    os.replace(tmp_path, output_filename)
    return output_filename


# --- 3. Helper: Flush Dictionary (Trees) ---
def flush_merger_trees_smart(merger_trees_dict, output_folder):
    """
    Persist a dictionary of 1D arrays to `<key>.npy` files inside a folder.

    Parameters
    - merger_trees_dict: dict[str, np.ndarray]
        Each key-value pair is saved as `<key>.npy`. All arrays are expected to be 1D.
    - output_folder: str
        Destination directory. Created if it does not exist.
    """
    os.makedirs(output_folder, exist_ok=True)
    print(f"Flushing merger trees to folder: {output_folder}...")

    for key, data in merger_trees_dict.items():
        filename = os.path.join(output_folder, f"{key}.npy")
        print(f"  Saving {key} ({data.shape}, {data.dtype})")
        np.save(filename, data)


# --- 4. MASTER FUNCTION ---
def flush_and_free_ram(halo_masses, merger_trees, mass_path, tree_folder):
    """
    Flush large arrays/dicts to disk and return lightweight, read-only handles.

    Parameters
    - halo_masses: np.ndarray
        Halo mass array (n_snapshots, n_halos), already in snapshot-major layout.
    - merger_trees: dict[str, np.ndarray]
        Dictionary of arrays representing the merger tree metadata; saved as separate .npy files.
    - mass_path: str
        Destination path for the halo mass .npy file.
    - tree_folder: str
        Destination folder where each item in `merger_trees` is saved as `<key>.npy`.

    Returns
    - masses_mmap: np.memmap
        Read-only memmap handle to the persisted halo masses, shape (n_snapshots, n_halos).
    - tree_loader: MergerTreeLoader
        Lazy loader that exposes merger tree arrays via attribute access and memory maps.

    Notes
    - Inputs are deleted from RAM and garbage collected after flushing to minimize memory footprint.
    """

    # 1. Flush Masses
    flush_halo_masses_to_disk(halo_masses, mass_path)
    
    # 2. Flush Trees
    flush_merger_trees_smart(merger_trees, tree_folder)
    
    print("Flushing complete. Releasing RAM...")
    
    # 3. DELETE INPUTS & GC
    del halo_masses
    del merger_trees
    gc.collect()
    
    print("RAM Cleared. Reloading inputs as Lazy Pointers.")
    
    # 4. Return Smart Pointers
    # Masses: Standard numpy memmap
    masses_mmap = np.load(mass_path, mmap_mode='r')
    
    # Trees: Our custom lazy loader object
    tree_loader = MergerTreeLoader(tree_folder)
    
    return masses_mmap, tree_loader


# =============================================================================
# SECTION 7: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
 
    # Module-level configuration
    # Site selected via BAQARO_SOURCE_DIR. Valid: "local", "machine_cosma",
    # "machine_igm" (see utils/my_dir.py).
    DEFAULT_SOURCE_DIR = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")

    notes_file = None

    min_snap = 0

    nbound_threshold = 40
    halo_filtering_mode = "global"

    # Sim-aware defaults (single source of truth — sim_config). All three
    # honour their BAQARO_* env overrides:
    #   max_snap           BAQARO_MAX_SNAP        (default = active sim's z=0 snap)
    #   fold_subhalo_mass  BAQARO_FOLD_SUBHALO_MASS (mass-conservation pass;
    #                      "_foldmass" appended to name_file_base when True)
    #   merger_delay_mode  BAQARO_MERGER_DELAY_MODE (see get_merger_trees(); mode
    #                      name appended to name_file_base when != "instant_old")
    # (tdyn_fraction_default is sim-dependent: 0.20 at L2800N5040, 0.25 at 10080.)
    from baqaro.utils.sim_config import (
        boxsize, N_particles_per_side, simulation_name, tdyn_fraction_default,
        max_snap, fold_subhalo_mass, merger_delay_mode,
    )

    # set the paths of input/output data
    path_sim = get_input_path_HBT_data(source=DEFAULT_SOURCE_DIR)
    path_out = get_output_path(source=DEFAULT_SOURCE_DIR)

    mass_resolution = get_mass_resolution_simulation(boxsize, N_particles_per_side)

    folder_hbt = os.path.join(path_sim, f"{simulation_name}/HBT_compressed")
    redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")
    path_in = os.path.join(folder_hbt, "OrderedSubSnap_{snap_nr:03d}.hdf5")

    # --- FILE I/O SETUP ---

    name_file_base = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
        simulation_name, max_snap, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
    )
    if fold_subhalo_mass:
        name_file_base += "_foldmass"
    if merger_delay_mode != "instant_old":
        name_file_base += "_{}".format(merger_delay_mode)
    if notes_file is not None:
        name_file_base += "_{}".format(notes_file)

    name_file_accretion = "accretion_rates_{}".format(name_file_base)
    name_file_specific_accretion = "specific_accretion_rates_{}".format(name_file_base)
    name_file_specific_cold_accretion = "specific_cold_accretion_rates_{}".format(name_file_base)
    name_file_halo_masses = "halo_masses_{}".format(name_file_base)
    name_file_trees = "merger_trees_{}".format(name_file_base) # Folder name
    
    # OUTPUT PATHS
    # Masses: .npy file
    path_file_halo_masses = os.path.join(path_out, "halo_histories", f"{name_file_halo_masses}.npy")
    
    # Trees: FOLDER path (Removed .hdf5 extension)
    path_file_trees = os.path.join(path_out, "halo_histories", name_file_trees)
    
    # Rates: .npy file
    path_file_accretion = os.path.join(path_out, "halo_histories", f"{name_file_accretion}.npy")
    path_file_specific_accretion = os.path.join(path_out, "halo_histories", f"{name_file_specific_accretion}.npy")
    path_file_specific_cold_accretion = os.path.join(path_out, "halo_histories", f"{name_file_specific_cold_accretion}.npy")


    ###############################################
    # COMPUTING PRELIMINARY HALO PROPERTIES
    ################################################
    start_time_global = time.time()

    # defining basic redshift and time arrays
    snapshots = np.arange(min_snap, max_snap + 1)
    redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]
    ages_of_the_universe = cosmo.age(redshifts)
    delta_times_snapshots = np.diff(ages_of_the_universe, prepend=0.01)  # in Gyr

    # --- 1. LOAD RAW DATA ---
    print("STARTING HALO MASS HISTORIES AND MERGER TREES ROUTINE")
    start_time = time.time()
    
    halo_masses_all, merger_trees = get_halo_mass_histories(
        path_in, snapshots, min_snap, max_snap,
        merger_delay_mode=merger_delay_mode,
        source=DEFAULT_SOURCE_DIR, sim=simulation_name,
    )
    
    total_number_of_objects = merger_trees['track_ids'].shape[0]
    print(f"Total objects: {total_number_of_objects}")
    print(f"Raw Mass Array Size: {halo_masses_all.nbytes / 1e9:.2f} GB")
    print(f"Loading time: {time.time() - start_time:.2f} s")


    # --- 2. FILTER & FLUSH (RAM OPTIMIZATION) ---
    print("\nSTARTING FILTERING AND FLUSHING ROUTINE")
    time_filter = time.time()

    halo_mass_threshold = nbound_threshold * mass_resolution / mass_units 

    # A. Apply Filter (Result in RAM)
    print("  Applying resolution filter...")
    masses_ram, trees_ram = apply_resolution_filter_combined(
        halo_masses_all, merger_trees, halo_mass_threshold,
        mode=halo_filtering_mode, fold_subhalo_mass=fold_subhalo_mass,
    )

    # B. Flush to Disk (Frees RAM)
    # This saves masses to .npy and trees to a folder of .npy files
    print("  Flushing to disk (C-order, snapshot-major)...")
    
    halo_masses_mmap, tree_loader = flush_and_free_ram(
        halo_masses=masses_ram,
        merger_trees=trees_ram,
        mass_path=path_file_halo_masses,  # .npy file
        tree_folder=path_file_trees       # Folder path
    )
    
    print(f"Filtering & Flushing time: {time.time() - time_filter:.2f} s")
    print(f"Masses map shape: {halo_masses_mmap.shape}")


    # --- 3. CALCULATE RATES (DISK-BASED) ---
    print("\nSTARTING COLD ACCRETION RATE CALCULATION")
    time_rates = time.time()

    # A. Calculate Time Windows
    print("  Calculating time windows...")
    src_indices, effective_dts = calculate_dynamical_time_windows(
        delta_times_snapshots, 
        redshifts, 
        tdyn_fraction=tdyn_fraction_default
    )

    # B. Calculate Rates (snapshot-by-snapshot, disk-backed)
    print(f"  Computing rates and saving to disk...")

    rates_mmap = get_accretion_rates(
        halo_masses_all=halo_masses_mmap,
        source_indices=src_indices,
        effective_dts=effective_dts,
        pathfile=path_file_accretion,
    )

    print(f"  Computing specific rates and saving to disk...")

    specific_rates_mmap = get_specific_accretion_rates(
        halo_masses_all=halo_masses_mmap,
        source_indices=src_indices,
        effective_dts=effective_dts,
        pathfile=path_file_specific_accretion,
        clipping=False,
    )

    print(f"  Computing specific cold rates and saving to disk...")

    specific_cold_rates_mmap = get_specific_cold_accretion_rates(
        halo_masses_all=halo_masses_mmap,
        source_indices=src_indices,
        effective_dts=effective_dts,
        redshifts=redshifts,
        mass_units=mass_units,
        pathfile=path_file_specific_cold_accretion,
        clipping=False,
    )

    print(f"Rate Calculation time: {time.time() - time_rates:.2f} s")
    print(f"TOTAL PIPELINE TIME: {time.time() - start_time_global:.2f} s")

    # --- OPTIONAL: QUICK VERIFICATION ---
    # Check a middle snapshot to ensure we don't have all zeros/epsilons
    check_snap = max_snap // 2
    sample_data = rates_mmap[check_snap, :10]
    print(f"\n[Verification] Sample rates at snap {check_snap}: {sample_data}")