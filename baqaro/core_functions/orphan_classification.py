"""Orphan merger-target classification under the "old" and "new" schemes.

This is the single source of truth for ``classify_orphans`` and the per-snap
catalogue readers it needs. It lives in ``core_functions`` because the
**default** ``merger_delay_mode="instant_new"`` path in
``halo_mass_histories_saver.get_merger_trees`` calls it — i.e. it is
production code, not a diagnostic.

Nothing here reads module-level configuration: the caller supplies the
catalogue path ``template``, so the module is portable across simulations and
machines.
"""
# These helpers live in core_functions so that the production halo pipeline
# has no dependency on any diagnostic tooling; other modules import them from
# here.

import h5py
import numpy as np

__all__ = [
    "KNOWN_SCHEMES",
    "classify_orphans",
    "open_snap",
    "per_snap_groupby",
    "read_field_at_snap_per_orphan",
]

KNOWN_SCHEMES = ("old", "new")


# -----------------------------------------------------------------------------
# Catalogue readers
# -----------------------------------------------------------------------------

def open_snap(template, snap, rdcc_mb=128):
    """Open an OrderedSubSnap file with an enlarged HDF5 chunk cache."""
    return h5py.File(
        template.format(snap_nr=snap),
        "r",
        rdcc_nbytes=rdcc_mb * 1024 * 1024,
        rdcc_nslots=10007,
    )


def per_snap_groupby(snap_per_item, item_idx=None):
    """
    Yield (snap, item_indices) pairs grouping ``item_idx`` by their
    snapshot in ``snap_per_item``. If ``item_idx`` is None, the indices
    are 0..n-1 (positions in ``snap_per_item``).

    Used by every per-snap loop to minimise file opens.
    """
    n = len(snap_per_item)
    if item_idx is None:
        item_idx = np.arange(n)
    order = np.argsort(snap_per_item, kind="stable")
    snap_sorted = snap_per_item[order]
    idx_sorted = item_idx[order]
    uniq = np.unique(snap_sorted)
    cuts = np.append(np.searchsorted(snap_sorted, uniq), n)
    for k, s in enumerate(uniq):
        yield int(s), idx_sorted[cuts[k]:cuts[k + 1]]


def read_field_at_snap_per_orphan(template, orphan_rows, snap_per_orphan,
                                  field_name, default=-1, desc=None):
    """
    For each orphan in ``orphan_rows``, look up the value of
    ``Subhalos/{field_name}`` at its OWN snapshot (``snap_per_orphan``,
    same length as ``orphan_rows``). Returns a per-orphan array.

    Used to read NestedParentTrackId at each orphan's death snapshot
    (or analogous per-orphan-per-snap lookups). Performs one full-column
    read per unique snapshot value.
    """
    from tqdm import tqdm
    n = orphan_rows.size
    result = np.full(n, default, dtype=np.int64)
    desc_label = desc or f"reading {field_name}"
    for s, group in tqdm(list(per_snap_groupby(snap_per_orphan)), desc=desc_label):
        if s < 0:
            continue
        with open_snap(template, s) as f:
            n_at_s = f["Subhalos/TrackId"].shape[0]
            col = f[f"Subhalos/{field_name}"][:]
        rows = orphan_rows[group]
        valid = rows < n_at_s
        if valid.any():
            result[group[valid]] = col[rows[valid]]
    return result


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

def classify_orphans(scheme, sink_id, desc_id, nest_id_final, snap_death,
                     snap_sink, *, template=None):
    """
    Classify orphans under the chosen merger scheme.

    Parameters
    ----------
    scheme : "old" | "new"
    sink_id, desc_id, nest_id_final : (N,) int arrays at the final snapshot
    snap_death, snap_sink : (N,) int arrays
    template : str, only required for ``scheme="new"`` -- catalog path
        format string used to read NestedParentTrackId at each orphan's
        death snap (a single per-snap pass over the catalogue).

    Returns
    -------
    dict with:
        is_orphan       : (N,) bool
        sink_mask       : (N,) bool   sink waterfall (same in both schemes)
        desc_mask       : (N,) bool   Descendant fall-through (same)
        nest_mask       : (N,) bool   NestedParent fall-through (scheme-specific)
        lost_mask       : (N,) bool   no target on any channel (scheme-specific)
        target_id       : (N,) int    chosen merger target (-1 where lost)
        merger_snap     : (N,) int    snapshot at which to fire the merger
        nest_id_used    : (N,) int    nest_id used in classification
                                       (== nest_id_final for old, == nest_at_death for new)
        dts_mask        : (N,) bool   sunk + sink > death (= disrupt-then-sink)
        aligned_mask    : (N,) bool   sunk + sink == death
    """
    if scheme not in KNOWN_SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}; expected one of {KNOWN_SCHEMES}")

    is_orphan = snap_death != -1
    sink_mask = is_orphan & (sink_id != -1)
    desc_mask = is_orphan & (sink_id == -1) & (desc_id != -1)

    # Subset that needs a NestedParent target: sink == -1 AND desc == -1.
    need_nest_mask = is_orphan & (sink_id == -1) & (desc_id == -1)

    if scheme == "old":
        nest_id_used = nest_id_final
    else:
        if template is None:
            raise ValueError("scheme='new' requires a catalog ``template``")
        # Per-snap pass: NestedParentTrackId at each orphan's death snap.
        # We widen the lookup to ALL orphans where (sink==-1 & desc==-1),
        # not just those with nest_id_final != -1, so we also catch
        # orphans whose nest was set at death but later reset to -1
        # (the small ``set→-1`` subset reported by id_evolution_at_death).
        rows = np.flatnonzero(need_nest_mask)
        per_orphan = read_field_at_snap_per_orphan(
            template, rows, snap_death[rows].astype(np.int64),
            "NestedParentTrackId",
            desc="classify_orphans: nest_at_death",
        )
        nest_id_used = np.full_like(nest_id_final, -1)
        nest_id_used[rows] = per_orphan

    nest_mask = need_nest_mask & (nest_id_used != -1)
    lost_mask = need_nest_mask & (nest_id_used == -1)

    # Build target_id and merger_snap (full-N arrays).
    target_id = np.full_like(sink_id, -1)
    target_id[sink_mask] = sink_id[sink_mask]
    target_id[desc_mask] = desc_id[desc_mask]
    target_id[nest_mask] = nest_id_used[nest_mask]

    merger_snap = snap_death.copy()
    if scheme == "new":
        merger_snap[sink_mask] = snap_sink[sink_mask]
    # else: "old" keeps merger_snap = snap_death everywhere.

    dts_mask     = sink_mask & (snap_sink > snap_death)
    aligned_mask = sink_mask & (snap_sink == snap_death)

    return dict(
        is_orphan=is_orphan,
        sink_mask=sink_mask, desc_mask=desc_mask,
        nest_mask=nest_mask, lost_mask=lost_mask,
        target_id=target_id, merger_snap=merger_snap,
        nest_id_used=nest_id_used,
        dts_mask=dts_mask, aligned_mask=aligned_mask,
    )
