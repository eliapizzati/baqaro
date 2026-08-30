"""
Stratified mass-binned subsampling of N-body merger tree roots.
================================================================

Given the full halo catalogue with merger tree pointers, pick ``N_b``
"root" halos at a chosen snapshot ``z_root`` per log-mass bin, walk the
tree backward via the closure-under-mergers BFS, and assign every kept
halo the inverse-probability weight

    w_b = N_b_total / N_b_kept

(or ``1`` for bins kept in full).  Every summary statistic computed on
the kept set with these weights is an unbiased estimator of the same
statistic computed on the full simulation, in expectation.

The mathematical scheme is the same single-ensemble, root-set-weighted
estimator used by EPS-SAM papers (Ricarte & Natarajan 2018, Delphi,
CAT/Trinca+2022), but the conditional mass function P(progenitor | root)
comes from the actual simulation merger trees rather than from an
analytic EPS recipe.

Closure under mergers
---------------------
A "kept" subset must include, for every kept root, *all* halos that
ultimately merge into it — otherwise the merger book-keeping in the BH
evolution kernel breaks.  This module reuses
:func:`select_merger_branches.build_reverse_index` to obtain the CSR
reverse index, then performs a single Numba-JITted BFS that both marks
the closure and propagates the root weight to every progenitor.

Public API
----------
build_subsampled_subset
    Build a subset mask + per-halo weights in one call.
save_subset / load_subset
    On-disk persistence (``.npz``).
make_subset_tag
    Build the filename tag that encodes the subsample config — guarantees
    different N_b / seeds / bin schemes land in different files.
"""

import hashlib
import os
import time
import uuid
import numpy as np
from typing import Optional, Tuple, Union, Sequence

from numba import njit

from baqaro.core_functions.select_merger_branches import (
    build_reverse_index,
)


@njit(cache=True)
def _bfs_propagate_weights(
    selected: np.ndarray,
    weights: np.ndarray,
    owner: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    target_indices: np.ndarray,
    target_weights: np.ndarray,
    alive_at_root: np.ndarray,
) -> None:
    """BFS from roots, marking the closed subset and propagating root weight.

    Modifies ``selected``, ``weights``, ``owner`` in place.  Each progenitor
    inherits the weight (and owner) of the *first* root that reaches it in
    BFS order — when two trees share a progenitor (rare but possible after
    closure under chain mergers), the discoverer wins.

    Re-resolved orphans
    --------------------
    Halos that are alive at z_root with ``merger_track_ids != -1`` (HBT-HERONS
    "re-resolved orphans" that re-emerged as central after a merger event)
    have *two* independent paths into the subset: stratification as a root,
    and closure from their merger target. Inheriting the target's weight via
    closure would double-count them by a factor up to 2× at z<z_root. This
    BFS therefore skips closure-walking into any halo with
    ``alive_at_root=True`` that isn't *already* a selected root — such halos
    are included iff the root stratification picked them. The skip is also
    propagated to their sub-trees, which is consistent because re-resolved
    orphans never trigger a merger event in main_evolution.py
    (``mask_dying_now`` requires ``death_index == i``, never True when
    ``death_index = -1``), so the descendant's BH bookkeeping is unaffected.
    """
    n_halos = len(selected)
    queue = np.empty(n_halos, dtype=np.int64)
    queue_start = 0
    queue_end = len(target_indices)

    for i in range(len(target_indices)):
        tid = target_indices[i]
        queue[i] = tid
        selected[tid] = True
        weights[tid] = target_weights[i]
        owner[tid] = i

    while queue_start < queue_end:
        current_idx = queue[queue_start]
        current_weight = weights[current_idx]
        current_owner = owner[current_idx]
        queue_start += 1

        start = indptr[current_idx]
        end = indptr[current_idx + 1]

        for j in range(start, end):
            src_idx = indices[j]
            if selected[src_idx]:
                continue
            if alive_at_root[src_idx]:
                # Re-resolved orphan with an independent z_root existence.
                # Do not include via closure — let the root stratification
                # handle it. See docstring for the inclusion-probability
                # argument.
                continue
            selected[src_idx] = True
            weights[src_idx] = current_weight
            owner[src_idx] = current_owner
            queue[queue_end] = src_idx
            queue_end += 1


def build_subsampled_subset(
    *,
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    halo_masses_at_root: np.ndarray,
    alive_at_root: np.ndarray,
    log_mass_bins: np.ndarray,
    N_b_per_bin: Union[int, Sequence[int], np.ndarray],
    keep_all_above_log_M: Optional[float] = None,
    rng_seed: int = 42,
    chunk_id: int = 0,
    n_chunks: int = 1,
    reverse_index: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Build a stratified subsampled subset of merger tree roots.

    Parameters
    ----------
    track_ids : (n_halos,) int
        Halo track IDs.  Must satisfy ``track_ids[i] == i`` (contiguous
        0..n_halos-1), per the existing pipeline convention.
    merger_track_ids : (n_halos,) int
        Forward merger map: ``merger_track_ids[i] = j`` means halo *i*
        merges into halo *j*; ``-1`` if disrupted or alive at the end.
    halo_masses_at_root : (n_halos,) float
        Halo mass at the chosen root snapshot, in *physical solar masses*
        (the caller is responsible for the unit conversion from the
        on-disk float32 array).
    alive_at_root : (n_halos,) bool
        True for halos that exist at the root snapshot.  Typically
        ``halo_masses_at_root > 0``, but can be derived from
        ``(birth_index <= root_snap) & ((death_index > root_snap) | (death_index < 0))``
        for explicitness.
    log_mass_bins : (n_bins + 1,) float
        Bin edges in ``log10(M / Msun)``.  Halos outside the outermost
        edges are not selected as roots (but can still be reached as
        progenitors of kept roots from inner bins).
    N_b_per_bin : int or length-``n_bins`` sequence
        Target sample size per bin.  Capped at the actual bin count, so
        sparse bins are kept in full with weight 1.  If a sequence is
        passed, ``N_b_per_bin[b]`` is used for bin ``b`` — enables
        mass-dependent sampling (e.g. thinner low-mass bins where
        observable BHs are rare). The sequence length must equal
        ``len(log_mass_bins) - 1``.
    keep_all_above_log_M : float, optional
        If set, every bin whose lower edge is ``>= keep_all_above_log_M``
        keeps *all* its halos (effectively ``N_b = N_total`` for that bin,
        weight 1). This is the canonical fix for noisy bright-end / massive
        observables under stratified subsampling: the abundant low-mass
        bins still take only ``N_b_per_bin`` halos, but rare massive bins
        are fully resolved. Above the pivot mass, the subset reproduces the
        full simulation exactly. Default: ``None`` (no special treatment;
        N_b_per_bin applies everywhere).
    rng_seed : int, default 42
        Seed for the per-bin random choice.
    chunk_id, n_chunks : int, default 0, 1
        Chunked multi-node partition. With ``n_chunks == 1`` (default)
        this is a no-op and the returned subset is identical to the
        un-chunked build. With ``n_chunks > 1`` the SELECTED roots are
        split into ``n_chunks`` disjoint groups by round-robin *within
        each mass bin*, and only group ``chunk_id`` is closed + returned.
        Because ``rng_seed`` is fixed, every chunk sees the identical
        full selection before partitioning, so the chunks form a clean
        disjoint cover. Per-halo weights are left at the full-``N_b``
        value (``n_total/n_keep``) — NOT recomputed against the smaller
        chunk root count — so per-chunk weighted summary histograms
        SUM exactly to the single-run (``n_chunks=1``) result (this is
        what makes the multinode chunk partitioning exact).
    reverse_index : tuple, optional
        Precomputed CSR ``(indptr, indices)`` from
        :func:`select_merger_branches.build_reverse_index`.  Pass this in
        to avoid the ~2 s rebuild on each call.

    Returns
    -------
    dict with keys:
      ``subset_mask`` : (n_halos,) bool — closed kept subset.
      ``weights``     : (n_halos,) float64 — per-halo inverse-prob weight
                        (0 for not-kept; same weight for a root and all
                        of its progenitors).
      ``root_mask``   : (n_halos,) bool — which kept halos are roots
                        (alive at z_root and selected directly).
      ``root_bin``    : (n_halos,) int16 — mass-bin index of each root's
                        z_root mass; -1 for non-roots.
      ``n_per_bin_total`` : (n_bins,) int — count of alive halos per bin.
      ``n_per_bin_kept``  : (n_bins,) int — count actually kept per bin.
      ``weights_per_bin`` : (n_bins,) float — w_b assigned to each bin.
      ``log_mass_bins``   : echoed back for storage.
      ``meta`` : dict with N_b_per_bin, rng_seed, n_roots, n_total_kept,
                 n_halos_full.
    """
    n_halos = len(track_ids)
    if len(merger_track_ids) != n_halos:
        raise ValueError("merger_track_ids must match track_ids length")
    if len(halo_masses_at_root) != n_halos:
        raise ValueError("halo_masses_at_root must match track_ids length")
    if len(alive_at_root) != n_halos:
        raise ValueError("alive_at_root must match track_ids length")

    rng = np.random.default_rng(rng_seed)

    if reverse_index is None:
        reverse_index = build_reverse_index(merger_track_ids)
    indptr, indices = reverse_index

    # Compute log mass at z_root, masked to alive halos with positive mass.
    valid = alive_at_root & (halo_masses_at_root > 0)
    log_M = np.full(n_halos, -np.inf, dtype=np.float64)
    log_M[valid] = np.log10(halo_masses_at_root[valid].astype(np.float64))

    n_bins = len(log_mass_bins) - 1
    # bin_indices[i] in [0, n_bins) iff log_M[i] falls inside the binning;
    # halos in -inf or above the last edge get values outside this range.
    bin_indices_raw = np.digitize(log_M, log_mass_bins) - 1
    bin_indices = np.where(
        (bin_indices_raw >= 0) & (bin_indices_raw < n_bins) & valid,
        bin_indices_raw,
        -1,
    ).astype(np.int32)

    # Normalize N_b_per_bin to a length-n_bins int64 array; scalar input
    # is broadcast. Validates the array case has the right length.
    _nb_in = N_b_per_bin
    if np.isscalar(_nb_in):
        N_b_array = np.full(n_bins, int(_nb_in), dtype=np.int64)
    else:
        N_b_array = np.asarray(_nb_in, dtype=np.int64)
        if N_b_array.shape != (n_bins,):
            raise ValueError(
                f"N_b_per_bin array length {N_b_array.shape} != n_bins {n_bins}"
            )

    n_per_bin_total = np.zeros(n_bins, dtype=np.int64)
    n_per_bin_kept = np.zeros(n_bins, dtype=np.int64)
    weights_per_bin = np.zeros(n_bins, dtype=np.float64)

    root_indices_list = []
    root_weights_list = []
    root_bins_list = []

    for b in range(n_bins):
        in_bin = np.flatnonzero(bin_indices == b)
        n_total = len(in_bin)
        n_per_bin_total[b] = n_total
        if n_total == 0:
            continue
        # Bins above the pivot mass are kept in full (weight 1) — eliminates
        # the bright-end / massive-halo variance that stratified subsampling
        # otherwise inflates for rare contributors.
        if (
            keep_all_above_log_M is not None
            and log_mass_bins[b] >= keep_all_above_log_M
        ):
            n_keep = n_total
        else:
            n_keep = min(int(N_b_array[b]), n_total)
        if n_keep == n_total:
            kept = in_bin
        else:
            kept = rng.choice(in_bin, size=n_keep, replace=False)
        weight = float(n_total) / float(n_keep)
        n_per_bin_kept[b] = n_keep
        weights_per_bin[b] = weight
        root_indices_list.append(kept.astype(np.int64))
        root_weights_list.append(np.full(n_keep, weight, dtype=np.float64))
        root_bins_list.append(np.full(n_keep, b, dtype=np.int16))

    if len(root_indices_list) == 0:
        raise ValueError(
            "No alive halos fall inside the supplied mass bins at z_root."
        )

    target_indices = np.concatenate(root_indices_list)
    target_weights = np.concatenate(root_weights_list)
    target_bins = np.concatenate(root_bins_list)

    # --- Chunked multi-node partition (default no-op when n_chunks == 1) ---
    # Split the SELECTED roots into n_chunks disjoint groups via round-robin
    # WITHIN each mass bin, then keep only group ``chunk_id``. Round-robin in
    # selection order is deterministic and identical across chunks (rng_seed
    # is fixed), so the chunks are a disjoint, complete cover of the roots.
    # Weights are deliberately left untouched (full-N_b) so that pooling the
    # per-chunk weighted histograms is an exact SUM — see docstring + §6c.
    if n_chunks > 1:
        if not (0 <= chunk_id < n_chunks):
            raise ValueError(
                f"chunk_id={chunk_id} out of range for n_chunks={n_chunks} "
                f"(must satisfy 0 <= chunk_id < n_chunks)"
            )
        keep_chunk = np.zeros(len(target_indices), dtype=np.bool_)
        for b in range(n_bins):
            pos_in_bin = np.flatnonzero(target_bins == b)
            keep_chunk[pos_in_bin[chunk_id::n_chunks]] = True
        target_indices = target_indices[keep_chunk]
        target_weights = target_weights[keep_chunk]
        target_bins = target_bins[keep_chunk]
        if len(target_indices) == 0:
            raise ValueError(
                f"chunk {chunk_id}/{n_chunks} selected zero roots — reduce "
                "n_chunks or check that the mass bins are populated."
            )

    root_mask = np.zeros(n_halos, dtype=np.bool_)
    root_mask[target_indices] = True
    root_bin = np.full(n_halos, -1, dtype=np.int16)
    root_bin[target_indices] = target_bins

    selected = np.zeros(n_halos, dtype=np.bool_)
    weights = np.zeros(n_halos, dtype=np.float64)
    owner = np.full(n_halos, -1, dtype=np.int32)
    # Pass alive_at_root as int8 (Numba is happy with np.ndarray of bool, but
    # int8 keeps the call signature uniform across versions).
    _bfs_propagate_weights(
        selected, weights, owner, indptr, indices, target_indices, target_weights,
        np.ascontiguousarray(alive_at_root),
    )

    return {
        "subset_mask": selected,
        "weights": weights,
        "root_mask": root_mask,
        "root_bin": root_bin,
        "n_per_bin_total": n_per_bin_total,
        "n_per_bin_kept": n_per_bin_kept,
        "weights_per_bin": weights_per_bin,
        "log_mass_bins": log_mass_bins,
        "meta": {
            # Scalar form preserved when uniform (for backward compat).
            "N_b_per_bin": int(N_b_array[0]) if np.all(N_b_array == N_b_array[0]) else -1,
            "N_b_array": N_b_array,
            "rng_seed": int(rng_seed),
            "n_roots": int(len(target_indices)),
            "n_total_kept": int(selected.sum()),
            "n_halos_full": int(n_halos),
            # Chunk provenance (filename tag is authoritative; these are
            # informational and not persisted by save_subset).
            "chunk_id": int(chunk_id),
            "n_chunks": int(n_chunks),
        },
    }


def build_multinode_partition_subset(
    *,
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    resolved_mask: np.ndarray,
    chunk_id: int = 0,
    n_chunks: int = 1,
    reverse_index: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Build ONE chunk of a *whole-catalogue* (no-subsample) merger-tree partition.

    This is the ``BAQARO_USE_SUBSAMPLE=0`` analogue of
    :func:`build_subsampled_subset`, used by the dedicated multinode full-sim
    forward run (``main_evolution_chunked.py``). "multinode" because each chunk
    is sized to evolve on its own node and the chunks recombine afterward.
    Unlike the stratified subsample (an *estimator* of the z=0 population that
    keeps only z=0-survivor-rooted trees), this reproduces ``main_evolution.py``
    EXACTLY: every resolved halo is evolved, every per-halo weight is ``1.0``,
    and the union of the ``n_chunks`` chunks is the entire resolved catalogue.

    Partition scheme — terminal-rooted components
    ---------------------------------------------
    Roots here are the **terminals** of the merger graph: halos that merge into
    nothing, ``merger_track_ids == -1`` (filtered to ``resolved_mask`` to drop
    never-resolved HBT artifacts). The merger map is a functional graph — each
    halo has at most one forward target — so following ``merger_track_ids``
    from any halo ends at exactly one terminal. Each terminal's backward closure
    is therefore one connected component, the components are disjoint, and their
    union is every resolved halo. Terminals are split round-robin over the
    stable ascending track-id ordering::

        terminals = np.flatnonzero((merger_track_ids == -1) & resolved_mask)
        this_chunk = terminals[chunk_id::n_chunks]

    This is the crucial difference from the subsample: the terminal set includes
    BOTH z=0 survivors (``death_index == -1``) AND "dead-end" lineages that
    disrupt before z=0 without merging into any survivor (``death_index >= 0,
    merger_track_ids == -1``). ``main_evolution.py`` evolves those dead-ends too
    — they host BHs that contribute to intermediate-z statistics — so a true
    full-sim equivalent MUST include them. (The subsample legitimately drops
    them because its HT estimator targets the z=0 population.)

    Why no merger ever crosses a chunk boundary
    -------------------------------------------
    Both ends of a merger ``merger_track_ids[i] = j`` lie in the same component
    (they share the terminal *i*'s chain reaches), so both fall in the same
    chunk. The closure BFS is run with the orphan-skip DISABLED (every
    progenitor is walked exactly once): no skip is needed because the functional
    graph already gives each halo a unique terminal, so there is exactly one
    inclusion path per halo and no double-counting — even for re-resolved
    orphans (a z=0-alive halo with a stale ``merger_track_ids != -1`` is simply
    a non-terminal progenitor of its target's component).

    Parameters
    ----------
    track_ids : (n_halos,) int
        Halo track IDs. Must satisfy ``track_ids[i] == i`` (contiguous
        0..n_halos-1), per the pipeline convention. Only its length is used.
    merger_track_ids : (n_halos,) int
        Forward merger map (``-1`` = terminal: disrupted or alive at the end).
    resolved_mask : (n_halos,) bool
        Halos that ever crossed the resolution threshold, i.e.
        ``mass_at_root > 0`` (HBT's LastMaxMass freezes at peak after death, so
        the z=0 mass row is positive for every halo that was ever resolved).
        Drops the never-resolved artifacts that host no BH.
    chunk_id, n_chunks : int, default 0, 1
        Which chunk to build, and how many nodes to split across.
        ``n_chunks == 1`` returns the whole catalogue in one piece (still
        weight 1.0).
    reverse_index : tuple, optional
        Precomputed CSR ``(indptr, indices)`` from
        :func:`select_merger_branches.build_reverse_index`. Pass it in to
        avoid the rebuild — share ONE reverse index across all chunks.

    Returns
    -------
    dict with the SAME key schema as :func:`build_subsampled_subset` so the
    forward-run pipeline can treat both uniformly:
      ``subset_mask`` : (n_halos,) bool — this chunk's closed halo set.
      ``weights``     : (n_halos,) float64 — 1.0 for kept, 0.0 otherwise.
      ``root_mask``   : (n_halos,) bool — this chunk's terminal roots.
      ``root_bin``    : (n_halos,) int16 — all -1 (no mass bins in multinode mode).
      ``n_per_bin_total`` : (1,) int — total terminals in the WHOLE catalogue.
      ``n_per_bin_kept``  : (1,) int — terminals in THIS chunk.
      ``weights_per_bin`` : (1,) float — 1.0.
      ``log_mass_bins``   : empty array (no bins).
      ``meta`` : dict with mode="multinode", chunk_id, n_chunks, n_roots,
                 n_total_kept, n_halos_full.
    """
    n_halos = len(track_ids)
    if len(merger_track_ids) != n_halos:
        raise ValueError("merger_track_ids must match track_ids length")
    if len(resolved_mask) != n_halos:
        raise ValueError("resolved_mask must match track_ids length")
    if not (0 <= chunk_id < n_chunks):
        raise ValueError(
            f"chunk_id={chunk_id} out of range for n_chunks={n_chunks} "
            f"(must satisfy 0 <= chunk_id < n_chunks)"
        )

    if reverse_index is None:
        reverse_index = build_reverse_index(merger_track_ids)
    indptr, indices = reverse_index

    # Terminals = halos that merge into nothing (merger_track_ids == -1).
    #
    # ⚠ Do NOT add `& resolved_mask` here. `apply_resolution_filter_combined`
    # never remaps `merger_track_ids`, so a RESOLVED halo can point at a
    # never-resolved target, and rooting only on resolved terminals would leave
    # every resolved halo whose chain dead-ends at an unresolved terminal in no
    # chunk at all. Root on ALL terminals (resolved or not), so every halo's
    # chain reaches a root, then intersect the BFS result back with
    # `resolved_mask` below so unresolved halos don't enter storage. Each
    # resolved halo lands in exactly one component (the merger graph is
    # functional), so the partition is disjoint AND complete.
    terminal_mask = (merger_track_ids == -1)
    all_roots = np.flatnonzero(terminal_mask)
    n_roots_total = len(all_roots)
    if n_roots_total == 0:
        raise ValueError("No resolved terminal halos (merger_track_ids == -1).")
    target_indices = np.ascontiguousarray(all_roots[chunk_id::n_chunks])
    if len(target_indices) == 0:
        raise ValueError(
            f"chunk {chunk_id}/{n_chunks} selected zero terminals — reduce n_chunks."
        )
    target_weights = np.ones(len(target_indices), dtype=np.float64)

    root_mask = np.zeros(n_halos, dtype=np.bool_)
    root_mask[target_indices] = True
    root_bin = np.full(n_halos, -1, dtype=np.int16)  # no mass bins in multinode mode

    selected = np.zeros(n_halos, dtype=np.bool_)
    weights = np.zeros(n_halos, dtype=np.float64)
    owner = np.full(n_halos, -1, dtype=np.int32)
    # Skip DISABLED: pass an all-False alive_at_root so the BFS walks every
    # progenitor. The functional-graph structure already guarantees each halo
    # belongs to exactly one terminal's component, so there is no double-count.
    _no_skip = np.zeros(n_halos, dtype=np.bool_)
    _bfs_propagate_weights(
        selected, weights, owner, indptr, indices, target_indices, target_weights,
        _no_skip,
    )

    # Root on ALL terminals (above) but STORE only resolved halos: unresolved
    # halos are needed as BFS *transit* nodes (a resolved halo can reach its
    # terminal only by passing through one), yet they carry zero halo mass and
    # must not consume storage.
    selected &= resolved_mask
    weights[~resolved_mask] = 0.0
    root_mask &= resolved_mask

    return {
        "subset_mask": selected,
        "weights": weights,
        "root_mask": root_mask,
        "root_bin": root_bin,
        "n_per_bin_total": np.array([n_roots_total], dtype=np.int64),
        "n_per_bin_kept": np.array([len(target_indices)], dtype=np.int64),
        "weights_per_bin": np.array([1.0], dtype=np.float64),
        "log_mass_bins": np.empty(0, dtype=np.float64),
        "meta": {
            "mode": "multinode",
            "N_b_per_bin": -1,  # sentinel: not a stratified build
            "rng_seed": -1,     # multinode mode has no per-bin RNG
            "n_roots": int(len(target_indices)),
            "n_total_kept": int(selected.sum()),
            "n_halos_full": int(n_halos),
            "chunk_id": int(chunk_id),
            "n_chunks": int(n_chunks),
        },
    }


def make_multinode_partition_tag(
    *,
    root_snap: int,
    chunk_id: int = 0,
    n_chunks: int = 1,
) -> str:
    """Filename tag for a multinode whole-catalogue (no-subsample) partition chunk.

    Distinct ``multinode_`` prefix so these never collide with stratified
    subsample tags (:func:`make_subset_tag`, which begins ``root{snap}_flatN...``).
    The version suffix carries the partition version, and ``_chunk{k}of{N}`` is
    appended only when ``n_chunks > 1`` (so a single-piece build —
    ``n_chunks == 1`` — and the recombined product share the chunk-free base
    tag ``multinode_root{snap}_v4``).

    **v4** roots the BFS on ALL terminals and intersects back to
    ``resolved_mask``, so every resolved halo lands in exactly one chunk and
    the union of chunks is the entire resolved catalogue. v3 rooted only on
    resolved terminals; v3 caches stay readable under an explicit tag
    (``BAQARO_SUBSET_TAG=multinode_root144_v3``), while auto-derivation
    resolves to v4 so a new multinode run never appends to a v3 product.

    Examples
    --------
    >>> make_multinode_partition_tag(root_snap=144, chunk_id=3, n_chunks=16)
    'multinode_root144_v4_chunk3of16'
    >>> make_multinode_partition_tag(root_snap=144)
    'multinode_root144_v4'
    """
    chunk_str = f"_chunk{chunk_id}of{n_chunks}" if n_chunks > 1 else ""
    return f"multinode_root{root_snap}_v4{chunk_str}"


def make_subset_tag(
    *,
    root_snap: int,
    N_b: Union[int, Sequence[int], np.ndarray],
    n_bins: int,
    log_M_lo: float,
    log_M_hi: float,
    seed: int,
    keep_all_above_log_M: Optional[float] = None,
    chunk_id: int = 0,
    n_chunks: int = 1,
) -> str:
    """Build a filename tag encoding the subsample config.

    Used to guarantee different (N_b, bins, seed) configs land in
    different files so reruns don't clobber prior subsets.

    The ``v3`` suffix marks the version where the closure-skip criterion is
    keyed on ``death_index == -1`` (HBT's liveness flag) joined with
    ``mass > 0`` -- a mass-only criterion would skip dead progenitors, because
    HBT freezes their LastMaxMass at peak. Mixing v1/v2/v3 outputs in one
    comparison is invalid; tag-bumping keeps cached subsets separate.

    ``keep_all_above_log_M`` is appended as ``_keepAbove{X.X}`` so subsets
    with and without the keep-all rule don't collide.

    Variable N_b: when ``N_b`` is a sequence (mass-dependent schedule), the
    flat ``flatN{N_b}`` token is replaced with ``schedN-{8hex}`` where the
    hash is computed over the int64 schedule. Different schedules collide
    only at 2^-32 probability; identical schedules reuse the cache.

    ``chunk_id`` / ``n_chunks`` append a trailing ``_chunk{k}of{N}`` token
    ONLY when ``n_chunks > 1`` (chunked multi-node training). With the
    default ``n_chunks == 1`` the tag is byte-identical to the un-chunked
    convention, so single-node caches/outputs are unaffected. The pooled
    re-sum output is written under the chunk-free base tag, which by
    construction equals the single-node name (see ``pool_chunks.py``).
    """
    keep_str = ""
    if keep_all_above_log_M is not None:
        keep_str = f"_keepAbove{keep_all_above_log_M:.1f}"

    chunk_str = f"_chunk{chunk_id}of{n_chunks}" if n_chunks > 1 else ""

    if np.isscalar(N_b):
        nb_token = f"flatN{int(N_b)}"
    else:
        arr = np.asarray(N_b, dtype=np.int64)
        if np.all(arr == arr[0]):
            nb_token = f"flatN{int(arr[0])}"
        else:
            digest = hashlib.sha1(arr.tobytes()).hexdigest()[:8]
            nb_token = f"schedN-{digest}"

    return (
        f"root{root_snap}_{nb_token}_K{n_bins}"
        f"_logM{log_M_lo:.1f}to{log_M_hi:.1f}{keep_str}_seed{seed}_v3{chunk_str}"
    )


# RNG seed for the production stratified subsample. Env-overridable via
# BAQARO_SUBSAMPLE_SEED (default 42 -> bit-identical to all existing products).
# Changing it reshuffles which halos are kept and lands in a distinct subset tag
# (seedNN), so cached subsets/training files never collide. This constant is the
# single source of truth: resolve_subsample_params()/resolve_subset_tag() (the
# consumer-side tag resolver used by plotting/inference/emulation) read it, and
# the producers (main_training/main_evolution/pool_chunks) read the SAME env var
# with the SAME default, so producer- and consumer-derived tags never drift.
SUBSAMPLE_RNG_SEED = int(os.environ.get("BAQARO_SUBSAMPLE_SEED", "42"))


def resolve_subsample_params(root_snap=None):
    """Resolve the production subsample config from env vars + the active sim's
    SimSpec geometry — the SAME resolution ``main_evolution`` / ``main_training``
    perform inline. Single source of truth so the producer-written subset tag
    and any consumer-derived default tag can never drift.

    Honoured env vars (all optional; defaults are the May-2026 benchmark winner —
    500k uniform, no keep_above):
      BAQARO_SUBSAMPLE_NB (default 500000), BAQARO_SUBSAMPLE_NB_SCHEDULE,
      BAQARO_SUBSAMPLE_LOG_M_LO / _HI / _NBINS (override the SimSpec geometry),
      BAQARO_KEEP_ALL_ABOVE (default empty = None).

    ``root_snap`` defaults to the active sim's ``max_snap`` (honours
    BAQARO_MAX_SNAP via sim_config).

    Returns a kwargs dict for ``make_subset_tag`` / ``build_subsampled_subset``:
    ``root_snap``, ``N_b``, ``n_bins``, ``log_M_lo``, ``log_M_hi``, ``seed``,
    ``keep_all_above_log_M``.
    """
    from baqaro.utils import sim_config as _sc

    if root_snap is None:
        root_snap = _sc.max_snap
    log_M_lo = float(os.environ.get("BAQARO_SUBSAMPLE_LOG_M_LO", _sc.subsample_log_M_lo))
    log_M_hi = float(os.environ.get("BAQARO_SUBSAMPLE_LOG_M_HI", _sc.subsample_log_M_hi))
    n_bins = int(os.environ.get("BAQARO_SUBSAMPLE_NBINS", _sc.subsample_n_bins))
    N_b = int(os.environ.get("BAQARO_SUBSAMPLE_NB", "500000"))

    sched = os.environ.get("BAQARO_SUBSAMPLE_NB_SCHEDULE", "").strip()
    if sched:
        bin_lo = np.linspace(log_M_lo, log_M_hi, n_bins + 1)[:-1]
        steps = sorted(
            (float(_logM_s), int(_nb_s))
            for _nb_s, _logM_s in (tok.split("@") for tok in sched.split(","))
        )
        N_b = np.full(n_bins, steps[0][1], dtype=np.int64)
        for _logM_thresh, _nb_val in steps:
            N_b[bin_lo >= _logM_thresh] = _nb_val

    keep_env = os.environ.get("BAQARO_KEEP_ALL_ABOVE", "").strip()
    keep_all_above_log_M = float(keep_env) if keep_env else None

    return {
        "root_snap": root_snap,
        "N_b": N_b,
        "n_bins": n_bins,
        "log_M_lo": log_M_lo,
        "log_M_hi": log_M_hi,
        "seed": SUBSAMPLE_RNG_SEED,
        "keep_all_above_log_M": keep_all_above_log_M,
    }


def resolve_subset_tag(root_snap=None):
    """Production-default subset tag — what ``main_evolution`` / ``main_training``
    write for the current sim + subsample env. Lets consumers (plotting,
    inference, emulation) default to the fiducial subsampled run without the
    caller having to set ``BAQARO_SUBSET_TAG`` by hand.

    For the L2800N10080 z=0 default this returns
    ``root144_flatN500000_K22_logM10.0to15.5_seed42_v3``.
    """
    return make_subset_tag(**resolve_subsample_params(root_snap))


def save_subset(path: str, subset: dict) -> None:
    """Save subset dict to ``.npz``.

    Stores all per-halo arrays plus per-bin diagnostics.  The ``meta``
    dict is flattened into individual scalar attributes for transparency.
    """
    meta = subset["meta"]
    # N_b_array is stored verbatim so a future loader can recover the
    # per-bin schedule; meta_N_b_per_bin remains the scalar form when uniform
    # (-1 sentinel when the schedule is non-uniform).
    nb_arr = np.asarray(meta.get("N_b_array", np.array([meta["N_b_per_bin"]], dtype=np.int64)), dtype=np.int64)
    # Atomic write: savez to a unique tmp .npz then os.replace, so a killed job
    # can't leave a truncated .npz that np.load later reads as a corrupt/partial
    # subset. savez_compressed appends ".npz" if absent, so the tmp ends in .npz
    # and the final target is normalised the same way.
    final = path if path.endswith(".npz") else path + ".npz"
    tmp = f"{final[:-4]}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npz"
    np.savez_compressed(
        tmp,
        subset_mask=subset["subset_mask"],
        weights=subset["weights"],
        root_mask=subset["root_mask"],
        root_bin=subset["root_bin"],
        n_per_bin_total=subset["n_per_bin_total"],
        n_per_bin_kept=subset["n_per_bin_kept"],
        weights_per_bin=subset["weights_per_bin"],
        log_mass_bins=subset["log_mass_bins"],
        meta_N_b_per_bin=np.int64(meta["N_b_per_bin"]),
        meta_N_b_array=nb_arr,
        meta_rng_seed=np.int64(meta["rng_seed"]),
        meta_n_roots=np.int64(meta["n_roots"]),
        meta_n_total_kept=np.int64(meta["n_total_kept"]),
        meta_n_halos_full=np.int64(meta["n_halos_full"]),
    )
    os.replace(tmp, final)


def load_subset(path: str) -> dict:
    """Load subset dict from ``.npz``."""
    data = np.load(path)
    meta = {
        "N_b_per_bin": int(data["meta_N_b_per_bin"]),
        "rng_seed": int(data["meta_rng_seed"]),
        "n_roots": int(data["meta_n_roots"]),
        "n_total_kept": int(data["meta_n_total_kept"]),
        "n_halos_full": int(data["meta_n_halos_full"]),
    }
    # meta_N_b_array present in v3.1+ caches (variable-N_b schedule).
    # Older caches won't have it; reconstruct a uniform array from the
    # scalar value.
    if "meta_N_b_array" in data.files:
        meta["N_b_array"] = np.asarray(data["meta_N_b_array"], dtype=np.int64)
    else:
        n_bins = len(data["log_mass_bins"]) - 1
        meta["N_b_array"] = np.full(n_bins, meta["N_b_per_bin"], dtype=np.int64)
    return {
        "subset_mask": data["subset_mask"],
        "weights": data["weights"],
        "root_mask": data["root_mask"],
        "root_bin": data["root_bin"],
        "n_per_bin_total": data["n_per_bin_total"],
        "n_per_bin_kept": data["n_per_bin_kept"],
        "weights_per_bin": data["weights_per_bin"],
        "log_mass_bins": data["log_mass_bins"],
        "meta": meta,
    }


# ============================================================================
# Eager-load cache and per-snapshot index precompute helpers
# ============================================================================
#
# Both helpers are pure functions of the on-disk halo files + the subset
# definition. They produce byte-identical outputs to the previous inline
# eager-load + per-snap-flatnonzero loops; their only purpose is to make
# those steps fast enough to run at every script launch on the 10k box.
#
# extract_subset_slice_cached:
#   First call writes a small (~15 GB at L2800N10080 maxsnap=60) .npy file
#   containing full_array[:, subset_indices]. Subsequent calls do a single
#   sequential read of that file (~30 s) instead of ~700k fancy-index
#   gathers against the 370 GB mmap'd source (~11 min seek-limited).
#
# precompute_per_snapshot_indices:
#   Subset-restricts the merger-tree arrays once, then per-snap masks
#   operate on the (n_subset,)-sized restricted arrays instead of the
#   (n_halos_full,)-sized originals. ~50x fewer ops per snap, mapping back
#   to GLOBAL indices via subset_indices[...] preserves output identity.
# ============================================================================


def extract_subset_slice_cached(
    full_path: str,
    subset_indices: np.ndarray,
    cache_path: str,
    refresh: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Return ``full_array[:, subset_indices]`` for a (n_snaps, n_halos) .npy.

    On first call (or when ``refresh=True``), reads the full array via
    mmap row-by-row, gathers the subset columns, and writes the result
    to ``cache_path`` as a fresh .npy. On subsequent calls the cache is
    read directly (single sequential pass — orders of magnitude faster
    than the seek-limited fancy-index gather against the full file).

    Sanity check: a cache is invalidated and rebuilt if its shape does
    not match ``(n_snaps_in_full, len(subset_indices))``. Callers should
    pass a cache_path that already encodes the subset tag (and ideally
    the halos token), so different subsets land in different cache files.

    Parameters
    ----------
    full_path:
        Path to the source ``halo_masses_*.npy`` / ``specific_cold_accretion_rates_*.npy``.
    subset_indices:
        1D int array of GLOBAL halo indices to keep (typically
        ``np.flatnonzero(subset_mask)``).
    cache_path:
        Where to write the eager-loaded subset slice on first call.
    refresh:
        Force rebuild even if the cache exists.
    verbose:
        Print one-line status messages.

    Returns
    -------
    np.ndarray of shape (n_snaps, len(subset_indices)) with the same
    dtype as the source .npy.
    """
    n_sub = int(len(subset_indices))

    if (not refresh) and os.path.exists(cache_path):
        t0 = time.time()
        arr = np.load(cache_path)
        # Validate BOTH dims (docstring promises (n_snaps, n_sub)). n_sub is
        # known; n_snaps is validated against the source header when the source
        # file is present (a cheap header-only mmap open — no data paged in).
        # When the source isn't staged (cache-only deployments), trust the
        # cache's row count rather than forcing a dependency on the full file.
        expected_rows = None
        if os.path.exists(full_path):
            try:
                expected_rows = int(np.load(full_path, mmap_mode="r").shape[0])
            except Exception:
                expected_rows = None
        cols_ok = arr.ndim == 2 and arr.shape[1] == n_sub
        rows_ok = expected_rows is None or arr.shape[0] == expected_rows
        if cols_ok and rows_ok:
            if verbose:
                gb = arr.nbytes / 1e9
                print(
                    f"  Loaded eager subset cache {os.path.basename(cache_path)} "
                    f"in {time.time()-t0:.1f}s ({gb:.1f} GB, shape {arr.shape})"
                )
            return arr
        if verbose:
            want_rows = expected_rows if expected_rows is not None else "n_snaps"
            print(
                f"  Eager cache {cache_path} shape {arr.shape} != "
                f"({want_rows}, {n_sub}); rebuilding."
            )

    # Build from source mmap and write cache.
    t0 = time.time()
    full = np.load(full_path, mmap_mode="r")
    out = np.empty((full.shape[0], n_sub), dtype=full.dtype)
    for row in range(full.shape[0]):
        out[row] = full[row, subset_indices]
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # Atomic write: dump to a sibling tmp .npy and rename. Avoids leaving a
    # half-written file if the run is killed mid-extract. The tmp name is made
    # UNIQUE per process (pid + uuid) so two nodes cold-building the same cache
    # on a shared filesystem can't scribble over each other's tmp and publish a
    # corrupt file via os.replace — each renames its own private tmp; last
    # writer wins with a complete file. NOTE: ``np.save`` auto-appends ``.npy``
    # when the path doesn't already end in it, so the tmp name ends in ``.npy``.
    base = cache_path[:-4] if cache_path.endswith(".npy") else cache_path
    tmp_path = f"{base}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npy"
    np.save(tmp_path, out)
    os.replace(tmp_path, cache_path)
    if verbose:
        gb = out.nbytes / 1e9
        print(
            f"  Built+saved eager subset cache {os.path.basename(cache_path)} "
            f"in {time.time()-t0:.1f}s ({gb:.1f} GB, shape {out.shape})"
        )
    return out


def precompute_per_snapshot_indices(
    min_snap: int,
    max_snap: int,
    mt_birth_index: np.ndarray,
    mt_death_index: np.ndarray,
    mask_merged: np.ndarray,
    mask_disrupted: Optional[np.ndarray] = None,
    subset_indices: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> dict:
    """Per-snapshot integer index arrays of spawning / evolving / dying halos.

    Returns a dict with four sub-dicts keyed by snapshot:

      - ``spawning``     : halos born at this snap
      - ``evolving``     : halos born before this snap and still alive
      - ``merging``      : halos dying at this snap with a merger target
      - ``lost``         : halos dying at this snap with no merger target

    All output arrays are GLOBAL halo indices (i.e. positions in the
    full ``mt_birth_index``), sorted ascending — matching what the
    previous inline ``np.flatnonzero`` produced.

    When ``subset_indices`` is given, the function works on
    subset-restricted views of the merger-tree arrays and maps results
    back through ``subset_indices``. Each per-snap scan then costs
    O(n_subset) instead of O(n_halos_full), which is the difference
    between ~1.5 B and ~30 M bool-ops per snap on L2800N10080.

    When ``subset_indices`` is None, the function reproduces the
    full-sim behaviour with no extra copies.
    """
    t0 = time.time()
    spawning = {}
    evolving = {}
    merging = {}
    lost = {}

    if subset_indices is None:
        death_eff = mt_death_index.copy()
        death_eff[death_eff == -1] = np.iinfo(death_eff.dtype).max
        for s in range(min_snap, max_snap + 1):
            spawn = (mt_birth_index == s)
            born_before = (mt_birth_index < s) & (mt_birth_index != -1)
            alive = born_before & (death_eff > s)
            dying = born_before & (mt_death_index == s)
            spawning[s] = np.flatnonzero(spawn)
            evolving[s] = np.flatnonzero(alive)
            merging[s] = np.flatnonzero(dying & mask_merged)
            if mask_disrupted is not None:
                lost[s] = np.flatnonzero(dying & mask_disrupted)
    else:
        # Subset-restrict the merger-tree arrays ONCE.
        birth_sub = mt_birth_index[subset_indices]
        death_sub = mt_death_index[subset_indices]
        merged_sub = mask_merged[subset_indices]
        disrupted_sub = (
            mask_disrupted[subset_indices] if mask_disrupted is not None else None
        )
        death_eff_sub = death_sub.copy()
        death_eff_sub[death_eff_sub == -1] = np.iinfo(death_eff_sub.dtype).max

        for s in range(min_snap, max_snap + 1):
            spawn_local = np.flatnonzero(birth_sub == s)
            born_before = (birth_sub < s) & (birth_sub != -1)
            alive_local = np.flatnonzero(born_before & (death_eff_sub > s))
            dying = born_before & (death_sub == s)
            merging_local = np.flatnonzero(dying & merged_sub)
            spawning[s] = subset_indices[spawn_local]
            evolving[s] = subset_indices[alive_local]
            merging[s] = subset_indices[merging_local]
            if disrupted_sub is not None:
                lost_local = np.flatnonzero(dying & disrupted_sub)
                lost[s] = subset_indices[lost_local]

    if verbose:
        print(
            f"  Precomputed per-snapshot indices for {max_snap - min_snap + 1} "
            f"snapshots in {time.time()-t0:.2f}s"
        )

    return {
        "spawning": spawning,
        "evolving": evolving,
        "merging": merging,
        "lost": lost,
    }
