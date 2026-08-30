"""
Closed merger-tree subset selection.
=====================================

Given a set of "target" halos, find **all progenitors** that ultimately
merge into them.  This produces a "closed" subset of the full halo
catalogue — closed in the sense that every merger source and destination
is included, so BH mergers remain self-consistent after subsetting.

The merger tree is stored as a forward mapping::

    merger_track_ids[i] = j   means halo i merges into halo j
    merger_track_ids[i] = -1  means halo i is disrupted or still alive

To walk the tree *backwards* (from targets to progenitors) we build a
**reverse index** in CSR (Compressed Sparse Row) format:

    indptr, indices = build_reverse_index(merger_track_ids)

    # All halos that merge into halo j:
    sources = indices[indptr[j] : indptr[j+1]]

This is the expensive step (~2 s for 200 M halos) but only needs to be
done **once** per merger tree — the result can be cached to disk as an
``.npz`` file and reloaded in milliseconds.

With the reverse index in hand, a BFS (breadth-first search) from the
target halos walks all progenitor chains in sub-millisecond time.

Public API
----------
build_reverse_index
    One-time construction of the CSR reverse index.
get_merger_branch_mask
    Fast boolean mask of all progenitors (+ optionally the targets).
get_merger_branch_info
    Same as above but also returns merger depth and per-target counts.

Performance
-----------
Optimised for catalogues with 100 M+ halos:

* ``build_reverse_index``: vectorised NumPy (argsort + bincount).
* BFS kernels: Numba ``@njit(cache=True)`` — compiled once, reused.
* Typical query after warmup: <1 ms for ~50 targets in a 200 M halo tree.
"""

import numpy as np
from typing import List, Union, Tuple, Optional
from numba import njit


def build_reverse_index(merger_track_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a CSR reverse index: destination → list of sources.

    The forward merger tree says "halo *i* merges into halo *j*".  The
    reverse index answers the opposite question: "which halos merge into
    halo *j*?" — needed for the backwards BFS traversal.

    The result is stored in Compressed Sparse Row (CSR) format, the same
    layout used by ``scipy.sparse.csr_matrix``::

        indptr[j]   …  indptr[j+1]   mark the slice of ``indices``
        indices[indptr[j] : indptr[j+1]]  =  source halo IDs that merge into j

    This is the expensive step (~2 s for 200 M halos) but only needs to
    be done **once** per merger tree.  Save the result with
    ``np.savez(path, indptr=indptr, indices=indices)`` and reload later.

    Parameters
    ----------
    merger_track_ids : np.ndarray, shape (n_halos,)
        Forward merger mapping.  ``merger_track_ids[i] = j`` means halo
        *i* merges into halo *j*.  A value of ``-1`` means the halo is
        disrupted or still alive (no merger).

    Returns
    -------
    indptr : np.ndarray, shape (n_halos + 1,)
        CSR row-pointer array.
    indices : np.ndarray
        CSR column-index array containing source halo IDs.
    """
    n_halos = len(merger_track_ids)
    merger_track_ids = np.asarray(merger_track_ids, dtype=np.int64)

    # 1. Keep only valid mergers (dest >= 0)
    valid_mask = merger_track_ids >= 0
    valid_sources = np.where(valid_mask)[0].astype(np.int64)
    valid_dests = merger_track_ids[valid_mask]

    # 2. Sort by destination so sources for the same dest are contiguous
    sort_idx = np.argsort(valid_dests, kind='stable')
    sorted_dests = valid_dests[sort_idx]
    sorted_sources = valid_sources[sort_idx]

    # 3. Build indptr via bincount + cumsum (vectorised, no Python loop)
    counts = np.bincount(sorted_dests, minlength=n_halos)
    indptr = np.zeros(n_halos + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    # sorted_sources is already in CSR order — no scatter step needed
    return indptr, sorted_sources


@njit(cache=True)
def _propagate_selection_bfs(selected, indptr, indices, target_indices):
    """BFS walk from targets through the reverse index, marking all progenitors.

    Starting from the already-marked *target_indices*, this traverses the
    CSR reverse index level by level.  Every newly discovered source halo
    is marked in *selected* and appended to the queue.  The queue is a
    simple ring-buffer carved out of a pre-allocated array (no Python list
    allocation inside the Numba kernel).

    Parameters
    ----------
    selected : np.ndarray, shape (n_halos,), dtype bool
        Boolean mask — targets must already be True on entry.  Modified
        in-place; on exit, all progenitors are also True.
    indptr, indices : np.ndarray
        CSR reverse index from :func:`build_reverse_index`.
    target_indices : np.ndarray
        Integer indices of the target halos (BFS starting points).

    Returns
    -------
    selected : np.ndarray
        Same array, modified in-place.
    """
    n_halos = len(selected)
    # Pre-allocated queue (worst case: every halo is reachable)
    queue = np.empty(n_halos, dtype=np.int64)
    queue_start = 0
    queue_end = len(target_indices)
    queue[:queue_end] = target_indices

    while queue_start < queue_end:
        current_idx = queue[queue_start]
        queue_start += 1

        # Look up all halos that merge into current_idx
        start = indptr[current_idx]
        end = indptr[current_idx + 1]

        for j in range(start, end):
            src_idx = indices[j]
            if not selected[src_idx]:
                selected[src_idx] = True
                queue[queue_end] = src_idx
                queue_end += 1

    return selected


@njit(cache=True)
def _bfs_with_target_tracking(indptr, indices, target_indices, n_halos):
    """BFS that also records merger depth and which target "owns" each progenitor.

    Like :func:`_propagate_selection_bfs` but additionally tracks:

    * **depth** — how many merger steps separate each progenitor from its
      target (0 for targets themselves, 1 for direct mergers, etc.).
    * **owner** — index into *target_indices* identifying which target
      tree each progenitor belongs to.  When two target trees share a
      progenitor, the first one discovered (BFS order) wins.

    Parameters
    ----------
    indptr, indices : np.ndarray
        CSR reverse index.
    target_indices : np.ndarray
        Integer indices of the target halos.
    n_halos : int
        Total number of halos in the catalogue.

    Returns
    -------
    selected : np.ndarray, shape (n_halos,), dtype bool
        True for all targets and their progenitors.
    depth : np.ndarray, shape (n_halos,), dtype int32
        Merger depth (-1 if not selected, 0 for targets, 1+ for progenitors).
    n_progenitors : np.ndarray, shape (n_targets,), dtype int64
        Number of progenitors (depth > 0) belonging to each target.
    """
    n_targets = len(target_indices)

    selected = np.zeros(n_halos, dtype=np.bool_)
    depth = np.full(n_halos, -1, dtype=np.int32)
    owner = np.full(n_halos, -1, dtype=np.int32)

    queue = np.empty(n_halos, dtype=np.int64)
    queue_start = 0
    queue_end = n_targets

    # Seed the BFS with target halos
    for i in range(n_targets):
        tid = target_indices[i]
        queue[i] = tid
        selected[tid] = True
        depth[tid] = 0
        owner[tid] = i

    # BFS traversal
    while queue_start < queue_end:
        current_idx = queue[queue_start]
        current_depth = depth[current_idx]
        current_owner = owner[current_idx]
        queue_start += 1

        start = indptr[current_idx]
        end = indptr[current_idx + 1]

        for j in range(start, end):
            src_idx = indices[j]
            if not selected[src_idx]:
                selected[src_idx] = True
                depth[src_idx] = current_depth + 1
                owner[src_idx] = current_owner
                queue[queue_end] = src_idx
                queue_end += 1

    # Count progenitors per target (exclude the targets themselves)
    n_progenitors = np.zeros(n_targets, dtype=np.int64)
    for i in range(n_halos):
        if owner[i] >= 0 and depth[i] > 0:
            n_progenitors[owner[i]] += 1

    return selected, depth, n_progenitors


def get_merger_branch_mask(
    target_track_ids: Union[List[int], np.ndarray],
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    include_targets: bool = True,
    reverse_index: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Boolean mask selecting all progenitors of the given target halos.

    Walks the merger tree backwards via BFS: if A → B → C (target), both
    A and B are selected.  The result is a "closed" subset where every
    merger's source and destination are both included.

    Parameters
    ----------
    target_track_ids : array-like
        Track IDs of the target halos to trace back from.
    track_ids : np.ndarray, shape (n_halos,)
        All track IDs.  **Must** be contiguous 0 … N-1 (i.e.
        ``track_ids[i] == i``), so track IDs double as array indices.
    merger_track_ids : np.ndarray, shape (n_halos,)
        Forward merger mapping (``-1`` = disrupted / alive).
    include_targets : bool, default True
        If False the target halos themselves are excluded from the mask
        (only their progenitors are kept).
    reverse_index : tuple of (indptr, indices), optional
        Pre-built CSR reverse index from :func:`build_reverse_index`.
        Pass this to skip the expensive rebuild on every call.

    Returns
    -------
    mask : np.ndarray, shape (n_halos,), dtype bool

    Examples
    --------
    >>> # Build reverse index ONCE (expensive, ~2 s for 200 M halos)
    >>> reverse_index = build_reverse_index(merger_track_ids)
    >>>
    >>> # Then query many times (fast, <1 ms each)
    >>> mask = get_merger_branch_mask([100, 200], track_ids,
    ...                              merger_track_ids,
    ...                              reverse_index=reverse_index)
    """
    target_track_ids = np.asarray(target_track_ids, dtype=np.int64)
    n_halos = len(track_ids)

    selected = np.zeros(n_halos, dtype=np.bool_)

    # track_ids[i] == i  →  target track IDs *are* array indices
    target_indices = target_track_ids
    selected[target_indices] = True

    if reverse_index is None:
        indptr, indices = build_reverse_index(merger_track_ids)
    else:
        indptr, indices = reverse_index

    selected = _propagate_selection_bfs(selected, indptr, indices, target_indices)

    if not include_targets:
        selected[target_indices] = False

    return selected


def get_merger_branch_info(
    target_track_ids: Union[List[int], np.ndarray],
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    reverse_index: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> dict:
    """Like :func:`get_merger_branch_mask` but returns richer diagnostics.

    A single BFS pass computes the selection mask, merger depth of every
    progenitor, and how many progenitors belong to each target.

    Parameters
    ----------
    target_track_ids : array-like
        Track IDs of the target halos.
    track_ids : np.ndarray, shape (n_halos,)
        All track IDs (must be contiguous 0 … N-1).
    merger_track_ids : np.ndarray, shape (n_halos,)
        Forward merger mapping.
    reverse_index : tuple of (indptr, indices), optional
        Pre-built CSR reverse index.

    Returns
    -------
    info : dict
        ``mask``
            Boolean mask, shape (n_halos,).
        ``selected_indices``
            Integer indices where ``mask`` is True.
        ``n_progenitors_per_target``
            1-D array, length = number of targets.  Counts progenitors
            only (depth > 0), excluding the target itself.
        ``merger_depth``
            Per-halo depth array (-1 = not selected, 0 = target,
            1 = direct merger into target, 2 = merger-of-merger, …).
        ``max_depth``
            Maximum depth across all selected halos.
        ``reverse_index``
            The ``(indptr, indices)`` tuple, for reuse in subsequent calls.
    """
    target_track_ids = np.asarray(target_track_ids, dtype=np.int64)
    n_halos = len(track_ids)
    
    # Build or use cached reverse index
    if reverse_index is None:
        indptr, indices = build_reverse_index(merger_track_ids)
    else:
        indptr, indices = reverse_index
    
    # Single BFS pass that computes everything
    selected, depth, n_progenitors = _bfs_with_target_tracking(
        indptr, indices, target_track_ids, n_halos
    )
    
    selected_indices = np.where(selected)[0]
    max_depth = int(np.max(depth)) if np.any(selected) else 0
    
    return {
        'mask': selected,
        'selected_indices': selected_indices,
        'n_progenitors_per_target': n_progenitors,
        'merger_depth': depth,
        'max_depth': max_depth,
        'reverse_index': (indptr, indices),
    }


# =============================================================================
# EXAMPLE / BENCHMARK
# =============================================================================
# Run as a script (`python select_merger_branches.py`) to benchmark the
# reverse-index build and BFS query on real merger tree data.
if __name__ == "__main__":
    import os
    import time
    from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
    from baqaro.utils.my_dir import get_output_path
    
    # Configuration - adjust these to match your setup
    source_dir = "machine_igm"
    max_snap = 38
    nbound_threshold = 40
    halo_filtering_mode = "global"
    from baqaro.utils.sim_config import tdyn_fraction_default
    
    path_out = get_output_path(source=source_dir)
    
    name_file_halos = "maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
        max_snap, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
    )
    name_folder_merger_trees = "merger_trees_{}".format(name_file_halos)
    path_file_trees = os.path.join(path_out, "halo_histories", name_folder_merger_trees)
    path_reverse = os.path.join(path_out, "halo_histories", f"reverse_index_{name_file_halos}.npz")
    
    # Load merger trees
    print("Loading merger trees from", path_file_trees)
    merger_trees = MergerTreeLoader(path_file_trees)
    
    track_ids = np.array(merger_trees.track_ids)
    merger_track_ids = np.array(merger_trees.merger_track_ids)
    
    print(f"\nTotal halos in tree: {len(track_ids):,}")
    
    # Example: Select some target halos
    target_track_ids = track_ids[:50]  # Replace with your actual targets
    print(f"Number of target halos: {len(target_track_ids)}")
    
    # ==========================================================================
    # KEY OPTIMIZATION: Build reverse index ONCE, save to file, reuse for all queries
    # ==========================================================================
    print("\n" + "="*60)
    if os.path.exists(path_reverse):
        print(f"Loading cached reverse index from {path_reverse}...")
        t0 = time.perf_counter()
        data = np.load(path_reverse)
        reverse_index = (data['indptr'], data['indices'])
        t_index = time.perf_counter() - t0
        print(f"  Reverse index load time: {t_index:.3f}s")
    else:
        print("Building reverse index (one-time cost)...")
        t0 = time.perf_counter()
        reverse_index = build_reverse_index(merger_track_ids)
        t_index = time.perf_counter() - t0
        print(f"  Reverse index build time: {t_index:.3f}s")
        print("Saving reverse index for next time...")
        np.savez(path_reverse, indptr=reverse_index[0], indices=reverse_index[1])
        print(f"  Saved to {path_reverse}")
    print(f"  Index memory: {(reverse_index[0].nbytes + reverse_index[1].nbytes) / 1e9:.2f} GB")
    print("="*60)
    
    # Warmup JIT (BFS function only - much faster now)
    print("\nWarming up Numba JIT...")
    t0 = time.perf_counter()
    _ = get_merger_branch_mask(target_track_ids[:1], track_ids, merger_track_ids,
                                reverse_index=reverse_index)
    print(f"  JIT warmup: {time.perf_counter() - t0:.4f}s")
    
    # Benchmark with cached index
    print("\nBenchmarking get_merger_branch_mask (with cached index)...")
    t0 = time.perf_counter()
    mask = get_merger_branch_mask(target_track_ids, track_ids, merger_track_ids,
                                   reverse_index=reverse_index)
    t1 = time.perf_counter()
    print(f"  Time: {t1 - t0:.6f}s  <-- This is blazing fast!")
    print(f"  Selected halos: {np.sum(mask):,}")
    
    # Multiple queries to show the benefit
    print("\nBenchmarking 100 queries with different targets...")
    t0 = time.perf_counter()
    for i in range(100):
        targets = track_ids[i*10:(i+1)*10]
        _ = get_merger_branch_mask(targets, track_ids, merger_track_ids,
                                    reverse_index=reverse_index)
    t1 = time.perf_counter()
    print(f"  100 queries in: {t1 - t0:.4f}s ({(t1-t0)/100*1000:.3f}ms per query)")
    
    # Benchmark info function
    print("\nBenchmarking get_merger_branch_info (with cached index)...")
    t0 = time.perf_counter()
    info = get_merger_branch_info(target_track_ids, track_ids, merger_track_ids,
                                   reverse_index=reverse_index)
    t1 = time.perf_counter()
    print(f"  Time: {t1 - t0:.4f}s")
    print(f"  Max merger depth: {info['max_depth']}")
    print(f"  Progenitors per target (first 5): {info['n_progenitors_per_target'][:5]}")

