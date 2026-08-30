"""The multinode partition must cover EVERY resolved halo.

`apply_resolution_filter_combined` never remaps `merger_track_ids`, so a
RESOLVED halo can point at a never-resolved target. The partition must
therefore root its BFS on ALL terminals, not only the resolved ones:
otherwise a resolved halo whose forward merger chain dead-ends at an
unresolved terminal belongs to no component and lands in no chunk. The
invariant locked here is that the union over chunks is exactly the resolved
catalogue.

These tests are pure-numpy (no catalogue, no sampler) and always run.
"""
import numpy as np
import pytest

from baqaro.core_functions.tree_subsample import (
    build_multinode_partition_subset,
)


def _toy_catalog():
    """A 7-halo catalogue with an unresolved-terminal chain.

    idx : merges_into : resolved?
      0  ->  1          yes     \
      1  ->  2          yes      |  chain dead-ends at 2, which is UNRESOLVED
      2  -> -1          NO      /   => 0 and 1 must still land in a chunk
      3  ->  4          yes     \
      4  -> -1          yes      |  ordinary resolved-terminal component
      5  ->  4          yes     /
      6  -> -1          yes         isolated resolved terminal
    """
    merger_track_ids = np.array([1, 2, -1, 4, -1, 4, -1], dtype=np.int64)
    resolved_mask = np.array([True, True, False, True, True, True, True])
    return merger_track_ids, resolved_mask


def _union_over_chunks(merger_track_ids, resolved_mask, n_chunks):
    """subset_mask union + a per-halo owner count, across all chunks."""
    n = len(merger_track_ids)
    union = np.zeros(n, dtype=bool)
    times_covered = np.zeros(n, dtype=int)
    for c in range(n_chunks):
        out = build_multinode_partition_subset(
            track_ids=np.arange(n, dtype=np.int64),
            merger_track_ids=merger_track_ids,
            resolved_mask=resolved_mask,
            chunk_id=c,
            n_chunks=n_chunks,
        )
        m = out["subset_mask"]
        union |= m
        times_covered += m.astype(int)
    return union, times_covered


def test_every_resolved_halo_lands_in_a_chunk():
    """COMPLETENESS: the union over chunks == the entire resolved catalogue.

    Halos 0 and 1 (resolved, but whose chain terminates at the UNRESOLVED
    halo 2) must appear in a chunk.
    """
    mt, res = _toy_catalog()
    for n_chunks in (1, 2, 3):
        union, _ = _union_over_chunks(mt, res, n_chunks)
        missing = np.flatnonzero(res & ~union)
        assert missing.size == 0, (
            f"n_chunks={n_chunks}: resolved halos {missing.tolist()} landed in NO "
            f"chunk. Halos 0/1 chain into the UNRESOLVED terminal 2, so a "
            f"terminal-only union drops them."
        )
        # ...and the union must be EXACTLY the resolved set, no unresolved halos.
        assert np.array_equal(union, res), (
            "union must equal resolved_mask exactly — unresolved halos are BFS "
            "transit nodes only and must not enter storage."
        )


def test_partition_is_disjoint():
    """DISJOINTNESS: no resolved halo is claimed by two chunks (no double-count)."""
    mt, res = _toy_catalog()
    for n_chunks in (2, 3):
        _, times = _union_over_chunks(mt, res, n_chunks)
        dup = np.flatnonzero(times > 1)
        assert dup.size == 0, (
            f"n_chunks={n_chunks}: halos {dup.tolist()} claimed by >1 chunk "
            f"(counts={times[dup].tolist()}) — would double-count their mass."
        )


def test_unresolved_halos_carry_zero_weight():
    """Unresolved transit nodes must never carry weight into the statistics."""
    mt, res = _toy_catalog()
    out = build_multinode_partition_subset(
        track_ids=np.arange(len(mt), dtype=np.int64),
        merger_track_ids=mt, resolved_mask=res, chunk_id=0, n_chunks=1,
    )
    assert np.all(out["weights"][~res] == 0.0)
    assert not out["subset_mask"][~res].any()
    # Every resolved halo kept in a 1-chunk build carries the full weight 1.0.
    assert np.all(out["weights"][res] == 1.0)


def test_all_resolved_terminals_case_unchanged():
    """Regression: when no unresolved terminal exists, behaviour is unchanged."""
    mt = np.array([1, -1, 3, -1], dtype=np.int64)
    res = np.array([True, True, True, True])
    union, times = _union_over_chunks(mt, res, 2)
    assert np.array_equal(union, res)
    assert np.all(times == 1)
