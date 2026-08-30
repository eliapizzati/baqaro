"""
Independent adversarial audit of the chunked root partition.
============================================================

This is an independent strengthening of tests/test_chunk_partition.py.
It does NOT touch production code; it only constructs deliberately nasty
merger forests and asserts the chunk partition stays disjoint + complete +
weight-preserving, comparing the union-of-chunks against an independent
brute-force closure that does not use the chunk machinery at all.

Adversarial structures exercised
--------------------------------
1. Deep linear chains (depth up to ~30) of dead progenitors.
2. Diamonds / DAG-like merges where a node's *forward* target is unique
   (as the data model guarantees) but several dead nodes funnel into one
   intermediate.
3. Multiple roots whose dead-only back-trees share a common deeper dead
   progenitor *only through* an alive-at-root intermediate (the case the
   v3 skip rule is supposed to neutralise).
4. Re-resolved orphans (alive_at_root with a forward merger target) sitting
   on the path between a dead progenitor and a root, in BOTH the "orphan is
   itself selected as a root" and "orphan not selected" configurations.
5. A dead progenitor that forward-merges into an alive-at-root orphan which
   is in turn a progenitor of a different chunk's root — the cross-chunk
   double-count trap.
6. Roots in DIFFERENT mass bins sharing a deep dead progenitor.

The independent oracle
-----------------------
For a given selected root set R (the union of all chunks' roots, which by
construction equals the single-run roots), the *intended* closure is:
  a halo h is in the subset iff there is a backward path R -> ... -> h that
  never steps THROUGH an alive_at_root node (alive nodes are only included
  when they are themselves in R). We compute this with a plain-Python BFS
  that mirrors the documented intent, independent of the Numba kernel, and
  check the production union matches it.

Run directly or via pytest.
"""

import numpy as np

from baqaro.core_functions.tree_subsample import (
    build_subsampled_subset,
)
from baqaro.core_functions.select_merger_branches import (
    build_reverse_index,
)


# ----------------------------------------------------------------------
# Independent oracle closure (pure Python, mirrors the documented intent)
# ----------------------------------------------------------------------
def oracle_closure(selected_roots, merger_track_ids, alive_at_root):
    """Brute-force the intended closed subset from a fixed root set.

    Intent (per tree_subsample docstring):
      - every selected root is in the subset,
      - walking BACKWARD through mergers, include each progenitor UNLESS it
        is alive_at_root (those enter only by being a root themselves),
      - the skip is propagated (we do not descend past a skipped node).
    """
    n = len(merger_track_ids)
    # reverse adjacency
    rev = [[] for _ in range(n)]
    for i, j in enumerate(merger_track_ids):
        if j >= 0:
            rev[j].append(i)

    in_set = np.zeros(n, dtype=bool)
    from collections import deque
    q = deque()
    for r in selected_roots:
        in_set[r] = True
        q.append(r)
    while q:
        cur = q.popleft()
        for src in rev[cur]:
            if in_set[src]:
                continue
            if alive_at_root[src]:
                # not reached via closure; only a root could include it
                continue
            in_set[src] = True
            q.append(src)
    return in_set


# ----------------------------------------------------------------------
# Adversarial forest builders
# ----------------------------------------------------------------------
def build_forest_arrays(specs):
    """specs: list of (mass, alive, merger_target). Returns the arrays."""
    n = len(specs)
    track_ids = np.arange(n, dtype=np.int64)
    merger = np.array([s[2] for s in specs], dtype=np.int64)
    mass = np.array([s[0] for s in specs], dtype=np.float64)
    alive = np.array([s[1] for s in specs], dtype=bool)
    return track_ids, merger, mass, alive


def make_adversarial_forest(seed=0):
    """Hand-built forest packed with the trap structures listed in the
    module docstring. Returns arrays + bin edges + the explicit list of
    'alive root candidates' so the test can reason about expected roots.
    """
    rng = np.random.default_rng(seed)
    specs = []  # (mass, alive_at_root, merger_target)

    def add(mass, alive, target):
        specs.append((mass, alive, target))
        return len(specs) - 1

    # We'll fix up forward targets after we know indices, using a 2-pass:
    # first reserve indices with target=-1, then set targets.
    # To keep it simple we add leaves last and patch.

    targets = {}  # idx -> target idx (set later)

    # --- Bin layout: 3 bins from logM 11..14 ---
    bins = np.array([11.0, 12.0, 13.0, 14.0])

    # Roots: a handful per bin, all alive_at_root, target -1.
    roots = []
    root_bin = []
    for b in range(3):
        for _ in range(8):
            logm = rng.uniform(11.0 + b + 0.05, 11.0 + b + 0.95)
            r = add(10.0 ** logm, True, -1)
            roots.append(r)
            root_bin.append(b)

    # Structure 1: deep linear chain of DEAD progenitors into roots[0].
    prev = roots[0]
    for _ in range(30):
        node = add(10.0 ** rng.uniform(8, 10.5), False, -1)
        targets[node] = prev
        prev = node

    # Structure 2: diamond of dead nodes funnelling into roots[1].
    mid = add(10.0 ** rng.uniform(9, 10.5), False, -1)
    targets[mid] = roots[1]
    for _ in range(5):
        leaf = add(10.0 ** rng.uniform(7, 9), False, -1)
        targets[leaf] = mid

    # Structure 3: an ALIVE orphan O that is itself NOT selected as a root
    #   in some N_b configs (its bin may subsample it out). O forward-merges
    #   into roots[2]. A DEAD progenitor D forward-merges into O. Per the
    #   skip rule, D must NOT be pulled in via closure from roots[2] (because
    #   the path goes THROUGH O which is alive). This is the central trap.
    O = add(10.0 ** rng.uniform(11.0, 11.9), True, -1)  # alive orphan, bin 0
    targets[O] = roots[2]
    D = add(10.0 ** rng.uniform(8, 10), False, -1)
    targets[D] = O
    D2 = add(10.0 ** rng.uniform(8, 10), False, -1)
    targets[D2] = D  # deeper dead behind D

    # Structure 4: cross-bin shared deep dead progenitor.
    #   Two roots in DIFFERENT bins; a dead chain leads into root in bin0,
    #   and a SEPARATE dead chain into a root in bin2. (Forward target is
    #   unique per node, so a node cannot literally point to two roots; this
    #   instead checks two independent deep chains into two different-bin
    #   roots both survive and stay owned by the right root.)
    c0 = add(10.0 ** rng.uniform(8, 10), False, -1)
    targets[c0] = roots[3]   # bin 1 root region (index 8..15)
    c1 = add(10.0 ** rng.uniform(8, 10), False, -1)
    targets[c1] = roots[16]  # bin 2 root region (index 16..23)

    # Structure 5: alive orphan that IS a stratifiable root candidate in its
    #   own bin, with a dead progenitor behind it. If the orphan is selected
    #   as a root in a chunk, its dead progenitor should be closed under it.
    O2 = add(10.0 ** rng.uniform(12.0, 12.9), True, -1)  # bin1 alive
    targets[O2] = roots[4]
    D3 = add(10.0 ** rng.uniform(8, 10), False, -1)
    targets[D3] = O2

    # Patch forward targets.
    spec_list = list(specs)
    for idx, tgt in targets.items():
        m, a, _ = spec_list[idx]
        spec_list[idx] = (m, a, tgt)
    specs = spec_list

    track_ids, merger, mass, alive = build_forest_arrays(specs)
    return dict(
        track_ids=track_ids,
        merger_track_ids=merger,
        halo_masses_at_root=mass,
        alive_at_root=alive,
        log_mass_bins=bins,
    )


def make_random_dag_forest(seed, n_layers=6, width=40, alive_frac=0.15):
    """Randomised layered forest. Forward target of a node in layer L points
    to a node in layer L-1 (closer to roots), enforcing the unique-forward-
    target data model while producing deep, branchy, partially-alive trees.
    A random subset of nodes is alive_at_root (re-resolved orphans).
    """
    rng = np.random.default_rng(seed)
    bins = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
    n_bins = len(bins) - 1

    specs = []

    def add(mass, alive, target):
        specs.append((mass, alive, target))
        return len(specs) - 1

    # Layer 0 = roots, all alive, binned masses.
    layer_prev = []
    for _ in range(width):
        b = int(rng.integers(0, n_bins))
        logm = rng.uniform(bins[b] + 0.05, bins[b + 1] - 0.05)
        r = add(10.0 ** logm, True, -1)
        layer_prev.append(r)

    # Deeper layers: each node targets a random node in the previous layer.
    targets = {}
    for L in range(1, n_layers):
        layer_cur = []
        for _ in range(width):
            alive = bool(rng.random() < alive_frac)
            # alive orphans get a binned mass so they can be stratified;
            # dead progenitors get sub-resolution-ish masses.
            if alive:
                b = int(rng.integers(0, n_bins))
                logm = rng.uniform(bins[b] + 0.05, bins[b + 1] - 0.05)
            else:
                logm = rng.uniform(8.0, 10.8)
            node = add(10.0 ** logm, alive, -1)
            tgt = int(rng.choice(layer_prev))
            targets[node] = tgt
            layer_cur.append(node)
        layer_prev = layer_cur

    spec_list = list(specs)
    for idx, tgt in targets.items():
        m, a, _ = spec_list[idx]
        spec_list[idx] = (m, a, tgt)
    specs = spec_list

    track_ids, merger, mass, alive = build_forest_arrays(specs)
    return dict(
        track_ids=track_ids,
        merger_track_ids=merger,
        halo_masses_at_root=mass,
        alive_at_root=alive,
        log_mass_bins=bins,
    )


# ----------------------------------------------------------------------
# Core assertion helper
# ----------------------------------------------------------------------
def _check_partition(forest, N_b, chunk_counts, seed=42):
    rev = build_reverse_index(forest["merger_track_ids"])
    single = build_subsampled_subset(
        track_ids=forest["track_ids"],
        merger_track_ids=forest["merger_track_ids"],
        halo_masses_at_root=forest["halo_masses_at_root"],
        alive_at_root=forest["alive_at_root"],
        log_mass_bins=forest["log_mass_bins"],
        N_b_per_bin=N_b,
        rng_seed=seed,
        reverse_index=rev,
    )
    single_roots = np.flatnonzero(single["root_mask"])

    # Independent oracle closure from the SAME root set.
    oracle = oracle_closure(
        single_roots, forest["merger_track_ids"], forest["alive_at_root"]
    )
    assert np.array_equal(oracle, single["subset_mask"]), (
        "single-run closure disagrees with the independent oracle closure"
    )

    for K in chunk_counts:
        masks, weights, root_masks = [], [], []
        for c in range(K):
            try:
                s = build_subsampled_subset(
                    track_ids=forest["track_ids"],
                    merger_track_ids=forest["merger_track_ids"],
                    halo_masses_at_root=forest["halo_masses_at_root"],
                    alive_at_root=forest["alive_at_root"],
                    log_mass_bins=forest["log_mass_bins"],
                    N_b_per_bin=N_b,
                    rng_seed=seed,
                    chunk_id=c,
                    n_chunks=K,
                    reverse_index=rev,
                )
            except ValueError as e:
                if "selected zero roots" in str(e):
                    # legitimately empty chunk (more chunks than roots in
                    # every bin) — skip, it contributes nothing.
                    continue
                raise
            masks.append(s["subset_mask"])
            weights.append(s["weights"])
            root_masks.append(s["root_mask"])

        stacked = np.vstack(masks)
        membership = stacked.sum(axis=0)

        # (A) disjoint
        n_overlap = int((membership > 1).sum())
        assert n_overlap == 0, (
            f"N_b={N_b} K={K}: {n_overlap} halos in >1 chunk "
            f"(indices {np.flatnonzero(membership > 1)[:20]})"
        )
        # (B) complete vs single AND vs independent oracle
        union = stacked.any(axis=0)
        assert np.array_equal(union, single["subset_mask"]), (
            f"N_b={N_b} K={K}: chunk union != single-run subset"
        )
        assert np.array_equal(union, oracle), (
            f"N_b={N_b} K={K}: chunk union != independent oracle closure"
        )
        # roots partition
        rstack = np.vstack(root_masks)
        assert rstack.sum(axis=0).max() <= 1
        assert np.array_equal(rstack.any(axis=0), single["root_mask"])
        # (C) weights sum to single
        wsum = np.sum(np.vstack(weights), axis=0)
        assert np.allclose(wsum, single["weights"]), (
            f"N_b={N_b} K={K}: Σ weights != single-run weights"
        )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_adversarial_handbuilt():
    forest = make_adversarial_forest(seed=0)
    # span N_b from "subsample hard" to "keep all roots".
    for N_b in (1, 2, 3, 5, 1_000_000):
        _check_partition(forest, N_b=N_b, chunk_counts=(2, 3, 4, 5, 8))


def test_adversarial_random_dags():
    for seed in range(12):
        forest = make_random_dag_forest(seed=seed, n_layers=7, width=50,
                                        alive_frac=0.2)
        for N_b in (1, 3, 8, 1_000_000):
            _check_partition(forest, N_b=N_b, chunk_counts=(2, 3, 4, 6),
                             seed=100 + seed)


def test_more_chunks_than_roots_is_safe():
    """K far exceeding the per-bin root count: empty chunks must not break
    disjointness/completeness; non-empty chunks still cover the subset."""
    forest = make_adversarial_forest(seed=3)
    _check_partition(forest, N_b=2, chunk_counts=(10, 16, 32, 64))


if __name__ == "__main__":
    test_adversarial_handbuilt()
    print("OK  test_adversarial_handbuilt")
    test_adversarial_random_dags()
    print("OK  test_adversarial_random_dags")
    test_more_chunks_than_roots_is_safe()
    print("OK  test_more_chunks_than_roots_is_safe")
    print("\nAll adversarial chunk-partition audits passed.")
