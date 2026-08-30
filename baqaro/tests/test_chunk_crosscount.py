"""
Targeted cross-chunk double-count probe.
========================================

The disjointness worry for chunking is: can a single DEAD progenitor be
pulled into two different chunks' closures? The data model gives every halo
a UNIQUE forward merger target, so its forward path is a single chain. A
dead progenitor p is therefore "owned" by whichever root/alive node its
forward chain first reaches. This test pins down the boundary cases:

  case A: p -> r1 directly, with r1 and another root r2 in the SAME bin.
          If chunking splits r1,r2 into different chunks, p must follow r1
          only (never appears in r2's chunk).
  case B: p -> O (alive orphan) -> r. With the v3 skip rule, p is NEVER
          pulled in via closure (regardless of which chunk owns r), and is
          included iff O is itself stratified as a root in some chunk; then
          p is closed under O in THAT chunk only.
  case C: a long dead chain whose head merges into an alive orphan that is
          a root candidate in a DIFFERENT bin than the deep root — ensures
          the per-bin round-robin never lets two chunks both claim the chain.

We additionally verify the round-robin actually DOES split same-bin roots
across chunks (otherwise the disjointness test would be vacuous).
"""

import numpy as np

from baqaro.core_functions.tree_subsample import (
    build_subsampled_subset,
)
from baqaro.core_functions.select_merger_branches import (
    build_reverse_index,
)


def _arrays(specs, bins):
    n = len(specs)
    return dict(
        track_ids=np.arange(n, dtype=np.int64),
        merger_track_ids=np.array([s[2] for s in specs], dtype=np.int64),
        halo_masses_at_root=np.array([s[0] for s in specs], dtype=np.float64),
        alive_at_root=np.array([s[1] for s in specs], dtype=bool),
        log_mass_bins=bins,
    )


def _run(forest, N_b, chunk_id, n_chunks, rev, seed=42):
    return build_subsampled_subset(
        track_ids=forest["track_ids"],
        merger_track_ids=forest["merger_track_ids"],
        halo_masses_at_root=forest["halo_masses_at_root"],
        alive_at_root=forest["alive_at_root"],
        log_mass_bins=forest["log_mass_bins"],
        N_b_per_bin=N_b,
        rng_seed=seed,
        chunk_id=chunk_id,
        n_chunks=n_chunks,
        reverse_index=rev,
    )


def test_same_bin_split_then_dead_prog_follows_one_root():
    """Many roots in ONE bin (so round-robin definitely splits them), each
    with a private dead progenitor. With N_b large enough to keep ALL roots,
    every chunk's closure is disjoint and the dead progenitor follows its
    own root. We assert the split is real (>=2 chunks get roots)."""
    bins = np.array([12.0, 13.0])
    specs = []
    roots = []
    for k in range(20):
        logm = 12.1 + 0.04 * k
        ridx = len(specs)
        specs.append((10.0 ** logm, True, -1))   # root, alive
        roots.append(ridx)
        # one dead progenitor per root, target patched below
        specs.append((10.0 ** (9.0 + 0.01 * k), False, ridx))
    forest = _arrays(specs, bins)
    rev = build_reverse_index(forest["merger_track_ids"])

    single = _run(forest, 10_000, 0, 1, rev)
    K = 4
    masks = [_run(forest, 10_000, c, K, rev)["subset_mask"] for c in range(K)]
    root_masks = [_run(forest, 10_000, c, K, rev)["root_mask"] for c in range(K)]

    stacked = np.vstack(masks)
    assert stacked.sum(axis=0).max() <= 1, "dead progenitor double-counted!"
    assert np.array_equal(stacked.any(axis=0), single["subset_mask"])
    # split is real: every chunk got some roots (20 roots / 4 chunks = 5 each)
    per_chunk_roots = [int(rm.sum()) for rm in root_masks]
    assert min(per_chunk_roots) >= 1, per_chunk_roots
    assert sum(per_chunk_roots) == 20


def test_dead_prog_behind_alive_orphan_never_closure_walked():
    """p -> O(alive) -> r. p must NEVER be selected via closure. It is only
    selected if O is stratified as a root. We force O out (N_b keeps fewer
    than the alive count in O's bin is impossible with 1 alive there, so we
    instead place O in a bin with many alive competitors and N_b=1 so O is
    very likely dropped — then check p is absent whenever O is absent)."""
    # bin0: r (the deep root). bin1: 12 alive orphans incl. O; N_b small.
    bins = np.array([11.0, 12.0, 13.0])
    specs = []
    r = len(specs); specs.append((10.0 ** 11.5, True, -1))      # bin0 root
    # 12 alive orphans in bin1, each merging into r.
    orphans = []
    for k in range(12):
        oidx = len(specs)
        specs.append((10.0 ** (12.1 + 0.05 * k), True, r))
        orphans.append(oidx)
    # a dead progenitor behind EACH orphan
    dead_behind = {}
    for oidx in orphans:
        didx = len(specs)
        specs.append((10.0 ** 9.5, False, oidx))
        dead_behind[oidx] = didx
    forest = _arrays(specs, bins)
    rev = build_reverse_index(forest["merger_track_ids"])

    # Try several seeds and N_b=1 (keep 1 of 12 orphans). For each, the dead
    # progenitor behind a NON-selected orphan must be absent in the subset;
    # behind a selected orphan it must be present (closed under that orphan).
    for seed in range(40):
        s = _run(forest, 1, 0, 1, rev, seed=seed)
        mask = s["subset_mask"]
        rootm = s["root_mask"]
        for oidx, didx in dead_behind.items():
            if rootm[oidx]:
                assert mask[didx], (
                    f"seed{seed}: dead prog {didx} behind SELECTED orphan "
                    f"{oidx} should be closed in"
                )
            else:
                assert not mask[oidx], "non-selected orphan leaked into subset"
                assert not mask[didx], (
                    f"seed{seed}: dead prog {didx} behind UNSELECTED orphan "
                    f"{oidx} leaked via closure from root {r} — SKIP RULE BUG"
                )


def test_chunked_with_orphan_in_loop_disjoint():
    """Same orphan structure, but now CHUNKED. The orphan bin gets split by
    round-robin; r's bin is a single root. Across chunks the subset must
    stay disjoint and the union must equal the single run, with the dead
    progenitors honouring the skip rule in every chunk."""
    bins = np.array([11.0, 12.0, 13.0])
    specs = []
    r = len(specs); specs.append((10.0 ** 11.5, True, -1))
    orphans = []
    for k in range(12):
        oidx = len(specs)
        specs.append((10.0 ** (12.1 + 0.05 * k), True, r))
        orphans.append(oidx)
    for oidx in list(orphans):
        specs.append((10.0 ** 9.5, False, oidx))
    forest = _arrays(specs, bins)
    rev = build_reverse_index(forest["merger_track_ids"])

    for N_b in (1, 3, 7, 10_000):
        single = _run(forest, N_b, 0, 1, rev)
        for K in (2, 3, 5):
            masks, weights = [], []
            for c in range(K):
                try:
                    s = _run(forest, N_b, c, K, rev)
                except ValueError as e:
                    if "selected zero roots" in str(e):
                        continue
                    raise
                masks.append(s["subset_mask"]); weights.append(s["weights"])
            stacked = np.vstack(masks)
            assert stacked.sum(axis=0).max() <= 1, f"N_b={N_b} K={K} overlap"
            assert np.array_equal(stacked.any(axis=0), single["subset_mask"])
            assert np.allclose(np.vstack(weights).sum(axis=0), single["weights"])


if __name__ == "__main__":
    test_same_bin_split_then_dead_prog_follows_one_root()
    print("OK  test_same_bin_split_then_dead_prog_follows_one_root")
    test_dead_prog_behind_alive_orphan_never_closure_walked()
    print("OK  test_dead_prog_behind_alive_orphan_never_closure_walked")
    test_chunked_with_orphan_in_loop_disjoint()
    print("OK  test_chunked_with_orphan_in_loop_disjoint")
    print("\nAll cross-count probes passed.")
