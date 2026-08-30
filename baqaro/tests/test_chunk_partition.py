"""
Correctness gate for chunked multi-node training (Phase 1).
===========================================================

These tests pin down the *estimator-level* correctness of the chunked
root partition added to ``tree_subsample.build_subsampled_subset``. They
are the precondition for any chunked production launch.

What "correct" means here (and what it does NOT mean)
-----------------------------------------------------
The per-chunk weighted summary histograms are pooled by a plain SUM in
``emulation/pool_chunks.py``. That sum equals the single-run (n_chunks=1)
estimator **iff**, across the chunks:

  (A) every kept halo appears in exactly ONE chunk   — disjoint cover,
  (B) the union of chunks equals the single-run subset — completeness,
  (C) each kept halo carries the SAME (full-N_b) weight as in the
      single run — weight preservation.

If (A)+(B)+(C) hold then for any binning,
    Σ_chunks Σ_{i in chunk} w_i·1[x_i in bin]
  = Σ_{i in single-run subset} w_i·1[x_i in bin],
i.e. the pooled weighted histogram is *algebraically identical* to the
single-run weighted histogram — for whatever per-halo values x_i each
run happens to produce.

NOTE on the per-halo values x_i (BH mass / L_bol): these are NOT
bit-identical between a chunk run and a single run, because
``run_single_model`` draws ERDF etas from one per-model RNG stream shared
across all evolving halos — changing the halo set changes each halo's
draw position. So a chunked pool is a *different, equally-valid,
unbiased* Monte-Carlo realization, not a bitwise copy of a single-node
run. That is fine for emulator training. These tests therefore verify
(A)+(B)+(C) exactly, and do NOT assert bitwise pool==single on stats.

Run directly (``python -m ...tests.test_chunk_partition``) or via pytest.
"""

import numpy as np

from baqaro.core_functions.tree_subsample import (
    build_subsampled_subset,
    build_multinode_partition_subset,
    make_subset_tag,
    make_multinode_partition_tag,
)


# ----------------------------------------------------------------------
# Synthetic merger forest
# ----------------------------------------------------------------------
def make_forest(
    *,
    n_roots_per_bin: int,
    log_M_lo: float,
    log_M_hi: float,
    n_bins: int,
    seed: int,
    n_orphan_satellites: int = 30,
):
    """Build a small, fully-specified merger forest for testing.

    Returns a dict with the arrays ``build_subsampled_subset`` consumes
    plus the bin edges. Structure:

    * ``n_roots_per_bin`` roots per mass bin, alive at z_root (death=-1),
      masses uniform inside the bin.
    * Each root gets 0–4 dead progenitors (death>=0) merging into it, and
      each of those gets 0–2 dead grand-progenitors — a clean forest where
      every progenitor has exactly one path to exactly one root.
    * ``n_orphan_satellites`` "re-resolved orphans": alive at z_root
      (death=-1) AND with merger_track_ids pointing at a random root —
      these exercise the v3 BFS skip rule (must be included only if
      stratified as a root, never walked into via closure).
    """
    rng = np.random.default_rng(seed)
    bins = np.linspace(log_M_lo, log_M_hi, n_bins + 1)

    mass = []          # mass at z_root (physical Msun)
    alive = []         # alive_at_root
    merger = []        # forward merger target (-1 if root/disrupted)

    def add(m, al, mt):
        idx = len(mass)
        mass.append(m)
        alive.append(al)
        merger.append(mt)
        return idx

    roots = []
    for b in range(n_bins):
        for _ in range(n_roots_per_bin):
            logm = rng.uniform(bins[b] + 0.02, bins[b + 1] - 0.02)
            r = add(10.0 ** logm, True, -1)
            roots.append(r)
            for _ in range(int(rng.integers(0, 5))):
                p = add(10.0 ** rng.uniform(8.0, 11.0), False, r)
                for _ in range(int(rng.integers(0, 3))):
                    add(10.0 ** rng.uniform(7.0, 10.0), False, p)

    # Re-resolved orphans: alive at root, point at a random root.
    for _ in range(n_orphan_satellites):
        tgt = int(rng.choice(roots))
        add(10.0 ** rng.uniform(10.0, 12.0), True, tgt)

    n = len(mass)
    track_ids = np.arange(n, dtype=np.int64)
    merger_track_ids = np.asarray(merger, dtype=np.int64)
    halo_masses_at_root = np.asarray(mass, dtype=np.float64)
    alive_at_root = np.asarray(alive, dtype=bool)
    return dict(
        track_ids=track_ids,
        merger_track_ids=merger_track_ids,
        halo_masses_at_root=halo_masses_at_root,
        alive_at_root=alive_at_root,
        log_mass_bins=bins,
    )


def _build(forest, *, N_b, chunk_id, n_chunks, seed=42):
    return build_subsampled_subset(
        track_ids=forest["track_ids"],
        merger_track_ids=forest["merger_track_ids"],
        halo_masses_at_root=forest["halo_masses_at_root"],
        alive_at_root=forest["alive_at_root"],
        log_mass_bins=forest["log_mass_bins"],
        N_b_per_bin=N_b,
        keep_all_above_log_M=None,
        rng_seed=seed,
        chunk_id=chunk_id,
        n_chunks=n_chunks,
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_nchunks1_identical_to_default():
    """n_chunks=1 (explicit) must reproduce the un-chunked build exactly."""
    forest = make_forest(
        n_roots_per_bin=40, log_M_lo=11.0, log_M_hi=15.0, n_bins=8, seed=1
    )
    default = build_subsampled_subset(
        track_ids=forest["track_ids"],
        merger_track_ids=forest["merger_track_ids"],
        halo_masses_at_root=forest["halo_masses_at_root"],
        alive_at_root=forest["alive_at_root"],
        log_mass_bins=forest["log_mass_bins"],
        N_b_per_bin=15,
        rng_seed=42,
    )  # no chunk args at all
    explicit = _build(forest, N_b=15, chunk_id=0, n_chunks=1)
    assert np.array_equal(default["subset_mask"], explicit["subset_mask"])
    assert np.array_equal(default["weights"], explicit["weights"])
    assert np.array_equal(default["root_mask"], explicit["root_mask"])


def test_partition_disjoint_complete_weightpreserving():
    """For several N_b and n_chunks: assert (A) disjoint, (B) complete,
    (C) weights preserved, and the headline sum-of-weights identity."""
    forest = make_forest(
        n_roots_per_bin=50, log_M_lo=11.0, log_M_hi=15.5, n_bins=9, seed=7
    )
    for N_b in (10, 25, 1_000_000):  # incl. N_b >> bin count (keeps all)
        single = _build(forest, N_b=N_b, chunk_id=0, n_chunks=1)
        for K in (2, 3, 5, 7):
            masks = []
            weights = []
            root_masks = []
            for c in range(K):
                s = _build(forest, N_b=N_b, chunk_id=c, n_chunks=K)
                masks.append(s["subset_mask"])
                weights.append(s["weights"])
                root_masks.append(s["root_mask"])
            stacked = np.vstack(masks)
            membership = stacked.sum(axis=0)

            # (A) disjoint: no halo in more than one chunk.
            assert membership.max() <= 1, (
                f"N_b={N_b} K={K}: {int((membership > 1).sum())} halos "
                "appear in more than one chunk (closures overlap!)"
            )
            # (B) complete: union of chunks == single-run subset.
            union = stacked.any(axis=0)
            assert np.array_equal(union, single["subset_mask"]), (
                f"N_b={N_b} K={K}: union of chunk subsets != single-run subset"
            )
            # roots also partition cleanly.
            root_stack = np.vstack(root_masks)
            assert root_stack.sum(axis=0).max() <= 1
            assert np.array_equal(root_stack.any(axis=0), single["root_mask"])

            # (C) weight preservation: where a halo is kept in a chunk, its
            # weight equals the single-run weight.
            for c in range(K):
                m = masks[c]
                assert np.allclose(weights[c][m], single["weights"][m])

            # Headline identity: per-halo weight summed across chunks equals
            # the single-run per-halo weight. This is exactly the statement
            # that pooled weighted histograms == single-run weighted
            # histograms for ANY binning of ANY per-halo quantity.
            wsum = np.sum(np.vstack(weights), axis=0)
            assert np.allclose(wsum, single["weights"]), (
                f"N_b={N_b} K={K}: Σ_chunks weight != single-run weight"
            )


def test_chunk_tag_token():
    """Tag is unchanged for n_chunks=1, gains _chunk{k}of{N} otherwise, and
    the pooled (chunk-free) tag equals the single-node tag."""
    base = dict(root_snap=144, N_b=500_000, n_bins=22, log_M_lo=10.0,
                log_M_hi=15.5, seed=42)
    single = make_subset_tag(**base)
    assert make_subset_tag(**base, chunk_id=0, n_chunks=1) == single
    chunked = make_subset_tag(**base, chunk_id=2, n_chunks=4)
    assert chunked == single + "_chunk2of4"
    # stripping the chunk token recovers the single-node (pool output) tag.
    assert chunked.rsplit("_chunk", 1)[0] == single


# ----------------------------------------------------------------------
# Multinode whole-catalogue (no-subsample) partition — for main_evolution_chunked
# ----------------------------------------------------------------------
# These gate ``build_multinode_partition_subset``: the BAQARO_USE_SUBSAMPLE=0
# path where EVERY alive-at-root halo is a root, weights are all 1.0, and the
# union of all chunks is the ENTIRE catalogue (roots + their full merger
# closure). Same (A) disjoint / (B) complete contract as the stratified path,
# plus the stronger multinode completeness: union == every halo in the forest.
def _build_multinode(forest, *, chunk_id, n_chunks):
    return build_multinode_partition_subset(
        track_ids=forest["track_ids"],
        merger_track_ids=forest["merger_track_ids"],
        resolved_mask=forest["halo_masses_at_root"] > 0,
        chunk_id=chunk_id,
        n_chunks=n_chunks,
    )


def test_multinode_nchunks1_is_whole_catalogue():
    """n_chunks=1 multinode build keeps every resolved halo (each belongs to
    exactly one terminal's component) with weight 1.0; roots are the terminals
    (merger_track_ids == -1), not the alive-at-root set."""
    forest = make_forest(
        n_roots_per_bin=40, log_M_lo=11.0, log_M_hi=15.0, n_bins=8, seed=3
    )
    whole = _build_multinode(forest, chunk_id=0, n_chunks=1)
    n = len(forest["track_ids"])
    # Every halo's merger chain ends at a terminal, so the closure is the whole
    # catalogue (all masses > 0 in this forest).
    assert whole["subset_mask"].all(), (
        f"{int((~whole['subset_mask']).sum())} halos not covered by the "
        "n_chunks=1 closure"
    )
    assert np.array_equal(whole["weights"], np.ones(n))
    # Roots are the terminals (merger_track_ids == -1) — NOT every alive halo:
    # re-resolved orphans are alive yet have merger != -1, so they enter via
    # closure, not as roots.
    assert np.array_equal(whole["root_mask"], forest["merger_track_ids"] == -1)


def test_multinode_partition_disjoint_complete():
    """For several n_chunks: (A) disjoint, (B) union == single build == whole
    catalogue, (C) weights all 1.0, and terminals partition cleanly."""
    forest = make_forest(
        n_roots_per_bin=50, log_M_lo=11.0, log_M_hi=15.5, n_bins=9, seed=11,
        n_orphan_satellites=40,
    )
    single = _build_multinode(forest, chunk_id=0, n_chunks=1)
    n = len(forest["track_ids"])
    terminals = forest["merger_track_ids"] == -1
    for K in (2, 3, 4, 8):
        masks, root_masks = [], []
        for c in range(K):
            s = _build_multinode(forest, chunk_id=c, n_chunks=K)
            # (C) every kept halo has weight 1.0.
            assert np.array_equal(s["weights"][s["subset_mask"]],
                                  np.ones(int(s["subset_mask"].sum())))
            masks.append(s["subset_mask"])
            root_masks.append(s["root_mask"])
        stacked = np.vstack(masks)
        membership = stacked.sum(axis=0)
        # (A) disjoint: each halo belongs to exactly one terminal's component.
        assert membership.max() <= 1, (
            f"K={K}: {int((membership > 1).sum())} halos appear in >1 chunk"
        )
        # (B) complete: union of chunks == single build == whole catalogue.
        union = stacked.any(axis=0)
        assert np.array_equal(union, single["subset_mask"])
        assert union.sum() == n
        # terminals partition cleanly across chunks.
        root_stack = np.vstack(root_masks)
        assert root_stack.sum(axis=0).max() <= 1
        assert np.array_equal(root_stack.any(axis=0), terminals)
        assert sum(int(rm.sum()) for rm in root_masks) == int(terminals.sum())


def test_multinode_includes_disrupted_deadends():
    """Dead-end lineages (disrupted terminals, merger=-1, death>=0) that never
    reach z=0 MUST be included — main_evolution.py evolves them, so a true
    full-sim equivalent must too. Never-resolved (mass=0) halos are excluded.

    Forest (track_id = index):
      0  survivor root   death=-1 merger=-1 mass>0   (terminal, kept as root)
      1  prog of 0       death=3  merger=0  mass>0   (closure of 0)
      2  dead-end term   death=2  merger=-1 mass>0   (terminal, kept as root)
      3  prog of 2       death=1  merger=2  mass>0   (closure of 2)
      4  never-resolved  death=2  merger=-1 mass=0   (dropped by resolved_mask)
    """
    merger = np.array([-1, 0, -1, 2, -1], dtype=np.int64)
    mass = np.array([1e13, 1e11, 1e12, 1e10, 0.0], dtype=np.float64)
    forest = dict(
        track_ids=np.arange(5, dtype=np.int64),
        merger_track_ids=merger,
        halo_masses_at_root=mass,
    )
    whole = _build_multinode(forest, chunk_id=0, n_chunks=1)
    # Resolved halos {0,1,2,3} kept; never-resolved 4 dropped.
    assert np.array_equal(whole["subset_mask"], np.array([1, 1, 1, 1, 0], bool))
    # Both terminals (survivor 0 AND dead-end 2) are roots; not the mass=0 one.
    assert np.array_equal(whole["root_mask"], np.array([1, 0, 1, 0, 0], bool))
    # The dead-end terminal and its progenitor are present.
    assert whole["subset_mask"][2] and whole["subset_mask"][3]

    # Split: each terminal's component lands in one chunk; union == resolved.
    c0 = _build_multinode(forest, chunk_id=0, n_chunks=2)
    c1 = _build_multinode(forest, chunk_id=1, n_chunks=2)
    assert (c0["subset_mask"] & c1["subset_mask"]).sum() == 0
    union = c0["subset_mask"] | c1["subset_mask"]
    assert np.array_equal(union, np.array([1, 1, 1, 1, 0], bool))


def test_multinode_partition_tag_token():
    """multinode_ tag: chunk-free for n_chunks=1, gains _chunk{k}of{N}
    otherwise, and never collides with a stratified subsample tag.

    v3 -> v4: the partition itself changed (v4 roots on ALL terminals, so
    every resolved halo -- including those whose chain dead-ends at an
    unresolved terminal -- lands in exactly one chunk). The tag bump
    keeps v3 and v4 products from ever being mixed under one name — existing
    on-disk v3 products are still readable via an explicit BAQARO_SUBSET_TAG.
    """
    single = make_multinode_partition_tag(root_snap=144)
    assert single == "multinode_root144_v4"
    assert make_multinode_partition_tag(root_snap=144, chunk_id=0, n_chunks=1) == single
    chunked = make_multinode_partition_tag(root_snap=144, chunk_id=3, n_chunks=16)
    assert chunked == "multinode_root144_v4_chunk3of16"
    assert chunked.rsplit("_chunk", 1)[0] == single
    # disjoint namespace from stratified tags (those start "root{snap}_flatN").
    strat = make_subset_tag(root_snap=144, N_b=500_000, n_bins=22,
                            log_M_lo=10.0, log_M_hi=15.5, seed=42)
    assert single.startswith("multinode_") and not strat.startswith("multinode_")


if __name__ == "__main__":
    test_nchunks1_identical_to_default()
    print("OK  test_nchunks1_identical_to_default")
    test_partition_disjoint_complete_weightpreserving()
    print("OK  test_partition_disjoint_complete_weightpreserving")
    test_chunk_tag_token()
    print("OK  test_chunk_tag_token")
    test_multinode_nchunks1_is_whole_catalogue()
    print("OK  test_multinode_nchunks1_is_whole_catalogue")
    test_multinode_partition_disjoint_complete()
    print("OK  test_multinode_partition_disjoint_complete")
    test_multinode_includes_disrupted_deadends()
    print("OK  test_multinode_includes_disrupted_deadends")
    test_multinode_partition_tag_token()
    print("OK  test_multinode_partition_tag_token")
    print("\nAll chunk-partition gate tests passed.")
