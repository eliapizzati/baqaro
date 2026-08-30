"""Equivalence test for the eager-load cache + precomp speedups.

Loads a small cached subset (L2800N5040 maxsnap=14, foldmass+instant_new)
and verifies that:

  1. ``extract_subset_slice_cached`` returns arrays byte-identical to
     the original inline row-by-row gather from the mmap'd source, both
     on the cold-rebuild path and on the warm-cache read path.
  2. ``precompute_per_snapshot_indices`` returns per-snap arrays that
     match the original inline full-N boolean-mask + flatnonzero
     pattern (modulo dtype: both produce sorted ascending indices, but
     int64 vs the original's int64 — same dtype).

Run from the repo root via::

    python -m baqaro.tests.diag_precomp_speedups

Sets a few env vars internally so it does not depend on the caller's
``BAQARO_*`` state.
"""

import os
import sys
import time

# Make all imports below pick up the test sim regardless of caller env.
os.environ["BAQARO_SIM"] = "L2800N5040"
os.environ["BAQARO_MAX_SNAP"] = "14"
os.environ["BAQARO_FOLD_SUBHALO_MASS"] = "1"
os.environ["BAQARO_MERGER_DELAY_MODE"] = "instant_new"
os.environ["BAQARO_USE_SUBSAMPLE"] = "1"
os.environ["BAQARO_SOURCE_DIR"] = "machine_igm"

import numpy as np

from baqaro.utils.my_dir import get_output_path
from baqaro.utils.sim_config import (
    simulation_name, max_snap as cfg_max_snap,
)
from baqaro.core_functions.halo_mass_histories_saver import (
    MergerTreeLoader,
)
from baqaro.core_functions.tree_subsample import (
    load_subset,
    extract_subset_slice_cached,
    precompute_per_snapshot_indices,
)


SOURCE = "machine_igm"
NBOUND_THRESHOLD = 40
HALO_FILTERING_MODE = "global"
TDYN_FRACTION = 0.2
FOLD = True
MERGER_DELAY_MODE = "instant_new"
SUBSET_TAG = "root14_flatN500000_K18_logM11.0to15.5_seed42_v3"
MAX_SNAP = 14
MIN_SNAP = 0


def main() -> int:
    print(f"sim_config.simulation_name = {simulation_name}, max_snap = {cfg_max_snap}")
    path_out = get_output_path(SOURCE)

    name_file_halos = (
        f"{simulation_name}_maxsnap{MAX_SNAP}_nboundthresh{NBOUND_THRESHOLD}"
        f"_halofilter_{HALO_FILTERING_MODE}_tdynfraction_{TDYN_FRACTION}"
    )
    if FOLD:
        name_file_halos += "_foldmass"
    if MERGER_DELAY_MODE != "instant_old":
        name_file_halos += f"_{MERGER_DELAY_MODE}"

    path_halo_masses = os.path.join(
        path_out, "halo_histories", f"halo_masses_{name_file_halos}.npy"
    )
    path_cold_rates = os.path.join(
        path_out, "halo_histories",
        f"specific_cold_accretion_rates_{name_file_halos}.npy",
    )
    path_trees = os.path.join(path_out, "halo_histories", f"merger_trees_{name_file_halos}")

    subset_path = os.path.join(
        path_out, "halo_subsets", f"subset_{name_file_halos}_{SUBSET_TAG}.npz"
    )

    for p in (path_halo_masses, path_cold_rates, subset_path):
        if not os.path.exists(p):
            print(f"MISSING input: {p}", file=sys.stderr)
            return 2

    # ---- Load subset + merger tree arrays. ----
    print(f"Loading subset cache from {subset_path}")
    subset = load_subset(subset_path)
    subset_mask = subset["subset_mask"]
    subset_indices = np.flatnonzero(subset_mask)
    n_subset = len(subset_indices)
    n_full = len(subset_mask)
    print(f"  n_subset = {n_subset:,} / n_full = {n_full:,}")

    print(f"Loading merger tree from {path_trees}")
    mt = MergerTreeLoader(path_trees)
    mt_birth_index = np.array(mt.snapshot_indexes_of_birth)
    mt_death_index = np.array(mt.snapshot_indexes_of_death)
    mt_merger_ids = np.array(mt.merger_track_ids)
    mask_merged = mt_merger_ids != -1
    mask_disrupted = mt_merger_ids == -1

    # =================================================================
    # TEST 1: extract_subset_slice_cached  ↔  inline row-by-row gather
    # =================================================================
    print("\n[TEST 1] Eager-load cache vs inline gather")

    # OLD inline path: row-by-row fancy gather from the mmap'd source.
    print("  Building OLD-style reference (inline mmap fancy-index)...")
    t0 = time.time()
    full = np.load(path_halo_masses, mmap_mode="r")
    old_masses = np.empty((full.shape[0], n_subset), dtype=full.dtype)
    for r in range(full.shape[0]):
        old_masses[r] = full[r, subset_indices]
    print(f"    {time.time()-t0:.1f}s  shape={old_masses.shape} dtype={old_masses.dtype}")

    cache_path = os.path.join(
        path_out, "halo_subsets",
        f"subset_{name_file_halos}_{SUBSET_TAG}__halo_masses.npy",
    )

    # NEW cold path: nuke any pre-existing cache, force the helper to
    # rebuild via the mmap → row-by-row gather → save .npy code path.
    if os.path.exists(cache_path):
        print(f"  Removing pre-existing cache: {cache_path}")
        os.remove(cache_path)
    print("  Building NEW-style array (cold rebuild)...")
    new_masses_cold = extract_subset_slice_cached(
        path_halo_masses, subset_indices, cache_path
    )
    if not np.array_equal(old_masses, new_masses_cold):
        print("  ❌ cold-rebuild output differs from inline reference")
        return 1
    print("  ✅ cold rebuild matches inline reference (byte-identical)")

    # NEW warm path: re-call the helper; it should hit the cache.
    print("  Reading NEW-style array (warm cache)...")
    new_masses_warm = extract_subset_slice_cached(
        path_halo_masses, subset_indices, cache_path
    )
    if not np.array_equal(old_masses, new_masses_warm):
        print("  ❌ warm-cache output differs from inline reference")
        return 1
    print("  ✅ warm cache matches inline reference (byte-identical)")

    # And the same for the cold-accretion-rates file.
    print("  Same test, cold-accretion-rates source:")
    t0 = time.time()
    full_r = np.load(path_cold_rates, mmap_mode="r")
    old_rates = np.empty((full_r.shape[0], n_subset), dtype=full_r.dtype)
    for r in range(full_r.shape[0]):
        old_rates[r] = full_r[r, subset_indices]
    print(f"    OLD inline gather: {time.time()-t0:.1f}s")

    cache_path_r = os.path.join(
        path_out, "halo_subsets",
        f"subset_{name_file_halos}_{SUBSET_TAG}__specific_cold_accretion_rates.npy",
    )
    if os.path.exists(cache_path_r):
        os.remove(cache_path_r)
    new_rates_cold = extract_subset_slice_cached(
        path_cold_rates, subset_indices, cache_path_r
    )
    if not np.array_equal(old_rates, new_rates_cold):
        print("  ❌ cold-rebuild rates differ from inline reference")
        return 1
    new_rates_warm = extract_subset_slice_cached(
        path_cold_rates, subset_indices, cache_path_r
    )
    if not np.array_equal(old_rates, new_rates_warm):
        print("  ❌ warm-cache rates differ from inline reference")
        return 1
    print("  ✅ cold-accretion-rates eager-load matches on both paths")

    # =================================================================
    # TEST 2: precompute_per_snapshot_indices  ↔  inline full-N flatnonzero
    # =================================================================
    print("\n[TEST 2] Precomp per-snap indices vs inline full-N flatnonzero")

    # OLD inline (main_evolution-style: 4-way split).
    print("  Building OLD-style precomp (inline full-N flatnonzero)...")
    t0 = time.time()
    death_eff = mt_death_index.copy()
    death_eff[death_eff == -1] = np.iinfo(death_eff.dtype).max
    old_spawn, old_alive, old_merge, old_lost = {}, {}, {}, {}
    for s in range(MIN_SNAP, MAX_SNAP + 1):
        spawn = (mt_birth_index == s)
        born_before = (mt_birth_index < s) & (mt_birth_index != -1)
        alive = born_before & (death_eff > s)
        dying = born_before & (mt_death_index == s)
        dying_merged = dying & mask_merged
        dying_lost = dying & mask_disrupted
        spawn &= subset_mask
        alive &= subset_mask
        dying_merged &= subset_mask
        dying_lost &= subset_mask
        old_spawn[s] = np.flatnonzero(spawn)
        old_alive[s] = np.flatnonzero(alive)
        old_merge[s] = np.flatnonzero(dying_merged)
        old_lost[s] = np.flatnonzero(dying_lost)
    print(f"    {time.time()-t0:.2f}s")

    # NEW (subset-restricted ops + index map-back).
    print("  Building NEW-style precomp (subset-restricted)...")
    t0 = time.time()
    new = precompute_per_snapshot_indices(
        min_snap=MIN_SNAP,
        max_snap=MAX_SNAP,
        mt_birth_index=mt_birth_index,
        mt_death_index=mt_death_index,
        mask_merged=mask_merged,
        mask_disrupted=mask_disrupted,
        subset_indices=subset_indices,
        verbose=False,
    )
    print(f"    {time.time()-t0:.2f}s")

    for s in range(MIN_SNAP, MAX_SNAP + 1):
        for label, ref, got in (
            ("spawning", old_spawn[s], new["spawning"][s]),
            ("evolving", old_alive[s], new["evolving"][s]),
            ("merging", old_merge[s], new["merging"][s]),
            ("lost",    old_lost[s],  new["lost"][s]),
        ):
            if not np.array_equal(ref, got):
                print(
                    f"  ❌ snap {s} {label}: ref shape={ref.shape}, "
                    f"got shape={got.shape} — first 10 diff(s) "
                    f"{np.setdiff1d(ref, got)[:10]} | "
                    f"{np.setdiff1d(got, ref)[:10]}"
                )
                return 1
    print("  ✅ all per-snap index arrays match for every category & snapshot")

    # =================================================================
    # TEST 3: also verify the full-sim branch of the precomp helper
    #          reproduces the inline pattern when subset_indices=None.
    # =================================================================
    print("\n[TEST 3] Precomp full-sim path (subset_indices=None)")
    new_full = precompute_per_snapshot_indices(
        min_snap=MIN_SNAP,
        max_snap=MAX_SNAP,
        mt_birth_index=mt_birth_index,
        mt_death_index=mt_death_index,
        mask_merged=mask_merged,
        mask_disrupted=mask_disrupted,
        subset_indices=None,
        verbose=False,
    )
    # Reference: redo inline WITHOUT subset_mask AND'ing.
    old_spawn_full, old_alive_full, old_merge_full, old_lost_full = {}, {}, {}, {}
    for s in range(MIN_SNAP, MAX_SNAP + 1):
        spawn = (mt_birth_index == s)
        born_before = (mt_birth_index < s) & (mt_birth_index != -1)
        alive = born_before & (death_eff > s)
        dying = born_before & (mt_death_index == s)
        old_spawn_full[s] = np.flatnonzero(spawn)
        old_alive_full[s] = np.flatnonzero(alive)
        old_merge_full[s] = np.flatnonzero(dying & mask_merged)
        old_lost_full[s] = np.flatnonzero(dying & mask_disrupted)
    for s in range(MIN_SNAP, MAX_SNAP + 1):
        for label, ref, got in (
            ("spawning", old_spawn_full[s], new_full["spawning"][s]),
            ("evolving", old_alive_full[s], new_full["evolving"][s]),
            ("merging", old_merge_full[s], new_full["merging"][s]),
            ("lost",    old_lost_full[s],  new_full["lost"][s]),
        ):
            if not np.array_equal(ref, got):
                print(f"  ❌ snap {s} {label} differs (full-sim path)")
                return 1
    print("  ✅ full-sim precomp matches inline reference")

    print("\nAll equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
