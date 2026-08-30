"""
Multinode whole-catalogue (no-subsample) fiducial forward run — ONE chunk.
==========================================================================

A *separate* forward-model entry point, kept apart from ``main_evolution.py``,
for the occasional big final run: ONE fiducial parameter set evolved over the
ENTIRE halo catalogue at full simulation resolution (no subsampling), split
into ``BAQARO_N_CHUNKS`` disjoint, merger-closed chunks. Each chunk runs on its
own node; the per-chunk outputs are recombined afterward (``concat_chunks.py``
/ ``pool_chunks.py``, Phase 2).

Run ONE chunk:

    BAQARO_USE_SUBSAMPLE=0 BAQARO_N_CHUNKS=16 BAQARO_CHUNK_ID=<k> \
    BAQARO_BESTFIT_NAME=<fiducial> python -m \
    baqaro.core_functions.main_evolution_chunked

Tree-splitting is exact because the chunks are a disjoint, merger-closed
partition of the catalogue (see ``tree_subsample.build_multinode_partition_subset``):
no merger crosses a chunk boundary, so every per-halo quantity is what the
single-node run would have produced.

What one chunk produces (all at the 13 training snapshots
``utils.sim_config.snapshots_to_save_default``):
  1. Summary statistics (QLF/BHMF/CERDF/QHMF) — same schema as training,
     weight ≡ 1.0, pooled additively across chunks (``pool_chunks.py``).
  2. Per-halo BH catalogue (global track_id, M_BH, L_bol) for every halo
     hosting a BH (M_BH > 0), dumped at those snapshots and concatenated
     across chunks (``concat_chunks.py``).
  3. Merger catalogue (binary events + z=0 survivors), unioned.

Two structural choices, both matching ``main_evolution.py`` (the per-halo
ground truth the whole pipeline is calibrated to):
  * RAM-ROLLING — only the previous/current snapshot's M_BH + L_bol rows are
    kept live (``bh_prev`` / ``bh_curr``), unlike main_evolution which holds
    all 145 snapshots (3.76 TB at full-sim). The catalogue is streamed at the
    anchors only.
  * FRESH-ZERO each snapshot — every snapshot's state starts at zero and only
    spawning / evolving / merge-dest halos are written, so a halo that died
    (disrupted or merged) is zero thereafter (no dead-halo carry-forward).

RNG — Option A (per-chunk shared stream). See the RNG-seam note below
"RNG convention". The whole-chunk realization is reproducible for a fixed
(BAQARO_RNG_SEED, N_chunks) but NOT invariant to the chunk count: the summary
stats are unaffected, the per-halo + merger catalogues are a valid but
chunk-count-dependent realization. ALL randomness flows through the
``_make_run_rng`` seam + the single ``erdf.rng`` / ``spawn_BHs(rng=)`` it feeds,
so switching to Option B (per-track-id counter-based) is a contained change.
"""

import os
# Set Numba thread count before any numba import (via bh_accretion_fast).
os.environ.setdefault("NUMBA_NUM_THREADS", str(max(1, os.cpu_count() // 2)))

import time
import numpy as np
from numpy.random import default_rng, SeedSequence

from qhtools.utils import natconst as nc, my_utils

from baqaro.core_functions.bh_seeding import spawn_BHs
from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast as evolve_BHs
from baqaro.core_functions.merger_catalog import MergerCatalogBuilder
from baqaro.utils.sim_config import (
    env_bool, FIDUCIAL_GROWTH_SUM_MAX, FIDUCIAL_MADAU_FEFF_CORRECTION,
)


# ==============================================================================
# RNG SEAM (Option A — per-chunk shared stream)
# ==============================================================================
def _make_run_rng(base_seed: int, chunk_id: int):
    """Build the per-chunk RNG for Option A (shared per-chunk stream).

    Seeded deterministically from ``(base_seed, chunk_id)`` so a fixed
    ``(BAQARO_RNG_SEED, N_chunks)`` is fully reproducible. The per-halo
    realization still depends on the chunk count (the stream is consumed in
    iteration order across the chunk's evolving halos) — that is the Option A
    contract. To switch to Option B (per-track-id, chunk-count-independent),
    replace this seam and the kernel draws; the rest of the engine is
    RNG-agnostic.
    """
    return default_rng(SeedSequence([int(base_seed), int(chunk_id)]))


# ==============================================================================
# RUN IDENTITY / NAMING — single source of truth (shared with concat_chunks.py)
# ==============================================================================
def resolve_identity():
    """Resolve all naming-relevant tokens from env + registry + sim_config.

    The ONE place that decides notes_file / bestfit / growth-cap / feffcorr /
    nonaccreting tokens, so the engine and the recombiner (``concat_chunks.py``)
    can never disagree on filenames. Also carries the matched registry entry so
    the param resolver can reuse it.
    """
    from baqaro.utils.sim_config import (
        max_snap, tdyn_fraction_default, fold_subhalo_mass, merger_delay_mode,
        simulation_name,
    )
    erdf_model = "log_normal_evol_halo_mass"
    bestfit_name = os.environ.get("BAQARO_BESTFIT_NAME", None)
    registry_entry = None
    if bestfit_name:
        from baqaro.core_functions.bestfit_registry import BESTFIT_REGISTRY
        registry_entry = BESTFIT_REGISTRY.get(bestfit_name)

    notes_env_set = "BAQARO_NOTES_FILE" in os.environ
    notes_file = (os.environ["BAQARO_NOTES_FILE"] or None) if notes_env_set else None
    if not notes_env_set and registry_entry is not None:
        notes_file = registry_entry.get("notes") or None

    nonaccreting_zero_rate = env_bool("BAQARO_NONACCRETING_ZERO_RATE", True)
    if not nonaccreting_zero_rate:
        notes_file = (notes_file + "_clipfloor") if notes_file else "clipfloor"

    growth_sum_max = float(os.environ.get("BAQARO_GROWTH_SUM_MAX", str(FIDUCIAL_GROWTH_SUM_MAX)))
    madau_feff_correction = (
        os.environ.get("BAQARO_MADAU_FEFF_CORRECTION",
                       "1" if FIDUCIAL_MADAU_FEFF_CORRECTION else "0").strip().lower()
        not in ("0", "", "false", "no", "off")
    )
    return {
        "simulation_name": simulation_name, "erdf_model": erdf_model,
        "max_snap": int(max_snap), "min_snap": 0,
        "fold_subhalo_mass": bool(fold_subhalo_mass), "merger_delay_mode": merger_delay_mode,
        "tdyn_fraction_default": float(tdyn_fraction_default),
        "nbound_threshold": 40, "halo_filtering_mode": "global",
        "notes_file": notes_file, "bestfit_name": bestfit_name,
        "nonaccreting_zero_rate": nonaccreting_zero_rate,
        "growth_sum_max": growth_sum_max, "madau_feff_correction": madau_feff_correction,
        "subsample_root_snap": int(max_snap), "registry_entry": registry_entry,
    }


def make_name_file(identity, chunk_id, n_chunks):
    """Build ``name_file`` for one chunk (mirrors main_evolution.py's token chain,
    with the multinode tag in the subset-tag slot). ``n_chunks == 1`` gives the
    chunk-free combined name (what concat_chunks.py writes)."""
    from baqaro.core_functions.tree_subsample import make_multinode_partition_tag
    tag = make_multinode_partition_tag(
        root_snap=identity["subsample_root_snap"], chunk_id=chunk_id, n_chunks=n_chunks)
    # NB: the historical `_6d` token ("6 free parameters") was dropped —
    # it was constant in every name, never varied, and the emulator/chain names had
    # already dropped it. All on-disk products were renamed to match.
    nf = f"{identity['simulation_name']}_erdf_{identity['erdf_model']}_maxsnap_{identity['max_snap']}"
    if identity["fold_subhalo_mass"]:
        nf += "_foldmass"
    if identity["merger_delay_mode"] != "instant_old":
        nf += f"_{identity['merger_delay_mode']}"
    if identity["notes_file"] is not None:
        nf += f"_{identity['notes_file']}"
    if identity["bestfit_name"]:
        nf += f"_bestfit_{identity['bestfit_name']}"
    # Use the standard ``_sub_<tag>`` slot (NOT a bare ``_<tag>``) so the
    # combined dense bh_evolution file is a DROP-IN for the plotting loader,
    # which always assembles ``..._sub_{BAQARO_SUBSET_TAG}...``. The multinode
    # run is a (weight-1) union-subset, so ``_sub_`` is accurate. To load it:
    # BAQARO_SUBSET_TAG=multinode_root{snap}_v3 (+ the usual notes/bestfit/g/feff).
    nf += f"_sub_{tag}"
    g = identity["growth_sum_max"]
    if g != 50.0:
        nf += f"_g{int(g)}" if float(g).is_integer() else f"_g{g:g}"
    if identity["madau_feff_correction"]:
        nf += "_feffcorr"
    # Model-changing physics toggles — same token chain as
    # main_evolution.py, read from the env at call time so a non-default toggle
    # never overwrites the fiducial chunked product.
    from baqaro.utils.sim_config import physics_toggle_suffix
    nf += physics_toggle_suffix()
    return nf


def output_paths(identity, chunk_id, n_chunks, path_out):
    """The per-chunk (or combined, if n_chunks==1) output paths + name_file.

    ``stats``/``catalog``/``merger`` are the per-chunk intermediates the engine
    writes. ``bh_evolution`` is the COMBINED, standard-schema per-halo file the
    recombiner (concat_chunks) emits — same layout as ``main_evolution.py`` so
    the plotting ``DataLoader`` reads it unchanged (n_chunks=1 → chunk-free name).
    """
    nf = make_name_file(identity, chunk_id, n_chunks)
    d = os.path.join(path_out, "evolution")
    return {
        "name_file": nf,
        "stats": os.path.join(d, f"bh_evolution_stats_{nf}.hdf5"),
        "catalog": os.path.join(d, f"bh_evolution_catalog_{nf}.hdf5"),
        "merger": os.path.join(d, f"merger_catalog_{nf}.hdf5"),
        "bh_evolution": os.path.join(d, f"bh_evolution_{nf}.hdf5"),
    }


# ==============================================================================
# PURE EVOLUTION KERNEL (file-I/O-free → unit-testable on synthetic forests)
# ==============================================================================
def evolve_chunk(
    *,
    # storage layout
    subset_indices,            # (n_storage,) GLOBAL track id of each storage slot
    global_to_storage,         # (n_full,) GLOBAL -> storage position, -1 if absent
    n_storage,
    # halo data (storage-sliced rows): shape (n_snaps_total, n_storage)
    halo_masses_all,
    halo_specific_cold_accretion_rates_all,
    # merger-tree (GLOBAL, full-N)
    mt_track_ids,
    mt_merger_ids,
    mt_death_index,
    # precomputed per-snapshot GLOBAL index arrays (already subset-restricted)
    precomp_spawning,
    precomp_evolving,
    precomp_merging,
    # snapshot grid
    snapshots,                 # iterable of snapshot indices to run
    snap_to_zindex,            # {snap: row} for the stat grid
    n_z,
    anchor_set,                # set of snapshots to dump the per-halo catalogue
    redshifts,
    delta_times_snapshots,
    # physics
    erdf,
    rng_run,
    transfer_function,
    time_step_for_accretion,
    accretion_on,
    merger_on,
    nonaccreting_zero_rate,
    rad_efficiency_0,
    rad_efficiency_model,
    growth_sum_max,
    madau_feff_correction,
    logfseed,
    sigmaseed,
    # stats bins / normalizations
    qlf_bins, qlf_normalization,
    bhmf_bins, bhmf_normalization,
    cerdf_bins, cerdf_normalization,
    qhmf_bins, qhmf_normalization,
    log_L_thresholds, Lcuts, lowest_cut,
    log_csi_10, mass_units_f32,
    # sinks / numerics
    anchor_sink=None,          # callable(snap, z, track_id[], M_BH_phys[], Lbol_phys[])
    record_merger_catalog=True,
    use_parallel_kernel=True,  # prange — correct for the single-process node run
    logger=None,
):
    """Evolve one chunk RAM-rolling; return stats + survivors (+ merger cat).

    ``anchor_sink`` is called once per anchor snapshot with the live-BH
    (M_BH > 0) GLOBAL track ids and PHYSICAL M_BH / L_bol — production wires it
    to an HDF5 writer; tests collect into a dict. Returns a dict with the four
    ``log_*`` stat arrays, the finalized merger-catalogue dict (code units, or
    None), and the z=0 survivor track ids + masses (code units).
    """
    n_lbins = len(qlf_bins) - 1
    n_mbins = len(bhmf_bins) - 1
    n_Lthr = len(log_L_thresholds)
    log_qlfs = np.zeros((n_z, n_lbins), dtype=np.float32)
    log_bhmfs = np.zeros((n_z, n_mbins), dtype=np.float32)
    log_cerdfs = np.zeros((n_z, n_Lthr, n_mbins), dtype=np.float32)
    log_qhmfs = np.zeros((n_z, n_Lthr, n_mbins), dtype=np.float32)

    def _to_storage(x):
        return global_to_storage[x]

    bh_prev = np.zeros(n_storage, dtype=np.float32)
    bh_curr = np.zeros(n_storage, dtype=np.float32)
    Lbol_curr = np.zeros(n_storage, dtype=np.float32)
    _buf_bh_prev = np.empty(n_storage, dtype=np.float32)
    _buf_log_rates = np.empty(n_storage, dtype=np.float32)
    _buf_Lbols_out = np.empty(n_storage, dtype=np.float64)
    _rng_buffers = {
        "z_last": np.empty(n_storage, dtype=np.float32),
        "u": np.empty(n_storage, dtype=np.float64),
        "z_approx": np.empty(n_storage, dtype=np.float32),
    }
    # Storage positions are int32 — which SILENTLY WRAPS at n_storage >= 2**31
    # (2.15 B), corrupting every merger mapping with negative/aliased positions
    # rather than failing. Reachable only by a 1-chunk full-box z=0 run (3.24 B
    # halos), currently shielded by RAM infeasibility — but the shield is
    # circumstantial, so assert it.
    assert n_storage < 2 ** 31, (
        f"n_storage={n_storage:,} >= 2**31: int32 storage positions would wrap "
        f"silently. Raise n_chunks (so each chunk stays under 2.15 B halos) or "
        f"switch _merger_mapping / tree_subsample's `owner` to int64.")
    _merger_mapping = np.full(n_storage, -1, dtype=np.int32)
    _bh_phys = np.empty(n_storage, dtype=np.float32)
    _Lbol_phys = np.empty(n_storage, dtype=np.float32)
    _halo_phys = np.empty(n_storage, dtype=np.float32)
    _edd = np.zeros(n_storage, dtype=np.float32)

    catalog_builder = MergerCatalogBuilder() if (merger_on and record_merger_catalog) else None

    for i in snapshots:
        i = int(i)
        if time_step_for_accretion == 0:
            n_steps = 0
        else:
            n_steps = max(1, int(delta_times_snapshots[i] * 1e3 / time_step_for_accretion))

        current_halo_masses = halo_masses_all[i]
        if accretion_on:
            current_rates = halo_specific_cold_accretion_rates_all[i]

        idx_spawning = precomp_spawning[i]
        idx_evolving = precomp_evolving[i]

        # Fresh-zero each snapshot — matches main_evolution.py. A halo that died
        # (disrupted or merged into a target) is ZERO from its death snapshot
        # onward. Do NOT carry bh_prev forward (that would keep dead halos'
        # masses forever and double-count merged sources).
        bh_curr[:] = 0.0
        Lbol_curr[:] = 0.0

        # STEP 1 — seed new BHs
        if idx_spawning.size > 0:
            sp = _to_storage(idx_spawning)
            bh_curr[sp] = spawn_BHs(current_halo_masses[sp], logfseed,
                                    sigmaseed=sigmaseed, rng=rng_run)

        # STEP 2 — accretion
        n_evolving = idx_evolving.size
        ev = _to_storage(idx_evolving)
        if accretion_on and n_evolving > 0:
            bh_in = _buf_bh_prev[:n_evolving]
            bh_in[:] = bh_prev[ev]
            log_halo_rates = _buf_log_rates[:n_evolving]
            np.take(current_rates, ev, out=log_halo_rates)
            np.clip(log_halo_rates, 0.0 if nonaccreting_zero_rate else 1e-8, 1e3,
                    out=log_halo_rates)
            with np.errstate(divide="ignore"):
                np.log10(log_halo_rates, out=log_halo_rates)
            bh_out, Lbol_out = evolve_BHs(
                transfer_function, bh_in, delta_times_snapshots[i], erdf,
                n_steps=n_steps, log_halo_rates_array=log_halo_rates,
                rad_efficiency=rad_efficiency_0, rad_efficiency_model=rad_efficiency_model,
                parallelize=False, num_workers=1, backend="threading",
                use_parallel_kernel=use_parallel_kernel,
                fw_p_threshold=0.0,  # exact table (fiducial); 0.7 default = FW approx
                L_bols_out=_buf_Lbols_out[:n_evolving], rng_buffers=_rng_buffers,
                growth_max=growth_sum_max, madau_feff_correction=madau_feff_correction,
            )
            bh_curr[ev] = bh_out
            Lbol_curr[ev] = Lbol_out
        elif (not accretion_on) and n_evolving > 0:
            bh_curr[ev] = bh_prev[ev]

        # STEP 3 — mergers + disruptions (with merger-catalogue recording)
        if merger_on:
            idx_merging = precomp_merging[i]
            if idx_merging.size > 0:
                src_ids = mt_track_ids[idx_merging]
                dest_ids = mt_merger_ids[idx_merging]
                # Targets must be resolved AND still alive: halo mass is
                # frozen LastMaxMass ("ever resolved"), so
                # a long-dead target would swallow the source BH into a row
                # that is never carried forward. Dying-at-this-snap targets
                # are kept (they are sources; chain resolution forwards).
                dest_death = mt_death_index[dest_ids]
                # A target that is not in storage maps to -1, which NumPy
                # would read as the LAST storage slot: the deposit would land
                # on an unrelated halo and the catalogue would record that
                # halo's mass as M2. Never-resolved targets are exactly this
                # case in the multinode partition (it intersects the closure
                # with resolved_mask, so the source is in storage and the
                # target is not). Abort such mergers (mass lost), like the
                # unresolved-target ones.
                dest_pos_all = _to_storage(dest_ids)
                in_storage = dest_pos_all >= 0
                alive = in_storage & (
                    current_halo_masses[np.where(in_storage, dest_pos_all, 0)] > 0.0
                ) & ((dest_death == -1) | (dest_death >= i))
                src_ids = src_ids[alive]
                dest_ids = dest_ids[alive]
                if src_ids.size > 0:
                    src_pos = _to_storage(src_ids)
                    dest_pos = _to_storage(dest_ids)
                    # Record binary events BEFORE chain resolution (immediate
                    # dest), pre-merger masses: src from prev, dest post-accretion.
                    if catalog_builder is not None:
                        catalog_builder.record_snapshot_mergers(
                            z_snap=float(redshifts[i]),
                            src_indices=src_ids, dest_indices=dest_ids,
                            src_masses=bh_prev[src_pos],
                            dest_masses_initial=bh_curr[dest_pos],
                            track_ids=mt_track_ids,
                        )
                    # Resolve chains A->B->C in storage space.
                    _merger_mapping[src_pos] = dest_pos
                    final_dest = dest_pos.copy()
                    for _ in range(src_pos.size):
                        nxt = _merger_mapping[final_dest]
                        chain = nxt != -1
                        if not np.any(chain):
                            break
                        final_dest[chain] = nxt[chain]
                    _merger_mapping[src_pos] = -1
                    np.add.at(bh_curr, final_dest, bh_prev[src_pos])

        # On-the-fly summary statistics at the stat grid (weight ≡ 1.0).
        if i in snap_to_zindex:
            iz = snap_to_zindex[i]
            np.multiply(bh_curr, mass_units_f32, out=_bh_phys)
            np.multiply(Lbol_curr, mass_units_f32, out=_Lbol_phys)
            np.multiply(current_halo_masses, mass_units_f32, out=_halo_phys)
            _edd[:] = 0.0
            valid_bh = _bh_phys > 0.0
            _edd[valid_bh] = (_Lbol_phys[valid_bh] / _bh_phys[valid_bh]) / log_csi_10

            qlf, _ = np.histogram(_Lbol_phys, qlf_bins)
            with np.errstate(divide="ignore"):
                log_qlfs[iz, :] = np.where(qlf * qlf_normalization > 0.0,
                                           np.log10(qlf * qlf_normalization), -np.inf)
            bhmf, _ = np.histogram(_bh_phys[valid_bh], bhmf_bins)
            with np.errstate(divide="ignore"):
                log_bhmfs[iz, :] = np.where(bhmf * bhmf_normalization > 0.0,
                                            np.log10(bhmf * bhmf_normalization), -np.inf)

            above_lowest = _Lbol_phys > lowest_cut
            edd_above = _edd[above_lowest]
            halo_above = _halo_phys[above_lowest]
            Lbol_above = _Lbol_phys[above_lowest]
            for j in range(n_Lthr):
                if j == 0:
                    edd_sel, halo_sel = edd_above, halo_above
                else:
                    cut = Lbol_above > Lcuts[j]
                    edd_sel, halo_sel = edd_above[cut], halo_above[cut]
                with np.errstate(divide="ignore"):
                    cerdf, _ = np.histogram(edd_sel, cerdf_bins)
                    log_cerdfs[iz, j, :] = np.where(cerdf * cerdf_normalization > 0.0,
                                                    np.log10(cerdf * cerdf_normalization), -np.inf)
                    qhmf, _ = np.histogram(halo_sel, qhmf_bins)
                    log_qhmfs[iz, j, :] = np.where(qhmf * qhmf_normalization > 0.0,
                                                   np.log10(qhmf * qhmf_normalization), -np.inf)

        # Per-halo BH catalogue at anchor snapshots (M_BH > 0 only).
        if anchor_sink is not None and i in anchor_set:
            live = np.flatnonzero(bh_curr > 0.0)  # storage positions
            anchor_sink(
                i, float(redshifts[i]),
                subset_indices[live].astype(np.int64),
                (bh_curr[live] * mass_units_f32).astype(np.float32),
                (Lbol_curr[live] * mass_units_f32).astype(np.float32),
            )

        bh_prev, bh_curr = bh_curr, bh_prev

    # Survivors at the final snapshot — bh_prev holds the last state after swap.
    survivor_storage = np.flatnonzero(bh_prev > 0.0)
    survivor_track_ids = mt_track_ids[subset_indices[survivor_storage]].astype(np.int64)
    survivor_M_z0_code = bh_prev[survivor_storage].astype(np.float32)

    return {
        "log_qlfs": log_qlfs,
        "log_bhmfs": log_bhmfs,
        "log_cerdfs": log_cerdfs,
        "log_qhmfs": log_qhmfs,
        "merger_catalog": catalog_builder.finalize() if catalog_builder is not None else None,
        "survivor_track_ids": survivor_track_ids,
        "survivor_M_z0_code": survivor_M_z0_code,
    }


# ==============================================================================
# DRIVER — config resolution, file I/O, output writers
# ==============================================================================
def main():
    """Run ONE chunk of the multinode whole-catalogue forward model.

    The chunk index and count come from ``BAQARO_CHUNK_INDEX`` /
    ``BAQARO_N_CHUNKS``; every node runs this with a different index and writes
    its own stats, catalog and merger files. Combine them afterwards with
    ``concat_chunks.py``.
    """
    import h5py
    from qhtools.utils.cosmology import cosmo
    from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    from baqaro.core_functions.erdf_core_functions import Erdf
    from baqaro.core_functions.tree_subsample import (
        build_multinode_partition_subset, make_multinode_partition_tag,
        extract_subset_slice_cached, precompute_per_snapshot_indices,
    )
    from baqaro.utils.logging import set_logger
    from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
    from baqaro.utils.local_utils import get_mass_resolution_simulation
    from baqaro.utils.my_units import mass_units
    from baqaro.utils.provenance import write_run_provenance

    t_global = time.time()

    # ---------------------------------------------------------------- config
    source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
    base_rng_seed = int(os.environ.get("BAQARO_RNG_SEED", "12345"))

    if env_bool("BAQARO_USE_SUBSAMPLE", False):
        raise SystemExit(
            "main_evolution_chunked is the no-subsample multinode path; set "
            "BAQARO_USE_SUBSAMPLE=0. For subsampled runs use main_evolution.py."
        )
    n_chunks = int(os.environ.get("BAQARO_N_CHUNKS", "1"))
    chunk_id = int(os.environ.get("BAQARO_CHUNK_ID", "0"))
    if n_chunks < 1 or not (0 <= chunk_id < n_chunks):
        raise ValueError(f"bad BAQARO_N_CHUNKS={n_chunks} / BAQARO_CHUNK_ID={chunk_id}")
    print(f"MULTINODE forward run: chunk {chunk_id} of {n_chunks}", flush=True)

    # Identity / naming — single source of truth (shared with concat_chunks).
    identity = resolve_identity()
    simulation_name = identity["simulation_name"]
    erdf_model = identity["erdf_model"]
    max_snap = identity["max_snap"]
    min_snap = identity["min_snap"]
    fold_subhalo_mass = identity["fold_subhalo_mass"]
    merger_delay_mode = identity["merger_delay_mode"]
    tdyn_fraction_default = identity["tdyn_fraction_default"]
    nbound_threshold = identity["nbound_threshold"]
    halo_filtering_mode = identity["halo_filtering_mode"]
    notes_file = identity["notes_file"]
    bestfit_name = identity["bestfit_name"]
    nonaccreting_zero_rate = identity["nonaccreting_zero_rate"]
    growth_sum_max = identity["growth_sum_max"]
    madau_feff_correction = identity["madau_feff_correction"]
    subsample_root_snap = identity["subsample_root_snap"]
    _registry_entry = identity["registry_entry"]

    # 6 free params — three-tier resolution (env > registry entry > defaults).
    _DEFAULTS = {
        "log_eta_mean_0": -0.866391, "log_eta_mean_evol": 0.812765,
        "std_0": 0.444504, "logfseed": -5.669447, "sigmaseed": 0.461684,
        "logtcoherence": 6.060304,
    }

    def _resolve(p, env):
        if env in os.environ:
            return float(os.environ[env])
        if _registry_entry is not None and p in _registry_entry:
            return float(_registry_entry[p])
        return float(_DEFAULTS[p])

    erdf_params_dict = {
        "log_eta_mean_0":    _resolve("log_eta_mean_0",    "BAQARO_LOG_ETA_MEAN_0"),
        "log_eta_mean_evol": _resolve("log_eta_mean_evol", "BAQARO_LOG_ETA_MEAN_EVOL"),
        "std_0":             _resolve("std_0",             "BAQARO_STD_0"),
    }
    logfseed = _resolve("logfseed", "BAQARO_LOGFSEED")
    sigmaseed = _resolve("sigmaseed", "BAQARO_SIGMASEED")
    logtcoherence = _resolve("logtcoherence", "BAQARO_LOGTCOHERENCE")
    time_step_for_accretion = 10 ** (logtcoherence - 6)
    if env_bool("BAQARO_TAU_COH_ZERO", False):
        time_step_for_accretion = 0.0
    print(f"Params: {erdf_params_dict}, logfseed={logfseed}, sigmaseed={sigmaseed}, "
          f"logtcoherence={logtcoherence} -> tstep={time_step_for_accretion} Myr", flush=True)

    accretion_on = True
    merger_on = env_bool("BAQARO_MERGER_ON", True)
    rad_efficiency_0 = 0.1
    rad_efficiency_model = os.environ.get("BAQARO_RAD_EFFICIENCY_MODEL", "madau+")

    from baqaro.utils.sim_config import boxsize, N_particles_per_side

    path_sim = get_input_path_HBT_data(source=source_dir)
    path_out = get_output_path(source=source_dir)
    get_mass_resolution_simulation(boxsize, N_particles_per_side)  # validates config
    redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")

    multinode_tag = make_multinode_partition_tag(
        root_snap=subsample_root_snap, chunk_id=chunk_id, n_chunks=n_chunks)
    paths = output_paths(identity, chunk_id, n_chunks, path_out)
    name_file = paths["name_file"]
    path_stats, path_catalog, path_merger = paths["stats"], paths["catalog"], paths["merger"]
    os.makedirs(os.path.join(path_out, "evolution"), exist_ok=True)
    logger = set_logger(os.path.join(path_out, "logs", f"bh_evolution_chunked_{name_file}.log"))

    # ---------------------------------------------------------- halo data
    snapshots = np.arange(min_snap, max_snap + 1)
    redshifts = np.asarray(np.loadtxt(redshift_file))[snapshots]
    ages = cosmo.age(redshifts)
    delta_times_snapshots = np.diff(ages, prepend=0.01)

    name_file_halos = (
        f"{simulation_name}_maxsnap{max_snap}_nboundthresh{nbound_threshold}"
        f"_halofilter_{halo_filtering_mode}_tdynfraction_{tdyn_fraction_default}"
    )
    if fold_subhalo_mass:
        name_file_halos += "_foldmass"
    if merger_delay_mode != "instant_old":
        name_file_halos += f"_{merger_delay_mode}"
    hist_dir = os.path.join(path_out, "halo_histories")
    path_file_halo_masses = os.path.join(hist_dir, f"halo_masses_{name_file_halos}.npy")
    path_file_cold = os.path.join(hist_dir, f"specific_cold_accretion_rates_{name_file_halos}.npy")
    path_file_trees = os.path.join(hist_dir, f"merger_trees_{name_file_halos}")

    print(f"Loading merger trees from {path_file_trees}", flush=True)
    mt = MergerTreeLoader(path_file_trees)
    mt_birth_index = np.array(mt.snapshot_indexes_of_birth)
    mt_death_index = np.array(mt.snapshot_indexes_of_death)
    mt_track_ids = np.array(mt.track_ids)
    mt_merger_ids = np.array(mt.merger_track_ids)
    mask_merged = mt_merger_ids != -1
    mask_disrupted = mt_merger_ids == -1

    rev_path = os.path.join(hist_dir, f"reverse_index_{name_file_halos}.npz")
    if os.path.exists(rev_path):
        print(f"Loading reverse index from {rev_path}", flush=True)
        rd = np.load(rev_path)
        reverse_index = (rd["indptr"], rd["indices"])
    else:
        from baqaro.core_functions.select_merger_branches import build_reverse_index
        print("Building reverse index (cache via prep_reverse_index)...", flush=True)
        reverse_index = build_reverse_index(mt_merger_ids)

    print("Reading halo masses at root snapshot for resolved_mask...", flush=True)
    _hm = np.load(path_file_halo_masses, mmap_mode="r")
    masses_at_root = np.array(_hm[subsample_root_snap]).astype(np.float64) * mass_units
    del _hm
    # resolved_mask = ever crossed the resolution threshold. HBT's LastMaxMass
    # freezes at peak after death, so the z=0 mass row is > 0 for every halo
    # that was ever resolved (including disrupted dead-ends). The terminal-rooted
    # partition then covers the ENTIRE resolved catalogue.
    resolved_mask = masses_at_root > 0
    del masses_at_root

    t_sub = time.time()
    subset = build_multinode_partition_subset(
        track_ids=mt_track_ids, merger_track_ids=mt_merger_ids,
        resolved_mask=resolved_mask, chunk_id=chunk_id, n_chunks=n_chunks,
        reverse_index=reverse_index,
    )
    subset_indices = np.flatnonzero(subset["subset_mask"])
    n_subset = int(len(subset_indices))
    n_full = int(len(subset["subset_mask"]))
    print(f"Chunk {chunk_id}/{n_chunks}: {n_subset:,} halos "
          f"({100*n_subset/n_full:.3f}% of {n_full:,}) in {time.time()-t_sub:.1f}s", flush=True)
    global_to_storage = np.full(n_full, -1, dtype=np.int64)
    global_to_storage[subset_indices] = np.arange(n_subset, dtype=np.int64)

    subsets_dir = os.path.join(path_out, "halo_subsets")
    os.makedirs(subsets_dir, exist_ok=True)
    print("Eager-loading chunk halo data into RAM...", flush=True)
    t_e = time.time()
    halo_masses_all = extract_subset_slice_cached(
        full_path=path_file_halo_masses, subset_indices=subset_indices,
        cache_path=os.path.join(subsets_dir, f"subset_{name_file_halos}_{multinode_tag}__halo_masses.npy"))
    halo_cold_all = extract_subset_slice_cached(
        full_path=path_file_cold, subset_indices=subset_indices,
        cache_path=os.path.join(subsets_dir, f"subset_{name_file_halos}_{multinode_tag}__specific_cold_accretion_rates.npy"))
    print(f"  Eager-loaded {(halo_masses_all.nbytes+halo_cold_all.nbytes)/1e9:.1f} GB in {time.time()-t_e:.1f}s", flush=True)

    print("Precomputing per-snapshot index arrays...", flush=True)
    _pre = precompute_per_snapshot_indices(
        min_snap=min_snap, max_snap=max_snap,
        mt_birth_index=mt_birth_index, mt_death_index=mt_death_index,
        mask_merged=mask_merged, mask_disrupted=mask_disrupted,
        subset_indices=subset_indices)

    # ------------------------------------------------------ stats grid + bins
    # This is the MULTINODE forward-run grid (distinct from the training grid;
    # see utils/sim_config.py). Overridable per run via
    # BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS; falls back to the training grid when
    # the SimSpec leaves the multinode grid unset.
    from baqaro.utils.sim_config import multinode_evolution_snapshots_default
    snapshots_to_save = [s for s in multinode_evolution_snapshots_default if s <= max_snap]
    snap_to_zindex = {int(s): iz for iz, s in enumerate(snapshots_to_save)}
    n_z = len(snapshots_to_save)
    _anchor_env = os.environ.get("BAQARO_CATALOG_ANCHOR_SNAPS", "").strip()
    if _anchor_env:
        anchor_snaps = sorted(int(s) for s in _anchor_env.split(",") if s != "")
        bad = [s for s in anchor_snaps if s not in snap_to_zindex]
        if bad:
            raise ValueError(f"BAQARO_CATALOG_ANCHOR_SNAPS {bad} not in snapshots_to_save")
        anchor_set = set(anchor_snaps)
    else:
        anchor_set = set(snap_to_zindex)
    print(f"snapshots_to_save: {snapshots_to_save}; anchors: {sorted(anchor_set)}", flush=True)

    num_lbins = num_mbins = 40
    log_L_thresholds = np.linspace(45.5, 47.5, 21)
    qlf_bins = np.logspace(my_utils.to_solar(44), my_utils.to_solar(48.5), num_lbins + 1)
    qlf_normalization = 1.0 / ((np.log10(qlf_bins[1]) - np.log10(qlf_bins[0])) * boxsize ** 3)
    bhmf_bins = np.logspace(6.5, 11.0, num_mbins + 1)
    bhmf_normalization = 1.0 / ((np.log10(bhmf_bins[1]) - np.log10(bhmf_bins[0])) * boxsize ** 3)
    cerdf_bins = np.logspace(-3, 2, num_mbins + 1)
    cerdf_normalization = 1.0 / ((np.log10(cerdf_bins[1]) - np.log10(cerdf_bins[0])) * boxsize ** 3)
    from baqaro.utils.sim_config import (
        log_M_halo_qhmf_lo as _qlo, log_M_halo_qhmf_hi as _qhi)
    qhmf_lo = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_LO", _qlo))
    qhmf_hi = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_HI", _qhi))
    qhmf_bins = np.logspace(qhmf_lo, qhmf_hi, num_mbins + 1)
    qhmf_normalization = 1.0 / ((np.log10(qhmf_bins[1]) - np.log10(qhmf_bins[0])) * boxsize ** 3)
    Lcuts = 10 ** my_utils.to_solar(log_L_thresholds.astype(np.float64))

    qlf_c = np.sqrt(qlf_bins[1:] * qlf_bins[:-1])
    bhmf_c = np.sqrt(bhmf_bins[1:] * bhmf_bins[:-1])
    cerdf_c = np.sqrt(cerdf_bins[1:] * cerdf_bins[:-1])
    qhmf_c = np.sqrt(qhmf_bins[1:] * qhmf_bins[:-1])

    # ------------------------------------------------------ transfer + erdf
    transfer_function = None if time_step_for_accretion == 0 else \
        load_3d_sampler("Universal_Lognormal_Sampler_final.npz")
    if env_bool("BAQARO_FORCE_NO_SAMPLER", False):
        transfer_function = None
    rng_run = _make_run_rng(base_rng_seed, chunk_id)
    erdf = Erdf(erdf_model, erdf_params_dict, rng=rng_run)

    # ------------------------------------------------------ run + stream catalog
    cat_file = h5py.File(path_catalog, "w")
    cat_file.attrs["name_file"] = name_file
    cat_file.attrs["chunk_id"] = int(chunk_id)
    cat_file.attrs["n_chunks"] = int(n_chunks)
    cat_file.attrs["mass_units"] = float(mass_units)
    cat_file.attrs["note"] = ("Per-halo BH catalogue (M_BH>0) at anchor snapshots. "
                              "track_id GLOBAL; M_BH, Lbol physical (code*mass_units).")

    def _anchor_sink(snap, z, track_id, M_BH, Lbol):
        g = cat_file.create_group(f"snap_{snap:03d}")
        g.attrs["snapshot"] = int(snap)
        g.attrs["redshift"] = float(z)
        g.create_dataset("track_id", data=track_id)
        g.create_dataset("M_BH", data=M_BH)
        g.create_dataset("Lbol", data=Lbol)
        cat_file.flush()
        print(f"  [catalog] snap {snap} (z={z:.3f}): {track_id.size:,} live BHs", flush=True)

    print(f"Starting evolution: n_storage={n_subset}, n_snaps={len(snapshots)}", flush=True)
    t_loop = time.time()
    out = evolve_chunk(
        subset_indices=subset_indices, global_to_storage=global_to_storage, n_storage=n_subset,
        halo_masses_all=halo_masses_all, halo_specific_cold_accretion_rates_all=halo_cold_all,
        mt_track_ids=mt_track_ids, mt_merger_ids=mt_merger_ids,
        mt_death_index=mt_death_index,
        precomp_spawning=_pre["spawning"], precomp_evolving=_pre["evolving"],
        precomp_merging=_pre["merging"],
        snapshots=snapshots, snap_to_zindex=snap_to_zindex, n_z=n_z, anchor_set=anchor_set,
        redshifts=redshifts, delta_times_snapshots=delta_times_snapshots,
        erdf=erdf, rng_run=rng_run, transfer_function=transfer_function,
        time_step_for_accretion=time_step_for_accretion, accretion_on=accretion_on,
        merger_on=merger_on, nonaccreting_zero_rate=nonaccreting_zero_rate,
        rad_efficiency_0=rad_efficiency_0, rad_efficiency_model=rad_efficiency_model,
        growth_sum_max=growth_sum_max, madau_feff_correction=madau_feff_correction,
        logfseed=logfseed, sigmaseed=sigmaseed,
        qlf_bins=qlf_bins, qlf_normalization=qlf_normalization,
        bhmf_bins=bhmf_bins, bhmf_normalization=bhmf_normalization,
        cerdf_bins=cerdf_bins, cerdf_normalization=cerdf_normalization,
        qhmf_bins=qhmf_bins, qhmf_normalization=qhmf_normalization,
        log_L_thresholds=log_L_thresholds, Lcuts=Lcuts, lowest_cut=Lcuts[0],
        log_csi_10=10 ** nc.log_csi, mass_units_f32=np.float32(mass_units),
        anchor_sink=_anchor_sink, record_merger_catalog=merger_on, logger=logger,
    )
    cat_file.close()
    print(f"Evolution loop done in {time.time()-t_loop:.1f}s", flush=True)

    # ------------------------------------------------------ write stats
    common_attrs = {
        "engine": "main_evolution_chunked", "simulation_name": simulation_name,
        "max_snap": int(max_snap), "min_snap": int(min_snap), "erdf_model": erdf_model,
        "fold_subhalo_mass": bool(fold_subhalo_mass), "merger_delay_mode": merger_delay_mode,
        "tdyn_fraction_default": float(tdyn_fraction_default),
        "nbound_threshold": int(nbound_threshold), "halo_filtering_mode": halo_filtering_mode,
        "notes_file": notes_file, "bestfit_name": bestfit_name, "name_file": name_file,
        "use_subsample": False, "subset_tag": multinode_tag,
        "multinode_chunk_id": int(chunk_id), "multinode_n_chunks": int(n_chunks),
        "rng_option": "A_per_chunk_stream", "rng_base_seed": int(base_rng_seed),
        "merger_on": bool(merger_on), "accretion_on": bool(accretion_on),
        "rad_efficiency_model": rad_efficiency_model, "rad_efficiency_0": float(rad_efficiency_0),
        "time_step_for_accretion_Myr": float(time_step_for_accretion),
        "logtcoherence": float(logtcoherence),
        "log_eta_mean_0": float(erdf_params_dict["log_eta_mean_0"]),
        "log_eta_mean_evol": float(erdf_params_dict["log_eta_mean_evol"]),
        "std_0": float(erdf_params_dict["std_0"]), "logfseed": float(logfseed),
        "sigmaseed": float(sigmaseed) if sigmaseed is not None else None,
        "growth_sum_max": float(growth_sum_max),
        "madau_feff_correction": bool(madau_feff_correction),
    }
    print(f"Writing stats to {path_stats}", flush=True)
    with h5py.File(path_stats, "w") as f:
        f.create_dataset("log_qlfs", data=out["log_qlfs"])
        f.create_dataset("log_bhmfs", data=out["log_bhmfs"])
        f.create_dataset("log_cerdfs", data=out["log_cerdfs"])
        f.create_dataset("log_qhmfs", data=out["log_qhmfs"])
        f.create_dataset("snapshots_to_save", data=np.asarray(snapshots_to_save, dtype=np.int64))
        f.create_dataset("redshifts_saved", data=np.asarray([redshifts[s] for s in snapshots_to_save]))
        f.create_dataset("log_lbins", data=my_utils.to_ergs(np.log10(qlf_c)))
        f.create_dataset("log_mbins_bhmf", data=np.log10(bhmf_c))
        f.create_dataset("log_etabins", data=np.log10(cerdf_c))
        f.create_dataset("log_mbins_qhmf", data=np.log10(qhmf_c))
        f.create_dataset("log_L_threshold", data=log_L_thresholds)
        write_run_provenance(f, common_attrs)

    # ------------------------------------------------------ write merger catalog
    cat = out["merger_catalog"]
    if cat is not None:
        survivor_track_ids = out["survivor_track_ids"]
        survivor_M_z0 = (out["survivor_M_z0_code"].astype(np.float64) * mass_units).astype(np.float32)
        print(f"Writing merger catalogue to {path_merger} "
              f"({cat['z'].size:,} events, {survivor_track_ids.size:,} survivors)", flush=True)
        with h5py.File(path_merger, "w") as f:
            f.create_dataset("z", data=cat["z"])
            f.create_dataset("M1", data=(cat["M1_Msun"].astype(np.float64) * mass_units).astype(np.float32))
            f.create_dataset("M2", data=(cat["M2_Msun"].astype(np.float64) * mass_units).astype(np.float32))
            f.create_dataset("progenitor1_id", data=cat["progenitor1_id"])
            f.create_dataset("progenitor2_id", data=cat["progenitor2_id"])
            f.create_dataset("descendant_id", data=cat["descendant_id"])
            f.create_dataset("halo_id", data=cat["halo_id"])
            f.create_dataset("survivor_id", data=survivor_track_ids)
            f.create_dataset("survivor_M_z0", data=survivor_M_z0)
            f.attrs["V_sim_Mpc3"] = float(boxsize) ** 3
            f.attrs["boxsize"] = float(boxsize)
            f.attrs["boxsize_units_assumed"] = "Mpc (no h factor)"
            f.attrs["simulation_name"] = simulation_name
            f.attrs["max_snap"] = int(max_snap)
            f.attrs["min_snap"] = int(min_snap)
            f.attrs["notes_file"] = notes_file or ""
            f.attrs["erdf_model"] = erdf_model
            f.attrs["nbound_threshold"] = int(nbound_threshold)
            f.attrs["halo_filtering_mode"] = halo_filtering_mode
            f.attrs["tdyn_fraction_default"] = float(tdyn_fraction_default)
            f.attrs["catalog_schema_version"] = "1.0"
            f.attrs["multinode_chunk_id"] = int(chunk_id)
            f.attrs["multinode_n_chunks"] = int(n_chunks)

    print(f"Chunk {chunk_id}/{n_chunks} DONE in {time.time()-t_global:.1f}s total.", flush=True)


if __name__ == "__main__":
    main()
