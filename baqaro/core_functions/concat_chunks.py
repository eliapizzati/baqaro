"""
Recombine multinode forward-run chunks into single combined files (Phase 2).
=============================================================================

``main_evolution_chunked`` writes, per chunk, three HDF5s carrying the
``_chunk{k}of{N}`` token (inside the ``multinode_*`` subset tag):

  * ``bh_evolution_stats_*``    — QLF/BHMF/CERDF/QHMF at the 13 snapshots,
  * ``bh_evolution_catalog_*``  — per-halo M_BH/L_bol (M_BH>0) per anchor,
  * ``merger_catalog_*``        — binary merger events + z=0 survivors.

This script combines them into the chunk-free single-node name. Three exact
recombines (the chunks are a disjoint + complete partition, weight ≡ 1.0):

  STATS  — additive: un-log (10**x, -inf→0), sum densities (float64), re-log.
  PER-HALO — assemble a STANDARD ``bh_evolution_{name_file}.hdf5`` (same schema
           as ``main_evolution.py``: ``black_hole_masses_all`` / ``Lbols_all``
           of shape ``(n_anchors, n_union)`` + a ``subset/`` group with
           ``subset_indices`` = the union of all live track_ids and ``weights``
           = 1) so the plotting ``DataLoader`` reads it with NO new code — it is
           just a 13-snapshot, weight-1 subsample run. Each chunk's per-anchor
           live BHs are scattered into the union columns (0 where not live).
  MERGER  — concatenate binary events; concatenate survivor lists.

Usage (env-driven — same env vars as the array job):

    BAQARO_USE_SUBSAMPLE=0 BAQARO_N_CHUNKS=16 BAQARO_BESTFIT_NAME=<fid> \
        python -m baqaro.core_functions.concat_chunks
"""

import os
import re
import h5py
import time
import contextlib

import numpy as np

from baqaro.core_functions.main_evolution_chunked import (
    resolve_identity, output_paths,
)

_STAT_KEYS = ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs")
_STAT_PASSTHROUGH = (
    "snapshots_to_save", "redshifts_saved", "log_lbins", "log_mbins_bhmf",
    "log_etabins", "log_mbins_qhmf", "log_L_threshold",
)


# ----------------------------------------------------------------------
# STATS — additive pool of the flat (n_z, n_bins) histograms
# ----------------------------------------------------------------------
def _pool_logdensity(arrays):
    """Un-log, sum densities (float64), re-log with -inf empty-bin convention."""
    acc = None
    for x in arrays:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(over="ignore"):
            dens = np.where(np.isfinite(x), np.power(10.0, x), 0.0)
        acc = dens if acc is None else acc + dens
    with np.errstate(divide="ignore"):
        return np.where(acc > 0.0, np.log10(acc), -np.inf).astype(np.float32)


def combine_stats(chunk_paths, out_path, verbose=True):
    """Pool per-chunk stats files -> ``out_path``."""
    pooled = {}
    with h5py.File(chunk_paths[0], "r") as f0:
        for k in _STAT_KEYS:
            stacks = []
            for p in chunk_paths:
                with h5py.File(p, "r") as f:
                    stacks.append(f[k][()])
            pooled[k] = _pool_logdensity(stacks)
        passthrough = {k: f0[k][()] for k in _STAT_PASSTHROUGH if k in f0}
        prov = dict(f0["provenance"].attrs) if "provenance" in f0 else {}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as out:
        for k in _STAT_KEYS:
            out.create_dataset(k, data=pooled[k])
        for k, v in passthrough.items():
            out.create_dataset(k, data=v)
        g = out.create_group("provenance")
        for k, v in prov.items():
            g.attrs[k] = v
        _tag = re.sub(r"_chunk\d+of\d+$", "", str(prov.get("subset_tag", "")))
        g.attrs["subset_tag"] = _tag
        g.attrs["engine"] = "main_evolution_chunked_combined"
        g.attrs["combined_n_chunks"] = len(chunk_paths)
        # Strip chunk-0-SPECIFIC keys. The block
        # above copies chunk 0's provenance wholesale, so the COMBINED file used to
        # advertise `multinode_chunk_id=0` — i.e. it described itself as chunk 0,
        # violating the self-describing-provenance contract and misleading anyone
        # doing forensics on it. (`combine_merger` already popped the id; the stats
        # path did not.)
        for _k in ("multinode_chunk_id", "chunk_id"):
            if _k in g.attrs:
                del g.attrs[_k]
        # The env snapshot is inherently chunk 0's (they differ only by CHUNK_ID);
        # say so rather than let it read as the combined run's own.
        if "swift_env_json" in g.attrs:
            g.attrs["swift_env_json_source"] = "chunk0 (chunks differ only by BAQARO_CHUNK_ID)"
    os.replace(tmp, out_path)
    if verbose:
        print(f"  stats   -> {os.path.basename(out_path)} ({len(chunk_paths)} chunks pooled)")


# ----------------------------------------------------------------------
# PER-HALO — assemble a standard bh_evolution_*.hdf5 (dense union-subset)
# ----------------------------------------------------------------------
def _anchor_redshift_time_arrays(stats_path, identity, source_dir):
    """Return (snapshots, redshifts, ages, delta_times) for the anchor snapshots,
    reading the saved snapshot list from the combined stats file and computing
    ages / per-snapshot Δt from the sim's output_list.txt (so Δt is the TRUE
    snapshot dt, not an anchor-to-anchor difference)."""
    from qhtools.utils.cosmology import cosmo
    from baqaro.utils.my_dir import get_input_path_HBT_data
    with h5py.File(stats_path, "r") as f:
        snaps = np.asarray(f["snapshots_to_save"]).astype(np.int64)
    path_sim = get_input_path_HBT_data(source=source_dir)
    zfile = os.path.join(path_sim, f"{identity['simulation_name']}/output_list.txt")
    z_full = np.asarray(np.loadtxt(zfile))
    ages_full = cosmo.age(z_full)
    dt_full = np.diff(ages_full, prepend=0.01)
    return snaps, z_full[snaps], ages_full[snaps], dt_full[snaps]


def _read_params_from_stats(stats_path):
    """Pull the run parameters out of the stats provenance group, in the shape
    the plotting DataLoader's load_simulation_metadata expects."""
    with h5py.File(stats_path, "r") as f:
        a = dict(f["provenance"].attrs) if "provenance" in f else {}
    return a


def combine_into_bh_evolution(catalog_paths, stats_path, out_path, identity,
                              source_dir, verbose=True):
    """Build a STANDARD bh_evolution_*.hdf5 from the per-chunk sparse catalogues.

    Resolves the anchor redshift/age/Δt arrays + run params, then delegates the
    union + dense scatter + write to :func:`write_dense_bh_evolution`.
    """
    snaps, redshifts, ages, dts = _anchor_redshift_time_arrays(
        stats_path, identity, source_dir)
    params = _read_params_from_stats(stats_path)
    write_dense_bh_evolution(catalog_paths, snaps, redshifts, ages, dts, params,
                             out_path, verbose=verbose)


def write_dense_bh_evolution(catalog_paths, snaps, redshifts, ages, dts, params,
                             out_path, verbose=True):
    """Assemble the dense standard ``bh_evolution`` file from per-chunk sparse
    catalogues + the resolved anchor arrays.

    Designed for the full z=0 box (union ~ few×10^9 halos) and for an
    ARBITRARILY DENSE anchor grid: peak RAM is independent of ``n_anchor``.

    Two passes, both streaming:
      1. union: exploit the fact that the chunks are a DISJOINT partition of the
         halos. Reduce each chunk to its own ``np.unique`` (bounded by the chunk
         size, ~n_full/n_chunks) and free that chunk's raw ids before moving on;
         the per-chunk uniques are disjoint, so the global union is just their
         concatenation, sorted.
      2. scatter: create the dense datasets on DISK, then fill them ONE ANCHOR
         ROW AT A TIME. All chunk files are held open across the anchor loop
         (via ExitStack), so this costs one file-open per chunk in total — it
         does NOT reintroduce the per-anchor reopen cost of the original
         row-by-row variant.

    MEMORY: ``union`` (8 B × n_union ≈ 18 GB at the full box) + two float32 row
    buffers (4 B × n_union ≈ 9 GB each) + one anchor's worth of chunk ids/values.
    ≈ 40–60 GB total, FLAT in n_anchor.

    The previous implementation materialised the whole ``(n_anchor, n_union)``
    array pair in RAM (``2 × n_anchor × n_union × 4 B``): 306 GB at 17 anchors
    (job peaked ~486 GB with the id bookkeeping) and ~740 GB at 41 — over the
    720 GB the recombine node requests, i.e. it OOMed rather than merely ran
    slowly. Hence the rewrite.
    """
    snaps = np.asarray(snaps).astype(np.int64)
    n_anchor = len(snaps)
    anchor_groups = [f"snap_{int(s):03d}" for s in snaps]
    nc = len(catalog_paths)

    # Pass 1 — union. Per-chunk unique (disjoint partition ⇒ the per-chunk
    # uniques are themselves disjoint), so we never hold every anchor's raw ids
    # at once. `np.unique` also sorts, and `np.searchsorted` below needs a
    # sorted union, so the final concatenation is sorted once.
    t0 = time.time()
    per_chunk = []
    for ci, p in enumerate(catalog_paths):
        ids = []
        with h5py.File(p, "r") as f:
            for sg in anchor_groups:
                if sg in f and "track_id" in f[sg]:
                    ids.append(f[sg]["track_id"][()])
        per_chunk.append(np.unique(np.concatenate(ids)) if ids
                         else np.empty(0, np.int64))
        del ids
        if verbose:
            print(f"  [union] chunk {ci + 1}/{nc}: {per_chunk[-1].size:,} live "
                  f"({time.time() - t0:.0f}s)", flush=True)
    union = np.sort(np.concatenate(per_chunk)) if per_chunk else np.empty(0, np.int64)
    del per_chunk
    n_union = int(union.size)
    if verbose:
        print(f"  union built: {n_union:,} live BHs in {time.time() - t0:.0f}s", flush=True)

    # Pass 2 — stream the dense scatter, one anchor ROW at a time. The datasets
    # are created up front (disk allocation, not RAM) and filled row by row.
    # M_BH / L_bol are already PHYSICAL (engine applied mass_units), matching
    # main_evolution's black_hole_masses_all / Lbols_all convention.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    t1 = time.time()
    with h5py.File(tmp, "w") as out:
        out.create_dataset("snapshots", data=snaps)
        out.create_dataset("redshifts", data=redshifts)
        out.create_dataset("ages_of_the_universe", data=ages)
        out.create_dataset("delta_times_snapshots", data=dts)
        d_bh = out.create_dataset("black_hole_masses_all",
                                  shape=(n_anchor, n_union), dtype=np.float32)
        d_lb = out.create_dataset("Lbols_all",
                                  shape=(n_anchor, n_union), dtype=np.float32)

        with contextlib.ExitStack() as stack:
            files = [stack.enter_context(h5py.File(p, "r")) for p in catalog_paths]
            row_bh = np.empty(n_union, dtype=np.float32)
            row_lb = np.empty(n_union, dtype=np.float32)
            for r, sg in enumerate(anchor_groups):
                row_bh[:] = 0.0
                row_lb[:] = 0.0
                for f in files:
                    if sg not in f or "track_id" not in f[sg]:
                        continue
                    cols = np.searchsorted(union, f[sg]["track_id"][()])  # union sorted
                    row_bh[cols] = f[sg]["M_BH"][()]
                    row_lb[cols] = f[sg]["Lbol"][()]
                d_bh[r, :] = row_bh
                d_lb[r, :] = row_lb
                if verbose:
                    print(f"  [scatter] anchor {r + 1}/{n_anchor} (snap {int(snaps[r])}) "
                          f"({time.time() - t1:.0f}s)", flush=True)
            del row_bh, row_lb

        # subset/ group — makes the DataLoader treat this as a (weight-1)
        # subsample: subset_indices are the GLOBAL track ids (== global halo
        # indices) so get_Halo_mass re-indexes the full-N halo_masses .npy.
        grp = out.create_group("subset")
        grp.create_dataset("subset_indices", data=union.astype(np.int64))
        grp.create_dataset("weights", data=np.ones(n_union, dtype=np.float64))
        grp.attrs["tag"] = re.sub(r"_chunk\d+of\d+$", "",
                                  str(params.get("subset_tag", "")))
        grp.attrs["multinode"] = True

        # parameters/ group — exactly what load_simulation_metadata reads.
        pg = out.create_group("parameters")
        pg.create_dataset("erdf_model", data=str(params.get("erdf_model",
                          "log_normal_evol_halo_mass")))
        eg = pg.create_group("erdf_params")
        for k in ("log_eta_mean_0", "log_eta_mean_evol", "std_0"):
            if k in params:
                eg.attrs[k] = float(params[k])
        if "logfseed" in params:
            pg.create_dataset("logfseed", data=float(params["logfseed"]))
        if params.get("sigmaseed") is not None:
            pg.create_dataset("sigmaseed", data=float(params["sigmaseed"]))
        if "time_step_for_accretion_Myr" in params:
            pg.create_dataset("time_step_for_accretion",
                              data=float(params["time_step_for_accretion_Myr"]))

        # provenance mirror (chunk-free).
        prov = out.create_group("provenance")
        for k, v in params.items():
            prov.attrs[k] = v
        prov.attrs["subset_tag"] = grp.attrs["tag"]
        prov.attrs["engine"] = "main_evolution_chunked_combined"
        prov.attrs["combined_n_chunks"] = len(catalog_paths)
    os.replace(tmp, out_path)
    if verbose:
        print(f"  bh_evol -> {os.path.basename(out_path)} "
              f"(dense {n_anchor}x{n_union}, weight-1 union-subset)")


# ----------------------------------------------------------------------
# MERGER CATALOGUE — concatenate events + survivors
# ----------------------------------------------------------------------
_MERGER_EVENT_KEYS = ("z", "M1", "M2", "progenitor1_id", "progenitor2_id",
                      "descendant_id", "halo_id")
_MERGER_SURVIVOR_KEYS = ("survivor_id", "survivor_M_z0")


def combine_merger(chunk_paths, out_path, verbose=True):
    """Concatenate per-chunk merger catalogues -> ``out_path``."""
    parts = {k: [] for k in _MERGER_EVENT_KEYS + _MERGER_SURVIVOR_KEYS}
    root_attrs = {}
    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            root_attrs = dict(f.attrs)
            for k in parts:
                if k in f:
                    parts[k].append(f[k][()])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as out:
        for k, chunks in parts.items():
            data = np.concatenate(chunks) if chunks else np.empty(0)
            out.create_dataset(k, data=data)
        for k, v in root_attrs.items():
            out.attrs[k] = v
        out.attrs["combined_n_chunks"] = len(chunk_paths)
        out.attrs.pop("multinode_chunk_id", None)
    os.replace(tmp, out_path)
    if verbose:
        n_ev = sum(len(a) for a in parts["z"])
        n_su = sum(len(a) for a in parts["survivor_id"])
        print(f"  merger  -> {os.path.basename(out_path)} ({n_ev:,} events, {n_su:,} survivors)")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def combine_all(*, chunk_path_sets, combined_paths, identity, source_dir, verbose=True):
    """Run all three combiners. ``chunk_path_sets`` maps
    {'stats','catalog','merger'} -> list of per-chunk paths; ``combined_paths``
    maps {'stats','bh_evolution','merger'} -> the single output path."""
    combine_stats(chunk_path_sets["stats"], combined_paths["stats"], verbose)
    combine_into_bh_evolution(chunk_path_sets["catalog"], combined_paths["stats"],
                              combined_paths["bh_evolution"], identity, source_dir,
                              verbose)
    if all(os.path.exists(p) for p in chunk_path_sets["merger"]):
        combine_merger(chunk_path_sets["merger"], combined_paths["merger"], verbose)
    elif verbose:
        print("  merger  -> skipped (no per-chunk merger catalogues found)")


def main():
    """Combine this run's ``BAQARO_N_CHUNKS`` chunk files into one set.

    Reads the chunk identity from the environment (the same variables the
    chunked forward run was launched with), locates the per-chunk stats,
    catalog and merger files, and writes the combined products alongside them.
    Requires at least two chunks.
    """
    from baqaro.utils.my_dir import get_output_path
    source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
    path_out = get_output_path(source=source_dir)
    n_chunks = int(os.environ.get("BAQARO_N_CHUNKS", "1"))
    if n_chunks < 2:
        raise SystemExit(f"concat_chunks: BAQARO_N_CHUNKS={n_chunks} — need >= 2 to combine.")

    identity = resolve_identity()
    chunk_path_sets = {kind: [] for kind in ("stats", "catalog", "merger")}
    for c in range(n_chunks):
        paths = output_paths(identity, c, n_chunks, path_out)
        for kind in chunk_path_sets:
            chunk_path_sets[kind].append(paths[kind])
    combined = output_paths(identity, 0, 1, path_out)  # n_chunks=1 → chunk-free name

    for kind in ("stats", "catalog"):
        missing = [p for p in chunk_path_sets[kind] if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                f"concat_chunks: missing {kind} chunk files:\n  " + "\n  ".join(missing))

    print(f"Combining {n_chunks} multinode chunks -> {combined['name_file']}")
    combine_all(chunk_path_sets=chunk_path_sets, combined_paths=combined,
                identity=identity, source_dir=source_dir)
    print("Done.")


if __name__ == "__main__":
    main()
