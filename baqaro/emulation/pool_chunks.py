"""
Re-sum chunked training outputs into a single training HDF5 (Phase 1).
======================================================================

The chunked multi-node training path (``BAQARO_N_CHUNKS>1`` in
``main_training.py``) writes one training HDF5 per node, each holding the
weighted summary statistics of a disjoint root-partition of the N_b
subsample (filename carries a ``_chunk{k}of{N}`` token). This script pools
them back into ONE file written under the chunk-free single-node name, so
``main_emulation`` / ``loading_helpers`` consume it with no changes.

Why a plain SUM is exact (estimator level)
------------------------------------------
Each chunk's roots are a disjoint, complete partition of the single-run
roots, and every kept halo keeps the *full-N_b* inverse-probability weight
(``tree_subsample.build_subsampled_subset`` with ``n_chunks>1``; see its
docstring and ``tests/test_chunk_partition.py``). Hence for every bin

    Σ_chunks Σ_{i in chunk} w_i·1[x_i in bin]
  = Σ_{i in single-run subset} w_i·1[x_i in bin].

So summing the per-chunk weighted number-density histograms reproduces the
single-run weighted histogram. The summary stats are stored as
``log10(density)`` with empty bins = ``-inf`` (see ``run_single_model``);
we therefore un-log (``10**x``, ``-inf -> 0``), sum across chunks in
float64, and re-log with the identical empty-bin convention.

NOTE: this is *not* a bitwise copy of a single-node run — ``run_single_model``
draws ERDF etas from one per-model RNG stream shared across all evolving
halos, so per-halo BH masses differ between a chunk run and a single run
(different, equally-valid MC realization). The pooled file is a valid,
unbiased training realization; that is exactly what the emulator wants.

Row matching
------------
``run_id`` differs per chunk (the chunk changes ``n_storage`` in the hashed
config), so rows are matched by their **parameter vector** (the Sobol
sequence is identical across chunks). Only parameter points that are
``status==DONE`` in *every* chunk are pooled; any point missing or failed in
some chunk is written with ``status=-1`` (excluded by ``loading_helpers``).

Usage
-----
Env-driven (same env vars as the array job — recommended for a job array):

    BAQARO_SIM=L2800N10080 BAQARO_MAX_SNAP=144 BAQARO_USE_SUBSAMPLE=1 \
    BAQARO_SUBSAMPLE_NB=500000 BAQARO_NOTES_FILE=fid1_z0 BAQARO_N_CHUNKS=4 \
        python -m baqaro.emulation.pool_chunks

Explicit paths (manual / testing):

    python -m baqaro.emulation.pool_chunks \
        --out  COMBINED.hdf5  --chunks  CHUNK0.hdf5 CHUNK1.hdf5 ...
"""

import argparse
import os
import shutil
import sys

import h5py
import numpy as np

# Statistic datasets pooled across chunks.
_POOL_KEYS = ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs")
STATUS_DONE = 1
STATUS_FAILED = -1

# Parameter-vector match granularity (matches make_run_id's dedup decimals;
# distinct Sobol points are well separated, so this never collides).
_PARAM_DECIMALS = 6


# ----------------------------------------------------------------------
# Path derivation (mirrors main_training.py naming)
# ----------------------------------------------------------------------
def _subsample_geometry_from_env():
    """Return (N_b, n_bins, log_M_lo, log_M_hi, keep_all_above, rng_seed) the
    same way main_training.py derives them (env + SimSpec defaults)."""
    from baqaro.core_functions.tree_subsample import resolve_subsample_params
    p = resolve_subsample_params()
    return p["N_b"], p["n_bins"], p["log_M_lo"], p["log_M_hi"], p["keep_all_above_log_M"], p["seed"]


def training_path(*, notes, max_snap, chunk_id=0, n_chunks=1, source_dir=None):
    """Full path of a training HDF5, mirroring main_training.py naming.

    Single source of truth for the chunk/combined/reference filenames used by
    this script AND compare_training.py. The constants below (erdf_model,
    fold_subhalo_mass, merger_delay_mode) are copied verbatim from
    main_training.py — keep in sync. ``chunk_id``/``n_chunks`` feed
    make_subset_tag, so n_chunks=1 yields the chunk-free single-node name.
    Subset geometry (N_b, bins, mass range, keep_above) is read from env.
    """
    from baqaro.utils.my_dir import get_output_path
    from baqaro.utils.sim_config import (
        simulation_name,
        fold_subhalo_mass,
        merger_delay_mode,
        growth_feff_suffix,
    )
    from baqaro.core_functions.tree_subsample import make_subset_tag

    source_dir = source_dir or os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
    path_out = get_output_path(source=source_dir)
    erdf_model = "log_normal_evol_halo_mass"

    N_b, n_bins, log_M_lo, log_M_hi, keep_all_above, rng_seed = _subsample_geometry_from_env()
    tag = make_subset_tag(
        root_snap=max_snap, N_b=N_b, n_bins=n_bins,
        log_M_lo=log_M_lo, log_M_hi=log_M_hi, seed=rng_seed,
        keep_all_above_log_M=keep_all_above, chunk_id=chunk_id, n_chunks=n_chunks,
    )
    nf = "{}_erdf_{}_maxsnap_{}".format(simulation_name, erdf_model, max_snap)
    if fold_subhalo_mass:
        nf += "_foldmass"
    if merger_delay_mode != "instant_old":
        nf += "_{}".format(merger_delay_mode)
    if notes is not None:
        nf += "_{}".format(notes)
    nf += f"_sub_{tag}"
    # Growth-cap + madau-feff tokens (_g before _feffcorr), single canonical
    # home in utils.sim_config.growth_feff_suffix. Without these the pooler
    # would derive the uncapped/uncorrected name and fail to find capped or
    # _feffcorr chunk files.
    nf += growth_feff_suffix()
    return os.path.join(path_out, "training", f"training_data_emulation_{nf}.hdf5")


def derive_paths_from_env():
    """Reconstruct the chunk + combined training HDF5 paths from env vars.

    Combined path uses the chunk-free (single-node) name; chunk paths carry
    the ``_chunk{k}of{N}`` token. Each chunk path is existence-checked so a
    naming drift fails loudly rather than silently pooling nothing.
    """
    notes = os.environ.get("BAQARO_NOTES_FILE") or None
    from baqaro.utils.sim_config import max_snap, env_bool
    use_subsample = env_bool("BAQARO_USE_SUBSAMPLE", False)
    n_chunks = int(os.environ.get("BAQARO_N_CHUNKS", "1"))
    if n_chunks < 2:
        raise SystemExit(
            f"pool_chunks: BAQARO_N_CHUNKS={n_chunks} — nothing to pool "
            "(need >= 2). Did you forget to set it?"
        )
    if not use_subsample:
        raise SystemExit("pool_chunks: BAQARO_USE_SUBSAMPLE=1 required (Phase 1).")

    combined_path = training_path(notes=notes, max_snap=max_snap, n_chunks=1)
    chunk_paths = [
        training_path(notes=notes, max_snap=max_snap, chunk_id=c, n_chunks=n_chunks)
        for c in range(n_chunks)
    ]
    missing = [p for p in chunk_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "pool_chunks: the following expected chunk files are missing — "
            "check that the array job completed and the naming matches:\n  "
            + "\n  ".join(missing)
        )
    return combined_path, chunk_paths


# ----------------------------------------------------------------------
# Pooling core
# ----------------------------------------------------------------------
def _param_key(params_row):
    return tuple(np.round(np.asarray(params_row, dtype=np.float64), _PARAM_DECIMALS))


def _done_param_index(f):
    """Map param-key -> row index for status==DONE rows of file ``f``."""
    statuses = f["runs/status"][:]
    params = f["runs/params"][:]
    idx = {}
    for row in np.flatnonzero(statuses == STATUS_DONE):
        idx[_param_key(params[row])] = int(row)
    return idx


def _info_n_storage(f):
    """Best-effort read of n_storage from the file's config JSON (info only)."""
    try:
        import json
        cfg = json.loads(f["schema/config_json"][()])
        return int(cfg.get("n_storage", -1))
    except Exception:
        return -1


def pool(chunk_paths, combined_path, verbose=True):
    """Pool per-chunk training HDF5s into ``combined_path``.

    Returns (n_pooled, n_incomplete).
    """
    n_chunks = len(chunk_paths)
    if n_chunks < 2:
        raise ValueError("pool() needs >= 2 chunk files")

    if verbose:
        print(f"Pooling {n_chunks} chunks -> {combined_path}")
        for c, p in enumerate(chunk_paths):
            print(f"  chunk {c}: {os.path.basename(p)}")

    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    tmp_path = combined_path + ".tmp"

    # Build per-chunk DONE param-index (read-only).
    chunk_files = [h5py.File(p, "r") for p in chunk_paths]
    try:
        done_idx = [_done_param_index(f) for f in chunk_files]
        nstore = [_info_n_storage(f) for f in chunk_files]
        for c, di in enumerate(done_idx):
            print(f"  chunk {c}: {len(di)} DONE rows")
        if verbose and all(n > 0 for n in nstore):
            print(f"  per-chunk n_storage: {nstore}  (sum={sum(nstore):,})")

        # Seed the combined file (schema + metadata + emulation bins + the runs
        # table) from the MOST-COMPLETE chunk, NOT blindly chunk 0. The combined
        # file's row set is the seed chunk's row set; a point absent from the
        # seed never appears in the output. Seeding from the chunk with the most
        # DONE rows makes the common failure mode — one chunk crashed/resumed
        # with fewer points — correct: its missing points become status=-1
        # (not DONE-in-all) and ARE visible/counted, instead of being silently
        # truncated away (audit finding). We then overwrite pooled rows + status.
        seed = int(np.argmax([len(di) for di in done_idx]))

        # Loudly flag any residual divergence: points DONE in some chunk but
        # absent from the seed chunk's table cannot be emitted (the combined
        # table is the seed's). This only happens if chunks ran DIFFERENT Sobol
        # sets — in production they don't.
        union_keys = set().union(*[set(di) for di in done_idx])
        missing_from_seed = union_keys - set(done_idx[seed])
        if missing_from_seed:
            print(f"  [WARN] {len(missing_from_seed)} param point(s) are DONE in "
                  f"some chunk but ABSENT from the most-complete chunk {seed}; "
                  f"they will NOT appear in the combined file. This means the "
                  f"chunks computed DIFFERENT Sobol sets — check per-chunk "
                  f"BAQARO_NUM_SIMULATIONS / resume state.")
        elif verbose:
            print(f"  seeding combined file from chunk {seed} "
                  f"({len(done_idx[seed])} DONE rows; all chunks' points covered)")

        shutil.copyfile(chunk_paths[seed], tmp_path)

        with h5py.File(tmp_path, "r+") as out:
            out_params = out["runs/params"][:]
            out_status = out["runs/status"][:]
            n_rows = out_params.shape[0]

            # Pre-read each chunk's pooled datasets fully (they're modest:
            # n_rows x n_z x n_bins float32). Avoids per-row HDF5 reads.
            chunk_data = [
                {k: f[f"runs/{k}"][:] for k in _POOL_KEYS if f"runs/{k}" in f}
                for f in chunk_files
            ]
            keys_present = [k for k in _POOL_KEYS if f"runs/{k}" in out]

            n_pooled = 0
            n_incomplete = 0
            new_status = out_status.copy()
            err_dset = out["runs/error"] if "runs/error" in out else None

            # Accumulators written back in bulk per key.
            pooled = {k: out[f"runs/{k}"][:] for k in keys_present}

            for row in range(n_rows):
                key = _param_key(out_params[row])
                rows_in_chunks = []
                ok = True
                for c in range(n_chunks):
                    r = done_idx[c].get(key)
                    if r is None:
                        ok = False
                        break
                    rows_in_chunks.append(r)
                if not ok:
                    new_status[row] = STATUS_FAILED
                    n_incomplete += 1
                    if err_dset is not None:
                        err_dset[row] = "pool: param point not DONE in all chunks"
                    continue

                for k in keys_present:
                    acc = None
                    for c in range(n_chunks):
                        x = chunk_data[c][k][rows_in_chunks[c]].astype(np.float64)
                        dens = np.power(10.0, x)        # -inf -> 0
                        acc = dens if acc is None else acc + dens
                    with np.errstate(divide="ignore"):
                        relog = np.where(acc > 0.0, np.log10(acc), -np.inf)
                    pooled[k][row] = relog.astype(np.float32)
                new_status[row] = STATUS_DONE
                n_pooled += 1

            for k in keys_present:
                out[f"runs/{k}"][:] = pooled[k]
            out["runs/status"][:] = new_status

            # Provenance.
            out.attrs["pooled_n_chunks"] = n_chunks
            out.attrs["pooled_from"] = np.array(
                [os.path.basename(p) for p in chunk_paths], dtype=object
            )
            # The seed chunk's `provenance/` group was copied in wholesale by
            # shutil.copyfile, so it describes a single chunk. Refresh it to
            # describe the COMBINED file: strip the `_chunkKofN` suffix from the
            # subset tag (the combined file is chunk-free) and mark it pooled.
            import re as _re
            if "provenance" in out:
                g = out["provenance"]
                _tag = _re.sub(r"_chunk\d+of\d+$", "", str(g.attrs.get("subset_tag", "")))
                g.attrs["subset_tag"] = _tag
                g.attrs["engine"] = "main_training_pooled"
                g.attrs["pooled_n_chunks"] = int(n_chunks)
                g.attrs["pooled_n_rows"] = int(n_pooled)
                out.attrs["subset_tag"] = _tag          # fix root mirror
                out.attrs["engine"] = "main_training_pooled"
            out.flush()
    finally:
        for f in chunk_files:
            f.close()

    os.replace(tmp_path, combined_path)
    if verbose:
        print(f"\nWrote {combined_path}")
        print(f"  pooled (DONE in all chunks): {n_pooled}")
        print(f"  incomplete (status=-1):      {n_incomplete}")
    return n_pooled, n_incomplete


def main(argv=None):
    """CLI entry point: pool per-chunk training HDF5s into one combined file.

    Either pass ``--chunks`` explicitly together with ``--out``, or give
    neither and let both be resolved from the environment, matching the names
    the chunked training run wrote.
    """
    ap = argparse.ArgumentParser(description="Pool chunked training HDF5s.")
    ap.add_argument("--out", help="combined output path (single-node name)")
    ap.add_argument("--chunks", nargs="+", help="explicit chunk HDF5 paths")
    args = ap.parse_args(argv)

    if args.chunks:
        if not args.out:
            raise SystemExit("--out is required when --chunks is given")
        combined_path, chunk_paths = args.out, args.chunks
        missing = [p for p in chunk_paths if not os.path.exists(p)]
        if missing:
            raise SystemExit("missing chunk files:\n  " + "\n  ".join(missing))
    else:
        combined_path, chunk_paths = derive_paths_from_env()

    pool(chunk_paths, combined_path)


if __name__ == "__main__":
    main()
