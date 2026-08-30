"""Write the training HDF5 incrementally and safely.

A training run is thousands of independent model evaluations spread over many
worker processes and, on a cluster, many nodes. It can be interrupted, resumed,
and extended, so the output file is built incrementally rather than assembled at
the end -- and the bookkeeping to make that safe is what lives here:

* stable per-run IDs, so a resumed run recognises what it already has
  (``make_run_id``, ``load_existing_run_ids``);
* a start/done/failed record per row, so an interrupted run leaves partial rows
  identifiable rather than silently truncated (``purge_incomplete_runs``);
* a schema written once and checked on reopen, so a file cannot be extended
  with rows that mean something different (``init_training_file``,
  ``write_training_metadata``).

The metadata block deliberately excludes the large precomputed arrays passed
around in the config: serialising 100M-element arrays to JSON would hang the
writer.
"""

import os
import time
import json
import hashlib
import h5py
import numpy as np

STATUS_STARTED = 0
STATUS_DONE = 1
STATUS_FAILED = -1

_CONFIG_EXCLUDE_KEYS = {
    "halo_masses_all",
    "halo_specific_cold_accretion_rates_all",
    "transfer_function",
    "mt_birth_index",
    "mt_death_index",
    "mt_track_ids",
    "mt_merger_ids",
    "mask_merged",
    "precomp_spawning",
    "precomp_evolving",
    "precomp_merging",
    "precomp_lost",
    # subset arrays (None when use_subsample=False, big arrays when on)
    "global_to_storage",
    "subset_indices",
    "subset_weights",
    "subset_weights_compact",
    "output_datasets",
    "logger",
}

_ARRAY_DATASETS = {
    "snapshots": "metadata/snapshots",
    "snapshots_to_save": "metadata/snapshots_to_save",
    "redshifts": "metadata/redshifts",
    "delta_times_snapshots": "metadata/delta_times_snapshots",
    "log_lbins": "emulation/log_lbins",
    "log_mbins_qbhmf": "emulation/log_mbins_qbhmf",
    "log_mbins_bhmf": "emulation/log_mbins_bhmf",
    "log_bins_cerdf": "emulation/log_bins_cerdf",
    "log_mbins_qhmf": "emulation/log_mbins_qhmf",
    "log_L_thresholds": "emulation/log_L_thresholds",
}

_PARAM_ATTR_KEYS = {
    "erdf_model": "erdf_model",
    "logfseed": "logfseed",
    "sigmaseed": "sigmaseed",
    "time_step_for_accretion": "time_step_for_accretion",
    "rad_efficiency_0": "rad_efficiency_0",
    "rad_efficiency_model": "rad_efficiency_model",
    "nbound_threshold": "nbound_threshold",
    "halo_filtering_mode": "halo_filtering_mode",
    "tdyn_fraction_default": "tdyn_fraction_default",
    "accretion_on": "accretion_on",
    "merger_on": "merger_on",
    "parallelize": "parallelize",
    "num_workers": "num_workers",
    "backend": "backend",
    "rng_seed_bh": "rng_seed_bh",
    "mass_units": "mass_units",
    "n_z": "n_z",
    "n_lbins": "n_lbins",
    "n_Lthr": "n_Lthr",
    "n_mbins": "n_mbins",
}


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

def _json_dumps_sorted(obj) -> str:
    obj = _to_serializable(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def make_run_id(params: np.ndarray, config: dict, decimals: int = 6) -> str:
    """
    Stable run id: hash(params_rounded + config).
    Increase decimals if you need more uniqueness.
    """
    p = np.round(np.asarray(params, dtype=np.float64), decimals=decimals).tolist()
    payload = {"params": p, "config": config}
    h = hashlib.sha1(_json_dumps_sorted(payload).encode("utf-8")).hexdigest()
    return h

def ensure_dset(f: h5py.File, path: str, shape, maxshape, dtype, chunks=True):
    """
    Get-or-create an HDF5 dataset.

    For existing datasets, verifies the static dims (everything beyond the
    extendable axis 0) match the requested ``shape``. Raises ValueError on
    mismatch, so the caller's subsequent per-row write cannot broadcast-fail
    at runtime without an actionable message. ``init_training_file`` is the
    canonical upstream guard (see ``_existing_schema_mismatches``); this is the
    last-line defense.
    """
    if path in f:
        existing = f[path]
        # Verify static dims (everything past axis 0). axis 0 is the runs
        # axis — it grows as we append, so the original `shape[0]=0` will
        # not equal the current row count.
        req_static = tuple(shape[1:])
        cur_static = tuple(existing.shape[1:])
        if req_static != cur_static:
            raise ValueError(
                f"ensure_dset: existing dataset {path!r} has static shape "
                f"{cur_static} but the caller requested {req_static}. "
                f"This usually means the training file's schema (n_z, n_mbins, n_lbins, n_Lthr) "
                f"no longer matches the current run. Use init_training_file with "
                "on_schema_mismatch='archive_and_recreate' (or set BAQARO_TRAINING_OVERWRITE=1) "
                "to start fresh."
            )
        return existing
    grp_path = os.path.dirname(path)
    if grp_path and grp_path not in f:
        f.require_group(grp_path)
    return f.create_dataset(path, shape=shape, maxshape=maxshape, dtype=dtype, chunks=chunks)

def _existing_schema_mismatches(path_file, param_names, output_datasets,
                                param_ranges=None):
    """
    If ``path_file`` exists, compare its per-output-dataset shapes (excluding
    the leading row axis), param-name list, and PARAM RANGES to what the
    current run wants. Returns a list of human-readable mismatch descriptions
    (empty list ⇒ OK).

    Used by ``init_training_file`` to refuse an append onto a file whose
    schema doesn't match (e.g. after a snapshots_to_save bump).

    ``schema/param_ranges`` is written ONCE at creation (``if
    "schema/param_ranges" not in f``) and is NOT inert metadata: it flows
    file → ``loading_helpers`` → ``GeneralEmulatorGP.param_ranges`` →
    ``likelihoods_and_priors.log_prior_uniform``, i.e. **it IS the MCMC prior
    box**. A range change is therefore a hard mismatch like a param_name
    change, so a file can never claim one box while containing points from
    two (the `BAQARO_PRIOR_*_LO/HI` workflow).

    NB the documented `local_box` append is NOT affected: `local_bounds` steers
    only the SAMPLING inside `run_training_set`; the `param_dict` handed to this
    file stays the global box, so the ranges still match.
    """
    if not os.path.exists(path_file):
        return []
    diffs = []
    try:
        with h5py.File(path_file, "r") as f:
            # Param names: must match exactly in order (params indexing
            # downstream uses positional access).
            if "schema/param_names" in f:
                existing_pn = [
                    (s.decode() if isinstance(s, bytes) else s)
                    for s in f["schema/param_names"][:]
                ]
                if list(existing_pn) != list(param_names):
                    diffs.append(
                        f"param_names mismatch: existing={existing_pn} vs requested={list(param_names)}"
                    )
            # Param RANGES: the recorded prior box must match the requested one.
            if param_ranges is not None and "schema/param_ranges" in f:
                existing_pr = np.asarray(f["schema/param_ranges"][:], dtype="f8")
                requested_pr = np.asarray(param_ranges, dtype="f8")
                if (existing_pr.shape != requested_pr.shape
                        or not np.allclose(existing_pr, requested_pr,
                                           rtol=0, atol=1e-12)):
                    _rows = []
                    for _i, _nm in enumerate(param_names):
                        if (_i < len(existing_pr) and _i < len(requested_pr)
                                and not np.allclose(existing_pr[_i], requested_pr[_i],
                                                    rtol=0, atol=1e-12)):
                            _rows.append(
                                f"      {_nm}: file={list(existing_pr[_i])} "
                                f"vs requested={list(requested_pr[_i])}")
                    diffs.append(
                        "param_ranges mismatch (this file's box IS the MCMC prior "
                        "— appending under a different box would make every "
                        "downstream chain sample the wrong prior):\n"
                        + "\n".join(_rows)
                        + "\n      If you meant to train a different box, use a NEW "
                          "BAQARO_NOTES_FILE_TRAINING (a separate file), which is how "
                          "the 'narrowprior' runs are done."
                    )
            # Output datasets: per-row shape (axis 0 = runs, can grow) must
            # equal the requested ``dims``.
            for name, dims in output_datasets.items():
                key = f"runs/{name}"
                if key in f:
                    existing_dims = tuple(f[key].shape[1:])
                    if existing_dims != tuple(dims):
                        diffs.append(
                            f"{name}: existing per-row shape {existing_dims} vs requested {tuple(dims)}"
                        )
    except (OSError, KeyError) as e:
        # Corrupted or unreadable file — treat as a mismatch so we don't
        # silently bulldoze it.
        diffs.append(f"could not read existing file ({e})")
    return diffs


def init_training_file(
    path_file: str,
    param_dict: dict,
    n_z: int,
    n_lbins: int,
    n_Lthr: int,
    n_mbins: int,
    output_datasets: dict | None = None,
    on_schema_mismatch: str = "raise",
):
    """
    Creates the HDF5 file with extendable datasets if it doesn't exist.
    Safe to call multiple times.

    Parameters:
        path_file: Path to HDF5 file
        param_dict: Dictionary mapping parameter names to [min, max] ranges
        n_z, n_lbins, n_Lthr, n_mbins: Dimensions for output arrays
        output_datasets: Optional dict mapping dataset names to shape tuples
            (excluding the leading run axis). If None, uses the default schema
            (log_qlfs, log_qbhmfs, log_cerdfs, log_qhmfs).
        on_schema_mismatch: What to do if ``path_file`` exists with
            incompatible per-row shapes or param_names. One of:
              * ``"raise"`` (default — safest) — raise ValueError with a
                clear remediation message. Use this when you can't afford
                to lose data.
              * ``"archive_and_recreate"`` — rename the existing file with
                a ``.YYYYMMDD-HHMMSS.archived.hdf5`` suffix and create a
                fresh file. Use this for explicit, opt-in clean restarts.
              * ``"ignore"`` — legacy behavior. Append regardless of
                schema mismatch. Only useful if you're sure the schema
                matches but the check is over-eager. Not recommended.
    """
    os.makedirs(os.path.dirname(path_file), exist_ok=True)

    # Extract param names and ranges from dict
    param_names = list(param_dict.keys())
    param_ranges = np.array(list(param_dict.values()))

    # Default output datasets if not specified
    if output_datasets is None:
        output_datasets = {
            "log_qlfs":   (n_z, n_lbins),
            "log_qbhmfs": (n_z, n_Lthr, n_mbins),
            "log_cerdfs": (n_z, n_Lthr, n_mbins),
            "log_qhmfs":  (n_z, n_Lthr, n_mbins),
        }

    # --- Schema-compatibility pre-flight ---------------------------------
    diffs = _existing_schema_mismatches(path_file, param_names, output_datasets,
                                        param_ranges=param_ranges)
    if diffs:
        msg_lines = [
            f"Existing training file at {path_file} is incompatible with the current run:",
            *[f"  - {d}" for d in diffs],
        ]
        if on_schema_mismatch == "raise":
            msg_lines += [
                "",
                "REMEDIATION:",
                "  1. (safe)    rename the old file: `mv <file> <file>.archived.hdf5`",
                "  2. (opt-in)  set BAQARO_TRAINING_OVERWRITE=1 (auto-archives + recreates)",
                "  3. (rare)    use a different BAQARO_NOTES_FILE to land in a new path",
                "  4. (legacy)  pass on_schema_mismatch='ignore' to suppress this check",
            ]
            raise ValueError("\n".join(msg_lines))
        elif on_schema_mismatch == "archive_and_recreate":
            ts = time.strftime("%Y%m%d-%H%M%S")
            archive = f"{path_file}.{ts}.archived.hdf5"
            print("\n".join(msg_lines))
            print(f"  → archiving existing file to: {archive}")
            os.rename(path_file, archive)
        elif on_schema_mismatch == "ignore":
            print("\n".join(msg_lines))
            print("  (on_schema_mismatch='ignore' — appending anyway; per-row writes may fail)")
        else:
            raise ValueError(
                f"on_schema_mismatch={on_schema_mismatch!r} not in "
                "{'raise', 'archive_and_recreate', 'ignore'}"
            )

    with h5py.File(path_file, "a") as f:
        # --- metadata / schema ---
        ensure_dset(f, "schema/version", shape=(), maxshape=(), dtype="i8", chunks=False)
        if f["schema/version"][()] == 0:
            f["schema/version"][...] = 1

        # Store param names once
        if "schema/param_names" not in f:
            dt = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("schema/param_names", data=np.array([str(name) for name in param_names], dtype=object), dtype=dt)

        # Store param ranges once
        if "schema/param_ranges" not in f:
            f.create_dataset("schema/param_ranges", data=np.asarray(param_ranges, dtype="f8"))

        # --- runs table (extendable on axis 0) ---
        # Core identifiers
        dt_str = h5py.string_dtype(encoding="utf-8")
        ensure_dset(f, "runs/run_id", shape=(0,), maxshape=(None,), dtype=dt_str)
        ensure_dset(f, "runs/status", shape=(0,), maxshape=(None,), dtype="i1")
        ensure_dset(f, "runs/design", shape=(0,), maxshape=(None,), dtype=dt_str)  # e.g. "global_sobol", "local_box"
        ensure_dset(f, "runs/t_start", shape=(0,), maxshape=(None,), dtype="f8")
        ensure_dset(f, "runs/t_end", shape=(0,), maxshape=(None,), dtype="f8")
        ensure_dset(f, "runs/runtime_s", shape=(0,), maxshape=(None,), dtype="f8")
        ensure_dset(f, "runs/error", shape=(0,), maxshape=(None,), dtype=dt_str)

        # Params
        ensure_dset(f, "runs/params", shape=(0, len(param_names)), maxshape=(None, len(param_names)), dtype="f8")

        # Outputs (store per run as rows)
        for name, dims in output_datasets.items():
            ensure_dset(f, f"runs/{name}", shape=(0,) + dims, maxshape=(None,) + dims, dtype="f4")


def write_training_metadata(path_file: str, config: dict):
    """
    Stores config and key metadata into the training HDF5 file.
    Safe to call multiple times; will not overwrite existing datasets.
    """
    if config is None:
        return

    os.makedirs(os.path.dirname(path_file), exist_ok=True)

    with h5py.File(path_file, "a") as f:
        # Ensure groups exist
        f.require_group("schema")
        f.require_group("metadata")
        f.require_group("parameters")
        f.require_group("emulation")

        # Store selected arrays as datasets (if not already present)
        for key, dset_path in _ARRAY_DATASETS.items():
            if key in config and dset_path not in f:
                f.create_dataset(dset_path, data=np.asarray(config[key]))

        # Store parameters as attributes
        params_grp = f["parameters"]
        for key, attr_name in _PARAM_ATTR_KEYS.items():
            if key in config:
                params_grp.attrs[attr_name] = _to_serializable(config[key])

        # Store ERDF params template as attributes
        erdf_params = config.get("erdf_params_template")
        if erdf_params is not None:
            erdf_grp = f.require_group("parameters/erdf_params_template")
            for k, v in erdf_params.items():
                erdf_grp.attrs[k] = _to_serializable(v)

        # Store sanitized full config as JSON
        safe_config = {k: v for k, v in config.items() if k not in _CONFIG_EXCLUDE_KEYS}
        safe_config = _to_serializable(safe_config)

        dt = h5py.string_dtype(encoding="utf-8")
        if "schema/config_json" not in f:
            f.create_dataset("schema/config_json", data=_json_dumps_sorted(safe_config), dtype=dt)
        else:
            f["schema/config_json"][...] = _json_dumps_sorted(safe_config)

def _resize_rowwise(dset, new_n):
    old_n = dset.shape[0]
    if new_n != old_n:
        dset.resize((new_n,) + dset.shape[1:])

def purge_incomplete_runs(path_file: str) -> int:
    """
    Remove all rows with status != STATUS_DONE from the HDF5 file.
    Compacts datasets in-place. Returns number of rows removed.
    """
    if not os.path.exists(path_file):
        return 0
    with h5py.File(path_file, "a") as f:
        if "runs/status" not in f:
            return 0
        statuses = f["runs/status"][...]
        keep = statuses == STATUS_DONE
        n_remove = int((~keep).sum())
        if n_remove == 0:
            return 0

        n_keep = int(keep.sum())
        all_keys = [k for k in f["runs"].keys()]
        for key in all_keys:
            dset = f[f"runs/{key}"]
            data = dset[...][keep]
            dset.resize((n_keep,) + dset.shape[1:])
            dset[:] = data

        print(f"Purged {n_remove} incomplete/failed rows, {n_keep} DONE rows remain")
    return n_remove


def load_existing_run_ids(path_file: str, only_done: bool = True) -> set[str]:
    """
    Load run IDs from the HDF5 file.
    
    Parameters:
        path_file: Path to HDF5 file
        only_done: If True, only return run IDs with STATUS_DONE. 
                   If False, return all run IDs regardless of status.
    
    Returns:
        Set of run_id strings
    """
    if not os.path.exists(path_file):
        return set()
    with h5py.File(path_file, "r") as f:
        if "runs/run_id" not in f:
            return set()
        
        run_ids = f["runs/run_id"][...].astype(str)
        
        if only_done and "runs/status" in f:
            statuses = f["runs/status"][...]
            # Only include runs that completed successfully
            done_mask = statuses == STATUS_DONE
            return set(run_ids[done_mask].tolist())
        else:
            return set(run_ids.tolist())

# ---------------------------------------------------------------------------
# Open-file-handle cores (no per-call open/flush/close).
#
# The single-writer training loop (run_training_set.writer_loop) keeps ONE
# h5py.File handle open for the whole run and calls these directly with
# flush=False, flushing periodically instead. Reopening + flushing the file on
# EVERY append (the old path-based behaviour below) made each op slower as the
# file grew on Lustre; the single writer then fell behind on its shared
# START/DONE queue and starved START replies past the timeout, orphaning
# hundreds of rows in late (large-file) batches. Keeping the handle open
# removes that per-op cost entirely. The path-based wrappers are retained for
# low-contention callers (the serial run_training_batch).
# ---------------------------------------------------------------------------

def append_run_started_f(
    f: "h5py.File",
    run_id: str,
    params: np.ndarray,
    design: str,
    *,
    flush: bool = True,
) -> int:
    """Append a STARTED row into an already-open file. Returns row index."""
    t0 = time.time()
    run_ids = f["runs/run_id"]
    n = run_ids.shape[0]
    new_n = n + 1

    # resize all run datasets (iterate whatever datasets init_training_file created)
    for key in f["runs"].keys():
        _resize_rowwise(f[f"runs/{key}"], new_n)

    f["runs/run_id"][n] = run_id
    f["runs/status"][n] = STATUS_STARTED
    f["runs/design"][n] = design
    f["runs/t_start"][n] = t0
    f["runs/t_end"][n] = np.nan
    f["runs/runtime_s"][n] = np.nan
    f["runs/error"][n] = ""
    f["runs/params"][n, :] = np.asarray(params, dtype=np.float64)

    if flush:
        f.flush()
    return n


def write_run_outputs_and_mark_done_f(
    f: "h5py.File",
    row: int,
    outputs: dict,
    *,
    flush: bool = True,
):
    """Write outputs + mark DONE into an already-open file."""
    t1 = time.time()
    t0 = float(f["runs/t_start"][row])

    for key, arr in outputs.items():
        dset_path = f"runs/{key}"
        if dset_path in f:
            f[dset_path][row] = arr.astype(np.float32)

    f["runs/status"][row] = STATUS_DONE
    f["runs/t_end"][row] = t1
    f["runs/runtime_s"][row] = max(0.0, t1 - t0)
    f["runs/error"][row] = ""
    if flush:
        f.flush()


def mark_failed_f(f: "h5py.File", row: int, err: Exception, *, flush: bool = True):
    """Mark a row FAILED into an already-open file."""
    f["runs/status"][row] = STATUS_FAILED
    f["runs/t_end"][row] = time.time()
    f["runs/error"][row] = repr(err)[:2000]
    if flush:
        f.flush()


def append_run_started(
    path_file: str,
    run_id: str,
    params: np.ndarray,
    design: str,
) -> int:
    """
    Appends a new row with status=STARTED. Returns row index.
    Path-based wrapper (opens/flushes/closes) for low-contention callers.
    """
    with h5py.File(path_file, "a") as f:
        return append_run_started_f(f, run_id, params, design, flush=True)


def write_run_outputs_and_mark_done(
    path_file: str,
    row: int,
    outputs: dict,
):
    """
    Write output arrays from a completed run into the HDF5 file.
    Matches output dict keys to existing ``runs/`` datasets — works for
    any combination of statistics (log_qlfs, log_bhmfs, log_qbhmfs, etc.).
    Path-based wrapper for low-contention callers.
    """
    with h5py.File(path_file, "a") as f:
        write_run_outputs_and_mark_done_f(f, row, outputs, flush=True)


def mark_failed(path_file: str, row: int, err: Exception):
    """Path-based wrapper for low-contention callers."""
    with h5py.File(path_file, "a") as f:
        mark_failed_f(f, row, err, flush=True)