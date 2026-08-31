"""Read training sets and trained emulators back off disk.

The counterpart to :mod:`writing_helpers`. Two entry points:

``load_training_data``
    Pull the parameter matrix and the summary statistics (QLF, cERDF, BHMF,
    QHMF) out of a training HDF5, with the floor/NaN handling the raw arrays
    need before they can be PCA-reduced.
``load_emulators``
    Unpickle the fitted GP+PCA emulators for a given run identity.

Both resolve their filenames from the same environment-driven run identity the
producers used, so a reader and a writer configured alike agree on the path
without it having to be passed around.
"""

import os
import numpy as np
import h5py

import time

from baqaro.emulation.emulation_core_functions import GeneralEmulatorGP




def load_training_data(filepath, physical_floor=-9.5, pca_floor=-12.0,
                       only_done=True, design_filter=None):
    """
    Loads and cleans training data from z0 HDF5 file.

    Z0 variant: reads log_bhmfs (all BHs, no luminosity cut, shape
    (n_runs, n_z, n_mbins)) instead of log_qbhmfs.

    Parameters:
        filepath: Path to HDF5 training file
        physical_floor: Threshold below which bins are considered empty (default: -9.5)
        pca_floor: Value to assign to empty/floored bins for PCA (default: -12.0)
        only_done: If True, only load runs with STATUS_DONE=1 (default: True)
        design_filter: If specified, only load runs with this design type
                      (e.g., "global_sobol" or "local_box"). If None, load all.

    Returns:
        Dictionary with training data including params, outputs, and metadata
    """
    print(f"Loading training data from: {filepath}")

    def _read_dataset(fh, keys):
        for key in keys:
            if key in fh:
                return np.asarray(fh[key])
        return None

    def _read_param_names(fh, keys):
        for key in keys:
            if key in fh:
                data = fh[key]
                if data.shape == ():
                    val = data[()]
                    return [val.decode("utf-8") if isinstance(val, bytes) else str(val)]
                return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in data[...]]
        return None

    with h5py.File(filepath, "r") as f:
        # Load metadata (new schema) with legacy fallback
        snapshots = _read_dataset(f, ["metadata/snapshots", "snapshots"])
        redshifts = _read_dataset(f, ["metadata/redshifts", "redshifts"])
        snapshots_to_save = _read_dataset(f, ["metadata/snapshots_to_save", "snapshots_to_save"])
        redshift_keys = _read_dataset(f, ["emulation/redshift_keys"])

        log_lbins = _read_dataset(f, ["emulation/log_lbins"])
        log_bins_cerdf = _read_dataset(f, ["emulation/log_bins_cerdf"])
        log_mbins_bhmf = _read_dataset(f, ["emulation/log_mbins_bhmf"])
        log_mbins_qhmf = _read_dataset(f, ["emulation/log_mbins_qhmf"])

        log_L_thresholds = _read_dataset(f, ["emulation/log_L_thresholds"])

        # Load parameter schema (new or legacy)
        param_names = _read_param_names(f, ["schema/param_names", "emulation/params_names"])

        # Load parameter ranges from file (with fallback)
        if "schema/param_ranges" in f:
            param_ranges = np.asarray(f["schema/param_ranges"])
        elif "emulation/params_ranges" in f:
            param_ranges = np.asarray(f["emulation/params_ranges"])
        else:
            # Fallback: will compute from data after loading
            param_ranges = None

        # Load run data (new or legacy)
        all_params = _read_dataset(f, ["runs/params", "emulation/params"])
        all_statuses = _read_dataset(f, ["runs/status"])
        all_designs = _read_dataset(f, ["runs/design"])

        if all_params is None:
            raise KeyError("No parameters found. Expected 'runs/params' or 'emulation/params'.")

        if param_names is None:
            raise KeyError("No parameter names found. Expected 'schema/param_names' or 'emulation/params_names'.")

        if all_statuses is None:
            all_statuses = np.ones(len(all_params), dtype=int)
        if all_designs is None:
            all_designs = np.array([""] * len(all_params))
        else:
            all_designs = all_designs.astype(str)

        # Load outputs — z0 schema uses log_bhmfs instead of log_qbhmfs
        all_log_qlfs = _read_dataset(f, ["runs/log_qlfs", "emulation/log_qlfs"])
        all_log_bhmfs = _read_dataset(f, ["runs/log_bhmfs"])
        all_log_cerdfs = _read_dataset(f, ["runs/log_cerdfs", "emulation/log_cerdfs"])
        all_log_qhmfs = _read_dataset(f, ["runs/log_qhmfs", "emulation/log_qhmfs"])

    # Build filter mask
    mask = np.ones(len(all_statuses), dtype=bool)

    if only_done:
        mask &= (all_statuses == 1)  # STATUS_DONE

    if design_filter is not None:
        mask &= (all_designs == design_filter)

    n_total = len(all_statuses)
    n_selected = mask.sum()

    filter_msg = []
    if only_done:
        filter_msg.append(f"STATUS_DONE={mask.sum()}/{n_total}")
    if design_filter:
        filter_msg.append(f"design={design_filter}")

    print(f"  Filtering: {n_selected}/{n_total} runs ({', '.join(filter_msg) if filter_msg else 'all'})")

    # Apply filter
    params = all_params[mask]
    designs = all_designs[mask]
    log_qlfs = all_log_qlfs[mask]
    log_bhmfs = all_log_bhmfs[mask]
    log_cerdfs = all_log_cerdfs[mask]
    log_qhmfs = all_log_qhmfs[mask]

    # Compute param_ranges if not in file (fallback for older files)
    if param_ranges is None:
        param_ranges = np.array([[params[:, i].min(), params[:, i].max()]
                                 for i in range(params.shape[1])])
        print(f"  param_ranges not found in file, computed from data")

    # Prepare output dictionary
    # Compute snapshots_to_save / redshift_keys if missing
    if snapshots_to_save is None:
        if redshift_keys is not None and redshifts is not None:
            idxs = []
            for rk in redshift_keys:
                matches = np.where(np.isclose(redshifts, rk))[0]
                if len(matches) == 0:
                    raise ValueError(f"Could not map redshift_key={rk} onto redshifts array")
                idxs.append(matches[0])
            snapshots_to_save = np.asarray(idxs, dtype=int)
        elif redshifts is not None:
            snapshots_to_save = np.arange(len(redshifts))

    if redshift_keys is None and redshifts is not None and snapshots_to_save is not None:
        redshift_keys = redshifts[snapshots_to_save]

    missing_bins = []
    if log_lbins is None:
        missing_bins.append("log_lbins")
    if log_bins_cerdf is None:
        missing_bins.append("log_bins_cerdf")
    if log_mbins_bhmf is None:
        missing_bins.append("log_mbins_bhmf")
    if log_mbins_qhmf is None:
        missing_bins.append("log_mbins_qhmf")
    if log_L_thresholds is None:
        missing_bins.append("log_L_thresholds")
    if missing_bins:
        raise KeyError(f"Missing bin definitions: {', '.join(missing_bins)}")

    data = {
        # Metadata
        "snapshots": snapshots,
        "redshifts": redshifts,
        "snapshots_to_save": snapshots_to_save,
        "redshift_keys": redshift_keys,

        # Bins (log10 of bin centers)
        "log_lbins": log_lbins,
        "log_bins_cerdf": log_bins_cerdf,
        "log_mbins_bhmf": log_mbins_bhmf,
        "log_mbins_qhmf": log_mbins_qhmf,
        "log_L_thresholds": log_L_thresholds,

        # Parameters
        "params": params,
        "param_names": param_names,
        "param_ranges": param_ranges,
        "designs": designs,

        # Outputs kept in storage format: (n_runs, n_z, ...)
        "log_qlfs": log_qlfs,      # (n_runs, n_z, n_lbins)
        "log_bhmfs": log_bhmfs,    # (n_runs, n_z, n_mbins)
        "log_cerdfs": log_cerdfs,  # (n_runs, n_z, n_Lthr, n_mbins)
        "log_qhmfs": log_qhmfs,    # (n_runs, n_z, n_Lthr, n_mbins)
    }

    # Clean Data: replace NaN/Inf, then apply monotonic floor clamping.
    # Strategy: values below physical_floor are considered "empty". Once a curve
    # crosses below physical_floor (scanning from left or right towards the
    # interior), everything beyond that point stays floored — no "up and down".
    # All floored values are set to pca_floor (well below physical_floor) so
    # the GP sees a smooth, flat baseline rather than a sharp cliff.
    keys_stats = ["log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs"]
    for k in keys_stats:
        arr = data[k]
        # First replace NaN/Inf. posinf -> physical_floor (NOT 0.0): a +inf
        # log-density is unphysical (0.0 == log10 Phi = 0 => Phi=1 Mpc^-3 dex^-1,
        # absurdly high) and mapping it to 0.0 produced a "valid" extreme value
        # that survived all flooring and poisoned the PCA basis. Treat it as
        # empty so the flooring below sends it to pca_floor.
        arr = np.nan_to_num(arr, nan=physical_floor, posinf=physical_floor, neginf=physical_floor)
        # Floor everything at or below physical_floor to pca_floor
        arr = np.where(arr <= physical_floor, pca_floor, arr)

        # Monotonic floor clamping: keep only the longest contiguous block
        # of valid bins (above physical_floor). Everything else is floored.
        # This ensures no "up and down" — no isolated valid bins surrounded
        # by floor values (which the PCA would have to spend components on).
        # An isolated valid bin beyond an interior hole is therefore floored
        # too; the training targets are defined with this rule.
        shape = arr.shape
        n_bins = shape[-1]
        flat = arr.reshape(-1, n_bins)

        for i in range(flat.shape[0]):
            row = flat[i]
            above = row > physical_floor

            if not above.any():
                flat[i, :] = pca_floor
                continue

            # Find longest contiguous run of valid bins
            # Diff trick: transitions are at boundaries of contiguous blocks
            padded = np.concatenate(([False], above, [False]))
            diffs = np.diff(padded.astype(int))
            starts = np.where(diffs == 1)[0]   # block start indices
            ends = np.where(diffs == -1)[0]     # block end indices (exclusive)
            lengths = ends - starts
            best = np.argmax(lengths)
            best_start, best_end = starts[best], ends[best]

            # Floor everything outside the longest valid block
            if best_start > 0:
                flat[i, :best_start] = pca_floor
            if best_end < n_bins:
                flat[i, best_end:] = pca_floor

        data[k] = flat.reshape(shape)

    print(f"  Loaded {params.shape[0]} runs with {len(param_names)} parameters")
    print(f"  Output shapes: QLF={data['log_qlfs'].shape}, BHMF={data['log_bhmfs'].shape}, QHMF={data['log_qhmfs'].shape}")

    return data



def load_emulators(path_out, name_file, flags):
    """Loads requested emulators from .xz pickle files."""
    emulators = {}
    mapping = {
        "qlf": f"emulator_qlf_{name_file}.xz",
        "bhmf": f"emulator_bhmf_{name_file}.xz",
        "cerdf": f"emulator_cerdf_{name_file}.xz",
        "qhmf": f"emulator_qhmf_{name_file}.xz"
    }

    for key, enabled in flags.items():
        if enabled:
            path = os.path.join(path_out, "emulators", mapping[key])
            # Portable (HDF5) fallbacks: the same emulator as plain arrays,
            # loadable with numpy + h5py alone (see emulation/portable_emulator.py).
            # Preferred outright with BAQARO_EMULATOR_FORMAT=portable.
            portable_candidates = [
                os.path.join(path_out, "emulators",
                             mapping[key].replace(".xz", ".hdf5")),
                os.path.join(path_out, "emulators", f"emulator_{key}.hdf5"),
            ]
            prefer_portable = os.environ.get("BAQARO_EMULATOR_FORMAT", "") == "portable"
            portable = next((c for c in portable_candidates if os.path.exists(c)), None)
            t0 = time.time()
            if portable is not None and (prefer_portable or not os.path.exists(path)):
                from baqaro.emulation.portable_emulator import PortableEmulator
                print(f"Loading {key.upper()} emulator (portable) from {portable}...")
                emulators[key] = PortableEmulator.load(portable)
                print(f"Loaded in {time.time()-t0:.2f}s")
                continue
            print(f"Loading {key.upper()} emulator from {path}...")
            try:
                emulators[key] = GeneralEmulatorGP.load(path)
                print(f"Loaded in {time.time()-t0:.2f}s")
            except FileNotFoundError:
                print(f"Warning: Emulator file not found: {path} "
                      f"(and no portable .hdf5 beside it)")
                emulators[key] = None
        else:
            emulators[key] = None

    return emulators
