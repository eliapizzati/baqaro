

"""
Single forward-model evaluation for emulator training.
=======================================================

Evolves BHs snapshot-by-snapshot for one parameter point and computes
summary statistics (QLF, QBHMF, CERDF, QHMF) on-the-fly at selected
redshifts.  Returns the statistics as arrays ready for HDF5 storage.

Performance notes
-----------------
* Uses ``evolve_BHs_fast`` (Numba) with pre-allocated buffers — same
  optimisations as ``main_evolution.py``.
* Integer indexing via ``np.flatnonzero`` for gather/scatter.
* Only 1-D state vectors are kept (no full snapshot × halo arrays).
* All 2-D halo arrays are ``(n_snapshots, n_halos)`` C-order — access
  a snapshot with ``arr[i]``.

Dead-BH retirement
------------------
* Tracks dying at snapshot ``i`` (merged, aborted-merged, disrupted) get
  their BH zeroed at ``i`` — matching ``main_evolution.py``'s live-only
  per-snapshot accounting. Carrying a dead BH forward at its pre-death mass
  would double-count merged mass.
* Merger targets are additionally filtered on true liveness
  (``death_index``), not just frozen ``LastMaxMass > 0`` — otherwise a
  deposit onto a long-dead target would park mass in a zeroed slot
  forever. The production drivers apply the same filter, so training and
  production merger semantics are identical.

Implementation notes
--------------------
* Uses precomputed per-snapshot index arrays from config
  (``precomp_spawning``, ``precomp_evolving``, ``precomp_merging``,
  ``precomp_lost``)
  instead of computing masks over 200M-element arrays per snapshot.
  Saves ~62s per model under 40-worker memory bandwidth contention.
* Statistics: replaced ``(200M, 21)`` boolean threshold matrix + 4
  sorted array copies (~5 GB peak) with single boolean mask for
  lowest threshold + sub-selection from small arrays (~200 MB peak).
  Reduced statistics time from 242s → 25s per model.
* Added detailed profiling instrumentation (mmap, seed, gather,
  accretion, scatter, merger, stats) printed per model.
* Overall: 423s → 146s per model (2.9x speedup).
"""

import numpy as np
from qhtools.utils import natconst as nc, my_utils

from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast as evolve_BHs
from baqaro.core_functions.bh_seeding import spawn_BHs
from baqaro.core_functions.erdf_core_functions import Erdf


def run_one_model(params: np.ndarray, config: dict) -> dict:
    """Run one forward-model evaluation and return summary statistics.

    Memory-efficient: keeps only 1-D state vectors (bh_prev, bh_curr,
    Lbol_curr), computes requested summary stats on-the-fly at
    ``snapshots_to_save``, and returns per-run output arrays ready to be
    appended into HDF5.

    Parameters
    ----------
    params : np.ndarray, shape (n_params,)
        Parameter values for this run (order matches ``config["param_names"]``).
    config : dict
        Run configuration.  See ``main_training.py`` for the full list of
        required keys.  Key arrays:

        * ``halo_masses_all`` — mmap, shape ``(n_snapshots, n_halos)``
        * ``halo_specific_cold_accretion_rates_all`` — mmap, same shape
        * ``mt_birth_index``, ``mt_death_index``, ``mt_track_ids``,
          ``mt_merger_ids``, ``mask_merged`` — 1-D merger tree arrays
        * ``transfer_function`` — precomputed lognormal-sum sampler

    Returns
    -------
    dict with keys ``"log_qlfs"``, ``"log_bhmfs"``, ``"log_cerdfs"``,
    ``"log_qhmfs"`` — float32 arrays of log10 number densities.
    """
    import time
    from numpy.random import default_rng

    # -----------------------------------------------------------------
    # Unpack parameters and build ERDF
    # -----------------------------------------------------------------
    t_total_start = time.time()

    param_names = config["param_names"]
    print(f"  [MODEL] Starting with params: {dict(zip(param_names, np.round(params, 4)))}", flush=True)

    sampled = dict(zip(param_names, params))
    erdf_params_dict = dict(config["erdf_params_template"])
    for name, val in sampled.items():
        if name in erdf_params_dict:
            erdf_params_dict[name] = float(val)

    erdf_model = config["erdf_model"]
    rng_seed = config.get("rng_seed_bh", 12345)
    rng_run_local = default_rng(rng_seed)
    erdf = Erdf(erdf_model, erdf_params_dict, rng=rng_run_local)

    # -----------------------------------------------------------------
    # Output arrays
    # -----------------------------------------------------------------
    n_z = config["n_z"]
    n_lbins = config["n_lbins"]
    n_Lthr = config["n_Lthr"]
    n_mbins = config["n_mbins"]

    log_qlfs = np.zeros((n_z, n_lbins), dtype=np.float32)
    log_bhmfs = np.zeros((n_z, n_mbins), dtype=np.float32)
    log_cerdfs = np.zeros((n_z, n_Lthr, n_mbins), dtype=np.float32)
    log_qhmfs = np.zeros((n_z, n_Lthr, n_mbins), dtype=np.float32)

    # -----------------------------------------------------------------
    # Unpack frequently used config
    # -----------------------------------------------------------------
    accretion_on = config["accretion_on"]
    merger_on = config["merger_on"]

    # Time stepping
    time_step_for_accretion = config["time_step_for_accretion"]
    if "logtcoherence" in sampled:
        time_step_for_accretion = 10 ** float(sampled["logtcoherence"]) / 1e6
        print(f"  [DEBUG] logtcoherence={sampled['logtcoherence']:.3f} -> time_step_for_accretion={time_step_for_accretion:.3f} Myr", flush=True)

    # Radiative efficiency
    rad_efficiency_0 = config["rad_efficiency_0"]
    rad_efficiency_model = config["rad_efficiency_model"]
    # Madau f_eff correction. Default off so existing
    # training products are bit-identical; set by main_training from the
    # BAQARO_MADAU_FEFF_CORRECTION env var. See core_functions/madau_feff.py.
    madau_feff_correction = config.get("madau_feff_correction", False)
    # Per-snapshot growth-sum cap (numerical guardrail mode). Default 50.0 never
    # bites; set by main_training from BAQARO_GROWTH_SUM_MAX so a capped emulator
    # matches a capped forward run. See core_functions/main_evolution.py.
    growth_sum_max = config.get("growth_sum_max", 50.0)

    # Seeding
    logfseed = config["logfseed"]
    sigmaseed = config.get("sigmaseed", None)
    if "logfseed" in sampled:
        logfseed = float(sampled["logfseed"])
        print(f"  [DEBUG] logfseed={logfseed:.3f}", flush=True)
    if "sigmaseed" in sampled:
        sigmaseed = float(sampled["sigmaseed"])
        print(f"  [DEBUG] sigmaseed={sigmaseed:.3f}", flush=True)

    # Numba prange doesn't survive fork — always run serial kernels in workers
    fw_approx = config.get("fw_approx", False)

    # Bins / normalisations
    qlf_bins = config["qlf_bins"]
    qlf_normalization = config["qlf_normalization"]
    bhmf_bins = config["bhmf_bins"]
    bhmf_normalization = config["bhmf_normalization"]
    cerdf_bins = config["cerdf_bins"]
    cerdf_normalization = config["cerdf_normalization"]
    qhmf_bins = config["qhmf_bins"]
    qhmf_normalization = config["qhmf_normalization"]
    log_L_thresholds = config["log_L_thresholds"]

    # Snapshot selection
    snapshots = config["snapshots"]
    snap_to_zindex = config["snap_to_zindex"]

    # Cosmology arrays
    redshifts = config["redshifts"]
    delta_times_snapshots = config["delta_times_snapshots"]

    # Units / constants
    mass_units_f32 = np.float32(config["mass_units"])
    log_csi_10 = 10 ** nc.log_csi

    # Global arrays (passed by reference — no copy)
    halo_masses_all = config["halo_masses_all"]           # subset-sized in subsample mode
    halo_specific_cold_accretion_rates_all = config["halo_specific_cold_accretion_rates_all"]
    transfer_function = config["transfer_function"]

    # mt_track_ids / mt_merger_ids stay full-N (merger pointers use global IDs).
    # mt_birth_index and mask_merged are encoded in precomp_* and not
    # accessed in the worker; mt_death_index IS needed for the merger-target
    # liveness filter (halo mass is frozen LastMaxMass, so
    # ``mass > 0`` means "ever resolved", not "alive").
    mt_track_ids = config["mt_track_ids"]
    mt_merger_ids = config["mt_merger_ids"]
    mt_death_index = config["mt_death_index"]

    logger = config.get("logger", None)

    # -----------------------------------------------------------------
    # Storage layout (subset-aware)
    # -----------------------------------------------------------------
    # ``n_storage`` is the size of BH/Lbol/_buf_* arrays. In full-sim
    # mode it equals the total halo count. In subsample mode it's
    # n_subset (typically 0.01-25% of total), and ``global_to_storage``
    # maps a global track ID to its position in the compact arrays.
    # ``_to_storage(x)`` is identity in full-sim mode.
    use_subsample_local = config.get("use_subsample", False)
    if use_subsample_local:
        n_storage = int(config["n_storage"])
        global_to_storage = config["global_to_storage"]
        subset_weights_compact = config["subset_weights_compact"]
    else:
        n_storage = halo_masses_all.shape[1]
        global_to_storage = None
        subset_weights_compact = None

    def _to_storage(x):
        if global_to_storage is None:
            return x
        return global_to_storage[x]

    bh_prev = np.zeros(n_storage, dtype=np.float32)
    bh_curr = np.zeros(n_storage, dtype=np.float32)
    Lbol_curr = np.zeros(n_storage, dtype=np.float32)

    # Pre-allocated buffers for evolve_BHs_fast (avoid repeated allocation).
    # Sized to n_storage — upper bound on n_evolving per snap.
    _buf_bh_prev = np.empty(n_storage, dtype=np.float32)
    _buf_log_rates = np.empty(n_storage, dtype=np.float32)
    _buf_Lbols_out = np.empty(n_storage, dtype=np.float64)
    _rng_buffers = {
        'z_last': np.empty(n_storage, dtype=np.float32),
        'u': np.empty(n_storage, dtype=np.float64),
        'z_approx': np.empty(n_storage, dtype=np.float32),
    }

    # -----------------------------------------------------------------
    # Per-snapshot index arrays (precomputed once in main_training.py,
    # shared across all workers via fork COW — zero per-worker cost).
    # In subsample mode the arrays are already restricted to subset_mask.
    # -----------------------------------------------------------------
    _precomp_spawning = config["precomp_spawning"]
    _precomp_evolving = config["precomp_evolving"]
    _precomp_merging = config.get("precomp_merging", {})
    _precomp_lost = config.get("precomp_lost", {})   # disrupted halos

    # Pre-allocate merger chain mapping array, sized to n_storage so
    # subsample mode pays only n_subset * 4 bytes instead of N * 8 bytes.
    # Dtype tracks the index space we write into it:
    #   - subsample mode: storage positions are 0..n_subset-1 with -1 sentinel,
    #     int32 fits (asserted upstream when global_to_storage is built).
    #   - full-sim mode: storage positions are global IDs (since _to_storage is
    #     identity), which can exceed 2**31 at L2800N10080 — keep int64 by
    #     following mt_track_ids.dtype.
    _merger_mapping_dtype = np.int32 if use_subsample_local else mt_track_ids.dtype
    _merger_mapping = np.full(n_storage, -1, dtype=_merger_mapping_dtype)

    # Pre-allocate statistics buffers (reused every saved snapshot)
    _bh_phys = np.empty(n_storage, dtype=np.float32)
    _Lbol_phys = np.empty(n_storage, dtype=np.float32)
    _halo_phys = np.empty(n_storage, dtype=np.float32)
    _edd = np.zeros(n_storage, dtype=np.float32)

    # Precompute luminosity cuts once (identical every snapshot)
    Lcuts = 10 ** my_utils.to_solar(log_L_thresholds.astype(np.float64))
    lowest_cut = Lcuts[0]

    # -----------------------------------------------------------------
    # Main snapshot evolution loop
    # -----------------------------------------------------------------
    t_setup = time.time() - t_total_start
    t0 = time.time()
    first_snap = int(snapshots[0]) if len(snapshots) > 0 else 0
    n_snaps = len(snapshots)

    # Timing accumulators
    t_mmap = 0.0       # reading mmap rows
    t_seed = 0.0       # seeding BHs
    t_gather = 0.0     # gather into buffers + log10
    t_accretion = 0.0  # evolve_BHs_fast kernel
    t_scatter = 0.0    # scatter back
    t_merger = 0.0     # merger step
    t_stats = 0.0      # on-the-fly statistics
    n_stats_computed = 0

    print(f"  [PROFILE] setup={t_setup:.3f}s, n_storage={n_storage}, n_snaps={n_snaps}, "
          f"use_subsample={use_subsample_local}", flush=True)

    for i in snapshots:
        i = int(i)

        if logger is not None:
            logger.info("#" * 50)
            logger.info(f"Snapshot {i}, z={redshifts[i]:.2f}")
            logger.info(f"Δt={delta_times_snapshots[i]*1e3:.2f} Myr")

        # Sub-step count
        dt_myr = float(delta_times_snapshots[i] * 1e3)
        n_steps = max(1, int(dt_myr / time_step_for_accretion))
        if i == first_snap:
            print(f"  [DEBUG] snap={i}: dt={dt_myr:.2f} Myr, timestep={time_step_for_accretion:.3f} Myr -> n_steps={n_steps}", flush=True)

        # Read snapshot data from mmap — layout is (n_snapshots, n_halos)
        _t = time.time()
        current_halo_masses = halo_masses_all[i]
        if accretion_on:
            current_rates = halo_specific_cold_accretion_rates_all[i]
        t_mmap += time.time() - _t

        # Precomputed index arrays (no per-snapshot mask computation)
        idx_spawning = _precomp_spawning[i]
        idx_evolving = _precomp_evolving[i]

        # Carry masses forward
        if i == first_snap:
            bh_curr[:] = 0.0
        else:
            bh_curr[:] = bh_prev
        Lbol_curr[:] = 0.0

        # --------------------------------------------------------------
        # STEP 1: Seed new BHs
        # --------------------------------------------------------------
        # idx_spawning is GLOBAL; map to storage positions so we index
        # eager-loaded subset rows correctly (identity in full-sim mode).
        _t = time.time()
        if idx_spawning.size > 0:
            idx_spawning_storage = _to_storage(idx_spawning)
            halo_slice = current_halo_masses[idx_spawning_storage]
            bh_seeded = spawn_BHs(halo_slice, logfseed, sigmaseed=sigmaseed, rng=rng_run_local)
            bh_curr[idx_spawning_storage] = bh_seeded
        t_seed += time.time() - _t

        # --------------------------------------------------------------
        # STEP 2: Accretion (evolve_BHs_fast with pre-allocated buffers)
        # --------------------------------------------------------------
        n_evolving = idx_evolving.size
        idx_evolving_storage = _to_storage(idx_evolving)

        if accretion_on and n_evolving > 0:
            # Gather into pre-allocated buffers (sliced to actual size)
            _t = time.time()
            bh_in = _buf_bh_prev[:n_evolving]
            bh_in[:] = bh_prev[idx_evolving_storage]

            log_halo_rates = _buf_log_rates[:n_evolving]
            np.take(current_rates, idx_evolving_storage, out=log_halo_rates)
            np.clip(log_halo_rates, 1e-8, 1e3, out=log_halo_rates)
            np.log10(log_halo_rates, out=log_halo_rates)
            t_gather += time.time() - _t

            _t = time.time()
            bh_out, Lbol_out = evolve_BHs(
                transfer_function,
                bh_in,
                delta_times_snapshots[i],
                erdf,
                n_steps=n_steps,
                log_halo_rates_array=log_halo_rates,
                rad_efficiency=rad_efficiency_0,
                rad_efficiency_model=rad_efficiency_model,
                parallelize=False,
                num_workers=1,
                backend="threading",
                use_parallel_kernel=False,
                fw_p_threshold=0.7 if fw_approx else 0.0,
                L_bols_out=_buf_Lbols_out[:n_evolving],
                rng_buffers=_rng_buffers,
                growth_max=growth_sum_max,
                madau_feff_correction=madau_feff_correction,
            )
            t_accretion += time.time() - _t

            # Scatter back to storage positions
            _t = time.time()
            bh_curr[idx_evolving_storage] = bh_out
            Lbol_curr[idx_evolving_storage] = Lbol_out
            t_scatter += time.time() - _t

        # --------------------------------------------------------------
        # STEP 3: Mergers and disruptions
        # --------------------------------------------------------------
        # Chain resolution operates in STORAGE-position space so the
        # mapping array is n_storage-sized (saves ~3 GB in subsample mode).
        _t = time.time()
        if merger_on:
            idx_merging = _precomp_merging[i]

            if idx_merging.size > 0:
                src_ids = mt_track_ids[idx_merging]
                dest_ids = mt_merger_ids[idx_merging]

                # Filter unresolved AND already-dead targets. Halo mass is
                # frozen LastMaxMass, so ``> 0`` alone means "ever resolved",
                # not "alive" — a target that died at an earlier snapshot
                # would swallow the source BH into a slot that is never
                # carried forward. A target
                # dying AT this snapshot is kept: it is itself a merger
                # source, and chain resolution forwards the deposit.
                dest_death = mt_death_index[dest_ids]
                alive = (current_halo_masses[_to_storage(dest_ids)] > 0.0) \
                    & ((dest_death == -1) | (dest_death >= i))
                src_ids = src_ids[alive]
                dest_ids = dest_ids[alive]

                if src_ids.size > 0:
                    # Convert to storage positions once
                    src_pos = _to_storage(src_ids)
                    dest_pos = _to_storage(dest_ids)

                    # Closure guarantees every merger src/dest is in the subset,
                    # so _to_storage never returns the -1 sentinel here. Guard it
                    # cheaply: a -1 would alias storage slot [-1] (the last BH)
                    # and mis-deposit. Subsample mode
                    # only (identity map in full-sim -> global_to_storage None).
                    if global_to_storage is not None:
                        assert src_pos.min() >= 0 and dest_pos.min() >= 0, (
                            "merger src/dest outside the subset storage map "
                            "(closure invariant violated)"
                        )

                    # Resolve merger chains A → B → C in storage space.
                    _merger_mapping[src_pos] = dest_pos
                    final_dest = dest_pos.copy()
                    for _ in range(src_pos.size):
                        nxt = _merger_mapping[final_dest]
                        chain = nxt != -1
                        if not np.any(chain):
                            break
                        final_dest[chain] = nxt[chain]
                    _merger_mapping[src_pos] = -1  # reset for next snapshot
                    dest_pos = final_dest

                    np.add.at(bh_curr, dest_pos, bh_prev[src_pos])

        # --------------------------------------------------------------
        # STEP 3b: Retire dying BHs
        # --------------------------------------------------------------
        # Every track dying at this snapshot stops existing: merged sources
        # (their mass now lives in the descendant), aborted-merger sources
        # (target unresolved/dead -> mass lost), and disrupted halos. This
        # mirrors main_evolution.py, which zero-initializes each snapshot
        # row and writes only live entries.
        # Ordering: AFTER the np.add.at deposit, which reads bh_prev — so
        # zeroing bh_curr cannot corrupt chain-intermediate deposits.
        idx_dying_merged = _precomp_merging.get(i)
        if idx_dying_merged is not None and idx_dying_merged.size > 0:
            bh_curr[_to_storage(mt_track_ids[idx_dying_merged])] = 0.0
        idx_lost = _precomp_lost.get(i)
        if idx_lost is not None and idx_lost.size > 0:
            bh_curr[_to_storage(mt_track_ids[idx_lost])] = 0.0
        t_merger += time.time() - _t

        # --------------------------------------------------------------
        # On-the-fly statistics at selected snapshots
        # --------------------------------------------------------------
        if i in snap_to_zindex:
            _t = time.time()
            iz = snap_to_zindex[i]

            # Convert to physical units in-place into preallocated buffers.
            # current_halo_masses is the eager-loaded subset row in
            # subsample mode (shape n_storage), full mmap row otherwise.
            np.multiply(bh_curr, mass_units_f32, out=_bh_phys)
            np.multiply(Lbol_curr, mass_units_f32, out=_Lbol_phys)
            np.multiply(current_halo_masses, mass_units_f32, out=_halo_phys)
            hist_weights_all = subset_weights_compact  # None in full-sim mode

            # Eddington ratios (reset, then fill valid entries)
            _edd[:] = 0.0
            valid_bh = _bh_phys > 0.0
            _edd[valid_bh] = (_Lbol_phys[valid_bh] / _bh_phys[valid_bh]) / log_csi_10

            # QLF
            qlf, _ = np.histogram(_Lbol_phys, qlf_bins, weights=hist_weights_all)
            qlf_norm = qlf * qlf_normalization
            with np.errstate(divide="ignore"):
                log_qlfs[iz, :] = np.where(qlf_norm > 0.0, np.log10(qlf_norm), -np.inf).astype(np.float32)

            # BHMF — all BHs with mass > 0 (no luminosity cut)
            bhmf_weights = hist_weights_all[valid_bh] if hist_weights_all is not None else None
            bhmf, _ = np.histogram(_bh_phys[valid_bh], bhmf_bins, weights=bhmf_weights)
            bhmf_norm = bhmf * bhmf_normalization
            with np.errstate(divide="ignore"):
                log_bhmfs[iz, :] = np.where(bhmf_norm > 0.0, np.log10(bhmf_norm), -np.inf).astype(np.float32)

            # Threshold-dependent statistics (CERDF, QHMF)
            # Single boolean mask for the loosest threshold
            above_lowest = _Lbol_phys > lowest_cut
            edd_above = _edd[above_lowest]
            halo_above = _halo_phys[above_lowest]
            Lbol_above = _Lbol_phys[above_lowest]
            w_above = hist_weights_all[above_lowest] if hist_weights_all is not None else None

            for j in range(len(log_L_thresholds)):
                if j == 0:
                    edd_sel = edd_above
                    halo_sel = halo_above
                    w_sel = w_above
                else:
                    cut_mask = Lbol_above > Lcuts[j]
                    edd_sel = edd_above[cut_mask]
                    halo_sel = halo_above[cut_mask]
                    w_sel = w_above[cut_mask] if w_above is not None else None

                with np.errstate(divide="ignore"):
                    cerdf, _ = np.histogram(edd_sel, cerdf_bins, weights=w_sel)
                    cerdf_norm = cerdf * cerdf_normalization
                    log_cerdfs[iz, j, :] = np.where(cerdf_norm > 0.0, np.log10(cerdf_norm), -np.inf).astype(np.float32)

                    qhmf, _ = np.histogram(halo_sel, qhmf_bins, weights=w_sel)
                    qhmf_norm = qhmf * qhmf_normalization
                    log_qhmfs[iz, j, :] = np.where(qhmf_norm > 0.0, np.log10(qhmf_norm), -np.inf).astype(np.float32)

            t_stats += time.time() - _t
            n_stats_computed += 1

        # Swap buffers (no allocation)
        bh_prev, bh_curr = bh_curr, bh_prev

    runtime_loop = time.time() - t0
    runtime_total = time.time() - t_total_start

    print(f"  [PROFILE] Snapshot loop: {runtime_loop:.2f}s over {n_snaps} snapshots", flush=True)
    print(f"    mmap_read   = {t_mmap:.3f}s", flush=True)
    print(f"    seeding     = {t_seed:.3f}s", flush=True)
    print(f"    gather+log  = {t_gather:.3f}s", flush=True)
    print(f"    accretion   = {t_accretion:.3f}s  (evolve_BHs_fast kernel)", flush=True)
    print(f"    scatter     = {t_scatter:.3f}s", flush=True)
    print(f"    mergers     = {t_merger:.3f}s", flush=True)
    print(f"    statistics  = {t_stats:.3f}s  ({n_stats_computed} redshifts)", flush=True)
    print(f"  [MODEL] Total: {runtime_total:.2f}s (setup={t_setup:.3f}s + loop={runtime_loop:.2f}s)", flush=True)

    return {
        "log_qlfs": log_qlfs,
        "log_bhmfs": log_bhmfs,
        "log_cerdfs": log_cerdfs,
        "log_qhmfs": log_qhmfs,
    }
