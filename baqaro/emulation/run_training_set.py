





"""
Parallel and serial batch runners for training data generation.

Implementation notes
--------------------
- Worker init: ``_worker_config`` set as module-level global BEFORE fork;
  ``_worker_init()`` takes no args.  Workers inherit config via COW — avoids
  pickling 100M+ precomp arrays (was causing slow pool creation).
- Workers override: ``use_parallel_kernel=False`` (Numba prange doesn't
  survive fork — dead thread pool), ``parallelize=False``, ``num_workers=1``.
- Memory backpressure: ``_memory_pressure()`` uses ``mem.available`` instead
  of ``mem.free`` — ``.free`` excludes reclaimable page cache and kernel slab
  (was ~93 GB on this machine), giving false low readings.
- Added ``precomp_spawning/evolving/merging`` to ``_CONFIG_EXCLUDE_KEYS``
  for run_id hashing (prevents JSON serialization of huge arrays).
- ``worker_task`` prints traceback on failure for easier debugging.
"""

import os
import time
import queue
import numpy as np
import h5py
import multiprocessing as mp
import psutil

from baqaro.emulation.create_parameter_training import create_emulation_parameters_sobol
from baqaro.emulation.run_single_model import run_one_model
from baqaro.emulation.writing_helpers import append_run_started, load_existing_run_ids, make_run_id, mark_failed, write_run_outputs_and_mark_done, init_training_file, purge_incomplete_runs, append_run_started_f, write_run_outputs_and_mark_done_f, mark_failed_f





def writer_loop(path_file: str, start_q: mp.Queue, work_q: mp.Queue, from_writer: mp.Queue):
    """
    Single-writer process. Only this process touches HDF5.

    Two-queue PRIORITY protocol (the fix for residual START starvation):
      - ``start_q`` carries ("START", run_id, params, design) and is drained
        FIRST and completely on every loop iteration. A START append is cheap;
        replying ("ROW", run_id, row) promptly keeps the main loop (which blocks
        on each row before submitting compute) from ever waiting behind a burst
        of slow output writes.
      - ``work_q`` carries the expensive ("DONE", run_id, row, outputs) and
        ("FAIL", run_id, row, errrepr), plus the terminal ("STOP",). Processed
        one at a time AFTER start_q is empty.
    With a single shared queue, a START enqueued behind N DONE writes had to wait
    for all of them (each writing big per-row output arrays); under a startup
    burst that blew even the 600 s ROW timeout and orphaned rows. Splitting the
    queues removes that head-of-line blocking.

    Persistent open-file design: the HDF5 file is opened ONCE for the whole run
    and flushed periodically (every BAQARO_WRITER_FLUSH_EVERY ops or
    BAQARO_WRITER_FLUSH_INTERVAL seconds), instead of open/append/flush/close on
    every message. Workers never read the file (params in, outputs out via
    queues), so no inter-process flush is needed for correctness — flushing is
    purely for crash durability.
    """
    flush_every = int(os.environ.get("BAQARO_WRITER_FLUSH_EVERY", "32"))
    flush_interval = float(os.environ.get("BAQARO_WRITER_FLUSH_INTERVAL", "15"))

    with h5py.File(path_file, "a") as f:
        ops_since_flush = 0
        last_flush = time.time()

        def _maybe_flush(force=False):
            nonlocal ops_since_flush, last_flush
            if force or ops_since_flush >= flush_every or (time.time() - last_flush) >= flush_interval:
                f.flush()
                ops_since_flush = 0
                last_flush = time.time()

        def _drain_starts():
            """Service every pending START immediately (cheap, replies fast)."""
            nonlocal ops_since_flush
            while True:
                try:
                    _, run_id, params, design = start_q.get_nowait()
                except queue.Empty:
                    return
                row = append_run_started_f(f, run_id, params, design=design, flush=False)
                from_writer.put(("ROW", run_id, row))
                ops_since_flush += 1
                _maybe_flush()

        while True:
            # Priority 1: clear all STARTs before touching the work queue.
            _drain_starts()

            # Priority 2: one unit of expensive work (or wait briefly for it).
            try:
                msg = work_q.get(timeout=0.05)
            except queue.Empty:
                continue
            tag = msg[0]

            if tag == "STOP":
                # All DONE/FAIL precede STOP on work_q (FIFO); STARTs are acked
                # synchronously by the main loop, so start_q is already empty —
                # drain once more for safety, then force-flush and exit.
                _drain_starts()
                _maybe_flush(force=True)
                break

            if tag == "DONE":
                _, run_id, row, outputs = msg
                try:
                    write_run_outputs_and_mark_done_f(f, row, outputs, flush=False)
                except Exception as e:
                    # If the write fails, record failure (best effort)
                    try:
                        mark_failed_f(f, row, e, flush=False)
                    except Exception:
                        pass
                ops_since_flush += 1
                _maybe_flush()
                continue

            if tag == "FAIL":
                _, run_id, row, errrepr = msg
                try:
                    mark_failed_f(f, row, RuntimeError(errrepr), flush=False)
                except Exception:
                    pass
                ops_since_flush += 1
                _maybe_flush()
                continue


_worker_config = None  # set BEFORE pool creation, inherited via fork COW

def _setup_worker_config(config: dict):
    """Set module-level config BEFORE forking the pool.

    Workers inherit this via fork COW — no pickling needed.
    The pool initializer just applies worker-specific overrides.
    """
    global _worker_config
    _worker_config = dict(config)

def _worker_init():
    """Pool initializer (no args — config already inherited via fork)."""
    pass  # _worker_config is already set in the parent before fork

def worker_task(params: np.ndarray, design: str, run_id: str, row: int):
    """
    Compute-only worker. Does not touch HDF5.
    Uses the process-global _worker_config set by the pool initializer.
    Returns a tuple that the main process forwards to writer.
    """
    import traceback
    try:
        outputs = run_one_model(params, _worker_config)
        return ("DONE", run_id, row, outputs)
    except Exception as e:
        traceback.print_exc()
        raise







def _design_tag(design: str, local_bounds: dict | None) -> str:
    return "local_box" if local_bounds is not None else design

def _generate_candidates(
    n_new: int,
    num_parameters: int,
    param_dict: dict,
    rng: np.random.Generator | None,
    local_bounds: dict | None,
    n_skip: int = 0,
) -> tuple[np.ndarray, list]:
    """
    Uses your Sobol sampler. If local_bounds is provided, sample within that box;
    otherwise sample within param_dict.
    
    Parameters:
        n_new: Number of new points to generate
        num_parameters: Dimensionality
        param_dict: Parameter ranges
        rng: Random number generator
        local_bounds: If provided, use these ranges instead of param_dict
        n_skip: Number of Sobol points to skip (for continuing sequence)
    
    Returns:
        candidates: array of shape (n_new, num_parameters)
        param_names: list of parameter names
    """
    bounds = local_bounds if local_bounds is not None else param_dict
    print(f"DEBUG: Calling create_emulation_parameters_sobol with n_new={n_new}, n_skip={n_skip}, bounds={list(bounds.keys())}", flush=True)
    result = create_emulation_parameters_sobol(n_new, num_parameters, bounds, rng=rng, n_skip=n_skip)
    print(f"DEBUG: create_emulation_parameters_sobol returned, unpacking...", flush=True)
    return result


def run_training_batch(
    path_file: str,
    num_parameters: int,
    param_dict: dict,        # {param_name: (min, max)}
    config: dict,            # defines model; used in run_id hash
    n_new: int,
    design: str = "global_sobol",
    rng: np.random.Generator | None = None,
    local_bounds: dict | None = None,
    append: bool = True,   # Automatically continue/append to existing Sobol sequence
    overwrite: bool = False,        # If True, delete existing file and start fresh
) -> int:
    """
    Serial version: generate candidates -> skip existing -> run_one_model -> append to HDF5.
    Returns number of newly accepted runs.
    
    NOTE: Disables parallelization in evolve_BHs to avoid nested parallelization overhead.
    Parallelization should be done at the outer loop (multiple models) if needed.
    
    Parameters:
        append: If True and design="global_sobol", automatically counts existing
                completed global_sobol runs and continues/appends to the sequence.
                If False, regenerates same Sobol points (duplicate detection skips).
        overwrite: If True, deletes the existing HDF5 file and starts from scratch.
                  WARNING: This will erase all previous training data!
    """
    print("\n" + "="*80)
    print("TRAINING BATCH STARTED")
    print("="*80)
    t_batch_start = time.time()
    
    # Handle file overwrite if requested
    if overwrite and os.path.exists(path_file):
        print(f"⚠️  OVERWRITE MODE: Deleting existing file {path_file}")
        response = input("Are you sure? This will erase all data! Type 'YES' to confirm: ")
        if response == "YES":
            os.remove(path_file)
            print(f"✓ File deleted, starting fresh")
        else:
            print(f"✗ Overwrite cancelled, will append to existing file")
            overwrite = False

    # Ensure HDF5 schema exists (runs datasets, schema/param_names, etc.)
    if not os.path.exists(path_file):
        init_training_file(
            path_file=path_file,
            param_dict=param_dict,
            n_z=config["n_z"],
            n_lbins=config["n_lbins"],
            n_Lthr=config["n_Lthr"],
            n_mbins=config["n_mbins"],
            output_datasets=config.get("output_datasets"),
        )
    else:
        with h5py.File(path_file, "r") as f:
            if "runs/run_id" not in f:
                init_training_file(
                    path_file=path_file,
                    param_dict=param_dict,
                    n_z=config["n_z"],
                    n_lbins=config["n_lbins"],
                    n_Lthr=config["n_Lthr"],
                    n_mbins=config["n_mbins"],
                    output_datasets=config.get("output_datasets"),
                )

    design_tag = _design_tag(design, local_bounds)

    # Auto-detect how many Sobol points to skip for proper continuation.
    # Purge incomplete/failed rows first, then skip all DONE runs to
    # continue the Sobol sequence with genuinely new points.
    n_skip = 0
    if append and design_tag in ("global_sobol", "local_box") and os.path.exists(path_file) and not overwrite:
        purge_incomplete_runs(path_file)
        with h5py.File(path_file, "r") as f:
            if "runs/design" in f and "runs/status" in f:
                designs = f["runs/design"][...].astype(str)
                statuses = f["runs/status"][...]
                n_done = int(((designs == design_tag) & (statuses == 1)).sum())
                n_skip = n_done
                print(f"✓ Found {n_skip} COMPLETED {design_tag} runs, will skip them in sequence")

    # Time: candidate generation
    t_gen_start = time.time()
    candidates, param_names = _generate_candidates(n_new, num_parameters, param_dict, rng, local_bounds, n_skip=n_skip)
    t_gen = time.time() - t_gen_start
    print(f"✓ Generated {len(candidates)} candidates in {t_gen:.3f}s")
    print(f"  - Parameters: {param_names}")
    print(f"  - design: {design_tag}")
    if n_skip > 0:
        print(f"  - Sobol continuation: skipped first {n_skip} points")

    # Time: loading existing run IDs
    t_load_start = time.time()
    existing = load_existing_run_ids(path_file)
    t_load = time.time() - t_load_start
    print(f"✓ Loaded {len(existing)} existing run IDs in {t_load:.3f}s")
    
    accepted = 0
    skipped = 0
    failed = 0
    
    local_config = dict(config)
    
    # Create sanitized config for run_id hashing (exclude large arrays).
    # Use the CANONICAL set from writing_helpers — keeping a duplicate
    # inline copy here previously caused a parent-side serialization hang
    # on L2800N10080 subsample mode (30M-element subset_indices arrays
    # being JSON-encoded for every model's SHA1 hash).
    from baqaro.emulation.writing_helpers import _CONFIG_EXCLUDE_KEYS
    config_for_hash = {k: v for k, v in config.items() if k not in _CONFIG_EXCLUDE_KEYS}
    print(f"✓ Created sanitized config for run_id (excluded {len(_CONFIG_EXCLUDE_KEYS)} large arrays)")
    
    print(f"\nRunning {n_new} candidate models with design={design_tag}...\n")
    print(f"DEBUG: About to start loop. candidates.shape = {candidates.shape}", flush=True)
    
    model_times = []
    write_times = []
    
    for idx, params in enumerate(candidates):
        print(f"DEBUG: [{idx+1:3d}/{n_new}] Starting iteration...", flush=True)
        params = np.asarray(params, dtype=np.float64)
        print(f"DEBUG: [{idx+1:3d}/{n_new}] Computing run_id...", flush=True)
        run_id = make_run_id(params, config=config_for_hash, decimals=6)
        print(f"DEBUG: [{idx+1:3d}/{n_new}] run_id={run_id[:12]}..., checking if exists", flush=True)
        
        if run_id in existing:
            skipped += 1
            print(f"[{idx+1:3d}/{n_new}] SKIP  row=?      run_id={run_id[:12]}... (already exists)", flush=True)
            continue

        # Time: append started
        print(f"DEBUG: [{idx+1:3d}/{n_new}] Appending run_started to HDF5...", flush=True)
        t_append_start = time.time()
        row = append_run_started(path_file, run_id, params, design=design_tag)
        t_append = time.time() - t_append_start
        print(f"DEBUG: [{idx+1:3d}/{n_new}] Appended row={row}, took {t_append:.3f}s", flush=True)
        existing.add(run_id)

        try:
            # Time: model execution
            print(f"DEBUG: [{idx+1:3d}/{n_new}] Starting run_one_model...", flush=True)
            t_model_start = time.time()
            outputs = run_one_model(params, local_config)
            t_model = time.time() - t_model_start
            model_times.append(t_model)
            
            # Time: write outputs
            t_write_start = time.time()
            write_run_outputs_and_mark_done(path_file, row, outputs)
            t_write = time.time() - t_write_start
            write_times.append(t_write)
            
            accepted += 1
            total_time = t_append + t_model + t_write
            print(f"[{idx+1:3d}/{n_new}] DONE  row={row:5d}  run_id={run_id[:12]}...  "
                  f"model={t_model:.2f}s  write={t_write:.3f}s  total={total_time:.2f}s")
        except Exception as e:
            mark_failed(path_file, row, e)
            failed += 1
            print(f"[{idx+1:3d}/{n_new}] FAIL  row={row:5d}  run_id={run_id[:12]}...  "
                  f"err={type(e).__name__}: {str(e)[:40]}")

    # Compute summary statistics
    t_batch_total = time.time() - t_batch_start
    
    if model_times:
        avg_model_time = np.mean(model_times)
        min_model_time = np.min(model_times)
        max_model_time = np.max(model_times)
        std_model_time = np.std(model_times)
    else:
        avg_model_time = min_model_time = max_model_time = std_model_time = 0.0
    
    if write_times:
        avg_write_time = np.mean(write_times)
        total_write_time = np.sum(write_times)
    else:
        avg_write_time = total_write_time = 0.0
    
    print("\n" + "="*80)
    print("TRAINING BATCH SUMMARY")
    print("="*80)
    print(f"Results:       {accepted} accepted  |  {skipped} skipped  |  {failed} failed")
    print(f"Total time:    {t_batch_total:.2f}s")
    print(f"  - Setup:     {t_gen + t_load:.3f}s (candidate gen + load existing)")
    print(f"  - Models:    {np.sum(model_times):.2f}s ({accepted} runs)")
    print(f"  - I/O:       {total_write_time:.3f}s ({accepted} writes)")
    
    if accepted > 0:
        print(f"\nModel timing stats (per run):")
        print(f"  - Average:   {avg_model_time:.2f}s ± {std_model_time:.2f}s")
        print(f"  - Min/Max:   {min_model_time:.2f}s / {max_model_time:.2f}s")
        print(f"  - Write:     {avg_write_time:.3f}s (avg)")
        print(f"  - Throughput: {accepted / t_batch_total:.2f} runs/sec")
    
    print("="*80 + "\n")
    
    return accepted


def run_training_batch_parallel(
    path_file: str,
    num_parameters: int,
    param_dict: dict,        # {param_name: (min, max)}
    config: dict,
    n_new: int,
    design: str = "global_sobol",
    rng: np.random.Generator | None = None,
    local_bounds: dict | None = None,
    append: bool = True,
    overwrite: bool = False,
    n_workers: int = 8,
    max_in_flight: int | None = None,
    memory_floor_gb: float | None = None,
) -> int:
    """
    Parallel version:
      - generate candidates -> skip existing
      - single writer process appends to HDF5
      - worker processes compute run_one_model and send results back

    Parameters
    ----------
    memory_floor_gb : float or None
        Minimum system-wide available RAM (in GB) to maintain. When available
        memory drops below this floor, job submission pauses until running
        workers finish and free memory.
        Uses ``mem.total - mem.used`` (i.e. checks actual used memory) rather
        than ``mem.available``, because ``.available`` double-counts COW-shared
        pages after fork() — it looks fine at first but real RSS balloons as
        workers write to private pages.
        If None, no memory-based backpressure is applied.

    Returns number of newly accepted runs.

    Parameters:
        append: If True, automatically counts existing completed runs of this
                design type and continues the Sobol sequence.
        overwrite: If True, deletes the existing HDF5 file and starts from scratch.
                  WARNING: This will erase all previous training data!
    """
    print("\n" + "="*80)
    print("TRAINING BATCH PARALLEL STARTED")
    print("="*80)
    t_batch_start = time.time()

    # Handle file overwrite if requested
    if overwrite and os.path.exists(path_file):
        print(f"⚠️  OVERWRITE MODE: Deleting existing file {path_file}")
        response = input("Are you sure? This will erase all data! Type 'YES' to confirm: ")
        if response == "YES":
            os.remove(path_file)
            print(f"✓ File deleted, starting fresh")
        else:
            print(f"✗ Overwrite cancelled, will append to existing file")
            overwrite = False

    # Ensure HDF5 schema exists (runs datasets, schema/param_names, etc.)
    if not os.path.exists(path_file):
        init_training_file(
            path_file=path_file,
            param_dict=param_dict,
            n_z=config["n_z"],
            n_lbins=config["n_lbins"],
            n_Lthr=config["n_Lthr"],
            n_mbins=config["n_mbins"],
            output_datasets=config.get("output_datasets"),
        )
    else:
        with h5py.File(path_file, "r") as f:
            if "runs/run_id" not in f:
                init_training_file(
                    path_file=path_file,
                    param_dict=param_dict,
                    n_z=config["n_z"],
                    n_lbins=config["n_lbins"],
                    n_Lthr=config["n_Lthr"],
                    n_mbins=config["n_mbins"],
                    output_datasets=config.get("output_datasets"),
                )

    design_tag = _design_tag(design, local_bounds)

    # Auto-detect how many Sobol points to skip for proper continuation.
    # Purge incomplete/failed rows first, then skip all DONE runs to
    # continue the Sobol sequence with genuinely new points.
    n_skip = 0
    if append and design_tag in ("global_sobol", "local_box") and os.path.exists(path_file) and not overwrite:
        purge_incomplete_runs(path_file)
        with h5py.File(path_file, "r") as f:
            if "runs/design" in f and "runs/status" in f:
                designs = f["runs/design"][...].astype(str)
                statuses = f["runs/status"][...]
                n_done = int(((designs == design_tag) & (statuses == 1)).sum())
                n_skip = n_done
                print(f"✓ Found {n_skip} COMPLETED {design_tag} runs, will skip them in sequence")

    # Time: candidate generation
    t_gen_start = time.time()
    candidates, param_names = _generate_candidates(n_new, num_parameters, param_dict, rng, local_bounds, n_skip=n_skip)
    t_gen = time.time() - t_gen_start
    print(f"✓ Generated {len(candidates)} candidates in {t_gen:.3f}s")
    print(f"  - Parameters: {param_names}")
    print(f"  - design: {design_tag}")
    if n_skip > 0:
        print(f"  - Sobol continuation: skipped first {n_skip} points")

    if max_in_flight is None:
        max_in_flight = n_workers

    # Pool always creates n_workers processes upfront, each holding mmap views
    # of the large halo arrays.  If n_workers > max_in_flight, the excess idle
    # workers still compete for disk I/O (page faults) and can cause sustained
    # D-state thrashing even though mem.available looks fine.
    if n_workers > max_in_flight:
        print(f"[WARN] n_workers={n_workers} > max_in_flight={max_in_flight}; "
              f"reducing n_workers to {max_in_flight} to avoid I/O thrashing")
        n_workers = max_in_flight

    memory_floor_bytes = int(memory_floor_gb * 1e9) if memory_floor_gb is not None else None
    if memory_floor_bytes is not None:
        mem = psutil.virtual_memory()
        headroom = (mem.total - mem.used) / 1e9
        print(f"✓ Memory backpressure enabled: floor={memory_floor_gb:.0f} GB headroom "
              f"(currently used={mem.used/1e9:.0f} GB / total={mem.total/1e9:.0f} GB, "
              f"headroom={headroom:.0f} GB)")

    # Time: loading existing run IDs
    t_load_start = time.time()
    existing = load_existing_run_ids(path_file)
    t_load = time.time() - t_load_start
    print(f"✓ Loaded {len(existing)} existing run IDs in {t_load:.3f}s")

    # Create sanitized config for run_id hashing (exclude large arrays).
    # Use the CANONICAL set from writing_helpers — keeping a duplicate
    # inline copy here previously caused a parent-side serialization hang
    # on L2800N10080 subsample mode (30M-element subset_indices arrays
    # being JSON-encoded for every model's SHA1 hash).
    from baqaro.emulation.writing_helpers import _CONFIG_EXCLUDE_KEYS
    config_for_hash = {k: v for k, v in config.items() if k not in _CONFIG_EXCLUDE_KEYS}
    print(f"✓ Created sanitized config for run_id (excluded {len(_CONFIG_EXCLUDE_KEYS)} large arrays)")

    print(f"\nRunning {n_new} candidate models with design={design_tag} ({n_workers} workers)...\n")

    # multiprocessing context
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    # Two queues: cheap STARTs on their own priority queue, expensive
    # DONE/FAIL/STOP on the work queue (see writer_loop docstring).
    start_q = ctx.Queue()
    work_q = ctx.Queue()
    from_writer = ctx.Queue()

    writer = ctx.Process(target=writer_loop, args=(path_file, start_q, work_q, from_writer), daemon=True)
    writer.start()

    # Set config as module global BEFORE fork — workers inherit via COW, no pickling.
    _setup_worker_config(config)
    pool = ctx.Pool(processes=n_workers, initializer=_worker_init)

    pending: dict[str, mp.pool.ApplyResult] = {}
    pending_since: dict[str, float] = {}  # submit timestamp per in-flight task
    run_id_to_row: dict[str, int] = {}
    _parked_rows: dict[str, int] = {}  # parks out-of-order ROW replies

    # Per-task wall-clock timeout. A worker killed by the OOM
    # reaper never returns a result, so ar.ready() stays False forever and the
    # drain loop would spin indefinitely. If a task exceeds this budget it is
    # marked FAILED and dropped from `pending` so the batch finishes (the row
    # gets status=-1 and is retried on the next resume). Generous default 1h —
    # no single model legitimately takes that long; set BAQARO_TASK_TIMEOUT=0 to
    # disable.
    _tt = float(os.environ.get("BAQARO_TASK_TIMEOUT", "3600"))
    task_timeout_s = _tt if _tt > 0 else None

    submitted = 0
    accepted = 0
    skipped = 0
    failed = 0

    # Seconds to wait for the single writer process to reply with a ROW
    # assignment. With the persistent open file + dedicated priority START queue
    # (drained before any DONE write), a ROW reply should be near-instant; this
    # timeout is now just a safety net against a genuinely stuck writer rather
    # than the expected-contention budget it used to be. Override via
    # BAQARO_WRITER_TIMEOUT.
    _WRITER_TIMEOUT = float(os.environ.get("BAQARO_WRITER_TIMEOUT", "600"))
    # Max time to wait for the writer to drain its full FIFO backlog at
    # shutdown before warning (it never gives up — see below).
    _WRITER_SHUTDOWN_TIMEOUT = float(
        os.environ.get("BAQARO_WRITER_SHUTDOWN_TIMEOUT", "1800")
    )
    print(
        f"[WRITER] ROW-reply timeout={_WRITER_TIMEOUT:.0f}s, "
        f"shutdown-drain warn-after={_WRITER_SHUTDOWN_TIMEOUT:.0f}s",
        flush=True,
    )

    def _get_row_for(run_id: str) -> int:
        """Read ROW replies until we find the one for run_id.
        Parks out-of-order replies so they are not lost.
        """
        if run_id in _parked_rows:
            return _parked_rows.pop(run_id)
        deadline = time.time() + _WRITER_TIMEOUT
        while True:
            remaining = max(deadline - time.time(), 0.0)
            if remaining == 0.0:
                raise RuntimeError("Writer timeout waiting for ROW reply")
            try:
                tag, rid_back, row = from_writer.get(timeout=min(remaining, 1.0))
            except Exception:
                raise RuntimeError("Writer timeout waiting for ROW reply")
            if tag != "ROW":
                raise RuntimeError(f"Unexpected writer tag: {tag}")
            if rid_back == run_id:
                return row
            _parked_rows[rid_back] = row  # out-of-order: park for later

    def drain_completed():
        nonlocal accepted, failed
        done_ids = [rid for rid, ar in pending.items() if ar.ready()]
        for rid in done_ids:
            ar = pending.pop(rid)
            pending_since.pop(rid, None)
            row = run_id_to_row[rid]
            try:
                msg = ar.get()          # ("DONE", run_id, row, outputs)
                work_q.put(msg)
                accepted += 1
                print(f"[DONE]   row={row:5d}  run_id={rid[:12]}...")
            except Exception as e:
                work_q.put(("FAIL", rid, row, repr(e)))
                failed += 1
                print(f"[FAIL]   row={row:5d}  run_id={rid[:12]}...  "
                      f"err={type(e).__name__}: {str(e)[:40]}")

        # Timeout sweep: a worker killed by the OOM reaper never
        # delivers a result, so its ar.ready() stays False forever and the drain
        # loops below would spin indefinitely. Drop such tasks as FAILED so the
        # batch can finish; the row lands status=-1 and is retried on resume.
        if task_timeout_s is not None and pending:
            now = time.time()
            stale = [rid for rid in list(pending)
                     if now - pending_since.get(rid, now) > task_timeout_s]
            for rid in stale:
                pending.pop(rid, None)
                pending_since.pop(rid, None)
                row = run_id_to_row.get(rid, -1)
                work_q.put(("FAIL", rid, row,
                            f"task exceeded BAQARO_TASK_TIMEOUT={task_timeout_s:.0f}s "
                            "(worker likely OOM-killed or hung)"))
                failed += 1
                print(f"[TIMEOUT] row={row:5d}  run_id={rid[:12]}...  "
                      f"exceeded {task_timeout_s:.0f}s -> marked FAILED", flush=True)

    try:
        for idx, params in enumerate(candidates):
            params = np.asarray(params, dtype=np.float64)
            run_id = make_run_id(params, config=config_for_hash, decimals=6)
            if run_id in existing:
                skipped += 1
                continue

            # backpressure: count-based AND memory-based
            # Uses mem.used (actual committed memory) rather than checking
            # mem.available, because .available double-counts COW-shared pages
            # after fork().  Workers share mmap'd data at first, but as each
            # worker writes to its own BH arrays the kernel COW-faults those
            # pages into private RSS.  .available doesn't see this coming, so
            # it stays high while real usage silently balloons past physical RAM.
            def _memory_pressure():
                if memory_floor_bytes is None:
                    return False
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                # Two triggers:
                # 1) Headroom check: total - used < floor
                headroom_low = (mem.total - mem.used) < memory_floor_bytes
                # 2) Swap guard: only trigger if swap is large AND available RAM is low
                #    Stale swap (e.g. 15 GB leftover) is harmless when available >> floor
                swap_pressure = swap.used > 20e9 and mem.available < memory_floor_bytes
                return headroom_low or swap_pressure

            while len(pending) >= max_in_flight or (len(pending) > 0 and _memory_pressure()):
                drain_completed()
                if _memory_pressure() and len(pending) > 0:
                    mem = psutil.virtual_memory()
                    swap = psutil.swap_memory()
                    print(f"[MEM] Waiting: used={mem.used/1e9:.1f} GB / "
                          f"total={mem.total/1e9:.1f} GB "
                          f"(free={mem.free/1e9:.1f} GB, "
                          f"swap={swap.used/1e9:.1f} GB), "
                          f"in-flight={len(pending)}", flush=True)
                    time.sleep(5.0)  # longer sleep when memory-constrained
                else:
                    time.sleep(0.05)

            # append STARTED row via writer (priority queue → prompt ROW reply)
            start_q.put(("START", run_id, params, design_tag))
            try:
                row = _get_row_for(run_id)
            except Exception as e:
                print(f"[ERROR]  Writer failed to reply for run_id={run_id[:12]}...: {e}")
                failed += 1
                continue

            run_id_to_row[run_id] = row
            existing.add(run_id)
            submitted += 1

            # submit compute (workers use _worker_config set by initializer, no pickling of config)
            ar = pool.apply_async(worker_task, (params, design_tag, run_id, row))
            pending[run_id] = ar
            pending_since[run_id] = time.time()

            print(f"[SUBMIT] row={row:5d}  run_id={run_id[:12]}...  ({submitted} submitted, {len(pending)} in-flight)")

        # finish remaining
        while pending:
            drain_completed()
            time.sleep(0.05)

    finally:
        # Drain any remaining completed results before shutting down
        pool.close()
        pool.join()
        drain_completed()

        work_q.put(("STOP",))
        # Block until the writer has flushed its ENTIRE FIFO backlog: every
        # queued DONE write precedes the STOP we just sent. A short join here
        # would let the `finally` return and the daemon writer be terminated
        # mid-backlog, dropping the last computed rows. Wait
        # indefinitely (the scheduler wall-clock is the real bound), logging if
        # it stalls so a genuine filesystem hang is still visible.
        _join_start = time.time()
        while writer.is_alive():
            writer.join(timeout=30)
            waited = time.time() - _join_start
            if writer.is_alive() and waited > _WRITER_SHUTDOWN_TIMEOUT:
                print(
                    f"[WARN] writer still draining backlog after {waited:.0f}s "
                    "at shutdown; continuing to wait to avoid losing rows",
                    flush=True,
                )
                _join_start = time.time()  # reset so we warn periodically

    # Summary statistics
    t_batch_total = time.time() - t_batch_start

    # The in-flight counters (accepted/skipped/failed) only see pool-side
    # outcomes — they cannot detect write-back failures (status=-1) that
    # happen between worker return and HDF5 append. Read the ground truth
    # from the file's runs/status array, so the summary is self-consistent.
    truth = {"done": None, "failed": None, "running": None}
    try:
        with h5py.File(path_file, "r") as _f:
            _status = _f["runs/status"][:]
            truth["done"]    = int((_status == 1).sum())
            truth["failed"]  = int((_status == -1).sum())
            truth["running"] = int((_status == 0).sum())
    except Exception:
        pass

    print("\n" + "="*80)
    print("TRAINING BATCH PARALLEL SUMMARY")
    print("="*80)
    print(f"In-flight pool counters: {accepted} accepted  |  {skipped} skipped  |  {failed} failed")
    if truth["done"] is not None:
        print(f"On-disk row status     : {truth['done']} done (status=1)  |  "
              f"{truth['failed']} failed (status=-1)  |  {truth['running']} stuck running (status=0)")
        if truth["failed"] > 0 or truth["running"] > 0:
            print(f"  ⚠ {truth['failed'] + truth['running']} row(s) did NOT complete successfully despite pool reporting them — "
                  "inspect runs/error for messages.")
    print(f"Total time:    {t_batch_total:.2f}s")
    print(f"  - Setup:     {t_gen + t_load:.3f}s (candidate gen + load existing)")

    # Throughput now uses the true done-count when available.
    _done_for_rate = truth["done"] if truth["done"] is not None else accepted
    if _done_for_rate > 0:
        print(f"  - Throughput: {_done_for_rate / t_batch_total:.2f} runs/sec (based on status=1 rows)")

    print("="*80 + "\n")

    return accepted