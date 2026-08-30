"""
Correctness gate for the persistent single-writer (run_training_set.writer_loop).
=================================================================================

The writer keeps ONE open handle for the whole run and flushes periodically;
STOP force-flushes. (Reopening the file on every append lets the writer fall
behind its START/DONE queue on a slow filesystem, and starved START replies
turn into orphaned rows.) These tests pin down that the persistent writer:

  1. assigns a unique row per START and replies with it (no orphans),
  2. writes the right outputs to the right row even when DONE arrives
     out-of-order relative to row index,
  3. records FAIL correctly,
  4. persists EVERYTHING after STOP even when periodic flushing is effectively
     disabled (proves the STOP force-flush + close durability — the
     shutdown-drop guard), and
  5. produces a file byte-equivalent to the old path-based write functions
     (the refactor is behaviour-preserving).

They drive the REAL ``writer_loop`` in a child process via the same fork
Queues production uses, so they exercise the actual shipped code path.
"""
import os
import time
import numpy as np
import h5py
import multiprocessing as mp
import pytest

from baqaro.emulation.run_training_set import writer_loop
from baqaro.emulation.writing_helpers import (
    init_training_file,
    append_run_started,
    write_run_outputs_and_mark_done,
    mark_failed,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_STARTED,
)

N_Z, N_LBINS = 2, 3
PARAM_DICT = {"a": [0.0, 1.0], "b": [0.0, 1.0]}
OUTPUT_DATASETS = {"log_qlfs": (N_Z, N_LBINS)}


def _init(path):
    init_training_file(
        path_file=str(path),
        param_dict=PARAM_DICT,
        n_z=N_Z, n_lbins=N_LBINS, n_Lthr=1, n_mbins=2,
        output_datasets=OUTPUT_DATASETS,
    )


def _params(i):
    return np.array([i * 0.01, 1.0 - i * 0.01], dtype=np.float64)


def _outputs(i):
    # distinct, recoverable payload per row so we can detect cross-contamination
    return {"log_qlfs": np.full((N_Z, N_LBINS), float(i), dtype=np.float64)}


def _drive_writer(path, ops, *, flush_every=None, flush_interval=None):
    """Run the real writer_loop in a child process, replaying ``ops``.

    ops: list of ("START", i) | ("DONE", i) | ("FAIL", i). START must precede
    the matching DONE/FAIL for the same i. Returns {i: row}.
    """
    if flush_every is not None:
        os.environ["BAQARO_WRITER_FLUSH_EVERY"] = str(flush_every)
    if flush_interval is not None:
        os.environ["BAQARO_WRITER_FLUSH_INTERVAL"] = str(flush_interval)

    ctx = mp.get_context("fork")
    start_q = ctx.Queue()
    work_q = ctx.Queue()
    from_writer = ctx.Queue()
    writer = ctx.Process(target=writer_loop, args=(str(path), start_q, work_q, from_writer), daemon=True)
    writer.start()

    row_of = {}
    parked = {}

    def _get_row(run_id):
        if run_id in parked:
            return parked.pop(run_id)
        while True:
            tag, rid, row = from_writer.get(timeout=30)
            assert tag == "ROW"
            if rid == run_id:
                return row
            parked[rid] = row

    try:
        for op in ops:
            kind, i = op
            run_id = f"run{i}"
            if kind == "START":
                start_q.put(("START", run_id, _params(i), "global_sobol"))
                row_of[i] = _get_row(run_id)
            elif kind == "DONE":
                work_q.put(("DONE", run_id, row_of[i], _outputs(i)))
            elif kind == "FAIL":
                work_q.put(("FAIL", run_id, row_of[i], f"boom{i}"))
            else:
                raise ValueError(kind)
        work_q.put(("STOP",))
        writer.join(timeout=60)
        assert not writer.is_alive(), "writer did not exit after STOP"
        assert writer.exitcode == 0, f"writer exitcode={writer.exitcode}"
    finally:
        if writer.is_alive():
            writer.terminate()
    return row_of


def test_basic_all_done(tmp_path):
    path = tmp_path / "t.hdf5"
    _init(path)
    n = 50
    ops = [("START", i) for i in range(n)] + [("DONE", i) for i in range(n)]
    row_of = _drive_writer(path, ops)

    with h5py.File(path, "r") as f:
        status = f["runs/status"][:]
        rids = f["runs/run_id"][:].astype(str)
        qlfs = f["runs/log_qlfs"][:]
        assert len(status) == n
        assert (status == STATUS_DONE).all(), "orphaned/failed rows present"
        assert sorted(rids.tolist()) == sorted(f"run{i}" for i in range(n))
        # each row carries ITS OWN payload
        for i, row in row_of.items():
            assert np.allclose(qlfs[row], float(i)), f"row {row} has wrong outputs"
    # rows are unique
    assert len(set(row_of.values())) == n


def test_out_of_order_done(tmp_path):
    """DONE arriving in scrambled order must still land on the right row."""
    path = tmp_path / "t.hdf5"
    _init(path)
    n = 40
    order = list(range(n))
    rng = np.random.default_rng(0)
    rng.shuffle(order)
    ops = [("START", i) for i in range(n)] + [("DONE", i) for i in order]
    row_of = _drive_writer(path, ops)
    with h5py.File(path, "r") as f:
        status = f["runs/status"][:]
        qlfs = f["runs/log_qlfs"][:]
        assert (status == STATUS_DONE).all()
        for i, row in row_of.items():
            assert np.allclose(qlfs[row], float(i))


def test_fail_recorded(tmp_path):
    path = tmp_path / "t.hdf5"
    _init(path)
    ops = [("START", 0), ("START", 1), ("START", 2),
           ("DONE", 0), ("FAIL", 1), ("DONE", 2)]
    row_of = _drive_writer(path, ops)
    with h5py.File(path, "r") as f:
        status = f["runs/status"][:]
        err = f["runs/error"][:].astype(str)
        assert status[row_of[0]] == STATUS_DONE
        assert status[row_of[1]] == STATUS_FAILED
        assert status[row_of[2]] == STATUS_DONE
        assert "boom1" in err[row_of[1]]


def test_stop_flushes_everything_without_periodic_flush(tmp_path):
    """With periodic flush effectively disabled, STOP must still persist all
    rows (force-flush + file close). This is the shutdown-drop guard: the old
    code's short join could kill the writer before the backlog was durable."""
    path = tmp_path / "t.hdf5"
    _init(path)
    n = 64
    ops = [("START", i) for i in range(n)] + [("DONE", i) for i in range(n)]
    # flush_every huge + interval huge -> no periodic flush fires; only STOP does
    row_of = _drive_writer(path, ops, flush_every=10**9, flush_interval=10**9)
    with h5py.File(path, "r") as f:
        status = f["runs/status"][:]
        assert len(status) == n
        assert (status == STATUS_DONE).all(), "rows lost despite STOP"


def test_persistent_equals_path_based(tmp_path):
    """The persistent _f cores (via writer_loop) must produce a file identical
    to the old path-based open/flush/close functions for the same sequence."""
    n = 30
    # (a) persistent writer
    p_fast = tmp_path / "fast.hdf5"
    _init(p_fast)
    ops = [("START", i) for i in range(n)] + [("DONE", i) for i in range(n)]
    row_of = _drive_writer(p_fast, ops)

    # (b) reference: path-based functions, same order, in-process
    p_ref = tmp_path / "ref.hdf5"
    _init(p_ref)
    ref_row = {}
    for i in range(n):
        ref_row[i] = append_run_started(str(p_ref), f"run{i}", _params(i), "global_sobol")
    for i in range(n):
        write_run_outputs_and_mark_done(str(p_ref), ref_row[i], _outputs(i))

    assert row_of == ref_row, "row assignment diverged"
    with h5py.File(p_fast, "r") as ff, h5py.File(p_ref, "r") as fr:
        for key in ("status", "run_id", "design", "params", "log_qlfs"):
            a = ff[f"runs/{key}"][:]
            b = fr[f"runs/{key}"][:]
            if a.dtype.kind in ("U", "S", "O"):
                assert a.astype(str).tolist() == b.astype(str).tolist(), key
            else:
                assert np.array_equal(a, b), key


def test_start_priority_over_done_backlog(tmp_path):
    """A START queued AFTER a big backlog of expensive DONE writes must still
    get its ROW reply promptly — the two-queue priority fix. With a single
    shared FIFO the START would wait behind the whole backlog (the residual
    starvation seen on chunk 3). We assert the late START is answered in well
    under the time it takes to drain the backlog."""
    path = tmp_path / "prio.hdf5"
    big = (300, 300)  # ~360 KB float32 per row -> DONE writes are non-trivial
    init_training_file(
        str(path), PARAM_DICT, n_z=big[0], n_lbins=big[1], n_Lthr=1, n_mbins=2,
        output_datasets={"log_qlfs": big},
    )
    payload = {"log_qlfs": np.ones(big, dtype=np.float64)}
    N = 150

    os.environ["BAQARO_WRITER_FLUSH_EVERY"] = "1"  # force each DONE to hit disk
    ctx = mp.get_context("fork")
    start_q, work_q, from_writer = ctx.Queue(), ctx.Queue(), ctx.Queue()
    writer = ctx.Process(target=writer_loop, args=(str(path), start_q, work_q, from_writer), daemon=True)
    writer.start()
    try:
        # Pre-create N rows we can DONE.
        rows = []
        for i in range(N):
            start_q.put(("START", f"pre{i}", _params(i), "global_sobol"))
            tag, rid, row = from_writer.get(timeout=30)
            assert tag == "ROW"
            rows.append(row)

        # Flood the work queue with N expensive DONE writes...
        for i in range(N):
            work_q.put(("DONE", f"pre{i}", rows[i], payload))
        # ...then immediately queue a NEW START and time its ROW reply.
        t0 = time.time()
        start_q.put(("START", "late", _params(999), "global_sobol"))
        tag, rid, row = from_writer.get(timeout=60)
        t_row = time.time() - t0
        assert tag == "ROW" and rid == "late"

        work_q.put(("DONE", "late", row, payload))
        work_q.put(("STOP",))
        writer.join(timeout=120)
        t_total = time.time() - t0
        assert writer.exitcode == 0

        # The late START was answered well before the DONE backlog finished
        # draining. (FIFO single-queue would give t_row ~ t_total.)
        assert t_row < 0.5 * t_total, f"START not prioritized: t_row={t_row:.3f}s, t_total={t_total:.3f}s"

        with h5py.File(path, "r") as f:
            assert (f["runs/status"][:] == STATUS_DONE).all()
            assert len(f["runs/status"][:]) == N + 1
    finally:
        if writer.is_alive():
            writer.terminate()
        os.environ.pop("BAQARO_WRITER_FLUSH_EVERY", None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
