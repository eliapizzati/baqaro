"""
Unit test for emulation/pool_chunks.py.
=======================================

Builds tiny synthetic per-chunk training HDF5s with known log-density rows
and verifies that pooling:
  * sums densities exactly: pooled = log10(Σ_chunks 10**log_density),
  * preserves the empty-bin convention (all -inf -> -inf),
  * matches rows by parameter vector regardless of row order,
  * marks a point status=-1 when it is not DONE in every chunk.

No halo data or sampler needed.
"""

import os
import tempfile

import h5py
import numpy as np

from baqaro.emulation.pool_chunks import pool, _POOL_KEYS

# tiny schema
N_Z, N_LBINS, N_MBINS, N_LTHR = 2, 3, 3, 2
_DIMS = {
    "log_qlfs": (N_Z, N_LBINS),
    "log_bhmfs": (N_Z, N_MBINS),
    "log_cerdfs": (N_Z, N_LTHR, N_MBINS),
    "log_qhmfs": (N_Z, N_LTHR, N_MBINS),
}
STATUS_DONE, STATUS_FAILED = 1, -1


def _make_chunk_file(path, params, statuses, data):
    """data[key] has shape (n_rows, *dims)."""
    n = len(params)
    with h5py.File(path, "w") as f:
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("schema/param_names",
                         data=np.array(["a", "b"], dtype=object), dtype=dt)
        f.create_dataset("runs/params", data=np.asarray(params, dtype="f8"))
        f.create_dataset("runs/status", data=np.asarray(statuses, dtype="i1"))
        f.create_dataset("runs/run_id",
                         data=np.array([f"id{i}" for i in range(n)], dtype=object),
                         dtype=dt)
        f.create_dataset("runs/design",
                         data=np.array(["global_sobol"] * n, dtype=object), dtype=dt)
        f.create_dataset("runs/error", data=np.array([""] * n, dtype=object), dtype=dt)
        for k in _POOL_KEYS:
            f.create_dataset(f"runs/{k}", data=data[k].astype("f4"))


def _rand_logdens(rng, dims, frac_empty=0.3):
    """Random log10 densities in [-12,-2], with some bins set to -inf."""
    x = rng.uniform(-12.0, -2.0, size=dims)
    empty = rng.random(size=dims) < frac_empty
    x[empty] = -np.inf
    return x


def test_pool_sums_densities_and_handles_status():
    rng = np.random.default_rng(0)
    # three shared param points + one extra only-in-chunk0 region exercised
    # via "missing in a chunk" below.
    params = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    n = len(params)

    # Per-chunk, per-key log densities for each of the n points.
    truth = {pi: {} for pi in range(n)}
    chunks_data = []
    for c in range(3):
        data = {}
        for k in _POOL_KEYS:
            arr = np.stack([_rand_logdens(rng, _DIMS[k]) for _ in range(n)])
            data[k] = arr
            for pi in range(n):
                truth[pi].setdefault(k, []).append(arr[pi])
        chunks_data.append(data)

    with tempfile.TemporaryDirectory() as d:
        paths = []
        # chunk 0: all DONE, natural order
        _make_chunk_file(os.path.join(d, "c0.hdf5"), params,
                         [STATUS_DONE] * n, chunks_data[0])
        paths.append(os.path.join(d, "c0.hdf5"))
        # chunk 1: all DONE, REVERSED row order (tests param matching)
        order = [2, 1, 0]
        _make_chunk_file(
            os.path.join(d, "c1.hdf5"),
            [params[i] for i in order],
            [STATUS_DONE] * n,
            {k: chunks_data[1][k][order] for k in _POOL_KEYS},
        )
        paths.append(os.path.join(d, "c1.hdf5"))
        # chunk 2: point index 1 is FAILED -> that point can't be pooled
        st2 = [STATUS_DONE, STATUS_FAILED, STATUS_DONE]
        _make_chunk_file(os.path.join(d, "c2.hdf5"), params, st2, chunks_data[2])
        paths.append(os.path.join(d, "c2.hdf5"))

        out = os.path.join(d, "combined.hdf5")
        n_pooled, n_incomplete = pool(paths, out, verbose=False)

        assert n_pooled == 2          # points 0 and 2
        assert n_incomplete == 1      # point 1 (failed in chunk 2)

        with h5py.File(out, "r") as f:
            status = f["runs/status"][:]
            out_params = f["runs/params"][:]
            # combined rows are in chunk0 order: points 0,1,2
            assert status[0] == STATUS_DONE
            assert status[1] == STATUS_FAILED
            assert status[2] == STATUS_DONE

            for row in (0, 2):
                pi = row
                for k in _POOL_KEYS:
                    # expected = log10(sum over 3 chunks of 10**log)
                    dens = sum(np.power(10.0, truth[pi][k][c].astype(np.float64))
                               for c in range(3))
                    with np.errstate(divide="ignore"):
                        expected = np.where(dens > 0, np.log10(dens), -np.inf)
                    got = f[f"runs/{k}"][row].astype(np.float64)
                    # compare finite entries tightly; -inf entries must match
                    fin = np.isfinite(expected)
                    assert np.array_equal(np.isneginf(expected), np.isneginf(got))
                    assert np.allclose(got[fin], expected[fin], atol=1e-5, rtol=1e-5)


def test_all_empty_bins_stay_neginf():
    params = [[0.0, 0.0]]
    allneg = {k: np.full((1,) + _DIMS[k], -np.inf) for k in _POOL_KEYS}
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for c in range(2):
            p = os.path.join(d, f"c{c}.hdf5")
            _make_chunk_file(p, params, [STATUS_DONE], allneg)
            paths.append(p)
        out = os.path.join(d, "combined.hdf5")
        pool(paths, out, verbose=False)
        with h5py.File(out, "r") as f:
            for k in _POOL_KEYS:
                assert np.all(np.isneginf(f[f"runs/{k}"][0]))


if __name__ == "__main__":
    test_pool_sums_densities_and_handles_status()
    print("OK  test_pool_sums_densities_and_handles_status")
    test_all_empty_bins_stay_neginf()
    print("OK  test_all_empty_bins_stay_neginf")
    print("\nAll pool_chunks tests passed.")
