"""Diagnostic: confirm worker processes draw reproducible random numbers.

The training pipeline forks a pool of workers, each of which must get an
independent but SEED-DETERMINED random stream -- otherwise a rerun of the same
training point would not reproduce, and the emulator would be fitted to noise
that cannot be regenerated.

Script-style: executes on import. Run it directly, never under pytest.
"""

import numpy as np
from multiprocessing import Pool
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def worker_compute(x_chunk, seed):
    rng = np.random.default_rng(seed)
    # Deterministic transformation using RNG
    noise = rng.standard_normal(len(x_chunk))
    return x_chunk * np.exp(noise * 0.01)


def run_once(x, num_workers):
    # Mimic bh_accretion_test_new child seed derivation
    parent_rng = np.random.default_rng(12345)
    child_seeds = [int(parent_rng.integers(0, 2**32)) for _ in range(num_workers)]

    chunks = np.array_split(x, num_workers)
    if HAS_JOBLIB:
        results = Parallel(n_jobs=num_workers, backend="loky")(
            delayed(worker_compute)(chunk, seed) for chunk, seed in zip(chunks, child_seeds)
        )
    else:
        with Pool(processes=num_workers) as pool:
            results = pool.starmap(worker_compute, list(zip(chunks, child_seeds)))
    return np.concatenate(results)


def main():
    num_workers = 4  # Keep constant as requested
    x = np.ones(100000, dtype=np.float64)

    y1 = run_once(x, num_workers)
    y2 = run_once(x, num_workers)

    # Check exact equality
    if np.array_equal(y1, y2):
        print("Determinism OK: two runs match exactly with constant workers.")
    else:
        max_abs_diff = np.max(np.abs(y1 - y2))
        print(f"Determinism FAILED: max abs diff {max_abs_diff}")
        raise SystemExit(1)

    # Optional: show non-determinism if worker count changes
    y3 = run_once(x, num_workers + 1)
    if np.array_equal(y1, y3):
        print("Unexpected: results matched even with different workers.")
    else:
        print("As expected: different worker count changes RNG path (non-deterministic across worker counts).")


if __name__ == "__main__":
    main()
