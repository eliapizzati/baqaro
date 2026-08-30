"""Diagnostic: compare the orphan merger-target resolution schemes.

Traces the "old" fallback order (Sink -> Descendant -> NestedParent at the final
snapshot) against the "new" one on small hand-built cases, so the difference
between the two schemes can be inspected directly rather than inferred from
population statistics.

Script-style: executes on import. Run it directly, never under pytest.
"""

import numpy as np


def resolve_old_loop(sink, desc, nest):
    out = sink.copy()
    for i in range(out.shape[0]):
        if out[i] == -1:
            out[i] = desc[i]
        if out[i] == -1:
            out[i] = nest[i]
    return out


def resolve_new_vectorized(sink, desc, nest):
    out = sink.copy()
    mask = out == -1
    if mask.any():
        out[mask] = desc[mask]
        mask2 = out == -1
        if mask2.any():
            out[mask2] = nest[mask2]
    return out


def resolve_parallel_sim(sink, desc, nest, num_workers, backend="threading"):
    # Simulate chunked parallel behavior by resolving chunks independently then concatenating
    n = sink.shape[0]
    if n == 0:
        return sink.copy()
    num_workers = max(1, min(num_workers, n))
    idx = np.linspace(0, n, num=num_workers+1, dtype=int)
    out_chunks = []
    for i in range(len(idx) - 1):
        s, e = int(idx[i]), int(idx[i+1])
        if s < e:
            ch = resolve_new_vectorized(sink[s:e], desc[s:e], nest[s:e])
            out_chunks.append(ch)
    return np.concatenate(out_chunks, axis=0) if out_chunks else sink.copy()


def random_arrays(n, seed=0):
    rng = np.random.default_rng(seed)
    # IDs in [0, 1e6), with -1 used as sentinel. Control frequency of -1 in each array.
    sink = rng.integers(-1, 1_000_000, size=n, endpoint=False, dtype=np.int64)
    desc = rng.integers(-1, 1_000_000, size=n, endpoint=False, dtype=np.int64)
    nest = rng.integers(-1, 1_000_000, size=n, endpoint=False, dtype=np.int64)
    # Increase sentinel frequency in sink/desc/nest independently
    # Force about p of entries to -1
    for arr, p in ((sink, 0.35), (desc, 0.25), (nest, 0.15)):
        mask = rng.random(n) < p
        arr[mask] = -1
    return sink, desc, nest


def run_suite():
    sizes = [0, 1, 3, 10, 999, 10_000, 12345]
    seeds = [1, 2, 98765]
    worker_settings = [1, 2, 3, 8]

    for n in sizes:
        for seed in seeds:
            sink, desc, nest = random_arrays(n, seed)
            ref = resolve_old_loop(sink, desc, nest)
            vec = resolve_new_vectorized(sink, desc, nest)
            if not np.array_equal(ref, vec):
                raise AssertionError(f"Vectorized mismatch for n={n}, seed={seed}")
            for w in worker_settings:
                par = resolve_parallel_sim(sink, desc, nest, num_workers=w)
                if not np.array_equal(ref, par):
                    raise AssertionError(f"Parallel mismatch for n={n}, seed={seed}, workers={w}")
    print("All merger logic equivalence tests passed.")


if __name__ == "__main__":
    run_suite()
