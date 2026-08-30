"""Benchmark black-hole seeding throughput.

Compares:
1) Baseline implementation (simple normal draw + power + multiply)
2) Optimized spawn_BHs implementation in core_functions/bh_seeding.py

Usage examples:
  python baqaro/tests/benchmark_bh_seeding.py --n 50000000 --dtype float32 --mode both
  python baqaro/tests/benchmark_bh_seeding.py --n 700000000 --dtype float32 --mode scatter --repeats 3
"""

import argparse
import time
import numpy as np

from baqaro.core_functions.bh_seeding import spawn_BHs


def baseline_spawn_bhs(m_halos, logfseed, sigmaseed=None, rng=None):
    """Reference implementation matching the pre-optimization math."""
    if sigmaseed is None:
        return m_halos * 10 ** logfseed

    if sigmaseed < 0:
        raise ValueError("sigmaseed must be >= 0 or None")

    if rng is None:
        rng = np.random.default_rng()

    logfseed_scattered = rng.normal(loc=logfseed, scale=sigmaseed, size=np.shape(m_halos))
    return m_halos * 10 ** logfseed_scattered


def _run_timed(func, m_halos, logfseed, sigmaseed, seed, repeats):
    times = []
    out = None

    # Warm-up to avoid one-time setup effects in timed region.
    rng = np.random.default_rng(seed)
    out = func(m_halos, logfseed, sigmaseed=sigmaseed, rng=rng)
    _ = float(np.mean(out))

    for i in range(repeats):
        rng = np.random.default_rng(seed + i)
        t0 = time.perf_counter()
        out = func(m_halos, logfseed, sigmaseed=sigmaseed, rng=rng)
        t1 = time.perf_counter()
        _ = float(np.mean(out))
        times.append(t1 - t0)

    return np.asarray(times, dtype=np.float64), out


def _summarize(label, times, n_obj):
    t_mean = float(np.mean(times))
    t_std = float(np.std(times))
    obj_per_s = n_obj / t_mean
    mobj_per_s = obj_per_s / 1e6
    return {
        "label": label,
        "time_mean_s": t_mean,
        "time_std_s": t_std,
        "objects_per_s": obj_per_s,
        "million_objects_per_s": mobj_per_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark BH seeding performance")
    parser.add_argument("--n", type=int, default=10_000_000,
                        help="Number of halo objects")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Input halo mass dtype")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of timed repeats")
    parser.add_argument("--seed", type=int, default=12345,
                        help="RNG seed")
    parser.add_argument("--logfseed", type=float, default=-4.2,
                        help="Mean log10 seed fraction")
    parser.add_argument("--sigmaseed", type=float, default=0.3,
                        help="Scatter in dex for scatter mode")
    parser.add_argument("--mode", choices=["deterministic", "scatter", "both"], default="both",
                        help="Which benchmark mode(s) to run")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    n = args.n

    print("=" * 80)
    print("BH seeding benchmark")
    print("=" * 80)
    print(f"N objects        : {n:,}")
    print(f"dtype            : {args.dtype}")
    print(f"repeats          : {args.repeats}")
    print(f"logfseed         : {args.logfseed}")
    print(f"sigmaseed (dex)  : {args.sigmaseed}")

    # Realistic positive halo masses, log-uniform over a broad dynamic range.
    rng_init = np.random.default_rng(args.seed)
    logm = rng_init.uniform(10.0, 14.0, size=n).astype(dtype, copy=False)
    m_halos = np.power(dtype(10.0), logm, dtype=dtype)

    modes = [args.mode] if args.mode != "both" else ["deterministic", "scatter"]

    for mode in modes:
        sig = None if mode == "deterministic" else args.sigmaseed

        print("\n" + "-" * 80)
        print(f"Mode: {mode}")
        print("-" * 80)

        times_base, out_base = _run_timed(
            baseline_spawn_bhs,
            m_halos,
            args.logfseed,
            sig,
            seed=args.seed,
            repeats=args.repeats,
        )
        times_opt, out_opt = _run_timed(
            spawn_BHs,
            m_halos,
            args.logfseed,
            sig,
            seed=args.seed,
            repeats=args.repeats,
        )

        s_base = _summarize("baseline", times_base, n)
        s_opt = _summarize("optimized", times_opt, n)

        speedup = s_base["time_mean_s"] / s_opt["time_mean_s"]

        print(
            f"Baseline  : {s_base['time_mean_s']:.4f} +/- {s_base['time_std_s']:.4f} s  "
            f"({s_base['million_objects_per_s']:.2f} M obj/s)"
        )
        print(
            f"Optimized : {s_opt['time_mean_s']:.4f} +/- {s_opt['time_std_s']:.4f} s  "
            f"({s_opt['million_objects_per_s']:.2f} M obj/s)"
        )
        print(f"Speedup   : {speedup:.3f}x")

        # Sanity checks: exact equality in deterministic mode, statistical similarity in scatter mode.
        if sig is None:
            max_abs = float(np.max(np.abs(out_base - out_opt)))
            print(f"Deterministic max |delta|: {max_abs:.6e}")
        else:
            mb = float(np.mean(out_base))
            mo = float(np.mean(out_opt))
            sb = float(np.std(out_base))
            so = float(np.std(out_opt))
            print(f"Scatter mean/std baseline : {mb:.6e} / {sb:.6e}")
            print(f"Scatter mean/std optimized: {mo:.6e} / {so:.6e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
