"""
Compare two training HDF5s statistic-by-statistic (chunked-pool vs reference).
=============================================================================

Used to validate chunked multi-node training: the pooled chunked output
should reproduce a single-node run *statistically* (it is a different,
equally-valid MC realization — see pool_chunks.py,
so it is NOT bit-identical). This script quantifies the agreement.

For each of log_qlfs / log_bhmfs / log_cerdfs / log_qhmfs it:
  * matches rows by parameter vector (rounded), intersection only,
  * compares ONLY if the stored bin edges match between the two files
    (so a schema-drifted axis — e.g. old 30-bin QLF vs new 40-bin — is
    auto-skipped rather than silently mis-aligned),
  * reports, over matched points and finite bins:
      median |Δ|, p90 |Δ|, max |Δ|   (Δ in dex of log10 number density),
      mean Δ (signed)                  -> detects systematic bias,
      and the same restricted to "high-signal" bins (log density > -8),
      which carry most objects and should agree very tightly.

Interpretation
--------------
* High-signal bins matching to ~<=0.02 dex with mean |bias| ~<= a few e-3
  dex => chunking introduces no bias; differences are MC noise. PASS.
* A nonzero systematic mean Δ that does NOT shrink in high-signal bins
  would indicate a real pooling/weight bug. FAIL.

Usage
-----
Env-driven (compares pooled BAQARO_NOTES_FILE vs BAQARO_REF_NOTES):
    BAQARO_SIM=... BAQARO_MAX_SNAP=59 BAQARO_USE_SUBSAMPLE=1 BAQARO_SUBSAMPLE_NB=500000 \
    BAQARO_NOTES_FILE=chunktest_z3_n2 BAQARO_REF_NOTES=chunktest_z3_n1 \
        python -m baqaro.emulation.compare_training

Explicit paths:
    python -m baqaro.emulation.compare_training --a A.hdf5 --b B.hdf5
"""

import argparse
import os

import h5py
import numpy as np

_STATS = ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs")
# Bin-edge dataset that defines each stat's axis (for the match check).
_BIN_DSET = {
    "log_qlfs": "emulation/log_lbins",
    "log_bhmfs": "emulation/log_mbins_bhmf",
    "log_cerdfs": "emulation/log_bins_cerdf",
    "log_qhmfs": "emulation/log_mbins_qhmf",
}
_PARAM_DECIMALS = 6
_HIGH_SIGNAL = -8.0   # log10 density above this => well-populated bin


def _load(path):
    with h5py.File(path, "r") as f:
        out = {
            "status": f["runs/status"][:],
            "params": f["runs/params"][:],
        }
        for k in _STATS:
            out[k] = f[f"runs/{k}"][:] if f"runs/{k}" in f else None
        out["_bins"] = {}
        for k, dset in _BIN_DSET.items():
            out["_bins"][k] = f[dset][:] if dset in f else None
    return out


def _done_index(d):
    idx = {}
    for row in np.flatnonzero(d["status"] == 1):
        idx[tuple(np.round(d["params"][row], _PARAM_DECIMALS))] = int(row)
    return idx


def _bins_match(a, b, k):
    ba, bb = a["_bins"].get(k), b["_bins"].get(k)
    if ba is None or bb is None:
        return False
    return ba.shape == bb.shape and np.allclose(ba, bb, rtol=1e-5, atol=1e-8)


def compare(path_a, path_b, verbose=True):
    """Compare two training HDF5s over the models both of them completed.

    Reports, per summary statistic, the largest and typical absolute
    difference between the two files on their common set of finished models,
    so a chunked-and-pooled run can be checked against a single-node
    reference. Returns the per-statistic difference summary.
    """
    a, b = _load(path_a), _load(path_b)
    if verbose:
        print(f"A (test) : {os.path.basename(path_a)}")
        print(f"B (ref)  : {os.path.basename(path_b)}")
    ia, ib = _done_index(a), _done_index(b)
    common = sorted(set(ia) & set(ib))
    if verbose:
        print(f"matched DONE points: {len(common)} "
              f"(A has {len(ia)} done, B has {len(ib)} done)\n")
    if not common:
        print("NO common DONE parameter points — cannot compare.")
        return {}

    rows_a = np.array([ia[k] for k in common])
    rows_b = np.array([ib[k] for k in common])

    results = {}
    print(f"{'stat':<12} {'bins?':<6} {'pts':>5} {'med|Δ|':>9} {'p90|Δ|':>9} "
          f"{'max|Δ|':>9} {'meanΔ':>10}   {'hi-sig med|Δ|':>13} {'hi-sig meanΔ':>13}")
    print("-" * 104)
    for k in _STATS:
        if a[k] is None or b[k] is None:
            print(f"{k:<12} {'n/a':<6} (absent in one file)")
            continue
        if a[k].shape[1:] != b[k].shape[1:]:
            print(f"{k:<12} {'SHAPE':<6} A{a[k].shape[1:]} vs B{b[k].shape[1:]} -> skip")
            continue
        if not _bins_match(a, b, k):
            print(f"{k:<12} {'NO':<6} bin edges differ -> skip (schema drift)")
            continue

        A = a[k][rows_a].astype(np.float64)
        B = b[k][rows_b].astype(np.float64)
        both_finite = np.isfinite(A) & np.isfinite(B)
        d = (A - B)[both_finite]
        if d.size == 0:
            print(f"{k:<12} {'YES':<6} (no jointly-finite bins)")
            continue
        ad = np.abs(d)
        # high-signal subset
        hi = (np.minimum(A, B) > _HIGH_SIGNAL) & both_finite
        dhi = (A - B)[hi]
        med_hi = np.median(np.abs(dhi)) if dhi.size else np.nan
        mean_hi = np.mean(dhi) if dhi.size else np.nan

        results[k] = dict(
            n_pts=len(common), med=np.median(ad), p90=np.percentile(ad, 90),
            max=ad.max(), mean=np.mean(d), med_hi=med_hi, mean_hi=mean_hi,
            n_hi=int(dhi.size),
        )
        print(f"{k:<12} {'YES':<6} {len(common):>5} {np.median(ad):>9.4f} "
              f"{np.percentile(ad,90):>9.4f} {ad.max():>9.4f} {np.mean(d):>+10.5f}   "
              f"{med_hi:>13.4f} {mean_hi:>+13.5f}")

    # verdict heuristic
    print("\nVERDICT (heuristic):")
    ok = True
    for k, r in results.items():
        bias_ok = abs(r["mean_hi"]) < 5e-3 if np.isfinite(r["mean_hi"]) else True
        tight_ok = r["med_hi"] < 0.02 if np.isfinite(r["med_hi"]) else True
        flag = "OK " if (bias_ok and tight_ok) else "CHECK"
        if not (bias_ok and tight_ok):
            ok = False
        print(f"  {k:<12} {flag}  (high-signal med|Δ|={r['med_hi']:.4f} dex, "
              f"bias={r['mean_hi']:+.5f} dex over {r['n_hi']} bins)")
    print("\n=> " + ("PASS: chunked pool matches reference within MC noise, no bias."
                     if ok else
                     "REVIEW: at least one stat shows tight-bin disagreement/bias — inspect."))
    return results


def _env_paths():
    from baqaro.emulation.pool_chunks import training_path
    from baqaro.utils.sim_config import max_snap
    notes_a = os.environ.get("BAQARO_NOTES_FILE")
    notes_b = os.environ.get("BAQARO_REF_NOTES")
    if not notes_a or not notes_b:
        raise SystemExit("Set BAQARO_NOTES_FILE (test/pooled) and BAQARO_REF_NOTES (reference).")
    return (training_path(notes=notes_a, max_snap=max_snap, n_chunks=1),
            training_path(notes=notes_b, max_snap=max_snap, n_chunks=1))


def main(argv=None):
    """CLI entry point: compare ``--a`` against ``--b``.

    With neither given, both paths are resolved from the environment the same
    way the training pipeline builds them.
    """
    ap = argparse.ArgumentParser(description="Compare two training HDF5s.")
    ap.add_argument("--a", help="test file (e.g. pooled chunked)")
    ap.add_argument("--b", help="reference file (e.g. single-node)")
    args = ap.parse_args(argv)
    if args.a and args.b:
        pa, pb = args.a, args.b
    else:
        pa, pb = _env_paths()
    for p in (pa, pb):
        if not os.path.exists(p):
            raise SystemExit(f"missing file: {p}")
    compare(pa, pb)


if __name__ == "__main__":
    main()
