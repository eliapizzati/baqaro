"""
LAYER 4 - Evaluate the candidate fixes for the madau-sampler bias.
=================================================================

Three treatments of the (n_steps-1)-sub-step HISTORY growth, evaluated on
IDENTICAL eta draws so the only difference is how (1 - eps(eta)) is handled:

  direct : Sum_j eta_j (1 - eps(eta_j))            (ground truth = Branch A)
  approx : (1 - eps(median)) * Sum_j eta_j         (current Branch B / mu_scale)
  fixed  : f_eff(mu,sigma) * approx                (the f_eff correction, mean-matched)

where f_eff = E[eta(1-eps)] / (E[eta](1-eps(median))) makes the MEAN of `fixed`
equal the mean of `direct` BY CONSTRUCTION. The question (see the derivation in
madau_feff.py) is whether a mean-only scale ALSO closes the tail (p84/p99), given that Madau
(1-eps) INCREASES with eta so the exact sum is heavier-tailed than a uniformly
scaled lognormal sum.

This layer uses a direct Monte-Carlo sum-of-lognormals (the "ideal" table, free
of the real table's finite tail-resolution error documented as the residual) so it isolates the MADAU correction alone.

Two evaluations:
  (a) history_block_recovery  - single-block percentile recovery over (mu,sigma,N).
  (b) compounded_fix_eval     - multi-snapshot evolve on real rate histories
                                (same draws, 3 treatments) -> residual BHMF shift
                                before vs after the fix, vs z.

Pure numpy; needs only the rate cache for (b). Fast.

Run: python -m baqaro.tests.madau_sampler_bias.test_layer4_fix_eval
"""
import os
import numpy as np
import pytest

import qhtools.utils.natconst as nc
from . import _common as C

OUTDIR = os.path.join(os.path.dirname(__file__), "out")
RE = 0.1                       # base radiative efficiency
INV_1M_RE = 1.0 / (1.0 - RE)
_GC_PREFAC = (1 - RE) / RE / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms * nc.year * 1e9


def growth_const(delta_t_gyr, n_eff):
    """Per-sub-step growth coefficient, matching evolve_BHs_fast Branch B."""
    return _GC_PREFAC * (delta_t_gyr / n_eff)


def _history_treatments(mu, sigma, n_hist, size, rng):
    """Return (direct, approx, fixed) history growth-exponent SUMS (no gc).

    All three on the SAME eta draws. n_hist = n_steps-1 history sub-steps.
    """
    z = rng.standard_normal((size, n_hist))
    eta = 10.0 ** (mu + sigma * z)                       # (size, n_hist)
    one_minus = C.one_minus_eps(eta, RE)
    direct = np.sum(eta * one_minus, axis=1) * INV_1M_RE
    median = 10.0 ** mu
    sum_eta = np.sum(eta, axis=1)
    approx = (median * C.one_minus_eps(median, RE) * INV_1M_RE) * (sum_eta / median)
    # = (1-eps(median))*INV_1M_RE * sum_eta -- the mu_scale form
    feff = C.f_eff_fix(mu, sigma, RE)
    fixed = approx * feff
    return direct, approx, fixed, feff


def history_block_recovery():
    """(a) Single-block percentile recovery over (mu, sigma, n_hist)."""
    print("\nLAYER 4a: history-block percentile recovery (table vs fix vs direct)")
    print("Residual = treatment - direct, in dex of the history sum. Want ~0.\n")
    rng = np.random.default_rng(7)
    size = 300_000
    pcts = (50, 84, 99, 99.9)
    print(f"{'mu':>5} {'sigma':>5} {'n_hist':>6} {'f_eff':>6} | "
          + " ".join(f"{'ap'+str(p):>8}" for p in pcts) + " |"
          + " ".join(f"{'fx'+str(p):>8}" for p in pcts))
    for mu in (-1.0, 0.0):
        for sigma in (0.3, 0.5, 0.7, 1.0):
            for n_hist in (50, 500):
                d, a, f, feff = _history_treatments(mu, sigma, n_hist, size, rng)
                ld = np.log10(d)
                ra = {p: np.percentile(np.log10(a), p) - np.percentile(ld, p) for p in pcts}
                rf = {p: np.percentile(np.log10(f), p) - np.percentile(ld, p) for p in pcts}
                print(f"{mu:5.1f} {sigma:5.2f} {n_hist:6d} {feff:6.3f} | "
                      + " ".join(f"{ra[p]:+8.4f}" for p in pcts) + " |"
                      + " ".join(f"{rf[p]:+8.4f}" for p in pcts))
    print("\n(ap = current approx residual; fx = after the f_eff correction, which zeroes the mean/p50;\n"
          " any remaining fx-p99 is the tail-shape residual a mean-only scale can't fix.)")


def _evolve_three(rate_cache, redshifts, dt, std_0, logtcoh, col_idx, seed=12345,
                  checkpoints=None, growth_max=4.6, i_start=1,
                  birth_snap=None, seed_mass=None):
    """Multi-snapshot numpy evolve, three treatments on identical draws.

    Returns dict snap -> (M_direct, M_approx, M_fixed). The ERDF mu per halo
    is FID eta0 + evol*log_rate (matches production); sigma = std_0 fixed.
    """
    from . import test_layer3_compounding as L3
    if checkpoints is None:
        checkpoints = L3.CHECKPOINTS
    eta0 = L3.FID["log_eta_mean_0"]
    evol = L3.FID["log_eta_mean_evol"]
    tau = 10.0 ** (logtcoh - 6.0)
    n = col_idx.size
    rng = np.random.default_rng(seed)
    if birth_snap is not None:
        Md = np.zeros(n, dtype=np.float64)
        early = (birth_snap >= 0) & (birth_snap <= i_start)
        Md[early] = seed_mass[early]
    else:
        Md = np.full(n, L3.M_SEED, dtype=np.float64)
    Ma = Md.copy(); Mf = Md.copy()
    snaps = {}
    for i in range(i_start, redshifts.size):
        if birth_snap is not None and i > i_start:
            born_now = birth_snap == i
            if np.any(born_now):
                Md[born_now] = seed_mass[born_now]
                Ma[born_now] = seed_mass[born_now]
                Mf[born_now] = seed_mass[born_now]
        row = np.asarray(rate_cache[i])[col_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.where(row > 0.0, np.log10(np.maximum(row, 1e-30)), -30.0)
        mu = eta0 + evol * lr                              # per-halo median log-eta
        n_steps = max(1, int(dt[i] * 1e3 / tau))
        n_eff = min(n_steps, 1001)
        gc = growth_const(dt[i], n_eff)
        n_hist = n_eff - 1
        accreting = np.isfinite(mu) & (mu > -25)
        # last step (exact, identical for all three)
        zl = rng.standard_normal(n)
        eta_last = 10.0 ** (mu + std_0 * zl)
        gl = eta_last * C.one_minus_eps(eta_last, RE) * INV_1M_RE * gc
        if n_hist >= 1:
            z = rng.standard_normal((n, n_hist))
            eta = 10.0 ** (mu[:, None] + std_0 * z)
            sum_eta = np.sum(eta, axis=1)
            direct_h = np.sum(eta * C.one_minus_eps(eta, RE), axis=1) * INV_1M_RE
            median = 10.0 ** mu
            approx_h = C.one_minus_eps(median, RE) * INV_1M_RE * sum_eta
            # per-halo f_eff via the analytic factor (sigma fixed -> depends on mu)
            feff = C.f_eff_fix_vec(mu, std_0, RE)
            fixed_h = approx_h * feff
        else:
            direct_h = approx_h = fixed_h = np.zeros(n)
        gd = np.clip(gl + direct_h * gc, None, growth_max)
        ga = np.clip(gl + approx_h * gc, None, growth_max)
        gf = np.clip(gl + fixed_h * gc, None, growth_max)
        gd = np.where(accreting, gd, 0.0)
        ga = np.where(accreting, ga, 0.0)
        gf = np.where(accreting, gf, 0.0)
        Md *= np.exp(gd); Ma *= np.exp(ga); Mf *= np.exp(gf)
        if i in checkpoints:
            snaps[i] = (Md.copy(), Ma.copy(), Mf.copy())
    return snaps


def compounded_fix_eval(growth_max=4.6):
    """(b) Compounded residual BHMF shift before vs after the fix, vs z."""
    from . import test_layer3_compounding as L3
    rate_path = C.find_rate_cache(L3.RATE_PATTERN)
    if rate_path is None or not os.path.exists(L3.WEIGHTS_CACHE):
        print("[skip 4b] rate/weights cache missing")
        return
    rate_cache = np.load(rate_path, mmap_mode="r")
    w = np.load(L3.WEIGHTS_CACHE)
    redshifts, dt = C.load_cadence("L2800N10080", 71)
    rng = np.random.default_rng(0)
    col = np.sort(rng.choice(w.size, size=400_000, replace=False))
    snaps = _evolve_three(rate_cache, redshifts, dt, L3.FID["std_0"],
                          L3.FID["logtcoherence"], col, growth_max=growth_max)
    ww = w[col]
    print(f"\nLAYER 4b: compounded weighted-BHMF residual vs direct, fiducial, "
          f"g_max={growth_max} (400k cols)")
    print("approx = current (biased); fixed = after the f_eff correction. shift = treatment-direct [dex]\n")
    print(f"{'z':>5} | {'ap_p50':>7} {'ap_p84':>7} {'ap_p99':>7} | "
          f"{'fx_p50':>7} {'fx_p84':>7} {'fx_p99':>7}")
    for snap in sorted(snaps):
        Md, Ma, Mf = snaps[snap]
        grown = (Ma > L3.M_SEED * 1.0001) | (Md > L3.M_SEED * 1.0001)
        lmd = np.log10(Md[grown]); lma = np.log10(Ma[grown]); lmf = np.log10(Mf[grown])
        wg = ww[grown]
        ap = {p: L3._weighted_pctile(lma, wg, p) - L3._weighted_pctile(lmd, wg, p)
              for p in (50, 84, 99)}
        fx = {p: L3._weighted_pctile(lmf, wg, p) - L3._weighted_pctile(lmd, wg, p)
              for p in (50, 84, 99)}
        print(f"{L3.CHECKPOINTS[snap]:5.2f} | {ap[50]:+7.4f} {ap[84]:+7.4f} {ap[99]:+7.4f} | "
              f"{fx[50]:+7.4f} {fx[84]:+7.4f} {fx[99]:+7.4f}")


def main():
    history_block_recovery()
    compounded_fix_eval(growth_max=4.6)
    compounded_fix_eval(growth_max=50.0)


# ----------------------------------------------------------------------
# pytest
# ----------------------------------------------------------------------
def test_feff_corrects_mean_exactly():
    """The f_eff correction makes the history-sum MEAN match direct to <0.5%."""
    rng = np.random.default_rng(3)
    for mu in (-1.0, 0.0):
        for sigma in (0.3, 0.7, 1.0):
            d, a, f, _ = _history_treatments(mu, sigma, 200, 400_000, rng)
            assert abs(np.mean(f) / np.mean(d) - 1.0) < 0.005, (mu, sigma)
            # and it improves on the uncorrected approx mean
            assert abs(np.mean(f) / np.mean(d) - 1.0) <= abs(np.mean(a) / np.mean(d) - 1.0)


def test_feff_improves_but_may_leave_tail():
    """The f_eff correction closes p50 (mean); quantify the residual tail (p99)."""
    rng = np.random.default_rng(5)
    d, a, f, _ = _history_treatments(0.0, 1.0, 500, 400_000, rng)
    ld = np.log10(d)
    res_a_p50 = abs(np.percentile(np.log10(a), 50) - np.percentile(ld, 50))
    res_f_p50 = abs(np.percentile(np.log10(f), 50) - np.percentile(ld, 50))
    assert res_f_p50 < res_a_p50, "fix should improve the median"


if __name__ == "__main__":
    main()
