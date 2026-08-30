"""
LAYER 2 - Single-snapshot kernel bias: real Branch B (table) vs Branch A (direct).
=================================================================================

Calls the PRODUCTION kernel ``evolve_BHs_fast`` on a synthetic population at a
fixed (mu, sigma), once with the transfer table (Branch B, the approximation)
and once with ``tf=None`` (Branch A, exact direct sub-step sampling). Same RNG
seed / Erdf, so the ONLY difference is the engine path.

It isolates two effects:

  (A) The MADAU mean bias. The robust statistic is the growth-EXPONENT mean
      ratio  R_exp = mean(ln M/M0)_table / mean(...)_direct. For madau+ this is
      < 1 (under-growth) and matches Layer 1 via the last-step-exact dilution
      R_exp = bias + (1 - bias)/n_steps   (the last sub-step is exact in both).
      For constant eps it is ~1 (the table mean is exact).

  (B) The table's INTRINSIC tail-resolution error, SEPARATE from madau: visible
      in the constant-eps p99.9 diff at high sigma (the inverse-CDF table has
      finite tail resolution; cf. the sub-step cap residual). Reported
      alongside so it is not mistaken for the madau bias.

Per-snapshot dex diffs are tiny by construction (one snapshot); the OBSERVABLE
bias is the COMPOUNDED effect measured in Layer 3. L_bol diffs are reported to
confirm "QLF largely spared".

Run: python -m baqaro.tests.madau_sampler_bias.test_layer2_kernel_snapshot
"""
import numpy as np
import pytest

from . import _common as C

_TF = None


def _sampler():
    global _TF
    if _TF is None:
        _TF = C.load_sampler()
    return _TF


def compare_one(tf, mu, sigma, n_steps, delta_t, size=200_000, rad_model="madau+",
                seed=12345):
    ln_tab, L_tab = C.run_kernel_growth(tf, mu, sigma, n_steps, delta_t,
                                        size=size, rad_model=rad_model, seed=seed)
    ln_dir, L_dir = C.run_kernel_growth(None, mu, sigma, n_steps, delta_t,
                                        size=size, rad_model=rad_model, seed=seed)
    # growth-exponent mean ratio (robust; this is what Layer 1 predicts)
    R_exp = float(np.mean(ln_tab) / np.mean(ln_dir))
    # per-snapshot percentile diff in dex of final M  (= (g_t - g_d)/ln10)
    pcts = (50, 84, 99, 99.9)
    dM = {p: float(np.percentile(ln_tab, p) - np.percentile(ln_dir, p)) / C.LN10
          for p in pcts}
    Ldex = C.pdiff_dex(L_tab, L_dir, pcts=(50, 99))
    return dict(R_exp=R_exp, dM=dM, Ldex=Ldex)


def layer1_dilution_pred(mu, sigma, n_steps, rad_model):
    """Layer-1 mean bias diluted by the exact last sub-step."""
    if rad_model == "constant":
        return 1.0
    bias = (C.per_substep_growth_mean_approx(mu, sigma)
            / C.per_substep_growth_mean_exact(mu, sigma))
    return bias + (1.0 - bias) / n_steps


def main():
    tf = _sampler()
    delta_t = 0.10           # z~2-ish snapshot
    mu = -1.0
    print(f"\nLAYER 2: kernel Branch B (table) vs Branch A (direct), dt={delta_t} Gyr, "
          f"mu={mu}, size=200k")
    print("R_exp = mean ln-growth ratio (table/direct); want madau<1, constant~1.")
    print("dM_pXX = per-snapshot dex diff of final M; dL = L_bol dex diff.\n")
    for model in ("madau+", "constant"):
        print(f"=== {model} ===")
        print(f"{'sigma':>6} {'n_steps':>8} | {'R_exp':>8} {'L1_pred':>8} | "
              f"{'dM_p50':>8} {'dM_p84':>8} {'dM_p99':>8} {'dM_p99.9':>9} | "
              f"{'dL_p50':>7} {'dL_p99':>7}")
        for sigma in (0.3, 0.5, 0.7, 1.0):
            for n_steps in (10, 50, 500):
                r = compare_one(tf, mu, sigma, n_steps, delta_t, rad_model=model)
                pred = layer1_dilution_pred(mu, sigma, n_steps, model)
                print(f"{sigma:6.2f} {n_steps:8d} | {r['R_exp']:8.4f} {pred:8.4f} | "
                      f"{r['dM'][50]:+8.4f} {r['dM'][84]:+8.4f} {r['dM'][99]:+8.4f} "
                      f"{r['dM'][99.9]:+9.4f} | {r['Ldex'][50]:+7.4f} {r['Ldex'][99]:+7.4f}")
        print()
    print("Reading: constant-eps R_exp~1 confirms the table mean is exact; the\n"
          "constant-eps dM_p99.9 blowup at sigma=1.0 is the table's intrinsic tail\n"
          "error (NOT madau). The madau R_exp<1 deficit (and its sigma growth) is\n"
          "the Layer-1 mean bias, reproduced by the real kernel.")


# ----------------------------------------------------------------------
# pytest
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def tf():
    try:
        return _sampler()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")


@pytest.mark.parametrize("sigma", [0.3, 0.5, 1.0])
def test_constant_eps_mean_is_exact(tf, sigma):
    """Constant eps: table growth-exponent mean matches direct (<0.5%)."""
    r = compare_one(tf, mu=-1.0, sigma=sigma, n_steps=200, delta_t=0.1,
                    rad_model="constant")
    assert abs(r["R_exp"] - 1.0) < 0.005, f"constant R_exp={r['R_exp']} (sigma={sigma})"
    assert abs(r["dM"][50]) < 0.01


def test_madau_undergrows_and_matches_layer1(tf):
    """madau+: table under-grows the mean; magnitude grows with sigma and
    matches the analytical Layer-1 prediction (with last-step dilution)."""
    prev = -1.0
    for sigma in (0.3, 0.5, 1.0):
        r = compare_one(tf, mu=-1.0, sigma=sigma, n_steps=200, delta_t=0.1,
                        rad_model="madau+")
        pred = layer1_dilution_pred(-1.0, sigma, 200, "madau+")
        assert r["R_exp"] < 1.0, f"expected under-growth at sigma={sigma}"
        assert abs(r["R_exp"] - pred) < 0.01, (
            f"kernel R_exp={r['R_exp']:.4f} != Layer1 pred {pred:.4f} (sigma={sigma})")
        deficit = 1.0 - r["R_exp"]
        assert deficit > prev, "deficit should grow with sigma"
        prev = deficit


def test_qlf_median_largely_spared(tf):
    """L_bol median (QLF proxy) bias << mass bias (uses exact last sub-step)."""
    r = compare_one(tf, mu=-1.0, sigma=0.5, n_steps=200, delta_t=0.1,
                    rad_model="madau+")
    assert abs(r["Ldex"][50]) < 0.01, f"L_bol p50 not spared: {r['Ldex']}"


if __name__ == "__main__":
    main()
