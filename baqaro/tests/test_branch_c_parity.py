"""
Fast-vs-reference parity for Branch C (tau = 0 analytical limit).
=================================================================

Locks fast/reference parity for the tau->0 limit: both engines accumulate
the arithmetic mean E[eta] = 10^mu * exp(0.5*(ln10)^2*sigma^2) (not the
median 10^mu), in constant-eps, madau+ and f_eff modes, and both honour the
same legacy toggle.

Branch C mass growth is fully analytical (no transfer table, no randomness
in the mass update), so the two engines must agree element-wise. Runs
without the sampler npz — pure Branch C (n_steps=0, transfer_function=None).
"""
import numpy as np
import pytest


SIGMA = 0.5
LN10_SQ_HALF = 2.6516504294495533


def _grow(engine, rad_efficiency_model, madau_feff_correction, size=10_000):
    from baqaro.core_functions.erdf_core_functions import Erdf

    erdf = Erdf(
        "log_normal_evol_halo_mass",
        {"log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": SIGMA},
        rng=np.random.default_rng(7),
    )
    # Spread of mu values via the halo-rate dependence
    log_rates = np.linspace(-1.0, 1.0, size).astype(np.float32)
    M0 = np.full(size, 1e6, dtype=np.float32)

    if engine == "fast":
        from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
        M_out, _ = evolve_BHs_fast(
            None, M0, 0.135, erdf,
            rad_efficiency=0.1, rad_efficiency_model=rad_efficiency_model,
            n_steps=0, log_halo_rates_array=log_rates,
            parallelize=False, num_workers=1, use_parallel_kernel=False,
            fw_p_threshold=0.0, madau_feff_correction=madau_feff_correction,
        )
    else:
        from baqaro.core_functions.bh_accretion import evolve_BHs
        M_out, _ = evolve_BHs(
            None, M0, 0.135, erdf,
            rad_efficiency=0.1, rad_efficiency_model=rad_efficiency_model,
            n_steps=0, log_halo_rates_array=log_rates,
            parallelize=False, num_workers=1,
            madau_feff_correction=madau_feff_correction,
        )
    return np.log(M_out.astype(np.float64) / 1e6)  # growth exponent


@pytest.mark.parametrize("model,feff", [
    ("constant", False),
    ("madau+", False),
    ("madau+", True),
])
def test_branch_c_fast_vs_reference_parity(model, feff, monkeypatch):
    monkeypatch.delenv("BAQARO_BRANCH_C_LEGACY", raising=False)
    g_fast = _grow("fast", model, feff)
    g_ref = _grow("reference", model, feff)
    # Analytical mass update -> element-wise agreement (float32 roundoff only).
    np.testing.assert_allclose(g_ref, g_fast, rtol=2e-4, atol=1e-7,
                               err_msg=f"Branch C parity broken ({model}, feff={feff})")


def test_branch_c_legacy_toggle(monkeypatch):
    """BAQARO_BRANCH_C_LEGACY=1 reverts BOTH engines to the geometric mean;
    the arith/legacy growth-exponent ratio is exp(0.5*(ln10)^2*sigma^2)."""
    monkeypatch.delenv("BAQARO_BRANCH_C_LEGACY", raising=False)
    g_arith = {e: _grow(e, "constant", False) for e in ("fast", "reference")}
    monkeypatch.setenv("BAQARO_BRANCH_C_LEGACY", "1")
    g_leg = {e: _grow(e, "constant", False) for e in ("fast", "reference")}

    expected_ratio = np.exp(LN10_SQ_HALF * SIGMA**2)
    for e in ("fast", "reference"):
        np.testing.assert_allclose(
            g_arith[e] / g_leg[e], expected_ratio, rtol=2e-4,
            err_msg=f"{e}: arith/legacy growth ratio != exp(0.5 ln^2(10) sigma^2)")
    np.testing.assert_allclose(g_leg["reference"], g_leg["fast"], rtol=2e-4, atol=1e-7,
                               err_msg="legacy-mode parity broken")


if __name__ == "__main__":
    class _MP:
        def delenv(self, k, raising=True):
            import os
            os.environ.pop(k, None)
        def setenv(self, k, v):
            import os
            os.environ[k] = v
    mp = _MP()
    for model, feff in [("constant", False), ("madau+", False), ("madau+", True)]:
        test_branch_c_fast_vs_reference_parity(model, feff, mp)
        print(f"OK  parity ({model}, feff={feff})")
    test_branch_c_legacy_toggle(mp)
    print("OK  legacy toggle")
