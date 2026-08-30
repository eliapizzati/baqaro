"""
Regression tests for the f_eff mean correction.
===============================================

The transfer-table Branch B applies the madau+ efficiency once at the median eta
(``mu_scale``), which under-grows the bright end. The
correction multiplies ``mu_scale`` by ``f_eff(mu,sigma) = E[eta(1-eps)]/((1-eps(median))E[eta])``,
restoring the per-sub-step mean. It is OFF by default (env
``BAQARO_MADAU_FEFF_CORRECTION`` / ``evolve_BHs_fast(madau_feff_correction=...)``)
so existing runs stay bit-identical.

These tests assert:
  1. f_eff >= 1 and -> 1 in the constant-eps (tiny sigma) limit.
  2. OFF (default) is BIT-IDENTICAL to the uncorrected behaviour, and the toggle
     never touches the constant-eps path.
  3. ON makes the table mean match exact direct sampling (Branch A) under madau+,
     while OFF leaves it biased low -- across mu, sigma, and both kernel modes.

The kernel tests need the lognormal sampler npz; they skip cleanly if absent.
"""
import os
import numpy as np
import pytest

from baqaro.core_functions.madau_feff import feff_per_halo
from baqaro.core_functions.erdf_core_functions import Erdf
from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast


# ----------------------------------------------------------------------
# 1. f_eff helper properties (no sampler needed)
# ----------------------------------------------------------------------
def test_feff_ge_one_and_constant_limit():
    mus = np.array([-3.0, -1.0, 0.0, 0.5])
    for sigma in (0.3, 0.54, 1.0):
        f = feff_per_halo(mus, sigma)
        assert np.all(f >= 1.0 - 1e-9), f"f_eff must be >=1 (sigma={sigma}): {f}"
    # tiny sigma -> eps ~ constant over the (degenerate) distribution -> f_eff ~ 1
    f_narrow = feff_per_halo(np.array([-1.0, 0.0]), 0.02)
    assert np.all(np.abs(f_narrow - 1.0) < 1e-3), f_narrow
    # variable-sigma (B-2D) path agrees with scalar at matching sigma to within
    # the sigma-grid bilinear interpolation error (~1e-4).
    f2d = feff_per_halo(np.array([-1.0, 0.0]), np.array([0.54, 0.54]))
    f1d = feff_per_halo(np.array([-1.0, 0.0]), 0.54)
    assert np.allclose(f2d, f1d, atol=1e-3)


# ----------------------------------------------------------------------
# kernel-level tests
# ----------------------------------------------------------------------
def _sampler():
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    return load_3d_sampler("Universal_Lognormal_Sampler_final.npz")


@pytest.fixture(scope="module")
def tf():
    if os.environ.get("BAQARO_SOURCE_DIR") is None and not os.environ.get("CI_FORCE"):
        pass  # default machine_igm path; still try
    try:
        return _sampler()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")


def _run(tf, mu, sigma, n_steps, dt, size=150_000, model="madau+", feff=None,
         seed=12345, par=False):
    erdf = Erdf("log_normal_evol_halo_mass",
                {"log_eta_mean_0": mu, "log_eta_mean_evol": 0.0, "std_0": sigma},
                rng=np.random.default_rng(seed))
    M = np.full(size, 1e6, dtype=np.float32)
    rates = np.zeros(size, dtype=np.float32)
    Mo, _ = evolve_BHs_fast(tf, M, dt, erdf, rad_efficiency=0.1, n_steps=n_steps,
                            log_halo_rates_array=rates, rad_efficiency_model=model,
                            parallelize=False, use_parallel_kernel=par,
                            madau_feff_correction=feff)
    return np.log(np.asarray(Mo, dtype=np.float64) / 1e6)


def test_off_is_bit_identical(tf):
    """Default (None) == explicit OFF, byte for byte (mu_scale *= 1.0 is exact)."""
    g_none = _run(tf, -1.0, 0.5, 200, 0.1, feff=None)
    g_off = _run(tf, -1.0, 0.5, 200, 0.1, feff=False)
    assert np.array_equal(g_none, g_off)


def test_constant_eps_untouched(tf):
    """The toggle must never change the constant-eps path (f_eff==1 there)."""
    g_off = _run(tf, -1.0, 0.5, 200, 0.1, model="constant", feff=False)
    g_on = _run(tf, -1.0, 0.5, 200, 0.1, model="constant", feff=True)
    assert np.array_equal(g_off, g_on)


@pytest.mark.parametrize("mu,sigma", [(-1.0, 0.3), (-1.0, 0.7), (0.0, 0.5)])
def test_on_matches_direct_off_is_biased(tf, mu, sigma):
    """madau+: ON makes the table mean match exact direct; OFF is biased low."""
    g_dir = _run(None, mu, sigma, 200, 0.1)           # Branch A exact ground truth
    g_off = _run(tf, mu, sigma, 200, 0.1, feff=False)
    g_on = _run(tf, mu, sigma, 200, 0.1, feff=True)
    r_off = np.mean(g_off) / np.mean(g_dir)
    r_on = np.mean(g_on) / np.mean(g_dir)
    # OFF under-grows (more at larger sigma); ON restores the mean to direct.
    assert r_off < 1.0, f"OFF should be biased low (got {r_off:.4f})"
    assert abs(r_on - 1.0) < 0.005, f"ON should match direct (got {r_on:.4f})"
    assert abs(r_on - 1.0) < abs(r_off - 1.0), "fix must improve on the bias"


def test_parallel_kernel_path_also_fixed(tf):
    """The prange kernel (single-process production path) honours the fix too."""
    g_dir = _run(None, -1.0, 0.5, 200, 0.1)
    g_on = _run(tf, -1.0, 0.5, 200, 0.1, feff=True, par=True)
    assert abs(np.mean(g_on) / np.mean(g_dir) - 1.0) < 0.005


if __name__ == "__main__":
    test_feff_ge_one_and_constant_limit()
    print("OK  f_eff helper properties")
    t = _sampler()
    test_off_is_bit_identical(t); print("OK  OFF bit-identical")
    test_constant_eps_untouched(t); print("OK  constant-eps untouched")
    for mu, s in [(-1.0, 0.3), (-1.0, 0.7), (0.0, 0.5)]:
        test_on_matches_direct_off_is_biased(t, mu, s)
    print("OK  ON matches direct / OFF biased")
    test_parallel_kernel_path_also_fixed(t); print("OK  parallel kernel fixed")
