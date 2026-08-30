"""
Regression test for the consistent sub-step cap in bh_accretion_fast.py.
=============================================================================

The transfer-function table spans at most N=1000 summands, so for
n_steps-1 > 1000 the history-sum lookup saturates at 1000 terms. The
EFFECTIVE sub-step count (`n_eff = min(n_steps, n_max_table+1)`) must be used
for BOTH the table lookup AND growth_const, so the mean growth is
n_steps-independent (the physical invariant: finer sub-stepping cannot change
the time-averaged accretion).

This test asserts the invariant: with delta_t and ERDF fixed, the mean log
growth is flat across n_steps that straddle the N=1000 cap.

Needs the real sampler npz (~764 MB, not in git). Skips if absent.
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BAQARO_SOURCE_DIR") is None,
    reason="needs BAQARO_SOURCE_DIR + the lognormal sampler npz on disk",
)


def _load_tf():
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    return load_3d_sampler("Universal_Lognormal_Sampler_final.npz")


def test_mean_growth_flat_across_substep_cap():
    from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
    from baqaro.core_functions.erdf_core_functions import Erdf

    try:
        tf = _load_tf()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")

    size = 100_000
    delta_t = 0.135  # Gyr
    log_rate = np.zeros(size, dtype=np.float32)  # fixed sSAR -> fixed mu,sigma

    growths = []
    # n_steps straddling the N=1000 table cap.
    for n_steps in (200, 1000, 2000, 13500, 42690):
        erdf = Erdf(
            "log_normal_evol_halo_mass",
            {"log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": 0.5},
            rng=np.random.default_rng(42),
        )
        M_out, _ = evolve_BHs_fast(
            tf, np.full(size, 1e6, dtype=np.float32), delta_t, erdf,
            rad_efficiency=0.1, rad_efficiency_model="constant",
            n_steps=n_steps, log_halo_rates_array=log_rate,
            parallelize=False, num_workers=1, use_parallel_kernel=False,
            fw_p_threshold=0.0,
        )
        growths.append(float(np.mean(np.log(M_out.astype(np.float64) / 1e6))))

    growths = np.array(growths)
    ref = growths[0]
    # Mean growth must be n_steps-independent: all within 1% of the n=200 value.
    assert np.all(np.abs(growths / ref - 1.0) < 0.01), (
        f"mean growth not flat across n_steps (sub-step cap regression): {growths}"
    )


if __name__ == "__main__":
    test_mean_growth_flat_across_substep_cap()
    print("OK  test_mean_growth_flat_across_substep_cap")
