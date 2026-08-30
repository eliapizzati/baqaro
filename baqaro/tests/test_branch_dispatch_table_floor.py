"""
Regression test for the Branch A/B dispatch at the transfer-table LOWER edge
(the mirror of the sub-step cap at the opposite table edge).
=============================================================================

The transfer table's floor is N=5 summands and Branch B looks up the
history sum for N = n_steps-1, so Branch B is exact only for n_steps >= 6
(an N=4 lookup would clamp to the N=5 plane while ``growth_const``
normalizes by delta_t/5). The dispatch must therefore send ``n_steps < 6``
to Branch A (direct sampling, exact at any small n).

This test asserts the physical invariant at the lower edge: with delta_t and
ERDF fixed, the mean log growth is flat across n_steps = 4, 5, 6, 7, 10.
Both engines checked.

Needs the real sampler npz (~764 MB, not in git). Skips if absent.
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BAQARO_SOURCE_DIR") is None,
    reason="needs BAQARO_SOURCE_DIR + the lognormal sampler npz on disk",
)

N_STEPS_GRID = (4, 5, 6, 7, 10)


def _load_tf():
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    return load_3d_sampler("Universal_Lognormal_Sampler_final.npz")


def _mean_log_growth_fast(tf, n_steps, size=200_000):
    from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
    from baqaro.core_functions.erdf_core_functions import Erdf

    erdf = Erdf(
        "log_normal_evol_halo_mass",
        {"log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": 0.5},
        rng=np.random.default_rng(42),
    )
    M_out, _ = evolve_BHs_fast(
        tf, np.full(size, 1e6, dtype=np.float32), 0.135, erdf,
        rad_efficiency=0.1, rad_efficiency_model="constant",
        n_steps=n_steps, log_halo_rates_array=np.zeros(size, dtype=np.float32),
        parallelize=False, num_workers=1, use_parallel_kernel=False,
        fw_p_threshold=0.0,
    )
    return float(np.mean(np.log(M_out.astype(np.float64) / 1e6)))


def _mean_log_growth_reference(tf, n_steps, size=200_000):
    from baqaro.core_functions.bh_accretion import evolve_BHs
    from baqaro.core_functions.erdf_core_functions import Erdf

    erdf = Erdf(
        "log_normal_evol_halo_mass",
        {"log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": 0.5},
        rng=np.random.default_rng(42),
    )
    M_out, _ = evolve_BHs(
        tf, np.full(size, 1e6, dtype=np.float32), 0.135, erdf,
        rad_efficiency=0.1, rad_efficiency_model="constant",
        n_steps=n_steps, log_halo_rates_array=np.zeros(size, dtype=np.float32),
        parallelize=False, num_workers=1,
    )
    return float(np.mean(np.log(M_out.astype(np.float64) / 1e6)))


def test_mean_growth_flat_across_table_floor_fast_engine():
    try:
        tf = _load_tf()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")

    growths = np.array([_mean_log_growth_fast(tf, n) for n in N_STEPS_GRID])
    ref = growths[0]
    # Mean growth must be n_steps-independent (2% tolerance).
    assert np.all(np.abs(growths / ref - 1.0) < 0.02), (
        f"mean growth not flat across the table floor: "
        f"n_steps={N_STEPS_GRID}, growths={growths}"
    )


def test_mean_growth_flat_across_table_floor_reference_engine():
    try:
        tf = _load_tf()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")

    growths = np.array([_mean_log_growth_reference(tf, n) for n in N_STEPS_GRID])
    ref = growths[0]
    assert np.all(np.abs(growths / ref - 1.0) < 0.02), (
        f"mean growth not flat across the table floor ("
        f"reference engine): n_steps={N_STEPS_GRID}, growths={growths}"
    )


if __name__ == "__main__":
    os.environ.setdefault("BAQARO_SOURCE_DIR", "machine_igm")
    test_mean_growth_flat_across_table_floor_fast_engine()
    print("OK  test_mean_growth_flat_across_table_floor_fast_engine")
    test_mean_growth_flat_across_table_floor_reference_engine()
    print("OK  test_mean_growth_flat_across_table_floor_reference_engine")
