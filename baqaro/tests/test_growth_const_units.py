"""
Regression tests for the growth_const units (ls/ms) + reference sub-step cap.
==============================================================================

Invariant: an Eddington accretor e-folds on the Salpeter time. `growth_const`
in BOTH bh_accretion.py and bh_accretion_fast.py combines `10^log_csi / c^2`
(10^log_csi in Lsun/Msun, the same units as L_bol) with the Lsun->erg/s
(nc.ls) and g->Msun (nc.ms) conversions; dropping either factor would make an
Eddington accretor e-fold on the wrong timescale for its (correct) luminosity.

The first test is the compensation-proof invariant: a BH at exactly Eddington
(eta=1) MUST e-fold on t_Sal = eps/(1-eps) * t_Edd, computed from the model's
own constants. No downstream calibration can change this. It needs no sampler
(Branch C), so it always runs.

The second test guards the sub-step cap newly ported into
bh_accretion.py (was only in the fast path): reference mean growth must be
n_steps-independent across the N=1000 table cap.
"""
import os
import numpy as np
import pytest

import qhtools.utils.natconst as nc
from baqaro.core_functions.erdf_core_functions import Erdf
from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
from baqaro.core_functions.bh_accretion import evolve_BHs


def _salpeter_growth(eps, dt_gyr):
    """Correct ln(M_new/M0) for an eta=1 accretor over dt, from natconst."""
    t_edd_yr = nc.ms * nc.cc**2 / (10**nc.log_csi * nc.ls) / nc.year   # ~451 Myr
    return (1.0 - eps) / eps / t_edd_yr * (dt_gyr * 1e9)


@pytest.mark.parametrize("engine,is_fast", [("fast", True), ("reference", False)])
def test_eddington_efold_equals_salpeter(engine, is_fast):
    """eta=1, eps=0.1 BH must e-fold on the Salpeter time (ls/ms unit fix)."""
    eps, M0, dt = 0.1, 1.0e8, 0.05
    G_correct = _salpeter_growth(eps, dt)   # ~0.998 over 50 Myr

    # std_0 == 0.0 is LOAD-BEARING: it makes the ERDF deterministic so eta is
    # exactly 10^log_eta_mean_0 = 1 (exactly Eddington). The Branch C arithmetic
    # mean is E[eta] = 10^mu * exp(0.5*ln10^2*sigma^2), so ANY sigma > 0 inflates
    # the growth (0.1 -> +2.7%) and breaks this invariant. Do NOT bump std_0
    # here, and do NOT add a lower-edge sigma clip to the fixed-sigma engine path
    # (that was proposed once and correctly declined — see the
    # "deliberately NOT clipped" note in bh_accretion_fast.py).
    erdf = Erdf("log_normal_evol_halo_mass",
                {"log_eta_mean_0": 0.0, "log_eta_mean_evol": 0.0, "std_0": 0.0},
                rng=np.random.default_rng(1))
    M = np.array([M0], dtype=np.float32)
    rates = np.zeros(1, dtype=np.float32)
    # n_steps=0 -> Branch C (analytical mean), no sampler, scalar-sigma safe.
    kw = dict(rad_efficiency=eps, n_steps=0, log_halo_rates_array=rates,
              rad_efficiency_model="constant", parallelize=False)
    if is_fast:
        M_out, _ = evolve_BHs_fast(None, M, dt, erdf, use_parallel_kernel=False, **kw)
    else:
        M_out, _ = evolve_BHs(None, M, dt, erdf, **kw)

    G = float(np.log(np.asarray(M_out)[0] / M0))
    # Assert within 2% of the Salpeter value.
    assert abs(G / G_correct - 1.0) < 0.02, (
        f"{engine}: Eddington e-fold inconsistent with Salpeter "
        f"(G={G:.4f}, correct={G_correct:.4f}, ratio={G/G_correct:.4f}). "
        f"growth_const likely missing the nc.ls/nc.ms factor."
    )


@pytest.mark.skipif(os.environ.get("BAQARO_SOURCE_DIR") is None,
                    reason="needs BAQARO_SOURCE_DIR + the lognormal sampler npz")
def test_reference_substep_cap_flat():
    """Reference (bh_accretion.py) mean growth flat across the N=1000 cap."""
    from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
    try:
        tf = load_3d_sampler("Universal_Lognormal_Sampler_final.npz")
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")

    size, dt = 50_000, 0.135
    log_rate = np.zeros(size, dtype=np.float32)
    growths = []
    for n_steps in (200, 2000, 13500):   # straddle the 1000 cap
        erdf = Erdf("log_normal_evol_halo_mass",
                    {"log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": 0.5},
                    rng=np.random.default_rng(42))
        M_out, _ = evolve_BHs(
            tf, np.full(size, 1e6, dtype=np.float32), dt, erdf,
            rad_efficiency=0.1, rad_efficiency_model="constant",
            n_steps=n_steps, log_halo_rates_array=log_rate, parallelize=False)
        growths.append(float(np.mean(np.log(np.asarray(M_out, dtype=np.float64) / 1e6))))
    growths = np.array(growths)
    assert np.all(np.abs(growths / growths[0] - 1.0) < 0.02), (
        f"reference mean growth not flat across sub-step cap: {growths}")


if __name__ == "__main__":
    for eng, fast in (("fast", True), ("reference", False)):
        test_eddington_efold_equals_salpeter(eng, fast)
        print(f"OK  eddington e-fold == salpeter [{eng}]")
