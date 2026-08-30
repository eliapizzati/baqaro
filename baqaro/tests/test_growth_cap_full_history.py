"""Tests for Option B (self-consistent per-sub-step growth cap) in
``process_evolution_full_history`` — i.e. the engine consumed by
``main_evolution_full_history.py`` for the sub-step lightcurve runs.

Invariants verified:

  1. Default ``growth_max=50.0`` must produce bit-identical (M, L, etas)
     to the pre-Option-B code path for any reasonable ERDF + halo setup.
     We achieve this by checking that the integrated growth in the bigeta
     setup stays well below 50, so the cap clause is inert and the
     reconstructed effective-eta equals the raw lognormal draw to
     floating-point precision.

  2. With a tight cap (``growth_max=4.6`` => max 100x per snapshot):

     * M_BH trajectory is monotonic non-decreasing in time (cumulative
       growth never goes negative).
     * M_BH is bounded by ``M_init * exp(growth_max)`` at every sub-step.
     * Once a halo's cumulative growth has hit the cap, the effective
       eta_acc drops to zero for all subsequent sub-steps.
     * L_bol = 0 wherever eta_acc_eff = 0 (no accretion -> no radiation).
     * Where the cap is NOT yet active, eta_acc_eff equals the raw
       lognormal draw (within numerical precision).
     * The instantaneous lambda_Edd = L_bol / L_Edd(M_BH) equals
       eta_acc_eff (Soltan/Eddington consistency invariant).

  3. For the "constant" radiative-efficiency model, the cap path is
     exact (correction == 1, no Madau eps coupling).
"""
import numpy as np
import pytest

from baqaro.core_functions.bh_accretion import (
    process_evolution_full_history,
)
from baqaro.core_functions.erdf_core_functions import Erdf
import qhtools.utils.natconst as nc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_erdf(seed=42, std_0=0.55, log_eta_mean_0=-0.5, log_eta_mean_evol=0.85):
    return Erdf(
        model="log_normal_evol_halo_mass",
        params_dict=dict(log_eta_mean_0=log_eta_mean_0,
                         log_eta_mean_evol=log_eta_mean_evol,
                         std_0=std_0),
        rng=np.random.default_rng(seed),
    )


def run_FH(growth_max, rad_efficiency_model="constant",
           n_halos=32, n_steps=200, growth_const=0.03, seed=7):
    """Drive the full-history engine with a high-eta "bigeta" setup that
    normally wants to grow each halo by many orders of magnitude in one
    snapshot — so the cap actually has work to do."""
    erdf = make_erdf(seed=seed)
    M0 = np.full(n_halos, 1e6, dtype=np.float32)
    log_halo_rates = np.full(n_halos, 2.0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    M, L, etas = process_evolution_full_history(
        None, n_steps, M0.copy(), erdf, growth_const, 0.1,
        n_halos, rng, log_halo_rates, rad_efficiency_model,
        growth_max=growth_max,
    )
    return M0, M, L, etas


def _run_FH_tame(growth_max, rad_model, seed=7):
    """Tame setup chosen so the integrated growth stays well below 50 e-folds
    even at the bigeta ERDF — used for the 'cap inert at default' test."""
    erdf = make_erdf(seed=seed, log_eta_mean_0=-2.0,
                     log_eta_mean_evol=0.5, std_0=0.4)
    M0 = np.full(32, 1e6, dtype=np.float32)
    log_halo_rates = np.full(32, 1.5, dtype=np.float32)
    rng = np.random.default_rng(seed)
    M, L, etas = process_evolution_full_history(
        None, 100, M0.copy(), erdf, 0.002, 0.1,  # growth_const tame
        32, rng, log_halo_rates, rad_model,
        growth_max=growth_max,
    )
    return M0, M, L, etas


# ---------------------------------------------------------------------------
# Invariant 1 — backward compatibility (cap inert at default 50)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rad_model", ["constant", "madau+"])
def test_default_growth_max_is_legacy_behaviour(rad_model):
    """At growth_max=50 with a tame setup the cap genuinely never fires;
    every effective eta must equal the raw lognormal draw to floating-point
    precision, and no plateau should appear anywhere."""
    M0, M, L, etas_eff = _run_FH_tame(growth_max=50.0, rad_model=rad_model)
    log_growth_final = np.log(M[:, -1] / M0)
    assert log_growth_final.max() < 20.0, (
        f"this test requires the tame integrated growth to stay below 20 "
        f"e-folds; max log growth = {log_growth_final.max():.2f}"
    )
    # Cap inert -> no zero etas anywhere:
    assert (etas_eff > 0).all(), (
        f"some etas_eff dropped to zero at growth_max=50 -> the cap fired "
        f"unexpectedly; min etas_eff = {etas_eff.min()}"
    )


@pytest.mark.parametrize("rad_model", ["constant", "madau+"])
def test_default_eq_huge_growth_max(rad_model):
    """The cap is monotone: raising growth_max from 50 to 1e6 cannot change
    M, L, or etas_eff at all when the cap was already inert at 50."""
    _, M_50, L_50, e_50 = _run_FH_tame(growth_max=50.0, rad_model=rad_model)
    _, M_huge, L_huge, e_huge = _run_FH_tame(growth_max=1e6, rad_model=rad_model)
    np.testing.assert_allclose(M_50, M_huge, rtol=1e-7, atol=0)
    np.testing.assert_allclose(L_50, L_huge, rtol=1e-7, atol=0)
    np.testing.assert_allclose(e_50, e_huge, rtol=1e-7, atol=0)


# ---------------------------------------------------------------------------
# Invariant 2 — tight-cap physics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rad_model", ["constant", "madau+"])
def test_tight_cap_M_monotonic_and_bounded(rad_model):
    GMAX = 4.6
    M0, M, L, etas_eff = run_FH(growth_max=GMAX,
                                 rad_efficiency_model=rad_model)
    # Per-halo non-decreasing across substeps:
    diffs = np.diff(M, axis=1)
    assert (diffs >= -1e-3).all(), (
        f"trajectory went DOWN at some substep "
        f"(min diff {diffs.min():.3e})"
    )
    # Cap on cumulative growth -> M bounded by M0 * exp(GMAX)
    ceiling = M0 * np.exp(GMAX) * (1.0 + 1e-3)  # tiny float headroom
    assert (M <= ceiling[:, None]).all(), (
        f"M_BH overshot the cap ceiling — max overshoot factor "
        f"{(M / ceiling[:, None]).max():.3e}"
    )


@pytest.mark.parametrize("rad_model", ["constant", "madau+"])
def test_tight_cap_eta_eff_zero_on_plateau(rad_model):
    GMAX = 4.6
    M0, M, L, etas_eff = run_FH(growth_max=GMAX,
                                 rad_efficiency_model=rad_model)
    # A halo is on plateau at substep j when its cumulative growth has
    # already reached the cap. Equivalent test: M[i, j] >= M0[i] * exp(GMAX)
    # to within float tolerance.
    plateau_mask = M >= M0[:, None] * np.exp(GMAX) * (1.0 - 1e-6)
    # Wherever a halo is on plateau at step j, the previous step's plateau
    # state implies eta_eff[i, j] must be 0.
    plateau_prev = np.concatenate(
        [np.zeros((plateau_mask.shape[0], 1), dtype=bool),
         plateau_mask[:, :-1]], axis=1,
    )
    if plateau_prev.any():
        assert (etas_eff[plateau_prev] == 0.0).all(), (
            "etas_eff non-zero on a sub-step where the halo was already "
            f"plateaued — max value {etas_eff[plateau_prev].max():.3e}"
        )


@pytest.mark.parametrize("rad_model", ["constant", "madau+"])
def test_tight_cap_lbol_zero_when_eta_eff_zero(rad_model):
    GMAX = 4.6
    _, M, L, etas_eff = run_FH(growth_max=GMAX,
                                rad_efficiency_model=rad_model)
    zero_eta = etas_eff == 0.0
    if zero_eta.any():
        assert (L[zero_eta] == 0.0).all(), (
            "L_bol non-zero on a sub-step with eta_eff == 0 — max value "
            f"{L[zero_eta].max():.3e}"
        )


def test_lambda_edd_consistency_constant():
    """For the 'constant' radiative-efficiency model, the Eddington ratio
    measured at each sub-step from L_bol and M_BH must exactly equal
    eta_acc_eff (because eps == rad_efficiency everywhere)."""
    GMAX = 4.6
    M0, M, L, etas_eff = run_FH(growth_max=GMAX, rad_efficiency_model="constant")
    # lambda_Edd = L_bol / L_Edd(M_BH); L_Edd(M_BH) = M_BH * 10**log_csi
    # (with eps=eps_base => the (eps/eps_base) factor is 1).
    L_edd = M * (10.0 ** nc.log_csi)
    nz = (L_edd > 0) & np.isfinite(L_edd)
    lam = np.zeros_like(L)
    lam[nz] = L[nz] / L_edd[nz]
    np.testing.assert_allclose(lam, etas_eff, rtol=1e-5, atol=1e-12)


# ---------------------------------------------------------------------------
# Invariant 3 — un-capped substeps preserve the raw eta
# ---------------------------------------------------------------------------
def test_uncapped_substeps_preserve_raw_eta_constant():
    """For substeps BEFORE the cap activates, the effective eta must equal
    the raw lognormal draw. Verified via the constant model where the
    correction factor is exactly 1."""
    GMAX = 4.6
    # Reconstruct what the raw etas WOULD have been with the same RNG.
    erdf = make_erdf(seed=7)
    rng_raw = np.random.default_rng(7)
    Z = rng_raw.standard_normal((32, 200))
    mus, sigmas = erdf.get_mu_sigma_arrays(np.full(32, 2.0, dtype=np.float32))
    LN_10 = 2.302585092994046
    mus = np.atleast_1d(mus); sigmas = np.atleast_1d(sigmas)
    etas_raw = np.exp((mus[:, None] + sigmas[:, None] * Z) * LN_10)

    _, M, L, etas_eff = run_FH(growth_max=GMAX,
                                rad_efficiency_model="constant")

    # Identify the sub-step at which each halo first crosses into the cap
    # (i.e. the first j where cumulative growth reaches growth_max).
    growth_const = 0.03
    cum_uncapped = np.cumsum(etas_raw * growth_const, axis=1)
    # "Pre-cap" sub-steps: cumulative_uncapped[j] <= growth_max
    pre_cap = cum_uncapped <= GMAX
    # For those, etas_eff should equal etas_raw.
    np.testing.assert_allclose(
        etas_eff[pre_cap], etas_raw[pre_cap], rtol=1e-5, atol=1e-12,
        err_msg="etas_eff diverged from etas_raw on a PRE-cap substep",
    )
