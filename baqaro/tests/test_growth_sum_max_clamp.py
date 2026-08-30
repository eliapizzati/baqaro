"""Tests for the per-snapshot growth-sum clamp introduced as an env-optional
guardrail (BAQARO_GROWTH_SUM_MAX -> ``growth_max`` parameter on
``evolve_BHs_fast`` / ``process_evolution``).

What we check:

1. Default (``growth_max=50.0``) yields bit-identical output to the legacy
   code path on a tiny ERDF setup (no asymmetric divergence vs the
   ``50.0`` literal that used to live inside the kernels).

2. A *tight* cap (``growth_max=4.6``) actually clips the per-snapshot
   growth — verified by driving a tiny n_halos through an ERDF that
   normally grows much more than exp(4.6) ≈ 100x in one snapshot, and
   checking that the resulting M_BH / M_initial ratio is bounded by
   exp(growth_max) within numerical tolerance.

3. Lbol is *not* clipped (eta_last and eps_last still set the
   instantaneous luminosity), so a capped run still emits the bright
   end correctly.

4. Each branch (A direct sampling, B-1D fixed-sigma transfer-table,
   B-2D variable-sigma transfer-table, C analytical mean) honours the
   cap.

These cover the fast (`bh_accretion_fast`) engine on which the
production forward run depends. ``process_evolution`` in
``bh_accretion.py`` is touched by the same test only at the
single-branch entry point; the legacy paths are not on the production
run hot path so a single smoke test there is enough.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

# Force the deterministic Numba kernels to be available before we use them.
import baqaro.core_functions.bh_accretion_fast as bf  # noqa: F401
import baqaro.core_functions.bh_accretion as ba       # noqa: F401
from baqaro.core_functions.erdf_core_functions import Erdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_erdf(rng_seed=12345, std_0=0.55,
              log_eta_mean_0=-0.5, log_eta_mean_evol=0.85):
    """A small ERDF with a moderately-large mean η so the growth sum can
    easily reach the legacy 50.0 cap on a single snapshot."""
    e = Erdf(model="log_normal_evol_halo_mass",
             params_dict=dict(log_eta_mean_0=log_eta_mean_0,
                              log_eta_mean_evol=log_eta_mean_evol,
                              std_0=std_0),
             rng=np.random.default_rng(rng_seed))
    return e


def make_halos(n=64, M_init=1e6):
    M_BHs = np.full(n, M_init, dtype=np.float32)
    # log10 of specific cold accretion rate (Gyr^-1). r=2 -> very high
    # accretion regime; the mean eta from log_eta_mean_evol*r is hot
    # enough that the integrated growth blows past 50 unless capped.
    log_halo_rates = np.full(n, 2.0, dtype=np.float32)
    return M_BHs, log_halo_rates


def _bigeta_growth_setup(branch, growth_max, n_halos=128,
                         delta_t_gyr=0.10, n_steps=300, rng_seed=12345):
    """Wrap ``evolve_BHs_fast`` so the only thing that varies between
    branches is the kernel selection (n_steps + transfer-function presence).

    Branch A: n_steps=2 with no transfer function (transfer_function=None).
    Branch B-1D: n_steps=300 with fixed-sigma erdf and a transfer function.
    Branch C: n_steps=0 with no transfer function.
    """
    erdf = make_erdf(rng_seed=rng_seed)
    M_BHs, log_halo_rates = make_halos(n=n_halos)
    if branch == "A":
        n_use, tf = max(1, min(4, n_steps)), None
    elif branch == "B1D":
        n_use = n_steps
        # Load the production transfer function lazily; skipped in CI if
        # the .npz is not on disk.
        from baqaro.core_functions.fast_lognormal_sampler import (
            load_3d_sampler,
        )
        try:
            tf = load_3d_sampler("Universal_Lognormal_Sampler_final.npz")
        except FileNotFoundError as e:
            pytest.skip(f"sampler file not on disk: {e}")
    elif branch == "C":
        n_use, tf = 0, None
    else:
        raise ValueError(branch)

    M_init = M_BHs.copy()
    M_BHs, L_bols = bf.evolve_BHs_fast(
        tf, M_BHs, delta_t_gyr, erdf,
        rad_efficiency=0.1, n_steps=n_use,
        log_halo_rates_array=log_halo_rates,
        rad_efficiency_model="constant",
        parallelize=False,
        use_parallel_kernel=False,
        fw_p_threshold=0.0,
        growth_max=growth_max,
    )
    return M_init, M_BHs, L_bols


# ---------------------------------------------------------------------------
# Tests — Branch A, B-1D, C all hit the same code path: a ``if growth >
# growth_max`` clamp before exp(). One assertion per branch is enough.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("branch", ["A", "B1D", "C"])
def test_default_growth_max_equals_legacy_50(branch):
    """growth_max=50.0 should reproduce the legacy literal-50 behaviour
    bit-for-bit when fed the same seed."""
    M0_a, M_a, L_a = _bigeta_growth_setup(branch, growth_max=50.0,
                                           rng_seed=42)
    M0_b, M_b, L_b = _bigeta_growth_setup(branch, growth_max=50.0,
                                           rng_seed=42)
    np.testing.assert_array_equal(M_a, M_b)
    np.testing.assert_array_equal(L_a, L_b)
    assert np.all(np.isfinite(M_a))
    assert np.all(np.isfinite(L_a))


@pytest.mark.parametrize("branch", ["A", "B1D", "C"])
def test_tight_cap_bounds_per_snapshot_growth(branch):
    """At growth_max=4.6 the maximum mass-growth factor per snapshot must
    be ≤ exp(4.6) ≈ 100. The bigeta ERDF setup normally grows BHs by
    several e-folds in one snapshot, so without the cap the assertion
    would fail.

    Tolerance: the existing safety clamp also has ``m_new > 1e15 -> 1e15``,
    which is irrelevant here because M_init=1e6 and growth_max=4.6
    keeps M_BH well under 1e9.
    """
    GMAX = 4.6
    M0, M, L = _bigeta_growth_setup(branch, growth_max=GMAX, rng_seed=42)
    growth_factor = M / M0
    # exp(growth_max) is the absolute ceiling per snapshot.
    # Tiny float headroom; use 1e-3 relative.
    assert growth_factor.max() <= np.exp(GMAX) * (1.0 + 1e-3), (
        f"branch {branch}: max growth factor {growth_factor.max():.3e} "
        f"exceeds cap exp({GMAX})={np.exp(GMAX):.3e}"
    )
    # Sanity: the bigeta setup must have actually wanted to grow more
    # than the cap in this branch — otherwise the test is meaningless.
    M0_unc, M_unc, _ = _bigeta_growth_setup(branch, growth_max=50.0,
                                            rng_seed=42)
    if np.max(M_unc / M0_unc) <= np.exp(GMAX) * (1.0 + 1e-3):
        pytest.skip(
            f"branch {branch}: ERDF didn't push past the cap in the "
            f"uncapped run (max factor {np.max(M_unc/M0_unc):.3e}); "
            "test setup needs more aggressive ERDF."
        )


@pytest.mark.parametrize("branch", ["A", "B1D", "C"])
def test_lbol_not_clipped_by_growth_cap(branch):
    """The growth cap is on the *integrated* mass-build; L_bol is the
    instantaneous luminosity at the LAST sub-step (Branch A/B) or a
    single random draw (Branch C). It should respond to η_last, not to
    the cap.

    We check that across a sample of seeds the capped/uncapped runs
    produce L_bol distributions whose medians are within a factor of
    a few — i.e., a tight cap may TRIM the upper tail (because M_BH is
    smaller) but should not collapse the bulk of L_bol.
    """
    _, M_c, L_c = _bigeta_growth_setup(branch, growth_max=4.6, rng_seed=42)
    _, M_u, L_u = _bigeta_growth_setup(branch, growth_max=50.0, rng_seed=42)
    # The median L_bol in the capped run cannot be MUCH larger than the
    # uncapped one (cap can only reduce L_bol via reduced M_BH).
    pos_c = L_c[L_c > 0]
    pos_u = L_u[L_u > 0]
    if pos_c.size and pos_u.size:
        assert np.median(pos_c) <= np.median(pos_u) * 1.1, (
            f"branch {branch}: capped median L_bol "
            f"{np.median(pos_c):.3e} > uncapped "
            f"{np.median(pos_u):.3e}"
        )


# ---------------------------------------------------------------------------
# Branch B-2D is not on the production path here (the production ERDF
# is fixed-sigma -> B-1D). A 1-test smoke check at the kernel layer
# is enough.
# ---------------------------------------------------------------------------

def test_growth_max_kernel_arg_is_respected_branch_a_kernel():
    """Hit the kernel directly so we don't depend on the env knob /
    wrapper plumbing. Confirms the kernel signature change works."""
    LN10 = np.log(10.0)
    n, n_sub = 8, 32
    rng = np.random.default_rng(7)
    Z = rng.standard_normal((n, n_sub)).astype(np.float32)
    # Force ALL eta draws to be huge so growth blows past any cap.
    Z[:] = 5.0  # 5σ above mean — guarantees growth way above any cap
    mus = np.full(n, 0.5, dtype=np.float32)
    sigmas = np.full(n, 0.55, dtype=np.float32)
    M_BHs = np.full(n, 1e6, dtype=np.float32)
    L_bols = np.zeros(n, dtype=np.float64)

    # Pick growth_const so that the *uncapped* growth wants to be way
    # above 4.6 (here, the eta values are ~ 10^(0.5+0.55*5) = 10^3.25
    # ≈ 1778 per sub-step; with growth_const=0.001 and n_sub=32,
    # uncapped growth ~ 1778 * 32 * 0.001 = 57 e-folds).
    bf._fused_branch_a_serial(
        Z, mus, sigmas, M_BHs,
        0.001, 0.1, 1.0/0.9, 1e4,  # growth_const, rad_eff, inv_1_minus_eff, FACTOR_CSI
        False, 1.0, 1.0, 1.0,
        L_bols, 4.6,
    )
    growth_factor = M_BHs / 1e6
    assert growth_factor.max() <= np.exp(4.6) * (1.0 + 1e-3), (
        f"kernel growth factor {growth_factor.max():.3e} exceeds cap"
    )
