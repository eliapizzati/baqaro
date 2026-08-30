"""
Tests for the three opt-in likelihood-definition options. Every one is an
ENV-GATED OPTION; the default path must stay byte-identical to the production
pipeline.

  BAQARO_CERDF_BIN_INTEGRATED : model cERDF bin-averaged with the data's
                                sum(p*dEta)=1 measure (vs the point-sampled
                                + centers-span trapezoid default).
  BAQARO_QLF_PRECISE_FAINT_CUT: faint QLF cut honored at 0.5-dex granularity
                                (the default rounds a x.5 floor down to the
                                integer-dex bin edge).
  BAQARO_QLF_PHI_MIN          : configurable log Phi censoring floor (the
                                default -9 censors the brightest survey
                                points at every z).

The QLF-loader tests spawn subprocesses (the module reads CSVs at import
and env flags are read at import time).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np


# ----------------------------------------------------------------------
# BAQARO_CERDF_BIN_INTEGRATED — _model_pdf_on_obs_bins
# ----------------------------------------------------------------------
def _legacy_point_sampled(pdf_model, grid, centers):
    m = np.interp(centers, grid, pdf_model)
    integral = np.trapezoid(m, centers)
    if integral > 0:
        m = m / integral
    return m


def test_m8_point_mode_is_bit_identical_to_legacy():
    from baqaro.inference.likelihoods_and_priors import _model_pdf_on_obs_bins
    rng = np.random.default_rng(3)
    grid = np.linspace(-4.0, 2.0, 61)
    pdf = np.exp(-0.5 * ((grid + 0.8) / 0.6) ** 2) * (1 + 0.1 * rng.random(61))
    edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])   # production bigeta bins
    centers = 0.5 * (edges[1:] + edges[:-1])

    out = _model_pdf_on_obs_bins(pdf, grid, centers, edges, bin_integrated=False)
    np.testing.assert_array_equal(out, _legacy_point_sampled(pdf, grid, centers))


def test_m8_integrated_mode_properties():
    from baqaro.inference.likelihoods_and_priors import _model_pdf_on_obs_bins
    grid = np.linspace(-4.0, 2.0, 601)
    edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths = np.diff(edges)

    # (a) For a LINEAR pdf the bin average equals the value at the bin center,
    # so integrated == point values up to the different normalization measure.
    pdf_lin = 2.0 + 0.5 * grid
    out = _model_pdf_on_obs_bins(pdf_lin, grid, centers, edges, bin_integrated=True)
    expected = 2.0 + 0.5 * centers
    expected = expected / np.sum(expected * widths)
    np.testing.assert_allclose(out, expected, rtol=1e-10)

    # (b) Data-measure normalization: sum(p_i * dEta_i) == 1.
    pdf_g = np.exp(-0.5 * ((grid + 0.8) / 0.6) ** 2)
    out_g = _model_pdf_on_obs_bins(pdf_g, grid, centers, edges, bin_integrated=True)
    assert abs(np.sum(out_g * widths) - 1.0) < 1e-12

    # (c) Curved pdf: integrated and point-sampled measures genuinely differ
    # (this is the distortion the option removes).
    out_p = _model_pdf_on_obs_bins(pdf_g, grid, centers, edges, bin_integrated=False)
    assert np.max(np.abs(out_g / out_p - 1.0)) > 0.02

    # (d) Bins fully outside the emulator grid get zero mass.
    edges_out = np.array([5.0, 6.0, 7.0])
    centers_out = 0.5 * (edges_out[1:] + edges_out[:-1])
    out_o = _model_pdf_on_obs_bins(pdf_g, grid, centers_out, edges_out, bin_integrated=True)
    np.testing.assert_array_equal(out_o, np.zeros(2))


def test_m8_projection_matrix_matches_direct_path():
    """The precomputed bin-mass mat-vec (MCMC hot-loop fast path) must equal
    the direct cumulative-trapezoid computation to float roundoff."""
    from baqaro.inference.likelihoods_and_priors import (
        _model_pdf_on_obs_bins, _bin_mass_projection)
    rng = np.random.default_rng(11)
    grid = np.linspace(-3.0, 2.0, 40)   # production emulator grid
    edges = np.array([-2.5, -1.5, -0.5, 0.5, 1.5])
    centers = 0.5 * (edges[1:] + edges[:-1])
    P = _bin_mass_projection(grid, edges)
    for _ in range(20):
        pdf = rng.random(40) * np.exp(-0.5 * ((grid + rng.uniform(-1.5, 0)) / 0.5) ** 2)
        direct = _model_pdf_on_obs_bins(pdf, grid, centers, edges, True)
        fast = _model_pdf_on_obs_bins(pdf, grid, centers, edges, True, bin_proj=P)
        np.testing.assert_allclose(fast, direct, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------
# QLF loader options (subprocess: env read at import time)
# ----------------------------------------------------------------------
_SNIPPET = textwrap.dedent("""
    import numpy as np
    from baqaro.obs_data import qlf_obs_data as q
    z2 = q.data_qlf_global_binned['2.0']
    raw02 = q.data_qlf_global_raw['0.2']
    print("FIRSTCENTER", z2.x[0])
    print("PHIMIN", q.QLF_PHI_MIN)
    print("RAWMAXL02", np.max(raw02.x))
    print("NRAW02", len(raw02.x))
""")


def _run_loader(env_extra):
    env = dict(os.environ)
    # Make sure inherited likelihood-definition vars don't leak into the test.
    for k in ("BAQARO_QLF_PRECISE_FAINT_CUT", "BAQARO_QLF_PHI_MIN",
              "BAQARO_QLF_MIN_LBOL_FLOOR", "BAQARO_QLF_MIN_LBOL_SHIFT",
              "BAQARO_QLF_MIN_LBOL_PER_Z"):
        env.pop(k, None)
    env.update(env_extra)
    out = subprocess.run([sys.executable, "-c", _SNIPPET], env=env,
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] in ("FIRSTCENTER", "PHIMIN", "RAWMAXL02", "NRAW02"):
            vals[parts[0]] = float(parts[1])
    return vals


def test_m9_m10_default_path_unchanged():
    v = _run_loader({})
    # The BAQARO_QLF_MIN_LBOL_PER_Z *default* is
    # "2.0:45.0,3.0:45.0,4.0:45.5,5.0:45.5", so z=2 is cut at 45.0 exactly (the
    # non-empty per-z string also implies the precise 0.5-dex grid) -> first
    # z=2 bin is (45.0, 45.5), center 45.25.
    assert abs(v["FIRSTCENTER"] - 45.25) < 1e-9
    assert v["PHIMIN"] == -9.0


def test_m9_precise_faint_cut_starts_at_documented_floor():
    # Clearing the per-z map restores the dict floors (z=2 -> 45.5), the
    # configuration the precise-cut flag exists for: without it
    # `start = np.floor(45.5)` rounds the cut down to 45.0.
    v = _run_loader({"BAQARO_QLF_MIN_LBOL_PER_Z": "",
                     "BAQARO_QLF_PRECISE_FAINT_CUT": "1"})
    # 45.5 honored -> first z=2 bin is (45.5, 46.0), center 45.75.
    assert abs(v["FIRSTCENTER"] - 45.75) < 1e-9
    # ...and without the flag the same floors get floored back to 45.0 (the legacy
    # bug this flag guards against), so the flag is what makes the difference.
    v_legacy = _run_loader({"BAQARO_QLF_MIN_LBOL_PER_Z": ""})
    assert abs(v_legacy["FIRSTCENTER"] - 45.25) < 1e-9


def test_m10_disabling_phi_floor_recovers_bright_points():
    v0 = _run_loader({})
    v1 = _run_loader({"BAQARO_QLF_PHI_MIN": "none"})
    assert np.isinf(v1["PHIMIN"]) and v1["PHIMIN"] < 0
    # The censored points are the BRIGHTEST: disabling the floor must extend
    # the raw z=0.2 range brighter and add points.
    assert v1["RAWMAXL02"] > v0["RAWMAXL02"] + 0.05
    assert v1["NRAW02"] > v0["NRAW02"]


if __name__ == "__main__":
    test_m8_point_mode_is_bit_identical_to_legacy()
    print("OK  m8 point mode bit-identical")
    test_m8_integrated_mode_properties()
    print("OK  m8 integrated mode properties")
    test_m9_m10_default_path_unchanged()
    print("OK  m9/m10 default path unchanged")
    test_m9_precise_faint_cut_starts_at_documented_floor()
    print("OK  m9 precise faint cut")
    test_m10_disabling_phi_floor_recovers_bright_points()
    print("OK  m10 phi floor disable")
