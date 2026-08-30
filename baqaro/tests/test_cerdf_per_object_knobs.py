"""Regression tests for the unbinned (per-object) cERDF estimator.

Covers:

  * `precompute_cerdf_inputs` must emit 'log_bins': it is what
    `log_likelihood_cerdf_from_prediction` -- the function the VECTORIZED MCMC
    path calls, and vectorize=True is the default -- reads, so without it
    CERDF_LIKELIHOOD_MODE="per_object" cannot run.  Guard it.

  * The flexibility knobs (BAQARO_CERDF_SCATTER_DEX, BAQARO_CERDF_OUTLIER_FRAC)
    must default to the historical behaviour bit-for-bit, and must do the
    right thing when switched on.

  * `log_likelihood_cerdf` (emulator call + core) and
    `log_likelihood_cerdf_from_prediction` (core alone) must agree, since the
    non-vectorized and vectorized MCMC paths use one each.

Sampler-free: builds a synthetic cERDF prediction cube, so these always run.
"""
import numpy as np
import pytest

from baqaro.inference import likelihoods_and_priors as L


N_Z, N_L, N_BINS = 3, 4, 60
LOG_BINS = np.linspace(-3.0, 1.5, N_BINS)
Z_GRID = np.array([1.0, 2.0, 3.0])
L_THR = np.array([46.0, 46.5, 47.0, 47.5])


class _FakeEmulator:
    """Minimal stand-in exposing what the cERDF likelihood touches."""

    def __init__(self, cube):
        self._cube = cube
        self.axis_data = {
            "redshift": Z_GRID,
            "log_L_threshold": L_THR,
            "log_bins": LOG_BINS,
        }

    def predict_mean_only(self, params):
        return self._cube[None, ...]


def _make_cube(support_max=None):
    """A cumulative CERDF cube: decreasing in L threshold, lognormal in eta.

    ``support_max`` truncates the model to zero density above that log_eta,
    which is how we manufacture an object that lands in a genuinely
    zero-probability region (see the floor-cliff test).
    """
    rng = np.random.default_rng(7)
    cube = np.zeros((N_Z, N_L, N_BINS))
    for iz in range(N_Z):
        for il in range(N_L):
            # Gaussian in log_eta, amplitude falling with L threshold so that
            # CERDF(>Lmin) - CERDF(>Lmax) > 0 -- the differential is positive.
            amp = 10.0 ** (-0.8 * il)
            mu = -0.6 + 0.15 * iz
            g = amp * np.exp(-0.5 * ((LOG_BINS - mu) / 0.55) ** 2)
            if support_max is not None:
                g = np.where(LOG_BINS > support_max, 0.0, g)
            cube[iz, il] = np.log10(g + 1e-300)
    return cube, rng


@pytest.fixture
def setup():
    cube, rng = _make_cube()
    emu = _FakeEmulator(cube)
    n = 400
    z = rng.uniform(0.8, 3.2, n)
    lbol = rng.uniform(46.1, 47.4, n)
    eta = rng.normal(-0.6, 0.6, n)
    return emu, cube, z, lbol, eta


def test_precompute_emits_log_bins(setup, monkeypatch):
    """The vectorized path reads cfg['log_bins']; it must be there."""
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)
    emu, _, z, lbol, eta = setup
    cfg = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    assert "log_bins" in cfg, "missing 'log_bins' -> per_object mode KeyErrors"
    np.testing.assert_allclose(cfg["log_bins"], LOG_BINS)
    assert cfg["scatter_dex"] == 0.3     # historical default
    assert cfg["outlier_frac"] == 0.0    # off by default


def test_vectorized_path_runs(setup, monkeypatch):
    """The exact call the vectorized MCMC makes. Regression for the KeyError."""
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)
    emu, cube, z, lbol, eta = setup
    cfg = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    ll = L.log_likelihood_cerdf_from_prediction(cube, cfg)
    assert np.isfinite(ll)


def test_emulator_and_prediction_paths_agree(setup, monkeypatch):
    """Non-vectorized and vectorized MCMC paths must give the same number."""
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)
    emu, cube, z, lbol, eta = setup
    cfg = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    ll_emu = L.log_likelihood_cerdf(np.zeros(6), emu, cfg)
    ll_pred = L.log_likelihood_cerdf_from_prediction(cube, cfg)
    assert ll_emu == pytest.approx(ll_pred, rel=1e-12)


def test_scatter_dex_env_is_honoured(setup, monkeypatch):
    """BAQARO_CERDF_SCATTER_DEX must reach the vectorized path via the cfg.

    It cannot be passed as a kwarg through the dispatcher, so if it is not
    carried in cerdf_cfg it is ignored (the 0.3 default would apply,
    unreachable from main_mcmc).
    """
    emu, cube, z, lbol, eta = setup
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)

    monkeypatch.setenv("BAQARO_CERDF_SCATTER_DEX", "0.3")
    cfg_03 = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    monkeypatch.setenv("BAQARO_CERDF_SCATTER_DEX", "0.6")
    cfg_06 = L.precompute_cerdf_inputs(z, lbol, eta, emu)

    assert cfg_06["scatter_dex"] == 0.6
    ll_03 = L.log_likelihood_cerdf_from_prediction(cube, cfg_03)
    ll_06 = L.log_likelihood_cerdf_from_prediction(cube, cfg_06)
    # A broader model PDF is flatter, so a data set concentrated near its peak
    # loses likelihood.  The point is only that the knob CHANGES the answer.
    assert not np.isclose(ll_03, ll_06), "BAQARO_CERDF_SCATTER_DEX was ignored"


def test_outlier_frac_off_is_bit_identical(setup, monkeypatch):
    """f=0 must reproduce the likelihood without the outlier mixture exactly."""
    emu, cube, z, lbol, eta = setup
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.setenv("BAQARO_CERDF_OUTLIER_FRAC", "0.0")
    cfg = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    ll_env = L.log_likelihood_cerdf_from_prediction(cube, cfg)

    cfg_bare = dict(cfg)
    cfg_bare.pop("outlier_frac")       # simulate a cfg built before the knob
    ll_bare = L.log_likelihood_cerdf_from_prediction(cube, cfg_bare)
    assert ll_env == pytest.approx(ll_bare, rel=1e-15)


def test_outlier_mixture_caps_the_floor_cliff(monkeypatch):
    """The mixture must bound the per-object penalty far above ln(pdf_floor).

    Without it, one QSO in a genuinely zero-probability region costs
    ln(1e-30) ~ -69 and can outvote the entire QLF.  With f>0 the worst case
    is ln(f / (eta_max - eta_min)), which is O(-6).

    NOTE on scope: this is the mixture's *worst case*, and manufacturing it
    requires a model with hard-zero support (here: truncated above log_eta=0.5)
    and no convolution.  On the REAL catalogue at the candidate optimum, zero
    of the 293,466 QSOs fall below p=1e-6 -- the 0.3 dex convolution smears the
    model PDF over the whole grid, so the cliff never actually fires.  The
    mixture is therefore a safeguard for DE excursions into bad parameter
    regions, NOT a fix for a live pathology in the production fit.  Do not
    claim otherwise.
    """
    cube, _ = _make_cube(support_max=0.5)
    emu = _FakeEmulator(cube)
    # No convolution, so the hard-zero region stays hard zero.
    monkeypatch.setenv("BAQARO_CERDF_SCATTER_DEX", "0.0")

    z = np.array([2.0, 2.0]); lbol = np.array([46.2, 46.2])
    eta = np.array([-0.6, 1.45])          # second object is outside support

    monkeypatch.setenv("BAQARO_CERDF_OUTLIER_FRAC", "0.0")
    cfg_off = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    monkeypatch.setenv("BAQARO_CERDF_OUTLIER_FRAC", "0.02")
    cfg_on = L.precompute_cerdf_inputs(z, lbol, eta, emu)

    ll_off = L.log_likelihood_cerdf_from_prediction(cube, cfg_off)
    ll_on = L.log_likelihood_cerdf_from_prediction(cube, cfg_on)

    # Off: the outlier is pinned at pdf_floor and costs ~ln(1e-30) = -69.
    assert ll_off < -60, f"expected the floor cliff to fire; got {ll_off}"
    # On: its penalty is bounded by the uniform background.
    p_bg = 0.02 / (LOG_BINS[-1] - LOG_BINS[0])
    assert ll_on > ll_off + 50, "mixture should relieve the outlier penalty"
    assert ll_on > len(eta) * np.log(p_bg), "penalty exceeded the mixture floor"


def test_outlier_frac_rejects_bad_values(setup, monkeypatch):
    emu, _, z, lbol, eta = setup
    monkeypatch.setenv("BAQARO_CERDF_OUTLIER_FRAC", "1.0")
    with pytest.raises(ValueError, match="OUTLIER_FRAC"):
        L.precompute_cerdf_inputs(z, lbol, eta, emu)


def _emu_with_low_z():
    cube, _ = _make_cube()
    emu = _FakeEmulator(cube)
    emu.axis_data["redshift"] = np.array([0.5, 1.0, 2.0])
    emu._cube = np.concatenate([cube[:1], cube], axis=0)[:3]
    return emu


def test_z_slice_min_disables_low_z_cells_but_keeps_the_data(monkeypatch):
    """Low-z emulator slices stop being CELLS, but their QSOs are KEPT.

    The data's own z floor (CERDF_Z_MIN) is a separate knob: objects below the
    lowest allowed slice are RE-SNAPPED onto it, not discarded.
    """
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_Z_SLICE_DROP", raising=False)
    emu = _emu_with_low_z()

    z = np.array([0.5, 0.55, 1.0, 1.1, 2.0, 2.0])   # first two would snap to z=0.5
    lbol = np.full(6, 46.2)
    eta = np.full(6, -0.6)

    monkeypatch.setenv("BAQARO_CERDF_Z_SLICE_MIN", "0")     # unrestricted
    cfg_all = L.precompute_cerdf_inputs(z, lbol, eta, emu)
    assert len(cfg_all["log_etas"]) == 6
    assert 0 in cfg_all["unique_keys"][:, 0], "z=0.5 slice should be in use here"

    monkeypatch.setenv("BAQARO_CERDF_Z_SLICE_MIN", "1.0")   # disable the z=0.5 slice
    cfg_cut = L.precompute_cerdf_inputs(z, lbol, eta, emu)

    # ALL SIX objects survive -- the sample is not cut.
    assert len(cfg_cut["log_etas"]) == 6, "QSOs must be kept, not dropped"
    # But no cell sits on the z=0.5 slice any more.
    assert 0 not in cfg_cut["unique_keys"][:, 0], "z=0.5 slice still used as a cell"
    # The two low-z objects were re-snapped onto z=1.0.
    assert (emu.axis_data["redshift"][cfg_cut["redshift_idx"]] >= 1.0).all()
    assert emu.axis_data["redshift"][cfg_cut["redshift_idx"][0]] == 1.0


def test_z_slice_drop_opt_in_removes_them(monkeypatch):
    """BAQARO_CERDF_Z_SLICE_DROP=1 discards instead of re-snapping."""
    monkeypatch.delenv("BAQARO_CERDF_SCATTER_DEX", raising=False)
    monkeypatch.delenv("BAQARO_CERDF_OUTLIER_FRAC", raising=False)
    emu = _emu_with_low_z()
    z = np.array([0.5, 0.55, 1.0, 1.1, 2.0, 2.0])

    monkeypatch.setenv("BAQARO_CERDF_Z_SLICE_MIN", "1.0")
    monkeypatch.setenv("BAQARO_CERDF_Z_SLICE_DROP", "1")
    cfg = L.precompute_cerdf_inputs(z, np.full(6, 46.2), np.full(6, -0.6), emu)
    assert len(cfg["log_etas"]) == 4, "DROP=1 should remove the two low-z QSOs"


def test_over_range_bright_qsos_land_in_the_top_cell(setup, monkeypatch):
    """A QSO brighter than the emulator's top L threshold must be BRACKETED into
    the top cell, not silently misplaced.

    main_mcmc used to exclude these (598 real objects, the brightest in the
    sample). BAQARO_CERDF_KEEP_BRIGHT=1 keeps them, and this is the guarantee the
    likelihood side must honour: `lmin_idx` is clipped to len(thresholds)-2, so an
    over-range QSO gets the top bracket [47.4, 47.5] rather than an out-of-bounds
    index or a wraparound.
    """
    for v in ("BAQARO_CERDF_SCATTER_DEX", "BAQARO_CERDF_OUTLIER_FRAC",
              "BAQARO_CERDF_Z_SLICE_MIN", "BAQARO_CERDF_Z_SLICE_DROP"):
        monkeypatch.delenv(v, raising=False)
    emu, cube, _, _, _ = setup
    top = L_THR[-1]          # 47.5
    n_thr = len(L_THR)

    z = np.array([2.0, 2.0, 2.0])
    lbol = np.array([46.2, top, top + 0.7])   # in-range, exactly at top, over-range
    eta = np.full(3, -0.6)
    cfg = L.precompute_cerdf_inputs(z, lbol, eta, emu)

    # every object keeps a valid bracket, and the two top ones share the TOP cell
    assert (cfg["lmin_idx"] <= n_thr - 2).all(), "lmin_idx escaped the grid"
    assert (cfg["lmax_idx"] <= n_thr - 1).all(), "lmax_idx escaped the grid"
    assert cfg["lmin_idx"][2] == n_thr - 2, "over-range QSO not put in the top cell"
    assert cfg["lmax_idx"][2] == n_thr - 1
    assert cfg["lmin_idx"][1] == cfg["lmin_idx"][2], "at-top and over-top must agree"
    # and the likelihood still evaluates finitely with them in
    assert np.isfinite(L.log_likelihood_cerdf_from_prediction(cube, cfg))


def test_z_slice_min_default_is_one(monkeypatch):
    """Default must be 1.0 -- per_object mode has no legacy to preserve."""
    for v in ("BAQARO_CERDF_Z_SLICE_MIN", "BAQARO_CERDF_Z_SLICE_DROP",
              "BAQARO_CERDF_SCATTER_DEX", "BAQARO_CERDF_OUTLIER_FRAC"):
        monkeypatch.delenv(v, raising=False)
    emu = _emu_with_low_z()
    z = np.array([0.5, 1.0, 2.0])
    cfg = L.precompute_cerdf_inputs(z, np.full(3, 46.2), np.full(3, -0.6), emu)
    assert len(cfg["log_etas"]) == 3, "data must be kept by default"
    assert 0 not in cfg["unique_keys"][:, 0], "z=0.5 slice should be off by default"
