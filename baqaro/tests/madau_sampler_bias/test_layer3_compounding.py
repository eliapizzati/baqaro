"""
LAYER 3 - Realistic compounding: observable-level bias (BHMF / QLF / rho_BH).
============================================================================

Evolves the FULL root71 subsample population through the real per-snapshot cold
accretion-rate histories, once with the transfer table (Branch B, production)
and once with exact direct sampling (Branch A), under the fiducial ERDF. The two
runs share seed mass and rate histories; the ONLY difference is the engine path,
so the gap is the pure madau-sampler bias compounded over cosmic time.

It is the minimal reproduction of the bias, extended to the OBSERVABLES:
  * BHMF: weighted percentile shift of log10 M_BH (table - direct) vs z.
  * rho_BH: total weighted M_BH ratio (table/direct) vs z (Soltan-relevant).
  * QLF: weighted percentile shift of log10 L_bol vs z (expected ~spared).
and maps the dependence on (std_0, logtcoherence).

Mergers and real seeding are omitted: both are identical between the two engine
paths and (being multiplicative / additive in the same way) cancel in the dex
SHIFT. Weighting uses the subset HT weights so
the reported shift is the OBSERVABLE shift, not an unweighted proxy.

Heavy (~7M halos x ~70 snaps x 2 engines per config). Standalone driver does the
fiducial run + a small (std_0, logtcoh) map. A light pytest variant subsamples
columns and asserts sign + rough magnitude.

Run: python -m baqaro.tests.madau_sampler_bias.test_layer3_compounding
"""
import os
import numpy as np
import pytest

from . import _common as C
from baqaro.core_functions.bestfit_registry import get_bestfit
from baqaro.core_functions.erdf_core_functions import Erdf

OUTDIR = os.path.join(os.path.dirname(__file__), "out")

# NB this key predates the growth_const fix, so `get_bestfit` warns. That is
# deliberate and harmless here: the test measures how the f_eff bias COMPOUNDS
# across snapshots, which depends on (mu, sigma) and the snapshot cadence, not
# on the parameters being the currently adopted ones. Keeping the original key
# keeps the reference numbers below comparable to when they were measured.
os.environ.setdefault("BAQARO_ALLOW_STALE_BESTFIT", "1")  # deliberate, see above
FID = get_bestfit("qcc_fid1_z0_chunked_noshift_v1")
RATE_PATTERN = "*root71_flatN20000*specific_cold_accretion_rates.npy"
MASS_PATTERN = "*root71_flatN20000*__halo_masses.npy"
WEIGHTS_CACHE = os.path.join(OUTDIR, "storage_weights_root71_flatN20000.npy")
CHECKPOINTS = {40: 5.88, 51: 3.94, 59: 3.00, 71: 2.00}  # snap -> z label
M_SEED = 1.0e3   # legacy uniform toy seed (only if realistic seeds not supplied)
RATE_FLOOR_LOG = -30.0  # non-accreting halos frozen (log10 rate -> -30 => eta~0)


def compute_seeds(col_idx, logfseed=None, sigmaseed=None, seed=99):
    """Per-column (birth_snap, seed_mass) matching production seeding.

    A BH seeds when its halo first resolves (first snap with halo_mass>0) at
    M_BH = M_halo(birth) * 10**(logfseed + N(0,sigmaseed)). This is essential:
    the specific cold-accretion rate is HIGHEST at early times, so seeding every
    halo at snap 1 (the toy default) lets late-forming halos accrete an early
    high-rate epoch they never see in production -> runaway to unphysical masses
    AND an over-biased bright end (the bias accumulates with total growth).
    Returns (birth_snap[ncols] int32, seed_mass[ncols] float32). Columns that
    never resolve get birth_snap=-1, seed_mass=0 (stay at 0, excluded downstream).
    """
    logfseed = FID["logfseed"] if logfseed is None else logfseed
    sigmaseed = FID["sigmaseed"] if sigmaseed is None else sigmaseed
    mass_path = C.find_rate_cache(MASS_PATTERN)
    if mass_path is None:
        return None, None
    hm = np.asarray(np.load(mass_path, mmap_mode="r")[:, col_idx])  # (n_snap, ncols)
    resolved = hm > 0.0
    ever = resolved.any(axis=0)
    birth = np.where(ever, resolved.argmax(axis=0), -1)
    rng = np.random.default_rng(seed)
    scatter = rng.standard_normal(col_idx.size) * float(sigmaseed)
    m_halo_birth = np.where(ever, hm[np.clip(birth, 0, None), np.arange(col_idx.size)], 0.0)
    seed_mass = (m_halo_birth * 10.0 ** (float(logfseed) + scatter)).astype(np.float32)
    seed_mass[~ever] = 0.0
    return birth.astype(np.int32), seed_mass


def _weighted_pctile(logM, w, p):
    """Weighted percentile of logM (1D), weights w. p in percent."""
    order = np.argsort(logM)
    lm = logM[order]
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    return float(np.interp(p / 100.0, cw, lm))


def _evolve(tf, rate_cache, redshifts, dt, std_0, logtcoh, col_idx, seed=12345,
            checkpoints=CHECKPOINTS, i_start=1, growth_max=50.0,
            birth_snap=None, seed_mass=None):
    """Evolve the selected columns under ONE engine path.

    Returns dict snap -> (M array, L array) at checkpoints. tf=None -> direct.
    ``growth_max`` is the per-snapshot ln-growth cap (production fiducial uses
    4.6 = ~100x/snap; 50.0 = effectively uncapped). The cap interacts strongly
    with the bias because it clamps the high-eta tail where the bias concentrates.
    """
    from baqaro.core_functions.bh_accretion_fast import evolve_BHs_fast
    erdf = Erdf("log_normal_evol_halo_mass",
                {"log_eta_mean_0": FID["log_eta_mean_0"],
                 "log_eta_mean_evol": FID["log_eta_mean_evol"],
                 "std_0": float(std_0)},
                rng=np.random.default_rng(seed))
    tau = 10.0 ** (logtcoh - 6.0)   # Myr
    n = col_idx.size
    # Realistic seeding (birth_snap, seed_mass) when supplied; else legacy uniform
    # toy seed at snap i_start (over-grows -- see compute_seeds docstring).
    if birth_snap is not None:
        M = np.zeros(n, dtype=np.float32)
        early = (birth_snap >= 0) & (birth_snap <= i_start)   # born at/before start
        M[early] = seed_mass[early]
    else:
        M = np.full(n, M_SEED, dtype=np.float32)
    L = np.zeros(n, dtype=np.float64)
    snaps = {}
    for i in range(i_start, redshifts.size):
        if birth_snap is not None and i > i_start:
            born_now = birth_snap == i
            if np.any(born_now):
                M[born_now] = seed_mass[born_now]
        row = np.asarray(rate_cache[i])[col_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rate = np.where(row > 0.0, np.log10(np.maximum(row, 1e-30)),
                                RATE_FLOOR_LOG).astype(np.float32)
        # Cap effective sub-steps at the transfer-table maximum (1001) for BOTH
        # engines. The production table path caps internally (n_eff);
        # capping the direct ground truth the same way isolates the MADAU bias
        # from the separate table cap-scatter residual (Layer 2) and bounds the
        # direct loop cost at short tau_coh. At fiducial tau (~0.5 Myr) n_steps<1001
        # so this is a no-op there.
        n_steps = min(max(1, int(dt[i] * 1e3 / tau)), 1001)
        M, L = evolve_BHs_fast(
            tf, M, float(dt[i]), erdf, rad_efficiency=0.1, n_steps=n_steps,
            log_halo_rates_array=log_rate, rad_efficiency_model="madau+",
            parallelize=False, use_parallel_kernel=True, growth_max=growth_max)
        if i in checkpoints:
            snaps[i] = (M.copy(), L.copy())
    return snaps


def run_config(tf, rate_cache, redshifts, dt, w, std_0, logtcoh, col_idx,
               seed=12345, growth_max=50.0, birth_snap=None, seed_mass=None):
    """Table vs direct for one (std_0, logtcoh); return per-checkpoint diffs."""
    kw = dict(seed=seed, growth_max=growth_max, birth_snap=birth_snap,
              seed_mass=seed_mass)
    tab = _evolve(tf, rate_cache, redshifts, dt, std_0, logtcoh, col_idx, **kw)
    dirr = _evolve(None, rate_cache, redshifts, dt, std_0, logtcoh, col_idx, **kw)
    ww = w[col_idx]
    res = {}
    for snap in sorted(tab):
        Mt, Lt = tab[snap]
        Md, Ld = dirr[snap]
        # BHMF over all born BHs (M>0). With realistic seeding "born" = seed_mass>0;
        # legacy toy uses the grown-above-uniform-seed criterion.
        if birth_snap is not None:
            grown = (Mt > 0.0) | (Md > 0.0)
        else:
            grown = (Mt > M_SEED * 1.0001) | (Md > M_SEED * 1.0001)
        lmt, lmd = np.log10(Mt[grown]), np.log10(Md[grown])
        wg = ww[grown]
        dB = {p: _weighted_pctile(lmt, wg, p) - _weighted_pctile(lmd, wg, p)
              for p in (50, 84, 99)}
        # rho_BH: total weighted mass ratio
        rho_ratio = float(np.sum(ww * Mt) / np.sum(ww * Md))
        # QLF: weighted L percentile shift (only emitting BHs)
        emit = (Lt > 1.0) | (Ld > 1.0)
        llt, lld = np.log10(np.maximum(Lt[emit], 1.0)), np.log10(np.maximum(Ld[emit], 1.0))
        we = ww[emit]
        dQ = {p: _weighted_pctile(llt, we, p) - _weighted_pctile(lld, we, p)
              for p in (50, 99)}
        res[snap] = dict(dBHMF=dB, drho_pct=(rho_ratio - 1.0) * 100.0, dQLF=dQ)
    return res


def _load_inputs():
    rate_path = C.find_rate_cache(RATE_PATTERN)
    if rate_path is None or not os.path.exists(WEIGHTS_CACHE):
        return None
    rate_cache = np.load(rate_path, mmap_mode="r")
    w = np.load(WEIGHTS_CACHE)
    redshifts, dt = C.load_cadence("L2800N10080", 71)
    return rate_cache, w, redshifts, dt


def main():
    inp = _load_inputs()
    if inp is None:
        print("[skip] rate cache or weights cache missing")
        return
    rate_cache, w, redshifts, dt = inp
    tf = C.load_sampler()
    n_tot = w.size
    # Column budget: full population by default; BAQARO_BIAS_NCOLS subsamples for
    # speed (BHMF percentile shifts are robust to it; rho_BH is noisier).
    ncols = int(os.environ.get("BAQARO_BIAS_NCOLS", "0"))
    if ncols and ncols < n_tot:
        rng = np.random.default_rng(0)
        col_idx = np.sort(rng.choice(n_tot, size=ncols, replace=False))
        print(f"[BAQARO_BIAS_NCOLS] subsampling {ncols:,} of {n_tot:,} columns")
    else:
        col_idx = np.arange(n_tot)  # full population

    print(f"\nLAYER 3: full root71 population ({n_tot:,} halos), fiducial ERDF "
          f"(eta0={FID['log_eta_mean_0']:.3f}, evol={FID['log_eta_mean_evol']:.3f})")
    print("Observable bias = table - direct (weighted). dBHMF/dQLF in dex, drho in %.\n")

    # ---- Fiducial run vs z, at uncapped AND production growth cap ----
    std0 = FID["std_0"]; ltc = FID["logtcoherence"]
    for gmax, tag in ((50.0, "UNCAPPED g_max=50"), (4.6, "PRODUCTION g_max=4.6")):
        print(f"=== FIDUCIAL [{tag}]  std_0={std0:.3f}  logtcoh={ltc:.3f} "
              f"(tau={10**(ltc-6):.3f} Myr) ===")
        res = run_config(tf, rate_cache, redshifts, dt, w, std0, ltc, col_idx,
                         growth_max=gmax)
        print(f"{'z':>6} {'snap':>5} | {'dBHMF50':>8} {'dBHMF84':>8} {'dBHMF99':>8} | "
              f"{'drhoBH%':>8} | {'dQLF50':>7} {'dQLF99':>7}")
        for snap in sorted(res):
            r = res[snap]
            print(f"{CHECKPOINTS[snap]:6.2f} {snap:5d} | "
                  f"{r['dBHMF'][50]:+8.4f} {r['dBHMF'][84]:+8.4f} {r['dBHMF'][99]:+8.4f} | "
                  f"{r['drho_pct']:+8.2f} | {r['dQLF'][50]:+7.4f} {r['dQLF'][99]:+7.4f}")
        print()

    # ---- (std_0, logtcoh) map at final z, at the production cap ----
    # Short-tau (logtcoh=4.5) configs loop ~1000 substeps in the direct engine;
    # use a smaller column budget for the map (BHMF percentile shifts are robust).
    map_cols = col_idx if col_idx.size <= 300_000 else np.sort(
        np.random.default_rng(1).choice(col_idx, size=300_000, replace=False))
    print(f"=== (std_0, logtcoh) dependence at z=2 (snap 71), production g_max=4.6, "
          f"{map_cols.size:,} cols ===")
    print(f"{'std_0':>6} {'logtcoh':>8} {'tau[Myr]':>9} | {'dBHMF50':>8} {'dBHMF84':>8} "
          f"{'dBHMF99':>8} | {'drhoBH%':>8}")
    for std0 in (0.32, 0.54, 0.70):
        for ltc in (4.5, 5.7, 6.5):
            res = run_config(tf, rate_cache, redshifts, dt, w, std0, ltc, map_cols,
                             growth_max=4.6)
            r = res[71]
            print(f"{std0:6.2f} {ltc:8.2f} {10**(ltc-6):9.3f} | "
                  f"{r['dBHMF'][50]:+8.4f} {r['dBHMF'][84]:+8.4f} {r['dBHMF'][99]:+8.4f} | "
                  f"{r['drho_pct']:+8.2f}")
    _plot_fiducial_vs_z(tf, rate_cache, redshifts, dt, w, col_idx)


def _plot_fiducial_vs_z(tf, rate_cache, redshifts, dt, w, col_idx):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped: {e}]")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for gmax, ls in ((50.0, "-"), (4.6, "--")):
        res = run_config(tf, rate_cache, redshifts, dt, w, FID["std_0"],
                         FID["logtcoherence"], col_idx, growth_max=gmax)
        zs = [CHECKPOINTS[s] for s in sorted(res)]
        lab = f"g_max={gmax:g}"
        for p, mk in ((50, "o"), (84, "s"), (99, "^")):
            ax.plot(zs, [res[s]["dBHMF"][p] for s in sorted(res)], mk + ls,
                    label=f"BHMF p{p} ({lab})")
    ax.axhline(0, color="k", lw=0.6)
    ax.invert_xaxis()
    ax.set_xlabel("redshift"); ax.set_ylabel("table - direct  [dex]")
    ax.set_title("Madau-sampler bias compounding (fiducial)\n"
                 "Layer 3, weighted; solid=uncapped, dashed=production g_max=4.6")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "layer3_bias_vs_z.png")
    fig.savefig(path, dpi=120)
    print(f"\n[saved] {path}")


# ----------------------------------------------------------------------
# pytest (light: subsample columns)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def inputs():
    inp = _load_inputs()
    if inp is None:
        pytest.skip("rate cache / weights cache unavailable")
    try:
        tf = C.load_sampler()
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"sampler npz unavailable: {e}")
    return (tf, *inp)


def test_fiducial_bhmf_undergrows_and_grows_with_time(inputs):
    tf, rate_cache, w, redshifts, dt = inputs
    rng = np.random.default_rng(0)
    col_idx = np.sort(rng.choice(w.size, size=400_000, replace=False))
    res = run_config(tf, rate_cache, redshifts, dt, w, FID["std_0"],
                     FID["logtcoherence"], col_idx)
    # bright end under-grows at z=2
    assert res[71]["dBHMF"][99] < -0.01, res[71]["dBHMF"]
    # compounds: |p99 shift| at z=2 >= at z=6
    assert abs(res[71]["dBHMF"][99]) >= abs(res[40]["dBHMF"][99])
    # QLF median much more spared than BHMF p99
    assert abs(res[71]["dQLF"][50]) < abs(res[71]["dBHMF"][99])


if __name__ == "__main__":
    main()
