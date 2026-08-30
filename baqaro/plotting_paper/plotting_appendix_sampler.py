"""
APPENDIX FIGURE: validation of the fast sum-of-lognormals accretion sampler.
============================================================================

The accretion engine builds BH mass over ``n_steps`` independent sub-steps per
snapshot, each contributing a lognormal Eddington-ratio draw. To avoid an
O(n_steps) inner loop, the sum of (n_steps-1) lognormals is drawn in one shot
from a precomputed 3D inverse-CDF transfer table (the "Universal Lognormal
Sampler"). This appendix figure benchmarks that machinery in two panels:

  (a) Distribution fidelity. For three representative (N, sigma) regimes the
      table-drawn sum-of-lognormals PDF is overlaid on a brute-force Monte-Carlo
      sum of N independent lognormals. The table reproduces the true
      distribution body AND the bright tail (which sets the QLF bright end).

  (b) Evolved BH mass function, with vs without the sampler. A REAL
      ``main_evolution`` forward run at the adopted fiducial parameters, down to
      z=2, is evolved twice on the same (small) subsample: once with the fast
      transfer-table sampler (production path), once with the table forced to
      None so the engine falls back to brute-force direct sub-step sampling
      (Branch A). The two BHMFs agree to <0.01 dex everywhere, confirming the
      sampler is a faithful drop-in for the exact direct sampling -- which, at
      the fiducial short coherence time, is the only computationally feasible
      path at production scale.

      EFFICIENCY MODE -- the demo flavour tracks the adopted fiducial:
        * feffcorr-ON fiducial (madau+, the adopted g6.21 fiducial): the runs use
          the PRODUCTION ``madau+`` efficiency with the f_eff correction ON, so this
          panel validates the FULL production path -- Branch B (table + median-eta
          eps + feffcorr) against the exact Branch A (per-sub-step eps, the
          reference). Branch A never touches ``feff_arr``, so the without-sampler
          reference stays exact regardless of the correction; agreement to
          <0.01 dex demonstrates the f_eff correction restores the table mean.
        * feffcorr-OFF fiducial (an uncapped or constant-epsilon run): the runs use CONSTANT epsilon
          (``BAQARO_RAD_EFFICIENCY_MODEL=constant``), where per-sub-step growth is
          exactly linear in eta so the table is mathematically exact and the raw
          sum-of-lognormals machinery is validated in isolation.
      Panel (a) (the pure sum-of-lognormals PDF) is efficiency-independent and
      exact either way.

The two forward runs are produced by ``main_evolution`` (see RUN COMMANDS below);
this script only reads them. Panel (a) needs the transfer-table npz on disk
(``BAQARO_SOURCE_DIR``, default machine_igm).

RUN COMMANDS (produce the panel-(b) inputs once, then render). The runs stop at
z=4 (``BAQARO_MAX_SNAP=50``) so the direct Branch A stays feasible even at the FULL
production subsample (``BAQARO_SUBSAMPLE_NB=500000`` -> root50 flatN500000, ~12.9M
halos): z=4 is 0.44x the z=2 sub-step budget, so big-N direct costs ~0.8x the old
z=2 / N=20000 set. The plotting side defaults its glob to ``fiducial_data.RESOLVED``
(bestfit + ``_g{cap}`` + ``_feffcorr`` + the ``samplerdemo_madau``/``samplerdemo_const``
notes base) at ``maxsnap_50`` / ``flatN500000``, so a repoint auto-selects the
matching demo set. For the adopted fiducial (madau+ + feffcorr):
    env BAQARO_SIM=L2800N10080 BAQARO_MAX_SNAP=50 BAQARO_FOLD_SUBHALO_MASS=1 \
        BAQARO_MERGER_DELAY_MODE=instant_new BAQARO_BESTFIT_NAME=qcc_ck22final_v1 \
        BAQARO_USE_SUBSAMPLE=1 BAQARO_SUBSAMPLE_NB=500000 BAQARO_GROWTH_SUM_MAX=6.21 \
        BAQARO_MADAU_FEFF_CORRECTION=1 BAQARO_NOTES_FILE=samplerdemo_madau \
        python -m baqaro.core_functions.main_evolution            # with sampler (Branch B)
    env ... BAQARO_MADAU_FEFF_CORRECTION=1 \
        BAQARO_NOTES_FILE=samplerdemo_madau_nosampler BAQARO_FORCE_NO_SAMPLER=1 \
        python -m baqaro.core_functions.main_evolution            # without (exact Branch A)
Three accretion seeds are averaged: default RNG (seed1), BAQARO_RNG_SEED=67890
(seed2, NOTES ...seed2), BAQARO_RNG_SEED=99999 (seed3, NOTES ...seed3), each with
its own `_nosampler` twin. (A constant-eps set for an earlier fiducial lived at
``maxsnap_71`` / ``flatN20000`` / ``samplerdemo_const`` — override
``BAQARO_APPENDIX_SAMPLER_MAXSNAP`` / ``_NBTOK`` / ``_ZS`` to render it.)

    env BAQARO_SAVE_FIGS=1 BAQARO_HEADLESS=1 \
        python -m baqaro.plotting_paper.plotting_appendix_sampler   # default (pinned) fiducial
    env BAQARO_PAPER_BESTFIT_NAME=<other> BAQARO_PAPER_GROWTH_SUM_MAX=<cap> \
        BAQARO_PAPER_MADAU_FEFF_CORRECTION=1 BAQARO_SAVE_FIGS=1 BAQARO_HEADLESS=1 \
        python -m baqaro.plotting_paper.plotting_appendix_sampler   # another registered fit
"""

import os
import glob

import numpy as np
import h5py
import matplotlib.gridspec as gridspec
from matplotlib.transforms import blended_transform_factory

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import plt, z_scalar_mapper
from baqaro.plotting_common.plot_config import save_fig, maybe_show
from baqaro.utils.my_dir import get_output_path
from baqaro.core_functions.fast_lognormal_sampler import load_3d_sampler
# Pinned fiducial identity — so panel (b)'s demo runs default to the SAME
# bestfit params + growth cap as the adopted fiducial (a repoint moves them too).
from baqaro.plotting_paper.fiducial_data import RESOLVED as _PAPER_RESOLVED

name_fig = "appendix_sampler_validation"

# ---------------------------------------------------------------------------
# Shared constants / styling
# ---------------------------------------------------------------------------
SAMPLER_NPZ = "Universal_Lognormal_Sampler_final.npz"
SEED = 12345
CASE_COLORS = ["#4477AA", "#228833", "#AA3377"]   # Tol bright: blue/green/purple
SAMPLER_COLOR = "#EE6677"                          # Tol red for the "with sampler" curve

# Tokens identifying the panel-(b) main_evolution runs (see RUN COMMANDS).
#
# ===========================================================================
# WHY PANEL (b) NEEDS 6 RUNS, NOT 1
# ===========================================================================
# Panel (b) costs 2 ARMS x 3 SEEDS = 6 `main_evolution` runs. Both multipliers
# are load-bearing; neither is boilerplate.
#
#   x2 ARMS -- the panel IS a comparison. It validates the transfer-table
#     sampler (Branch B, dashed) against exact direct per-sub-step Monte Carlo
#     (Branch A, solid, produced with BAQARO_FORCE_NO_SAMPLER=1). The two arms
#     must be identical in EVERY respect except the sampler: same params, same
#     halos, same seed. A single run has nothing to compare against, and if
#     either arm's glob comes back empty the whole panel degrades to
#     an "inputs not found" placeholder (see the `if not gw_s1 or not gn_s1`
#     branch below) -- it does NOT raise. A missing arm therefore looks like a
#     rendering glitch rather than an error.
#
#   x3 SEEDS -- the seeds ARE the error bar. The effect being measured (sampler
#     bias, ~0.05-0.2 dex at the bright end) is comparable to per-realization
#     shot noise there, so one pair cannot separate "the sampler is biased"
#     from "this realization fluctuated". `_stack_seeds` below hard-codes
#     `phi_spread = 0` when n_seeds < 2, so a single-seed render produces a
#     ratio panel with NO uncertainty at all and any excursion reads as a
#     detection. 3 seeds + the c4 small-sample correction is the published
#     configuration: a bigger sample would shrink the shot noise but still
#     yield no empirical spread estimate, and the bright-end bins stay
#     few-object regardless.
#
#     Seeds are: default RNG (seed1, no BAQARO_RNG_SEED), 67890 (seed2),
#     99999 (seed3) -- each with its own `_nosampler` twin.
#
# SAMPLE SIZE: the constant-eps set ran at `flatN20000`/maxsnap_71; the madau+
# sets run at `flatN500000` (25x larger) / maxsnap_50 (z=4.08). Both use 3 seeds.
#
# CONSTANT-eps vs MADAU+ IS A REAL CHOICE, not a config detail:
#   * Under madau+ without the f_eff correction, Branch B applies eps once at
#     the median eta (`core_functions/madau_feff.py` documents the correction);
#     a CONSTANT-eps panel makes the sampler mathematically exact, so the
#     figure isolates the SAMPLER alone.
#   * With `_feffcorr` ON a madau+ render is again a valid sampler test and
#     comes out at ratio ~1.00 -- but it is a DIFFERENT test from the
#     constant-eps one. Do not swap them when comparing appendices across
#     fiducials.
# ===========================================================================
_EVO_DIR = os.path.join(str(get_output_path(
    source=os.environ.get("BAQARO_SOURCE_DIR", "machine_igm"))), "evolution")
#
# The demo run flavour tracks the adopted fiducial's efficiency config:
#   * feffcorr ON  (madau+, the adopted fiducial) → `samplerdemo_madau_*`
#     runs at madau+ efficiency with the f_eff correction ON. This validates the
#     FULL production path: Branch B (table + median-eta eps + feffcorr) vs the
#     exact Branch A (per-sub-step eps, no correction — the reference). The runs
#     carry the `_feffcorr` token. feffcorr is a no-op on Branch A (it never
#     touches feff_arr), so the without-sampler reference stays exact.
#   * feffcorr OFF (a constant-epsilon run) → `samplerdemo_const_*` runs at
#     CONSTANT epsilon, where the sum-of-lognormals table is mathematically exact
#     and the raw sampler machinery is validated in isolation (no `_feffcorr`).
# The rest of the identity — bestfit params + growth cap — also defaults to the
# fiducial (via fiducial_data.RESOLVED), so a repoint auto-selects the matching
# demo set. Override any piece with BAQARO_APPENDIX_SAMPLER_{BESTFIT,GTOK,NOTES}.
# When the runs are absent, panel (b) renders an "inputs not found" placeholder.
_DEMO_FEFF = str(_PAPER_RESOLVED.get("BAQARO_MADAU_FEFF_CORRECTION", "0")).strip().lower() \
    not in ("0", "", "false", "no", "off")


def _fiducial_gtok():
    g = float(_PAPER_RESOLVED.get("BAQARO_GROWTH_SUM_MAX", "50.0"))
    tok = "" if g == 50.0 else ("_g%d" % int(g) if float(g).is_integer() else "_g%g" % g)
    if _DEMO_FEFF:
        tok += "_feffcorr"
    return tok


_DEMO_BESTFIT = os.environ.get(
    "BAQARO_APPENDIX_SAMPLER_BESTFIT", _PAPER_RESOLVED["BAQARO_BESTFIT_NAME"])
_DEMO_GTOK = os.environ.get("BAQARO_APPENDIX_SAMPLER_GTOK", _fiducial_gtok())
_DEMO_NOTES_BASE = os.environ.get(
    "BAQARO_APPENDIX_SAMPLER_NOTES",
    "samplerdemo_madau" if _DEMO_FEFF else "samplerdemo_const")
# maxsnap / subsample-size / z-anchors are tied to the demo flavour so each
# fiducial resolves to its OWN run set (all env-overridable):
#   * feffcorr ON (madau): z=4 (snap 50) at the FULL production
#     subsample (flatN500000, ~12.9M halos) — stopping at z=4 makes the direct
#     Branch A cheap enough (0.44x the z=2 sub-step budget) for big N, giving a
#     clean faint/mid BHMF. z-anchors 6/5/4.
#   * feffcorr OFF (const): the z=2 (snap 71) small-N set
#     (flatN20000). z-anchors 6/5/4/3/2.
_DEMO_MAXSNAP = os.environ.get("BAQARO_APPENDIX_SAMPLER_MAXSNAP", "50" if _DEMO_FEFF else "71")
_DEMO_NBTOK = os.environ.get("BAQARO_APPENDIX_SAMPLER_NBTOK",
                             "flatN500000" if _DEMO_FEFF else "flatN20000")


def _demo_glob(variant):
    """Glob for one demo variant ('' | 'seed2' | 'nosampler' | 'nosampler_seed2' | ...).

    Trailing `*` before `.hdf5`: the existing demo files encode "nosampler" in
    the NOTES token, and a `BAQARO_FORCE_NO_SAMPLER=1` run ALSO appends an
    automatic `_nosampler` physics-toggle token AFTER `{_DEMO_GTOK}`. The
    wildcard accepts both naming generations.
    """
    mid = (f"_{_DEMO_NOTES_BASE}_{variant}_bestfit_" if variant
           else f"_{_DEMO_NOTES_BASE}_bestfit_")
    return (f"bh_evolution_*maxsnap_{_DEMO_MAXSNAP}_*{mid}{_DEMO_BESTFIT}"
            f"_sub_*{_DEMO_NBTOK}*{_DEMO_GTOK}*.hdf5")


_EVO_GLOB_WITH_S1 = _demo_glob("")
_EVO_GLOB_WITHOUT_S1 = _demo_glob("nosampler")
_EVO_GLOB_WITH_S2 = _demo_glob("seed2")
_EVO_GLOB_WITHOUT_S2 = _demo_glob("nosampler_seed2")
_EVO_GLOB_WITH_S3 = _demo_glob("seed3")
_EVO_GLOB_WITHOUT_S3 = _demo_glob("nosampler_seed3")


# ===========================================================================
# Panel (a): table-drawn PDF vs brute-force MC sum-of-lognormals
# ===========================================================================
def _batched_direct_mc(n_samples, n_test, sig_nat, rng, batch=2_000_000):
    """Brute-force direct MC sum-of-lognormals at large n_samples (1e8+),
    batched so the (n_samples, n_test) draw never blows memory."""
    out = np.empty(n_samples)
    pos = 0
    while pos < n_samples:
        b = min(batch, n_samples - pos)
        X = rng.lognormal(0.0, sig_nat, (b, n_test))
        out[pos:pos + b] = X.sum(axis=1)
        pos += b
    return out


def _batched_transfer(sampler, n_samples, n_test, sig_dex, rng, batch=2_000_000):
    """Same shape as the direct-MC path, batched."""
    out = np.empty(n_samples)
    pos = 0
    while pos < n_samples:
        b = min(batch, n_samples - pos)
        out[pos:pos + b] = sampler(
            n_arr=np.full(b, n_test),
            mu_dex_arr=np.zeros(b),
            sigma_dex_arr=np.full(b, sig_dex),
            rng=rng,
        )
        pos += b
    return out


def _panel_a_cache_path():
    """Cache path for the left-panel histogram results. Lives next to the
    transfer-table .npz under {path_out}/transfer_functions/."""
    base = os.path.join(str(get_output_path(
        source=os.environ.get("BAQARO_SOURCE_DIR", "machine_igm"))),
        "transfer_functions")
    return os.path.join(base, "appendix_sampler_panel_a_cache.npz")


def _load_panel_a_cache(cache_path, cases, n_samples, bins):
    """Return cached histograms if compatible, else None. Cache is invalidated
    if any of (cases, n_samples, bins) changes."""
    if not os.path.exists(cache_path):
        return None
    if int(os.environ.get("BAQARO_APPENDIX_SAMPLER_REFIT", "0")):
        return None
    try:
        d = np.load(cache_path, allow_pickle=False)
        cached_cases = d["cases"]
        if (cached_cases.shape != (len(cases), 2)
                or not np.allclose(cached_cases, np.asarray(cases))):
            return None
        if int(d["n_samples"]) != n_samples:
            return None
        if not np.array_equal(d["bins"], bins):
            return None
        return {
            "truth_pdf": d["truth_pdf"],     # (n_cases, n_bins-1)
            "samp_pdf":  d["samp_pdf"],
            "ratios":    d["ratios"],
        }
    except Exception as exc:
        print(f"  panel (a) cache load failed ({exc}); regenerating")
        return None


def plot_distribution_fidelity(sub_axes, sampler):
    """Panel: transfer-table draws vs direct Monte-Carlo, for three (N, sigma) cases.

    Compares the sum-of-lognormals the sampler returns against brute-force
    sampling of the same distribution, over a range wide enough to show the
    bright tail (which is what drives the BHMF bright end).
    """
    cases = [(10, 0.35), (50, 0.5), (200, 0.65)]
    # 2e5 -> 1e8 so the validation histograms have a clean bright tail (the
    # tail drives the BHMF bright end). Drawn in batches so the temporary
    # (n_samples, n_test) array never blows memory. The 1e8 MC takes ~15-20
    # min; histograms are CACHED to disk so re-renders for styling tweaks
    # take seconds. Set BAQARO_APPENDIX_SAMPLER_REFIT=1 to force a re-draw.
    n_samples = 100_000_000

    # Shared log10(S / <S>) range so the x-axis is truly shared across all
    # three subpanels — normalising by E[S] = N·exp(sigma^2/2) places all
    # three at mean 1.
    SHARED_LO_LOG, SHARED_HI_LOG = np.log10(0.2), np.log10(30.0)
    # 120 bins so the per-bin width stays ~0.018 dex (same as before) despite
    # the wider range.
    shared_bins = np.logspace(SHARED_LO_LOG, SHARED_HI_LOG, 120)
    # Display range is slightly tighter than the binning so the bright tail
    # has room to fade out without showing empty space at the very edge.
    DISPLAY_HI_LOG = np.log10(25.0)

    cache_path = _panel_a_cache_path()
    cache = _load_panel_a_cache(cache_path, cases, n_samples, shared_bins)
    if cache is not None:
        print(f"  panel (a): loaded cached histograms from {cache_path}")
    else:
        print(f"  panel (a): cache miss -- drawing {n_samples:,} samples per "
              f"case per path (will cache result for next render)")

    # If cache is missing, accumulate fresh histograms; if hit, just slice.
    new_truth_pdf = np.empty((len(cases), shared_bins.size - 1))
    new_samp_pdf  = np.empty((len(cases), shared_bins.size - 1))
    new_ratios    = np.empty(len(cases))

    for k, (ax, (n_test, sig_dex)) in enumerate(zip(sub_axes, cases)):
        color = CASE_COLORS[k]

        sig_nat = sig_dex * np.log(10.0)
        # E[X] for a lognormal with mu_nat=0: exp(sigma_nat^2 / 2)
        expected_S = n_test * np.exp(sig_nat ** 2 / 2.0)

        if cache is not None:
            truth_density = cache["truth_pdf"][k]
            samp_density  = cache["samp_pdf"][k]
            ratio         = float(cache["ratios"][k])
        else:
            print(f"    case {k+1}/{len(cases)}: N={n_test}, sigma={sig_dex} dex")
            rng_truth = np.random.default_rng(SEED + k)
            S_truth = _batched_direct_mc(n_samples, n_test, sig_nat, rng_truth) / expected_S
            rng_samp = np.random.default_rng(SEED + 1000 + k)
            S_samp = _batched_transfer(sampler, n_samples, n_test, sig_dex, rng_samp) / expected_S
            # Pre-bin to density so the cache size stays trivial.
            truth_counts, _ = np.histogram(S_truth, bins=shared_bins)
            samp_counts,  _ = np.histogram(S_samp,  bins=shared_bins)
            widths = np.diff(shared_bins)
            truth_density = truth_counts / (S_truth.size * widths)
            samp_density  = samp_counts  / (S_samp.size  * widths)
            ratio = float(S_samp.mean() / S_truth.mean())
            new_truth_pdf[k] = truth_density
            new_samp_pdf[k]  = samp_density
            new_ratios[k]    = ratio
            # Free the 1e8-sample arrays before the next case.
            del S_truth, S_samp

        # stairs takes pre-binned density arrays so we can render straight
        # from cache without ever holding the 1e8 samples in memory again.
        ax.stairs(truth_density, shared_bins, fill=True,
                  color="0.75", edgecolor="0.45", linewidth=0.8,
                  label="direct MC")
        ax.stairs(samp_density, shared_bins, fill=False,
                  color=color, linewidth=1.8,
                  label="transfer table")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** SHARED_LO_LOG, 10 ** DISPLAY_HI_LOG)
        ax.set_ylabel("PDF")
        # Two-line info label at the bottom-centre. Math symbols are bold
        # via mathtext \mathbf{}; surrounding plain-text words ("dex",
        # "ratio") get bolded by matplotlib's fontweight='bold' on the
        # text() call itself (mathtext's \mathbf{} would italicise them
        # as variable-name math, which we don't want).
        info = (rf"$\mathbf{{N={n_test}, \sigma={sig_dex}}}$ dex"
                "\n"
                rf"$\mathbf{{\langle S\rangle}}$ ratio $\mathbf{{= {ratio:.3f}}}$")
        ax.text(0.5, 0.06, info,
                transform=ax.transAxes, va="bottom", ha="center", fontsize=11,
                fontweight="bold", color=color)
        ax.set_ylim(1e-9 * ax.get_ylim()[1], ax.get_ylim()[1] * 1.5)
        if k == len(cases) - 1:
            ax.set_xlabel(r"sum of lognormals, $S\,/\,\langle S\rangle$")
            ax.legend(loc="upper right", fontsize=11, handlelength=1.6,
                      borderaxespad=0.4, markerfirst=False)

    # Persist the freshly-computed histograms so the next render skips the
    # ~15-min 1e8 MC. The cache key is (cases, n_samples, bins).
    if cache is None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            cases=np.asarray(cases, dtype=float),
            n_samples=np.int64(n_samples),
            bins=shared_bins,
            truth_pdf=new_truth_pdf,
            samp_pdf=new_samp_pdf,
            ratios=new_ratios,
        )
        print(f"  panel (a): cached histograms -> {cache_path}")


# ===========================================================================
# Panel (b): real main_evolution BHMF at z=2, with vs without the sampler
# ===========================================================================
def _bhmf_from_run(path, bins, snap_indices):
    """Weighted BHMF (comoving number density per dex) at given snapshots.

    Returns (phi, phi_err, redshifts) where phi / phi_err have shape
    (n_snaps_requested, n_bins) and phi_err is the Horvitz-Thompson 1-sigma
    per bin (sqrt(sum w^2) / V / dlog) -- the small-number/weight noise
    that dominates the bright end where only tens of BHs survive.
    """
    with h5py.File(path, "r") as f:
        w = np.asarray(f["subset/weights"], dtype=np.float64)
        box = float(f.attrs.get("boxsize", 2800.0))
        zs = np.asarray(f["redshifts"][:])
        bh_arr = np.asarray(f["black_hole_masses_all"][snap_indices], dtype=np.float64)
    V = box ** 3                       # Mpc^3 (boxsize convention: no h factor)
    dlog = bins[1] - bins[0]
    n_snaps = bh_arr.shape[0]
    n_bins = bins.size - 1
    phi = np.zeros((n_snaps, n_bins))
    phi_err = np.zeros((n_snaps, n_bins))
    for i in range(n_snaps):
        bh = bh_arr[i]
        m = bh > 0
        lm = np.log10(bh[m])
        counts, _ = np.histogram(lm, bins, weights=w[m])
        sumw2, _ = np.histogram(lm, bins, weights=w[m] ** 2)
        phi[i] = counts / V / dlog
        phi_err[i] = np.sqrt(sumw2) / V / dlog
    return phi, phi_err, zs[snap_indices]


def _resolve_snap_indices(path, target_zs, tol=0.5):
    """Map target_zs -> snapshot indices in the HDF5 (nearest within tol)."""
    with h5py.File(path, "r") as f:
        zs = np.asarray(f["redshifts"][:])
    out = []
    for tz in target_zs:
        i = int(np.argmin(np.abs(zs - tz)))
        if abs(zs[i] - tz) > tol:
            continue
        out.append(i)
    return out


# Redshifts to overplot in the right panel, tied to the demo flavour's run depth:
#   * feffcorr ON (madau): big-N runs stop at z=4 (snap 50) -> anchors 6/5/4.
#   * feffcorr OFF (g4.6): const runs reach z=2 (snap 71) -> anchors 6/5/4/3/2.
# Override with BAQARO_APPENDIX_SAMPLER_ZS="6,5,4".
PANEL_B_TARGET_ZS = [float(x) for x in os.environ.get(
    "BAQARO_APPENDIX_SAMPLER_ZS", "6,5,4" if _DEMO_FEFF else "6,5,4,3,2").split(",")]


def _bhmf_seed_average(paths, bins, snap_idx):
    """Average BHMF over the supplied list of HDF5 paths (one per seed).

    Returns (phi_mean, phi_spread, zs) where phi_spread is |max - min| / 2
    across seeds per bin -- a noisy but unbiased 1-sigma estimate of the
    mean from N=2 independent realisations. Used as the shaded band so the
    reader sees realization-to-realization variation directly, rather than
    the per-run HT estimator.
    """
    per_run = []
    z_actual = None
    for p in paths:
        phi, _err, zs = _bhmf_from_run(p, bins, snap_idx)
        per_run.append(phi)
        z_actual = zs
    stack = np.stack(per_run, axis=0)          # (n_seeds, n_z, n_bins)
    n_seeds = len(per_run)
    phi_mean = stack.mean(axis=0)
    # 1-sigma on the mean = SEM = s / sqrt(N), with s the sample std (ddof=1).
    # Even s(ddof=1) is biased LOW at tiny N (E[s] = c4(N) * sigma), so divide by
    # Hartley's c4 to make the estimator unbiased. MC-verified against the true
    # SEM at N=2 and N=3:
    #     estimator          N=2      N=3
    #     s/sqrt(N)          0.80x    0.89x
    #     s/(c4*sqrt(N))     1.00x    1.00x     <- this one
    # With a single seed there is no spread to report -> 0.
    _C4 = {2: 0.7978845608, 3: 0.8862269255, 4: 0.9213177319, 5: 0.9399856030}
    if n_seeds < 2:
        phi_spread = np.zeros_like(phi_mean)
    else:
        c4 = _C4.get(n_seeds, 1.0 - 1.0 / (4.0 * n_seeds))   # asymptotic for N>5
        phi_spread = stack.std(axis=0, ddof=1) / (c4 * np.sqrt(n_seeds))
    return phi_mean, phi_spread, z_actual


def plot_real_evolution_bhmf(ax, ax_res):
    """Panel: BHMF from real forward runs, sampler vs direct per-sub-step draws.

    Overlays the two engines' z=0 BHMFs (and their residual) using the paired
    runs on disk, so the table-vs-exact difference is shown on the actual model
    rather than on a synthetic distribution.
    """
    gw_s1 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITH_S1))
    gn_s1 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITHOUT_S1))
    gw_s2 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITH_S2))
    gn_s2 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITHOUT_S2))
    gw_s3 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITH_S3))
    gn_s3 = glob.glob(os.path.join(_EVO_DIR, _EVO_GLOB_WITHOUT_S3))
    if not gw_s1 or not gn_s1:
        ax.text(0.5, 0.5, "panel (b) inputs not found\n"
                "run main_evolution (see module docstring)",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color="firebrick")
        ax_res.set_visible(False)
        return

    snap_idx = _resolve_snap_indices(gw_s1[0], PANEL_B_TARGET_ZS)
    bins = np.arange(4.5, 11.5 + 0.25, 0.25)
    cen = 0.5 * (bins[:-1] + bins[1:])

    # Average across the 3 seeds (12345 + 67890 + 99999) so the shaded band
    # reflects empirical realisation-to-realisation spread rather than the
    # HT estimate. Seeds that aren't on disk are skipped.
    with_paths    = gw_s1 + (gw_s2 if gw_s2 else []) + (gw_s3 if gw_s3 else [])
    without_paths = gn_s1 + (gn_s2 if gn_s2 else []) + (gn_s3 if gn_s3 else [])
    phi_with,    err_with,    z_actual = _bhmf_seed_average(with_paths,    bins, snap_idx)
    phi_without, err_without, _        = _bhmf_seed_average(without_paths, bins, snap_idx)
    n_seeds = len(with_paths)
    print(f"  panel (b): averaging over {n_seeds} seed(s) per sampling mode")

    def _logphi(p):
        out = np.full_like(p, np.nan)
        out[p > 0] = np.log10(p[p > 0])
        return out

    def _logphi_band(p, err):
        """1-sigma upper/lower bounds in log10-phi (HT shot noise)."""
        upper = p + err
        lower = p - err
        log_hi = np.full_like(p, np.nan)
        log_lo = np.full_like(p, np.nan)
        m_hi = upper > 0
        log_hi[m_hi] = np.log10(upper[m_hi])
        m_lo = lower > 0
        log_lo[m_lo] = np.log10(lower[m_lo])
        log_lo[~m_lo] = -9.0
        return log_lo, log_hi

    # Use the main-text redshift colormap so this figure's redshift colours
    # match qlf/cerdf/qhmf/clustering (z_scalar_mapper: rainbow_PuRd, vmax=7).
    z_colors = [z_scalar_mapper.to_rgba(z) for z in z_actual]

    # Plot per-z: lighter shaded band for shot noise, solid = without sampler
    # (reference), dashed = with sampler (drop-in).
    z_handles = []
    for i, z_i in enumerate(z_actual):
        col = z_colors[i]
        lo_w, hi_w = _logphi_band(phi_without[i], err_without[i])
        lo_s, hi_s = _logphi_band(phi_with[i], err_with[i])
        ax.stairs(hi_w, bins, baseline=lo_w, fill=True, color=col, alpha=0.10,
                  linewidth=0)
        ax.stairs(hi_s, bins, baseline=lo_s, fill=True, color=col, alpha=0.10,
                  linewidth=0)
        ax.stairs(_logphi(phi_without[i]), bins, fill=False, color=col, lw=2.0,
                  baseline=None)
        ax.stairs(_logphi(phi_with[i]), bins, fill=False, color=col, lw=1.6,
                  ls="--", baseline=None)
        z_handles.append(plt.Line2D([0], [0], color=col, lw=2.0, ls="-",
                                     label=f"$z = {round(z_i)}$"))

    # Style convention (solid / dashed); separate from the z-colour legend.
    style_handles = [
        plt.Line2D([0], [0], color="0.2", lw=2.0, ls="-",  label="direct MC"),
        plt.Line2D([0], [0], color="0.2", lw=1.6, ls="--", label="transfer table"),
    ]
    ax.set_ylabel(r"$\log_{10}\,\phi_{\rm BHMF}\ [{\rm Mpc^{-3}\,dex^{-1}}]$")
    ax.set_ylim(-9.0, -1.0)
    leg_z = ax.legend(handles=z_handles, loc="upper right", fontsize=12,
                      markerfirst=False, handlelength=1.6)
    ax.add_artist(leg_z)
    ax.legend(handles=style_handles, loc="lower left", fontsize=12,
              markerfirst=False, handlelength=1.8)
    ax.tick_params(labelbottom=False)

    # Residual = ratio (with sampler / without) at each z, with HT shot noise.
    ax_res.axhline(1.0, color="k", lw=0.6, alpha=0.5)
    ax_res.axhspan(0.9, 1.1, color="0.85", alpha=0.6, lw=0)
    for i in range(len(z_actual)):
        col = z_colors[i]
        good = (phi_with[i] > 0) & (phi_without[i] > 0)
        ratio = np.full_like(phi_with[i], np.nan)
        ratio[good] = phi_with[i][good] / phi_without[i][good]
        rerr = np.full_like(phi_with[i], np.nan)
        rerr[good] = ratio[good] * np.sqrt(
            (err_with[i][good] / phi_with[i][good]) ** 2
            + (err_without[i][good] / phi_without[i][good]) ** 2)
        # Aligned with the top-panel histogram bin centres (no x-stagger),
        # so the eye can trace the ratio directly to the corresponding bar.
        ax_res.errorbar(cen[good], ratio[good], yerr=rerr[good],
                        fmt="o", ms=3.2, color=col, lw=0,
                        elinewidth=1.0, capsize=1.4, ecolor=col)
    ax_res.set_ylim(0.6, 1.4)
    ax_res.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
    ax_res.set_ylabel("ratio")
    # Cap the BHMF x-axis (shared by ax + ax_res) at log M_BH = 11 by default —
    # above that the subsample is shot-noise-limited. Override with
    # BAQARO_APPENDIX_SAMPLER_XMAX.
    _xmax = float(os.environ.get("BAQARO_APPENDIX_SAMPLER_XMAX", "11"))
    ax_res.set_xlim(bins[0], _xmax)


# ===========================================================================
# Assemble the figure
# ===========================================================================
def main():
    """Assemble and save the sampler-validation appendix figure."""
    print("Loading transfer table ...")
    sampler = load_3d_sampler(SAMPLER_NPZ)

    fig = plt.figure(figsize=(12.5, 4.8))
    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18, width_ratios=[1, 1.35])

    inner_a = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0],
                                               hspace=0.06)
    ax_a = [fig.add_subplot(inner_a[i]) for i in range(3)]
    for i in range(2):
        ax_a[i].tick_params(axis="x", which="both", labelbottom=False)

    inner_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                               hspace=0.05, height_ratios=[3, 1])
    ax_b = fig.add_subplot(inner_b[0])
    ax_b_res = fig.add_subplot(inner_b[1], sharex=ax_b)

    plot_distribution_fidelity(ax_a, sampler)
    plot_real_evolution_bhmf(ax_b, ax_b_res)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.96)

    save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
    maybe_show()


if __name__ == "__main__":
    main()
