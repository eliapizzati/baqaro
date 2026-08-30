"""
APPENDIX FIGURE: emulator held-out training validation (95% train / 5% test).
==============================================================================

Honest train/test validation of the GP+PCA emulators on a held-out
fraction of the production training set. Replaces the older
`plotting_appendix_emulator_cv.py` (which leaned on a pre-computed CV
report) and `plotting_appendix_emulator_vs_real.py` (which compared the
production emulator to a fresh forward run at the fiducial, conflating
emulator error with forward-model noise from a single new realisation).

What the figure shows
---------------------
For each of QLF, cERDF, QHMF (the three observables fed to the
likelihood) we:

  1. Load the production training set
     (`training_data_emulation_..._clean_z0_g6.21_K22_final_...hdf5`, via ``fiducial_data``).
  2. Apply the production "bad sim" filter from `main_emulation.py`:
     drop any sim with >80% bins below `physical_floor` in any of the
     four tracked stats (per-stat OR, deduplicated across stats).
  3. Apply Savitzky-Golay smoothing per-quantity with the
     ``SG_PARAMS_BY_QUANTITY`` settings the production emulator uses.
  4. Randomly split the remaining good runs into 95% TRAIN / 5% TEST
     (fixed seed for reproducibility).
  5. Fit a fresh PCA+GP emulator on the train slice (n_components
     chosen to capture >99.8% of variance, matching `main_emulation.py`).
  6. Predict on the TEST slice and emit TWO independent figures:
       • ``appendix_emulator_curves.pdf`` — actual curves at one
         (z, L_thr) slice per stat: a handful of test samples
         overplotted as true (solid) + emulator-predicted (dashed)
         in matching colours.
       • ``appendix_emulator_scatter.pdf`` — predicted vs. true
         hexbin density per stat, flattening all valid above-floor
         bin pairs, with 1:1 line and ±0.1 dex reference band.
         RMSE + median |error| inset per panel.

Both figures are emitted in a single invocation. The slow GP fits are
cached to disk (``{path_out}/emulators/appendix_validation_<name>.npz``)
so re-running the script for plotting tweaks takes seconds instead of
hours. Set ``BAQARO_APPENDIX_VALIDATION_REFIT=1`` to force a re-fit.

Runtime: 5-10 h overnight at N≈5000 train points, `BAQARO_GP_N_RESTARTS=0`
(kernel-init only — measured to give the same accuracy as nr=3).

Pinned through ``fiducial_data`` so the figure tracks whatever the rest
of the paper is using; override the training HDF5 with
``BAQARO_APPENDIX_VALIDATION_TRAINING_FILE``.

Run (paper export):
    env BAQARO_SAVE_FIGS=1 BAQARO_HEADLESS=1 BAQARO_GP_N_RESTARTS=0 \
        python -m baqaro.plotting_paper.plotting_appendix_emulator_validation
"""

import os

import numpy as np

from baqaro.plotting_paper import fiducial_data as fd
from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import plt
from baqaro.plotting_common.plot_config import save_fig, maybe_show

from baqaro.emulation.loading_helpers import load_training_data
from baqaro.emulation.sg_smoothing import (
    smooth_function_rows, SG_PARAMS_BY_QUANTITY,
)
from baqaro.emulation.emulation_core_functions import (
    GeneralEmulatorPCA, GeneralEmulatorGP,
)

# nr=0 (kernel-init start only) gives accuracy indistinguishable from nr=3 on
# this training set and is ~5-7x faster.
os.environ.setdefault("BAQARO_GP_N_RESTARTS", "0")


name_fig_scatter = "appendix_emulator_scatter"
name_fig_curves  = "appendix_emulator_curves"

# Cache the (true, pred) test-set arrays per stat so the slow GP fits are
# only run once; subsequent script invocations (e.g. plotting tweaks) reuse
# the cache and re-render in seconds. Set BAQARO_APPENDIX_VALIDATION_REFIT=1
# to force a re-fit even when the cache exists.
def _cache_path():
    cache_dir = os.path.join(str(fd.path_out), "emulators")
    return os.path.join(cache_dir, f"appendix_validation_{fd.name_file_mcmc}.npz")


def _sg_cache_path():
    """Cache for the SG-smoothed (qlf, cerdf, qhmf) arrays — ~3 GB total, but
    keyed on the training file so it survives across renders. Re-renders
    (~minutes -> seconds) avoid the slow `smooth_function_rows` pass."""
    cache_dir = os.path.join(str(fd.path_out), "emulators")
    return os.path.join(cache_dir,
                         f"appendix_validation_sg_{fd.name_file_mcmc}.npz")


def _sg_cache_fingerprint(training_file):
    """A small fingerprint identifying the inputs that determine the cache
    contents. Anything in here changing -> cache miss."""
    st = os.stat(training_file)
    sg = tuple((q, *SG_PARAMS_BY_QUANTITY[q]) for q in QUANTS)
    return np.array([
        training_file,
        int(st.st_size),
        int(st.st_mtime_ns),
        repr(sg),
        f"floor={PHYSICAL_FLOOR},pca={PCA_FLOOR}",
    ], dtype=object)

QUANTS = ["qlf", "cerdf", "qhmf"]
LABELS = {"qlf": "QLF", "cerdf": "cERDF", "qhmf": "QHMF"}
KEYS   = {"qlf": "log_qlfs", "cerdf": "log_cerdfs", "qhmf": "log_qhmfs"}
# Bottom-row axis label = the actual log10 quantity being scattered.
AXLAB  = {"qlf":   r"$\log_{10}\Phi_{\rm QLF}$ [Mpc$^{-3}$ dex$^{-1}$]",
          "cerdf": r"$\log_{10}\,p(\log\lambda_{\rm Edd}|L_{\rm bol},z)$",
          "qhmf":  r"$\log_{10}\Phi_{\rm QHMF}$ [Mpc$^{-3}$ dex$^{-1}$]"}
# Top-row x-axis label per stat — the observable axis the curves are drawn over.
XLAB_CURVE = {"qlf":   r"$\log_{10}(L_{\rm bol}/[{\rm erg\,s^{-1}}])$",
              "cerdf": r"$\log_{10}\lambda_{\rm Edd}$",
              "qhmf":  r"$\log_{10}(M_{\rm halo}/M_\odot)$"}

PHYSICAL_FLOOR = -9.5
PCA_FLOOR      = -10.0
PCA_MAX_COMP   = 15
PCA_VAR_TARGET = 0.998
TEST_FRAC      = 0.05    # 95 / 5 split
SPLIT_SEED     = 0

# Per-quantity slice used in the top-row curve overlays.
# QLF has no L_thr axis; CERDF and QHMF do.
SLICE_Z       = 2.0       # target redshift for the top-row slice
SLICE_LOG_LTH = 46.0      # target log10 L_threshold [erg/s] (CERDF, QHMF)
N_CURVE_SAMPLES = 25      # number of test samples to overplot per stat

HEXBIN_GRIDSIZE = 70
HEXBIN_CMAP     = "viridis"


def _resolve_training_file():
    """Find the training HDF5 matching the production fiducial emulator."""
    override = os.environ.get("BAQARO_APPENDIX_VALIDATION_TRAINING_FILE", "").strip()
    if override:
        return override
    nfm_train = fd.name_file_mcmc.replace("_smooth", "")
    candidate = os.path.join(str(fd.path_out), "training",
                              f"training_data_emulation_{nfm_train}.hdf5")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Could not resolve training file for {fd.name_file_mcmc}.\n"
            f"  Tried: {candidate}\n"
            "  Set BAQARO_APPENDIX_VALIDATION_TRAINING_FILE to override."
        )
    return candidate


def _production_bad_sim_filter(data):
    """Mirror the >80%-empty filter from `main_emulation.py` (lines 191-247).

    A simulation is dropped if ANY tracked stat (QLF / BHMF / CERDF / QHMF)
    has more than 80% of its bins at/below `physical_floor + 0.1`. The OR
    across stats is dedupliated. Returns a boolean keep-mask.
    """
    n_runs = data["params"].shape[0]
    bad = set()
    for key in ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs"):
        arr = np.asarray(data[key])
        global_floor_frac = float(np.mean(arr <= PHYSICAL_FLOOR + 0.1))
        # main_emulation.py skips a stat from filtering if it's >95% globally
        # floored (i.e. unmeaningful) — replicate that.
        if global_floor_frac > 0.95:
            continue
        emptiness = np.mean(arr.reshape(n_runs, -1) <= PHYSICAL_FLOOR + 0.1, axis=1)
        bad.update(np.where(emptiness > 0.80)[0].tolist())
    keep = np.ones(n_runs, dtype=bool)
    if bad:
        keep[np.array(sorted(bad))] = False
    print(f"  production-style filter: dropped {(~keep).sum()} / {n_runs} sims "
          f"(>80% floored in some stat)")
    return keep


def _select_n_components(stat_arr):
    """Smallest n s.t. cumulative explained variance >= PCA_VAR_TARGET."""
    flat = stat_arr.reshape(stat_arr.shape[0], -1)
    flat = np.nan_to_num(flat, nan=PCA_FLOOR, neginf=PCA_FLOOR) - PCA_FLOOR
    from sklearn.decomposition import PCA as _PCA
    n_try = min(PCA_MAX_COMP, flat.shape[0] - 1, flat.shape[1] - 1)
    pca = _PCA(n_components=n_try)
    pca.fit(flat)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_opt = int(np.argmax(cum >= PCA_VAR_TARGET)) + 1
    if cum[-1] < PCA_VAR_TARGET:
        n_opt = n_try
    return max(2, min(n_opt, n_try))


def _fit_holdout(stat_arr, params, train_idx, test_idx, label):
    """Train PCA+GP on (stat_arr[train_idx], params[train_idx]); predict on test."""
    n_components = _select_n_components(stat_arr[train_idx])
    print(f"  [{label}] n_components = {n_components}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")

    pca = GeneralEmulatorPCA(n_components=n_components,
                              floor_value=PCA_FLOOR)
    Y_train_weights = pca.fit_transform(stat_arr[train_idx])

    gp = GeneralEmulatorGP(pca_model=pca, verbose=False, kernel_type="matern32")
    gp.fit(params[train_idx], Y_train_weights, normalize_params=True)

    mu, _ = gp.predict(params[test_idx])
    return mu


def _pick_sample_indices(true_arr_slice, n_pick):
    """Pick ``n_pick`` test-sample indices that span the dynamic range of
    ``true_arr_slice`` (1D summary per sample = mean above-floor value)."""
    summary = []
    for i in range(true_arr_slice.shape[0]):
        v = true_arr_slice[i]
        m = v > PHYSICAL_FLOOR
        summary.append(float(v[m].mean()) if m.any() else float("nan"))
    summary = np.asarray(summary)
    finite = np.isfinite(summary)
    if finite.sum() < n_pick:
        return np.where(finite)[0]
    valid_idx = np.where(finite)[0]
    order = np.argsort(summary[valid_idx])
    # Pick evenly along the sorted range.
    picks = np.linspace(0, order.size - 1, n_pick).round().astype(int)
    return valid_idx[order[picks]]


def _flatten_for_scatter(true_arr, pred_arr):
    """Flatten + keep only bins with real signal on BOTH sides — FOR THE HEXBIN.

    Both-real is the right mask for the *density plot* (a floored value sits at
    PHYSICAL_FLOOR, far off the panel's axes, and would just smear the colour
    scale). It is not the mask for the error metric — use `_error_stats`.
    """
    t = true_arr.reshape(-1)
    p = pred_arr.reshape(-1)
    mask = (t > PHYSICAL_FLOOR) & (p > PHYSICAL_FLOOR)
    return t[mask], p[mask]


def _error_stats(true_arr, pred_arr):
    """Emulator error metrics.

    Bins floored on EITHER side are discordant: the emulator either erased a
    real signal (true real, pred floored) or invented one (true floored, pred
    real). These are reported separately from the concordant-bin accuracy.

    Reporting choice. We do NOT simply fold them into the RMSE, because the
    "error" of such a bin is |floor - real|, which depends on where PHYSICAL_FLOOR
    happens to sit — a floor-dependent, and therefore arbitrary, headline number.
    Instead we report the two things that are each well-defined:

      * rmse / mae  — over CONCORDANT bins (both sides real). This is the accuracy
        of the emulator *where it agrees the bin has signal*.
      * n_discordant / frac_discordant — the RATE at which it erases or invents
        signal. Floor-independent.

    A reader needs BOTH.

    Returns (rmse, mae, n_concordant, n_discordant, frac_discordant).
    """
    t = np.asarray(true_arr).reshape(-1)
    p = np.asarray(pred_arr).reshape(-1)
    t_real = t > PHYSICAL_FLOOR
    p_real = p > PHYSICAL_FLOOR
    concordant = t_real & p_real
    informative = t_real | p_real          # at least one side has signal
    discordant = t_real ^ p_real           # exactly one side floored
    n_info = int(informative.sum())
    n_disc = int(discordant.sum())
    err = p[concordant] - t[concordant]
    rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else float("nan")
    mae = float(np.median(np.abs(err))) if err.size else float("nan")
    frac = (n_disc / n_info) if n_info else float("nan")
    return rmse, mae, int(concordant.sum()), n_disc, frac


# ----------------------------------------------------------------------------
# Plot panels
# ----------------------------------------------------------------------------
def _plot_curve_panel(ax, ax_res, x_bins, true_curves, pred_curves,
                       slice_text, q_key, xlab):
    """Curve overlay (main) + residual (small bottom subpanel).

    true_curves / pred_curves: (n_picked, n_bins) — already sliced to the
    desired (z[, L_thr]) cell. ``x_bins`` is the observable axis. The
    residual subpanel shows (pred - true) per sample in dex with a
    ±0.1 dex reference band.
    """
    n_picked = true_curves.shape[0]
    cmap = plt.get_cmap("viridis", max(n_picked, 2))

    # --- main panel: true (solid) + emulator (dashed) ---
    for k in range(n_picked):
        col = cmap(k)
        t = true_curves[k]; p = pred_curves[k]
        mt = t > PHYSICAL_FLOOR
        mp = p > PHYSICAL_FLOOR
        if mt.any():
            ax.plot(x_bins[mt], t[mt], "-",  color=col, lw=1.2, alpha=0.85)
        if mp.any():
            ax.plot(x_bins[mp], p[mp], "--", color=col, lw=1.2, alpha=0.85)
    handles = [plt.Line2D([0], [0], color="0.2", lw=1.8, ls="-",  label="true"),
               plt.Line2D([0], [0], color="0.2", lw=1.8, ls="--", label="emulator")]
    ax.legend(handles=handles, loc="lower left", fontsize=11, frameon=True,
              handlelength=2.0)
    ax.set_ylabel(AXLAB[q_key], fontsize=13)
    ax.tick_params(axis="x", labelbottom=False)       # share x with residual
    ax.tick_params(axis="y", labelsize=11)
    # Slice info as inset text (no title — matches the scatter figure style).
    ax.text(0.97, 0.97, slice_text, transform=ax.transAxes,
            fontsize=12, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55",
                      lw=0.7, alpha=0.93))

    # --- residual subpanel: pred - true in dex ---
    ax_res.axhline(0.0, color="k", lw=0.7, alpha=0.55)
    ax_res.axhspan(-0.2, 0.2, color="0.85", alpha=0.55, lw=0)
    for k in range(n_picked):
        col = cmap(k)
        t = true_curves[k]; p = pred_curves[k]
        m = (t > PHYSICAL_FLOOR) & (p > PHYSICAL_FLOOR)
        if m.any():
            ax_res.plot(x_bins[m], p[m] - t[m], "-", color=col, lw=1.1, alpha=0.85)
    ax_res.set_xlabel(xlab, fontsize=13)
    ax_res.set_ylabel(r"$\Delta$ [dex]", fontsize=12)
    ax_res.set_ylim(-1.0, 1.0)
    ax_res.tick_params(axis="both", labelsize=10)


def _plot_scatter_panel(ax, t, p, label, n_test, norm=None, stats=None):
    """Predicted vs true hexbin. Returns the hexbin artist so a SHARED
    colorbar can be added by the caller (norm should be the shared LogNorm).

    ``t``/``p`` are the BOTH-REAL bins (they set the hexbin axes). ``stats``, when
    given, is ``_error_stats(true, pred)`` computed over every INFORMATIVE bin —
    including the floored-vs-real ones the hexbin cannot show. The quoted RMSE /
    med|Delta| come from ``stats``, NOT from the plotted subset, so the reported
    numbers include the discordant bins.
    """
    if t.size == 0:
        ax.text(0.5, 0.5, "no valid bins", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="0.4")
        return None

    lo = float(np.floor(min(t.min(), p.min())))
    hi = float(np.ceil(max(t.max(), p.max())))

    hb = ax.hexbin(t, p, gridsize=HEXBIN_GRIDSIZE, cmap=HEXBIN_CMAP,
                   norm=norm,
                   extent=(lo, hi, lo, hi),
                   mincnt=1, linewidths=0)

    xs = np.array((lo, hi))
    ax.plot(xs, xs, "-", color="white", lw=1.8, zorder=3)
    ax.plot(xs, xs, "-", color="0.15", lw=1.1, zorder=4)
    for off in (0.1, -0.1):
        ax.plot(xs, xs + off, "--", color="0.45", lw=0.8, zorder=4)

    if stats is not None:
        rmse, mae, n_conc, n_disc, frac_disc = stats
    else:   # legacy fallback: plotted subset only, no discordant rate
        err = p - t
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.median(np.abs(err)))
        n_conc, n_disc, frac_disc = t.size, 0, 0.0
    txt = (f"RMSE $= {rmse:.03f}$ dex\n"
           f"med$|\\Delta|= {mae:.03f}$ dex\n"
           f"$N_{{\\rm test}} = {n_test}$,  "
           f"$N_{{\\rm bins}} = {n_conc:,}$")
    # NB: the floored-vs-real (discordant) rate — how often the emulator erases a
    # real bin or invents signal — is still computed in `stats`/`frac_disc` and
    # printed to the log, but is intentionally NOT annotated on the panel.
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=11.5,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.55",
                      lw=0.7, alpha=0.93))

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"true {AXLAB[label]}", fontsize=13)
    ax.set_ylabel(f"emulator {AXLAB[label]}", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_aspect("equal", adjustable="box")
    return hb


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    """Render the two emulator-validation appendix figures.

    Loads the training set, refits the PCA+GP on a held-out split, and writes
    ``appendix_emulator_curves`` (predicted vs true curves at one held-out
    point) and ``appendix_emulator_scatter`` (predicted vs true across the
    whole held-out set).
    """
    training_file = _resolve_training_file()
    print(f"Loading training data: {training_file}")
    data = load_training_data(training_file,
                              physical_floor=PHYSICAL_FLOOR,
                              pca_floor=PCA_FLOOR,
                              only_done=True)

    keep_mask = _production_bad_sim_filter(data)
    params = np.asarray(data["params"])[keep_mask]
    # NB: the stat arrays' z-axis indexes `redshift_keys` (the 13 saved
    # snapshots), NOT the full 145-entry `redshifts` array.
    redshift_keys = np.asarray(data["redshift_keys"])
    log_lbins = np.asarray(data["log_lbins"])             # QLF bins
    log_bins_cerdf = np.asarray(data["log_bins_cerdf"])   # cERDF bins
    log_mbins_qhmf = np.asarray(data["log_mbins_qhmf"])   # QHMF bins
    log_L_thresholds = np.asarray(data["log_L_thresholds"])  # for CERDF / QHMF

    # Resolve slice indices for the curve-overlay row.
    iz = int(np.argmin(np.abs(redshift_keys - SLICE_Z)))
    ilth = int(np.argmin(np.abs(log_L_thresholds - SLICE_LOG_LTH)))
    print(f"  curve-overlay slice: z={redshift_keys[iz]:.2f} (idx {iz}); "
          f"log L_thr={log_L_thresholds[ilth]:.2f} (idx {ilth})")

    # SG smoothing matching the production emulator — cached because it's
    # the dominant cost on each render (~10 min for cERDF/QHMF). Cache
    # is invalidated if training-file mtime/size, SG params, or floors change.
    sg_cache_path = _sg_cache_path()
    fingerprint = _sg_cache_fingerprint(training_file)
    force_sg_refit = bool(int(os.environ.get("BAQARO_APPENDIX_VALIDATION_SG_REFIT", "0")))
    stats = {}
    if os.path.exists(sg_cache_path) and not force_sg_refit:
        sg = np.load(sg_cache_path, allow_pickle=True)
        if (sg["fingerprint"].shape == fingerprint.shape
                and (sg["fingerprint"] == fingerprint).all()):
            for q in QUANTS:
                stats[q] = np.asarray(sg[q])
            print(f"  SG cache hit -> {sg_cache_path}")
        else:
            print("  SG cache fingerprint mismatch -> regenerating")
    if not stats:
        for q in QUANTS:
            win, poly = SG_PARAMS_BY_QUANTITY[q]
            raw = np.asarray(data[KEYS[q]])[keep_mask]
            smoothed = smooth_function_rows(raw, win, poly, floor=PHYSICAL_FLOOR)
            stats[q] = smoothed
            print(f"  [{q}] SG-smoothed ({win=}, {poly=}), shape={smoothed.shape}")
        np.savez(sg_cache_path, fingerprint=fingerprint, **stats)
        print(f"  SG cache written -> {sg_cache_path}")

    n_runs = params.shape[0]
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(n_runs)
    n_test = max(1, int(np.round(TEST_FRAC * n_runs)))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    print(f"Split: train={train_idx.size}  test={test_idx.size}  "
          f"(seed={SPLIT_SEED}, test_frac={TEST_FRAC})")

    # Per-stat fit + collect predictions, with on-disk cache so the slow GP
    # fits run once (~hours) and figure re-renders are seconds.
    cache_path = _cache_path()
    force_refit = bool(int(os.environ.get("BAQARO_APPENDIX_VALIDATION_REFIT", "0")))
    fits = {}
    if os.path.exists(cache_path) and not force_refit:
        print(f"\nLoading cached fit results from {cache_path}")
        cache = np.load(cache_path)
        # Sanity-check the cache matches the current split / shapes.
        cached_test = np.asarray(cache["test_idx"])
        if cached_test.size != test_idx.size or not np.array_equal(cached_test, test_idx):
            print("  cache test_idx mismatch -> refitting")
        else:
            for q in QUANTS:
                fits[q] = {
                    "true": np.asarray(cache[f"{q}_true"]),
                    "pred": np.asarray(cache[f"{q}_pred"]),
                }
                t, p = _flatten_for_scatter(fits[q]["true"], fits[q]["pred"])
                fits[q]["t_flat"] = t
                fits[q]["p_flat"] = p
                fits[q]["stats"] = _error_stats(fits[q]["true"], fits[q]["pred"])
            print("  cache OK; skipping GP fits")
    if not fits:
        for q in QUANTS:
            print(f"\nFitting {q.upper()} (PCA+GP on train, predict on test)...")
            pred = _fit_holdout(stats[q], params, train_idx, test_idx, q)
            true = stats[q][test_idx]
            t, p = _flatten_for_scatter(true, pred)
            st = _error_stats(true, pred)
            rmse, mae, n_conc, n_disc, frac_disc = st
            # Report the accuracy AND the discordant rate — the latter is what the
            # old both-real-only metric hid entirely.
            print(f"  [{q}] RMSE = {rmse:.04f} dex, med|d| = {mae:.04f} dex "
                  f"(over {n_conc:,} concordant bins)")
            print(f"        floored-vs-real: {n_disc:,} bins = {100*frac_disc:.2f}% "
                  f"of informative bins  <-- emulator erased or invented signal; "
                  f"NOT in the RMSE above, and previously not reported at all")
            fits[q] = {"true": true, "pred": pred, "t_flat": t, "p_flat": p, "stats": st}
        # Persist
        np.savez(cache_path,
                 test_idx=test_idx,
                 **{f"{q}_true": fits[q]["true"] for q in QUANTS},
                 **{f"{q}_pred": fits[q]["pred"] for q in QUANTS})
        print(f"  cached fit results -> {cache_path}")

    # ---- FIGURE 1: curve overlays (2 rows × 3 cols; main + residual) ------
    import matplotlib.gridspec as gridspec
    fig_c = plt.figure(figsize=(15.0, 5.4))
    outer = gridspec.GridSpec(
        2, 3, figure=fig_c,
        height_ratios=[3, 1], hspace=0.04, wspace=0.24,
        left=0.05, right=0.985, bottom=0.10, top=0.985,
    )
    for col, q in enumerate(QUANTS):
        true = fits[q]["true"]
        pred = fits[q]["pred"]
        if q == "qlf":
            t_slice = true[:, iz, :]; p_slice = pred[:, iz, :]
            x_bins = log_lbins
            slice_text = f"{LABELS[q]} at $z = {redshift_keys[iz]:.2f}$"
        else:  # cerdf, qhmf
            t_slice = true[:, iz, ilth, :]; p_slice = pred[:, iz, ilth, :]
            x_bins = log_bins_cerdf if q == "cerdf" else log_mbins_qhmf
            slice_text = (f"{LABELS[q]} at $z = {redshift_keys[iz]:.2f}$\n"
                          f"$\\log\\,L_{{\\rm thr}} = "
                          f"{log_L_thresholds[ilth]:.1f}$")
        picks = _pick_sample_indices(t_slice, N_CURVE_SAMPLES)
        ax_main = fig_c.add_subplot(outer[0, col])
        ax_res  = fig_c.add_subplot(outer[1, col], sharex=ax_main)
        _plot_curve_panel(ax_main, ax_res, x_bins,
                          t_slice[picks], p_slice[picks],
                          slice_text=slice_text, q_key=q,
                          xlab=XLAB_CURVE[q])
    save_fig(fig_c, plot_config.FIGURES_DIR, name_fig_curves, force_dir=True)

    # ---- FIGURE 2: predicted-vs-true hexbin scatter (1 row × 3 panels) ----
    # Compute a shared LogNorm across all three panels so the colorbar is
    # meaningful side-by-side. We get vmin/vmax by binning ONCE in numpy,
    # collecting counts across stats, and taking the global range.
    from matplotlib.colors import LogNorm

    counts_all = []
    for q in QUANTS:
        t = fits[q]["t_flat"]; p = fits[q]["p_flat"]
        if t.size == 0:
            continue
        # Same gridding as hexbin will use, only to derive the colour limits.
        # We approximate hexbin's range with a 2D histogram on a square grid.
        lo = float(np.floor(min(t.min(), p.min())))
        hi = float(np.ceil(max(t.max(), p.max())))
        H, _, _ = np.histogram2d(
            t, p, bins=HEXBIN_GRIDSIZE,
            range=[[lo, hi], [lo, hi]],
        )
        counts_all.append(H[H > 0])
    if counts_all:
        flat = np.concatenate(counts_all)
        vmin = float(max(1.0, flat.min()))
        vmax = float(flat.max())
    else:
        vmin, vmax = 1.0, 10.0
    shared_norm = LogNorm(vmin=vmin, vmax=vmax)

    fig_s, axs_s = plt.subplots(
        1, 3, figsize=(14.5, 4.6),
        gridspec_kw={"wspace": 0.25, "left": 0.055, "right": 0.92,
                     "bottom": 0.13, "top": 0.97},
    )
    last_hb = None
    for col, q in enumerate(QUANTS):
        hb = _plot_scatter_panel(axs_s[col], fits[q]["t_flat"], fits[q]["p_flat"],
                                  label=q, n_test=test_idx.size,
                                  norm=shared_norm,
                                  stats=fits[q].get("stats"))
        if hb is not None:
            last_hb = hb

    # Single shared colorbar to the right of all three panels.
    if last_hb is not None:
        cax = fig_s.add_axes([0.935, 0.13, 0.013, 0.84])
        cbar = fig_s.colorbar(last_hb, cax=cax)
        cbar.set_label("number of test bins per hex", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    save_fig(fig_s, plot_config.FIGURES_DIR, name_fig_scatter, force_dir=True)

    maybe_show()


if __name__ == "__main__":
    main()
