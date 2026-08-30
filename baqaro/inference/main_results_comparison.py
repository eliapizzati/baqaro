"""Multi-run posterior-predictive comparison.

Overlay several MCMC chains (different likelihood combinations) on the SAME
QLF / BHMF / CERDF / correlation posterior-predictive panels, one colour per
run. Sibling of ``main_results_corner_comparison.py`` (which overlays the
corner posteriors); both share ``RUNS`` + the model identifier via
``comparison_config.py`` so colours/labels match across every comparison figure.

For each run we load its chain, draw ``NUM_SAMPLES`` posterior samples, push
them through the emulator(s), and draw the **median + 16/84 band** in the run's
colour. Observational data is drawn once per panel (black). Per-sample spaghetti
is intentionally omitted (too busy with several runs overlaid).

Env
---
  BAQARO_PLOT_FLAGS=qlf,bhmf,cerdf,corr   which figures (default: all)
  BAQARO_NUM_SAMPLES=100                  posterior draws per run
  BAQARO_MCMC_NOTES=minL05                compare a re-fit set (preserves baseline)
  BAQARO_SIM / BAQARO_MAX_SNAP / BAQARO_FOLD_SUBHALO_MASS / BAQARO_MERGER_DELAY_MODE
  BAQARO_NOTES_FILE_EMULATION / BAQARO_SUBSET_TAG   (model identifier; see comparison_config)
  BAQARO_HEADLESS=1                       skip plt.show(), save PDFs
"""
import os
import numpy as np
import matplotlib
if os.environ.get("BAQARO_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from baqaro.emulation.loading_helpers import load_emulators
from baqaro.plotting_common.plot_config import z_scalar_mapper, fig_ext
from baqaro.inference.comparison_config import (
    RUNS, name_file, short_name, path_out, source_dir, notes_suffix, load_flat_samples,
    comparison_plots_dir, active_likelihoods,
)


# ==========================================
# CONFIG
# ==========================================
_plot_env = os.environ.get("BAQARO_PLOT_FLAGS", "").strip()
if _plot_env:
    _en = {s.strip() for s in _plot_env.split(",") if s.strip()}
    PLOT_FLAGS = {k: (k in _en) for k in ("qlf", "bhmf", "cerdf", "corr")}
else:
    PLOT_FLAGS = {"qlf": True, "bhmf": True, "cerdf": True, "corr": True}

NUM_SAMPLES = int(os.environ.get("BAQARO_NUM_SAMPLES", "100"))
HEADLESS = os.environ.get("BAQARO_HEADLESS", "0") == "1"
rng = np.random.default_rng(12345)


# ==========================================
# LOAD EMULATORS + PER-RUN POSTERIOR PREDICTIONS
# ==========================================
emulator_flags = {
    "qlf": PLOT_FLAGS["qlf"],
    "bhmf": PLOT_FLAGS["bhmf"],
    "cerdf": PLOT_FLAGS["cerdf"],
    "qhmf": PLOT_FLAGS["corr"],
}
emulators = load_emulators(path_out, name_file, emulator_flags)

runs_data = []
for run in RUNS:
    sub = load_flat_samples(run, n_draw=NUM_SAMPLES, rng=rng)
    if sub is None:
        print(f"WARNING: no chain for '{run['label']}' ({active_likelihoods(run)}"
              f"{('_' + run['notes']) if run.get('notes') else ''}) — skipping")
        continue
    preds = {}
    if PLOT_FLAGS["qlf"]:
        preds["qlf"] = emulators["qlf"].predict_mean_only(sub)
    if PLOT_FLAGS["bhmf"]:
        preds["bhmf"] = emulators["bhmf"].predict_mean_only(sub)
    if PLOT_FLAGS["cerdf"]:
        preds["cerdf"] = emulators["cerdf"].predict_mean_only(sub)
    if PLOT_FLAGS["corr"]:
        preds["qhmf"] = emulators["qhmf"].predict_mean_only(sub)
    runs_data.append({"label": run["label"], "color": run["color"], "preds": preds})
    print(f"  loaded '{run['label']}': {sub.shape[0]} posterior draws")

if not runs_data:
    raise RuntimeError("No MCMC chains found for any run. Check BAQARO_MCMC_NOTES / model env vars.")


# ==========================================
# SHARED HELPERS
# ==========================================
def _run_legend(fig, ncol=None):
    handles = [Line2D([], [], color=r["color"], lw=2.5, label=r["label"]) for r in runs_data]
    handles.append(Line2D([], [], color="black", marker="o", ls="", label="Data"))
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles),
               fontsize=10, frameon=True)


def _median_band(stack):
    """median, 16, 84 over axis 0 (the posterior-sample axis)."""
    return (np.median(stack, axis=0),
            np.percentile(stack, 16, axis=0),
            np.percentile(stack, 84, axis=0))


_saved = []


def _finalize(fig, kind):
    if HEADLESS or os.environ.get("BAQARO_SAVE_FIGS", "0") == "1":
        d = comparison_plots_dir()
        p = os.path.join(d, f"comparison_{kind}_{short_name}{notes_suffix()}.{fig_ext()}")
        fig.savefig(p, bbox_inches="tight")
        _saved.append(p)
        print(f"saved: {p}")


def _matched_idx(emul_z, z, tol=0.5):
    i = int(np.argmin(np.abs(emul_z - z)))
    return i if abs(emul_z[i] - z) < tol else None


# ==========================================
# QLF
# ==========================================
if PLOT_FLAGS["qlf"]:
    from baqaro.obs_data.qlf_obs_data import (
        data_qlf_global_raw, data_qlf_global_sys_err_binned,
    )
    print("\n--- QLF comparison ---")
    emu = emulators["qlf"]
    emul_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    log_bins = emu.axis_data["log_bins"]
    redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    nrows, ncols = 2, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.75 * ncols, 3.4 * nrows),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    for i, z in enumerate(redshift_keys):
        ax = axs[i]
        z_str = f"{z:.1f}"
        if z_str in data_qlf_global_raw:
            o = data_qlf_global_raw[z_str]
            ax.errorbar(o.x, o.data, yerr=o.err, fmt='o', color='black', ms=4,
                        capsize=2, elinewidth=0.8, alpha=0.25, zorder=0)
        if z_str in data_qlf_global_sys_err_binned:
            ob = data_qlf_global_sys_err_binned[z_str]
            ax.errorbar(ob.x, ob.data, yerr=ob.err, fmt='s', color='black', ms=6,
                        capsize=3, elinewidth=1.5, alpha=0.85, zorder=11)
        ei = _matched_idx(emul_z, z)
        if ei is not None:
            for r in runs_data:
                med, lo, hi = _median_band(r["preds"]["qlf"][:, ei, :])
                ax.plot(log_bins, med, color=r["color"], lw=2.0, zorder=5)
                ax.fill_between(log_bins, lo, hi, color=r["color"], alpha=0.18, zorder=2)
        ax.text(0.95, 0.95, f"z = {z:.1f}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
        if i % ncols == 0:
            ax.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]")
        else:
            ax.tick_params(labelleft=False)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\log_{10}$ L$_\mathrm{bol}$ [erg/s]")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_xlim(43.5, 48.5)
        ax.set_ylim(-9.8, -2.5)
    _run_legend(fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.10, hspace=0.05, wspace=0.05)
    _finalize(fig, "qlf")


# ==========================================
# BHMF
# ==========================================
if PLOT_FLAGS["bhmf"]:
    from baqaro.obs_data.abhmf_obs_data import data_bhmf_global
    try:
        from baqaro.plotting_common.plot_config import bhmf_obs_marker
    except Exception:
        bhmf_obs_marker = lambda label: "s"
    print("\n--- BHMF comparison ---")
    emu = emulators["bhmf"]
    emul_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    log_bins = emu.axis_data["log_bins"]
    redshift_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    nrows, ncols = 2, 4
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.75 * ncols, 3.4 * nrows),
                            sharex=True, sharey=True)
    axs = axs.flatten()
    for i, z in enumerate(redshift_keys):
        ax = axs[i]
        # obs: fuzzy match by z parsed from label (tol 0.3)
        for obs in data_bhmf_global.values():
            try:
                oz = float(obs.label.split("z=")[-1])
            except (ValueError, IndexError):
                continue
            if abs(oz - z) <= 0.3:
                ax.errorbar(obs.x, obs.data, yerr=obs.errs, fmt=bhmf_obs_marker(obs.label),
                            color="black", ms=5, capsize=2, elinewidth=1.0, fillstyle="none",
                            ls="none", alpha=0.8, zorder=11)
        ei = _matched_idx(emul_z, z)
        if ei is not None:
            for r in runs_data:
                med, lo, hi = _median_band(r["preds"]["bhmf"][:, ei, :])
                ax.plot(log_bins, med, color=r["color"], lw=2.0, zorder=5)
                ax.fill_between(log_bins, lo, hi, color=r["color"], alpha=0.18, zorder=2)
        ax.text(0.95, 0.95, f"z = {z:.1f}", transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
        if i % ncols == 0:
            ax.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]")
        else:
            ax.tick_params(labelleft=False)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$\log_{10}$ M$_\mathrm{BH}$ [M$_\odot$]")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_xlim(6.5, 11.0)
        ax.set_ylim(-10.5, -1.5)
    _run_legend(fig)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.10, hspace=0.05, wspace=0.05)
    _finalize(fig, "bhmf")


# ==========================================
# CERDF
# ==========================================
if PLOT_FLAGS["cerdf"]:
    from qhtools.utils import my_utils
    import qhtools.utils.natconst as nc
    from baqaro.obs_data.qso_obs_data_setup import (
        redshifts_data, logL_Bols_data, logM_BHs_data,
    )
    print("\n--- CERDF comparison ---")
    emu = emulators["cerdf"]
    emul_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    log_bins_cerdf = emu.axis_data["log_bins"]
    log_L_thresholds = emu.axis_data["log_L_threshold"]

    redshift_keys = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    logL_bins = np.linspace(45.5, 47.5, 5)
    logL_bin_centers = 0.5 * (logL_bins[1:] + logL_bins[:-1])
    n_L = len(logL_bin_centers)
    n_z = len(redshift_keys)
    log_etas_data = my_utils.to_solar(logL_Bols_data) - logM_BHs_data - nc.log_csi
    bins_data = np.linspace(-3, 1.5, 25)
    bins_data_c = 0.5 * (bins_data[1:] + bins_data[:-1])
    bin_w = np.abs(np.diff(log_bins_cerdf[:2])[0])
    sigma_bins = 0.3 / bin_w

    def _cerdf_pdfs(pred_cerdf, zidx, lmin_idx, lmax_idx):
        """Per-sample normalized+convolved differential ERDF PDFs."""
        out = []
        for j in range(pred_cerdf.shape[0]):
            diff = np.clip(10 ** pred_cerdf[j, zidx, lmin_idx, :] - 10 ** pred_cerdf[j, zidx, lmax_idx, :], 0.0, None)
            integ = np.trapezoid(diff, log_bins_cerdf)
            if integ > 0:
                pdf = gaussian_filter1d(diff / integ, sigma_bins, mode="constant", cval=0.0)
                pdf /= np.trapezoid(pdf, log_bins_cerdf)
            else:
                pdf = np.zeros_like(diff)
            out.append(pdf)
        return np.asarray(out)

    fig, axes = plt.subplots(n_z, n_L, figsize=(3.4 * n_L, 3.0 * n_z),
                             sharex=True, sharey=True, squeeze=False)
    for iz, z in enumerate(redshift_keys):
        zi = int(np.abs(emul_z - z).argmin())
        for iL in range(n_L):
            ax = axes[iz, iL]
            lmin_v, lmax_v = logL_bins[iL], logL_bins[iL + 1]
            lmin_i = int(np.abs(log_L_thresholds - lmin_v).argmin())
            lmax_i = int(np.abs(log_L_thresholds - lmax_v).argmin())
            # obs histogram once (black)
            m = ((redshifts_data < z + 0.5) & (redshifts_data > z - 0.5)
                 & (logL_Bols_data > lmin_v) & (logL_Bols_data < lmax_v))
            if np.sum(m) > 3:
                c, _ = np.histogram(log_etas_data[m], bins=bins_data, density=True)
                ax.step(bins_data_c, c, where="mid", lw=2.0, alpha=0.8, color="black", zorder=10)
                # Observed median Eddington ratio (dotted black).
                ax.axvline(np.median(log_etas_data[m]), ls=":", lw=1.8, color="black",
                           alpha=0.85, zorder=11)
            for r in runs_data:
                pdfs = _cerdf_pdfs(r["preds"]["cerdf"], zi, lmin_i, lmax_i)
                valid = np.any(pdfs > 0, axis=1)
                if np.sum(valid) == 0:
                    continue
                med, lo, hi = _median_band(pdfs[valid])
                ax.plot(log_bins_cerdf, med, color=r["color"], lw=2.0, zorder=5)
                ax.fill_between(log_bins_cerdf, lo, hi, color=r["color"], alpha=0.15, zorder=2)
                # Model median Eddington ratio for this run (dashed, run colour):
                # invert the CDF of the median PDF at the 0.5 quantile.
                cdf = np.cumsum(med[:-1] * np.diff(log_bins_cerdf))
                if cdf[-1] > 0:
                    cdf_norm = np.concatenate(([0.0], cdf / cdf[-1]))
                    ax.axvline(np.interp(0.5, cdf_norm, log_bins_cerdf), ls="--", lw=1.6,
                               color=r["color"], alpha=0.85, zorder=6)
            if iz == 0:
                ax.set_title(rf"$\log L=[{lmin_v:.1f},{lmax_v:.1f}]$", fontsize=9)
            ax.text(0.95, 0.95, f"z={z:.1f}", transform=ax.transAxes, fontsize=9,
                    fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            if iL == 0:
                ax.set_ylabel("PDF")
            else:
                ax.tick_params(labelleft=False)
            if iz == n_z - 1:
                ax.set_xlabel(r"$\log_{10}\,\lambda_{\rm Edd}$")
            else:
                ax.tick_params(labelbottom=False)
            ax.set_xlim(-3, 1.5)
            ax.set_ylim(0, 2.0)
    _run_legend(fig)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.06, hspace=0.05, wspace=0.05)
    _finalize(fig, "cerdf")


# ==========================================
# CORRELATION
# ==========================================
if PLOT_FLAGS["corr"]:
    from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle, get_corr_from_triangle_cross
    from qhtools.clustering.projected_correlation_functions import get_projected_wp, get_volume_averaged_xi
    # Corr datasets MUST mirror main_mcmc.py, or this figure overlays different
    # observations than the chains were fit to.
    #
    #
    # FIT at z=2.5 = data_ef_ext_restricted (Table 3, masked to EF15's own
    # 4 < rp < 25 h^-1 Mpc fit range, 10 bins).
    # PLOTTED  = data_ef_ext (Table 3, all 18 bins) — show the data, not just the
    # part we chose to fit; the fitted window is shaded in the panel. Safe because
    # the restricted set is a pure row-mask of the full one and they share pimax and
    # log_L_threshold, so the halo-model inputs are identical.
    from baqaro.obs_data import corr_obs_data as _corr_mod
    from baqaro.obs_data.corr_obs_data import data_eiger
    from baqaro.inference.likelihoods_and_priors import precompute_corr_inputs
    print("\n--- Correlation comparison ---")
    emu = emulators["qhmf"]

    _z25_key = (os.environ.get("BAQARO_CORR_Z25", "data_ef_ext_restricted").strip()
                or "data_ef_ext_restricted")
    # z=4 default = ALL-FIELDS Shen+07 with x2-inflated errors (matching
    # main_mcmc / main_results_analysis / plotting_clustering).
    # Revert with BAQARO_CORR_Z4=data_shen_highz.
    _z4_key = (os.environ.get("BAQARO_CORR_Z4", "data_shen_highz_allfields_err2").strip()
               or "data_shen_highz_allfields_err2")
    _z6_key = os.environ.get("BAQARO_CORR_Z6", "data_aspire_cov").strip() or "data_aspire_cov"
    _z25_plot_key = os.environ.get("BAQARO_CORR_Z25_PLOT", "").strip() or (
        "data_ef_ext" if _z25_key == "data_ef_ext_restricted" else _z25_key)
    _corr_z25 = getattr(_corr_mod, _z25_key)
    _corr_z25_plot = getattr(_corr_mod, _z25_plot_key)
    _corr_z4 = getattr(_corr_mod, _z4_key)
    _corr_z6 = getattr(_corr_mod, _z6_key)
    print(f"[corr] fit datasets: z=2.5={_z25_key}  z=4={_z4_key}  z=6={_z6_key}")
    if _z25_plot_key != _z25_key:
        print(f"[corr] z=2.5 PLOTTED as {_z25_plot_key} ({len(_corr_z25_plot.x)} bins); "
              f"FIT used {_z25_key} ({len(_corr_z25.x)} bins)")

    _corr_z25_plot.redshift = 2.5; _corr_z25_plot.corr_type = "auto"
    _corr_z25_plot.logM_min = 11.5; _corr_z25_plot.logM_max = 14.5
    _corr_z4.redshift = 4.0; _corr_z4.corr_type = "auto"
    _corr_z4.logM_min = 11.5; _corr_z4.logM_max = 14.5
    # z~6 cross-correlation: the selected ASPIRE variant drives the model (z=6.1
    # cached halo-model inputs, emulator's nearest slice is z=6.14, and it's what
    # the MCMC fit); EIGER is overlaid as a legacy second measurement.
    _corr_z6.redshift = 6.1; _corr_z6.corr_type = "cross"
    _corr_z6.logM_min = 10.5; _corr_z6.logM_max = 14.0
    data_eiger.redshift = 6.1; data_eiger.corr_type = "cross"
    data_eiger.logM_min = 10.5; data_eiger.logM_max = 14.0
    corr_datasets = {2.5: _corr_z25_plot, 4.0: _corr_z4, 6.1: _corr_z6}
    # Extra data-only overlays per primary dataset (same panel, distinct marker).
    extra_obs = {id(_corr_z6): [data_eiger]}
    # rp window actually fitted at z=2.5, so the panel can shade it.
    fitted_rp_range = {id(_corr_z25_plot): (float(_corr_z25.x.min()),
                                            float(_corr_z25.x.max()))} \
        if _z25_plot_key != _z25_key else {}

    target_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    datas_corr = []
    for _i, tz in enumerate(target_z):
        best = min(corr_datasets, key=lambda z: abs(z - tz))
        # Uniqueness guard (back-ported from plotting_paper/plotting_results_comparison):
        # a dataset binds to exactly ONE panel -- the emulator slice CLOSEST to it.
        # Otherwise a dataset with two nearby slices (z=4 sits between emulator
        # z=3.937, gap 0.063, and z=3.534, gap 0.466) is drawn twice AND the panel
        # cap silently drops a real one (z=6.1).
        _closest = int(np.argmin(np.abs(target_z - best)))
        datas_corr.append(corr_datasets[best]
                          if (abs(best - tz) < 0.5 and _closest == _i) else None)
    print("Precomputing correlation inputs...")
    pre = precompute_corr_inputs(datas_corr)
    log_L_thr = emu.axis_data["log_L_threshold"]
    log_mbins_qhmf = emu.axis_data["log_bins"]

    active = [i for i, d in enumerate(datas_corr) if d is not None]
    # Emulator z axis is DESCENDING; the paper figures order panels low-z -> high-z.
    active.sort(key=lambda i: datas_corr[i].redshift)
    if active:
        ncols = min(len(active), 3)
        nrows = int(np.ceil(len(active) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.75 * ncols, 3.5 * nrows),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        for panel, iz in enumerate(active):
            ax = axes[panel]
            dc = datas_corr[iz]; inp = pre[iz]
            ctype = getattr(dc, "corr_type", "auto")
            # Shade the rp window actually FITTED, when the panel shows more data
            # than the likelihood used (z=2.5: EF15's own 4 < rp < 25 h^-1 Mpc).
            # Without this the figure implies every plotted point constrained the
            # model, which is not true.
            _fit_lo_hi = fitted_rp_range.get(id(dc))
            if _fit_lo_hi is not None:
                ax.axvspan(_fit_lo_hi[0], _fit_lo_hi[1], color="grey", alpha=0.12,
                           zorder=0, label="fitted range")
            ax.errorbar(dc.x, dc.data, yerr=dc.err, fmt="o", color="black", ms=6,
                        capsize=3, elinewidth=1.5, alpha=0.85, zorder=10,
                        label=getattr(dc, "label", "Data"))
            # Extra measurements overlaid in the same panel (e.g. ASPIRE next to
            # EIGER): distinct open markers, plotted as data only.
            _extra_markers = ["s", "D", "^"]
            _has_extra = False
            for k, extra in enumerate(extra_obs.get(id(dc), [])):
                _has_extra = True
                ax.errorbar(extra.x, extra.data, yerr=extra.err, fmt=_extra_markers[k % 3],
                            color="black", mfc="none", ms=6, capsize=3, elinewidth=1.2,
                            alpha=0.85, zorder=10, label=getattr(extra, "label", "Data2"))
            if _has_extra:
                ax.legend(loc="lower left", fontsize=8, frameon=True)
            log_m_axis = inp["log_m_axis"]; rbins = inp["rbins"]; rpbins = inp["rpbins"]
            mf_fit = inp["mf_fit"]; tri = inp["triangle_fit"]
            thr_idx = np.abs(log_L_thr - inp["log_L_threshold"]).argmin()
            for r in runs_data:
                pred_qhmf = r["preds"]["qhmf"]
                ys = []
                for j in range(pred_qhmf.shape[0]):
                    sl = pred_qhmf[j, iz, thr_idx, :]
                    if np.all(sl <= -9.5):
                        continue
                    qi = np.interp(log_m_axis, log_mbins_qhmf, sl, left=-10.0, right=-10.0)
                    if ctype == "auto":
                        lr, xi = get_corr_from_triangle(
                            log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                            triangle=tri, log_m_axis=log_m_axis, qhmf=10 ** qi)
                        wp = get_projected_wp(rpbins, xi, 10 ** lr, pimax=dc.pimax)
                        xx, yy = rpbins, wp / rpbins
                    else:
                        qg = np.copy(mf_fit) / 5.0; qg[log_m_axis < 10.7] = 0.0
                        lr, xic = get_corr_from_triangle_cross(
                            log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                            triangle=tri, log_m_axis=log_m_axis, qhmf1=10 ** qi, qhmf2=qg)
                        xv = get_volume_averaged_xi(rpbins, xic, 10 ** lr, pimax=dc.pimax)
                        xx = 10 ** (0.5 * (np.log10(rpbins[:-1]) + np.log10(rpbins[1:]))); yy = xv
                    ys.append(yy)
                if ys:
                    ys = np.asarray(ys)
                    med, lo, hi = _median_band(ys)
                    ax.plot(xx, med, color=r["color"], lw=2.0, zorder=5)
                    ax.fill_between(xx, lo, hi, color=r["color"], alpha=0.18, zorder=2)
            ax.text(0.95, 0.95, f"z={dc.redshift:.1f} ({ctype})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.09, 95.0); ax.set_ylim(1.1e-1, 1e3)
            if panel % ncols == 0:
                ax.set_ylabel(r"$w_p(r_p)/r_p$  /  $\bar\xi$")
            else:
                ax.tick_params(labelleft=False)
            if panel // ncols == nrows - 1:
                ax.set_xlabel(r"$r_p$ [Mpc/h]")
            else:
                ax.tick_params(labelbottom=False)
        for j in range(len(active), len(axes)):
            axes[j].axis("off")
        _run_legend(fig)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.12, hspace=0.05, wspace=0.05)
        _finalize(fig, "corr")
    else:
        print("No correlation data matched. Skipping corr figure.")


if _saved:
    print("\nSaved figures:")
    for p in _saved:
        print(f"  {p}")
if not HEADLESS:
    plt.show()
