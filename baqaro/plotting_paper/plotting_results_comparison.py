"""
PAPER FIGURE: multi-run posterior-predictive comparison.

Overlays the four likelihood-combination chains (QLF / QLF+CERDF / QLF+Corr /
QLF+CERDF+Corr) on the SAME posterior-predictive panels, one colour per run.
For each run: load its chain, draw NUM_SAMPLES posterior samples, push them
through the emulator(s), and draw the **median + 16/84 band** in the run's
colour; observational data is drawn once per panel (black). Per-sample spaghetti
is omitted (too busy with four runs overlaid).

Produces up to four figures (``BAQARO_PLOT_FLAGS`` selects a subset):
  ``comparison_qlf`` / ``comparison_bhmf`` / ``comparison_cerdf`` / ``comparison_corr``.

DATA — chains + emulators come from the MCMC fiducial (the adopted fiducial's
``ck22final_100k`` chains on the ``clean_z0_g6.21_K22_final_smooth`` emulators).
``fiducial_data`` is imported FIRST so
it pins ``BAQARO_*`` (incl. ``BAQARO_NOTES_FILE_EMULATION`` / ``BAQARO_MCMC_NOTES``)
before ``comparison_config`` reads the environment — so ``comparison_config``'s
``RUNS`` / ``name_file`` / chain loader resolve to THIS fiducial, and the
emulators come from ``fiducial_data.load_paper_emulators``.

Adapted for the paper from ``inference/main_results_comparison.py``: same four
panels and posterior-predictive logic; the only changes are the fiducial-pinned
data source and routing the save to the git-tracked ``figures_paper/`` (PDF by
default) under stable basenames.

Env: ``BAQARO_PLOT_FLAGS=qlf,bhmf,cerdf,corr`` (default all),
``BAQARO_NUM_SAMPLES=100`` (posterior draws per run; lower for a fast corr check).
"""

import os

import numpy as np

# Pin the fiducial FIRST: sets BAQARO_* (incl. the emulator/MCMC notes) so the
# comparison_config import below resolves to this fiducial, and provides the
# emulator loader.
from baqaro.plotting_paper.fiducial_data import load_paper_emulators

from baqaro.plotting_paper import plot_config  # noqa: F401  (rcParams + Agg + PDF default)
from baqaro.plotting_common.plot_config import save_fig, maybe_show

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

from baqaro.inference.comparison_config import (
    RUNS, load_flat_samples, active_likelihoods,
)


# ==============================================================================
# CONFIG
# ==============================================================================
_plot_env = os.environ.get("BAQARO_PLOT_FLAGS", "").strip()
if _plot_env:
    _en = {s.strip() for s in _plot_env.split(",") if s.strip()}
    PLOT_FLAGS = {k: (k in _en) for k in ("qlf", "bhmf", "cerdf", "corr")}
else:
    PLOT_FLAGS = {"qlf": True, "bhmf": True, "cerdf": True, "corr": True}

NUM_SAMPLES = int(os.environ.get("BAQARO_NUM_SAMPLES", "100"))
rng = np.random.default_rng(12345)


# ==============================================================================
# LOAD EMULATORS + PER-RUN POSTERIOR PREDICTIONS
# ==============================================================================
emulator_flags = {
    "qlf": PLOT_FLAGS["qlf"], "bhmf": PLOT_FLAGS["bhmf"],
    "cerdf": PLOT_FLAGS["cerdf"], "qhmf": PLOT_FLAGS["corr"],
}
emulators = load_paper_emulators(emulator_flags)

runs_data = []
for run in RUNS:
    sub = load_flat_samples(run, n_draw=NUM_SAMPLES, rng=rng)
    if sub is None:
        print(f"WARNING: no chain for '{run['label']}' ({active_likelihoods(run)}) — skipping")
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
    raise RuntimeError("No MCMC chains found for any run. Check the MCMC fiducial / BAQARO_PAPER_MCMC_*.")


# ==============================================================================
# SHARED HELPERS
# ==============================================================================
def _run_legend(fig, ncol=None, data_style="marker"):
    handles = [Line2D([], [], color=r["color"], lw=2.5, label=r["label"]) for r in runs_data]
    # The "Data" proxy matches how the data is DRAWN in that figure: a line for
    # the cerdf step histograms, a white-filled square for the qlf (the binned
    # likelihood set), a filled point for the other errorbar figures (bhmf/corr).
    if data_style == "line":
        handles.append(Line2D([], [], color="black", lw=2.0, label="Data"))
    elif data_style == "square":
        handles.append(Line2D([], [], marker="s", mfc="white", mec="black",
                              markeredgewidth=1.5, ms=8, ls="", label="Data"))
    else:
        handles.append(Line2D([], [], color="black", marker="o", ls="", label="Data"))
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles),
               fontsize=10, frameon=True)


def _median_band(stack):
    """median, 16, 84 over the posterior-sample axis (axis 0)."""
    return (np.median(stack, axis=0), np.percentile(stack, 16, axis=0),
            np.percentile(stack, 84, axis=0))


def _matched_idx(emul_z, z, tol=0.5):
    i = int(np.argmin(np.abs(emul_z - z)))
    return i if abs(emul_z[i] - z) < tol else None


def _save(fig, kind):
    save_fig(fig, plot_config.FIGURES_DIR, f"comparison_{kind}", force_dir=True)


# ==============================================================================
# QLF — TWO figures with two layouts:
#   * `comparison_qlf` (primary)         — 2×4 grid mirroring `qlf_individual`:
#     LF per panel, one panel per redshift; obs in black, each chain's median
#     + 16/84 band in its corner-palette colour.
#   * `comparison_qlf_evolution`         — single-panel ND vs z in 3 L_bol bins,
#     mirroring `qlf_evolution`: Shen+20 obs bands in BLACK (alpha varies by
#     bin), chains in their corner colour, linestyle distinguishes the L_bol
#     bin within a chain.
# ==============================================================================
if PLOT_FLAGS["qlf"]:
    from qhtools.utils import my_utils
    from baqaro.obs_data.qlf_obs_data import (
        data_qlf_global_raw as data_qlf_global,
        data_qlf_global_sys_err_binned,
    )
    from baqaro.obs_data.qlf_uv_obs_data import (
        data_z2_UV_Kulkarni_sys_err, data_z4_UV_Kulkarni_sys_err,
        data_z5_UV_Niida_hsc, data_z5_UV_Niida_sdss,
        data_z6_UV_Schindler, data_z7_UV_Wang, data_z7_UV_Matsuoka,
    )
    from baqaro.obs_data.lrds_agn_obs_data import (
        lrd_qlf_data, bulichi_qlf_data,
    )
    from baqaro.obs_data.qlf_shen_model import plot_phi_obs_A, plot_phi_obs_B
    print("\n--- QLF comparison (qlf_individual 2×4 grid) ---")
    emu = emulators["qlf"]
    emul_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    log_bins = np.asarray(emu.axis_data["log_bins"], dtype=float)
    keys_grid = ["0.2", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0"]
    redshift_keys_grid = np.asarray([float(k) for k in keys_grid])
    # Mirror main_mcmc's QLF redshift mask so the emphasized "data actually used
    # in the likelihood" squares show ONLY what the fiducial actually fit. z=7 is
    # masked by default (mask_redshifts_qlf=[7.0]); BAQARO_QLF_INCLUDE_Z7=1 unmasks
    # it, the same toggle as the fit. The per-z faint-end floors are already
    # baked into data_qlf_global_sys_err_binned via the module defaults, so at
    # z<=6 the binned dict already equals the fitted point set — nothing else to
    # filter. Masked-z panels keep the model curve + faint context data, but drop
    # the emphasized red squares (the panel would otherwise imply z=7 was fitted).
    _mask_qlf_z = [] if os.environ.get("BAQARO_QLF_INCLUDE_Z7", "0") == "1" else [7.0]
    def _qlf_z_masked(zf):
        return any(abs(zf - mz) <= 0.1 for mz in _mask_qlf_z)
    nrows, ncols = 2, 4
    fig_grid, axs_grid = plt.subplots(nrows, ncols, figsize=(10, 5.6),
                                      sharex=True, sharey=True)
    axs_grid = axs_grid.flatten()
    OBS_COLOR = "black"
    show_z7_bol = os.environ.get("BAQARO_QLF_SHOW_Z7_BOL", "0") == "1"
    # Same obs DATASETS + marker SHAPES as qlf_individual (colours/alphas are the
    # comparison's own — obs stays neutral so the 4 chain colours read). The UV
    # family shares the open square; JWST sets are keyed by survey; z=5 adds
    # Niida+20's two samples (HSC faint end + SDSS bright end) and z=7 adds
    # Matsuoka+23 alongside Wang+19.
    _uv_by_z = {
        "2.0": [(data_z2_UV_Kulkarni_sys_err, "s")],
        "4.0": [(data_z4_UV_Kulkarni_sys_err, "s")],
        "5.0": [(data_z5_UV_Niida_hsc, "s"), (data_z5_UV_Niida_sdss, "s")],
        "6.0": [(data_z6_UV_Schindler, "s")],
        "7.0": [(data_z7_UV_Wang, "s"), (data_z7_UV_Matsuoka, "s")],
    }
    _lrd_markers = {"Greene+24": "p", "Matthee+24": "h"}
    _OPEN = dict(mfc="none", mec="0.35", mew=1.0)
    _FILL = dict(mfc="0.7", mec="0.35", mew=0.8)
    _WHITE = dict(mfc="white", mec="0.35", mew=1.5)
    obs_legend_spec = [
        ("o", _FILL, "Shen+20 (bolometric)"),
        ("s", _OPEN, "Kulkarni+19, Niida+20, Schindler+23, Wang+19, Matsuoka+23 (UV)"),
        ("X", _WHITE, "Bulichi+26 (JWST mid-IR)"),
        ("p", _WHITE, "Greene+24 (JWST broad lines)"),
        ("h", _WHITE, "Matthee+24 (JWST broad lines)"),
    ]
    for i, (key, z, ax) in enumerate(zip(keys_grid, redshift_keys_grid, axs_grid)):
        # ---- model: per-chain median + 16-84 band ----
        ei = _matched_idx(emul_z, z)
        if ei is not None:
            for r in runs_data:
                med, lo, hi = _median_band(r["preds"]["qlf"][:, ei, :])
                ax.plot(log_bins, med, color=r["color"], lw=2.0, zorder=10)
                ax.fill_between(log_bins, lo, hi, color=r["color"],
                                alpha=0.18, zorder=5)
        # ---- bolometric global compilation (raw, non-binned — faint). z=7 is
        # hidden by default (duplicates Matsuoka+23 UV), as in qlf_individual. ----
        if key in data_qlf_global and not (key == "7.0" and not show_z7_bol):
            obs_entry = data_qlf_global[key]
            ax.errorbar(obs_entry.x, obs_entry.data, yerr=obs_entry.err, fmt="o",
                        markersize=4, capsize=2, elinewidth=0.8, alpha=0.25,
                        color=OBS_COLOR, mec=OBS_COLOR, zorder=1)
        # ---- binned compilation (the QLF data actually used in the
        # likelihood — stand-out, top zorder so chains can't bury it) ----
        if key in data_qlf_global_sys_err_binned and not _qlf_z_masked(z):
            ob = data_qlf_global_sys_err_binned[key]
            # The likelihood set: strong black, fully opaque, on top — white-filled
            # squares so they read as the fitted data above the fainter context.
            ax.errorbar(ob.x, ob.data, yerr=ob.err, fmt="s",
                        markersize=6.5, capsize=3, elinewidth=1.7, alpha=1.0,
                        color=OBS_COLOR, mec=OBS_COLOR, mfc="white",
                        markeredgewidth=1.8, zorder=30)
        # ---- per-z UV points (open square; z=7 shows Wang+19 AND Matsuoka+23).
        # Non-Shen data: stronger alpha than the faint Shen bolometric compilation
        # (only the binned likelihood squares below are the strong-black set). ----
        for obs_d, uv_mk in _uv_by_z.get(key, []):
            ax.errorbar(my_utils.to_ergs(obs_d.x), obs_d.data, obs_d.err,
                        c=OBS_COLOR, mec=OBS_COLOR, marker=uv_mk, alpha=0.4,
                        capsize=2, elinewidth=0.8, fillstyle="none",
                        linestyle="none", markersize=4, zorder=2)
        # ---- JWST LRDs: UPPER LIMITS in L_bol (leftward 0.5-dex arrows), marker
        # SHAPE keyed by survey — same convention as qlf_individual. ----
        if key in lrd_qlf_data:
            for lrd in lrd_qlf_data[key]:
                mk = next((m for surv, m in _lrd_markers.items()
                           if lrd.label.startswith(surv)), "o")
                ax.errorbar(lrd.x, lrd.data, yerr=[lrd.err_down, lrd.err_up],
                            fmt=mk, markersize=6, capsize=2, elinewidth=1.0,
                            ecolor=OBS_COLOR, alpha=0.4, color="white",
                            mec=OBS_COLOR, markeredgewidth=1.5,
                            linestyle="none", zorder=6)
                for xi, yi in zip(np.atleast_1d(lrd.x), np.atleast_1d(lrd.data)):
                    ax.annotate("", xy=(xi - 0.5, yi), xytext=(xi, yi), zorder=6,
                                arrowprops=dict(arrowstyle="->", color=OBS_COLOR,
                                                lw=1.2, alpha=0.4,
                                                mutation_scale=11,
                                                shrinkA=0, shrinkB=0))
        # ---- Bulichi+26 mid-IR AGN (marker "X", same shape as qlf_individual) ----
        if key in bulichi_qlf_data:
            bul = bulichi_qlf_data[key]
            ax.errorbar(bul.x, bul.data, yerr=[bul.err_down, bul.err_up],
                        fmt="X", markersize=7, capsize=2, elinewidth=1.0,
                        alpha=0.4, color="white", mec=OBS_COLOR,
                        markeredgewidth=1.5, linestyle="none", zorder=6)
        # ---- z label (bottom-left, bold) ----
        ax.text(0.05, 0.05, f"$z = {key}$", transform=ax.transAxes,
                fontweight="bold", color=OBS_COLOR, ha="left", va="bottom")
        # ---- axis labels / tick gating ----
        is_left = (i % ncols == 0)
        is_bottom = (i // ncols == nrows - 1)
        if is_left:
            ax.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]", labelpad=-1)
        else:
            ax.tick_params(labelleft=False)
        if is_bottom:
            ax.set_xlabel(r"$\log_{10}\, L_{\rm bol}$ [erg s$^{-1}$]", labelpad=-3)
        else:
            ax.tick_params(labelbottom=False)
    for a in axs_grid:
        a.set_xlim(43.5, 48.35)
        a.set_ylim(-9.8, -2.5)
        a.minorticks_on()
    _run_legend(fig_grid, data_style="square")
    # Observation SHAPE legend (gray proxies) under the grid — same marker->dataset
    # mapping as qlf_individual so the shapes are identifiable, in two facility-
    # grouped rows (pre-JWST / JWST). Colours here are legend-neutral gray.
    def _obs_handles(labels):
        return [Line2D([], [], ls="", marker=mk, ms=7, label=lbl, **st)
                for mk, st, lbl in obs_legend_spec if lbl in labels]
    _jwst = [lbl for _, _, lbl in obs_legend_spec if "JWST" in lbl]
    _pre = [lbl for _, _, lbl in obs_legend_spec if "JWST" not in lbl]
    _lk = dict(loc="lower center", frameon=False, fontsize=10, handletextpad=0.35,
               columnspacing=1.6, borderaxespad=0.0, borderpad=0.0)
    fig_grid.legend(handles=_obs_handles(_pre), bbox_to_anchor=(0.5, 0.045),
                    ncol=len(_pre), **_lk)
    fig_grid.legend(handles=_obs_handles(_jwst), bbox_to_anchor=(0.5, 0.0),
                    ncol=len(_jwst), **_lk)
    fig_grid.subplots_adjust(left=0.08, right=0.98, top=0.905, bottom=0.16,
                             hspace=0.05, wspace=0.05)
    _save(fig_grid, "qlf")

    print("\n--- QLF comparison (qlf_evolution layout) ---")
    emu = emulators["qlf"]
    emul_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    log_bins = np.asarray(emu.axis_data["log_bins"], dtype=float)
    dlogL_emu = float(np.median(np.diff(log_bins)))

    log_Lbins = np.linspace(45.5, 48.5, 4)   # 3 bins: [45.5,46.5], [46.5,47.5], [47.5,48.5]
    # Bin-distinguishing line styles (same for all chains).
    LSTYLES = ["-", "--", ":"]
    BIN_MARKERS = ["o", "s", "^"]
    z_keys = np.asarray([0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    fig_qlf, ax_qlf = plt.subplots(1, 1, figsize=(6.5, 4.8))

    # Shen+20 obs bands — all in BLACK, distinguished by alpha so the three
    # bins are still readable.
    bin_band_alphas = [0.35, 0.22, 0.12]
    for j in range(len(log_Lbins) - 1):
        Phi_A, Err_A, Z_A = plot_phi_obs_A(Lmin=log_Lbins[j], Lmax=log_Lbins[j + 1])
        Phi_B, Err_B, Z_B = plot_phi_obs_B(Lmin=log_Lbins[j], Lmax=log_Lbins[j + 1])
        min_obs = np.minimum(Phi_A - Err_A, Phi_B - Err_B)
        max_obs = np.maximum(Phi_A + Err_A, Phi_B + Err_B)
        ax_qlf.fill_between(Z_A, max_obs, min_obs, color="black",
                            alpha=bin_band_alphas[j], edgecolor="none", zorder=1)

    # Per-chain median + 16-84 band of the L-bin number density at each z.
    for r in runs_data:
        qlf_pred = r["preds"]["qlf"]  # (n_samples, n_z_emul, n_logbins)
        for j in range(len(log_Lbins) - 1):
            lo_L, hi_L = log_Lbins[j], log_Lbins[j + 1]
            in_bin = (log_bins >= lo_L) & (log_bins < hi_L)
            if not np.any(in_bin):
                continue
            z_plot, med_plot, p16_plot, p84_plot = [], [], [], []
            for z in z_keys:
                ei = _matched_idx(emul_z, z)
                if ei is None:
                    continue
                # Integrate Phi over the L window for each posterior sample.
                phi_slice = 10 ** qlf_pred[:, ei, in_bin]  # (n_samples, n_in_bin)
                ndens_samples = np.sum(phi_slice, axis=1) * dlogL_emu
                ok = np.isfinite(ndens_samples) & (ndens_samples > 0)
                if not np.any(ok):
                    continue
                log_nd = np.log10(ndens_samples[ok])
                z_plot.append(emul_z[ei])
                med_plot.append(np.median(log_nd))
                p16_plot.append(np.percentile(log_nd, 16))
                p84_plot.append(np.percentile(log_nd, 84))
            if not z_plot:
                continue
            z_plot = np.asarray(z_plot)
            med_plot = np.asarray(med_plot)
            p16_plot = np.asarray(p16_plot)
            p84_plot = np.asarray(p84_plot)
            ax_qlf.plot(z_plot, med_plot, color=r["color"], lw=2.0,
                        ls=LSTYLES[j], zorder=5)
            ax_qlf.fill_between(z_plot, p16_plot, p84_plot, color=r["color"],
                                alpha=0.15, zorder=2)

    # Legend: stack chain colours + L-bin styles separately.
    chain_handles = [Line2D([], [], color=r["color"], lw=2.5, label=r["label"])
                     for r in runs_data]
    bin_handles = []
    for j in range(len(log_Lbins) - 1):
        lab = (rf"$\log_{{10}}\,L_{{\rm bol}}/\mathrm{{erg\,s^{{-1}}}}="
               rf"{log_Lbins[j]:.1f}$–${log_Lbins[j + 1]:.1f}$")
        bin_handles.append(Line2D([], [], color="black", lw=2.0,
                                  ls=LSTYLES[j], label=lab))
    obs_proxy = Line2D([], [], color="black", lw=8.0, alpha=0.30,
                       label="Shen+20")
    leg1 = ax_qlf.legend(handles=chain_handles, loc="lower left",
                         fontsize=9, frameon=True, framealpha=0.85,
                         handlelength=1.6, labelspacing=0.25, borderpad=0.4)
    ax_qlf.add_artist(leg1)
    ax_qlf.legend(handles=bin_handles + [obs_proxy], loc="upper right",
                  fontsize=9, frameon=True, framealpha=0.85,
                  handlelength=1.8, labelspacing=0.25, borderpad=0.4)

    ax_qlf.set_xlim(0, 7.2)
    ax_qlf.set_ylim(-9.8, -3.2)
    ax_qlf.set_xlabel(r"Redshift $z$", labelpad=4)
    ax_qlf.set_ylabel(r"$\log_{10}\,\Phi$ [dex$^{-1}$ cMpc$^{-3}$]", labelpad=-1)
    ax_qlf.minorticks_on()
    fig_qlf.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.13)
    _save(fig_qlf, "qlf_evolution")


# ==============================================================================
# BHMF
# ==============================================================================
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
    _save(fig, "bhmf")


# ==============================================================================
# CERDF
# ==============================================================================
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

    # cerdf_distribution layout: rows = L_bol bins, cols = redshifts (z=1..6).
    redshift_keys = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    keys = [f"{z:.1f}" for z in redshift_keys]
    logL_bins = np.linspace(45.5, 47.5, 5)   # 4 L_bol bins
    n_L = len(logL_bins) - 1
    n_z = len(redshift_keys)
    log_etas_data = my_utils.to_solar(logL_Bols_data) - logM_BHs_data - nc.log_csi
    bins_data = np.linspace(-3, 1.5, 25)
    bins_data_c = 0.5 * (bins_data[1:] + bins_data[:-1])
    bin_w = np.abs(np.diff(log_bins_cerdf[:2])[0])
    sigma_bins = 0.3 / bin_w

    def _cerdf_pdfs(pred_cerdf, zidx, lmin_idx, lmax_idx):
        """Per-sample normalized + convolved differential ERDF PDFs."""
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

    fig, axes = plt.subplots(n_L, n_z, figsize=(10, 6.5),
                             sharex=True, sharey=True)
    for iz, (key, z) in enumerate(zip(keys, redshift_keys)):
        zi = int(np.abs(emul_z - z).argmin())
        for iL in range(n_L):
            ax = axes[iL, iz]
            lmin_v, lmax_v = logL_bins[iL], logL_bins[iL + 1]
            lmin_i = int(np.abs(log_L_thresholds - lmin_v).argmin())
            lmax_i = int(np.abs(log_L_thresholds - lmax_v).argmin())
            # ---- observations (single black step + dotted median) ----
            m = ((redshifts_data < z + 0.5) & (redshifts_data > z - 0.5)
                 & (logL_Bols_data > lmin_v) & (logL_Bols_data < lmax_v))
            if np.sum(m) > 3:
                c, _ = np.histogram(log_etas_data[m], bins=bins_data, density=True)
                ax.step(bins_data_c, c, where="mid", lw=2.5, alpha=0.7,
                        color="black", zorder=10)
                ax.axvline(np.median(log_etas_data[m]), ls=":", lw=2.0,
                           color="black", alpha=0.7, zorder=11)
            # ---- per-chain median + 16-84 band ----
            for r in runs_data:
                pdfs = _cerdf_pdfs(r["preds"]["cerdf"], zi, lmin_i, lmax_i)
                valid = np.any(pdfs > 0, axis=1)
                if np.sum(valid) == 0:
                    continue
                med, lo, hi = _median_band(pdfs[valid])
                ax.plot(log_bins_cerdf, med, color=r["color"], lw=1.8,
                        alpha=0.9, zorder=5)
                ax.fill_between(log_bins_cerdf, lo, hi, color=r["color"],
                                alpha=0.15, zorder=2)
                cdf = np.cumsum(med[:-1] * np.diff(log_bins_cerdf))
                if cdf[-1] > 0:
                    cdf_norm = np.concatenate(([0.0], cdf / cdf[-1]))
                    ax.axvline(np.interp(0.5, cdf_norm, log_bins_cerdf),
                               ls="--", lw=1.4, color=r["color"], alpha=0.7,
                               zorder=6)
            # ---- labels ----
            if iL == 0:
                ax.set_title(f"$z = {key}$", fontsize=13)
            if iz == 0:
                lmin_str = f"{lmin_v:.0f}" if lmin_v == int(lmin_v) else f"{lmin_v:.1f}"
                lmax_str = f"{lmax_v:.0f}" if lmax_v == int(lmax_v) else f"{lmax_v:.1f}"
                ax.set_ylabel(rf"$L_{{bol}} \in [{lmin_str}, {lmax_str}]$"
                              "\n" + r"$P(\log \lambda_{\rm Edd})$", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            if iL == n_L - 1:
                ax.set_xlabel(r"$\log_{10}\,\lambda_{\rm Edd}$", labelpad=-1)
            else:
                ax.tick_params(labelbottom=False)

    for a in axes.flatten():
        a.set_xlim(-2.5, 1.5)
        a.set_ylim(0., 1.9)
        a.minorticks_on()

    # Figure-level chain-colour legend at top. Data is a step histogram here, so
    # its legend proxy is a LINE, not a point.
    _run_legend(fig, data_style="line")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.10,
                        hspace=0.05, wspace=0.05)
    _save(fig, "cerdf")


# ==============================================================================
# CORRELATION
# ==============================================================================
if PLOT_FLAGS["corr"]:
    from qhtools.clustering.qhmf_to_corr import get_corr_from_triangle, get_corr_from_triangle_cross
    from qhtools.clustering.projected_correlation_functions import get_projected_wp, get_volume_averaged_xi
    # DATA matched to the main-text correlation_plot figure (plotting_clustering.py):
    #   z=2.5 -> data_ef_ext (Eftekharzadeh+15, full sample, 18 bins, diagonal);
    #   z=4.0 -> data_shen_highz_allfields_err2 (Shen+07 ALL fields, errors x2 —
    #           the same set the likelihood uses by default (BAQARO_CORR_Z4 in
    #           inference/main_mcmc.py); the figure resolves it via
    #           BAQARO_CLUST_Z4_DATA, and the good-fields x1 data_shen_highz is
    #           the alternative);
    #   z=6.1 -> data_aspire (cross) + data_eiger overlay.
    from baqaro.obs_data import corr_obs_data as _corr_mod
    from baqaro.obs_data.corr_obs_data import (
        data_ef_ext, data_aspire, data_eiger,
    )
    from baqaro.obs_data.corr_obs_data import GAL_LOGM_CUT_FIGURE
    from baqaro.inference.likelihoods_and_priors import precompute_corr_inputs
    print("\n--- Correlation comparison ---")
    emu = emulators["qhmf"]
    _z4_key = (os.environ.get("BAQARO_CLUST_Z4_DATA", "data_shen_highz_allfields_err2").strip()
               or "data_shen_highz_allfields_err2")
    data_shen_z4 = getattr(_corr_mod, _z4_key)
    data_ef_ext.redshift = 2.5; data_ef_ext.corr_type = "auto"
    data_ef_ext.logM_min = 11.5; data_ef_ext.logM_max = 14.5
    data_shen_z4.redshift = 4.0; data_shen_z4.corr_type = "auto"
    data_shen_z4.logM_min = 11.5; data_shen_z4.logM_max = 14.5
    data_aspire.redshift = 6.1; data_aspire.corr_type = "cross"
    data_aspire.logM_min = 10.5; data_aspire.logM_max = 14.0
    data_eiger.redshift = 6.1; data_eiger.corr_type = "cross"
    data_eiger.logM_min = 10.5; data_eiger.logM_max = 14.0
    corr_datasets = {2.5: data_ef_ext, 4.0: data_shen_z4, 6.1: data_aspire}
    extra_obs = {id(data_aspire): [data_eiger]}

    target_z = np.asarray(emu.axis_data["redshift"], dtype=float)
    datas_corr = []
    for i, tz in enumerate(target_z):
        best = min(corr_datasets, key=lambda z: abs(z - tz))
        # Uniqueness guard (same as the QLF matcher in main_mcmc): a dataset
        # binds to exactly ONE panel — the emulator slice CLOSEST to it — else a
        # dataset with two nearby slices (e.g. z=4 sits between emulator z=3.937
        # gap 0.063 and z=3.534 gap 0.466) is plotted twice and the ncols cap
        # drops a real panel (here z=6.1). Without this, comparison_corr
        # showed z=4 twice and no z=6.
        closest_slice = int(np.argmin(np.abs(target_z - best)))
        if abs(best - tz) < 0.5 and closest_slice == i:
            datas_corr.append(corr_datasets[best])
        else:
            datas_corr.append(None)
    print("Precomputing correlation inputs...")
    pre = precompute_corr_inputs(datas_corr)
    log_L_thr = emu.axis_data["log_L_threshold"]
    log_mbins_qhmf = emu.axis_data["log_bins"]

    active = [i for i, d in enumerate(datas_corr) if d is not None]
    # The emulator's z axis is DESCENDING (7.3, 6.1, 5, 4, 3, 2.5, ...), so
    # `active` is in high-z -> low-z order. The paper figure `correlation_plot`
    # uses LOW-z -> HIGH-z (2.5, 4, 6.1) — match that here.
    active.sort(key=lambda i: datas_corr[i].redshift)
    if active:
        # correlation_plot layout: 1×3, modestly taller than the paper figure
        # so the top-of-figure run-colour legend has its own band.
        ncols = 3
        nrows = 1
        # Same canvas as `correlation_plot.pdf` (10 x 2.9) — the run-colour
        # legend at top is tucked into the slim 0.04-height header below.
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.9),
                                 sharex=True, sharey=True, squeeze=False)
        axes = axes.flatten()
        for panel, iz in enumerate(active):
            if panel >= ncols:
                break
            ax = axes[panel]
            dc = datas_corr[iz]; inp = pre[iz]
            ctype = getattr(dc, "corr_type", "auto")
            # ---- per-chain model curves (lowest zorder, obs sits on top) ----
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
                        # Galaxy proxy for the z=6 cross: use the FIGURE cut
                        # (GAL_LOGM_CUT_FIGURE), matching correlation_plot's model
                        # curve. The /5.0 scalar cancels in the mass-weight
                        # normalisation.
                        qg = np.copy(mf_fit) / 5.0
                        qg[log_m_axis < GAL_LOGM_CUT_FIGURE] = 0.0
                        lr, xic = get_corr_from_triangle_cross(
                            log_rbins_centers=np.log10(rbins), log_mbins_centers=log_m_axis,
                            triangle=tri, log_m_axis=log_m_axis, qhmf1=10 ** qi, qhmf2=qg)
                        xv = get_volume_averaged_xi(rpbins, xic, 10 ** lr, pimax=dc.pimax)
                        xx = 10 ** (0.5 * (np.log10(rpbins[:-1]) + np.log10(rpbins[1:]))); yy = xv
                    ys.append(yy)
                if ys:
                    ys = np.asarray(ys)
                    med, lo, hi = _median_band(ys)
                    ax.plot(xx, med, color=r["color"], lw=2.2, zorder=5)
                    ax.fill_between(xx, lo, hi, color=r["color"], alpha=0.18,
                                    zorder=2)
            # ---- observations (always BLACK on top) ----
            ax.errorbar(dc.x, dc.data, yerr=dc.err, fmt="o", color="black",
                        ms=5, capsize=2, elinewidth=1.0, alpha=0.9, zorder=20,
                        label=getattr(dc, "label", "Data"))
            # ---- EIGER overlay at z~6 (same convention as correlation_plot) ----
            _has_extra = False
            for extra in extra_obs.get(id(dc), []):
                _has_extra = True
                ax.errorbar(extra.x, extra.data, yerr=extra.err, fmt="s",
                            color="black", mfc="none", ms=5, capsize=2,
                            elinewidth=0.9, alpha=0.9, zorder=20,
                            label=getattr(extra, "label", "Data2"))
            # ---- z label in corner, no white box ----
            label_txt = (f"$z = {dc.redshift:.1f}$, cross" if ctype == "cross"
                         else f"$z = {dc.redshift:.1f}$, auto")
            if ctype == "cross":
                ax.text(0.95, 0.95, label_txt, transform=ax.transAxes,
                        fontweight="bold", color="black", ha="right", va="top")
            else:
                ax.text(0.05, 0.95, label_txt, transform=ax.transAxes,
                        fontweight="bold", color="black", ha="left", va="top")
            # z=6.1 panel: ASPIRE / EIGER marker legend in bottom-right.
            if _has_extra:
                ax.legend(handles=[
                    Line2D([], [], color="black", marker="o", ls="", label="ASPIRE"),
                    Line2D([], [], color="black", marker="s", mfc="none", ls="", label="EIGER"),
                ], fontsize=9, loc="lower right", markerfirst=False,
                handletextpad=0.5, handlelength=0.9, borderpad=0.4, frameon=False)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(0.09, 95.0); ax.set_ylim(1.1e-1, 1e3)
            if panel == 0:
                ax.set_ylabel(r"$w_p(r_p)/r_p,\, \chi_{\mathrm{V}}$", labelpad=-1)
            else:
                ax.tick_params(labelleft=False)
            # h-FREE comoving Mpc (separations are /cosmo.h at load).
            ax.set_xlabel(r"$r_p$ [cMpc]", labelpad=-3)
            ax.minorticks_on()
        for j in range(min(len(active), ncols), len(axes)):
            axes[j].axis("off")
        _run_legend(fig)
        # Match correlation_plot.pdf margins (top=0.96, bottom=0.19) and shave
        # the legend into the tight top band so the panel proportions stay
        # identical to the main-text figure.
        fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.19,
                            hspace=0.05, wspace=0.05)
        _save(fig, "corr")
    else:
        print("No correlation data matched. Skipping corr figure.")

maybe_show()
