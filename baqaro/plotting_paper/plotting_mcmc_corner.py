"""
PAPER FIGURE: posterior corner plot, overlaying likelihood combinations.

Overlays the MCMC posteriors from several likelihood combinations (QLF only,
QLF+Corr, QLF+CERDF+Corr by default) on one corner plot, so the tightening and
shifts from adding data constraints are visible. The diagonal titles report the
median ± 16/84% of the JOINT (last) run.

DATA — the chains come from the MCMC fiducial pinned in
``plotting_paper/fiducial_data.py`` (``MCMC_FIDUCIAL`` → ``name_file_mcmc`` +
``mcmc_notes``): the adopted fiducial's inference (``qcc_ck22final_v1``: the
``ck22final_100k`` chains on the ``clean_z0_g6.21_K22_final_smooth`` emulators). Each run's
chain is ``{mcmc_dir}/mcmc_{name_file_mcmc}_{active}_{mcmc_notes}.h5`` where
``active`` is the ``+``-joined likelihood tag. Override the selection with
``BAQARO_PAPER_MCMC_<X>``.

Adapted for the paper from the working version of this figure. Two
substantive changes: (1) the chain ``name_file`` / notes come from
``fiducial_data`` (the earlier version hardcoded an L2800N5040 emulator
name_file with no foldmass/subset tokens, which would mis-resolve the z=0
chains); (2) the parameter names are hardcoded in the canonical sampling order
(``main_training.param_definitions_emulation``) instead of loading the 206 MB
QLF emulator just to read ``.param_names``. Save routes to ``figures_paper/``.
"""

import os

import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from baqaro.plotting_paper.fiducial_data import (
    name_file_mcmc,
    mcmc_dir,
    mcmc_notes,
    RESOLVED as _PAPER_RESOLVED,
)
from baqaro.core_functions.bestfit_registry import get_bestfit

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = os.environ.get("BAQARO_PAPER_CORNER_NAME", "mcmc_corner")

# Canonical 6-param sampling order (= main_training.param_definitions_emulation
# key order, which the emulator.param_names and chain columns follow). Hardcoded
# so the figure doesn't need the 206 MB emulator just for labels.
param_names = [
    "log_eta_mean_0",
    "log_eta_mean_evol",
    "std_0",
    "logtcoherence",
    "logfseed",
    "sigmaseed",
]
ndim = len(param_names)

# Final paper symbols (Table of free parameters). Note logfseed is shown as the
# seed MASS log10 M_seed = 10.5 + logfseed (see LOG_MSEED_OFFSET below).
PARAM_LABELS = {
    "log_eta_mean_0":    r"$\log\,\eta_{\rm av,0}$",
    "log_eta_mean_evol": r"$\eta_{\rm av,slope}$",
    "std_0":             r"$\sigma_{\rm acc}$",
    "std_evol":          r"$\sigma_{\rm evol}$",
    "logtcoherence":     r"$\log(\tau_{\rm coh}/{\rm yr})$",
    "logfseed":          r"$\log\,M_{\rm seed}$",
    "sigmaseed":         r"$\sigma_{\rm seed}$",
}

# Which likelihood combinations to overlay. `active` is the `+`-joined tag in
# the chain filename. The diagonal titles use TITLE_RUN_INDEX (-1 = last = joint).
RUNS = [
    {"active": "qlf",            "label": "QLF only",           "color": "#0072B2"},
    {"active": "qlf+cerdf",      "label": "QLF + cERDF",        "color": "#D55E00"},
    {"active": "qlf+corr",       "label": "QLF + Corr",         "color": "#CC79A7"},
    {"active": "qlf+cerdf+corr", "label": "QLF + Corr + cERDF", "color": "#009E73"},
]
TITLE_RUN_INDEX = -1
MANUAL_BURN_IN = 2000  # used when autocorr fails (10k step chains often need this)

# The adopted paper fiducial (QLF+CERDF+Corr joint, no-shift) — marked as a star
# on every panel + dashed verticals on the diagonals. Defaults to whatever
# PAPER_FIDUCIAL points at (via RESOLVED, so BAQARO_PAPER_BESTFIT_NAME wins), so
# a repoint moves the star with the figures instead of stranding it on the old
# model. BAQARO_PAPER_CORNER_BESTFIT still overrides for a one-off comparison.
FIDUCIAL_BESTFIT = os.environ.get(
    "BAQARO_PAPER_CORNER_BESTFIT", _PAPER_RESOLVED["BAQARO_BESTFIT_NAME"])
FID_FACE = "gold"            # yellow star fill
FID_EDGE = "black"           # star outline
FID_LINE = "black"           # diagonal verticals

# Reparametrisation for the paper: the seed fraction logfseed = log10(M_BH/M_halo)
# at seeding is reported as the seed MASS at the pivot halo mass 10^10.5 Msun,
# i.e. log10 M_seed = 10.5 + log10 f_seed. Applied to both the chains and the
# fiducial below so the corner shows log10 M_seed directly.
LOG_MSEED_OFFSET = 10.5
FSEED_IDX = param_names.index("logfseed")

_bf = get_bestfit(FIDUCIAL_BESTFIT)
fid_values = [float(_bf[p]) for p in param_names]
fid_values[FSEED_IDX] += LOG_MSEED_OFFSET
print(f"Fiducial ({FIDUCIAL_BESTFIT}): {dict(zip(param_names, fid_values))}")


# ==============================================================================
# LOAD CHAINS
# ==============================================================================
loaded_runs = []
for run in RUNS:
    mcmc_filename = f"mcmc_{name_file_mcmc}_{run['active']}"
    if mcmc_notes:
        mcmc_filename += f"_{mcmc_notes}"
    mcmc_path = f"{mcmc_dir}/{mcmc_filename}.h5"

    if not os.path.exists(mcmc_path):
        print(f"WARNING: {mcmc_path} not found — skipping '{run['label']}'")
        continue

    print(f"Loading: {mcmc_path}")
    reader = emcee.backends.HDFBackend(mcmc_path, read_only=True)
    try:
        tau = reader.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau))
        # NOTE: thin is for VISUAL smoothness only, NOT independent-sample counting.
        # The old thin=0.5*min(tau) left the least-converged chain (qlf+corr) with
        # only ~11k points, so the 0.95 contour broke into lumpy islands ("bubbles").
        # Thinning a converged chain is unbiased, so for the density estimate we keep
        # FAR more samples (target ~200k) — smooth contours, medians unchanged.
        # Burn-in (2*max(tau)) is unchanged.
        n_after_burn = (reader.iteration - burnin) * reader.shape[0]
        thin = max(1, n_after_burn // 200_000)
        print(f"  tau: {tau}, burn-in: {burnin}, plot-thin: {thin} "
              f"(indep-sample thin would be {max(1, int(0.5 * np.min(tau)))})")
    except Exception:
        print(f"  Chain too short for autocorr — manual burn-in={MANUAL_BURN_IN}")
        burnin, thin = MANUAL_BURN_IN, 1

    # Drop NON-MIXING walkers: walkers with a catastrophically low MEAN log-prob
    # (median - max(6*MAD, 5 lnL)) are not sampling the equilibrium distribution
    # and are excluded from the density estimate. This is a no-op for well-mixed
    # chains (their walkers cluster within ~0.3 lnL).
    chain = reader.get_chain(discard=burnin, thin=thin)          # (steps, walkers, ndim)
    lp_w = reader.get_log_prob(discard=burnin).mean(axis=0)      # per-walker mean lnP
    _med = np.median(lp_w)
    _mad = 1.4826 * np.median(np.abs(lp_w - _med))
    stuck = lp_w < _med - max(6.0 * _mad, 5.0)
    if stuck.any():
        print(f"  dropping {int(stuck.sum())} non-mixing walker(s) "
              f"{list(np.where(stuck)[0])} (mean lnP << ensemble)")
        chain = chain[:, ~stuck, :]
    samples = chain.reshape(-1, chain.shape[-1])
    samples[:, FSEED_IDX] += LOG_MSEED_OFFSET   # logfseed -> log10 M_seed
    print(f"  Posterior shape: {samples.shape}")
    loaded_runs.append({"samples": samples, "label": run["label"], "color": run["color"]})

if len(loaded_runs) == 0:
    raise RuntimeError("No MCMC chains found. Check MCMC_FIDUCIAL / RUNS / BAQARO_PAPER_MCMC_*.")


# ==============================================================================
# CORNER PLOT
# ==============================================================================
print(f"\nPlotting {len(loaded_runs)} runs...")
labels = [PARAM_LABELS.get(p, p) for p in param_names]

fig = None
for i, run in enumerate(loaded_runs):
    kwargs = dict(
        labels=labels if i == 0 else None,
        label_kwargs={"fontsize": 11, "labelpad": 6},   # tick label/axis label gap
        show_titles=False,
        smooth=1.2, smooth1d=1.2,
        color=run["color"],
        plot_datapoints=False, plot_density=False,
        fill_contours=True, no_fill_contours=True,
        levels=[0.68, 0.95],
        hist_kwargs={"alpha": 0.9, "linewidth": 1.6},
        contour_kwargs={"linewidths": 1.5},
        contourf_kwargs={"alpha": 0.25},
        # Fix the per-panel axis range to the FULL prior box (instead of corner's
        # default data-extent auto-zoom) — without this every posterior looks
        # like it's railing because the panel is just very zoomed in.
        # Order: log_eta_mean_0, log_eta_mean_evol, std_0, logtcoherence,
        # log10 M_seed (= logfseed + 10.5), sigmaseed.
        # These MUST equal main_training.param_definitions_emulation (the box the
        # emulator was trained on and the MCMC prior enforces). sigmaseed was drawn
        # from 0.15 while the real prior floor is 0.20 — so a posterior railing AT the floor rendered as a spurious
        # empty gap between the rail and the axis, instead of visibly hitting it.
        range=[(-2.5, -0.5), (0.2, 2.0), (0.2, 1.0),
               (3.5, 7.3),
               (-7.5 + LOG_MSEED_OFFSET, -3.5 + LOG_MSEED_OFFSET),
               (0.2, 1.5)],
        labelpad=0.10, max_n_ticks=3,                    # corner's own axis-label distance from frame
    )
    if fig is None:
        fig = corner.corner(run["samples"], fig=plt.figure(figsize=(5.6, 5.6)), **kwargs)
    else:
        corner.corner(run["samples"], fig=fig, **kwargs)

    # Strip the outermost contourf level (the fill outside the 95% contour) so
    # the 2D panels keep a white background. corner's contourf uses levels
    # [0, V_95, V_68, H_max] => 3 filled paths; drop the first.
    axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)
    for row in range(ndim):
        for col in range(row):
            for child in axes_grid[row, col].get_children():
                if hasattr(child, "levels") and child.filled:
                    paths = child.get_paths()
                    fcs = child.get_facecolors()
                    if len(paths) >= 3:
                        child.set_paths(paths[1:])
                        child.set_facecolors(fcs[1:])

# Fiducial model (the parameter set used everywhere else in the paper): a star
# on every 2-D panel and a dashed vertical on each diagonal. (Per-parameter
# median±16/84 numbers are tabulated alongside the figure output, so
# the space-hungry diagonal titles are dropped here.)
axes_grid = np.array(fig.get_axes()).reshape(ndim, ndim)

# Peak-normalise each diagonal's 1D marginals to unit height. corner scales the
# diagonal y-axis to the tallest (narrow joint) curve, which dwarfs the broad
# QLF-only marginal to near-invisibility; equal peaks make all three readable.
for i in range(ndim):
    axd = axes_grid[i, i]
    for ln in axd.get_lines():
        y = np.asarray(ln.get_ydata(), dtype=float)
        if y.size > 2 and np.isfinite(y).any() and np.nanmax(y) > 0:
            ln.set_ydata(y / np.nanmax(y))
    axd.set_ylim(0, 1.12)

for i in range(ndim):
    axes_grid[i, i].axvline(fid_values[i], color=FID_LINE, ls="--", lw=1.3, zorder=9)
    for j in range(i):
        axes_grid[i, j].plot(
            fid_values[j], fid_values[i], marker="*", markersize=11,
            markerfacecolor=FID_FACE, markeredgecolor=FID_EDGE, markeredgewidth=0.8,
            linestyle="none", zorder=10,
        )

for ax in fig.get_axes():
    ax.set_facecolor("white")
    ax.tick_params(labelsize=9, length=2.5, width=0.5, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

legend_handles = [
    Patch(facecolor=run["color"], alpha=0.4, edgecolor=run["color"], label=run["label"])
    for run in loaded_runs
]
legend_handles.append(Line2D(
    [], [], marker="*", markersize=10, markerfacecolor=FID_FACE,
    markeredgecolor=FID_EDGE, markeredgewidth=0.8, linestyle="none", label="Fiducial"))
fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.955, 0.945),
           fontsize=10, frameon=False, markerfirst=False)

fig.subplots_adjust(left=0.11, right=0.985, top=0.985, bottom=0.10, hspace=0.07, wspace=0.07)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
