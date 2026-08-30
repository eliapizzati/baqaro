"""
Corner plot comparing posteriors from different likelihood combinations.

Overlays multiple MCMC chains on the same corner plot so you can see how
different data constraints (QLF-only, CERDF-only, QLF+CERDF, QLF+corr, etc.)
shift and tighten the posteriors.

Usage
-----
Edit RUNS below to list the likelihood combinations you want to compare.
Each entry specifies the LIKELIHOOD_FLAGS that were active when the chain
was produced, a display label, and a color.
"""

import os
import corner
import matplotlib.pyplot as plt
import numpy as np

from baqaro.emulation.loading_helpers import load_emulators
from baqaro.plotting_common.plot_config import fig_ext
from baqaro.inference.comparison_config import (
    RUNS, name_file, short_name, path_out, source_dir, notes_suffix, mcmc_path, load_flat_samples,
    comparison_plots_dir,
)


# ==========================================
# CONFIGURATION
# ==========================================
# RUNS (which likelihood combinations to compare + colours/labels), the model
# identifier (name_file), source_dir, path_out, the chain-path resolver and the
# flat-sample loader all come from comparison_config — shared with
# main_results_comparison.py so colours/labels/chain paths stay in sync.
# BAQARO_MCMC_NOTES (e.g. =minL05) is honoured there and tags the output PDF.

# Manual burn-in fallback (used when autocorrelation estimate fails)
MANUAL_BURN_IN = 2000  # 10k-step chains often don't reach 50*tau → autocorr fails → fallback kicks in

# ==========================================
# LOAD ONE EMULATOR (for param names/ranges)
# ==========================================
emulator_flags = {"qlf": True, "bhmf": False, "cerdf": False, "qhmf": False}
emulators = load_emulators(path_out, name_file, emulator_flags)

_any_emulator = next(v for v in emulators.values() if v is not None)
param_names = _any_emulator.param_names
param_ranges = _any_emulator.param_ranges
ndim = len(param_names)
print(f"\nParameters ({ndim}D): {param_names}")
print(f"Ranges: {param_ranges}")


# ==========================================
# LOAD ALL CHAINS
# ==========================================

loaded_runs = []

for run in RUNS:
    samples = load_flat_samples(run, manual_burnin=MANUAL_BURN_IN)
    if samples is None:
        print(f"WARNING: {mcmc_path(run)} not found — skipping '{run['label']}'")
        continue
    print(f"Loaded '{run['label']}': posterior shape {samples.shape}")
    loaded_runs.append({
        "samples": samples,
        "label": run["label"],
        "color": run["color"],
    })

if len(loaded_runs) == 0:
    raise RuntimeError("No MCMC chains found. Check RUNS configuration and file paths.")


# ==========================================
# CORNER PLOT
# ==========================================

print(f"\nPlotting {len(loaded_runs)} runs...")

fig = None

for i, run in enumerate(loaded_runs):
    kwargs = dict(
        labels=param_names if i == 0 else None,
        show_titles=False,
        smooth=1.0,
        smooth1d=1.0,
        color=run["color"],
        plot_datapoints=False,
        plot_density=False,
        fill_contours=False,
        no_fill_contours=True,
        levels=[0.68, 0.95],
        # Normalise each run's 1D marginal to unit total weight. corner leaves
        # matplotlib's default (raw COUNTS), and these chains differ in length by
        # ~4x (22.7k / 41.6k / 10.9k / 33.7k flat samples), so un-normalised
        # marginals differ in height purely by sample count, not by probability.
        #
        # Done with `weights` rather than hist_kwargs={"density": True}: with
        # smooth1d set, corner feeds hist_kwargs to ax.plot() (a Line2D), which
        # rejects `density`. Weights are applied when the histogram is built, so
        # they work on BOTH the plain and the smoothed code path.
        weights=np.full(len(run["samples"]), 1.0 / len(run["samples"])),
        hist_kwargs={"alpha": 0.4, "linewidth": 1.5},
        contour_kwargs={"linewidths": 1.5},
    )

    if fig is None:
        fig = corner.corner(run["samples"], **kwargs)
    else:
        corner.corner(run["samples"], fig=fig, **kwargs)

# Add legend manually (corner doesn't support it natively)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=run["color"], alpha=0.4, edgecolor=run["color"], label=run["label"])
    for run in loaded_runs
]
fig.legend(handles=legend_handles, loc="upper right", fontsize=12,
           frameon=True, framealpha=0.9, edgecolor="gray")

# Save to the run-specific plots dir using the same naming convention as
# main_evolution / main_training so the file is unambiguously tagged with
# the model identifier (helpful when comparing across sims / bestfits).
_save_dir = os.environ.get(
    "BAQARO_PLOTS_OUTDIR",
    comparison_plots_dir("mcmc_corner_comparison"),
)
os.makedirs(_save_dir, exist_ok=True)
_save_path = os.path.join(_save_dir, f"corner_comparison_{short_name}{notes_suffix()}.{fig_ext()}")
fig.savefig(_save_path, bbox_inches="tight")
print(f"saved: {_save_path}")

if os.environ.get("BAQARO_HEADLESS", "0") != "1":
    plt.show()
