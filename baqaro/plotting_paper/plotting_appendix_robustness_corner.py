"""PAPER FIGURE (appendix): posterior robustness corner.

Overlays the fiducial qcc posterior with every systematics/robustness VARIATION
chain, to show the 6 parameters are stable under the analysis choices we scanned:
  * temperature   T = 1000, 10000   (fiducial 5000)
  * QLF faint cut  min-L +/- 0.5 dex
  * QLF sys-err    floor = 0.2, 0.4  (fiducial 0.3)
  * z=4 clustering  excluded / all-fields (x1) / all-fields (x2 err)

Same size + layout + style as the main-text corner (`plotting_mcmc_corner.py`):
6-parameter triangle, figsize 5.6x5.6, paper labels, log M_seed = 10.5 + logfseed
reparam, prior-box axis ranges. The FIDUCIAL is a filled reference band; every
variation is a 68%-contour line, coloured by family (z=4 greens, min-L oranges,
temperature pinks, sys-err blues). Missing chains are skipped with a warning
(so the figure renders during a partial run).

Chains share the fiducial emulator identity (name_file_mcmc); they differ only in
the MCMC-notes tag. Reads the chains directly from mcmc_dir.
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

from baqaro.plotting_paper import fiducial_data as fd
from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show

name_fig = os.environ.get("BAQARO_PAPER_ROBUSTNESS_NAME", "appendix_robustness_corner")

# --- parameters, labels, reparam (identical to plotting_mcmc_corner) ----------
LOG_MSEED_OFFSET = 10.5
labels = [r"$\log\,\eta_{\rm av,0}$", r"$\eta_{\rm av,slope}$", r"$\sigma_{\rm acc}$",
          r"$\log(\tau_{\rm coh}/{\rm yr})$", r"$\log\,M_{\rm seed}$", r"$\sigma_{\rm seed}$"]
prange = [(-2.5, -0.5), (0.2, 2.0), (0.2, 1.0), (3.5, 7.3),
          (-7.5 + LOG_MSEED_OFFSET, -3.5 + LOG_MSEED_OFFSET), (0.2, 1.5)]

# --- the runs. (label, mcmc-notes, colour, is_fiducial?) ----------------------
# Fiducial = thick BLACK contour lines (68+95%); variations = thin coloured
# lines (68%), grouped by systematic family so the legend reads as 3 blocks:
#   QLF sys-err (blues) | z=4 clustering (greens) | cERDF temperature (warm).
#
# Named sets — pick with BAQARO_ROBUSTNESS_SET (default "ck22final", the
# adopted fiducial `qcc_ck22final_v1`). Older sets need their own emulator
# identity via the BAQARO_PAPER_MCMC_* overrides. The variation tags differ
# between suites (e.g. `z4err1` vs `z4allf` for the z=4-original-errors run),
# so each set is an explicit list rather than a prefix swap.
_RUN_SETS = {
    "unbT5000": [
        ("Fiducial (QLF sys. error = 0.3,\n"
         r"Corr $z{=}4$ errors $\times 2$, cERDF $T = 5000$)",
         "unbT5000", "black", True),
        ("QLF, sys. error = 0.2",           "unbT5000_syserr0p2", "#88CCEE", False),
        ("QLF, sys. error = 0.4",           "unbT5000_syserr0p4", "#4477AA", False),
        (r"Corr, $z{=}4$ excluded",         "unbT5000_noz4",      "#117733", False),
        (r"Corr, $z{=}4$ original errors",  "unbT5000_z4allf",    "#44AA99", False),
        (r"cERDF, $T = 1000$",              "unbT5000_T1000",     "#EE7733", False),
        (r"cERDF, $T = 10000$",             "unbT5000_T10000",    "#CC3311", False),
    ],
    "ck22g621": [
        ("Fiducial (QLF sys. error = 0.3,\n"
         r"Corr $z{=}4$ errors $\times 2$, cERDF $T = 5000$)",
         "cleanK22_g6.21_10k", "black", True),
        ("QLF, sys. error = 0.2",           "ck22g621_syserr0p2", "#88CCEE", False),
        ("QLF, sys. error = 0.4",           "ck22g621_syserr0p4", "#4477AA", False),
        (r"Corr, $z{=}4$ excluded",         "ck22g621_noz4",      "#117733", False),
        (r"Corr, $z{=}4$ original errors",  "ck22g621_z4err1",    "#44AA99", False),
        (r"cERDF, $T = 1000$",              "ck22g621_T1000",     "#EE7733", False),
        (r"cERDF, $T = 10000$",             "ck22g621_T10000",    "#CC3311", False),
    ],
    # ck22final = FINAL 8192-pt emulators (clean_z0_g6.21_K22_final_smooth). Same
    # structure as ck22g621; baseline chain tag ck22final_10k, variations
    # ck22final_<var>. Render with
    # BAQARO_PAPER_MCMC_NOTES_FILE_EMULATION=clean_z0_g6.21_K22_final_smooth.
    "ck22final": [
        ("Fiducial (QLF sys. error = 0.3,\n"
         r"Corr $z{=}4$ errors $\times 2$, cERDF $T = 5000$)",
         "ck22final_10k", "black", True),
        ("QLF, sys. error = 0.2",           "ck22final_syserr0p2", "#88CCEE", False),
        ("QLF, sys. error = 0.4",           "ck22final_syserr0p4", "#4477AA", False),
        (r"Corr, $z{=}4$ excluded",         "ck22final_noz4",      "#117733", False),
        (r"Corr, $z{=}4$ original errors",  "ck22final_z4err1",    "#44AA99", False),
        (r"cERDF, $T = 1000$",              "ck22final_T1000",     "#EE7733", False),
        (r"cERDF, $T = 10000$",             "ck22final_T10000",    "#CC3311", False),
    ],
    # _ml = augmented (8191-pt) emulators.
    # Render with BAQARO_PAPER_MCMC_NOTES_FILE_EMULATION=clean_z0_g6.21_K22_ml_smooth.
    "ck22g621ml": [
        ("Fiducial (QLF sys. error = 0.3,\n"
         r"Corr $z{=}4$ errors $\times 2$, cERDF $T = 5000$)",
         "cleanK22_g6.21_ml_10k", "black", True),
        ("QLF, sys. error = 0.2",           "ck22g621ml_syserr0p2", "#88CCEE", False),
        ("QLF, sys. error = 0.4",           "ck22g621ml_syserr0p4", "#4477AA", False),
        (r"Corr, $z{=}4$ excluded",         "ck22g621ml_noz4",      "#117733", False),
        (r"Corr, $z{=}4$ original errors",  "ck22g621ml_z4err1",    "#44AA99", False),
        (r"cERDF, $T = 1000$",              "ck22g621ml_T1000",     "#EE7733", False),
        (r"cERDF, $T = 10000$",             "ck22g621ml_T10000",    "#CC3311", False),
    ],
}
# Default tracks the adopted fiducial (qcc_ck22final_v1). Older sets are still
# available via BAQARO_ROBUSTNESS_SET=<name> (+ the BAQARO_PAPER_MCMC_* overrides
# that repoint the emulator identity).
RUNS = _RUN_SETS[os.environ.get("BAQARO_ROBUSTNESS_SET", "ck22final")]

_base = fd.name_file_mcmc          # emulator identity shared by every chain
_combo = "qlf+cerdf+corr"


# Only overlay COMPLETE chains. A partial chain (still sampling) has an
# artificially BROAD marginal — its walkers haven't reached the stationary
# distribution — which, because corner area-normalises each 1D histogram, shows
# up as a spuriously LOW/wide curve. Require essentially the full 10k steps so a
# half-run variation never appears mid-convergence. Lower via the env only for
# a deliberate quick preview.
MIN_STEPS = int(os.environ.get("BAQARO_ROBUSTNESS_MIN_STEPS", "9500"))


def _load(notes):
    p = os.path.join(fd.mcmc_dir, f"mcmc_{_base}_{_combo}_{notes}.h5")
    if not os.path.exists(p):
        print(f"  [robustness] SKIP (missing): {notes}")
        return None
    try:
        with h5py.File(p, "r") as f:
            # emcee's HDFBackend PRE-ALLOCATES the full (nsteps, nwalkers, ndim)
            # array and fills it as it runs; the unwritten tail is all zeros
            # (which pass an isfinite check). Use the real filled count from the
            # `iteration` attr so a still-running chain isn't read as complete.
            nfilled = int(f["mcmc"].attrs.get("iteration", f["mcmc/chain"].shape[0]))
            ch = f["mcmc/chain"][:nfilled]
    except (OSError, KeyError) as e:
        print(f"  [robustness] SKIP (unreadable, likely mid-write): {notes} ({e})")
        return None
    if nfilled < MIN_STEPS:
        print(f"  [robustness] SKIP (only {nfilled} steps < {MIN_STEPS}, still running): {notes}")
        return None
    s = ch[nfilled // 2:].reshape(-1, ch.shape[2]).copy()
    if s.shape[0] < 100:
        print(f"  [robustness] SKIP (too few post-burn samples): {notes}")
        return None
    s[:, 4] += LOG_MSEED_OFFSET
    return s


fig = None
drawn = []
for lbl, notes, col, filled in RUNS:
    s = _load(notes)
    if s is None:
        continue
    # No fills, and ALL chains show the SAME single 68% contour (a filled/2-contour
    # fiducial next to 1-contour variations reads inconsistently). The fiducial is
    # distinguished only by a thicker BLACK line, variations by thin coloured lines.
    kw = dict(
        labels=labels if fig is None else None,
        label_kwargs={"fontsize": 11, "labelpad": 6},
        range=prange, smooth=1.2, smooth1d=1.2, color=col,
        plot_datapoints=False, plot_density=False,
        fill_contours=False, no_fill_contours=True,
        levels=[0.68],
        hist_kwargs={"alpha": 0.95, "linewidth": 2.4 if filled else 1.3},
        contour_kwargs={"linewidths": 2.4 if filled else 1.3,
                        "alpha": 1.0 if filled else 0.9},
        max_n_ticks=3, show_titles=False,
    )
    try:
        if fig is None:
            fig = corner.corner(s, fig=plt.figure(figsize=(5.6, 5.6)), **kw)
        else:
            corner.corner(s, fig=fig, **kw)
    except ValueError as e:
        print(f"  [robustness] corner FAILED for {notes}: {e}")
        continue
    drawn.append((lbl, col, filled))

# Shrink tick labels so the 45-deg rotated ticks don't collide with the axis
# labels at this compact (fig-corner) size.
for ax in fig.get_axes():
    ax.tick_params(labelsize=7)
    for lab in ax.get_xticklabels():
        lab.set_rotation(45); lab.set_ha("right")
    for lab in ax.get_yticklabels():
        lab.set_rotation(45); lab.set_va("top")

# legend + layout matching plotting_mcmc_corner (upper-right, markerfirst=False)
handles = [Line2D([], [], color=c, lw=2.4 if f else 1.6, label=l) for l, c, f in drawn]
_leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.955, 0.985),
                  fontsize=10, frameon=False, markerfirst=False)
# right-align the label text (flush against the handles), incl. the 2-line fiducial
for _t in _leg.get_texts():
    _t.set_ha("right")
    _t.set_multialignment("right")
fig.subplots_adjust(left=0.135, right=0.985, top=0.985, bottom=0.125,
                    hspace=0.07, wspace=0.07)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
