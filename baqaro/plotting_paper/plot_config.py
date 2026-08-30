"""
Shared plotting configuration for the PAPER figures (``plotting_paper/``).

Every script in this folder imports this module to get:
  - consistent paper-quality matplotlib rcParams,
  - shared colormaps + a redshift-to-colour ``ScalarMappable``,
  - ``FIGURES_DIR`` — the git-tracked ``figures_paper/`` directory at the
    repository root (so the rendered figures travel with the repo),
  - ``save_figure(fig, name, ...)`` — writes ``<name>.pdf`` (and any other
    requested formats) into ``FIGURES_DIR`` and, unless headless, shows it.

Headless use
------------
On a display-less server set ``BAQARO_HEADLESS=1`` (or simply run without a
``DISPLAY``) and the module forces the non-interactive ``Agg`` backend and
``save_figure`` skips ``plt.show()``. Figures are always written to
``FIGURES_DIR`` regardless of backend.

This mirrors ``plotting_common/plot_config.py`` (same rcParams, colormaps,
and BHMF obs-marker convention) so existing scripts port over with only an
import swap, but adds the paper output-directory plumbing.
"""

import os
from pathlib import Path

import matplotlib

# ---------------------------------------------------------------------------
# Backend selection — must happen BEFORE pyplot is imported anywhere.
# ---------------------------------------------------------------------------
HEADLESS = os.environ.get("BAQARO_HEADLESS", "0") == "1" or not os.environ.get("DISPLAY")
if HEADLESS:
    matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# PDF is the default output format for the paper figures (vector, the standard
# for figures_paper/). The shared save_fig honours fig_ext(), which reads
# BAQARO_FIG_FORMAT (png otherwise); we setdefault it to pdf here so every paper
# script gets PDF without a flag. Override with BAQARO_FIG_FORMAT=png for a quick
# raster preview. (setdefault => an explicit BAQARO_FIG_FORMAT still wins.)
# ---------------------------------------------------------------------------
os.environ.setdefault("BAQARO_FIG_FORMAT", "pdf")

from matplotlib import pyplot as plt  # noqa: E402
from baqaro.utils import tol_colors  # noqa: E402


# ---------------------------------------------------------------------------
# Paper-quality matplotlib settings (same baseline as plotting_common).
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    # Slightly enlarged so labels/legends read larger relative to the
    # panel area; constrained_layout shrinks the data panel a hair to fit.
    "font.size": 14.0,
    "font.family": 'sans-serif',
    #        "font.sans-serif": ['Helvetica'],
    "axes.titlesize": 13.,
    "axes.labelsize": 13.,
    "xtick.labelsize": 12.,
    "ytick.labelsize": 12.,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.,
    "ytick.minor.size": 2.,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 12.0,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    #        "text.usetex": True
})


# ---------------------------------------------------------------------------
# Output directory: the git-tracked figures_paper/ at the repo root.
# This file lives at <repo>/baqaro/plotting_paper/plot_config.py,
# so parents[2] is the repo root.
# ---------------------------------------------------------------------------
_figures_dir_env = os.environ.get("BAQARO_FIGURES_DIR")
if _figures_dir_env:
    FIGURES_DIR = Path(_figures_dir_env).expanduser().resolve()
else:
    FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures_paper"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared colormaps + redshift colorbar mapper.
#
# ``z_scalar_mapper`` (vmax=7) is the scale used by the population-statistics
# figures (qlf / qhmf / cerdf / clustering), whose science range stops at z=7 —
# keep it fixed so those colours never move.
#
# The model-ingredient figures that sample many redshifts (halo rate, f_cold)
# instead span the full ``REDSHIFT_TARGETS`` range (z up to 8.7) and build a
# local ``Normalize`` from it via ``cmap_z`` — see ``plotting_halo_rate.py`` /
# ``plotting_models_fcold.py``. Sharing the ``REDSHIFT_TARGETS`` list keeps
# those two figures sampling (and colouring) the same set of redshifts.
# ---------------------------------------------------------------------------
cmap_z = tol_colors.tol_cmap(colormap='rainbow_PuRd')
cmap_M = matplotlib.colormaps.get_cmap("cividis")

z_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=7.0)
z_scalar_mapper = matplotlib.cm.ScalarMappable(cmap=cmap_z, norm=z_norm)

# Redshifts the model-ingredient figures colour-code at (high -> low); matches
# the hardcoded list in plotting_halo_rate.py so f_cold samples the same z.
REDSHIFT_TARGETS = [8.7, 7.3, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]


# ---------------------------------------------------------------------------
# BHMF observational-data styling (ported from plotting_common.plot_config so
# future paper BHMF panels stay consistent — see that file for the rationale).
# ---------------------------------------------------------------------------
BHMF_OBS_MARKERS = {
    "He": "s",            # He et al. 2024 (HSC+SDSS, z~4)
    "Lai": "o",           # Lai et al. 2024 (XQz5+, z~5)
    "Wu": "^",            # Wu, J. et al. 2022 (z~6)
    "Matthee": "D",       # Matthee et al. 2024 (JWST LRD, z~5)
    "Taylor": "X",        # Taylor et al. 2025 (JWST BLAGN, 3.5<z<6)
    "Kelly&Shen13": "v",  # Kelly & Shen 2013
}
BHMF_OBS_DEFAULT_MARKER = "s"


def bhmf_obs_marker(label):
    """Return a marker style for a BHMF obs dataset given its label.

    The leading author token is parsed from ``label`` (text before the first
    space or '+'), then looked up in ``BHMF_OBS_MARKERS``.
    """
    token = label.split("+")[0].split()[0]
    return BHMF_OBS_MARKERS.get(token, BHMF_OBS_DEFAULT_MARKER)


# ---------------------------------------------------------------------------
# Saving helper — single choke-point so every paper figure lands in the same
# committed directory with the same defaults.
# ---------------------------------------------------------------------------
def save_figure(fig, name, formats=("pdf",), show=None):
    """Save ``fig`` into ``FIGURES_DIR`` as ``<name>.<ext>`` for each format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        Basename without extension (e.g. ``"halo_accretion_rate"``).
    formats : iterable of str
        Extensions to write (default ``("pdf",)``; pass e.g. ``("pdf", "png")``).
    show : bool or None
        Force ``plt.show()`` on/off. ``None`` (default) shows only when not
        headless.

    Returns
    -------
    list[pathlib.Path]
        The paths written.
    """
    paths = []
    for ext in formats:
        out = FIGURES_DIR / f"{name}.{ext}"
        fig.savefig(out)
        paths.append(out)
        print(f"[plotting_paper] wrote {out}")

    do_show = (not HEADLESS) if show is None else show
    if do_show:
        plt.show()
    return paths
