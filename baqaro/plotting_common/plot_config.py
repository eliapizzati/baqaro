"""Shared matplotlib configuration and figure-saving policy.

Imported (often only for its side effect) by every figure-producing module, so
that working figures and publication figures share one set of rcParams,
colour schemes and redshift normalisation rather than drifting apart.

Provides:

* **rcParams and colour schemes** -- including the colour-blind-safe Paul Tol
  palettes vendored in :mod:`~baqaro.utils.tol_colors`, and
  ``z_scalar_mapper`` / ``cmap_z`` for colouring curves by redshift
  consistently across figures.
* **``save_fig(fig, path, name)``** -- saving is OPT-IN: it writes nothing
  unless ``BAQARO_SAVE_FIGS=1``. That way a script can be imported or run
  interactively without scattering files, and a batch run that means to save
  has to say so. Format follows ``BAQARO_FIG_FORMAT`` (``png`` by default for
  quick viewing; set ``pdf`` for publication output).
* **``maybe_show()``** -- shows the figure unless running headless.

``LOCAL_PLOTS_DIR`` is the default destination, resolved relative to this
file's location so it lands in ``<repo>/plots`` regardless of the working
directory.
"""

import numpy as np
import os
import matplotlib
from baqaro.utils import tol_colors


def fig_ext():
    """File extension for saved figures.

    Default ``png`` (fast, viewable for everyday/interactive work). Set
    ``BAQARO_FIG_FORMAT=pdf`` for production / vector output (papers,
    talks). Recognised: png, pdf, svg, eps; anything else -> png.
    """
    ext = os.environ.get("BAQARO_FIG_FORMAT", "png").strip().lower().lstrip(".")
    return ext if ext in ("png", "pdf", "svg", "eps") else "png"


# Repo-local plots dir (default save destination). plot_config.py lives at
# <repo>/baqaro/plotting_common/plot_config.py, so parents[2] is
# the repo root that holds plots/ and figures_paper/.
LOCAL_PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plots",
)


# ---------------------------------------------------------------------------
# Named figure sizes
# ---------------------------------------------------------------------------
# There was no convention: 24 distinct `figsize=` literals across the working
# figures and 20 across plotting_paper, several differing only in formatting
# ((6, 5) vs (6.0, 5.0)). These name the sizes that actually recur,
# so a new figure picks an existing one instead of inventing a 25th.
#
# PUBLICATION sizes are sized for the paper's column/page width; SCREEN sizes
# are for the working figures, which are meant to be read on a monitor and are
# deliberately larger. Neither is "right" — pick by where
# the figure is going.
FIG_SINGLE = (6.0, 5.0)        #: one panel, publication
FIG_WIDE_2PANEL = (10.5, 4.6)  #: two panels side by side, publication
FIG_CORNER = (5.6, 5.6)        #: square — corner plots
FIG_SCREEN_WIDE = (15.0, 7.0)  #: multi-panel grid, screen/working figures
FIG_SCREEN_MED = (8.0, 6.0)    #: medium, screen/working figures


def resolve_plots_dir(path_plots=None, subdir=None, source=None):
    """The single place that decides WHERE a non-paper figure is written.

    Use this instead of calling ``get_plots_path()`` inline: scripts that build
    their own destination drift away from the rest of the pipeline, which is how
    figures ended up split across two trees.

    The rule, matching :func:`save_fig`:
      * default → the repo-local ``plots/`` dir (gitignored, co-located), or
        ``BAQARO_LOCAL_PLOTS_DIR`` if set;
      * ``BAQARO_PLOTS_ON_DATA3=1`` → the production plots tree
        (``path_plots`` if the caller passed one, else ``get_plots_path``).

    ``plotting_paper`` is deliberately outside this rule: its figures are the
    publication set and always go to the git-tracked ``figures_paper/`` via
    ``plotting_paper.plot_config.save_figure``.

    Parameters
    ----------
    path_plots : str, optional
        Explicit production-tree root. Resolved from ``get_plots_path(source)``
        when omitted and ``BAQARO_PLOTS_ON_DATA3`` is on.
    subdir : str, optional
        Leaf directory appended to the destination (e.g. ``"training_diagnostics"``).
    source : str, optional
        Passed to ``get_plots_path`` when it has to be resolved.

    Returns
    -------
    str
        The directory, already created.
    """
    if os.environ.get("BAQARO_PLOTS_ON_DATA3", "0") == "1":
        if path_plots is None:
            from baqaro.utils.my_dir import get_plots_path
            path_plots = get_plots_path(source=source)
        dest = path_plots
    else:
        dest = os.environ.get("BAQARO_LOCAL_PLOTS_DIR") or LOCAL_PLOTS_DIR
    if subdir:
        dest = os.path.join(dest, subdir)
    os.makedirs(dest, exist_ok=True)
    return dest

def figure_rng(seed=None):
    """Seeded RNG for figures that convolve the model with a measurement error.

    Several comparison figures blur the model by the observational scatter (e.g.
    the ~0.3 dex virial-mass error) before histogramming it against the data.
    That draw MUST be reproducible: with a bare ``np.random.normal`` every run
    produces a different realisation, so the figure cannot be regenerated and a
    genuine change cannot be told apart from a fresh draw.

    Override with ``BAQARO_FIG_SEED`` to inspect realisation-to-realisation
    scatter deliberately; leave it unset for the reproducible default.
    """
    if seed is None:
        seed = int(os.environ.get("BAQARO_FIG_SEED", "42"))
    return np.random.default_rng(seed)


def annotate_panel_z(ax, z_label, z_model, snap=None, *, x=0.95, y=0.95,
                     ha="right", va="top", fontsize=12, dy=None,
                     color="black", sub_color="0.35", bbox=None,
                     as_title=False, **kw):
    """Panel redshift label with the MODEL's actual snapshot redshift beneath it.

    A panel is labelled with the redshift of the data it is compared against,
    but the model curve is drawn at whichever stored snapshot is nearest — and
    on a reduced-snapshot run that is not the same number. The fiducial forward
    run keeps 41 of 145 snapshots, so "z = 7" is drawn at z=7.005, "z = 6" at
    z=5.879, and a QLF/QHMF panel pinned to the emulator ladder at z=6.708.
    Printing both, one under the other, makes each panel say which snapshot
    produced its curve instead of leaving it to be inferred.

    Working figures only. ``plotting_paper`` deliberately labels with the
    nominal redshift alone — that is what the plotted DATA is, and the paper
    text carries the snapshot table.

    Parameters
    ----------
    z_label : str
        The comparison label, drawn bold on the first line (e.g. ``"z = 7.0"``).
    z_model : float
        Redshift of the snapshot the model curve was actually taken from.
    snap : int, optional
        True snapshot number, appended to the second line when given.
    x, y, ha, va : positioning, as for :meth:`~matplotlib.axes.Axes.text`, in
        axes coordinates. The second line is placed below the first for both
        ``va="top"`` and ``va="bottom"``.
    as_title : bool
        Draw as a column header above the axes instead of inside it, for grids
        that label their columns with ``set_title`` rather than in-panel text.

    Returns
    -------
    tuple
        The two ``Text`` artists (main, sub).
    """
    if dy is None:
        dy = 0.055 * (fontsize / 12.0) + 0.02
    sub = f"model: z = {z_model:.3f}"
    if snap is not None:
        sub += f"  (snap {int(snap)})"
    if as_title:
        # Column-header form: the sub-line sits just above the axes and the
        # padded title above that, so the order on the page is still
        # comparison-redshift then model-redshift.
        ax.set_title(z_label, fontsize=fontsize, pad=fontsize + 6)
        t_sub = ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=fontsize * 0.72, color=sub_color,
                        **kw)
        return ax.title, t_sub
    y_main, y_sub = (y, y - dy) if va == "top" else (y + dy, y)
    t_main = ax.text(x, y_main, z_label, transform=ax.transAxes,
                     fontsize=fontsize, fontweight="bold", color=color,
                     ha=ha, va=va, bbox=bbox, **kw)
    t_sub = ax.text(x, y_sub, sub, transform=ax.transAxes,
                    fontsize=fontsize * 0.72, color=sub_color,
                    ha=ha, va=va, bbox=bbox, **kw)
    return t_main, t_sub


def save_fig(fig, path_plots, name, dpi=200, bbox_inches="tight", force_dir=False):
    """Standard figure save for every plotting script.

    Writes ``{dest}/{name}.{fig_ext()}`` (``name`` may include a subdirectory,
    which is created) **only when ``BAQARO_SAVE_FIGS=1``** — saving is opt-in so
    bare interactive runs just display. Format follows ``fig_ext()`` (png
    default; ``BAQARO_FIG_FORMAT=pdf`` for production).

    Destination:
      * default → the repo-local ``plots/`` dir (``LOCAL_PLOTS_DIR``) — fast,
        co-located, gitignored.
      * ``BAQARO_PLOTS_ON_DATA3=1`` → the caller's ``path_plots`` argument — the
        production plots tree for the pipeline scripts.
      * ``force_dir=True`` → ALWAYS the caller's ``path_plots``, ignoring the
        local-default / ``BAQARO_PLOTS_ON_DATA3`` toggle. Used by ``plotting_paper``
        so paper figures always land in the git-tracked ``figures_paper/`` dir.

    Returns the path written, or ``None`` if saving was skipped.
    """
    if os.environ.get("BAQARO_SAVE_FIGS", "0") != "1":
        return None
    if force_dir:
        # force_dir bypasses the standard entirely: plotting_paper needs its
        # figures in the git-tracked figures_paper/ dir no matter what.
        dest = path_plots
    else:
        # Same rule as resolve_plots_dir(), delegated so there is exactly one
        # implementation of "where do figures go".
        dest = resolve_plots_dir(path_plots=path_plots)
    ext = fig_ext()
    out = os.path.join(dest, "{}.{}".format(name, ext))
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out, bbox_inches=bbox_inches, dpi=dpi if ext == "png" else None)
    print(" -> {}".format(out), flush=True)
    return out


def maybe_show():
    """Call ``plt.show()`` unless ``BAQARO_HEADLESS=1`` (batch / SSH runs).

    The one display call each script makes at the very end, so headless
    pipeline runs don't block on a GUI window.
    """
    import matplotlib.pyplot as plt
    if os.environ.get("BAQARO_HEADLESS", "0") != "1":
        plt.show()


# Matplotlib global settings
matplotlib.rcParams.update({
    "font.size": 13.0,
    "font.family": 'sans-serif',
    #        "font.sans-serif": ['Helvetica'],
    "axes.titlesize": 12.,
    "axes.labelsize": 12.,
    "xtick.labelsize": 12.,
    "ytick.labelsize": 12.,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.,
    "ytick.minor.size": 2.,
    "legend.fontsize": 12.0,
    "legend.frameon": False,
    #       "figure.dpi": 200,
    "savefig.dpi": 150,
    #        "text.usetex": True
})


# Shared colormaps
cmap_z = tol_colors.tol_cmap(colormap='rainbow_PuRd')
cmap_M = matplotlib.colormaps.get_cmap("cividis")


# Efficient colorbar setup: Create once, reuse everywhere
z_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=7.0)
z_scalar_mapper = matplotlib.cm.ScalarMappable(cmap=cmap_z, norm=z_norm)


# --------------------------------------------------------------------------
# BHMF observational-data styling
# --------------------------------------------------------------------------
# Per-source marker so that different papers can be told apart in a single
# redshift panel (where every dataset is otherwise drawn in the same
# redshift-coded colour). Keyed by the leading author token of the dataset
# label (e.g. "He+2024 z=4.0" -> "He"). Used wherever several BHMF datasets
# share one redshift panel.
BHMF_OBS_MARKERS = {
    "He": "s",            # He et al. 2024 (HSC+SDSS, z~4)
    "Lai": "o",           # Lai et al. 2024 (XQz5+, z~5)
    "Wu": "^",            # Wu, J. et al. 2022 (z~6)
    "Matthee": "D",       # Matthee et al. 2024 (JWST LRD, z~5)
    "Taylor": "X",        # Taylor et al. 2025 (JWST BLAGN, 3.5<z<6)
    "Kelly&Shen13": "v",  # Kelly & Shen 2013
    "Schulze": "P",       # Schulze & Wisotzki 2010 (active BHMF, z~0)
}
BHMF_OBS_DEFAULT_MARKER = "s"


def bhmf_obs_marker(label):
    """Return a marker style for a BHMF obs dataset given its label.

    The leading author token is parsed from ``label`` (text before the
    first space or '+'), then looked up in ``BHMF_OBS_MARKERS``.
    """
    token = label.split("+")[0].split()[0]
    return BHMF_OBS_MARKERS.get(token, BHMF_OBS_DEFAULT_MARKER)
