"""
PAPER FIGURE: single merger-tree lightcurve (sub-step BH history).

Three stacked panels sharing a cosmic-time x-axis (with a redshift top axis):
  * top    — the merger tree of one target BH back to its progenitors, each
             branch coloured/thickened by log10 M_BH;
  * middle — the target's bolometric lightcurve L_bol (left axis) and BH mass
             M_BH (right axis) on the fine sub-step grid;
  * bottom — the target's Eddington ratio lambda_Edd, over a 16-84% reference
             band of the whole first-born target population.

DATA — note this figure does NOT use the main paper fiducial. Sub-step tracks
come from a SEPARATE, small ``main_evolution_full_history`` run (a first-born
selection at z=0), not the big root144 subsample forward run. That dedicated
full-history fiducial is pinned in ``plotting_paper/fiducial_data.py`` as
``FULL_HISTORY_FIDUCIAL`` and resolved to ``path_file_full_history_fid`` (token
``firstborn100``, bestfit inherited from ``PAPER_FIDUCIAL``, no subset tag). Override any
token with ``BAQARO_PAPER_FH_<X>``. This is a single-object figure, so there is
no population number-density estimate and no subsample weighting applies.

Which tree to draw:
  * ``BAQARO_PAPER_TREE_RANKS`` — comma list (e.g. "0,1,7,11"): SCAN mode, one
    figure per rank saved as ``full_history_single_tree_rank{r}`` so you can
    pick the best-looking tree;
  * ``BAQARO_PAPER_TREE_IDX``  — a direct full-history array index (single figure);
  * ``BAQARO_PAPER_TREE_RANK`` — else the rank-th first-born target. **The paper
    fiducial is rank 14.**

Tree de-cluttering (dense trees don't fit the panel): short leaf branches are
pruned via ``BAQARO_PAPER_TREE_MIN_LEN`` (lifetime in snapshots, default 45; 0 =
off) and optionally ``BAQARO_PAPER_TREE_MIN_LOGM`` (peak log M_BH, default off —
note BH mass is monotonic so progenitors that merge while small get stripped
aggressively, usually leaving only the backbone).

Adapted for the paper from the working version of this figure
(same tree walker / layout / sub-step colour scheme); the changes are the
data source (paper full-history fiducial), the tree selection env vars,
z=0-inclusive redshift ticks, and routing the save to the git-tracked
``figures_paper/`` (PDF by default).
"""

import os

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import NullLocator

from qhtools.utils.cosmology import cosmo
import qhtools.utils.natconst as nc
from qhtools.utils import my_utils

# Paper full-history fiducial (NOT load_data_to_plot — see fiducial_data.py).
from baqaro.plotting_paper.fiducial_data import (
    load_simulation_metadata_full_history,
    path_file_full_history_fid,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show

# This figure embeds large RASTERIZED data layers (the tree LineCollection +
# the sub-step lightcurves). Their resolution in the vector PDF is set by
# savefig.dpi (plot_config defaults it to 300, which looks soft). Bump to 600 so
# the rasterized content is crisp while text/axes stay vector.
plt.rcParams["savefig.dpi"] = 600
# The figure is compact (~6.8x5.7"), so the default 12-13 pt rcParams text reads
# too large here — dial it down for this script only.
plt.rcParams.update({
    "font.size": 9,         # in-panel labels (L_bol, M_BH, eta_acc, lambda_Edd)
    "axes.labelsize": 9,    # axis titles
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "full_history_single_tree"

# Redshift ticks for the top axis — z=0-inclusive (the fiducial reaches z=0).
Z_TICKS = [0, 0.5, 1, 2, 3, 5, 8, 15]

# Sub-step track colours.
col_lbol = "#06437F"     # steel blue   (L_bol)
col_mass = "#a00c1d"     # dark red     (M_BH)
col_halo = "#2e7d32"     # green        (M_halo / 1e5, middle-panel overlay)
col_eta = "#5e4fa2"      # muted purple (lambda_Edd, radiative)
col_eta_acc = "#9a9a9a"  # light grey   (eta_acc, accretion-rate Eddington ratio)

# Truncated inferno for the tree branches (shared with merger_trees_single).
# Top clipped at 0.95 (saturated yellow) rather than 1.0: inferno's extreme end
# is near-white (#fcffa4), so high-mass (~1e9 Msun) branches became
# indistinguishable from the white background.
_inferno = plt.cm.inferno
_colors_trunc = _inferno(np.linspace(0.15, 0.95, 256))
cmap_tree = LinearSegmentedColormap.from_list("inferno_trunc", _colors_trunc)


# ==============================================================================
# 1. REVERSE INDEX BUILDER
# ==============================================================================
def build_reverse_map_on_the_fly(track_ids, descendant_ids):
    """Invert descendant pointers into a descendant -> [progenitors] map.

    Built here rather than loaded because this figure needs only one tree, so
    materialising the full reverse index would cost far more than it saves.
    """
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    progenitor_map = {}
    for i, desc_id in enumerate(descendant_ids):
        if desc_id in id_to_idx:
            desc_idx = id_to_idx[desc_id]
            if desc_idx not in progenitor_map:
                progenitor_map[desc_idx] = []
            progenitor_map[desc_idx].append(i)
    return progenitor_map


# ==============================================================================
# 2. TREE WALKER
# ==============================================================================
def build_tree_nodes(target_array_idx, progenitor_map, snap_birth, snap_death):
    """Breadth-first walk of one merger tree from a target halo backwards.

    Returns the node list (index, depth, parent) for every progenitor reachable
    from ``target_array_idx``, used to lay the tree out for drawing.
    """
    tree_nodes = []
    queue = [(target_array_idx, 0, -1)]
    visited = set()

    while queue:
        curr_idx, depth, parent_node_idx = queue.pop(0)
        if curr_idx in visited:
            continue
        visited.add(curr_idx)

        tree_nodes.append({
            "array_idx": curr_idx,
            "depth": depth,
            "birth": snap_birth[curr_idx],
            "death": snap_death[curr_idx],
            "parent_node_idx": parent_node_idx,
        })

        if curr_idx in progenitor_map:
            for prog_idx in progenitor_map[curr_idx]:
                queue.append((prog_idx, depth + 1, len(tree_nodes) - 1))

    return tree_nodes


def prune_small_branches(tree_nodes, min_len, min_log_mbh, max_snap, loader):
    """Iteratively drop *small* leaf branches so dense trees stay legible.

    A leaf (no children in the tree) is removed if it is too short-lived
    (lifetime ``death - birth`` < ``min_len`` snapshots) OR too low-mass (peak
    ``log10 M_BH`` < ``min_log_mbh``; BH mass is ~monotonic so the peak is the
    mass at the branch's last snapshot). Repeating trims whole bushy twig
    layers from the tips inward without ever orphaning a surviving branch — the
    root (node 0) and any node with children are always kept. The mass cut is
    the effective declutter knob: it strips the swarm of faint low-mass
    progenitors while keeping the massive backbone and significant mergers.
    Both cuts off (``min_len<=0`` and ``min_log_mbh=-inf``) -> no-op.
    """
    if (min_len <= 0 and min_log_mbh == -np.inf) or not tree_nodes:
        return tree_nodes

    def peak_logm(n):
        # Read at death-1, NOT at the death snapshot. The per-snapshot BH array
        # is ZERO at a track's death snap for every MERGED track — production
        # zero-inits each snapshot row and writes only live entries, and a dying
        # track is excluded from the evolving set (its mass lives on in the
        # descendant). death-1 is the last snapshot the BH was actually alive.
        s = (n["death"] - 1) if n["death"] != -1 else max_snap
        s = min(max(s, 0), max_snap)
        m = loader.get_BH_mass(s)[n["array_idx"]]
        return np.log10(m) if m > 0 else -np.inf

    while True:
        children = {i: [] for i in range(len(tree_nodes))}
        for i, n in enumerate(tree_nodes):
            p = n["parent_node_idx"]
            if p != -1:
                children[p].append(i)

        remove = set()
        for i, n in enumerate(tree_nodes):
            if i == 0 or children[i]:          # keep root + any node with kids
                continue
            birth = n["birth"]
            if birth == -1:
                remove.add(i)
                continue
            death = n["death"] if n["death"] != -1 else max_snap
            too_short = (death - birth) < min_len
            too_faint = peak_logm(n) < min_log_mbh
            if too_short or too_faint:
                remove.add(i)

        if not remove:
            return tree_nodes

        keep = [i for i in range(len(tree_nodes)) if i not in remove]
        remap = {old: new for new, old in enumerate(keep)}
        pruned = []
        for old in keep:
            n = dict(tree_nodes[old])
            p = n["parent_node_idx"]
            n["parent_node_idx"] = remap[p] if (p != -1 and p in remap) else -1
            pruned.append(n)
        tree_nodes = pruned


# ==============================================================================
# 3. PLOTTING HELPERS
# ==============================================================================
_TIME_AT_Z_MIN = cosmo.age(0.001)


def time_to_redshift(time):
    """Cosmic time [Gyr] -> redshift, clamped to the axis range."""
    t = np.asanyarray(time)
    t_clamped = np.clip(t, 0.002, _TIME_AT_Z_MIN)
    return cosmo.age(t_clamped, inverse=True)


def redshift_to_time(z):
    """Redshift -> cosmic time [Gyr], clamped away from z=0 singularities."""
    z_val = np.asanyarray(z)
    z_clamped = np.maximum(z_val, 0.001)
    return cosmo.age(z_clamped)


def add_redshift_axis(ax, z_ticks=None):
    """Add a secondary top axis labelled in redshift to a time-axis plot."""
    ax3 = ax.secondary_xaxis("top", functions=(time_to_redshift, redshift_to_time))
    ax3.set_xlabel(r"Redshift $z$", labelpad=0)
    if z_ticks is None:
        z_ticks = Z_TICKS
    ax3.set_xticks(z_ticks)
    ax3.set_xticklabels([f"{z:g}" for z in z_ticks])
    # Drop the auto minor ticks. On a secondary axis with a nonlinear
    # time<->redshift mapping the minor locator places them at uniform *time*
    # steps, which land at irregular, hard-to-read redshift positions (bunched
    # at high z). NullLocator removes them entirely (minorticks_off alone does
    # not stick on a secondary axis). The only ticks left are the labelled
    # majors at Z_TICKS.
    ax3.xaxis.set_minor_locator(NullLocator())
    ax3.tick_params(axis="x", which="minor", length=0)  # belt-and-suspenders
    return ax3


def get_layout(nodes, spacing, root_lane=0.9, top_scale=1.0, below_scale=1.0):
    """Vertical positions for each branch.

    ``spacing[i]`` is the vertical slot height for node *i* (driven by its
    thickness): thin/low-mass branches get a small slot so they pack tightly,
    thick/massive branches get a large slot so they breathe. The MAIN branch
    (root) gets its OWN dedicated lane of height ``root_lane`` (its children
    split below/above with that gap reserved) so no sub-branch runs along it,
    without grabbing the full thick-branch slot. The whole layout is then
    mirrored vertically (flip of the above/below regions).

    ``top_scale`` / ``below_scale`` (<1) multiply the spacing of the root's TOP
    and BELOW child-groups (the halves that end up at the top / bottom after the
    flip), compressing each section independently of branch mass (unlike SP_MIN).
    """
    valid = [i for i, n in enumerate(nodes) if n["birth"] != -1]
    children_map = {i: [] for i in range(len(nodes))}
    for i in valid:
        p = nodes[i]["parent_node_idx"]
        if p != -1:
            children_map[p].append(i)
    pos = np.zeros(len(nodes))

    # EVERY node gets its own dedicated lane (height = its spacing) placed
    # between two halves of its children — this guarantees a node never overlaps
    # its own child. ``scale`` multiplies the spacing of an entire subtree.
    def assign_pos(idx, next_leaf, lane=None, scale=1.0):
        children = children_map[idx]
        if not children:
            s = spacing[idx] * scale
            pos[idx] = next_leaf + 0.5 * s
            return next_leaf + s
        half = len(children) // 2
        for c in children[:half]:
            next_leaf = assign_pos(c, next_leaf, scale=scale)
        own = (spacing[idx] if lane is None else lane) * scale
        pos[idx] = next_leaf + 0.5 * own
        next_leaf += own
        for c in children[half:]:
            next_leaf = assign_pos(c, next_leaf, scale=scale)
        return next_leaf

    if not nodes:
        return pos

    # Root handled explicitly so its two child-groups can use different vertical
    # compression: the TOP group (-> top after flip) at full spacing, the BELOW
    # group (-> bottom after flip) shrunk by ``below_scale``.
    root_children = children_map[0]
    if not root_children:
        pos[0] = 0.0
    else:
        rh = len(root_children) // 2
        next_leaf = 0.0
        for c in root_children[:rh]:                       # top after flip
            next_leaf = assign_pos(c, next_leaf, scale=top_scale)
        pos[0] = next_leaf + 0.5 * root_lane
        next_leaf += root_lane
        for c in root_children[rh:]:                       # bottom after flip
            next_leaf = assign_pos(c, next_leaf, scale=below_scale)

    # Flip above <-> below: mirror the layout vertically.
    if valid:
        v = pos[valid]
        mid = v.max() + v.min()
        pos[valid] = mid - pos[valid]
    return pos


# ==============================================================================
# 4. COMBINED LIGHTCURVE + TREE FIGURE
# ==============================================================================
def plot_combined_lightcurve_and_tree(
    target_idx,
    full_history_loader,
    loader,
    tree_nodes,
    redshifts,
):
    """Draw the paper figure: one object's lightcurve above its merger tree.

    The two panels share a cosmic-time axis so a luminosity episode can be read
    directly against the merger that triggered it.
    """
    # Full page width (X), squished in Y — a wide, short figure. The top (tree)
    # panel gets the lion's share of the height so the many branches don't overlap.
    fig = plt.figure(figsize=(7.0, 5.2))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1.7, 1.0, 0.6], hspace=0.0,
        left=0.12, right=0.92, top=0.93, bottom=0.10,
    )

    times = full_history_loader.times_full_history
    lbol = full_history_loader.Lbols_full_history[:, target_idx]
    mbh_high_res = full_history_loader.black_hole_masses_full_history[:, target_idx]

    snapshot_times = np.array([cosmo.age(z) for z in redshifts])

    # Time axis: cap at the run's final (lowest) redshift so the axis matches
    # the data extent.
    t_end = cosmo.age(redshifts[-1])
    mbh_lo, mbh_hi = 4.5, 10.0
    xlim = (0.1, t_end * 1.02)

    # ===== TOP PANEL: MERGER TREE =====
    ax_tree = fig.add_subplot(gs[0])

    if tree_nodes:
        max_snap = len(redshifts) - 1

        # Per-node representative thickness = peak log M_BH (BH mass is ~monotonic
        # so the peak is the mass at the branch's last snapshot). This drives the
        # vertical SPACING: thin/low-mass branches get a small slot (packed),
        # thick/massive branches get a large slot (more room).
        node_logm = np.full(len(tree_nodes), np.nan)
        for _i, _n in enumerate(tree_nodes):
            if _n["birth"] == -1:
                continue
            # death-1, not death: the BH array is ZERO at a merged track's death
            # snap (same convention as `peak_logm` above; keep the two in lockstep).
            _se = (_n["death"] - 1) if _n["death"] != -1 else max_snap
            _se = min(max(_se, 0), max_snap)
            _m = loader.get_BH_mass(_se)[_n["array_idx"]]
            if _m > 0:
                node_logm[_i] = np.log10(_m)
        _fin = np.isfinite(node_logm)

        # ---- Vertical spacing per branch --------------------------------------
        # Massive branches get a wider lane; thin ones pack tight.
        #
        # The spacing is RANK-based by default, with a small dynamic range: on
        # a bottom-heavy mass distribution (this tree: 39 branches, percentiles
        # 3.6/4.5/5.0/5.1/6.6/8.0/9.3) a mass-anchored scheme pins most branches
        # at the floor while a few take SP_MAX each, doubling the y-extent and
        # crushing the low-mass bundle.
        #
        # SP_MODE:
        #   "uniform" (DEFAULT) every branch gets SP_MIN. This is the layout the
        #             figure has always actually had — and it is the one that reads
        #             well. NOTE the branch COLOUR and THICKNESS are unaffected by
        #             any of this: they are sampled per-segment along each branch,
        #             not from the death snap, so they were always correct. The
        #             death-snap read fed only (a) this spacing and (b) the
        #             BAQARO_PAPER_TREE_MIN_LOGM prune threshold. Keeping the
        #             death-1 fix therefore costs nothing visually here while making
        #             the MIN_LOGM knob work for the first time.
        #   "rank"    lane grows with the branch's RANK in peak mass — evenly spread
        #             lanes regardless of how skewed the mass distribution is.
        #   "mass"    the original linear-in-log-mass mapping (percentile anchor).
        #             On a bottom-heavy tree this pins ~40% of branches at the floor
        #             and gives a few branches enormous lanes — it looks bad. Kept
        #             only for reference.
        # Tune with BAQARO_PAPER_TREE_SP_MIN / _SP_MAX / _SP_MODE / _SP_PCT.
        SP_MODE = os.environ.get("BAQARO_PAPER_TREE_SP_MODE", "uniform").strip().lower()
        SP_MIN = float(os.environ.get("BAQARO_PAPER_TREE_SP_MIN", "0.75"))
        SP_MAX = float(os.environ.get("BAQARO_PAPER_TREE_SP_MAX", "1.7"))
        SP_PCT = float(os.environ.get("BAQARO_PAPER_TREE_SP_PCT", "40"))

        _n_nodes = len(tree_nodes)
        if SP_MODE == "uniform" or not _fin.any():
            _sp_frac = np.zeros(_n_nodes)
        elif SP_MODE == "rank":
            # Rank each branch by peak mass (NaN -> lowest). Ranks are spread
            # uniformly over [0, 1], so the lane widths are well-conditioned
            # regardless of how the masses bunch up.
            _filled = np.nan_to_num(node_logm, nan=np.nanmin(node_logm[_fin]) - 1.0)
            _rank = np.argsort(np.argsort(_filled)).astype(float)
            _sp_frac = _rank / max(_n_nodes - 1, 1)
        else:   # "mass" — the original scheme
            lm_lo = float(np.percentile(node_logm[_fin], SP_PCT))
            lm_hi = float(np.max(node_logm[_fin]))
            if lm_hi - lm_lo < 0.5:
                lm_lo, lm_hi = lm_hi - 0.5, lm_hi + 0.1
            _sp_frac = np.clip(
                (np.nan_to_num(node_logm, nan=lm_lo) - lm_lo) / (lm_hi - lm_lo + 1e-6),
                0.0, 1.0)
        spacing = SP_MIN + (SP_MAX - SP_MIN) * _sp_frac

        y_pos = get_layout(tree_nodes, spacing, root_lane=1.3, top_scale=0.85, below_scale=0.7)

        lines, colors, connector_lines = [], [], []

        for i, node in enumerate(tree_nodes):
            if node["birth"] == -1:
                continue
            idx = node["array_idx"]

            s_start, s_end = node["birth"], node["death"]
            if s_end == -1 or s_end > max_snap:
                s_end = max_snap
            if s_start < 0:
                s_start = 0
            if s_start > s_end:
                s_start = s_end

            segments = np.arange(s_start, s_end + 1)
            if len(segments) < 2:
                segments = [s_start, s_start]

            for k in range(len(segments) - 1):
                s0, s1 = segments[k], segments[k + 1]
                t0, t1 = snapshot_times[s0], snapshot_times[s1]
                if t0 == t1:
                    t1 += 0.01
                lines.append([(t0, y_pos[i]), (t1, y_pos[i])])

                # Colour each segment by the mass at its START snapshot (s0).
                # Reading the END (s1) makes a mass jump at s1 — e.g. a merger
                # that lands on the descendant at the progenitor's death snap
                # (np.add.at at snapshot i in main_evolution_full_history) — bleed
                # into the [s0, s1] segment, so the brightening appears ONE
                # snapshot before the merger. Start-colouring puts the jump at the
                # correct snapshot, and also reads the branch's last real mass
                # (M(death-1)) on its final segment instead of the death-snap
                # entry (0 / post-merger), so no death special-case is needed.
                m = loader.get_BH_mass(s0)[idx]
                colors.append(np.log10(m) if m > 0 else np.nan)

            if node["parent_node_idx"] != -1:
                t_merge = snapshot_times[s_end]
                connector_lines.append(
                    [(t_merge, y_pos[i]), (t_merge, y_pos[node["parent_node_idx"]])]
                )

        # Colour + width normalisation ADAPTIVE to this tree's own mass range so
        # the many low-mass branches spread across the colormap and the width
        # scale instead of all clipping to the identical darkest/thinnest line.
        # (The old fixed vmin=6.54 mapped every seed/small progenitor at or below
        # it to the same colour AND width.) vmin = 1st percentile (robust to a
        # single extreme seed), vmax = peak mass of the backbone.
        colors = np.array(colors, dtype=float)
        finite = np.isfinite(colors)
        if finite.any():
            vmin = float(np.percentile(colors[finite], 1.0))
            vmax = float(np.max(colors[finite]))
            if vmax - vmin < 0.5:                 # near-flat tree guard
                vmin, vmax = vmax - 0.5, vmax + 0.1
        else:
            vmin, vmax = 6.5, 10.0
        colors[~finite] = vmin
        frac = np.clip((colors - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        widths = 1.1 + 3.2 * frac

        # rasterized=True: the tree is thousands of individually-coloured line
        # segments — keeping them vector makes a path-heavy PDF that some viewers
        # (PDF.js / VSCode preview) render blank. Rasterising the data artists at
        # savefig.dpi (300) keeps the figure crisp while text/axes stay vector.
        lc = LineCollection(
            lines, array=colors, cmap=cmap_tree,
            norm=Normalize(vmin, vmax), linewidths=widths, capstyle="round",
            rasterized=True,
        )
        ax_tree.add_collection(lc)

        ax_tree.add_collection(LineCollection(
            connector_lines, colors="#888888", linewidths=1.2,
            alpha=0.5, linestyles="--", zorder=0, rasterized=True,
        ))

        valid_ys = [y_pos[i] for i, n in enumerate(tree_nodes) if n["birth"] != -1]
        if valid_ys:
            pad = 0.5
            ax_tree.set_ylim(min(valid_ys) - pad, max(valid_ys) + pad)

        # Colorbar: small, vertical, to the left of the tree panel.
        tree_pos = ax_tree.get_position()
        cbar_h = tree_pos.height * 0.8
        cbar_y0 = tree_pos.y0 + (tree_pos.height - cbar_h) / 2
        cax = fig.add_axes([tree_pos.x0 - 0.04, cbar_y0, 0.012, cbar_h])
        cbar = fig.colorbar(lc, cax=cax, orientation="vertical")
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.set_ylabel(r"$\log_{10}\, M_{\rm BH}$", fontsize=9, labelpad=6, rotation=90)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.minorticks_on()

    ax_tree.set_yticks([])
    ax_tree.spines["left"].set_visible(False)
    ax_tree.spines["right"].set_visible(False)
    ax_tree.spines["top"].set_visible(False)
    plt.setp(ax_tree.get_xticklabels(), visible=False)
    # Kill the tree panel's OWN x-ticks (the global xtick.top=True rcParam draws
    # the cosmic-time ticks on top, right under the redshift secondary axis,
    # where they look like stray subticks). Only the redshift axis (ax3) should
    # tick the top.
    ax_tree.tick_params(axis="x", which="both", top=False, bottom=False, length=0)
    ax_tree.set_xlim(xlim)

    # ===== MIDDLE PANEL: LIGHTCURVE + MASS =====
    ax_lbol = fig.add_subplot(gs[1], sharex=ax_tree)

    with np.errstate(divide="ignore"):
        log_lbol_ergs = my_utils.to_ergs(np.log10(lbol))
    # L_bol==0 (non-accreting) -> log=-inf leaves a GAP; floor below the panel
    # so the line visibly PLUNGES to the bottom at each drop-out.
    log_lbol_ergs = np.where(np.isfinite(log_lbol_ergs), log_lbol_ergs,
                             my_utils.to_ergs(mbh_lo + nc.log_csi) - 0.4)
    ax_lbol.plot(times, log_lbol_ergs,
                 lw=1.0, alpha=0.7, drawstyle="steps-mid", color=col_lbol,
                 rasterized=True)

    for lref in [45., 46., 47.]:
        ax_lbol.axhline(y=lref, color="grey", lw=0.6, ls="--", alpha=0.4)

    ax_lbol.set_ylabel(r"$\log_{10}\, L_{\rm bol}$ [erg s$^{-1}$]", color=col_lbol, labelpad=3)
    ax_lbol.tick_params(axis="y", colors=col_lbol)
    ax_lbol.set_ylim(my_utils.to_ergs(mbh_lo + nc.log_csi),
                     my_utils.to_ergs(mbh_hi + nc.log_csi))
    ax_lbol.minorticks_on()
    plt.setp(ax_lbol.get_xticklabels(), visible=False)

    # Twin axis: BH mass.
    ax_mass = ax_lbol.twinx()
    with np.errstate(divide="ignore"):
        log_mbh = np.log10(mbh_high_res)  # -inf before birth (M_BH=0); not drawn
    ax_mass.plot(times, log_mbh,
                 lw=1.5, alpha=0.9, drawstyle="steps-mid", color=col_mass,
                 rasterized=True)
    ax_mass.set_ylabel(r"$\log_{10}\, M_{\rm BH}$ [$M_\odot$]", color=col_mass, labelpad=-1)
    ax_mass.tick_params(axis="y", colors=col_mass)
    ax_mass.set_ylim(mbh_lo, mbh_hi)
    ax_mass.set_xlim(xlim)
    ax_mass.minorticks_on()

    ax_lbol.text(0.02, 0.93, r"$L_{\rm bol}$", color=col_lbol, fontsize=11,
                 fontweight="bold", transform=ax_lbol.transAxes, ha="left", va="top")
    ax_lbol.text(0.145, 0.93, r"$M_{\rm BH}$", color=col_mass, fontsize=11,
                 fontweight="bold", transform=ax_lbol.transAxes, ha="left", va="top")

    # Redshift axis on top of the tree panel.
    add_redshift_axis(ax_tree, z_ticks=Z_TICKS)

    # ===== BOTTOM PANEL: EDDINGTON RATIOS (eta_acc + lambda_Edd) =====
    ax_eta = fig.add_subplot(gs[2], sharex=ax_tree)

    # eta_acc = M_dot_acc / M_dot_Edd — the ERDF accretion-rate Eddington ratio
    # (stored per sub-step). lambda_Edd = L_bol / L_Edd — the radiative one; they
    # differ by the Madau efficiency ratio eps(eta)/eps0 (lambda_Edd <= eta_acc).
    eta_acc = full_history_loader.etas_full_history[:, target_idx].astype(float)
    eta_acc[eta_acc <= 1e-20] = np.nan
    safe_mass = np.maximum(mbh_high_res, 1e-10)
    lam_edd = lbol / safe_mass / 10**nc.log_csi
    with np.errstate(divide="ignore", invalid="ignore"):
        log_eta_acc = np.log10(eta_acc)
        log_lam_edd = np.log10(lam_edd)

    ax_eta.axhline(y=0., color="grey", lw=0.6, ls="--", alpha=0.4)
    ax_eta.plot(times, log_eta_acc,
                lw=1.0, alpha=0.8, drawstyle="steps-mid", color=col_eta_acc,
                rasterized=True)
    ax_eta.plot(times, log_lam_edd,
                lw=1.0, alpha=0.8, drawstyle="steps-mid", color=col_eta,
                rasterized=True)

    ax_eta.set_ylabel(r"$\log_{10}$ Edd. ratio", labelpad=3)
    ax_eta.set_ylim(-4.5, 2.0)
    ax_eta.set_xlabel(r"Cosmic time [Gyr]", labelpad=-1)
    ax_eta.minorticks_on()

    ax_eta.text(0.985, 0.90, r"$\eta_{\rm acc}$", color=col_eta_acc,
                fontweight="bold", transform=ax_eta.transAxes, ha="right", va="top")
    ax_eta.text(0.985, 0.62, r"$\lambda_{\rm Edd}$", color=col_eta,
                fontweight="bold", transform=ax_eta.transAxes, ha="right", va="top")

    return fig


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("Loading full-history fiducial:", path_file_full_history_fid)
    with h5py.File(path_file_full_history_fid, "r") as file:
        data = load_simulation_metadata_full_history(file)
        full_loader = data["full_history_loader"]
        loader = data["loader"]
        target_indices = data["original_targets_new_indices"]
        redshifts = data["redshifts"]

        grp = file["merger_trees"]
        mt_track_ids = grp["track_ids"][:]
        mt_descendants = grp["merger_track_ids"][:]
        mt_birth = grp["snapshot_indexes_of_birth"][:]
        mt_death = grp["snapshot_indexes_of_death"][:]

        progenitor_map = build_reverse_map_on_the_fly(mt_track_ids, mt_descendants)

        # Prune small twig branches so dense trees fit. The mass cut
        # (BAQARO_PAPER_TREE_MIN_LOGM, peak log M_BH) is the main declutter knob;
        # the length cut (BAQARO_PAPER_TREE_MIN_LEN, snapshots) trims brief twigs.
        # Set either to its off value (logm=-inf via "", len=0) to disable.
        min_branch_len = int(os.environ.get("BAQARO_PAPER_TREE_MIN_LEN", "50"))
        _minlogm_env = os.environ.get("BAQARO_PAPER_TREE_MIN_LOGM", "").strip()
        min_branch_logm = float(_minlogm_env) if _minlogm_env else -np.inf
        max_snap_for_prune = len(redshifts) - 1

        def render_one(storage_idx, save_name, close=False):
            full_tree = build_tree_nodes(storage_idx, progenitor_map, mt_birth, mt_death)
            n_full = len(full_tree)
            tree_nodes = prune_small_branches(
                full_tree, min_branch_len, min_branch_logm, max_snap_for_prune, loader)

            # --- target-object stats (for the paper / figure note) ---
            ms = max_snap_for_prune
            births = [n["birth"] for n in full_tree if n["birth"] != -1]
            first_snap = min(births) if births else -1          # earliest BH in tree
            seed_snap = full_tree[0]["birth"]                   # the target's own seed
            z_first = float(redshifts[first_snap]) if first_snap >= 0 else float("nan")
            z_seed = float(redshifts[seed_snap]) if seed_snap >= 0 else float("nan")
            mbh_z0 = float(loader.get_BH_mass(ms)[storage_idx])
            logmbh_z0 = np.log10(mbh_z0) if mbh_z0 > 0 else float("nan")
            print(f"  [{save_name}] array idx {storage_idx}: tree {n_full} -> "
                  f"{len(tree_nodes)} nodes (min_len={min_branch_len}, "
                  f"min_logM={min_branch_logm})")
            print(f"  [{save_name}] STATS  original_nodes={n_full}  kept_nodes={len(tree_nodes)}  "
                  f"mergers(full)={n_full - 1}  z_first_BH={z_first:.2f}(snap{first_snap})  "
                  f"z_target_seed={z_seed:.2f}(snap{seed_snap})  "
                  f"logMBH_z0={logmbh_z0:.2f}")

            fig = plot_combined_lightcurve_and_tree(
                storage_idx, full_loader, loader, tree_nodes, redshifts,
            )
            save_fig(fig, plot_config.FIGURES_DIR, save_name, force_dir=True)
            if close:
                plt.close(fig)
            return fig

        # SCAN mode: BAQARO_PAPER_TREE_RANKS="0,1,5,9" renders one figure per rank
        # (suffixed _rank{r}) so you can pick the best-looking tree. Otherwise a
        # single figure: BAQARO_PAPER_TREE_IDX (direct) wins, else
        # BAQARO_PAPER_TREE_RANK (default 0).
        ranks_env = os.environ.get("BAQARO_PAPER_TREE_RANKS", "").strip()
        if ranks_env:
            ranks = [int(x) for x in ranks_env.split(",") if x.strip() != ""]
            print(f"Scanning {len(ranks)} first-born targets (of {len(target_indices)}): {ranks}")
            for r in ranks:
                render_one(int(target_indices[r]), f"{name_fig}_rank{r}", close=True)
        else:
            _idx_env = os.environ.get("BAQARO_PAPER_TREE_IDX")
            if _idx_env not in (None, ""):
                target_storage = int(_idx_env)
            else:
                rank = int(os.environ.get("BAQARO_PAPER_TREE_RANK", "14"))
                target_storage = int(target_indices[rank])
            render_one(target_storage, name_fig)
            maybe_show()
