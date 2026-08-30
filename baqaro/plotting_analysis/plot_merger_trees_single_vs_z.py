"""ANALYSIS FIGURE: one merger tree drawn against redshift.

Renders a single halo's merger tree with redshift on the x-axis, so the epoch
at which each branch joins is directly readable. The companion
``plot_merger_trees_single.py`` draws the same tree by topology against
snapshot index instead, which shows the branching structure more clearly but
not when things happen.

Saves ``merger_tree_{target_id}`` when ``BAQARO_SAVE_FIGS=1``.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
import os

from baqaro.utils.my_units import mass_units

from baqaro.plotting_common.plot_config import save_fig, maybe_show
from baqaro.plotting_analysis._merger_tree_layout import (
    build_tree_debug, get_layout,
)



def plot_debug_tree(nodes, redshifts, halo_masses=None, title=""):
    """Draw the tree with redshift on the x-axis and topology on the y-axis.

    The epoch at which each branch joins is directly readable, which the
    topology-ordered companion in ``plot_merger_trees_single.py`` cannot show.
    """
    fig, ax = plt.subplots(figsize=(6, 6.5))

    if not nodes:
        return fig

    y_pos = get_layout(nodes)  # topology position is now the y-axis
    max_snap = len(redshifts) - 1

    lines = []
    colors = []
    widths = []
    connector_lines = []

    # Mass scaling
    if halo_masses is not None:
        all_masses = []
        for n in nodes:
            if n['birth'] == -1:
                continue
            d = n['death']
            if d < 0:
                d = max_snap
            m = halo_masses[min(d, max_snap), n['track_id']] * mass_units
            if m > 0:
                all_masses.append(m)
        if all_masses:
            vmin, vmax = np.log10(min(all_masses)), np.log10(max(all_masses))
        else:
            vmin, vmax = 8, 12
    else:
        vmin, vmax = 0, 1

    for i, node in enumerate(nodes):
        if node['birth'] == -1:
            continue

        y = y_pos[i]
        start = node['birth']
        end = node['death']

        if end == -1:
            end = max_snap
        if start < 0:
            start = 0
        if end > max_snap:
            end = max_snap
        if start > end:
            start = end

        # Horizontal line segments (the branch) — snapshot on x-axis
        segments_steps = np.arange(start, end + 1)
        if len(segments_steps) < 2:
            segments_steps = [start, start + 0.5]

        for k in range(len(segments_steps) - 1):
            x0, x1 = segments_steps[k], segments_steps[k + 1]
            lines.append([(x0, y), (x1, y)])

            if halo_masses is not None:
                snap_idx = int(min(x1, max_snap))
                m = halo_masses[snap_idx, node['track_id']] * mass_units
                log_m = np.log10(m) if m > 0 else vmin
                colors.append(log_m)
                w = 2.5 + 10.0 * (log_m - vmin) / (vmax - vmin + 1e-6)
                widths.append(max(2.5, min(w, 12.0)))
            else:
                colors.append(node['depth'])
                widths.append(4.0)

        # Vertical connector (merger)
        parent_idx = node['parent_list_idx']
        if parent_idx != -1:
            parent_node = nodes[parent_idx]
            if parent_node['birth'] != -1:
                parent_y = y_pos[parent_idx]
                x_merge = end
                connector_lines.append([(x_merge, y), (x_merge, parent_y)])

    # Truncated inferno: skip the darkest 15% so branches are always visible
    _inferno = plt.cm.inferno
    _colors_trunc = _inferno(np.linspace(0.15, 1.0, 256))
    cmap_trunc = LinearSegmentedColormap.from_list('inferno_trunc', _colors_trunc)

    # Main branches
    lc = LineCollection(
        lines, array=np.array(colors), cmap=cmap_trunc,
        norm=Normalize(vmin, vmax), linewidths=widths, capstyle='round',
    )
    ax.add_collection(lc)

    # Merger connectors
    lc_conn = LineCollection(
        connector_lines, colors='#888888', linewidths=1.5,
        alpha=0.6, linestyles='--', zorder=0,
    )
    ax.add_collection(lc_conn)

    # Redshift on the x-axis (snapshot coordinates, labelled as z)
    ax.set_xlim(-2, max_snap + 1)

    # Evenly spaced ticks in snapshot space, labelled with redshift
    n_ticks = 8
    z_ticks = np.linspace(0, max_snap, n_ticks, dtype=int)
    z_ticks = np.unique(z_ticks)
    z_labels = [f"{redshifts[t]:.1f}" for t in z_ticks]
    ax.set_xticks(z_ticks)
    ax.set_xticklabels(z_labels)
    ax.set_xlabel("Redshift")

    # Remove y-axis (topology has no physical meaning)
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Zoom y to valid data
    valid_ys = [y_pos[i] for i, n in enumerate(nodes) if n['birth'] != -1]
    if valid_ys:
        pad = 0.5
        ax.set_ylim(min(valid_ys) - pad, max(valid_ys) + pad)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Colorbar at the top, horizontal
    if halo_masses is not None:
        cbar = fig.colorbar(
            lc, ax=ax, orientation='horizontal', location='top',
            pad=0.03, fraction=0.03, aspect=35,
        )
        cbar.set_label(r"$\log_{10}(M_{\rm halo}\;/\;{\rm M}_\odot)$")

    if title:
        ax.set_title(title, pad=35)

    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(left=0.02, right=0.98)
    return fig

# =============================================================================
# MAIN DEBUG RUNNER
# =============================================================================
if __name__ == "__main__":
    from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
    from baqaro.core_functions.select_merger_branches import build_reverse_index
    from baqaro.utils.my_dir import get_output_path, get_input_path_HBT_data, get_plots_path

    # --- CONFIG ---
    # Which tree to draw. Env-driven and shared by BOTH single-tree modules, so
    # the topology view and the vs-redshift view show the SAME tree and can be
    # read side by side. They used to hardcode different indices (9 here, 38 in
    # the sibling), which quietly made the two views incomparable.
    target_index_to_plot = int(os.environ.get("BAQARO_TREE_INDEX", "9"))
    source_dir = "machine_igm"
    # This is an L2800N5040 figure, so run it with BAQARO_SIM=L2800N5040.
    # Override via BAQARO_MAX_SNAP (L2800N5040 maxes: {14, 38, 78}).
    # Default = the sim's own production max_snap (sim_config), the one value
    # guaranteed to have halo files on disk. Was hardcoded "78", a max_snap that
    # exists for NO simulation here -- the module died on a missing tree dir.
    from baqaro.utils.sim_config import max_snap as _sim_max_snap
    max_snap = int(os.environ.get("BAQARO_MAX_SNAP", str(_sim_max_snap)))

    # --- LOAD ---
    print("1. Loading Data...")
    path_sim = get_input_path_HBT_data(source=source_dir)
    from baqaro.utils.sim_config import (
        simulation_name, halo_histories_name,
    )

    path_out = get_output_path(source=source_dir)
    path_plots = get_plots_path(source=source_dir)
    # Build the filename through the shared helper so the `_foldmass` /
    # `_{merger_delay_mode}` tokens and the per-sim tdyn_fraction are applied.
    # Composing it inline gives a token-less `..._tdynfraction_0.2` name, which
    # would load the legacy (no-fold, instant_old) arrays instead. The reverse index is
    # rebuilt below if absent, so no cached .npz is required.
    name_file = halo_histories_name(max_snap_=max_snap)
    path_trees = os.path.join(path_out, "halo_histories", f"merger_trees_{name_file}")
    path_mass = os.path.join(path_out, "halo_histories", f"halo_masses_{name_file}.npy")
    path_reverse = os.path.join(path_out, "halo_histories", f"reverse_index_{name_file}.npz")

    
    trees = MergerTreeLoader(path_trees)
    track_ids = np.array(trees.track_ids)
    merger_track_ids = np.array(trees.merger_track_ids)
    snap_birth = np.array(trees.snapshot_indexes_of_birth)
    snap_death = np.array(trees.snapshot_indexes_of_death)
    halo_masses = np.load(path_mass, mmap_mode='r')
    
    # Redshifts — read from the active sim's output_list (sim-aware).
    path_sim = get_input_path_HBT_data(source=source_dir)
    try:
        z_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")
        redshifts = np.loadtxt(z_file)[0:max_snap+1]
    except Exception:
        redshifts = np.linspace(10, 0, max_snap+1)

    print("2. Building Reverse Index...")
    # 3. Load/Build Reverse Index (Cached)
    if os.path.exists(path_reverse):
        print(f"Loading cached reverse index from {path_reverse}...")
        data = np.load(path_reverse)
        reverse_index = (data['indptr'], data['indices'])
    else:
        print("Building reverse index (this may take time)...")
        reverse_index = build_reverse_index(merger_track_ids)
        print("Saving reverse index for next time...")
        np.savez(path_reverse, indptr=reverse_index[0], indices=reverse_index[1])

    
    # --- SELECT TARGET ---
    target_id = track_ids[target_index_to_plot]
    print(f"\n--- DEBUGGING TARGET: {target_id} ---")
    
    # --- BUILD TREE ---
    nodes = build_tree_debug(target_id, track_ids, merger_track_ids, snap_birth, snap_death, reverse_index)
    
    print(f"Tree contains {len(nodes)} nodes.")

    # --- PLOT ---
    print("3. Plotting...")
    fig = plot_debug_tree(nodes, redshifts, halo_masses, title=None)

    save_fig(fig, path_plots, f"merger_tree_{target_id}")
    maybe_show()
