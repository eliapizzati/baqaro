"""ANALYSIS FIGURE: one halo merger tree, drawn.

Renders a single tree from the HBT-HERONS catalogue -- branches positioned by
mass and time -- to make the tree structure the forward model walks visible.
Useful for sanity-checking merger timing and the orphan-handling schemes, which
are otherwise only observable through their statistical effect.

Halo files are resolved via ``sim_config.halo_histories_name`` so the
``_foldmass`` / merger-mode tokens always match the run being inspected.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import os

from baqaro.utils.my_units import mass_units

from baqaro.plotting_common.plot_config import save_fig, maybe_show
from baqaro.plotting_analysis._merger_tree_layout import (
    build_tree_debug, get_layout,
)



def plot_debug_tree(nodes, redshifts, halo_masses=None, title=""):
    """Draw the tree with topology on the x-axis and snapshot on the y-axis.

    Shows branching STRUCTURE most clearly, at the cost of not showing when
    branches join. ``plot_merger_trees_single_vs_z.py`` draws the same tree
    against redshift for the complementary view.
    """
    fig = plt.figure(figsize=(14, 8))
    # Create main axis with space for colorbar on the right
    ax = fig.add_axes([0.05, 0.1, 0.8, 0.85])
    
    if not nodes: return fig
        
    x_pos = get_layout(nodes)
    max_snap = len(redshifts) - 1
    
    lines = []
    colors = []
    widths = []
    connector_lines = []
    
    # Mass scaling helper
    if halo_masses is not None:
        all_masses = []
        for n in nodes:
            if n['birth'] == -1: continue # Skip invalid for scaling
            d = n['death']
            if d < 0: d = max_snap
            m = halo_masses[min(d, max_snap), n['track_id']] * mass_units
            if m > 0: all_masses.append(m)
        if all_masses:
            vmin, vmax = np.log10(min(all_masses)), np.log10(max(all_masses))
        else:
            vmin, vmax = 8, 12
    else:
        vmin, vmax = 0, 1

    print(f"\n--- Plotting Valid Nodes (Skipping birth=-1) ---")
    
    for i, node in enumerate(nodes):
        # --- FILTERING STEP ---
        # If birth is -1, this halo was filtered out by the resolution script.
        if node['birth'] == -1:
            continue

        x = x_pos[i]
        start = node['birth']
        end = node['death']
        
        # Standardize "Alive at End"
        # If birth is valid (checked above), but death is -1, it means "Alive".
        if end == -1: 
            end = max_snap
            
        # Sanity Clamps
        if start < 0: start = 0
        if end > max_snap: end = max_snap
        if start > end: start = end # Prevent negative duration
            
        # 1. Vertical Line (The Halo)
        segments_steps = np.arange(start, end + 1)
        if len(segments_steps) < 2: segments_steps = [start, start + 0.5]
            
        for k in range(len(segments_steps)-1):
            y0, y1 = segments_steps[k], segments_steps[k+1]
            lines.append([(x, y0), (x, y1)])
            
            if halo_masses is not None:
                snap_idx = int(min(y1, max_snap))
                m = halo_masses[snap_idx, node['track_id']] * mass_units
                log_m = np.log10(m) if m > 0 else vmin
                colors.append(log_m)
                w = 0.5 + 12.0 * (log_m - vmin) / (vmax - vmin + 1e-6)
                widths.append(max(0.5, min(w, 12.0)))
            else:
                colors.append(node['depth'])
                widths.append(3.0)
                
        # 2. Horizontal Line (The Merger)
        parent_idx = node['parent_list_idx']
        
        # Only draw connection if parent exists AND is valid
        if parent_idx != -1:
            parent_node = nodes[parent_idx]
            if parent_node['birth'] != -1:
                parent_x = x_pos[parent_idx]
                y_merge = end
                connector_lines.append([(x, y_merge), (parent_x, y_merge)])

        # 3. Label (ID)
        # Intentionally omitted to avoid plotting halo IDs

    # Add to plot
    lc = LineCollection(lines, array=np.array(colors), cmap='viridis', norm=Normalize(vmin, vmax), linewidths=widths, capstyle='round')
    ax.add_collection(lc)
    
    lc_conn = LineCollection(connector_lines, colors='gray', linewidths=1.5, alpha=0.5, zorder=0)
    ax.add_collection(lc_conn)
    
    # Axes
    ax.set_ylim(-2, max_snap + 1)
    ax.invert_yaxis()
    ax.set_ylabel("Snapshot")
    ax.set_xlabel("Merger Tree Topology")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = np.unique(np.linspace(0, max_snap, 8, dtype=int))
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"z={redshifts[t]:.1f}" for t in ticks])

    # Colorbar for mass (log10) - positioned to the right
    if halo_masses is not None:
        cax = fig.add_axes([0.92, 0.1, 0.025, 0.85])
        cbar = fig.colorbar(lc, cax=cax)
        cbar.set_label("log10(Halo Mass)", fontsize=11)
    
    # Zoom to valid data
    valid_xs = [x_pos[i] for i, n in enumerate(nodes) if n['birth'] != -1]
    if valid_xs:
        pad = 0.5
        ax.set_xlim(min(valid_xs)-pad, max(valid_xs)+pad)
    
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
    # Override via BAQARO_MAX_SNAP env var. Available (the `_foldmass` arrays;
    # the token-less legacy set was deleted):
    #   L2800N5040  : {14, 38, 78}
    #   L2800N10080 : {39, 40, 50, 60, 71, 144}
    # Default = the sim's own production max_snap (sim_config), the one value
    # guaranteed to have arrays on disk for whichever BAQARO_SIM is active. The old
    # literal 14 resolved only for L2800N5040, never for the default sim.
    from baqaro.utils.sim_config import max_snap as _sim_max_snap
    max_snap = int(os.environ.get("BAQARO_MAX_SNAP", str(_sim_max_snap)))
    
    # --- LOAD ---
    print("1. Loading Data...")
    path_out = get_output_path(source=source_dir)

    path_plots = get_plots_path(source=source_dir)
    # Parameters matching those used in halo_mass_histories_saver
    nbound_threshold = 40
    halo_filtering_mode = "global"


    # Simulation parameters
    path_sim = get_input_path_HBT_data(source=source_dir)
    from baqaro.utils.sim_config import (
        simulation_name, tdyn_fraction_default,
        halo_histories_name,
    )

    # Use the shared builder: an inline name without the `_foldmass` /
    # `_{merger_delay_mode}` tokens would load the legacy (no-fold, instant_old)
    # trees instead (both variants are on disk).
    name_file_halos = halo_histories_name(
        max_snap_=max_snap,
        nbound_threshold=nbound_threshold,
        halo_filtering_mode=halo_filtering_mode,
        tdyn_fraction=tdyn_fraction_default,
    )


    path_trees = os.path.join(path_out, "halo_histories", f"merger_trees_{name_file_halos}")
    path_mass = os.path.join(path_out, "halo_histories", f"halo_masses_{name_file_halos}.npy")
    path_reverse = os.path.join(path_out, "halo_histories", f"reverse_index_{name_file_halos}.npz")

    
    trees = MergerTreeLoader(path_trees)
    track_ids = np.array(trees.track_ids)
    merger_track_ids = np.array(trees.merger_track_ids)
    snap_birth = np.array(trees.snapshot_indexes_of_birth)
    snap_death = np.array(trees.snapshot_indexes_of_death)
    halo_masses = np.load(path_mass, mmap_mode='r')
    
    # Redshifts
    path_sim = get_input_path_HBT_data(source=source_dir)
    try:
        redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")
        redshifts = np.loadtxt(redshift_file)[0:max_snap+1]
    except:
        print("Warning: Using dummy redshifts.")
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
    
    # --- PRINT TEXT DIAGNOSTIC ---
    # This is crucial. If the plot is ugly, look at this output.
    print(f"Tree contains {len(nodes)} nodes.")
    print(f"{'IDX':<5} {'TrackID':<10} {'Depth':<6} {'Birth':<6} {'Death':<6} {'ParentListIdx':<15}")
    print("-" * 60)
    for i in range(min(20, len(nodes))): # Print first 20 nodes
        n = nodes[i]
        print(f"{n['list_idx']:<5} {n['track_id']:<10} {n['depth']:<6} {n['birth']:<6} {n['death']:<6} {n['parent_list_idx']:<15}")
    if len(nodes) > 20: print("... (more nodes)")

    # --- PLOT ---
    print("\n3. Plotting...")
    fig = plot_debug_tree(nodes, redshifts, halo_masses, title="Single Tree Debug")
    save_fig(fig, path_plots, f"merger_tree_topology_{target_id}")
    maybe_show()
