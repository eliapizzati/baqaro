"""
Visualize merger histories of selected halos.
Production version: Includes robust filtering, headless plotting, and caching.
"""

import numpy as np

from baqaro.utils.my_units import mass_units
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from typing import List, Tuple, Optional, Dict
import os

from baqaro.plotting_common.plot_config import save_fig, maybe_show

# =============================================================================
# 1. DATA EXTRACTION
# =============================================================================

def get_merger_tree_structure(
    target_track_id: int,
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    snapshot_of_birth: np.ndarray,
    snapshot_of_death: np.ndarray,
    reverse_index: Tuple[np.ndarray, np.ndarray],
) -> List[Dict]:
    """
    Builds a tree structure for a single target halo.
    Returns a list of node dictionaries, filtering out invalid (birth=-1) halos.
    """
    indptr, indices = reverse_index
    
    # Structure: List of dicts. 
    # Each dict has: {list_idx, track_id, depth, birth, death, parent_list_idx}
    tree_nodes = []
    
    # Queue for BFS: (track_id, depth, parent_list_index)
    queue = [(target_track_id, 0, -1)]
    visited = set()
    
    while queue:
        current_id, depth, parent_idx = queue.pop(0)
        
        if current_id in visited:
            continue
        visited.add(current_id)
        
        # Vital Stats
        birth = snapshot_of_birth[current_id]
        death = snapshot_of_death[current_id]
        
        # --- CRITICAL FIX: RESOLUTION FILTER ---
        # If birth is -1, this halo was filtered out by the resolution script.
        # We skip it and its branch entirely (or just this node).
        # Usually if a parent is invalid, we stop the branch.
        if birth == -1:
            continue
            
        current_list_idx = len(tree_nodes)
        
        tree_nodes.append({
            'list_idx': current_list_idx,
            'track_id': current_id,
            'depth': depth,
            'birth': birth,
            'death': death,
            'parent_list_idx': parent_idx
        })
        
        # Find Progenitors (children in tree view)
        if current_id < len(indptr) - 1:
            start = indptr[current_id]
            end = indptr[current_id+1]
            progenitors = indices[start:end]
            
            for prog_id in progenitors:
                # Add progenitors to queue
                queue.append((prog_id, depth + 1, current_list_idx))
                
    return tree_nodes

# =============================================================================
# 2. LAYOUT ENGINE
# =============================================================================

def _calculate_tree_layout(nodes: List[Dict]) -> np.ndarray:
    """
    Calculates X-positions. Leaves are stacked side-by-side.
    Parents are centered above children.
    """
    n_nodes = len(nodes)
    if n_nodes == 0: return np.array([])
    
    # Rebuild adjacency for the filtered list
    children_map = {i: [] for i in range(n_nodes)}
    for i, node in enumerate(nodes):
        p = node['parent_list_idx']
        if p != -1:
            children_map[p].append(i)
            
    x_positions = np.zeros(n_nodes, dtype=np.float64)
    next_leaf_x = 0.0

    # ITERATIVE post-order walk.
    # This was a recursive `assign_x`, which blows Python's default 1000-frame
    # recursion limit on a deep merger tree (tree depth == number of snapshots the
    # lineage spans, easily >1000 for a long chain at maxsnap=144) — a RecursionError
    # mid-figure. The iterative form has no depth limit and is otherwise identical:
    # leaves are assigned left-to-right, parents are centred on their children.
    if nodes:
        # Phase 1: post-order sequence (children fully processed before parent).
        post_order = []
        stack = [(0, False)]
        while stack:
            idx, expanded = stack.pop()
            if expanded:
                post_order.append(idx)
                continue
            stack.append((idx, True))
            # push children in reverse so they pop in original order
            for child in reversed(children_map[idx]):
                stack.append((child, False))
        # Phase 2: assign in post-order — every child already has its x.
        for idx in post_order:
            children = children_map[idx]
            if not children:
                x_positions[idx] = next_leaf_x
                next_leaf_x += 1.0
            else:
                x_positions[idx] = sum(x_positions[c] for c in children) / len(children)

    return x_positions

# =============================================================================
# 3. PLOTTING FUNCTIONS
# =============================================================================

def plot_single_merger_tree(
    nodes: List[Dict],
    redshifts: np.ndarray,
    halo_masses: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
) -> plt.Axes:
    """Draw one merger tree onto ``ax``, coloured by halo mass where available.

    Shared by the grid and timeline figures below; pass ``ax=None`` to get a
    freshly created one. Returns the axes.
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
    if not nodes:
        ax.text(0.5, 0.5, "No valid progenitors", ha='center', transform=ax.transAxes)
        return ax

    # 1. Layout
    x_positions = _calculate_tree_layout(nodes)
    max_snap = len(redshifts) - 1
    
    # 2. Mass Scaling
    if halo_masses is not None:
        all_masses = []
        for n in nodes:
            d = n['death']
            if d < 0: d = max_snap
            # code units -> Msun (x mass_units = 1e7). Missing => a 7-DEX mislabel;
            # the sibling plot_merger_trees_single.py already converts.
            m = halo_masses[min(d, max_snap), n['track_id']] * mass_units
            if m > 0: all_masses.append(m)
        
        if all_masses:
            vmin, vmax = np.log10(min(all_masses)), np.log10(max(all_masses))
            # Ensure visible contrast
            if vmax - vmin < 1.0: vmax = vmin + 1.0
        else:
            vmin, vmax = 8, 12
    else:
        vmin, vmax = 0, 1

    segments = []
    colors = []
    linewidths = []
    conn_segments = []
    
    # 3. Build Lines
    for i, node in enumerate(nodes):
        x = x_positions[i]
        start = node['birth']
        end = node['death']
        
        # Fix "Alive" status
        if end < 0: end = max_snap
        
        # Sanity clamps
        if start < 0: start = 0
        if end > max_snap: end = max_snap
        if start > end: start = end # Fix negative duration
            
        # A. Vertical Segments
        steps = np.arange(start, end + 1)
        if len(steps) < 2: steps = [start, start + 0.5]
            
        for k in range(len(steps)-1):
            y0, y1 = steps[k], steps[k+1]
            segments.append([(x, y0), (x, y1)])
            
            if halo_masses is not None:
                snap_idx = int(min(y1, max_snap))
                m = halo_masses[snap_idx, node['track_id']] * mass_units
                val = np.log10(m) if m > 0 else vmin
                colors.append(val)
                # Clamp width: 0.5 to 5.0
                w = 0.5 + 4.5 * ((val - vmin) / (vmax - vmin + 1e-6))
                linewidths.append(np.clip(w, 0.5, 5.0))
            else:
                colors.append(node['depth'])
                linewidths.append(2.0)
                
        # B. Horizontal Connections
        p_idx = node['parent_list_idx']
        if p_idx != -1:
            # Connect at death time
            parent_x = x_positions[p_idx]
            conn_segments.append([(x, end), (parent_x, end)])

    # 4. Render
    lc_halos = LineCollection(segments, array=np.array(colors), cmap=cmap, norm=Normalize(vmin, vmax), linewidths=linewidths, capstyle='round')
    ax.add_collection(lc_halos)
    
    lc_conn = LineCollection(conn_segments, colors='gray', linewidths=1.0, alpha=0.5, zorder=0)
    ax.add_collection(lc_conn)
    
    # 5. Axes
    ax.set_ylim(-1, max_snap + 1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlabel("Merger History")
    ax.set_ylabel("Snapshot")
    
    # Redshift Axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = np.unique(np.linspace(0, max_snap, 8, dtype=int))
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"z={redshifts[t]:.1f}" for t in ticks])
    ax2.set_ylabel("Redshift")
    
    # Auto-Zoom X
    if len(x_positions) > 0:
        pad = 0.5
        ax.set_xlim(x_positions.min()-pad, x_positions.max()+pad)
    else:
        ax.set_xlim(-1, 1)

    if title:
        ax.set_title(title, fontsize=10)
        
    return ax

def plot_merger_tree_grid(
    target_track_ids: np.ndarray,
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    snapshot_of_birth: np.ndarray,
    snapshot_of_death: np.ndarray,
    redshifts: np.ndarray,
    reverse_index: Tuple[np.ndarray, np.ndarray],
    halo_masses: Optional[np.ndarray] = None,
    ncols: int = 5,
    path_plots: Optional[str] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """Panel grid of merger trees, one panel per target track.

    The sample view: several trees side by side, to compare branching richness
    across halo masses rather than study any one tree in detail.
    """

    n_targets = len(target_track_ids)
    nrows = int(np.ceil(n_targets / ncols))
    
    # Auto-size figure based on count
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    
    print(f"Generating grid for {n_targets} targets...")
    
    for i, target_id in enumerate(target_track_ids):
        ax = axes[i]
        
        # Build clean tree
        nodes = get_merger_tree_structure(
            target_id, track_ids, merger_track_ids,
            snapshot_of_birth, snapshot_of_death, reverse_index
        )
        
        # Title logic
        if nodes:
            root_mass = 0
            if halo_masses is not None:
                 d = nodes[0]['death']
                 if d < 0: d = len(redshifts) - 1
                 root_mass = halo_masses[d, target_id] * mass_units
            title = f"ID {target_id}\n(M={root_mass:.2e})"
        else:
            title = f"ID {target_id} (Empty)"
            
        plot_single_merger_tree(nodes, redshifts, halo_masses, ax=ax, title=title)
        
        if i % 10 == 0:
            print(f"  Processed {i}/{n_targets}...")
    
    # Hide empty subplots
    for i in range(n_targets, len(axes)):
        axes[i].axis('off')

    if path_plots is not None and name is not None:
        save_fig(fig, path_plots, name)

    return fig

def plot_timeline_summary(
    target_track_ids: np.ndarray,
    track_ids: np.ndarray,
    merger_track_ids: np.ndarray,
    snapshot_of_birth: np.ndarray,
    snapshot_of_death: np.ndarray,
    redshifts: np.ndarray,
    reverse_index: Tuple[np.ndarray, np.ndarray],
    halo_masses: Optional[np.ndarray] = None,
    path_plots: Optional[str] = None,
    name: Optional[str] = None,
) -> plt.Figure:
    """One row per target track, marking when each progenitor merges in.

    Compresses a tree to its merger EPOCHS, which makes assembly timing
    comparable across halos in a way the tree drawings do not.
    """

    # Calculate height dynamically
    fig, ax = plt.subplots(figsize=(12, len(target_track_ids) * 0.4 + 2))
    
    y_ticks = []
    y_labels = []
    
    print("Generating timeline summary...")
    
    for y_pos, tid in enumerate(target_track_ids):
        nodes = get_merger_tree_structure(
            tid, track_ids, merger_track_ids,
            snapshot_of_birth, snapshot_of_death, reverse_index
        )
        
        if not nodes: continue
        
        # Main Branch (Root)
        root = nodes[0]
        start_z = redshifts[root['birth']]
        end_z = redshifts[max(0, redshifts.shape[0]-1)] if root['death'] < 0 else redshifts[root['death']]
        
        # Draw main life bar
        ax.hlines(y_pos, end_z, start_z, colors='black', alpha=0.3, linewidth=2, zorder=2)
        
        # Draw Mergers (Direct children of root)
        for node in nodes:
            if node['parent_list_idx'] == 0: # Is a direct child
                death_snap = node['death']
                if death_snap < 0: death_snap = len(redshifts)-1 # "Alive" satellite
                
                merge_z = redshifts[death_snap]
                
                # Size bubble by mass
                size = 80
                if halo_masses is not None:
                    m = halo_masses[death_snap, node['track_id']] * mass_units
                    if m > 0:
                        size = (np.log10(m) - 8) * 40
                        size = max(20, min(size, 500))
                
                sc = ax.scatter([merge_z], [y_pos], s=size, c='teal', alpha=0.6, edgecolors='none', zorder=1)

        y_ticks.append(y_pos)
        y_labels.append(f"ID {tid}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel("Redshift")
    ax.set_title("Merger History Timeline (Bubbles = Mergers)")
    ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()

    if path_plots is not None and name is not None:
        save_fig(fig, path_plots, name)

    return fig

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    from baqaro.core_functions.halo_mass_histories_saver import MergerTreeLoader
    from baqaro.core_functions.select_merger_branches import build_reverse_index
    from baqaro.utils.my_dir import get_output_path, get_input_path_HBT_data, get_plots_path


    # --- CONFIG ---
    source_dir = "machine_igm"
    # Override via BAQARO_MAX_SNAP env var. Available (the `_foldmass` arrays;
    # the token-less legacy set was deleted):
    #   L2800N5040  : {14, 38, 78}
    #   L2800N10080 : {39, 40, 50, 60, 71, 144}
    # Default = the sim's own production max_snap (sim_config), which is the one
    # value guaranteed to have arrays on disk for whichever BAQARO_SIM is active.
    # (The old hardcoded 18 resolved for NEITHER sim: L2800N10080 has no maxsnap18
    # at all, and L2800N5040 snap 18 was legacy-only -- deleted.)
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


    # 1. Load Redshifts
    path_sim = get_input_path_HBT_data(source=source_dir)
    redshift_file = os.path.join(path_sim, f"{simulation_name}/output_list.txt")
    try:
        snapshots = np.arange(0, max_snap + 1)
        full_z = np.loadtxt(redshift_file)
        redshifts = full_z[snapshots] if len(full_z) > max_snap else full_z
    except:
        print("Warning: Using dummy redshifts.")
        redshifts = np.linspace(10, 0, max_snap + 1)

    # 2. Load Data
    print("Loading merger trees...")
    merger_trees = MergerTreeLoader(path_trees)
    track_ids = np.array(merger_trees.track_ids)
    merger_track_ids = np.array(merger_trees.merger_track_ids)
    snap_birth = np.array(merger_trees.snapshot_indexes_of_birth)
    snap_death = np.array(merger_trees.snapshot_indexes_of_death)
    
    print("Loading masses...")
    halo_masses = np.load(path_mass, mmap_mode='r')

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

    # 4. Plotting
    target_track_ids = track_ids[:20] # Plot first 20
    
    plot_merger_tree_grid(
        target_track_ids, track_ids, merger_track_ids,
        snap_birth, snap_death, redshifts, reverse_index,
        halo_masses=halo_masses, ncols=5,
        path_plots=path_plots, name="final_merger_grid"
    )

    plot_timeline_summary(
        target_track_ids, track_ids, merger_track_ids,
        snap_birth, snap_death, redshifts, reverse_index,
        halo_masses=halo_masses,
        path_plots=path_plots, name="final_timeline"
    )

    print("Done.")
    maybe_show()