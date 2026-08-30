"""Merger-tree construction and layout, shared by the two single-tree figures.

``plot_merger_trees_single.py`` and ``plot_merger_trees_single_vs_z.py`` draw the
same tree on different x-axes -- topology/snapshot index versus redshift -- and
carried BYTE-IDENTICAL copies of ``build_tree_debug`` and ``get_layout`` (76
lines) until they were lifted here. Only ``plot_debug_tree``
genuinely differs between them, which is the whole point of having two modules.

Keep this module free of matplotlib: it computes the tree and its coordinates,
the callers draw it.
"""

import numpy as np


def build_tree_debug(target_id, track_ids, merger_track_ids, snap_birth, snap_death, reverse_index):
    """Breadth-first walk of one halo's merger tree, rooted at ``target_id``.

    Uses ``reverse_index`` -- a CSR-style ``(indptr, indices)`` pair mapping a
    track to the tracks that merged INTO it -- so each node's progenitors are a
    contiguous slice rather than a scan over the full catalogue.

    Returns a list of node dicts (``track_id``, ``depth``, ``birth``, ``death``,
    ``list_idx``, ``parent_list_idx``). Nodes whose birth snapshot is -1 are kept
    in the list but flagged invalid, so the caller can report them rather than
    drop branches unreported; ``get_layout`` skips them.
    """
    indptr, indices = reverse_index
    tree_nodes = []
    
    # Queue: (track_id, depth, parent_list_index)
    queue = [(target_id, 0, -1)]
    visited = set()
    
    while queue:
        current_id, depth, parent_idx = queue.pop(0)
        
        if current_id in visited: continue
        visited.add(current_id)
        
        # Get Vital Stats
        birth = snap_birth[current_id]
        death = snap_death[current_id]
        
        # Add to list
        current_list_idx = len(tree_nodes)
        
        # We store it, but we might filter it later
        tree_nodes.append({
            'list_idx': current_list_idx,
            'track_id': current_id,
            'depth': depth,
            'birth': birth,
            'death': death,
            'parent_list_idx': parent_idx
        })
        
        # Find Progenitors
        if current_id < len(indptr) - 1:
            start = indptr[current_id]
            end = indptr[current_id+1]
            progenitors = indices[start:end]
            
            for prog_id in progenitors:
                queue.append((prog_id, depth + 1, current_list_idx))
                
    return tree_nodes


def get_layout(nodes):
    """Horizontal position of every valid node, so branches do not overlap.

    Leaves take consecutive integer slots left to right; an internal node sits at
    the mean of its children, which centres each branch over its own subtree.
    Invalid nodes (``birth == -1``) are excluded from the layout.

    Returns an array indexed like ``nodes`` (invalid entries left at 0).
    """
    # Only layout VALID nodes
    valid_indices = [i for i, n in enumerate(nodes) if n['birth'] != -1]
    
    # Map old list index to new valid index to fix parent pointers
    # (Simplified: we just rebuild adjacency for valid nodes only)
    
    children_map = {i: [] for i in range(len(nodes))}
    for i in valid_indices:
        p = nodes[i]['parent_list_idx']
        if p != -1 and nodes[p]['birth'] != -1: # Only if parent is also valid
            children_map[p].append(i)
            
    x_pos = np.zeros(len(nodes))
    next_leaf = 0.0
    
    def assign_x(idx):
        """Place node ``idx``, recursing depth-first into its children."""
        nonlocal next_leaf
        children = children_map[idx]
        
        if not children:
            x_pos[idx] = next_leaf
            next_leaf += 1.0
        else:
            child_xs = []
            for child in children:
                assign_x(child)
                child_xs.append(x_pos[child])
            x_pos[idx] = sum(child_xs) / len(child_xs)
            
    # Find the root(s) - usually index 0
    if nodes and nodes[0]['birth'] != -1:
        assign_x(0)
        
    return x_pos
