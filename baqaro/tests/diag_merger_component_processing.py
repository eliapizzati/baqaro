"""
Quick test of merger component processing approach.
Creates synthetic merger tree data and validates the batching logic.
"""

import numpy as np
import time
import tempfile
import os

# Import the functions we want to test. This was a bare sibling-file import
# that only resolved when run with `tests/` as the working directory; it is now
# an absolute package import of the renamed prototype module.
from baqaro.tests.diag_merger_component_prototype import (
    build_merger_connectivity_graph,
    create_component_batches,
    get_batch_indices
)


def create_synthetic_merger_trees(n_objects=1000000, merge_fraction=0.3, 
                                  component_structure="mixed"):
    """
    Create synthetic merger tree data for testing.
    
    Parameters:
    -----------
    n_objects : int
        Total number of objects
    merge_fraction : float
        Fraction of objects that merge (0-1)
    component_structure : str
        "isolated": Most objects are isolated (single-object components)
        "mixed": Mix of large and small components
        "clustered": Few large components
    
    Returns:
    --------
    merger_trees : dict
        Synthetic merger tree data
    """
    
    print(f"\nCreating synthetic merger trees ({n_objects:,} objects)...")
    
    track_ids = np.arange(n_objects, dtype=int)
    merger_track_ids = np.full(n_objects, -1, dtype=int)
    
    n_mergers = int(n_objects * merge_fraction)
    
    if component_structure == "isolated":
        # Most objects isolated, few small mergers
        for i in range(n_mergers):
            src = np.random.randint(0, n_objects)
            dest = np.random.randint(0, n_objects)
            if src != dest:
                merger_track_ids[src] = dest
    
    elif component_structure == "clustered":
        # Create a few large hub-and-spoke components
        n_hubs = 10
        hubs = np.random.choice(n_objects, size=n_hubs, replace=False)
        
        for i in range(n_mergers):
            src = np.random.randint(0, n_objects)
            hub = np.random.choice(hubs)
            if src not in hubs:
                merger_track_ids[src] = hub
    
    else:  # "mixed"
        # Mix of chain mergers and isolated objects
        for i in range(n_mergers):
            src = np.random.randint(0, n_objects)
            # 50% chance to merge with neighbor, 50% with random
            if np.random.rand() < 0.5 and src < n_objects - 1:
                dest = src + 1
            else:
                dest = np.random.randint(0, n_objects)
            
            if src != dest:
                merger_track_ids[src] = dest
    
    merger_trees = {
        "track_ids": track_ids,
        "merger_track_ids": merger_track_ids,
        "snapshot_indexes_of_birth": np.zeros(n_objects, dtype=int),
        "snapshot_indexes_of_death": np.full(n_objects, -1, dtype=int)
    }
    
    n_actual_mergers = np.sum(merger_track_ids != -1)
    print(f"  Created {n_actual_mergers:,} merger events ({n_actual_mergers/n_objects*100:.1f}%)")
    
    return merger_trees


def test_connectivity_graph(merger_trees, n_objects, cache_dir=None):
    """Test connectivity graph building."""
    
    print("\n" + "="*70)
    print("TEST 1: Connectivity Graph Building")
    print("="*70)
    
    start = time.time()
    labels, n_components, component_sizes = build_merger_connectivity_graph(
        merger_trees, n_objects, cache_dir=cache_dir
    )
    elapsed = time.time() - start
    
    # Validate results
    assert len(labels) == n_objects, "Labels array size mismatch"
    assert len(component_sizes) == n_components, "Component sizes mismatch"
    assert np.sum(component_sizes) == n_objects, "Total objects mismatch"
    
    print(f"\n✅ Test passed in {elapsed:.2f} seconds")
    print(f"   Found {n_components:,} components")
    print(f"   Isolated objects: {np.sum(component_sizes == 1):,}")
    print(f"   Multi-object components: {np.sum(component_sizes > 1):,}")
    
    return labels, n_components, component_sizes


def test_batching(labels, n_components, component_sizes, n_batches=10):
    """Test component batching."""
    
    print("\n" + "="*70)
    print("TEST 2: Component Batching")
    print("="*70)
    
    start = time.time()
    batches, batch_sizes = create_component_batches(
        labels, n_components, component_sizes, n_batches=n_batches
    )
    elapsed = time.time() - start
    
    # Validate results
    assert len(batches) == n_batches, "Wrong number of batches"
    assert len(batch_sizes) == n_batches, "Batch sizes mismatch"
    
    # Check that all components are assigned
    all_components = set()
    for batch in batches:
        all_components.update(batch)
    assert len(all_components) == n_components, "Not all components assigned"
    
    # Check that batch sizes sum to total
    total_batch_objects = sum(batch_sizes)
    total_objects = sum(component_sizes)
    assert total_batch_objects == total_objects, "Batch size sum mismatch"
    
    # Check load balance
    max_size = max(batch_sizes)
    min_size = min(batch_sizes)
    avg_size = np.mean(batch_sizes)
    balance = (avg_size / max_size) * 100
    
    print(f"\n✅ Test passed in {elapsed:.2f} seconds")
    print(f"   Load balance: {balance:.1f}%")
    print(f"   Size ratio (max/min): {max_size/max(min_size,1):.2f}x")
    
    return batches, batch_sizes


def test_batch_isolation(labels, batches, merger_trees):
    """Test that batches are truly independent (no mergers cross batch boundaries)."""
    
    print("\n" + "="*70)
    print("TEST 3: Batch Isolation (Critical for Correctness)")
    print("="*70)
    
    merger_track_ids = merger_trees["merger_track_ids"]
    
    violations = 0
    
    for batch_id, component_ids in enumerate(batches):
        # Get all objects in this batch
        batch_indices = get_batch_indices(labels, component_ids)
        batch_set = set(batch_indices)
        
        # Check that all mergers stay within batch
        for obj_idx in batch_indices:
            merger_partner = merger_track_ids[obj_idx]
            if merger_partner != -1:  # Has a merger
                if merger_partner not in batch_set:
                    violations += 1
                    if violations <= 5:  # Only print first few
                        print(f"  ⚠️  Object {obj_idx} in batch {batch_id} merges with "
                              f"{merger_partner} (outside batch)")
    
    if violations == 0:
        print(f"\n✅ Test passed - All mergers contained within batches")
    else:
        print(f"\n❌ Test FAILED - {violations} mergers cross batch boundaries!")
        raise AssertionError("Batch isolation violated!")


def test_caching(merger_trees, n_objects, cache_dir):
    """Test that caching works correctly."""
    
    print("\n" + "="*70)
    print("TEST 4: Caching")
    print("="*70)
    
    # First call - should build graph
    print("First call (building graph)...")
    start = time.time()
    labels1, n_comp1, sizes1 = build_merger_connectivity_graph(
        merger_trees, n_objects, cache_dir=cache_dir
    )
    time1 = time.time() - start
    
    # Second call - should load from cache
    print("\nSecond call (loading from cache)...")
    start = time.time()
    labels2, n_comp2, sizes2 = build_merger_connectivity_graph(
        merger_trees, n_objects, cache_dir=cache_dir
    )
    time2 = time.time() - start
    
    # Validate that results are identical
    assert np.array_equal(labels1, labels2), "Cached labels differ!"
    assert n_comp1 == n_comp2, "Cached n_components differ!"
    assert np.array_equal(sizes1, sizes2), "Cached sizes differ!"
    
    speedup = time1 / time2
    
    print(f"\n✅ Test passed")
    print(f"   First call:  {time1:.2f} seconds")
    print(f"   Second call: {time2:.2f} seconds")
    print(f"   Speedup: {speedup:.1f}x")


def run_all_tests():
    """Run all tests."""
    
    print("="*70)
    print("MERGER COMPONENT PROCESSING - TEST SUITE")
    print("="*70)
    
    # Test parameters
    n_objects = 100000  # 100K for quick testing
    n_batches = 10
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    print(f"Using temporary cache directory: {cache_dir}")
    
    try:
        # Create synthetic data
        merger_trees = create_synthetic_merger_trees(
            n_objects=n_objects,
            merge_fraction=0.3,
            component_structure="mixed"
        )
        
        # Test 1: Connectivity graph
        labels, n_components, component_sizes = test_connectivity_graph(
            merger_trees, n_objects, cache_dir=cache_dir
        )
        
        # Test 2: Batching
        batches, batch_sizes = test_batching(
            labels, n_components, component_sizes, n_batches=n_batches
        )
        
        # Test 3: Batch isolation (critical!)
        test_batch_isolation(labels, batches, merger_trees)
        
        # Test 4: Caching
        test_caching(merger_trees, n_objects, cache_dir)
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        print("\nApproach is validated and ready for production use!")
        print(f"Tested with {n_objects:,} objects, {n_components:,} components, {n_batches} batches")
        
    finally:
        # Cleanup cache
        import shutil
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"\nCleaned up temporary cache directory")


if __name__ == "__main__":
    run_all_tests()
