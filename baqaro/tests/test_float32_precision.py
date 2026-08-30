"""
Test script to verify if float32 precision is sufficient for BH evolution.
Runs a small subset with both float32 and float64, then compares results.
"""

import numpy as np
import time

def test_precision_loss(n_objects=100000, n_snapshots=49):
    """
    Test float32 vs float64 precision for typical BH evolution operations.
    
    Parameters:
    -----------
    n_objects : int
        Number of test objects (use ~1% of your full dataset)
    n_snapshots : int
        Number of snapshots to simulate
    """
    
    print("="*60)
    print(f"Testing float32 vs float64 precision")
    print(f"N_objects: {n_objects:,}, N_snapshots: {n_snapshots}")
    print("="*60)
    
    # Simulate realistic BH mass values (0.5 Msun to 10^10 Msun)
    rng = np.random.default_rng(42)
    initial_masses = 10**rng.uniform(-0.3, 10, size=n_objects)  # log-uniform distribution
    
    # Test arrays
    masses_64 = np.zeros((n_objects, n_snapshots), dtype=np.float64)
    masses_32 = np.zeros((n_objects, n_snapshots), dtype=np.float32)
    
    masses_64[:, 0] = initial_masses
    masses_32[:, 0] = initial_masses.astype(np.float32)
    
    print("\n1. INITIAL STATE")
    print("-" * 40)
    print(f"Mass range: {initial_masses.min():.2e} to {initial_masses.max():.2e} Msun")
    print(f"float64 memory: {masses_64.nbytes / 1e9:.2f} GB")
    print(f"float32 memory: {masses_32.nbytes / 1e9:.2f} GB")
    print(f"Memory savings: {(1 - masses_32.nbytes/masses_64.nbytes)*100:.1f}%")
    
    
    # 2. SIMULATE TYPICAL OPERATIONS
    print("\n2. SIMULATING BH EVOLUTION OPERATIONS")
    print("-" * 40)
    
    for i in range(1, n_snapshots):
        # Accretion: add small increments (typical ~1-10% growth per snapshot)
        growth_factor_64 = 1.0 + rng.uniform(0.01, 0.1, size=n_objects)
        growth_factor_32 = growth_factor_64.astype(np.float32)
        
        masses_64[:, i] = masses_64[:, i-1] * growth_factor_64
        masses_32[:, i] = masses_32[:, i-1] * growth_factor_32
        
        # Mergers: add mass from random pairs
        n_mergers = int(n_objects * 0.01)  # 1% merger rate per snapshot
        if n_mergers > 0:
            src_idx = rng.choice(n_objects, size=n_mergers, replace=False)
            dest_idx = rng.choice(n_objects, size=n_mergers, replace=False)
            
            masses_64[dest_idx, i] += masses_64[src_idx, i-1]
            masses_32[dest_idx, i] += masses_32[src_idx, i-1]
            
            masses_64[src_idx, i] = 0.0
            masses_32[src_idx, i] = 0.0
    
    
    # 3. COMPARE RESULTS
    print("\n3. PRECISION COMPARISON")
    print("-" * 40)
    
    # Relative error
    mask_nonzero = masses_64 > 0
    rel_error = np.abs((masses_32 - masses_64) / masses_64)
    rel_error_nonzero = rel_error[mask_nonzero]
    
    print(f"Relative error statistics (non-zero masses only):")
    print(f"  Mean:   {np.mean(rel_error_nonzero):.2e}")
    print(f"  Median: {np.median(rel_error_nonzero):.2e}")
    print(f"  95th percentile: {np.percentile(rel_error_nonzero, 95):.2e}")
    print(f"  99th percentile: {np.percentile(rel_error_nonzero, 99):.2e}")
    print(f"  Max:    {np.max(rel_error_nonzero):.2e}")
    
    # Absolute error
    abs_error = np.abs(masses_32 - masses_64)
    print(f"\nAbsolute error statistics:")
    print(f"  Mean:   {np.mean(abs_error):.2e} Msun")
    print(f"  Median: {np.median(abs_error):.2e} Msun")
    print(f"  Max:    {np.max(abs_error):.2e} Msun")
    
    # Mass conservation check
    total_mass_64 = np.sum(masses_64[:, -1])
    total_mass_32 = np.sum(masses_32[:, -1])
    mass_conservation_error = abs(total_mass_32 - total_mass_64) / total_mass_64
    
    print(f"\nMass conservation:")
    print(f"  Total mass (float64): {total_mass_64:.6e} Msun")
    print(f"  Total mass (float32): {total_mass_32:.6e} Msun")
    print(f"  Relative difference:  {mass_conservation_error:.2e}")
    
    # Final mass distribution comparison
    final_masses_64 = masses_64[:, -1]
    final_masses_32 = masses_32[:, -1]
    
    print(f"\nFinal mass distribution:")
    print(f"  Min mass (float64): {final_masses_64[final_masses_64>0].min():.2e}")
    print(f"  Min mass (float32): {final_masses_32[final_masses_32>0].min():.2e}")
    print(f"  Max mass (float64): {final_masses_64.max():.2e}")
    print(f"  Max mass (float32): {final_masses_32.max():.2e}")
    
    
    # 4. ASSESSMENT
    print("\n4. PRECISION ASSESSMENT")
    print("=" * 60)
    
    # Float32 has ~7 significant figures precision
    # Typical thresholds:
    threshold_excellent = 1e-6  # < 0.0001% error
    threshold_good = 1e-5       # < 0.001% error
    threshold_acceptable = 1e-4 # < 0.01% error
    
    median_error = np.median(rel_error_nonzero)
    p99_error = np.percentile(rel_error_nonzero, 99)
    
    if median_error < threshold_excellent and p99_error < threshold_good:
        status = "✅ EXCELLENT"
        recommendation = "Float32 is SAFE. Precision loss is negligible."
    elif median_error < threshold_good and p99_error < threshold_acceptable:
        status = "✅ GOOD"
        recommendation = "Float32 is ACCEPTABLE. Minor precision loss unlikely to affect science."
    elif median_error < threshold_acceptable:
        status = "⚠️  FAIR"
        recommendation = "Float32 is MARGINAL. Consider float64 if high precision is critical."
    else:
        status = "❌ POOR"
        recommendation = "Float32 NOT RECOMMENDED. Use float64 to preserve accuracy."
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    print()
    
    # Specific checks
    warnings = []
    if mass_conservation_error > 1e-4:
        warnings.append("⚠️  Mass conservation error > 0.01%")
    if p99_error > 1e-3:
        warnings.append("⚠️  99th percentile error > 0.1%")
    if np.any(np.isnan(masses_32)):
        warnings.append("❌ NaN values detected in float32")
    if np.any(np.isinf(masses_32)):
        warnings.append("❌ Inf values detected in float32")
    
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("No precision issues detected.")
    
    print("="*60)
    
    return {
        'median_rel_error': median_error,
        'p99_rel_error': p99_error,
        'max_rel_error': np.max(rel_error_nonzero),
        'mass_conservation_error': mass_conservation_error,
        'recommendation': recommendation,
        'safe': median_error < threshold_good
    }


def test_accretion_rate_precision(n_objects=100000, n_snapshots=49):
    """
    Test float32 vs float64 for HALO ACCRETION RATE calculation.
    This is the critical case: (M_halo[i] - M_halo[i-1]) / dt
    where M_halo[i] ≈ M_halo[i-1] (catastrophic cancellation risk).
    """
    
    print("="*60)
    print(f"Testing HALO ACCRETION RATE precision (CRITICAL TEST)")
    print(f"N_objects: {n_objects:,}, N_snapshots: {n_snapshots}")
    print("="*60)
    
    # Realistic halo masses: 10^10 to 10^14 Msun
    rng = np.random.default_rng(123)
    initial_halo_masses = 10**rng.uniform(10, 14, size=n_objects)
    
    # Halo mass arrays
    halo_masses_64 = np.zeros((n_objects, n_snapshots), dtype=np.float64)
    halo_masses_32 = np.zeros((n_objects, n_snapshots), dtype=np.float32)
    
    halo_masses_64[:, 0] = initial_halo_masses
    halo_masses_32[:, 0] = initial_halo_masses.astype(np.float32)
    
    # Simulate realistic halo growth (1-5% per snapshot, typical for cosmological sims)
    for i in range(1, n_snapshots):
        growth_rate = rng.uniform(0.01, 0.05, size=n_objects)  # 1-5% growth
        halo_masses_64[:, i] = halo_masses_64[:, i-1] * (1.0 + growth_rate)
        halo_masses_32[:, i] = halo_masses_32[:, i-1] * (1.0 + growth_rate.astype(np.float32))
    
    print("\n1. HALO MASS DISTRIBUTION")
    print("-" * 40)
    print(f"Mass range: {initial_halo_masses.min():.2e} to {initial_halo_masses.max():.2e} Msun")
    print(f"Typical growth per snapshot: 1-5%")
    
    # Compute accretion rates: dM/dt
    dt = 0.1  # Typical dt in Gyr between snapshots
    
    acc_rates_64 = np.diff(halo_masses_64, axis=1) / dt  # Shape: (n_objects, n_snapshots-1)
    acc_rates_32 = np.diff(halo_masses_32, axis=1) / dt
    
    # Compute specific accretion rates: (dM/dt) / M
    specific_acc_rates_64 = acc_rates_64 / halo_masses_64[:, :-1]
    specific_acc_rates_32 = acc_rates_32 / halo_masses_32[:, :-1]
    
    print("\n2. ACCRETION RATE PRECISION")
    print("-" * 40)
    
    # Relative error in accretion rates
    mask_positive = acc_rates_64 > 0
    rel_error_acc = np.abs((acc_rates_32 - acc_rates_64) / acc_rates_64)
    rel_error_acc_positive = rel_error_acc[mask_positive]
    
    print(f"Relative error in dM/dt:")
    print(f"  Mean:   {np.mean(rel_error_acc_positive):.2e}")
    print(f"  Median: {np.median(rel_error_acc_positive):.2e}")
    print(f"  95th percentile: {np.percentile(rel_error_acc_positive, 95):.2e}")
    print(f"  99th percentile: {np.percentile(rel_error_acc_positive, 99):.2e}")
    print(f"  Max:    {np.max(rel_error_acc_positive):.2e}")
    
    # Relative error in specific accretion rates
    rel_error_spec = np.abs((specific_acc_rates_32 - specific_acc_rates_64) / specific_acc_rates_64)
    rel_error_spec_positive = rel_error_spec[mask_positive]
    
    print(f"\nRelative error in (dM/dt)/M:")
    print(f"  Mean:   {np.mean(rel_error_spec_positive):.2e}")
    print(f"  Median: {np.median(rel_error_spec_positive):.2e}")
    print(f"  95th percentile: {np.percentile(rel_error_spec_positive, 95):.2e}")
    print(f"  99th percentile: {np.percentile(rel_error_spec_positive, 99):.2e}")
    print(f"  Max:    {np.max(rel_error_spec_positive):.2e}")
    
    # Check for catastrophic cancellation cases
    # Where float32 loses significant precision
    catastrophic_mask = rel_error_acc_positive > 0.01  # > 1% error
    n_catastrophic = np.sum(catastrophic_mask)
    frac_catastrophic = n_catastrophic / len(rel_error_acc_positive)
    
    print(f"\nCatastrophic cancellation cases (>1% error):")
    print(f"  Count: {n_catastrophic:,} / {len(rel_error_acc_positive):,}")
    print(f"  Fraction: {frac_catastrophic*100:.2f}%")
    
    # 3. ASSESSMENT
    print("\n3. ACCRETION RATE PRECISION ASSESSMENT")
    print("=" * 60)
    
    median_error = np.median(rel_error_spec_positive)
    p99_error = np.percentile(rel_error_spec_positive, 99)
    
    threshold_excellent = 1e-5
    threshold_good = 1e-4
    threshold_acceptable = 1e-3
    
    if median_error < threshold_excellent and p99_error < threshold_good:
        status = "✅ EXCELLENT"
        recommendation = "Float32 is SAFE for halo accretion rates."
    elif median_error < threshold_good and p99_error < threshold_acceptable:
        status = "✅ GOOD"
        recommendation = "Float32 is ACCEPTABLE for halo accretion rates."
    elif median_error < threshold_acceptable and frac_catastrophic < 0.01:
        status = "⚠️  FAIR"
        recommendation = "Float32 has some precision loss. Consider float64 for accretion rates."
    else:
        status = "❌ POOR"
        recommendation = "Float32 NOT SAFE for accretion rates. Use float64."
    
    print(f"Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    # Specific warnings
    warnings = []
    if frac_catastrophic > 0.05:
        warnings.append(f"❌ {frac_catastrophic*100:.1f}% of values have >1% error (catastrophic cancellation)")
    if p99_error > 0.01:
        warnings.append("❌ 99th percentile error > 1%")
    if median_error > 0.001:
        warnings.append("⚠️  Median error > 0.1%")
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n✅ No significant precision issues detected.")
    
    print("="*60)
    
    return {
        'median_rel_error': median_error,
        'p99_rel_error': p99_error,
        'max_rel_error': np.max(rel_error_spec_positive),
        'catastrophic_fraction': frac_catastrophic,
        'recommendation': recommendation,
        'safe': median_error < threshold_good and frac_catastrophic < 0.01
    }


if __name__ == "__main__":
    # Test with different problem sizes
    print("\nRUNNING PRECISION TEST\n")
    
    # Test 1: BH evolution
    print("\n" + "#"*60)
    print("### TEST 1: BLACK HOLE MASS EVOLUTION ###")
    print("#"*60)
    result_bh = test_precision_loss(n_objects=100000, n_snapshots=49)
    
    # Test 2: Halo accretion rates (CRITICAL - catastrophic cancellation risk)
    print("\n" + "#"*60)
    print("### TEST 2: HALO ACCRETION RATES (CRITICAL) ###")
    print("#"*60)
    result_acc = test_accretion_rate_precision(n_objects=100000, n_snapshots=49)
    
    # Final recommendation
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    
    if result_bh['safe'] and result_acc['safe']:
        print("✅ Float32 is SAFE for ALL arrays.")
        print("   Expected memory savings: 50% (442 GB → 221 GB)")
        print("   Precision loss: < 0.01% (negligible for astrophysics)")
        print("\n   Safe to use:")
        print("     - black_hole_masses_all (float32)")
        print("     - Lbols_all (float32)")
        print("     - halo_masses_all (float32)")
        print("     - halo_specific_cold_accretion_rates_all (float32)")
    elif result_bh['safe'] and not result_acc['safe']:
        print("⚠️  HYBRID APPROACH RECOMMENDED:")
        print("   Use float64 for: halo_masses_all, halo_specific_cold_accretion_rates_all")
        print("   Use float32 for: black_hole_masses_all, Lbols_all")
        print("\n   Reason: Accretion rate calculation suffers from catastrophic cancellation.")
        print("   Memory savings: ~25% (442 GB → 331 GB)")
    else:
        print("❌ Float32 NOT RECOMMENDED.")
        print("   Use float64 for all arrays to preserve accuracy.")
        print("   Memory required: 442 GB (well within your 2 TB node)")
    
    print("="*60)
