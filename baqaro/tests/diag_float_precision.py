"""
Test precision of float32 vs float64 for accretion rate calculations
"""
import numpy as np

# Typical scenario: small mass change relative to large halo mass
M_prev_64 = np.array([1e12, 1e14, 1e15], dtype=np.float64)
M_curr_64 = M_prev_64 * 1.001  # 0.1% growth
dt = 0.1  # Gyr

# Accretion rates in float64
dM_64 = M_curr_64 - M_prev_64
specific_rate_64 = dM_64 / (dt * M_prev_64)

# Now in float32
M_prev_32 = M_prev_64.astype(np.float32)
M_curr_32 = (M_prev_64 * 1.001).astype(np.float32)
dM_32 = M_curr_32 - M_prev_32
specific_rate_32 = dM_32 / (dt * M_prev_32)

print("=" * 60)
print("PRECISION TEST: float32 vs float64")
print("=" * 60)

for i, M in enumerate([1e12, 1e14, 1e15]):
    diff_abs = np.abs(specific_rate_64[i] - specific_rate_32[i])
    diff_rel = diff_abs / np.abs(specific_rate_64[i]) if specific_rate_64[i] != 0 else 0
    
    print(f"\nMass: {M:.1e}")
    print(f"  Specific rate (float64): {specific_rate_64[i]:.6e}")
    print(f"  Specific rate (float32): {specific_rate_32[i]:.6e}")
    print(f"  Absolute difference:     {diff_abs:.6e}")
    print(f"  Relative error:          {diff_rel*100:.4f}%")

# Test with very small growth (< 1%)
print("\n" + "=" * 60)
print("EDGE CASE: Tiny accretion (0.01% growth)")
print("=" * 60)

M_base = np.array([1e15], dtype=np.float64)
M_tiny_growth = M_base * 1.0001  # 0.01% growth

dM_64 = (M_tiny_growth - M_base)[0]
rate_64 = dM_64 / (0.1 * M_base[0])

dM_32 = (M_tiny_growth.astype(np.float32) - M_base.astype(np.float32))[0]
rate_32 = dM_32 / (0.1 * M_base.astype(np.float32)[0])

print(f"dM (float64): {dM_64:.6e}")
print(f"dM (float32): {float(dM_32):.6e}")
print(f"Rate (float64): {rate_64:.6e}")
print(f"Rate (float32): {float(rate_32):.6e}")
print(f"Relative error: {np.abs(rate_64 - float(rate_32)) / rate_64 * 100:.4f}%")

# Test f_hot computation
print("\n" + "=" * 60)
print("TEST: f_hot computation")
print("=" * 60)

mass_units = 1.0
fcold_log_m_half = 12.8
fcold_smooth = -1.07
const_factor = mass_units / (10**fcold_log_m_half)

M_test = np.logspace(10, 15, 10)
M_test_32 = M_test.astype(np.float32)

f_hot_64 = 1.0 / (1.0 + (M_test * const_factor)**fcold_smooth)
f_hot_32 = 1.0 / (1.0 + (M_test_32 * const_factor)**fcold_smooth)

max_rel_error = np.max(np.abs(f_hot_64 - f_hot_32) / np.abs(f_hot_64 + 1e-10))
print(f"Max relative error in f_hot: {max_rel_error*100:.4f}%")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
if max_rel_error > 0.01:  # > 1%
    print("⚠️  Use float64: precision loss with float32 is significant (>1%)")
elif max_rel_error > 0.001:  # > 0.1%
    print("⚠️  Use float64: precision loss is borderline (~0.1-1%)")
else:
    print("✓ float32 acceptable: precision loss <0.1%")
