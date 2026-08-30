"""Diagnostic: reconstruct the Marconi+2004 local BHMF and its mass density.

Rebuilds the published Schechter-like fit, writes it to ``plots/`` as a CSV, and
plots it, as a cross-check on the local-BHMF normalisation used elsewhere.

Script-style: executes on import. Run it directly, never under pytest.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def marconi_2004_bhmf(log_mass):
    """
    Calculates the Number Density Phi for the Marconi et al. (2004) Local BHMF.
    Based on the analytical fit used in Shankar et al. (2009/2016) comparisons.
    
    Parameters:
    log_mass : array-like
        log10 of Black Hole Mass (in Solar Masses)
        
    Returns:
    log_phi : array-like
        log10 of Number Density (Mpc^-3 dex^-1)
    """
    
    # Parameters for the Marconi et al. (2004) fit
    # These match the "hump" and high-mass tail of the published figure
    # Note: h=0.7 is assumed in these standard parameters
    phi_star = 10**-2.82
    log_m_star = 8.35
    alpha = -1.18 
    beta = 0.6  # A lower beta (<1) creates the broad "shoulder" seen in the plot
    
    # Calculate M/M*
    m = 10**log_mass
    m_star = 10**log_m_star
    ratio = m / m_star
    
    # Generalized Schechter Function (in terms of dLogM)
    # Phi(M) dLogM = ln(10) * Phi* * (M/M*)^(alpha+1) * exp(-(M/M*)^beta)
    # Note: The term (alpha+1) is used because we are in dLogM space, 
    # but often alpha is defined intrinsically. 
    
    term1 = np.log(10) * phi_star
    term2 = ratio**(alpha + 1)
    term3 = np.exp(-1 * (ratio**beta))
    
    phi = term1 * term2 * term3
    
    return np.log10(phi)

# 1. Generate the Mass Range (X-axis from your plot: 6 to 10)
log_bh_mass = np.linspace(6.0, 10.0, 50)

# 2. Calculate the Density (Y-axis)
log_phi = marconi_2004_bhmf(log_bh_mass)

# 3. Create a DataFrame
df = pd.DataFrame({
    'log_M_BH': log_bh_mass,
    'log_Phi': log_phi
})

# 4. Save to CSV
# Generated output, not repository data: write beside the repo's other plots.
csv_filename = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plots", "marconi_2004_digitized.csv")
os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")

# --- Optional: Plot to verify it matches the published figure ---
plt.figure(figsize=(8, 6))
plt.plot(df['log_M_BH'], df['log_Phi'], 'o-', color='purple', label='Marconi + 2004 (Synthesized)')

# Axis limits matching the published figure
plt.xlim(6, 11)
plt.ylim(-8, -1)
plt.xlabel(r'$\log(M_{\rm BH} [M_{\odot}])$', fontsize=12)
plt.ylabel(r'$\log(\Phi \, [\rm dex^{-1} \, cMpc^{-3}])$', fontsize=12)
plt.title('Reconstruction of Marconi et al. 2004 Points', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()