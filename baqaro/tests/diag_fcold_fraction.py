"""Diagnostic: cold-accretion fraction against halo mass and redshift.

Plots f_cold(M200, z) as the model applies it, to see where the sigmoid turns
over relative to the halo masses that actually host quasars.

Script-style: executes on import. Run it directly, never under pytest.
"""

import numpy as np
import matplotlib.pyplot as plt

def get_accretion_fractions(M200, z):
    """
    Calculates the hot and cold accretion fractions based on the parametrization.
    Returns both (f_cold, f_hot).
    """
    # Ensure inputs are arrays
    M200 = np.atleast_1d(M200)
    z = np.atleast_1d(z)
    
    # --- Eq (14): z_tilde ---
    z_tilde = np.log10(1 + z)
    
    # --- Eq (13): a(z) ---
    a_z = np.zeros_like(z, dtype=float)
    mask1 = (z >= 0) & (z < 2)
    mask2 = (z >= 2) & (z < 4)
    mask3 = (z >= 4)
    
    # 0 <= z < 2
    exponent1 = -1.26 * z_tilde[mask1] + 1.29 * (z_tilde[mask1]**2)
    a_z[mask1] = -1.86 * (10**exponent1)
    
    # 2 <= z < 4
    exponent2 = 0.81 * z_tilde[mask2] - 0.42 * (z_tilde[mask2]**2)
    a_z[mask2] = -0.46 * (10**exponent2)
    
    # z >= 4
    a_z[mask3] = -1.07
    
    # --- Eq (15): M_1/2(z) ---
    log_factor = np.zeros_like(z, dtype=float)
    
    # 0 <= z < 2
    log_factor[mask1] = -0.15 + 0.22 * z[mask1] + 0.07 * (z[mask1]**2)
    # 2 <= z < 4
    log_factor[mask2] = -0.25 + 0.53 * z[mask2] - 0.07 * (z[mask2]**2)
    # z >= 4
    log_factor[mask3] = 0.72 + 0.01 * z[mask3]
    
    M_half_z = (10**12) * (10**log_factor)
    
    # --- Eq (12): f_acc,hot ---
    # We use broadcasting here. 
    # If M200 is shape (N,) and z is scalar, result is (N,)
    f_hot = 1.0 / (1.0 + (M200 / M_half_z)**a_z)
    f_cold = 1.0 - f_hot
    
    return f_cold, f_hot

def recreate_plot():
    # 1. Setup the Mass range (X-axis)
    # Log range from 11 to 14 as seen in the plot
    log_M200 = np.linspace(11.0, 14.0, 100)
    M200 = 10**log_M200

    # 2. Define Redshifts and Styles to match the image
    # Note: The image shows analytic lines (fits) and data points (simulation).
    # We are plotting the analytic lines defined by the equations.
    zs_styles = [
        {'z': 0, 'label': 'z=0', 'c': '#999933', 'ls': '-'},       # Olive/Yellow
        {'z': 1, 'label': 'z=1', 'c': '#FFCC66', 'ls': '--'},      # Light Orange
        {'z': 2, 'label': 'z=2', 'c': '#FF8800', 'ls': '--'},      # Orange (long dash)
        {'z': 3, 'label': 'z=3', 'c': '#FF4400', 'ls': '-.'},      # Red-Orange
        {'z': 4, 'label': 'z=4', 'c': '#880022', 'ls': '-.'},      # Dark Red/Burgundy
        {'z': 5, 'label': 'z=5', 'c': '#6600CC', 'ls': ':'},       # Purple
        {'z': 6, 'label': 'z=6', 'c': 'blue',    'ls': '-'},       # Blue
    ]

    # 3. Create Plot
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    for item in zs_styles:
        z_val = item['z']
        # Calculate fractions
        _, f_hot = get_accretion_fractions(M200, z_val)
        
        # Adjust dash pattern for z=2 to look like "long dashes" if needed
        # and z=4 to look like dash-dot-dot
        linestyle = item['ls']
        if z_val == 2:
            linestyle = (0, (5, 5)) # Custom long dash
        if z_val == 4:
            linestyle = (0, (3, 1, 1, 1)) # Dash-dot-dot-ish

        ax.plot(log_M200, f_hot, 
                label=item['label'], 
                color=item['c'], 
                linestyle=linestyle, 
                linewidth=1.5)

    # 4. Styling and Formatting
    ax.set_xlabel(r'$\log_{10} M_{200} \ [M_{\odot}]$', fontsize=12)
    ax.set_ylabel(r'$f_{\text{acc,hot}}$', fontsize=12)
    
    ax.set_xlim(11.0, 14.0)
    ax.set_ylim(0.0, 1.02)

    # Add the text annotation in top left
    ax.text(11.1, 0.93, 'Ref-L100N1504', fontsize=12, color='black')

    # Tick formatting (minor ticks and direction)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=10)

    # Legend
    ax.legend(frameon=False, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    recreate_plot()