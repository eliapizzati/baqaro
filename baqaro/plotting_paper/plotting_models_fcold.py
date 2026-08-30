"""
PAPER FIGURE: cold accretion fraction f_cold(M_halo) at several redshifts.

f_cold is the Correa et al. (2018, MNRAS 473, 538) fitting formula for the
hot-mode accretion suppression in massive haloes:

    f_cold = 1 - 1 / (1 + (M_halo / M_1/2(z))^a(z))

with a redshift-dependent (piecewise-in-z) power-law index a(z) and
characteristic mass M_1/2(z). This script plots the SAME function the model
applies to the simulation specific accretion rates — it calls the canonical
implementation ``get_cold_accretion_fraction`` directly rather than
re-hardcoding the coefficients, so the figure can never drift from the model.

saves into the git-tracked ``figures_paper/`` via the shared paper config.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from baqaro.core_functions.halo_mass_histories_saver import (
    get_cold_accretion_fraction,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import cmap_z, REDSHIFT_TARGETS
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
name_fig = "models_fcold"

# Halo mass grid (in solar masses; mass_units=1.0 -> values already in M_sun).
log_M_min, log_M_max = 9.5, 14.5
log_M_halo = np.linspace(log_M_min, log_M_max, 500)
M_halo = 10.0 ** log_M_halo

# Same redshifts (and colour scale) as the halo-rate figure: span the shared
# REDSHIFT_TARGETS range with cmap_z, exactly mirroring plotting_halo_rate.py.
redshifts_plot = np.array(REDSHIFT_TARGETS)
z_lo, z_hi = redshifts_plot.min(), redshifts_plot.max()


def _z_color(z):
    return cmap_z((z - z_lo) / (z_hi - z_lo)) if z_hi > z_lo else cmap_z(0.5)


# ------------------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))

for z in redshifts_plot:
    f_cold = get_cold_accretion_fraction(M_halo, z, mass_units=1.0)
    ax.plot(log_M_halo, f_cold, lw=2.0, color=_z_color(z), label=f"z = {z:g}")
    # Dotted vertical at M_1/2(z): the mass where f_cold = 0.5 (f_cold decreases
    # with mass, so reverse for np.interp). Same style as the median lines in the
    # halo-rate figure.
    log_M_half = np.interp(0.5, f_cold[::-1], log_M_halo[::-1])
    ax.axvline(log_M_half, ls=":", lw=1.5, color=_z_color(z), zorder=-2)

# Reference: f_cold = 1 (pure cold accretion).
ax.axhline(1.0, color="black", ls=":", lw=1.2, zorder=-3)

ax.set_xlabel(r"log$_{10}$ M$_\mathrm{halo}$ [M$_\odot$]", labelpad=-1)
ax.set_ylabel(r"cold accretion fraction, f$_\mathrm{cold}$", labelpad=-1)
ax.set_xlim(log_M_min, log_M_max)
ax.set_ylim(-0.03, 1.08)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=cmap_z, norm=plt.Normalize(vmin=z_lo, vmax=z_hi)),
    ax=ax,
)
cbar.set_label("Redshift")
cbar.set_ticks(np.arange(z_lo, z_hi, 1.0))

fig.subplots_adjust(left=0.115, right=0.985, top=0.96, bottom=0.16)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
