"""
PAPER FIGURE: spin-dependent radiative efficiency epsilon(eta_acc).

The accretion-rate-dependent radiative efficiency follows the Madau et al.
(2014) spin-dependent fit:

    eps(eta) = eps_base * A / eta * (0.985/(1.6/eta + B) + 0.015/(1.6/eta + C))

At low Eddington ratio eps drops (radiatively inefficient flows); it
approaches the spin-set ceiling at high eta. This script calls the canonical
``get_madau_efficiency_epsilon`` used by the BH-evolution engine directly, so
the figure tracks the actual model coefficients (spin a ~= 0.67, encoded in
A, B, C) instead of a hardcoded copy.

Split out of an earlier combined analysis figure, which packed f_cold +
epsilon into one figure with a hardcoded a=0.6 set; this module and
``plotting_models_fcold.py`` supersede it. Saves into the
git-tracked ``figures_paper/`` via the shared paper config.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from baqaro.core_functions.bh_accretion import (
    get_madau_efficiency_epsilon,
)

from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
name_fig = "models_radiative_efficiency"

# Base (high-eta ceiling) radiative efficiency used by the model.
rad_efficiency_base = 0.1

# Eddington-ratio grid in log10.
log_eta_min, log_eta_max = -3.0, 2.5
log_eta = np.linspace(log_eta_min, log_eta_max, 500)
eta = 10.0 ** log_eta


# ------------------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.6))

# matplotlib's mathtext maps BOTH \epsilon and \varepsilon to U+03B5 (the curly
# variant), so the lunate epsilon has to be given as the literal U+03F5 glyph.
EPS = "ϵ"

eps = get_madau_efficiency_epsilon(eta, rad_efficiency=rad_efficiency_base)
ax.plot(log_eta, eps, color="black", lw=2.0,
        label=r"$%s(\eta_\mathrm{acc})$ model" % EPS)

# Reference: constant base efficiency.
ax.axhline(rad_efficiency_base, ls=":", lw=1.5, color="black",
           label=r"$%s = %.1f$" % (EPS, rad_efficiency_base))

ax.set_xlabel(r"$\log_{10}\,\eta_\mathrm{acc}$", labelpad=-0.5)
ax.set_ylabel(r"radiative efficiency, $%s$" % EPS)
ax.set_xlim(log_eta_min, log_eta_max)
ax.set_yscale("log")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.legend()

fig.subplots_adjust(left=0.135, right=0.97, top=0.95, bottom=0.25)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
