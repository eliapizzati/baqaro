"""
PAPER FIGURE: model BH specific accretion rate vs. halo specific accretion rate.

The ERDF model relates the Eddington ratio of each BH to the halo specific
cold accretion rate r = f_cold * M_dot_halo / M_halo via

    log10(eta) ~ Normal(mu(r), sigma(r))
    mu(r)    = log_eta_mean_0 + log_eta_mean_evol * log10(r)
    sigma(r) = std_0

Definitions used here:

    eta       = M_dot_acc / M_dot_Edd          (gross Eddington-ratio = ERDF variable)
    M_dot_Edd = L_Edd / (eps0 * c^2)           (Eddington rate, eps0 = 0.1 fixed)

We plot the GROSS specific accretion rate on the left axis, so both axes carry
the same (gross) quantity and the twin axis is a clean constant offset that maps
EXACTLY onto eta:

    M_dot_acc / M_BH [Gyr^-1] = eta / t_Sal0   (left)
    M_dot_acc / M_dot_Edd     = eta            (right)

with the canonical Salpeter time t_Sal0 = eps0 * sigma_T c / (4 pi G m_p)
~= 0.045 Gyr, i.e. M_dot_Edd / M_BH = 1/t_Sal0 ~= 22.18 Gyr^-1.  The right axis
is the left axis shifted down by log10(1/t_Sal0[Gyr]) ~= 1.346 dex.

(The mass actually retained is (1-eps)*M_dot_acc; we deliberately plot the gross
accretion rate so the right axis is exactly the ERDF variable eta, with no
eta-dependent (1-eps) factor distorting the secondary-axis mapping.)

The ERDF parameters are read from the evolution HDF5 selected by
``load_data_to_plot``'s env vars (``BAQARO_SIM``, ``BAQARO_MAX_SNAP``, ...).  This
is a pure model-curve figure (no population number-density estimate), so no
subsample weighting is involved.

Adapted for the paper from the working version of this figure; the
only change is the output plumbing — figures land in the git-tracked
``figures_paper/`` via the shared paper config (``save_fig(..., force_dir=True)``).
"""

import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import qhtools.utils.natconst as nc

# Pinned fiducial run (NOT load_data_to_plot — see plotting_paper/fiducial_data.py).
# This import must precede anything that reads BAQARO_*; it force-pins the env.
from baqaro.plotting_paper.fiducial_data import (
    path_file,
    load_simulation_metadata,
)

from baqaro.plotting_paper import plot_config  # noqa: F401  (rcParams)
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------
name_fig = "bh_specific_rate_vs_halo_rate"

# Radiative efficiency used in the model (matches main_evolution.py default).
rad_efficiency = 0.1

# x-axis range: log10(halo specific accretion rate) in Gyr^-1.
log_r_min, log_r_max = -3.5, 1.5
log_r = np.linspace(log_r_min, log_r_max, 400)


# ------------------------------------------------------------------------------
# CONSTANTS: 1/t_Edd in Gyr^-1
# ------------------------------------------------------------------------------
# L_Edd / (M_BH c^2) = 4 pi G m_p / (sigma_T c) — independent of M_BH.
L_Edd_per_M_cgs = 4.0 * np.pi * nc.gg * nc.mp * nc.cc / nc.st  # [erg s^-1 g^-1]
inv_t_Edd_Gyr = L_Edd_per_M_cgs / nc.cc**2 * nc.year * 1e9      # ~2.22 Gyr^-1


# ------------------------------------------------------------------------------
# LOAD ERDF PARAMETERS FROM THE EVOLUTION HDF5
# ------------------------------------------------------------------------------
with h5py.File(path_file, "r") as f:
    data = load_simulation_metadata(f)
    erdf = data["erdf"]
    erdf_params = data["erdf_params"]
    erdf_model_str = data["erdf_model_str"]

print("ERDF model :", erdf_model_str)
print("ERDF params:", dict(erdf_params))


# ------------------------------------------------------------------------------
# EVALUATE THE ERDF AND CONVERT TO ACCRETION-RATE UNITS
# ------------------------------------------------------------------------------
# mu(r), sigma(r) for the requested log halo rates.
mus, sigmas = erdf.get_mu_sigma_arrays(log_r, dtype=np.float64)
if np.ndim(sigmas) == 0:
    sigmas_arr = np.full_like(mus, float(sigmas))
else:
    sigmas_arr = np.asarray(sigmas)

# Constant offset (in dex) between log10(eta) and log10(M_dot_acc/M_BH [Gyr^-1]).
# Gross rate: M_dot_acc/M_BH = eta * (M_dot_Edd/M_BH) = eta / t_Sal0, with
# M_dot_Edd/M_BH = inv_t_Edd/eps0 = 1/t_Sal0. (No (1-eps) factor; see docstring.)
log_offset_sBHAR = np.log10(inv_t_Edd_Gyr / rad_efficiency)

# Median + 1/2-sigma bands of log10(M_dot_acc/M_BH) [Gyr^-1].
log_sBHAR_median = mus + log_offset_sBHAR
log_sBHAR_lo1 = log_sBHAR_median - sigmas_arr
log_sBHAR_hi1 = log_sBHAR_median + sigmas_arr
log_sBHAR_lo2 = log_sBHAR_median - 2.0 * sigmas_arr
log_sBHAR_hi2 = log_sBHAR_median + 2.0 * sigmas_arr


# ------------------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.3))

color_band = "#3a6fb1"
color_line = "#1f3b73"

ax.fill_between(log_r, log_sBHAR_lo2, log_sBHAR_hi2,
                color=color_band, alpha=0.12, lw=0, label=r"$\pm 2\sigma$")
ax.fill_between(log_r, log_sBHAR_lo1, log_sBHAR_hi1,
                color=color_band, alpha=0.30, lw=0, label=r"$\pm 1\sigma$")
ax.plot(log_r, log_sBHAR_median, color=color_line, lw=2.4, label="Median")

# Reference: 1:1 relation between halo and BH specific rates.
ax.plot(log_r, log_r, color="black", ls=":", lw=1.2,
        label=r"$\dot{M}_\mathrm{BH,acc}/M_\mathrm{BH} = "
              r"f_\mathrm{cold}\,\dot{M}_\mathrm{halo}/M_\mathrm{halo}$")

# Reference: Eddington accretion rate (eta = M_dot_acc/M_dot_Edd = 1), which on
# the left (gross specific-rate) axis sits at log10(M_dot_Edd/M_BH) =
# log10(inv_t_Edd_Gyr/eps0) = log10(1/t_Sal0) ~= 1.346.
log_y_mEdd1 = np.log10(inv_t_Edd_Gyr / rad_efficiency)
ax.axhline(log_y_mEdd1, color="darkred", ls="--", lw=1.2, zorder=-2)
ax.text(log_r_min + 0.25, log_y_mEdd1 + 0.08,
        r"$\eta_\mathrm{acc}=1$",
        ha="left", va="bottom", color="darkred", fontsize=13)

ax.set_xlim(log_r_min, log_r_max)
ax.set_ylim(log_sBHAR_lo2.min() - 0.3, max(log_sBHAR_hi2.max(), log_y_mEdd1) + 0.3)

ax.set_xlabel(
    r"log$_{10}\,f_\mathrm{cold}\,\dot{M}_\mathrm{halo}/M_\mathrm{halo}$ [Gyr$^{-1}$]",
    labelpad=-1,
)
ax.set_ylabel(
    r"log$_{10}\,\dot{M}_\mathrm{BH,acc}/M_\mathrm{BH}$ [Gyr$^{-1}$]",
    labelpad=3,
)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Twin axis: the gross Eddington-ratio eta = M_dot_acc/M_dot_Edd (the ERDF
# variable). Since the left axis is the gross specific rate M_dot_acc/M_BH, the
# two differ by the constant M_dot_Edd/M_BH = 1/t_Sal0 ≈ 22.18 Gyr^-1, i.e. a
# shift of log10(1/t_Sal0[Gyr]) ≈ 1.346 dex.
shift = np.log10(inv_t_Edd_Gyr / rad_efficiency)


def _to_edd(y):
    return y - shift


def _from_edd(y):
    return y + shift


# The specific-rate (primary) axis lives on the RIGHT spine; the eta_acc secondary
# axis lives on the LEFT. The primary axis would otherwise draw its OWN y ticks
# (rate units) on both spines, colliding with the eta_acc ticks (offset by the
# constant 1.346 dex) and producing a garbled double set of subticks — so restrict
# the primary ticks/label to the right spine only.
ax.tick_params(axis="y", which="both", left=False, right=True,
               labelleft=False, labelright=True)
ax.yaxis.set_label_position("right")

ax2 = ax.secondary_yaxis("left", functions=(_to_edd, _from_edd))
ax2.set_ylabel(r"$\log_{10}\,\eta_\mathrm{acc}=\log_{10}(\dot{M}_\mathrm{BH,acc}/\dot{M}_\mathrm{Edd})$", labelpad=-1)
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax.legend(loc="lower right", fontsize=10, handlelength=2.2, markerfirst=False)

fig.subplots_adjust(left=0.12, right=0.875, top=0.97, bottom=0.13)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
