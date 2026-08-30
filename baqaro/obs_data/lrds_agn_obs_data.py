"""Number densities of little red dots and faint high-z AGN (z ~ 4 - 7).

JWST-era measurements of an AGN population far fainter than the quasars the
QLF is built from, kept separate because their selection, bolometric
corrections and obscuration corrections are much less settled -- these are
comparison data, not likelihood inputs.

Sources: Greene+24, Harikane+23, Kokorev+24, Maiolino+23, Matthee+24,
Bulichi+26. Points are exposed both as raw number densities and, where a
luminosity axis exists, wrapped in :class:`~.qlf_obs_data.Qlf_data` so the
same plotting helpers work on them.
"""

import numpy as np

from baqaro.obs_data.qlf_obs_data import Qlf_data



# Greene+24

log_Lbols_greene24_z5 = np.asarray([44.0, 45.0, 46.0])
numdens_greene24_z5 = np.asarray([1.0e-5, 4.2e-5, 1.0e-5])
err_numdens_greene24_z5 = np.asarray([0.4e-5, 0.7e-5, 0.6e-5])


log_Lbols_greene24_z7 = np.asarray([45.0, 46.0])
numdens_greene24_z7 = np.asarray([2.6e-5, 1.3e-5])
err_numdens_greene24_z7 = np.asarray([0.5e-5, 0.5e-5])


redshifts_greene24 = [5.5, 7.5]
err_redshifts_greene24 = [1.0, 1.0]
integrated_numdens_greene24 = [6.2e-5, 3.9e-5]



# Kokorev+24



log_Lbols_kokorev24_z7 = np.asarray([44.0, 45.0, 46.0, 47.0])
log_Lbols_kokorev24_z5 = np.asarray([44.0, 45.0, 46.0, 47.0])


numdens_kokorev24_z7 = np.asarray([10**-5.48, 10**-4.51, 10**-4.76, 10**-5.47])
err_numdens_kokorev24_z7 = np.asarray([0.2*10**-5.48, 10**-4.51*0.23, 10**-4.76*0.15, 10**-5.47* 0.29]) * np.log(10)

numdens_kokorev24_z5 = np.asarray([10**-4.37, 10**-4.48, 10**-5.10, 10**-5.62])
err_numdens_kokorev24_z5 = np.asarray([10**-4.37*0.20, 10**-4.48*0.10, 10**-5.10*0.18, 0.2*10**-5.62]) * np.log(10)





# Matthee+24


log_Halpha_matthee24 = np.asarray([42.5, 42.9, 43.5])
log_Lbols_matthee24 = 44 + np.log10(10.33) + (1/1.157) * (log_Halpha_matthee24 - np.log10(5.25e42))

numdens_matthee24 = np.asarray([10**-4.20, 10**-4.74, 10**-5.36])
err_numdens_matthee24 = np.asarray([10**-4.20*0.14, 10**-4.74*0.30,  10**-5.36*0.53]) * np.log(10)







# Harikane+23

# Volume bin 1 harikane =  [1111.11111111 4761.9047619  2222.22222222    0.        ]
# Volume bin 2 harikane =  [ 37453.1835206       0.          95238.0952381  208333.33333333]
# Volume average harikane =  [ 6450.9416643      0.         14547.85934907     0.        ]
# Phi Lbol bin 1 harikane =  [0.0027  0.00042 0.00045     nan]
# Phi Lbol bin 2 harikane =  [2.67e-05      nan 1.05e-05 4.80e-06]
#

log_Lbols_harikane23 = np.asarray([44.5, 45.5])
numdens_harikane23 = np.asarray([4.5e-4, 1.05e-5])
err_numdens_harikane23 = np.asarray([0.7*4.5e-4, 0.7e-5])





# Bulichi+26 (mid-IR AGN)
# Bin centers from luminosity bin edges; phi in Mpc^-3 dex^-1 with
# asymmetric linear errors. Upper-limit bins are not included.

# 0.5 < z < 1.5
log_Lbols_bulichi26_z1 = np.asarray([44.2, 44.6, 45.0, 45.4])
numdens_bulichi26_z1 = np.asarray([1.87e-4, 1.02e-4, 5.81e-5, 1.44e-5])
err_down_numdens_bulichi26_z1 = np.asarray([0.43e-4, 0.29e-4, 2.14e-5, 0.96e-5])
err_up_numdens_bulichi26_z1   = np.asarray([0.51e-4, 0.36e-4, 2.91e-5, 1.55e-5])

# 1.5 < z < 2.5
log_Lbols_bulichi26_z2 = np.asarray([44.2, 44.6, 45.0, 45.4, 45.8, 46.2, 46.6])
numdens_bulichi26_z2 = np.asarray([1.39e-4, 1.04e-4, 5.23e-5, 4.77e-5, 2.13e-5, 1.83e-5, 1.01e-5])
err_down_numdens_bulichi26_z2 = np.asarray([0.32e-4, 0.27e-4, 1.77e-5, 1.53e-5, 1.04e-5, 0.92e-5, 0.59e-5])
err_up_numdens_bulichi26_z2   = np.asarray([0.39e-4, 0.30e-4, 2.22e-5, 2.06e-5, 1.52e-5, 1.37e-5, 0.97e-5])

# 2.5 < z < 3.5
log_Lbols_bulichi26_z3 = np.asarray([44.6, 45.0, 45.4, 45.8, 46.2])
numdens_bulichi26_z3 = np.asarray([1.45e-4, 1.25e-4, 8.44e-5, 4.38e-5, 2.21e-5])
err_down_numdens_bulichi26_z3 = np.asarray([0.34e-4, 0.29e-4, 2.42e-5, 1.52e-5, 1.00e-5])
err_up_numdens_bulichi26_z3   = np.asarray([0.37e-4, 0.32e-4, 2.72e-5, 1.97e-5, 1.30e-5])

# 3.5 < z < 4.5
log_Lbols_bulichi26_z4 = np.asarray([45.0, 45.4, 45.8, 46.2])
numdens_bulichi26_z4 = np.asarray([6.44e-5, 6.78e-5, 4.79e-5, 1.93e-5])
err_down_numdens_bulichi26_z4 = np.asarray([2.21e-5, 2.24e-5, 1.79e-5, 1.02e-5])
err_up_numdens_bulichi26_z4   = np.asarray([2.99e-5, 2.78e-5, 2.34e-5, 1.48e-5])

# 4.5 < z < 6
log_Lbols_bulichi26_z5 = np.asarray([44.55, 45.25, 45.95])
numdens_bulichi26_z5 = np.asarray([2.50e-5, 2.05e-5, 1.77e-5])
err_down_numdens_bulichi26_z5 = np.asarray([1.65e-5, 0.87e-5, 0.68e-5])
err_up_numdens_bulichi26_z5   = np.asarray([2.69e-5, 1.10e-5, 0.90e-5])





# ==============================================================================
# Qlf_data objects (same format as qlf_obs_data)
# x = log_Lbol, data = log10(phi [dlogL^-1 cMpc^-3]), errs in dex
# ==============================================================================

def _to_log_err(phi, err):
    """Convert symmetric linear error to log-space error (dex)."""
    return err / (phi * np.log(10))


def _make_qlf_data_sym(log_Lbol, phi, err_phi, label):
    """Build Qlf_data from linear phi with symmetric linear errors."""
    log_phi = np.log10(phi)
    err_dex = _to_log_err(phi, err_phi)
    return Qlf_data(log_Lbol, log_phi, [err_dex, err_dex], label)


def _make_qlf_data_asym(log_Lbol, phi, err_down, err_up, label):
    """Build Qlf_data from linear phi with asymmetric linear errors."""
    log_phi = np.log10(phi)
    err_down_dex = log_phi - np.log10(phi - err_down)
    err_up_dex = np.log10(phi + err_up) - log_phi
    return Qlf_data(log_Lbol, log_phi, [err_down_dex, err_up_dex], label)


# --- Greene+24 ---
qlf_greene24_z5 = _make_qlf_data_sym(
    log_Lbols_greene24_z5, numdens_greene24_z5, err_numdens_greene24_z5,
    r"Greene+24 $z\approx5.5$"
)
qlf_greene24_z7 = _make_qlf_data_sym(
    log_Lbols_greene24_z7, numdens_greene24_z7, err_numdens_greene24_z7,
    r"Greene+24 $z\approx7.5$"
)

# --- Kokorev+24 ---
qlf_kokorev24_z5 = _make_qlf_data_sym(
    log_Lbols_kokorev24_z5, numdens_kokorev24_z5, err_numdens_kokorev24_z5,
    r"Kokorev+24 $z\approx5$"
)
qlf_kokorev24_z7 = _make_qlf_data_sym(
    log_Lbols_kokorev24_z7, numdens_kokorev24_z7, err_numdens_kokorev24_z7,
    r"Kokorev+24 $z\approx7$"
)

# --- Matthee+24 ---
qlf_matthee24 = _make_qlf_data_sym(
    log_Lbols_matthee24, numdens_matthee24, err_numdens_matthee24,
    r"Matthee+24 $z\approx5$"
)

# (Maiolino+23 is not included — see the note at the top of this file.)

# --- Harikane+23 ---
qlf_harikane23 = _make_qlf_data_sym(
    log_Lbols_harikane23, numdens_harikane23, err_numdens_harikane23,
    r"Harikane+23 $z\approx5$"
)

# --- Bulichi+26 (mid-IR AGN) ---
qlf_bulichi26_z1 = _make_qlf_data_asym(
    log_Lbols_bulichi26_z1, numdens_bulichi26_z1,
    err_down_numdens_bulichi26_z1, err_up_numdens_bulichi26_z1,
    r"Bulichi+26 $0.5<z<1.5$"
)
qlf_bulichi26_z2 = _make_qlf_data_asym(
    log_Lbols_bulichi26_z2, numdens_bulichi26_z2,
    err_down_numdens_bulichi26_z2, err_up_numdens_bulichi26_z2,
    r"Bulichi+26 $1.5<z<2.5$"
)
qlf_bulichi26_z3 = _make_qlf_data_asym(
    log_Lbols_bulichi26_z3, numdens_bulichi26_z3,
    err_down_numdens_bulichi26_z3, err_up_numdens_bulichi26_z3,
    r"Bulichi+26 $2.5<z<3.5$"
)
qlf_bulichi26_z4 = _make_qlf_data_asym(
    log_Lbols_bulichi26_z4, numdens_bulichi26_z4,
    err_down_numdens_bulichi26_z4, err_up_numdens_bulichi26_z4,
    r"Bulichi+26 $3.5<z<4.5$"
)
qlf_bulichi26_z5 = _make_qlf_data_asym(
    log_Lbols_bulichi26_z5, numdens_bulichi26_z5,
    err_down_numdens_bulichi26_z5, err_up_numdens_bulichi26_z5,
    r"Bulichi+26 $4.5<z<6$"
)

# --- Convenience dict: all LRD QLFs grouped by approximate redshift ---
lrd_qlf_data = {
    '5.0': [qlf_greene24_z5, qlf_matthee24],
    '6.0': [qlf_greene24_z5, qlf_matthee24],
    '7.0': [qlf_greene24_z7],
}

# --- Bulichi+26 mid-IR AGN, grouped by redshift bin center ---
bulichi_qlf_data = {
    '1.0': qlf_bulichi26_z1,
    '2.0': qlf_bulichi26_z2,
    '3.0': qlf_bulichi26_z3,
    '4.0': qlf_bulichi26_z4,
    '5.0': qlf_bulichi26_z5,
}
