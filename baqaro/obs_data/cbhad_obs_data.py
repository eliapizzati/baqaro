"""
Cosmic Black Hole Accretion-rate Density (BHAD / rho_dot_BH) observational data.

Companion to ``cbhmd_obs_data.py`` (cosmic BH *mass* density). Each entry is a
``CBHAD_data`` object holding the cosmic BH accretion-rate density as a function
of redshift, in **log10(rho_dot_BH / [Msun yr^-1 Mpc^-3])**.

Convention (matches the model's Soltan estimator and plotting_bhar.py):
    rho_dot_BH = (1 - eps)/eps * u_bol / c^2 ,   eps = 0.1
where u_bol(z) = integral L phi(L,z) dlogL is the bolometric AGN luminosity
density. Some literature reports the *gas* accretion-rate density
(Mdot_acc = rho_dot_BH/(1-eps)) or uses a different eps — see per-entry notes;
we always store the BH *mass-growth* rate (1-eps) Mdot_acc at eps=0.1 so the
curves are directly comparable to the model's rho_dot_BH.

PROJECT RULE: transcribe RAW values verbatim from the
source papers, or reconstruct from a published parametric form — never invent
numbers.
"""

import numpy as np

import qhtools.utils.natconst as nc
from baqaro.obs_data.qlf_shen_model import parameter_fit_B, L_sun


class CBHAD_data:
    """Cosmic BH accretion-rate density container (mirrors CBHMD_data)."""

    def __init__(self, z_data, y_data, errs, label, z_errs=None):
        self.z = np.asarray(z_data)
        self.data = np.asarray(y_data)  # log10(rho_dot_BH) [Msun yr^-1 Mpc^-3]
        self.errs = errs
        self.err_down = np.asarray(errs[0])
        self.err_up = np.asarray(errs[1])
        err_down_finite = np.where(np.isfinite(self.err_down), self.err_down, 0)
        err_up_finite = np.where(np.isfinite(self.err_up), self.err_up, 0)
        self.err = (err_down_finite + err_up_finite) / 2.
        # Optional asymmetric redshift (bin-width) error bars; None for the
        # smooth-curve / line-style entries that carry no z uncertainty.
        if z_errs is not None:
            self.z_err_down = np.asarray(z_errs[0])
            self.z_err_up = np.asarray(z_errs[1])
        else:
            self.z_err_down = None
            self.z_err_up = None
        self.label = label


data_cbhad = {}

# Eps and unit conversion: u_bol[erg/s/Mpc^3] -> rho_dot_BH[Msun/yr/Mpc^3].
_EPS = 0.1
# (1-eps)/eps * u/c^2 [g/s/Mpc^3] * year[s] / Msun[g] = Msun/yr/Mpc^3
_U_TO_BHAD = (1.0 - _EPS) / _EPS / nc.cc**2 * nc.year / nc.ms


# ==========================================================================
# Shen et al. 2020 (ApJS 249, 17; arXiv:2001.02696) — bolometric QLF global
# fit B. BHAD reconstructed by integrating the double-power-law bolometric LF
# (the project's preferred "reconstruct from the parametric form" approach;
# exact + reproducible, uses the canonical coefficients in qlf_shen_model.py).
# Verified: integrating this BHAD over cosmic time reproduces the cbhmd_obs_data
# Shen20 rho_BH(z=0) to ~0.02 dex. The wide [35,52] logL range is load-bearing
# at low z (z~0: L* ~ 1e44 erg/s, so a [42,49] window clips the faint end and
# underestimates BHAD by ~0.1 dex).
# fit B (NOT A): fit B's gamma1 decreases with z and stays < 1, so the faint-end
# luminosity-density integrand L*phi ~ L^(1-gamma1) always converges. Fit A's
# gamma1 has a quadratic upturn crossing 1 at z>~6, which makes the integral
# DIVERGE on the wide range (spurious high-z BHAD). The two agree to <0.04 dex
# where both are valid (z<6).
# Not a binned measurement -> stored as a smooth curve, no error band.
# ==========================================================================
def _shen20_bhad(z):
    """log10 rho_dot_BH [Msun/yr/Mpc^3] from the Shen+20 fit-B bolometric LF."""
    g1, g2, logLstar, logphi = parameter_fit_B(z)
    Lstar_erg = 10.0 ** logLstar * L_sun          # erg/s
    phi_star = 10.0 ** logphi                      # Mpc^-3 dex^-1
    logL = np.linspace(35.0, 52.0, 6000)           # log10(L_bol / erg s^-1)
    L = 10.0 ** logL
    ratio = L / Lstar_erg
    phi = phi_star / (ratio ** g1 + ratio ** g2)   # Mpc^-3 dex^-1
    u_bol = np.trapezoid(L * phi, logL)            # erg/s/Mpc^3
    return np.log10(u_bol * _U_TO_BHAD)


_z_shen = np.arange(0.0, 7.001, 0.2)
_bhad_shen = np.array([_shen20_bhad(z) for z in _z_shen])
_zeros_shen = np.zeros_like(_bhad_shen)

data_cbhad['Shen20'] = CBHAD_data(
    z_data=_z_shen, y_data=_bhad_shen,
    errs=[_zeros_shen, _zeros_shen],
    label="Shen+2020 (bol. QLF)",
)


# ==========================================================================
# Aird et al. 2015 (MNRAS 451, 1892; arXiv:1501.01982) — X-ray-derived cosmic
# BHAD. DIGITIZED from the paper's BHAD-vs-z figure; no error bars given.
# eps=0.1 as published.  [DIGITIZED — approximate]
# ==========================================================================
_aird15_raw = np.array([
    [0.08976660682226201, -5.450832381263261],
    [0.4488330341113105, -4.856407431586965],
    [0.7989228007181326, -4.538490832925303],
    [1.1669658886894074, -4.330436320113161],
    [1.5350089766606823, -4.247381807301018],
    [1.9120287253141826, -4.247667428322725],
    [2.217235188509874, -4.293353190794843],
    [2.6840215439856374, -4.418706816821718],
    [3.0520646319569114, -4.562925031282303],
    [3.4829443447037693, -4.741281758337414],
    [3.9766606822262114, -4.980292149502204],
    [5.053859964093356, -5.511411239867255],
])
_aird15_raw = _aird15_raw[np.argsort(_aird15_raw[:, 0])]
_z_aird = _aird15_raw[:, 0]
_bhad_aird = _aird15_raw[:, 1]
_zeros_aird = np.zeros_like(_bhad_aird)

data_cbhad['Aird15'] = CBHAD_data(
    z_data=_z_aird, y_data=_bhad_aird,
    errs=[_zeros_aird, _zeros_aird],
    label="Aird+2015 (X-ray)",
)


# ==========================================================================
# Ananna et al. 2019 (ApJ 871, 240; arXiv:1810.02298) — BHAD from the X-ray AGN
# population-synthesis model (obscured + Compton-thick corrected). DIGITIZED
# from the BHAD-vs-z figure; no error bars given. eps=0.1 as published.
# [DIGITIZED — approximate]
# ==========================================================================
_ananna19_raw = np.array([
    [0.09874326750448814, -5.155384636309233],
    [0.430879712746858, -4.6821514063435075],
    [0.5924596050269297, -4.504243512322507],
    [0.8707360861759426, -4.292333115717317],
    [1.2208258527827647, -4.091840759479898],
    [1.6157989228007177, -3.9216854360480933],
    [1.8940754039497303, -3.8650780697459335],
    [2.9174147217235182, -4.055247266198792],
    [3.922800718132854, -4.5938877101354665],
    [4.999999999999999, -5.109855285349002],
])
_ananna19_raw = _ananna19_raw[np.argsort(_ananna19_raw[:, 0])]
_z_an19 = _ananna19_raw[:, 0]
_bhad_an19 = _ananna19_raw[:, 1]
_zeros_an19 = np.zeros_like(_bhad_an19)

data_cbhad['Ananna19'] = CBHAD_data(
    z_data=_z_an19, y_data=_bhad_an19,
    errs=[_zeros_an19, _zeros_an19],
    label="Ananna+2019 (X-ray)",
)


# ==========================================================================
# Ueda et al. 2014 (ApJ 786, 104; arXiv:1402.1836) — BHAD from the hard-X-ray
# (2-10 keV) AGN luminosity function with absorption / Compton-thick correction.
# DIGITIZED from the BHAD-vs-z figure; no error bars given. eps=0.1 as published.
# [DIGITIZED — approximate]
# ==========================================================================
_ueda14_raw = np.array([
    [0.09874326750448814, -5.4508391817637785],
    [0.2872531418312385, -5.1744668407594805],
    [0.4488330341113105, -4.928377128556662],
    [0.8258527827648112, -4.553662749578369],
    [1.1759425493716338, -4.326655241825799],
    [1.5529622980251345, -4.194365105271748],
    [1.9120287253141826, -4.13781894347424],
    [3.007181328545781, -4.343194059082749],
    [5.1077199281867145, -5.435694467112779],
])
_ueda14_raw = _ueda14_raw[np.argsort(_ueda14_raw[:, 0])]
_z_ueda = _ueda14_raw[:, 0]
_bhad_ueda = _ueda14_raw[:, 1]
_zeros_ueda = np.zeros_like(_bhad_ueda)

data_cbhad['Ueda14'] = CBHAD_data(
    z_data=_z_ueda, y_data=_bhad_ueda,
    errs=[_zeros_ueda, _zeros_ueda],
    label="Ueda+2014 (X-ray)",
)


# ==========================================================================
# Yang, G. et al. 2023 (ApJL 950, L5; CEERS) — BHAD from JWST/MIRI MID-INFRARED
# (mid-IR-selected AGN, recovering heavily obscured / Compton-thick sources that
# X-ray surveys miss; a conservative lower limit from summed AGN disk
# luminosities). This is a JWST MID-IR measurement, NOT X-ray. DIGITIZED binned
# points with asymmetric z (bin width) and y (BHAD) error bars. eps=0.1 as
# published. [DIGITIZED — approximate]
# Central (z, log rho_dot_BH); errors stored as +/- offsets from the central.
# ==========================================================================
_yang23_z = np.array([0.43985637, 1.46319569, 2.52244165, 3.60861759, 4.27289048])
_yang23_y = np.array([-5.11397639, -4.09960013, -4.29358441, -4.14289212, -4.45021354])
# y lower / upper digitized -> down/up offsets
_yang23_y_lo = np.array([-5.48518851, -4.49353952, -4.72540259, -4.35501333, -4.63960748])
_yang23_y_hi = np.array([-4.91700669, -3.89505468, -4.10040259, -4.00652168, -4.32142566])
_yang23_yerr_down = _yang23_y - _yang23_y_lo
_yang23_yerr_up = _yang23_y_hi - _yang23_y
# z lower / upper edges (bin width) -> down/up offsets
_yang23_z_lo = np.array([0.0, 1.19389587, 2.19928187, 2.99820467, 3.99461400])
_yang23_z_hi = np.array([1.19389587, 2.19928187, 2.99820467, 4.00359066, 5.0])
_yang23_zerr_down = _yang23_z - _yang23_z_lo
_yang23_zerr_up = _yang23_z_hi - _yang23_z

data_cbhad['Yang23'] = CBHAD_data(
    z_data=_yang23_z, y_data=_yang23_y,
    errs=[_yang23_yerr_down, _yang23_yerr_up],
    z_errs=[_yang23_zerr_down, _yang23_zerr_up],
    label="Yang+2023 (JWST mid-IR)",
)


