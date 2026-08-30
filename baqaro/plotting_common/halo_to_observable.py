"""Halo mass -> observable conversions for the local scaling-relation figures.

Turns a halo mass into the quantities the M_BH-sigma and M_BH-M_star figures are
plotted against:

  ``mstar_universemachine_b18``  M_halo -> M_star, UniverseMachine (Behroozi+18)
  ``sigma_rn18``                 M_halo -> stellar velocity dispersion sigma,
                                 following Ricarte & Natarajan 2018a: Moster+13
                                 SMHM for the stellar mass, Mosleh+13 sizes, an
                                 NFW halo (Dutton & Maccio 2014 concentration),
                                 and a Hernquist stellar profile inside R_e.

These two relations were extracted into a dependency-free module so that the
paper figures can import them without pulling in the exploratory analysis
they were first written for.

Consumers: the paper's ``plotting_local_relations`` figure, and the working
M_BH-sigma / M_BH-M_star figures.

These are LITERATURE relations, not fitted parts of this model -- treat the
coefficients as fixed and cite the papers above rather than re-tuning them.
"""

import numpy as np
from astropy import constants as const
from colossus.halo.concentration import concentration as col_conc

from qhtools.utils.cosmology import cosmo
from qhtools.utils.my_utils import log_mstar_behroozi_18


G_KPC_KMS2_PER_MSUN = const.G.to("kpc km2 / (Msun s2)").value


H_HUBBLE = cosmo.h


_MOSTER13 = dict(
    M10=11.590, M11=1.195,
    N10=0.0351, N11=-0.0247,
    beta10=1.376, beta11=-0.826,
    gamma10=0.608, gamma11=0.329,
)


def moster13_stellar_mass(log10_Mh, z):
    """Return log10(M_*/Msun) given log10(M_h/Msun) and redshift z."""
    Mh = 10.0 ** np.asarray(log10_Mh)
    zz = np.asarray(z)
    a = zz / (1.0 + zz)

    log10_M1 = _MOSTER13["M10"] + _MOSTER13["M11"] * a
    N = _MOSTER13["N10"] + _MOSTER13["N11"] * a
    beta = _MOSTER13["beta10"] + _MOSTER13["beta11"] * a
    gamma = _MOSTER13["gamma10"] + _MOSTER13["gamma11"] * a

    M1 = 10.0 ** log10_M1
    ratio = Mh / M1
    Mstar = Mh * 2.0 * N / (ratio ** (-beta) + ratio ** gamma)
    return np.log10(Mstar)


def mosleh13_Re_z0(log10_Mstar):
    """Effective radius R_e [kpc] for red galaxies at z=0 (RN18 Eq. 7)."""
    Mstar = 10.0 ** np.asarray(log10_Mstar)
    return 10.0 ** (-0.314) * Mstar ** 0.042 * (1.0 + Mstar / 10.0 ** 10.537) ** 0.76


def Re_of_Mstar_z(log10_Mstar, z):
    """RN18 Eqs. 7-9: R_e(M_*, z) = R_e(M_*, 0) (1+z)^gamma(M_*),
    with gamma = max(0, (log10 M_* - 10.75) / 0.85)."""
    gamma = np.maximum(0.0, (np.asarray(log10_Mstar) - 10.75) / 0.85)
    return mosleh13_Re_z0(log10_Mstar) * (1.0 + np.asarray(z)) ** gamma


def R200c_kpc(M200c_Msun, z):
    """Physical R_200c [kpc] for M_200c in Msun, using rho_c(z) from
    qhtools/colossus (physical Msun h^2 / kpc^3 -> Msun / kpc^3 by * h^2)."""
    rho_c_phys = cosmo.rho_c(z) * H_HUBBLE ** 2
    return (3.0 * M200c_Msun / (4.0 * np.pi * 200.0 * rho_c_phys)) ** (1.0 / 3.0)


def c200c_dutton14(M200c_Msun, z):
    """Dutton & Maccio 2014 c_200c (the c-M relation RN18 use).

    colossus.concentration() accepts an array of masses for a scalar
    redshift; arrays of z are looped (unique-z micro-batching)."""
    M = np.atleast_1d(np.asarray(M200c_Msun, dtype=float))
    M_over_h = M * H_HUBBLE
    z_arr = np.broadcast_to(z, M.shape)

    if z_arr.ndim == 0 or np.unique(z_arr).size == 1:
        out = col_conc(M_over_h.ravel(), "200c", float(np.unique(z_arr)[0]),
                       model="dutton14")
        return np.asarray(out).reshape(M.shape) if M.shape else float(out)

    out = np.empty_like(M)
    for zi in np.unique(z_arr):
        m = z_arr == zi
        out[m] = col_conc(M_over_h[m], "200c", float(zi), model="dutton14")
    return out


def _nfw_mu(x):
    """NFW enclosed-mass kernel: M(<r)/[4 pi rho_s r_s^3] = ln(1+x) - x/(1+x)."""
    return np.log1p(x) - x / (1.0 + x)


_HERNQUIST_RE_OVER_A = 1.8153


def f_star_within_re(M200c_Msun, log10_Mstar, Re_kpc, z):
    """f_*(R_e) = M_*(<R_e) / M_DM(<R_e).

    Stars: Hernquist profile, M_*(<R_e) = M_* (R_e / (R_e + a))^2,
    a = R_e / 1.8153.
    DM: NFW with c_200c from Dutton-Maccio 2014. (RN18 use Dehnen-
    McLaughlin; we substitute NFW. Differences are ~10-20% at R_e.)
    """
    Mstar = 10.0 ** np.asarray(log10_Mstar)

    a = Re_kpc / _HERNQUIST_RE_OVER_A
    Mstar_in = Mstar * (Re_kpc / (Re_kpc + a)) ** 2

    R200 = R200c_kpc(M200c_Msun, z)
    c200 = c200c_dutton14(M200c_Msun, z)
    rs = R200 / c200
    x_re = Re_kpc / rs
    MDM_in = M200c_Msun * _nfw_mu(x_re) / _nfw_mu(c200)

    return Mstar_in / MDM_in


def sigma_rn18(log10_Mh, z):
    """RN18 Eq. 10: sigma [km/s] as a function of M_h and z."""
    log10_Mh = np.asarray(log10_Mh, dtype=float)
    z = np.asarray(z, dtype=float)
    Mh = 10.0 ** log10_Mh

    log10_Mstar = moster13_stellar_mass(log10_Mh, z)
    Re = Re_of_Mstar_z(log10_Mstar, z)
    Mstar = 10.0 ** log10_Mstar

    f_star = f_star_within_re(Mh, log10_Mstar, Re, z)

    one_plus_Fej = 1.0 / 0.58
    inside = (G_KPC_KMS2_PER_MSUN * Mstar / Re) * (one_plus_Fej + 0.86 / f_star)
    return 0.389 * np.sqrt(inside)


def mstar_universemachine_b18(log10_Mh, z):
    """Return log10(M_*/Msun) from UniverseMachine (Behroozi+18) SMHM.

    Thin wrapper around qhtools.utils.my_utils.log_mstar_behroozi_18.
    The underlying numba-jit function vectorises over its first arg for a
    scalar z; for arrays of z we batch by unique-z."""
    log10_Mh = np.asarray(log10_Mh, dtype=float)
    z_arr = np.broadcast_to(z, log10_Mh.shape)

    if z_arr.ndim == 0 or np.unique(z_arr).size == 1:
        return np.asarray(log_mstar_behroozi_18(log10_Mh, float(np.unique(z_arr)[0])))

    out = np.empty_like(log10_Mh)
    for zi in np.unique(z_arr):
        m = z_arr == zi
        out[m] = log_mstar_behroozi_18(log10_Mh[m], float(zi))
    return out
