"""
Observed quasar linear bias b_Q(z) — literature compilation.
=====================================================================================

A table of measured large-scale linear quasar bias vs redshift, for overlaying on
the model b_Q(z) figure (``plotting_paper/plotting_quasar_bias.py``).

Provenance and reliability
--------------------------
The table is compiled BY HAND from the published papers, not machine-read, and
the entries differ in how directly they measure a bias. Entries flagged
``approx=True`` are the least certain.

* DESI DR2 (Charles+26) is the precise anchor over 2 < z < 3.5
  (b(2.48) = 3.61 +/- 0.01, z-bins from its fit and text). Not approximate.
* White+12 (3.8 +/- 0.3), Laurent+17 (2.45 +/- 0.05), Eftekharzadeh+15
  (3.54 +/- 0.10), Croom+05 (b(0.53) = 1.13 +/- 0.18, b(2.48) = 4.24 +/- 0.53),
  He+18 low-L (5.93 +/- 1.4), Ross+09 (b(1.27) = 2.06 +/- 0.03) and
  da Angela+08 (2QZ+2SLAQ, 1.5 +/- 0.2 at z = 1.4) were each checked against
  their papers.
* Shen+07 is high-z with large errors, and its b is inferred from the quoted
  r0 rather than measured directly (approximate).
* ⚠ EIGER (Eilers+24) and ASPIRE (Huang+26) report a HOST HALO MASS, not a
  bias. The value here is b_h(M200m, z) from Tinker10 -- the same relation the
  model itself uses -- evaluated at their log M_halo,min. They are therefore
  host-mass-IMPLIED biases, NOT direct clustering-bias measurements, and any
  figure caption using them must say so.

Cosmology: most of these assume sigma8 ~ 0.8, close enough to FLAMINGO's 0.808
that no rescaling is applied (DESI uses Planck sigma8 = 0.81).

Each entry carries an approximate effective bolometric luminosity ``logLbol``
[log10 erg/s] so the plotter can colour obs points by the model's L_bol bins
(converted from M_i / M1450 where the survey quotes magnitudes — again approximate).
"""

from dataclasses import dataclass


@dataclass
class BiasPoint:
    """One published quasar linear-bias measurement.

    ``approx=True`` marks entries that are not direct clustering-bias
    measurements -- inferred from a quoted r0, or converted from a host halo
    mass -- and which therefore should not be read as equivalent to the rest.
    ``logLbol`` is an approximate effective bolometric luminosity so the
    plotter can colour points by the model's luminosity bins.
    """
    z: float
    b: float
    err: float                       # symmetric 1σ
    ref: str
    logLbol: float = float("nan")    # approx effective log10 L_bol [erg/s]
    approx: bool = False
    M1450: float = float("nan")      # mean M1450; if set, the plotter derives
                                     # logLbol from it via qhtools (Runnoe+12)
    uplim: bool = False              # upper limit (plot with a down-arrow)


# Low / intermediate z (b ~ 1-4): 2QZ / BOSS / eBOSS. All verified against the papers.
DATA_LOWZ = [
    # Croom+05 (2QZ) b(z)=0.53+0.289(1+z)^2; two published binned anchors.
    BiasPoint(0.53, 1.13, 0.18, "Croom+05", 45.8, approx=False),
    BiasPoint(2.48, 4.24, 0.53, "Croom+05", 46.2, approx=False),
    # Ross+09 (SDSS DR5), z̄=1.27 (full-sample fit b evolves 1.4@z0.5 -> 3@z2.2).
    BiasPoint(1.27, 2.06, 0.03, "Ross+09", 45.9, approx=False),
    # da Angela+08 (2QZ+2SLAQ), z=1.4.
    BiasPoint(1.40, 1.50, 0.20, "da Angela+08", 46.0, approx=False),
    # Laurent+17 (eBOSS DR14), z̄=1.55.
    BiasPoint(1.55, 2.45, 0.05, "Laurent+17", 46.0, approx=False),
    # White+12 (BOSS DR9), z~2.4.
    BiasPoint(2.40, 3.80, 0.30, "White+12", 46.4, approx=False),
    # Eftekharzadeh+15 (final BOSS), full 2.2<z<2.8 headline value.
    BiasPoint(2.50, 3.54, 0.10, "Eftekharzadeh+15", 46.4, approx=False),
]

# DESI DR2 (Charles+2026, arXiv:2606.27115) — Table 4, 'equal numbers' scheme.
# The actual 12 redshift-luminosity binned measurements (z̄, mean M1450, b_Q±err),
# ~713,700 quasars total over 2.0<z<3.5. logLbol derived per-point from M1450 via
# qhtools (Runnoe+12) in the plotter. Stat errors shown (±0.04-0.29; systematic
# ~1.4-5.9× larger). This is the precise anchor + the L-dependence measurement.
_DESI = "DESI DR2 (Charles+26)"
DATA_DESI = [
    BiasPoint(2.133, 3.06, 0.04, _DESI, approx=False, M1450=-22.17),
    BiasPoint(2.146, 3.12, 0.04, _DESI, approx=False, M1450=-23.16),
    BiasPoint(2.151, 3.31, 0.05, _DESI, approx=False, M1450=-24.53),
    BiasPoint(2.462, 3.45, 0.06, _DESI, approx=False, M1450=-22.55),
    BiasPoint(2.482, 3.66, 0.07, _DESI, approx=False, M1450=-23.53),
    BiasPoint(2.493, 3.73, 0.06, _DESI, approx=False, M1450=-24.90),
    BiasPoint(2.863, 4.12, 0.11, _DESI, approx=False, M1450=-23.01),
    BiasPoint(2.881, 4.29, 0.12, _DESI, approx=False, M1450=-23.99),
    BiasPoint(2.888, 4.50, 0.12, _DESI, approx=False, M1450=-25.30),
    BiasPoint(3.248, 5.12, 0.23, _DESI, approx=False, M1450=-23.43),
    BiasPoint(3.267, 4.69, 0.29, _DESI, approx=False, M1450=-24.41),
    BiasPoint(3.277, 5.31, 0.25, _DESI, approx=False, M1450=-25.65),
]

# DESI DR2 'same boundaries' scheme (Table 4) — same M1450 bin EDGES at every z
# (so the bright bin is a fixed cut). Poorer luminosity-model fit (chi2_red=4.16)
# than 'equal numbers', kept only as a binning-scheme COMPARISON. Same (z, mean
# M1450, b_Q +/- err).
DATA_DESI_SB = [
    BiasPoint(2.138, 3.06, 0.03, _DESI, approx=False, M1450=-22.55),
    BiasPoint(2.149, 3.28, 0.05, _DESI, approx=False, M1450=-24.13),
    BiasPoint(2.155, 3.08, 0.27, _DESI, approx=False, M1450=-25.72),
    BiasPoint(2.467, 3.48, 0.04, _DESI, approx=False, M1450=-22.75),
    BiasPoint(2.487, 3.68, 0.05, _DESI, approx=False, M1450=-24.16),
    BiasPoint(2.500, 4.01, 0.19, _DESI, approx=False, M1450=-25.78),
    BiasPoint(2.862, 4.14, 0.11, _DESI, approx=False, M1450=-22.95),
    BiasPoint(2.882, 4.19, 0.07, _DESI, approx=False, M1450=-24.21),
    BiasPoint(2.890, 5.43, 0.19, _DESI, approx=False, M1450=-25.84),
    BiasPoint(3.239, 5.40, 0.44, _DESI, approx=False, M1450=-23.11),
    BiasPoint(3.264, 4.95, 0.15, _DESI, approx=False, M1450=-24.29),
    BiasPoint(3.279, 6.03, 0.31, _DESI, approx=False, M1450=-25.89),
]

# High z (z>=3): Shen+07, He+18, and the z~6 JWST samples.
# NOTE: EIGER (Eilers+24) and ASPIRE (Huang+26) do NOT quote a linear bias — they
# report host halo masses. The b below is b_h(M200m) via Tinker10 (SAME relation
# as the model) from their reported log M_halo,min: EIGER log M=12.43 @ z=6.25 ->
# b≈15.3; ASPIRE log M=12.13 @ z=6.6 -> b≈13.6. So these are HOST-MASS-IMPLIED
# biases (err from the M_halo uncertainty), not direct clustering-bias measurements.
DATA_HIGHZ = [
    # Shen+07 (SDSS DR5; AJ 133,2222). Reports b_eff DIRECTLY (its Table 6 / abstract):
    # central values b_eff=7.9 @ z~3.1 and 14.2 @ z=4.0 are the PAPER'S OWN (they are
    # number-density-derived: Martini-Weinberg Phi(z)+HMF -> M_min -> Jing98 bias, NOT
    # fitted from the clustering amplitude -- hence approx=True). The paper quotes NO
    # error on b_eff; the errors here are r0-UNCERTAINTY PROPAGATION (gamma=2 => b ∝ r0,
    # so db/b = dr0/r0): r0=16.9±1.7 (10.1%) -> ±0.80; r0=24.3±2.4 (9.9%) -> ±1.40.
    # ±0.80 matches the Timlin+18 compilation (7.90±0.80). (Prev ±2.5 on the high bin
    # was inconsistent -- it was half the Table-5 duty-cycle spread, not r0-prop.)
    BiasPoint(3.20, 7.9, 0.8, "Shen+07", 46.8, approx=True),
    BiasPoint(4.00, 14.2, 1.4, "Shen+07", 47.0, approx=True),
    # He+18 (HSC z~4) LOW-luminosity cross-corr bias (verified). Their high-L point
    # b=2.73(+2.44/-2.55) is barely constrained -> omitted.
    BiasPoint(3.80, 5.93, 1.4, "He+18", 45.9, approx=False),
    # EIGER (Eilers+24) & ASPIRE (Huang+26): HOST-MASS-IMPLIED bias b_h(M200m) via
    # Tinker10 from their logMh,min (12.43@z6.25, 12.13@z6.6) -- NOT a direct bias.
    BiasPoint(6.25, 15.3, 1.7, "EIGER (Eilers+24)", 47.0, approx=True),
    BiasPoint(6.60, 13.6, 3.4, "ASPIRE (Huang+26)", 46.7, approx=True),
]

# Additional compilation datasets — ALL VERIFIED against the paper tables
# (P&N06 Table 3; Myers+07 Paper I Table 1; Ikeda+15). NOTE both
# Porciani&Norberg AND Myers assume sigma8=0.9 (vs FLAMINGO 0.808) — b would
# scale up ~11% under a uniform sigma8 rescaling (NOT applied, consistent across
# both 2QZ-era datasets).
# Shen+09 dropped: it is a luminosity/mass/colour/radio study at 0.4<z<2.5, NOT a
# clean b(z) evolution table (that evolution is Ross+09, already included).
DATA_EXTRA = [
    # Porciani & Norberg 2006 (2QZ) — Table 3 constant-bias b, 6 z-bins (sigma8=0.9).
    BiasPoint(0.933, 1.57, 0.29, "Porciani&Norberg+06", 46.0, approx=False),
    BiasPoint(1.185, 1.76, 0.39, "Porciani&Norberg+06", 46.0, approx=False),
    BiasPoint(1.410, 2.13, 0.31, "Porciani&Norberg+06", 46.1, approx=False),
    BiasPoint(1.602, 2.33, 0.42, "Porciani&Norberg+06", 46.1, approx=False),
    BiasPoint(1.796, 3.02, 0.49, "Porciani&Norberg+06", 46.2, approx=False),
    BiasPoint(1.987, 4.13, 0.52, "Porciani&Norberg+06", 46.2, approx=False),
    # Myers+2007 (Paper I, ApJ 658,85; astro-ph/0612190) — Table 1, the 5
    # redshift sub-bins, 1-parameter model, Gamma=0.21 sigma8=0.9 (same sigma8 as
    # P&N06). Full-sample average b=2.41+/-0.09 (= abstract). z=0.75 & z=2.28 are
    # the photo-z catastrophic-failure end bins (footnotes b/c: values shifted
    # >=10%), least reliable. VERIFIED from the paper table.
    BiasPoint(0.75, 1.93, 0.14, "Myers+06", 45.8, approx=False),
    BiasPoint(1.20, 2.05, 0.11, "Myers+06", 46.0, approx=False),
    BiasPoint(1.53, 2.23, 0.13, "Myers+06", 46.0, approx=False),
    BiasPoint(1.87, 2.81, 0.13, "Myers+06", 46.1, approx=False),
    BiasPoint(2.28, 2.84, 0.40, "Myers+06", 46.2, approx=False),
    # Ikeda+2015 (COSMOS z~4 low-L quasar-LBG cross) — 86% UPPER LIMITS
    # (b<5.63 total, b<10.50 spectroscopic; -24<M1450<-22, <z>~3.9).
    BiasPoint(3.90, 5.63, 0.0, "Ikeda+15", 45.7, approx=False, uplim=True),
    BiasPoint(3.90, 10.50, 0.0, "Ikeda+15", 46.0, approx=False, uplim=True),
]

DATA_ALL = DATA_LOWZ + DATA_DESI + DATA_HIGHZ + DATA_EXTRA


def arrays(subset=None):
    """Return (z, b, err, refs, logLbol) numpy arrays for a subset (default all)."""
    import numpy as np
    d = subset if subset is not None else DATA_ALL
    return (np.array([p.z for p in d]),
            np.array([p.b for p in d]),
            np.array([p.err for p in d]),
            np.array([p.ref for p in d], dtype=object),
            np.array([p.logLbol for p in d]))
