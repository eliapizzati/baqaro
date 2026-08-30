"""Observed Eddington-ratio distribution functions (ERDFs) of quasars.

Absolute number density  dPhi/dlog(lambda_Edd)  [Mpc^-3 dex^-1]  — NOT a
normalised PDF. This is the form the observational papers publish, and it is the
form the model must be compared in, because the *normalisation* carries the
information about how many quasars there are at each Eddington ratio.

⚠ THE SELECTION TRAP. An ERDF is only meaningful together with the LUMINOSITY
RANGE of the sample it was measured from. A flux-limited survey preferentially
detects fast accretors at fixed M_BH, so an ERDF measured from a bright sample is
NOT the ERDF of a faint one. Every entry below therefore carries `log_L_bol_min`,
and the model MUST be cut at the same threshold before comparison. Comparing to
"all model black holes" is meaningless.
"""

import numpy as np


class ERDF_data:
    """Binned ERDF: dPhi/dlog(lambda_Edd) in Mpc^-3 dex^-1."""

    def __init__(self, x_data, y_data, err_low=None, err_up=None, label="",
                 redshift=None, log_L_bol_min=None, log_L_bol_max=None,
                 n_objects=None, reference="", n_per_bin=None):
        self.x = np.asarray(x_data)            # log10 lambda_Edd (bin centres)
        self.data = np.asarray(y_data)         # Phi [Mpc^-3 dex^-1]
        self.err_low = None if err_low is None else np.asarray(err_low)
        self.err_up = None if err_up is None else np.asarray(err_up)
        self.label = label
        self.redshift = redshift
        # The sample's luminosity limits — REQUIRED for a fair model cut.
        self.log_L_bol_min = log_L_bol_min
        self.log_L_bol_max = log_L_bol_max
        self.n_objects = n_objects
        self.reference = reference
        # Per-bin object counts. A bin with n == 0 is a genuine NON-DETECTION and
        # can be drawn as an upper limit; a bin that is simply ABSENT from the
        # paper's table is not — that is absence of information. Keeping the counts
        # is what lets the figure tell the two apart.
        self.n_per_bin = None if n_per_bin is None else np.asarray(n_per_bin)

    def upper_limit(self, i, n_sigma_counts=3.0):
        """95% Poisson upper limit (Gehrels 1986: 3.0 counts for n=0) on Phi in an
        EMPTY bin, converted using the mean 1/Vmax of the nearest populated bin."""
        if self.n_per_bin is None or self.n_per_bin[i] > 0:
            return None
        pop = np.flatnonzero(self.n_per_bin > 0)
        if pop.size == 0:
            return None
        j = pop[np.argmin(np.abs(pop - i))]
        mean_inv_vmax = self.data[j] / self.n_per_bin[j]
        return float(n_sigma_counts * mean_inv_vmax)

    @property
    def err(self):
        if self.err_low is None and self.err_up is None:
            return None
        lo = self.err_low if self.err_low is not None else self.err_up
        up = self.err_up if self.err_up is not None else self.err_low
        return np.vstack([lo, up])


data_erdf_global = {}


# ===========================================================================
#  He et al. 2024 (ApJ 962, 152; arXiv:2311.08922)
#  "Black hole mass and Eddington ratio distribution functions of less-luminous
#   quasars at z~4 in the Subaru Hyper Suprime-Cam wide field"
#
#  Sample: 52 HSC-SSP + 1462 SDSS DR7 broad-line quasars, z = 3.50 - 4.25.
#  The SAME paper supplies our z=4 BHMF (see abhmf_obs_data.py) — so the two
#  panels of the high-z figure are measured from one sample, which is why they
#  pair naturally.
#
#  Values below are taken VERBATIM from the authors' own data release,
#      https://github.com/wanqqq31/BHMF_hsc_z4  ->  BHMF_ERDF/binERDF.csv
#  (the `log Edd`, `Phi_tot`, `Phi_tot_err` columns; Phi in Mpc^-3 dex^-1,
#  0.3-dex bins). NOT digitised from a figure, NOT recalled — downloaded.
#
#  Cross-check: the same repo's binBHMF.csv reproduces our
#  independent Table-8 transcription in abhmf_obs_data.py to 3 significant
#  figures (1.21e-06 and 9.92e-07 at logM = 7.5, 7.8), so the source is the one
#  we already trust for the BHMF.
#
#  LUMINOSITY RANGE — read the caveat, the naive numbers are from the wrong sample.
#  He+2024 Section 3.3: "The resulting bolometric luminosity of the 80 QOP=4
#  quasars spans a range of 45.41 < log Lbol/erg s^-1 < 46.93." Those 80 objects
#  are their 3 < z < 4.5 spectroscopic table (tab7), NOT the ERDF sample, which is
#  the 52 HSC quasars at 3.50 <= z <= 4.25 (+ 1462 SDSS).
#    * log_L_bol_min = 45.41 IS correct for the ERDF sample — the faintest object
#      (J095931.01+021332.89, z=3.638) does fall inside the ERDF window. This is
#      the only value consumed (it sets the model's cut), so the figure is right.
#    * log_L_bol_max is set to None ON PURPOSE. 46.93 (an earlier value here) is
#      wrong twice over: that object sits at z=4.317, outside the ERDF window
#      (the in-window max is 46.80), and an upper limit is MEANINGLESS anyway for
#      an HSC+SDSS COMBINED ERDF whose SDSS arm runs far brighter.
#  (Checked against the authors' tab7/tab5 and the paper text.)
# ===========================================================================

_x_he24 = np.array([-2.00, -1.70, -1.40, -1.10, -0.80, -0.50,
                    -0.20, 0.10, 0.40, 0.70, 1.00])
_y_he24 = np.array([0.00e+00, 3.23e-10, 1.04e-07, 3.00e-07, 6.14e-07, 2.22e-06,
                    2.42e-06, 3.75e-07, 1.75e-07, 1.10e-09, 0.00e+00])
_e_he24 = np.array([0.00e+00, 1.45e-10, 1.03e-07, 1.56e-07, 2.71e-07, 7.13e-07,
                    9.67e-07, 1.86e-07, 1.22e-07, 2.66e-10, 0.00e+00])

# Per-bin counts (N_tot column of binERDF.csv). The two Phi==0 bins (log Edd =
# -2.00, +1.00) have N_tot = 0: they are GENUINE NON-DETECTIONS, not missing data,
# so they are KEPT and drawn as upper limits: with 1514 objects and a 1/Vmax
# correction, He's 95% limit at log lambda = +1 is ~1e-4 in normalised PDF
# units, the strongest single constraint in the figure.
_n_he24 = np.array([0, 5, 19, 115, 356, 469, 286, 174, 73, 17, 0])

data_erdf_global["4.0"] = ERDF_data(
    x_data=_x_he24,
    y_data=_y_he24,
    err_low=_e_he24,
    err_up=_e_he24,
    n_per_bin=_n_he24,
    label="He+2024 z=4.0 (HSC+SDSS)",
    redshift=4.0,
    log_L_bol_min=45.41,          # faintest ERDF-sample quasar (see caveat above)
    log_L_bol_max=None,           # meaningless for an HSC+SDSS combined ERDF
    n_objects=52 + 1462,
    reference="He et al. 2024, ApJ 962, 152 (arXiv:2311.08922); "
              "github.com/wanqqq31/BHMF_hsc_z4 BHMF_ERDF/binERDF.csv",
)


# ===========================================================================
#  Lai, S., Onken, C. A., Wolf, C., Bian, F., Fan, X. 2024
#  (MNRAS 531, 2245; arXiv:2405.10721)
#  "Supermassive black holes are growing slowly by z~5"  (XQz5+ sample)
#
#  ⚠ FIRST AUTHOR IS **LAI**, not Miller — verified against the arXiv PDF title
#  page. The paper is sometimes miscited; MNRAS 531, 2245 with this
#  title is Lai, Onken, Wolf, Bian & Fan.
#
#  This is the SAME PAPER that supplies our z=5 BHMF (abhmf_obs_data.py, their
#  Table 3). So — exactly as with He+2024 at z=4 and Wu+2022 at z=6 — the BHMF
#  and the ERDF at this redshift come from ONE sample. All three redshifts of the
#  high-z figure now have that property.
#
#  Values are their **Table 2, "Binned Eddington ratio distribution function for
#  XQz5+ at z~5, estimated by the 1/Vmax approach"**, read from the arXiv PDF.
#  Phi is quoted in units of 1e-9 Mpc^-3 dex^-1 -> converted to Mpc^-3 dex^-1
#  here so every entry in this module shares one unit.
#
#  VALIDATION: parsing their Table 1 (all 72 quasars) gives a sample
#  mean log(lambda) = -0.20 +/- 0.24, reproducing the abstract's quoted
#  "log lambda ~ -0.20 +/- 0.24" exactly; and the N_QSO column of Table 2 sums to
#  72, the full sample. Both checks passed.
#
#  ⚠ LUMINOSITY RANGE — THIS IS A **BRIGHT** SAMPLE. XQz5+ is "the most luminous
#  quasars between 4.5 < z < 5.3". Table 1 lists log L_3000, not L_bol; deriving
#  L_bol self-consistently from their own log M_BH and log(lambda) columns
#  (log L_bol = log lambda + log M_BH + log_csi) gives
#      log L_bol = 46.63 - 48.26,  median 47.56.
#  That is ~1.2 dex BRIGHTER at the faint end than He+2024's 45.41. A model cut
#  at He's 45.41 would include quasars fainter than ANY object in this sample and
#  would bias the model's lambda distribution LOW. THE MODEL MUST BE CUT PER
#  DATASET — see `log_L_bol_min` below, which the figure reads.
# ===========================================================================

_x_lai24 = np.array([-0.63, -0.45, -0.26, -0.08, 0.10, 0.29, 0.47])
_n_lai24 = np.array([5, 12, 24, 17, 8, 5, 1])              # sums to 72 = N_sample
_y_lai24 = np.array([0.70, 1.98, 2.44, 1.74, 1.48, 0.57, 0.25]) * 1e-9
_e_lai24 = np.array([0.34, 0.84, 0.58, 0.48, 0.78, 0.29, 0.25]) * 1e-9

# Lai's Table 2 has NO zero-count bins (N = 5,12,24,17,8,5,1). Their table simply
# STOPS at log lambda = -0.63 .. +0.47 — outside that range they publish nothing.
# That is ABSENCE OF INFORMATION, not a non-detection, so Lai gets NO upper limits.
data_erdf_global["5.0"] = ERDF_data(
    x_data=_x_lai24,
    y_data=_y_lai24,
    err_low=_e_lai24,
    err_up=_e_lai24,
    n_per_bin=_n_lai24,
    label="Lai+2024 z~5 (XQz5+)",
    redshift=5.0,
    log_L_bol_min=46.63,          # derived from their own M_BH + lambda columns
    log_L_bol_max=48.26,
    n_objects=72,
    reference="Lai S. et al. 2024, MNRAS 531, 2245 (arXiv:2405.10721), Table 2 "
              "(1/Vmax binned ERDF); L range derived from their Table 1",
)


# ---------------------------------------------------------------------------
#  He+2024 INTRINSIC (forward-modelled, selection-corrected) ERDF
#  From the same release: BHMF_ERDF/intrinsicERDF.csv — a 0.1-dex grid with
#  asymmetric errors, the authors' MLE fit rather than the 1/Vmax binned
#  estimate above. Kept as a smooth band for the figure; the BINNED points are
#  the primary comparison because they are the direct measurement.
#  (Loaded lazily from the CSV only if a caller asks for it — we deliberately do
#  NOT hardcode 31 rows of a fitted curve here.)
# ---------------------------------------------------------------------------


# ===========================================================================
#  INDIVIDUAL-OBJECT Eddington-ratio samples
#
#  These are NOT ERDFs — they are the raw lambda_Edd values of the quasars in a
#  survey, with no 1/Vmax or selection correction. They can only be compared to
#  the model as a *shape*, and ONLY after cutting the model at the same L_bol
#  range. Kept separate from the binned ERDFs above so the distinction cannot be
#  lost by accident.
# ===========================================================================


class ERDF_sample:
    """Individual quasar lambda_Edd measurements (no completeness correction)."""

    def __init__(self, log_lam, log_lam_err=None, log_L_bol=None, log_M_BH=None,
                 label="", redshift=None, z_min=None, z_max=None, reference=""):
        self.log_lam = np.asarray(log_lam)
        self.log_lam_err = None if log_lam_err is None else np.asarray(log_lam_err)
        self.log_L_bol = None if log_L_bol is None else np.asarray(log_L_bol)
        self.log_M_BH = None if log_M_BH is None else np.asarray(log_M_BH)
        self.label = label
        self.redshift = redshift
        self.z_min = z_min
        self.z_max = z_max
        self.reference = reference

    @property
    def n_objects(self):
        return int(self.log_lam.size)

    @property
    def log_L_bol_min(self):
        return None if self.log_L_bol is None else float(self.log_L_bol.min())

    @property
    def log_L_bol_max(self):
        return None if self.log_L_bol is None else float(self.log_L_bol.max())


data_erdf_samples = {}


# ---------------------------------------------------------------------------
#  Wu, J., Shen, Y., Jiang, L., et al. 2022 (MNRAS 517, 2659; arXiv:2210.02518)
#  "Demographics of z~6 quasars in the black hole mass-luminosity plane"
#
#  This is the SAME PAPER that supplies our z=6 BHMF (abhmf_obs_data.py, their
#  Table B.3) — so, exactly as with He+2024 at z=4, the BHMF and the Eddington
#  ratios come from ONE sample.
#
#  Values are the 34 rows of their **Table B.2 ("The BH mass sample")**, parsed
#  from the published PDF (arXiv:2210.02518), NOT recalled and NOT digitised from
#  a figure. Surveys: SDSS_M (20) + SDSS_O (9) + SDSS_S82 (5).
#  Independently re-verified element-by-element against the PDF:
#  all four arrays exact, 34 rows, no omissions or duplicates.
#
#  ⚠⚠ THE ANALYSIS SAMPLE IS **29**, NOT 34.
#  Wu+2022 use only the SDSS_M + SDSS_O objects ("SDSS_MO"). Their Section 2:
#      "We will use this SDSS_MO BH mass sample (29 objects) to jointly constrain
#       the 2D demographics"
#      "The SDSS_S82 sample ... is too small and highly incomplete in terms of BH
#       mass measurements."
#  Every ERDF/BHMF result in the paper uses the 29. THIS MATTERS HERE: the 5 S82
#  objects are array indices 29-33, and index 29 (J0005-0006, log M = 8.03) has
#  **log lambda = +0.61 — the single highest Eddington ratio in the whole array**.
#  It alone sets the high-lambda edge, which is precisely the feature the
#  madau+ vs constant-eps comparison turns on. Using all 34 would let a quasar
#  the authors deliberately discard drive our conclusion.
#  => `data_erdf_samples["6.0"]` is the **29-object SDSS_MO** sample.
#     `data_erdf_samples["6.0_all34"]` keeps the full table for reference.
#
#  ⚠ log_csi CONVENTION MISMATCH.  Wu+2022 Section 2 adopt
#  L_Edd = 1.3e38 erg/s per Msun (log_csi = 4.5310) while ours is nc.log_csi =
#  4.5138. That is a SYSTEMATIC +0.017 dex offset which any model comparison
#  against _WU22_LOGLAM inherits. It is small (well inside the bin width) but it
#  is a bias, not agreement.
#  He+2024 use 1.26e38 -> only 0.004 dex from ours, so their ERDF x-axis is fine.
#
#  ⚠ Wu+2022 explicitly warn that their 1/Vmax binned ERDF "is highly incomplete
#  at log lambda < -0.5". These raw object values inherit that: they are a
#  FLUX-LIMITED sample, biased AGAINST low lambda. Use for the shape of the
#  HIGH-lambda side, and do not read the faint tail as a measurement.
# ---------------------------------------------------------------------------

_WU22_LOGL = np.array([
    47.180, 48.210, 47.193, 47.621, 46.932, 47.196, 46.986, 47.370, 47.311,
    47.453, 47.282, 46.999, 47.533, 47.060, 46.988, 47.400, 47.200, 47.210,
    46.975, 47.464, 46.964, 46.969, 46.986, 47.195, 46.909, 46.739, 47.249,
    47.007, 46.760, 46.737, 47.311, 46.579, 46.975, 47.087])
_WU22_LOGM = np.array([
    9.36, 10.33, 9.29, 9.61, 9.37, 9.52, 9.73, 9.46, 9.81, 9.49, 9.76, 9.73,
    9.82, 9.84, 9.13, 9.41, 8.97, 9.42, 9.32, 9.66, 9.19, 9.91, 9.40, 9.58,
    9.53, 9.43, 9.31, 9.17, 9.27, 8.03, 10.05, 8.62, 9.32, 9.02])
_WU22_LOGLAM = np.array([
    -0.28, -0.22, -0.20, -0.09, -0.54, -0.42, -0.86, -0.19, -0.60, -0.14,
    -0.58, -0.83, -0.39, -0.88, -0.24, -0.10, +0.13, -0.31, -0.45, -0.30,
    -0.33, -1.04, -0.51, -0.50, -0.72, -0.79, -0.16, -0.28, -0.61, +0.61,
    -0.85, -0.14, -0.45, -0.03])
_WU22_LOGLAM_ERR = np.array([
    0.05, 0.09, 0.11, 0.08, 0.12, 0.06, 0.10, 0.05, 0.10, 0.14, 0.09, 0.08,
    0.09, 0.05, 0.06, 0.05, 0.13, 0.08, 0.15, 0.15, 0.07, 0.13, 0.19, 0.23,
    0.08, 0.10, 0.04, 0.37, 0.10, 0.06, 0.12, 0.03, 0.17, 0.12])

# Survey flag per row, in Table B.2 order: 20 SDSS_M, then 9 SDSS_O, then 5 S82.
_WU22_SURVEY = np.array(["SDSS_M"] * 20 + ["SDSS_O"] * 9 + ["SDSS_S82"] * 5)
_WU22_MO = _WU22_SURVEY != "SDSS_S82"          # the authors' 29-object sample

data_erdf_samples["6.0"] = ERDF_sample(
    log_lam=_WU22_LOGLAM[_WU22_MO],
    log_lam_err=_WU22_LOGLAM_ERR[_WU22_MO],
    log_L_bol=_WU22_LOGL[_WU22_MO],
    log_M_BH=_WU22_LOGM[_WU22_MO],
    label="Wu+2022 z~6 (SDSS_MO, 29)",
    redshift=6.0, z_min=5.73, z_max=6.42,   # S82 held the 5.71 minimum
    reference="Wu J. et al. 2022, MNRAS 517, 2659 (arXiv:2210.02518), Table B.2, "
              "SDSS_M + SDSS_O only — the 29-object sample the paper actually "
              "analyses (S82 excluded by the authors as 'too small and highly "
              "incomplete')",
)

# The full 34-row table, kept for reference / sensitivity checks ONLY. Do not use
# it as the default: it includes the 5 S82 objects the authors discard, one of
# which (log lambda = +0.61) is the highest Eddington ratio in the table and would
# single-handedly set the high-lambda edge of any comparison.
data_erdf_samples["6.0_all34"] = ERDF_sample(
    log_lam=_WU22_LOGLAM,
    log_lam_err=_WU22_LOGLAM_ERR,
    log_L_bol=_WU22_LOGL,
    log_M_BH=_WU22_LOGM,
    label="Wu+2022 z~6 (all 34, incl. S82)",
    redshift=6.0, z_min=5.71, z_max=6.42,
    reference="Wu J. et al. 2022, Table B.2, ALL rows (incl. the 5 SDSS_S82 the "
              "paper excludes) — reference only",
)


# ---------------------------------------------------------------------------
#  DELIBERATELY NOT INCLUDED
#
#  Willott et al. 2010 (AJ 140, 546; arXiv:1006.1342) report the z~6 Eddington
#  ratio distribution as a LOGNORMAL FIT (peak lambda = 1.07, sigma = 0.28 dex,
#  17 quasars). It is a MODEL FIT, not a measurement — plotting it next to the
#  model would compare our model to someone else's model. Excluded on purpose:
#  this module carries REAL DATA ONLY. If the fit is
#  ever wanted as context, add it to the FIGURE, clearly labelled as a fit, not
#  to this data module.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  SDSS DR16Q (Wu & Shen 2022, ApJS 263, 42) — the catalogue our cERDF
#  likelihood is ALREADY fit to (obs_data/qso_obs_data_setup.py).
#
#  Built on demand rather than hardcoded: it is 750k rows on disk, and using the
#  same loader guarantees the figure and the likelihood see identical numbers.
#  Coverage above log L_bol > 46:  z~4: 8,602   z~5: 496   z~6: 67 quasars.
#
#  ⚠ Same caveat as Wu+2022: flux-limited, NO completeness correction. Shape
#  only, and only against a model cut at the same L_bol.
#
#  NB "Wu & Shen 2022" (this DR16Q catalogue) is a DIFFERENT paper from
#  "Wu J. et al. 2022" (the z~6 demographics paper above). The abhmf module
#  already carries a warning about exactly this name collision.
# ---------------------------------------------------------------------------

def get_dr16q_sample(z_centre, dz=0.5, log_L_bol_min=46.0):
    """Individual DR16Q lambda_Edd values in a redshift slice above an L cut."""
    import qhtools.utils.natconst as nc
    from qhtools.utils import my_utils
    from baqaro.obs_data.qso_obs_data_setup import (
        redshifts_data, logL_Bols_data, logM_BHs_data)

    z = np.asarray(redshifts_data)
    lL = np.asarray(logL_Bols_data)
    lM = np.asarray(logM_BHs_data)
    sel = (np.abs(z - z_centre) < dz) & (lL > log_L_bol_min) & np.isfinite(lM)
    if sel.sum() == 0:
        return None
    loglam = my_utils.to_solar(lL[sel]) - lM[sel] - nc.log_csi
    return ERDF_sample(
        log_lam=loglam,
        log_L_bol=lL[sel],
        log_M_BH=lM[sel],
        label=f"DR16Q z~{z_centre:.0f}",
        redshift=z_centre,
        z_min=z_centre - dz, z_max=z_centre + dz,
        reference="Wu & Shen 2022, ApJS 263, 42 (SDSS DR16Q); "
                  "the catalogue the cERDF likelihood is fit to",
    )


__all__ = ["ERDF_data", "ERDF_sample", "data_erdf_global", "data_erdf_samples",
           "get_dr16q_sample"]
