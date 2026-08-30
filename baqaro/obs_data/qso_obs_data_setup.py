"""SDSS DR16Q quasar sample — the observational input to the cERDF likelihood.

Loads the Wu & Shen (2022) DR16Q quasar property catalogue and exposes the
three columns the conditional Eddington-ratio distribution is built from:
``redshifts_data``, ``logL_Bols_data``, ``logM_BHs_data`` (~727k objects after
cuts). :func:`bin_qso_eddington_ratios` reduces them to the binned cERDF the
likelihood compares against.

REQUIRES AN EXTERNAL FILE
-------------------------
``<data root>/input_data/dr16q_prop_May01_2024.fits`` (~3.1 GB). It is far too
large to version, so it is NOT in this repository -- obtain the DR16Q quasar
property catalogue from the SDSS data release and place it there. The data root
is the active site's (see :mod:`baqaro.utils.my_dir`) and can be
pointed anywhere with ``BAQARO_DATA_DIR``.

Importing this module reads that file, so every consumer of the cERDF
likelihood depends on it being present.
"""

# read the file dr16q_prop_May01_2024.fits and extract the columns
# 'logM_BH', 'logL_Bol', 'logL_Edd', 'logM_BH_err', 'logL_Bol_err', 'logL_Edd_err'

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import os # Import the os module
from baqaro.utils.my_dir import get_data_path

# Resolve the catalogue against the active site (BAQARO_SOURCE_DIR), so this
# is not pinned to one machine. BAQARO_DATA_DIR overrides the site default.
input_data_dir = os.path.join(
    get_data_path(os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")), 'input_data')

# Small tables committed with the package.
_BUNDLED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
filename = os.path.join(input_data_dir, 'dr16q_prop_May01_2024.fits')

# Define the columns you want to extract
desired_columns = ['LOGLBOL', 'LOGLBOL_ERR', 'LOGMBH', 'LOGMBH_ERR', 'Z_SYS'] # Add error columns if needed, e.g., 'logM_BH_err'

with fits.open(filename) as hdul:
    # The catalogue's table lives in HDU 1.
    full_data = Table(hdul[1].data)
    df = full_data[desired_columns].to_pandas()

# Extract the relevant columns from the DataFrame
logM_BHs = df['LOGMBH']
logL_Bols = df['LOGLBOL']
redshifts = df['Z_SYS']

mask_zeros = (logM_BHs != 0) & (logL_Bols != 0)
mask_z_space = (redshifts < 5.5) & (redshifts > 0.5)
mask_MBH = (logM_BHs > 6.5) & (logM_BHs < 10.5)
logM_BHs = logM_BHs[mask_zeros & mask_z_space & mask_MBH]
logL_Bols = logL_Bols[mask_zeros & mask_z_space & mask_MBH]
redshifts = redshifts[mask_zeros & mask_z_space & mask_MBH]


# High-z (z ~ 6) quasar compilation: the database of all z > 5.9 quasars from
# Fan, Banados & Simcoe (2023, ARA&A 61, 373), with the MgII-based BH masses
# and L3000 tabulated there. Unlike the DR16Q catalogue above it is small
# enough to bundle, so it is read from the package rather than the external
# data root.
df = pd.read_csv(os.path.join(_BUNDLED_DATA_DIR, 'ARAA_qso_database.csv'))

redshifts_qso = df['redshift']
bh_masses_qso = df['BHmass']*1e8
Lbol_qso =  5.15*df['L3000']*1e46


logM_BHs_z6 = np.log10(bh_masses_qso)
logL_Bols_z6 = np.log10(Lbol_qso)

logM_BHs_data = np.concatenate((logM_BHs, logM_BHs_z6))
logL_Bols_data = np.concatenate((logL_Bols, logL_Bols_z6))
redshifts_data = np.concatenate((redshifts, redshifts_qso))


mask_valid = (logM_BHs_data > 6.5) & (logM_BHs_data < 11.) & (logL_Bols_data > 43.5) & (logL_Bols_data < 48.5) & (redshifts_data > 0.5) & (redshifts_data < 7.5)

logM_BHs_data = logM_BHs_data[mask_valid]
logL_Bols_data = logL_Bols_data[mask_valid]
redshifts_data = redshifts_data[mask_valid]


# ==============================================================================
# BINNED EDDINGTON RATIO HISTOGRAMS
# ==============================================================================

def bin_qso_eddington_ratios(redshifts, logL_Bols, log_etas,
                             z_centers, logL_bins=None,
                             log_eta_bins=None, min_count=5,
                             z_halfwidth=0.5, sys_frac=0.15,
                             sys_abs=0.20):
    """
    Bin observed QSOs into (redshift x luminosity) cells and build normalised
    log_eta histograms in each cell.

    This is the data preparation step for the binned CERDF likelihood.  The
    output is a list of cell dicts containing observed histograms with Poisson
    errors, ready for chi-squared comparison against model PDFs.

    Parameters
    ----------
    redshifts : array, shape (N,)
        Observed QSO redshifts (already filtered for valid range).
    logL_Bols : array, shape (N,)
        Observed log bolometric luminosities (erg/s).
    log_etas : array, shape (N,)
        Observed log Eddington ratios.
    z_centers : array-like
        Redshift bin centres (typically the emulator's redshift grid).
    logL_bins : array or None
        Luminosity bin edges (log10 erg/s).
        Default: [45.5, 46.0, 46.5, 47.0, 47.5].
    log_eta_bins : array or None
        Eddington ratio bin edges for the histogram.
        Default: 18 bins from -3.0 to 1.5 (width 0.25 dex).
    min_count : int
        Minimum QSO count in a (z, L) cell to include it.  Cells with
        fewer QSOs are skipped (too noisy for chi2).  Default 5.
    z_halfwidth : float
        Half-width of the redshift window around each z_center.  Default 0.5.
    sys_frac : float
        Fractional systematic error floor added in quadrature to Poisson
        errors.  The floor is ``sys_frac * obs_pdf``.  Default 0.15 (15%).
    sys_abs : float
        Absolute systematic error floor (in PDF density units) added in
        quadrature.  Prevents tail bins (where obs_pdf ~ 0 but the model
        may predict non-zero density) from dominating the chi2.  Represents
        irreducible model uncertainty from emulator interpolation, BH mass
        scatter calibration, and finite-volume effects.  Default **0.20**.

    Returns
    -------
    cells : list of dict
        Each dict contains:
          'i_z'          : int, index into z_centers
          'z_center'     : float, redshift bin centre
          'lmin_val'     : float, lower luminosity bin edge
          'lmax_val'     : float, upper luminosity bin edge
          'counts'       : int array, histogram counts per eta bin
          'N_total'      : int, total QSOs in this cell
          'obs_pdf'      : float array, normalised histogram (PDF)
          'sigma_pdf'    : float array, Poisson error on the PDF
          'valid_mask'   : bool array, True where counts > 0
    log_eta_bins : array
        Histogram bin edges used.
    log_eta_centers : array
        Histogram bin centres.
    d_eta : float
        Histogram bin width.
    """
    if logL_bins is None:
        logL_bins = np.linspace(45.5, 47.5, 5)
    if log_eta_bins is None:
        log_eta_bins = np.linspace(-3.0, 1.5, 19)

    logL_bins = np.asarray(logL_bins)
    log_eta_bins = np.asarray(log_eta_bins)
    log_eta_centers = 0.5 * (log_eta_bins[1:] + log_eta_bins[:-1])
    d_eta = log_eta_bins[1] - log_eta_bins[0]
    z_centers = np.asarray(z_centers)

    cells = []

    for i_z, z_c in enumerate(z_centers):
        mask_z = np.abs(redshifts - z_c) < z_halfwidth

        for i_L in range(len(logL_bins) - 1):
            lmin_val = logL_bins[i_L]
            lmax_val = logL_bins[i_L + 1]

            mask_L = (logL_Bols >= lmin_val) & (logL_Bols < lmax_val)
            mask = mask_z & mask_L

            N_total = int(np.sum(mask))
            if N_total < min_count:
                continue

            log_etas_cell = log_etas[mask]
            counts, _ = np.histogram(log_etas_cell, bins=log_eta_bins)

            # Normalised histogram (PDF)
            obs_pdf = counts.astype(float) / (N_total * d_eta)

            # Poisson error: sigma_stat = sqrt(n) / (N * d_eta).
            # Add systematic floor in quadrature: sigma_sys = sys_frac * obs_pdf.
            # This accounts for BH mass measurement uncertainty smearing the
            # histogram shape and emulator interpolation error.
            # Empty bins get inf so they contribute zero to chi2.
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_stat = np.where(
                    counts > 0,
                    np.sqrt(counts.astype(float)) / (N_total * d_eta),
                    np.inf
                )
                sigma_sys = sys_frac * obs_pdf
                sigma_pdf = np.sqrt(sigma_stat**2 + sigma_sys**2 + sys_abs**2)

            valid_mask = counts > 0

            cells.append({
                'i_z': i_z,
                'z_center': z_c,
                'lmin_val': lmin_val,
                'lmax_val': lmax_val,
                'counts': counts,
                'N_total': N_total,
                'obs_pdf': obs_pdf,
                'sigma_pdf': sigma_pdf,
                'valid_mask': valid_mask,
            })

    return cells, log_eta_bins, log_eta_centers, d_eta
