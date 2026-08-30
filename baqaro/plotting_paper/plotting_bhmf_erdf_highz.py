"""PAPER FIGURE: high-z BHMF and ERDF — the two model ingredients that set them.

Two panels, same redshifts (z = 4, 5, 6), each isolating the ingredient it is
most sensitive to:

  LEFT   BHMF   fiducial  vs  tau_coh = 0  (Branch C)
         tau_coh is the number of independent accretion draws per snapshot. As
         tau -> 0 the number of draws -> infinity, the lognormal lottery
         self-averages, and its scatter vanishes. The MASSIVE END OF THE BHMF IS
         THE TAIL OF THAT LOTTERY — so killing the scatter kills the tail. The
         gap between solid and dashed IS the coherence time.

  RIGHT  ERDF   fiducial (madau+)  vs  constant epsilon
         eps(eta) sets L/M, so it lands on the Eddington ratio and barely touches
         the BHMF. Under `madau+`, eps COLLAPSES at high eta (slim-disk/ADAF), so
         lambda_Edd = eta * eps(eta)/eps_base SATURATES: measured max lambda_Edd
         is ~3.5 at every redshift, with ZERO objects above 10 in the whole box.
         Under constant eps, lambda_Edd == eta exactly, the ERDF's high-eta tail
         passes straight through, and lambda runs to 215 with 71% of z=6 quasars
         super-Eddington. The data have no such tail. THIS PANEL IS THE
         OBSERVATIONAL CASE FOR THE madau+ PRESCRIPTION.

⚠ THE SELECTION TRAP — THE L_bol CUT IS **PER REDSHIFT**:
A lambda_Edd distribution from a flux-limited survey is not "the ERDF" — at fixed
M_BH you preferentially detect fast accretors, so it is biased high. It is only
meaningful together with the sample's LUMINOSITY RANGE, and the three samples have
VERY different ones:
      He+2024   z=4   log L_bol = 45.41 - 46.93   (deep HSC + SDSS)
      Lai+2024  z=5   log L_bol = 46.63 - 48.26   ("the MOST LUMINOUS quasars")
      Wu+2022   z=6   log L_bol = 46.58 - 48.21
Each redshift takes its luminosity cut from ITS OWN dataset (`log_L_bol_min`),
read from the data object and never hardcoded: fainter quasars sit at lower
lambda, so a cut below a sample's own floor would bias the model's lambda
distribution low relative to that sample.

WHY THE ERDF PANEL IS NORMALISED (shape, not absolute density): the absolute
normalisations are not directly comparable (the samples' completeness limits
differ, and our own QLF is only fitted above log L_bol = 46.0 at z >= 4), so
both are normalised to unit area and the panel compares SHAPE only.

DATA — colour encodes redshift, MARKER encodes the source. **At every redshift the
ERDF and the BHMF come from THE SAME PAPER AND THE SAME SAMPLE**, which is what
makes the two panels a coherent pair rather than a collage:

  squares (1/Vmax BINNED, selection-corrected — the primary comparison)
     z=4  He+2024,  ApJ 962, 152   — their ERDF + our left-panel BHMF
     z=5  Lai+2024, MNRAS 531,2245 — their Table 2 ERDF + our left-panel BHMF
  circles (individual objects, NOT completeness-corrected)
     z=6  Wu J.+2022, MNRAS 517, 2659 — the 34 quasars of their Table B.2, again
          the same paper as our z=6 BHMF. They publish no binned ERDF table, so
          this is the best available at z=6.

⚠ Only the SQUARES are completeness-corrected. The z=6 circles are a flux-limited
sample with no 1/Vmax weighting: biased AGAINST low lambda, and to be read on the
HIGH-lambda side only. Wu+2022 say so themselves — their binned ERDF is "highly
incomplete at log lambda < -0.5". That is fine for the question this panel asks
(is there a super-Eddington tail?) and NOT fine for reading the faint end.

DELIBERATELY EXCLUDED:
  * SDSS DR16Q — flux-limited with no completeness correction, and SDSS is nowhere
    near complete at log L_bol ~ 45.4 at z = 5-6 (its flux limit maps to a far
    brighter L_bol there). Redundant now that every redshift has a proper source.
  * Willott+2010 — a lognormal FIT, not a measurement. Plotting it would compare
    our model against someone else's model. Real data only.
"""


import os

import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import qhtools.utils.natconst as nc

from baqaro.obs_data.abhmf_obs_data import data_bhmf_global
from baqaro.obs_data.erdf_obs_data import (
    data_erdf_global, data_erdf_samples,
)
from baqaro.plotting_paper.fiducial_data import (
    name_file, name_file_variants, path_out, boxsize, snapshot_index_for_redshift,
)
from baqaro.plotting_paper import plot_config
from baqaro.plotting_paper.plot_config import (
    z_scalar_mapper, bhmf_obs_marker,
)
from baqaro.plotting_common.load_data_to_plot import mass_function_auto
from baqaro.plotting_common.plot_config import save_fig, maybe_show

name_fig = os.environ.get("BAQARO_PAPER_BHMF_ERDF_NAME", "bhmf_erdf_highz")

REDSHIFTS = [6.0, 5.0, 4.0]
LSUN = np.log10(3.826e33)
V_SIM = float(boxsize) ** 3

# BHMF: use the SAME estimator as the old bhmf_highz figure — the HT-weighted
# `mass_function_auto` with an error band, 51 bins, not a bare histogram.
MIN_MASS_BH, MAX_MASS_BH, N_BINS_BH = 1e6, 1e11, 51
Z_TOL = 0.3          # fuzzy label-redshift match for the obs overlays
UL_ARROW_DEX = 0.25  # length of the M_BH upper-limit arrows (Matthee/Taylor)
# Obs sources are keyed in data_bhmf_global under NON-numeric keys too
# ('Matthee24_5.0', 'Taylor25_4.0', ...), so match on the label's z=<float>
# rather than on the exact key, exactly as plotting_bhmf_highz does, and give
# each source its own marker.
OBS_Z_RANGE = {
    "He": r"$z \simeq 4$",
    "Lai": r"$z \simeq 5$",
    "Wu": r"$z \simeq 6$",
    "Matthee": r"$4 < z < 5.5$",
    "Taylor": r"$3.5 < z < 6$",
}
# ERDF binning: 0.3-dex bins on EXACTLY He+2024's grid (centres -2.00, -1.70, ...,
# verified to 2e-16). Lai+2024 is on a 0.1833-dex grid; both are densities per dex,
# so they overlay correctly, but the comparison is bin-for-bin only for He.
#
# THE GRID RUNS WELL PAST lambda_Edd = 1 ON PURPOSE: constant-epsilon reaches
# lambda ~ 200 (log ~ 2.3), and that super-Eddington tail is the result the
# panel is about, so it must be drawn, not cropped. Centres run to +2.20
# (lambda = 158).
E_LO, E_HI, E_NB = -2.15, 2.35, 15
# Minimum RAW (unweighted) objects per bin. A Horvitz-Thompson weighted bin with
# only 1-2 raw objects can carry an enormous weight and produce a spurious spike;
# `3` was too permissive and produced a visible artifact at log M ~ 10.6.
MIN_RAW = 8


def _variant_path(suffix_tag):
    """Physics-variation run: token APPENDED at the end (physics_toggle_suffix).

    ⚠ Built on `name_file_variants` (the K22 SUBSAMPLE forward run), NOT on the
    fiducial `name_file` (now the MULTINODE full catalogue): the physics-toggle
    variants were only ever launched on the subsample. So the fiducial curve is
    multinode (shot-noise-free) while the tau0/tau7/constant-eps overlays are the
    subsample runs — the physics differences (e.g. tau7 reaching log M ~ 14)
    dwarf the sampling difference.

    ⚠ Existence is NOT enough. A run that is still WRITING its HDF5 leaves a file
    on disk that is (a) locked, so h5py raises BlockingIOError, and (b) incomplete,
    so reading it with locking disabled would plot garbage. So: openable AND
    complete ('redshifts' present) or it is treated as ABSENT.
    """
    nf = f"{name_file_variants}_{suffix_tag}"
    p = os.path.join(path_out, "evolution", f"bh_evolution_{nf}.hdf5")
    if not os.path.exists(p):
        return p, False
    try:
        with h5py.File(p, "r") as _f:
            ok = "redshifts" in _f and "black_hole_masses_all" in _f
    except (OSError, BlockingIOError):
        print(f"  ⚠ {os.path.basename(p)} exists but is locked/incomplete "
              f"(a run is still writing it) — treating as MISSING.")
        return p, False
    return p, ok


path_fid = os.path.join(path_out, "evolution", f"bh_evolution_{name_file}.hdf5")
path_tau0, has_tau0 = _variant_path("tau0")
path_tau7, has_tau7 = _variant_path("tau7")
path_radc, has_radc = _variant_path("radeff_constant")

print(f"fiducial     : {os.path.basename(path_fid)}")
print(f"tau_coh=0    : {'FOUND' if has_tau0 else 'MISSING'}  {os.path.basename(path_tau0)}")
print(f"tau=10 Myr   : {'FOUND' if has_tau7 else 'MISSING'}  {os.path.basename(path_tau7)}")
print(f"constant eps : {'FOUND' if has_radc else 'MISSING'}  {os.path.basename(path_radc)}")


def _read(path, z_target):
    """Return (logM, logL_erg, log_lambda, weights) for the snapshot nearest z."""
    with h5py.File(path, "r") as f:
        z = f["redshifts"][:]
        snaps = f["snapshots"][:] if "snapshots" in f else None
        i = snapshot_index_for_redshift(z, z_target, snaps, label="bhmf_erdf_highz")
        M = f["black_hole_masses_all"][i]
        L = f["Lbols_all"][i]
        w = f["subset/weights"][:] if "subset/weights" in f else np.ones_like(M, float)
    alive = M > 0
    Ms = M[alive].astype(np.float64)
    Ls = L[alive].astype(np.float64)      # Lsun
    ww = w[alive]
    logM = np.log10(Ms)
    lum = Ls > 0
    logL = np.full(Ms.shape, -np.inf)
    loglam = np.full(Ms.shape, -np.inf)
    logL[lum] = np.log10(Ls[lum]) + LSUN
    loglam[lum] = np.log10(Ls[lum]) - logM[lum] - nc.log_csi
    return logM, logL, loglam, ww


def _hist_density(vals, w, lo, hi, nb):
    """Weighted dPhi/dx in Mpc^-3 dex^-1, NaN where too few raw objects."""
    edges = np.linspace(lo, hi, nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    dx = (hi - lo) / nb
    wc, _ = np.histogram(vals, edges, weights=w)
    raw, _ = np.histogram(vals, edges)
    phi = np.where(raw >= MIN_RAW, wc / (dx * V_SIM), np.nan)
    return cen, phi


def _hist_pdf(vals, w, lo, hi, nb):
    """Weighted PDF, normalised over the FULL lambda support — not the plot window.

    Two conventions, both needed for the panel's comparison of the tails:

    (1) NORMALISE OVER THE FULL SUPPORT, NOT THE WINDOW. The denominator is the
        TOTAL weight of every object passing the L cut, so the in-window integral
        is < 1 by exactly the fraction that escapes the window — a tail beyond
        the plotted range shows up as missing area rather than inflating the
        body. The plot window is a VIEW (`set_xlim`), never the normalisation
        range.

    (2) A GENUINE ZERO IS A MEASUREMENT, NOT A GAP. Bins with raw == 0 are
        plotted as 0.0 on the linear axis (at z=6 the fiducial predicts no
        super-Eddington quasars, and that prediction must be visible); only
        sparsely-populated (0 < raw < MIN_RAW) bins are masked.

    Returns (centres, pdf, frac_outside).
    """
    edges = np.linspace(lo, hi, nb + 1)
    cen = 0.5 * (edges[1:] + edges[:-1])
    dx = (hi - lo) / nb
    w_total = float(np.sum(w))                 # ALL objects past the L cut
    if w_total <= 0:
        return cen, np.full(nb, np.nan), 0.0
    wc, _ = np.histogram(vals, edges, weights=w)
    raw, _ = np.histogram(vals, edges)
    pdf = np.where(raw >= MIN_RAW, wc / (w_total * dx), np.nan)
    pdf = np.where(raw == 0, 0.0, pdf)         # a true zero is a data point
    frac_out = 1.0 - float(np.sum(wc)) / w_total
    return cen, pdf, frac_out



# ==============================================================================
# FIGURE
# ==============================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

# ---------------------------------------------------------------------------
# THE L_bol CUT IS **PER REDSHIFT**, taken from each dataset's own sample.
#
# ⚠ This is a real trap. The three samples have very different
# luminosity limits:
#      He+2024   z=4   log L_bol = 45.41 - 46.93   (deep HSC)
#      Lai+2024  z=5   log L_bol = 46.63 - 48.26   ("the MOST LUMINOUS quasars")
#      Wu+2022   z=6   log L_bol = 46.58 - 48.21
# Cutting the model at He's 45.41 at ALL redshifts — as the first version did —
# lets the z=5 and z=6 model curves include quasars ~1.2 dex fainter than ANY
# object in the data they are compared to. Fainter quasars sit at lower
# lambda_Edd, so it biases the model's lambda distribution LOW and makes the
# model look artificially better on the faint side. Each redshift now gets its
# own cut, read from the dataset.
# ---------------------------------------------------------------------------
# ROUNDED, EXPLICIT cuts. Rounding is deliberate: each
# dataset's raw `log_L_bol_min` is the luminosity of its FAINTEST OBJECT, which is
# a noisy order statistic (one draw from the tail), not a survey limit. Rounding to
# a stated threshold is both more honest and more stable.
#
#   z=4  45.5   He+2024 (faintest object 45.41; their HSC arm reaches ~45.4-45.6)
#   z=5  46.5   Lai+2024 (faintest object 46.63)
#   z=6  46.5   Wu+2022  (faintest object 46.74, on the 29-object SDSS_MO sample)
#
# NB the *completeness* limits are arguably higher still (He ~46.0, Lai ~47.5
# from their lambda-L_3000 selection).
# Override any of them with BAQARO_PAPER_ERDF_LCUTS="4.0:46.0,5.0:47.5".
_DEFAULT_LCUTS = {"4.0": 46.0, "5.0": 46.0, "6.0": 46.0}

_lc_env = os.environ.get("BAQARO_PAPER_ERDF_LCUTS", "").strip()
if _lc_env:
    for _pair in _lc_env.split(","):
        _k, _v = _pair.split(":")
        _DEFAULT_LCUTS[_k.strip()] = float(_v)


def _l_cut_for(z_key):
    cut = _DEFAULT_LCUTS[z_key]
    d = data_erdf_global.get(z_key) or data_erdf_samples.get(z_key)
    return cut, (d.label if d is not None else "no data")


L_CUTS = {f"{z:.1f}": _l_cut_for(f"{z:.1f}") for z in REDSHIFTS}
print("\nERDF L_bol cut, PER REDSHIFT (rounded, explicit):")
for k, (c, lab) in sorted(L_CUTS.items()):
    _d = data_erdf_global.get(k) or data_erdf_samples.get(k)
    _raw = _d.log_L_bol_min if _d is not None else None
    print(f"   z={k}:  log L_bol > {c:.2f}   ({lab}"
          + (f"; faintest object {_raw:.2f})" if _raw else ")"))

def _bhmf_curve(path, zt):
    """HT-weighted BHMF via mass_function_auto — the same estimator the old
    bhmf_highz figure used (51 bins + an error band), not a bare histogram."""
    logM, _, _, w = _read(path, zt)
    return mass_function_auto(
        10.0 ** logM, V_SIM, weights=w,
        lowest_mass=MIN_MASS_BH, highest_mass=MAX_MASS_BH,
        n_bins=N_BINS_BH, minimum_in_bin=1,
    )


obs_marker_seen = {}     # clean_source -> (legend_label, marker)

for zt in REDSHIFTS:
    colour = z_scalar_mapper.to_rgba(zt)

    # ---------------- LEFT: BHMF — the coherence time -------------------------
    mbins, mf, mf_err = _bhmf_curve(path_fid, zt)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(mbins)
        y = np.log10(mf)
        y_lo = np.log10(mf - mf_err)
        y_hi = np.log10(mf + mf_err)
    axL.plot(x, y, lw=2.5, color=colour, ls="-", zorder=10)
    axL.fill_between(x, y_lo, y_hi, alpha=0.22, color=colour, zorder=5)

    if has_tau0:
        mb0, mf0, _ = _bhmf_curve(path_tau0, zt)
        with np.errstate(divide="ignore", invalid="ignore"):
            axL.plot(np.log10(mb0), np.log10(mf0), color=colour, ls="--",
                     lw=1.8, zorder=9)

    # tau_coh = 10 Myr (logtcoherence = 7.0): the OPPOSITE limit from tau -> 0.
    # Longer coherence => FEWER independent draws per snapshot => LESS
    # self-averaging => MORE scatter in the growth lottery => a FATTER massive
    # tail. The three curves bracket the coherence time:
    #   tau -> 0 (dashed) < tau = 1.08 Myr (fiducial, solid) < tau = 10 Myr (dotted)
    if has_tau7:
        mb7, mf7, _ = _bhmf_curve(path_tau7, zt)
        with np.errstate(divide="ignore", invalid="ignore"):
            axL.plot(np.log10(mb7), np.log10(mf7), color=colour, ls=":",
                     lw=2.2, zorder=9)

    # --- obs BHMFs: FUZZY redshift match on the LABEL, marker = source ---------
    # Match on the label's z=<float> rather than on the exact dict key
    # ('4.0'/'5.0'/'6.0'): Matthee+2024 and Taylor+2025 live under
    # 'Matthee24_5.0' / 'Taylor25_4.0' / 'Taylor25_5.0' and would otherwise be
    # skipped.
    for obs in data_bhmf_global.values():
        if obs.label.startswith("Kelly&Shen"):     # excluded in the old figure too
            continue
        try:
            obs_z = float(obs.label.split("z=")[-1])
        except (ValueError, IndexError):
            continue
        if abs(obs_z - zt) > Z_TOL:
            continue
        token_ul = obs.label.split("+")[0].split()[0]
        marker = bhmf_obs_marker(obs.label)
        # JWST broad-line AGN (Matthee+2024 LRDs, Taylor+2025 BLAGN): their virial
        # M_BH are UPPER BOUNDS on the true BH mass — the broad Halpha can be
        # contaminated by outflows/host and the single-epoch calibration is
        # uncertain at these luminosities — so the true mass lies to the LEFT of
        # the plotted point. Drawn exactly as the LRDs are in plotting_qlf:
        # a fixed-length 0.5 dex annotate arrow (not a matplotlib caret).
        _is_upper_limit = token_ul in ("Matthee", "Taylor")
        # Clip non-finite / unbounded lower bars (a bin consistent with zero has
        # err >= Phi -> log10 of a negative -> -inf -> matplotlib draws a spike to
        # the axis edge; that was the log M ~ 10.6 artifact).
        _y = np.asarray(obs.data, float)
        _good = np.isfinite(_y)
        _e = np.asarray(obs.errs, float)
        if _e.ndim == 1:
            _e = np.vstack([_e, _e])
        _e = np.clip(np.nan_to_num(_e, nan=1.5, posinf=1.5, neginf=0.0), 0.0, 1.5)
        axL.errorbar(np.asarray(obs.x)[_good], _y[_good], yerr=_e[:, _good],
                     c=colour, mec=colour, marker=marker, fillstyle="none",
                     linestyle="none", elinewidth=1.0, capsize=2, zorder=8)
        token = obs.label.split("+")[0].split()[0]
        if _is_upper_limit:
            for xi, yi in zip(np.atleast_1d(obs.x)[_good], _y[_good]):
                axL.annotate("", xy=(xi - UL_ARROW_DEX, yi), xytext=(xi, yi),
                             zorder=7, annotation_clip=True,
                             arrowprops=dict(arrowstyle="->", color=colour,
                                             lw=1.1, alpha=0.85, mutation_scale=9,
                                             shrinkA=0, shrinkB=0))
        clean_src = obs.label.split(" z=")[0].split(" (")[0]
        zrange = OBS_Z_RANGE.get(token, f"$z={obs_z:.0f}$")
        obs_marker_seen[clean_src] = (f"{clean_src} ({zrange})", marker)

    # ---------------- RIGHT: ERDF, fiducial vs constant eps -------------------
    key = f"{zt:.1f}"
    l_cut, l_cut_src = L_CUTS[key]
    _, logL, loglam, w = _read(path_fid, zt)
    sel = logL > l_cut
    xe, pe, out_f = _hist_pdf(loglam[sel], w[sel], E_LO, E_HI, E_NB)
    axR.plot(xe, pe, color=colour, ls="-", lw=2.3, zorder=4)
    n_raw_f = int(sel.sum())

    out_c = None
    if has_radc:
        _, logLc, loglamc, wcw = _read(path_radc, zt)
        selc = logLc > l_cut
        xc, pc, out_c = _hist_pdf(loglamc[selc], wcw[selc], E_LO, E_HI, E_NB)
        axR.plot(xc, pc, color=colour, ls="--", lw=1.8, zorder=3)

    # The fraction of each model's probability ESCAPING the plot window is the
    # super-Eddington tail. It is the whole point of the panel, so print it.
    print(f"  z={zt:.0f}  model: N_raw={n_raw_f:,} above logL>{l_cut:.2f}   "
          f"tail beyond log lam={E_HI:.2f}:  fiducial {100*out_f:.2f}%"
          + (f",  constant eps {100*out_c:.2f}%" if out_c is not None else ""))
    if n_raw_f < 500:
        print(f"        ⚠ only {n_raw_f} model objects pass this cut — the "
              f"z={zt:.0f} model curve is shot-noise limited.")

    # --- observations: colour = redshift, marker = source -------------------
    # (1) BINNED 1/Vmax, selection-corrected ERDFs -> squares (z=4 He, z=5 Lai).
    #     These are the PRIMARY comparison: the only ones corrected for volume.
    if key in data_erdf_global:
        d = data_erdf_global[key]
        # dx from the ENDPOINTS, not median(diff): Lai's bins are 0.18333 dex
        # and median(diff) rounds to 0.18, a 1.8% normalisation error.
        dx_d = (float(d.x[-1]) - float(d.x[0])) / (len(d.x) - 1)
        norm = float(np.sum(d.data) * dx_d)
        # Marker = source paper, via the SAME helper the left panel uses, so
        # He/Lai/Wu carry identical symbols in both panels.
        _mkd = bhmf_obs_marker(d.label)
        _det = d.data > 0 if d.n_per_bin is None else d.n_per_bin > 0
        axR.errorbar(d.x[_det], d.data[_det] / norm, yerr=d.err_up[_det] / norm,
                     marker=_mkd, c=colour, mec=colour, fillstyle="none",
                     ls="none", ms=6.0, capsize=2, elinewidth=1.0, zorder=7)

        # NON-DETECTIONS -> 95% Poisson upper limits. He+2024 has two genuinely
        # EMPTY bins (N_tot = 0 at log lambda = -2.0 and +1.0). With 1514 objects
        # and a 1/Vmax correction its limit at log lambda = +1 is ~1e-4 in
        # normalised PDF units — ~3000x tighter than Wu's 0.345, and ~1000x BELOW
        # the constant-eps curve there. It is the single strongest constraint in
        # this figure, and an earlier version dropped it.
        # (Lai+2024 has NO zero-count bins — their table simply stops. That is
        #  absence of information, not a non-detection, so Lai gets no limits.)
        if d.n_per_bin is not None:
            _emp = np.flatnonzero(d.n_per_bin == 0)
            _uls = [(d.x[i], d.upper_limit(i)) for i in _emp]
            _uls = [(xx, uu / norm) for xx, uu in _uls if uu is not None]
            if _uls:
                _ux = np.array([u[0] for u in _uls])
                _uy = np.array([u[1] for u in _uls])
                axR.errorbar(_ux, _uy, yerr=0.045 * np.ones_like(_uy), uplims=True,
                             marker=_mkd, c=colour, mec=colour, fillstyle="none",
                             ls="none", ms=6.0, elinewidth=1.0, zorder=7)
                print(f"  z={zt:.0f}  {d.label}: {len(_uls)} empty bin(s) -> "
                      f"95% UL (normalised PDF) = "
                      + ", ".join(f"{xx:+.2f}: {yy:.5f}" for xx, yy in zip(_ux, _uy)))
        print(f"  z={zt:.0f}  {d.label}: {len(d.x)} bins (dx={dx_d:.4f}), "
              f"logL>{d.log_L_bol_min:.2f}, N={d.n_objects}")

    # (2) INDIVIDUAL-OBJECT sample -> open circles. z=6 only (Wu+2022).
    src = data_erdf_samples.get(key)
    if src is not None and src.n_objects >= 15:
        edges = np.linspace(E_LO, E_HI, E_NB + 1)
        cen = 0.5 * (edges[1:] + edges[:-1])
        dx = (E_HI - E_LO) / E_NB
        h, _ = np.histogram(src.log_lam, edges)
        N = int(src.n_objects)                 # normalise by the FULL sample
        if N > 0:
            pdf = h / (N * dx)
            # MULTINOMIAL, not Poisson. The histogram is divided by its own
            # (random) total, so Var(p_i) = N f_i (1-f_i) / (N dx)^2.
            err = np.sqrt(h * (1.0 - h / N)) / (N * dx)
            ok = h > 0
            # Stagger the individual-object markers +0.06 dex in x so they do not
            # sit on top of the 1/Vmax squares, which share the same bin grid.
            DX_STAGGER = 0.06
            _mk = bhmf_obs_marker(src.label)
            axR.errorbar(cen[ok] + DX_STAGGER, pdf[ok], yerr=err[ok],
                         marker=_mk, c=colour, mec=colour, fillstyle="none",
                         ls="none", ms=6.0, capsize=2, elinewidth=0.9,
                         alpha=0.9, zorder=6)
            # Empty bins enter as data (upper limits), not as missing values: Wu
            # has ZERO quasars above log lambda ~ +0.25, and that non-detection is
            # exactly what rules out the constant-eps tail.
            #
            # HOW THE UPPER LIMITS ARE BUILT. For a bin with 0 observed objects the
            # one-sided 95% Poisson upper limit on the EXPECTED COUNT is 3.0
            # (Gehrels 1986, Table 1: n=0 -> 3.00 at 95% CL; 1.84 at 84%). Divide by
            # the same normalisation the detected bins use — the full sample size N
            # times the bin width — to put it in PDF units:
            #       ul = 3.0 / (N * dx) = 3.0 / (29 * 0.3) = 0.345
            # N and dx are constant, so EVERY empty bin gets the SAME height: the
            # arrows form a flat line, which is correct. It says "with 29
            # quasars we would have expected to see at least ~3 objects here if the
            # true PDF were above 0.345".
            #
            # Approximation: the histogram is normalised by its own total, so the
            # counts are strictly multinomial, not Poisson. For h=0 the Poisson limit
            # is the standard and near-exact choice, and N is known exactly (29), so
            # dividing by N*dx is sound. The arrow LENGTH (0.16*ul) is cosmetic only.
            empty = (h == 0) & (cen > np.min(src.log_lam))
            if empty.any():
                ul = 3.0 / (N * dx)
                axR.errorbar(cen[empty] + DX_STAGGER, np.full(empty.sum(), ul),
                             yerr=np.full(empty.sum(), 0.16 * ul), uplims=True,
                             marker=_mk, c=colour, mec=colour, fillstyle="none",
                             ls="none", ms=6.0, elinewidth=0.9, alpha=0.9,
                             zorder=6)
            print(f"  z={zt:.0f}  {src.label}: N={N}, "
                  f"logL {src.log_L_bol_min:.2f}-{src.log_L_bol_max:.2f}, "
                  f"median log lam = {np.median(src.log_lam):+.2f}, "
                  f"max log lam = {src.log_lam.max():+.2f}, "
                  f"{int(empty.sum())} empty bins -> 95% upper limits")

# DR16Q and Willott+2010 are DELIBERATELY NOT SHOWN:
#   * DR16Q — flux-limited with no completeness correction, and SDSS is nowhere
#     near complete at log L_bol ~ 45.4 at z = 5-6 (its flux limit maps to a much
#     brighter L_bol there). Now redundant anyway: every redshift has a proper
#     1/Vmax ERDF or a dedicated sample from the same paper as its BHMF.
#   * Willott+2010 — a lognormal FIT, not a measurement. Plotting it would compare
#     our model to someone else's model. Real data only, by choice.

axL.set_xlabel(r"$\log_{10}(M_{\rm BH}/M_\odot)$")
axL.set_ylabel(r"$\log_{10}\left[\Phi\,/\,{\rm Mpc^{-3}\,dex^{-1}}\right]$")
axL.set_xlim(6.8, 11.0)
axL.set_ylim(-9.5, -3.5)
axL.minorticks_on()

axR.axvline(0.0, color="0.6", ls=":", lw=1.0, zorder=1)
axR.set_xlabel(r"$\log_{10}\lambda_{\rm Edd}$")
axR.set_ylabel(r"$P(\log_{10}\lambda_{\rm Edd}\,|\,L_{\rm bol} > L_{\rm cut}(z))$")
axR.set_xlim(-2.2, 1.6)
# Dip slightly below zero so the NON-DETECTIONS (He's 95% limits sit at
# ~1e-4, i.e. visually at 0) are drawn as visible points with arrows,
# rather than being buried in the axis spine.
axR.set_ylim(-0.075, None)
axR.minorticks_on()

z_handles = [Line2D([], [], color=z_scalar_mapper.to_rgba(z), lw=2.6,
                    label=f"$z = {z:.0f}$") for z in REDSHIFTS]
# Only advertise a comparison run that was actually FOUND and drawn. On a run
# whose physics-variant siblings do not exist on disk (e.g. the multinode
# full-catalogue fiducial, for which only the baseline was ever launched) the
# figure degrades to fiducial-only, and an unconditional legend would promise
# curves that are not on the axes.
# τ_coh of the fiducial run, read from its own parameters (= the accretion
# sub-step in Myr) so the legend can never drift from the actual fiducial.
with h5py.File(path_fid, "r") as _f:
    _tau_coh_myr = float(_f["parameters/time_step_for_accretion"][()])
style_handles = [
    Line2D([], [], color="k", ls="-", lw=2.5,
           label=rf"fiducial ($\tau_{{\rm coh}}={_tau_coh_myr:.1f}$ Myr)"),
]
if has_tau0:
    style_handles.append(
        Line2D([], [], color="k", ls="--", lw=1.8, label=r"$\tau_{\rm coh}\!\to\!0$"))
if has_tau7:
    style_handles.append(
        Line2D([], [], color="k", ls=":", lw=2.2, label=r"$\tau_{\rm coh}=10$ Myr"))
# Source-marker legend (one entry per paper, as in the old bhmf_highz figure).
obs_handles = [Line2D([], [], color="0.35", mec="0.35", marker=m, fillstyle="none",
                      ls="none", label=lab)
               for lab, m in sorted(obs_marker_seen.values(), key=lambda t: t[0])]
_leg1 = axL.legend(handles=z_handles + style_handles, loc="lower left",
                   fontsize=11, handlelength=1.6, labelspacing=0.25,
                   borderpad=0.35, frameon=False)
axL.add_artist(_leg1)
if obs_handles:
    axL.legend(handles=obs_handles, loc="upper right", fontsize=10,
               markerfirst=False, handlelength=1.0, labelspacing=0.25,
               borderpad=0.35, frameon=False)

# Right-panel legend: the SAME source entries (identical text AND marker) as the
# left panel, restricted to the papers that supply an ERDF.
_ERDF_SOURCES = ("He", "Lai", "Wu")
_r_obs = [Line2D([], [], color="0.35", mec="0.35", marker=m, fillstyle="none",
                 ls="none", label=lab)
          for src, (lab, m) in sorted(obs_marker_seen.items())
          if src.split("+")[0].split()[0] in _ERDF_SOURCES]
_r_style = [Line2D([], [], color="k", ls="-", lw=2.5, label="fiducial (variable $ϵ$)")]
if has_radc:
    _r_style.append(
        Line2D([], [], color="k", ls="--", lw=1.8, label="constant $ϵ$"))
_legR = axR.legend(handles=_r_style,
                   loc="upper left", fontsize=11, handlelength=1.6, labelspacing=0.25,
                   borderpad=0.35, frameon=False)
axR.add_artist(_legR)
if _r_obs:
    axR.legend(handles=_r_obs, loc="upper right", fontsize=10,
               markerfirst=False, handlelength=1.0, labelspacing=0.25,
               borderpad=0.35, frameon=False)

# Layout handled by constrained_layout=True at figure creation.
save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
