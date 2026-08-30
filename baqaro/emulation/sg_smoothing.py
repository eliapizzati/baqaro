"""Savitzky-Golay denoising of emulator summary-statistic output.

Operates on any summary statistic (QLF, BHMF, CERDF, QHMF) along its
last/observable axis. Historically QLF-only — hence the legacy
``smooth_qlf`` name, kept as a thin alias — but the algorithm is
quantity-agnostic; use :func:`smooth_function_rows` with the per-quantity
params in :data:`SG_PARAMS_BY_QUANTITY`.

Why this exists
---------------
Subsampled training QLFs carry bright-end shot noise (a handful of
high-weight halos per bright luminosity bin). The TRUE QLF (full-sim) is
smooth, so denoising the noisy emulator output toward smoothness is
legitimate. Savitzky-Golay (local polynomial fit) is used rather than a
boxcar/Gaussian because it preserves the steep bright-end slope and
curvature — validated on the L2800N5040 full-sim QLF, where SG(window=7,
poly=2) distorts the already-smooth truth by only ~0.0005 dex median
(signed bright-end bias ~0) while cutting subsample roughness ~5x. The
same validation extends to the other quantities (see
``SG_PARAMS_BY_QUANTITY``).

Apply mode "1b": smooth the emulator's prediction at use-time. Keeps the
SG params tunable per-run (env vars) with no retraining. Apply mode "1a"
would instead smooth the training data before fitting (baked in) — that's
the `_v4smooth` retrain experiment, not this module.

Usage
-----
    from baqaro.emulation.sg_smoothing import smooth_function_rows, sg_params_from_env
    win, poly = sg_params_from_env()          # win == 0 => smoothing disabled
    if win > 0:
        log_qlf = smooth_function_rows(log_qlf, win, poly)
"""
import os
import numpy as np
from scipy.signal import savgol_filter

# Bins at/below this are "empty" (floor); never smoothed across them.
DEFAULT_FLOOR = -9.5


def sg_params_from_env():
    """Read the SG window/polyorder from the environment.

    A window of ``0`` means smoothing is DISABLED, which is the default; the
    polyorder is read independently and so is still reported (as 2) in that
    case. Callers should branch on the window alone.

    BAQARO_QLF_SG_WINDOW (odd int >= 5) enables smoothing; default off (0).
    BAQARO_QLF_SG_POLY   (default 2).
    """
    win = int(os.environ.get("BAQARO_QLF_SG_WINDOW", "0"))
    poly = int(os.environ.get("BAQARO_QLF_SG_POLY", "2"))
    return win, poly


def _smooth_row(row, window, polyorder, floor):
    """SG-smooth one QLF row, but only over the contiguous block of real
    (> floor) bins, and only if that block has no interior floor gaps and
    is long enough. Floor bins are left untouched."""
    out = row.copy()
    real = np.where(row > floor + 0.1)[0]
    if real.size < max(window, polyorder + 2):
        return out
    a, b = real[0], real[-1] + 1
    seg = row[a:b]
    if not np.all(seg > floor + 0.1):       # interior floor gap -> skip (edge case)
        return out
    w = window
    if w > seg.size:                         # clamp to an odd window <= segment
        w = seg.size if seg.size % 2 == 1 else seg.size - 1
    if w < polyorder + 2 or w % 2 == 0:
        return out
    out[a:b] = savgol_filter(seg, w, polyorder)
    return out


# Validated per-quantity SG (window, polyorder), checked against
# full-catalogue L2800N5040 truth.
# Monotonic declines (QLF, BHMF) tolerate the stronger (7,2); peaked PDFs
# (QHMF host-mass fn, CERDF) need the gentler (5,3) or SG clips the peak
# (SG(7,2) flattened the CERDF peak by -0.036 dex).
SG_PARAMS_BY_QUANTITY = {
    "qlf":   (7, 2),
    "bhmf":  (7, 2),
    "qhmf":  (5, 3),
    "cerdf": (5, 3),
}


def smooth_function_rows(arr, window, polyorder=2, floor=DEFAULT_FLOOR):
    """Savitzky-Golay denoise along the LAST axis of any summary statistic.

    Works for QLF/BHMF ``(n_z, n_bins)`` and CERDF/QHMF
    ``(n_z, n_Lthr, n_bins)`` alike (the observable axis is last in the
    training schema). Smooths only contiguous real (> ``floor``) segments,
    leaving empty bins untouched. ``window <= 0`` is a no-op.

    This is the generalized form of :func:`smooth_qlf`; use the
    per-quantity ``(window, polyorder)`` in :data:`SG_PARAMS_BY_QUANTITY`.
    """
    return smooth_qlf(arr, window, polyorder, floor)


def smooth_qlf(log_qlf, window, polyorder=2, floor=DEFAULT_FLOOR):
    """Savitzky-Golay denoise a summary statistic along its last axis.

    Named ``smooth_qlf`` for back-compat; despite the name it operates on
    any ``(..., n_bins)`` array (see :func:`smooth_function_rows`).

    Parameters
    ----------
    log_qlf : ndarray
        ``(..., n_lbins)`` — last axis is luminosity. Handles 1D
        ``(n_lbins,)``, 2D ``(n_z, n_lbins)``, and batched
        ``(n_batch, n_z, n_lbins)``.
    window : int
        SG window length (odd, >= polyorder+2). ``window <= 0`` is a no-op
        (returns input unchanged) so callers can leave it env-gated.
    polyorder : int
        SG polynomial order (default 2 — preserves QLF slope+curvature).
    floor : float
        Empty-bin sentinel; bins at/below ``floor`` are not smoothed.

    Returns
    -------
    ndarray, same shape as input.
    """
    if window is None or window <= 0:
        return log_qlf
    arr = np.asarray(log_qlf, dtype=float)
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        out[i] = _smooth_row(flat[i], window, polyorder, floor)
    return out.reshape(arr.shape)


def wrap_emulator_predict(emulator, window, polyorder=2, floor=DEFAULT_FLOOR):
    """Monkey-patch ``emulator.predict_mean_only`` to SG-smooth its QLF
    output along the last (luminosity) axis. No-op when ``window <= 0``.

    Returns the emulator (patched in place) for chaining. Idempotent-safe:
    stores the original on ``_raw_predict_mean_only``.
    """
    if window is None or window <= 0:
        return emulator
    if getattr(emulator, "_qlf_sg_wrapped", False):
        return emulator
    raw = emulator.predict_mean_only
    emulator._raw_predict_mean_only = raw

    def _patched(X_new, _raw=raw, _w=window, _p=polyorder, _f=floor):
        return smooth_qlf(_raw(X_new), _w, _p, _f)

    emulator.predict_mean_only = _patched
    emulator._qlf_sg_wrapped = True
    return emulator
