"""
Black hole seeding module.
==========================

Seeds black holes in newly-resolved halos with mass proportional to
the halo mass:

    M_BH = M_halo * 10^(logfseed + scatter)

where ``scatter ~ Normal(0, sigmaseed)`` in dex (optional).

Called once per snapshot for halos that appear (``mask_spawning_now``)
in the main evolution loop (``main_evolution.py``).
"""

import numpy as np


def spawn_BHs(M_halos, logfseed, sigmaseed=None, rng=None):
    """Seed black holes from halo masses.

    Parameters
    ----------
    M_halos : array_like
        Halo masses (code units, typically float32 from mmap).
    logfseed : float
        log10 of the deterministic seed mass fraction:
        ``M_BH = M_halo * 10^logfseed``.
        Typical value: -4.2 (i.e. M_BH ~ 10^{-4.2} * M_halo).
    sigmaseed : float or None, optional
        Lognormal scatter (1-sigma, in dex) around ``logfseed``.
        Each halo draws its own ``logf ~ Normal(logfseed, sigmaseed)``.
        If None (default), all halos get the same deterministic fraction.
    rng : numpy.random.Generator or None, optional
        Random number generator. Required when ``sigmaseed`` is not None.
        If None, a fresh default generator is created (not reproducible).

    Returns
    -------
    M_BHs : ndarray
        Seeded black hole masses, same shape and dtype as input.

    Notes
    -----
    The function preserves the input dtype (float32 or float64) throughout
    to avoid unnecessary type promotion. When ``sigmaseed`` is provided,
    it uses the ``out=`` parameter of ``rng.standard_normal`` to write
    directly into the output buffer (zero extra allocation).
    """

    halo_arr = np.asarray(M_halos)
    # Preserve input precision — float32 from mmap, float64 otherwise
    if halo_arr.dtype == np.float32:
        work_dtype = np.float32
    elif np.issubdtype(halo_arr.dtype, np.floating):
        work_dtype = np.float64
    else:
        work_dtype = np.float32

    halo_arr = halo_arr.astype(work_dtype, copy=False)

    if sigmaseed is None:
        # Deterministic fast path: one vector multiply, no RNG needed.
        return halo_arr * work_dtype(10.0**logfseed)

    if sigmaseed < 0:
        raise ValueError("sigmaseed must be >= 0 or None")

    if rng is None:
        rng = np.random.default_rng()

    # Scattered path — all in-place to avoid temporary arrays:
    #   1. z ~ N(0,1)            — written directly into output buffer
    #   2. z = logfseed + sigmaseed * z   — per-halo log10(fraction)
    #   3. z = 10^z              — convert to linear fraction
    #   4. z *= M_halo           — scale by halo mass
    M_BHs = np.empty_like(halo_arr, dtype=work_dtype)
    rng.standard_normal(size=halo_arr.shape, dtype=work_dtype, out=M_BHs)
    M_BHs *= work_dtype(sigmaseed)
    M_BHs += work_dtype(logfseed)
    np.power(work_dtype(10.0), M_BHs, out=M_BHs)
    M_BHs *= halo_arr

    return M_BHs
