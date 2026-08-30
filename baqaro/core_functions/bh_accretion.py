"""
BH accretion evolution — pure NumPy vectorized (no Numba).
==========================================================

The **reference engine**. ``bh_accretion_fast.py`` (fused Numba kernels,
~5-10x faster) is what the per-snapshot production driver
``main_evolution.py`` uses, and the two are kept in lock-step, with parity
regressions in ``tests/`` (``test_growth_const_units``,
``test_branch_c_parity``, ``test_branch_dispatch_table_floor``).

⚠ THIS FILE IS ACTIVE CODE — do not treat it as dead. Live importers:
- ``main_evolution_full_history.py`` → ``evolve_BHs_full_history``
  (the sub-step full-history engine has NO fast counterpart; this IS
  the production path for every full-history / lightcurve run)
- ``plotting_paper/`` (bhar, cosmic_bhmd,
  merger_fraction, radiative_efficiency) → ``get_madau_efficiency_epsilon``
- the engine regression tests above → ``evolve_BHs``

Physical model
--------------
Same as bh_accretion_fast.py (see its module docstring for full details):

Each BH grows over ``n_steps`` sub-timesteps. At each sub-step, an
Eddington ratio ``eta`` is drawn from a lognormal ERDF. The cumulative
growth determines the new mass:

    M_new = M_old * exp( sum_j [ eta_j * f(eps_j) * growth_const ] )

Two branches:

**Branch A** (n_steps < 6 or no transfer function):
    Direct sampling — draw (size, n_steps) standard normals, vectorize
    the entire computation with NumPy broadcasting.

**Branch B** (n_steps >= 6 with transfer function; the table floor is
N=5 summands, so the N = n_steps-1 history lookup needs n_steps >= 6):
    Transfer function path — sample the last sub-step directly, then
    draw the sum of (n_steps-1) lognormals from the precomputed table
    via ``transfer_function()`` closure call.

Key differences from bh_accretion_fast.py:
- Uses temporary arrays + NumPy vectorization instead of fused loops
- Parallelization via joblib ``array_split`` + ``concatenate`` (copies data)
- No pre-allocated buffer support
- No Numba dependency

Also contains ``evolve_BHs_full_history`` / ``process_evolution_full_history``
which return per-sub-step mass, luminosity, and eta arrays of shape
(n_halos, n_steps).
"""

import os
import time
import numpy as np
import numexpr as ne

from joblib import Parallel, delayed
from multiprocessing import cpu_count

import qhtools.utils.natconst as nc

# Madau effective-efficiency correction (the f_eff correction; see
# madau_feff.py). OFF by default so the reference engine
# stays bit-identical for existing callers/tests. NOTE: only the table path
# (process_evolution Branch B) carries the bias; process_evolution_full_history
# samples eps per sub-step and is EXACT, so the correction is a no-op there.
# (feff_per_halo is imported lazily inside process_evolution to avoid a circular
# import: madau_feff imports get_madau_efficiency_epsilon from this module.)
_MADAU_FEFF_ENV = (os.environ.get("BAQARO_MADAU_FEFF_CORRECTION", "0").strip().lower()
                   not in ("0", "", "false", "no", "off"))


# ==============================================================================
# HELPER: Madau+ spin-dependent radiative efficiency
# ==============================================================================

def get_madau_efficiency_epsilon(etas, rad_efficiency=0.1):
    """Compute radiative efficiency eps(eta) from the Madau+ (2014) fit.

    The spin-dependent radiative efficiency is a function of the
    Eddington ratio:

        eps(eta) = eps_base * A / eta * (0.985/(1.6/eta + B) + 0.015/(1.6/eta + C))

    where A, B, C are precomputed fitting constants.

    Parameters
    ----------
    etas : ndarray
        Eddington ratios (can be 1D or 2D).
    rad_efficiency : float
        Base radiative efficiency eps_base (default 0.1).

    Returns
    -------
    eps : ndarray
        Radiative efficiency, clamped to [0, 0.999].
    """
    A = 1.8260922439282448
    B = 0.7586284639465684
    C = 0.01611606486191978

    inv_etas = 1.0 / np.maximum(etas, 1e-20)

    term1 = 0.985 / (1.6 * inv_etas + B)
    term2 = 0.015 / (1.6 * inv_etas + C)

    eps = (rad_efficiency * A) * inv_etas * (term1 + term2)

    return np.clip(eps, 0.0, 0.999)


# ==============================================================================
# CORE: Single-snapshot evolution (one chunk of halos)
# ==============================================================================

def process_evolution(transfer_function, n_steps, M_BHs, erdf, growth_const,
                      rad_efficiency, size, rng, log_halo_rates_array,
                      rad_efficiency_model, print_times=False,
                      growth_max=50.0, madau_feff_correction=False):
    """Evolve a chunk of BH masses over one snapshot interval.

    This is the inner workhorse — called directly (single-process) or by
    individual joblib workers (parallel mode).

    Parameters
    ----------
    transfer_function : callable or None
        Sum-of-lognormals sampler closure from ``fast_lognormal_sampler``.
        If None, Branch A (direct sampling) is always used.
    n_steps : int
        Number of sub-timesteps.
    M_BHs : ndarray (float32)
        BH masses. Modified in-place.
    erdf : Erdf
        ERDF model providing ``.get_mu_sigma_arrays()``.
    growth_const : float
        Pre-computed growth constant per sub-step:
        ``(1-eps)/eps / c^2 * L_Edd_factor * dt_substep``.
    rad_efficiency : float
        Base radiative efficiency.
    size : int
        Number of halos (== len(M_BHs)).
    rng : numpy.random.Generator
        Random number generator for this chunk.
    log_halo_rates_array : ndarray or None
        log10(specific cold accretion rate) per halo.
    rad_efficiency_model : str
        "constant" or "madau+".
    print_times : bool
        Print timing information.

    Returns
    -------
    M_BHs : ndarray
        Updated BH masses (same array, modified in-place).
    L_bols : ndarray
        Bolometric luminosities at the last sub-step.
    """

    time_start = time.time()

    # --- ERDF parameters: per-halo mu and sigma ---
    # float32 to halve memory at 100M+ halos
    mus, sigmas = erdf.get_mu_sigma_arrays(log_halo_rates_array, dtype=np.float32)

    if np.ndim(sigmas) > 0:
        np.clip(sigmas, np.float32(0.1), np.float32(3.0), out=sigmas)
    # Scalar (fixed) sigma is intentionally left unclipped — see the matching
    # note in bh_accretion_fast.py: clipping the lower edge would break the
    # deterministic std_0=0 Salpeter invariant (tests/test_growth_const_units).

    LN_10 = np.float32(2.302585092994046)
    growth_const = np.float32(growth_const)

    growth_sum = np.zeros(size, dtype=np.float32)
    etas_last = np.zeros(size, dtype=np.float32)

    # ==================================================================
    # BRANCH C: Analytical mean growth (n_steps == 0)
    # ==================================================================
    # Zero coherence time: all draws are independent. The proper tau->0
    # analytical limit accumulates the ARITHMETIC mean
    #   E[eta] = 10^mu * exp(0.5 * (ln10)^2 * sigma^2),
    # not the geometric mean 10^mu. Using the median here instead makes this
    # engine disagree with the fast one by exp(0.5*ln(10)^2*sigma^2) ~ 2.2x in
    # eta at sigma=0.54, so the two are cross-checked at tau=0.
    # BAQARO_BRANCH_C_LEGACY=1 reverts BOTH engines to the geometric-mean
    # path, for bit-identical reproduction of runs made with that
    # convention. L_bol is computed from a single independent random
    # eta draw (unchanged).

    if n_steps == 0:
        use_arith_mean = os.environ.get("BAQARO_BRANCH_C_LEGACY", "0") != "1"
        eta_med = np.exp(mus * LN_10)
        if use_arith_mean:
            LN10_SQ_HALF = np.float32(2.6516504294495533)  # 0.5 * (ln 10)^2
            eta_growth = eta_med * np.exp(LN10_SQ_HALF * sigmas * sigmas)
        else:
            eta_growth = eta_med

        if rad_efficiency_model == "constant":
            growth_sum = eta_growth * growth_const
            eps_last = rad_efficiency

        elif rad_efficiency_model == "madau+":
            # eps evaluated at the MEDIAN eta — consistent with the fast
            # kernel and with the f_eff definition (f_eff multiplies the
            # exact (1 - eps(median)) prefactor; see madau_feff.py).
            eps_med = get_madau_efficiency_epsilon(eta_med, rad_efficiency)
            correction = (1.0 - eps_med) / (1.0 - rad_efficiency)
            growth_sum = eta_growth * correction * growth_const
            if madau_feff_correction:
                from .madau_feff import feff_per_halo
                sig_for_feff = sigmas if np.ndim(sigmas) else float(sigmas)
                growth_sum = growth_sum * feff_per_halo(
                    mus, sig_for_feff, rad_efficiency)

        else:
            raise ValueError(f"Unknown model: {rad_efficiency_model}")

        np.clip(growth_sum, None, growth_max, out=growth_sum)
        M_BHs *= np.exp(growth_sum)
        np.clip(M_BHs, None, 1e15, out=M_BHs)

        # Single random eta draw for L_bol
        Z_lbol = rng.standard_normal(size).astype(np.float32)
        etas_last = np.exp((mus + sigmas * Z_lbol) * LN_10)

        if rad_efficiency_model == "madau+":
            eps_last = get_madau_efficiency_epsilon(etas_last, rad_efficiency)
        else:
            eps_last = rad_efficiency

        log_csi = nc.log_csi
        factor = (eps_last / rad_efficiency) * (10**log_csi)
        L_bols = M_BHs * etas_last * factor
        np.clip(L_bols, None, 1e20, out=L_bols)

        if print_times:
            print(f"TIME (BRANCH C): {time.time() - time_start:.4f}s")

        return M_BHs, L_bols

    # ==================================================================
    # BRANCH A: Direct sub-step sampling (n_steps < 6)
    # ==================================================================
    # Draw all standard normals as a 2D array and vectorize with
    # NumPy broadcasting. Simple but allocates (size, n_steps) temp arrays.

    # n_steps <= 5 MUST use direct sampling: the transfer table's floor is
    # N=5 summands, so Branch B's N = n_steps-1 history lookup is only exact
    # for n_steps >= 6. Dispatching n_steps == 5 to Branch B makes the N=4
    # lookup clamp to the N=5 plane -> mean growth exponent 6/5 too large.
    if n_steps < 6 or transfer_function is None:

        # Z[i, j] ~ N(0, 1) for halo i, sub-step j
        Z = rng.standard_normal((size, n_steps)).astype(np.float32)

        # eta[i, j] = 10^(mu_i + sigma_i * Z[i,j])
        # sigmas is a scalar for fixed-sigma ERDF models; only column-ify
        # per-halo arrays (scalar * Z broadcasts on its own).
        sigmas_col = sigmas[:, None] if np.ndim(sigmas) > 0 else sigmas
        log_arg = (mus[:, None] + sigmas_col * Z) * LN_10
        etas = np.exp(log_arg)

        etas_last = etas[:, -1]

        if rad_efficiency_model == "constant":
            etas_sum = np.sum(etas, axis=1)
            growth_sum = etas_sum * growth_const
            eps_last = rad_efficiency

        elif rad_efficiency_model == "madau+":
            eps_all = get_madau_efficiency_epsilon(etas, rad_efficiency)
            correction = (1.0 - eps_all) / (1.0 - rad_efficiency)
            growth_sum = np.sum(etas * correction, axis=1) * growth_const
            eps_last = eps_all[:, -1]

        else:
             raise ValueError(f"Unknown model: {rad_efficiency_model}")

        if print_times:
            print(f"TIME (NAIVE N={n_steps}): {time.time() - time_start:.4f}s")

    # ==================================================================
    # BRANCH B: Transfer function path (n_steps >= 6)
    # ==================================================================
    # Sample only the last sub-step directly. The sum of the remaining
    # (n_steps-1) sub-steps is drawn from the precomputed 3D table
    # via the transfer_function() closure.

    else:
        # --- Last sub-step: sample directly ---
        Z_last = rng.standard_normal(size).astype(np.float32)
        etas_last = np.exp((mus + sigmas * Z_last) * LN_10)

        if rad_efficiency_model == "constant":
            eff_eta_last = etas_last
            eps_last = rad_efficiency

        elif rad_efficiency_model == "madau+":
            eps_last = get_madau_efficiency_epsilon(etas_last, rad_efficiency)
            correction = (1.0 - eps_last) / (1.0 - rad_efficiency)
            eff_eta_last = etas_last * correction

        growth_sum += (eff_eta_last * growth_const)

        # --- History (n_steps - 1): transfer function lookup ---
        # For Madau+, remap mu to effective mu (accounting for median
        # efficiency correction) before the table lookup.
        if rad_efficiency_model == "madau+":
            median_eta = np.power(10.0, mus)
            eps_med = get_madau_efficiency_epsilon(median_eta, rad_efficiency)
            correction_med = (1.0 - eps_med) / (1.0 - rad_efficiency)
            median_eff_eta = median_eta * correction_med
            # f_eff correction: scale the table-history mean by
            # f_eff(mu,sigma) so it matches exact per-sub-step direct sampling.
            # OFF by default -> bit-identical. Mirrors bh_accretion_fast.py.
            if madau_feff_correction:
                from .madau_feff import feff_per_halo
                sig_for_feff = sigmas if np.ndim(sigmas) else float(sigmas)
                median_eff_eta = median_eff_eta * feff_per_halo(
                    mus, sig_for_feff, rad_efficiency)
            mus_eff = np.log10(median_eff_eta)
            sigmas_eff = sigmas
        else:
            mus_eff = mus
            sigmas_eff = sigmas

        # Sub-step cap: the transfer table spans at most n_max_table
        # summands; cap the EFFECTIVE count so the looked-up summands match the
        # growth_const timestep (computed identically in evolve_BHs from the same
        # transfer_function + n_steps). Mirrors bh_accretion_fast.py.
        if hasattr(transfer_function, "_table_data"):
            n_max_table = int(round(
                10.0 ** float(transfer_function._table_data['n_log_vals'][-1])))
            n_eff = min(n_steps, n_max_table + 1)
        else:
            n_eff = n_steps

        # Call the sum-of-lognormals sampler closure. Use n_eff-1 (capped) so the
        # looked-up summand count matches the growth_const timestep.
        etas_history_val = transfer_function(
            n_arr=[n_eff - 1],
            mu_dex_arr=mus_eff,
            sigma_dex_arr=sigmas_eff,
            rng=rng
        )

        growth_sum += (etas_history_val * growth_const)

        if print_times:
            print(f"TIME (TRANSFER N={n_steps}): {time.time() - time_start:.4f}s")

    # ==================================================================
    # COMMON: Apply growth, compute luminosity
    # ==================================================================

    # Clamp growth to prevent overflow: exp(50) ~ 5e21 (still finite).
    # ``growth_max`` is overridable via BAQARO_GROWTH_SUM_MAX (see main_evolution).
    np.clip(growth_sum, None, growth_max, out=growth_sum)
    M_BHs *= np.exp(growth_sum)

    # Cap masses: 10^15 M_sun is far beyond any physical BH
    np.clip(M_BHs, None, 1e15, out=M_BHs)

    # Luminosity from last sub-step only (instantaneous):
    # L_bol = M_BH * eta_last * (eps_last / eps_base) * 10^log_csi
    log_csi = nc.log_csi
    factor = (eps_last / rad_efficiency) * (10**log_csi)
    L_bols = M_BHs * etas_last * factor

    np.clip(L_bols, None, 1e20, out=L_bols)

    if print_times:
        print(f"TOTAL TIME: {time.time() - time_start:.4f}s")

    return M_BHs, L_bols


# ==============================================================================
# PUBLIC API: evolve_BHs (single-snapshot, optional parallelization)
# ==============================================================================

def evolve_BHs(transfer_function, M_BHs, delta_t, erdf,
               rad_efficiency=0.1, n_steps=100,
               log_halo_rates_array=None, rad_efficiency_model="constant",
               parallelize=True, num_workers=None, backend="threading",
               print_info=False, growth_max=50.0, madau_feff_correction=None,
               **kwargs):
    """Evolve BH masses over one snapshot interval.

    Parameters
    ----------
    transfer_function : callable or None
        Sum-of-lognormals sampler from ``fast_lognormal_sampler.load_3d_sampler``.
    M_BHs : ndarray
        BH masses at previous snapshot.
    delta_t : float
        Snapshot time interval in Gyr.
    erdf : Erdf
        ERDF model instance.
    rad_efficiency : float
        Base radiative efficiency (default 0.1).
    n_steps : int
        Number of sub-timesteps (default 100).
    log_halo_rates_array : ndarray or None
        log10(specific cold accretion rate) per halo.
    rad_efficiency_model : str
        "constant" or "madau+".
    parallelize : bool
        Use joblib threading (default True).
    num_workers : int or None
        Number of workers (default: cpu_count).
    backend : str
        joblib backend.
    print_info : bool
        Verbose output.
    **kwargs
        Ignored (for compatibility with bh_accretion_fast interface).

    Returns
    -------
    M_BHs : ndarray
        Updated BH masses.
    L_bols : ndarray
        Bolometric luminosities at the last sub-step.
    """

    if rad_efficiency_model not in ["constant", "madau+"]:
        raise ValueError("rad_efficiency_model must be either 'constant' or 'madau+'")

    if log_halo_rates_array is not None and len(log_halo_rates_array) != len(M_BHs):
        raise ValueError("The number of halo masses must be equal to the number of BHs")

    if print_info:
        print("ENTERING EVOLUTION ROUTINE")
        print("Evolving black hole mass array size in GB" , M_BHs.nbytes / 1e9)

    L_bols = np.zeros(len(M_BHs))
    # Branch C (n_steps == 0): growth_const is for the full delta_t interval
    # (no sub-stepping, analytical mean growth). Avoid divide-by-zero.
    #
    # Sub-step cap (mirrors bh_accretion_fast.py): the
    # transfer-function table spans at most n_max_table summands. For
    # n_steps-1 beyond that the history-sum lookup SATURATES, so we cap the
    # EFFECTIVE sub-step count for BOTH the table lookup (n_arr below) AND the
    # growth_const timestep, keeping mean growth n_steps-independent.
    if n_steps == 0:
        n_eff = 0
        timestep = delta_t
    else:
        if transfer_function is not None and n_steps >= 6 \
                and hasattr(transfer_function, "_table_data"):
            n_max_table = int(round(
                10.0 ** float(transfer_function._table_data['n_log_vals'][-1])))
            n_eff = min(n_steps, n_max_table + 1)
        else:
            n_eff = n_steps
        timestep = delta_t / n_eff
    total_number_of_evolving_objects = len(M_BHs)

    if len(M_BHs) == 0:
        if print_info:
            print("No black holes to evolve, returning empty arrays")
        return np.array([]), np.array([])

    size = total_number_of_evolving_objects
    if print_info:
        print("COMPUTING SIZE", size)

    # Growth constant: dimensionless growth per sub-step at eta=1 (or total for Branch C)
    # growth = (1-eps)/eps / c^2 * 10^log_csi * (ls/ms) * dt_seconds
    # nc.ls/nc.ms converts the Eddington factor 10^log_csi (Lsun/Msun, same units
    # as L_bol) into a cgs mass-growth rate (Lsun->erg/s, g->Msun). MUST stay
    # consistent with the L_bol formula `M * eta * (eps/eps_base) * 10^log_csi`.
    growth_const = (1 - rad_efficiency) / rad_efficiency / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms * timestep * nc.year * 1e9
    if print_info:
        print("GROWTH CONSTANT (growth assuming eta=1)", growth_const, "* nsteps =", growth_const * n_steps)
        print("np.exp(growth_const * n_steps)", np.exp(growth_const * n_steps))

    # Resolve the madau f_eff correction toggle: explicit arg wins, else the
    # BAQARO_MADAU_FEFF_CORRECTION env default (OFF -> bit-identical).
    feff_on = _MADAU_FEFF_ENV if madau_feff_correction is None else bool(madau_feff_correction)

    time_here = time.time()
    if parallelize:
        # --- Joblib parallelization via array_split + concatenate ---
        # Each worker gets a copy of its chunk (not a view — this is slower
        # than the slice-view approach in bh_accretion_fast.py).
        num_workers = cpu_count() if num_workers is None else num_workers
        num_workers = min(num_workers, len(M_BHs))
        MBHs_chunks = np.array_split(M_BHs, num_workers)
        if print_info:
            print("PARALLELIZING WITH", num_workers, "WORKERS")
            print("MBHs_chunks AV SIZE", size/num_workers)

        if log_halo_rates_array is not None:
            log_halo_rates_chunks = np.array_split(log_halo_rates_array, num_workers)
        else:
            log_halo_rates_chunks = [None] * num_workers

        size_chunks = [len(chunk) for chunk in MBHs_chunks]

        # Each worker needs its own RNG: per-worker RNGs via SeedSequence.spawn
        # (provably independent streams). The fast engine draws centrally, not
        # per worker.
        rngs_local = erdf.rng.spawn(num_workers)

        results = Parallel(n_jobs=num_workers, backend=backend)(
            delayed(process_evolution)(transfer_function, n_steps, M_BH_chunk, erdf, growth_const, rad_efficiency, size_chunk, rng_local, log_halo_rates_chunk, rad_efficiency_model, growth_max=growth_max, madau_feff_correction=feff_on)
            for M_BH_chunk, size_chunk, rng_local, log_halo_rates_chunk in zip(MBHs_chunks, size_chunks, rngs_local, log_halo_rates_chunks)
        )
        M_BHs_list, L_bols_list = zip(*results)

        M_BHs = np.concatenate(M_BHs_list)
        L_bols = np.concatenate(L_bols_list)

    else:
        M_BHs, L_bols = process_evolution(transfer_function, n_steps, M_BHs, erdf, growth_const, rad_efficiency, size, erdf.rng, log_halo_rates_array, rad_efficiency_model, growth_max=growth_max, madau_feff_correction=feff_on)

    if print_info:
        print("TIME TO EVOLVE BHs", time.time() - time_here)

    return M_BHs, L_bols


# ==============================================================================
# FULL HISTORY VARIANT: returns per-sub-step mass, luminosity, and eta
# ==============================================================================

def process_evolution_full_history(transfer_function, n_steps, M_BHs, erdf,
                                   growth_const, rad_efficiency, size, rng,
                                   log_halo_rates_array, rad_efficiency_model,
                                   print_times=False, growth_max=50.0):
    """Evolve a chunk of BHs and return full per-sub-step history.

    Unlike ``process_evolution`` which only returns the final mass and
    luminosity, this function returns arrays of shape (n_halos, n_steps)
    tracking the evolution at every sub-step.

    Always uses Branch A (direct sampling for all sub-steps) — the
    transfer function is NOT used here because we need individual
    per-step values, not just the sum.

    Parameters
    ----------
    (Same as process_evolution)

    Returns
    -------
    M_BHs_history : ndarray, shape (size, n_steps)
        BH masses at each sub-step.
    L_bols : ndarray, shape (size, n_steps)
        Luminosities at each sub-step.
    etas : ndarray, shape (size, n_steps)
        Eddington ratios at each sub-step.
    """

    time_start = time.time()

    # --- ERDF parameters ---
    mus, sigmas = erdf.get_mu_sigma_arrays(log_halo_rates_array)

    # mus is always 1-D (one per halo).  sigmas can be scalar (fixed-sigma
    # model like log_normal_evol_halo_mass) or 1-D (variable-sigma model).
    # Ensure both are 1-D so broadcasting with the 2-D Z array works.
    mus = np.atleast_1d(mus)
    sigmas = np.atleast_1d(sigmas)
    np.clip(sigmas, 0.1, 3.0, out=sigmas)

    LN_10 = 2.302585092994046

    # --- Sample all sub-steps at once ---
    # Z[i, j] ~ N(0,1), shape (size, n_steps)
    Z = rng.standard_normal((size, n_steps))

    # eta_raw[i, j] = 10^(mu_i + sigma_i * Z[i,j])   — the unconditional
    # ERDF draws, BEFORE any cap is applied.
    log_arg = (mus[:, None] + sigmas[:, None] * Z) * LN_10
    etas_raw = np.exp(log_arg)

    # --- Per-substep growth contribution (uncapped) ---
    if rad_efficiency_model == "constant":
        # eps == rad_efficiency everywhere -> correction == 1
        correction = np.ones_like(etas_raw)
        delta_g = etas_raw * growth_const
    elif rad_efficiency_model == "madau+":
        eps_raw = get_madau_efficiency_epsilon(etas_raw, rad_efficiency)
        correction = (1.0 - eps_raw) / (1.0 - rad_efficiency)
        delta_g = etas_raw * correction * growth_const
    else:
        raise ValueError(f"Unknown model: {rad_efficiency_model}")

    if print_times:
        print(f"TIME (NAIVE N={n_steps}): {time.time() - time_start:.4f}s")

    # === SELF-CONSISTENT PER-SUB-STEP CAP (Option B) ==========================
    # Per-snapshot integrated-growth cap. Default growth_max=50.0 leaves the
    # legacy behaviour intact (exp(50) ~ 5e21 is a finite-but-astronomical
    # guard); lowering it via BAQARO_GROWTH_SUM_MAX (e.g. =4.6 -> max 100x
    # growth per snap) progressively freezes the per-sub-step lightcurve.
    #
    # The implementation is SELF-CONSISTENT: when the cumulative growth hits
    # the cap, the per-sub-step EFFECTIVE Eddington ratio (eta_acc_eff) drops
    # to whatever "remaining growth budget" was left, then to zero for the
    # rest of the snapshot. L_bol uses eta_acc_eff (not the raw lognormal
    # draw), and we recompute the Madau radiative efficiency at the effective
    # eta. The net result:
    #   * M_BH plateaus at M_init * exp(growth_max)
    #   * eta_acc_eff = 0 for sub-steps past the plateau onset
    #   * L_bol = 0 during the plateau (BH stops accreting -> stops radiating)
    #   * lambda_Edd = L_bol / L_Edd(M_BH) ≡ eta_acc_eff (consistency invariant)
    # This matches the physical interpretation that feedback (or gas depletion)
    # cuts off BOTH growth and radiation at the cap.
    #
    # NOTE: the FAST engine in bh_accretion_fast.py is Option A — its L_bol
    # is computed from a single final-substep raw eta and the plateaued
    # M_BH, so L_bol there stays bright during a plateau. The two engines
    # are population-statistic-consistent for default growth_max=50 (the cap
    # almost never fires); for tight caps they diverge slightly on the bright
    # tail because Option B (here) zeroes L_bol on plateaus. Lightcurves
    # MUST use this engine to get Soltan-consistent (M, L) histories; the
    # fast path is for population-only forward runs.
    growth_cumsum_uncapped = np.cumsum(delta_g, axis=1)
    growth_cumsum = np.minimum(growth_cumsum_uncapped, growth_max)

    # Effective per-sub-step growth: diff of the CAPPED cumulative growth.
    # `prepend=0.0` makes the first substep's diff equal to its own delta.
    delta_g_eff = np.diff(growth_cumsum, axis=1, prepend=0.0)

    # Reconstruct the EFFECTIVE eta that "lands" as accretion at each step:
    #   delta_g_eff = eta_acc_eff * correction * growth_const  -> solve
    # Strictly, correction depends on eta — but at default 50.0 the cap never
    # triggers and eta_acc_eff == etas_raw to floating-point precision; for
    # the tight-cap regime, the residual self-consistency error from using the
    # raw `correction` is < 0.1% in any practical regime (Madau eps varies
    # slowly with eta).
    with np.errstate(divide="ignore", invalid="ignore"):
        etas_eff = np.where(
            correction > 0,
            delta_g_eff / (correction * growth_const),
            0.0,
        )

    # Recompute eps at the effective eta so L_bol's (eps/eps_base) factor is
    # consistent with the actually-accreted rate. For the "constant" model
    # this is a no-op (eps == rad_efficiency).
    if rad_efficiency_model == "madau+":
        eps_eff = get_madau_efficiency_epsilon(etas_eff, rad_efficiency)
    else:
        eps_eff = np.full_like(etas_eff, rad_efficiency)

    # --- Mass history (uses capped cumulative growth) ---
    M_BHs_history = M_BHs[:, None] * np.exp(growth_cumsum)
    np.clip(M_BHs_history, None, 1e15, out=M_BHs_history)

    # --- Luminosity history (uses EFFECTIVE eta + EFFECTIVE eps) ---
    # L_bol[i, j] = M_BH[i, j] * eta_acc_eff[i, j] * (eps_eff / eps_base) * 10^logξ
    log_csi = nc.log_csi
    factor = (eps_eff / rad_efficiency) * (10**log_csi)
    L_bols = M_BHs_history * etas_eff * factor
    np.clip(L_bols, None, 1e20, out=L_bols)

    if print_times:
        print(f"TOTAL TIME: {time.time() - time_start:.4f}s")

    # Return the EFFECTIVE eta (what physically drove the accretion). At
    # default growth_max=50.0 this is identical to the raw ERDF draws.
    return M_BHs_history, L_bols, etas_eff


def evolve_BHs_full_history(transfer_function, M_BHs, delta_t, erdf,
                            rad_efficiency=0.1, n_steps=100,
                            log_halo_rates_array=None,
                            rad_efficiency_model="constant",
                            parallelize=True, num_workers=None,
                            backend="threading", print_info=False,
                            growth_max=50.0, madau_feff_correction=None):
    """Evolve BHs and return full per-sub-step history arrays.

    ``madau_feff_correction`` is accepted for API parity with ``evolve_BHs`` but
    is a **physical no-op here**: the full-history engine
    (``process_evolution_full_history``) samples the madau eps(eta) at EVERY
    sub-step (direct sampling), so it is already exact and never carried the
    transfer-table median-eps bias that the f_eff correction removes. In other words, the
    full-history output equals the *corrected* (feff-on) fast-path model
    regardless of this flag. It is threaded through only so callers / provenance
    can record intent consistently. See core_functions/madau_feff.py.

    Parameters
    ----------
    (Same as evolve_BHs)

    Returns
    -------
    M_BHs_history : ndarray, shape (n_halos, n_steps)
        BH masses at each sub-step.
    L_bols : ndarray, shape (n_halos, n_steps)
        Luminosities at each sub-step.
    etas : ndarray, shape (n_halos, n_steps)
        Eddington ratios at each sub-step.
    """

    if rad_efficiency_model not in ["constant", "madau+"]:
        raise ValueError("rad_efficiency_model must be either 'constant' or 'madau+'")

    if log_halo_rates_array is not None and len(log_halo_rates_array) != len(M_BHs):
        raise ValueError("The number of halo masses must be equal to the number of BHs")

    if print_info:
        print("ENTERING EVOLUTION ROUTINE")
        print("Evolving black hole mass array size in GB" , M_BHs.nbytes / 1e9)

    L_bols = np.zeros(len(M_BHs))
    timestep = delta_t / n_steps
    total_number_of_evolving_objects = len(M_BHs)

    if len(M_BHs) == 0:
        if print_info:
            print("No black holes to evolve, returning empty arrays")
        return np.array([]), np.array([]), np.array([])

    size = total_number_of_evolving_objects
    if print_info:
        print("COMPUTING SIZE", size)

    # Growth constant per sub-step. * nc.ls / nc.ms: Lsun->erg/s, g->Msun unit
    # conversion so mass build-up is consistent with L_bol (see process_evolution).
    # Direct-sampling path (no transfer table) -> no sub-step cap needed here.
    growth_const = (1 - rad_efficiency) / rad_efficiency / nc.cc**2 * 10**nc.log_csi * nc.ls / nc.ms * timestep * nc.year * 1e9
    if print_info:
        print("GROWTH CONSTANT (growth assuming eta=1)", growth_const, "* nsteps =", growth_const * n_steps)
        print("np.exp(growth_const * n_steps)", np.exp(growth_const * n_steps))

    time_here = time.time()
    if parallelize:
        num_workers = cpu_count() if num_workers is None else num_workers
        num_workers = min(num_workers, len(M_BHs))
        MBHs_chunks = np.array_split(M_BHs, num_workers)
        if print_info:
            print("PARALLELIZING WITH", num_workers, "WORKERS")
            print("MBHs_chunks AV SIZE", size/num_workers)

        if log_halo_rates_array is not None:
            log_halo_rates_chunks = np.array_split(log_halo_rates_array, num_workers)
        else:
            log_halo_rates_chunks = [None] * num_workers

        size_chunks = [len(chunk) for chunk in MBHs_chunks]

        # Per-worker RNGs via SeedSequence.spawn (provably independent
        # streams). The fast engine draws centrally, not per worker.
        rngs_local = erdf.rng.spawn(num_workers)

        results = Parallel(n_jobs=num_workers, backend=backend)(
            delayed(process_evolution_full_history)(transfer_function, n_steps, M_BH_chunk, erdf, growth_const, rad_efficiency, size_chunk, rng_local, log_halo_rates_chunk, rad_efficiency_model, growth_max=growth_max)
            for M_BH_chunk, size_chunk, rng_local, log_halo_rates_chunk in zip(MBHs_chunks, size_chunks, rngs_local, log_halo_rates_chunks)
        )
        M_BHs_list, L_bols_list, etas_list = zip(*results)

        M_BHs = np.concatenate(M_BHs_list)
        L_bols = np.concatenate(L_bols_list)
        etas = np.concatenate(etas_list)

    else:
        M_BHs, L_bols, etas = process_evolution_full_history(transfer_function, n_steps, M_BHs, erdf, growth_const, rad_efficiency, size, erdf.rng, log_halo_rates_array, rad_efficiency_model, growth_max=growth_max)

    if print_info:
        print("TIME TO EVOLVE BHs", time.time() - time_here)

    return M_BHs, L_bols, etas
