"""
Eddington Ratio Distribution Function (ERDF).
==============================================

The ERDF defines the probability distribution of Eddington ratios (eta)
for accreting black holes, parameterized by the halo's cold gas specific
accretion rate.

For a halo with log-rate ``r = log10(sSAR_cold)``, the ERDF gives:

    log10(eta) ~ Normal(mu(r), sigma(r))

where:

    mu(r) = log_eta_mean_0 + log_eta_mean_evol * r
    sigma(r) = std_0                                  (fixed-sigma model)
    sigma(r) = std_0 + std_evol * r                   (variable-sigma model)

Active models
-------------
- ``log_normal_evol_halo_mass`` (3 params): fixed sigma across all halos.
  Returns scalar sigma, triggering the fast B-1D kernel path in
  ``bh_accretion_fast.py`` (2 table reads per halo instead of 8).
- ``log_normal_evol_halo_mass_dd`` (4 params): sigma varies with halo rate.
  Returns array sigma, uses the B-2D kernel path (8 table reads per halo).

Legacy models (defined in ``get_names`` but not used by active code):
- ``constant``, ``uniform``, ``log_normal``: no halo-rate dependence
- ``schechter*``, ``broken_power_law*``: alternative functional forms

Usage
-----
::

    erdf = Erdf("log_normal_evol_halo_mass", {
        "log_eta_mean_0": -1.0,
        "log_eta_mean_evol": 0.8,
        "std_0": 0.5,
    }, rng=rng)

    mus, sigmas = erdf.get_mu_sigma_arrays(log_halo_rates, dtype=np.float32)
"""

import numpy as np
from numpy.random import default_rng


class Erdf:
    """Eddington Ratio Distribution Function.

    Stores the ERDF model type and parameters, and provides methods to
    compute per-halo mu and sigma arrays for the accretion evolution kernels.

    Attributes
    ----------
    model : str
        ERDF model name (e.g. ``"log_normal_evol_halo_mass"``).
    params : dict
        Model parameters keyed by name (e.g. ``{"log_eta_mean_0": -1.0, ...}``).
    rng : numpy.random.Generator
        Random number generator used for all stochastic sampling in the
        accretion module. Shared with ``bh_accretion_fast.py`` via ``erdf.rng``.
    """

    def __init__(self, model, params_dict, duty_cycle=None, rng=None):
        """Initialize the ERDF.

        Parameters
        ----------
        model : str
            ERDF model name. Active models:
            ``"log_normal_evol_halo_mass"`` (3 params, fixed sigma),
            ``"log_normal_evol_halo_mass_dd"`` (4 params, variable sigma).
        params_dict : dict
            Parameter values keyed by name. Keys must match ``get_names()``.
        duty_cycle : float or None, optional
            Reserved for future use. Not currently implemented.
        rng : numpy.random.Generator or None, optional
            Random number generator. If None, a default generator with
            fixed seed 483295 is created (reproducible but not configurable).
        """
        self.model = model
        self.params = params_dict
        self.rng = rng if rng is not None else default_rng(483295)

    def get_names(self):
        """Return the parameter names for the current ERDF model.

        Returns
        -------
        names : list of str
            Ordered parameter names matching the dict keys expected
            by ``params_dict``.
        """
        # --- Lognormal models (active) ---
        if self.model == "log_normal_evol_halo_mass":
            return ["log_eta_mean_0", "log_eta_mean_evol", "std_0"]

        elif self.model == "log_normal_evol_halo_mass_dd":
            return ["log_eta_mean_0", "log_eta_mean_evol", "std_0", "std_evol"]

        # --- Base models (legacy) ---
        elif self.model == "constant":
            return ["log_eta"]

        elif self.model == "uniform":
            return ["log_eta_mean", "std"]

        elif self.model == "log_normal":
            return ["log_eta_mean", "std"]

        # --- Schechter models (legacy) ---
        elif self.model == "schechter":
            return ["log_eta_min", "log_eta_break", "alpha"]

        elif self.model == "schechter_evol_halo_mass":
            return ["log_eta_min", "log_eta_break", "alpha_0", "alpha_evol"]

        elif self.model == "schechter_evol_halo_mass_dd":
            return ["log_eta_min", "log_eta_break", "log_eta_evol", "alpha_0", "alpha_evol"]

        # --- Broken power law models (legacy) ---
        elif self.model == "broken_power_law":
            return ["log_eta_min", "log_eta_break", "log_eta_max", "alpha", "beta"]

        elif self.model == "broken_power_law_evol_halo_mass":
            return ["log_eta_min", "log_eta_break", "log_eta_max", "alpha_0", "beta", "alpha_evol"]

        else:
            raise ValueError(f"Model not implemented: {self.model}")

    def get_mu_sigma_arrays(self, log_halo_rates_array, dtype=np.float32):
        """Compute per-halo ERDF mean (mu) and scatter (sigma) arrays.

        These are passed to the accretion kernels in ``bh_accretion_fast.py``
        to draw Eddington ratios: ``log10(eta) ~ Normal(mu, sigma)``.

        Parameters
        ----------
        log_halo_rates_array : array_like
            log10(specific cold accretion rate) for each halo. Shape (n_halos,).
        dtype : numpy dtype, optional
            Output dtype (default float32). Using float32 halves memory
            at ~200M halos with negligible precision loss.

        Returns
        -------
        mus : ndarray, shape (n_halos,)
            Per-halo mean of log10(eta).
            ``mu_i = log_eta_mean_0 + log_eta_mean_evol * log_rate_i``
        sigmas : scalar or ndarray
            Scatter of log10(eta) in dex.
            - ``log_normal_evol_halo_mass``: returns a **scalar** (float).
              This signals to ``bh_accretion_fast.py`` that the B-1D fast
              path can be used (pre-interpolate sigma dimension once).
            - ``log_normal_evol_halo_mass_dd``: returns an **array** of
              shape (n_halos,). The B-2D path with full bilinear
              interpolation is used.

        Raises
        ------
        ValueError
            If the current model doesn't support this method.
        """
        log_halo_rates_array = np.asarray(log_halo_rates_array, dtype=dtype)

        # Unpack by KEY, not by dict insertion order: a reordered params_dict
        # literal (or a differently-ordered h5py attrs read) would otherwise
        # silently swap mu/sigma across the whole forward model. Keys match
        # get_names() for each model.
        if self.model == "log_normal_evol_halo_mass":
            mean0 = self.params["log_eta_mean_0"]
            mean_evol = self.params["log_eta_mean_evol"]
            std0 = self.params["std_0"]
            mus = dtype(mean0) + dtype(mean_evol) * log_halo_rates_array
            # A zero halo rate arrives as log10(0) = -inf and must give
            # eta = 0 (mu = -inf) for ANY slope. With mean_evol == 0 the
            # product 0 * (-inf) is NaN, which the growth cap cannot catch
            # and which mergers then spread through np.add.at.
            mus = np.where(np.isneginf(log_halo_rates_array), dtype(-np.inf), mus)
            sigmas = dtype(std0)  # scalar → triggers B-1D fast path

        elif self.model == "log_normal_evol_halo_mass_dd":
            mean0 = self.params["log_eta_mean_0"]
            mean_evol = self.params["log_eta_mean_evol"]
            std0 = self.params["std_0"]
            std_evol = self.params["std_evol"]
            mus = dtype(mean0) + dtype(mean_evol) * log_halo_rates_array
            sigmas = dtype(std0) + dtype(std_evol) * log_halo_rates_array
            # Same zero-rate guard as above; a -inf sigma is meaningless.
            zero_rate = np.isneginf(log_halo_rates_array)
            mus = np.where(zero_rate, dtype(-np.inf), mus)
            sigmas = np.where(zero_rate, dtype(std0), sigmas)

        else:
            raise ValueError(f"get_mu_sigma_arrays not implemented for model: {self.model}")

        return mus, sigmas
