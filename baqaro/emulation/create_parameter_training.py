"""Sample the 6-D parameter space that the training set spans.

The emulator is only as good as the coverage of the points it is trained on, so
the design matters: these build low-discrepancy samples (Latin hypercube or
Sobol) rather than a random draw, which would leave gaps and clumps at the few-
thousand-point budget a training run can afford.

``create_emulation_parameters_sobol`` is what production uses. Sobol sequences
are *extensible*: skipping the first ``n_skip`` points and taking the next block
appends new points that stay well-distributed against the ones already run, so a
training set can be grown without discarding it and starting over.
"""

import numpy as np
from scipy.stats import qmc


def create_emulation_parameters_lhs(num_simulations, num_lhs_parameters, parameter_definitions, rng=None):
    """
    Create Latin Hypercube Sampling (LHS) parameters for emulation.

    Parameters:
    num_simulations : int
        The number of simulations to generate parameters for.
    num_lhs_parameters : int
        The number of parameters to sample.
    parameter_definitions : dict
        A dictionary defining the parameter names and their ranges as [lower_bound, upper_bound].
    rng : np.random.Generator, optional
        A random number generator instance for reproducibility. If None, a default RNG is used.

    Returns:
    lhs_params_rescaled : np.ndarray
        A 2D array of shape (num_simulations, num_lhs_parameters) containing the sampled parameters.
    param_names_lhs : list
        A list of parameter names corresponding to the columns in `lhs_params_rescaled`.
    """
    if rng is None:
        rng = np.random.default_rng(42379)

    param_names_lhs = list(parameter_definitions.keys())

    # Latin Hypercube Sampling for 4D
    # This creates samples in the range [0, 1] for each parameter dimension
    # samples_lhs_normalized = lhs(num_lhs_parameters, samples=num_simulations, criterion='maximin')
    sampler = qmc.LatinHypercube(d=num_lhs_parameters, rng=rng)
    samples_lhs_normalized = sampler.random(n=num_simulations)

    parameter_ranges_lhs = np.array([parameter_definitions[name] for name in param_names_lhs])

    # Rescale samples from [0, 1] to their defined parameter ranges (vectorized)
    lower_bounds = parameter_ranges_lhs[:, 0]
    upper_bounds = parameter_ranges_lhs[:, 1]
    lhs_params_rescaled = lower_bounds + (upper_bounds - lower_bounds) * samples_lhs_normalized

    # 'lhs_params_rescaled' is now a (num_samples x num_lhs_parameters) array
    # where each row is a set of 4 parameters for one run.

    print(f"Generated {num_lhs_parameters}D LHS parameters (shape: {lhs_params_rescaled.shape}):")
    for i, name in enumerate(param_names_lhs):
        print(f"  Parameter '{name}': min={np.min(lhs_params_rescaled[:, i]):.2f}, max={np.max(lhs_params_rescaled[:, i]):.2f}")
    # print(lhs_params_rescaled)

    return lhs_params_rescaled, param_names_lhs



def create_emulation_parameters_sobol(num_simulations, num_sobol_parameters, parameter_definitions, rng=None, n_skip=0):
    """
    Create Sobol sequence parameters for emulation.

    Parameters:
    num_simulations : int
        The number of NEW simulations to generate parameters for.
    num_sobol_parameters : int
        The number of parameters to sample.
    parameter_definitions : dict
        A dictionary defining the parameter names and their ranges as [lower_bound, upper_bound].
    rng : np.random.Generator, optional
        A random number generator instance for reproducibility. If None, a default RNG is used.
    n_skip : int, optional
        Number of initial Sobol points to skip. Use this to continue a previous sequence.
        For example, if you already have 128 points, set n_skip=128 to get the next batch.

    Returns:
    sobol_params_rescaled : np.ndarray
        A 2D array of shape (num_simulations, num_sobol_parameters) containing the sampled parameters.
    param_names_sobol : list
        A list of parameter names corresponding to the columns in `sobol_params_rescaled`.
    """
    if rng is None:
        rng = np.random.default_rng(42379)

    param_names_sobol = list(parameter_definitions.keys())

    # Sobol sampling in [0, 1]
    sampler = qmc.Sobol(d=num_sobol_parameters, scramble=True, seed=rng)
    
    # Generate total points needed (existing + new), then take only the new ones
    total_samples_needed = n_skip + num_simulations
    samples_sobol_normalized = sampler.random(n=total_samples_needed)
    
    # Skip the first n_skip points (already generated previously)
    if n_skip > 0:
        samples_sobol_normalized = samples_sobol_normalized[n_skip:]
        print(f"Skipped first {n_skip} Sobol points, generating points [{n_skip}:{total_samples_needed}]")
    
    parameter_ranges_sobol = np.array([parameter_definitions[name] for name in param_names_sobol])

    # Rescale samples from [0, 1] to their defined parameter ranges (vectorized)
    lower_bounds = parameter_ranges_sobol[:, 0]
    upper_bounds = parameter_ranges_sobol[:, 1]
    sobol_params_rescaled = lower_bounds + (upper_bounds - lower_bounds) * samples_sobol_normalized

    print(f"Generated {num_sobol_parameters}D Sobol parameters (shape: {sobol_params_rescaled.shape}):")
    for i, name in enumerate(param_names_sobol):
        print(f"  Parameter '{name}': min={np.min(sobol_params_rescaled[:, i]):.2f}, max={np.max(sobol_params_rescaled[:, i]):.2f}")

    return sobol_params_rescaled, param_names_sobol



def sample_in_box(num_simulations, num_sobol_parameters, parameter_definitions_local, rng=None):
    """
    Local refinement in a sub-box. Make sure parameter_definitions_local are within global bounds.
    """
    return create_emulation_parameters_sobol(num_simulations, num_sobol_parameters, parameter_definitions_local, rng=rng)