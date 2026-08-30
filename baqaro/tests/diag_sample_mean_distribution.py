"""Diagnostic: sampling distribution of the mean of a lognormal.

Small numerical check that the arithmetic mean of lognormal draws converges as
expected -- the property the tau -> 0 accretion limit relies on.

Script-style: executes on import. Run it directly, never under pytest.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from scipy.stats import norm

# Reproducibility
np.random.seed(0)

# Lognormal parameters (parameters of underlying normal)
mu = 0.0     # mean of ln X
sigma = 1.0  # std of ln X

# Sample sizes to illustrate
sample_sizes = [1, 5, 30, 100]
replicates = 10000  # number of Monte Carlo repetitions

# Theoretical mean of the lognormal
mean_X = exp(mu + sigma**2 / 2)

for N in sample_sizes:
    # Simulate replicates of sample means
    sample_means = np.mean(np.random.lognormal(mu, sigma, size=(replicates, N)), axis=1)
    
    # Theoretical variance of the sample mean
    var_X = (exp(sigma**2) - 1) * exp(2 * mu + sigma**2)
    var_mean = var_X / N
    std_mean = sqrt(var_mean)
    
    # Plotting
    fig, ax = plt.subplots()
    ax.hist(sample_means, bins=60, density=True, alpha=0.6, edgecolor='black')
    
    # Overlay normal approximation from CLT
    x_vals = np.linspace(sample_means.min(), sample_means.max(), 500)
    ax.plot(x_vals, norm.pdf(x_vals, mean_X, std_mean), linewidth=2)
    
    # Reference lines and labels
    ax.axvline(mean_X, linestyle='--', linewidth=1.5)
    ax.set_title(f"Distribution of Sample Mean (N = {N})")
    ax.set_xlabel(r"$\bar{{X}}_N$")
    ax.set_ylabel("Density")
    plt.show()