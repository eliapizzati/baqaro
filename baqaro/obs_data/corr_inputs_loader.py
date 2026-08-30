"""Repo-local loader for the cached correlation-model inputs.

Thin wrapper around :func:`qhtools.clustering.get_corr_inputs` that fixes the
data directory to the bundled artifacts (``obs_data/corr_inputs/``, produced by
``build_corr_inputs.py``). These functions are drop-in replacements for
the original clustering code's ``get_input_quantities`` (auto / cross) and
return the same 5-tuple ``(log_m_axis, rbins, out_bins, mf_fit, triangle_fit)``.

Regenerate the artifacts with::

    python -m baqaro.obs_data.build_corr_inputs
"""

import os

from qhtools.clustering import get_corr_inputs

DATA_DIR = os.path.join(os.path.dirname(__file__), "corr_inputs")


def get_input_quantities_auto(redshift, len_mbins=51, len_rbins=101,
                              log_M_min=11.5, log_M_max=14.5):
    """Auto-correlation inputs (projected wp/rp) for a cached config.

    Defaults match the committed artifacts, so ``get_input_quantities_auto(2.5)``
    and ``(4.0)`` work as-is. Any other combination raises FileNotFoundError
    listing what IS cached -- it is a lookup, not a computation.
    """
    return get_corr_inputs(DATA_DIR, "auto", redshift, len_mbins, len_rbins,
                           log_M_min, log_M_max)


def get_input_quantities_cross(redshift, len_mbins=51, len_rbins=101,
                               log_M_min=10.5, log_M_max=14.0):
    """Cross-correlation inputs (volume-averaged xi) for a cached config.

    Defaults match the committed z=6.1 artifact, so
    ``get_input_quantities_cross(6.1)`` works as-is. See
    :func:`get_input_quantities_auto` for the lookup semantics.
    """
    return get_corr_inputs(DATA_DIR, "cross", redshift, len_mbins, len_rbins,
                           log_M_min, log_M_max)
