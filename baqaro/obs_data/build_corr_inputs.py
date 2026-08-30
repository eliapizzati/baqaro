"""Regenerate the cached correlation-model inputs. **Normally not needed.**

The clustering likelihood needs an HMF fit and a halo-model triangle for each
redshift it evaluates. Computing those requires the ``swift_qso_model`` package
and the raw FLAMINGO simulation data, so instead they are precomputed once and
**committed to this repository** as small ``.npz`` artifacts under
``obs_data/corr_inputs/`` — one per production config, ~2 MB each:

    corr_inputs_auto_z2.5_m51_r101_logM11.5_14.5.npz
    corr_inputs_auto_z4.0_m51_r101_logM11.5_14.5.npz
    corr_inputs_cross_z6.1_m51_r101_logM10.5_14.0.npz

Everything downstream reads those files through
``obs_data/corr_inputs_loader.py``, which needs nothing but numpy. So running
the clustering likelihood, or reproducing any clustering figure, does **not**
require this script, ``swift_qso_model``, or the simulation data.

Run it only to add a NEW config to ``CONFIGS`` below, or to refresh the cache
against updated simulation data — and only on a machine that has both of the
above:

    python -m baqaro.obs_data.build_corr_inputs

Each config takes seconds to ~10 s (the cross-correlation does colossus
halo-bias loops).
"""

import os

import numpy as np

from qhtools.clustering.corr_inputs import corr_inputs_filename

OUT_DIR = os.path.join(os.path.dirname(__file__), "corr_inputs")

# Production configs: (kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)
#
# These mirror exactly what the clustering consumers request
# (likelihoods_and_priors via main_mcmc, the reference implementation, both plotting_clustering) —
# and every one of them is ALREADY BUILT and committed under corr_inputs/, so
# this list is here to document what exists, not as work to be done. A config
# added here has no cached artifact until the script is rerun.
CONFIGS = [
    ("auto", 2.5, 51, 101, 11.5, 14.5),
    ("auto", 4.0, 51, 101, 11.5, 14.5),
    ("cross", 6.1, 51, 101, 10.5, 14.0),
]


def build_one(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max):
    """Compute and cache one config's correlation inputs as an .npz.

    Requires ``swift_qso_model`` and the raw simulation data, which is why the
    import is deferred to here rather than module scope.
    """
    # Deferred import: ONLY the producer needs swift_qso_model (+ the raw sim
    # data). Keeping it out of module scope means merely importing
    # this module — or having it on the path — does not require swift_qso_model.
    from swift_qso_model.core_functions.final_quantities import (
        get_input_quantities as get_input_quantities_auto,
    )
    from swift_qso_model.big_runs.final_quantities import (
        get_input_quantities as get_input_quantities_cross,
    )

    fn = get_input_quantities_auto if kind == "auto" else get_input_quantities_cross
    log_m_axis, rbins, out_bins, mf_fit, triangle_fit = fn(
        redshift=redshift, len_mbins=len_mbins, len_rbins=len_rbins,
        log_M_min=log_M_min, log_M_max=log_M_max,
    )
    path = os.path.join(
        OUT_DIR, corr_inputs_filename(kind, redshift, len_mbins, len_rbins, log_M_min, log_M_max)
    )
    np.savez(
        path,
        log_m_axis=log_m_axis,
        rbins=rbins,
        out_bins=out_bins,
        mf_fit=mf_fit,
        triangle_fit=triangle_fit,
        # provenance metadata (ignored by the loader)
        kind=kind,
        redshift=redshift,
        len_mbins=len_mbins,
        len_rbins=len_rbins,
        log_M_min=log_M_min,
        log_M_max=log_M_max,
    )
    size_mb = os.path.getsize(path) / 1e6
    print(f"  wrote {os.path.basename(path)}  triangle={triangle_fit.shape}  ({size_mb:.2f} MB)")


def main():
    """Rebuild every config in ``CONFIGS`` into ``obs_data/corr_inputs/``."""
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Building correlation inputs into {OUT_DIR}")
    for cfg in CONFIGS:
        build_one(*cfg)
    print("Done.")


if __name__ == "__main__":
    main()
