"""plotting_common: shared plotting infrastructure.

The pieces every figure-producing module needs, regardless of whether the
figure is a working diagnostic or one that appears in the paper. Kept separate
so the exploratory scripts and the publication set can live in different places
without either importing from the other.

``plot_config``
    matplotlib defaults, colour schemes, the redshift colour mapper, and
    ``save_fig`` / ``maybe_show``. Saving is opt-in via ``BAQARO_SAVE_FIGS=1``,
    so importing or running a figure script never writes to disk by accident.
``load_data_to_plot``
    The ``DataLoader`` that resolves a run from the ``BAQARO_*`` identity
    variables and reads its evolved population, plus the mass-function helpers.
    This is where the Horvitz-Thompson subsample weights are applied, so
    anything summing over objects should go through it rather than reading the
    HDF5 directly.
``local_relation_helpers``
    Selection floors, the weight-aware median/percentile envelope, and the
    published-relation overlays that the local M_BH-sigma / M_BH-M_star figures
    share. The published fits themselves live in
    :mod:`~baqaro.obs_data.m_sigma_m_star_obs_data`.
"""
