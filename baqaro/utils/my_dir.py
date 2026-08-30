"""Filesystem locations for input catalogues, run outputs and figures.

The pipeline reads a multi-terabyte subhalo catalogue and writes forward-run
products of comparable size, so none of that lives inside the repository. This
module is the single place that decides where those trees are.

A *source* names a site: ``"local"``, ``"machine_cosma"`` or ``"machine_igm"``.
Every accessor takes one and returns the corresponding root. Callers select the
site with the ``BAQARO_SOURCE_DIR`` environment variable, which is threaded
through the entry points.

The built-in per-site defaults are the paths used by the original runs and are
almost certainly wrong on any other machine. Rather than editing this file,
point the three environment variables at your own trees:

===========================  ================================================
``BAQARO_DATA_DIR``           input catalogues (the HBT-HERONS subhalo data)
``BAQARO_OUTPUT_DIR``         forward-run / training / emulator products
``BAQARO_PLOTS_DIR``          figure output
===========================  ================================================

When set, each overrides the selected site's default. When unset, the defaults
below apply unchanged.
"""

import os

# --- per-site defaults -----------------------------------------------------
# Site-specific; override with the environment variables documented above.

# A workstation checkout, with data and outputs beside the repository.
main_dir_local = "/Users/eliapizzati/projects/swift_qso/baqaro"
plot_dir = os.path.join(main_dir_local, "plots")
out_dir_local = os.path.join(main_dir_local, "outputs")
data_dir_local = "/Users/eliapizzati/projects/swift_qso/data"

# COSMA (Durham).
out_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/outputs/bh_evolution"
plot_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/plots/bh_evolution"
data_dir_machine_cosma = "/cosma8/data/dp004/dc-pizz1/projects/swift_qso/data"

# IGM (Leiden).
out_dir_machine_igm = "/data3/pizzati/projects/swift_qso/output_data/bh_evolution"
plot_dir_machine_igm = "/data3/pizzati/projects/swift_qso/plots/bh_evolution"
data_dir_machine_igm = "/data3/pizzati/projects/swift_qso/data/"

KNOWN_SOURCES = ("local", "machine_cosma", "machine_igm")

_DATA_DIRS = {
    "local": data_dir_local,
    "machine_cosma": data_dir_machine_cosma,
    "machine_igm": data_dir_machine_igm,
}
_OUT_DIRS = {
    "local": out_dir_local,
    "machine_cosma": out_dir_machine_cosma,
    "machine_igm": out_dir_machine_igm,
}
_PLOT_DIRS = {
    "local": plot_dir,
    "machine_cosma": plot_dir_machine_cosma,
    "machine_igm": plot_dir_machine_igm,
}


def _resolve(table, source, env_var):
    """Look up ``source`` in ``table``, letting ``env_var`` override it."""
    if source not in table:
        raise ValueError(
            f"Invalid source {source!r}. Choose from {list(KNOWN_SOURCES)}."
        )
    return os.environ.get(env_var) or table[source]


def get_input_path_HBT_data(source="local"):
    """Directory holding the HBT-HERONS subhalo catalogue.

    Parameters
    ----------
    source : str
        One of ``KNOWN_SOURCES``.

    Returns
    -------
    str
        Path to the ``HBT_runs_FLAMINGO`` directory. Overridden by
        ``BAQARO_DATA_DIR``, which replaces the site's data root.
    """
    return os.path.join(_resolve(_DATA_DIRS, source, "BAQARO_DATA_DIR"),
                        "HBT_runs_FLAMINGO")


def get_data_path(source="local"):
    """Root directory for auxiliary input data that is not the HBT catalogue.

    Holds externally-obtained tables too large to version, such as the SDSS
    DR16Q quasar property catalogue used by the conditional-ERDF likelihood.

    Parameters
    ----------
    source : str
        One of ``KNOWN_SOURCES``.

    Returns
    -------
    str
        Data root. Overridden by ``BAQARO_DATA_DIR``.
    """
    return _resolve(_DATA_DIRS, source, "BAQARO_DATA_DIR")


def get_output_path(source="local"):
    """Root directory for run products (evolution, training, emulators).

    Parameters
    ----------
    source : str
        One of ``KNOWN_SOURCES``.

    Returns
    -------
    str
        Output root. Overridden by ``BAQARO_OUTPUT_DIR``.
    """
    return _resolve(_OUT_DIRS, source, "BAQARO_OUTPUT_DIR")


def get_plots_path(source="local"):
    """Root directory for figure output.

    Parameters
    ----------
    source : str
        One of ``KNOWN_SOURCES``.

    Returns
    -------
    str
        Plots root. Overridden by ``BAQARO_PLOTS_DIR``.
    """
    return _resolve(_PLOT_DIRS, source, "BAQARO_PLOTS_DIR")
