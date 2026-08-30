"""Guards for the TWO snapshot (z) grids in utils/sim_config.py.

The training grid (emulation/main_training.py) and the multinode forward-run
grid (core_functions/main_evolution_chunked.py) are separate constants, so
that widening the forward-run grid cannot
silently re-index the training HDF5s / emulators. These tests lock:
  - the two grids resolve independently,
  - the back-compat alias still points at the training grid,
  - each is env-overridable in isolation,
  - a malformed env grid raises rather than silently dropping entries.

sim_config resolves the grids at IMPORT time, so each test reloads the module
under a patched environment (importlib.reload).
"""
import importlib
import os

import pytest


def _reload(env):
    """Reload sim_config with `env` applied on top of the current environ."""
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        import baqaro.utils.sim_config as sc
        return importlib.reload(sc)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_l10080_two_grids_distinct_and_superset():
    sc = _reload({"BAQARO_SIM": "L2800N10080",
                  "BAQARO_TRAINING_SNAPSHOTS": None,
                  "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS": None})
    train = set(sc.training_snapshots_default)
    multi = set(sc.multinode_evolution_snapshots_default)
    assert len(train) == 17
    assert len(multi) == 41
    # The multinode grid must reproduce the training z ladder exactly, then refine.
    assert train <= multi, "multinode grid must be a superset of the training grid"
    # Back-compat alias still means "the training grid".
    assert list(sc.snapshots_to_save_default) == list(sc.training_snapshots_default)


def test_env_overrides_are_independent():
    sc = _reload({"BAQARO_SIM": "L2800N10080",
                  "BAQARO_TRAINING_SNAPSHOTS": "10,20,30",
                  "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS": None})
    # training overridden; multinode falls back to its SimSpec default (unchanged).
    assert sc.training_snapshots_default == [10, 20, 30]
    assert len(sc.multinode_evolution_snapshots_default) == 41
    # alias tracks the (overridden) training grid.
    assert sc.snapshots_to_save_default == [10, 20, 30]

    sc = _reload({"BAQARO_SIM": "L2800N10080",
                  "BAQARO_TRAINING_SNAPSHOTS": None,
                  "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS": "5,7,9,9,7"})
    # multinode overridden (sorted + de-duplicated); training untouched.
    assert sc.multinode_evolution_snapshots_default == [5, 7, 9]
    assert len(sc.training_snapshots_default) == 17


def test_malformed_grid_raises():
    with pytest.raises(ValueError):
        _reload({"BAQARO_SIM": "L2800N10080",
                 "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS": "5,notanint,9"})


def test_l5040_multinode_falls_back_to_training():
    # L2800N5040 leaves the multinode grid unset (None) in its SimSpec ⇒ the two
    # grids coincide (the pre-split behaviour).
    sc = _reload({"BAQARO_SIM": "L2800N5040",
                  "BAQARO_TRAINING_SNAPSHOTS": None,
                  "BAQARO_MULTINODE_EVOLUTION_SNAPSHOTS": None})
    assert (sc.multinode_evolution_snapshots_default
            == sc.training_snapshots_default)
