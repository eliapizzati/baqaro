"""Run-identity provenance + read-side guard for the clustering_direct .npz caches.

All the clustering/bias caches (``clustering_direct_z*.npz``, ``quasar_bias_vs_z.npz``,
``quasar_bias_direct_*.npz``) live in ONE shared directory, so without a stamp there
is no way to tell which forward run produced a given cache — a subsample model vs a
multinode full-catalogue direct measurement look identical on disk, and a wrong-run
re-measurement silently overwrites the right one.

These helpers close that gap:
  * ``is_full_catalogue(loader)`` — the physically meaningful test for whether direct
    pair counting is valid (weights all == 1). NOTE the multinode run is stored with a
    ``subset/`` group but unit weights, so it IS a full catalogue; a real HT-weighted
    subsample is NOT.
  * ``provenance_dict(kind, loader, requires_full_cat)`` — the stamp to merge into a
    ``np.savez`` at WRITE time (call from the measurement scripts).
  * ``verify_cache(npz, label, expect_full_cat)`` — the READ-side guard (call from the
    plotting scripts). Prints the source run for the caption; raises if a cache that
    must be full-catalogue was produced on a subsample; warns (does not crash) on an
    unstamped legacy cache.
"""

import re

import numpy as np


def bestfit_of(name_file):
    """Extract the ``_bestfit_<name>`` model token from a run's name_file (the
    piece that identifies the physical model, independent of the subset tag /
    subsample-vs-multinode storage). Returns None if absent."""
    m = re.search(r"_bestfit_(.+?)(?:_sub_|$)", name_file or "")
    return m.group(1) if m else None


def source_bestfit(npz):
    """The bestfit-model token of a cache's source run, or None if unstamped."""
    if "prov_source_name_file" not in getattr(npz, "files", []):
        return None
    return bestfit_of(str(npz["prov_source_name_file"].item()))

# Keys written into every stamped cache (all scalar / short strings so np.savez
# stores them as 0-d arrays; read back with ``.item()``).
PROV_KEYS = (
    "prov_kind",              # "direct_clustering" | "direct_bias" | "model_bias"
    "prov_source_name_file",  # full name_file of the source forward run (encodes everything)
    "prov_subset_tag",        # subset tag ("" for full-sim)
    "prov_full_catalogue",    # bool: source had unit weights (direct pair counting valid)
    "prov_requires_full_cat", # bool: this measurement is only meaningful on a full catalogue
)


def is_full_catalogue(loader):
    """True if the run is a full catalogue (unit weights) — the only case where direct
    pair counting is valid. The multinode run (subset storage, weights==1) qualifies;
    a Horvitz-Thompson-weighted subsample does not."""
    if getattr(loader, "weights", None) is None:
        return True
    w = np.asarray(loader.weights)
    return bool(np.allclose(w[w > 0], 1.0))


def require_full_catalogue(loader, what="Direct pair counting"):
    """Write-side guard: raise if ``loader`` is a down-weighted subsample."""
    if not is_full_catalogue(loader):
        w = np.asarray(loader.weights)
        raise RuntimeError(
            f"{what} needs the FULL-CATALOGUE run (unit weights); this run has "
            f"non-trivial HT weights (max={w.max():.1f}) — it is a down-weighted "
            "subsample. Point BAQARO_SUBSET_TAG at the multinode full-cat run "
            "(e.g. multinode_root144_v4) with BAQARO_USE_SUBSAMPLE=0.")


def provenance_dict(kind, loader, requires_full_cat):
    """Stamp to merge into a measurement's ``np.savez(...)`` — records the exact
    source forward run so the cache is self-identifying."""
    from baqaro.plotting_common.load_data_to_plot import (
        name_file, subset_tag,
    )
    return {
        "prov_kind": kind,
        "prov_source_name_file": name_file,
        "prov_subset_tag": subset_tag or "",
        "prov_full_catalogue": bool(is_full_catalogue(loader)),
        "prov_requires_full_cat": bool(requires_full_cat),
    }


def verify_cache(npz, label="cache", expect_full_cat=None):
    """Read-side guard for a loaded ``.npz``. Returns a short source string.

    - ``expect_full_cat=True`` and the cache came from a subsample -> RuntimeError.
    - unstamped legacy cache -> warn (returns "unknown (unstamped)"), does not crash.
    - otherwise prints and returns the source run identity (for figure captions).
    """
    if "prov_source_name_file" not in npz.files:
        print(f"  [{label}] WARNING: unstamped cache — source run UNKNOWN "
              "(pre-provenance file; re-measure to stamp it).")
        return "unknown (unstamped)"
    src = str(npz["prov_source_name_file"].item())
    full = bool(npz["prov_full_catalogue"]) if "prov_full_catalogue" in npz.files else None
    backfilled = ("  [backfilled stamp]" if "prov_backfilled" in npz.files else "")
    if expect_full_cat and full is False:
        raise RuntimeError(
            f"[{label}] this measurement requires a FULL-CATALOGUE source, but the "
            f"cache was produced on a SUBSAMPLE run:\n    {src}\n"
            "Re-measure on the multinode full-cat run.")
    kind = "full-cat" if full else "subsample" if full is not None else "?"
    print(f"  [{label}] source: {kind} :: {src}{backfilled}")
    return src
