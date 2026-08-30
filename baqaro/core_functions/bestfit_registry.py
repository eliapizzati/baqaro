"""
Best-fit parameter registry for `main_evolution.py`.
=====================================================

A single source-of-truth dict keyed by a short ``bestfit_name``. Each
entry holds the 6 free parameters that inference / optimization produces,
plus minimal context: ``label``, ``sim``, ``max_snap``, and ``notes`` —
the ``notes_file`` token its forward runs were produced with (older
entries pin a fixed token so their on-disk filenames stay reproducible;
newer entries default to ``None`` = no notes token).

How `main_evolution.py` consumes this:

1. The user sets ``BAQARO_BESTFIT_NAME=<key>`` (also tags the output
   filename ``_bestfit_<key>``).
2. If ``<key>`` is in ``BESTFIT_REGISTRY``, the 6 params are loaded
   from the entry (overriding the hardcoded defaults).
3. Individual env vars (``BAQARO_LOG_ETA_MEAN_0`` etc.) still take final
   precedence — useful for one-off tweaks of a baseline entry without
   forking a new registry entry.
4. If ``<key>`` is not in the registry, the name is treated as a
   free-form filename tag — useful when you want to run a custom set
   without committing it to the registry yet.

The ``notes`` field is resolved into ``notes_file`` with the same
3-tier precedence: ``BAQARO_NOTES_FILE`` env (presence wins; ``""`` =
explicit no-token) > entry ``notes`` > ``None``. So re-running a
registered bestfit reproduces the exact on-disk filename of its
original run.

To add a new entry: append a new key with the 6 params + minimal
context (``label``/``sim``/``max_snap``/``notes``), ideally with a
one-line comment naming the inference run that produced it.

Two things to know before using an entry
----------------------------------------
**Not every entry is comparable with current runs.** Parameters fitted
before the growth-normalisation change in the accretion engine belong to a
different normalisation of the model and are not comparable with any current
run. Those keys are listed in ``PRE_GROWTHFIX_BESTFITS`` and ``get_bestfit``
warns when one is selected; they are kept only so that runs already on disk
under those names stay identifiable.

**A run's own file is the better provenance source.** Since every forward
run writes a self-describing ``provenance/`` group, the parameters of any
product on disk can be read back directly with
``h5py.File(path)["provenance"].attrs`` — the registry is a convenience
for *launching* runs, not the archive of record.
"""

import os
import warnings


BESTFIT_REGISTRY: dict = {
    "qcc_fid1_z0_chunked_noshift_v1": {
        "label": "joint QLF+cERDF+clustering fit, L2800N10080 max_snap=144; predates the growth-normalisation change",
        "sim": "L2800N10080", "max_snap": 144,
        "notes": "newfast_test11",
        "log_eta_mean_0": -1.002291, "log_eta_mean_evol": 0.841191, "std_0": 0.535679,
        "logtcoherence": 5.703964, "logfseed": -6.314546, "sigmaseed": 0.450976,
    },
}




#: Entries whose parameters were fitted before the growth-normalisation change
#: in the accretion engine. They belong to a different normalisation of the
#: model and are not comparable with any current run; they are kept only so
#: that runs already on disk under these names remain identifiable.
#:
#: ``get_bestfit`` warns when one is selected; set
#: ``BAQARO_ALLOW_STALE_BESTFIT=1`` to silence it for a deliberate reproduction.
PRE_GROWTHFIX_BESTFITS = frozenset({
    "qcc_fid1_z0_chunked_noshift_v1",
})


def _warn_if_pre_growthfix(name: str) -> None:
    """Warn when ``name`` predates the growth-normalisation change (see the frozenset above)."""
    if name not in PRE_GROWTHFIX_BESTFITS:
        return
    if os.environ.get("BAQARO_ALLOW_STALE_BESTFIT", "0") == "1":
        return
    warnings.warn(
        f"bestfit {name!r} predates the growth-normalisation change in the "
        "accretion engine; its parameters are not comparable with current "
        "runs. Set BAQARO_ALLOW_STALE_BESTFIT=1 to silence this.",
        UserWarning, stacklevel=3,
    )


def get_bestfit(name: str) -> dict:
    """Return the registry entry for ``name``, or raise KeyError with a
    helpful list of available keys."""
    if name not in BESTFIT_REGISTRY:
        raise KeyError(
            f"BESTFIT_NAME={name!r} not in registry. "
            f"Known keys: {sorted(BESTFIT_REGISTRY.keys())}"
        )
    _warn_if_pre_growthfix(name)
    return BESTFIT_REGISTRY[name]


def list_bestfits() -> None:
    """Pretty-print every registered best-fit. Handy CLI:
    ``python -c "from baqaro.core_functions.bestfit_registry import list_bestfits; list_bestfits()"``
    """
    for key, entry in BESTFIT_REGISTRY.items():
        print(f"{key:25s}  {entry.get('sim', '?'):12s}  max_snap={entry.get('max_snap', '?')}  — {entry.get('label', '')}")


def resolve_notes_file(bestfit_name=None):
    """Consumer-side ``notes_file`` resolution — the read-side mirror of
    ``main_evolution``'s three-tier logic, so a plotting/analysis script lands
    on the exact filename the forward run produced.

    Precedence (highest first), env *presence* winning:
      3. ``BAQARO_NOTES_FILE`` env  (``""`` => None, i.e. explicit no-token)
      2. the matched registry entry's ``notes`` field (legacy entries pin
         ``"newfast_test11"``; new ones default to ``None``)
      1. ``None`` (no ``_{notes}`` token)

    ``bestfit_name`` defaults to ``BAQARO_BESTFIT_NAME`` from the environment,
    so callers can simply do ``notes_file = resolve_notes_file()``.
    """
    import os
    if "BAQARO_NOTES_FILE" in os.environ:
        return os.environ["BAQARO_NOTES_FILE"] or None
    if bestfit_name is None:
        bestfit_name = os.environ.get("BAQARO_BESTFIT_NAME")
    if bestfit_name and bestfit_name in BESTFIT_REGISTRY:
        return BESTFIT_REGISTRY[bestfit_name].get("notes") or None
    return None






BESTFIT_REGISTRY["qcc_bugfix_z0_v1"] = {
    "label": "joint QLF+cERDF+clustering DE optimum, L2800N10080 max_snap=144",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": "newfast_test11",
    "log_eta_mean_0": -1.330021, "log_eta_mean_evol": 0.83582, "std_0": 0.512473,
    "logtcoherence": 5.80737, "logfseed": -5.359161, "sigmaseed": 0.295085,
}




BESTFIT_REGISTRY["qcc_bugfix_z0_v2"] = {
    "label": "joint QLF+cERDF+clustering DE optimum at the default cERDF binning (z={1,2,3,3.94,5.02,6.14}, 0.5-dex eta bins)",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": "bugfix_z0",
    "log_eta_mean_0": -1.363512, "log_eta_mean_evol": 0.840281, "std_0": 0.530303,
    "logtcoherence": 5.848908, "logfseed": -5.300048, "sigmaseed": 0.240855,
}



BESTFIT_REGISTRY["qcc_bugfix_z0_v3_bigeta"] = {
    "label": "joint QLF+cERDF+clustering DE optimum with 1.0-dex cERDF eta bins (BAQARO_CERDF_LOG_ETA_BINS=-2.5,-1.5,-0.5,0.5,1.5), growth cap 4.6; an earlier fiducial, replaced by qcc_ck22final_v1",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.260446, "log_eta_mean_evol": 0.842368, "std_0": 0.535071,
    "logtcoherence": 5.520492, "logfseed": -6.540536, "sigmaseed": 0.559988,
}

BESTFIT_REGISTRY["qlfcerdf_bugfix_z0_v3_bigeta"] = {
    "label": "QLF+cERDF DE optimum with 1.0-dex cERDF eta bins, growth cap 4.6",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.392527, "log_eta_mean_evol": 0.834289, "std_0": 0.550883,
    "logtcoherence": 4.500600, "logfseed": -6.133257, "sigmaseed": 0.792460,
}




BESTFIT_REGISTRY["qcc_bugfix_z0_g691_v1"] = {
    "label": "joint QLF+cERDF+clustering DE optimum; growth cap 6.91, f_eff correction on",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": "bugfix_z0_chunked_g6.91",
    "log_eta_mean_0": -1.23721, "log_eta_mean_evol": 0.820709, "std_0": 0.499799,
    "logtcoherence": 6.053045, "logfseed": -6.254738, "sigmaseed": 0.463139,
}


# z=2.5 clustering: the published EF15 covariance matrix is indefinite, so
# EF15 Table 3 with diagonal errors (`data_ef_ext_restricted`) is used; it is
# the main_mcmc default. The entries below also use the merged top QLF bin
# (BAQARO_QLF_MERGE_TOP_SLIVER=1), the bin-integrated cERDF
# (BAQARO_CERDF_BIN_INTEGRATED=1) and per-z QLF faint floors
# (BAQARO_QLF_MIN_LBOL_PER_Z), with the K24 subsample geometry
# (BAQARO_SUBSAMPLE_LOG_M_HI=16.0, BAQARO_SUBSAMPLE_NBINS=24).
BESTFIT_REGISTRY["qcc_tab3_v1"] = {
    "label": (
        "joint QLF+cERDF+clustering fit with the binned cERDF; EF15 Table 3 at "
        "z=2.5, merged top QLF bin, bin-integrated cERDF, per-z QLF floors; "
        "growth cap 6.91, f_eff on; K24 subsample"
    ),
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.234752, "log_eta_mean_evol": 0.830318, "std_0": 0.495234,
    "logtcoherence": 5.930498, "logfseed": -6.296859, "sigmaseed": 0.507857,
}

# ---------------------------------------------------------------------------
#  Unbinned (per-object) cERDF
# ---------------------------------------------------------------------------
# Same joint fit as qcc_tab3_v1; the only change is the cERDF estimator,
# per-object (unbinned) instead of binned, tempered at T = 5000. T=5000 sits
# on a measured plateau (see the paper's appendix on the cERDF likelihood)
# and is quoted as a systematic.
# Config: BAQARO_CERDF_MODE=per_object, BAQARO_CERDF_TEMP=5000,
#         BAQARO_CERDF_Z_SLICE_MIN=1.0; scatter_dex=0.3, outlier_frac=0 (defaults).
BESTFIT_REGISTRY["qcc_unb_T5000_v1"] = {
    "label": (
        "joint QLF+cERDF+clustering fit with the unbinned cERDF (T=5000, "
        "z-slices >= 1.0); otherwise as qcc_tab3_v1, plus the full ASPIRE "
        "covariance at z=6; growth cap 6.91, f_eff on; K24 subsample. An "
        "earlier fiducial, replaced by qcc_ck22final_v1"
    ),
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.197032, "log_eta_mean_evol": 0.830494, "std_0": 0.468825,
    "logtcoherence": 6.033639, "logfseed": -6.387826, "sigmaseed": 0.513948,
}




BESTFIT_REGISTRY["qcc_cleanK22_g6.21_v1"] = {
    "label": "joint QLF+cERDF+clustering DE optimum; K22 subsample, growth cap 6.21, unbinned cERDF (T=5000)",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.220108, "log_eta_mean_evol": 0.83942, "std_0": 0.497011,
    "logtcoherence": 5.896001, "logfseed": -6.691491, "sigmaseed": 0.598355,
}




BESTFIT_REGISTRY["qcc_ck22final_v1"] = {
    "label": "joint QLF+cERDF+clustering fit; the adopted fiducial (what plotting_paper/fiducial_data.py resolves to by default)",
    "sim": "L2800N10080", "max_snap": 144,
    "notes": None,
    "log_eta_mean_0": -1.235476, "log_eta_mean_evol": 0.832736, "std_0": 0.507524,
    "logtcoherence": 5.894683, "logfseed": -6.469369, "sigmaseed": 0.505643,
}


