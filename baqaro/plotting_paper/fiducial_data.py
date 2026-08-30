"""
Pins the FIDUCIAL data run for the PAPER figures (``plotting_paper/``).

``plotting_common/load_data_to_plot.py`` reads ``BAQARO_SIM`` /
``BAQARO_MAX_SNAP`` / ``BAQARO_NOTES_FILE`` / ``BAQARO_BESTFIT_NAME`` /
``BAQARO_SUBSET_TAG`` / ... **at import time** and builds the evolution-HDF5
path + ``DataLoader`` from them.  The exploratory scripts
deliberately let those vary so you can experiment.  The PAPER figures must instead always load ONE settled run.

This module is the single choke-point for that.  It **force-sets** the
run-identity env vars to the fiducial values *before* importing
``load_data_to_plot``, so whatever ``BAQARO_*`` you have exported while
experimenting elsewhere cannot leak into the paper figures.
It then re-exports the resolved paths + loader helper.  **Paper scripts import
data handles from here, never from ``load_data_to_plot`` directly.**

Re-pointing the paper
---------------------
Edit ``PAPER_FIDUCIAL`` below — one place, all paper figures move together.

One-off override (``BAQARO_PAPER_*`` escape hatch)
-------------------------------------------------
To regenerate the paper figures against a *different* run without editing this
file (e.g. to compare against the legacy L2800N5040 z=0 production run), set the
parallel ``BAQARO_PAPER_<X>`` var, which wins over the fiducial default for that
key only::

    env BAQARO_PAPER_SIM=L2800N5040 BAQARO_PAPER_MAX_SNAP=78 \
        BAQARO_PAPER_FOLD_SUBHALO_MASS=0 BAQARO_PAPER_MERGER_DELAY_MODE=instant_old \
        BAQARO_PAPER_BESTFIT_NAME= BAQARO_PAPER_SUBSET_TAG= \
        python -m baqaro.plotting_paper.plotting_erdf_model

Plain ``BAQARO_SIM`` etc. are IGNORED here by design — that's the whole point.

Note: ``BAQARO_SOURCE_DIR`` (machine_igm vs machine_cosma) is a *machine*
choice, not part of the run identity, so it is left to the ambient
environment.
"""

import os

# utils.fiducial is a LEAF module: plain literals, no env reads, no package
# imports -- so importing it here, BEFORE os.environ is pinned below, cannot
# bind anything to the ambient environment. (Importing utils.sim_config here
# instead WOULD: its simulation_name/max_snap aliases are env-derived at import,
# which is exactly the ordering trap the pinning below exists to avoid.)
from baqaro.utils import fiducial as _fid


# ---------------------------------------------------------------------------
# THE FIDUCIAL RUN.  Edit here to re-point every paper figure at once.
# Keys are the BAQARO_* vars load_data_to_plot / sim_config consume; values are
# the pinned fiducial.  Empty string neutralises a token: load_data_to_plot
# uses `if bestfit_name:` / `if subset_tag:`, so "" => no _bestfit / _sub token
# (i.e. full-sim legacy production output).
# ---------------------------------------------------------------------------
# THE ADOPTED FIDUCIAL: L2800N10080 z=0 (snap 144), the joint QLF + cERDF +
# clustering best fit `qcc_ck22final_v1` (utils/fiducial.BESTFIT_NAME): the
# per-object cERDF likelihood tempered at T=5000, fitted on the K22 stratified
# subsample at growth cap g=6.21 (~500x/snap) with the madau f_eff correction ON.
# Params (bestfit_registry): eta0=-1.235476, eta_evol=0.832736, std0=0.507524,
# logtcoh=5.894683, logfseed=-6.469369, sigmaseed=0.505643.
# The products, all named from utils/fiducial:
#   * training HDF5   notes `clean_z0_g6.21_K22_final`
#   * emulators       notes `clean_z0_g6.21_K22_final_smooth`
#   * chains          mcmc_notes `ck22final_100k` (MCMC_FIDUCIAL below)
#   * population run  the multinode full-catalogue forward run
#                     (`multinode_root144_v4`, foldmass + instant_new, no notes token)
# Forward run, emulators and chains share one subset geometry (K22) and one cap
# (g6.21), so no figure mixes two model definitions.
PAPER_FIDUCIAL = {
    "BAQARO_SIM": _fid.SIM,
    "BAQARO_MAX_SNAP": str(_fid.MAX_SNAP),
    # The multinode full-catalogue forward run: the population figures
    # (QLF/cERDF/QHMF/BHMF/clustering/merger/…) read the FULL 2.255-billion-halo
    # catalogue (`multinode_root144_v4`, weights all == 1, no subsample shot
    # noise) rather than the K22 500k/bin subsample. Same model (qcc_ck22final_v1,
    # g6.21, feffcorr) — only the halo sampling differs. The run carries no notes
    # token, so BAQARO_NOTES_FILE="". 41 snapshots saved (z=8.68→0);
    # snapshot_index_for_redshift picks the closest + warns on any mis-snap.
    "BAQARO_NOTES_FILE": _fid.PAPER_NOTES_FILE,
    "BAQARO_FOLD_SUBHALO_MASS": "1" if _fid.FOLD_SUBHALO_MASS else "0",
    "BAQARO_MERGER_DELAY_MODE": _fid.MERGER_DELAY_MODE,
    "BAQARO_BESTFIT_NAME": _fid.BESTFIT_NAME,
    # Multinode full-catalogue run (all halos, weights==1). To fall back to the
    # K22 500k/bin subsample forward run set
    # BAQARO_PAPER_SUBSET_TAG=root144_flatN500000_K22_logM10.0to15.5_seed42_v3.
    "BAQARO_SUBSET_TAG": _fid.PAPER_SUBSET_TAG,
    # growth cap g=6.21 (~500x/snap). Set BAQARO_PAPER_GROWTH_SUM_MAX=50.0 to
    # address an uncapped run.
    "BAQARO_GROWTH_SUM_MAX": str(_fid.GROWTH_SUM_MAX),
    # madau f_eff correction ON: the fiducial's forward run + emulators
    # are all _feffcorr, and must never be mixed with uncorrected products. Pinned so an
    # ambient value cannot swap the model unnoticed.
    "BAQARO_MADAU_FEFF_CORRECTION": "1" if _fid.MADAU_FEFF_CORRECTION else "0",
}


def _resolve(key, default):
    """Fiducial default, overridable per-key by the parallel BAQARO_PAPER_<X>."""
    paper_key = "BAQARO_PAPER_" + key[len("BAQARO_"):]
    return os.environ.get(paper_key, default)


# Force-set the BAQARO_* vars BEFORE load_data_to_plot (and sim_config, which it
# imports) read the environment.  This must run before the import below.
RESOLVED = {key: _resolve(key, default) for key, default in PAPER_FIDUCIAL.items()}
for _key, _val in RESOLVED.items():
    os.environ[_key] = _val


# ---------------------------------------------------------------------------
# The three sub-fiducials below (full-history / duty-cycle / high-z lightcurves)
# are DIFFERENT on-disk products of the SAME physical model. They legitimately
# differ in max_snap, selection_tag and subset_tag — but the model identity
# (which best-fit params, which notes_file, which cap / f_eff state) must track
# PAPER_FIDUCIAL, or a repoint leaves Figs 5 / 16 / 19 rendering the
# OLD model beside the new one. Inherit those four keys by default; every
# key stays individually overridable via the BAQARO_PAPER_{FH,DC,LC_FH}_<X> hatch.
# ---------------------------------------------------------------------------
# notes_file is a LAUNCH-TIME filename label, not model identity. It is EMPTY
# for every fiducial product. One rule: notes is "" unless a run genuinely
# needs disambiguating (e.g. the samplerdemo_* / detest_* runs, where it still does
# real work). The constant is kept purely as the BAQARO_PAPER_FH_NOTES_FILE override
# hook for pointing at an older generation. TRUE model identity (bestfit_name /
# growth / f_eff) tracks PAPER_FIDUCIAL via _MODEL_IDENTITY below.
_FULL_HISTORY_NOTES = os.environ.get("BAQARO_PAPER_FH_NOTES_FILE", "")

_MODEL_IDENTITY = {
    "notes_file": _FULL_HISTORY_NOTES,
    "bestfit_name": RESOLVED["BAQARO_BESTFIT_NAME"],
    "growth_sum_max": RESOLVED["BAQARO_GROWTH_SUM_MAX"],
    "feffcorr": RESOLVED["BAQARO_MADAU_FEFF_CORRECTION"],
}


# ---------------------------------------------------------------------------
# Now build the fiducial paths / loader via the canonical machinery and
# re-export.  Imported AFTER the env is pinned, so it sees the fiducial.
# ---------------------------------------------------------------------------
from baqaro.plotting_common.load_data_to_plot import (  # noqa: E402
    path_file,
    path_file_full_history,
    path_out,
    path_plots,
    path_file_halo_masses,
    name_file,
    name_file_halos,
    max_snap,
    boxsize,
    simulation_name,
    source_dir,
    subset_tag,
    erdf_model,
    load_simulation_metadata,
    load_simulation_metadata_full_history,
    luminosity_function_auto,
    mass_function_auto,
    weighted_median,
    weighted_percentile,
    snapshot_index_for_redshift,
    full_simulation_redshifts,
)


# ---------------------------------------------------------------------------
# FULL-HISTORY fiducial (sub-step BH tracks for the single-tree figure).
# These come from a SEPARATE, small `main_evolution_full_history` run — a
# first-born selection at z=0 — NOT the big root144 subsample forward run that
# PAPER_FIDUCIAL points at. So it has its own notes_file / bestfit and no subset
# tag, and is built EXPLICITLY here (the env pin above can't express it without
# clobbering the main fiducial). Override any token with BAQARO_PAPER_FH_<X>;
# pick the tree to draw with BAQARO_PAPER_TREE_IDX (default = first target).
# ---------------------------------------------------------------------------
FULL_HISTORY_FIDUCIAL = {
    "sim": "L2800N10080",
    # firstborn100 z=0 run (max_snap=144): the single-tree figure's pruner
    # keeps branches with lifetimes of >= 50 snapshots, which leaves only the
    # trunk on a 71-snapshot (z=2) tree. The firstborn100 z=0 trees span the
    # full 144 snapshots, so the rank-14 backbone has enough mergers to fill
    # the panel.
    "max_snap": "144",
    "foldmass": "1",
    "merger_delay_mode": "instant_new",
    "selection_tag": "firstborn100",
    "subset_tag": "",
    # notes_file / bestfit_name / growth_sum_max / feffcorr inherited from
    # PAPER_FIDUCIAL — see _MODEL_IDENTITY.
    **_MODEL_IDENTITY,
}


def _resolve_fh(key, default):
    """Full-history fiducial default, overridable by BAQARO_PAPER_FH_<KEY>."""
    return os.environ.get("BAQARO_PAPER_FH_" + key.upper(), default)


FH_RESOLVED = {k: _resolve_fh(k, v) for k, v in FULL_HISTORY_FIDUCIAL.items()}


def _build_name_file(d):
    """Reproduce load_data_to_plot's name_file token order for a token dict."""
    # NB: no `_6d` ("6 free parameters") token: it was constant in every name,
    # never varied, and neither the forward-run nor the emulator/chain names carry it.
    nf = "{}_erdf_{}_maxsnap_{}".format(d["sim"], erdf_model, d["max_snap"])
    if d["foldmass"] == "1":
        nf += "_foldmass"
    if d["merger_delay_mode"] != "instant_old":
        nf += "_{}".format(d["merger_delay_mode"])
    # Full-history selection tag (auto-derived by main_evolution_full_history);
    # only the FH/duty-cycle dicts carry it. Comes before the optional notes.
    if d.get("selection_tag"):
        nf += "_{}".format(d["selection_tag"])
    if d["notes_file"]:
        nf += "_{}".format(d["notes_file"])
    if d["bestfit_name"]:
        nf += "_bestfit_{}".format(d["bestfit_name"])
    if d["subset_tag"]:
        nf += "_sub_{}".format(d["subset_tag"])
    return nf + _growth_feff_tokens(d)


def _growth_feff_tokens(d):
    """`_g{cap}[_feffcorr]` built from a token dict, NOT from the environment.

    Mirrors ``utils.sim_config.growth_feff_suffix`` (which reads the env), but
    each fiducial dict here describes a DIFFERENT on-disk product and so carries
    its own cap/f_eff state. For the adopted fiducial the states coincide (the
    forward run, the emulators and the chains all carry `_g6.21_feffcorr`), but
    the evolution-side cap must still never leak into the MCMC/emulator name: a
    BAQARO_PAPER_GROWTH_SUM_MAX override repoints the forward run only, while
    the emulator identity is set in MCMC_FIDUCIAL.
    """
    tok = ""
    _g = float(d.get("growth_sum_max", "50.0"))
    if _g != 50.0:
        tok += "_" + (f"g{int(_g)}" if float(_g).is_integer() else f"g{_g:g}")
    if str(d.get("feffcorr", "0")).strip().lower() not in ("0", "", "false", "no", "off"):
        tok += "_feffcorr"
    return tok


# ---------------------------------------------------------------------------
# SUBSAMPLE forward-run name_file (for PHYSICS-VARIANT overlays).
# The fiducial `name_file` above is now the MULTINODE full-catalogue run, but the
# physics-toggle variants (`_tau0` / `_tau7` / `_radeff_constant` / `_nomerge`)
# were only ever launched on the K22 500k/bin SUBSAMPLE. So a figure that draws a
# MULTINODE fiducial curve against SUBSAMPLE variation curves (e.g.
# `plotting_bhmf_erdf_highz`) must build the variant paths from THIS name_file,
# not from the multinode one. Same model (`qcc_ck22final_v1`, g6.21, f_eff);
# only the halo sampling differs (both carry an empty notes token). `_build_name_file`
# with no selection_tag reproduces the forward-run token order exactly.
_VARIANTS_SUBSET = {
    "sim": RESOLVED["BAQARO_SIM"],
    "max_snap": RESOLVED["BAQARO_MAX_SNAP"],
    "foldmass": RESOLVED["BAQARO_FOLD_SUBHALO_MASS"],
    "merger_delay_mode": RESOLVED["BAQARO_MERGER_DELAY_MODE"],
    "selection_tag": "",
    # Empty for every fiducial product (see _FULL_HISTORY_NOTES).
    "notes_file": os.environ.get("BAQARO_PAPER_VARIANTS_NOTES_FILE", ""),
    "bestfit_name": RESOLVED["BAQARO_BESTFIT_NAME"],
    "subset_tag": os.environ.get(
        "BAQARO_PAPER_VARIANTS_SUBSET_TAG",
        "root144_flatN500000_K22_logM10.0to15.5_seed42_v3"),
    "growth_sum_max": RESOLVED["BAQARO_GROWTH_SUM_MAX"],
    "feffcorr": RESOLVED["BAQARO_MADAU_FEFF_CORRECTION"],
}
name_file_variants = _build_name_file(_VARIANTS_SUBSET)


name_file_full_history_fid = _build_name_file(FH_RESOLVED)
path_file_full_history_fid = os.path.join(
    path_out, "evolution",
    "bh_evolution_full_history_{}.hdf5".format(name_file_full_history_fid),
)


# ---------------------------------------------------------------------------
# DUTY-CYCLE fiducial (z~6 duty-cycle distribution figure). This is a THIRD
# distinct full-history run: a mass-threshold selection that evolves ALL alive
# halos above log M_halo > 11.5, so the f_duty distribution of the L_bol>10^46.5
# QSO sample is measured on a COMPLETE sample (not first-born, not
# tree_subsample). Same fiducial BH params as PAPER_FIDUCIAL, foldmass +
# instant_new, no subset.
#
# max_snap = 40 (z=5.879), NOT 39. Two reasons:
#   1. The QLF/BHMF comparison must sit at the snapshot the rest of the paper
#      calls z=6 — `snapshot_index_for_redshift(6.0)` = 40, since
#      |6.0-5.879| < |6.0-6.145| (the nearest snapshot is kept).
#   2. It makes the z=6.145 QSO sample MORE COMPLETE. Selection is on halo mass
#      at the tree root, so a maxsnap39 run drops halos below 10^11.5 at z=6.14
#      that nonetheless host an L>10^46.5 quasar there. Rooting at snap 40 admits
#      them (LastMaxMass is monotonic, so the snap-40 cut is a strict superset):
#      the bright sample went 230 -> 269 objects.
# The QSO selection itself STAYS at snap 39: a z-target of 6.1 picks
# argmin|z-6.1| = 39, matching the observed sample (median z=6.275).
# Override any token with BAQARO_PAPER_DC_<X>.
# ---------------------------------------------------------------------------
DUTY_CYCLE_FIDUCIAL = {
    "sim": "L2800N10080",
    "max_snap": "40",
    "foldmass": "1",
    "merger_delay_mode": "instant_new",
    "selection_tag": "massthr11.5",   # all halos with log M_halo > 11.5 at the root snap
    "subset_tag": "",
    # Model identity inherited from PAPER_FIDUCIAL — see _MODEL_IDENTITY.
    **_MODEL_IDENTITY,
}


# ---------------------------------------------------------------------------
# HIGH-Z LIGHTCURVES fiducial (the lightcurves_highz figure). DISTINCT from
# the single-tree FH fiducial:
#
#   * single-tree         -> firstborn100 z=0 (max_snap=144); the pruner needs
#                            ≥50-snap lifetimes so trees must span the full
#                            144 snapshots to fill the panel.
#   * lightcurves_highz   -> massthr13.5 z=2 (max_snap=71); the script
#                            picks 6 hand-curated halo indices that represent
#                            distinctive luminous z~6 QSO progenitors from
#                            inside the merger closure of massive z=2 hosts
#                            (pinned by storage id).
#                            Those IDs are baked in as the default in the
#                            script via BAQARO_PAPER_LC_IDS; this fiducial just
#                            points at the right HDF5 file.
#
# Override any token with BAQARO_PAPER_LC_FH_<X>.
# ---------------------------------------------------------------------------
LIGHTCURVES_HIGHZ_FIDUCIAL = {
    "sim": "L2800N10080",
    "max_snap": "71",
    "foldmass": "1",
    "merger_delay_mode": "instant_new",
    # The fiducial LC run was produced with selection tag `massthr13.5` (no
    # `_sub` suffix); the tag here must match the file on disk. Override via
    # BAQARO_PAPER_LC_FH_SELECTION_TAG.
    "selection_tag": "massthr13.5",
    "subset_tag": "",
    # Model identity inherited from PAPER_FIDUCIAL — see _MODEL_IDENTITY.
    **_MODEL_IDENTITY,
}


def _resolve_lc_fh(key, default):
    """LC-high-z fiducial default, overridable by BAQARO_PAPER_LC_FH_<KEY>."""
    return os.environ.get("BAQARO_PAPER_LC_FH_" + key.upper(), default)


LC_FH_RESOLVED = {k: _resolve_lc_fh(k, v)
                  for k, v in LIGHTCURVES_HIGHZ_FIDUCIAL.items()}

name_file_lc_highz_fid = _build_name_file(LC_FH_RESOLVED)
path_file_lc_highz_fid = os.path.join(
    path_out, "evolution",
    "bh_evolution_full_history_{}.hdf5".format(name_file_lc_highz_fid),
)


def _resolve_dc(key, default):
    """Duty-cycle fiducial default, overridable by BAQARO_PAPER_DC_<KEY>."""
    return os.environ.get("BAQARO_PAPER_DC_" + key.upper(), default)


DC_RESOLVED = {k: _resolve_dc(k, v) for k, v in DUTY_CYCLE_FIDUCIAL.items()}

name_file_duty_cycle_fid = _build_name_file(DC_RESOLVED)
path_file_duty_cycle_fid = os.path.join(
    path_out, "evolution",
    "bh_evolution_full_history_{}.hdf5".format(name_file_duty_cycle_fid),
)


# ---------------------------------------------------------------------------
# MCMC fiducial (posterior corner figure). Chains are named by the EMULATOR
# name_file (built by inference/main_mcmc.py — note: NO `_6d` token, NO
# `_bestfit`) + the active-likelihood combo + optional notes. The adopted
# fiducial's inference: the `qcc_ck22final_v1` chains — the per-object cERDF at
# T=5000 — on the cap+f_eff-consistent `clean_z0_g6.21_K22_final_smooth`
# emulators (mcmc_notes `ck22final_100k`). The per-combo active-likelihood tag
# (qlf / qlf+cerdf / ...) is appended by the corner script. Override with
# BAQARO_PAPER_MCMC_<X>.
# CONSISTENT with PAPER_FIDUCIAL: this suite's forward run, emulators, and chains
# ALL live on the SAME K22 subset at the SAME g6.21 cap.
# ---------------------------------------------------------------------------
MCMC_FIDUCIAL = {
    "sim": "L2800N10080",
    "max_snap": "144",
    "foldmass": "1",
    "merger_delay_mode": "instant_new",
    "notes_file_emulation": "clean_z0_g6.21_K22_final_smooth",
    "subset_tag": "root144_flatN500000_K22_logM10.0to15.5_seed42_v3",
    # ck22final production 100k chains (the paper corner).
    "mcmc_notes": "ck22final_100k",
    # Cap + f_eff CONSISTENT: these emulators WERE trained with the growth cap and
    # the madau f_eff correction, so their name carries both `_g6.21` and
    # `_feffcorr`.
    "growth_sum_max": "6.21",
    "feffcorr": "1",
}


def _resolve_mcmc(key, default):
    """MCMC fiducial default, overridable by BAQARO_PAPER_MCMC_<KEY>."""
    return os.environ.get("BAQARO_PAPER_MCMC_" + key.upper(), default)


MCMC_RESOLVED = {k: _resolve_mcmc(k, v) for k, v in MCMC_FIDUCIAL.items()}


def _build_mcmc_name_file(d):
    """Reproduce inference/main_mcmc.py's emulator/chain name_file (NO `_6d`)."""
    nf = "{}_erdf_{}_maxsnap_{}".format(d["sim"], erdf_model, d["max_snap"])
    if d["foldmass"] == "1":
        nf += "_foldmass"
    if d["merger_delay_mode"] != "instant_old":
        nf += "_{}".format(d["merger_delay_mode"])
    if d["notes_file_emulation"]:
        nf += "_{}".format(d["notes_file_emulation"])
    if d["subset_tag"]:
        nf += "_sub_{}".format(d["subset_tag"])
    # main_mcmc.py:218-219 appends the same `_g{cap}[_feffcorr]` tokens, but
    # keyed on the EMULATOR's training run — read them from this dict, not the
    # env (which carries PAPER_FIDUCIAL's evolution-side cap). Override with
    # BAQARO_PAPER_MCMC_GROWTH_SUM_MAX / BAQARO_PAPER_MCMC_FEFFCORR.
    return nf + _growth_feff_tokens(d)


name_file_mcmc = _build_mcmc_name_file(MCMC_RESOLVED)
mcmc_notes = MCMC_RESOLVED["mcmc_notes"]
mcmc_dir = os.path.join(path_out, "mcmc")

# Propagate the MCMC emulator notes to the ambient env so that
# inference/comparison_config.py (which reads BAQARO_NOTES_FILE_EMULATION /
# BAQARO_MCMC_NOTES at import) resolves to THIS fiducial's chains + emulators when
# a paper script imports it AFTER this module. The shared sim/max_snap/fold/
# merger/subset vars are already pinned by PAPER_FIDUCIAL above, so
# comparison_config.name_file then equals name_file_mcmc.
os.environ["BAQARO_NOTES_FILE_EMULATION"] = MCMC_RESOLVED["notes_file_emulation"]
os.environ["BAQARO_MCMC_NOTES"] = MCMC_RESOLVED["mcmc_notes"]
# CRITICAL: the population run now lives on the MULTINODE full catalogue, so
# PAPER_FIDUCIAL force-set the ambient BAQARO_SUBSET_TAG to `multinode_root144_v4`
# (line ~107). But the emulators + chains were trained on the K22 500k/bin
# SUBSAMPLE, and comparison_config.py reads the ambient BAQARO_SUBSET_TAG to build
# their name_file. Re-pin it here to the MCMC fiducial's (K22) subset so
# comparison_config.name_file == name_file_mcmc. The population path_file/loader
# were already resolved above (import time) against the multinode tag, so this
# re-pin is safe — nothing downstream re-reads BAQARO_SUBSET_TAG for the run paths.
os.environ["BAQARO_SUBSET_TAG"] = MCMC_RESOLVED["subset_tag"]


def load_paper_emulators(flags):
    """Load the MCMC-fiducial GP emulators for the requested quantities.

    ``flags`` is a dict like ``{"qlf": True, "bhmf": False, "cerdf": True,
    "qhmf": True}``. Returns the ``load_emulators`` dict (each value an emulator
    or ``None``). Used by the posterior-predictive comparison figure. The heavy
    ``emulation`` import is deferred to here so the (many) evolution-only paper
    scripts don't pull it in.
    """
    from baqaro.emulation.loading_helpers import load_emulators
    return load_emulators(str(path_out), name_file_mcmc, flags)


__all__ = [
    "PAPER_FIDUCIAL",
    "RESOLVED",
    "FULL_HISTORY_FIDUCIAL",
    "FH_RESOLVED",
    "DUTY_CYCLE_FIDUCIAL",
    "DC_RESOLVED",
    "MCMC_FIDUCIAL",
    "MCMC_RESOLVED",
    "path_file",
    "path_file_full_history",
    "path_file_full_history_fid",
    "name_file_full_history_fid",
    "name_file_variants",
    "path_file_duty_cycle_fid",
    "name_file_duty_cycle_fid",
    "LIGHTCURVES_HIGHZ_FIDUCIAL",
    "LC_FH_RESOLVED",
    "path_file_lc_highz_fid",
    "name_file_lc_highz_fid",
    "name_file_mcmc",
    "mcmc_dir",
    "mcmc_notes",
    "load_paper_emulators",
    "path_out",
    "path_plots",
    "path_file_halo_masses",
    "name_file",
    "name_file_halos",
    "max_snap",
    "boxsize",
    "simulation_name",
    "source_dir",
    "subset_tag",
    "load_simulation_metadata",
    "load_simulation_metadata_full_history",
    "luminosity_function_auto",
    "mass_function_auto",
    "weighted_median",
    "weighted_percentile",
]


def print_fiducial():
    """Print the pinned fiducial selection + the resolved evolution paths."""
    print("[plotting_paper] FIDUCIAL data run:")
    for key, val in RESOLVED.items():
        tag = "" if val == PAPER_FIDUCIAL[key] else "  (BAQARO_PAPER override)"
        print(f"    {key} = {val!r}{tag}")
    print(f"    -> path_file = {path_file}")
    print("[plotting_paper] FULL-HISTORY fiducial (single-tree figure):")
    for key, val in FH_RESOLVED.items():
        tag = "" if val == FULL_HISTORY_FIDUCIAL[key] else "  (BAQARO_PAPER_FH override)"
        print(f"    {key} = {val!r}{tag}")
    print(f"    -> path_file_full_history_fid = {path_file_full_history_fid}")
    print("[plotting_paper] DUTY-CYCLE fiducial (z~6 duty-cycle figure):")
    for key, val in DC_RESOLVED.items():
        tag = "" if val == DUTY_CYCLE_FIDUCIAL[key] else "  (BAQARO_PAPER_DC override)"
        print(f"    {key} = {val!r}{tag}")
    print(f"    -> path_file_duty_cycle_fid = {path_file_duty_cycle_fid}")
    print("[plotting_paper] MCMC fiducial (corner figure):")
    for key, val in MCMC_RESOLVED.items():
        tag = "" if val == MCMC_FIDUCIAL[key] else "  (BAQARO_PAPER_MCMC override)"
        print(f"    {key} = {val!r}{tag}")
    print(f"    -> chains: {mcmc_dir}/mcmc_{name_file_mcmc}_<combo>_{mcmc_notes}.h5")
