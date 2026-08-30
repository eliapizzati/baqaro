"""Emulator fidelity check: REAL main_evolution vs emulator prediction at
registered best-fit parameter sets.

For each requested bestfit (a key in ``BESTFIT_REGISTRY``) this:
  1. looks up the 6 params from the registry,
  2. finds the matching ``main_evolution`` output (``_bestfit_<key>_sub_<tag>``),
  3. bins the REAL QLF / BHMF / CERDF / QHMF with the SAME bins+normalizations
     the emulator was trained on (from the training-schema ``config_json``),
  4. predicts the emulator at the same params,
  5. overlays real (black dashed) vs emulator (colour solid) per redshift and
     prints a median-absolute-error / RMS table (the robust + outlier-sensitive
     fidelity metrics).

This is the standard version of earlier one-off scripts: env-driven model identifier, registry-driven bestfits, glob-based
file discovery, optional QHMF (needs the halo-mass subset cache).

Env
---
  BAQARO_BESTFIT_NAMES=qcc_ck22final_v1   (registry keys; comma-sep)
  BAQARO_PLOT_FLAGS=qlf,bhmf,cerdf,qhmf                          (default: all available)
  BAQARO_SIM / BAQARO_MAX_SNAP / BAQARO_FOLD_SUBHALO_MASS / BAQARO_MERGER_DELAY_MODE
  BAQARO_NOTES_FILE_EMULATION  (emulator notes; e.g. newfast_fid3_10k_narrowprior_smooth)
  BAQARO_NOTES_FILE_TRAINING   (training-schema notes; defaults to EMULATION with
                               a trailing '_smooth' stripped, since smoothing is
                               an emulation-time op and the schema lives in the
                               unsmoothed training file)
  BAQARO_SUBSET_TAG            (subsample tag; e.g. root60_flatN500000_K22_logM10.0to15.5_seed42_v3)
  BAQARO_HEADLESS=1            (skip plt.show, save PDFs)
"""
import os
import glob
import json
import numpy as np
import h5py
import matplotlib
if os.environ.get("BAQARO_HEADLESS", "0") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baqaro.emulation.emulation_core_functions import GeneralEmulatorGP
from baqaro.core_functions.bestfit_registry import get_bestfit, BESTFIT_REGISTRY
from baqaro.utils.my_dir import get_output_path, get_plots_path
from baqaro.plotting_common.plot_config import fig_ext, resolve_plots_dir
from baqaro.utils.sim_config import (
    simulation_name,
    fold_subhalo_mass as fold,
    merger_delay_mode as merger_mode,
    growth_feff_suffix,
)
import qhtools.utils.natconst as nc
from qhtools.utils.my_utils import to_solar


# ==========================================
# CONFIG (env-driven model identifier; mirrors main_mcmc / comparison_config)
# ==========================================
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
# Defaults track sim_config, NOT a frozen literal, so a bare run resolves to
# the fiducial emulator family.
from baqaro.utils.sim_config import (
    max_snap as _sim_max_snap, FIDUCIAL_NOTES_EMULATION as _fid_notes_emul,
)
max_snap = int(os.environ.get("BAQARO_MAX_SNAP", str(_sim_max_snap)))
erdf_model = "log_normal_evol_halo_mass"
# Default to the production subsampled run's tag (auto-derived from
# sim_config geometry + subsample env defaults) so callers needn't set
# BAQARO_SUBSET_TAG. Env presence wins: BAQARO_SUBSET_TAG="" => full-sim.
if "BAQARO_SUBSET_TAG" in os.environ:
    subset_tag = os.environ["BAQARO_SUBSET_TAG"]
else:
    from baqaro.core_functions.tree_subsample import resolve_subset_tag
    subset_tag = resolve_subset_tag(max_snap)
# Cross-comparison overrides: validate a z=0 (maxsnap=144/root144) emulator against
# a SHORTER real run (e.g. maxsnap=71/root71 to z=2). The emulator + training schema
# still load at max_snap/subset_tag, but the REAL main_evolution file is found at
# these evo_* values, and schema snapshots beyond evo_max_snap are skipped (the
# short run doesn't have them). Default = identical to max_snap/subset_tag.
evo_max_snap = int(os.environ.get("BAQARO_EVO_MAX_SNAP", str(max_snap)))
evo_subset_tag = os.environ.get("BAQARO_EVO_SUBSET_TAG", subset_tag)
notes_emul = os.environ.get("BAQARO_NOTES_FILE_EMULATION", _fid_notes_emul)
notes_train = os.environ.get("BAQARO_NOTES_FILE_TRAINING",
                             notes_emul[:-7] if notes_emul.endswith("_smooth") else notes_emul)

PARAM_ORDER = ["log_eta_mean_0", "log_eta_mean_evol", "std_0", "logtcoherence", "logfseed", "sigmaseed"]
PHYS = -9.5  # log-floor: bins at/below this are "empty" in both real and emulator
ZMAX_PLOT = 6.3
_COLORS = ["#0072C1", "#E69F00", "#E60068", "#009E73", "#7B3FF2", "#56B4E9"]

out = get_output_path(source=source_dir)  # already ends in .../output_data/bh_evolution
EVO_DIR = os.path.join(out, "evolution")
EMU_DIR = os.path.join(out, "emulators")
TRAIN_DIR = os.path.join(out, "training")
HALO_DIR = os.path.join(out, "halo_subsets")


def _name_file(notes):
    nf = f"{simulation_name}_erdf_{erdf_model}_maxsnap_{max_snap}"
    if fold:
        nf += "_foldmass"
    if merger_mode != "instant_old":
        nf += f"_{merger_mode}"
    if notes is not None:
        nf += f"_{notes}"
    if subset_tag:
        nf += f"_sub_{subset_tag}"
    nf += growth_feff_suffix()
    return nf


NAME_EMUL = _name_file(notes_emul)
NAME_TRAIN = _name_file(notes_train)
# Short name for figure filenames (matches comparison_config / corner /
# results_comparison convention) — strips the verbose _erdf_<model> and
# _sub_<tag> tokens that otherwise make the saved name ~150 chars.
SHORT_NAME = f"{simulation_name}_snap{max_snap}"
if notes_emul:
    SHORT_NAME += f"_{notes_emul}"

# Which bestfits to compare (registry keys).
_bf_env = os.environ.get("BAQARO_BESTFIT_NAMES", "").strip()
_bf_env_all = _bf_env.lower() == "all"
if _bf_env_all:
    _bf_env = ""
if _bf_env:
    BESTFITS = [k.strip() for k in _bf_env.split(",") if k.strip()]
elif _bf_env_all:
    # BAQARO_BESTFIT_NAMES=all -> every registry entry matching sim + max_snap,
    # including ones with no forward run (they are skipped with a warning).
    BESTFITS = [k for k, e in BESTFIT_REGISTRY.items()
                if e.get("sim") == simulation_name and e.get("max_snap") == max_snap]
else:
    # Default: registry entries matching this sim + max_snap that ALSO have a
    # forward run on disk. Matching on sim/max_snap alone selected 62 entries of
    # which only 3 had a run, so a bare invocation hunted 59 non-existent files
    # and then overlaid every survivor -- producing multi-MB figures nobody
    # asked for. Filtering on "a run exists" makes a bare run plot exactly what
    # it is able to plot.
    _evo_dir = os.path.join(out, "evolution")
    BESTFITS = [k for k, e in BESTFIT_REGISTRY.items()
                if e.get("sim") == simulation_name and e.get("max_snap") == max_snap
                and glob.glob(os.path.join(_evo_dir, f"*_bestfit_{k}_*"))]
if not BESTFITS:
    raise SystemExit("No bestfits selected (set BAQARO_BESTFIT_NAMES or check sim/max_snap).")

_plot_env = os.environ.get("BAQARO_PLOT_FLAGS", "").strip()
WANT = ({s.strip() for s in _plot_env.split(",") if s.strip()}
        if _plot_env else {"qlf", "bhmf", "cerdf", "qhmf"})


# ==========================================
# TRAINING SCHEMA (bins + normalizations + axes)
# ==========================================
_train_path = os.path.join(TRAIN_DIR, f"training_data_emulation_{NAME_TRAIN}.hdf5")
if not os.path.exists(_train_path):
    raise SystemExit(f"Training schema not found: {_train_path}\n(set BAQARO_NOTES_FILE_TRAINING?)")
with h5py.File(_train_path, "r") as f:
    sch = json.loads(np.asarray(f["schema/config_json"]).item())
sn = sch["snapshots_to_save"]
zs = np.asarray(sch["redshifts"])[sn]
qbins = np.asarray(sch["qlf_bins"]);  qn = float(sch["qlf_normalization"]);  Lc = np.asarray(sch["log_lbins"])
mbins = np.asarray(sch["bhmf_bins"]); mn = float(sch["bhmf_normalization"]); Mc = np.asarray(sch["log_mbins_bhmf"])
cbins = np.asarray(sch["cerdf_bins"]); cn = float(sch["cerdf_normalization"]); Ec = np.asarray(sch["log_bins_cerdf"])
hbins = np.asarray(sch["qhmf_bins"]); hn = float(sch["qhmf_normalization"]); Hc = np.asarray(sch["log_mbins_qhmf"])
Lthr = np.asarray(sch["log_L_thresholds"]); MU = float(sch["mass_units"])
LOGCSI10 = 10 ** nc.log_csi
Lcuts = 10 ** np.array([to_solar(t) for t in Lthr])
NZ = len(sn)


# ==========================================
# EMULATORS (load only those requested + present)
# ==========================================
def _load_emu(q):
    p = os.path.join(EMU_DIR, f"emulator_{q}_{NAME_EMUL}.xz")
    if not os.path.exists(p):
        print(f"  [skip {q}] emulator not found: {os.path.basename(p)}")
        return None
    e = GeneralEmulatorGP.load(p); e.precompute_alpha(); return e

EMU = {q: _load_emu(q) for q in ("qlf", "bhmf", "cerdf", "qhmf") if q in WANT}
EMU = {q: e for q, e in EMU.items() if e is not None}

# Halo-mass subset cache (needed for QHMF real binning).
_halo_glob = glob.glob(os.path.join(HALO_DIR, f"subset_{simulation_name}_maxsnap{evo_max_snap}_*{evo_subset_tag}*__halo_masses.npy"))
halo_all = np.load(_halo_glob[0], mmap_mode="r") if _halo_glob else None
if "qhmf" in EMU and halo_all is None:
    print("  [warn] no halo-mass subset cache found -> QHMF real binning disabled")


def _evolution_file(key):
    g = glob.glob(os.path.join(EVO_DIR, f"bh_evolution_*maxsnap_{evo_max_snap}_*_bestfit_{key}_sub_*{evo_subset_tag}*.hdf5"))
    return g[0] if g else None


def _real_and_pred(key, evf, par):
    """Return {stat: (x, real_logphi[NZ,...], emu_logphi[NZ,...])}."""
    res = {}
    preds = {q: EMU[q].predict_mean_only(par.reshape(1, -1))[0] for q in EMU}
    dq = np.full((NZ, len(Lc)), -np.inf); db = np.full((NZ, len(Mc)), -np.inf)
    dc = np.full((NZ, len(Lthr), len(Ec)), -np.inf); dh = np.full((NZ, len(Lthr), len(Hc)), -np.inf)
    with h5py.File(evf, "r") as f:
        w = np.asarray(f["subset/weights"], float)
        n_snap_real = f["Lbols_all"].shape[0]
        for iz, s in enumerate(sn):
            if s >= n_snap_real:
                continue  # schema snapshot beyond this (shorter) real run -> leave -inf, skipped downstream
            L = np.asarray(f["Lbols_all"][s], float)
            bh = np.asarray(f["black_hole_masses_all"][s], float)
            if "qlf" in EMU:
                c, _ = np.histogram(L, qbins, weights=w)
                with np.errstate(divide="ignore"): dq[iz] = np.where(c * qn > 0, np.log10(c * qn), -np.inf)
            if "bhmf" in EMU:
                mok = bh > 0
                c2, _ = np.histogram(bh[mok], mbins, weights=w[mok])
                with np.errstate(divide="ignore"): db[iz] = np.where(c2 * mn > 0, np.log10(c2 * mn), -np.inf)
            if "cerdf" in EMU or ("qhmf" in EMU and halo_all is not None):
                edd = np.zeros_like(bh); vb = bh > 0; edd[vb] = (L[vb] / bh[vb]) / LOGCSI10
                hm = np.asarray(halo_all[s], float) * MU if halo_all is not None else None
                above = L > Lcuts[0]
                ea, la, wa = edd[above], L[above], w[above]
                ha = hm[above] if hm is not None else None
                for j in range(len(Lthr)):
                    if j == 0:
                        es, ws = ea, wa; hs = ha
                    else:
                        cm = la > Lcuts[j]; es, ws = ea[cm], wa[cm]; hs = ha[cm] if ha is not None else None
                    if "cerdf" in EMU:
                        with np.errstate(divide="ignore"):
                            cc, _ = np.histogram(es, cbins, weights=ws); dc[iz, j] = np.where(cc * cn > 0, np.log10(cc * cn), -np.inf)
                    if "qhmf" in EMU and hs is not None:
                        with np.errstate(divide="ignore"):
                            qq, _ = np.histogram(hs, hbins, weights=ws); dh[iz, j] = np.where(qq * hn > 0, np.log10(qq * hn), -np.inf)
    if "qlf" in EMU:  res["qlf"] = (Lc, dq, preds["qlf"])
    if "bhmf" in EMU: res["bhmf"] = (Mc, db, preds["bhmf"])
    if "cerdf" in EMU: res["cerdf"] = (Ec, dc, preds["cerdf"])
    if "qhmf" in EMU and halo_all is not None: res["qhmf"] = (Hc, dh, preds["qhmf"])
    return res


def _medae_rms(pred, direct, reg=None):
    real = (direct > PHYS + 0.1) & np.isfinite(pred)
    if reg is not None: real &= reg
    if not real.any(): return (np.nan, np.nan, 0)
    d = (pred - direct)[real]
    return (np.median(np.abs(d)), np.sqrt(np.mean(d ** 2)), int(real.sum()))


# ==========================================
# RUN: discover files, compute, tabulate
# ==========================================
RES = {}
print(f"\nEmulator: {NAME_EMUL}")
print(f"Stats: {sorted(EMU)}   Bestfits: {BESTFITS}\n")
print("========== emulator vs REAL main_evolution — medAE/RMS (dex) ==========")
for ki, key in enumerate(BESTFITS):
    evf = _evolution_file(key)
    if evf is None:
        print(f"[{key}] NO main_evolution output found (run it with BAQARO_BESTFIT_NAME={key}) — skipping")
        continue
    par = np.array([get_bestfit(key)[p] for p in PARAM_ORDER])
    RES[key] = _real_and_pred(key, evf, par)
    print(f"\n--- {key} ---  ({BESTFIT_REGISTRY[key].get('label', '')})")
    for iz, zz in enumerate(zs):
        if zz > ZMAX_PLOT: continue
        line = f"  z={zz:4.2f} "
        if "qlf" in RES[key]:
            _, dq, pq = RES[key]["qlf"]; regQ = (Lc >= 45.5) & (Lc <= 47.5)
            m, r, _ = _medae_rms(pq[iz], dq[iz], regQ); line += f" QLF medAE={m:.3f}/RMS={r:.3f}"
        if "bhmf" in RES[key]:
            _, db, pb = RES[key]["bhmf"]; m, r, _ = _medae_rms(pb[iz], db[iz]); line += f"  BHMF medAE={m:.3f}/RMS={r:.3f}"
        if "cerdf" in RES[key]:
            _, dc, pc = RES[key]["cerdf"]; m, _, _ = _medae_rms(pc[iz, 0], dc[iz, 0]); line += f"  CERDF medAE={m:.3f}"
        if "qhmf" in RES[key]:
            _, dh, ph = RES[key]["qhmf"]; m, _, _ = _medae_rms(ph[iz, 0], dh[iz, 0]); line += f"  QHMF medAE={m:.3f}"
        print(line)

if not RES:
    raise SystemExit("\nNo main_evolution outputs matched any bestfit — nothing to plot.")


# ==========================================
# FIGURES: one per quantity, rows=bestfit, cols=redshift
# ==========================================
keys_ok = [k for k in BESTFITS if k in RES]
zsel = [iz for iz, zz in enumerate(zs) if zz <= ZMAX_PLOT]
QUANT_LABELS = {
    "qlf":   (r"$\log_{10}$ L$_{\rm bol}$ [erg/s]", r"$\log\,\phi_{\rm QLF}$", (43.5, 48.5), (-9.8, -2.5), False),
    "bhmf":  (r"$\log_{10}$ M$_{\rm BH}$ [M$_\odot$]", r"$\log\,\phi_{\rm BHMF}$", (6.0, 11.0), (-10.5, -1.5), False),
    "cerdf": (r"$\log_{10}\,\lambda_{\rm Edd}$ (faintest L-thr)", r"$\log\,\phi_{\rm CERDF}$", None, None, True),
    "qhmf":  (r"$\log_{10}$ M$_{\rm halo}$ [M$_\odot$] (faintest L-thr)", r"$\log\,\phi_{\rm QHMF}$", None, None, True),
}
_saved = []
_present = [s for s in ("qlf", "bhmf", "cerdf", "qhmf") if any(s in RES[k] for k in keys_ok)]
for q in _present:
    xl, yl, xlim, ylim, is3d = QUANT_LABELS[q]
    nr, nc = len(keys_ok), len(zsel)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.6 * nr), squeeze=False, sharex=True, sharey=True)
    for ri, key in enumerate(keys_ok):
        if q not in RES[key]:
            continue
        x, real, emu = RES[key][q]
        color = _COLORS[ri % len(_COLORS)]
        for ci, iz in enumerate(zsel):
            ax = axs[ri][ci]
            r_arr = real[iz, 0] if is3d else real[iz]
            e_arr = emu[iz, 0] if is3d else emu[iz]
            mreal = r_arr > PHYS + 0.1
            ax.plot(x[mreal], r_arr[mreal], ls="--", lw=2.0, color="black", zorder=5)
            ax.plot(x, e_arr, ls="-", lw=2.0, color=color, alpha=0.9, zorder=4)
            if ri == 0: ax.set_title(f"z = {zs[iz]:.2f}", fontsize=10)
            if ci == 0: ax.set_ylabel(f"{key}\n{yl}", fontsize=7)
            if ri == nr - 1: ax.set_xlabel(xl, fontsize=8)
            if xlim: ax.set_xlim(*xlim)
            if ylim: ax.set_ylim(*ylim)
    fig.suptitle(f"{q.upper()}: REAL main_evolution (black dashed) vs emulator (colour solid)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if os.environ.get("BAQARO_HEADLESS", "0") == "1" or os.environ.get("BAQARO_SAVE_FIGS", "0") == "1":
        d = resolve_plots_dir(subdir="emulator_vs_real", source=source_dir)
        p = os.path.join(d, f"emulator_vs_real_{q}_{SHORT_NAME}.{fig_ext()}")
        fig.savefig(p, bbox_inches="tight"); _saved.append(p); print(f"saved: {p}")

if _saved:
    print("\nSaved:")
    for p in _saved: print(f"  {p}")
if os.environ.get("BAQARO_HEADLESS", "0") != "1":
    plt.show()
