"""
PAPER FIGURE: quasar bias b_Q vs redshift, in bolometric-luminosity bins.

Model curve = the effective large-scale linear bias b_Q(z, Lbin) = weight-averaged
Tinker10 host-halo bias of the pinned forward run's quasars. **SELF-CONTAINED &
SWITCH-FRIENDLY:** the model line is generated in-script from the fiducial run and
cached PER-RUN as ``quasar_bias_model_{name_file}.npz``; switching the fiducial
(subsample / multinode / another fit, via BAQARO_PAPER_*) auto-picks or regenerates the
matching cache. No dependence on the ``clustering_direct/`` follow-up tools. The
optional direct-b_Q overlay here (``BAQARO_QBIAS_NO_DIRECT=0``) still reads the
multinode caches written by ``clustering_direct/measure_bias_direct.py`` (the
DIRECT ξ_QQ bias), model-consistency-guarded.

Observations overlaid from the literature compilation
(obs_data/quasar_bias_obs_data.py — VERIFY those values), incl. DESI DR2
(Charles+26) Table-4 points. Everything coloured by (effective) log L_bol; obs
that quote M1450 get L_bol from qhtools (Runnoe+12). A low-z zoom inset resolves
the crowded 2<z<3.5 region.

Env knobs: BAQARO_QBIAS_MODEL_ZTARGETS (comma z-grid for the model line, default a
27-pt grid mapped to nearest stored snapshot); BAQARO_QBIAS_NO_DIRECT (default 1).

Run:
  env BAQARO_SAVE_FIGS=1 BAQARO_HEADLESS=1 \\
      python -m baqaro.plotting_paper.plotting_quasar_bias
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

from baqaro.obs_data.quasar_bias_obs_data import DATA_ALL, DATA_DESI_SB
from qhtools.utils.magnitude_conversion import get_log_Lbol_from_M1450, get_M1450_from_log_Lbol
from baqaro.utils.my_dir import get_output_path
from baqaro.clustering_direct.cache_provenance import (
    verify_cache, source_bestfit, provenance_dict,
)
# Pinned paper fiducial (imported FIRST so the BAQARO_* run identity is set before
# we read it; switch it via BAQARO_PAPER_*). path_file + name_file identify the
# forward run the MODEL bias line is generated from.
from baqaro.plotting_paper.fiducial_data import (
    path_file, name_file, load_simulation_metadata, snapshot_index_for_redshift,
)
from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show

# Paper default: the CLEAN version — model curves + obs only, no this-work direct
# points (those need a matching-model multinode run).
# Opt into the direct overlay with BAQARO_QBIAS_NO_DIRECT=0 (saved under a distinct
# name so both coexist on disk).
NO_DIRECT = os.environ.get("BAQARO_QBIAS_NO_DIRECT", "1") == "1"
name_fig = "quasar_bias_vs_z" if NO_DIRECT else "quasar_bias_vs_z_direct"
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")
outdir = os.environ.get("BAQARO_CLUST_OUTDIR") or os.path.join(
    get_output_path(source_dir), "clustering_direct")

# --- MODEL bias line: b_Q(z, Lbin) = Σ w_i b_h(M_host,i, z)/Σ w_i (weight-averaged
# Tinker10 host bias). Generated SELF-CONTAINED from the pinned forward run and
# cached PER-RUN (keyed by name_file), so switching the fiducial (subsample /
# multinode / another fit) auto-(re)generates the right line — no dependence on the
# clustering_direct follow-up tools. ---
_MODEL_LBINS = [(45.5, 46.0), (46.0, 46.5), (46.5, 47.0), (47.0, 48.0)]
# z-target grid, mapped to nearest STORED snapshot: 145-snap subsample -> ~27
# unique cols; 13-snap multinode -> 13 (self-adapts to the run's coverage).
# Override with BAQARO_QBIAS_MODEL_ZTARGETS="0,1,2,..." (comma list).
_zt_env = os.environ.get("BAQARO_QBIAS_MODEL_ZTARGETS", "").strip()
_MODEL_ZTARGETS = ([float(x) for x in _zt_env.split(",")] if _zt_env else
                   [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5,
                    2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.3, 4.6, 5.0, 5.4, 5.8, 6.15,
                    6.5, 7.0])


def _build_model_bias(out_path):
    """Compute + cache the model b_Q(z, Lbin) from the pinned forward run."""
    import h5py
    from colossus.lss import bias
    from qhtools.utils.cosmology import cosmo
    from qhtools.utils import my_utils
    print(f"  [bias model] cache miss -> generating from "
          f"{os.path.basename(path_file)}", flush=True)
    with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as f:
        meta = load_simulation_metadata(f)
        loader = meta["loader"]
        redshifts = np.asarray(meta["redshifts"])
        snapshots = meta["snapshots"]
        cols = sorted({snapshot_index_for_redshift(
            redshifts, t, snapshots, label="quasar_bias") for t in _MODEL_ZTARGETS})
        nb, nz = len(_MODEL_LBINS), len(cols)
        z_arr = np.array([redshifts[c] for c in cols], dtype=float)
        bQ_ = np.full((nb, nz), np.nan)
        n_ = np.zeros((nb, nz), dtype=np.int64)
        med_ = np.full((nb, nz), np.nan)
        for j, col in enumerate(cols):
            zc = float(redshifts[col])
            Lbol = np.asarray(loader.get_Lbol(col))
            Mh = np.asarray(loader.get_Halo_mass(col))
            w = (np.asarray(loader.weights, dtype=float)
                 if loader.weights is not None else np.ones_like(Lbol, dtype=float))
            for b, (llo, lhi) in enumerate(_MODEL_LBINS):
                m = ((Lbol > 10 ** my_utils.to_solar(llo))
                     & (Lbol <= 10 ** my_utils.to_solar(lhi)))
                n = int(m.sum()); n_[b, j] = n
                if n < 20:
                    continue
                Mm = Mh[m]; wm = w[m]
                bh = bias.haloBias(Mm * cosmo.h, zc, model="tinker10", mdef="200m")
                bQ_[b, j] = np.sum(wm * bh) / np.sum(wm)
                med_[b, j] = np.log10(my_utils.get_weighted_median(Mm, wm)
                                      if hasattr(my_utils, "get_weighted_median")
                                      else np.median(Mm))
            print(f"    col {col:3d} z={zc:.2f}", flush=True)
        prov = provenance_dict("model_bias", loader, requires_full_cat=False)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, redshifts=z_arr, cols=np.array(cols),
             lbins_lo=np.array([b[0] for b in _MODEL_LBINS]),
             lbins_hi=np.array([b[1] for b in _MODEL_LBINS]),
             b_Q=bQ_, n_qso=n_, med_logMh=med_, **prov)
    print(f"  [bias model] cached -> {out_path}", flush=True)


# Per-run cache -> switching the fiducial picks/generates the matching file.
_model_cache = os.path.join(outdir, f"quasar_bias_model_{name_file}.npz")
if not os.path.exists(_model_cache):
    _build_model_bias(_model_cache)
d = np.load(_model_cache, allow_pickle=True)
verify_cache(d, label="bias model")
_MODEL_BF = source_bestfit(d)      # model identity the direct points must match
z = d["redshifts"]; bQ = d["b_Q"]; lo = d["lbins_lo"]; hi = d["lbins_hi"]
lbin_c = 0.5 * (lo + hi)
order = np.argsort(z)


def _load_direct(fname, label):
    """Load a DIRECT b_Q cache if present, valid (full-cat), AND from the SAME
    model as the bias-model curve. A direct measurement only exists for a run with
    a multinode full-cat version; if the plot's model has none, the direct
    points are correctly SKIPPED rather than borrowed from a different model's
    multinode run."""
    p = os.path.join(outdir, fname)
    if NO_DIRECT or not os.path.exists(p):
        return None
    npz = np.load(p, allow_pickle=True)
    verify_cache(npz, label=label, expect_full_cat=True)
    bf = source_bestfit(npz)
    if _MODEL_BF is not None and bf is not None and bf != _MODEL_BF:
        print(f"  [{label}] SKIP: source model '{bf}' != plot model '{_MODEL_BF}' "
              "(no matching-model multinode run — direct points not shown).")
        return None
    return npz


# Direct b_Q from measured xi_QQ (open circles) / cross-corr (open squares).
DIRECT = _load_direct("quasar_bias_direct_xiQQ.npz", "direct b_Q (auto)")
DIRECT_X = _load_direct("quasar_bias_direct_cross.npz", "direct b_Q (cross)")

# DISCRETE colour scale = the model's L_bol bin edges, so each bin is one colour.
LBOUNDS = np.array([45.5, 46.0, 46.5, 47.0, 48.0])   # log10 L_bol bin edges [erg/s]
_base = plt.get_cmap("plasma")
DISC_COLORS = [_base(x) for x in (0.06, 0.38, 0.66, 0.90)]  # 4 distinct colours
cmap = ListedColormap(DISC_COLORS)
norm = BoundaryNorm(LBOUNDS, cmap.N, clip=True)     # obs below 45.5 / above 48 clamp
sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

# per-survey marker (fixed so DESI stays a distinct diamond regardless of order)
REF_MARKER = {
    "Croom+05": "o", "Ross+09": "s", "da Angela+08": "^", "Laurent+17": "<",
    "White+12": ">", "Eftekharzadeh+15": "H", "DESI DR2 (Charles+26)": "D",
    "Shen+07": "P", "He+18": "X", "EIGER (Eilers+24)": "*", "ASPIRE (Huang+26)": "h",
    "Porciani&Norberg+06": "d", "Myers+06": "p", "Ikeda+15": "v",
}
# surveys to hide in the clean (no-direct) version (data kept, just not plotted)
SUPPRESS_REFS = {"Ikeda+15", "He+18"} if NO_DIRECT else set()
refs = []
for p in DATA_ALL:
    if p.ref not in refs and p.ref not in SUPPRESS_REFS:
        refs.append(p.ref)
# order the obs legend entries by (representative) redshift, low -> high, so the
# legend reads in the same direction as the data along the x-axis.
_ref_meanz = {r: np.mean([p.z for p in DATA_ALL if p.ref == r]) for r in refs}
refs.sort(key=lambda r: _ref_meanz[r])
ref_marker = {r: REF_MARKER.get(r, "o") for r in refs}


def point_logL(p):
    """log10 L_bol for one observed point, from M1450 when available.

    Prefers the M1450 -> L_bol conversion in qhtools (authoritative, and the
    same one the model uses) and falls back to the compilation's own quoted
    bolometric luminosity.
    """
    if np.isfinite(p.M1450):
        return get_log_Lbol_from_M1450(p.M1450)   # authoritative, via qhtools
    return p.logLbol


# DESI DR2 luminosity-dependent fit (Charles+26 Eq 4.2, 'equal numbers'):
#   b_Q(z,M1450) = a[(1+z)^2 - 6.565] + b - m(M1450+24),  valid 2<z<3.5.
A_D, B_D, M_D = 0.2056, 2.5381, 0.0998
_zf = np.linspace(2.0, 3.5, 60)


def draw(a, ms=6.0):
    """Draw the model bias curves and the observed points onto one axis."""
    # model curves (solid, one per L bin)
    for b in range(len(lo)):
        gg = np.isfinite(bQ[b])[order]
        a.plot(z[order][gg], bQ[b][order][gg], "-", lw=2.2,
               color=cmap(norm(lbin_c[b])), zorder=5)
    # DESI DR2 fit at OUR L bins (dashed, colour-matched; L_bol -> M1450 via qhtools)
    for b in range(len(lo)):
        M1450 = get_M1450_from_log_Lbol(float(lbin_c[b]))
        a.plot(_zf, A_D * ((1 + _zf) ** 2 - 6.565) + B_D - M_D * (M1450 + 24.0),
               "--", lw=1.5, color=cmap(norm(lbin_c[b])), zorder=6)
    # observations
    for p in DATA_ALL:
        if p.ref in SUPPRESS_REFS:
            continue
        lL = point_logL(p)
        c = cmap(norm(lL)) if np.isfinite(lL) else "0.5"
        if getattr(p, "uplim", False):        # upper limit -> down-arrow
            a.errorbar(p.z, p.b, yerr=p.b * 0.22, uplims=True, fmt=ref_marker[p.ref],
                       ms=ms, mfc=c, mec="k", ecolor=c, elinewidth=0.9, mew=0.6, zorder=8)
        else:
            a.errorbar(p.z, p.b, yerr=p.err, fmt=ref_marker[p.ref], ms=ms, mfc=c, mec="k",
                       ecolor=c, elinewidth=0.9, capsize=2, alpha=0.9, zorder=8,
                       mew=0.6 if not p.approx else 0.4)
    # DESI DR2 'same boundaries' scheme — open diamonds (binning-scheme comparison)
    if not NO_DIRECT:
        for p in DATA_DESI_SB:
            c = cmap(norm(point_logL(p)))
            a.errorbar(p.z, p.b, yerr=p.err, fmt="D", ms=ms - 1.5, mfc="none", mec=c,
                       ecolor=c, elinewidth=0.7, capsize=1.5, mew=1.0, alpha=0.85, zorder=7)
    # direct b_Q from the measured xi_QQ — open circles (this work). Shown only
    # where the auto-corr is well-measured (z <= AUTO_ZMAX); at higher z the
    # pair-starved auto is shot-noise-dominated and the CROSS squares take over.
    AUTO_ZMAX = 3.5
    if DIRECT is not None:
        zd = DIRECT["redshifts"]; bd = DIRECT["b_Q"]; be = DIRECT["b_Q_err"]
        lcd = 0.5 * (DIRECT["lbins_lo"] + DIRECT["lbins_hi"])
        for bb in range(bd.shape[0]):
            for jj in range(bd.shape[1]):
                if np.isfinite(bd[bb, jj]) and zd[jj] <= AUTO_ZMAX:
                    c = cmap(norm(lcd[bb]))
                    a.errorbar(zd[jj] + (bb - 1.5) * 0.045, bd[bb, jj], yerr=be[bb, jj],
                               fmt="o", ms=ms + 1.5, mfc="white", mec=c, ecolor=c,
                               mew=1.7, capsize=2, zorder=11)
    # direct b_Q from the CROSS-corr xi_Q,ref (dense lower-mass tracer) — open
    # squares (this work). The high-z / bright-bin measurement that survives the
    # auto-corr shot noise. Tiny +z offset so it doesn't sit exactly on the circles.
    if DIRECT_X is not None:
        zx = DIRECT_X["redshifts"]; bx = DIRECT_X["b_Q"]; bxe = DIRECT_X["b_Q_err"]
        lcx = 0.5 * (DIRECT_X["lbins_lo"] + DIRECT_X["lbins_hi"])
        for bb in range(bx.shape[0]):
            for jj in range(bx.shape[1]):
                if np.isfinite(bx[bb, jj]):
                    c = cmap(norm(lcx[bb]))
                    a.errorbar(zx[jj] + (bb - 1.5) * 0.045 + 0.12, bx[bb, jj], yerr=bxe[bb, jj],
                               fmt="s", ms=ms + 0.5, mfc="white", mec=c, ecolor=c,
                               mew=1.7, capsize=2, zorder=12)


fig, ax = plt.subplots(figsize=(6.0, 5.0))   # match the other single-panel paper figs
draw(ax, ms=6.0)
ax.set_xlabel(r"redshift $z$")
ax.set_ylabel(r"quasar bias $b_Q$")
ax.set_xlim(-0.2, 8.0)
ax.set_ylim(0.5, 22.0)
ax.minorticks_on()

cbar = fig.colorbar(sm, ax=ax, pad=0.015, boundaries=LBOUNDS, ticks=LBOUNDS,
                    spacing="uniform", drawedges=True)
cbar.set_label(r"$\log_{10} L_{\rm bol}\,[{\rm erg\,s^{-1}}]$")

fig.tight_layout()

# --- model/fit LINES: small legend INSIDE the panel, top-right (empty high-b band;
# the inset takes the upper-left). ---
model_handles = [
    Line2D([], [], color="k", lw=2.2, label="BAQARO"),
    Line2D([], [], color="0.4", lw=1.5, ls="--", label="DESI DR2 fit"),
]
ax.legend(handles=model_handles, loc="upper right", fontsize=10.0, frameon=False,
          markerfirst=False, handletextpad=0.5, handlelength=1.9,
          labelspacing=0.5, borderpad=0.5)

# --- obs POINTS: bottom external legend (3 cols x 4 rows). ---
obs_handles = ([] if NO_DIRECT else [
    Line2D([], [], ls="", marker="o", mfc="white", mec="k", mew=1.5, ms=7,
           label=r"direct $b_Q$ ($\xi_{QQ}$ auto, this work)"),
    Line2D([], [], ls="", marker="s", mfc="white", mec="k", mew=1.5, ms=6,
           label=r"direct $b_Q$ ($\xi_{Q\,{\rm h}}$ cross, this work)"),
    Line2D([], [], ls="", marker="D", mfc="none", mec="k", mew=1.0, ms=6,
           label="DESI DR2 (same bounds)"),
]) + [Line2D([], [], ls="", marker=ref_marker[r], mfc="0.7", mec="k", ms=6, label=r)
     for r in refs]
fig.legend(handles=obs_handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
           ncol=3, fontsize=9.5, frameon=False, handletextpad=0.4,
           columnspacing=0.8, labelspacing=0.25)
fig.subplots_adjust(bottom=0.235)

# --- zoom inset over the crowded low-z region (all SDSS/BOSS + DESI). Placed in the
# empty upper-left band (b>11, where no data/curves sit). Created LAST (after
# tight_layout/subplots_adjust finalise the axes); connectors hidden. ---
axins = ax.inset_axes([0.08, 0.58, 0.40, 0.39])
draw(axins, ms=5.0)
axins.set_xlim(1.2, 3.4)
axins.set_ylim(1.3, 5.7)
axins.tick_params(labelsize=8)
axins.minorticks_on()
ind = ax.indicate_inset_zoom(axins, edgecolor="0.45", alpha=0.6)
_conns = getattr(ind, "connectors", None) or (ind[1] if isinstance(ind, tuple) else [])
for _c in _conns:
    _c.set_visible(False)

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
