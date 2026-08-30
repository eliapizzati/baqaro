"""
PAPER FIGURE: gallery of high-z single-BH lightcurves.

A collection of individual black-hole sub-step lightcurves (one cell per BH),
focused on the **high-redshift assembly epoch**. Each cell stacks two panels:
  * TOP: bolometric luminosity L_bol (left, steel blue) + BH mass M_BH (right
    twin, dark red), with the Salpeter e-folding reference (full + 1/10 rate)
    from the seed.
  * BOTTOM: the Eddington ratio λ_Edd = L_bol/(M_BH·10^log_csi) (purple) over
    the intrinsic ERDF accretion-rate η (gray).
A secondary top axis maps cosmic time → redshift. The time↔z conversion is
clamped at both ends (see ``time_to_redshift``/``redshift_to_time``) so the
secondary axis never runs out of bounds; the x-window is additionally trimmed
to the high-z epoch (z ≥ ``Z_FOCUS_MIN``).

The drawn BHs are the ``N`` most massive at z≈6 among the first-born targets
(the luminous z~6 QSO progenitors). Individual objects → **no subsample
weighting** (➖ N/A).

Reads the dedicated full-history paper fiducial (``path_file_full_history_fid``
via ``fiducial_data`` — the first-born ``main_evolution_full_history`` run, same
as the single-tree figure), NOT the big subsample forward run.

Env overrides:
  * ``BAQARO_PAPER_LC_RANKS`` — comma list of z≈6 mass ranks to draw
    (default ``0,1,2,3,4,5``; 0 = most massive).
  * ``BAQARO_PAPER_LC_ZMIN``  — high-z focus floor on the time axis
    (default ``3.0``; the right edge is cosmic time at this redshift).
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.ticker import NullLocator

from qhtools.utils.cosmology import cosmo
import qhtools.utils.natconst as nc
from qhtools.utils import my_utils

# High-z lightcurves use their OWN full-history fiducial (massthr13.5_sub z=2),
# distinct from the single-tree FH fiducial (firstborn100 z=0). See the
# docstring of LIGHTCURVES_HIGHZ_FIDUCIAL in fiducial_data.py — the
# z=2 file is needed because the 6 hand-curated IDs (DEFAULT_LC_IDS below)
# come from there.
from baqaro.plotting_paper.fiducial_data import (
    load_simulation_metadata_full_history,
    path_file_lc_highz_fid as path_file_full_history_fid,
)
# Fast column-cache (build once with build_lc_highz_cache.py) — lets a re-render
# skip the ~159 GB/array full-history eager load. Optional: absent cache or a
# selected object outside it falls back to the slow per-column FH read.
from baqaro.plotting_paper.build_lc_highz_cache import lc_highz_cache_path
# Full-N halo_masses mmap + code-unit factor — for the efficient column gather
# (load_data_to_plot is already imported, env-pinned, by fiducial_data above).
from baqaro.plotting_common.load_data_to_plot import (
    path_file_halo_masses,
    mass_units,
)
from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
# Grid + output name are mode-aware. The "gallery" browse mode renders a large
# grid of random candidates from ONE mass band (each cell labelled with its
# storage idx) so you can eyeball many draws and then lock the ones you like in
# via BAQARO_PAPER_LC_MODE=explicit BAQARO_PAPER_LC_IDS=...  The gallery saves to a
# band+seed-stamped filename so it never overwrites the real 3x2 figure.
_mode_top = os.environ.get("BAQARO_PAPER_LC_MODE", "explicit").strip().lower()
if _mode_top == "gallery":
    N_COL = int(os.environ.get("BAQARO_PAPER_LC_GALLERY_NCOL", "4"))
    N_ROW = int(os.environ.get("BAQARO_PAPER_LC_GALLERY_NROW", "3"))
    _gallery_band = os.environ.get("BAQARO_PAPER_LC_GALLERY_BAND", "top").strip().lower()
    _gallery_seed = int(os.environ.get("BAQARO_PAPER_LC_RANDOM_SEED", "42"))
    name_fig = f"lightcurves_highz_gallery_{_gallery_band}_seed{_gallery_seed}"
else:
    N_COL, N_ROW = 3, 2  # 6 lightcurves (top row >1e9, bottom ~1e8)
    name_fig = "lightcurves_highz"

RANKS = [int(r) for r in os.environ.get("BAQARO_PAPER_LC_RANKS", "0,1,2,3,4,5").split(",")]
Z_FOCUS_MIN = float(os.environ.get("BAQARO_PAPER_LC_ZMIN", "3.0"))
Z_SELECT = 6.0  # rank the targets by M_BH at this redshift

# Fixed y-windows (shared by every cell). L_bol is coupled to M_BH via log_csi
# so both axes reach the seed epoch together. The range is extended so
# the figure captures both the seed scale at the formation epoch and the
# late-time growth into the z=2 massive-host regime.
MBH_LO, MBH_HI = 3.5, 10.5

# Colours (match the working version of this figure).
COL_LBOL = "#06437F"   # steel blue
COL_MASS = "#a00c1d"   # dark red
COL_ETA = "#5e4fa2"    # muted purple — Eddington ratio
COL_MDOT = "0.65"      # gray — intrinsic ERDF accretion rate (behind)
COL_HALO = "#2ca02c"   # bright green — seed-mass track f_seed*M_halo (per-snapshot)

T_SALPETER = 0.045  # Gyr, Eddington e-folding time
Z_TICKS = [3, 4, 5, 6, 8, 12, 20]


# ==============================================================================
# TIME-LIMIT HANDLING (clamped time<->redshift; borrowed from the high-z LC)
# ==============================================================================
_TIME_AT_Z_MIN = cosmo.age(0.001)


def time_to_redshift(time):
    """Cosmic time [Gyr] -> redshift, clamped to avoid out-of-bounds."""
    t = np.asanyarray(time)
    t_clamped = np.clip(t, 0.002, _TIME_AT_Z_MIN)
    return cosmo.age(t_clamped, inverse=True)


def redshift_to_time(z):
    """Redshift -> cosmic time [Gyr], clamped at z<0."""
    z_clamped = np.maximum(np.asanyarray(z), 0.001)
    return cosmo.age(z_clamped)


def add_redshift_axis(ax):
    """Secondary top axis labelled in redshift (high-z ticks)."""
    sax = ax.secondary_xaxis("top", functions=(time_to_redshift, redshift_to_time))
    sax.set_xlabel(r"Redshift $z$", labelpad=2)
    sax.set_xticks(Z_TICKS)
    sax.set_xticklabels([f"{z:g}" for z in Z_TICKS])
    sax.xaxis.set_minor_locator(NullLocator())  # no minors on the nonlinear z axis
    return sax


# ==============================================================================
# FIGURE
# ==============================================================================
T_LEFT = 0.15
T_RIGHT = float(cosmo.age(Z_FOCUS_MIN))

# Figure scales with the grid (3x2 -> the canonical 12x6); gallery grids grow.
fig = plt.figure(figsize=(4.0 * N_COL, 3.0 * N_ROW))
outer = gridspec.GridSpec(N_ROW, N_COL, figure=fig, wspace=0.07, hspace=0.13,
                          left=0.055, right=0.95, top=0.91, bottom=0.10)

with h5py.File(path_file_full_history_fid, "r",
               rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as file:
    data = load_simulation_metadata_full_history(file)
    loader = data["loader"]
    redshifts = data["redshifts"]
    fh = data["full_history_loader"]
    targets = data["original_targets_new_indices"]

    # Rank by M_BH at z≈6 across ALL halos in the FH file (closure included),
    # not just the original targets. With a mass_threshold-at-z=2 FH file the
    # original targets are z=2 hosts which may not even exist at z=6 — the
    # luminous z=6 progenitors live in the closure. This branch (default since
    # reliably picks the brightest z=6 BHs.
    # Override with BAQARO_PAPER_LC_RANK_SOURCE=targets to restrict to the
    # original-target selection (old behaviour).
    # Selection mode (default = "two_tier"):
    #   "two_tier":     row-banded random sample by M_BH at z=Z_SELECT. The TOP
    #                   row draws N_COL BHs from [10^TOP_MIN, 10^TOP_MAX) (default
    #                   [1e9, 1e9.2)), the BOTTOM row from [10^BOT_MIN, 10^BOT_MAX)
    #                   (default [1e8, 1e8.2)). NARROW windows just above each round
    #                   number so the rows are genuinely ~1e9 / ~1e8 BHs (not
    #                   "anywhere in the decade"). Bands via
    #                   BAQARO_PAPER_LC_{TOP,BOT}_LOGMBH_{MIN,MAX}.
    #   "gallery":      BROWSE mode — fill a big grid (NROW x NCOL via
    #                   BAQARO_PAPER_LC_GALLERY_{NROW,NCOL}, default 3x4) with random
    #                   draws from ONE band (BAQARO_PAPER_LC_GALLERY_BAND = top|bot|
    #                   "lo,hi"), each cell labelled with its storage id. Saves to a
    #                   band+seed-stamped name (never overwrites the real figure).
    #                   Bump BAQARO_PAPER_LC_RANDOM_SEED for fresh draws; then lock
    #                   your picks in via mode=explicit BAQARO_PAPER_LC_IDS=...
    #   "random_above": uniformly random N_COL*N_ROW BHs with M_BH at z=Z_SELECT
    #                   above BAQARO_PAPER_LC_LOGMBH_MIN (default 9.0). Gives a
    #                   representative sample of the bright z~6 QSO progenitor
    #                   population without bias toward the absolute-most-massive.
    #   "ranks":        the legacy top-N-by-mass behaviour (RANKS env var).
    #   "explicit":     plot a hand-picked ordered list of storage indices, read
    #                   from BAQARO_PAPER_LC_IDS=id0,id1,id2,id3,id4,id5
    #                   (cell order is top-left, top-mid, top-right, bot-left,
    #                   bot-mid, bot-right — row-major). Useful when paging
    #                   through `_seedN.pdf` galleries to assemble a final.
    # Both rank modes filter unphysical runaway BHs above BAQARO_PAPER_LC_LOGMBH_CAP
    # (default 10^11 Msun; the largest observed z=6 BH ~ 10^10 Msun).
    # PAPER FIDUCIAL — the 6 hand-curated z~6 QSO progenitor IDs in the fiducial
    # massthr13.5 z=2 FH file, pinned by storage id and forced with mode=explicit
    # so the layout is identical across re-renders (no need to set
    # BAQARO_PAPER_LC_IDS). Cell order is row-major (TL, TM, TR, BL, BM, BR):
    # top row (~1e9 Msun): 271713, 484, 6594 (logM 9.03/9.15/9.03); bottom row
    # (~1e8 Msun): 74375, 17924, 2131 (logM 8.14/8.07/8.01). They were selected
    # from the `both`-band overview gallery (BAQARO_PAPER_LC_GALLERY_BAND=both).
    DEFAULT_LC_IDS = "271713,484,6594,74375,17924,2131"
    mode = os.environ.get("BAQARO_PAPER_LC_MODE", "explicit").strip().lower()
    _explicit_ids = os.environ.get("BAQARO_PAPER_LC_IDS", DEFAULT_LC_IDS).strip()
    rank_source = os.environ.get("BAQARO_PAPER_LC_RANK_SOURCE", "all").strip().lower()
    _logmbh_cap = float(os.environ.get("BAQARO_PAPER_LC_LOGMBH_CAP", "11.0"))
    _logmbh_min = float(os.environ.get("BAQARO_PAPER_LC_LOGMBH_MIN", "9.0"))
    _rand_seed  = int(os.environ.get("BAQARO_PAPER_LC_RANDOM_SEED", "42"))
    # two_tier mode: per-row M_BH(z=Z_SELECT) bands. NARROW windows just above
    # each round number so the rows are genuinely ~1e9 (top) and ~1e8 (bottom),
    # NOT "anywhere in the decade" (a full [1e8,1e9) band lets a random draw land
    # at ~1e8.6, which is not a "1e8 BH"). Widen via the env vars if a band is
    # too sparse. Defaults: top [1e9, 1e9.2), bottom [1e8, 1e8.2).
    _top_min = float(os.environ.get("BAQARO_PAPER_LC_TOP_LOGMBH_MIN", "9.0"))
    _top_max = float(os.environ.get("BAQARO_PAPER_LC_TOP_LOGMBH_MAX", "9.2"))
    _bot_min = float(os.environ.get("BAQARO_PAPER_LC_BOT_LOGMBH_MIN", "8.0"))
    _bot_max = float(os.environ.get("BAQARO_PAPER_LC_BOT_LOGMBH_MAX", "8.2"))
    _mbh_cap = 10.0 ** _logmbh_cap
    _mbh_min = 10.0 ** _logmbh_min
    snap_sel = int(np.argmin(np.abs(redshifts - Z_SELECT)))

    mbh_all = loader.get_BH_mass(snap_sel) if rank_source != "targets" else None
    mbh_pool = (mbh_all if mbh_all is not None
                else loader.get_BH_mass(snap_sel)[targets])
    pool_idx_map = (np.arange(len(mbh_pool)) if rank_source != "targets" else targets)

    if mode == "gallery":
        # Browse mode: fill the whole grid with random draws from ONE band so you
        # can pick the nicest examples. Band selected by BAQARO_PAPER_LC_GALLERY_BAND
        # ("top" uses the top tier's [10^_top_min,10^_top_max), "bot" the bottom);
        # any other value is parsed as "lo,hi" log10 M_BH edges.
        _gb = os.environ.get("BAQARO_PAPER_LC_GALLERY_BAND", "top").strip().lower()
        rng = np.random.default_rng(_rand_seed)
        if _gb == "both":
            # OVERVIEW mode: browse BOTH paper tiers in one figure. The top
            # ceil(N_ROW/2) rows are drawn from the >1e9 band [10^_top_min,
            # 10^_top_max), the bottom rows from the >1e8 band [10^_bot_min,
            # 10^_bot_max). Each cell is labelled with its storage id + logM_BH(z≈6);
            # pick your favourites, then lock them with
            #   BAQARO_PAPER_LC_MODE=explicit BAQARO_PAPER_LC_IDS=<top ids…,bot ids…>
            # (row-major). Widen a tier for more candidates via
            # BAQARO_PAPER_LC_{TOP,BOT}_LOGMBH_{MIN,MAX}.
            n_top_rows = (N_ROW + 1) // 2
            bands = ((n_top_rows, _top_min, _top_max, "top >1e9"),
                     (N_ROW - n_top_rows, _bot_min, _bot_max, "bot >1e8"))
            chosen = []
            for _nrows, _lo, _hi, _tier in bands:
                n_want = _nrows * N_COL
                cand = np.flatnonzero((mbh_pool >= 10.0 ** _lo) & (mbh_pool < 10.0 ** _hi))
                if len(cand) < n_want:
                    print(f"WARNING: only {len(cand)} BHs in {_tier} band "
                          f"[10^{_lo:.2f},10^{_hi:.2f}); plotting all of them.")
                    picked = cand
                else:
                    picked = rng.choice(cand, size=n_want, replace=False)
                picked = picked[np.argsort(mbh_pool[picked])[::-1]]  # big -> small
                for pp in picked:
                    chosen.append((len(chosen), int(pool_idx_map[pp]), float(mbh_pool[pp])))
        else:
            # Single band: whole grid from ONE band ("top"/"bot"/"lo,hi").
            if _gb == "top":
                _glo, _ghi = _top_min, _top_max
            elif _gb == "bot":
                _glo, _ghi = _bot_min, _bot_max
            else:
                _glo, _ghi = (float(x) for x in _gb.split(","))
            band = (mbh_pool >= 10.0 ** _glo) & (mbh_pool < 10.0 ** _ghi)
            cand = np.flatnonzero(band)
            n_pick = N_ROW * N_COL
            if len(cand) < n_pick:
                print(f"WARNING: only {len(cand)} BHs in gallery band "
                      f"[10^{_glo:.2f}, 10^{_ghi:.2f}); plotting all of them.")
                picked = cand
            else:
                picked = rng.choice(cand, size=n_pick, replace=False)
            picked = picked[np.argsort(mbh_pool[picked])[::-1]]  # big -> small
            chosen = [(r, int(pool_idx_map[picked[r]]), float(mbh_pool[picked[r]]))
                      for r in range(len(picked))]
    elif mode == "two_tier":
        # Per-row mass bands: top row from [10^_top_min, 10^_top_max), bottom row
        # from [10^_bot_min, 10^_bot_max). N_COL objects per row, random within
        # the band (seeded), sorted big->small so cells go from massive to light.
        rng = np.random.default_rng(_rand_seed)
        chosen = []
        for _tier, _lo, _hi in (("top", _top_min, _top_max), ("bot", _bot_min, _bot_max)):
            band = (mbh_pool >= 10.0 ** _lo) & (mbh_pool < 10.0 ** _hi)
            cand = np.flatnonzero(band)
            if len(cand) < N_COL:
                print(f"WARNING: only {len(cand)} BHs in {_tier} band "
                      f"[10^{_lo:.1f}, 10^{_hi:.1f}); plotting all of them.")
                picked = cand
            else:
                picked = rng.choice(cand, size=N_COL, replace=False)
            picked = picked[np.argsort(mbh_pool[picked])[::-1]]  # big -> small
            for pp in picked:
                chosen.append((len(chosen), int(pool_idx_map[pp]), float(mbh_pool[pp])))
    elif mode == "explicit":
        explicit_ids = [int(s) for s in _explicit_ids.split(",") if s.strip()]
        chosen = []
        for r, sid in enumerate(explicit_ids):
            # storage_idx → pool position. For rank_source=="all", pool_idx_map
            # is np.arange(n_subset), so pool_position == storage_idx directly.
            if rank_source == "targets":
                # Find the target slot that corresponds to this storage idx
                target_pos = np.flatnonzero(targets == sid)
                if len(target_pos) == 0:
                    print(f"WARNING: idx {sid} not in original_targets; skipping")
                    continue
                pool_pos = int(target_pos[0])
            else:
                pool_pos = int(sid)   # storage idx == pool position when rank_source=='all'
            chosen.append((r, sid, float(mbh_pool[pool_pos])))
    elif mode == "ranks":
        mbh_for_rank = np.where(mbh_pool < _mbh_cap, mbh_pool, 0.0)
        order_local = np.argsort(mbh_for_rank)[::-1]
        chosen = [(r, int(pool_idx_map[order_local[r]]),
                   float(mbh_pool[order_local[r]])) for r in RANKS]
    else:
        valid_mask = (mbh_pool >= _mbh_min) & (mbh_pool < _mbh_cap)
        valid_local_idx = np.flatnonzero(valid_mask)
        n_pick = N_ROW * N_COL
        rng = np.random.default_rng(_rand_seed)
        if len(valid_local_idx) < n_pick:
            print(f"WARNING: only {len(valid_local_idx)} BHs in mass window "
                  f"[10^{_logmbh_min:.1f}, 10^{_logmbh_cap:.1f}); plotting all of them.")
            picked = valid_local_idx
        else:
            picked = rng.choice(valid_local_idx, size=n_pick, replace=False)
        # Sort the picked by M_BH descending so cells go big→small
        picked = picked[np.argsort(mbh_pool[picked])[::-1]]
        chosen = [(r, int(pool_idx_map[picked[r]]),
                   float(mbh_pool[picked[r]])) for r in range(len(picked))]

    n_capped = int((mbh_pool >= _mbh_cap).sum())
    print(f"Selected at z={redshifts[snap_sel]:.1f} (mode={mode}, "
          f"M_BH cap={_mbh_cap:.0e} filtered {n_capped}, "
          f"M_BH floor={_mbh_min:.0e}): "
          + ", ".join(f"r{r} idx{idx} logM={np.log10(m):.2f}" for r, idx, m in chosen),
          flush=True)

    # Halo-mass histories for the chosen targets. Prefer the FH file's own
    # `halo_masses_all` (n_snap, n_subset) — already pre-subsetted to the closure
    # halos and shape-matched to the FH redshift axis (critical when FH's max_snap
    # differs from the paper fiducial's, e.g. FH at z=2 max_snap=71 vs paper at
    # z=0 max_snap=144). Fall back to the external (n_snap, n_halos) mmap if the
    # in-file dataset is absent (older FH files). Shown as the seed-mass track
    # f_seed*M_halo (M_BH = M_halo * 10^logfseed) on the per-snapshot grid.
    storage_cols = np.asarray([idx for _, idx, _ in chosen], dtype=np.int64)
    # h5py fancy indexing requires increasing order — sort, read, then permute back.
    _sorted_pos = np.argsort(storage_cols)
    _inv_perm = np.argsort(_sorted_pos)
    if "halo_masses_all" in file:
        _tmp = np.asarray(file["halo_masses_all"][:, storage_cols[_sorted_pos].tolist()])
        halo_mass_grid = _tmp[:, _inv_perm]
    else:
        halo_mmap = np.load(path_file_halo_masses, mmap_mode="r")
        global_cols = data["original_indices"][storage_cols]
        halo_mass_grid = np.asarray(halo_mmap[:, global_cols])
    # NB halo_masses_all in the FH file is already in physical Msun (the
    # script's "Rescaling to physical units" step), so no `* mass_units` here.
    # Reference line = the model SEED-MASS track M_BH^seed = M_halo * f_seed,
    # with f_seed = 10^logfseed (bh_seeding: M_BH = M_halo * 10^logfseed). This
    # is the deterministic mass each halo seeds, so the red M_BH curve starts on
    # it. log10(M_seed) = log10(M_halo) + logfseed. Read logfseed from the run's
    # provenance (fallback -5.0 = the legacy M_halo/1e5 reference).
    try:
        logfseed = float(file["provenance"].attrs["logfseed"])
    except Exception:
        print("WARNING: logfseed not in provenance; using legacy -5.0 (M_halo/1e5)")
        logfseed = -5.0
    with np.errstate(divide="ignore"):
        mh_seed = np.log10(halo_mass_grid) + logfseed
    t_snap = cosmo.age(redshifts)

    # Fast path: load the chosen objects' sub-step columns from the pre-extracted
    # cache (build_lc_highz_cache.py) instead of eager-loading the ~159 GB/array
    # full-history datasets. Used only if the cache covers EVERY chosen object
    # (cols are matched by storage index via searchsorted on the sorted
    # cache_cols). Otherwise (no cache, BAQARO_PAPER_LC_NO_CACHE=1, or a selection
    # below the cache floor) fall back to the slow per-column FH read.
    name_file_fh = file.attrs["name_file"]
    if isinstance(name_file_fh, bytes):
        name_file_fh = name_file_fh.decode("utf-8")
    cache_path = lc_highz_cache_path(name_file_fh)
    use_cache = (os.path.exists(cache_path)
                 and os.environ.get("BAQARO_PAPER_LC_NO_CACHE", "") != "1")
    lbol_grid = mbh_grid = eta_grid = None
    if use_cache:
        cz = np.load(cache_path)
        cache_cols = cz["cache_cols"]
        pos = np.searchsorted(cache_cols, storage_cols)
        pos_clamped = np.clip(pos, 0, len(cache_cols) - 1)
        covered = (pos < len(cache_cols)) & (cache_cols[pos_clamped] == storage_cols)
        if covered.all():
            # Column j of each grid corresponds to chosen cell j (same order as
            # storage_cols), so the loop indexes them by `cell`.
            mbh_grid = cz["mbh"][:, pos]
            lbol_grid = cz["lbol"][:, pos]
            eta_grid = cz["eta"][:, pos]
            times = cz["times"]
            print(f"Using FH column cache ({len(cache_cols)} objects): {cache_path}",
                  flush=True)
        else:
            use_cache = False
            missing = storage_cols[~covered].tolist()
            print(f"FH cache present but missing {int((~covered).sum())} selected "
                  f"object(s) {missing} (below cache floor?) — using slow "
                  f"full-history load. Rebuild with a lower "
                  f"BAQARO_PAPER_LC_CACHE_FLOOR to cover them.", flush=True)
    if not use_cache:
        if not os.path.exists(cache_path):
            print(f"No FH column cache at {cache_path} — using slow full-history "
                  f"load (~159 GB/array). Build it once with "
                  f"`python -m baqaro.plotting_paper.build_lc_highz_cache`.",
                  flush=True)
        times = fh.times_full_history

    for cell, (rank, idx, m_sel) in enumerate(chosen):
        r, c = divmod(cell, N_COL)
        is_left, is_right = (c == 0), (c == N_COL - 1)
        is_top, is_bottom = (r == 0), (r == N_ROW - 1)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[cell], height_ratios=[3, 1], hspace=0.05)
        ax_lbol = fig.add_subplot(inner[0])
        ax_eta = fig.add_subplot(inner[1], sharex=ax_lbol)
        ax_mass = ax_lbol.twinx()

        if use_cache:
            lbol = lbol_grid[:, cell]
            mbh = mbh_grid[:, cell]
            etas = eta_grid[:, cell]
        else:
            lbol = fh.Lbols_full_history[:, idx]
            mbh = fh.black_hole_masses_full_history[:, idx]
            etas = fh.etas_full_history[:, idx]

        # --- L_bol (left) ---
        with np.errstate(divide="ignore"):
            log_lbol_ergs = my_utils.to_ergs(np.log10(lbol))
        # L_bol==0 (non-accreting plateaus) -> log = -inf, which would leave a
        # GAP; floor it just below the panel so the line visibly PLUNGES to the
        # bottom at each drop-out instead of vanishing.
        _y_bottom = my_utils.to_ergs(MBH_LO + nc.log_csi)
        log_lbol_ergs = np.where(np.isfinite(log_lbol_ergs), log_lbol_ergs,
                                 _y_bottom - 0.4)
        ax_lbol.plot(times, log_lbol_ergs, lw=1.2, alpha=0.75,
                     drawstyle="steps-mid", color=COL_LBOL, zorder=4)
        ax_lbol.set_xlim(T_LEFT, T_RIGHT)
        ax_lbol.set_ylim(my_utils.to_ergs(MBH_LO + nc.log_csi),
                         my_utils.to_ergs(MBH_HI + nc.log_csi))
        ax_lbol.tick_params(axis="y", colors=COL_LBOL)
        ax_lbol.minorticks_on()
        # Kill the parent's own TOP x-ticks (rcParams xtick.top=True) — the
        # secondary redshift axis owns the top spine on the top row; left on,
        # their linear-time spacing showed through as bogus "redshift subticks".
        ax_lbol.tick_params(axis="x", which="both", top=False)

        # --- M_BH (right twin) ---
        with np.errstate(divide="ignore"):  # log10(0) before the seed -> -inf (clipped)
            log_mbh = np.log10(mbh)
        ax_mass.plot(times, log_mbh, lw=1.5, alpha=0.9,
                     drawstyle="steps-mid", color=COL_MASS, zorder=5)
        ax_mass.set_ylim(MBH_LO, MBH_HI)
        ax_mass.set_xlim(ax_lbol.get_xlim())
        ax_mass.tick_params(axis="y", colors=COL_MASS)
        # Minor ticks on the right log M_BH axis (mirrors ax_lbol/ax_eta).
        # Keep x-axis minors off the twin (top=False keeps them quiet).
        ax_mass.minorticks_on()
        ax_mass.tick_params(axis="x", which="both", top=False, labeltop=False)

        # Salpeter growth from the seed (full rate + 1/10).
        spawned = np.where(mbh > 0.0)[0]
        if spawned.size:
            i0 = spawned[0]
            dt = times - times[i0]
            with np.errstate(divide="ignore"):
                ax_mass.plot(times, np.log10(mbh[i0] * np.exp(dt / T_SALPETER)),
                             lw=1.3, color="gray", alpha=0.25, ls="-")
                ax_mass.plot(times, np.log10(mbh[i0] * np.exp(dt / T_SALPETER / 10)),
                             lw=1.3, color="gray", alpha=0.25, ls="-")
            # Filled circle marking the START of the M_BH track (seed spawn).
            ax_mass.plot(times[i0], log_mbh[i0], marker="o", ms=5.5,
                         color=COL_MASS, mec="white", mew=0.8, ls="none", zorder=8)

        # --- Host halo mass / 1e5 (per-snapshot, dotted) ---
        # High zorder + white outline so it reads over the dense L_bol scatter
        # (and isn't lost under the M_BH curve when the two nearly coincide).
        ax_mass.plot(t_snap, mh_seed[:, cell], lw=2.0, color=COL_HALO,
                     alpha=1.0, ls=":", zorder=6,
                     path_effects=[pe.Stroke(linewidth=3.4, foreground="white", alpha=0.75),
                                   pe.Normal()])
        # Filled circle marking the START of the f_seed*M_halo track (first
        # resolved-halo snapshot).
        _seed_valid = np.where(np.isfinite(mh_seed[:, cell]))[0]
        if _seed_valid.size:
            _j0 = _seed_valid[0]
            ax_mass.plot(t_snap[_j0], mh_seed[_j0, cell], marker="o", ms=5.5,
                         color=COL_HALO, mec="white", mew=0.8, ls="none", zorder=8)

        # --- Eddington ratio (bottom) ---
        safe_mass = np.maximum(mbh, 1e-10)
        etas_eff = lbol / safe_mass / 10 ** nc.log_csi
        with np.errstate(divide="ignore"):
            ax_eta.plot(times, np.log10(np.maximum(etas, 1e-10)), lw=0.9, alpha=0.6,
                        drawstyle="steps-mid", color=COL_MDOT, zorder=1)
            ax_eta.plot(times, np.log10(etas_eff), lw=1.0, alpha=0.85,
                        drawstyle="steps-mid", color=COL_ETA, zorder=2)
        ax_eta.axhline(0.0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax_eta.set_xlim(ax_lbol.get_xlim())
        ax_eta.set_ylim(-1.5, 1.5)
        ax_eta.minorticks_on()

        # --- Per-cell label: the selection-epoch mass (+ storage idx in gallery
        # mode, so you can read off the IDs to lock in via LC_IDS). ---
        _lbl = rf"$\log_{{10}} M_{{\rm BH}}^{{z=6}} = {np.log10(m_sel):.2f}$"
        if mode == "gallery":
            _lbl = rf"id {idx}" + "\n" + _lbl
        ax_lbol.text(0.97, 0.06, _lbl,
                     transform=ax_lbol.transAxes, fontsize=12, ha="right", va="bottom",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        # --- Edge-only labelling (shared fixed y-windows) ---
        if is_left:
            ax_lbol.set_ylabel(r"$\log_{10}\, L_{\rm bol}$ [erg s$^{-1}$]",
                               color=COL_LBOL, labelpad=-1.5)
            # Two lines + smaller: the eta panel is only 1/4 of the cell height,
            # so the one-line form overflows into ax_lbol's ylabel.
            ax_eta.set_ylabel("$\\log_{10}$\nEdd. ratio", labelpad=-1,
                              fontsize=10.5, linespacing=1.0)
        else:
            ax_lbol.tick_params(labelleft=False)
            ax_eta.tick_params(labelleft=False)
        if is_right:
            ax_mass.set_ylabel(r"$\log_{10}\, M_{\rm BH}$ [$M_\odot$]",
                               color=COL_MASS, labelpad=-1.5)
        else:
            ax_mass.tick_params(labelright=False)

        plt.setp(ax_lbol.get_xticklabels(), visible=False)  # time label lives on eta panel
        if is_bottom:
            ax_eta.set_xlabel(r"Cosmic time [Gyr]", labelpad=2)
        else:
            ax_eta.tick_params(labelbottom=False)
        if is_top:
            add_redshift_axis(ax_lbol)

        # --- Colour-coded text keys ---
        # L_bol / M_BH / f_seed M_halo key, drawn only in the BOTTOM row (the
        # bottom-row keys serve the whole column).
        if not is_top:
            ax_lbol.text(0.03, 0.95, r"$L_{\rm bol}$", color=COL_LBOL, fontweight="bold",
                         transform=ax_lbol.transAxes, ha="left", va="top")
            ax_lbol.text(0.03, 0.82, r"$M_{\rm BH}$", color=COL_MASS, fontweight="bold",
                         transform=ax_lbol.transAxes, ha="left", va="top")
            ax_lbol.text(0.03, 0.69, r"$f_{\rm seed}\,M_{\rm halo}$", color=COL_HALO, fontweight="bold",
                         transform=ax_lbol.transAxes, ha="left", va="top")
        # η_acc / λ_Edd key in the TOP-LEFT cell's η (bottom) subpanel — λ_Edd at
        # bottom-left, η_acc at top-left (same x position).
        if cell == 0:
            ax_eta.text(0.03, 0.06, r"$\lambda_{\rm Edd}$", color=COL_ETA, fontweight="bold",
                        transform=ax_eta.transAxes, ha="left", va="bottom")
            ax_eta.text(0.03, 0.95, r"$\eta_{\rm acc}$", color=COL_MDOT, fontweight="bold",
                        transform=ax_eta.transAxes, ha="left", va="top")

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
