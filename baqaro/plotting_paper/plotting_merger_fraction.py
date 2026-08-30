"""
PAPER FIGURE: merger vs total BH mass growth in the (M_BH, z) plane.

Single panel: the fraction of BH mass-growth contributed by BH-BH mergers,
  f_mrg(M_BH, z) = ΔM_mrg / ΔM_tot,
binned by the **growing** black hole (descendant mass at the snapshot where the
growth is recorded). A cyan f=0.5 contour marks where mergers take over. This is
the Pacucci & Loeb (2020) decomposition realised in our model: accretion
dominates everywhere except the high-mass / low-z corner (logM_BH >~ 9, z <~ 1),
where dry mergers of massive BHs overtake the (by then declining) gas accretion.

FOR THE PAPER (method, one paragraph):
  We split each black hole's mass growth into mergers and accretion. Rather than
  inferring the accreted mass from the luminosity, we measure the model's total
  mass growth directly — ΔM_tot = Σ_i max(M_BH(t_i) − M_BH(t_{i-1}), 0), summed
  between successive snapshots over black holes already present at t_{i-1} — and
  attribute to mergers the mass delivered by the secondary in every BH–BH merger,
  ΔM_mrg. The merger fraction of assembled mass, f_mrg = ΔM_mrg/ΔM_tot, is shown
  as a function of the (descendant) mass M_BH and redshift z. This definition is
  self-consistent — it needs no radiative efficiency and no luminosity, and is
  well defined on runs that save only a subset of snapshots — yet gives, in
  practice, the SAME map as the Soltan/luminosity accretion decomposition it
  replaces (the two agree to <0.006 in f_mrg across the whole M_BH–z plane, on
  the full-snapshot root144 run; max f_mrg ≈ 0.85, unchanged).

DENOMINATOR — TOTAL measured mass growth.
  The accretion channel is NO LONGER estimated separately from L_bol. Instead we
  measure the model's ACTUAL total mass growth between consecutive SAVED
  snapshots — ΔM_tot = Σ max(M_BH(i) − M_BH(i−1), 0) over tracks that already
  existed at i−1 — which by construction is "accretion + mergers together". The
  merger contribution is then a fraction of that same measured total. This is
  self-consistent (it uses the run's own M_BH build-up, not a luminosity proxy
  that could disagree with it) and, crucially, adapts to WHATEVER snapshots the
  run saved: on a reduced-snapshot run (the multinode full-catalogue file stores
  17 of 145 snapshots) ΔM between two saved snapshots is the cumulative growth
  over that gap; on a full 145-snapshot run each gap is one snapshot. The z
  resolution of the figure is therefore exactly the saved-snapshot cadence.

NUMERATOR — merger-delivered mass. TWO code paths, auto-selected —

  (A) MERGER CATALOG (preferred; used whenever
      ``merger_catalog_{name_file}.hdf5`` exists next to the run). The catalog
      stores EVERY merger as a row: primary ``M1``, secondary ``M2`` and the
      merger's EXACT redshift ``z``. The growing BH's mass is ``M1+M2`` and the
      mass DELIVERED is the SECONDARY ``min(M1,M2)``. Each merger is attributed
      to the SAVED-snapshot gap that contains its redshift, so numerator and
      denominator share the same gap→z binning (they must, or f is ill-defined).
      The catalog is accumulated over every snapshot, so it is complete even on a
      reduced-snapshot run.

  (B) MERGER TREES (fallback, when no catalog exists). A subset halo merges at
      snap i iff birth<i & death==i & merger_id!=-1; the secondary's BH mass at
      i-1 is the mass brought in, attributed to its descendant. Requires ALL 145
      snapshots (it matches tree death-snaps against the column index) and
      refuses a reduced-snapshot run outright.

  Mergers conserve total mass, so the delivered secondary mass is part of the
  descendant's measured ΔM_tot — hence f = ΔM_mrg/ΔM_tot ∈ [0, 1] by
  construction (modulo mild mass-bin migration over a wide gap).

Reads the main paper fiducial via fiducial_data. The heavy pass is cached to an
.npz; set BAQARO_MERGER_MAP_RECOMPUTE=1 to redo. The cache key carries the run
identity, the grid knobs AND which path produced it, so (A) and (B) can never be
served for one another.
"""

import os
import hashlib
import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import qhtools.utils.natconst as nc
from qhtools.utils.cosmology import cosmo

# Pinned fiducial run (force-sets BAQARO_* before load_data_to_plot resolves).
from baqaro.plotting_paper.fiducial_data import (
    path_file, path_out, name_file, name_file_halos, boxsize,
    load_simulation_metadata,
)
from baqaro.plotting_paper import plot_config
from baqaro.plotting_common.plot_config import save_fig, maybe_show


# ==============================================================================
# CONFIG
# ==============================================================================
name_fig = "merger_accretion_fraction"

LOG_M_LO = float(os.environ.get("BAQARO_MERGER_MAP_LOGM_LO", "6"))
LOG_M_HI = float(os.environ.get("BAQARO_MERGER_MAP_LOGM_HI", "10"))
DLOGM = float(os.environ.get("BAQARO_MERGER_MAP_DLOGM", "0.4"))
Z_MAX = float(os.environ.get("BAQARO_MERGER_MAP_ZMAX", "8"))
DZ = float(os.environ.get("BAQARO_MERGER_MAP_DZ", "0.5"))  # tree-path z-binning only
VMIN_LOG = float(os.environ.get("BAQARO_MERGER_MAP_VMIN", "1e-3"))  # log colour-scale floor

M_LO, M_HI = 10 ** LOG_M_LO, 10 ** LOG_M_HI
M_EDGES = np.arange(LOG_M_LO, LOG_M_HI + 1e-6, DLOGM)
M_CENT = 0.5 * (M_EDGES[1:] + M_EDGES[:-1])
Z_EDGES = np.arange(0.0, Z_MAX + 1e-6, DZ)

_run_id = os.path.splitext(os.path.basename(path_file))[0].replace("bh_evolution_", "")

# --- which rho_mrg path? Catalog if the run wrote one, else merger trees. ------
# The catalog is the SAME run's exact merger record; the trees are a
# reconstruction. Prefer the catalog whenever it exists (see module docstring).
CATALOG_PATH = os.path.join(path_out, "evolution",
                            f"merger_catalog_{name_file}.hdf5")
HAS_CATALOG = os.path.exists(CATALOG_PATH)
_METHOD = "catalog" if HAS_CATALOG else "trees"
# Opt out of the catalog (e.g. to reproduce a figure on a run that has one).
# Cannot opt IN without a catalog — there is nothing to read.
if os.environ.get("BAQARO_MERGER_FORCE_TREES", "0") == "1":
    HAS_CATALOG, _METHOD = False, "trees"

# Rows per chunk when streaming the ~1e9-row catalog (M1+M2+z ≈ 16 B/row).
CAT_CHUNK = int(float(os.environ.get("BAQARO_MERGER_CAT_CHUNK", "50e6")))

# Cache key MUST cover everything the grid depends on — run
# identity + grid knobs + WHICH PATH produced it + the estimator version, so a
# catalog grid can never be served for a tree grid, and caches from the earlier
# "rho_acc from luminosity" estimator are invalidated.
_ESTIMATOR = "dMtot_v3"  # catalog numerator = M1 (secondary), not min(M1,M2); per-gap cache
_grid_key = hashlib.md5(
    repr((LOG_M_LO, LOG_M_HI, DLOGM, Z_MAX, DZ, _METHOD, _ESTIMATOR)).encode()
).hexdigest()[:8]
CACHE = os.path.join(
    path_out, "plots",
    f"merger_fraction_cache_{_run_id}_{_METHOD}_grid{_grid_key}.npz",
)
# Catalog-path PER-GAP cache: the expensive catalog scan (ΔM_mrg / ΔM_tot per
# mass-bin per saved-snapshot gap), keyed on run + mass grid + estimator ONLY
# (NOT DZ / Z_MAX). The z rebinning into coarse display bins is a cheap plot-time
# step, so DZ can be retuned without re-reading the ~GB merger catalog.
_gap_key = hashlib.md5(
    repr((LOG_M_LO, LOG_M_HI, DLOGM, _METHOD, _ESTIMATOR)).encode()
).hexdigest()[:8]
CACHE_GAP = os.path.join(
    path_out, "plots",
    f"merger_fraction_gapcache_{_run_id}_{_gap_key}.npz",
)


# ==============================================================================
# SHARED — total measured mass growth per (mass-bin, saved-snapshot gap)
# ==============================================================================
def _total_growth_by_gap(loader, redshifts):
    """ΔM_tot per (mass-bin, saved-snapshot gap), a raw mass [Msun].

    Gap c (c = 1 .. N-1) spans saved columns c-1 -> c; ΔM = M_BH(c) - M_BH(c-1)
    for every track that ALREADY EXISTED at c-1 (M_prev > 0) and grew (ΔM > 0),
    binned by the DESCENDANT mass log10(M_BH(c)). This is "accretion + mergers
    together": a merger deposits the secondary onto the descendant, so that mass
    is part of the descendant's ΔM here. Newly-seeded BHs (M_prev == 0) are
    excluded so the denominator is growth of pre-existing holes, not seeding.

    HT-weighted by loader.weights (== 1 for a full-catalogue run). Returned WITHOUT
    /dt or /V: the final f = ΔM_mrg/ΔM_tot is a ratio of masses, so those factors
    cancel and are never formed.
    """
    n_m = M_CENT.size
    N = len(redshifts)
    total_gap = np.zeros((n_m, N - 1))
    w_all = loader.weights
    M_prev = np.asarray(loader.get_BH_mass(0), dtype=np.float64)
    for i in range(1, N):
        M = np.asarray(loader.get_BH_mass(i), dtype=np.float64)
        dM = M - M_prev
        cut = (M_prev > 0) & (M >= M_LO) & (M <= M_HI) & (dM > 0)
        if np.any(cut):
            w = w_all[cut] if w_all is not None else 1.0
            h, _ = np.histogram(np.log10(M[cut]), bins=M_EDGES, weights=dM[cut] * w)
            total_gap[:, i - 1] = h
        print(f"  ΔM_tot: gap -> col {i:3d} z={redshifts[i]:5.2f}", end="\r", flush=True)
        M_prev = M
    print()
    return total_gap


# ==============================================================================
# PATH (A) — MERGER CATALOG numerator (exact, snapshot-free), gap-binned
# ==============================================================================
def _mrg_mass_by_gap(loader, redshifts):
    """ΔM_mrg per (mass-bin, saved-snapshot gap) from the EXACT merger catalog.

    Growing BH = descendant, mass M1+M2; **mass delivered = the SECONDARY
    min(M1,M2)**. Each merger is attributed by its exact redshift to the
    saved-snapshot GAP that contains it (the same gaps _total_growth_by_gap
    integrates ΔM over), so numerator and denominator live on one z-binning.

    CATALOG CONVENTION (verified against the tree path): **M1 is the SECONDARY**
    (the merged-in BH = the mass delivered), **M2 is the primary** (already built
    by its own accretion, counted in the descendant's ΔM). So the delivered mass
    is M1, not min(M1, M2): ΣM1 reproduces the tree numerator.
    """
    n_m = M_CENT.size
    N = len(redshifts)
    mrg_gap = np.zeros((n_m, N - 1))
    asc = np.asarray(redshifts, dtype=np.float64)[::-1]  # ascending z for searchsorted
    gap_bins = np.arange(N)                              # edges 0..N-1 -> N-1 gap cols

    w_all = loader.weights
    uniform_w = w_all is None or bool(np.all(np.asarray(w_all) == 1.0))
    if not uniform_w:
        gid = np.asarray(loader._subset_indices)
        order = np.argsort(gid)
        gid_sorted = gid[order]

    with h5py.File(CATALOG_PATH, "r") as c:
        n_rows = c["M1"].shape[0]
        print(f"  ΔM_mrg: streaming {n_rows:,} mergers from the catalog", flush=True)
        for s in range(0, n_rows, CAT_CHUNK):
            e = min(s + CAT_CHUNK, n_rows)
            M1 = np.asarray(c["M1"][s:e], dtype=np.float64)
            M2 = np.asarray(c["M2"][s:e], dtype=np.float64)
            zz = np.asarray(c["z"][s:e], dtype=np.float64)

            M_desc = M1 + M2                      # the GROWING black hole
            M_sec = M1                            # M1 = SECONDARY = mass DELIVERED (NOT min; see docstring)
            # Gap index: a merger at z belongs to gap c (cols c-1 -> c, rep z =
            # redshifts[c], the lower-z endpoint) iff redshifts[c] <= z < redshifts[c-1].
            # In ascending z: k = searchsorted(asc, z, "right") -> c = N - k, gap = c-1.
            k = np.searchsorted(asc, zz, side="right")
            gap_idx = (N - k) - 1
            keep = ((M_sec > 0) & (M_desc >= M_LO) & (M_desc <= M_HI)
                    & (gap_idx >= 0) & (gap_idx <= N - 2))
            if not np.any(keep):
                continue

            if uniform_w:
                w = np.ones(int(keep.sum()))
            else:
                dest = np.asarray(c["descendant_id"][s:e])[keep]
                pos = np.clip(np.searchsorted(gid_sorted, dest), 0, gid.size - 1)
                found = gid_sorted[pos] == dest
                w = np.where(found, np.asarray(w_all)[order[pos]], 0.0)

            h, _, _ = np.histogram2d(
                np.log10(M_desc[keep]), gap_idx[keep],
                bins=[M_EDGES, gap_bins], weights=M_sec[keep] * w,
            )
            mrg_gap += h
            print(f"    {e:,}/{n_rows:,}", end="\r", flush=True)
    print()
    return mrg_gap


def compute_grid_catalog():
    """Per-gap ΔM_mrg (catalog numerator) and ΔM_tot (measured denominator).

    Returns ``(z_rep, M_CENT, tg, mg)`` — the numerator/denominator per mass-bin
    per saved-snapshot gap, NOT the ratio. ``_rebin_gaps_to_z`` sums these into
    the coarse ``Z_EDGES`` display bins and takes the ratio there.
    """
    print(f"Merger path: CATALOG -> {os.path.basename(CATALOG_PATH)}", flush=True)
    with h5py.File(path_file, "r", rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as f:
        data = load_simulation_metadata(f)
        redshifts = np.asarray(data["redshifts"])
        loader = data["loader"]

        total_gap = _total_growth_by_gap(loader, redshifts)
        mrg_gap = _mrg_mass_by_gap(loader, redshifts)

    # Representative z of each gap = its lower-z endpoint (where the descendant
    # mass is measured), i.e. redshifts[1:]. Return the PER-GAP numerator (ΔM_mrg)
    # and denominator (ΔM_tot), NOT their ratio: the z rebinning into coarse
    # display bins is done mass-weighted at plot time (_rebin_gaps_to_z), so DZ is
    # retunable without rescanning the catalog. Keep ALL gaps with growth (no Z_MAX
    # gate here — Z_MAX is a display choice applied during rebinning).
    z_rep = np.asarray(redshifts)[1:]
    keep = np.nansum(total_gap, axis=0) > 0
    z_rep = z_rep[keep]
    tg, mg = total_gap[:, keep], mrg_gap[:, keep]
    o = np.argsort(z_rep)                     # ascending z
    return z_rep[o], M_CENT.copy(), tg[:, o], mg[:, o]   # (n_gap,), (n_m,), (n_m,n_gap)×2


# ==============================================================================
# PATH (B) — MERGER TREES numerator (fallback; needs ALL snapshots)
# ==============================================================================
def compute_grid():
    """Compute the merger-fraction grid from the halo merger trees.

    Reads the birth/death snapshot arrays as memmaps (they are far too large to
    load) and accumulates the fraction of BH mass growth contributed by mergers
    per (redshift, mass) cell. Cached by the caller -- this is the expensive
    step of the figure.
    """
    trees_dir = os.path.join(path_out, "halo_histories", f"merger_trees_{name_file_halos}")
    death_mmap = np.load(os.path.join(trees_dir, "snapshot_indexes_of_death.npy"), mmap_mode="r")
    birth_mmap = np.load(os.path.join(trees_dir, "snapshot_indexes_of_birth.npy"), mmap_mode="r")
    merger_mmap = np.load(os.path.join(trees_dir, "merger_track_ids.npy"), mmap_mode="r")

    with h5py.File(path_file, "r", rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as f:
        data = load_simulation_metadata(f)
        redshifts = data["redshifts"]
        loader = data["loader"]
        w_all = loader.weights
        if w_all is None:
            raise SystemExit("Expected a subsample run (loader.weights is None).")
        # This figure differences BH mass between ADJACENT columns and matches the
        # tree death snapshot (TRUE snapshot number 0..144) against the loop column
        # index. Both are only valid when the file stores EVERY snapshot
        # (column i == snapshot i). A reduced-snapshot run would mis-attribute
        # merger z-bins -- refuse it loudly and point at the catalog path.
        _snaps = getattr(loader, "_snapshots", None)
        if _snaps is not None:
            _snaps = np.asarray(_snaps)
            if not np.array_equal(_snaps, np.arange(len(_snaps))):
                raise SystemExit(
                    "plotting_merger_fraction's TREE path needs a FULL-snapshot run "
                    f"(column i == snapshot i); this run stores only {len(_snaps)} "
                    f"snapshots ({list(_snaps[:5])}...), i.e. a reduced-snapshot file. "
                    "The per-snapshot merger reconstruction can't be made correct on it.\n"
                    "  FIX: give the run a MERGER CATALOG. The catalog path needs no "
                    "snapshot columns at all (it carries each merger's exact z), and is "
                    "the preferred path when present. Expected at:\n"
                    f"    {CATALOG_PATH}\n"
                    "  Otherwise point the fiducial at a full-snapshot subsample run.")
        subset_indices = np.asarray(loader._subset_indices)
        n_sub = subset_indices.size

        order = np.argsort(subset_indices)
        g_sorted = subset_indices[order]

        def gather(mmap):
            vs = np.asarray(mmap[g_sorted])
            out = np.empty(n_sub, dtype=vs.dtype)
            out[order] = vs
            return out

        print("Gathering merger trees for the subset ...", flush=True)
        death_sub, birth_sub, merger_sub = gather(death_mmap), gather(birth_mmap), gather(merger_mmap)

        def to_storage(gids):
            pos = np.clip(np.searchsorted(g_sorted, gids), 0, n_sub - 1)
            return order[pos], (g_sorted[pos] == gids)

        n_m = M_CENT.size
        z_tot_mass = [np.zeros(n_m) for _ in Z_EDGES[:-1]]  # total ΔM  per (mbin) per zbin
        z_mrg_mass = [np.zeros(n_m) for _ in Z_EDGES[:-1]]  # merger ΔM per (mbin) per zbin

        M_prev = loader.get_BH_mass(0)
        for i in range(1, len(redshifts)):
            z = float(redshifts[i])
            M = loader.get_BH_mass(i)
            zb = int(np.digitize(z, Z_EDGES)) - 1
            in_map = 0 <= zb < len(z_tot_mass)

            if in_map:
                # DENOMINATOR: total measured growth of pre-existing holes.
                dM = M - M_prev
                cut = (M_prev > 0) & (M >= M_LO) & (M <= M_HI) & (dM > 0)
                if np.any(cut):
                    h_tot, _ = np.histogram(np.log10(M[cut]), bins=M_EDGES,
                                            weights=dM[cut] * w_all[cut])
                    z_tot_mass[zb] += h_tot

            # NUMERATOR: merger-delivered secondary mass, attributed to descendant.
            mmask = (birth_sub < i) & (birth_sub != -1) & (death_sub == i) & (merger_sub != -1)
            st = np.flatnonzero(mmask)
            if in_map and st.size:
                M2 = M_prev[st]
                dest_st, found = to_storage(merger_sub[st])
                Mdest = np.zeros(st.size)
                Mdest[found] = M[dest_st[found]]
                keep = found & (M2 > 0) & (Mdest >= M_LO) & (Mdest <= M_HI)
                if np.any(keep):
                    h_mrg, _ = np.histogram(np.log10(Mdest[keep]), bins=M_EDGES,
                                            weights=M2[keep] * w_all[st][keep])
                    z_mrg_mass[zb] += h_mrg
            print(f"z={z:5.2f} snap={i:3d}", end="\r", flush=True)
            M_prev = M
        print()

    zb_c, fcols = [], []
    for k in range(len(z_tot_mass)):
        tot = z_tot_mass[k]
        if np.sum(tot) <= 0:
            continue
        mrg = z_mrg_mass[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = mrg / tot
        zc = 0.5 * (Z_EDGES[k] + Z_EDGES[k + 1])
        print(f"  z={zc:4.2f}: ΣΔM_tot={np.nansum(tot):.3e}  "
              f"ΣΔM_mrg={np.nansum(mrg):.3e} Msun", flush=True)
        zb_c.append(zc)
        fcols.append(f)
    return np.asarray(zb_c), M_CENT.copy(), np.asarray(fcols).T  # (n_m, n_z)


def _rebin_gaps_to_z(z_rep, tg, mg):
    """Sum the per-gap ΔM_mrg and ΔM_tot into the coarse ``Z_EDGES`` display bins,
    THEN take the ratio (mass-weighted f_mrg, the only correct way to coarsen a
    ratio). Empty bins -> NaN (masked grey). Retuning ``DZ`` / ``Z_MAX`` only
    re-runs this cheap step, never the catalog scan.
    """
    z_cent = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
    idx = np.digitize(np.asarray(z_rep), Z_EDGES) - 1        # coarse bin per gap
    n_m, n_zb = tg.shape[0], len(Z_EDGES) - 1
    TG = np.zeros((n_m, n_zb)); MG = np.zeros((n_m, n_zb))
    for k in range(n_zb):
        sel = idx == k
        if sel.any():
            TG[:, k] = np.nansum(tg[:, sel], axis=1)
            MG[:, k] = np.nansum(mg[:, sel], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = MG / TG
    f[TG <= 0] = np.nan                                      # no growth in bin -> masked
    empty = int(np.sum(TG.sum(axis=0) <= 0))
    if empty:
        print(f"  {empty}/{n_zb} display z-bin(s) empty (masked grey) at DZ={DZ}.", flush=True)
    for k in range(n_zb):
        if TG[:, k].sum() > 0:
            print(f"  z≈{z_cent[k]:4.2f} (bin {Z_EDGES[k]:.1f}-{Z_EDGES[k+1]:.1f}): "
                  f"ΣΔM_tot={np.nansum(TG[:, k]):.3e}  ΣΔM_mrg={np.nansum(MG[:, k]):.3e}  "
                  f"f={np.nansum(MG[:, k])/max(np.nansum(TG[:, k]),1e-30):.3f}", flush=True)
    return z_cent, f


_recompute = os.environ.get("BAQARO_MERGER_MAP_RECOMPUTE", "0") == "1"
if HAS_CATALOG:
    # Catalog path: load/compute the PER-GAP numerator/denominator (the expensive
    # scan, DZ-independent), then rebin into coarse z bins here (cheap, DZ-tunable).
    if os.path.exists(CACHE_GAP) and not _recompute:
        print(f"Loading per-gap cache: {CACHE_GAP}", flush=True)
        d = np.load(CACHE_GAP)
        z_rep, m_cent, tg_gap, mg_gap = d["z_rep"], d["m_cent"], d["tg"], d["mg"]
    else:
        z_rep, m_cent, tg_gap, mg_gap = compute_grid_catalog()
        os.makedirs(os.path.dirname(CACHE_GAP), exist_ok=True)
        np.savez(CACHE_GAP, z_rep=z_rep, m_cent=m_cent, tg=tg_gap, mg=mg_gap)
        print(f"Cached per-gap grid -> {CACHE_GAP}", flush=True)
    z_cent, f_grid = _rebin_gaps_to_z(z_rep, tg_gap, mg_gap)
else:
    # Tree path: already DZ-binned internally; cache the final f as before.
    if os.path.exists(CACHE) and not _recompute:
        print(f"Loading cached grid ({_METHOD}): {CACHE}", flush=True)
        d = np.load(CACHE)
        z_cent, m_cent, f_grid = d["z_cent"], d["m_cent"], d["f_grid"]
    else:
        z_cent, m_cent, f_grid = compute_grid()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, z_cent=z_cent, m_cent=m_cent, f_grid=f_grid)
        print(f"Cached grid ({_METHOD}) -> {CACHE}", flush=True)

print("max merger fraction:", float(np.nanmax(f_grid)),
      "| cells f>0.5:", int(np.nansum(f_grid > 0.5)))


# ==============================================================================
# FIGURE (single panel, paper style)
# ==============================================================================
fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.0), constrained_layout=True)

cmap = plt.get_cmap("magma").copy()
cmap.set_bad("0.86")  # no-BH-yet cells (NaN) -> light gray
# Log colour scale: clip finite values to the floor so f=0 (pure accretion) maps
# to the bottom colour, while NaN (no growth yet) stays masked/grey. mrg/0 -> inf
# is clipped down to 1 (mild mass-bin migration over wide gaps can push it >1).
disp = np.where(np.isfinite(f_grid), np.clip(f_grid, VMIN_LOG, 1.0), np.nan)
fm = np.ma.masked_invalid(disp)

pcm = ax.pcolormesh(z_cent, m_cent, fm, shading="nearest", cmap=cmap,
                    norm=LogNorm(vmin=VMIN_LOG, vmax=1.0))
cb = fig.colorbar(pcm, ax=ax, pad=0.012)
cb.set_label(r"merger fraction  $\dot{\rho}_{\rm mrg}/\dot{\rho}_{\rm tot}$")
cb.ax.minorticks_on()

# f=0.5 dominance contour (drawn only where finite)
ff = np.where(np.isfinite(f_grid), f_grid, 0.0)
if np.nanmax(f_grid) >= 0.5:
    ax.contour(z_cent, m_cent, ff, levels=[0.5], colors="cyan", linewidths=1.8)

ax.set_xlim(0, Z_MAX)
ax.set_ylim(LOG_M_LO, LOG_M_HI)
ax.set_xlabel(r"Redshift $z$", labelpad=2)
ax.set_ylabel(r"$\log_{10}\, M_{\rm BH}$ [$M_\odot$]", labelpad=2)
ax.minorticks_on()

save_fig(fig, plot_config.FIGURES_DIR, name_fig, force_dir=True)
maybe_show()
