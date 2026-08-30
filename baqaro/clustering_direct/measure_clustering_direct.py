"""
Direct spatial clustering measurement from the FULL-CATALOGUE forward run.
====================================================================================

Measures the **real-space** correlation function xi(r) *directly* from the 3D HBT
subhalo positions (Corrfunc / Landy-Szalay), instead of predicting it from the
QHMF through the analytic halo-model "triangle" (which is what
``plotting_paper/plotting_clustering.py`` does). Panels mirror that figure:

  * z = 2.5, 4.0 -> quasar AUTO-correlation
  * z = 6.1      -> quasar x galaxy CROSS-correlation

Samples (both chosen to match the analytic figure so the two are comparable):
  * Quasars: obs-matched L_bol threshold per panel, read from ``corr_obs_data``
    (z=2.5 data_ef_ext, z=4.0 data_shen_highz, z=6.1 = to_solar(46.5) as in the
    paper script's override).
  * z=6.1 galaxies: halos with log10(M_halo/Msun) > 10.75 — the same cut the
    analytic cross tracer uses (``qhmf_gal[log_m_axis < 10.75] = 0`` in
    plotting_clustering.py). Optionally subsampled for speed (the cross-corr
    amplitude is independent of the tracer number density).

Why the full catalogue: spatial clustering needs the TRUE positions of every
quasar in the box. The mass-stratified subsample (Horvitz-Thompson weights)
cannot provide that, so this analysis *requires* the multinode full-catalogue
run (weights == 1, real positions).

Units: HBT ``ComovingAveragePosition`` is h-free comoving Mpc, box = 2800 cMpc.
The obs/triangle separations in ``corr_obs_data`` are ALSO h-free cMpc (they are
the published h^-1 Mpc values divided by cosmo.h), so we measure natively in
cMpc with box=2800 and reuse ``data.pimax`` / ``rp_arr`` directly — NO h factor.
Only real-space positions are used (no redshift-space distortions), matching the
analytic prediction and the obs deprojection.

Pair counting is done directly with Corrfunc (natural estimator with an
analytic RR in the periodic box; see ``measure_natural_xi``).
Output: one ``.npz`` per panel with the measured xi(r) on r-bin centers (+ optional
jackknife errors). Project + overlay on obs with ``plotting_clustering_direct.py``.

Run identity comes from the BAQARO_* env (same as load_data_to_plot); a bare
run resolves to the fiducial. Point it at the full-catalogue run, e.g.:

  env BAQARO_USE_SUBSAMPLE=0 BAQARO_SUBSET_TAG=multinode_root144_v3 \\
      BAQARO_CLUST_Z=6.1 BAQARO_CLUST_NJACK=0 \\
      python -m baqaro.clustering_direct.measure_clustering_direct

Env knobs:
  BAQARO_CLUST_Z          comma list of panels to run (default "2.5,4.0,6.1")
  BAQARO_CLUST_NJACK      per-side jackknife slices (0 = no errors; default 0)
  BAQARO_CLUST_GAL_MAX    max galaxies for the z=6.1 cross (subsample; default 3_000_000)
  BAQARO_CLUST_NTHREADS   Corrfunc threads (default 32)
  BAQARO_CLUST_RLO/RHI/RNB  r-bin range/count in cMpc (default 0.05 / 200 / 40)
  BAQARO_CLUST_OUTDIR     output dir for the .npz (default {output}/clustering_direct)
"""

import os
import numpy as np
import h5py

# --- run selection (reads BAQARO_* env at import, same as the plotting layer) ---
from baqaro.plotting_common.load_data_to_plot import (
    path_file,
    load_simulation_metadata,
    simulation_name,
    source_dir,
    snapshot_index_for_redshift,
)
from baqaro.utils.my_dir import get_input_path_HBT_data, get_output_path
from baqaro.clustering_direct.cache_provenance import (
    provenance_dict, require_full_catalogue,
)
from baqaro.obs_data.corr_obs_data import (
    data_ef_ext, data_shen_highz, data_aspire,
)
from qhtools.utils import my_utils

# Pair counting: Corrfunc directly. In a PERIODIC
# box the random pair count RR is analytic (uniform density, no edges/mask), so
# we use the NATURAL estimator xi = DD/RR_analytic - 1 and never build a random
# catalogue (that would only add RR shot noise + the O(N_rand) cost). Auto:
# Corrfunc.theory.xi (periodic, analytic RR built in). Cross: Corrfunc.theory.DD
# with autocorr=0 for the D1D2 counts, normalised by N1*N2*V_shell/V_box here.
# Both conventions were validated locally in tests/scratch (uniform -> xi~0,
# clustered -> xi>0; DD-natural matches theory.xi to <1e-4).
from Corrfunc.theory import xi as corrfunc_xi, DD as corrfunc_DD

BOX_CMPC = 2800.0                      # h-free comoving Mpc (position + box units)

# z=6.1 galaxy-tracer halo-mass cut for the FIGURE / DIRECT-MEASUREMENT path.
#
# **10.75 — adopted deliberately.** Read from the shared
# helper so the analytic halo-model curve (plotting_paper/plotting_clustering.py) and
# this direct Corrfunc measurement can never drift apart: the paper's z=6.1 panel
# overlays them, and if the two used different galaxy selections a ~9% offset would
# appear that reads as a halo-model failure but is pure bookkeeping, which is what
# this shared constant prevents.
#
# Changing this INVALIDATES the cached clustering_direct_z6.1.npz (its stored
# `logMh_gal_cut` records the value it was measured with; plotting_clustering refuses
# to overlay a cache whose cut disagrees).
from baqaro.obs_data.corr_obs_data import GAL_LOGM_CUT_FIGURE

LOG_MHALO_GAL_CUT = GAL_LOGM_CUT_FIGURE

# ---- per-panel config: obs object + auto/cross + L_bol threshold [Lsun] -------
# z=6.1 uses the paper script's explicit override to_solar(46.5) (data_aspire's
# own threshold is fainter); z=2.5 / 4.0 use each obs object's log_L_threshold.
PANELS = {
    "2.5": dict(obs=data_ef_ext,    kind="auto",  log_L_thr=data_ef_ext.log_L_threshold),
    "4.0": dict(obs=data_shen_highz, kind="auto", log_L_thr=data_shen_highz.log_L_threshold),
    "6.1": dict(obs=data_aspire,    kind="cross", log_L_thr=my_utils.to_solar(46.5)),
    # TEST panel: the SAME ASPIRE cross, measured one snapshot EARLIER,
    # to see how sensitive the cross-correlation is to the snapshot choice.
    #
    # ⚠ The requested z=6.5 is NOT measurable on the multinode v4 run: snap 38
    # (z=6.421, the true nearest to 6.5) was never saved, so a "6.5" target
    # argmins onto snap 37 (z=6.708), 0.21 away. This key is therefore named for
    # the TRUE redshift of the snapshot it uses, so the cached .npz can never be
    # mistaken for a z=6.5 measurement. Snap 38 IS in the 41-snapshot re-run
    # union — once it exists, add a "6.421" panel for the real z≈6.5 test.
    "6.708": dict(obs=data_aspire,  kind="cross", log_L_thr=my_utils.to_solar(46.5)),
}


def _env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def gather_positions(true_snap, global_ids, box=BOX_CMPC):
    """Return (3, N) h-free comoving-Mpc positions for the given GLOBAL track ids.

    ``global_ids`` are TrackIds into the full catalogue (row == TrackId in each
    OrderedSubSnap file). We read ``Subhalos/ComovingAveragePosition`` at the TRUE
    snapshot number. Small samples use an h5py fancy read; large samples read the
    whole (N,3) dataset once and index in memory (cheaper than millions of
    scattered reads).
    """
    template = os.path.join(
        get_input_path_HBT_data(source_dir), simulation_name,
        "HBT_compressed", "OrderedSubSnap_{snap_nr:03d}.hdf5",
    )
    path = template.format(snap_nr=int(true_snap))
    global_ids = np.ascontiguousarray(global_ids)
    with h5py.File(path, "r", rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as f:
        dset = f["Subhalos/ComovingAveragePosition"]      # (N, 3) float64
        N = dset.shape[0]
        if global_ids.max() >= N:
            raise ValueError(
                f"global id {global_ids.max()} >= N_subhalos {N} at snap {true_snap} "
                "— sample contains halos not yet born at this snapshot.")
        # Heuristic: h5py fancy reads are per-selection and get slow for many
        # rows, so only use them for small samples (quasars); for large samples
        # (galaxies) a single sequential read of the whole (N,3) dataset + an
        # in-memory index is much faster than millions of scattered reads.
        if global_ids.size <= 500_000:
            pos = np.asarray(dset[global_ids])            # (Nsel, 3), sorted ids
        else:
            full = np.asarray(dset[:])                    # (N, 3) sequential read
            pos = full[global_ids]
            del full
    # wrap into [0, box) for safety, then to (3, N)
    pos = np.mod(pos, box)
    return np.ascontiguousarray(pos.T)                    # (3, Nsel)


def build_sample_positions(loader, subset_indices, col, true_snap, mask):
    """storage-order boolean ``mask`` on Lbol/Mhalo -> (3, N) positions."""
    storage_rows = np.flatnonzero(mask)
    global_ids = subset_indices[storage_rows]             # storage -> GLOBAL track id
    return gather_positions(true_snap, global_ids), storage_rows.size


def measure_natural_xi(pos1, pos2, box, edges, nthreads):
    """Real-space xi(r) via Corrfunc + analytic RR (periodic box, NO randoms).

    ``pos1`` (and optional ``pos2``) are (3, N) h-free comoving-Mpc arrays.
    Returns (r_centers, xi) on the ``edges`` r-bins.
      * auto  (pos2 is None): Corrfunc.theory.xi (analytic RR = N(N-1)V_shell/V_box).
      * cross (pos2 given)   : D1D2 = Corrfunc.theory.DD(autocorr=0), then
                               xi = D1D2 / (N1*N2*V_shell/V_box) - 1.
    """
    X1 = np.ascontiguousarray(pos1[0]); Y1 = np.ascontiguousarray(pos1[1]); Z1 = np.ascontiguousarray(pos1[2])
    rlo, rhi = edges[:-1], edges[1:]
    r_geom = 0.5 * (rlo + rhi)
    if pos2 is None:
        res = corrfunc_xi(box, nthreads, edges, X1, Y1, Z1)
        xi = np.asarray(res["xi"], dtype=float)
    else:
        X2 = np.ascontiguousarray(pos2[0]); Y2 = np.ascontiguousarray(pos2[1]); Z2 = np.ascontiguousarray(pos2[2])
        res = corrfunc_DD(0, nthreads, edges, X1, Y1, Z1,
                          X2=X2, Y2=Y2, Z2=Z2, periodic=True, boxsize=box)
        n1, n2 = X1.size, X2.size
        Vshell = 4.0 / 3.0 * np.pi * (rhi**3 - rlo**3)
        RR = n1 * n2 * Vshell / box**3
        xi = np.asarray(res["npairs"], dtype=float) / RR - 1.0
    ravg = np.asarray(res["ravg"], dtype=float)          # 0 unless output_ravg
    r_c = np.where(ravg > 0, ravg, r_geom)
    return r_c, xi


def main():
    """Measure xi(r) directly from the full-catalogue run and cache it to .npz.

    Natural estimator with an analytic RR term, so no random catalogue is needed.
    Auto-correlation for the quasar-only panels, quasar x galaxy cross for the
    z~6 ASPIRE panel. Optional jackknife errors via ``BAQARO_CLUST_NJACK``.

    Snapshot selection goes through ``snapshot_index_for_redshift``, which warns
    when the run's saved grid cannot reach a requested redshift -- the reason the
    z~6.5 panel is keyed ``"6.708"``, after the snapshot it actually uses.

    Configured entirely through ``BAQARO_CLUST_*``.
    """
    z_list = [z.strip() for z in _env("BAQARO_CLUST_Z", "2.5,4.0,6.1").split(",")]
    njack = int(_env("BAQARO_CLUST_NJACK", "0"))
    # Use ALL galaxies above the cut by default: the cross amplitude is
    # number-density-independent, but more galaxies multiply the pair counts and
    # crush the small-scale shot noise (the quasar count is the fixed limiter).
    gal_max = int(_env("BAQARO_CLUST_GAL_MAX", "100000000"))
    nthreads = int(_env("BAQARO_CLUST_NTHREADS", "32"))
    r_lo = float(_env("BAQARO_CLUST_RLO", "0.05"))
    r_hi = float(_env("BAQARO_CLUST_RHI", "200.0"))
    r_nb = int(_env("BAQARO_CLUST_RNB", "40"))
    # Optional quasar L_bol threshold override [log10 erg/s], applied to ALL
    # panels in this run. Use it to beat pair-starvation at z=4 (the bright
    # Shen cut leaves only ~3k quasars; quasar bias is ~luminosity-independent
    # over ~1-2 dex, so a fainter cut measures the same clustering with far more
    # pairs). Default: each panel's obs-matched threshold.
    qso_thr_erg = os.environ.get("BAQARO_CLUST_QSO_LOGLBOL_ERG")
    qso_thr_erg = float(qso_thr_erg) if qso_thr_erg not in (None, "") else None
    outdir = _env("BAQARO_CLUST_OUTDIR", os.path.join(get_output_path(source_dir), "clustering_direct"))
    os.makedirs(outdir, exist_ok=True)

    rng = np.random.default_rng(42)

    print(f"[clustering-direct] run file:\n  {path_file}")
    print(f"[clustering-direct] panels={z_list} njack={njack} gal_max={gal_max} "
          f"nthreads={nthreads} rbins=({r_lo},{r_hi},{r_nb}) outdir={outdir}")

    with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as f:
        meta = load_simulation_metadata(f)
        loader = meta["loader"]
        redshifts = meta["redshifts"]
        subset_indices = loader._subset_indices
        snapshots = loader._snapshots
        if subset_indices is None or snapshots is None:
            raise RuntimeError("Run has no subset/subset_indices or snapshots — "
                               "this analysis needs the full-catalogue multinode file.")
        # Direct pair counting requires the TRUE positions of EVERY quasar; a
        # down-weighted subsample is invalid (see cache_provenance).
        require_full_catalogue(loader, what="Direct clustering")

        for zkey in z_list:
            cfg = PANELS[zkey]
            z_target = float(zkey)
            # Warns if the run's saved snapshot grid cannot reach z_target as
            # closely as the full simulation grid could (same guard the paper
            # plotting scripts use) -- a reduced-snapshot run otherwise snaps
            # silently, however far away.
            col = snapshot_index_for_redshift(
                redshifts, z_target, snapshots, label=f"clustering-direct z={zkey}")
            true_snap = int(snapshots[col])
            data = cfg["obs"]
            log_L_thr = cfg["log_L_thr"] if qso_thr_erg is None else my_utils.to_solar(qso_thr_erg)
            print(f"\n=== z={zkey} ({cfg['kind']}): file z={redshifts[col]:.3f} "
                  f"col={col} true_snap={true_snap} logL_thr={log_L_thr:.3f} pimax={data.pimax:.3g} ===")

            # --- quasar sample ---
            Lbol = loader.get_Lbol(col)
            qso_mask = Lbol > 10 ** log_L_thr
            qso_pos, n_qso = build_sample_positions(loader, subset_indices, col, true_snap, qso_mask)
            print(f"  N_qso = {n_qso:,}")
            if n_qso < 50:
                print("  !! too few quasars — skipping panel"); continue

            gal_pos = None
            n_gal = 0
            n_gal_used = 0
            if cfg["kind"] == "cross":
                Mh = loader.get_Halo_mass(col)               # storage order, solar
                gal_mask = Mh > 10 ** LOG_MHALO_GAL_CUT
                gal_storage = np.flatnonzero(gal_mask)
                n_gal = gal_storage.size
                if n_gal > gal_max:                          # subsample (amplitude is n-independent)
                    gal_storage = np.sort(rng.choice(gal_storage, size=gal_max, replace=False))
                n_gal_used = gal_storage.size
                gal_global = subset_indices[gal_storage]
                gal_pos = gather_positions(true_snap, gal_global)
                print(f"  N_gal = {n_gal:,} (using {n_gal_used:,}, cut logMh>{LOG_MHALO_GAL_CUT})")

            # --- measure real-space xi(r) directly (Corrfunc, analytic RR) ---
            edges = np.logspace(np.log10(r_lo), np.log10(r_hi), r_nb + 1)
            r_c, xi = measure_natural_xi(qso_pos, gal_pos, BOX_CMPC, edges, nthreads)
            xi_lo = xi_hi = None
            if njack > 0:
                # TODO: subvolume jackknife on the natural estimator (random-free).
                print("  (jackknife errors not yet implemented for the analytic-RR "
                      "path; central xi only)")

            out = os.path.join(outdir, f"clustering_direct_z{zkey}.npz")
            _prov = provenance_dict("direct_clustering", loader, requires_full_cat=True)
            np.savez(
                out,
                zkey=zkey, kind=cfg["kind"], z_file=redshifts[col], true_snap=true_snap,
                log_L_thr=log_L_thr, pimax=data.pimax, box_cmpc=BOX_CMPC,
                r_centers=r_c, xi=xi,
                xi_lo=(xi_lo if xi_lo is not None else np.array([])),
                xi_hi=(xi_hi if xi_hi is not None else np.array([])),
                n_qso=n_qso, n_gal=n_gal, n_gal_used=n_gal_used,
                logMh_gal_cut=LOG_MHALO_GAL_CUT,
                **_prov,
            )
            print(f"  -> {out}   xi[0:5]={np.round(xi[:5],3)}")

    print("\n[clustering-direct] done.")


if __name__ == "__main__":
    main()
