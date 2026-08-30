"""
Direct quasar bias b_Q from the measured auto-correlation xi_QQ(r).
=====================================================================================

For a grid of (redshift, L_bol bin), measure the quasar AUTO-correlation xi_QQ(r)
directly from the 3D positions (Corrfunc natural estimator, analytic RR — same as
clustering_direct/measure_clustering_direct), then read off the large-scale linear
bias by amplitude-matching to the linear matter correlation:

    xi_QQ(r) = b_Q^2 * xi_mm(r, z)   on large (linear) scales
    b_Q = sqrt( <xi_QQ> / <xi_mm> )  averaged over r in [LINEAR_RMIN, LINEAR_RMAX]

This is the DIRECT-clustering bias (real-space; no RSD), complementary to the
effective host-halo bias b_Q=<b_h>_QHMF (Tinker10, needs no pair counting) that
the paper figure (plotting_paper/plotting_quasar_bias.py) now computes inline.
The two should agree where xi_QQ is well measured; the direct one is noisier for
sparse/bright bins (auto-corr ~ N^2).

xi_mm(r,z) from colossus with the FLAMINGO cosmology (qhtools.utils.cosmology).
Error from a jackknife over the linear-range r-bins.

Run identity from BAQARO_* env; a bare run resolves to the fiducial. Point it
at the full-catalogue run, e.g.:
  env BAQARO_USE_SUBSAMPLE=0 BAQARO_SUBSET_TAG=multinode_root144_v3 \\
      python -m baqaro.clustering_direct.measure_bias_direct

Two modes (BAQARO_BIASDIR_MODE, default "auto"):
  auto  — b_Q from the quasar AUTO-corr xi_QQ (pair count ~ N_Q^2; noisy for
          sparse/bright/high-z bins). Writes quasar_bias_direct_xiQQ.npz.
  cross — b_Q from the CROSS-corr xi_Q,ref against a dense LOWER-mass halo
          tracer (pair count ~ N_Q*N_ref; stays measured where the auto starves).
          Self-calibrates b_ref from the tracer's own auto xi_rr, so
          b_Q = S_Qr / sqrt(S_rr*S_mm). Writes quasar_bias_direct_cross.npz.
          This mirrors how the z~6 JWST surveys (EIGER/ASPIRE/He+18) do it.

Env knobs:
  BAQARO_BIASDIR_MODE     "auto" (default) or "cross"
  BAQARO_BIASDIR_Z        auto-mode redshifts (default "2.0,2.5,3.0")
  BAQARO_BIASDIR_CROSS_Z  cross-mode redshifts (default "3.0,4.0,5.0,6.1")
  BAQARO_BIASDIR_LBINS    log L_bol bins (default "45.5-46,46-46.5,46.5-47,47-48")
  BAQARO_BIASDIR_QSO_MAX  cap quasars per bin (subsample; default 500000)
  BAQARO_BIASDIR_MINN     skip a bin with fewer quasars (auto 300, cross 50)
  BAQARO_BIASDIR_REF_LOGM cross tracer log10 M_halo window (default "11,12")
  BAQARO_BIASDIR_REF_MAX  cross tracer cap (subsample; default 4000000)
  BAQARO_BIASDIR_RLIN     "rmin,rmax" cMpc linear range for b_Q (default "10,40")
  BAQARO_CLUST_NTHREADS   Corrfunc threads (default 32)
"""

import os
import numpy as np
import h5py

from baqaro.plotting_common.load_data_to_plot import (
    path_file, load_simulation_metadata, source_dir, simulation_name,
    snapshot_index_for_redshift,
)
from baqaro.utils.my_dir import get_output_path, get_input_path_HBT_data
from baqaro.clustering_direct.cache_provenance import (
    provenance_dict, require_full_catalogue,
)
from baqaro.clustering_direct.measure_clustering_direct import (
    measure_natural_xi, BOX_CMPC,
)
from qhtools.utils import my_utils
from qhtools.utils.cosmology import cosmo                  # FLAMINGO colossus cosmology


def read_full_positions(true_snap):
    """Read the WHOLE ComovingAveragePosition (N,3) for a snapshot into RAM as
    float32 (one sequential read, chunked), so all L-bins can be indexed in
    memory. Far faster than repeated scattered fancy reads of ~1e5-1e6 rows from
    the multi-GB chunked dataset."""
    template = os.path.join(get_input_path_HBT_data(source_dir), simulation_name,
                            "HBT_compressed", "OrderedSubSnap_{snap_nr:03d}.hdf5")
    with h5py.File(template.format(snap_nr=int(true_snap)), "r",
                   rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as f:
        dset = f["Subhalos/ComovingAveragePosition"]      # (N, 3) float64
        N = dset.shape[0]
        pos = np.empty((N, 3), dtype=np.float32)
        step = 100_000_000
        for i0 in range(0, N, step):
            pos[i0:i0 + step] = dset[i0:i0 + step]
    return pos


def read_reference_ids(true_snap, logm_lo, logm_hi, ref_max, rng):
    """Global row IDs of a LOWER-mass halo tracer sample for cross-correlation:
    ALIVE subhalos (SnapshotIndexOfDeath == -1) with instantaneous bound mass in
    [logm_lo, logm_hi] (log10 Msun). HBT ``Mbound`` is in 1e10-Msun units, so the
    raw window is 10**(logM-10). Sequential chunked reads of Mbound+death only
    (positions are indexed later from the already-in-RAM pos_full). Returns
    (global_ids, n_in_window)."""
    template = os.path.join(get_input_path_HBT_data(source_dir), simulation_name,
                            "HBT_compressed", "OrderedSubSnap_{snap_nr:03d}.hdf5")
    raw_lo, raw_hi = 10.0 ** (logm_lo - 10.0), 10.0 ** (logm_hi - 10.0)
    with h5py.File(template.format(snap_nr=int(true_snap)), "r",
                   rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as f:
        mb_d = f["Subhalos/Mbound"]; dz_d = f["Subhalos/SnapshotIndexOfDeath"]
        N = mb_d.shape[0]
        keep = np.empty(N, dtype=bool)
        step = 100_000_000
        for i0 in range(0, N, step):
            i1 = min(i0 + step, N)
            mb = mb_d[i0:i1]
            dz = dz_d[i0:i1]
            keep[i0:i1] = (dz == -1) & (mb >= raw_lo) & (mb < raw_hi)
    ids = np.flatnonzero(keep)
    n_win = ids.size
    if n_win > ref_max:
        ids = np.sort(rng.choice(ids, size=ref_max, replace=False))
    return ids, n_win


def _env(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def parse_lbins(s):
    """Parse a ``"45.5-46,46-46.5"`` env string into ``[(lo, hi), ...]`` in log L_bol."""
    out = []
    for tok in s.split(","):
        lo, hi = tok.split("-")
        out.append((float(lo), float(hi)))
    return out


def main():
    """Measure b_Q per luminosity bin and per redshift, and cache it to .npz.

    For each z target and L_bol bin: measure xi_QQ from the run's quasar
    positions, divide by the linear matter xi_mm at the SAME redshift, and fit a
    constant over the linear range (default 10-40 cMpc). Writes one provenance-
    stamped .npz per configuration; see ``cache_provenance.py`` for the read-side
    guard that stops a cache being reused against a different run.

    Configured entirely through ``BAQARO_BIASDIR_*`` / ``BAQARO_CLUST_NTHREADS``.
    """
    z_targets = [float(z) for z in _env("BAQARO_BIASDIR_Z", "2.0,2.5,3.0").split(",")]
    lbins = parse_lbins(_env("BAQARO_BIASDIR_LBINS", "45.5-46,46-46.5,46.5-47,47-48"))
    qso_max = int(_env("BAQARO_BIASDIR_QSO_MAX", "500000"))
    min_n = int(_env("BAQARO_BIASDIR_MINN", "300"))
    r_lin_lo, r_lin_hi = [float(x) for x in _env("BAQARO_BIASDIR_RLIN", "10,40").split(",")]
    nthreads = int(_env("BAQARO_CLUST_NTHREADS", "32"))
    outdir = os.path.join(get_output_path(source_dir), "clustering_direct")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    edges = np.logspace(np.log10(0.5), np.log10(120.0), 26)   # r-bins [cMpc]
    r_geom = 0.5 * (edges[:-1] + edges[1:])

    print(f"[bias-direct] {path_file}")
    print(f"[bias-direct] z={z_targets} lbins={lbins} qso_max={qso_max} rlin=({r_lin_lo},{r_lin_hi})")

    with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as f:
        meta = load_simulation_metadata(f)
        loader = meta["loader"]
        redshifts = np.asarray(meta["redshifts"])
        subset_indices = loader._subset_indices
        require_full_catalogue(loader, what="Direct bias")
        snapshots = loader._snapshots

        nb, nz = len(lbins), len(z_targets)
        z_arr = np.zeros(nz)
        bQ = np.full((nb, nz), np.nan)
        bQ_err = np.full((nb, nz), np.nan)
        n_qso = np.zeros((nb, nz), dtype=np.int64)
        out = os.path.join(outdir, "quasar_bias_direct_xiQQ.npz")

        _prov = provenance_dict("direct_bias", loader, requires_full_cat=True)
        def _save():                       # incremental: persist after each z
            np.savez(out, redshifts=z_arr,
                     lbins_lo=np.array([b[0] for b in lbins]),
                     lbins_hi=np.array([b[1] for b in lbins]), b_Q=bQ, b_Q_err=bQ_err,
                     n_qso=n_qso, r_lin=np.array([r_lin_lo, r_lin_hi]), **_prov)

        for j, zt in enumerate(z_targets):
            col = snapshot_index_for_redshift(
                redshifts, zt, snapshots, label="bias-direct")
            z = float(redshifts[col]); z_arr[j] = z
            true_snap = int(snapshots[col])
            Lbol = np.asarray(loader.get_Lbol(col))
            xi_mm = np.asarray(cosmo.correlationFunction(r_geom * cosmo.h, z=z))
            print(f"\n=== z={zt} (file z={z:.3f}, snap {true_snap}) ===", flush=True)
            pos_full = read_full_positions(true_snap)      # (N,3) float32, read ONCE
            print(f"  positions in RAM: {pos_full.shape}", flush=True)

            for b, (lo, hi) in enumerate(lbins):
                mask = (Lbol > 10 ** my_utils.to_solar(lo)) & (Lbol <= 10 ** my_utils.to_solar(hi))
                storage = np.flatnonzero(mask)
                n = storage.size
                n_qso[b, j] = n
                if n < min_n:
                    print(f"  L[{lo},{hi}]: n={n} (<{min_n}, skip)"); continue
                if n > qso_max:                       # subsample (auto-corr amplitude unbiased)
                    storage = np.sort(rng.choice(storage, size=qso_max, replace=False))
                gids = subset_indices[storage]
                pos = np.ascontiguousarray(np.mod(pos_full[gids], BOX_CMPC).T)  # (3,Nsel) cMpc
                r_c, xi = measure_natural_xi(pos, None, BOX_CMPC, edges, nthreads)

                sel = (r_c >= r_lin_lo) & (r_c <= r_lin_hi) & (xi > 0) & (xi_mm > 0)
                if sel.sum() < 3:
                    print(f"  L[{lo},{hi}]: n={n} but <3 usable linear bins, skip"); continue
                ratio = xi[sel] / xi_mm[sel]
                b2 = xi[sel].sum() / xi_mm[sel].sum()           # amplitude match
                bQ[b, j] = np.sqrt(b2)
                # jackknife over the linear r-bins
                idx = np.flatnonzero(sel)
                jk = np.array([np.sqrt(np.delete(xi[idx], k).sum() / np.delete(xi_mm[idx], k).sum())
                               for k in range(idx.size)])
                bQ_err[b, j] = np.sqrt((idx.size - 1) / idx.size * np.sum((jk - jk.mean()) ** 2))
                print(f"  L[{lo},{hi}]: n={n:>9,d} (used {storage.size:,})  "
                      f"b_Q={bQ[b,j]:.2f} +/- {bQ_err[b,j]:.2f}  "
                      f"(nbins={sel.sum()}, ratio spread {ratio.min():.2f}-{ratio.max():.2f})")
            del pos_full                                    # free before next snapshot
            _save()                                          # persist progress through this z
            print(f"  [saved through z={z:.3f}] -> {out}", flush=True)

        print(f"\n[bias-direct] -> {out}")


def main_cross():
    """CROSS-correlation bias: b_Q from xi_Q,ref against a dense LOWER-mass halo
    tracer, instead of the pair-starved quasar auto-corr. For rare/bright/high-z
    bins the cross pair count is N_Q*N_ref (not N_Q^2), so it stays well-measured
    where the auto-corr is dominated by shot noise. This is how the z~6 JWST
    surveys (EIGER, ASPIRE, He+18) measure quasar bias.

    Self-calibrated (no halo-model assumption):
      xi_Qr = b_Q b_ref xi_mm   (cross)   -> b_Q b_ref = S_Qr / S_mm
      xi_rr = b_ref^2  xi_mm    (ref auto) -> b_ref     = sqrt(S_rr / S_mm)
      => b_Q = S_Qr / sqrt(S_rr * S_mm)    (S_* = sum over linear r-bins)
    b_ref is measured from the sim itself, so only xi_mm (colossus) is external."""
    z_targets = [float(z) for z in _env("BAQARO_BIASDIR_CROSS_Z", "3.0,4.0,5.0,6.1").split(",")]
    lbins = parse_lbins(_env("BAQARO_BIASDIR_LBINS", "45.5-46,46-46.5,46.5-47,47-48"))
    qso_max = int(_env("BAQARO_BIASDIR_QSO_MAX", "500000"))
    min_n = int(_env("BAQARO_BIASDIR_MINN", "50"))       # cross tolerates far fewer QSO
    ref_lo, ref_hi = [float(x) for x in _env("BAQARO_BIASDIR_REF_LOGM", "11,12").split(",")]
    ref_max = int(_env("BAQARO_BIASDIR_REF_MAX", "4000000"))
    r_lin_lo, r_lin_hi = [float(x) for x in _env("BAQARO_BIASDIR_RLIN", "10,40").split(",")]
    nthreads = int(_env("BAQARO_CLUST_NTHREADS", "32"))
    outdir = os.path.join(get_output_path(source_dir), "clustering_direct")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(42)

    edges = np.logspace(np.log10(0.5), np.log10(120.0), 26)
    r_geom = 0.5 * (edges[:-1] + edges[1:])

    print(f"[bias-cross] {path_file}")
    print(f"[bias-cross] z={z_targets} ref_logM=[{ref_lo},{ref_hi}] ref_max={ref_max} "
          f"qso_max={qso_max} rlin=({r_lin_lo},{r_lin_hi})")

    with h5py.File(path_file, "r", rdcc_nbytes=512 * 1024 * 1024, rdcc_nslots=10007) as f:
        meta = load_simulation_metadata(f)
        loader = meta["loader"]
        redshifts = np.asarray(meta["redshifts"])
        subset_indices = loader._subset_indices
        require_full_catalogue(loader, what="Direct bias")
        snapshots = loader._snapshots

        nb, nz = len(lbins), len(z_targets)
        z_arr = np.zeros(nz)
        bQ = np.full((nb, nz), np.nan)
        bQ_err = np.full((nb, nz), np.nan)
        n_qso = np.zeros((nb, nz), dtype=np.int64)
        b_ref_arr = np.full(nz, np.nan)
        n_ref_arr = np.zeros(nz, dtype=np.int64)
        out = os.path.join(outdir, "quasar_bias_direct_cross.npz")

        _prov = provenance_dict("direct_bias", loader, requires_full_cat=True)
        def _save():
            np.savez(out, redshifts=z_arr,
                     lbins_lo=np.array([b[0] for b in lbins]),
                     lbins_hi=np.array([b[1] for b in lbins]), b_Q=bQ, b_Q_err=bQ_err,
                     n_qso=n_qso, b_ref=b_ref_arr, n_ref=n_ref_arr,
                     ref_logm=np.array([ref_lo, ref_hi]), r_lin=np.array([r_lin_lo, r_lin_hi]), **_prov)

        for j, zt in enumerate(z_targets):
            col = snapshot_index_for_redshift(
                redshifts, zt, snapshots, label="bias-direct")
            z = float(redshifts[col]); z_arr[j] = z
            true_snap = int(snapshots[col])
            Lbol = np.asarray(loader.get_Lbol(col))
            xi_mm = np.asarray(cosmo.correlationFunction(r_geom * cosmo.h, z=z))
            sel_mm = (r_geom >= r_lin_lo) & (r_geom <= r_lin_hi) & (xi_mm > 0)
            print(f"\n=== z={zt} (file z={z:.3f}, snap {true_snap}) ===", flush=True)
            pos_full = read_full_positions(true_snap)
            ref_gids, n_win = read_reference_ids(true_snap, ref_lo, ref_hi, ref_max, rng)
            n_ref_arr[j] = ref_gids.size
            pos_ref = np.ascontiguousarray(np.mod(pos_full[ref_gids], BOX_CMPC).T)  # (3,Nref)
            print(f"  positions in RAM: {pos_full.shape}; reference halos "
                  f"logM[{ref_lo},{ref_hi}]: {n_win:,} in window, using {ref_gids.size:,}", flush=True)

            # reference AUTO once per snapshot -> b_ref (self-calibration)
            _, xi_rr = measure_natural_xi(pos_ref, None, BOX_CMPC, edges, nthreads)
            sel_r = sel_mm & (xi_rr > 0)
            S_rr = xi_rr[sel_r].sum(); S_mm_r = xi_mm[sel_r].sum()
            b_ref = np.sqrt(S_rr / S_mm_r)
            b_ref_arr[j] = b_ref
            print(f"  reference auto: b_ref={b_ref:.2f} (nbins={sel_r.sum()})", flush=True)

            for b, (lo, hi) in enumerate(lbins):
                mask = (Lbol > 10 ** my_utils.to_solar(lo)) & (Lbol <= 10 ** my_utils.to_solar(hi))
                storage = np.flatnonzero(mask)
                n = storage.size
                n_qso[b, j] = n
                if n < min_n:
                    print(f"  L[{lo},{hi}]: n={n} (<{min_n}, skip)"); continue
                if n > qso_max:
                    storage = np.sort(rng.choice(storage, size=qso_max, replace=False))
                gids = subset_indices[storage]
                pos_q = np.ascontiguousarray(np.mod(pos_full[gids], BOX_CMPC).T)
                _, xi_qr = measure_natural_xi(pos_q, pos_ref, BOX_CMPC, edges, nthreads)

                sel = sel_mm & (xi_qr > 0) & (xi_rr > 0)
                if sel.sum() < 3:
                    print(f"  L[{lo},{hi}]: n={n} but <3 usable linear bins, skip"); continue
                idx = np.flatnonzero(sel)
                # b_Q = S_Qr / sqrt(S_rr * S_mm), self-calibrated on the reference auto
                def _bq(mask_idx):
                    return (xi_qr[mask_idx].sum()
                            / np.sqrt(xi_rr[mask_idx].sum() * xi_mm[mask_idx].sum()))
                bQ[b, j] = _bq(idx)
                jk = np.array([_bq(np.delete(idx, k)) for k in range(idx.size)])
                bQ_err[b, j] = np.sqrt((idx.size - 1) / idx.size * np.sum((jk - jk.mean()) ** 2))
                print(f"  L[{lo},{hi}]: n={n:>9,d} (used {storage.size:,})  "
                      f"b_Q={bQ[b,j]:.2f} +/- {bQ_err[b,j]:.2f}  (b_ref={b_ref:.2f}, nbins={sel.sum()})",
                      flush=True)
            del pos_full, pos_ref
            _save()
            print(f"  [saved through z={z:.3f}] -> {out}", flush=True)

        print(f"\n[bias-cross] -> {out}")


if __name__ == "__main__":
    if _env("BAQARO_BIASDIR_MODE", "auto") == "cross":
        main_cross()
    else:
        main()
