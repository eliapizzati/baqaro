"""
Phase-2/4 gate for the multinode recombiner (``core_functions/concat_chunks``).
==============================================================================

Closes the loop on the synthetic sim from ``test_evolution_chunked``:
run it as N chunks, write each chunk's per-anchor sparse catalogue + stats +
merger files, then:

  * STATS   — ``combine_stats``: pooled == single run (exact, accretion off).
  * MERGER  — ``combine_merger``: events + survivors == single run.
  * PER-HALO — ``write_dense_bh_evolution``: builds the STANDARD
    ``bh_evolution`` file (dense union-subset, weight 1) and we verify it
    through the REAL plotting ``DataLoader``:
      - dense scatter is faithful (n_chunks=1 and n_chunks=3 give identical
        dense arrays; nonzero entries == the single-run catalogue),
      - ``get_BH_mass(pos)`` returns the row at position ``pos``,
      - ``get_Halo_mass(pos)`` reads the CORRECT snapshot row from the full-N
        halo .npy (the position→snapshot translation — the bug this guards),
      - ``weights`` are all 1, ``subset_indices`` is the sorted union.

Run directly (``python -m ...tests.test_concat_chunks``) or via pytest.
"""

import os
import tempfile

import h5py
import numpy as np

from baqaro.core_functions import concat_chunks as cc
from baqaro.utils.my_units import mass_units
from baqaro.tests.test_evolution_chunked import (
    make_sim, _stats_cfg, _run_chunk,
)


# ---- mini-writers replicating main_evolution_chunked.main()'s output schema --
def _write_stats(path, out, tag, snaps):
    with h5py.File(path, "w") as f:
        for k in ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs"):
            f.create_dataset(k, data=out[k])
        f.create_dataset("snapshots_to_save", data=np.asarray(snaps, np.int64))
        g = f.create_group("provenance")
        g.attrs["subset_tag"] = tag
        g.attrs["erdf_model"] = "log_normal_evol_halo_mass"


def _write_catalog(path, catalog, tag):
    with h5py.File(path, "w") as f:
        f.attrs["subset_tag"] = tag
        for snap, d in catalog.items():
            grp = f.create_group(f"snap_{int(snap):03d}")
            grp.attrs["snapshot"] = int(snap)
            grp.create_dataset("track_id", data=d["track_id"])
            grp.create_dataset("M_BH", data=d["M_BH"])
            grp.create_dataset("Lbol", data=d["Lbol"])


def _write_merger(path, out, chunk_id):
    cat = out["merger_catalog"]
    with h5py.File(path, "w") as f:
        f.create_dataset("z", data=cat["z"])
        f.create_dataset("M1", data=cat["M1_Msun"].astype(np.float32))
        f.create_dataset("M2", data=cat["M2_Msun"].astype(np.float32))
        f.create_dataset("progenitor1_id", data=cat["progenitor1_id"])
        f.create_dataset("progenitor2_id", data=cat["progenitor2_id"])
        f.create_dataset("descendant_id", data=cat["descendant_id"])
        f.create_dataset("halo_id", data=cat["halo_id"])
        f.create_dataset("survivor_id", data=out["survivor_track_ids"])
        f.create_dataset("survivor_M_z0", data=out["survivor_M_z0_code"].astype(np.float32))
        f.attrs["multinode_chunk_id"] = int(chunk_id)
        f.attrs["simulation_name"] = "synthetic"


def _run_and_write(sim, cfg, save, tmp, *, K, tag):
    """Run K chunks, write per-chunk stats/catalog/merger files; return paths."""
    sp, cp, mp = [], [], []
    for c in range(K):
        o, cat = _run_chunk(sim, chunk_id=c, n_chunks=K, snapshots_to_save=save,
                            anchors=save, cfg=cfg)
        s = os.path.join(tmp, f"{tag}_stats_{c}.hdf5")
        ca = os.path.join(tmp, f"{tag}_cat_{c}.hdf5")
        m = os.path.join(tmp, f"{tag}_mrg_{c}.hdf5")
        _write_stats(s, o, f"multinode_root5_v3{'' if K == 1 else f'_chunk{c}of{K}'}", save)
        _write_catalog(ca, cat, "multinode_root5_v3")
        _write_merger(m, o, c)
        sp.append(s); cp.append(ca); mp.append(m)
    return sp, cp, mp


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_stats_and_merger_combine_match_single():
    sim, cfg, save = make_sim(), _stats_cfg(), [1, 3, 5]
    tmp = tempfile.mkdtemp(prefix="concat_sm_")
    (rs,), (rc,), (rm,) = _run_and_write(sim, cfg, save, tmp, K=1, tag="ref")
    sp, cp, mp = _run_and_write(sim, cfg, save, tmp, K=3, tag="k3")

    comb_stats = os.path.join(tmp, "comb_stats.hdf5")
    comb_mrg = os.path.join(tmp, "comb_mrg.hdf5")
    cc.combine_stats(sp, comb_stats, verbose=False)
    cc.combine_merger(mp, comb_mrg, verbose=False)

    with h5py.File(comb_stats, "r") as fc, h5py.File(rs, "r") as fr:
        for k in ("log_qlfs", "log_bhmfs", "log_cerdfs", "log_qhmfs"):
            a, b = fc[k][()], fr[k][()]
            assert np.array_equal(np.isneginf(a), np.isneginf(b))
            fin = np.isfinite(b)
            assert np.allclose(a[fin], b[fin], atol=1e-6)
        assert fc["provenance"].attrs["subset_tag"] == "multinode_root5_v3"

    def mset(p):
        with h5py.File(p, "r") as f:
            ev = sorted(zip(np.round(f["z"][()], 6), np.round(f["M1"][()], 3),
                            np.round(f["M2"][()], 3), f["progenitor1_id"][()],
                            f["descendant_id"][()]))
            sv = set(zip(f["survivor_id"][()], np.round(f["survivor_M_z0"][()], 3)))
        return ev, sv
    assert mset(comb_mrg) == mset(rm)


def test_dense_bh_evolution_and_loader_indexing():
    """Build the dense standard bh_evolution file and read it via the REAL
    DataLoader, locking the position→snapshot halo-mass indexing."""
    from baqaro.plotting_common.load_data_to_plot import DataLoader

    sim, cfg, save = make_sim(), _stats_cfg(), [1, 3, 5]
    tmp = tempfile.mkdtemp(prefix="concat_dense_")
    # synthetic anchor redshift/age/dt arrays (injected — no real sim needed)
    z_full = np.linspace(6.0, 0.0, sim["n_snap"])
    redshifts = z_full[save]
    ages = np.linspace(0.9, 13.0, sim["n_snap"])[save]
    dts = np.full(len(save), 0.1)
    params = {"erdf_model": "log_normal_evol_halo_mass", "subset_tag": "multinode_root5_v3"}

    # single-run reference catalogue + a 3-chunk set
    (_,), (rc,), (_,) = _run_and_write(sim, cfg, save, tmp, K=1, tag="ref")
    _, cp3, _ = _run_and_write(sim, cfg, save, tmp, K=3, tag="k3")

    dense1 = os.path.join(tmp, "bh_evolution_ref.hdf5")
    dense3 = os.path.join(tmp, "bh_evolution_k3.hdf5")
    cc.write_dense_bh_evolution([rc], save, redshifts, ages, dts, params, dense1, verbose=False)
    cc.write_dense_bh_evolution(cp3, save, redshifts, ages, dts, params, dense3, verbose=False)

    # n_chunks=1 and n_chunks=3 dense arrays are identical (accretion off → exact).
    with h5py.File(dense1, "r") as f1, h5py.File(dense3, "r") as f3:
        assert np.array_equal(f1["subset/subset_indices"][()], f3["subset/subset_indices"][()])
        assert np.array_equal(f1["black_hole_masses_all"][()], f3["black_hole_masses_all"][()])
        assert np.array_equal(f1["Lbols_all"][()], f3["Lbols_all"][()])

    # write a synthetic full-N halo_masses .npy (code units) for get_Halo_mass
    halo_npy = os.path.join(tmp, "halo_masses.npy")
    np.save(halo_npy, sim["masses"])  # shape (n_snap, n_halos)

    # single-run catalogue values for cross-check
    with h5py.File(rc, "r") as f:
        cat = {int(f[g].attrs["snapshot"]):
               {"track_id": f[g]["track_id"][()], "M_BH": f[g]["M_BH"][()]}
               for g in f if g.startswith("snap_")}

    with h5py.File(dense1, "r") as f:
        union = f["subset/subset_indices"][()]
        assert np.array_equal(union, np.sort(union))                      # sorted
        assert np.array_equal(f["subset/weights"][()], np.ones(union.size))  # weight 1
        assert np.array_equal(f["snapshots"][()], np.asarray(save))
        loader = DataLoader(f)
        loader.load_halo_masses(halo_npy)   # full-N path (no subset cache)
        for pos, snap in enumerate(save):
            bh = loader.get_BH_mass(pos)
            assert bh.shape == (union.size,)
            # nonzero BH entries reproduce the single-run catalogue at this anchor
            live_cols = np.searchsorted(union, cat[snap]["track_id"])
            assert np.allclose(bh[live_cols], cat[snap]["M_BH"])
            assert np.isclose(bh.sum(), cat[snap]["M_BH"].sum())  # no stray nonzeros
            # get_Halo_mass(pos) must read SNAPSHOT `snap` (not row `pos`) from
            # the full-N .npy, re-indexed by the union. pos != snap here.
            hm = loader.get_Halo_mass(pos)
            expected = sim["masses"][snap][union] * mass_units
            assert np.allclose(hm, expected), (
                f"pos={pos} snap={snap}: get_Halo_mass read the wrong snapshot row")


if __name__ == "__main__":
    test_stats_and_merger_combine_match_single()
    print("OK  test_stats_and_merger_combine_match_single")
    test_dense_bh_evolution_and_loader_indexing()
    print("OK  test_dense_bh_evolution_and_loader_indexing")
    print("\nAll concat-chunks gate tests passed.")
