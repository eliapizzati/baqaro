"""
Phase-1 gate for the multinode forward engine (``main_evolution_chunked``).
===========================================================================

Exercises the pure evolution kernel ``evolve_chunk`` on a small synthetic sim
held entirely in RAM (no on-disk halo files), validating that splitting the
catalogue into chunks and recombining reproduces the single-run result.

To make the cross-chunk check EXACT (not merely statistical), we run with
accretion OFF and deterministic seeding (``sigmaseed=None``): then every
halo's M_BH is a deterministic function of its seed mass + merger history,
independent of chunk count and RNG. The partition is disjoint + complete
(``test_chunk_partition``), so:

  * pooled BHMF (un-log, sum densities, re-log over chunks) == single-run BHMF,
  * concatenated per-halo catalogue (union over chunks) == single-run catalogue
    (same track ids AND masses) at every anchor,
  * the union of survivors == single-run survivors.

Also pins the two structural guarantees that distinguish this engine from a
naive carry-forward:

  * FRESH-ZERO dead halos: a disrupted dead-end BH appears at an intermediate
    anchor but is GONE at z=0 (its lineage left no survivor).
  * mass conservation across a merger chain.

Run directly (``python -m ...tests.test_evolution_chunked``) or via pytest.
"""

import numpy as np
from numpy.random import default_rng

from qhtools.utils import my_utils

from baqaro.core_functions.tree_subsample import (
    build_multinode_partition_subset, precompute_per_snapshot_indices,
)
from baqaro.core_functions.main_evolution_chunked import evolve_chunk


# ----------------------------------------------------------------------
# Synthetic sim with per-snapshot histories
# ----------------------------------------------------------------------
def make_sim():
    """Small forest with merger chains, a dead-end disrupted lineage, and a
    never-resolved halo. Returns the arrays evolve_chunk + the partition need.

    LastMaxMass convention: a halo's mass is its (constant) peak from birth
    onward and PERSISTS after death (monotonic), so ``mass[snap] > 0`` for any
    snap >= birth — matching HBT.
    """
    n_snap = 6
    birth, death, merger, peak = [], [], [], []

    def add(b, d, m, pk):
        birth.append(b); death.append(d); merger.append(m); peak.append(pk)
        return len(birth) - 1

    # survivor roots (death=-1, merger=-1)
    r0 = add(0, -1, -1, 1.0e12)
    r1 = add(0, -1, -1, 2.0e12)
    r2 = add(1, -1, -1, 1.5e12)
    # progenitors merging into roots (+ a chain into r0)
    p0 = add(0, 2, r0, 5.0e11)
    add(0, 1, p0, 2.0e11)            # grand-prog: chain gp -> p0 -> r0
    add(1, 3, r1, 4.0e11)
    add(2, 4, r2, 3.0e11)
    # dead-end disrupted terminal (merger=-1, death>=0) + its progenitor
    d0 = add(0, 3, -1, 8.0e11)
    add(0, 2, d0, 2.0e11)
    add(3, 4, -1, 6.0e11)            # second dead-end terminal
    # never-resolved halo (mass 0) — must be excluded by resolved_mask
    add(0, 2, -1, 0.0)

    n = len(birth)
    birth = np.array(birth, np.int64)
    death = np.array(death, np.int64)
    merger = np.array(merger, np.int64)
    peak = np.array(peak, np.float64)

    masses = np.zeros((n_snap, n), dtype=np.float32)
    for h in range(n):
        masses[birth[h]:, h] = peak[h]      # peak persists from birth onward
    cold = np.zeros((n_snap, n), dtype=np.float32)  # unused (accretion off)

    return dict(
        n_snap=n_snap, n=n,
        track_ids=np.arange(n, dtype=np.int64),
        mt_birth_index=birth, mt_death_index=death, mt_merger_ids=merger,
        masses=masses, cold=cold,
    )


# ----------------------------------------------------------------------
# Fixed stats config (shared across runs)
# ----------------------------------------------------------------------
def _stats_cfg():
    boxsize = 100.0
    bhmf_bins = np.logspace(6.5, 11.0, 41)
    qlf_bins = np.logspace(my_utils.to_solar(44), my_utils.to_solar(48.5), 41)
    cerdf_bins = np.logspace(-3, 2, 41)
    qhmf_bins = np.logspace(10.5, 14.5, 41)
    log_L_thresholds = np.linspace(45.5, 47.5, 21)
    Lcuts = 10 ** my_utils.to_solar(log_L_thresholds.astype(np.float64))

    def norm(bins):
        return 1.0 / ((np.log10(bins[1]) - np.log10(bins[0])) * boxsize ** 3)

    return dict(
        qlf_bins=qlf_bins, qlf_normalization=norm(qlf_bins),
        bhmf_bins=bhmf_bins, bhmf_normalization=norm(bhmf_bins),
        cerdf_bins=cerdf_bins, cerdf_normalization=norm(cerdf_bins),
        qhmf_bins=qhmf_bins, qhmf_normalization=norm(qhmf_bins),
        log_L_thresholds=log_L_thresholds, Lcuts=Lcuts, lowest_cut=Lcuts[0],
        log_csi_10=1.0e4, mass_units_f32=np.float32(1.0),
    )


def _run_chunk(sim, *, chunk_id, n_chunks, snapshots_to_save, anchors, cfg,
               accretion_on=False):
    """Build one chunk's partition + precomp, run evolve_chunk, return the
    output dict plus the collected anchor catalogue."""
    subset = build_multinode_partition_subset(
        track_ids=sim["track_ids"], merger_track_ids=sim["mt_merger_ids"],
        resolved_mask=sim["masses"][-1] > 0,   # LastMaxMass at z=0 row
        chunk_id=chunk_id, n_chunks=n_chunks,
    )
    subset_indices = np.flatnonzero(subset["subset_mask"])
    n_storage = subset_indices.size
    g2s = np.full(sim["n"], -1, dtype=np.int64)
    g2s[subset_indices] = np.arange(n_storage)

    pre = precompute_per_snapshot_indices(
        min_snap=0, max_snap=sim["n_snap"] - 1,
        mt_birth_index=sim["mt_birth_index"], mt_death_index=sim["mt_death_index"],
        mask_merged=sim["mt_merger_ids"] != -1,
        mask_disrupted=sim["mt_merger_ids"] == -1,
        subset_indices=subset_indices, verbose=False,
    )

    catalog = {}

    def sink(snap, z, tid, mbh, lbol):
        catalog[snap] = dict(track_id=tid.copy(), M_BH=mbh.copy(), Lbol=lbol.copy())

    snap_to_zindex = {int(s): iz for iz, s in enumerate(snapshots_to_save)}
    out = evolve_chunk(
        subset_indices=subset_indices, global_to_storage=g2s, n_storage=n_storage,
        halo_masses_all=sim["masses"][:, subset_indices],
        halo_specific_cold_accretion_rates_all=sim["cold"][:, subset_indices],
        mt_track_ids=sim["track_ids"], mt_merger_ids=sim["mt_merger_ids"],
        mt_death_index=sim["mt_death_index"],
        precomp_spawning=pre["spawning"], precomp_evolving=pre["evolving"],
        precomp_merging=pre["merging"],
        snapshots=np.arange(sim["n_snap"]), snap_to_zindex=snap_to_zindex,
        n_z=len(snapshots_to_save), anchor_set=set(anchors),
        redshifts=np.linspace(6.0, 0.0, sim["n_snap"]),
        delta_times_snapshots=np.full(sim["n_snap"], 0.1),
        erdf=None, rng_run=default_rng(0), transfer_function=None,
        time_step_for_accretion=1.0, accretion_on=accretion_on, merger_on=True,
        nonaccreting_zero_rate=True, rad_efficiency_0=0.1,
        rad_efficiency_model="constant", growth_sum_max=50.0,
        madau_feff_correction=False, logfseed=-3.0, sigmaseed=None,
        anchor_sink=sink, **cfg,
    )
    return out, catalog


def _pool_log(stat_list):
    """Un-log, sum densities across chunks, re-log (the pool_chunks recipe)."""
    acc = None
    for s in stat_list:
        with np.errstate(over="ignore"):
            dens = np.where(np.isfinite(s), np.power(10.0, s, dtype=np.float64), 0.0)
        acc = dens if acc is None else acc + dens
    with np.errstate(divide="ignore"):
        return np.where(acc > 0.0, np.log10(acc), -np.inf)


def _cat_to_dict(catalog):
    """Flatten {snap: {...}} -> {(snap, track_id): M_BH} for set comparison."""
    out = {}
    for snap, d in catalog.items():
        for tid, m in zip(d["track_id"], d["M_BH"]):
            out[(int(snap), int(tid))] = float(m)
    return out


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_pool_and_concat_match_single_run():
    """Accretion OFF + deterministic seeding -> pooled BHMF and concatenated
    catalogue EXACTLY reproduce the single-run result, for several n_chunks."""
    sim = make_sim()
    cfg = _stats_cfg()
    save = [1, 3, 5]
    single, cat1 = _run_chunk(sim, chunk_id=0, n_chunks=1, snapshots_to_save=save,
                              anchors=save, cfg=cfg)
    cat1_flat = _cat_to_dict(cat1)
    surv1 = set(int(t) for t in single["survivor_track_ids"])

    for K in (2, 3):
        outs, cats = [], {}
        surv_union = set()
        for c in range(K):
            o, cat = _run_chunk(sim, chunk_id=c, n_chunks=K, snapshots_to_save=save,
                                anchors=save, cfg=cfg)
            outs.append(o)
            surv_union |= set(int(t) for t in o["survivor_track_ids"])
            for snap, d in cat.items():
                cats.setdefault(snap, {"track_id": [], "M_BH": [], "Lbol": []})
                for k in ("track_id", "M_BH", "Lbol"):
                    cats[snap][k].append(d[k])
        cats = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in cats.items()}

        # BHMF pools exactly (to float roundoff).
        pooled = _pool_log([o["log_bhmfs"] for o in outs])
        s1 = single["log_bhmfs"]
        same_inf = np.isneginf(pooled) == np.isneginf(s1)
        assert same_inf.all(), f"K={K}: BHMF empty-bin pattern differs"
        fin = np.isfinite(s1)
        assert np.allclose(pooled[fin], s1[fin], atol=1e-6), f"K={K}: BHMF pool != single"

        # Catalogue concat == single (same (snap, track_id) -> M_BH map).
        assert _cat_to_dict(cats) == cat1_flat, f"K={K}: concatenated catalogue != single"
        # Survivors union == single.
        assert surv_union == surv1, f"K={K}: survivor union != single"


def test_deadend_zeroed_at_z0_but_present_intermediate():
    """A disrupted dead-end BH (track 7, death=3) is present at an intermediate
    anchor but ABSENT at z=0 — validates fresh-zero dead-halo handling and that
    dead-ends contribute to intermediate-z stats."""
    sim = make_sim()
    cfg = _stats_cfg()
    save = [1, 3, 5]
    _, cat = _run_chunk(sim, chunk_id=0, n_chunks=1, snapshots_to_save=save,
                        anchors=save, cfg=cfg)
    deadend = 7  # d0: birth=0, death=3, merger=-1
    present_at_1 = deadend in set(int(t) for t in cat[1]["track_id"])
    present_at_5 = deadend in set(int(t) for t in cat[5]["track_id"])
    assert present_at_1, "dead-end BH should host a BH while alive (snap 1)"
    assert not present_at_5, "dead-end BH must be GONE at z=0 (no carry-forward)"


def test_merger_mass_conservation():
    """Total BH mass at z=0 == seeds that flowed into surviving roots. Concretely:
    each survivor's z=0 M_BH equals the sum of seed masses of every halo in its
    component that survived to (or merged before) z=0, with disrupted lineages
    removed. Here we check global conservation: z=0 total == sum over survivors,
    and that it equals the analytic expectation for this forest."""
    sim = make_sim()
    cfg = _stats_cfg()
    out, cat = _run_chunk(sim, chunk_id=0, n_chunks=1, snapshots_to_save=[5],
                          anchors=[5], cfg=cfg)
    fseed = 10 ** (-3.0)
    # r0 component that survives to z=0: r0 + p0 + gp(prog of p0) all flow into r0
    #   (gp->p0 at snap1, p0->r0 at snap2). r1: r1 + its prog (merges at 3).
    #   r2: r2 + its prog (merges at 4). Dead-ends d0(+prog) and the 2nd dead-end
    #   leave NO survivor mass. Never-resolved contributes nothing.
    peak = {0: 1.0e12, 1: 2.0e12, 2: 1.5e12, 3: 5.0e11, 4: 2.0e11,
            5: 4.0e11, 6: 3.0e11}
    expected_total = fseed * sum(peak.values())
    z0_total = float(cat[5]["M_BH"].sum())
    assert np.isclose(z0_total, expected_total, rtol=1e-5), (
        f"z=0 BH mass {z0_total:.6e} != expected {expected_total:.6e}")
    # And it equals the survivor-list total.
    surv_total = float(out["survivor_M_z0_code"].sum())  # mass_units=1
    assert np.isclose(surv_total, expected_total, rtol=1e-5)


def test_merger_onto_long_dead_target_is_aborted():
    """A merger whose target died at an EARLIER snapshot must be ABORTED
    (source BH mass lost), not deposited into the dead target's row. Halo
    mass is frozen LastMaxMass, so a ``mass > 0`` test alone would pass
    long-dead targets; the target must also be alive at the merger snapshot.
    """
    n_snap = 6
    #        r0 (survivor)   t0 (dies snap 2, disrupted)   x0 (merges into t0 at snap 4)
    birth = np.array([0, 0, 0], np.int64)
    death = np.array([-1, 2, 4], np.int64)
    merger = np.array([-1, -1, 1], np.int64)   # x0 -> t0, but t0 died at 2 < 4
    peak = np.array([1.0e12, 8.0e11, 5.0e11])

    masses = np.zeros((n_snap, 3), dtype=np.float32)
    for h in range(3):
        masses[birth[h]:, h] = peak[h]         # LastMaxMass persists after death
    sim = dict(
        n_snap=n_snap, n=3,
        track_ids=np.arange(3, dtype=np.int64),
        mt_birth_index=birth, mt_death_index=death, mt_merger_ids=merger,
        masses=masses, cold=np.zeros_like(masses),
    )

    cfg = _stats_cfg()
    save = [3, 4, 5]
    out, cat = _run_chunk(sim, chunk_id=0, n_chunks=1, snapshots_to_save=save,
                          anchors=save, cfg=cfg)

    ids_at = {s: set(int(t) for t in cat[s]["track_id"]) for s in save}
    # snap 3: r0 alive, x0 alive, t0 long dead.
    assert ids_at[3] == {0, 2}, f"snap 3 live BHs {ids_at[3]} != {{0, 2}}"
    # snap 4: the x0 -> t0 merger must be aborted — no BH on the dead t0
    # track, and x0 is dead.
    assert ids_at[4] == {0}, f"snap 4 live BHs {ids_at[4]} != {{0}} (deposit onto a dead target?)"
    assert ids_at[5] == {0}
    assert set(int(t) for t in out["survivor_track_ids"]) == {0}


def test_merger_into_never_resolved_target_is_aborted_not_teleported():
    """A resolved halo merging into a NEVER-resolved target (mass 0, and
    birth = death = -1 after the resolution filter) must be ABORTED. The
    multinode partition intersects the closure with ``resolved_mask``, so the
    target is not in storage and ``_to_storage`` returns -1 -- which NumPy
    reads as the LAST storage slot. The source BH must be lost, not
    deposited onto whichever halo happens to own that slot.
    """
    n_snap = 6
    #     r0, r1 survivors; A resolved, dies at 3 merging into U; U never
    #     resolved; S survivor with the HIGHEST track id -> last storage slot.
    birth = np.array([0, 0, 0, -1, 0], np.int64)
    death = np.array([-1, -1, 3, -1, -1], np.int64)
    merger = np.array([-1, -1, 3, -1, -1], np.int64)      # A -> U
    peak = np.array([1.0e12, 2.0e12, 7.0e11, 0.0, 3.0e12])
    A, U, S = 2, 3, 4

    masses = np.zeros((n_snap, 5), dtype=np.float32)
    for h in range(5):
        if peak[h] > 0:
            masses[birth[h]:, h] = peak[h]
    sim = dict(
        n_snap=n_snap, n=5,
        track_ids=np.arange(5, dtype=np.int64),
        mt_birth_index=birth, mt_death_index=death, mt_merger_ids=merger,
        masses=masses, cold=np.zeros_like(masses),
    )
    save = [2, 3, 5]
    out, cat = _run_chunk(sim, chunk_id=0, n_chunks=1, snapshots_to_save=save,
                          anchors=save, cfg=_stats_cfg())

    # U must not be in storage at all (that is what makes the case reachable).
    assert U not in set(int(t) for t in out["survivor_track_ids"])
    ids_at = {s: set(int(t) for t in cat[s]["track_id"]) for s in save}
    assert ids_at[2] == {0, 1, A, S}
    assert ids_at[3] == {0, 1, S}, f"snap 3 live BHs {ids_at[3]}"
    assert ids_at[5] == {0, 1, S}

    # S keeps exactly its own seed mass: A's BH is lost, not teleported.
    seed = 10.0 ** -3.0                              # logfseed = -3, mass units 1
    surv = dict(zip((int(t) for t in out["survivor_track_ids"]),
                    (float(m) for m in out["survivor_M_z0_code"])))
    assert np.isclose(surv[S], seed * peak[S], rtol=1e-6), (
        f"last-slot survivor S carries {surv[S]:.4g}, expected its own seed "
        f"{seed * peak[S]:.4g}: the aborted merger teleported A's BH into it")
    assert np.isclose(surv[0], seed * peak[0], rtol=1e-6)
    assert np.isclose(surv[1], seed * peak[1], rtol=1e-6)


if __name__ == "__main__":
    test_pool_and_concat_match_single_run()
    print("OK  test_pool_and_concat_match_single_run")
    test_deadend_zeroed_at_z0_but_present_intermediate()
    print("OK  test_deadend_zeroed_at_z0_but_present_intermediate")
    test_merger_mass_conservation()
    print("OK  test_merger_mass_conservation")
    test_merger_onto_long_dead_target_is_aborted()
    print("OK  test_merger_onto_long_dead_target_is_aborted")
    test_merger_into_never_resolved_target_is_aborted_not_teleported()
    print("OK  test_merger_into_never_resolved_target_is_aborted_not_teleported")
    print("\nAll multinode evolution gate tests passed.")
