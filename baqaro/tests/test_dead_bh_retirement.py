"""
Regression test for dead-BH retirement in the training forward model.
=====================================================================

``run_single_model.run_one_model`` must retire every BH whose track dies at
snapshot i (merged, aborted-merged, disrupted) after the merger deposit, and
must filter merger targets on TRUE liveness (``death_index``), not just
frozen ``LastMaxMass > 0`` -- the same semantics as production
``main_evolution.py``, which zero-initializes each snapshot row and writes
only live entries, so that training and production BHMFs share one
definition.

This test drives the real ``run_one_model`` end-to-end on a hand-built
6-halo merger tree with accretion OFF (pure seed/merge/retire
bookkeeping, fully deterministic) and asserts the per-snapshot BHMF bin
occupancy:

  H0: born 0, survives                    (3e8 Msun seed)
  H1: born 0, merges into H0 at snap 3    (3e6)
  H2: born 0, disrupted at snap 3         (3e4)
  H3: born 0, disrupted at snap 1         (3e10)
  H4: born 0, merges at snap 3 into H3 — a target that died at snap 1
      (the frozen-LastMaxMass dead-target trap): must be ABORTED, mass
      lost, nothing parked in H3's slot   (3e2)
  H5: born 2, survives                    (3e12)

Pure numpy — no sampler npz, no halo data. Runs everywhere.
"""
import numpy as np


def _build_config():
    from baqaro.core_functions.tree_subsample import (
        precompute_per_snapshot_indices,
    )

    n_halos = 6
    snaps = np.arange(5)

    mt_track_ids = np.arange(n_halos, dtype=np.int64)
    mt_merger_ids = np.array([-1, 0, -1, -1, 3, -1], dtype=np.int64)
    mt_birth_index = np.array([0, 0, 0, 0, 0, 2], dtype=np.int32)
    mt_death_index = np.array([-1, 3, 3, 1, 3, -1], dtype=np.int32)
    mask_merged = mt_merger_ids != -1

    # LastMaxMass-style halo masses: constant from birth on, FROZEN after
    # death (this is what makes ``mass > 0`` == "ever resolved", the trap).
    # Mid-bin values (3ex, not 1ex): float32(1e12) rounds below the float64
    # 10^12 bin edge and would land one bin short.
    seed_masses = np.array([3e8, 3e6, 3e4, 3e10, 3e2, 3e12], dtype=np.float32)
    halo_masses_all = np.zeros((len(snaps), n_halos), dtype=np.float32)
    for h in range(n_halos):
        halo_masses_all[mt_birth_index[h]:, h] = seed_masses[h]

    precomp = precompute_per_snapshot_indices(
        min_snap=0, max_snap=4,
        mt_birth_index=mt_birth_index,
        mt_death_index=mt_death_index,
        mask_merged=mask_merged,
        mask_disrupted=~mask_merged,
        subset_indices=None,
        verbose=False,
    )

    bhmf_bins = 10.0 ** np.arange(0.0, 16.0)  # one bin per decade
    config = {
        "param_names": [],
        "erdf_params_template": {
            "log_eta_mean_0": -1.0, "log_eta_mean_evol": 0.9, "std_0": 0.5,
        },
        "erdf_model": "log_normal_evol_halo_mass",
        "rng_seed_bh": 12345,
        "n_z": len(snaps),
        "n_lbins": 10,
        "n_Lthr": 1,
        "n_mbins": len(bhmf_bins) - 1,
        "accretion_on": False,          # pure seed/merge/retire bookkeeping
        "merger_on": True,
        "time_step_for_accretion": 1.0,
        "rad_efficiency_0": 0.1,
        "rad_efficiency_model": "constant",
        "logfseed": 0.0,                # M_BH = M_halo exactly
        "sigmaseed": None,              # deterministic seeding
        "qlf_bins": 10.0 ** np.linspace(0.0, 20.0, 11),
        "qlf_normalization": 1.0,
        "bhmf_bins": bhmf_bins,
        "bhmf_normalization": 1.0,      # -> 10**log_bhmf = raw counts
        "cerdf_bins": np.linspace(-5.0, 2.0, len(bhmf_bins)),  # n_mbins bins (shared output axis)
        "cerdf_normalization": 1.0,
        "qhmf_bins": bhmf_bins,
        "qhmf_normalization": 1.0,
        "log_L_thresholds": np.array([40.0]),
        "snapshots": snaps,
        "snap_to_zindex": {int(s): int(s) for s in snaps},
        "redshifts": np.zeros(len(snaps)),
        "delta_times_snapshots": np.full(len(snaps), 0.1),
        "mass_units": 1.0,
        "halo_masses_all": halo_masses_all,
        "halo_specific_cold_accretion_rates_all": np.zeros_like(halo_masses_all),
        "transfer_function": None,
        "mt_track_ids": mt_track_ids,
        "mt_merger_ids": mt_merger_ids,
        "mt_death_index": mt_death_index,
        "precomp_spawning": precomp["spawning"],
        "precomp_evolving": precomp["evolving"],
        "precomp_merging": precomp["merging"],
        "precomp_lost": precomp["lost"],
        "use_subsample": False,
    }
    return config


def _counts(log_bhmf_row):
    """log10(count) row (with -inf empties) -> integer counts per bin."""
    out = np.zeros_like(log_bhmf_row, dtype=np.int64)
    finite = np.isfinite(log_bhmf_row)
    out[finite] = np.rint(10.0 ** log_bhmf_row[finite].astype(np.float64))
    return out


def test_dead_bhs_are_retired_and_dead_targets_aborted():
    from baqaro.emulation.run_single_model import run_one_model

    result = run_one_model(np.array([]), _build_config())
    counts = np.array([_counts(row) for row in result["log_bhmfs"]])

    def bins_with(row, expected_logm):
        """Assert occupancy: exactly one BH in each 1-dex bin [10^m, 10^(m+1))."""
        expected = np.zeros(counts.shape[1], dtype=np.int64)
        for m in expected_logm:
            expected[m] += 1
        np.testing.assert_array_equal(row, expected)

    # snap 0: all 5 early halos seeded (H5 not yet born)
    bins_with(counts[0], [8, 6, 4, 10, 2])
    # snap 1: H3 disrupted -> retired
    bins_with(counts[1], [8, 6, 4, 2])
    # snap 2: H5 seeded
    bins_with(counts[2], [8, 6, 4, 2, 12])
    # snap 3: H1 merged into H0 (1e8+1e6 stays in the log10=8 bin, source
    # zeroed), H2 disrupted, H4's target H3 is long-dead -> merger ABORTED,
    # mass lost, nothing parked in H3's slot. Only H0 and H5 remain
    # (stale H1, H2, H4 entries or H4's mass in H3 would add counts here).
    bins_with(counts[3], [8, 12])
    # snap 4: unchanged
    bins_with(counts[4], [8, 12])


if __name__ == "__main__":
    test_dead_bhs_are_retired_and_dead_targets_aborted()
    print("OK  test_dead_bhs_are_retired_and_dead_targets_aborted")
