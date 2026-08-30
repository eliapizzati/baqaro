"""Unit tests for the binary-decomposition logic in ``merger_catalog.py``.

Each test traces a small hand-crafted merger pattern at a single snapshot and
verifies that:

  * the recorded binary events sum to the correct total per dest halo,
  * the chain order is right (upstream merged before its downstream becomes a
    source itself),
  * the M2 of each binary equals the running mass of the dest at that moment.

Run::

    pytest baqaro/tests/test_merger_catalog.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from baqaro.core_functions.merger_catalog import (
    MergerCatalogBuilder,
    _decompose_to_binaries,
)


def test_single_pairwise_merger():
    """A->B, no chain. Single binary event with M1=M_A, M2=M_B."""
    out = _decompose_to_binaries(
        src_indices=np.array([10]),
        dest_indices=np.array([20]),
        src_masses=np.array([1.0e8]),
        dest_masses_initial=np.array([3.0e8]),
    )
    assert out["src_idx"].tolist() == [10]
    assert out["dest_idx_at_event"].tolist() == [20]
    assert out["M1"][0] == pytest.approx(1.0e8)
    assert out["M2"][0] == pytest.approx(3.0e8)


def test_two_sources_into_same_dest_mass_descending():
    """A->D, B->D with M_A < M_B. Heavier source (B) recorded first.

    Expected sequence:
      event 1: (B, D)  -> M1=M_B,  M2=M_D_initial
      event 2: (A, D)  -> M1=M_A,  M2=M_D_initial + M_B
    """
    out = _decompose_to_binaries(
        src_indices=np.array([1, 2]),         # halo idx 1 = A, halo idx 2 = B
        dest_indices=np.array([10, 10]),
        src_masses=np.array([1.0e8, 5.0e8]),  # A=1e8, B=5e8
        dest_masses_initial=np.array([2.0e8, 2.0e8]),
    )
    # Sort by depth-desc then mass-desc -> B first (heavier), then A.
    assert out["src_idx"].tolist() == [2, 1]
    assert out["M1"].tolist() == [5.0e8, 1.0e8]
    assert out["M2"][0] == pytest.approx(2.0e8)        # initial dest mass
    assert out["M2"][1] == pytest.approx(2.0e8 + 5.0e8)  # after B merged in
    # final dest mass = initial + M_A + M_B
    final_M_D = out["M2"][1] + out["M1"][1]
    assert final_M_D == pytest.approx(2.0e8 + 5.0e8 + 1.0e8)


def test_chain_merger_AB_BC():
    """A->B, B->C in one snapshot. Two binary events in chain order, with
    running mass tracking for sources too:
      event 1: (A, B)   -> M1=M_A, M2=M_B (B about to leave as a source)
      event 2: (B, C)   -> M1 = M_B + M_A (running, after A merged in),
                           M2 = M_C
    The running-mass convention means each binary's M1 reflects what the
    source actually carries at the moment of merger -- including any
    upstream mergers earlier this snapshot. Without this, the (B, C)
    event would record M1 = M_B at i-1 only, missing M_A's contribution.
    """
    out = _decompose_to_binaries(
        src_indices=np.array([10, 20]),       # idx 10 = A, idx 20 = B
        dest_indices=np.array([20, 30]),       # A->B, B->C
        src_masses=np.array([1.0e8, 2.0e8]),   # M_A=1e8, M_B=2e8 (at i-1)
        dest_masses_initial=np.array([2.0e8, 4.0e8]),  # B=2e8 initial, C=4e8 initial
    )
    # A has depth 1 (its dest B is a source), B has depth 0. Process A first.
    assert out["src_idx"].tolist() == [10, 20]
    assert out["dest_idx_at_event"].tolist() == [20, 30]
    # First binary: (A, B) with M_A=1e8 entering B at initial mass 2e8.
    assert out["M1"][0] == pytest.approx(1.0e8)
    assert out["M2"][0] == pytest.approx(2.0e8)
    # Second binary: (B, C) -- B's running mass at this moment = M_B + M_A = 3e8
    # (A merged into B in event 1, accumulating). M2 = C's still at initial 4e8.
    assert out["M1"][1] == pytest.approx(3.0e8)
    assert out["M2"][1] == pytest.approx(4.0e8)


def test_chain_merger_with_diamond_AC_BC_AthenC():
    """A->B, B->C, D->C (diamond + chain).
    Expected processing order:
      A has depth 1 (A->B, B is source), B and D have depth 0.
      Within depth-0 bucket: heavier first.
    Order: A, then (B and D), heavier of (B, D) first.
    """
    M_A, M_B, M_D = 1.0e8, 5.0e8, 3.0e8
    M_B_init, M_C_init = 5.0e8, 4.0e8  # B's initial dest mass, C's initial dest mass
    out = _decompose_to_binaries(
        src_indices=np.array([10, 20, 40]),       # A=10, B=20, D=40
        dest_indices=np.array([20, 30, 30]),
        src_masses=np.array([M_A, M_B, M_D]),
        dest_masses_initial=np.array([M_B_init, M_C_init, M_C_init]),
    )
    # A first (depth 1)
    assert out["src_idx"][0] == 10
    assert out["M1"][0] == pytest.approx(M_A)
    assert out["M2"][0] == pytest.approx(M_B_init)
    # Then B vs D (depth 0). B is heavier -> B first.
    assert out["src_idx"][1] == 20
    assert out["src_idx"][2] == 40
    # B's binary: (B, C) -- B's running mass = M_B_init + M_A (A merged into B
    # at event 0); C still at initial mass.
    assert out["M1"][1] == pytest.approx(M_B_init + M_A)
    assert out["M2"][1] == pytest.approx(M_C_init)
    # D's binary: (D, C) -- D's running mass = M_D (D was never a dest before);
    # C now has C_init + B's running mass (from previous event).
    assert out["M1"][2] == pytest.approx(M_D)
    assert out["M2"][2] == pytest.approx(M_C_init + M_B_init + M_A)


def test_no_mergers():
    out = _decompose_to_binaries(
        src_indices=np.empty(0, dtype=np.int64),
        dest_indices=np.empty(0, dtype=np.int64),
        src_masses=np.empty(0, dtype=np.float64),
        dest_masses_initial=np.empty(0, dtype=np.float64),
    )
    assert out["M1"].size == 0
    assert out["M2"].size == 0


def test_total_mass_conservation_per_dest():
    """For each terminal dest, sum of (M1) over all upstream events = total mass added.

    Build a wider tree: A->B, B->C, D->C, E->C, F->E, G->F.
    Terminal: C (and B exits, E exits, F exits into upstream to C).
    Total mass that should arrive at C = M_A + M_B + M_D + M_E + M_F + M_G.
    """
    # Structure:
    #   G -> F -> E -> C
    #   A -> B -> C
    #   D -> C
    # With masses M_A=1, M_B=2, M_D=3, M_E=4, M_F=5, M_G=6 (in 1e8 units).
    src   = np.array([10, 20, 40, 50, 60, 70])  # A, B, D, E, F, G
    dest  = np.array([20, 30, 30, 30, 50, 60])  # ->B, ->C, ->C, ->C, ->E, ->F
    masses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * 1e8
    # Initial dest masses (only matters for the actual dest nodes; pick non-zero
    # values for B, C, E, F):
    dest_init = np.array([2.0e8, 0.0, 0.0, 4.0e8, 5.0e8, 6.0e8])
    # NOTE: dest_init[k] is the initial dest mass for the dest at row k. For
    # rows referring to the same dest, we take the value from the first row
    # (this matches main_evolution.py's per-snapshot dest mass lookup).
    out = _decompose_to_binaries(
        src_indices=src,
        dest_indices=dest,
        src_masses=masses,
        dest_masses_initial=dest_init,
    )
    # Total events = 6 (A->B, B->C, D->C, E->C, F->E, G->F)
    assert out["src_idx"].size == 6
    # Mass conservation: under the running-mass convention, the final mass
    # accumulated at terminal C must equal the sum of all initial source
    # masses + initial dest masses for any halo that ends up flowing into C.
    # Source masses: M_A + M_B + M_D + M_E + M_F + M_G = 1+2+3+4+5+6 = 21.
    # Initial dest mass for C (the only terminal): 0.
    # Plus initial dest masses for intermediate halos B, E, F that end up at C
    # via chain: B_init=2, E_init=4, F_init=5. But those are also INITIAL
    # source masses for B, E, F themselves (since they're sources too); their
    # initial src mass equals their initial dest mass when they're both src
    # and dest. So we don't double-count -- final at C = Σ initial source
    # masses of every halo whose chain terminus is C = 1+2+3+4+5+6 = 21.
    # Verified by tracing: M1 of E->C = M_E + M_F + M_G = 15 (since F and G
    # merged into E earlier); M1 of B->C = M_B + M_A = 3; M1 of D->C = 3.
    # 15 + 3 + 3 = 21.
    mask_C = out["dest_idx_at_event"] == 30
    mass_into_C = out["M1"][mask_C].sum()
    assert mass_into_C == pytest.approx((1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0) * 1e8)


def test_builder_concatenates_across_snapshots():
    builder = MergerCatalogBuilder()
    track_ids = np.array([100, 101, 102, 103, 104, 105], dtype=np.int64)

    # Snapshot 1: idx 0 -> idx 1 (A->B)
    builder.record_snapshot_mergers(
        z_snap=2.0,
        src_indices=np.array([0]),
        dest_indices=np.array([1]),
        src_masses=np.array([1.0e8]),
        dest_masses_initial=np.array([2.0e8]),
        track_ids=track_ids,
    )
    # Snapshot 2: idx 2 -> idx 3, idx 4 -> idx 3 (two into one dest)
    builder.record_snapshot_mergers(
        z_snap=1.0,
        src_indices=np.array([2, 4]),
        dest_indices=np.array([3, 3]),
        src_masses=np.array([1.0e8, 5.0e8]),
        dest_masses_initial=np.array([3.0e8, 3.0e8]),
        track_ids=track_ids,
    )
    out = builder.finalize()
    assert out["z"].tolist() == [2.0, 1.0, 1.0]
    # First binary uses idx 0 -> track_id 100, dest idx 1 -> track_id 101
    assert out["progenitor1_id"][0] == 100
    assert out["progenitor2_id"][0] == 101
    assert out["descendant_id"][0] == 101
    # Second snapshot: heavier source (idx 4, M=5e8) first
    assert out["progenitor1_id"][1] == 104
    assert out["progenitor1_id"][2] == 102


def test_builder_empty_finalize():
    builder = MergerCatalogBuilder()
    out = builder.finalize()
    assert out["z"].size == 0
    assert out["M1_Msun"].size == 0
    assert out["progenitor1_id"].dtype == np.int64
