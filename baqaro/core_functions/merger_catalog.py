"""Merger catalog recorder: turns per-snapshot src/dest pairs into binary events.

Used by ``main_evolution.py`` (when ``record_merger_catalog=True``) to produce
a standalone HDF5 catalogue of black-hole binary mergers, for analyses that need
the merger history rather than the evolved population.

The non-trivial bit is decomposing two cases into a sequence of binary events:

    1. **Chain mergers** in one snapshot, e.g. A->B and B->C: the simulation
       chain-resolves these to A->C for mass-accumulation efficiency, but
       physically there are TWO binary events: (A, B) -> AB and (AB, C) -> ABC.

    2. **Multiple sources into the same destination** in one snapshot, e.g.
       A->D, B->D, C->D: physically three sequential binary events.

Algorithm:

  * For each source, compute its **chain depth**: number of source-edges between
    it and the first non-source destination. Sources whose immediate dest is
    NOT itself a source have depth 0.
  * Process sources in (depth descending, mass descending) order. Deeper chains
    are processed first so the upstream mass is already accumulated into the
    intermediate node before that node leaves as a source itself.
  * Track ``running_M[dest]`` as we go; each binary records (M_src, running_M[dest])
    then ``running_M[dest] += M_src``.

This module is independent of the rest of ``core_functions/`` and has no
numpy/Numba performance constraint -- per-snapshot event counts are
typically <~10^5, well within Python-loop territory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class MergerCatalogBuilder:
    """In-memory accumulator for merger events across a multi-snapshot run.

    Call ``record_snapshot_mergers(...)`` from inside ``main_evolution.py``'s
    merger block (with chain-resolution-OFF inputs, i.e. *immediate* dest_ids).
    Call ``finalize()`` at the end of the run to get concatenated arrays
    ready to write to HDF5.
    """

    z_per_event: list[np.ndarray] = field(default_factory=list)
    M1_per_event: list[np.ndarray] = field(default_factory=list)
    M2_per_event: list[np.ndarray] = field(default_factory=list)
    prog1_id_per_event: list[np.ndarray] = field(default_factory=list)
    prog2_id_per_event: list[np.ndarray] = field(default_factory=list)
    descendant_id_per_event: list[np.ndarray] = field(default_factory=list)
    halo_id_per_event: list[np.ndarray] = field(default_factory=list)

    def record_snapshot_mergers(
        self,
        z_snap: float,
        src_indices: np.ndarray,         # halo indices (rows of black_hole_masses_all)
        dest_indices: np.ndarray,        # immediate (NOT chain-resolved) dest halo indices
        src_masses: np.ndarray,          # M_BH of src at snapshot i-1
        dest_masses_initial: np.ndarray, # M_BH of dest at snapshot i, post-accretion, pre-merger
        track_ids: np.ndarray,           # mt_track_ids array (long-lived halo IDs)
    ) -> int:
        """Decompose this snapshot's mergers into binary events; append to buffers.

        Returns the number of binary events recorded. Mutates internal lists.

        ``src_indices`` and ``dest_indices`` are halo-array indices (rows of
        ``black_hole_masses_all``). ``track_ids[idx]`` gives the persistent
        track-ID for halo at index ``idx``. The catalog stores track IDs (which
        survive across snapshots and are stable identifiers), not array indices.

        Inputs may have duplicates (multiple sources into the same dest). The
        function handles all decomposition cases correctly.
        """
        n = src_indices.size
        if n == 0:
            return 0
        if dest_indices.size != n or src_masses.size != n or dest_masses_initial.size != n:
            raise ValueError("src/dest/mass arrays must all have the same length.")

        events = _decompose_to_binaries(
            src_indices=np.asarray(src_indices, dtype=np.int64),
            dest_indices=np.asarray(dest_indices, dtype=np.int64),
            src_masses=np.asarray(src_masses, dtype=np.float64),
            dest_masses_initial=np.asarray(dest_masses_initial, dtype=np.float64),
        )

        n_ev = events["src_idx"].size
        if n_ev == 0:
            return 0

        z_arr = np.full(n_ev, float(z_snap), dtype=np.float64)
        prog1_id = track_ids[events["src_idx"]].astype(np.int64)
        prog2_id = track_ids[events["dest_idx_at_event"]].astype(np.int64)
        # descendant_id == dest's track_id, since the dest BH is the post-merger remnant
        desc_id = prog2_id.copy()
        halo_id = events["dest_idx_at_event"].astype(np.int64)

        self.z_per_event.append(z_arr)
        self.M1_per_event.append(events["M1"])
        self.M2_per_event.append(events["M2"])
        self.prog1_id_per_event.append(prog1_id)
        self.prog2_id_per_event.append(prog2_id)
        self.descendant_id_per_event.append(desc_id)
        self.halo_id_per_event.append(halo_id)

        return n_ev

    def finalize(self) -> dict[str, np.ndarray]:
        """Concatenate per-snapshot buffers; return a dict ready to dump to HDF5."""
        if not self.z_per_event:
            return {
                "z": np.empty(0, dtype=np.float64),
                "M1_Msun": np.empty(0, dtype=np.float64),
                "M2_Msun": np.empty(0, dtype=np.float64),
                "progenitor1_id": np.empty(0, dtype=np.int64),
                "progenitor2_id": np.empty(0, dtype=np.int64),
                "descendant_id": np.empty(0, dtype=np.int64),
                "halo_id": np.empty(0, dtype=np.int64),
            }
        return {
            "z":               np.concatenate(self.z_per_event),
            "M1_Msun":         np.concatenate(self.M1_per_event),
            "M2_Msun":         np.concatenate(self.M2_per_event),
            "progenitor1_id":  np.concatenate(self.prog1_id_per_event),
            "progenitor2_id":  np.concatenate(self.prog2_id_per_event),
            "descendant_id":   np.concatenate(self.descendant_id_per_event),
            "halo_id":         np.concatenate(self.halo_id_per_event),
        }

    def __len__(self) -> int:
        return sum(arr.size for arr in self.z_per_event)


# -- core algorithm --------------------------------------------------------


def _decompose_to_binaries(
    src_indices: np.ndarray,
    dest_indices: np.ndarray,
    src_masses: np.ndarray,
    dest_masses_initial: np.ndarray,
) -> dict[str, np.ndarray]:
    """Decompose a snapshot's source->dest pairs into a sequence of binary events.

    Handles:
      - chain mergers (A->B, B->C): two events (A,B), (B,C) where B's mass at
        the second event includes M_A.
      - multiple sources into the same dest (A->D, B->D): two events with
        running M_dest.

    Order: by chain depth descending, then by source mass descending. This
    ensures upstream nodes are processed before their downstream destinations
    leave as sources, and the heaviest binaries within a same-depth bucket are
    recorded first.

    Returns a dict with keys:
      - ``src_idx``           : halo-array index of M1 (the disappearing BH)
      - ``dest_idx_at_event`` : halo-array index of M2 (the receiving BH)
      - ``M1``                : mass of the source at the moment of merger
      - ``M2``                : running mass of the dest at the moment of merger
    """
    n = src_indices.size
    if n == 0:
        return {
            "src_idx": np.empty(0, dtype=np.int64),
            "dest_idx_at_event": np.empty(0, dtype=np.int64),
            "M1": np.empty(0, dtype=np.float64),
            "M2": np.empty(0, dtype=np.float64),
        }

    # Map: halo_idx -> list of source-event-indices having this halo as src
    # (so we can determine if a halo is a "source" in this snapshot).
    src_set = set(int(s) for s in src_indices)

    # Compute chain depth per source via memoised lookup.
    src_to_dest = {int(src_indices[k]): int(dest_indices[k]) for k in range(n)}
    depth_cache: dict[int, int] = {}

    def depth_of(node: int) -> int:
        if node in depth_cache:
            return depth_cache[node]
        if node not in src_set:
            depth_cache[node] = 0
            return 0
        # Use iterative chain-walk with cycle detection.
        path: list[int] = []
        cur = node
        while cur in src_set and cur not in depth_cache:
            path.append(cur)
            nxt = src_to_dest[cur]
            if nxt == cur:
                break  # self-loop, treat as terminal
            cur = nxt
        # cur is either not-a-source (terminal) or in the cache; assign depths.
        base = depth_cache.get(cur, 0)
        for k, p in enumerate(reversed(path)):
            depth_cache[p] = base + k + 1
        return depth_cache[node]

    depths = np.array([depth_of(int(s)) for s in src_indices], dtype=np.int64)

    # Sort by (depth desc, mass desc).
    order = np.lexsort((-src_masses, -depths))

    # Walk in this order, building up running mass per BH halo index. We
    # track running_M for BOTH srcs and dests so that chain mergers
    # (A->B, B->C in one snapshot) record the right M1: at the (B, C)
    # event, B's mass already includes A's contribution from the
    # preceding (A, B) event. Without this, M1 at (B, C) would just be
    # B's mass at i-1, missing the chain-accumulated contribution.
    running_M: dict[int, float] = {}
    for k in range(n):
        s_idx = int(src_indices[k])
        d_idx = int(dest_indices[k])
        if s_idx not in running_M:
            running_M[s_idx] = float(src_masses[k])
        if d_idx not in running_M:
            running_M[d_idx] = float(dest_masses_initial[k])

    out_src_idx = np.empty(n, dtype=np.int64)
    out_dest_idx = np.empty(n, dtype=np.int64)
    out_M1 = np.empty(n, dtype=np.float64)
    out_M2 = np.empty(n, dtype=np.float64)

    for write_pos, k in enumerate(order):
        s = int(src_indices[k])
        d = int(dest_indices[k])
        M1 = running_M[s]
        M2 = running_M[d]

        out_src_idx[write_pos] = s
        out_dest_idx[write_pos] = d
        out_M1[write_pos] = M1
        out_M2[write_pos] = M2

        running_M[d] += M1
        # src is gone; running_M[s] is no longer relevant.

    return {
        "src_idx": out_src_idx,
        "dest_idx_at_event": out_dest_idx,
        "M1": out_M1,
        "M2": out_M2,
    }
