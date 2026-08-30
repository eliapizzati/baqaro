"""
Run resolution and data loading for every figure.
================================================

This module is imported by every figure-producing script.
It provides:

1. **Module-level constants** — file paths, simulation parameters, and naming
   conventions that identify which evolution run to visualise.
2. **DataLoader / FullDataHistoryLoader** — lightweight lazy loaders that wrap
   an open ``h5py.File`` handle (and, for halo masses, a memory-mapped ``.npy``
   file).  Data is read from disk only when a specific snapshot is requested and
   is then cached for the lifetime of the loader.
3. **load_simulation_metadata / load_simulation_metadata_full_history** —
   convenience functions that read all small arrays and parameters from the
   HDF5 file in one call and return a dict ready for use by the plotting
   scripts.

Typical usage (inside a plotting script)::

    from baqaro.plotting_common.load_data_to_plot import (
        path_file, path_plots, boxsize, load_simulation_metadata,
    )
    import h5py

    with h5py.File(path_file, "r") as f:
        data = load_simulation_metadata(f)
        Lbols = data['loader'].get_Lbol(snapshot_index)      # solar luminosities
        M_BH  = data['loader'].get_BH_mass(snapshot_index)    # solar masses
        M_h   = data['loader'].get_Halo_mass(snapshot_index)  # solar masses

Unit conventions
----------------
* **BH masses and luminosities** stored in the evolution HDF5 are already in
  solar units (``mass_units = 1e7`` was applied during ``main_evolution.py``).
* **Halo masses** in the ``.npy`` files are in *code units*
  (``halo_mass_units_hbt`` was applied during
  ``halo_mass_histories_saver.py``).  ``get_Halo_mass`` multiplies by
  ``mass_units`` to return solar masses, matching the BH mass convention.
"""

import os

import numpy as np

from qhtools.utils.create_binned_functions import (
    create_mass_function,
    create_luminosity_function,
)
from baqaro.core_functions.erdf_core_functions import Erdf
from baqaro.utils.my_dir import get_output_path, get_plots_path
from baqaro.utils.my_units import mass_units



# ==============================================================================
# RUN IDENTIFICATION
# ==============================================================================
# These settings select *which* evolution output to load.  Every plotting
# script that imports this module will use the same run.

# notes_file resolution mirrors main_evolution (read-side): BAQARO_NOTES_FILE
# env (presence wins; "" => no token) > the matched bestfit registry entry's
# `notes` (older entries pin a fixed notes token, newer ones None) > None. So a
# plotting run lands on the exact filename the forward run produced — set
# BAQARO_BESTFIT_NAME and the notes come along automatically.
from baqaro.core_functions.bestfit_registry import resolve_notes_file
notes_file = resolve_notes_file()
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")  # machine choice, env-var-driven

# Halo-data variant tokens. Must match the saver run that produced the halo
# files and the main_evolution.py / main_training.py run that produced
# the BH evolution HDF5 we're about to load. fold_subhalo_mass /
# merger_delay_mode come from sim_config (env-overridable, centralized
# defaults; the fiducial is foldmass + instant_new) and are imported below
# with the other sim_config names. To read the legacy (no-fold, instant_old)
# products instead:
#   BAQARO_FOLD_SUBHALO_MASS=0 BAQARO_MERGER_DELAY_MODE=instant_old python ...

# `BAQARO_BESTFIT_NAME` — when main_evolution.py ran with a bestfit tag
# (e.g. "qcc_ck22final_v1"), the output filename has `_bestfit_<name>`
# between the notes and the subset tag. Set this env var to load that
# specific bestfit's evolution output.
# Defaults to the ADOPTED fiducial, so a BARE run reproduces the fiducial like
# every other pipeline stage. Previously this defaulted to None, which built a
# path with no `_bestfit_` token -- a run that was never produced -- so every
# figure script died with an opaque h5py FileNotFoundError unless the caller
# happened to know to set BAQARO_BESTFIT_NAME.
# Set BAQARO_BESTFIT_NAME="" for the (rare) genuinely untagged run.
from baqaro.utils.sim_config import FIDUCIAL_BESTFIT_NAME as _FID_BF
_bf_env = os.environ.get("BAQARO_BESTFIT_NAME")
bestfit_name = (_bf_env if _bf_env is not None else _FID_BF) or None

# `BAQARO_SUBSET_TAG` — when the evolution was subsampled, the output filename
# ends with `_sub_<tag>`. Defaults to the production subsampled run's tag
# (auto-derived from sim_config geometry + subsample env defaults), so callers
# needn't set it. Env presence wins: BAQARO_SUBSET_TAG="" forces full-sim (no
# `_sub` token); BAQARO_SUBSET_TAG=<tag> picks a specific subset.
# (resolve_subset_tag() defaults root_snap to sim_config.max_snap — used here
# because the local `max_snap` is defined further below; the two are identical.)
if "BAQARO_SUBSET_TAG" in os.environ:
    subset_tag = os.environ["BAQARO_SUBSET_TAG"]
else:
    from baqaro.core_functions.tree_subsample import resolve_subset_tag
    subset_tag = resolve_subset_tag()

# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================
from baqaro.utils.sim_config import (
    boxsize, N_particles_per_side, simulation_name, tdyn_fraction_default,
    log_M_halo_qhmf_lo as _qhmf_lo_sim, log_M_halo_qhmf_hi as _qhmf_hi_sim,
    max_snap, fold_subhalo_mass, merger_delay_mode, growth_feff_suffix,
)

# QHMF host-halo mass range — sim-dependent (L2800N5040: 11.5-14.5;
# L2800N10080: 10.5-14.5, 1 dex lower floor from the 8x finer particle
# mass). Pulled from sim_config so plotting scripts stop hardcoding the
# L2800N5040 window; override per-run with BAQARO_LOG_M_HALO_QHMF_LO/HI.
log_M_halo_qhmf_lo = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_LO", _qhmf_lo_sim))
log_M_halo_qhmf_hi = float(os.environ.get("BAQARO_LOG_M_HALO_QHMF_HI", _qhmf_hi_sim))

erdf_model = "log_normal_evol_halo_mass"  # ERDF model used in the evolution
# max_snap now comes from sim_config (env-overridable via BAQARO_MAX_SNAP,
# defaults to the active sim's z=0 snapshot) — imported above.

show_lightcurves = True

# ==============================================================================
# DERIVED FILE PATHS
# ==============================================================================
# --- Evolution HDF5 (BH masses, luminosities, parameters) ---
# NB: no `_6d` ("6 free parameters") token: it was constant in every name,
# never varied, and neither the forward-run nor the emulator/chain names carry it.
name_file = "{}_erdf_{}_maxsnap_{}".format(
    simulation_name,
    erdf_model,
    max_snap,
)
if fold_subhalo_mass:
    name_file += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file += "_{}".format(merger_delay_mode)
if notes_file is not None:
    name_file += "_{}".format(notes_file)
# Match main_evolution.py's name_file order: _bestfit_<name> AFTER notes_file
# and BEFORE _sub_<tag>.
if bestfit_name:
    name_file += "_bestfit_{}".format(bestfit_name)
if subset_tag:
    name_file += "_sub_{}".format(subset_tag)

# Read-side mirror of main_evolution's `_g<value>` + `_feffcorr` tokens, in the
# write order (`..._g4.6_feffcorr.hdf5`):
#   BAQARO_GROWTH_SUM_MAX != 50.0        -> `_g<value>`  (growth-cap mode)
#   BAQARO_MADAU_FEFF_CORRECTION truthy  -> `_feffcorr`  (the madau f_eff correction)
# Shared with main_evolution / main_training / main_emulation via the SINGLE
# sim_config helper — this block used to re-derive the token logic inline, which
# is exactly how the two can drift apart.
name_file += growth_feff_suffix()

name_fig = name_file  # Default figure-name tag, can be overridden by scripts

path_out = get_output_path(source=source_dir)
path_plots = get_plots_path(source=source_dir)

path_file = os.path.join(path_out, "evolution", f"bh_evolution_{name_file}.hdf5")

# Full-history runs are SELECTION-based, never subsampled: `main_evolution_full_history`
# picks its targets by a criterion (`massthr11.5`, `firstborn100`, ...) carried in
# notes_file, and writes NO `_sub_<tag>` token. Not one of the full-history files on
# disk has ever had one. Reusing `name_file` here therefore appended a subset tag to a
# path that can never exist, so every full-history consumer failed with a bare
# FileNotFoundError as soon as the subset tag became the zero-env default.
# Rebuild without it; the selection comes from BAQARO_NOTES_FILE.
_name_file_full_history = name_file.replace("_sub_{}".format(subset_tag), "") if subset_tag else name_file
path_file_full_history = os.path.join(
    path_out, "evolution", f"bh_evolution_full_history_{_name_file_full_history}.hdf5")

# --- Halo history .npy (halo masses, separate from evolution HDF5) ---
# These must match the parameters used in halo_mass_histories_saver.py.
nbound_threshold = 40
halo_filtering_mode = "global"
# tdyn_fraction_default comes from sim_config (0.20 at 5040, 0.25 at 10080).
name_file_halos = "{}_maxsnap{}_nboundthresh{}_halofilter_{}_tdynfraction_{}".format(
    simulation_name, max_snap, nbound_threshold, halo_filtering_mode, tdyn_fraction_default
)
if fold_subhalo_mass:
    name_file_halos += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file_halos += "_{}".format(merger_delay_mode)
path_file_halo_masses = os.path.join(path_out, "halo_histories", f"halo_masses_{name_file_halos}.npy")

# Pre-extracted subset slice of halo masses (written by main_evolution.py via
# extract_subset_slice_cached). For subsampled runs this is the PREFERRED
# source for get_Halo_mass: shape (n_snaps, n_subset), already in
# subset_indices order, so it avoids the scattered fancy-index read of
# subset_indices against the full-N (n_snaps, ~1.5B) mmap — and lets plots run
# when only the subset cache is on disk (full halo_masses_*.npy absent / still
# transferring). Naming mirrors main_evolution.py's eager cache.
if subset_tag:
    path_subset_halo_masses = os.path.join(
        path_out, "halo_subsets",
        f"subset_{name_file_halos}_{subset_tag}__halo_masses.npy",
    )
else:
    path_subset_halo_masses = None




# ------------------------------------------------------------------------------
# HELPER: Lazy Data Loader
# ------------------------------------------------------------------------------
class DataLoader:
    """Lazy, per-snapshot loader for large 2-D arrays.

    Wraps an open ``h5py.File`` for BH masses and luminosities (stored in
    the evolution HDF5) and, optionally, a memory-mapped ``.npy`` file for
    halo masses (produced by ``halo_mass_histories_saver.py``).

    Subsample-aware
    ----------------
    If the HDF5 contains a ``subset/`` group (written by ``main_evolution.py``
    when ``use_subsample=True``), the BH/Lbol arrays are already compact
    (shape ``(n_snaps, n_subset)``) and per-halo inverse-probability weights
    are exposed via :attr:`weights`. Halo masses are transparently re-indexed via
    ``subset/subset_indices`` so all per-halo arrays returned by the loader
    line up element-wise. The :attr:`is_subsample` flag tells callers whether
    they need to pass weights to histograms / mass functions.

    Parameters
    ----------
    file_handle : h5py.File
        Open HDF5 file containing ``black_hole_masses_all`` and
        ``Lbols_all`` datasets.  Must remain open for the lifetime of the
        loader.
    """

    def __init__(self, file_handle):
        self._file_handle = file_handle
        self._cache_bh_mass = {}
        self._cache_lbol = {}
        self._cache_halo_mass = {}
        self._halo_masses_mmap = None
        # True when ``_halo_masses_mmap`` is the pre-extracted subset slice
        # (shape (n_snaps, n_subset), already subset-ordered) rather than the
        # full-N array — get_Halo_mass then skips the subset_indices re-index.
        self._halo_masses_is_subset = False
        # Map a row position in this file's per-snapshot arrays -> the SNAPSHOT
        # NUMBER (== row in the full-N halo_masses .npy). For standard / subsample
        # runs the file stores every snapshot so ``snapshots == arange`` and this
        # is the identity. For a multinode run that stores only the anchor
        # snapshots (e.g. [35, 39, ..., 144]) it is NOT the identity, so
        # get_Halo_mass must translate position -> snapshot before indexing the
        # full-N .npy (whose rows are snapshot-numbered). See get_Halo_mass.
        self._snapshots = (np.asarray(file_handle["snapshots"])
                           if "snapshots" in file_handle else None)

        self.is_subsample = "subset" in file_handle
        if self.is_subsample:
            self._subset_indices = np.asarray(file_handle["subset/subset_indices"])
            self.weights = np.asarray(file_handle["subset/weights"])
            self.subset_tag = (
                file_handle["subset"].attrs.get("tag", "")
                if hasattr(file_handle["subset"], "attrs")
                else ""
            )
        else:
            self._subset_indices = None
            self.weights = None
            self.subset_tag = ""

    # ------------------------------------------------------------------
    # Data from the evolution HDF5
    # ------------------------------------------------------------------

    def get_BH_mass(self, snap_idx):
        """Return BH masses [M_sun] for snapshot *snap_idx*.

        Shape is ``(n_halos,)`` for full-sim runs or ``(n_subset,)`` for
        subsampled runs. Element *i* aligns with :attr:`weights` index *i*
        in subsample mode and with the global track ID *i* otherwise.
        """
        if snap_idx not in self._cache_bh_mass:
            self._cache_bh_mass[snap_idx] = self._file_handle["black_hole_masses_all"][snap_idx, :]
        return self._cache_bh_mass[snap_idx]

    def get_Lbol(self, snap_idx):
        """Return bolometric luminosities [L_sun] for snapshot *snap_idx*.

        Same indexing convention as :meth:`get_BH_mass`.
        """
        if snap_idx not in self._cache_lbol:
            self._cache_lbol[snap_idx] = self._file_handle["Lbols_all"][snap_idx, :]
        return self._cache_lbol[snap_idx]

    # ------------------------------------------------------------------
    # Halo masses from the external .npy file
    # ------------------------------------------------------------------

    def load_halo_masses(self, path, subset_cache_path=None):
        """Open the halo-mass ``.npy`` as a read-only memory map.

        Must be called before :meth:`get_Halo_mass`.  Files store masses in
        code units (``halo_mass_units_hbt`` applied in
        ``halo_mass_histories_saver.py``); :meth:`get_Halo_mass` converts to
        solar.

        Source preference:

        1. ``subset_cache_path`` (subsample runs) — the pre-extracted subset
           slice, ``(n_snaps, n_subset)`` already in ``subset_indices`` order.
           Preferred whenever present: ~n_full/n_subset× smaller and a plain
           sequential row read, vs. a scattered fancy-index of
           ``subset_indices`` against the full-N mmap. Also lets plots run
           when only the subset cache is on disk.
        2. ``path`` — the full-N ``halo_masses_*.npy``; re-indexed by
           ``subset_indices`` in subsample mode, used as-is in full-sim mode.
        3. Neither present — leave the mmap ``None``; :meth:`get_Halo_mass`
           raises only if actually called (QLF / BHMF / CERDF never call it,
           so those plots still work without any halo-mass file).
        """
        if (self._subset_indices is not None
                and subset_cache_path is not None
                and os.path.exists(subset_cache_path)):
            _cand = np.load(subset_cache_path, mmap_mode='r')
            # SHAPE-VALIDATE before trusting it.
            # The cache is indexed BY POSITION below, so it is only usable if it
            # has exactly one row per snapshot stored in THIS file and one column
            # per subset entry, in subset_indices order. Caches written under
            # this same tag by other run layouts (an `n_chunks=1` engine run's
            # eager cache, shaped (n_all_snaps, n_partition) in PARTITION order;
            # a combined multinode file with ~13 anchor rows and its own
            # subset_indices order) have other shapes and are rejected here.
            _n_rows_expected = (len(self._snapshots)
                                if self._snapshots is not None
                                else _cand.shape[0])
            _expected = (_n_rows_expected, len(self._subset_indices))
            if tuple(_cand.shape) != _expected:
                print(f"  [plotting] ⚠ IGNORING subset slice cache "
                      f"{os.path.basename(subset_cache_path)}: shape "
                      f"{tuple(_cand.shape)} != expected {_expected} "
                      f"(n_snapshots_in_file, n_subset). This usually means the "
                      f"cache belongs to a DIFFERENT run (e.g. a 1-chunk eager "
                      f"cache alongside a combined multinode file). Falling back "
                      f"to the full-N halo_masses file.")
                del _cand
            else:
                self._halo_masses_mmap = _cand
                self._halo_masses_is_subset = True
                print(f"  [plotting] halo masses from subset slice cache "
                      f"{os.path.basename(subset_cache_path)} "
                      f"(shape {self._halo_masses_mmap.shape})")
                return
        if os.path.exists(path):
            self._halo_masses_mmap = np.load(path, mmap_mode='r')
            self._halo_masses_is_subset = False
            return
        print(f"  [plotting] halo masses absent at {path} "
              f"(no subset slice cache either); get_Halo_mass() will raise if "
              f"called. QLF / BHMF / CERDF don't need it, so those still run.")
        self._halo_masses_mmap = None
        self._halo_masses_is_subset = False

    def get_Halo_mass(self, snap_idx):
        """Return halo masses [M_sun] for row position *snap_idx*.

        ``snap_idx`` is a POSITION into this file's per-snapshot arrays (what
        plotting gets from ``argmin(|redshifts - z|)``), NOT necessarily a
        snapshot number. The on-disk values are in code units; multiplying by
        ``mass_units`` (1e7) converts to solar masses. In subsample mode the
        full-N row is re-indexed with ``subset_indices`` so the returned array
        lines up element-wise with :meth:`get_BH_mass`.
        """
        if snap_idx not in self._cache_halo_mass:
            if self._halo_masses_mmap is None:
                raise RuntimeError(
                    "Halo masses not loaded (file absent). Provide the full "
                    "halo_masses_*.npy or the pre-extracted subset slice cache, "
                    "then call loader.load_halo_masses(path[, subset_cache_path])."
                )
            if self._halo_masses_is_subset:
                # The subset slice cache has one row per saved snapshot, in the
                # same order as the file's per-snapshot arrays — index by position.
                row = np.asarray(self._halo_masses_mmap[snap_idx])
            else:
                # The full-N .npy is indexed by SNAPSHOT NUMBER. Translate the
                # row position -> snapshot via the file's ``snapshots`` array
                # (identity when the file stores every snapshot; non-trivial for
                # a multinode run that stores only the anchor snapshots). Without
                # this, a 13-anchor file would read another snapshot's halos.
                file_row = (int(self._snapshots[snap_idx])
                            if self._snapshots is not None else snap_idx)
                row = np.asarray(self._halo_masses_mmap[file_row])
                if self._subset_indices is not None:
                    # Materialise just the subset entries so the result lines up
                    # element-wise with get_BH_mass / weights.
                    row = row[self._subset_indices]
            self._cache_halo_mass[snap_idx] = row * mass_units
        return self._cache_halo_mass[snap_idx]





class FullDataHistoryLoader:
    """Lazy loader for sub-timestep ("full history") arrays.

    ``main_evolution_full_history.py`` saves per-sub-step BH masses,
    luminosities, Eddington ratios, and a shared time axis into the
    ``full_history/`` HDF5 group.  On-disk layout is
    ``(total_steps, n_halos)`` — consistent with the global
    ``(n_snapshots, n_halos)`` convention.

    These arrays can be very large, so they are only read into RAM on
    first access via the corresponding property.

    Parameters
    ----------
    file_handle : h5py.File
        Open HDF5 file containing the ``full_history/`` group.
    """

    def __init__(self, file_handle):
        self._file_handle = file_handle
        self._cached_bh_masses = None
        self._cached_lbols = None
        self._cached_etas = None
        self._cached_times = None

    @property
    def black_hole_masses_full_history(self):
        """BH mass at every sub-step, shape ``(total_steps, n_halos)`` [M_sun]."""
        if self._cached_bh_masses is None:
            self._cached_bh_masses = np.asarray(self._file_handle["full_history/black_hole_masses_full_history"][:,:])
        return self._cached_bh_masses

    @property
    def Lbols_full_history(self):
        """Bolometric luminosity at every sub-step, shape ``(total_steps, n_halos)`` [L_sun]."""
        if self._cached_lbols is None:
            self._cached_lbols = np.asarray(self._file_handle["full_history/Lbols_full_history"][:,:])
        return self._cached_lbols

    @property
    def etas_full_history(self):
        """Eddington ratio at every sub-step, shape ``(total_steps, n_halos)``."""
        if self._cached_etas is None:
            self._cached_etas = np.asarray(self._file_handle["full_history/etas_full_history"][:,:])
        return self._cached_etas

    @property
    def times_full_history(self):
        """Time axis shared by all halos, shape ``(total_steps,)`` [Gyr]."""
        if self._cached_times is None:
            self._cached_times = np.asarray(self._file_handle["full_history/times_full_history"][:])
        return self._cached_times
            


# ------------------------------------------------------------------------------
# SHARED DATA LOADING FUNCTION
# ------------------------------------------------------------------------------
def load_simulation_metadata(file_handle):
    """Read simulation metadata and create a lazy loader from an open HDF5 file.

    This is the main entry point used by plotting scripts.  It reads all
    small arrays and scalar parameters eagerly (cheap) and creates a
    :class:`DataLoader` for the large per-snapshot arrays (read on demand).
    The halo-mass ``.npy`` file is also memory-mapped via the loader.

    Must be called inside a ``with h5py.File(...)`` context — the returned
    ``data['loader']`` holds a reference to *file_handle* so the file must
    stay open while the loader is in use.

    Parameters
    ----------
    file_handle : h5py.File
        Open HDF5 evolution file (``bh_evolution_*.hdf5``).
        All 2-D datasets must be in ``(n_snapshots, n_halos)`` layout.

    Returns
    -------
    data : dict
        Keys include:

        * ``snapshots``, ``redshifts``, ``ages_of_the_universe``,
          ``delta_times_snapshots`` — 1-D arrays, one element per snapshot.
        * ``erdf_model_str``, ``erdf_params``, ``erdf_duty_cycle``,
          ``time_step_for_accretion``, ``logfseed``, ``sigmaseed`` —
          scalar parameters that were used in the evolution run.
        * ``loader`` — :class:`DataLoader` instance (call
          ``.get_BH_mass(i)``, ``.get_Lbol(i)``, ``.get_Halo_mass(i)``).
        * ``erdf`` — initialised :class:`Erdf` object ready for evaluation.
    """
    data = {}

    # 1. Small 1-D arrays (read eagerly — negligible cost)
    data['snapshots'] = np.asarray(file_handle["snapshots"])
    data['redshifts'] = np.asarray(file_handle["redshifts"])
    data['ages_of_the_universe'] = np.asarray(file_handle["ages_of_the_universe"])
    data['delta_times_snapshots'] = np.asarray(file_handle["delta_times_snapshots"])

    # 2. Scalar / dict parameters
    erdf_model = file_handle["parameters/erdf_model"][()]
    data['erdf_model_str'] = erdf_model.decode("utf-8") if isinstance(erdf_model, bytes) else erdf_model
    data['erdf_params'] = dict(file_handle["parameters/erdf_params"].attrs)
    data['erdf_duty_cycle'] = file_handle["parameters/erdf_duty_cycle"][()] if "parameters/erdf_duty_cycle" in file_handle else None
    data['time_step_for_accretion'] = file_handle["parameters/time_step_for_accretion"][()]
    data['logfseed'] = file_handle["parameters/logfseed"][()]
    try:
        data['sigmaseed'] = file_handle["parameters/sigmaseed"][()]
    except KeyError:
        data['sigmaseed'] = None

    # 3. Lazy loader for large arrays (BH masses, Lbols from HDF5;
    #    halo masses from the external .npy file)
    data['loader'] = DataLoader(file_handle)
    data['loader'].load_halo_masses(
        path_file_halo_masses, subset_cache_path=path_subset_halo_masses
    )

    # 4. ERDF object (can evaluate mu/sigma for any halo accretion rate)
    data['erdf'] = Erdf(data['erdf_model_str'], data['erdf_params'], duty_cycle=data['erdf_duty_cycle'])

    print("CATALOGUES LOADED FROM FILE: ", path_file)
    print("ERDF MODEL", data['erdf_model_str'])
    print("ERDF PARAMS", data['erdf_params'])
    print("logfseed", data['logfseed'])
    print("sigmaseed", data['sigmaseed'])
    print("time_step_for_accretion", data['time_step_for_accretion'])

    return data





def load_simulation_metadata_full_history(file_handle):
    """Like :func:`load_simulation_metadata` but also sets up the full-history loader.

    The full-history HDF5 file (``bh_evolution_full_history_*.hdf5``) is a
    superset of the standard evolution file: it additionally contains
    per-sub-step arrays (BH mass, luminosity, Eddington ratio) for a
    *subset* of halos selected via ``select_merger_branches``.

    Parameters
    ----------
    file_handle : h5py.File
        Open full-history HDF5 file.
        All 2-D datasets must be in ``(n_steps, n_halos)`` layout.

    Returns
    -------
    data : dict
        Everything from :func:`load_simulation_metadata`, plus:

        * ``full_history_loader`` — :class:`FullDataHistoryLoader` instance
          (access ``.black_hole_masses_full_history``, etc.).
        * ``original_targets_new_indices`` — indices of the original target
          halos within the selected subset.
        * ``original_indices`` — mapping from subset indices back to the
          full halo catalogue.
        * ``subset_selection_mode``, ``n_targets``, ``n_selected`` —
          metadata describing how the subset was chosen.
    """
    data = load_simulation_metadata(file_handle)

    # Full-history lazy loader (sub-step resolution arrays)
    data['full_history_loader'] = FullDataHistoryLoader(file_handle)

    # Subset selection metadata (which halos were included in the full history)
    data['original_targets_new_indices'] = np.asarray(file_handle["subset_selection/original_targets_new_indices"])
    data["original_indices"] = np.asarray(file_handle["subset_selection/original_indices"])

    # Full-history files store their halo subset under `subset_selection/`, NOT
    # the tree_subsample `subset/` group that DataLoader.__init__ looks for, so
    # hand the loader the storage->global map here; get_Halo_mass(s)[storage_idx]
    # then addresses the right physical halo.
    data['loader']._subset_indices = data["original_indices"]

    data["subset_selection_mode"] = file_handle["subset_selection/mode"][()]
    data["n_targets"] = file_handle["subset_selection/n_targets"][()]
    data["n_selected"] = file_handle["subset_selection/n_selected"][()]

    return data


# ------------------------------------------------------------------------------
# Weighted binning helpers
# ------------------------------------------------------------------------------
#
# These mirror ``qhtools.utils.create_binned_functions.create_mass_function`` /
# ``create_luminosity_function`` but accept per-object weights so subsampled
# evolution outputs produce the same volumetric number densities as the
# corresponding full-sim run, in expectation.
#
# When weights is None the call delegates to the upstream unweighted function,
# so plotting scripts can stay sim/subset-agnostic via the dispatcher form.


def create_mass_function_weighted(
    masses,
    weights,
    box_volume,
    lowest_mass=1e8,
    highest_mass=1e16,
    n_bins=45,
    minimum_in_bin=3,
    return_n_counts=False,
):
    """Weighted analogue of ``qhtools.create_mass_function``.

    Per-bin sum of *weights* gives the unbiased halo count; the variance is
    estimated from the sum of ``weights**2`` (Horvitz-Thompson). The
    ``minimum_in_bin`` cut is applied to the *raw* halo count (not the
    weighted sum) — this keeps bins where you have several rare massive
    halos that each carry weight ~1 but drops bins with one heavily
    upweighted low-mass halo.
    """
    masses = np.ascontiguousarray(masses)
    weights = np.ascontiguousarray(weights)

    log_low, log_high = np.log10(lowest_mass), np.log10(highest_mass)
    bins = np.logspace(log_low, log_high, n_bins + 1)
    bin_width_dex = (log_high - log_low) / n_bins
    norm = 1.0 / (bin_width_dex * box_volume)

    weighted_counts, _ = np.histogram(masses, bins, weights=weights)
    w2_counts, _ = np.histogram(masses, bins, weights=weights ** 2)
    raw_counts, _ = np.histogram(masses, bins)

    valid = raw_counts >= minimum_in_bin
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    result = (
        bin_centers[valid],
        weighted_counts[valid] * norm,
        np.sqrt(w2_counts[valid]) * norm,
    )
    if return_n_counts:
        return result + (raw_counts[valid],)
    return result


def create_luminosity_function_weighted(
    quasar_luminosities,
    weights,
    box_volume,
    input_type="L_bol",
    lowest_lim=1e42,
    highest_lim=1e48,
    n_bins=51,
    minimum_in_bin=3,
):
    """Weighted analogue of ``qhtools.create_luminosity_function``.

    Mirrors the upstream sign conventions: log-spaced bins for ``L_bol``,
    linear-in-magnitude bins for ``M_1450``. Per-bin sum of *weights* gives
    the unbiased object count; variance from sum of ``weights**2``.
    """
    quasar_luminosities = np.ascontiguousarray(quasar_luminosities)
    weights = np.ascontiguousarray(weights)

    if input_type == "L_bol":
        log_low, log_high = np.log10(lowest_lim), np.log10(highest_lim)
        bins = np.logspace(log_low, log_high, n_bins + 1)
        bin_width = (log_high - log_low) / n_bins
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
    elif input_type == "M_1450":
        bins = np.linspace(lowest_lim, highest_lim, n_bins + 1)
        bin_width = (highest_lim - lowest_lim) / n_bins
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
    else:
        raise ValueError(
            f"input_type={input_type!r}; expected 'L_bol' or 'M_1450'."
        )

    norm = 1.0 / (bin_width * box_volume)
    weighted_counts, _ = np.histogram(quasar_luminosities, bins, weights=weights)
    w2_counts, _ = np.histogram(quasar_luminosities, bins, weights=weights ** 2)
    raw_counts, _ = np.histogram(quasar_luminosities, bins)

    valid = raw_counts >= minimum_in_bin
    return (
        bin_centers[valid],
        weighted_counts[valid] * norm,
        np.sqrt(w2_counts[valid]) * norm,
    )


def mass_function_auto(masses, box_volume, *, weights=None, **kwargs):
    """Dispatch to the (un)weighted mass function depending on *weights*.

    ``weights=None`` → unweighted ``create_mass_function`` (full-sim runs).
    ``weights`` array → :func:`create_mass_function_weighted` (subsampled).
    """
    if weights is None:
        return create_mass_function(masses, box_volume, **kwargs)
    return create_mass_function_weighted(masses, weights, box_volume, **kwargs)


def luminosity_function_auto(luminosities, box_volume, *, weights=None, **kwargs):
    """Dispatch to the (un)weighted luminosity function depending on *weights*."""
    if weights is None:
        return create_luminosity_function(luminosities, box_volume, **kwargs)
    return create_luminosity_function_weighted(
        luminosities, weights, box_volume, **kwargs
    )


def weighted_percentile(values, weights, p):
    """Empirical weighted percentile (``p`` ∈ [0, 1]).

    In subsample mode each halo carries an inverse-probability weight,
    so ``np.percentile`` on the raw subsample is biased toward "which
    halos we picked" rather than the underlying population. This
    helper returns the smallest value whose cumulative weight reaches
    ``p * total_weight``.

    Matches ``np.percentile(values, 100*p, method='lower')`` for equal
    weights. Falls back to ``np.percentile`` when ``weights`` is None
    (full-sim mode). Returns NaN for empty input.
    """
    values = np.asarray(values)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.percentile(values, 100.0 * p))
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    cum_w = np.cumsum(weights[order])
    total = cum_w[-1]
    if total <= 0:
        return float("nan")
    idx = int(np.searchsorted(cum_w, p * total))
    idx = min(idx, len(values) - 1)
    return float(values[order[idx]])


def weighted_median(values, weights=None):
    """Median of ``values`` under Horvitz-Thompson weights.

    Shorthand for :func:`weighted_percentile` at ``p=0.5``. See that
    function for the rationale and falsey-input handling.
    """
    return weighted_percentile(values, weights, 0.5)


def weighted_std(values, weights=None, ddof=1):
    """Standard deviation of ``values`` under HT weights.

    Uses the reliability-weighted unbiased estimator:
        var = sum(w_i * (x_i - x_mean)^2) / (sum_w - ddof * sum_w/N_eff_unbiased)
    Simplified, for ``ddof=1`` we use the Bessel-corrected form via
    effective sample size N_eff = (sum_w)^2 / sum_w^2:
        var = sum(w * (x - x_mean)^2) / sum_w  *  N_eff / (N_eff - 1)

    Falls back to ``np.std(values, ddof=ddof)`` when ``weights`` is None.
    Returns NaN for empty input.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    if weights is None:
        return float(np.std(values, ddof=ddof))
    weights = np.asarray(weights, dtype=float)
    sw = weights.sum()
    if sw <= 0:
        return float("nan")
    mean = (weights * values).sum() / sw
    var_biased = (weights * (values - mean) ** 2).sum() / sw
    if ddof == 0:
        return float(np.sqrt(var_biased))
    n_eff = sw * sw / (weights * weights).sum()
    if n_eff <= 1:
        return float("nan")
    return float(np.sqrt(var_biased * n_eff / (n_eff - 1)))


# ==============================================================================
# Snapshot selection — with the reduced-snapshot guard
# ==============================================================================
_FULL_Z_GRID_CACHE = {}
_MIS_SNAP_SEEN = set()

#: A target is reported as MIS-SNAPPED when the file's grid forces a redshift
#: this much worse (in |dz|) than the full simulation grid could have supplied.
#: 0.0 would fire on floating-point ties; 0.01 is below any real snapshot gap.
MIS_SNAP_DZ_TOL = 0.01


def full_simulation_redshifts(sim=None):
    """Redshift of EVERY snapshot of the simulation, from ``output_list.txt``.

    Array index == true snapshot number. Returns ``None`` if the file is not
    reachable (e.g. plotting on a machine without the sim tree mounted), which
    disables the mis-snap warning rather than breaking the plot.
    """
    sim = sim or simulation_name
    if sim not in _FULL_Z_GRID_CACHE:
        try:
            from baqaro.utils.my_dir import get_input_path_HBT_data
            p = os.path.join(get_input_path_HBT_data(source=source_dir), sim,
                             "output_list.txt")
            _FULL_Z_GRID_CACHE[sim] = np.loadtxt(p)
        except Exception:
            _FULL_Z_GRID_CACHE[sim] = None
    return _FULL_Z_GRID_CACHE[sim]


def snapshot_index_for_redshift(redshifts, z_target, snapshots=None, label=None,
                                restrict_to=None):
    """Index into the FILE's snapshot axis of the snapshot nearest ``z_target``.

    Drop-in replacement for ``np.abs(redshifts - z_target).argmin()`` that adds
    the guard bare argmin lacks. A run which stored only a SUBSET of the
    simulation's snapshots (the multinode full-catalogue files keep ~17 of 145)
    still returns *a* nearest snapshot, however far away it is.

    So: compare the match this file can offer against the best the FULL
    ``output_list.txt`` grid could have supplied, and WARN when the file's
    reduced grid is worse. Warn only — the caller still gets a usable index, and
    a full-snapshot run can never trigger it.

    ``snapshots`` is the file's TRUE snapshot numbers (``data["snapshots"]`` /
    ``loader._snapshots``); without it the check is skipped, since the file's
    row index cannot be mapped onto the simulation grid.

    ``restrict_to`` is an optional whitelist of TRUE snapshot numbers the search
    may return — pass ``training_snapshots_default`` to pin a forward-run figure
    to the EMULATOR's redshift ladder, so it and the emulator-based
    posterior-predictive figure draw the same panel at the same snapshot. (The
    emulator has no z≈7.0 rung: its high-z rungs are snap 35 / z=7.315 and snap
    37 / z=6.708, while a 41-snap forward run also holds snap 36 / z=7.005 — a
    0.30-in-z gap worth ~0.2 dex in the z≈7 QLF.) Requires ``snapshots``; the
    mis-snap guard then measures against the best the RESTRICTED full grid could
    offer, so deliberately snapping to the ladder is not reported — only a run
    that fails to store the ladder snapshot is.
    """
    redshifts = np.asarray(redshifts, dtype=float)

    allowed_snaps = None
    if restrict_to is not None:
        if snapshots is None:
            raise ValueError("snapshot_index_for_redshift: restrict_to needs the "
                             "file's true `snapshots` array to map row -> snapshot")
        allowed_snaps = np.asarray(list(restrict_to), dtype=int)
        cand = np.where(np.isin(np.asarray(snapshots, dtype=int), allowed_snaps))[0]
        if len(cand) == 0:
            raise ValueError(
                f"snapshot_index_for_redshift: this run stores none of the "
                f"restrict_to snapshots {allowed_snaps.tolist()}")
        i = int(cand[np.abs(redshifts[cand] - z_target).argmin()])
    else:
        i = int(np.abs(redshifts - z_target).argmin())

    if snapshots is None:
        return i
    full_z = full_simulation_redshifts()
    if full_z is None:
        return i

    got_dz = abs(float(redshifts[i]) - z_target)
    if allowed_snaps is None:
        best_snap = int(np.abs(full_z - z_target).argmin())
    else:
        # Best the restricted grid could supply if the run stored every allowed
        # snapshot — the honest reference once the caller has opted into the
        # ladder (against the unrestricted grid this would warn on every panel).
        in_range = allowed_snaps[allowed_snaps < len(full_z)]
        best_snap = int(in_range[np.abs(full_z[in_range] - z_target).argmin()])
    best_dz = abs(float(full_z[best_snap]) - z_target)
    if got_dz - best_dz <= MIS_SNAP_DZ_TOL:
        return i

    got_snap = int(np.asarray(snapshots)[i])
    key = (label, round(float(z_target), 4))
    if key not in _MIS_SNAP_SEEN:
        _MIS_SNAP_SEEN.add(key)
        where = f"[{label}] " if label else ""
        grid = "restrict_to grid" if allowed_snaps is not None else "full grid"
        print(
            f"  ⚠ {where}MIS-SNAPPED z={z_target:g}: this run stores "
            f"{len(redshifts)} of {len(full_z)} snapshots, so the nearest it can "
            f"offer is snap {got_snap} (z={redshifts[i]:.3f}, dz={redshifts[i] - z_target:+.3f}). "
            f"The {grid} has snap {best_snap} (z={full_z[best_snap]:.3f}, "
            f"dz={full_z[best_snap] - z_target:+.3f}). Re-run saving snap {best_snap} "
            f"to plot this target at the redshift it was asked for.",
            flush=True,
        )
    return i