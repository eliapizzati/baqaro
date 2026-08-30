"""The adopted fiducial, in ONE place.

Every constant here is a plain literal. This module deliberately reads **no**
environment variables and imports nothing from the package, which is what makes
it safe to import at any point — in particular *before*
``plotting_paper/fiducial_data.py`` pins ``os.environ``. Importing
``utils.sim_config`` there instead would bind its env-derived aliases
(``simulation_name``, ``max_snap``, …) to the ambient environment rather than
the pinned one, which is the ordering trap that module exists to avoid.

Consumers
---------
``utils.sim_config``
    Re-exports these as its ``FIDUCIAL_*`` names, which the pipeline entry
    points use for their zero-env defaults.
``plotting_paper.fiducial_data``
    Builds ``PAPER_FIDUCIAL`` from them, so the paper figures and the pipeline
    cannot disagree about what the fiducial is.

Moving the fiducial
-------------------
Change the values here and nowhere else. They previously lived in two places —
``sim_config``'s ``FIDUCIAL_*`` block and ``fiducial_data``'s ``PAPER_FIDUCIAL``
dict — which restated the same bestfit name, growth cap and f_eff flag as
separate literals and could therefore drift apart.

A repoint is deliberately a CODE change, not a filesystem one: it shows up in
``git diff``, and every output filename still carries the full token chain, so a
figure or log always says which model produced it. (There is a human-facing
``evolution/fiducial/`` symlink tree on disk with short names; nothing in the
code reads it, by design.)
"""

#: Simulation the fiducial was run on.
SIM = "L2800N10080"

#: Root snapshot: 144 == z=0.
MAX_SNAP = 144

#: Halo-data variant selectors (part of every halo-history filename).
FOLD_SUBHALO_MASS = True
MERGER_DELAY_MODE = "instant_new"

#: Registry key of the adopted best-fit parameter set. Forward-run outputs carry
#: it as the ``_bestfit_<name>`` filename token.
BESTFIT_NAME = "qcc_ck22final_v1"

#: Per-snapshot growth-sum cap. 50.0 disables the cap (and drops the ``_g``
#: token); anything else is a model change requiring re-DE + re-MCMC.
GROWTH_SUM_MAX = 6.21

#: Madau effective-efficiency correction -> ``_feffcorr`` token.
MADAU_FEFF_CORRECTION = True

#: Savitzky-Golay pre-smoothing of the training curves.
SMOOTH_TRAINING = True

#: Notes tokens for the training HDF5 and the emulator ``.xz``.
NOTES_TRAINING = "clean_z0_g6.21_K22_final"
NOTES_EMULATION = "clean_z0_g6.21_K22_final_smooth"

#: Subset tag of the run the PAPER population figures use. This is the multinode
#: FULL-CATALOGUE run (all 2.255e9 halos, weights == 1, no shot noise) — NOT the
#: stratified subsample, which needs its inverse-probability weights applied and
#: is meant to be differenced against its own baseline rather than read alone.
PAPER_SUBSET_TAG = "multinode_root144_v4"

#: Notes token on the paper's forward run (empty: the filename normalisation
#: dropped the redundant `_ck22final_` token).
PAPER_NOTES_FILE = ""
