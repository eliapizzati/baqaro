"""Run provenance — stamp a self-describing record into output HDF5 files.

Every forward run (``main_evolution`` / ``main_evolution_full_history``) writes a
``provenance/`` group whose attrs fully document HOW the file was produced: the
run-identity fields (sim, snap, fold/merger, notes, bestfit, subset/selection
tag, the 6 free params, …), plus the git revision, a UTC timestamp, the host,
the command line, and a snapshot of every ``BAQARO_*`` environment variable.

So to recover the parameters of any run, open it and read the attrs::

    import h5py
    with h5py.File(path) as f:
        print(dict(f["provenance"].attrs))          # full record
        print(f.attrs["bestfit_name"], f.attrs["subset_tag"])   # quick top-level

The key identity fields are also mirrored onto the file's ROOT attrs for a quick
``f.attrs[...]`` lookup without descending into the group.
"""

import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone

# Identity fields mirrored to the file's root group for quick lookup.
_ROOT_MIRROR = (
    "simulation_name", "max_snap", "min_snap", "notes_file", "bestfit_name",
    "fold_subhalo_mass", "merger_delay_mode", "subset_tag", "selection_tag",
    "name_file",
)


def git_revision(cwd=None):
    """Best-effort ``<sha>[-dirty]`` of the working tree. Returns "" on failure
    (not a git checkout, git absent, etc.) — never raises."""
    cwd = cwd or os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd,
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        # Compare against HEAD (not just the unstaged worktree) so a run
        # launched with STAGED-but-uncommitted edits is still marked -dirty.
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL,
        ) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return ""


def swift_env_snapshot():
    """All ``BAQARO_*`` env vars currently set (the run's effective config)."""
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith("BAQARO_")}


def _attr_safe(v):
    """Coerce a value to something h5py can store as an attribute."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return bool(v)
    return v


def write_run_provenance(h5file, fields, *, group="provenance", mirror_to_root=True):
    """Write a self-describing provenance record into an open ``h5py.File``.

    Parameters
    ----------
    h5file : open ``h5py.File`` (mode "w"/"a").
    fields : dict of scalar run-identity values (str / int / float / bool / None).
        ``None`` is stored as "". Build this in the caller from its run config.
    group : provenance group name (default ``"provenance"``).
    mirror_to_root : also copy the well-known identity fields (see ``_ROOT_MIRROR``)
        onto the file's root attrs for quick ``f.attrs[...]`` access.

    Auto-added attrs: ``git_revision``, ``created_utc``, ``hostname``,
    ``python``, ``argv``, ``swift_env_json``.
    """
    grp = h5file.require_group(group)
    for k, v in fields.items():
        grp.attrs[k] = _attr_safe(v)
    grp.attrs["git_revision"] = git_revision()
    grp.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
    grp.attrs["hostname"] = socket.gethostname()
    grp.attrs["python"] = sys.version.split()[0]
    grp.attrs["argv"] = " ".join(sys.argv)
    grp.attrs["swift_env_json"] = json.dumps(swift_env_snapshot())
    if mirror_to_root:
        for k in _ROOT_MIRROR:
            if k in fields:
                h5file.attrs[k] = _attr_safe(fields[k])
    return grp
