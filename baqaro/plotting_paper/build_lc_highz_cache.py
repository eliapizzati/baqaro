"""
Build a compact column-cache for the high-z lightcurve gallery figure.

``plotting_lightcurves_highz.py`` draws a handful of single-BH sub-step
lightcurves out of the full-history fiducial. Those sub-step arrays live in the
FH HDF5 as ``(n_steps, n_halos)`` **contiguous, uncompressed** datasets — for
the z=2 massthr fiducial that is ``(9849, 4051282)`` float32 ≈ **159 GB each**
(BH mass, L_bol, eta). The plotting loader reads them whole (``[:,:]``), so a
single re-render costs ~hundreds of GB of I/O and many minutes, even though the
figure only needs a few columns.

A per-object column read is NOT a shortcut: a column is strided by ~16 MB across
the whole 159 GB array, so reading even 6 columns is as slow as the full load.

This builder instead reads each array **once, sequentially, in row-chunks**
(RAM-bounded), extracts the columns for every "bright" object
(``M_BH(z=Z_SELECT) >= 10^CACHE_FLOOR``), and writes a small
``(n_steps, n_cache)`` npz. After that, ``plotting_lightcurves_highz.py`` loads
the npz (~hundreds of MB) in <1 s and never touches the 159 GB arrays — so you
can iterate on the figure (bands, labels, colours) freely.

Run once (slow — one full sequential pass over the FH file):

    env BAQARO_HEADLESS=1 python -m baqaro.plotting_paper.build_lc_highz_cache

Env knobs:
  * ``BAQARO_PAPER_LC_CACHE_FLOOR`` — log10 M_BH(z=Z_SELECT) floor for cached
    objects (default ``7.8``; must sit BELOW the lowest band you plan to plot).
  * ``BAQARO_PAPER_LC_CACHE_CHUNK`` — sequential row-block size (default ``512``;
    peak RAM ≈ CHUNK × n_halos × 4 B ≈ 8 GB at 512).
  * ``BAQARO_PAPER_LC_CACHE_FORCE`` — set to ``1`` to rebuild even if the cache
    already exists.

The cache path is derived from the FH run's ``name_file`` and matches what
``plotting_lightcurves_highz.py`` looks for; see ``lc_highz_cache_path``.
"""

import os
import numpy as np
import h5py

# Pins the paper fiducial env (must precede load_data_to_plot reads).
from baqaro.plotting_paper.fiducial_data import (
    load_simulation_metadata_full_history,
    path_file_lc_highz_fid,
    path_out,
)

# Selection epoch — MUST match Z_SELECT in plotting_lightcurves_highz.py.
Z_SELECT = 6.0

# Heavy sub-step datasets to cache, mapped to short npz keys consumed by the plot.
_FH_DATASETS = {
    "black_hole_masses_full_history": "mbh",
    "Lbols_full_history": "lbol",
    "etas_full_history": "eta",
}


def lc_highz_cache_path(name_file):
    """Cache filename for a given FH ``name_file`` (creates the dir)."""
    cache_dir = os.path.join(path_out, "full_history_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"lc_highz_cache_{name_file}.npz")


def build_cache():
    """Build the high-z lightcurve cache the paper figure reads.

    Scans the full-history run for objects above the luminosity floor
    (``BAQARO_PAPER_LC_CACHE_FLOOR``) in chunks of
    ``BAQARO_PAPER_LC_CACHE_CHUNK`` and writes the result once, so
    ``plotting_lightcurves_highz`` does not re-derive it on every render.
    Set ``BAQARO_PAPER_LC_CACHE_FORCE=1`` to rebuild an existing cache.
    """
    cache_floor = float(os.environ.get("BAQARO_PAPER_LC_CACHE_FLOOR", "7.8"))
    chunk = int(os.environ.get("BAQARO_PAPER_LC_CACHE_CHUNK", "512"))
    force = os.environ.get("BAQARO_PAPER_LC_CACHE_FORCE", "") == "1"

    with h5py.File(path_file_lc_highz_fid, "r",
                   rdcc_nbytes=256 * 1024 * 1024, rdcc_nslots=10007) as file:
        data = load_simulation_metadata_full_history(file)  # lazy — no heavy load
        loader = data["loader"]
        redshifts = data["redshifts"]

        name_file = file.attrs["name_file"]
        if isinstance(name_file, bytes):
            name_file = name_file.decode("utf-8")
        out_path = lc_highz_cache_path(name_file)
        if os.path.exists(out_path) and not force:
            print(f"Cache already exists (set BAQARO_PAPER_LC_CACHE_FORCE=1 to rebuild):\n  {out_path}")
            return out_path

        # Cheap selection: M_BH at the selection epoch (per-snapshot row read).
        snap_sel = int(np.argmin(np.abs(redshifts - Z_SELECT)))
        mbh_z6 = loader.get_BH_mass(snap_sel)
        cache_cols = np.flatnonzero(mbh_z6 >= 10.0 ** cache_floor).astype(np.int64)
        # np.flatnonzero returns ascending indices → cache_cols is sorted, which
        # the plot relies on for a searchsorted lookup.
        n_cache = int(cache_cols.size)

        grp = file["full_history"]
        n_steps = int(grp["times_full_history"].shape[0])
        n_halos = int(grp[next(iter(_FH_DATASETS))].shape[1])
        print(f"FH file: {path_file_lc_highz_fid}")
        print(f"  full-history arrays: ({n_steps}, {n_halos}) float32 "
              f"(~{n_steps * n_halos * 4 / 1e9:.0f} GB each)")
        print(f"  z(snap {snap_sel}) = {redshifts[snap_sel]:.2f}; "
              f"caching M_BH >= 1e{cache_floor:.1f}: {n_cache} objects "
              f"-> ({n_steps}, {n_cache}) x3 = "
              f"{n_steps * n_cache * 4 * 3 / 1e6:.0f} MB", flush=True)

        saved = {"cache_cols": cache_cols,
                 "times": np.asarray(grp["times_full_history"])}

        for ds_name, key in _FH_DATASETS.items():
            d = grp[ds_name]
            buf = np.empty((n_steps, n_cache), dtype=np.float32)
            for r0 in range(0, n_steps, chunk):
                r1 = min(r0 + chunk, n_steps)
                # Sequential read of a contiguous row-block (~chunk*16 MB), then
                # subselect the cached columns in RAM (avoids strided per-column I/O).
                block = d[r0:r1, :]
                buf[r0:r1] = block[:, cache_cols]
                print(f"  {ds_name}: rows {r1}/{n_steps}", flush=True)
            saved[key] = buf

        saved["cache_floor"] = np.float64(cache_floor)
        saved["snap_sel"] = np.int64(snap_sel)
        saved["z_select"] = np.float64(Z_SELECT)
        saved["name_file"] = name_file

        print(f"Writing cache -> {out_path}", flush=True)
        np.savez(out_path, **saved)
        print(f"Done. {n_cache} objects cached; "
              f"{os.path.getsize(out_path) / 1e6:.0f} MB on disk.")
        return out_path


if __name__ == "__main__":
    build_cache()
