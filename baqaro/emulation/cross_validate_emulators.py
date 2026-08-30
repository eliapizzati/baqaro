"""
Standalone Cross-Validation for the GP Emulators
================================================

Companion to `main_emulation.py`. When the emulators are fit with
`BAQARO_SKIP_CV=1` (fit + save only, fast), run this afterwards to get the
honest k-fold cross-validation RMSE for each quantity.

CV cannot be read off a saved emulator: an honest estimate requires
refitting the GP on each leave-one-fold-out split. This script therefore
reproduces the *exact* load -> filter -> PCA setup that `main_emulation.py`
uses (same floors, same bad-sim filter, same PCA-component selection,
same kernel), then calls `GeneralEmulatorGP.cross_validate_physical`. The
numbers it prints are identical to what `main_emulation.py` would have
printed inline — it just decouples the (expensive) CV from the fit.

It does NOT load or modify the saved `.xz` emulators.

Env vars (same identifier block as main_emulation.py)
-----------------------------------------------------
    BAQARO_SOURCE_DIR, BAQARO_SIM, BAQARO_MAX_SNAP, BAQARO_FOLD_SUBHALO_MASS,
    BAQARO_MERGER_DELAY_MODE, BAQARO_NOTES_FILE_TRAINING, BAQARO_SUBSET_TAG,
    BAQARO_MAX_PCA_COMPONENTS  (default 15)

CV-specific:
    BAQARO_CV_KFOLDS    (default 5)
    BAQARO_CV_QUANTITIES (comma list subset of qlf,cerdf,bhmf,qhmf; default all)
    BAQARO_CV_REPORT    (path for the JSON report; default
                        <out>/emulators/cv_report_<name>.json)

Example
-------
    env BAQARO_SIM=L2800N10080 BAQARO_MAX_SNAP=144 BAQARO_FOLD_SUBHALO_MASS=1 \\
        BAQARO_MERGER_DELAY_MODE=instant_new BAQARO_NOTES_FILE_TRAINING=fid1_z0 \\
        BAQARO_SUBSET_TAG=root144_flatN500000_K22_logM10.0to15.5_seed42_v3 \\
        python -u -m baqaro.emulation.cross_validate_emulators
"""

import os
# Headless-safe: hunt_cv_failure builds (but does not show) a figure.
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import time
import numpy as np

from baqaro.emulation.emulation_core_functions import (
    GeneralEmulatorGP, GeneralEmulatorPCA,
)
from baqaro.emulation.emulation_testing_helpers import hunt_cv_failure
from baqaro.emulation.loading_helpers import load_training_data
from baqaro.utils.my_dir import get_output_path
from baqaro.utils.sim_config import (
    simulation_name,
    max_snap,
    fold_subhalo_mass,
    merger_delay_mode,
    FIDUCIAL_NOTES_TRAINING,
    FIDUCIAL_NOTES_EMULATION,
    FIDUCIAL_SMOOTH_TRAINING,
)


# ----------------------------------------------------------------------
# 1. CONFIG — mirror main_emulation.py exactly
# ----------------------------------------------------------------------
source_dir = os.environ.get("BAQARO_SOURCE_DIR", "machine_igm")

notes_file_training = os.environ.get("BAQARO_NOTES_FILE_TRAINING") or FIDUCIAL_NOTES_TRAINING
notes_file_emulation = (os.environ.get("BAQARO_NOTES_FILE_EMULATION")
                        or os.environ.get("BAQARO_NOTES_FILE_TRAINING")
                        or FIDUCIAL_NOTES_EMULATION)

physical_floor = -9.5   # below this a bin is "empty"
pca_floor = -10.0       # value assigned to empty bins (matches main_emulation.py)
kernel_type = "matern32"
filter_training_data = True

# Default to the production subsampled run's tag (auto-derived from
# sim_config geometry + subsample env defaults) so callers needn't set
# BAQARO_SUBSET_TAG. Env presence wins: BAQARO_SUBSET_TAG="" => full-sim.
if "BAQARO_SUBSET_TAG" in os.environ:
    subset_tag = os.environ["BAQARO_SUBSET_TAG"]
else:
    from baqaro.core_functions.tree_subsample import resolve_subset_tag
    subset_tag = resolve_subset_tag(max_snap)
erdf_model = "log_normal_evol_halo_mass"

k_folds = int(os.environ.get("BAQARO_CV_KFOLDS", "5"))
_max_pca = int(os.environ.get("BAQARO_MAX_PCA_COMPONENTS", "15"))

# Same filename composition order as main_training / main_emulation:
# maxsnap -> foldmass -> merger_delay_mode -> notes -> subset_tag.
name_file = "{}_erdf_{}_maxsnap_{}".format(simulation_name, erdf_model, max_snap)
if fold_subhalo_mass:
    name_file += "_foldmass"
if merger_delay_mode != "instant_old":
    name_file += "_{}".format(merger_delay_mode)

name_file_training = name_file + ("_{}".format(notes_file_training) if notes_file_training else "")
name_file_emulation = name_file + ("_{}".format(notes_file_emulation) if notes_file_emulation else "")
if subset_tag:
    name_file_training += "_sub_{}".format(subset_tag)
    name_file_emulation += "_sub_{}".format(subset_tag)

# Growth-cap + madau-f_eff tokens (after _sub_, matching main_training /
# main_emulation) so CV reads the matching capped/_feffcorr training file.
from baqaro.utils.sim_config import growth_feff_suffix
_gf_suffix = growth_feff_suffix()
name_file_training += _gf_suffix
name_file_emulation += _gf_suffix

path_out = get_output_path(source=source_dir)
path_file_training = os.path.join(
    path_out, "training", f"training_data_emulation_{name_file_training}.hdf5"
)
report_path = os.environ.get(
    "BAQARO_CV_REPORT",
    os.path.join(path_out, "emulators", f"cv_report_{name_file_emulation}.json"),
)


# ----------------------------------------------------------------------
# 2. LOAD + FILTER — identical to main_emulation.py
# ----------------------------------------------------------------------
print("Loading data from", path_file_training)
data = load_training_data(path_file_training, physical_floor=physical_floor, pca_floor=pca_floor)
print("CATALOGUES LOADED")

log_qlfs = data["log_qlfs"]
log_cerdfs = data["log_cerdfs"]
log_bhmfs = data["log_bhmfs"]
log_qhmfs = data["log_qhmfs"]

redshift_keys = data["redshift_keys"]
log_lbins = data["log_lbins"]
log_bins_cerdf = data["log_bins_cerdf"]
log_mbins_bhmf = data["log_mbins_bhmf"]
log_mbins_qhmf = data["log_mbins_qhmf"]
log_L_thresholds = data["log_L_thresholds"]

params = data["params"]
param_names = data["param_names"]
param_ranges = data["param_ranges"]

name_emulators = ["QLF", "CERDF", "BHMF", "QHMF"]
all_data_arrays = [log_qlfs, log_cerdfs, log_bhmfs, log_qhmfs]
all_bin_arrays = [log_lbins, log_bins_cerdf, log_mbins_bhmf, log_mbins_qhmf]

if filter_training_data:
    print("\n--- FILTERING BAD SIMULATIONS (same rule as main_emulation) ---")
    bad_indices_global = []
    for arr, name in zip(all_data_arrays, name_emulators):
        global_floor_fraction = np.mean(arr <= physical_floor + 0.1)
        print(f"\n{name}: floored {global_floor_fraction:.2%}")
        if global_floor_fraction > 0.95:
            print(f"  Skipping {name} from filtering (globally floored).")
            continue
        emptiness = np.mean(arr.reshape(arr.shape[0], -1) <= physical_floor + 0.1, axis=1)
        bad_idx = np.where(emptiness > 0.80)[0]
        print(f"  Sims with >80% empty: {len(bad_idx)} / {len(emptiness)}")
        bad_indices_global.extend(bad_idx)

    unique_bad = np.unique(bad_indices_global)
    print(f"\nGLOBALLY removing {len(unique_bad)} simulations")
    keep = np.ones(params.shape[0], dtype=bool)
    keep[unique_bad] = False
    params = params[keep]
    all_data_arrays = [a[keep] for a in all_data_arrays]
    print(f"Remaining simulations after filtering: {params.shape[0]}")


# ----------------------------------------------------------------------
# 3. CROSS-VALIDATE each quantity (PCA refit on filtered data, then k-fold CV)
# ----------------------------------------------------------------------
_requested = os.environ.get("BAQARO_CV_QUANTITIES", "")
requested = {s.strip().lower() for s in _requested.split(",") if s.strip()} or None

report = {
    "training_file": path_file_training,
    "name_file_emulation": name_file_emulation,
    "n_runs_after_filter": int(params.shape[0]),
    "k_folds": k_folds,
    "max_pca_components": _max_pca,
    "physical_floor": physical_floor,
    "pca_floor": pca_floor,
    "kernel_type": kernel_type,
    "quantities": {},
}

summary_rows = []
for quantity, log_bins, name in zip(all_data_arrays, all_bin_arrays, name_emulators):
    if requested is not None and name.lower() not in requested:
        continue

    print(f"\n{'='*60}\n{name}: CV  (data shape {quantity.shape})")

    # Optional SG smoothing before PCA — must match main_emulation.py so the
    # CV measures the SAME (smoothed) emulator. Per-quantity (window, poly)
    # from SG_PARAMS_BY_QUANTITY. ON by default (FIDUCIAL_SMOOTH_TRAINING) to
    # match main_emulation.py; set BAQARO_SMOOTH_TRAINING=0 for the raw path.
    if os.environ.get("BAQARO_SMOOTH_TRAINING",
                      "1" if FIDUCIAL_SMOOTH_TRAINING else "0") == "1":
        from baqaro.emulation.sg_smoothing import (
            smooth_function_rows, SG_PARAMS_BY_QUANTITY,
        )
        _w, _p = SG_PARAMS_BY_QUANTITY.get(name.lower(), (7, 2))
        quantity = smooth_function_rows(quantity, _w, _p, floor=pca_floor)
        print(f"  SG-smoothed before fit (window={_w}, poly={_p})")

    # PCA — same component selection as main_emulation.py
    max_components = min(_max_pca, quantity.shape[0] - 1)
    emu_pca = GeneralEmulatorPCA(n_components=max_components, floor_value=pca_floor)
    emu_pca.fit(quantity)
    cumsum = np.cumsum(emu_pca.pca.explained_variance_ratio_)
    optimal_n = int(np.argmax(cumsum > 0.998) + 1) if np.any(cumsum > 0.998) else max_components
    emu_pca = GeneralEmulatorPCA(n_components=optimal_n, floor_value=pca_floor)
    # The PCA basis is fit on the full set; only the GP is refit per fold.
    training_weights = emu_pca.fit_transform(quantity)
    print(f"  PCA n_components={optimal_n} (var={np.sum(emu_pca.pca.explained_variance_ratio_):.4f})")

    axis_info = {"redshift": redshift_keys, "log_L_threshold": log_L_thresholds, "log_bins": log_bins}
    emu_gp = GeneralEmulatorGP(
        pca_model=emu_pca, param_names=param_names, param_ranges=param_ranges,
        axis_data=axis_info, kernel_type=kernel_type, verbose=True,
    )

    t0 = time.time()
    cv_results = emu_gp.cross_validate_physical(
        params, training_weights, Y_physical=quantity,
        k_folds=k_folds, normalize_params=True, verbose=True,
        real_threshold=physical_floor,
    )
    dt = time.time() - t0

    rmse = float(cv_results.get("rmse_physical", -1))
    med = float(cv_results.get("median_ae_physical", -1))
    mx = float(cv_results.get("max_error_physical", -1))
    print(f"\n{name}: RMSE={rmse:.4f}  MedianAE={med:.4f}  Max={mx:.4f} dex  ({dt:.1f}s, n_pca={optimal_n})")
    hunt_cv_failure(cv_results, floor_value=pca_floor, real_threshold=physical_floor)

    report["quantities"][name] = {
        "rmse_physical": rmse, "median_ae_physical": med, "max_error_physical": mx,
        "n_pca_components": int(optimal_n), "cv_seconds": dt,
    }
    summary_rows.append((name, rmse, med, mx, optimal_n))


# ----------------------------------------------------------------------
# 4. SUMMARY + JSON report
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"CROSS-VALIDATION SUMMARY  ({k_folds}-fold, {params.shape[0]} runs)")
print(f"{'Quantity':<8} {'RMSE':>8} {'MedAE':>8} {'MaxErr':>9} {'n_pca':>6}   verdict")
for name, rmse, med, mx, npca in summary_rows:
    verdict = ("EXCELLENT" if rmse < 0.05 else "GOOD" if rmse < 0.15
               else "FAIR" if rmse < 0.30 else "POOR")
    print(f"{name:<8} {rmse:8.4f} {med:8.4f} {mx:9.4f} {npca:6d}   {verdict}")

os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as fh:
    json.dump(report, fh, indent=2)
print(f"\nJSON report written to {report_path}")
