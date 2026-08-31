"""Gates for the portable (HDF5) emulator loader.

Self-contained: builds a tiny synthetic portable file following the exporter
schema, checks the prediction against an independent hand computation, and
checks that ``load_emulators`` falls back to the portable format when the
pickle is absent. No simulation data, no sampler, always runs.
"""

import os

import h5py
import numpy as np
import pytest

from baqaro.emulation.portable_emulator import PortableEmulator, matern
from baqaro.emulation.loading_helpers import load_emulators

RNG = np.random.default_rng(7)
N_PAR, N_TRAIN, N_COMP = 2, 5, 3
SHAPE = (2, 4)


def _write_synthetic(path, kernel_type="matern32"):
    Xt = RNG.normal(size=(N_TRAIN, N_PAR))
    with h5py.File(path, "w") as f:
        f.create_dataset("X_train", data=Xt)
        f.create_dataset("alpha", data=RNG.normal(size=(N_COMP, N_TRAIN)))
        f.create_dataset("kernel_amplitude", data=np.abs(RNG.normal(size=N_COMP)) + 0.1)
        f.create_dataset("kernel_metric", data=np.abs(RNG.normal(size=(N_COMP, N_PAR))) + 0.5)
        f.create_dataset("gp_mean", data=RNG.normal(size=N_COMP))
        f.create_dataset("scaler_mean", data=np.array([0.5, -1.0]))
        f.create_dataset("scaler_scale", data=np.array([2.0, 0.25]))
        f.create_dataset("pca_components", data=RNG.normal(size=(N_COMP, int(np.prod(SHAPE)))))
        f.create_dataset("pca_mean", data=RNG.normal(size=int(np.prod(SHAPE))))
        f.create_dataset("param_ranges", data=np.array([[-3.0, 3.0], [-3.0, 3.0]]))
        f.attrs["kernel_type"] = kernel_type
        f.attrs["floor_value"] = -12.0
        f.attrs["output_shape"] = np.asarray(SHAPE, dtype=np.int64)
        f.attrs["quantity"] = "qlf"
        f.attrs["param_names"] = ["a", "b"]
        ax = f.create_group("axes")
        ax.create_dataset("log_bins", data=np.linspace(44, 48, SHAPE[1]))
        ax.create_dataset("redshift", data=np.array([2.0, 6.0]))
    return Xt


def _predict_by_hand(f, X):
    """Independent implementation of the documented arithmetic."""
    xs = (X - f["scaler_mean"][:]) / f["scaler_scale"][:]
    Xt, alpha = f["X_train"][:], f["alpha"][:]
    amp, met, gm = f["kernel_amplitude"][:], f["kernel_metric"][:], f["gp_mean"][:]
    w = np.empty((xs.shape[0], alpha.shape[0]))
    for i in range(alpha.shape[0]):
        r2 = np.sum((xs[:, None, :] - Xt[None, :, :]) ** 2 / met[i], axis=-1)
        w[:, i] = gm[i] + amp[i] * matern(r2, str(f.attrs["kernel_type"])) @ alpha[i]
    flat = w @ f["pca_components"][:] + f["pca_mean"][:]
    return flat.reshape((-1, *tuple(f.attrs["output_shape"]))) + f.attrs["floor_value"]


@pytest.mark.parametrize("kernel_type", ["matern12", "matern32", "matern52"])
def test_prediction_matches_hand_computation(tmp_path, kernel_type):
    path = tmp_path / "emulator_qlf.hdf5"
    _write_synthetic(path, kernel_type)
    emu = PortableEmulator.load(path)
    X = RNG.uniform(-2, 2, size=(6, N_PAR))
    got = emu.predict_mean_only(X)
    with h5py.File(path, "r") as f:
        want = _predict_by_hand(f, X)
    assert got.shape == (6, *SHAPE)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_interface_contract(tmp_path):
    path = tmp_path / "emulator_qlf.hdf5"
    _write_synthetic(path)
    emu = PortableEmulator.load(path)
    assert emu.param_names == ["a", "b"]
    assert emu.param_ranges.shape == (N_PAR, 2)
    assert set(emu.axis_data) == {"log_bins", "redshift"}
    assert len(emu.gps) == N_COMP
    assert emu.precompute_alpha() is None            # no-op
    mu, std = emu.predict([[0.0, 0.0]])
    assert mu.shape == (1, *SHAPE) and np.all(std == 0.0)
    # the runner's save/clear/restore of cached alpha state must round-trip
    saved = emu._alpha_vectors
    emu._alpha_vectors = None
    with pytest.raises(RuntimeError):
        emu.predict_mean_only([[0.0, 0.0]])
    emu._alpha_vectors = saved
    assert emu.predict_mean_only([[0.0, 0.0]]).shape == (1, *SHAPE)
    with pytest.raises(ValueError):
        emu.predict_mean_only([[0.0, 0.0, 0.0]])


def test_load_emulators_falls_back_to_portable(tmp_path, capsys):
    emudir = tmp_path / "emulators"
    emudir.mkdir()
    _write_synthetic(emudir / "emulator_qlf_TESTNAME.hdf5")
    _write_synthetic(emudir / "emulator_cerdf.hdf5")     # release-style short name
    flags = {"qlf": True, "cerdf": True, "bhmf": True, "qhmf": False}
    emus = load_emulators(str(tmp_path), "TESTNAME", flags)
    assert isinstance(emus["qlf"], PortableEmulator)     # long-name portable
    assert isinstance(emus["cerdf"], PortableEmulator)   # short-name portable
    assert emus["bhmf"] is None                          # nothing on disk
    assert emus["qhmf"] is None                          # flag off
