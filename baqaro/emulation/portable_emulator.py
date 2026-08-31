"""Load and evaluate a portable (HDF5) emulator with numpy alone.

The trained emulators exist in two serialisations. The ``.xz`` pickles hold
live :class:`GeneralEmulatorGP` objects and need the exact Python, george and
scikit-learn versions they were written with. The portable ``.hdf5`` files
hold the same emulator as plain arrays -- the standardised training inputs,
the precomputed GP solution ``alpha = K^-1 (y - mean)`` per PCA component, the
kernel hyperparameters, and the PCA basis -- so a prediction is pure
arithmetic:

    xs   = (X - scaler_mean) / scaler_scale
    K*   = amp_i * matern(|xs - X_train|; metric_i)
    w_i  = gp_mean_i + K* @ alpha_i
    flat = w @ pca_components + pca_mean
    out  = flat.reshape(output_shape) + floor_value

:class:`PortableEmulator` evaluates exactly that, and exposes the interface
the inference uses (``predict_mean_only``, ``precompute_alpha``,
``axis_data``, ``param_names``, ``param_ranges``), so ``main_mcmc`` runs from
the portable files with no pickle anywhere. The files are written by the
release exporter; the mean prediction agrees with the pickle emulators to
floating-point precision. The one thing the portable files do not carry is
the training covariance, so no predictive variance is available -- ``predict``
returns a zero standard deviation.
"""

import os

import h5py
import numpy as np


def matern(r2, kind):
    """Matern covariance from the SQUARED scaled distance."""
    r = np.sqrt(np.maximum(r2, 0.0))
    if kind == "matern12":
        return np.exp(-r)
    if kind == "matern32":
        a = np.sqrt(3.0) * r
        return (1.0 + a) * np.exp(-a)
    if kind == "matern52":
        a = np.sqrt(5.0) * r
        return (1.0 + a + a * a / 3.0) * np.exp(-a)
    raise ValueError(f"unknown kernel type {kind!r}")


class PortableEmulator:
    """A GP+PCA emulator reconstructed from a portable HDF5 file.

    Duck-types the subset of :class:`GeneralEmulatorGP` that the inference
    pipeline touches. ``_alpha_vectors`` is a real attribute (the loaded
    ``alpha`` array) so the runner's save/clear/restore of cached state works
    unchanged; ``precompute_alpha`` is a no-op because the file already stores
    the solved system.
    """

    def __init__(self, path):
        with h5py.File(path, "r") as f:
            self.X_train = f["X_train"][:]                # (n_train, n_params), scaled
            self._alpha_vectors = f["alpha"][:]           # (n_comp, n_train)
            self.kernel_amplitude = f["kernel_amplitude"][:]
            self.kernel_metric = f["kernel_metric"][:]    # squared length scales
            self.gp_mean = f["gp_mean"][:]
            self.scaler_mean = f["scaler_mean"][:]
            self.scaler_scale = f["scaler_scale"][:]
            self.pca_components = f["pca_components"][:]
            self.pca_mean = f["pca_mean"][:]
            self.param_ranges = f["param_ranges"][:] if "param_ranges" in f else None
            self.axis_data = ({k: f["axes"][k][:] for k in f["axes"]}
                              if "axes" in f else {})
            self.kernel_type = str(f.attrs["kernel_type"])
            self.floor_value = float(f.attrs["floor_value"])
            self.output_shape = tuple(int(v) for v in f.attrs["output_shape"])
            self.param_names = [p.decode() if isinstance(p, bytes) else str(p)
                                for p in f.attrs["param_names"]]
            self.quantity = str(f.attrs.get("quantity", ""))
            self.provenance = (dict(f["provenance"].attrs)
                               if "provenance" in f else {})
        self.path = str(path)
        # len(emu.gps) is reported by the runner; one entry per PCA component.
        self.gps = [None] * self._alpha_vectors.shape[0]

    @classmethod
    def load(cls, path):
        return cls(path)

    def precompute_alpha(self):
        """No-op: the portable file already stores alpha = K^-1 (y - mean)."""
        return None

    def predict_mean_only(self, X_new):
        """Mean prediction in physical space, shape (n, *output_shape)."""
        if self._alpha_vectors is None:
            raise RuntimeError(
                "alpha vectors were cleared on this PortableEmulator; "
                "restore them before predicting.")
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        if X_new.shape[1] != len(self.param_names):
            raise ValueError(f"expected {len(self.param_names)} parameters "
                             f"({', '.join(self.param_names)}), got {X_new.shape[1]}")
        xs = (X_new - self.scaler_mean) / self.scaler_scale
        n_comp = self._alpha_vectors.shape[0]
        w = np.empty((xs.shape[0], n_comp))
        d = xs[:, None, :] - self.X_train[None, :, :]
        for i in range(n_comp):
            r2 = np.sum(d * d / self.kernel_metric[i], axis=-1)
            w[:, i] = (self.gp_mean[i]
                       + self.kernel_amplitude[i]
                       * matern(r2, self.kernel_type) @ self._alpha_vectors[i])
        flat = w @ self.pca_components + self.pca_mean
        return flat.reshape((-1, *self.output_shape)) + self.floor_value

    def predict(self, X_new):
        """(mean, std). The portable file carries no covariance: std is zero."""
        mu = self.predict_mean_only(X_new)
        return mu, np.zeros_like(mu)

    def __repr__(self):
        return (f"PortableEmulator({self.quantity!r}, "
                f"{self._alpha_vectors.shape[0]} components, "
                f"{self.X_train.shape[0]} training points)")
