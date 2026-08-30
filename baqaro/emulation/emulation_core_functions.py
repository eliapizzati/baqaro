"""
PCA + Gaussian Process Emulation
=================================

Two-stage emulation pipeline for high-dimensional astrophysical statistics:

1. **GeneralEmulatorPCA** — Dimensionality reduction via PCA.
   Accepts data of arbitrary shape (n_sims, dim1, dim2, ...), flattens for PCA,
   and reconstructs to the original shape. Data is linearly shifted by
   ``floor_value`` before PCA.

2. **GeneralEmulatorGP** — GP regression on PCA weights.
   Trains one independent GP (Matern 3/2 kernel via george) per PCA component.
   Provides predictions with uncertainty propagation back to physical space.

Typical usage::

    pca = GeneralEmulatorPCA(n_components=10, floor_value=-10.5)
    weights = pca.fit_transform(data)       # (n_sims, n_components)

    gp = GeneralEmulatorGP(pca_model=pca, param_names=names, ...)
    gp.fit(params, weights)                 # train GPs
    mu, std = gp.predict(new_params)        # predict in physical space
"""

import os
import numpy as np
import lzma
import pickle

import george
from george import kernels
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class GeneralEmulatorPCA:
    """
    PCA-based dimensionality reduction for emulation training data.

    Handles data of any shape (n_sims, dim1, dim2, ...) by flattening
    to (n_sims, n_features) for PCA. Before PCA, data is shifted so that
    floor_value maps to 0, ensuring PCA captures variation above the floor.

    Parameters
    ----------
    n_components : int
        Number of PCA components to retain.
    floor_value : float
        Value representing empty/unphysical bins in log-space data.
        Data is shifted by subtracting this value before PCA.

    The transform is a plain linear shift by ``floor_value``; a nonlinear
    output warp would require propagating the GP sigma through its Jacobian
    (``sigma_phys = |d(unwarp)/dz| * sigma_warped``) at the same time.
    """

    def __init__(self, n_components=10, floor_value=-10.0):
        self.n_components = n_components
        self.floor_value = floor_value

        self.pca = PCA(n_components=n_components)
        self.input_shape = None  # stored on first fit, e.g. (n_z, n_L, n_bins)
        self.mu_ = None          # PCA mean vector (set after fit)

    # --- Internal preprocessing pipeline (linear shift by floor_value) ---

    def _preprocess(self, data):
        """Clean -> shift -> flatten to (n_sims, n_features)."""
        data_clean = np.nan_to_num(data, nan=self.floor_value, neginf=self.floor_value)
        data_processed = data_clean - self.floor_value

        if self.input_shape is None:
            self.input_shape = data_processed.shape[1:]

        return data_processed.reshape(data_processed.shape[0], -1)

    def _postprocess(self, flat_data):
        """Reshape -> inverse shift back to physical space."""
        data_shaped = flat_data.reshape(flat_data.shape[0], *self.input_shape)
        return data_shaped + self.floor_value

    # --- Public API ---

    def fit(self, data):
        """Fit PCA on training data of any shape (n_sims, ...)."""
        self.input_shape = None
        X = self._preprocess(data)
        self.pca.fit(X)
        self.mu_ = self.pca.mean_
        var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA Training: {self.n_components} components explain {var*100:.2f}% of variance.")

    def transform(self, data):
        """Project data -> PCA weights. Shape: (n_sims, n_components)."""
        if self.mu_ is None:
            raise RuntimeError("PCA not fitted. Call fit() first.")
        return self.pca.transform(self._preprocess(data))

    def fit_transform(self, data):
        """Fit PCA and return weights in one step."""
        self.fit(data)
        return self.pca.transform(self._preprocess(data))

    def inverse_transform(self, weights):
        """Reconstruct physical data from PCA weights."""
        return self._postprocess(self.pca.inverse_transform(weights))

    def save(self, path):
        with lzma.open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with lzma.open(path, 'rb') as f:
            return pickle.load(f)


class GeneralEmulatorGP:
    """
    Gaussian Process emulator that operates on PCA weights.

    Trains one independent GP per PCA component using george.
    Input parameters are standardised to zero mean / unit variance
    before GP training.

    Parameters
    ----------
    pca_model : GeneralEmulatorPCA
        Fitted PCA model for forward/inverse transforms.
    param_names : list of str
        Physical parameter names, e.g. ["log_eta_mean_0", "std_0", ...].
    param_ranges : array-like, shape (n_params, 2)
        Prior bounds for each parameter.
    axis_data : dict
        Grid axes for the emulated statistic, e.g.
        ``{"redshift": z_arr, "log_L_threshold": L_arr, "log_bins": bins}``.
    kernel_type : str
        Kernel family: "matern32" (default, once differentiable),
        "matern52" (twice differentiable, smoother), or
        "matern12" (not differentiable, roughest — can track sharp transitions).
    verbose : bool
        Print progress during training.
    """

    KERNEL_MAP = {
        "matern12": kernels.ExpKernel,
        "matern32": kernels.Matern32Kernel,
        "matern52": kernels.Matern52Kernel,
    }

    def __init__(self, pca_model, param_names=None, param_ranges=None,
                 axis_data=None, kernel_type="matern32", verbose=False):
        self.pca_model = pca_model
        self.verbose = verbose
        self.kernel_type = kernel_type

        self.param_names = param_names
        self.param_ranges = param_ranges
        self.axis_data = axis_data

        self.gps = []       # one george.GP per PCA component
        self.train_y = []   # training targets for each GP (needed for george.predict)
        self.scaler = None  # StandardScaler for input parameters

        # Precomputed alpha vectors for fast prediction (set by precompute_alpha)
        self._alpha_vectors = None  # list of arrays, one per GP component
        self._X_train_scaled = None  # scaled training inputs (shared by all GPs)

    # --- Kernel construction ---

    def _build_kernel(self, n_params, y):
        """
        Build a kernel with variance initialised from data.

        Parameters
        ----------
        n_params : int
            Dimensionality of input space.
        y : array, shape (n_sims,)
            Training targets for this component (used to set initial variance).
        """
        if self.kernel_type not in self.KERNEL_MAP:
            raise ValueError(f"Unknown kernel_type '{self.kernel_type}'. "
                             f"Choose from: {list(self.KERNEL_MAP.keys())}")
        kernel_cls = self.KERNEL_MAP[self.kernel_type]
        initial_metric = np.ones(n_params)
        return np.var(y) * kernel_cls(metric=initial_metric, ndim=n_params)

    def _get_bounds(self, gp):
        """
        Generate hyperparameter bounds for L-BFGS-B optimisation.

        Returns a list of (lo, hi) tuples, one per hyperparameter.
        Bounds are tightened for metric (length-scale) and white_noise
        parameters to prevent degenerate solutions.
        """
        vector = gp.get_parameter_vector()
        names = gp.get_parameter_names()

        bounds = [(-20.0, 20.0)] * len(vector)

        if len(names) == len(vector):
            for i, name in enumerate(names):
                if "metric" in name:
                    bounds[i] = (-1.6, 6.0)    # length scale in [0.45, 20]
                elif "white_noise" in name:
                    bounds[i] = (-9.5, -1.0)   # noise in [1e-4, 0.37]

        return bounds

    # --- GP hyperparameter optimisation ---

    def _optimize_gp(self, gp, y, n_restarts=5, seed=42):
        """
        Optimise GP hyperparameters with multiple random restarts.

        Runs L-BFGS-B from the current hyperparameters plus ``n_restarts``
        random starting points. Keeps the result with the highest
        log-likelihood.

        Parameters
        ----------
        gp : george.GP
            GP object (modified in-place).
        y : array
            Training targets.
        n_restarts : int
            Number of additional random starting points beyond the default.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        best_p : array
            Best hyperparameter vector (already set on gp).
        """
        bounds = self._get_bounds(gp)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(y)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y)

        best_nll = np.inf
        best_p = gp.get_parameter_vector().copy()

        rng = np.random.default_rng(seed)
        bounds_arr = np.array(bounds)
        starts = [best_p.copy()]
        for _ in range(n_restarts):
            starts.append(rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1]))

        for p0 in starts:
            try:
                gp.set_parameter_vector(p0)
                res = minimize(neg_ln_like, p0, jac=grad_neg_ln_like,
                               method="L-BFGS-B", bounds=bounds)
                if res.fun < best_nll:
                    best_nll = res.fun
                    best_p = res.x.copy()
            except Exception:
                continue

        gp.set_parameter_vector(best_p)
        return best_p

    # --- Training ---

    def fit(self, X, Y_weights, y_err=None, normalize_params=True):
        """
        Train one GP per PCA component.

        Parameters
        ----------
        X : array, shape (n_sims, n_params)
            Input parameters.
        Y_weights : array, shape (n_sims, n_components)
            PCA weights from ``pca_model.fit_transform()``.
        y_err : float or None
            Observational error on weights (passed to ``gp.compute``).
        normalize_params : bool
            If True, standardise X to zero mean / unit variance.
        """
        if normalize_params:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            self.scaler = None
            X_scaled = X

        n_params = X_scaled.shape[1]
        n_components = Y_weights.shape[1]

        # n_restarts = extra random L-BFGS-B starts beyond the kernel-init start.
        # Each start is a full optimization, and at large N (e.g. 3483 z0 points)
        # one fun+grad eval is ~2.5s, so total fit time scales ~linearly with
        # (1 + n_restarts). Default 3 (4 starts) preserves the fid1 methodology;
        # set BAQARO_GP_N_RESTARTS=1 to roughly halve fit time for big z0 sets.
        n_restarts = int(os.environ.get("BAQARO_GP_N_RESTARTS", "3"))

        if self.verbose:
            print(f"Training {n_components} GPs... (n_restarts={n_restarts})")

        self.gps = []
        self.train_y = []

        for i in range(n_components):
            y = Y_weights[:, i]

            kernel = self._build_kernel(n_params, y)
            gp = george.GP(kernel, mean=np.mean(y), fit_mean=True,
                           white_noise=np.log(1e-3), fit_white_noise=True)
            gp.compute(X_scaled, yerr=y_err if y_err is not None else 1e-3)

            self._optimize_gp(gp, y, n_restarts=n_restarts, seed=42 + i)

            self.gps.append(gp)
            self.train_y.append(y)

        if self.verbose:
            print("GP Training Complete.")

    # --- Prediction ---

    def predict(self, X_new):
        """
        Predict with uncertainties in physical space.

        Parameters
        ----------
        X_new : array, shape (n_samples, n_params)
            New parameter points.

        Returns
        -------
        mu : array, shape (n_samples, dim1, dim2, ...)
            Mean prediction in physical space.
        std : array, shape (n_samples, dim1, dim2, ...)
            Standard deviation from GP variance propagated through PCA.
        """
        if not self.gps:
            raise RuntimeError("GP not trained.")

        X_new = np.atleast_2d(X_new)
        if self.scaler:
            X_new = self.scaler.transform(X_new)

        n_samples = X_new.shape[0]
        n_components = len(self.gps)

        pred_weights = np.zeros((n_samples, n_components))
        pred_vars = np.zeros((n_samples, n_components))

        for i, gp in enumerate(self.gps):
            mu, var = gp.predict(self.train_y[i], X_new, return_var=True)
            pred_weights[:, i] = mu
            pred_vars[:, i] = var

        mu_recon = self.pca_model.inverse_transform(pred_weights)

        # Propagate variance: Var_feature = sum_i(var_weight_i * eigenvector_i^2)
        eigenvectors = self.pca_model.pca.components_
        var_flat = np.dot(pred_vars, eigenvectors**2)
        std_recon = np.sqrt(var_flat).reshape(n_samples, *self.pca_model.input_shape)

        # The transform from PCA-weight space back to physical space is now purely
        # LINEAR (a shift by floor_value),
        # so propagating the variance through the eigenvectors as above is exact.
        # If a nonlinear output transform is ever reintroduced, sigma must be pushed
        # through its Jacobian here as well.
        #
        # KNOWN LIMITATION (dormant): mu_recon is not clamped to pca_floor, so a
        # linear reconstruction can dip below the training floor. Matches
        # predict_mean_only's unclamped behaviour; clamp both together if it matters.
        # (This variance path feeds only plotting — the MCMC uses predict_mean_only.)
        return mu_recon, std_recon

    def precompute_alpha(self):
        """
        Pre-extract GP alpha vectors for fast prediction.

        Computes alpha_i = K_train^{-1} @ (y_i - mean_i) once for each PCA
        component.  Subsequent calls to ``predict_mean_only`` use these cached
        vectors with a simple K_star @ alpha matrix multiply, avoiding the
        repeated Cholesky solve inside ``george.GP.predict()``.

        Call this once after loading a trained emulator (no retraining needed).
        """
        if not self.gps:
            raise RuntimeError("GP not trained.")

        # Store the scaled training inputs (needed for kernel evaluation)
        # We reconstruct from the first GP's _x attribute if available,
        # or from the stored training data
        self._alpha_vectors = []
        for i, gp in enumerate(self.gps):
            # george.predict() calls recompute() which re-factorizes the
            # kernel if hyperparameters changed since the last compute().
            # We must do the same to ensure the solver is up to date.
            gp.recompute()
            # Use george's internal _compute_alpha which handles all the
            # details (mean subtraction, caching, contiguous arrays).
            alpha = gp._compute_alpha(self.train_y[i], cache=False)
            self._alpha_vectors.append(alpha)

        # Store X_train_scaled for kernel evaluation
        self._X_train_scaled = self.gps[0]._x.copy()

    def predict_mean_only(self, X_new):
        """
        Fast mean-only prediction (no variance). Used in MCMC likelihood calls.

        If ``precompute_alpha()`` has been called, uses cached alpha vectors
        with direct kernel evaluation (K_star @ alpha), bypassing george's
        internal solver.  Otherwise falls back to ``gp.predict()``.
        """
        if not self.gps:
            raise RuntimeError("GP not trained.")

        X_new = np.atleast_2d(X_new)
        if self.scaler:
            X_new = self.scaler.transform(X_new)

        n_new = X_new.shape[0]
        n_components = len(self.gps)
        pred_weights = np.zeros((n_new, n_components))

        if self._alpha_vectors is not None:
            # Fast path: precomputed alpha vectors
            for i, gp in enumerate(self.gps):
                K_star = gp.kernel.get_value(X_new, self._X_train_scaled)
                mean_new = gp.mean.get_value(X_new)
                pred_weights[:, i] = mean_new + K_star @ self._alpha_vectors[i]
        else:
            # Fallback: standard george.predict
            for i, gp in enumerate(self.gps):
                pred_weights[:, i] = gp.predict(self.train_y[i], X_new,
                                                return_var=False, return_cov=False)

        return self.pca_model.inverse_transform(pred_weights)

    # --- Cross-validation ---

    def cross_validate_physical(self, X, Y_weights, Y_physical=None, k_folds=5,
                                y_err=None, normalize_params=True, verbose=True,
                                seed=42, real_threshold=None):
        """
        K-fold cross-validation with error metrics in physical space.

        For each fold, trains GPs on the training split, predicts the held-out
        split, reconstructs via PCA, and computes error statistics.

        Parameters
        ----------
        X : array, shape (n_sims, n_params)
            Input parameters.
        Y_weights : array, shape (n_sims, n_components)
            PCA weights.
        Y_physical : array, shape (n_sims, ...), optional
            Original physical-space data for computing errors in dex.
        k_folds : int
            Number of CV folds.
        y_err : float or None
            Observational error on weights.
        normalize_params : bool
            Standardise input parameters.
        verbose : bool
            Print progress.
        seed : int
            RNG seed for fold assignment.
        real_threshold : float or None
            If set, clamp true and predicted values at this threshold before
            computing errors. Prevents penalising the emulator for differences
            in the floored (unphysical) region.

        Returns
        -------
        results : dict
            Keys: 'rmse_weights', and if Y_physical is provided:
            'rmse_physical', 'median_ae_physical', 'max_error_physical',
            'predictions_physical' (tuple of true, pred arrays).
        """
        n_sims = X.shape[0]
        n_components = Y_weights.shape[1]

        # The PCA basis is fit once on the full set; the folds hold out only
        # the GP regression.
        if n_sims < k_folds:
            raise ValueError(f"Cannot perform {k_folds}-fold CV with only {n_sims} samples")

        # Deterministic fold assignment
        rng = np.random.default_rng(seed)
        indices = np.arange(n_sims)
        rng.shuffle(indices)

        fold_sizes = np.full(k_folds, n_sims // k_folds, dtype=int)
        fold_sizes[:n_sims % k_folds] += 1
        fold_indices = np.split(indices, np.cumsum(fold_sizes)[:-1])

        all_y_weight_true, all_y_weight_pred = [], []
        all_y_phys_true, all_y_phys_pred = [], []

        if verbose:
            print(f"\nPerforming {k_folds}-fold cross-validation...")

        for fold in range(k_folds):
            if verbose:
                print(f"  Fold {fold + 1}/{k_folds}...")

            test_idx = fold_indices[fold]
            train_idx = np.concatenate([fold_indices[i] for i in range(k_folds) if i != fold])

            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y_weights[train_idx], Y_weights[test_idx]

            if normalize_params:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled, X_test_scaled = X_train, X_test

            # Train GPs and predict held-out weights
            fold_weights_pred = np.zeros_like(Y_test)

            for i in range(n_components):
                y_tr = Y_train[:, i]

                kernel = self._build_kernel(X_train_scaled.shape[1], y_tr)
                gp = george.GP(kernel, mean=np.mean(y_tr), fit_mean=True,
                               white_noise=np.log(1e-3), fit_white_noise=True)
                gp.compute(X_train_scaled, yerr=y_err if y_err is not None else 1e-3)

                # Restarts for CV (k_folds x n_components GPs total). Honors
                # BAQARO_GP_N_RESTARTS like fit() so CV matches the production fit
                # config; nr=0 gives the same accuracy in our tests
                # and ~halves CV wall-time. Default 1 preserves prior behavior.
                _cv_nr = int(os.environ.get("BAQARO_GP_N_RESTARTS", "1"))
                self._optimize_gp(gp, y_tr, n_restarts=_cv_nr,
                                  seed=42 + fold * n_components + i)

                # Honor the caller's y_err here too (matches the pre-optimize
                # compute above); the hardcoded 1e-3 ignored it. Dormant while
                # y_err is None everywhere, but the two computes must agree.
                gp.compute(X_train_scaled, yerr=y_err if y_err is not None else 1e-3)
                fold_weights_pred[:, i] = gp.predict(y_tr, X_test_scaled,
                                                     return_var=False, return_cov=False)

            all_y_weight_true.append(Y_test)
            all_y_weight_pred.append(fold_weights_pred)

            if Y_physical is not None:
                all_y_phys_true.append(Y_physical[test_idx])
                all_y_phys_pred.append(self.pca_model.inverse_transform(fold_weights_pred))

        # --- Aggregate results ---
        all_y_weight_true = np.vstack(all_y_weight_true)
        all_y_weight_pred = np.vstack(all_y_weight_pred)

        rmse_weights = np.sqrt(np.mean((all_y_weight_true - all_y_weight_pred)**2, axis=0))
        results = {'rmse_weights': rmse_weights}

        if Y_physical is not None:
            all_y_phys_true = np.concatenate(all_y_phys_true, axis=0)
            all_y_phys_pred = np.concatenate(all_y_phys_pred, axis=0)

            if real_threshold is not None:
                y_true_clamped = np.maximum(all_y_phys_true, real_threshold)
                y_pred_clamped = np.maximum(all_y_phys_pred, real_threshold)
                residuals = y_true_clamped - y_pred_clamped
                if verbose:
                    print(f"  (Clamping at real_threshold={real_threshold} for error metrics)")
            else:
                residuals = all_y_phys_true - all_y_phys_pred

            global_rmse = np.sqrt(np.mean(residuals**2))
            median_ae = np.median(np.abs(residuals))
            max_error = np.max(np.abs(residuals))

            results['rmse_physical'] = global_rmse
            results['median_ae_physical'] = median_ae
            results['max_error_physical'] = max_error
            results['predictions_physical'] = (all_y_phys_true, all_y_phys_pred)

            if verbose:
                print("=" * 60)
                print("PHYSICAL SPACE VALIDATION (Real Units)")
                print(f"  Global RMSE:        {global_rmse:.4f} dex")
                print(f"  Median Abs Error:   {median_ae:.4f} dex")
                print(f"  Max Outlier:        {max_error:.4f} dex")
                print("=" * 60)

        return results

    # --- Serialisation ---

    def save(self, path):
        with lzma.open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with lzma.open(path, 'rb') as f:
            return pickle.load(f)
