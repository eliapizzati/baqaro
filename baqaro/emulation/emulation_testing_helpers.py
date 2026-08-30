"""Diagnostics for when an emulator fits badly.

Interactive helpers for the failure modes that actually occur: parameters that
were not scaled compatibly across the training set, a single bad simulation
poisoning the fit, and cross-validation errors concentrated somewhere specific
rather than spread evenly.

These are for use at a prompt while investigating -- they print and plot.
:mod:`cross_validate_emulators` is the batch equivalent that produces the
quantitative accuracy numbers.
"""

import numpy as np
import matplotlib.pyplot as plt


def check_parameter_scaling(params, param_names=None):
    """
    Checks if parameters cover huge dynamic ranges (e.g., 1e-5 to 1e2)
    which confuses the GP.
    """
    print("\n--- DIAGNOSTIC 0: PARAMETER SCALING ---")
    n_sims, n_params = params.shape
    
    # Sparsity check
    print(f"\n  Training samples: {n_sims}")
    print(f"  Parameter dimensions: {n_params}")
    samples_per_dim = n_sims ** (1.0 / n_params)
    print(f"  Effective samples per dimension: {samples_per_dim:.1f}")
    
    if samples_per_dim < 3:
        print(f"  ⚠️  WARNING: Very sparse! Need ~{3**n_params:.0f}+ samples for good coverage.")
        print(f"     With {n_sims} samples in {n_params}D, expect ~0.2-0.4 dex RMSE minimum.")
    elif samples_per_dim < 5:
        print(f"  ⚠️  Moderately sparse. Consider adding more training data.")
    else:
        print(f"  ✅ Reasonable sample density.")
    
    print()
    for i in range(n_params):
        p_min, p_max = np.min(params[:, i]), np.max(params[:, i])
        ratio = abs(p_max / (p_min + 1e-20))
        
        name = param_names[i] if param_names else f"Param {i}"
        
        print(f"  {name:<15}: Range [{p_min:.3e}, {p_max:.3e}]", end="")
        
        if ratio > 1000 and p_min > 0:
            print(f"  <- ⚠️  WARNING: spans {np.log10(ratio):.1f} orders of magnitude. Use LOG10 scale?")
        else:
            print("  (OK)")

def diagnose_emulator_failure(pca_model, gp_model, params, data_true, n_check=5):
    """
    Diagnose if the bottleneck is the PCA (shapes) or the GP (mapping).
    """
    print("\n--- DIAGNOSTIC 1: PCA RECONSTRUCTION (The Shape Limit) ---")
    
    # 1. Transform to weights and back (Theoretical Best Case)
    weights_perfect = pca_model.transform(data_true)
    data_recon_pca = pca_model.inverse_transform(weights_perfect)
    
    # Calculate Error in Dex (assuming data is log-space)
    resid_pca = data_true - data_recon_pca
    rmse_pca = np.sqrt(np.mean(resid_pca**2))
    max_err_pca = np.max(np.abs(resid_pca))
    
    print(f"  Global RMSE (PCA only): {rmse_pca:.4f} dex")
    print(f"  Max Outlier (PCA only): {max_err_pca:.4f} dex")
    
    if rmse_pca > 0.15:
        print("\n  ❌ CRITICAL FAILURE IN PCA:")
        print("     The emulator CANNOT describe your data shapes, even with perfect prediction.")
        print("     Fixes:")
        print("     1. Increase n_components.")
        print("     2. Check for 'Cliffs' (e.g., hard floor at -10.0).")
        return 
    else:
        print("\n  ✅ PCA is healthy. It can describe the shapes accurately.")

    print("\n--- DIAGNOSTIC 2: GP MAPPING (The Physics Link) ---")
    
    print(f"  Checking R² for the first {n_check} components (Signal vs Noise):")
    
    # Scale parameters same as training
    params_scaled = gp_model.scaler.transform(params)
    
    for i in range(min(n_check, weights_perfect.shape[1])):
        gp = gp_model.gps[i]
        y_true = weights_perfect[:, i]
        
        # --- FIX: GET TRAINING DATA FROM THE WRAPPER LIST ---
        y_train_for_comp = gp_model.train_y[i] 
        
        # Predict on training data
        y_pred, _ = gp.predict(y_train_for_comp, params_scaled, return_var=True)
        
        # Calculate R2
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        status = "✅" if r2 > 0.8 else "⚠️" if r2 > 0.5 else "❌"
        print(f"  Component {i+1}: R² = {r2:.4f} {status}")

    if r2 < 0.6:
         print("\n  ❌ CRITICAL FAILURE IN GP MAPPING:")
         print("     The GP cannot connect your parameters to the PCA weights.")
         print("     Likely Causes:")
         print("     1. Parameters need log-scaling (e.g. f_cold is 1e-5...1e0).")
         print("     2. Discontinuous physics (bimodality).")
         print("     3. Simulation noise is larger than the signal.")

    # --- NEW: Print learned hyperparameters ---
    print("\n--- DIAGNOSTIC 3: LEARNED HYPERPARAMETERS ---")
    for i, gp in enumerate(gp_model.gps[:3]):
        names = gp.get_parameter_names()
        values = gp.get_parameter_vector()
        print(f"  Component {i+1}:")
        for n, v in zip(names, values):
            if "metric" in n:
                length_scale = np.sqrt(np.exp(v))
                print(f"    {n}: log={v:.2f} → length_scale={length_scale:.3f}")
            elif "white_noise" in n:
                noise_var = np.exp(v)
                print(f"    {n}: log={v:.2f} → noise_frac={noise_var:.3f}")
            else:
                print(f"    {n}: {v:.3f}")




def identify_bad_simulation(emu_gp, params, data_true, floor_value=-9.9):
    """
    Hunts for the single worst prediction error in the entire dataset.
    Pinpoints the exact Simulation, Redshift, and Threshold causing the failure.
    """
    print("\n🔍 HUNTING FOR THE CATASTROPHIC FAILURE...")
    
    # 1. Get predictions for ALL points
    params_scaled = emu_gp.scaler.transform(params)
    weights_pred_matrix = np.zeros((params.shape[0], len(emu_gp.gps)))
    
    for i, gp in enumerate(emu_gp.gps):
        mu, _ = gp.predict(emu_gp.train_y[i], params_scaled, return_var=True)
        weights_pred_matrix[:, i] = mu
        
    # 2. Transform weights -> Physical Data
    data_pred = emu_gp.pca_model.inverse_transform(weights_pred_matrix)
    
    # 3. Calculate Residuals
    residuals = np.abs(data_true - data_pred)
    
    # --- CRITICAL: IGNORE "FLOOR-TO-FLOOR" NOISE ---
    # If both Truth and Pred are effectively at the floor, error is 0.
    # We DO care if Truth is Floor (-10) and Pred is Signal (-5).
    # We DO care if Truth is Signal (-5) and Pred is Floor (-10).
    mask_both_floor = (data_true <= floor_value) & (data_pred <= floor_value)
    residuals[mask_both_floor] = 0.0
    
    # 4. Find the coordinates of the SINGLE WORST PIXEL
    # unravel_index turns the flat index into (Sim, Z, Bin, Thresh)
    max_error_val = np.max(residuals)
    worst_idx = np.unravel_index(np.argmax(residuals), residuals.shape)
    
    sim_idx = worst_idx[0]
    z_idx = worst_idx[1]
    # bin_idx is where the spike actually happened
    
    # Handle 3D vs 4D data structures automatically
    if len(worst_idx) == 4:
        thresh_idx = worst_idx[3]
        # Slice the curve: [Sim, Z, :, Thresh]
        y_true_curve = data_true[sim_idx, z_idx, :, thresh_idx]
        y_pred_curve = data_pred[sim_idx, z_idx, :, thresh_idx]
        title_str = f"WORST FAILURE: Sim #{sim_idx} | z-idx {z_idx} | Thresh-idx {thresh_idx}"
    else:
        # 3D Data: [Sim, Z, :]
        thresh_idx = None
        y_true_curve = data_true[sim_idx, z_idx, :]
        y_pred_curve = data_pred[sim_idx, z_idx, :]
        title_str = f"WORST FAILURE: Sim #{sim_idx} | z-idx {z_idx}"

    print(f"\n🚨 FOUND CULPRIT:")
    print(f"   Simulation Index : {sim_idx}")
    print(f"   Max Error Found  : {max_error_val:.4f} dex")
    print(f"   Parameter Values : {params[sim_idx]}")
    
    # Check if this is the "Cliff/Overshoot" issue
    # (Truth is empty, Pred is ~ -5.0)
    bin_idx = worst_idx[2]
    val_true = y_true_curve[bin_idx]
    val_pred = y_pred_curve[bin_idx]
    
    print(f"   AT BIN {bin_idx}: Truth={val_true:.2f}, Pred={val_pred:.2f}")
    
    if val_true <= floor_value and val_pred > floor_value + 1.0:
        print("   -> DIAGNOSIS: OVERSHOOT (Hallucination). Emulator missed the cliff.")
    elif val_true > floor_value + 1.0 and val_pred <= floor_value:
        print("   -> DIAGNOSIS: DROPOUT. Emulator cut off too early.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_true_curve, 'k-', lw=2, label='True Simulation')
    plt.plot(y_pred_curve, 'r--', lw=2, label='Emulator Prediction')
    
    # Highlight the failure point
    plt.scatter(bin_idx, val_true, c='blue', s=100, zorder=10, label='Truth (Point)')
    plt.scatter(bin_idx, val_pred, c='red', s=100, zorder=10, marker='x', label='Pred (Point)')
    
    plt.axhline(floor_value, color='gray', linestyle=':', label='Floor Limit')
    
    plt.title(title_str)
    plt.xlabel("Bin Index")
    plt.ylabel("log10(Phi) [dex]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def hunt_cv_failure(cv_results, floor_value=-9.9, real_threshold=None):
    """
    Plots the single worst failure from the Cross-Validation (Test Set).
    This reveals the true generalization errors (like the 5.0 dex cliff miss).

    If real_threshold is set, both true and predicted values are clamped at
    that threshold before computing residuals, so errors in the extrapolated
    region (below the physical floor) don't dominate.
    """
    print("\n🔍 HUNTING FOR CV TEST FAILURE...")

    # 1. Unpack the CV predictions (Test Set Data)
    # These are the concatenated arrays of every simulation used as a test case
    y_true_all, y_pred_all = cv_results['predictions_physical']

    # 2. Calculate residuals (clamp at real_threshold if set)
    if real_threshold is not None:
        y_true_clamped = np.maximum(y_true_all, real_threshold)
        y_pred_clamped = np.maximum(y_pred_all, real_threshold)
        residuals = np.abs(y_true_clamped - y_pred_clamped)
    else:
        residuals = np.abs(y_true_all - y_pred_all)

    # 3. Mask floor-to-floor errors (Ignore "Ringing" in empty space)
    # We only care if one is Signal and the other is Floor
    mask_both_floor = (y_true_all <= floor_value) & (y_pred_all <= floor_value)
    residuals[mask_both_floor] = 0.0
    
    # 4. Find the single worst point
    max_error_val = np.max(residuals)
    worst_idx = np.unravel_index(np.argmax(residuals), residuals.shape)
    
    sim_idx = worst_idx[0]
    z_idx = worst_idx[1]
    
    # Handle dimensions (4D vs 3D data)
    if len(worst_idx) == 4:
        thresh_idx = worst_idx[3]
        # Slice the specific curve
        y_true = y_true_all[sim_idx, z_idx, :, thresh_idx]
        y_pred = y_pred_all[sim_idx, z_idx, :, thresh_idx]
        title = f"CV FAILURE: Sim {sim_idx} (Test Set) | z={z_idx} | Thresh={thresh_idx}"
    else:
        thresh_idx = None
        y_true = y_true_all[sim_idx, z_idx, :]
        y_pred = y_pred_all[sim_idx, z_idx, :]
        title = f"CV FAILURE: Sim {sim_idx} (Test Set) | z={z_idx}"
        
    print(f"🚨 FOUND WORST CV FAILURE:")
    print(f"   Max Error: {max_error_val:.4f} dex")
    
    # 5. Diagnose the Physics
    bin_idx = worst_idx[2]
    val_true = y_true[bin_idx]
    val_pred = y_pred[bin_idx]
    
    print(f"   At Bin {bin_idx}: Truth={val_true:.2f}, Pred={val_pred:.2f}")
    
    if val_true <= floor_value and val_pred > floor_value + 2.0:
        print("   -> DIAGNOSIS: OVERSHOOT (Hallucination). The emulator missed the cliff.")
        print("      It predicted signal where there was none.")
    elif val_true > floor_value + 2.0 and val_pred <= floor_value:
        print("   -> DIAGNOSIS: DROPOUT. The emulator cut off too early.")
    
    # 6. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, 'k-', lw=2, label='True Sim (Test Set)')
    plt.plot(y_pred, 'r--', lw=2, label='Emulator Pred')
    
    # Highlight the specific failure bin
    plt.scatter(bin_idx, val_true, c='blue', s=100, zorder=5, label='Truth')
    plt.scatter(bin_idx, val_pred, c='red', s=100, zorder=5, marker='x', label='Pred')
    
    plt.axhline(floor_value, color='gray', ls=':', label='Floor')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.show()

