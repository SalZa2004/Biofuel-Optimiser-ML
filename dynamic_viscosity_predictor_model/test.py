"""
Evaluate the trained dynamic viscosity model on a held-out test set.
Model was trained on log10(dynamic_viscosity).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from train import DynamicViscosityPredictor, FeatureSelector, featurize_df

# =============================================================================
# CONFIG
# =============================================================================

TEST_SIZE = 0.2
RANDOM_STATE = 42
CSV_PATH = "dynamic_viscosity.csv"
PLOT_PATH = "dynamic_viscosity_model/evaluation_plots.png"
PRED_PATH = "dynamic_viscosity_model/test_predictions.csv"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv(CSV_PATH)
    df.dropna(subset=["dynamic_viscosity", "SMILES"], inplace=True)

    print(f"Total samples: {len(df)}")
    print(f"Viscosity range: [{df.dynamic_viscosity.min():.3f}, "
          f"{df.dynamic_viscosity.max():.3f}]")

    # Stratify by log-viscosity to avoid distribution mismatch
    df["log_visc"] = np.log10(df["dynamic_viscosity"])
    df["log_bin"] = pd.qcut(df["log_visc"], q=5, duplicates="drop")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["log_bin"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(predictor, df, label="TEST"):
    print("\n" + "=" * 70)
    print(f"EVALUATING {label} SET")
    print("=" * 70)

    y_true = df["dynamic_viscosity"].values
    smiles = df["SMILES"].tolist()

    y_pred = np.array(predictor.predict(smiles))

    # Safety
    y_pred = np.clip(y_pred, 1e-6, None)

    # --- LOG SPACE (correctness) ---
    y_true_log = np.log10(y_true)
    y_pred_log = np.log10(y_pred)

    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2_log = r2_score(y_true_log, y_pred_log)

    # --- LINEAR SPACE (reported units) ---
    rmse_lin = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_lin = mean_absolute_error(y_true, y_pred)

    # Multiplicative error (physically meaningful)
    mult_error = np.maximum(y_pred / y_true, y_true / y_pred)
    median_mult = np.median(mult_error)

    print(f"\n{label} RESULTS — LOG SPACE (MODEL QUALITY)")
    print(f"RMSE (log10): {rmse_log:.4f}")
    print(f"MAE  (log10): {mae_log:.4f}")
    print(f"R²   (log10): {r2_log:.4f}")

    print(f"\n{label} RESULTS — LINEAR SPACE (REAL UNITS)")
    print(f"RMSE (viscosity units): {rmse_lin:.2f}")
    print(f"MAE  (viscosity units): {mae_lin:.2f}")
    print(f"Median multiplicative error: {median_mult:.2f}×")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_log": y_true_log,
        "y_pred_log": y_pred_log,
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_lin": rmse_lin,
        "mae_lin": mae_lin,
        "median_mult": median_mult
    }

# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(y_true_log, y_pred_log, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pred vs actual (log)
    ax = axes[0]
    ax.scatter(y_true_log, y_pred_log, alpha=0.5)
    minv = min(y_true_log.min(), y_pred_log.min())
    maxv = max(y_true_log.max(), y_pred_log.max())
    ax.plot([minv, maxv], [minv, maxv], "r--")
    ax.set_xlabel("Actual log10(viscosity)")
    ax.set_ylabel("Predicted log10(viscosity)")
    ax.set_title("Predicted vs Actual (log scale)")
    r2 = r2_score(y_true_log, y_pred_log)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(alpha=0.3)


    # Residuals
    ax = axes[1]
    residuals = y_true_log - y_pred_log
    ax.scatter(y_pred_log, residuals, alpha=0.5)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted log10(viscosity)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals (log space)")
    ax.grid(alpha=0.3)
    
    

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Plots saved to {save_path}")

# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(df, y_pred, save_path):
    out = df.copy()
    out["Predicted_dynamic_viscosity"] = y_pred
    out["Absolute_Error"] = np.abs(out.dynamic_viscosity - y_pred)
    out["Multiplicative_Error"] = np.maximum(
        y_pred / out.dynamic_viscosity,
        out.dynamic_viscosity / y_pred
    )

    out.to_csv(save_path, index=False)
    print(f"✓ Predictions saved to {save_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("DYNAMIC VISCOSITY MODEL EVALUATION")
    print("=" * 70)

    predictor = DynamicViscosityPredictor()

    train_df, test_df = load_and_split_data(TEST_SIZE, RANDOM_STATE)

    train_res = evaluate(predictor, train_df, label="TRAIN")
    test_res = evaluate(predictor, test_df, label="TEST")

    print("\nOVERFITTING CHECK")
    print(f"Train RMSE (log): {train_res['rmse_log']:.4f}")
    print(f"Test  RMSE (log): {test_res['rmse_log']:.4f}")

    plot_results(
        test_res["y_true_log"],
        test_res["y_pred_log"],
        PLOT_PATH
    )

    save_predictions(test_df, test_res["y_pred"], PRED_PATH)

    print("\n✓ EVALUATION COMPLETE")

if __name__ == "__main__":
    main()
