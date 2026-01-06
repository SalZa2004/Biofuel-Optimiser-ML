"""
Test the trained BP prediction model on a held-out test set.
Saves full evaluation results under model_performance/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from core.shared_features import FeatureSelector
from core.predictors.pure_component.generic import GenericPredictor

# ============================================================
# PATH SETUP
# ============================================================

# Location of this file
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root (one level up)
PROJECT_ROOT = os.path.dirname(TEST_DIR)
sys.path.append(PROJECT_ROOT)

# Performance output directory
PERFORMANCE_DIR = os.path.join(PROJECT_ROOT, "model_performance")
os.makedirs(PERFORMANCE_DIR, exist_ok=True)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def save_metrics_txt(metrics: dict, title: str, filename: str):
    """Save metrics dictionary to a .txt file."""
    path = os.path.join(PERFORMANCE_DIR, filename)
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{title}\n")
        f.write("=" * 70 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"✓ Metrics saved to {path}")


# ============================================================
# DATA LOADING
# ============================================================

def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv("bp_data.csv")

    # Basic cleaning
    df.dropna(subset=["bp", "SMILES"], inplace=True)

    print(f"Total samples: {len(df)}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples:  {len(test_df)}")

    return train_df, test_df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_training_set(predictor, train_df):
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TRAINING SET")
    print("=" * 70)

    predictions = predictor.predict(train_df["SMILES"].tolist())

    if len(predictions) != len(train_df):
        results = predictor.predict_with_details(train_df["SMILES"].tolist())
        results = results.merge(train_df[["SMILES", "bp"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]

        y_true = results["bp"].values
        y_pred = results["Predicted_BP"].values
    else:
        y_true = train_df["bp"].values
        y_pred = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def evaluate_test_set(predictor, test_df):
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 70)

    predictions = predictor.predict(test_df["SMILES"].tolist())

    if len(predictions) != len(test_df):
        results = predictor.predict_with_details(test_df["SMILES"].tolist())
        results = results.merge(test_df[["SMILES", "bp"]], on="SMILES", how="left")
        results = results[results["Valid"] == True]

        y_true = results["bp"].values
        y_pred = results["Predicted_BP"].values
    else:
        y_true = test_df["bp"].values
        y_pred = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return y_true, y_pred, metrics


# ============================================================
# PLOTTING
# ============================================================

def plot_results(y_true, y_pred):
    save_path = os.path.join(PERFORMANCE_DIR, "evaluation_plots.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Predicted vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].plot([min_v, max_v], [min_v, max_v], "r--")
    axes[0].set_xlabel("Actual BP")
    axes[0].set_ylabel("Predicted BP")
    axes[0].set_title("Predicted vs Actual")

    # Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_title("Residuals")

    # Error distribution
    axes[2].hist(residuals, bins=30, edgecolor="black")
    axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set_title("Error Distribution")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✓ Plots saved to {save_path}")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(test_df, y_true, y_pred):
    save_path = os.path.join(PERFORMANCE_DIR, "test_predictions.csv")

    df = test_df.reset_index(drop=True)
    df["Actual_BP"] = y_true
    df["Predicted_BP"] = y_pred
    df["Absolute_Error"] = np.abs(y_true - y_pred)
    df["Relative_Error_%"] = np.abs((y_true - y_pred) / y_true) * 100

    df.to_csv(save_path, index=False)
    print(f"✓ Predictions saved to {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BP MODEL EVALUATION")
    print("=" * 70)

    predictor = GenericPredictor()

    train_df, test_df = load_and_split_data()

    train_metrics = evaluate_training_set(predictor, train_df)
    save_metrics_txt(train_metrics, "TRAINING SET PERFORMANCE", "training_metrics.txt")

    y_true, y_pred, test_metrics = evaluate_test_set(predictor, test_df)
    save_metrics_txt(test_metrics, "TEST SET PERFORMANCE", "test_metrics.txt")

    # Overfitting check
    overfit_path = os.path.join(PERFORMANCE_DIR, "overfitting_check.txt")
    with open(overfit_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("OVERFITTING CHECK\n")
        f.write("=" * 70 + "\n")
        f.write(f"Train RMSE: {train_metrics['RMSE']:.4f}\n")
        f.write(f"Test RMSE:  {test_metrics['RMSE']:.4f}\n")
        f.write(f"Difference: {test_metrics['RMSE'] - train_metrics['RMSE']:.4f}\n")

        if test_metrics["RMSE"] > train_metrics["RMSE"] * 1.2:
            f.write("⚠ Possible overfitting detected\n")
        else:
            f.write("✓ Model generalization looks good\n")

    print(f"✓ Overfitting report saved to {overfit_path}")

    plot_results(y_true, y_pred)
    save_predictions(test_df, y_true, y_pred)

    print("\n✓ Evaluation complete!")
    print(f"Results saved in: {PERFORMANCE_DIR}")


if __name__ == "__main__":
    main()
