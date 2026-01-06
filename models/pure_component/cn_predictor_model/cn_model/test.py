"""
Generic model evaluation script for pure-component property predictors.
Uses the SAME inference pathway as production (featurize_df + predict_from_features).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from core.predictors.pure_component.generic import GenericPredictor
from core.shared_features import FeatureSelector, featurize_df


# ============================================================
# CONFIG
# ============================================================

PROPERTY_NAME = "CN"
TARGET_COLUMN = "Standardised_DCN"

MODEL_ROOT = Path(__file__).resolve().parent
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = MODEL_ROOT.parents[3]
PERFORMANCE_DIR = MODEL_ROOT / "model_performance"
PERFORMANCE_DIR.mkdir(exist_ok=True)

DB_PATH = PROJECT_ROOT / "data" / "database" / "database_main.db"


# ============================================================
# DATA
# ============================================================

def load_data():
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT
        F.Fuel_Name,
        F.SMILES,
        T.{TARGET_COLUMN} AS y
    FROM FUEL F
    JOIN TARGET T ON F.fuel_id = T.fuel_id
    WHERE T.{TARGET_COLUMN} IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df.dropna(subset=["SMILES", "y"], inplace=True)
    return df


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}"
    )

    mask = y_true != 0

    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": np.mean(
            np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        ) * 100
    }



# ============================================================
# PLOTTING
# ============================================================

def plot_results(y_true, y_pred):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_title("Residuals")

    axes[2].hist(residuals, bins=30, edgecolor="black")
    axes[2].axvline(0, color="r", linestyle="--")
    axes[2].set_title("Error Distribution")

    plt.tight_layout()
    plt.savefig(PERFORMANCE_DIR / "evaluation_plots.png", dpi=300)
    plt.close()


# ============================================================
# SAVE
# ============================================================

def save_outputs(df, y_pred, metrics):
    out = df.copy()
    out["Predicted"] = y_pred
    out["Absolute_Error"] = np.abs(df["y"] - y_pred)

    out.to_csv(PERFORMANCE_DIR / "test_predictions.csv", index=False)

    with open(PERFORMANCE_DIR / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading {PROPERTY_NAME} model...")

    predictor = GenericPredictor(
        MODEL_ROOT / "artifacts",
        PROPERTY_NAME
    )

    df = load_data()
    _, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    smiles_list = test_df["SMILES"].tolist()
    y_true = test_df["y"].values

    print("Featurizing test set...")
    X_full = featurize_df(smiles_list, return_df=False)

    if X_full is None:
        raise RuntimeError("Featurization failed for test set.")

    print("Running predictions...")
    raw_pred = predictor.predict_from_features(X_full)

    # Handle all known return types safely
    if isinstance(raw_pred, (list, tuple)):
        raw_pred = raw_pred[0]

    y_pred = np.asarray(raw_pred, dtype=float).reshape(-1)


    metrics = compute_metrics(y_true, y_pred)

    plot_results(y_true, y_pred)
    save_outputs(test_df, y_pred, metrics)

    print("\nEvaluation complete.")
    print(f"Results saved to {PERFORMANCE_DIR}")


if __name__ == "__main__":
    main()