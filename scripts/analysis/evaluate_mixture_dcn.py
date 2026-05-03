"""
Evaluate MixtureDCNPredictor against the formatted_mixtures.csv benchmark.

Usage (from project root):
    python scripts/evaluate_mixture_dcn.py

Output:
    results/mixture_dcn_actual_vs_pred.png
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless — must be set before importing pyplot
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor

DATA_PATH = ROOT / "data" / "database" / "formatted_mixtures.csv"
RESULTS_DIR = ROOT / "results"


def load_mixtures(path):
    mixtures = []
    with open(path) as f:
        for row in csv.DictReader(f):
            n = int(row["num_components"])

            smiles = [row["fuel1_inchi"]]
            if n >= 2:
                smiles.append(row["fuel2_inchi"])
            if n >= 3:
                smiles.append(row["fuel3_inchi"])

            frac1 = float(row["frac_fuel1 (molar)"])
            if n == 2:
                fracs = [frac1, 1.0 - frac1]
            else:
                frac2 = float(row["frac_fuel2 (molar)"])
                fracs = [frac1, frac2, round(1.0 - frac1 - frac2, 8)]

            mixtures.append({
                "smiles": smiles,
                "fracs": fracs,
                "actual": float(row["DCN"]),
                "name": row.get("Name", ""),
            })
    return mixtures


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading data from {DATA_PATH.relative_to(ROOT)} ...")
    mixtures = load_mixtures(DATA_PATH)
    print(f"  {len(mixtures)} mixtures loaded")

    print("Initialising predictor ...")
    predictor = MixtureDCNPredictor()

    print("Predicting ...")
    actual, predicted = [], []
    failed = 0

    for i, mix in enumerate(mixtures):
        try:
            pred = predictor.predict_mixture_dcn(mix["smiles"], mix["fracs"])
            actual.append(mix["actual"])
            predicted.append(pred)
        except Exception as e:
            print(f"  [SKIP {i}] {mix['name']}: {e}")
            failed += 1

    if not actual:
        print("No successful predictions — exiting.")
        return

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)

    print(f"\nMetrics  ({len(actual)}/{len(mixtures)} succeeded, {failed} failed)")
    print(f"  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(actual, predicted, s=55, alpha=0.75, edgecolors="k", linewidths=0.4)

    lo = min(actual.min(), predicted.min()) - 3
    hi = max(actual.max(), predicted.max()) + 3
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="Ideal (y = x)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Actual DCN", fontsize=13)
    ax.set_ylabel("Predicted DCN", fontsize=13)
    ax.set_title("Mixture DCN: Actual vs Predicted", fontsize=14)

    stats = f"MAE  = {mae:.2f}\nRMSE = {rmse:.2f}\nR²   = {r2:.4f}\nn = {len(actual)}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()

    out_path = RESULTS_DIR / "mixture_dcn_actual_vs_pred.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path.relative_to(ROOT)}")
    plt.close()


if __name__ == "__main__":
    main()
