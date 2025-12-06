import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
from boiling_point.data_latest import load_data
def evaluate_model(csv_filename):
    """
    Compute performance metrics (MAE, RMSE, R²) for predicted boiling points.

    Parameters
    ----------
    csv_filename : str
        Name of the CSV file in the 'data/predicted' folder, e.g., 'bp_1.csv'

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, R²
    """
    # Build full path relative to this script
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(BASE_DIR, "data", "predicted", csv_filename)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ["boiling_point", "bp_predicted"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    
    y_true = df["boiling_point"]
    y_pred = df["bp_predicted"]
    
    # Compute metrics
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_true, y_pred)
    
    # Print results
    print(f"Performance for {csv_filename}:")
    print("  Mean Absolute Error (MAE):", MAE)
    print("  Root Mean Squared Error (RMSE):", RMSE)
    print("  R² Score:", R2)
    N= 0
    for y_true, y_pred in zip(df["boiling_point"], df["bp_predicted"]):
        if y_pred - y_true > 50:
            N+=1
            
    print("  Number of large errors (>100K):", N)
    return {"MAE": MAE, "RMSE": RMSE, "R2": R2}

# Example usage in terminal:
# python
# from evaluate_model import evaluate_model
# evaluate_model('bp_1.csv')

# NOTE: Won't work for bp_3.csv and bp_4.csv. Use evaluate_gnn_model.py and evaluate_chemprop_model instead.