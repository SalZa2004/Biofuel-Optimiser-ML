import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load your CSV
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
csv_path = os.path.join(BASE_DIR, "data", "predicted", "bp_2.csv")
df = pd.read_csv(csv_path)

# Experimental and predicted values
y_true = df['EX_Boiling Point']
y_gcn = df['bp_predicted_gcn']
y_mpnn = df['bp_predicted_mpnn']

# Function to print metrics
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R²   = {r2:.3f}\n")

# Evaluate both models
evaluate(y_true, y_gcn, "GCN")
evaluate(y_true, y_mpnn, "MPNN")