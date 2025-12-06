import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load your CSV
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
csv_path = os.path.join(BASE_DIR, "data", "predicted", "bp_4.csv")
df = pd.read_csv(csv_path)

# Load CSV
df = pd.read_csv(csv_path)

y_true = df["EX_Boiling Point"]
y_pred = df["Tb"] - 273.15  # Convert from Kelvin to Celsius

# Compute metrics
MAE = mean_absolute_error(y_true, y_pred)
MSE = mean_squared_error(y_true, y_pred)
RMSE = np.sqrt(MSE)
R2 = r2_score(y_true, y_pred)

# Print results
print(f"Performance for {csv_path}:")
print("  Mean Absolute Error (MAE):", MAE)
print("  Root Mean Squared Error (RMSE):", RMSE)
print("  R² Score:", R2)