import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
import os
from model_loader import load_gcn_model, load_mpnn_model
from predictor import predict_boiling_point

# Input and output CSV paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(BASE_DIR, "..", "..", "data", "processed", "filtered_smiles_bp.csv")
output_csv = os.path.join(BASE_DIR, "..", "..", "data", "predicted", "bp_2.csv")

# Read CSV
df = pd.read_csv(input_csv)

# Load models
gcn_model = load_gcn_model()
mpnn_model = load_mpnn_model()

# Predict
smiles_list = df['SMILES'].tolist()
gcn_predictions = np.round(predict_boiling_point(gcn_model, smiles_list, mode='gcn'), 3)
mpnn_predictions = np.round(predict_boiling_point(mpnn_model, smiles_list, mode='mpnn'), 3)

# Add predictions to DataFrame
df["bp_predicted_gcn"] = gcn_predictions.flatten() - 273.15  # Convert from Kelvin to Celsius
df["bp_predicted_mpnn"] = mpnn_predictions.flatten() - 273.15  # Convert from Kelvin to Celsius

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")