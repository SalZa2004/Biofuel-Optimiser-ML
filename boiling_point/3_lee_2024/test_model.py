import pandas as pd
import os
import Pure_PropertyModel

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from boiling_point.data_latest import load_data

# Input and output CSV paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(BASE_DIR, "..", "..", "data", "processed", "filtered_smiles_bp.csv")
output_csv = os.path.join(BASE_DIR, "..", "..", "data", "predicted", "bp_3.csv")

# Load your CSV
df = load_data()

# Predict boiling point directly with a lambda
df['bp_predicted'] = df['SMILES'].apply(
    lambda smi: Pure_PropertyModel.JOBACK(smi)[1]  # TB is the second element
)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")