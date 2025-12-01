import pandas as pd
import os
import Pure_PropertyModel

# Input and output CSV paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(BASE_DIR, "..", "..", "data", "processed", "filtered_smiles_bp.csv")
output_csv = os.path.join(BASE_DIR, "..", "..", "data", "predicted", "bp_3.csv")

# Load your CSV
df = pd.read_csv(input_csv)

# Predict boiling point directly with a lambda
df['bp_predicted'] = df['SMILES'].apply(
    lambda smi: Pure_PropertyModel.JOBACK(smi)[1]  # TB is the second element
)

# Save to CSV
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")