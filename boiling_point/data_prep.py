from rdkit import Chem
import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(BASE_DIR, "..", "data", "processed", "complete_pure_data.csv")
output_csv = os.path.join(BASE_DIR, "..", "data", "processed", "filtered_smiles_bp.csv")

# Read CSV
df = pd.read_csv(input_csv)

# Check required columns
required_cols = ['SMILES', 'EX_Boiling Point']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Filter non-empty rows
df_filtered = df[df['SMILES'].notna() & df['EX_Boiling Point'].notna()].copy()

# Filter invalid SMILES
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumBonds() > 0

df_filtered = df_filtered[df_filtered['SMILES'].apply(is_valid_smiles)].copy()

# Save only SMILES and EX_Boiling Point
df_filtered[['SMILES', 'EX_Boiling Point']].to_csv(output_csv, index=False)
print(f"Filtered data saved to {output_csv}")
