import joblib
import pandas as pd
import os
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.base import BaseEstimator, RegressorMixin
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from boiling_point.data_latest import load_data
from boiling_point.evaluate_model import evaluate_model


# 1. Define MetaModel
class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)

# 2. Main execution
if __name__ == "__main__":

    ## PATHS
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Model and descriptor files (example folder inside boiling_point)
    model_path = os.path.join(BASE_DIR, "example", "04_modele_final_NBP.joblib")
    descriptor_path = os.path.join(BASE_DIR, "example", "noms_colonnes_247_TC.txt")

    # Input and output CSVs

    output_csv = os.path.join(BASE_DIR, "..", "..", "data", "predicted", "bp_1.csv")

    ## LOAD FILES

    # Model and descriptors
    NBP = joblib.load(model_path)

    with open(descriptor_path, 'r') as f:
        descriptor_list = [line.strip() for line in f]
    descriptor_list = descriptor_list[1:]  # remove header if present

    # Input CSV
    df = load_data()

    ## PREDICT

    # Define Mordred descriptor calculation
    def compute_mordred(smiles_list):
        calc = Calculator(descriptors, ignore_3D=False)
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        return calc.pandas(mols)

    # Compute descriptors
    df_desc = compute_mordred(df["SMILES"])
    X = df_desc[descriptor_list]

    # Clean data: convert to float, fill NaN with 0
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    # Predict boiling points
    df["bp_predicted"] = NBP.predict(X) - 273.15  # Convert from Kelvin to Celsius

    # Save results to CSV
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    evaluate_model("bp_1.csv")