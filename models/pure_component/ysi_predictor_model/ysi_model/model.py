import os
import joblib
import pandas as pd
import sys
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)
# import feature selector and features from correct modules
from core.shared_features import FeatureSelector, featurize_df


class YSIPredictor:


    def __init__(self):
        print("Loading YSI Predictor...")

        # Path to this file (ysi_model/model.py)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Path to artifacts directory
        artifact_dir = os.path.join(base_dir, "artifacts")

        # Full absolute paths
        model_path = os.path.join(artifact_dir, "model.joblib")
        selector_path = os.path.join(artifact_dir, "selector.joblib")

        # Print debug (helps confirm correct loading)
        print(">>> MODEL PATH:", model_path)
        print(">>> SELECTOR PATH:", selector_path)
        print(">>> MODEL EXISTS:", os.path.exists(model_path))
        print(">>> SELECTOR EXISTS:", os.path.exists(selector_path))

        # Load artifacts
        self.model = joblib.load(model_path)
        self.selector = FeatureSelector.load(selector_path)

        print("✓ Predictor ready!\n")

    def predict(self, smiles_list):
        """Inference on a list of SMILES strings."""

        # Handle single SMILES
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        # Featurize
        X_full = featurize_df(smiles_list, return_df=False)

        if X_full is None:
            print("⚠ Warning: No valid molecules found!")
            return []

        # Apply feature selection
        X_selected = self.selector.transform(X_full)

        # Predict
        predictions = self.model.predict(X_selected)
        return predictions.tolist()

    def predict_with_details(self, smiles_list):
        """Inference with valid/invalid info."""

        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        df = pd.DataFrame({"SMILES": smiles_list})
        X_full, df_valid = featurize_df(df, return_df=True)

        if X_full is None:
            return pd.DataFrame(columns=["SMILES", "Predicted_YSI", "Valid"])

        X_selected = self.selector.transform(X_full)
        predictions = self.model.predict(X_selected)

        df_valid["Predicted_YSI"] = predictions
        df_valid["Valid"] = True

        all_results = pd.DataFrame({"SMILES": smiles_list})
        all_results = all_results.merge(
            df_valid[["SMILES", "Predicted_YSI", "Valid"]],
            on="SMILES", how="left"
        )
        all_results["Valid"] = all_results["Valid"].fillna(False)

        return all_results
