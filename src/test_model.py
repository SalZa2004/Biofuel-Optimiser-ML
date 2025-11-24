from feature_selection import FeatureSelector, prepare_prediction_features
import joblib
import pandas as pd

# Load model + selector
model = joblib.load("extratrees_model.pkl")
selector = joblib.load("feature_selector.pkl")

# Load CSV
candidate_df = pd.read_csv("data/raw/candidates_3.csv")

# Clean SMILES
candidate_df["SMILES"] = (
    candidate_df["smiles"]
    .astype(str)
    .str.split(r"\s+")
    .str[0]
    .str.strip()
)

# Featurize + select
X_pred = prepare_prediction_features(candidate_df["SMILES"], selector)

# Predict
preds = model.predict(X_pred)

# Attach predictions
candidate_df["predicted_cn"] = preds

# Print nicely
print(candidate_df[["names", "predicted_cn"]])
