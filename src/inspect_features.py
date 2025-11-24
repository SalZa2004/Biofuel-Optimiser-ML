import joblib
import numpy as np
import pandas as pd

from feature_selection import FeatureSelector
from feature_engineering import DESCRIPTOR_NAMES

print("\n====================================================")
print("FEATURE SELECTION BREAKDOWN")
print("====================================================\n")

# Load selector
selector: FeatureSelector = joblib.load("feature_selector.pkl")

# Important attributes
n_morgan = selector.n_morgan
selected_idx = selector.selected_indices

# Morgan fingerprint bits (0 .. n_morgan-1)
mfp_mask = selected_idx < n_morgan
selected_mfp_bits = selected_idx[mfp_mask]

# Descriptor positions AFTER correlation filtering
desc_mask = selected_idx >= n_morgan
desc_positions_after_corr = selected_idx[desc_mask] - n_morgan

# Descriptor names
descriptor_names = [
    selector.remaining_descriptor_names[i]
    for i in desc_positions_after_corr
]

print(f"Total selected features: {len(selected_idx)}")
print(f"Morgan fingerprint bits: {len(selected_mfp_bits)}")
print(f"Descriptor features:      {len(descriptor_names)}\n")

print("Selected Morgan FP bits:")
print(selected_mfp_bits.tolist(), "\n")

print("Selected RDKit Descriptors:")
for name in descriptor_names:
    print(" -", name)

# Save for export
df_export = pd.DataFrame({
    "feature_type": (
        ["morgan_fp"] * len(selected_mfp_bits)
        + ["descriptor"] * len(descriptor_names)
    ),
    "feature": (
        [f"bit_{b}" for b in selected_mfp_bits]
        + descriptor_names
    )
})

df_export.to_csv("selected_features.csv", index=False)
print("\nSaved feature list to selected_features.csv")
