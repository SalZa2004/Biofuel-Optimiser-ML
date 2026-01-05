# applications/1_pure_predictor/results.py

import pandas as pd

def display_results(result: dict):
    """
    Display pure-component prediction results.
    """
    df = pd.DataFrame([result])
    print("\n=== PURE COMPONENT PROPERTY PREDICTION ===\n")
    print(df.to_string(index=False))
