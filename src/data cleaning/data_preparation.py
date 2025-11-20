import pandas as pd
from pathlib import Path
import numpy as np

# ------------------------------------
# CLEANING THE STANDARDIZED CSV FILE
# ------------------------------------

def clean_standardized_dataframe(csv_file: Path) -> pd.DataFrame:

    df = pd.read_csv(csv_file)
    df = df[~df.apply(lambda row: row.astype(str).str.contains("DNC", case=False, na=False)).any(axis=1)]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(how="all")

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    if "engine_type" in df.columns:
        df = df[df["engine_type"].notna() & (df["engine_type"].astype(str).str.strip() != "")]

    return df

def merge_cleaned_dataframes(cleaned_dir: Path) -> pd.DataFrame:

    all_dfs = []
    for csv_file in cleaned_dir.glob("Standardized_fuel_*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    return merged_df    


if __name__ == "__main__":

    csv_dir = Path("data/processed")
    merged_df = merge_cleaned_dataframes(csv_dir)
    merged_path = csv_dir / "Merged_Cleaned_Standardized_Data.csv"
    merged_df.to_csv(merged_path, index=False)  
    print(f"Saved merged cleaned data to: {merged_path.name}")
