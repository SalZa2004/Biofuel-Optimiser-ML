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


if __name__ == "__main__":

    csv_dir = Path("data/processed")
    cleaned_dir = Path("data/processed")
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_dir.glob("Standardized_*.csv"):

        cleaned_df = clean_standardized_dataframe(csv_file)
        cleaned_path = cleaned_dir / f"Cleaned_{csv_file.name}"
        cleaned_df.to_csv(cleaned_path, index=False)

        print(f"Saved cleaned file: {cleaned_path.name}")
