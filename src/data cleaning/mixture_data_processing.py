import pandas as pd
from pathlib import Path
import re

def expand_mixture_data():
    """
    Extract and expand mixture fuel data from the raw Excel file.
    Saves the expanded data to 'data/processed/RP_MIXTURE.csv'.
    
    """

    excel_file = "data/raw/RP_Database.xlsx"
    df_all = pd.read_excel(excel_file)

    # Keep only Mixtures
    df = df_all[df_all['is_pure'] == 'M']

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "mixture_M.csv", index=False)

    print("'mixture_M.csv' is created")

    #--------------------------------
    # EXPANDING MIXTURE COMPONENTS
    #--------------------------------
    file = output_dir / "mixture_M.csv"
    df = pd.read_csv(file)

    # Store base mixtures
    base_mixtures = {}

    for _, row in df.iterrows():
        name = row["name"].strip().upper()
        mix = row.get("mixture_components", "")
        if isinstance(mix, str) and ":" in mix:
            components = {}
            for item in re.split(r',\s*(?=[^,]*:)', mix):
                if ":" in item:
                    key_part, value_part = item.rsplit(":", 1)
                    key = key_part.strip().strip('"').strip("'")
                    try:
                        value = float(value_part.strip())
                    except ValueError:
                        value = None
                    components[key] = value/100
            base_mixtures[name] = components

    component_dicts = []

    for _, row in df.iterrows():
        mix = row["mixture_components"]
        components = {}

        if isinstance(mix, str) and ":" in mix:
            for item in re.split(r',\s*(?=[^,]*:)', mix):
                if ":" in item:
                    key_part, value_part = item.rsplit(":", 1)
                    key = key_part.strip().strip('"').strip("'")
                    try:
                        value = float(value_part.strip())
                    except ValueError:
                        value = None

                    if key.upper() in base_mixtures and value is not None:
                        for subcomp, subval in base_mixtures[key.upper()].items():
                            if subval is not None:
                                components[subcomp] = components.get(subcomp, 0) + subval * (value / 100)
                    else:
                        components[key] = value/100  

        component_dicts.append(components)

    components_df = pd.DataFrame(component_dicts)
    df_expanded = pd.concat([df, components_df], axis=1)

    # Remove unwanted columns after expansion
    cols_to_remove = ["is_pure", "smiles", "mixture_components", "feedstock", "subclass", "NOTES"]
    df_expanded = df_expanded.drop(columns=[col for col in cols_to_remove if col in df_expanded.columns])

    # Remove fossil and DNC again after expansion
    df_expanded = df_expanded[~df_expanded['name'].str.contains('FOSSIL', case=False, na=False)]
    df_expanded = df_expanded[~df_expanded.apply(lambda row: row.astype(str).str.contains('DNC', case=False, na=False)).any(axis=1)]

    # Mapping component names to SMILES for MultiIndex columns
    mapping = pd.read_csv("data/raw/component_smiles.csv")
    name_to_smiles = dict(zip(mapping["name"], mapping["smiles"]))

    new_columns = []
    for col in df_expanded.columns:
        if col in name_to_smiles:
            new_columns.append((name_to_smiles[col], col))  # (SMILES, Name)
        else:
            new_columns.append(("", col))  # Empty top-level if no mapping

    df_expanded.columns = pd.MultiIndex.from_tuples(new_columns)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "RP_MIXTURE.csv"
    df_expanded.to_csv(output_file, index=False)

    print(f"Done! Expanded mixture file saved as '{output_file}'.")


if __name__ == "__main__":
    
    expand_mixture_data()

