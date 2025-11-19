import pandas as pd
from pathlib import Path
import re

def expand_mixture_data():

    excel_file = "data/raw/Merged_data_2.xlsx"
    df_all = pd.read_excel(excel_file)

    # Keep only Mixtures
    df = df_all[df_all['is_pure'] == 'M']

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "mixture_M.csv", index=False)

    print("'mixture_M.csv' created")

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
                    k, v = item.rsplit(":", 1)
                    key = k.strip().strip('"').strip("'")
                    try:
                        value = float(v.strip()) / 100
                    except ValueError:
                        value = None
                    components[key] = value
            base_mixtures[name] = components

    # Expand all mixtures
    expanded_list = []
    for _, row in df.iterrows():
        mix = row["mixture_components"]
        components = {}

        if isinstance(mix, str) and ":" in mix:
            for item in re.split(r',\s*(?=[^,]*:)', mix):
                if ":" in item:
                    k, v = item.rsplit(":", 1)
                    key = k.strip().strip('"').strip("'")
                    try:
                        value = float(v.strip())
                    except ValueError:
                        value = None

                    # Nested mixtures
                    if key.upper() in base_mixtures and value is not None:
                        for sub, val in base_mixtures[key.upper()].items():
                            if val is not None:
                                components[sub] = components.get(sub, 0) + val * (value / 100)
                    else:
                        components[key] = value / 100

        expanded_list.append(components)

    components_df = pd.DataFrame(expanded_list)
    df_expanded = pd.concat([df, components_df], axis=1)

    # Remove unwanted columns
    remove_cols = ["is_pure", "smiles", "mixture_components", "feedstock", "subclass", "NOTES"]
    pattern_remove = df_expanded.columns[
        df_expanded.columns.str.contains("density", case=False, na=False)
        | df_expanded.columns.str.contains("viscosity", case=False, na=False)
    ]

    df_expanded = df_expanded.drop(columns=[c for c in remove_cols + list(pattern_remove) if c in df_expanded.columns])

    # Remove fossil and DNC
    df_expanded = df_expanded[~df_expanded["name"].str.contains("FOSSIL", case=False, na=False)]
    df_expanded = df_expanded[~df_expanded.apply(lambda r: r.astype(str).str.contains("DNC", case=False, na=False)).any(axis=1)]

    # Map component names to SMILES (multi-index columns)
    mapping = pd.read_csv("data/raw/component_smiles.csv")
    name_to_smiles = dict(zip(mapping["name"], mapping["smiles"]))

    new_columns = []
    for col in df_expanded.columns:
        if col in name_to_smiles:
            new_columns.append((name_to_smiles[col], col))  # (SMILES, Name)
        else:
            new_columns.append(("", col))

    df_expanded.columns = pd.MultiIndex.from_tuples(new_columns)

    # Save
    output_file = output_dir / "expanded_mixtures.csv"
    df_expanded.to_csv(output_file, index=False)

    print(f"Done! Expanded mixture file saved as '{output_file}'.")


if __name__ == "__main__":
    
    expand_mixture_data()
