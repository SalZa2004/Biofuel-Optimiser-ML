import pandas as pd
from pathlib import Path

def clean_pure_data():
    """
    Extract and clean pure fuel data from the raw Excel file.
    Saves the cleaned data to 'data/processed/expanded_pure.csv'.
    
    """
    excel_file = "data/raw/RP_Database.xlsx"
    df_all = pd.read_excel(excel_file)

    # Keep only Pure
    df = df_all[df_all["is_pure"] == "P"]

    # Rename columns
    df = df.rename(columns={"smiles": "SMILES"})
    df = df.rename(columns={"paper": "source"})

    # Remove rows containing "DNC"
    df = df[~df.apply(lambda row: row.astype(str).str.contains("DNC", case=False, na=False)).any(axis=1)]

    # Remove unwanted columns
    remove_cols = ["is_pure", "mixture_components", "feedstock", "NOTES"]
    df_cleaned = df.drop(columns=[col for col in remove_cols if col in df.columns])

    # Remove duplicate names
    df_cleaned = df_cleaned.drop_duplicates(subset="name", keep="first")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "RP_PURE.csv"
    df_cleaned.to_csv(output_file, index=False)

    print(f"Done! Expanded pure fuel file saved as '{output_file}'.")


def extract_online_pure_data():
    """
    Loads and cleans three external online pure-fuel datasets:
    - Detailed YSI Database Volume 2
    - Schweidtmann 2020 (D)CN
    - Chen 2024

    Saves cleaned versions into data/processed/ as:
        online_data_Harvard_YSI.csv
        online_data_Schweidtmann_DCN.csv
        online_data_Chen_Simulated.csv
    """

    df1 = pd.read_excel("data/raw/Detailed YSI Database Volume 2.xlsx")
    df2 = pd.read_excel(
        "data/raw/Schweidtmann 2020 IQT-DCN, (D)CN.xlsx",
        sheet_name="(D)CN_TransferLearning_Comp",
        usecols=range(5)  # columns A–E
    )
    df3 = pd.read_excel(
        "data/raw/Chen 2024.xlsx",
        header=1
    )

    # ------------
    # ADD SOURCES
    # ------------

    df1["paper"] = "Detailed YSI Database Volume 2"
    df2["paper"] = "Schweidtmann 2020 IQT-DCN (D)CN"
    df3["paper"] = "Chen 2024"

    # ------------------------
    # CLEAN df1 (YSI database)
    # ------------------------

    cols_to_remove_df1 = [
        "Ref #",
        "Original Upper endpoint species",
        "Original Upper endpoint value",
        "Original Lower endpoint species",
        "Original Lower endpoint value",
    ]

    df1 = df1.drop(columns=[c for c in cols_to_remove_df1 if c in df1.columns])
    df1 = df1.rename(columns={"Species": "name", "CAS #": "CAS"})

    # ------------------------
    # CLEAN df2 (Schweidtmann)
    # ------------------------
    
    # remove rows with missing SMILES
    df2 = df2.dropna(subset=['SMILES'])

    # rename columns
    df2 = df2.rename(columns={"Compounds": "name"})

    # ------------------------
    # CLEAN df3 (Chen 2024)
    # ------------------------

    cols_to_remove_1 = range(3, 17)   # B–O columns
    cols_to_remove_2 = range(35, 41)  # AK–AO columns
    cols_to_remove = list(cols_to_remove_1) + list(cols_to_remove_2)

    df3 = df3.drop(df3.columns[cols_to_remove], axis=1)
    df3 = df3.drop(columns={"Number"})
    df3 = df3.rename(columns={"Name": "name"})


    # ------------------------
    # SAVE CLEANED FILES
    # ------------------------

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df1.to_csv(output_dir / "online_data_Harvard_YSI.csv", index=False)
    df2.to_csv(output_dir / "online_data_Schweidtmann_DCN.csv", index=False)
    df3.to_csv(output_dir / "online_data_Chen_Simulated.csv", index=False)

    print("Online pure data extraction completed and saved.")

if __name__ == "__main__":
    
    clean_pure_data()
    extract_online_pure_data()
