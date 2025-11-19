import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_input_files():
    """Load the 4 input CSV files."""
    df1 = pd.read_csv('data/processed/online_data_Harvard_YSI.csv')
    df2 = pd.read_csv('data/processed/online_data_Schweidtmann_DCN.csv')
    df3 = pd.read_csv('data/processed/online_data_Chen_Simulated.csv')
    df4 = pd.read_csv('data/processed/RP_PURE.csv')
    return df1, df2, df3, df4

# ------------------
# MERGING DATASETS
# ------------------

def merge_all_files(df1, df2, df3, df4):
    """Perform full outer join on SMILES across all 4 datasets."""
    for df in [df1, df2, df3, df4]:
        df['SMILES'] = df['SMILES'].astype(str).str.strip()
    merged = (
        df1.merge(df2, on='SMILES', how='outer', suffixes=('_1', '_2'))
           .merge(df3, on='SMILES', how='outer', suffixes=('', '_3'))
           .merge(df4, on='SMILES', how='outer', suffixes=('', '_4'))
    )
    return merged


def combine_fuel_names(df):
    """
    Unify all fuel name columns into one 'Fuel Name' column.
    """
    name_cols = [c for c in df.columns if 'name' in c.lower()]
    df['Fuel Name'] = df[name_cols].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns=name_cols)
    return df


def combine_sources_columns(df):
    """
    Unify all source paper columns into one 'Source' column.
    """
    paper_cols = [col for col in df.columns if 'paper' in col.lower()]

    def combine_sources(row):
        items = [str(v).strip() for v in row[paper_cols] if pd.notna(v) and str(v).strip()]
        return ', '.join(sorted(set(items), key=items.index)) if items else None

    df['Source'] = df.apply(combine_sources, axis=1)
    df = df.drop(columns=paper_cols)
    return df


def combine_cas_columns(df):
    """
    Unify all CAS number columns into one 'CAS' column.
    """
    cas_cols = [c for c in df.columns if 'cas' in c.lower()]

    def combine_cas(row):
        items = [str(v).strip() for v in row[cas_cols] if pd.notna(v) and str(v).strip()]
        return ', '.join(sorted(set(items), key=items.index)) if items else None

    if cas_cols:
        df['CAS'] = df.apply(combine_cas, axis=1)
        df = df.drop(columns=cas_cols)
    return df


# ----------------
# STANDARDISATION 
# ----------------

def standardise_ysi(df):

    def get_ysi(row):
        if pd.notna(row.get('Unified YSI')) and str(row['Unified YSI']).strip():
            return row['Unified YSI']
        if pd.notna(row.get('EX_Yield Sooting Index')) and str(row['EX_Yield Sooting Index']).strip():
            return row['EX_Yield Sooting Index']
        if pd.notna(row.get('SI_Yield Sooting Index')) and str(row['SI_Yield Sooting Index']).strip():
            return row['SI_Yield Sooting Index']
        return None

    df['Standardised_YSI'] = df.apply(get_ysi, axis=1)
    return df


def standardise_dcn(df):

    def get_dcn(row):
        if pd.notna(row.get('measured (D)CN')) and str(row.get('measured (D)CN')).strip():
            return row['measured (D)CN']
        if pd.notna(row.get('EX_Cetane Number')) and str(row.get('EX_Cetane Number')).strip():
            return row['EX_Cetane Number']
        if pd.notna(row.get('SI_Cetane Number')) and str(row.get('SI_Cetane Number')).strip():
            return row['SI_Cetane Number']
        return None

    df['Standardised_DCN'] = df.apply(get_dcn, axis=1)
    return df

def standardise_fuel_names_order(df):
    df = df.sort_values(by='Fuel Name', ascending=True)
    return df

# -------------
# STATISTICS
# -------------

def print_statistics(df):
    print("\nFuel Name Statistics:")
    print(f"   Total:  {df['Fuel Name'].shape[0]}")
    print(f"   Unique: {df['Fuel Name'].nunique(dropna=True)}")
    print(f"   Duplicates: {df['Fuel Name'].shape[0] - df['Fuel Name'].nunique(dropna=True)}")

    print("\nSMILES Statistics:")
    print(f"   Total:  {df['SMILES'].shape[0]}")
    print(f"   Unique: {df['SMILES'].nunique(dropna=True)}")
    print(f"   Duplicates: {df['SMILES'].shape[0] - df['SMILES'].nunique(dropna=True)}")


def remove_redundant_smiles(df):
    original = df.copy()

    if 'Unified YSI' in df.columns:
        df = df.drop_duplicates(subset=['SMILES', 'Unified YSI'], keep='first')

    if 'measured (D)CN' in df.columns:
        df = df.drop_duplicates(subset=['SMILES', 'measured (D)CN'], keep='first')

    removed = original.merge(df, how='outer', indicator=True).query('_merge == "left_only"')
    removed = removed.drop(columns=['_merge'])

    if not removed.empty:
        removed.to_csv('data/processed/removed_redundant_smiles.csv', index=False)

    print(f"\nRemoved {removed.shape[0]} redundant SMILES rows.")
    return df


def count_ysi_sources(df):
    counts = {"Unified YSI": 0, "EX_Yield Sooting Index": 0, "SI_Yield Sooting Index": 0, "Missing": 0}
    for _, r in df.iterrows():
        if r['Standardised_YSI'] == r.get('Unified YSI'):
            counts["Unified YSI"] += 1
        elif r['Standardised_YSI'] == r.get('EX_Yield Sooting Index'):
            counts["EX_Yield Sooting Index"] += 1
        elif r['Standardised_YSI'] == r.get('SI_Yield Sooting Index'):
            counts["SI_Yield Sooting Index"] += 1
        else:
            counts["Missing"] += 1
    print("\nStandardised_YSI source breakdown:")
    for k, v in counts.items():
        print(f"   {k:30s}: {v}")
    return counts


def count_dcn_sources(df):
    counts = {"measured (D)CN": 0, "EX_Cetane Number": 0, "SI_Cetane Number": 0, "Missing": 0}
    for _, r in df.iterrows():
        if r['Standardised_DCN'] == r.get('measured (D)CN'):
            counts["measured (D)CN"] += 1
        elif r['Standardised_DCN'] == r.get('EX_Cetane Number'):
            counts["EX_Cetane Number"] += 1
        elif r['Standardised_DCN'] == r.get('SI_Cetane Number'):
            counts["SI_Cetane Number"] += 1
        else:
            counts["Missing"] += 1
    print("\nStandardised_DCN source breakdown:")
    for k, v in counts.items():
        print(f"   {k:30s}: {v}")
    return counts
     
# ---------------------------------------
# PLOTTING DATA SOURCE BREAKDOWN
# ---------------------------------------

def plot_data_source_breakdown(ysi_values, dcn_values):
    """
    Plot a stacked-bar comparison of YSI and DCN data source counts [Online, Experimental, Simulated, Missing].
    """

    if len(ysi_values) != 4 or len(dcn_values) != 4:
        raise ValueError("ysi_values and dcn_values must each contain exactly 4 values.")

    categories = ["Online", "Experimental", "Simulated", "Missing"]
    colors = ['#5DADE2', '#58D68D', '#F1948A', '#BB8FCE']   

    x = np.arange(2)   # positions for YSI and DCN
    width = 0.55

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 7))

    bottom_ysi = 0
    bottom_dcn = 0

    # Stacked bar chart
    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax.bar(x[0], ysi_values[i], width, bottom=bottom_ysi, color=color, edgecolor='white')
        ax.bar(x[1], dcn_values[i], width, bottom=bottom_dcn, color=color, edgecolor='white')

        # Add value labels inside each block
        ax.text(x[0], bottom_ysi + ysi_values[i] / 2, f"{ysi_values[i]}",
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(x[1], bottom_dcn + dcn_values[i] / 2, f"{dcn_values[i]}",
                ha='center', va='center', fontsize=12, fontweight='bold')

        bottom_ysi += ysi_values[i]
        bottom_dcn += dcn_values[i]

    # Legend
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(legend_handles, categories, title="Data Source",
              bbox_to_anchor=(1.05, 1), loc='upper left',
              fontsize=12, title_fontsize=13)

    # Axis
    ax.set_xticks(x)
    ax.set_xticklabels(['YSI', 'DCN'], fontsize=15, fontweight='bold')
    ax.set_ylabel("Number of Entries", fontsize=14, fontweight='bold')
    ax.set_title("Comparison of Standardised Data Sources for YSI and DCN",
                 fontsize=18, fontweight='bold', pad=20)

    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Load files
    df1, df2, df3, df4 = load_input_files()
   
    # Merge
    merged = merge_all_files(df1, df2, df3, df4)
    merged = combine_fuel_names(merged)
    merged = combine_sources_columns(merged)
    merged = combine_cas_columns(merged)

    # Standardise
    merged = standardise_ysi(merged)
    merged = standardise_dcn(merged)

    # Cleaning
    cleaned_df = remove_redundant_smiles(merged)
    cleaned_df = standardise_fuel_names_order(cleaned_df)

    # Statistics
    print_statistics(cleaned_df)
    count_ysi_sources(cleaned_df)
    count_dcn_sources(cleaned_df)
    
    # Save final merged file
    cleaned_df.to_csv('data/processed/complete_pure_data.csv', index=False)
    print("\nMerge complete and saved to complete_pure_data.csv")

    # Plots
    ysi_values = [441, 212, 708, 133]
    dcn_values = [479, 62, 692, 261]
    plot_data_source_breakdown(ysi_values, dcn_values)


