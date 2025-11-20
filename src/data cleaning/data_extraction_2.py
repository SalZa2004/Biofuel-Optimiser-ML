import pandas as pd
from pathlib import Path

# ---------------------------------------
# CREATING EMPTY STANDARDIZED DATAFRAME 
# ---------------------------------------

def create_empty_dataframe():

    columns = [
        "paper", # e.g., author names, year, title etc.
        "name", 
        "is_pure", # P/M
        "smiles",# SMILES representation of the molecule
        "mixture_components", # number of components in the mixture
        "fuel_type", # e.g., biodiesel, bio alcohol
        "is_biofuel", # Biofuel or not
        "feedstock", # e.g., waste oil, algae, etc.
        "subclass", # e.g., methyl ester, ethyl ester, etc.
        "density_at_15_kg_m3",
        "density_at_20_kg_m3",
        "density_at_25_kg_m3",
        "density_at_60_kg_m3",
        "dynamic_viscosity_at_20_mPa_s",
        "dynamic_viscosity_at_25_mPa_s",
        "dynamic_viscosity_at_60_mPa_s",
        "kinematic_viscosity_at_20_cST",
        "kinematic_viscosity_at_40_cST",
        "lower_heating_value_MJ_kg",
        "cetane_number",
        "DCN", # Derived Cetane Number
        "injection_timing_CAD_BTDC",
        "ignition_delay_CAD",
        "YSI", # Yield Sooting Index
        "total_particulate_mass_concentration_mg_m3",

    ]

    data_df = pd.DataFrame(columns=columns)

    raw_data_dir = Path("data/processed")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_data_dir / "fuel_template.csv"
    data_df.to_csv(output_file, index=False)

    print(f"Standardized CSV fuel template created successfully")


if __name__ == "__main__":


# --------------------------------------
# CREATING EMPTY STANDARDIZED DATAFRAME
# --------------------------------------

    create_empty_dataframe()

# -----------------------------------
# EXTRACTING DATA FROM SALVINA SHEET
# -----------------------------------

    sheet_name = "Salvina"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
    temp_df = standard_df.copy()

    mapping = {
        "paper": "Source_Paper",
        "name": "Fuel",
        "is_pure": "",
        "smiles": "",
        "mixture_components": "",
        "is_biofuel": "",
        "fuel_type": "",
        "feedstock": "Feedstock_Type",
        "subclass": "",
        "density_at_15_kg_m3": "",
        "density_at_20_kg_m3": "Density_kg_at 20 m3",
        "density_at_25_kg_m3": "",
        "density_at_60_kg_m3": "",
        "dynamic_viscosity_at_20_mPa_s": "Dynamic Viscosity at 19.7 (mPa·s)",
        "dynamic_viscosity_at_25_mPa_s": "",
        "dynamic_viscosity_at_60_mPa_s": "Dynamic_viscosity_at_59.7 (mPa s)",
        "kinematic_viscosity_at_20_cST": "",
        "kinematic_viscosity_at_40_cST": "",
        "lower_heating_value_MJ_kg": "Lower_Heating_Value_MJ_kg",
        "cetane_number": "Cetane_Number",
        "DCN": "",
        "injection_timing_CAD_BTDC": "CIN Injection Timing (CAD BTDC)",
        "ignition_delay_CAD": "CIN Ignition_Delay_CAD",
        "YSI": "",
        "total_particulate_mass_concentration_mg_m3": "CIN total_particulate_mass_concentration (gm-3)",
    }

    for std_col, src_col in mapping.items():
        if src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# -----------------------------------
# EXTRACTING DATA FROM INTAN SHEET
# -----------------------------------

    sheet_name = "Intan"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
    temp_df = standard_df.copy()

    mapping = {
        "paper": "Source_Paper",
        "name": "Fuel",
        "is_pure": "",
        "smiles": "",
        "mixture_components": "",
        "is_biofuel": "is_biofuel",
        "fuel_type": "",
        "feedstock": "",
        "subclass": "",
        "density_at_15_kg_m3": "Density at 15g_m3",
        "density_at_20_kg_m3": "Density at 20 kg_m3",
        "density_at_25_kg_m3": "",
        "density_at_60_kg_m3": "",
        "dynamic_viscosity_at_20_mPa_s": "Dynamic Viscosity at 20 (mPa·s)",
        "dynamic_viscosity_at_25_mPa_s": "",
        "dynamic_viscosity_at_60_mPa_s": "Dynamic_viscosity_at_60 (mPa s)",
        "kinematic_viscosity_at_20_cST": "",
        "kinematic_viscosity_at_40_cST": "Kinematic viscosity at 40",
        "lower_heating_value_MJ_kg": "Lower_Heating_Value_MJ_kg",
        "cetane_number": "Cetane_Number",
        "DCN": "",
        "injection_timing_CAD_BTDC": "CIN Injection Timing (CAD BTDC)",
        "ignition_delay_CAD": "CIN Ignition_Delay_CAD",
        "YSI": "",
        "total_particulate_mass_concentration_mg_m3": "",
    }

    for std_col, src_col in mapping.items():
        if src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ------------------------------------
# EXTRACTING DATA FROM MAIMOONA SHEET
# ------------------------------------

    sheet_name = "Maimoona"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    temp_df = standard_df.copy()
    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)

    max_row = len(df_sheet)
    row_1_indices = [i for i in list(range(1, 14)) + list(range(29, 54)) + list(range(81, 109)) if i < max_row]
    row_2_indices = [i for i in list(range(15, 28)) + list(range(55, 80)) + list(range(110, 138)) if i < max_row]

    subset_1_df = df_sheet.iloc[row_1_indices][["Source_Paper", "Fuel", "Biodiesel?", "Fuel_Type", "Boiling_Point_K", 
    "Melting_Point_K", "Density_at_20_kg_m3", "Density_at_60_kg_m3", "Dynamic_Viscosity_at_20_mPa.s", 
    "Dynamic_Viscosity_at_60_mPa.s", "Lower_Heating_Value_MJ_kg", "Cetane_Number", "Engine_Type", 
    "Engine_Speed_rpm", "Injection_Pressure_bar", "IMEP_bar", "SOI for constant injection timing (CAD BTDC)", 
    "Ignition_Delay_CAD", "Indicated_Thermal_Efficiency_%", "Injection_Duration_μs", "Peak_Pressure_bar", 
    "Peak_Heat_Release_Rate_J_deg", "Max_Temperature_K", "NOx_ppm ", "CO_ppm", "THC_ppm", 
    "Total_Particulate_Mass_Concentration_g_m3", "2_EHN_dosage_ppm"]]

    subset_2_df = df_sheet.iloc[row_2_indices][["SOC for constant ignition timing (CAD BTDC)", 
    "Ignition_Delay_CAD", "Indicated_Thermal_Efficiency_%", "Injection_Duration_μs", "Peak_Pressure_bar", 
    "Peak_Heat_Release_Rate_J_deg", "Max_Temperature_K", "NOx_ppm ", "CO_ppm", "THC_ppm", 
    "Total_Particulate_Mass_Concentration_g_m3"]]

    temp_df = pd.DataFrame({
        "paper": subset_1_df["Source_Paper"].values,
        "name": subset_1_df["Fuel"].values,
        "is_pure": None,
        "smiles": None,
        "mixture_components": None,
        "is_biofuel": subset_1_df["Biodiesel?"].values,
        "fuel_type": subset_1_df["Fuel_Type"].values,
        "feedstock": None,  
        "subclass": None,  
        "density_at_15_kg_m3": None,
        "density_at_20_kg_m3": subset_1_df["Density_at_20_kg_m3"].values,
        "density_at_25_kg_m3": None,
        "density_at_60_kg_m3": subset_1_df["Density_at_60_kg_m3"].values,
        "dynamic_viscosity_at_20_mPa_s": subset_1_df["Dynamic_Viscosity_at_20_mPa.s"].values,
        "dynamic_viscosity_at_25_mPa_s": None,
        "dynamic_viscosity_at_60_mPa_s": subset_1_df["Dynamic_Viscosity_at_60_mPa.s"].values,
        "kinematic_viscosity_at_20_cST": None,
        "kinematic_viscosity_at_40_cST": None,
        "lower_heating_value_MJ_kg": subset_1_df["Lower_Heating_Value_MJ_kg"].values,
        "cetane_number": subset_1_df["Cetane_Number"].values,
        "DCN": None,  
        "injection_timing_CAD_BTDC": subset_1_df["SOI for constant injection timing (CAD BTDC)"].values,
        "ignition_delay_CAD": subset_1_df["Ignition_Delay_CAD"].values,
        "YSI": None,  
        "total_particulate_mass_concentration_mg_m3": subset_1_df["Total_Particulate_Mass_Concentration_g_m3"].values,
    })

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ---------------------------------
# EXTRACTING DATA FROM NADIA 2SHEET
# ---------------------------------

    sheet_name = "Nadia 2"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    raw_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    source_paper_value = raw_df.iat[0, 0] if pd.notna(raw_df.iat[0, 0]) else "Unknown Source"

    try:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=[4, 5])
        df_sheet.columns = [
            "_".join([str(c) for c in col if str(c) != "nan"]).strip()
            for col in df_sheet.columns
        ]
    except ValueError:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)

    # print(df_sheet.columns.tolist())
    
    df_sheet = df_sheet.iloc[:19]
    temp_df = standard_df.copy()

    mapping = {
        "paper": "",
        "name": "Fuel Type_Unnamed: 0_level_1",
        "is_pure": "",
        "smiles": "",
        "mixture_components": "",
        "is_biofuel": "Feedstock_Type_Unnamed: 4_level_1",
        "fuel_type": "",
        "feedstock": "",
        "subclass": "",
        "density_at_15_kg_m3": "",
        "density_at_20_kg_m3": "Density at 20_Unnamed: 7_level_1",
        "density_at_25_kg_m3": "",
        "density_at_60_kg_m3": "",
        "dynamic_viscosity_at_20_mPa_s": "",
        "dynamic_viscosity_at_25_mPa_s": "",
        "dynamic_viscosity_at_60_mPa_s": "",
        "kinematic_viscosity_at_20_cST": "Kinematic_Viscosity_mm2_s at 20_Unnamed: 9_level_1",
        "kinematic_viscosity_at_40_cST": "",
        "lower_heating_value_MJ_kg": "",
        "cetane_number": "Cetane Number_Unnamed: 10_level_1",
        "DCN": "",
        "injection_timing_CAD_BTDC": "",
        "ignition_delay_CAD": "Constant injection timing (SOI at 7.5 CAD BTDC)_Average Ignition Delay (CAD)",
        "YSI": "",
        "total_particulate_mass_concentration_mg_m3": "Total_PM_ug_cc_10^-5_Constant_Injection_Timing",
    }

    for std_col, src_col in mapping.items():
        if src_col and src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    temp_df["paper"] = source_paper_value

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ---------------------------------
# EXTRACTING DATA FROM NADIA 1 SHEET
# ---------------------------------

    sheet_name = "Nadia 1"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    raw_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
    source_paper_value = raw_df.iat[0, 0] if pd.notna(raw_df.iat[0, 0]) else "Unknown Source"

    try:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=[2, 3])
        df_sheet.columns = [
            "_".join([str(c) for c in col if str(c) != "nan"]).strip()
            for col in df_sheet.columns
        ]
    except ValueError:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)

    # print(df_sheet.columns.tolist())
    
    df_sheet = df_sheet.iloc[:19]
    temp_df = standard_df.copy()
    
    mapping = {
        "paper": "",
        "name": "Fuel_Type_Unnamed: 0_level_1",
        "is_pure": "",
        "smiles": "",
        "mixture_components": "",
        "is_biofuel": "Feedstock_Type_Unnamed: 1_level_1",
        "fuel_type": "",
        "feedstock": "",
        "subclass": "",
        "density_at_15_kg_m3": "Density_g_cm3 @ 15°C_Unnamed: 11_level_1",
        "density_at_20_kg_m3": "",
        "density_at_25_kg_m3": "",
        "density_at_60_kg_m3": "",
        "dynamic_viscosity_at_20_mPa_s": "",
        "dynamic_viscosity_at_25_mPa_s": "",
        "dynamic_viscosity_at_60_mPa_s": "",
        "kinematic_viscosity_at_20_cST": "",
        "kinematic_viscosity_at_40_cST": "Kinematic_Viscosity_mm2_s @ 40°C_Unnamed: 12_level_1",
        "lower_heating_value_MJ_kg": "",
        "cetane_number": "",
        "DCN": "",
        "injection_timing_CAD_BTDC": "",
        "ignition_delay_CAD": "Ignition_Delay_CAD_Constant_Injection_Timing",
        "YSI": "",
        "total_particulate_mass_concentration_mg_m3": "Total_PM_concentration_g_cm3_Constant_Injection_Timing",
    }

    for std_col, src_col in mapping.items():
        if src_col and src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    temp_df["paper"] = source_paper_value

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# --------------------------------
# EXTRACTING DATA FROM YANG SHEET
# --------------------------------

    sheet_name = "Yang"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/fuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
    temp_df = standard_df.copy()

    mapping = {
        "paper": "Source_Paper",
        "name": "Fuel",
        "is_pure": "",
        "smiles": "",
        "mixture_components": "",
        "is_biofuel": "Biofuel or not",
        "fuel_type": "Fuel_Type",
        "feedstock": "Feedstock_Type",
        "subclass": "",
        "density_at_15_kg_m3": "",
        "density_at_20_kg_m3": "",
        "density_at_25_kg_m3": "Density_kg_m3 25",
        "density_at_60_kg_m3": "",
        "dynamic_viscosity_at_20_mPa_s": "",
        "dynamic_viscosity_at_25_mPa_s": "Dynamic Viscosity @25°C (mPa·s)",
        "dynamic_viscosity_at_60_mPa_s": "",
        "kinematic_viscosity_at_20_cST": "",
        "lower_heating_value_MJ_kg": "",
        "cetane_number": "",
        "DCN": "",
        "injection_timing_CAD_BTDC": "Injection Timing (CAD bTDC)",
        "ignition_delay_CAD": "Ignition_Delay_CAD",
        "YSI": "",
        "total_particulate_mass_concentration_mg_m3": "",
    }

    for std_col, src_col in mapping.items():
        if src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_fuel_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

#---------------------
# EXTRACTION COMPLETE
#---------------------

print("Data extraction complete.")