import pandas as pd
from pathlib import Path


# ---------------------------------------
# CREATING EMPTY STANDARDIZED DATAFRAME 
# ---------------------------------------

def create_empty_dataframe():

    columns = [
        "source_paper",
        "fuel_name",
        "is_biodiesel",
        "fuel_type",
        "feedstock_type",
        "boiling_point",
        "melting_point",
        "flash_point_C",
        "density_kg_m3",
        "dynamic_viscosity_at_20",
        "dynamic_viscosity_at_60",
        "lower_heating_value_MJ_kg",
        "cetane_number",
        "engine_type",
        "engine_speed",
        "injection_pressure_bar",
        "IMEP_bar",
        "CIN_injection_timing_CAD_BTDC",
        "CIN_ignition_delay_CAD",
        "CIN_indicated_thermal_efficiency",
        "CIN_injection_duration",
        "CIN_peak_pressure_bar",
        "CIN_peak_HRR_J_deg",
        "CIN_max_average_temp_K",
        "CIN_NOx_ppm",
        "CIN_CO_ppm",
        "CIN_THC_ppm",
        "CIN_particulate_mass_mg_m3",
        "CIG_ignition_timing_CAD_BTDC",
        "CIG_ignition_delay_CAD",
        "CIG_indicated_thermal_efficiency",
        "CIG_injection_duration",
        "CIG_peak_pressure_bar",
        "CIG_peak_HRR_J_deg",
        "CIG_max_average_temp_K",
        "CIG_NOx_ppm",
        "CIG_CO_ppm",
        "CIG_THC_ppm",
        "CIG_particulate_mass_mg_m3",
        "2_EHN_dosage_ppm",
        "CID_injection_timing_CAD_BTDC",
        "CID_ignition_timing_CAD_BTDC",
        "CID_ignition_delay_CAD",
        "CID_indicated_thermal_efficiency",
        "CID_injection_duration",
        "CID_peak_pressure_bar",
        "CID_peak_HRR_J_deg",
        "CID_max_average_temp_K",
        "CID_NOx_ppm",
        "CID_CO_ppm",
        "CID_THC_ppm",
        "CID_particulate_mass_mg_m3",
    ]

    data_df = pd.DataFrame(columns=columns)

    raw_data_dir = Path("data/processed")
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = raw_data_dir / "biofuel_template.csv"
    data_df.to_csv(output_file, index=False)

    print(f"Standardized CSV template created successfully")

# ---------------------------------------------------------------
# DATA EXTRACTION FROM EXCEL SHEETS INTO STANDARDIZED DATAFRAME
# ---------------------------------------------------------------

def extract_and_append_data(sheet_name, excel_file, template_file, output_file):

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)
    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
    temp_df = standard_df.copy()

    mapping = {
        "source_paper": "Source_Paper",
        "fuel_name": "Fuel",
        "is_biodiesel": "",
        "fuel_type": "",
        "feedstock_type": "Feedstock_Type",
        "boiling_point": "boiling_point_K",
        "melting_point": "",
        "flash_point_C": "flash_point_C",
        "density_kg_m3": "Density_kg_m3",
        "dynamic_viscosity_at_20": "Dynamic Viscosity (mPa·s)",
        "dynamic_viscosity_at_60": "Dynamic_viscosity_at_60 (mPa s)",
        "lower_heating_value_MJ_kg": "Lower_Heating_Value_MJ_kg",
        "cetane_number": "Cetane_Number",
        "engine_type": "Engine_Type",
        "engine_speed": "Engine_Speed_rpm",
        "injection_pressure_bar": "Injection_Pressure_bar",
        "IMEP_bar": "",
        "CIN_injection_timing_CAD_BTDC": "CIN Injection Timing (CAD BTDC)",
        "CIN_ignition_delay_CAD": "CIN Ignition_Delay_CAD",
        "CIN_indicated_thermal_efficiency": "CIN Indicated_thermal_efficiency",
        "CIN_injection_duration": "CIN Injection duration_(mu_s)",
        "CIN_peak_pressure_bar": "CIN Peak_Pressure_bar",
        "CIN_peak_HRR_J_deg": "CIN Peak_HRR_J_deg",
        "CIN_max_average_temp_K": "CIN Max-average_temp_(K)",
        "CIN_NOx_ppm": "CIN NOx_ppm",
        "CIN_CO_ppm": "CIN CO_ppm",
        "CIN_THC_ppm": "CIN THC_ppm",
        "CIN_particulate_mass_mg_m3": "CIN total_particulate_mass_concentration (gm-3)",
        "CIG_ignition_timing_CAD_BTDC": "",
        "CIG_ignition_delay_CAD": "CIG Ignition_Delay_CAD",
        "CIG_indicated_thermal_efficiency": "CIG Indicated_thermal_efficiency",
        "CIG_injection_duration": "CIG Injection duration_(mu_s)",
        "CIG_peak_pressure_bar": "CIG Peak_Pressure_bar",
        "CIG_peak_HRR_J_deg": "CIG Peak_HRR_J_deg",
        "CIG_max_average_temp_K": "CIG Max-average_temp_(K)",
        "CIG_NOx_ppm": "CIG NOx_ppm",
        "CIG_CO_ppm": "CIG CO_ppm",
        "CIG_THC_ppm": "CIG THC_ppm",
        "CIG_particulate_mass_mg_m3": "CIG total_particulate_mass_concentration (gm-3)",
        "2_EHN_dosage_ppm": "",
        "CID_injection_timing_CAD_BTDC": "",
        "CID_ignition_timing_CAD_BTDC": "",
        "CID_ignition_delay_CAD": "",
        "CID_indicated_thermal_efficiency": "",
        "CID_injection_duration": "",
        "CID_peak_pressure_bar": "",
        "CID_peak_HRR_J_deg": "",
        "CID_max_average_temp_K": "",
        "CID_NOx_ppm": "",
        "CID_CO_ppm": "",
        "CID_THC_ppm": "",
        "CID_particulate_mass_mg_m3": "",
    }

    for std_col, src_col in mapping.items():
        if src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")
    

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
    template_file = Path("data/processed/biofuel_template.csv")
    output_file = "data/processed"

    extract_and_append_data(sheet_name, excel_file, template_file, output_file)

# -----------------------------------
# EXTRACTING DATA FROM SALVINA SHEET
# -----------------------------------

    sheet_name = "Intan"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/biofuel_template.csv")
    output_file = "data/processed"

    extract_and_append_data(sheet_name, excel_file, template_file, output_file)

# --------------------------------
# EXTRACTING DATA FROM YANG SHEET
# --------------------------------

    sheet_name = "Yang"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/biofuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
    temp_df = standard_df.copy()

    mapping = {
        "source_paper": "Source_Paper",
        "fuel_name": "Fuel",
        "is_biodiesel": "Biofuel or not",
        "fuel_type": "Fuel_Type",
        "feedstock_type": "Feedstock_Type",
        "boiling_point": "boiling Point (K)",
        "melting_point": "",
        "flash_point_C": "",
        "density_kg_m3": "Density_kg_m3",
        "dynamic_viscosity_at_20": "Dynamic Viscosity @25°C (mPa·s)",
        "dynamic_viscosity_at_60": "",
        "lower_heating_value_MJ_kg": "Heating_Value_MJ_kg",
        "cetane_number": "Cetane_Number",
        "engine_type": "Engine_Type",
        "engine_speed": "Engine_Speed_rpm",
        "injection_pressure_bar": "Injection_Pressure_bar",
        "IMEP_bar": "",
        "CIN_injection_timing_CAD_BTDC": "Injection Timing (CAD bTDC)",
        "CIN_ignition_delay_CAD": "Ignition_Delay_CAD",
        "CIN_indicated_thermal_efficiency": "",
        "CIN_injection_duration": "",
        "CIN_peak_pressure_bar": "Peak_Pressure_bar",
        "CIN_peak_HRR_J_deg": "Heat_Release_Rate_J_deg",
        "CIN_max_average_temp_K": "",
        "CIN_NOx_ppm": "NOx_ppm estimated from graph",
        "CIN_CO_ppm": "CO_ppm",
        "CIN_THC_ppm": "THC_ppm",
        "CIN_particulate_mass_mg_m3": "Particulate_Number",
    }

    for std_col, src_col in mapping.items():
        if src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ------------------------------------
# EXTRACTING DATA FROM MAIMOONA SHEET
# ------------------------------------

    sheet_name = "Maimoona"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/biofuel_template.csv")
    output_file = "data/processed"

    template_path = Path(template_file)
    standard_df = pd.read_csv(template_path)

    temp_df = standard_df.copy()
    df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)

    max_row = len(df_sheet)
    row_1_indices = [i for i in list(range(1, 14)) + list(range(29, 54)) + list(range(81, 109)) if i < max_row]
    row_2_indices = [i for i in list(range(15, 28)) + list(range(55, 80)) + list(range(110, 138)) if i < max_row]

    subset_1_df = df_sheet.iloc[row_1_indices][["Source_Paper", "Fuel", "Biodiesel?", "Fuel_Type", "Boiling_Point_K", 
    "Melting_Point_K", "Density_at_20_kg_m3", "Dynamic_Viscosity_at_20_mPa.s", 
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
        "source_paper": subset_1_df["Source_Paper"].values,
        "fuel_name": subset_1_df["Fuel"].values,
        "is_biodiesel": subset_1_df["Biodiesel?"].values,
        "fuel_type": subset_1_df["Fuel_Type"].values,
        "feedstock_type": "",
        "boiling_point": subset_1_df["Boiling_Point_K"].values,
        "melting_point": subset_1_df["Melting_Point_K"].values,
        "flash_point_C": "",
        "density_kg_m3": subset_1_df["Density_at_20_kg_m3"].values,
        "dynamic_viscosity_at_20": subset_1_df["Dynamic_Viscosity_at_20_mPa.s"].values,
        "dynamic_viscosity_at_60": subset_1_df["Dynamic_Viscosity_at_60_mPa.s"].values,
        "lower_heating_value_MJ_kg": subset_1_df["Lower_Heating_Value_MJ_kg"].values,
        "cetane_number": subset_1_df["Cetane_Number"].values,
        "engine_type": subset_1_df["Engine_Type"].values,
        "engine_speed": subset_1_df["Engine_Speed_rpm"].values,
        "injection_pressure_bar": subset_1_df["Injection_Pressure_bar"].values,
        "IMEP_bar": subset_1_df["IMEP_bar"].values,
        "CIN_injection_timing_CAD_BTDC": subset_1_df["SOI for constant injection timing (CAD BTDC)"].values,
        "CIN_ignition_delay_CAD": subset_1_df["Ignition_Delay_CAD"].values,
        "CIN_indicated_thermal_efficiency": subset_1_df["Indicated_Thermal_Efficiency_%"].values,
        "CIN_injection_duration": subset_1_df["Injection_Duration_μs"].values,
        "CIN_peak_pressure_bar": subset_1_df["Peak_Pressure_bar"].values,
        "CIN_peak_HRR_J_deg": subset_1_df["Peak_Heat_Release_Rate_J_deg"].values,
        "CIN_max_average_temp_K": subset_1_df["Max_Temperature_K"].values,
        "CIN_NOx_ppm": subset_1_df["NOx_ppm "].values,
        "CIN_CO_ppm": subset_1_df["CO_ppm"].values,
        "CIN_THC_ppm": subset_1_df["THC_ppm"].values,
        "CIN_particulate_mass_mg_m3": subset_1_df["Total_Particulate_Mass_Concentration_g_m3"].values,
        "CIG_ignition_timing_CAD_BTDC": subset_2_df["SOC for constant ignition timing (CAD BTDC)"].values,
        "CIG_ignition_delay_CAD": subset_2_df["Ignition_Delay_CAD"].values,
        "CIG_indicated_thermal_efficiency": subset_2_df["Indicated_Thermal_Efficiency_%"].values,
        "CIG_injection_duration": subset_2_df["Injection_Duration_μs"].values,
        "CIG_peak_pressure_bar": subset_2_df["Peak_Pressure_bar"].values,
        "CIG_peak_HRR_J_deg": subset_2_df["Peak_Heat_Release_Rate_J_deg"].values,
        "CIG_max_average_temp_K": subset_2_df["Max_Temperature_K"].values,
        "CIG_NOx_ppm": subset_2_df["NOx_ppm "].values,
        "CIG_CO_ppm": subset_2_df["CO_ppm"].values,
        "CIG_THC_ppm": subset_2_df["THC_ppm"].values,
        "CIG_particulate_mass_mg_m3": subset_2_df["Total_Particulate_Mass_Concentration_g_m3"].values,
        "2_EHN_dosage_ppm": subset_1_df["2_EHN_dosage_ppm"].values,
    })

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ---------------------------------
# EXTRACTING DATA FROM NADIA 2SHEET
# ---------------------------------

    sheet_name = "Nadia 2"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/biofuel_template.csv")
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

    #print(df_sheet.columns.tolist())
    
    df_sheet = df_sheet.iloc[:19]
    temp_df = standard_df.copy()

    mapping = {
        "source_paper": "",
        "fuel_name": "Fuel Type_Unnamed: 0_level_1",
        "is_biodiesel": "Feedstock_Type_Unnamed: 4_level_1",
        "fuel_type": "",
        "feedstock_type": "",
        "boiling_point": "Boiling_Point_K_Unnamed: 5_level_1",
        "melting_point": "",
        "flash_point_C": "",
        "density_kg_m3": "Density_g_cm3_Unnamed: 7_level_1",
        "dynamic_viscosity_at_20": "",
        "dynamic_viscosity_at_60": "",
        "lower_heating_value_MJ_kg": "",
        "cetane_number": "Cetane Number_Unnamed: 9_level_1",
        "engine_type": "Engine_Type_Unnamed: 10_level_1",
        "engine_speed": "",
        "injection_pressure_bar": "",
        "IMEP_bar": "",
        "CIN_injection_timing_CAD_BTDC": "",
        "CIN_ignition_delay_CAD": "Constant injection timing (SOI at 7.5 CAD BTDC)_Average Ignition Delay (CAD)",
        "CIN_indicated_thermal_efficiency": "",
        "CIN_injection_duration": "",
        "CIN_peak_pressure_bar": "",
        "CIN_peak_HRR_J_deg": "PHRR_J_deg_Constant_Injection_Timing",
        "CIN_max_average_temp_K": "Maximum_in-cylinder_global_temperature_T_max_K_Constant_Injection_Timing",
        "CIN_NOx_ppm": "NOx_ppm_Constant_Injection_Timing",
        "CIN_CO_ppm": "",
        "CIN_THC_ppm": "",
        "CIN_particulate_mass_mg_m3": "Total_PM_ug_cc_10^-5_Constant_Injection_Timing",
        "CIG_ignition_timing_CAD_BTDC": "",
        "CIG_ignition_delay_CAD": "Constant ignition timing (SOC at TDC)_Average Ignition Delay (CAD)",
        "CIG_indicated_thermal_efficiency": "",
        "CIG_injection_duration": "",
        "CIG_peak_pressure_bar": "",
        "CIG_peak_HRR_J_deg": "PHRR_J_deg_Constant_Ignition_Timing",
        "CIG_max_average_temp_K": "Maximum_in-cylinder_global_temperature_T_max_K_Constant_Ignition_Timing",
        "CIG_NOx_ppm": "NOx_ppm_Constant_Ignition_Timing",
        "CIG_CO_ppm": "",
        "CIG_THC_ppm": "",
        "CIG_particulate_mass_mg_m3": "Total_PM_ug_cc_10^-5_Constant_Ignition_Timing",
    }

    for std_col, src_col in mapping.items():
        if src_col and src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    temp_df["source_paper"] = source_paper_value

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

# ---------------------------------
# EXTRACTING DATA FROM NADIA 1 SHEET
# ---------------------------------

    sheet_name = "Nadia 1"
    excel_file = "data/raw/latest master 2.xlsx"
    template_file = Path("data/processed/biofuel_template.csv")
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
        "source_paper": "",
        "fuel_name": "Fuel_Type_Unnamed: 0_level_1",
        "is_biodiesel": "Feedstock_Type_Unnamed: 1_level_1",
        "fuel_type": "",
        "feedstock_type": "",
        "boiling_point": "",
        "melting_point": "",
        "flash_point_C": "",
        "density_kg_m3": "Density_g_cm3 @ 15°C_Unnamed: 11_level_1",
        "dynamic_viscosity_at_20": "",
        "dynamic_viscosity_at_60": "",
        "lower_heating_value_MJ_kg": "",
        "cetane_number": "",
        "engine_type": "Engine_Type_Unnamed: 2_level_1",
        "engine_speed": "",
        "injection_pressure_bar": "",
        "IMEP_bar": "",
        "CIN_injection_timing_CAD_BTDC": "",
        "CIN_ignition_delay_CAD": "Ignition_Delay_CAD_Constant_Injection_Timing",
        "CIN_indicated_thermal_efficiency": "",
        "CIN_injection_duration": "",
        "CIN_peak_pressure_bar": "",
        "CIN_peak_HRR_J_deg": "PHRR_J_deg_Constant_Injection_Timing",
        "CIN_max_average_temp_K": "",
        "CIN_NOx_ppm": "NOx_ppm_Constant_Injection_Timing",
        "CIN_CO_ppm": "CO_ppm_Constant_Injection_Timing",
        "CIN_THC_ppm": "THC_ppm_Constant_Injection_Timing",
        "CIN_particulate_mass_mg_m3": "Total_PM_concentration_g_cm3_Constant_Injection_Timing",
        "CIG_ignition_timing_CAD_BTDC": "",
        "CIG_ignition_delay_CAD": "Ignition_Delay_CAD_Constant_Ignition_Timing",
        "CIG_indicated_thermal_efficiency": "",
        "CIG_injection_duration": "",
        "CIG_peak_pressure_bar": "",
        "CIG_peak_HRR_J_deg": "PHRR_J_deg_Constant_Ignition_Timing",
        "CIG_max_average_temp_K": "Maximum_in-cylinder_global_temperature_T_max_K_Constant_Ignition_Timing",
        "CIG_NOx_ppm": "NOx_ppm_Constant_Ignition_Timing",
        "CIG_CO_ppm": "CO_ppm_Constant_Ignition_Timing",
        "CIG_THC_ppm": "THC_ppm_Constant_Ignition_Timing",
        "CIG_particulate_mass_mg_m3": "Total_PM_concentration_g_cm3_Constant_Ignition_Timing",
    }

    for std_col, src_col in mapping.items():
        if src_col and src_col in df_sheet.columns:
            temp_df[std_col] = df_sheet[src_col]
        else:
            temp_df[std_col] = None  

    temp_df["source_paper"] = source_paper_value

    processed_dir = Path(output_file)
    processed_dir.mkdir(parents=True, exist_ok=True)

    sheet_output = processed_dir / f"Standardized_{sheet_name}.csv"
    temp_df.to_csv(sheet_output, index=False)

    print(f"Data from sheet '{sheet_name}' saved to {sheet_output}")

#---------------------
# EXTRACTION COMPLETE
#---------------------

print("Data extraction complete.")