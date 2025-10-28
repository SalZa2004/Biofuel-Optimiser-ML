erDiagram
	SOURCES {
		INTEGER source_id PK ""  
		TEXT source_paper  ""  
		INTEGER year  ""  
	}

	FUEL_TYPES {
		INTEGER fuel_id PK ""  
		INTEGER source_id FK ""  
		TEXT fuel_name  ""  
		TEXT is_biodiesel  ""  
		TEXT fuel_type  ""  
		TEXT feedstock_type  ""  
	}

	FUEL_PROPERTIES {
		INTEGER fuel_id PK,FK ""  
		REAL boiling_point_K  ""  
		REAL melting_point_K ""  
		REAL flash_point_C
		REAL density_kg_m3  ""  
		REAL dynamic_viscosity_at_20  ""  
		REAL dynamic_viscosity_at_60  ""  
		REAL lower_heating_value_MJ_kg  ""  
		REAL cetane_number  ""  
	}

	ENGINE_TYPES {
		INTEGER engine_id PK  ""  
		TEXT engine_type  ""  
		REAL engine_speed_rpm ""  
		REAL injection_pressure_bar  ""  
		REAL IMEP_bar  ""  
	}

	TEST_CONDITION_CONSTANT_IGNITION_TIMING {
		INTEGER test_id PK ""  
		INTEGER fuel_id FK ""  
		INTEGER engine_id FK ""  
		REAL injection_timing_CAD_BTDC  ""  
		REAL ignition_delay_CAD_BTDC  ""  
		REAL indicated_thermal_efficiency  "" 
		REAL injection_duration "" 
	}

	TEST_CONDITION_CONSTANT_INJECTION_TIMING {
		INTEGER test_id PK ""  
		INTEGER fuel_id FK ""  
		INTEGER engine_id FK ""  
		REAL ignition_timing_CAD_BTDC  ""  
		REAL ignition_delay_CAD  ""  
		REAL indicated_thermal_efficiency  ""
		REAL injection_duration ""
	}

	TEST_CONDITION_CONSTANT_IGNITION_DELAY {
		INTEGER test_id PK ""  
		INTEGER fuel_id FK ""  
		INTEGER engine_id FK ""  
		REAL injection_timing_CAD_BTDC  "" 
		REAL ignition_timing_CAD_BTDC  ""  
		REAL ignition_delay_CAD  ""  
		REAL indicated_thermal_efficiency  ""
		REAL injection_duration "" 
	}

	TEST_RESULTS_CONSTANT_IGNITION_TIMING {
		INTEGER result_id PK ""  
		INTEGER test_id FK ""  
		REAL peak_pressure_bar  ""  
		REAL peak_HRR_J_deg  "" 
		REAL max_average_temp_K "" 
		REAL NOx_ppm  ""  
		REAL CO_ppm  ""  
		REAL THC_ppm  ""  
		REAL particulate_mass_mg_m3  ""  
	}

	TEST_RESULTS_CONSTANT_INJECTION_TIMING {
		INTEGER result_id PK ""  
		INTEGER test_id FK ""  
		REAL peak_pressure_bar  ""  
		REAL peak_HRR_J_deg  ""  
		REAL max_average_temp_K ""
		REAL NOx_ppm  ""  
		REAL CO_ppm  ""  
		REAL THC_ppm  ""  
		REAL particulate_mass_g_m3  ""  
	}

	TEST_RESULTS_CONSTANT_IGNITION_DELAY {
		INTEGER result_id PK ""  
		INTEGER test_id FK ""  
		REAL peak_pressure_bar  ""  
		REAL peak_HRR_J_deg  ""  
		REAL max_average_temp_K ""
		REAL NOx_ppm  ""  
		REAL CO_ppm  ""  
		REAL THC_ppm  ""  
		REAL particulate_mass_mg_m3  ""  
	}


	SOURCES||--o{FUEL_TYPES:""
	FUEL_TYPES||--||FUEL_PROPERTIES:""
	FUEL_TYPES||--o{TEST_CONDITION_CONSTANT_IGNITION_TIMING:""
	FUEL_TYPES||--o{TEST_CONDITION_CONSTANT_INJECTION_TIMING:""
	FUEL_TYPES||--o{TEST_CONDITION_CONSTANT_IGNITION_DELAY:""
	ENGINE_TYPES||--o{TEST_CONDITION_CONSTANT_IGNITION_TIMING:""
	ENGINE_TYPES||--o{TEST_CONDITION_CONSTANT_INJECTION_TIMING:""
	ENGINE_TYPES||--o{TEST_CONDITION_CONSTANT_IGNITION_DELAY:""
	TEST_CONDITION_CONSTANT_IGNITION_TIMING||--||TEST_RESULTS_CONSTANT_IGNITION_TIMING:""
	TEST_CONDITION_CONSTANT_INJECTION_TIMING||--||TEST_RESULTS_CONSTANT_INJECTION_TIMING:""
	TEST_CONDITION_CONSTANT_IGNITION_DELAY||--||TEST_RESULTS_CONSTANT_IGNITION_DELAY:""

