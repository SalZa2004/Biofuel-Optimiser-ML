import sqlite3
from pathlib import Path
import pandas as pd


def create_master_db_structure(db_path):
    """
    Create the master database structure for the biofuel dataset

    Args:
        db_path (Path): Path to the SQLite database file.
    """

    SOURCES_sql = '''
    CREATE TABLE SOURCES (
        source_id INTEGER PRIMARY KEY,
        source_paper TEXT NOT NULL,
        year INTEGER
    );
    '''

    FUEL_TYPES_sql = '''
    CREATE TABLE FUEL_TYPES (
        fuel_id INTEGER PRIMARY KEY,
        source_id INTEGER,
        fuel_name TEXT NOT NULL,
        is_biodiesel TEXT,
        fuel_type TEXT,
        feedstock_type TEXT,
        FOREIGN KEY (source_id) REFERENCES SOURCES(source_id)
    );
    '''

    FUEL_PROPERTIES_sql = '''
    CREATE TABLE FUEL_PROPERTIES (
        fuel_id INTEGER PRIMARY KEY,
        boiling_point REAL,
        melting_point REAL,
        flash_point_C REAL,
        density_kg_m3 REAL,
        dynamic_viscosity_at_20 REAL,
        dynamic_viscosity_at_60 REAL,
        lower_heating_value_MJ_kg REAL,
        cetane_number REAL,
        FOREIGN KEY (fuel_id) REFERENCES FUEL_TYPES(fuel_id)
    );
    '''

    ENGINE_TYPES_sql = '''
    CREATE TABLE ENGINE_TYPES (
        engine_id INTEGER PRIMARY KEY,
        engine_type TEXT,
        engine_speed REAL,
        injection_pressure_bar REAL,
        IMEP_bar REAL
    );
    '''

    TEST_CONDITION_CONSTANT_IGNITION_TIMING_sql = '''
    CREATE TABLE TEST_CONDITION_CONSTANT_IGNITION_TIMING (
        test_id INTEGER PRIMARY KEY,
        fuel_id INTEGER,
        engine_id INTEGER,
        ignition_timing_CAD_BTDC REAL,
        ignition_delay_CAD REAL,
        indicated_thermal_efficiency REAL,
        injection_duration REAL,
        FOREIGN KEY (fuel_id) REFERENCES FUEL_TYPES(fuel_id),
        FOREIGN KEY (engine_id) REFERENCES ENGINE_TYPES(engine_id)
    );
    '''

    TEST_CONDITION_CONSTANT_INJECTION_TIMING_sql = '''
    CREATE TABLE TEST_CONDITION_CONSTANT_INJECTION_TIMING (
        test_id INTEGER PRIMARY KEY,
        fuel_id INTEGER,
        engine_id INTEGER,
        injection_timing_CAD_BTDC REAL,
        ignition_delay_CAD REAL,
        indicated_thermal_efficiency REAL,
        injection_duration REAL,
        FOREIGN KEY (fuel_id) REFERENCES FUEL_TYPES(fuel_id),
        FOREIGN KEY (engine_id) REFERENCES ENGINE_TYPES(engine_id)
    );
    '''

    TEST_CONDITION_CONSTANT_IGNITION_DELAY_sql = '''
    CREATE TABLE TEST_CONDITION_CONSTANT_IGNITION_DELAY (
        test_id INTEGER PRIMARY KEY,
        fuel_id INTEGER,
        engine_id INTEGER,
        injection_timing_CAD_BTDC REAL,
        ignition_timing_CAD_BTDC REAL,
        ignition_delay_CAD REAL,
        indicated_thermal_efficiency REAL,
        injection_duration REAL,
        ehn_dosage_ppm REAL,
        FOREIGN KEY (fuel_id) REFERENCES FUEL_TYPES(fuel_id),
        FOREIGN KEY (engine_id) REFERENCES ENGINE_TYPES(engine_id)
    );
    '''

    TEST_RESULTS_CONSTANT_IGNITION_TIMING_sql = '''
    CREATE TABLE TEST_RESULTS_CONSTANT_IGNITION_TIMING (
        result_id INTEGER PRIMARY KEY,
        test_id INTEGER,
        peak_pressure_bar REAL,
        peak_HRR_J_deg REAL,
        max_average_temp_K REAL,
        NOx_ppm REAL,
        CO_ppm REAL,
        THC_ppm REAL,
        particulate_mass_mg_m3 REAL,
        FOREIGN KEY (test_id) REFERENCES TEST_CONDITION_IGNITION_TIMING(test_id)
    );
    '''

    TEST_RESULTS_CONSTANT_INJECTION_TIMING_sql = '''
    CREATE TABLE TEST_RESULTS_CONSTANT_INJECTION_TIMING (
        result_id INTEGER PRIMARY KEY,
        test_id INTEGER,
        peak_pressure_bar REAL,
        peak_HRR_J_deg REAL,
        max_average_temp_K REAL,
        NOx_ppm REAL,
        CO_ppm REAL,
        THC_ppm REAL,
        particulate_mass_mg_m3 REAL,
        FOREIGN KEY (test_id) REFERENCES TEST_CONDITION_INJECTION_TIMING(test_id)
    );
    '''

    TEST_RESULTS_CONSTANT_IGNITION_DELAY_sql = '''
    CREATE TABLE TEST_RESULTS_CONSTANT_IGNITION_DELAY (
        result_id INTEGER PRIMARY KEY,
        test_id INTEGER,
        peak_pressure_bar REAL,
        peak_HRR_J_deg REAL,
        max_average_temp_K REAL,
        NOx_ppm REAL,
        CO_ppm REAL,
        THC_ppm REAL,
        particulate_mass_mg_m3 REAL,
        FOREIGN KEY (test_id) REFERENCES TEST_CONDITION_IGNITION_DELAY(test_id)
    );
    '''

    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute('PRAGMA foreign_keys = ON;')

        cursor.executescript('''
        DROP TABLE IF EXISTS TEST_RESULTS_CONSTANT_IGNITION_DELAY;
        DROP TABLE IF EXISTS TEST_RESULTS_CONSTANT_INJECTION_TIMING;
        DROP TABLE IF EXISTS TEST_RESULTS_CONSTANT_IGNITION_TIMING;
        DROP TABLE IF EXISTS TEST_CONDITION_CONSTANT_IGNITION_DELAY;
        DROP TABLE IF EXISTS TEST_CONDITION_CONSTANT_INJECTION_TIMING;
        DROP TABLE IF EXISTS TEST_CONDITION_CONSTANT_IGNITION_TIMING;
        DROP TABLE IF EXISTS ENGINE_TYPES;
        DROP TABLE IF EXISTS FUEL_PROPERTIES;
        DROP TABLE IF EXISTS FUEL_TYPES;
        DROP TABLE IF EXISTS SOURCES;
        ''')

        cursor.execute(SOURCES_sql)
        cursor.execute(FUEL_TYPES_sql)
        cursor.execute(FUEL_PROPERTIES_sql)
        cursor.execute(ENGINE_TYPES_sql)
        cursor.execute(TEST_CONDITION_CONSTANT_IGNITION_TIMING_sql)
        cursor.execute(TEST_CONDITION_CONSTANT_INJECTION_TIMING_sql)
        cursor.execute(TEST_CONDITION_CONSTANT_IGNITION_DELAY_sql)
        cursor.execute(TEST_RESULTS_CONSTANT_IGNITION_TIMING_sql)
        cursor.execute(TEST_RESULTS_CONSTANT_INJECTION_TIMING_sql)
        cursor.execute(TEST_RESULTS_CONSTANT_IGNITION_DELAY_sql)

        connection.commit()
        print("Biofuel database created successfully!")

    except sqlite3.Error as e:
        print(f"Error creating the database: {e}")
        if connection:
            connection.rollback()

    finally:
        if connection:
            connection.close()


def insert_standardized_data_into_db(db_path, csv_file):

    df = pd.read_csv(csv_file)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        for _, row in df.iterrows():
            
            cur.execute("""
                SELECT source_id FROM SOURCES WHERE source_paper = ?
            """, (row["source_paper"],))
            source = cur.fetchone()

            if source:
                source_id = source[0]
            else:
                cur.execute("""
                    INSERT INTO SOURCES (source_paper, year)
                    VALUES (?, ?)
                """, (row["source_paper"], None))
                source_id = cur.lastrowid

            engine_type = str(row["engine_type"]).strip().lower() if pd.notna(row["engine_type"]) else None
            engine_speed = round(float(row["engine_speed"]), 2) if pd.notna(row["engine_speed"]) else None
            injection_pressure = round(float(row["injection_pressure_bar"]), 2) if pd.notna(row["injection_pressure_bar"]) else None
            imep = round(float(row["IMEP_bar"]), 2) if pd.notna(row["IMEP_bar"]) else None

            cur.execute("""
                SELECT engine_id FROM ENGINE_TYPES
                WHERE engine_type = ? AND engine_speed IS ? 
                    AND injection_pressure_bar IS ? AND IMEP_bar IS ?
            """, (engine_type, engine_speed, injection_pressure, imep))
            engine = cur.fetchone()

            if engine:
                engine_id = engine[0]
            else:
                cur.execute("""
                    INSERT INTO ENGINE_TYPES (engine_type, engine_speed, injection_pressure_bar, IMEP_bar)
                    VALUES (?, ?, ?, ?)
                """, (engine_type, engine_speed, injection_pressure, imep))
                engine_id = cur.lastrowid

            cur.execute("""
                INSERT INTO FUEL_TYPES (source_id, fuel_name, is_biodiesel, fuel_type, feedstock_type)
                VALUES (?, ?, ?, ?, ?)
            """, (
                source_id,
                row["fuel_name"],
                row["is_biodiesel"],
                row["fuel_type"],
                row["feedstock_type"]
            ))
            fuel_id = cur.lastrowid

            cur.execute("""
                INSERT INTO FUEL_PROPERTIES (
                    fuel_id, boiling_point, melting_point, flash_point_C,
                    density_kg_m3, dynamic_viscosity_at_20, dynamic_viscosity_at_60,
                    lower_heating_value_MJ_kg, cetane_number
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fuel_id,
                row["boiling_point"],
                row["melting_point"],
                row["flash_point_C"],
                row["density_kg_m3"],
                row["dynamic_viscosity_at_20"],
                row["dynamic_viscosity_at_60"],
                row["lower_heating_value_MJ_kg"],
                row["cetane_number"]
            ))

            cur.execute("""
                INSERT INTO TEST_CONDITION_CONSTANT_INJECTION_TIMING (
                    fuel_id, engine_id, injection_timing_CAD_BTDC,
                    ignition_delay_CAD, indicated_thermal_efficiency, injection_duration
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fuel_id,
                engine_id,
                row["CIN_injection_timing_CAD_BTDC"],
                row["CIN_ignition_delay_CAD"],
                row["CIN_indicated_thermal_efficiency"],
                row["CIN_injection_duration"]
            ))
            cin_test_id = cur.lastrowid

            cur.execute("""
                INSERT INTO TEST_RESULTS_CONSTANT_INJECTION_TIMING (
                    test_id, peak_pressure_bar, peak_HRR_J_deg, max_average_temp_K,
                    NOx_ppm, CO_ppm, THC_ppm, particulate_mass_mg_m3
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cin_test_id,
                row["CIN_peak_pressure_bar"],
                row["CIN_peak_HRR_J_deg"],
                row["CIN_max_average_temp_K"],
                row["CIN_NOx_ppm"],
                row["CIN_CO_ppm"],
                row["CIN_THC_ppm"],
                row["CIN_particulate_mass_mg_m3"]
            ))

            cur.execute("""
                INSERT INTO TEST_CONDITION_CONSTANT_IGNITION_TIMING (
                    fuel_id, engine_id, ignition_timing_CAD_BTDC,
                    ignition_delay_CAD, indicated_thermal_efficiency, injection_duration
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fuel_id,
                engine_id,
                row["CIG_ignition_timing_CAD_BTDC"],
                row["CIG_ignition_delay_CAD"],
                row["CIG_indicated_thermal_efficiency"],
                row["CIG_injection_duration"]
            ))
            cig_test_id = cur.lastrowid

            cur.execute("""
                INSERT INTO TEST_RESULTS_CONSTANT_IGNITION_TIMING (
                    test_id, peak_pressure_bar, peak_HRR_J_deg, max_average_temp_K,
                    NOx_ppm, CO_ppm, THC_ppm, particulate_mass_mg_m3
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cig_test_id,
                row["CIG_peak_pressure_bar"],
                row["CIG_peak_HRR_J_deg"],
                row["CIG_max_average_temp_K"],
                row["CIG_NOx_ppm"],
                row["CIG_CO_ppm"],
                row["CIG_THC_ppm"],
                row["CIG_particulate_mass_mg_m3"]
            ))

        conn.commit()
        print(f"Inserted data from {csv_file.name} successfully")

    except Exception as e:
        conn.rollback()
        print(f"Error inserting from {csv_file.name}: {e}")

    finally:
        conn.close()

if __name__ == "__main__":


    db_path = Path("data/database/master_new.db")

    create_master_db_structure(db_path)

    csv_dir = Path("data/processed")

    for csv_file in csv_dir.glob("Cleaned_Standardized_*.csv"):
        insert_standardized_data_into_db(db_path, csv_file)



