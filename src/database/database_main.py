import sqlite3
from pathlib import Path
import pandas as pd

def create_fuel_database(db_path):
    """
    Create the SQLite database structure for the fuel dataset.

    Args:
        db_path (str or Path): Path to the SQLite database file.
    """

    db_path = Path(db_path)

    schema_sql = """
    DROP TABLE IF EXISTS TARGET;
    DROP TABLE IF EXISTS FUEL;
    DROP TABLE IF EXISTS SOURCE;

    CREATE TABLE SOURCE (
        source_id INTEGER PRIMARY KEY AUTOINCREMENT,
        Source TEXT NOT NULL
    );

    CREATE TABLE FUEL (
        fuel_id INTEGER PRIMARY KEY AUTOINCREMENT,
        Fuel_Name TEXT,
        SMILES TEXT,
        Formula TEXT,
        CAS TEXT,
        SI_Melting_Point REAL,
        EX_Melting_Point REAL,
        SI_Boiling_Point REAL,
        EX_Boiling_Point REAL,
        SI_Enthalpy_of_Vaporization REAL,
        EX_Enthalpy_of_Vaporization REAL,
        SI_Surface_Tension REAL,
        EX_Surface_Tension REAL,
        SI_Dynamic_Viscosity REAL,
        EX_Dynamic_Viscosity REAL,
        SI_Lower_Heating_Value REAL,
        EX_Lower_Heating_Value REAL,
        SI_Liquid_Density REAL,
        EX_Liquid_Density REAL,
        Sources TEXT
    );

    CREATE TABLE TARGET (
        fuel_id INTEGER,
        Original_Scale TEXT,
        Original_YSI REAL,
        Unified_YSI REAL,
        Unified_YSI_Error REAL,
        SI_Yield_Sooting_Index REAL,
        EX_Yield_Sooting_Index REAL,
        Standardised_YSI REAL,
        CN_measurement_type TEXT,
        measured_DCN REAL,
        SI_Cetane_Number REAL,
        EX_Cetane_Number REAL,
        Standardised_DCN REAL,
        FOREIGN KEY (fuel_id) REFERENCES FUEL (fuel_id)
    );
    """

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA foreign_keys = ON;")

        cursor.executescript(schema_sql)

        conn.commit()
        print(f"SQLite database created successfully: {db_path.name}")

    except sqlite3.Error as e:
        print(f"Database creation error: {e}")
        conn.rollback()

    finally:
        if conn:
            conn.close()


def import_merged_csv_to_database(csv_path, db_path):
    """
    Import merged fuel data from CSV into the SQLite database.

    Inserts rows into:
        - SOURCE
        - FUEL
        - TARGET

    Args:
        csv_path (str or Path): Path to merged_fuels.csv
        db_path (str or Path): Path to SQLite database file
    """

    csv_path = Path(csv_path)
    db_path = Path(db_path)

    df = pd.read_csv(csv_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON;")

    # ------------------------------
    # INSERT INTO SOURCE TABLE
    # ------------------------------
    source_set = set()

    for entry in df['Source'].dropna():
        parts = entry.split(",")           # split by comma
        parts = [p.strip() for p in parts] # remove extra spaces
        parts = [p for p in parts if p]    # remove empty entries
        source_set.update(parts)           # add to set (no duplicates)

    for source in source_set:
        cursor.execute(
            "INSERT OR IGNORE INTO SOURCE (Source) VALUES (?);",
            (source,)
        )

    conn.commit()

    # ------------------------------
    # INSERT INTO FUEL TABLE
    # ------------------------------
    for _, row in df.iterrows():

        cursor.execute("""
            INSERT INTO FUEL (
                Fuel_Name, SMILES, Formula, CAS,
                SI_Melting_Point, EX_Melting_Point,
                SI_Boiling_Point, EX_Boiling_Point,
                SI_Enthalpy_of_Vaporization, EX_Enthalpy_of_Vaporization,
                SI_Surface_Tension, EX_Surface_Tension,
                SI_Dynamic_Viscosity, EX_Dynamic_Viscosity,
                SI_Lower_Heating_Value, EX_Lower_Heating_Value,
                SI_Liquid_Density, EX_Liquid_Density,
                Sources
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            row.get("Fuel Name"),
            row.get("SMILES"),
            row.get("Formula"),
            row.get("CAS"),
            row.get("SI_Melting Point"),
            row.get("EX_Melting Point"),
            row.get("SI_Boiling Point"),
            row.get("EX_Boiling Point"),
            row.get("SI_Enthalpy of Vaporization"),
            row.get("EX_Enthalpy of Vaporization"),
            row.get("SI_Surface Tension"),
            row.get("EX_Surface tension"),
            row.get("SI_Dynamic Viscosity"),
            row.get("EX_Dynamic Viscosity"),
            row.get("SI_Lower Heating Value"),
            row.get("EX_Lower Heating Value"),
            row.get("SI_Liquid Density"),
            row.get("EX_Liquid Density"),
            row.get("Source")
        ))

    conn.commit()

    # ---------------------------
    # INSERT INTO TARGET TABLE
    # ---------------------------
    for _, row in df.iterrows():

        # Match correct FUEL record using SMILES
        cursor.execute("SELECT fuel_id FROM FUEL WHERE SMILES = ?;", (row.get("SMILES"),))
        result = cursor.fetchone()

        if not result:
            continue

        fuel_id = result[0]

        cursor.execute("""
            INSERT INTO TARGET (
                fuel_id,
                Original_Scale,
                Original_YSI,
                Unified_YSI,
                Unified_YSI_Error,
                SI_Yield_Sooting_Index,
                EX_Yield_Sooting_Index,
                Standardised_YSI,
                CN_measurement_type,
                measured_DCN,
                SI_Cetane_Number,
                EX_Cetane_Number,
                Standardised_DCN
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            fuel_id,
            row.get("Original Scale"),
            row.get("Original YSI"),
            row.get("Unified YSI"),
            row.get("Unified YSI Error"),
            row.get("SI_Yield Sooting Index"),
            row.get("EX_Yield Sooting Index"),
            row.get("Standardised_YSI"),
            row.get("measurement type"),
            row.get("measured (D)CN"),
            row.get("SI_Cetane Number"),
            row.get("EX_Cetane Number"),
            row.get("Standardised_DCN")
        ))

    conn.commit()
    conn.close()

    print(f"Data imported successfully into {db_path.name}!")


if __name__ == "__main__":

    create_fuel_database("data/database/database_main.db")
    import_merged_csv_to_database("data/processed/complete_pure_data.csv",
                                 "data/database/database_main.db")

    

