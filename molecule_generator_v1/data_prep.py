import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))   # goes from src/ → project root
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "database_main.db")

TARGET_CN = "cn"      # Cetane number
N_FOLDS = 5
TOP_K = 5
print("Connecting to SQLite database...")
conn = sqlite3.connect(DB_PATH)

query = """
SELECT 
    F.Fuel_Name,
    F.SMILES,
    T.Standardised_DCN AS cn
FROM FUEL F
LEFT JOIN TARGET T ON F.fuel_id = T.fuel_id
"""
df = pd.read_sql_query(query, conn)
conn.close()
df.dropna(subset=[TARGET_CN, "SMILES"], inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(df.head())
print(df.columns)

def load_data():
    return df