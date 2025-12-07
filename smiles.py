# export_smiles.py
import os
import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/salvina2004/biofuel-ml/data/database/database_main.db")
query = """
SELECT SMILES
FROM FUEL
WHERE SMILES IS NOT NULL
"""
df = pd.read_sql_query(query, conn)
df.to_csv("fuel_smiles.smi", index=False, header=False)
