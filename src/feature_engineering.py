import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
from tqdm import tqdm
import sqlite3
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdFingerprintGenerator
from tqdm import tqdm
import sqlite3
import os
import pandas as pd
from data_prep import load_data

df = load_data()
def morgan_fp_from_mol(mol, radius=2, n_bits=2048):
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = fpgen.GetFingerprint(mol)
    arr = np.array(list(fp.ToBitString()), dtype=int)
    return arr

desc_list = Descriptors._descList
DESCRIPTOR_NAMES = [d[0] for d in Descriptors._descList]
print(f"Total descriptors available: {len(desc_list)}")
desc_functions = [d[1] for d in desc_list]

def physchem_desc_from_mol(mol):
    try:
        desc = np.array([fn(mol) for fn in desc_functions], dtype=np.float32)
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        return desc
    except:
        return None


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_fp_from_mol(mol)
    desc = physchem_desc_from_mol(mol)
    if fp is None or desc is None:
        return None
    return np.hstack([fp, desc])

def featurize_df(df, smiles_col="SMILES", return_df=True):
    features = []
    valid_indices = []

    for i, smi in tqdm(enumerate(df[smiles_col]), total=len(df)):
        fv = featurize(smi)
        if fv is not None:
            features.append(fv)
            valid_indices.append(i)

    X = np.vstack(features)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    if return_df:
        return X, df_valid      # training mode
    else:
        return X                # prediction mode



# First, load and clean the data
df = load_data()
df.dropna(subset=["cn"], inplace=True)  # Remove rows with NaN in target FIRST

# Then featurize the cleaned data
X, df_valid = featurize_df(df, smiles_col="SMILES")

# Now create y from the validated df
y_cn = df_valid["cn"].values

print("Feature matrix shape:", X.shape)
print("Valid molecules:", len(df_valid))
print("Any NaN in y_cn:", np.isnan(y_cn).any())
print("NaN count in y_cn:", np.isnan(y_cn).sum())
# Remove rows with NaN in y_cn
nan_mask = ~np.isnan(y_cn)
X_clean = X[nan_mask]
y_clean = y_cn[nan_mask]
mask = (y_clean >= 0) & (y_clean <= 150)
X_clean = X_clean[mask]
y_clean = y_clean[mask]

print(f"Removed: {len(y_clean) - len(y_clean)} samples")
print(f"Remaining: {len(y_clean)} samples")



print(f"Removed {np.isnan(y_cn).sum()} NaN values")
print(f"New shapes - X: {X_clean.shape}, y: {y_clean.shape}")

def get_training_data():
    return X_clean, y_clean