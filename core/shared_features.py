import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "database_main.db")

def load_raw_data():
    """Load raw data from database."""
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
    
    # Clean data
    df.dropna(subset=["cn", "SMILES"], inplace=True)
    
    return df


# ============================================================================
# 2. FEATURIZATION MODULE
# ============================================================================
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm

# Get descriptor names globally
DESCRIPTOR_NAMES = [d[0] for d in Descriptors._descList]
desc_functions = [d[1] for d in Descriptors._descList]

# Module-level cached generator — avoids re-creating it on every call
_FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_fp_from_mol(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint."""
    fp = _FPGEN.GetFingerprint(mol)
    arr = np.empty(2048, dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr

def physchem_desc_from_mol(mol):
    """Calculate physicochemical descriptors."""
    try:
        desc = np.array([fn(mol) for fn in desc_functions], dtype=np.float32)
        desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        return desc
    except:
        return None

def featurize(smiles):
    """Convert SMILES to feature vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = morgan_fp_from_mol(mol)
    desc = physchem_desc_from_mol(mol)
    
    if fp is None or desc is None:
        return None
    
    return np.hstack([fp, desc])

def featurize_df(df, smiles_col="SMILES", return_df=True):
    """
    Featurize a DataFrame or list of SMILES (vectorized for speed).
    """
    # Handle different input types
    if isinstance(df, (list, np.ndarray)):
        df = pd.DataFrame({smiles_col: df})
    elif isinstance(df, pd.Series):
        df = pd.DataFrame({smiles_col: df})
    
    # Convert all SMILES to molecules in batch
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    
    features = []
    valid_indices = []
    
    # Process valid molecules
    for i, mol in enumerate(tqdm(mols, desc="Featurizing", disable=True)):
        if mol is None:
            continue
            
        try:
            fp = morgan_fp_from_mol(mol)
            desc = physchem_desc_from_mol(mol)
            
            if fp is not None and desc is not None:
                features.append(np.hstack([fp, desc]))
                valid_indices.append(i)
        except:
            continue
    
    if len(features) == 0:
        return (None, None) if return_df else None
    
    X = np.vstack(features)
    
    if return_df:
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        return X, df_valid
    else:
        return X


# ============================================================================
# 3. FEATURE SELECTOR CLASS
# ============================================================================
import joblib

class FeatureSelector:
    """Feature selection pipeline that can be saved and reused."""
    
    def __init__(self, n_morgan=2048, corr_threshold=0.95, top_k=300):
        self.n_morgan = n_morgan
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        
        # Filled during fit()
        self.corr_cols_to_drop = None
        self.selected_indices = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the feature selector on training data."""
        print("\n" + "="*70)
        print("FITTING FEATURE SELECTOR")
        print("="*70)
        
        # Step 1: Split Morgan and descriptors
        X_mfp = X[:, :self.n_morgan]
        X_desc = X[:, self.n_morgan:]
        
        print(f"Morgan fingerprints: {X_mfp.shape[1]}")
        print(f"Descriptors: {X_desc.shape[1]}")
        
        # Step 2: Remove correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        corr_matrix = desc_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        self.corr_cols_to_drop = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        
        print(f"Correlated descriptors removed: {len(self.corr_cols_to_drop)}")
        
        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1).values
        X_corr = np.hstack([X_mfp, desc_filtered])
        
        print(f"Features after correlation filter: {X_corr.shape[1]}")
        
        # Step 3: Feature importance selection
        from sklearn.ensemble import ExtraTreesRegressor
        
        print("Running feature importance selection...")
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_corr, y)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        self.selected_indices = indices[:self.top_k]
        
        print(f"Final selected features: {len(self.selected_indices)}")
        
        self.is_fitted = True
        return self
    
    def _build_absolute_indices(self, n_total_features: int):
        """Precompute absolute column indices into the raw feature array."""
        n_desc = n_total_features - self.n_morgan
        kept_desc_cols = sorted(set(range(n_desc)) - set(int(c) for c in self.corr_cols_to_drop))
        abs_indices = []
        for idx in self.selected_indices:
            if idx < self.n_morgan:
                abs_indices.append(int(idx))
            else:
                abs_indices.append(self.n_morgan + kept_desc_cols[int(idx) - self.n_morgan])
        self._absolute_transform_indices = np.array(abs_indices)

    def transform(self, X):
        """Apply the fitted feature selection to new data."""
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform!")

        if not hasattr(self, '_absolute_transform_indices'):
            self._build_absolute_indices(X.shape[1])

        return X[:, self._absolute_transform_indices]
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath='feature_selector.joblib'):
        """Save the fitted selector."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted selector!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        joblib.dump(self, filepath)
        print(f"✓ Feature selector saved to {filepath}")
    
    @staticmethod
    def load(filepath='feature_selector.joblib'):
        """Load a fitted selector."""
        selector = joblib.load(filepath)
        if not selector.is_fitted:
            raise RuntimeError("Loaded selector is not fitted!")
        print(f"✓ Feature selector loaded from {filepath}")
        return selector
