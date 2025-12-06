# ============================================================================
# 1. DATA LOADING MODULE
# ============================================================================
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "latest_fuel_database.db")

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
from tqdm import tqdm

# Get descriptor names globally
DESCRIPTOR_NAMES = [d[0] for d in Descriptors._descList]
desc_functions = [d[1] for d in Descriptors._descList]

def morgan_fp_from_mol(mol, radius=2, n_bits=2048):
    """Generate Morgan fingerprint."""
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = fpgen.GetFingerprint(mol)
    arr = np.array(list(fp.ToBitString()), dtype=int)
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
    Featurize a DataFrame or list of SMILES.
    
    Args:
        df: DataFrame with SMILES column, or list of SMILES strings
        smiles_col: Name of SMILES column (if df is DataFrame)
        return_df: If True, return (X, df_valid). If False, return only X
    
    Returns:
        X: Feature matrix
        df_valid: Valid DataFrame (only if return_df=True)
    """
    # Handle different input types
    if isinstance(df, (list, np.ndarray)):
        df = pd.DataFrame({smiles_col: df})
    elif isinstance(df, pd.Series):
        df = pd.DataFrame({smiles_col: df})
    
    features = []
    valid_indices = []
    
    for i, smi in tqdm(enumerate(df[smiles_col]), total=len(df), desc="Featurizing"):
        fv = featurize(smi)
        if fv is not None:
            features.append(fv)
            valid_indices.append(i)
    
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
    
    def transform(self, X):
        """Apply the fitted feature selection to new data."""
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform!")
        
        # Step 1: Split Morgan and descriptors
        X_mfp = X[:, :self.n_morgan]
        X_desc = X[:, self.n_morgan:]
        
        # Step 2: Remove same correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1).values
        X_corr = np.hstack([X_mfp, desc_filtered])
        
        # Step 3: Select same important features
        X_selected = X_corr[:, self.selected_indices]
        
        return X_selected
    
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


# ============================================================================
# 4. TRAINING PIPELINE
# ============================================================================
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, KFold

def prepare_training_data():
    """Load and prepare clean training data."""
    print("\n" + "="*70)
    print("PREPARING TRAINING DATA")
    print("="*70)
    
    # Load raw data
    df = load_raw_data()
    print(f"Raw samples: {len(df)}")
    
    # Featurize
    X, df_valid = featurize_df(df, return_df=True)
    y = df_valid["cn"].values
    
    # Remove any remaining NaN in target
    nan_mask = ~np.isnan(y)
    X = X[nan_mask]
    y = y[nan_mask]
    
    # Remove outliers
    mask = (y >= 0) & (y <= 150)
    X = X[mask]
    y = y[mask]
    
    print(f"Valid samples after cleaning: {len(y)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    return X, y

def objective(trial, X_train, y_train):
    """Optuna objective function for ExtraTrees."""
    
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 10, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 0.8),
        "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.02),
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1
    }
    
    model = ExtraTreesRegressor(**params)
    
    # 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv, 
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    
    # Return mean RMSE
    rmse = np.sqrt(-scores.mean())
    return rmse

def train_and_save_model(n_trials=100, 
                         output_dir='cn_model/artifacts'):
    """
    Complete training pipeline:
    1. Load and featurize data
    2. Fit and save feature selector
    3. Optimize hyperparameters
    4. Train final model with best params
    5. Save model
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    selector_path = os.path.join(output_dir, 'selector.joblib')
    model_path = os.path.join(output_dir, 'model.joblib')
    study_path = os.path.join(output_dir, 'optuna_study.joblib')
    
    # Step 1: Prepare data
    X_full, y = prepare_training_data()
    
    # Step 2: Create and fit feature selector
    print("\n" + "="*70)
    print("FEATURE SELECTION")
    print("="*70)
    
    selector = FeatureSelector(n_morgan=2048, corr_threshold=0.95, top_k=300)
    X_selected = selector.fit_transform(X_full, y)
    
    # Save selector
    selector.save(selector_path)
    
    # Step 3: Hyperparameter optimization
    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name="extratrees_cn_tuning"
    )
    
    print(f"Running {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, X_selected, y),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1
    )
    
    print(f"\n✓ Best CV RMSE: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Step 4: Train final model
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    best_params = study.best_params.copy()
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_selected, y)
    
    # Evaluate
    train_pred = final_model.predict(X_selected)
    train_rmse = np.sqrt(np.mean((y - train_pred)**2))
    train_mae = np.mean(np.abs(y - train_pred))
    
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Train MAE:  {train_mae:.4f}")
    
    # Step 5: Save model
    joblib.dump(final_model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save study for reference
    joblib.dump(study, study_path)
    print(f"✓ Optuna study saved to {study_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Artifacts saved in: {output_dir}/")
    print(f"  - model.joblib")
    print(f"  - selector.joblib")
    print(f"  - optuna_study.joblib")
    
    return final_model, selector, study


# ============================================================================
# 5. PREDICTION CLASS
# ============================================================================

class CetanePredictor:
    """
    Simple predictor class for cetane number prediction.
    
    Usage:
        predictor = CetanePredictor()
        cn_values = predictor.predict(["CCCCCCCC", "CC(C)C"])
    """
    
    def __init__(self, model_path="cn_model/artifacts/model.joblib", 
                 selector_path="cn_model/artifacts/selector.joblib"):
        """Load the trained model and feature selector."""
        print("Loading Cetane Predictor...")
        self.model = joblib.load(model_path)
        self.selector = FeatureSelector.load(selector_path)
        print("✓ Predictor ready!\n")
    
    def predict(self, smiles_list):
        """
        Predict cetane numbers for SMILES strings.
        
        Args:
            smiles_list: Single SMILES string, list of SMILES, or pandas Series
        
        Returns:
            List of predicted CN values
        """
        # Handle single SMILES
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Featurize
        X_full = featurize_df(smiles_list, return_df=False)
        
        if X_full is None:
            print("⚠ Warning: No valid molecules found!")
            return []
        
        # Apply feature selection
        X_selected = self.selector.transform(X_full)
        
        # Predict
        predictions = self.model.predict(X_selected)
        
        return predictions.tolist()
    
    def predict_with_details(self, smiles_list):
        """
        Predict with validation info.
        
        Returns:
            DataFrame with SMILES and predictions
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        # Featurize with validation
        df = pd.DataFrame({"SMILES": smiles_list})
        X_full, df_valid = featurize_df(df, return_df=True)
        
        if X_full is None:
            print("⚠ Warning: No valid molecules found!")
            return pd.DataFrame(columns=["SMILES", "Predicted_CN", "Valid"])
        
        # Apply feature selection
        X_selected = self.selector.transform(X_full)
        
        # Predict
        predictions = self.model.predict(X_selected)
        
        # Create results dataframe
        df_valid["Predicted_CN"] = predictions
        df_valid["Valid"] = True
        
        # Mark invalid molecules
        all_results = pd.DataFrame({"SMILES": smiles_list})
        all_results = all_results.merge(df_valid[["SMILES", "Predicted_CN", "Valid"]], 
                                        on="SMILES", how="left")
        all_results["Valid"] = all_results["Valid"].fillna(False)
        
        return all_results


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Training mode
        print("="*70)
        print("TRAINING MODE")
        print("="*70)
        
        model, selector, study = train_and_save_model(
            n_trials=100,
            output_dir='cn_model/artifacts'
        )
        
    else:
        # Prediction mode - check if model exists
        model_path = "cn_model/artifacts/model.joblib"
        selector_path = "cn_model/artifacts/selector.joblib"
        
        if not os.path.exists(model_path) or not os.path.exists(selector_path):
            print("="*70)
            print("MODEL NOT FOUND")
            print("="*70)
            print("\nNo trained model found. Please train first:")
            print("  python train.py train")
            print("\nThis will create:")
            print("  - cn_model/artifacts/model.joblib")
            print("  - cn_model/artifacts/selector.joblib")
            sys.exit(1)
        
        # Prediction example
        print("="*70)
        print("PREDICTION EXAMPLE")
        print("="*70)
        
        # Create predictor
        predictor = CetanePredictor()
        
        # Example SMILES
        test_smiles = [
            "CC(C)C",      # Isobutane
            "CCCCCCCC",    # Octane
            "C1CCCCC1",    # Cyclohexane
            "INVALID"      # Invalid SMILES
        ]
        
        print("Testing with example SMILES:")
        results = predictor.predict_with_details(test_smiles)
        print("\n" + results.to_string(index=False))
        
        print("\n" + "="*70)
        print("Simple prediction:")
        predictions = predictor.predict(["CCCCCCCC", "CC(C)C"])
        print(f"Octane CN: {predictions[0]:.2f}")
        print(f"Isobutane CN: {predictions[1]:.2f}")
        
        print("\n" + "="*70)
        print("Usage:")
        print("  Training:   python train.py train")
        print("  Prediction: python train.py")
        print("\nIn your code:")
        print("  from train import CetanePredictor")
        print("  predictor = CetanePredictor()")
        print("  predictions = predictor.predict(['CCCCCCCC'])")
        print("="*70)