"""
Feature selection pipeline that can be saved and reused for predictions.
"""

import numpy as np
import pandas as pd
import joblib
from feature_engineering import get_training_data

class FeatureSelector:
    """
    Feature selection pipeline that can be saved and reused.
    Handles correlation filtering and importance-based selection.
    """
    
    def __init__(self, n_morgan=512, corr_threshold=0.95, top_k=300):
        self.n_morgan = n_morgan
        self.corr_threshold = corr_threshold
        self.top_k = top_k

        # Filled during fit()
        self.corr_cols_to_drop = None
        self.selected_indices = None
        self.remaining_descriptor_names = None
        self.selected_descriptor_names = None
        self.selected_morgan_bits = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the feature selector on training data.
        
        Args:
            X: Full feature matrix (n_samples, n_features)
            y: Target values
        
        Returns:
            self
        """
        print("Fitting feature selector...")
        
        # Step 1: Split Morgan and descriptors
        X_mfp = X[:, :self.n_morgan]
        X_desc = X[:, self.n_morgan:]
        
        # Step 2: Remove correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        corr_matrix = desc_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        self.corr_cols_to_drop = [
            col for col in upper.columns if any(upper[col] > self.corr_threshold)
        ]
        
        print(f"  Descriptors before: {X_desc.shape[1]}")
        print(f"  Correlated removed: {len(self.corr_cols_to_drop)}")
        
        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1).values
        X_corr = np.hstack([X_mfp, desc_filtered])
        
        print(f"  After correlation filter: {X_corr.shape[1]}")
        
        # Step 3: Feature importance selection
        from sklearn.ensemble import ExtraTreesRegressor
        
        model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_corr, y)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        self.selected_indices = indices[:self.top_k]
        
        print(f"  Final selected features: {len(self.selected_indices)}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply the fitted feature selection to new data.
        
        Args:
            X: Full feature matrix (same structure as training)
        
        Returns:
            X_selected: Filtered feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureSelector must be fitted before transform!")
        
        # Step 1: Split Morgan and descriptors
        X_mfp = X[:, :self.n_morgan]
        X_desc = X[:, self.n_morgan:]
        
        # Step 2: Remove same correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        # Store full descriptor names BEFORE we drop correlated ones
        from feature_engineering import DESCRIPTOR_NAMES
        self.original_descriptor_names = DESCRIPTOR_NAMES

        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1).values
        X_corr = np.hstack([X_mfp, desc_filtered])
        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1)
        self.remaining_descriptor_names = desc_filtered.columns.tolist()

        
        # Step 3: Select same important features
        X_selected = X_corr[:, self.selected_indices]
        
        return X_selected
    
    def fit_transform(self, X, y):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def save(self, filepath='feature_selector.pkl'):
        """Save the fitted selector."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted selector!")
        joblib.dump(self, filepath)
        print(f"Feature selector saved to {filepath}")
    
    @staticmethod
    def load(filepath='feature_selector.pkl'):
        """Load a fitted selector."""
        selector = joblib.load(filepath)
        print(f"Feature selector loaded from {filepath}")
        return selector


# =============================================================================
# USAGE FOR TRAINING
# =============================================================================

def create_and_save_selector():
    """Create selector from training data and save it."""
    print("="*60)
    print("CREATING FEATURE SELECTOR")
    print("="*60)
    
    # Load full training data
    X, y = get_training_data()
    print(f"Full data shape: {X.shape}")
    
    # Create and fit selector
    selector = FeatureSelector(n_morgan=512, corr_threshold=0.95, top_k=300)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Selected data shape: {X_selected.shape}")
    
    # Save selector
    selector.save('feature_selector.pkl')
    
    return selector, X_selected, y


def get_selected_data():
    """
    Get selected features for training.
    Loads cached selector if available.
    """
    try:
        selector = FeatureSelector.load('feature_selector.pkl')
        X, y = get_training_data()
        X_selected = selector.transform(X)
        print(f"Using cached feature selector: {X_selected.shape}")
        return X_selected, y
    except:
        print("No cached selector found. Creating new one...")
        selector, X_selected, y = create_and_save_selector()
        return X_selected, y


# =============================================================================
# USAGE FOR PREDICTION
# =============================================================================

def prepare_prediction_features(smiles_list, selector):
    """
    Prepare features for prediction on new SMILES.
    
    Args:
        smiles_list: List of SMILES strings (or Series)
        selector: The pre-loaded FeatureSelector object (passed as argument)
        
    Returns:
        X_selected: Feature matrix ready for model prediction (numpy array)
    """
    from feature_engineering import featurize_df
    
    # 1. Standardize Input to DataFrame
    if isinstance(smiles_list, (list, np.ndarray, pd.Series)):
        input_df = pd.DataFrame({"SMILES": smiles_list})
    else:
        # If it's already a DataFrame, use it directly
        input_df = smiles_list

    # 2. Calculate Full Features (works on input_df regardless of original type)
    # Assumes featurize_df handles the input format.
    X_full = featurize_df(input_df, return_df=False)

    # 3. Apply the Selector (Use the object passed as an argument)
    if selector is not None and X_full is not None:
        try:
            # This is the feature matrix that matches your model's expected input shape
            X_selected = selector.transform(X_full)
        except Exception as e:
            # Safety fallback: If selection fails (e.g., column mismatch), use the raw features.
            # This is risky, but avoids a crash. The preferred solution is to fix 'featurize_df'
            print(f"⚠ Warning: Feature selection failed ({e}). Returning raw features.")
            X_selected = X_full
    else:
        # If no selector object was passed, return the full feature matrix.
        X_selected = X_full
    
    return X_selected

    



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Create and save selector
    selector, X_selected, y = create_and_save_selector()
    
    print("\n" + "="*60)
    print("TESTING LOAD AND TRANSFORM")
    print("="*60)
    
    # Test loading
    loaded_selector = FeatureSelector.load('feature_selector.pkl')
    
    # Test transform
    X_test, _ = get_training_data()
    X_test_selected = loaded_selector.transform(X_test)
    
    print(f"Test transform shape: {X_test_selected.shape}")
    print("✓ Feature selector working correctly!")