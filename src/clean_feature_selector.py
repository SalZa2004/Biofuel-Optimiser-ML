"""
Script to create a clean feature selector without any database dependencies.
Run this ONCE to extract just the important parts from your feature_selector.pkl
"""

import joblib
import numpy as np
import pandas as pd


class CleanFeatureSelector:
    """
    Minimal feature selector with NO dependencies on training code.
    Only stores the essential transformation logic.
    """
    
    def __init__(self, n_morgan=2048, corr_cols_to_drop=None, selected_indices=None):
        """
        Args:
            n_morgan: Number of Morgan fingerprint bits (should be 2048 for your case)
            corr_cols_to_drop: List of descriptor column indices to drop (from correlation filtering)
            selected_indices: Final indices to select after all filtering
        """
        self.n_morgan = n_morgan
        self.corr_cols_to_drop = corr_cols_to_drop or []
        self.selected_indices = selected_indices
        self.is_fitted = True if selected_indices is not None else False
    
    def transform(self, X):
        """
        Apply feature selection to new data.
        
        Args:
            X: Full feature matrix [Morgan FP | Descriptors]
            
        Returns:
            X_selected: Filtered feature matrix
        """
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted!")
        
        # Step 1: Split Morgan and descriptors
        X_mfp = X[:, :self.n_morgan]
        X_desc = X[:, self.n_morgan:]
        
        # Step 2: Remove correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        desc_filtered = desc_df.drop(columns=self.corr_cols_to_drop, axis=1).values
        
        # Step 3: Concatenate
        X_corr = np.hstack([X_mfp, desc_filtered])
        
        # Step 4: Select important features
        X_selected = X_corr[:, self.selected_indices]
        
        return X_selected


def extract_clean_selector(old_selector_path='feature_selector.pkl', 
                           new_selector_path='feature_selector_clean.pkl'):
    """
    Extract a clean selector from your existing one.
    This removes all dependencies on training scripts!
    """
    print("Loading old selector...")
    old_selector = joblib.load(old_selector_path)
    
    # Extract only the essential attributes
    print(f"  n_morgan: {old_selector.n_morgan}")
    print(f"  corr_cols_to_drop: {len(old_selector.corr_cols_to_drop)} columns")
    print(f"  selected_indices: {len(old_selector.selected_indices)} features")
    
    # Create clean version with ONLY the transformation logic
    clean_selector = CleanFeatureSelector(
        n_morgan=old_selector.n_morgan,
        corr_cols_to_drop=old_selector.corr_cols_to_drop,
        selected_indices=old_selector.selected_indices
    )
    
    # Save clean version
    joblib.dump(clean_selector, new_selector_path)
    print(f"\n✓ Clean selector saved to {new_selector_path}")
    print(f"  This has NO dependencies on your training code!")
    
    return clean_selector


def verify_clean_selector(clean_selector_path='feature_selector_clean.pkl',
                          test_data_shape=(10, 2256)):  # 2048 Morgan + 208 descriptors
    """
    Verify the clean selector works without any imports from training code.
    """
    print("\nVerifying clean selector...")
    
    # Create dummy data
    X_test = np.random.randn(*test_data_shape)
    
    # Load clean selector
    selector = joblib.load(clean_selector_path)
    
    # Transform
    X_selected = selector.transform(X_test)
    
    print(f"✓ Input shape: {X_test.shape}")
    print(f"✓ Output shape: {X_selected.shape}")
    print(f"✓ No database connection needed!")
    
    return X_selected


if __name__ == "__main__":
    print("="*60)
    print("CLEANING FEATURE SELECTOR")
    print("="*60)
    
    # Step 1: Extract clean selector
    clean_selector = extract_clean_selector(
        old_selector_path='feature_selector.pkl',
        new_selector_path='feature_selector_clean.pkl'
    )
    
    # Step 2: Verify it works
    print("\n" + "="*60)
    verify_clean_selector('feature_selector_clean.pkl')
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print("\nNow use 'feature_selector_clean.pkl' instead of 'feature_selector.pkl'")
    print("This clean version has NO dependencies on training scripts!")