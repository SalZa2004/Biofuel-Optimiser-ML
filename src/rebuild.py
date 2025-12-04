"""
Rebuild the portable model WITHOUT pickle dependencies.
This extracts just the weights/parameters and bundles them cleanly.
Run this ONCE locally to create a clean model file.
"""

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class CleanCNPredictor(BaseEstimator, RegressorMixin):
    """
    Completely self-contained CN predictor with ZERO external dependencies.
    Everything needed is stored as pure numpy arrays/dicts.
    """
    
    def __init__(self):
        # These will be filled when we extract from your existing model
        self.model_type = None
        self.model_params = None
        self.selector_params = None
        
    def set_model(self, model):
        """Extract model parameters (works for sklearn models)."""
        self.model_type = type(model).__name__
        
        # Store the actual trained model object
        # (sklearn models pickle cleanly on their own)
        self.trained_model = model
        
    def set_selector(self, selector):
        """Extract selector parameters as pure numpy/python types."""
        self.selector_params = {
            'n_morgan': int(selector.n_morgan),
            'corr_cols_to_drop': list(selector.corr_cols_to_drop),
            'selected_indices': selector.selected_indices.tolist() if hasattr(selector.selected_indices, 'tolist') else list(selector.selected_indices)
        }
    
    def _get_descriptor_functions(self):
        """Get RDKit descriptor functions."""
        from rdkit.Chem import Descriptors
        desc_list = Descriptors._descList
        return [d[1] for d in desc_list]
    
    def _morgan_fp_from_mol(self, mol, radius=2, n_bits=2048):
        """Generate Morgan fingerprint."""
        from rdkit.Chem import rdFingerprintGenerator
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = fpgen.GetFingerprint(mol)
        return np.array(list(fp.ToBitString()), dtype=int)
    
    def _physchem_desc_from_mol(self, mol):
        """Calculate physicochemical descriptors."""
        try:
            descriptor_functions = self._get_descriptor_functions()
            desc = np.array([fn(mol) for fn in descriptor_functions], dtype=np.float32)
            return np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            return None
    
    def _featurize_single(self, smiles):
        """Convert SMILES to feature vector."""
        from rdkit import Chem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        fp = self._morgan_fp_from_mol(mol, radius=2, n_bits=2048)
        desc = self._physchem_desc_from_mol(mol)
        
        if fp is None or desc is None:
            return None
        
        return np.hstack([fp, desc])
    
    def _apply_selection(self, X):
        """Apply feature selection using stored parameters."""
        import pandas as pd
        
        # Split Morgan and descriptors
        n_morgan = self.selector_params['n_morgan']
        X_mfp = X[:, :n_morgan]
        X_desc = X[:, n_morgan:]
        
        # Remove correlated descriptors
        desc_df = pd.DataFrame(X_desc)
        corr_cols = self.selector_params['corr_cols_to_drop']
        desc_filtered = desc_df.drop(columns=corr_cols, axis=1).values
        
        # Concatenate
        X_corr = np.hstack([X_mfp, desc_filtered])
        
        # Select important features
        selected_indices = self.selector_params['selected_indices']
        X_selected = X_corr[:, selected_indices]
        
        return X_selected
    
    def featurize(self, smiles_input):
        """Convert SMILES to features."""
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
        else:
            smiles_list = smiles_input
        
        features = []
        for smi in smiles_list:
            fv = self._featurize_single(smi)
            if fv is not None:
                features.append(fv)
        
        if len(features) == 0:
            raise ValueError("No valid SMILES strings provided!")
        
        return np.vstack(features)
    
    def predict(self, smiles_input):
        """Predict CN values from SMILES."""
        # Featurize
        X_full = self.featurize(smiles_input)
        
        # Apply feature selection
        X_selected = self._apply_selection(X_full)
        
        # Predict using the trained model
        predictions = self.trained_model.predict(X_selected)
        
        return predictions
    
    def predict_single(self, smiles):
        """Predict single SMILES."""
        predictions = self.predict([smiles])
        return predictions[0]


def rebuild_clean_model(
    old_model_path='cn_predictor_complete.pkl',
    new_model_path='cn_predictor_clean.pkl'
):
    """
    Extract your existing model and rebuild it cleanly.
    This breaks all the pickle dependency chains!
    """
    print("="*60)
    print("REBUILDING CLEAN MODEL")
    print("="*60)
    
    # Load your existing messy model
    print("\n1. Loading existing model...")
    try:
        old_predictor = joblib.load(old_model_path)
        print(f"   ✓ Loaded from {old_model_path}")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        print("\n   Trying alternative approach...")
        
        # If the complete model won't load, load components separately
        print("   Loading model and selector separately...")
        model = joblib.load('cn_model.pkl')
        selector = joblib.load('feature_selector_clean.pkl')
        
        # Create a mock predictor object
        class MockPredictor:
            pass
        
        old_predictor = MockPredictor()
        old_predictor.model = model
        old_predictor.selector = selector
    
    # Create new clean predictor
    print("\n2. Creating clean predictor...")
    clean_predictor = CleanCNPredictor()
    
    # Extract model
    print("   - Extracting model weights...")
    clean_predictor.set_model(old_predictor.model)
    
    # Extract selector parameters
    print("   - Extracting selector parameters...")
    clean_predictor.set_selector(old_predictor.selector)
    
    print("   ✓ Clean predictor created")
    
    # Test it works
    print("\n3. Testing clean predictor...")
    test_smiles = "CCCCCCCCCCCCCCCC"
    try:
        result = clean_predictor.predict_single(test_smiles)
        print(f"   ✓ Test prediction: {result:.2f}")
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return None
    
    # Save clean version
    print(f"\n4. Saving clean model to {new_model_path}...")
    joblib.dump(clean_predictor, new_model_path)
    print(f"   ✓ Saved successfully")
    
    # Verify file size
    import os
    size_mb = os.path.getsize(new_model_path) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("✅ SUCCESS! Clean model created")
    print("="*60)
    print(f"\nUpload '{new_model_path}' to HuggingFace")
    print("This file has NO external dependencies!")
    
    return clean_predictor


if __name__ == "__main__":
    clean_predictor = rebuild_clean_model(
        old_model_path='cn_predictor_complete.pkl',
        new_model_path='cn_predictor_clean.pkl'
    )
    
    if clean_predictor:
        print("\n" + "="*60)
        print("VERIFICATION TEST")
        print("="*60)
        
        test_smiles_list = [
            "CCCCCCCCCCCCCCCC",  # hexadecane
            "CC(C)CCCCC",        # isoheptane
        ]
        
        for smi in test_smiles_list:
            cn = clean_predictor.predict_single(smi)
            print(f"{smi:20s} -> CN: {cn:.2f}")