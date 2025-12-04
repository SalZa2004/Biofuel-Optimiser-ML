"""
Portable CN Predictor - Self-contained model for HuggingFace deployment
No external dependencies on feature_engineering.py or feature_selection.py needed!
"""

import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


class CNPredictorPortable:
    """
    Complete, portable CN predictor with all preprocessing bundled.
    Use this for deployment - no need for separate feature engineering scripts!
    """
    
    def __init__(self, model_path=None, selector_path=None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model .pkl file
            selector_path: Path to feature_selector.pkl file
        """
        self.model = None
        self.selector = None
        # Don't store descriptor functions - they can't be pickled!
        # We'll regenerate them on-the-fly
        
        # Load model and selector if paths provided
        if model_path:
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        
        if selector_path:
            self.selector = joblib.load(selector_path)
            print(f"✓ Feature selector loaded from {selector_path}")
    
    def _get_descriptor_functions(self):
        """
        Get RDKit descriptor functions (regenerated each time to avoid pickle issues).
        """
        desc_list = Descriptors._descList
        return [d[1] for d in desc_list]
    
    def _morgan_fp_from_mol(self, mol, radius=2, n_bits=2048):
        """Generate Morgan fingerprint from RDKit mol object."""
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = fpgen.GetFingerprint(mol)
        arr = np.array(list(fp.ToBitString()), dtype=int)
        return arr
    
    def _physchem_desc_from_mol(self, mol):
        """Calculate physicochemical descriptors from RDKit mol object."""
        try:
            # Get fresh descriptor functions each time (can't store them due to pickle)
            descriptor_functions = self._get_descriptor_functions()
            desc = np.array([fn(mol) for fn in descriptor_functions], dtype=np.float32)
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
            return desc
        except:
            return None
    
    def _featurize_single(self, smiles):
        """
        Convert single SMILES string to feature vector.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector (Morgan FP + descriptors) or None if invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get Morgan fingerprint (default 2048 bits)
        fp = self._morgan_fp_from_mol(mol, radius=2, n_bits=2048)
        
        # Get physicochemical descriptors
        desc = self._physchem_desc_from_mol(mol)
        
        if fp is None or desc is None:
            return None
        
        # Concatenate: [Morgan bits | Descriptors]
        return np.hstack([fp, desc])
    
    def featurize(self, smiles_input):
        """
        Convert SMILES to features (handles single string or list/array).
        
        Args:
            smiles_input: Single SMILES string, list of SMILES, or numpy array
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        # Handle single string
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
        else:
            smiles_list = smiles_input
        
        # Featurize all valid SMILES
        features = []
        valid_indices = []
        
        for i, smi in enumerate(smiles_list):
            fv = self._featurize_single(smi)
            if fv is not None:
                features.append(fv)
                valid_indices.append(i)
        
        if len(features) == 0:
            raise ValueError("No valid SMILES strings provided!")
        
        X = np.vstack(features)
        
        # Store valid indices in case caller needs them
        self.last_valid_indices = valid_indices
        
        return X
    
    def predict(self, smiles_input):
        """
        Predict CN values from SMILES strings.
        
        Args:
            smiles_input: Single SMILES string or list/array of SMILES
            
        Returns:
            Predicted CN values (numpy array)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded! Initialize with model_path or call load_model()")
        
        # Step 1: Featurize
        X_full = self.featurize(smiles_input)
        
        # Step 2: Apply feature selection (if selector exists)
        if self.selector is not None:
            X_selected = self.selector.transform(X_full)
        else:
            X_selected = X_full
            print("⚠️  Warning: No feature selector loaded, using all features")
        
        # Step 3: Predict
        predictions = self.model.predict(X_selected)
        
        return predictions
    
    def predict_single(self, smiles):
        """
        Convenience method for predicting single SMILES.
        
        Args:
            smiles: Single SMILES string
            
        Returns:
            Single predicted CN value (float)
        """
        predictions = self.predict([smiles])
        return predictions[0]
    
    def load_model(self, model_path):
        """Load trained model from file."""
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    def load_selector(self, selector_path):
        """Load feature selector from file."""
        self.selector = joblib.load(selector_path)
        print(f"✓ Feature selector loaded from {selector_path}")
    
    def save(self, save_path='cn_predictor_complete.pkl'):
        """
        Save complete predictor (includes model + selector + all logic).
        This creates a SINGLE file that's fully portable!
        
        Args:
            save_path: Where to save the complete predictor
        """
        if self.model is None:
            raise RuntimeError("Cannot save - no model loaded!")
        
        joblib.dump(self, save_path)
        print(f"✓ Complete predictor saved to {save_path}")
        print(f"  This file contains everything needed for inference!")
    
    @staticmethod
    def load(filepath='cn_predictor_complete.pkl'):
        """
        Load complete predictor from file.
        
        Args:
            filepath: Path to saved predictor
            
        Returns:
            CNPredictorPortable instance ready for predictions
        """
        predictor = joblib.load(filepath)
        print(f"✓ Complete predictor loaded from {filepath}")
        return predictor


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_create_portable_model():
    """
    Run this ONCE to create your portable model from existing files.
    """
    print("="*60)
    print("CREATING PORTABLE MODEL")
    print("="*60)
    
    # Load your existing model and selector
    predictor = CNPredictorPortable(
        model_path='cn_model.pkl',              # Your trained model
        selector_path='feature_selector.pkl'     # Your feature selector
    )
    
    # Save as single portable file
    predictor.save('cn_predictor_complete.pkl')
    
    print("\n✅ Done! Upload 'cn_predictor_complete.pkl' to HuggingFace")
    
    return predictor


def example_inference():
    """
    How to use the portable model for predictions.
    """
    print("="*60)
    print("INFERENCE EXAMPLE")
    print("="*60)
    
    # Load the complete predictor (one file!)
    predictor = CNPredictorPortable.load('cn_predictor_complete.pkl')
    
    # Single prediction
    smiles = "CCCCCCCCCCCCCCCC"  # hexadecane
    cn = predictor.predict_single(smiles)
    print(f"\nSingle prediction:")
    print(f"  SMILES: {smiles}")
    print(f"  Predicted CN: {cn:.2f}")
    
    # Batch prediction
    smiles_list = [
        "CCCCCCCCCCCCCCCC",    # hexadecane
        "CC(C)CCCCC",           # isoheptane
        "c1ccccc1",             # benzene
    ]
    predictions = predictor.predict(smiles_list)
    
    print(f"\nBatch predictions:")
    for smi, pred in zip(smiles_list, predictions):
        print(f"  {smi[:20]:20s} -> CN: {pred:.2f}")


def example_huggingface_inference():
    """
    Example of how someone would use your model on HuggingFace.
    This is what your README should show!
    """
    print("="*60)
    print("HUGGINGFACE USAGE EXAMPLE")
    print("="*60)
    
    # They just need to download your .pkl file and:
    from huggingface_hub import hf_hub_download
    
    # Download your model (replace with your actual repo)
    # model_path = hf_hub_download(
    #     repo_id="your-username/cn-predictor",
    #     filename="cn_predictor_complete.pkl"
    # )
    
    # For now, use local file
    model_path = "cn_predictor_complete.pkl"
    
    # Load and predict in 2 lines!
    predictor = CNPredictorPortable.load(model_path)
    cn_value = predictor.predict_single("CCCCCCCCCCCCCCCC")
    
    print(f"\n✅ Predicted CN: {cn_value:.2f}")
    print("\nThat's it! Super clean for users 🎉")


if __name__ == "__main__":
    # Step 1: Run this once to create portable model
    predictor = example_create_portable_model()
    
    # Step 2: Test it works
    print("\n" + "="*60)
    example_inference()
    
    # Step 3: Show HuggingFace usage
    print("\n" + "="*60)
    example_huggingface_inference()