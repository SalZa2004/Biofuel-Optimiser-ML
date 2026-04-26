import joblib
import numpy as np
from pathlib import Path
import sys
from core.shared_features import FeatureSelector

# --- FIX FOR JOBLIB / PICKLE ---
main_module = sys.modules.get("__main__")
if main_module is not None and not hasattr(main_module, "FeatureSelector"):
    setattr(main_module, "FeatureSelector", FeatureSelector)

class GenericPredictor:
    """Generic predictor that works for any property model."""
    
    # Properties that were trained in log10 space
    LOG_TRANSFORMED_PROPERTIES = {
        'dynamic viscosity',
        'dynamic_viscosity',
        'viscosity',
        'dynamicviscosity'
    }
    
    def __init__(self, model_dir: Path, property_name: str):
        """
        Initialize predictor from a model directory.
        
        Args:
            model_dir: Path to the model directory containing artifacts/
            property_name: Name of the property (for display purposes)
        """
        print(f"Loading {property_name} Predictor...")
        
        model_path = model_dir / "model.joblib"
        selector_path = model_dir / "selector.joblib"
        
        # Load artifacts
        self.model = joblib.load(model_path)
        self.selector = FeatureSelector.load(selector_path)
        self.property_name = property_name
        
        # Check if this property uses log transform
        self.uses_log_transform = property_name.lower() in self.LOG_TRANSFORMED_PROPERTIES
        
        if self.uses_log_transform:
            print(f"  → Using log₁₀ inverse transform")
        
        print(f"✓ {property_name} Predictor ready!")
    
    def predict_from_features(self, X_full):
        """Predict from pre-computed features."""
        if X_full is None or len(X_full) == 0:
            return []
        
        try:
            X_selected = self.selector.transform(X_full)
            predictions = self.model.predict(X_selected)
            
            # Apply inverse transform for log-trained models
            if self.uses_log_transform:
                # Model outputs log₁₀(viscosity), convert back to linear scale
                predictions = np.power(10.0, predictions)
            
            return predictions.tolist()
        except Exception as e:
            print(f"⚠ Warning: {self.property_name} prediction failed: {e}")
            return [None] * len(X_full)