import joblib
import numpy as np
from shared_features import FeatureSelector
from pathlib import Path

class GenericPredictor:
    """Generic predictor that works for any property model."""
    
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
        
        print(f"✓ {property_name} Predictor ready!")
    
    def predict_from_features(self, X_full):
        """Predict from pre-computed features (OPTIMIZED)."""
        if X_full is None or len(X_full) == 0:
            return []
        
        try:
            X_selected = self.selector.transform(X_full)
            predictions = self.model.predict(X_selected)
            return predictions.tolist()
        except Exception as e:
            print(f"⚠ Warning: {self.property_name} prediction failed: {e}")
            return [None] * len(X_full)