import numpy as np
from predictors.generic import GenericPredictor
from shared_features import featurize_df
from config import EvolutionConfig
from typing import List, Dict, Optional, Tuple, Callable
from predictors.hf_models import load_models


PREDICTOR_PATHS = load_models()
class PropertyPredictor:
    """Handles batch prediction for all molecular properties (OPTIMIZED)."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        
        # Initialize only the predictors we need
        self.predictors = {}
        
        # Always need CN predictor
        self.predictors['cn'] = GenericPredictor(
            PREDICTOR_PATHS['cn'], 
            'Cetane Number'
        )
        
        # Conditional predictors
        if config.minimize_ysi:
            self.predictors['ysi'] = GenericPredictor(
                PREDICTOR_PATHS['ysi'], 
                'YSI'
            )
        
        
        # Define validation rules
        self.validators = {
            'bp': lambda v: self.config.min_bp <= v <= self.config.max_bp,
            'density': lambda v: v > self.config.min_density,
            'lhv': lambda v: v > self.config.min_lhv,
            'dynamic_viscosity': lambda v: self.config.min_dynamic_viscosity < v <= self.config.max_dynamic_viscosity
        }
    
    def _safe_predict(self, predictions: List) -> List[Optional[float]]:
        """Safely convert predictions, handling None/NaN/inf values."""
        return [
            float(pred) if pred is not None and np.isfinite(pred) else None
            for pred in predictions
        ]
    
    def predict_all_properties(self, smiles_list: List[str]) -> Dict[str, List[Optional[float]]]:
        """
        Predict all properties for a batch of SMILES (OPTIMIZED).
        Featurizes ONCE and reuses features for all predictors.
        """
        if not smiles_list:
            return {prop: [] for prop in self.predictors.keys()}
        
        # OPTIMIZATION: Featurize only once per batch
        X_full = featurize_df(smiles_list, return_df=False)
        
        if X_full is None:
            return {prop: [None] * len(smiles_list) for prop in self.predictors.keys()}
        
        # Predict all properties using the same features
        results = {}
        for prop_name, predictor in self.predictors.items():
            predictions = predictor.predict_from_features(X_full)
            results[prop_name] = self._safe_predict(predictions)
        
        return results
    
    def is_valid(self, name, value):
        if value is None or name not in self.config.filters:
            return True
        lo, hi = self.config.filters[name]
        if lo is not None and value < lo:
            return False
        if hi is not None and value > hi:
            return False
        return True
