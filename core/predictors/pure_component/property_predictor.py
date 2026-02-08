import numpy as np
from .generic import GenericPredictor
from core.shared_features import featurize_df
from core.config import EvolutionConfig
from typing import List, Dict, Optional, Tuple, Callable
from .hf_models import load_models
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

PREDICTOR_PATHS = load_models()
class PropertyPredictor:
    """Handles batch prediction for all molecular properties."""
    
    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config
        
        # Initialize only the predictors we need
        self.predictors = {}
        
        # Always need CN predictor
        self.predictors['cn'] = GenericPredictor(
            PREDICTOR_PATHS['cn'], 
            'Cetane Number'
        )
        
        # Conditional predictors
        if config is None or config.minimize_ysi:
            self.predictors['ysi'] = GenericPredictor(
                PREDICTOR_PATHS['ysi'], 
                'YSI'
            )
        self._init_tanimoto_reference()

        
        if self.config is None:
            self.validators = {}
        else:
        # Define validation rules
            self.validators = {
                'bp': lambda v: self.config.min_bp <= v <= self.config.max_bp,
                'density': lambda v: v > self.config.min_density,
                'lhv': lambda v: v > self.config.min_lhv,
                'dynamic_viscosity': lambda v: self.config.min_dynamic_viscosity < v <= self.config.max_dynamic_viscosity
            }
    def _init_tanimoto_reference(self):
        """
        Load training-set fingerprints for Tanimoto similarity.
        Used as a proxy for prediction confidence.
        """
        # You MUST replace this with your actual training dataset source
        from core.data_prep import df  # or wherever your training df lives

        train_smiles = df["SMILES"].tolist()

        self._train_fps = []
        for s in train_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )
                self._train_fps.append(fp)
    
    def compute_tanimoto(self, smiles_list: List[str]) -> List[Optional[float]]:
        """
        Compute max Tanimoto similarity to training set
        for each SMILES.
        """
        results = []

        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                results.append(None)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=2048
            )

            sims = DataStructs.BulkTanimotoSimilarity(fp, self._train_fps)
            results.append(max(sims))

        return results

    
    def _safe_predict(self, predictions: List) -> List[Optional[float]]:
        """Safely convert predictions, handling None/NaN/inf values."""
        return [
            float(pred) if pred is not None and np.isfinite(pred) else None
            for pred in predictions
        ]
    
    def predict_all_properties(self, smiles_list: List[str]) -> Dict[str, List[Optional[float]]]:
        """
        Predict all properties for a batch of SMILES.
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
        
        results["tanimoto"] = self.compute_tanimoto(smiles_list)
        
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
