"""
Mixture DCN Predictor - CSV-based version

Uses your existing read_data() pipeline instead of direct Datapoint creation.
This is slower but guaranteed to work with your existing code structure.
"""

import pandas as pd
import numpy as np
import os
import sys
import torch
import tempfile
from typing import List
from pathlib import Path


class MixtureDCNPredictor:
    """
    Wrapper for your DCN prediction model.
    
    Uses CSV-based prediction (slower but compatible with your existing code).
    """
    
    def __init__(self, model_dir=None):
        """Initialize the DCN predictor."""
        if model_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_dir = base_dir / "predictors" / "mixture" / "solvation_predictor" / "trained_models" / "DCN"
        
        self.model_dir = str(model_dir)
        
        # Verify models exist
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
        if len(model_files) == 0:
            raise FileNotFoundError(f"No .pt model files found in {self.model_dir}")
        
        print(f"DCN Predictor initialized with {len(model_files)} models")
        
        # Initialize models (lazy loading)
        self.models = None
        self.scalers = None
        self.args = None
        self._is_initialized = False
    
    def _initialize_models(self):
        """Lazy initialization of models."""
        if self._is_initialized:
            return
        
        # Fix import paths
        from core.predictors.mixture import inp as mixture_inp
        import core.predictors.mixture.solvation_predictor as sp
        
        sys.modules['solvation_predictor.inp'] = mixture_inp
        sys.modules['solvation_predictor'] = sp
        
        # Import functions
        from core.predictors.mixture.solvation_predictor.train.train import load_checkpoint, load_scaler
        
        # Create args
        self.args = self._create_args()
        
        # Load models
        self.models = []
        self.scalers = []
        
        model_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.pt')])
        
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            scaler = load_scaler(model_path)
            self.scalers.append(scaler)
            model = load_checkpoint(model_path, self.args, logger=None)
            self.models.append(model)
        
        self._is_initialized = True
        print(f"✓ Loaded {len(self.models)} DCN models")
    
    def _create_args(self):
        """Create arguments object."""
        class Args:
            def __init__(self):
                self.max_num_mols = 11
                self.solute = False
                self.f_mol_size = 2
                self.num_targets = 1
                self.num_features = 0
                self.scale = "standard"
                self.property = "solvation"
                self.depth = 4
                self.mpn_hidden = 200
                self.mpn_dropout = 0.0
                self.mpn_activation = "LeakyReLU"
                self.mpn_bias = False
                self.aggregation = "mean"
                self.ffn_hidden = 500
                self.ffn_num_layers = 4
                self.ffn_dropout = 0.0
                self.ffn_activation = "LeakyReLU"
                self.ffn_bias = True
                self.attention = False
                self.att_hidden = 200
                self.att_dropout = 0.0
                self.att_bias = False
                self.att_activation = "ReLU"
                self.att_normalize = "sigmoid"
                self.att_first_normalize = False
                self.postprocess = False
                self.uncertainty = False
                self.ensemble_variance = False
                self.mix = False
                self.morgan_fingerprint = "None"
                self.add_hydrogens_to_solvent = False
                self.scale_features = False
                self.use_same_scaler_for_features = False
                self.max_molecules = -1
                self.delimiter = ','
                
                # Device
                self.cuda = torch.cuda.is_available() or torch.backends.mps.is_available()
                if self.cuda:
                    self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
                else:
                    self.device = torch.device('cpu')
        
        return Args()
    
    def _create_temp_csv(self, smiles_list: List[str], mole_fractions: List[float]) -> str:
        """Create temporary CSV file for prediction."""
        # Create row
        row = {}
        
        # Add SMILES (using 'inchi' column names as your code does)
        for i, smiles in enumerate(smiles_list):
            row[f'fuel{i+1}_inchi'] = smiles
        
        # Add mole fractions (N-1 fractions)
        for i in range(len(smiles_list) - 1):
            row[f'frac_fuel{i+1} (molar)'] = mole_fractions[i]
        
        # Dummy target
        row['DCN'] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([row])
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
    
    def predict_mixture_dcn(self, 
                           smiles_list: List[str], 
                           mole_fractions: List[float]) -> float:
        """
        Predict DCN for a single mixture.
        
        Args:
            smiles_list: List of SMILES for each component
            mole_fractions: Mole fraction for each component (must sum to 1.0)
        
        Returns:
            predicted_dcn: Derived cetane number
        """
        # Validate
        if len(smiles_list) != len(mole_fractions):
            raise ValueError(f"Length mismatch")
        if abs(sum(mole_fractions) - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0")
        
        # Initialize
        self._initialize_models()
        
        # Import
        from core.predictors.mixture.solvation_predictor.data.data import read_data, DatapointList
        from core.predictors.mixture.solvation_predictor.train.evaluate import predict
        
        # Create temp CSV
        temp_csv = self._create_temp_csv(smiles_list, mole_fractions)
        
        try:
            # Update args for this prediction
            self.args.input_file = temp_csv
            self.args.num_mols = len(smiles_list)
            self.args.solvent_headers = [f'fuel{i+1}_inchi' for i in range(len(smiles_list))]
            self.args.molefrac_headers = [f'frac_fuel{i+1} (molar)' for i in range(len(smiles_list) - 1)]
            self.args.target_headers = ['DCN']
            self.args.features_headers = []
            self.args.solute_headers = []
            
            # Read data
            data = read_data(self.args)
            
            if len(data) == 0:
                raise ValueError(f"Failed to parse SMILES")
            
            # Update f_mol_size
            self.args.f_mol_size = data[0].get_mol_encoder()[0].get_sizes()[2]
            self.args.num_features = len(data[0].features)
            
            data = DatapointList(data)
            
            # Predict with all models
            predictions = []
            
            for model, scaler in zip(self.models, self.scalers):
                if self.args.scale == "standard":
                    scaler.transform_standard(data)
                
                preds = predict(model=model, data=data, scaler=scaler, inp=self.args)
                predictions.append(preds[0][0])
            
            # Return ensemble average
            return float(np.mean(predictions))
        
        finally:
            # Clean up
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    
    def predict_batch_mixtures(self,
                               additive_smiles_list: List[str],
                               base_smiles: List[str],
                               base_mole_fractions: List[float],
                               additive_fraction: float) -> List[float]:
        """
        Predict DCN for multiple additives.
        
        Args:
            additive_smiles_list: List of additive SMILES
            base_smiles: Base fuel SMILES
            base_mole_fractions: Base fuel fractions (sum to 1.0)
            additive_fraction: Additive fraction
        
        Returns:
            List of DCN values (None for failures)
        """
        # Validate
        if abs(sum(base_mole_fractions) - 1.0) > 1e-6:
            raise ValueError("Base fractions must sum to 1.0")
        
        # Adjust base fractions
        base_fraction = 1.0 - additive_fraction
        adjusted_base = [f * base_fraction for f in base_mole_fractions]
        
        # Predict each additive
        results = []
        
        for additive_smiles in additive_smiles_list:
            try:
                mixture_smiles = [additive_smiles] + base_smiles
                mixture_fractions = [additive_fraction] + adjusted_base
                
                dcn = self.predict_mixture_dcn(mixture_smiles, mixture_fractions)
                results.append(dcn)
            except Exception as e:
                # Skip failures
                print(f"⚠ Skipping {additive_smiles[:30]}: {str(e)[:50]}")
                results.append(None)
        
        return results


# Base fuel library
class BaseFuelLibrary:
    """Library of base fuels."""
    
    @staticmethod
    def get_fossil_diesel():
        """Get fossil diesel composition."""
        smiles = [
            "CCCCCCCCCCCCCCCC",
            "CCCCCCCCCCCCCCCCC",
            "CCCCCCCCCCCCCCCCCC",
            "CC(C)CCCCCCCCCCCC",
            "CCCC(C)CCCCCCCCCC",
            "c1ccccc1CCCCCCCCCC",
            "Cc1ccccc1CCCCCCCCC",
            "C1CCCCC1CCCCCCCCCC",
        ]
        
        fractions = [0.15, 0.10, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10]
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_biodiesel():
        """Get biodiesel composition."""
        smiles = [
            "CCCCCCCCCCCCCCCCCC(=O)OC",
            "CCCCCCCCC/C=C/CCCCCCCC(=O)OC",
            "CCCCCC/C=C/C/C=C/CCCCCCC(=O)OC",
            "CCCCCCCCCCCCCCCC(=O)OC",
        ]
        
        fractions = [0.10, 0.50, 0.35, 0.05]
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_base_fuel(fuel_type: str):
        """Get base fuel by type."""
        if fuel_type == "fossil_diesel":
            return BaseFuelLibrary.get_fossil_diesel()
        elif fuel_type == "biodiesel":
            return BaseFuelLibrary.get_biodiesel()
        else:
            raise ValueError(f"Unknown fuel type: {fuel_type}")