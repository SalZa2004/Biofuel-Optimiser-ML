"""
WORKING Direct Datapoint Creation for DCN Prediction

Based on the actual data.py code, creates datapoints without CSV files.
"""

import numpy as np
import torch
from typing import List, Optional
from pathlib import Path
import sys
import os


class MixtureDCNPredictor:
    """
    Direct datapoint creation - properly handles MolencoderDatabase requirement.
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
        
        # Models (lazy loading)
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
        
        # Import required classes
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
    
    def _create_datapoint_direct(self,
                                mixture_smiles: List[str],
                                mixture_fractions: List[float]):
        """
        Create DataPoint directly using the actual DataPoint constructor.
        
        Key insight from data.py:
        DataPoint(smiles, targets, features, molefracs, inp, mol_encoders)
        
        Where mol_encoders is a MolencoderDatabase instance!
        """
        from core.predictors.mixture.solvation_predictor.data.data import DataPoint, MolencoderDatabase
        from rdkit import Chem
        
        # Validate SMILES
        for i, smiles in enumerate(mixture_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES at index {i}: {smiles}")
        
        # Validate fractions
        if abs(sum(mixture_fractions) - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1.0, got {sum(mixture_fractions)}")
        
        # Create MolencoderDatabase (THIS IS THE KEY!)
        mol_encoder_db = MolencoderDatabase()
        
        # DataPoint constructor signature from data.py:
        # __init__(self, smiles, targets, features, molefracs, inp: TrainArgs, mol_encoders: MolencoderDatabase)
        
        targets = [0.0]  # Dummy target (not used during prediction)
        features = []    # No additional features
        molefracs = mixture_fractions[:-1]  # N-1 fractions (last is implicit)
        
        try:
            datapoint = DataPoint(
                smiles=mixture_smiles,
                targets=targets,
                features=features,
                molefracs=molefracs,
                inp=self.args,
                mol_encoders=mol_encoder_db  # Pass the MolencoderDatabase instance
            )
            
            return datapoint
        
        except Exception as e:
            raise ValueError(f"Failed to create DataPoint: {e}")
    
    def predict_mixture_dcn(self,
                           mixture_smiles: List[str],
                           mixture_fractions: List[float]) -> float:
        """
        Predict DCN for a single mixture using direct datapoint creation.
        
        Args:
            mixture_smiles: List of SMILES strings for each component
            mixture_fractions: Mole fractions (must sum to 1.0)
        
        Returns:
            Predicted DCN value
        """
        # Initialize models
        self._initialize_models()
        
        # Import required classes
        from core.predictors.mixture.solvation_predictor.data.data import DatapointList
        from core.predictors.mixture.solvation_predictor.train.evaluate import predict
        
        # Update args for this prediction
        self.args.num_mols = len(mixture_smiles)
        self.args.solvent_headers = [f'fuel{i+1}_inchi' for i in range(len(mixture_smiles))]
        self.args.molefrac_headers = [f'frac_fuel{i+1} (molar)' for i in range(len(mixture_smiles) - 1)]
        self.args.target_headers = ['DCN']
        self.args.features_headers = []
        self.args.solute_headers = []
        
        # Create datapoint directly (NO FILE I/O!)
        datapoint = self._create_datapoint_direct(mixture_smiles, mixture_fractions)
        
        # Update f_mol_size from datapoint
        if not hasattr(self, '_f_mol_size_set'):
            self.args.f_mol_size = datapoint.get_mol_encoder()[0].get_sizes()[2]
            self.args.num_features = len(datapoint.features)
            self._f_mol_size_set = True
        
        # Create DatapointList
        data = DatapointList([datapoint])
        
        # Predict with ensemble
        predictions = []
        
        for model, scaler in zip(self.models, self.scalers):
            if self.args.scale == "standard":
                scaler.transform_standard(data)
            
            preds = predict(model=model, data=data, scaler=scaler, inp=self.args)
            predictions.append(preds[0][0])
        
        # Return ensemble average
        return float(np.mean(predictions))
    
    def predict_batch_mixtures(self,
                              additive_smiles_list: List[str],
                              base_smiles: List[str],
                              base_mole_fractions: List[float],
                              additive_fraction: float,
                              verbose: bool = False) -> List[Optional[float]]:
        """
        Predict DCN for multiple additives (TRUE BATCH PROCESSING).
        
        Args:
            additive_smiles_list: List of additive SMILES
            base_smiles: Base fuel SMILES
            base_mole_fractions: Base fuel mole fractions
            additive_fraction: Additive fraction
            verbose: Print progress
        
        Returns:
            List of DCN predictions
        """
        # Validate
        if abs(sum(base_mole_fractions) - 1.0) > 1e-6:
            raise ValueError(f"Base fractions must sum to 1.0")
        
        # Initialize
        self._initialize_models()
        
        # Import
        from core.predictors.mixture.solvation_predictor.data.data import DatapointList
        from core.predictors.mixture.solvation_predictor.train.evaluate import predict
        
        # Setup args
        num_components = 1 + len(base_smiles)
        self.args.num_mols = num_components
        self.args.solvent_headers = [f'fuel{i+1}_inchi' for i in range(num_components)]
        self.args.molefrac_headers = [f'frac_fuel{i+1} (molar)' for i in range(num_components - 1)]
        self.args.target_headers = ['DCN']
        self.args.features_headers = []
        self.args.solute_headers = []
        
        # Adjust base fractions
        base_ratio = 1.0 - additive_fraction
        adjusted_base_fractions = [f * base_ratio for f in base_mole_fractions]
        
        # Create all datapoints
        datapoints = []
        valid_indices = []
        
        for i, additive_smiles in enumerate(additive_smiles_list):
            try:
                mixture_smiles = [additive_smiles] + base_smiles
                mixture_fractions = [additive_fraction] + adjusted_base_fractions
                
                datapoint = self._create_datapoint_direct(mixture_smiles, mixture_fractions)
                datapoints.append(datapoint)
                valid_indices.append(i)
            
            except Exception as e:
                if verbose:
                    print(f"⚠ Skipping {additive_smiles[:30]}: {str(e)[:50]}")
                continue
        
        if len(datapoints) == 0:
            print("❌ No valid datapoints created")
            return [None] * len(additive_smiles_list)
        
        # Update f_mol_size
        if not hasattr(self, '_f_mol_size_set'):
            self.args.f_mol_size = datapoints[0].get_mol_encoder()[0].get_sizes()[2]
            self.args.num_features = len(datapoints[0].features)
            self._f_mol_size_set = True
        
        # Create DatapointList (BATCH!)
        data = DatapointList(datapoints)
        
        if verbose:
            print(f"  Predicting batch of {len(datapoints)} mixtures...")
        
        # Predict with ensemble (BATCHED!)
        all_predictions = []
        
        for model, scaler in zip(self.models, self.scalers):
            if self.args.scale == "standard":
                scaler.transform_standard(data)
            
            batch_preds = predict(model=model, data=data, scaler=scaler, inp=self.args)
            preds = [p[0] for p in batch_preds]
            all_predictions.append(preds)
        
        # Ensemble average
        ensemble_predictions = np.array(all_predictions).mean(axis=0)
        
        # Map back to original indices
        results = [None] * len(additive_smiles_list)
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = float(ensemble_predictions[i])
        
        if verbose:
            success_rate = len(valid_indices) / len(additive_smiles_list) * 100
            print(f"  ✓ Predicted {len(valid_indices)}/{len(additive_smiles_list)} ({success_rate:.1f}%)")
        
        return results


# Test if it works
if __name__ == "__main__":
    print("="*70)
    print("TESTING DIRECT DATAPOINT CREATION")
    print("="*70)
    
    # Initialize predictor
    predictor = MixtureDCNPredictor()
    
    # Test with simple molecules
    test_smiles = ['CCCCCCCC', 'CCCCCCCCCCCCCCCC']
    test_fractions = [0.5, 0.5]
    
    print(f"\nTest mixture: {test_smiles}")
    print(f"Fractions: {test_fractions}")
    
    try:
        dcn = predictor.predict_mixture_dcn(test_smiles, test_fractions)
        print(f"\n✓ SUCCESS!")
        print(f"Predicted DCN: {dcn:.2f}")
    except Exception as e:
        print(f"\n❌ FAILED:")
        print(f"Error: {e}")
        
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)

