"""
Real Mixture DCN Predictor

Integrates your actual DCN prediction model with the GA.
"""

import pandas as pd
import numpy as np
import os
import sys
from core.predictors.mixture import inp as mixture_inp
import core.predictors.mixture.solvation_predictor as sp

# Alias modules to match old import paths used by the trained models
sys.modules['solvation_predictor.inp'] = mixture_inp
sys.modules['solvation_predictor'] = sp

import torch
from typing import Optional, List, Dict
from pathlib import Path
from core.predictors.mixture.solvation_predictor.data.data import read_data
from core.predictors.mixture.solvation_predictor.train.evaluate import predict


class MixtureDCNPredictor:
    """
    Wrapper for your DCN prediction model.
    
    This adapts your existing DCN predictor for use in the GA.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the DCN predictor.
        
        Args:
            model_dir: Path to trained DCN models
                      If None, uses default location
        """
        if model_dir is None:
            # Default location
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
        
        # Initialize model (lazy loading - will load on first prediction)
        self.models = None
        self.scaler = None
        self.args = None
        self._is_initialized = False
    
    def _initialize_models(self):
        """Lazy initialization of models."""
        if self._is_initialized:
            return
        
        # Fix import path issues
        from core.predictors.mixture import inp as mixture_inp
        import core.predictors.mixture.solvation_predictor as sp
        
        sys.modules['solvation_predictor.inp'] = mixture_inp
        sys.modules['solvation_predictor'] = sp
        
        # Import required functions
        from core.predictors.mixture.solvation_predictor.train.train import load_checkpoint, load_scaler
        
        # Create minimal args object
        self.args = self._create_args()
        self.args.input_file = "dummy.csv"  # can be any placeholder
        self.args.output_dir = "/tmp"
        self.args.model_path_root = self.model_dir
        self.args.model_path = sorted([f for f in os.listdir(self.model_dir) if f.endswith(".pt")])

        # Load all models
        self.models = []
        self.scalers = []
        
        model_files = sorted([f for f in os.listdir(self.model_dir) if f.endswith('.pt')])
        
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            
            # Load scaler
            scaler = load_scaler(model_path)
            self.scalers.append(scaler)
            
            # Load model
            model = load_checkpoint(model_path, self.args, logger=None)
            self.models.append(model)
        
        self._is_initialized = True
        print(f"✓ Loaded {len(self.models)} DCN models")
    
    def _write_temp_mixture_csv(self, rows: List[Dict]) -> str:
        """
        Write mixtures to a temporary CSV in the exact schema expected by read_data.
        """
        df = pd.DataFrame(rows)

        tmp_path = Path(self.model_dir).parent / "_tmp_ga_mixtures.csv"
        df.to_csv(tmp_path, index=False)

        return str(tmp_path)

    
    def _create_args(self):
        """Create arguments object for the model."""
        class Args:
            def __init__(self):
                # Model parameters
                self.max_num_mols = 11  # Max components in a mixture
                self.solute = False
                self.f_mol_size = 2
                self.num_targets = 1
                self.num_features = 0
                
                # Scale parameters
                self.scale = "standard"
                self.property = "solvation"
                
                # MPN parameters
                self.depth = 4
                self.mpn_hidden = 200
                self.mpn_dropout = 0.0
                self.mpn_activation = "LeakyReLU"
                self.mpn_bias = False
                self.aggregation = "mean"
                
                # FFN parameters
                self.ffn_hidden = 500
                self.ffn_num_layers = 4
                self.ffn_dropout = 0.0
                self.ffn_activation = "LeakyReLU"
                self.ffn_bias = True
                
                # Attention parameters
                self.attention = False
                self.att_hidden = 200
                self.att_dropout = 0.0
                self.att_bias = False
                self.att_activation = "ReLU"
                self.att_normalize = "sigmoid"
                self.att_first_normalize = False
                
                # Other
                self.postprocess = False
                self.uncertainty = False
                self.ensemble_variance = False
                self.mix = False
                self.morgan_fingerprint = "None"
                self.add_hydrogens_to_solvent = False
                
                # Device
                self.cuda = torch.cuda.is_available() or torch.backends.mps.is_available()
                if self.cuda:
                    self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
                else:
                    self.device = torch.device('cpu')

                self.input_file = None      # placeholder, updated at prediction
                self.output_dir = None      # placeholder
                self.model_path_root = None # placeholder
                self.model_path = []        # placeholder
            
        return Args()
    
    def predict_mixture_dcn(self,
                        smiles_list: List[str],
                        mole_fractions: List[float]) -> float:

        assert len(smiles_list) == len(mole_fractions)
        assert abs(sum(mole_fractions) - 1.0) < 1e-6

        self._initialize_models()

        from core.predictors.mixture.solvation_predictor.data.data import read_data
        from core.predictors.mixture.solvation_predictor.train.evaluate import predict

        # Build CSV row
        row = {}
        for i, smi in enumerate(smiles_list):
            row[f"fuel{i+1}_inchi"] = smi
            row[f"fuel{i+1}_mol_frac"] = mole_fractions[i]

        csv_path = self._write_temp_mixture_csv([row])

        # Update args
        self.args.data_path = csv_path
        self.args.num_mols = len(smiles_list)

        # Read data the RIGHT way
        data = read_data(self.args)

        preds = []
        for model, scaler in zip(self.models, self.scalers):
            preds_model = predict(model=model, data=data, scaler=scaler, inp=self.args)
            preds.append(preds_model[0][0])

        return float(np.mean(preds))

    
def predict_batch_mixtures(self,
                           additive_smiles_list: List[str],
                           base_smiles: List[str],
                           base_mole_fractions: List[float],
                           additive_fraction: float) -> List[float]:

    assert abs(sum(base_mole_fractions) - 1.0) < 1e-6

    self._initialize_models()

    from core.predictors.mixture.solvation_predictor.data.data import read_data
    from core.predictors.mixture.solvation_predictor.train.evaluate import predict

    rows = []

    base_scale = 1.0 - additive_fraction
    scaled_base_fracs = [f * base_scale for f in base_mole_fractions]

    for add_smi in additive_smiles_list:
        row = {
            "fuel1_inchi": add_smi,
            "fuel1_mol_frac": additive_fraction
        }

        for i, (bs, bf) in enumerate(zip(base_smiles, scaled_base_fracs)):
            row[f"fuel{i+2}_inchi"] = bs
            row[f"fuel{i+2}_mol_frac"] = bf

        rows.append(row)

    csv_path = self._write_temp_mixture_csv(rows)

    self.args.data_path = csv_path
    self.args.num_mols = len(base_smiles) + 1

    data = read_data(self.args)

    all_model_preds = []
    for model, scaler in zip(self.models, self.scalers):
        preds = predict(model=model, data=data, scaler=scaler, inp=self.args)
        all_model_preds.append([p[0] for p in preds])

    # Ensemble average
    final_preds = []
    for i in range(len(additive_smiles_list)):
        final_preds.append(float(np.mean([m[i] for m in all_model_preds])))

    return final_preds



# ============================================================================
# BASE FUEL LIBRARY (Updated with realistic compositions)
# ============================================================================

class BaseFuelLibrary:
    """Library of common base fuels for blending."""
    
    @staticmethod
    def get_fossil_diesel():
        """
        Get representative fossil diesel composition.
        
        Based on typical diesel fuel composition:
        - ~30-40% n-alkanes
        - ~30-40% branched alkanes
        - ~20-30% aromatics
        - Small amount of cycloalkanes
        
        Returns:
            (smiles_list, mole_fractions)
        """
        # Representative components of diesel
        smiles = [
            # n-Alkanes (straight chain)
            "CCCCCCCCCCCCCCCC",              # n-Hexadecane (cetane)
            "CCCCCCCCCCCCCCCCC",             # n-Heptadecane
            "CCCCCCCCCCCCCCCCCC",            # n-Octadecane
            
            # Branched alkanes
            "CC(C)CCCCCCCCCCCC",             # 2-methylpentadecane
            "CCCC(C)CCCCCCCCCC",             # 4-methylpentadecane
            
            # Aromatics
            "c1ccccc1CCCCCCCCCC",            # n-Decylbenzene
            "Cc1ccccc1CCCCCCCCC",            # 1-Methyl-2-nonylbenzene
            
            # Cycloalkanes
            "C1CCCCC1CCCCCCCCCC",            # n-Decylcyclohexane
        ]
        
        fractions = [
            0.15,  # n-C16
            0.10,  # n-C17
            0.10,  # n-C18
            0.15,  # Branched 1
            0.15,  # Branched 2
            0.15,  # Aromatic 1
            0.10,  # Aromatic 2
            0.10,  # Cyclic
        ]
        
        # Normalize to ensure sum = 1.0
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_biodiesel():
        """
        Get representative biodiesel (FAME) composition.
        
        Based on soybean biodiesel composition.
        
        Returns:
            (smiles_list, mole_fractions)
        """
        # Common fatty acid methyl esters
        smiles = [
            "CCCCCCCCCCCCCCCCCC(=O)OC",              # Methyl stearate (C18:0)
            "CCCCCCCCC/C=C/CCCCCCCC(=O)OC",          # Methyl oleate (C18:1)
            "CCCCCC/C=C/C/C=C/CCCCCCC(=O)OC",        # Methyl linoleate (C18:2)
            "CCCCCCCCCCCCCCCC(=O)OC",                # Methyl palmitate (C16:0)
        ]
        
        fractions = [
            0.10,  # Stearate
            0.50,  # Oleate (most common in soy)
            0.35,  # Linoleate
            0.05,  # Palmitate
        ]
        
        total = sum(fractions)
        fractions = [f / total for f in fractions]
        
        return smiles, fractions
    
    @staticmethod
    def get_base_fuel(fuel_type: str):
        """
        Get base fuel by type.
        
        Args:
            fuel_type: "fossil_diesel" or "biodiesel"
        
        Returns:
            (smiles_list, mole_fractions)
        """
        if fuel_type == "fossil_diesel":
            return BaseFuelLibrary.get_fossil_diesel()
        elif fuel_type == "biodiesel":
            return BaseFuelLibrary.get_biodiesel()
        else:
            raise ValueError(f"Unknown fuel type: {fuel_type}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    
    print("="*70)
    print("MIXTURE DCN PREDICTOR - REAL IMPLEMENTATION")
    print("="*70)
    
    # Initialize predictor
    try:
        predictor = MixtureDCNPredictor()
    except Exception as e:
        print(f"\n✗ Error initializing predictor: {e}")
        print("\nMake sure:")
        print("  1. DCN models are in core/predictors/mixture/.../trained_models/DCN/")
        print("  2. All required modules are importable")
        sys.exit(1)
    
    # Get base fuel
    base_smiles, base_fractions = BaseFuelLibrary.get_fossil_diesel()
    
    print("\nBase fuel composition (Fossil Diesel):")
    for smi, frac in zip(base_smiles, base_fractions):
        print(f"  {frac*100:5.2f}%: {smi[:30]}...")
    
    # Test single prediction
    print("\n" + "="*70)
    print("Test 1: Single Mixture Prediction")
    print("="*70)
    
    additive = "CCCCCCCCCCCCCCCCCC(=O)OC"  # Methyl stearate (biodiesel component)
    additive_fraction = 0.15
    
    # Adjust base fractions
    base_fraction = 1.0 - additive_fraction
    adjusted_base = [f * base_fraction for f in base_fractions]
    
    # Predict
    mixture_smiles = [additive] + base_smiles
    mixture_fractions = [additive_fraction] + adjusted_base
    
    print(f"\nMixture composition:")
    print(f"  {additive_fraction*100:5.2f}%: {additive} (additive)")
    for smi, frac in zip(base_smiles[:3], adjusted_base[:3]):
        print(f"  {frac*100:5.2f}%: {smi[:30]}...")
    print(f"  ... (total {len(base_smiles)} base components)")
    
    try:
        dcn = predictor.predict_mixture_dcn(mixture_smiles, mixture_fractions)
        print(f"\n✓ Predicted mixture DCN: {dcn:.2f}")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test batch prediction
    print("\n" + "="*70)
    print("Test 2: Batch Prediction (GA use case)")
    print("="*70)
    
    test_additives = [
        "CCCCCCCCCCCCCCCCCC(=O)OC",              # Methyl stearate
        "CCCCCCCCC/C=C/CCCCCCCC(=O)OC",          # Methyl oleate
        "CCCCCCCCCCCCCCCCCCCC",                  # n-Eicosane
    ]
    
    print(f"\nTesting {len(test_additives)} additives at {additive_fraction*100:.0f}% blend:")
    
    try:
        dcn_values = predictor.predict_batch_mixtures(
            additive_smiles_list=test_additives,
            base_smiles=base_smiles,
            base_mole_fractions=base_fractions,
            additive_fraction=additive_fraction
        )
        
        print("\nResults:")
        for smi, dcn in zip(test_additives, dcn_values):
            print(f"  {smi[:40]:<40} → DCN: {dcn:.2f}")
        
        print(f"\n✓ Batch prediction successful!")
        
    except Exception as e:
        print(f"\n✗ Batch prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Ready for integration with GA!")
    print("="*70)