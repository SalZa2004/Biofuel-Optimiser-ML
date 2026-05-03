"""
Applicability Domain Setup for Your Mixture DCN Model

Customized for your model that uses:
- InChI strings (not SMILES)
- 2-11 component mixtures
- Mole fractions
- GNN architecture from solvation_predictor
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path
import sys
import os



from applicability_domain import ApplicabilityDomainChecker


class MixtureDCNEmbeddingExtractor:
    """
    Extract embeddings from your trained mixture DCN model ENSEMBLE.
    
    Uses all 10 models and averages their embeddings for more robust representation.
    """
    
    def __init__(self, model_dir: str, device='cpu'):
        """
        Args:
            model_dir: Directory containing all model .pt files
            device: 'cpu', 'cuda', or 'mps'
        """
        self.device = torch.device(device)
        
        # Import model loading function (aliases already set up above)
        from core.predictors.mixture.solvation_predictor.train.train import load_checkpoint, create_logger
        
        # Find all model files
        model_dir_path = Path(model_dir)
        self.model_files = sorted([f for f in model_dir_path.glob("*.pt")])
        
        if len(self.model_files) == 0:
            raise ValueError(f"No .pt model files found in {model_dir}")
        
        print(f"Found {len(self.model_files)} models in ensemble")
        
        # Initialize containers
        self.models = []
        self.hooks = []  # Initialize BEFORE loading models to prevent AttributeError
        self.embeddings = []
        
        # Load all models
        logging_obj = create_logger("embedding_extractor", ".")
        
        for i, model_path in enumerate(self.model_files):
            print(f"  Loading model {i+1}/{len(self.model_files)}: {model_path.name}")
            
            args = self._create_model_args(str(model_path))
            model = load_checkpoint(str(model_path), args, logger=logging_obj)
            model.to(self.device)
            model.eval()
            
            self.models.append(model)
        
        # Register hooks on all models
        self._register_hooks()
        
        print(f"✓ Loaded {len(self.models)} models")
        print(f"  Device: {self.device}")
    
    def _create_model_args(self, model_path, device=None):
        """Create minimal args needed for model loading / DataPoint / DataTensor."""
        _device = device  # capture for closure

        class ModelArgs:
            def __init__(self):
                self.model_path = model_path
                self.cuda = torch.cuda.is_available() or torch.backends.mps.is_available()

                # Model architecture
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

                self.num_targets = 1
                self.attention = False
                self.postprocess = False
                self.num_features = 0
                self.f_mol_size = 2
                self.num_mols = 11
                self.max_num_mols = 11

                # Additional attributes from inp.py
                self.property = "solvation"
                self.solute = False
                self.add_hydrogens_to_solvent = False
                self.uncertainty = False
                self.ensemble_variance = False
                self.mix = False
                self.morgan_fingerprint = "None"
                self.morgan_bits = 16
                self.morgan_radius = 2

                # Device — propagated so DataPoint/DataTensor match the model's device
                self.device = _device

                # Attention parameters
                self.att_hidden = 200
                self.att_dropout = 0.0
                self.att_bias = False
                self.att_activation = "ReLU"
                self.att_normalize = "sigmoid"
                self.att_first_normalize = False

        return ModelArgs()

    @classmethod
    def from_models(cls, models: list, model_dir: str = ".", device=None):
        """
        Create an extractor from already-loaded models (avoids loading them twice).

        Args:
            models: List of loaded PyTorch model objects (e.g. from MixtureDCNPredictor)
            model_dir: Path used only for args metadata (no files are read)
            device: If None, inferred from the first model's parameters
        """
        instance = cls.__new__(cls)

        # Infer device from the model if not specified
        if device is None:
            try:
                device = next(models[0].parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        instance.device = torch.device(device) if not isinstance(device, torch.device) else device
        instance.model_files = [Path(model_dir)]  # Needed only for _create_model_args path metadata
        instance.models = list(models)
        instance.hooks = []
        instance.embeddings = []

        instance._register_hooks()

        print(f"✓ Embedding extractor reusing {len(instance.models)} pre-loaded models")
        print(f"  Device: {instance.device}")
        return instance
    
                

    
    def _register_hooks(self):
        """Register hooks on all models to capture embeddings."""
        for model_idx, model in enumerate(self.models):
            # Create a separate embeddings list for each model
            model_embeddings = []
            
            # Pre-hook: captures the INPUT to a module (before any transformation).
            # Used on model.ffn to get the mole-fraction-weighted mixture vector
            # produced by mixture_forward() — one vector per mixture, not per molecule.
            def create_pre_hook_fn(emb_list):
                def hook_fn(module, input):
                    inp = input[0] if isinstance(input, tuple) else input
                    emb_list.append(inp.detach().cpu())
                return hook_fn

            # Register hook - prefer ffn pre-hook (true mixture-level representation)
            hook = None
            if hasattr(model, 'ffn'):
                # Captures the mole-fraction-weighted sum of MPN embeddings:
                # shape (N_mixtures, ffn_input_size) — fires once per forward pass
                hook = model.ffn.register_forward_pre_hook(
                    create_pre_hook_fn(model_embeddings)
                )
            elif hasattr(model, 'encoder'):
                hook = model.encoder.register_forward_hook(
                    create_pre_hook_fn(model_embeddings)
                )
            elif hasattr(model, 'predictor'):
                layers = list(model.children())
                hook = layers[-2].register_forward_hook(
                    create_pre_hook_fn(model_embeddings)
                )

            if hook is None:
                available = [name for name, _ in model.named_children()]
                raise RuntimeError(
                    f"Could not find a layer to hook on model {model_idx}. "
                    f"Available top-level children: {available}"
                )

            self.hooks.append(hook)
            self.embeddings.append(model_embeddings)
        
        print(f"  Registered hooks on {len(self.hooks)} models")
    
    def extract_embeddings_from_mixtures(self, 
                                        mixture_data: List[dict]) -> np.ndarray:
        """
        Extract embeddings from mixture dictionaries using ENSEMBLE AVERAGING.
        
        Args:
            mixture_data: List of dicts with keys:
                - 'inchis': List of InChI strings
                - 'fractions': List of mole fractions (N-1 for N components)
        
        Returns:
            embeddings: numpy array of shape (n_mixtures, embedding_dim)
                       (averaged across all models in ensemble)
        """
        from core.predictors.mixture.solvation_predictor.data.data import (
            DataPoint, DatapointList, MolencoderDatabase, DataTensor
        )

        # Clear previous embeddings
        for emb_list in self.embeddings:
            emb_list.clear()

        args = self._create_model_args(str(self.model_files[0]), device=self.device)

        with torch.no_grad():
            # Build DataPoints - one shared MolencoderDatabase caches mol graphs
            mol_encoder_db = MolencoderDatabase()
            datapoints = []
            for mix in mixture_data:
                dp = DataPoint(
                    smiles=mix['inchis'],  # Model accepts InChI in smiles field
                    targets=[0.0],         # Dummy target
                    features=[],
                    molefracs=mix['fractions'],
                    inp=args,
                    mol_encoders=mol_encoder_db,
                )
                datapoints.append(dp)

            data = DatapointList(datapoints)

            # Build tensors list: one DataTensor per molecule position (same as train.py)
            mol_encodings = [[] for _ in range(args.max_num_mols)]
            for d in datapoints:
                encoders = d.get_mol_encoder()
                # Pad short mixtures by repeating the first encoder
                while len(encoders) < args.max_num_mols:
                    encoders.append(encoders[0])
                for pos, enc in enumerate(encoders):
                    mol_encodings[pos].append(enc)

            tensors = [
                DataTensor(mol_enc_list, args, property=args.property)
                for mol_enc_list in mol_encodings
            ]

            # Forward pass through ALL models (hooks will capture MPN embeddings)
            for model in self.models:
                _ = model(data, tensors)
        
        # Collect embeddings from all models
        all_model_embeddings = []
        
        for model_idx, emb_list in enumerate(self.embeddings):
            if len(emb_list) > 0:
                # Concatenate batches for this model
                model_emb = torch.cat(emb_list, dim=0)
                all_model_embeddings.append(model_emb)
        
        if len(all_model_embeddings) == 0:
            raise RuntimeError("No embeddings captured! Check hook registration.")
        
        # Stack and average across models
        # Shape: (n_models, n_samples, embedding_dim)
        stacked_embeddings = torch.stack(all_model_embeddings, dim=0)
        
        # Average across models
        # Shape: (n_samples, embedding_dim)
        averaged_embeddings = stacked_embeddings.mean(dim=0)
        
        print(f"  Extracted embeddings from {len(all_model_embeddings)} models")
        print(f"  Shape per model: {all_model_embeddings[0].shape}")
        print(f"  Averaged shape: {averaged_embeddings.shape}")
        
        return averaged_embeddings.numpy()
    
    def __del__(self):
        """Remove all hooks when done."""
        for hook in self.hooks:
            hook.remove()


def load_training_data_for_ad(csv_path: str, max_samples: int = None) -> List[dict]:
    """
    Load training mixtures in format needed for embedding extraction.
    
    Args:
        csv_path: Path to formatted training CSV
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        List of mixture dictionaries
    """
    # Try different encodings (same as your prediction script)
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            print(f"  Trying encoding: {encoding}")
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"  ✓ Successfully read with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read CSV with any of these encodings: {encodings}")
    
    print(f"  Loaded {len(df)} rows from CSV")
    
    mixtures = []
    
    for idx, row in df.iterrows():
        if max_samples and idx >= max_samples:
            break
        
        # Extract InChI strings (try multiple column name formats)
        inchis = []
        
        # Format 1: "fuel1 inchi", "fuel2 inchi", etc. (with space)
        for i in range(1, 12):
            col = f'fuel{i} inchi'
            if col in df.columns and pd.notna(row.get(col)):
                inchi = str(row[col]).strip()
                if inchi and inchi != 'nan':
                    inchis.append(inchi)
        
        # Format 2: "fuel1_inchi", "fuel2_inchi", etc. (with underscore)
        if len(inchis) == 0:
            for i in range(1, 12):
                col = f'fuel{i}_inchi'
                if col in df.columns and pd.notna(row.get(col)):
                    inchi = str(row[col]).strip()
                    if inchi and inchi != 'nan':
                        inchis.append(inchi)
        
        if len(inchis) == 0:
            continue
        
        # Extract mole fractions (N-1 for N components)
        fractions = []
        
        # Format 1: "molar fraction fuel 1", etc.
        for i in range(1, len(inchis)):
            col = f'molar fraction fuel {i}'
            if col in df.columns and pd.notna(row.get(col)):
                fractions.append(float(row[col]))
        
        # Format 2: "frac_fuel1 (molar)", etc. (from formatted data)
        if len(fractions) == 0:
            for i in range(1, len(inchis)):
                col = f'frac_fuel{i} (molar)'
                if col in df.columns and pd.notna(row.get(col)):
                    fractions.append(float(row[col]))
        
        # Only add if we have fractions
        if len(fractions) == len(inchis) - 1:
            mixtures.append({
                'inchis': inchis,
                'fractions': fractions
            })
    
    print(f"  ✓ Extracted {len(mixtures)} valid mixtures")
    
    return mixtures


def train_ad_checker_for_mixture_model(
    training_csv: str,
    model_dir: str,  # Changed from model_path to model_dir
    output_path: str = "models/mixture_ad_checker.pkl",
    nu: float = 0.02,
    device: str = 'cpu'
):
    """
    Train AD checker for your mixture DCN model ENSEMBLE.
    
    Args:
        training_csv: Path to formatted training data CSV
        model_dir: Directory containing all .pt model files
        output_path: Where to save AD checker
        nu: Outlier fraction for One-Class SVM
        device: 'cpu', 'cuda', or 'mps'
    """
    print("="*70)
    print("TRAINING APPLICABILITY DOMAIN CHECKER FOR MIXTURE DCN MODEL")
    print("="*70)
    
    # Step 1: Load training data
    print("\nStep 1: Loading training data...")
    mixtures = load_training_data_for_ad(training_csv)
    print(f"✓ Loaded {len(mixtures)} training mixtures")
    
    # Step 2: Create embedding extractor with ENSEMBLE
    print("\nStep 2: Loading model ensemble and creating embedding extractor...")
    extractor = MixtureDCNEmbeddingExtractor(model_dir, device=device)
    
    # Step 3: Extract embeddings (averaged across all models)
    print("\nStep 3: Extracting embeddings from training set...")
    print("  (This may take several minutes with 10 models...)")
    
    train_embeddings = extractor.extract_embeddings_from_mixtures(mixtures)
    print(f"✓ Extracted embeddings: shape {train_embeddings.shape}")
    print(f"  (Averaged across {len(extractor.models)} models)")
    
    # Step 4: Train One-Class SVM
    print("\nStep 4: Training One-Class SVM...")
    
    ad_checker = ApplicabilityDomainChecker(
        nu=nu,
        kernel='rbf',
        gamma='scale'
    )
    
    ad_checker.fit(train_embeddings)
    
    # Step 5: Validate on training set
    print("\nStep 5: Validating on training set...")
    
    train_in_domain = ad_checker.is_in_domain(train_embeddings)
    train_confidence = ad_checker.get_confidence_scores(train_embeddings)
    
    print(f"  In domain: {train_in_domain.sum()}/{len(train_in_domain)} ({train_in_domain.mean()*100:.1f}%)")
    print(f"  Mean confidence: {train_confidence.mean():.1f}%")
    print(f"  Confidence range: [{train_confidence.min():.1f}, {train_confidence.max():.1f}]")
    
    # Step 6: Save AD checker
    print("\nStep 6: Saving AD checker...")
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ad_checker.save(output_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nAD checker saved to: {output_path}")
    print(f"Embedding extractor uses: {len(extractor.models)} models from {model_dir}")
    print("\nNext steps:")
    print("  1. Use this AD checker during evolution")
    print("  2. See mixture_evolution_with_ad.py for integration")
    print("="*70)
    
    return ad_checker, extractor


# =============================================================================
# MAIN SCRIPT
# =============================================================================

if __name__ == "__main__":
    # Configuration
    TRAINING_CSV = "data/database/mixture_training_dataset.csv"  # Your formatted training data
    MODEL_DIR = "models/mixture/dcn"  # Directory with all 10 models
    OUTPUT_PATH = "models/mixture_ad_checker.pkl"
    
    # Device selection
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Check if model directory exists
    if not Path(MODEL_DIR).exists():
        print(f"\n✗ ERROR: Model directory not found: {MODEL_DIR}")
        print("Please update MODEL_DIR to point to your DCN models directory")
        sys.exit(1)
    
    model_files = list(Path(MODEL_DIR).glob("*.pt"))
    print(f"Found {len(model_files)} model files in {MODEL_DIR}")
    
    if len(model_files) == 0:
        print(f"\n✗ ERROR: No .pt model files found in {MODEL_DIR}")
        sys.exit(1)
    
    # Train AD checker
    try:
        ad_checker, extractor = train_ad_checker_for_mixture_model(
            training_csv=TRAINING_CSV,
            model_dir=MODEL_DIR,  # Now using directory
            output_path=OUTPUT_PATH,
            nu=0.02,  # 2% outliers
            device=device
        )
        
        print("\n✓ Setup complete! You can now use the AD checker in your evolution code.")
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found")
        print(f"  {e}")
        print("\nPlease update the paths at the bottom of this script:")
        print(f"  TRAINING_CSV = '{TRAINING_CSV}'")
        print(f"  MODEL_DIR = '{MODEL_DIR}'")
        
    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}")
        print(f"  {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# USAGE IN EVOLUTION
# =============================================================================

"""
Once trained, use in your mixture evolution like this:

from mixture_evolution_with_ad import MixtureAwareMolecularEvolutionWithAD

# Load AD checker
ad_checker = ApplicabilityDomainChecker.load('models/mixture_ad_checker.pkl')

# Load embedding extractor (loads ALL 10 models)
extractor = MixtureDCNEmbeddingExtractor(
    'models/mixture/dcn',  # Directory!
    device='cpu'
)

# During evolution, for each batch of new additives:
new_mixtures = [
    {
        'inchis': [additive_inchi] + base_fuel_inchis,
        'fractions': [additive_fraction] + base_fuel_fractions
    }
    for additive_inchi in new_additive_inchis
]

# Extract embeddings (averaged across 10 models!)
embeddings = extractor.extract_embeddings_from_mixtures(new_mixtures)

# Check AD
in_domain = ad_checker.is_in_domain(embeddings)
confidence = ad_checker.get_confidence_scores(embeddings)

# Filter
for i, additive in enumerate(new_additives):
    if not in_domain[i]:
        print(f"{additive}: OUT OF AD! (confidence: {confidence[i]:.1f}%)")
        continue  # Skip this molecule
    
    # Use this molecule (high confidence prediction)
    ...


WHY USE ALL 10 MODELS FOR EMBEDDINGS?
-------------------------------------

Ensemble averaging gives MORE ROBUST embeddings:
- Single model: Embedding might be noisy
- 10 models averaged: Smoother, more stable representation
- Better AD detection: Less false positives/negatives

The embeddings capture the "consensus" of what all 10 models learned!
"""