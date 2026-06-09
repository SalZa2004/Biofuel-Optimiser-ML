"""
SIMPLE Step-by-Step Mixture Embedding Extractor

For training One-Class SVM on your mixture training data.
"""

import numpy as np
import torch
from typing import List
import pandas as pd


def extract_mixture_embeddings_simple(
    csv_path: str = "data/database/mixture_training_dataset.csv",
    model_dir: str = "models/mixture/dcn"
):
    """
    Extract mixture embeddings from your training data in 5 simple steps.
    
    Args:
        csv_path: Path to mixture_training_dataset.csv
        model_dir: Path to trained models directory
    
    Returns:
        X_train: (n_mixtures, embedding_dim) numpy array for your SVM
    """
    
    print("="*70)
    print("STEP-BY-STEP MIXTURE EMBEDDING EXTRACTION")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Load your mixture DCN predictor
    # =========================================================================
    print("\nStep 1: Loading DCN predictor...")
    
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    
    predictor = MixtureDCNPredictor(model_dir=model_dir)
    predictor._initialize_models()
    
    print(f"✓ Loaded {len(predictor.models)} models")
    
    # =========================================================================
    # STEP 2: Load your training data from CSV
    # =========================================================================
    print(f"\nStep 2: Loading training data from CSV...")
    
    # Try different encodings
    df = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"✓ Read CSV with {encoding} encoding")
            break
        except:
            continue
    
    if df is None:
        raise ValueError("Could not read CSV with any encoding")
    
    print(f"✓ Loaded {len(df)} rows from CSV")
    
    # =========================================================================
    # STEP 3: Convert CSV rows to mixture format
    # =========================================================================
    print("\nStep 3: Converting CSV to mixture format...")
    
    mixtures = []
    
    for idx, row in df.iterrows():
        # Extract InChI strings from fuel1 inchi, fuel2 inchi, etc.
        inchis = []
        for i in range(1, 12):  # Check up to fuel11
            col = f'fuel{i} inchi'
            if col in df.columns and pd.notna(row.get(col)):
                inchi = str(row[col]).strip()
                # Clean up InChI (remove quotes if present)
                inchi = inchi.strip('"').strip("'")
                if inchi and inchi != 'nan' and inchi != '':
                    inchis.append(inchi)
        
        if len(inchis) == 0:
            continue
        
        # Extract mole fractions from "molar fraction fuel 1", etc.
        fractions = []
        for i in range(1, len(inchis)):  # N-1 fractions
            col = f'molar fraction fuel {i}'
            if col in df.columns and pd.notna(row.get(col)):
                frac = float(row[col])
                fractions.append(frac)
            else:
                # If fraction column doesn't exist, skip this mixture
                break
        
        # Only add if we have the correct number of fractions (N-1 for N components)
        if len(fractions) == len(inchis) - 1:
            mixtures.append({
                'inchis': inchis,
                'fractions': fractions
            })
        elif len(inchis) == 2 and len(fractions) == 1:
            # Special case: binary mixture with 1 fraction
            mixtures.append({
                'inchis': inchis,
                'fractions': fractions
            })
    
    print(f"✓ Converted {len(mixtures)} valid mixtures")
    print(f"  Example mixture: {len(mixtures[0]['inchis'])} components")
    
    # =========================================================================
    # STEP 4: Extract embeddings using a hook
    # =========================================================================
    print("\nStep 4: Extracting mixture embeddings...")
    print("  (This is where we hook into the model)")
    
    all_embeddings = []
    
    # We'll extract from just the first model for simplicity
    # (You can average across all 10 later if you want)
    model = predictor.models[0]
    model.eval()
    
    # Hook to capture the mixture embedding
    captured_embeddings = []
    
    def hook_fn(module, input, output):
        """This captures the mixture embedding AFTER pooling."""
        # The input to FFN is the mixture embedding we want!
        captured_embeddings.append(input[0].detach().cpu())
    
    # Register hook on the FFN (it receives the mixture embedding as input)
    hook = model.ffn.register_forward_hook(hook_fn)
    
    # Process in batches
    batch_size = 50
    
    from core.predictors.mixture.solvation_predictor.data.data import (
        DataPoint, DatapointList, MolencoderDatabase, DataTensor
    )
    
    for batch_start in range(0, len(mixtures), batch_size):
        batch_end = min(batch_start + batch_size, len(mixtures))
        batch_mixtures = mixtures[batch_start:batch_end]
        
        # Create datapoints
        datapoints = []
        mol_encoder_db = MolencoderDatabase()
        
        for mix in batch_mixtures:
            try:
                dp = DataPoint(
                    smiles=mix['inchis'],
                    targets=[0.0],
                    features=[],
                    molefracs=mix['fractions'],
                    inp=predictor.args,
                    mol_encoders=mol_encoder_db
                )
                datapoints.append(dp)
            except Exception as e:
                # Skip invalid mixtures
                continue
        
        if len(datapoints) == 0:
            continue
        
        data = DatapointList(datapoints)
        
        # Create tensors (same as in evaluate.py)
        mol_encodings = []
        tensors = []
        for molecule in range(predictor.args.max_num_mols):
            mol_encodings.append([])
        
        for mol in mol_encodings:
            for d in datapoints:
                encoders = d.get_mol_encoder()
                if len(encoders) < predictor.args.max_num_mols:
                    for j in range(predictor.args.max_num_mols - len(encoders)):
                        encoders.append(encoders[0])
                mol.append(encoders[mol_encodings.index(mol)])
            tensors.append(DataTensor(mol, predictor.args, property=predictor.args.property))
        
        # Forward pass (hook will capture the mixture embedding)
        with torch.no_grad():
            _ = model(data, tensors)
        
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"    Processed {batch_end}/{len(mixtures)} mixtures...")
    
    # Remove hook
    hook.remove()
    
    # Stack all embeddings
    X_train = torch.cat(captured_embeddings, dim=0).numpy()
    
    print(f"✓ Extracted {X_train.shape[0]} embeddings")
    print(f"  Embedding dimension: {X_train.shape[1]}")
    
    # =========================================================================
    # STEP 5: Return X_train for your SVM
    # =========================================================================
    print("\nStep 5: Ready for SVM training!")
    print(f"  X_train shape: {X_train.shape}")
    print("="*70)
    
    return X_train


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Extract embeddings
    X_train = extract_mixture_embeddings_simple()
    
    print("\n" + "="*70)
    print("TRAINING ONE-CLASS SVM")
    print("="*70)
    
    # Your code from above!
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np

# STEP 1: Normalize the embeddings (IMPORTANT!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# STEP 2: Train SVM with better parameters
svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',  # Auto-calculate based on data (much better!)
    nu=0.02        # 2% outliers (from the paper)
)

svm.fit(X_train_scaled)

# STEP 3: Check the scores
scores = svm.decision_function(X_train_scaled)

print(f"\nDecision function scores:")
print(f"  Min: {scores.min():.3f}")
print(f"  Max: {scores.max():.3f}")
print(f"  Mean: {scores.mean():.3f}")
print(f"  Std: {scores.std():.3f}")

# STEP 4: Check predictions
predictions = svm.predict(X_train_scaled)
n_outliers = (predictions == -1).sum()
print(f"\nOutliers in training set: {n_outliers}/{len(X_train_scaled)} ({n_outliers/len(X_train_scaled)*100:.1f}%)")

# STEP 5: Save BOTH scaler and SVM
import pickle
with open('models/mixture/mixture_ad_svm.pkl', 'wb') as f:
    pickle.dump({
        'svm': svm,
        'scaler': scaler,
        'nu': 0.02,
        'embedding_dim': X_train.shape[1]
    }, f)

print("\n✓ Saved SVM + Scaler to models/mixture/mixture_ad_svm.pkl")
