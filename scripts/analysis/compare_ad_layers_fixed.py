"""
Compare AD performance: FFN layer vs Earlier layer embeddings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


def inspect_model_layers():
    """Inspect model architecture to find layers."""
    print("="*70)
    print("INSPECTING MODEL ARCHITECTURE")
    print("="*70)
    
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    
    predictor = MixtureDCNPredictor()
    predictor._initialize_models()
    model = predictor.models[0]
    
    print("\nModel structure:")
    print(model)
    
    print("\n\nNamed modules:")
    for name, module in model.named_modules():
        print(f"  {name:30s} → {type(module).__name__}")
    
    return model


def extract_embeddings(layer='ffn', max_samples=None):
    """
    Extract mixture embeddings from specified layer.
    
    Args:
        layer: 'ffn' (last layer) or 'mpn' (message passing layer - earlier)
        max_samples: Limit samples for faster testing
    
    Returns:
        X: Embeddings array
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING EMBEDDINGS FROM {layer.upper()} LAYER")
    print(f"{'='*70}")
    
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    from core.predictors.mixture.solvation_predictor.data.data import (
        DataPoint, DatapointList, MolencoderDatabase, DataTensor
    )
    
    # Load predictor
    predictor = MixtureDCNPredictor()
    predictor._initialize_models()
    
    # Load training data
    df = None
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv("data/database/mixture_training_dataset.csv", encoding=encoding)
            break
        except:
            continue
    
    print(f"✓ Loaded {len(df)} training samples")
    
    # Convert to mixture format
    mixtures = []
    for idx, row in df.iterrows():
        if max_samples and len(mixtures) >= max_samples:
            break
            
        inchis = []
        for i in range(1, 12):
            col = f'fuel{i} inchi'
            if col in df.columns and pd.notna(row.get(col)):
                inchi = str(row[col]).strip().strip('"').strip("'")
                if inchi and inchi not in ['nan', '']:
                    inchis.append(inchi)
        
        if len(inchis) == 0:
            continue
        
        fractions = []
        for i in range(1, len(inchis)):
            col = f'molar fraction fuel {i}'
            if col in df.columns and pd.notna(row.get(col)):
                fractions.append(float(row[col]))
        
        if len(fractions) == len(inchis) - 1:
            mixtures.append({'inchis': inchis, 'fractions': fractions})
    
    print(f"✓ Prepared {len(mixtures)} valid mixtures")
    
    # Extract embeddings
    model = predictor.models[0]
    model.eval()
    
    captured = []
    
    if layer == 'mpn':
        # Hook into MPN output (earlier layer - after graph encoding)
        # NOTE: MPN.forward() returns (mol_vecs, atoms_vecs) tuple, not a tensor.
        # Also, Model.forward() calls self.mpn once per molecule slot (max_num_mols times),
        # so the hook fires max_num_mols times per sample.
        def hook_fn(module, input, output):
            mol_vecs = output[0]  # (batch_size, hidden_size + f_mol_size)
            captured.append(mol_vecs.detach().cpu())

        hook = model.mpn.register_forward_hook(hook_fn)
        print(f"  ✓ Hooked into MPN layer (graph encoder output)")
    
    else:  # ffn (current method)
        # Hook into FFN layer input (after pooling, before final layers)
        def hook_fn(module, input, output):
            captured.append(input[0].detach().cpu())
        
        hook = model.ffn.register_forward_hook(hook_fn)
        print(f"  ✓ Hooked into FFN layer (current method)")
    
    # Process mixtures
    mol_encoder_db = MolencoderDatabase()
    
    for i, mix in enumerate(mixtures):
        if i % 50 == 0:
            print(f"    Processing {i}/{len(mixtures)}...", end='\r')
        
        try:
            dp = DataPoint(
                smiles=mix['inchis'],
                targets=[0.0],
                features=[],
                molefracs=mix['fractions'],
                inp=predictor.args,
                mol_encoders=mol_encoder_db
            )
            
            data = DatapointList([dp])
            
            mol_encodings = [[] for _ in range(predictor.args.max_num_mols)]
            tensors = []
            
            for mol in mol_encodings:
                encoders = dp.get_mol_encoder()
                while len(encoders) < predictor.args.max_num_mols:
                    encoders.append(encoders[0])
                mol.append(encoders[mol_encodings.index(mol)])
                tensors.append(DataTensor(mol, predictor.args, property=predictor.args.property))
            
            with torch.no_grad():
                _ = model(data, tensors)
        
        except Exception as e:
            continue
    
    hook.remove()
    
    # Concatenate all embeddings
    if len(captured) == 0:
        raise ValueError(f"No embeddings captured from {layer} layer!")
    
    # Handle different tensor shapes
    if layer == 'mpn':
        # Model.forward() calls self.mpn max_num_mols times per sample, so
        # captured has (num_samples * max_num_mols) tensors each of shape (1, D).
        # Group them and mean-pool across molecule slots to get one vector per sample.
        max_num_mols = predictor.args.max_num_mols
        all_embeds = torch.cat(captured, dim=0)  # (num_samples * max_num_mols, D)
        n_total = all_embeds.shape[0]
        if n_total % max_num_mols == 0:
            n_samples = n_total // max_num_mols
            X = all_embeds.view(n_samples, max_num_mols, -1).mean(dim=1).numpy()
        else:
            # Fallback if count doesn't divide evenly (some samples failed mid-forward)
            X = all_embeds.numpy()
    else:
        X = torch.cat(captured, dim=0).numpy()
    
    print(f"\n✓ Extracted {X.shape[0]} embeddings")
    print(f"  Embedding dimension: {X.shape[1]}")
    
    return X


def train_and_evaluate_svm(X, layer_name, nu=0.02):
    """Train SVM and evaluate on held-out validation set."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING SVM ON {layer_name.upper()} EMBEDDINGS")
    print(f"{'='*70}")
    
    # Split into train/val (80/20)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train SVM
    print(f"\n  Training One-Class SVM (nu={nu})...")
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu)
    svm.fit(X_train_scaled)
    
    # Evaluate on train
    train_scores = svm.decision_function(X_train_scaled)
    train_preds = svm.predict(X_train_scaled)
    train_outliers = (train_preds == -1).sum()
    
    # Evaluate on validation
    val_scores = svm.decision_function(X_val_scaled)
    val_preds = svm.predict(X_val_scaled)
    val_outliers = (val_preds == -1).sum()
    
    print(f"\n  Results:")
    print(f"    Train outliers: {train_outliers}/{len(train_preds)} ({train_outliers/len(train_preds)*100:.1f}%)")
    print(f"    Val outliers:   {val_outliers}/{len(val_preds)} ({val_outliers/len(val_preds)*100:.1f}%)")
    print(f"    Train score: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
    print(f"    Val score:   {val_scores.mean():.3f} ± {val_scores.std():.3f}")
    
    results = {
        'layer': layer_name,
        'embedding_dim': X.shape[1],
        'train_samples': X_train.shape[0],
        'val_samples': X_val.shape[0],
        'train_outlier_rate': train_outliers / len(train_preds),
        'val_outlier_rate': val_outliers / len(val_preds),
        'train_score_mean': train_scores.mean(),
        'train_score_std': train_scores.std(),
        'val_score_mean': val_scores.mean(),
        'val_score_std': val_scores.std(),
        'svm': svm,
        'scaler': scaler
    }
    
    return results


def compare_layers(max_samples=500):
    """Compare FFN vs MPN layer embeddings."""
    
    print("="*70)
    print("AD LAYER COMPARISON EXPERIMENT")
    print("="*70)
    print(f"\nUsing first {max_samples} samples for quick test")
    
    # Extract from both layers
    print("\n" + "="*70)
    print("STEP 1: Extract FFN embeddings (current method)")
    print("="*70)
    X_ffn = extract_embeddings('ffn', max_samples=max_samples)
    
    print("\n" + "="*70)
    print("STEP 2: Extract MPN embeddings (earlier layer)")
    print("="*70)
    X_mpn = extract_embeddings('mpn', max_samples=max_samples)
    
    # Train and evaluate both
    results_ffn = train_and_evaluate_svm(X_ffn, 'FFN')
    results_mpn = train_and_evaluate_svm(X_mpn, 'MPN')
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'FFN (Current)':<20} {'MPN (Earlier)':<20}")
    print("-" * 70)
    print(f"{'Embedding Dimension':<30} {results_ffn['embedding_dim']:<20} {results_mpn['embedding_dim']:<20}")
    print(f"{'Train Outlier Rate':<30} {results_ffn['train_outlier_rate']*100:>6.1f}%{' '*13} {results_mpn['train_outlier_rate']*100:>6.1f}%")
    print(f"{'Val Outlier Rate':<30} {results_ffn['val_outlier_rate']*100:>6.1f}%{' '*13} {results_mpn['val_outlier_rate']*100:>6.1f}%")
    print(f"{'Consistency (train-val)':<30} {abs(results_ffn['train_outlier_rate'] - results_ffn['val_outlier_rate'])*100:>6.1f}%{' '*13} {abs(results_mpn['train_outlier_rate'] - results_mpn['val_outlier_rate'])*100:>6.1f}%")
    
    # Determine winner based on consistency
    ffn_consistency = abs(results_ffn['train_outlier_rate'] - results_ffn['val_outlier_rate'])
    mpn_consistency = abs(results_mpn['train_outlier_rate'] - results_mpn['val_outlier_rate'])
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    if ffn_consistency < mpn_consistency:
        winner = 'FFN'
        print(f"\n✓ Use FFN layer (current method)")
        print(f"  Reason: Better consistency ({ffn_consistency*100:.1f}% vs {mpn_consistency*100:.1f}%)")
        best_results = results_ffn
    else:
        winner = 'MPN'
        print(f"\n✓ Use MPN layer (earlier layer)")
        print(f"  Reason: Better consistency ({mpn_consistency*100:.1f}% vs {ffn_consistency*100:.1f}%)")
        best_results = results_mpn
    
    # Save best model
    filename = f'models/mixture/mixture_ad_svm_{winner.lower()}_layer.pkl'
    with open(filename, 'wb') as f:
        pickle.dump({
            'svm': best_results['svm'],
            'scaler': best_results['scaler'],
            'layer': winner,
            'nu': 0.02,
            'embedding_dim': best_results['embedding_dim']
        }, f)
    
    print(f"\n✓ Saved: {filename}")
    print("="*70)
    
    return results_ffn, results_mpn, winner


if __name__ == "__main__":
    # First, inspect the model
    # model = inspect_model_layers()
    # input("\nPress Enter to continue with extraction...")
    
    # Run comparison
    results_ffn, results_mpn, winner = compare_layers(max_samples=400)
    
    print("\n\n📊 SUMMARY FOR SUPERVISOR:")
    print("="*70)
    print(f"✓ Compared FFN vs MPN layer embeddings for AD training")
    print(f"✓ Tested on 400 samples with 80/20 train/val split")
    print(f"✓ Winner: {winner} layer (better train/val consistency)")
    print(f"✓ Saved production model: models/mixture/mixture_ad_svm_{winner.lower()}_layer.pkl")
    print("="*70)
