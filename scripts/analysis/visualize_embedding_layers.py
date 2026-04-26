"""
Visualize Pre-FFN (graph embeddings) vs Post-FFN (task embeddings)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.predictors.mixture.solvation_predictor.data.data import (
    DataPoint, DatapointList, MolencoderDatabase, DataTensor
)


def extract_both_layers(max_samples=200):
    """
    Extract embeddings from BOTH:
    1. Pre-FFN: Pure graph/mixture representation
    2. Post-FFN: Task-specific representation
    """
    print("="*70)
    print("EXTRACTING EMBEDDINGS FROM BOTH LAYERS")
    print("="*70)
    
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
    dcn_targets = []
    complexities = []  # Number of components
    
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
        
        if not inchis:
            continue
        
        fractions = []
        for i in range(1, len(inchis)):
            col = f'molar fraction fuel {i}'
            if col in df.columns and pd.notna(row.get(col)):
                fractions.append(float(row[col]))
        
        if len(fractions) == len(inchis) - 1:
            mixtures.append({'inchis': inchis, 'fractions': fractions})
            
            # Get DCN target
            dcn = row.get('DCN', np.nan)
            if pd.isna(dcn):
                dcn = row.get('dcn', np.nan)
            dcn_targets.append(dcn if not pd.isna(dcn) else None)
            
            # Track complexity
            complexities.append(len(inchis))
    
    print(f"✓ Prepared {len(mixtures)} valid mixtures")
    
    # Extract from BOTH layers
    model = predictor.models[0]
    model.eval()
    
    pre_ffn_embeddings = []   # Graph representations
    post_ffn_embeddings = []  # Task representations
    
    # Hook Pre-FFN (readout/pooling output)
    def pre_ffn_hook(module, input, output):
        pre_ffn_embeddings.append(output.detach().cpu())
    
    # Hook Post-FFN (last hidden layer)
    def post_ffn_hook(module, input, output):
        # Get output of second-to-last layer (before final prediction)
        # This is the task-shaped representation
        post_ffn_embeddings.append(output.detach().cpu())
    
    # Register hooks
    # Pre-FFN: hook the readout (if it exists) or first FFN layer input
    try:
        hook_pre = model.readout.register_forward_hook(pre_ffn_hook)
        print("✓ Hooked Pre-FFN at readout layer")
    except AttributeError:
        # Fallback: hook FFN input
        hook_pre = model.ffn.register_forward_hook(
            lambda m, i, o: pre_ffn_embeddings.append(i[0].detach().cpu())
        )
        print("✓ Hooked Pre-FFN at FFN input")
    
    # Post-FFN: hook the last hidden layer (before output layer)
    # This is inside the FFN network
    last_hidden_layer = None
    for name, module in model.ffn.named_modules():
        if 'hidden' in name.lower() or 'ffn' in name.lower():
            last_hidden_layer = module
    
    if last_hidden_layer is not None:
        hook_post = last_hidden_layer.register_forward_hook(post_ffn_hook)
        print("✓ Hooked Post-FFN at last hidden layer")
    else:
        # Fallback: use FFN output
        hook_post = model.ffn.register_forward_hook(post_ffn_hook)
        print("✓ Hooked Post-FFN at FFN output")
    
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
        except:
            continue
    
    hook_pre.remove()
    hook_post.remove()
    
    # Stack embeddings
    X_pre = torch.cat(pre_ffn_embeddings, dim=0).numpy()
    
    # Post-FFN might have different capture patterns
    if post_ffn_embeddings:
        X_post = torch.cat(post_ffn_embeddings, dim=0).numpy()
    else:
        print("\n⚠ Post-FFN capture failed, using Pre-FFN for both")
        X_post = X_pre
    
    print(f"\n✓ Pre-FFN embeddings: {X_pre.shape}")
    print(f"✓ Post-FFN embeddings: {X_post.shape}")
    
    return X_pre, X_post, np.array(dcn_targets), np.array(complexities)


def visualize_embeddings(X_pre, X_post, dcn_targets, complexities):
    """Create comprehensive visualization."""
    
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Remove NaN targets
    valid_mask = ~np.isnan(dcn_targets)
    X_pre_valid = X_pre[valid_mask]
    X_post_valid = X_post[valid_mask]
    dcn_valid = dcn_targets[valid_mask]
    complexity_valid = complexities[valid_mask]
    
    print(f"✓ Using {valid_mask.sum()} samples with valid DCN targets")
    
    # t-SNE projection
    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    X_pre_2d = tsne.fit_transform(X_pre_valid)
    
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=30)
    X_post_2d = tsne2.fit_transform(X_post_valid)
    
    # Create 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Pre-FFN colored by DCN
    ax = axes[0, 0]
    scatter = ax.scatter(X_pre_2d[:, 0], X_pre_2d[:, 1], 
                        c=dcn_valid, cmap='viridis', 
                        s=30, alpha=0.6)
    ax.set_title('Pre-FFN: Graph Embeddings\n(Colored by DCN Target)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='DCN')
    
    # 2. Post-FFN colored by DCN
    ax = axes[0, 1]
    scatter = ax.scatter(X_post_2d[:, 0], X_post_2d[:, 1], 
                        c=dcn_valid, cmap='viridis', 
                        s=30, alpha=0.6)
    ax.set_title('Post-FFN: Task Embeddings\n(Colored by DCN Target)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='DCN')
    
    # 3. Pre-FFN colored by complexity
    ax = axes[1, 0]
    scatter = ax.scatter(X_pre_2d[:, 0], X_pre_2d[:, 1], 
                        c=complexity_valid, cmap='tab10', 
                        s=30, alpha=0.6)
    ax.set_title('Pre-FFN: Graph Embeddings\n(Colored by # Components)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='# Components')
    
    # 4. Post-FFN colored by complexity
    ax = axes[1, 1]
    scatter = ax.scatter(X_post_2d[:, 0], X_post_2d[:, 1], 
                        c=complexity_valid, cmap='tab10', 
                        s=30, alpha=0.6)
    ax.set_title('Post-FFN: Task Embeddings\n(Colored by # Components)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='# Components')
    
    plt.tight_layout()
    plt.savefig('results/embedding_layer_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved to results/embedding_layer_comparison.png")
    
    # PCA variance explained
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    pca_pre = PCA(n_components=min(50, X_pre_valid.shape[1]))
    pca_pre.fit(X_pre_valid)
    
    pca_post = PCA(n_components=min(50, X_post_valid.shape[1]))
    pca_post.fit(X_post_valid)
    
    ax.plot(np.cumsum(pca_pre.explained_variance_ratio_) * 100, 
            label='Pre-FFN (Graph)', linewidth=2)
    ax.plot(np.cumsum(pca_post.explained_variance_ratio_) * 100, 
            label='Post-FFN (Task)', linewidth=2)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax.set_title('Information Content: Pre-FFN vs Post-FFN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/embedding_pca_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved to results/embedding_pca_comparison.png")
    
    plt.close('all')
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Correlation with DCN
    from scipy.stats import spearmanr
    
    # Average distance to same-DCN points vs different-DCN points
    print("\nStructure analysis:")
    print(f"  Pre-FFN dimension: {X_pre.shape[1]}")
    print(f"  Post-FFN dimension: {X_post.shape[1]}")
    
    if X_pre.shape[1] != X_post.shape[1]:
        print("\n✓ Different dimensions - these ARE different layers!")
    else:
        print("\n⚠ Same dimensions - might be same layer")


if __name__ == "__main__":
    print("="*70)
    print("PRE-FFN vs POST-FFN EMBEDDING VISUALIZATION")
    print("="*70)
    
    X_pre, X_post, dcn_targets, complexities = extract_both_layers(max_samples=300)
    
    visualize_embeddings(X_pre, X_post, dcn_targets, complexities)
    
    print("\n✓ Visualization complete!")
    print("  Check results/embedding_layer_comparison.png")
    print("  Check results/embedding_pca_comparison.png")
