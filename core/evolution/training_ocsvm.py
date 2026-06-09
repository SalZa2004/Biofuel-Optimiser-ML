"""
Mixture Embedding Extraction + One-Class SVM Training
"""

import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')  # headless backend (no GUI)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math


# =============================================================================
# STEP 1: LOAD MODEL
# =============================================================================

def load_predictor(model_dir):
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor

    predictor = MixtureDCNPredictor(model_dir=model_dir)
    predictor._initialize_models()

    print(f"✓ Loaded {len(predictor.models)} models")
    return predictor


# =============================================================================
# STEP 2: LOAD CSV
# =============================================================================

def load_csv(csv_path):
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"✓ CSV loaded with {encoding}")
            return df
        except Exception:
            continue
    raise ValueError("Failed to load CSV with any encoding")


# =============================================================================
# STEP 3: CONVERT TO MIXTURES
# =============================================================================

def _detect_format(df):
    """Return 'training' or 'seed' based on column naming convention."""
    if 'fuel1 inchi' in df.columns:
        return 'training'   # e.g. 'fuel1 inchi', 'molar fraction fuel 1'
    if 'fuel1_inchi' in df.columns:
        return 'seed'       # e.g. 'fuel1_inchi', 'frac_fuel1 (molar)'
    raise ValueError(
        f"Unrecognised column format. Columns found: {df.columns.tolist()}"
    )


def _parse_row_training(row, df):
    """Parser for mixture_training_dataset.csv format."""
    inchis = []
    for i in range(1, 12):
        col = f'fuel{i} inchi'
        if col in df.columns and pd.notna(row.get(col)):
            val = str(row[col]).strip().strip('"').strip("'")
            if val and val != 'nan':
                inchis.append(val)

    if not inchis:
        return None

    fractions = []
    for i in range(1, len(inchis)):
        col = f'molar fraction fuel {i}'
        if col in df.columns and pd.notna(row.get(col)):
            fractions.append(float(row[col]))
        else:
            break

    if len(fractions) == len(inchis) - 1:
        return {'inchis': inchis, 'fractions': fractions}
    return None


def _parse_row_seed(row, df):
    """Parser for formatted_mixtures.csv format.

    Columns: fuel1_inchi, fuel2_inchi, ..., frac_fuel1 (molar), frac_fuel2 (molar)
    Stores N-1 fractions (last component fraction is implicit = 1 - sum).
    """
    inchis = []
    for i in range(1, 12):
        col = f'fuel{i}_inchi'
        if col in df.columns and pd.notna(row.get(col)):
            val = str(row[col]).strip().strip('"').strip("'")
            if val and val != 'nan':
                inchis.append(val)

    if not inchis:
        return None

    # Collect all available fractions (may be N-1 or fewer)
    fractions = []
    for i in range(1, len(inchis)):
        col = f'frac_fuel{i} (molar)'
        if col in df.columns and pd.notna(row.get(col)):
            fractions.append(float(row[col]))
        else:
            break

    if len(fractions) == len(inchis) - 1:
        return {'inchis': inchis, 'fractions': fractions}
    return None


def build_mixtures(df):
    fmt      = _detect_format(df)
    parser   = _parse_row_training if fmt == 'training' else _parse_row_seed
    mixtures = []

    for _, row in df.iterrows():
        result = parser(row, df)
        if result is not None:
            mixtures.append(result)

    print(f"✓ Built {len(mixtures)} mixtures (format: {fmt})")
    return mixtures


# =============================================================================
# STEP 4: EXTRACT EMBEDDINGS (ENSEMBLE AVERAGE)
# =============================================================================

def extract_embeddings(predictor, mixtures, batch_size=50):
    from core.predictors.mixture.solvation_predictor.data.data import (
        DataPoint, DatapointList, MolencoderDatabase, DataTensor
    )

    all_model_embeddings = []

    for model_idx, model in enumerate(predictor.models):
        print(f"\n  Extracting from model {model_idx + 1}/{len(predictor.models)}")

        model.eval()
        captured = []

        def hook_fn(module, input, output):
            captured.append(input[0].detach().cpu())

        hook = model.ffn.register_forward_hook(hook_fn)

        for i in range(0, len(mixtures), batch_size):
            batch = mixtures[i:i + batch_size]

            datapoints = []
            mol_db = MolencoderDatabase()

            for mix in batch:
                try:
                    dp = DataPoint(
                        smiles=mix['inchis'],
                        targets=[0.0],
                        features=[],
                        molefracs=mix['fractions'],
                        inp=predictor.args,
                        mol_encoders=mol_db
                    )
                    datapoints.append(dp)
                except Exception:
                    continue

            if len(datapoints) == 0:
                continue

            data = DatapointList(datapoints)
            mol_encodings = [[] for _ in range(predictor.args.max_num_mols)]
            tensors = []

            for m in range(len(mol_encodings)):
                for d in datapoints:
                    enc = d.get_mol_encoder()
                    if len(enc) < predictor.args.max_num_mols:
                        enc += [enc[0]] * (predictor.args.max_num_mols - len(enc))
                    mol_encodings[m].append(enc[m])
                tensors.append(DataTensor(
                    mol_encodings[m], predictor.args,
                    property=predictor.args.property
                ))

            with torch.no_grad():
                _ = model(data, tensors)

        hook.remove()

        model_emb = torch.cat(captured, dim=0).numpy()
        print(f"  → Shape: {model_emb.shape}")
        all_model_embeddings.append(model_emb)

    X = np.mean(all_model_embeddings, axis=0)
    print(f"\n✓ Final embedding shape: {X.shape}")
    print(f"✓ Embedding std:         {np.std(X):.4f}")
    return X


# =============================================================================
# STEP 5: TRAIN ONE-CLASS SVM WITH FIXED HYPERPARAMETER SELECTION
# =============================================================================

def train_ocsvm(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Restricted to what visual inspection confirmed works well:
    # gamma=scale is the only well-behaved option for this embedding scale.
    # nu range kept tight to favour conservative applicability domains.
    nu_candidates    = [0.02, 0.05]
    gamma_candidates = ['scale']

    results = []

    print("\nGrid search results:")
    print(f"  {'nu':>6}  {'gamma':>8}  {'actual_out':>10}  {'deviation':>10}  {'score_mean':>11}  {'score_penalty':>13}")

    for nu in nu_candidates:
        for gamma in gamma_candidates:
            svm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            svm.fit(X_scaled)

            preds    = svm.predict(X_scaled)
            scores   = svm.decision_function(X_scaled)
            frac_out = (preds == -1).mean()

            # Primary criterion: how close is actual outlier rate to target nu?
            deviation = abs(frac_out - nu)

            # Secondary criterion: penalise boundaries that are too loose
            # (high mean score) or incorrectly cutting into data (very negative).
            # A well-calibrated SVM has scores centred near zero.
            score_penalty = abs(scores.mean())

            # Combined score — deviation dominates; score_penalty breaks ties
            combined = deviation + 0.1 * score_penalty

            results.append({
                'svm':           svm,
                'nu':            nu,
                'gamma':         gamma,
                'frac_out':      frac_out,
                'deviation':     deviation,
                'score_mean':    round(float(scores.mean()), 4),
                'score_std':     round(float(scores.std()),  4),
                'score_penalty': score_penalty,
                'combined':      combined,
                'scores':        scores,
            })

            print(f"  {nu:>6.2f}  {str(gamma):>8}  "
                  f"{frac_out*100:>9.1f}%  "
                  f"{deviation:>10.4f}  "
                  f"{scores.mean():>11.4f}  "
                  f"{score_penalty:>13.4f}")

    # Select by combined score — tightest well-calibrated boundary wins
    best = min(results, key=lambda r: r['combined'])

    print(f"\n  Best → nu={best['nu']}, gamma={best['gamma']}, "
          f"actual_out={best['frac_out']*100:.1f}%")

    return best, scaler, X_scaled, results


# =============================================================================
# STEP 6: PLOT BOUNDARY COMPARISON
# =============================================================================

def plot_ocsvm_grid(X_scaled, results, best, save_path="ocsvm_grid.png"):
    print("\nGenerating boundary comparison plot...")

    pca = PCA(n_components=2, random_state=42)
    X_2d    = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X_2d[:, 0].min() - pad, X_2d[:, 0].max() + pad, 200),
        np.linspace(X_2d[:, 1].min() - pad, X_2d[:, 1].max() + pad, 200),
    )
    grid_high = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

    n_models = len(results)
    cols = min(3, n_models)
    rows = math.ceil(n_models / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax  = axes[idx]
        svm = r['svm']

        Z = svm.decision_function(grid_high).reshape(xx.shape)

        cf = ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
        ax.contour(xx, yy, Z, levels=[0], colors='black',
                   linewidths=1.5, linestyles='--')
        fig.colorbar(cf, ax=ax, shrink=0.8, label='decision score')

        preds   = svm.predict(X_scaled)
        inliers = preds == 1
        ax.scatter(X_2d[inliers,  0], X_2d[inliers,  1],
                   s=10, alpha=0.4, color='steelblue', label='inlier')
        ax.scatter(X_2d[~inliers, 0], X_2d[~inliers, 1],
                   s=20, alpha=0.9, color='crimson', marker='x',
                   linewidths=1.5, label='outlier')

        ax.set_title(
            f"nu={r['nu']}, γ={r['gamma']}\n"
            f"out={r['frac_out']*100:.1f}%  "
            f"score_mean={r['score_mean']:.3f}",
            fontsize=9
        )
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var)", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var)", fontsize=8)
        ax.legend(fontsize=7)

        if r['nu'] == best['nu'] and r['gamma'] == best['gamma']:
            for spine in ax.spines.values():
                spine.set_edgecolor('#1D9E75')
                spine.set_linewidth(2.5)

    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(
        "One-class SVM boundary comparison",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot → {save_path}")


# =============================================================================
# STEP 7: SAVE MODEL
# =============================================================================

def save_model(best, scaler, X, path):
    # 5th-percentile score as a soft threshold for inference-time flagging
    threshold = float(np.percentile(best['scores'], 5))

    with open(path, 'wb') as f:
        pickle.dump({
            'svm':           best['svm'],
            'scaler':        scaler,
            'nu':            best['nu'],
            'gamma':         best['gamma'],
            'threshold':     threshold,
            'embedding_dim': X.shape[1],
        }, f)

    print(f"\n✓ Saved model to {path}")
    print(f"  nu={best['nu']}, gamma={best['gamma']}, "
          f"threshold (p5)={threshold:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    predictor = load_predictor("models/mixture/dcn")

    # --- training dataset ---
    print("\n--- Training dataset ---")
    df_train    = load_csv("data/database/mixture_training_dataset.csv")
    mix_train   = build_mixtures(df_train)
    X_train     = extract_embeddings(predictor, mix_train)
    print(f"  Training embeddings: {X_train.shape}")

    # --- seed dataset ---
    print("\n--- Seed dataset ---")
    df_seed     = load_csv("data/database/formatted_mixtures.csv")
    mix_seed    = build_mixtures(df_seed)
    X_seed      = extract_embeddings(predictor, mix_seed)
    print(f"  Seed embeddings: {X_seed.shape}")

    # --- combine and deduplicate ---
    X = np.unique(np.vstack([X_train, X_seed]), axis=0)
    print(f"\n✓ Combined + deduplicated: {X.shape[0]} embeddings "
          f"(train={X_train.shape[0]}, seed={X_seed.shape[0]})")

    best, scaler, X_scaled, results = train_ocsvm(X)

    plot_ocsvm_grid(X_scaled, results, best, save_path="ocsvm_grid.png")

    save_model(best, scaler, X, "models/mixture/mixture_ocsvm.pkl")


if __name__ == "__main__":
    main()