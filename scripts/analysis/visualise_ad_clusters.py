"""
Visualise and compare embedding clusters for FFN vs MPN AD layers.

Produces a multi-panel figure saved to results/ad_cluster_comparison.png:
  Row 1 – UMAP coloured by SVM decision score  (inlier confidence)
  Row 2 – UMAP coloured by number of mixture components
  Row 3 – UMAP coloured by DCN target value
  Row 4 – PCA explained-variance curves  +  pairwise-distance histograms

Usage:
  python visualise_ad_clusters.py [--samples 400] [--nu 0.02] [--no-umap]
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# Try UMAP; fall back to t-SNE
try:
    import umap
    _HAVE_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    _HAVE_UMAP = False

# ── re-use extraction logic from compare_ad_layers_fixed.py ──────────────────
sys.path.insert(0, ".")
from compare_ad_layers_fixed import extract_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def reduce_2d(X: np.ndarray, method: str = "umap", random_state: int = 42) -> np.ndarray:
    """Reduce embedding matrix to 2-D for visualisation."""
    if method == "umap" and _HAVE_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=random_state,
                            n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(X)
    else:
        if method == "umap":
            print("  UMAP not installed – falling back to t-SNE")
        reducer = TSNE(n_components=2, random_state=random_state,
                       perplexity=min(30, len(X) - 1), max_iter=1000)
        return reducer.fit_transform(X)


def fit_svm(X: np.ndarray, nu: float = 0.02):
    """Fit One-Class SVM and return (scaler, svm, decision_scores)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    svm.fit(X_scaled)
    scores = svm.decision_function(X_scaled)
    return scaler, svm, scores


def load_metadata(max_samples: int):
    """
    Load per-mixture metadata (n_components, DCN target) aligned with the
    same row order that extract_embeddings uses.
    """
    for enc in ("utf-8", "latin-1", "iso-8859-1"):
        try:
            df = pd.read_csv("data/database/mixture_training_dataset.csv",
                             encoding=enc)
            break
        except Exception:
            continue

    n_components, dcn_values = [], []

    for _, row in df.iterrows():
        if len(n_components) >= max_samples:
            break

        inchis = []
        for i in range(1, 12):
            col = f"fuel{i} inchi"
            if col in df.columns and pd.notna(row.get(col)):
                val = str(row[col]).strip().strip('"').strip("'")
                if val and val not in ("nan", ""):
                    inchis.append(val)

        if len(inchis) == 0:
            continue

        fracs = []
        for i in range(1, len(inchis)):
            col = f"molar fraction fuel {i}"
            if col in df.columns and pd.notna(row.get(col)):
                fracs.append(float(row[col]))

        if len(fracs) != len(inchis) - 1:
            continue

        n_components.append(len(inchis))

        # Prefer experimental; fall back to synthetic
        dcn = np.nan
        for col in ("Experimental value (from literature)", "Synthetic value"):
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    dcn = float(row[col])
                    break
                except (ValueError, TypeError):
                    pass
        dcn_values.append(dcn)

    return np.array(n_components), np.array(dcn_values, dtype=float)

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_anomaly_detection(X, nu=0.02, test_split=0.3, random_state=42):
    np.random.seed(random_state)

    # ── Split data ─────────────────────────────────────
    n = len(X)
    idx = np.random.permutation(n)
    split = int((1 - test_split) * n)

    X_train = X[idx[:split]]
    X_test_normal = X[idx[split:]]

    # ── Generate synthetic anomalies ───────────────────
    noise_scale = 0.5 * X.std(axis=0)
    X_test_anom = X_test_normal + np.random.normal(0, noise_scale, X_test_normal.shape)

    # Combine test set
    X_test = np.vstack([X_test_normal, X_test_anom])
    y_true = np.hstack([np.ones(len(X_test_normal)), np.zeros(len(X_test_anom))])

    # ── Train SVM ──────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
    svm.fit(X_train_s)

    # ── Evaluate ───────────────────────────────────────
    X_test_s = scaler.transform(X_test)
    scores = svm.decision_function(X_test_s)

    # Higher = more normal → invert for anomaly score
    anomaly_scores = -scores

    roc = roc_auc_score(y_true, anomaly_scores)

    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = auc(recall, precision)

    return roc, pr_auc
# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

LAYER_COLORS = {"FFN": "#2196F3", "MPN": "#FF5722"}
LAYER_KEYS   = ["FFN", "MPN"]


def _scatter(ax, xy, values, cmap, title, vmin=None, vmax=None,
             svm_scores=None, nu=None):
    """Scatter with colorbar; optionally outline outliers."""
    vmin = vmin if vmin is not None else np.nanpercentile(values, 2)
    vmax = vmax if vmax is not None else np.nanpercentile(values, 98)

    sc = ax.scatter(xy[:, 0], xy[:, 1],
                    c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=10, alpha=0.6, linewidths=0)

    # Outline outliers if SVM scores provided
    if svm_scores is not None:
        outlier_mask = svm_scores < 0
        if outlier_mask.any():
            ax.scatter(xy[outlier_mask, 0], xy[outlier_mask, 1],
                       s=25, facecolors="none", edgecolors="red",
                       linewidths=0.6, alpha=0.8, label="SVM outlier")
            ax.legend(fontsize=7, markerscale=1.2, loc="upper right")

    plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([]);  ax.set_yticks([])
    return ax


def _pca_variance(ax, X_ffn, X_mpn, n_components=30):
    """Cumulative explained variance for both layers."""
    for X, label, color in zip(
            [X_ffn, X_mpn], LAYER_KEYS, [LAYER_COLORS["FFN"], LAYER_COLORS["MPN"]]):
        pca = PCA(n_components=min(n_components, X.shape[1], X.shape[0]))
        pca.fit(X)
        cum_var = np.cumsum(pca.explained_variance_ratio_) * 100
        ax.plot(range(1, len(cum_var) + 1), cum_var,
                marker="o", ms=3, label=label, color=color)

    ax.axhline(90, color="gray", lw=0.8, ls="--", label="90 %")
    ax.set_xlabel("PCs", fontsize=9)
    ax.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax.set_title("PCA explained variance", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _distance_hist(ax, X_ffn, X_mpn, n_sample=300):
    """Histogram of pairwise cosine distances (sub-sampled)."""
    from sklearn.metrics.pairwise import cosine_distances

    def _sample(X):
        idx = np.random.choice(len(X), size=min(n_sample, len(X)), replace=False)
        return X[idx]

    for X, label, color in zip(
            [X_ffn, X_mpn], LAYER_KEYS, [LAYER_COLORS["FFN"], LAYER_COLORS["MPN"]]):
        dists = cosine_distances(_sample(X)).ravel()
        dists = dists[dists > 1e-6]  # drop self-distances
        ax.hist(dists, bins=60, density=True, alpha=0.55,
                color=color, label=label)

    ax.set_xlabel("Cosine distance", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Pairwise distance distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(max_samples: int = 400, nu: float = 0.02, force_tsne: bool = False):
    method = "tsne" if (force_tsne or not _HAVE_UMAP) else "umap"
    method_label = "t-SNE" if method == "tsne" else "UMAP"

    print(f"Dimensionality reduction: {method_label}")
    print(f"Samples: {max_samples}  |  SVM nu: {nu}\n")

    # ── 1. Extract embeddings ────────────────────────────────────────────────
    print("Extracting FFN embeddings...")
    X_ffn = extract_embeddings("ffn", max_samples=max_samples)

    print("\nExtracting MPN embeddings...")
    X_mpn = extract_embeddings("mpn", max_samples=max_samples)

    n_ffn, n_mpn = X_ffn.shape[0], X_mpn.shape[0]
    n_shared = min(n_ffn, n_mpn)
    print(f"\nFFN: {n_ffn} samples  |  MPN: {n_mpn} samples")
    if n_ffn != n_mpn:
        print(f"  Trimming to {n_shared} for alignment")
        X_ffn = X_ffn[:n_shared]
        X_mpn = X_mpn[:n_shared]

    # ── 2. Fit SVMs ──────────────────────────────────────────────────────────
    print("\nFitting SVMs...")
    _, _, svm_scores_ffn = fit_svm(X_ffn, nu=nu)
    _, _, svm_scores_mpn = fit_svm(X_mpn, nu=nu)

    # ── 3. Load metadata ─────────────────────────────────────────────────────
    print("Loading metadata...")
    n_comps, dcn_vals = load_metadata(max_samples)
    n_comps = n_comps[:n_shared]
    dcn_vals = dcn_vals[:n_shared]

    # ── 4. 2-D projections ───────────────────────────────────────────────────
    print(f"Computing {method_label} projections (FFN)...")
    xy_ffn = reduce_2d(X_ffn, method=method)

    print(f"Computing {method_label} projections (MPN)...")
    xy_mpn = reduce_2d(X_mpn, method=method)

    # ── 5. Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 22))
    fig.suptitle(
        f"Embedding Cluster Comparison: FFN vs MPN layers\n"
        f"({method_label}, n={n_shared}, SVM nu={nu})",
        fontsize=13, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        hspace=0.45, wspace=0.25,
        height_ratios=[1, 1, 1, 0.85]
    )

    # ── Row 0: SVM decision score ────────────────────────────────────────────
    score_vmin = min(svm_scores_ffn.min(), svm_scores_mpn.min())
    score_vmax = max(svm_scores_ffn.max(), svm_scores_mpn.max())

    ax00 = fig.add_subplot(gs[0, 0])
    _scatter(ax00, xy_ffn, svm_scores_ffn, "RdYlGn",
             f"FFN – SVM decision score",
             vmin=score_vmin, vmax=score_vmax,
             svm_scores=svm_scores_ffn)

    ax01 = fig.add_subplot(gs[0, 1])
    _scatter(ax01, xy_mpn, svm_scores_mpn, "RdYlGn",
             f"MPN – SVM decision score",
             vmin=score_vmin, vmax=score_vmax,
             svm_scores=svm_scores_mpn)

    ffn_outlier_pct = (svm_scores_ffn < 0).mean() * 100
    mpn_outlier_pct = (svm_scores_mpn < 0).mean() * 100
    ax00.set_xlabel(f"Outliers: {ffn_outlier_pct:.1f}%  |  green = in-domain",
                    fontsize=8)
    ax01.set_xlabel(f"Outliers: {mpn_outlier_pct:.1f}%  |  green = in-domain",
                    fontsize=8)

    # ── Row 1: Number of mixture components ──────────────────────────────────
    unique_nc = sorted(np.unique(n_comps[~np.isnan(n_comps)]))
    cmap_nc = plt.cm.get_cmap("tab10", len(unique_nc))

    ax10 = fig.add_subplot(gs[1, 0])
    sc10 = ax10.scatter(xy_ffn[:, 0], xy_ffn[:, 1],
                        c=n_comps, cmap=cmap_nc,
                        vmin=min(unique_nc) - 0.5, vmax=max(unique_nc) + 0.5,
                        s=10, alpha=0.6, linewidths=0)
    cb10 = plt.colorbar(sc10, ax=ax10, pad=0.02, fraction=0.046,
                        ticks=unique_nc)
    cb10.set_label("# components", fontsize=8)
    ax10.set_title("FFN – mixture complexity", fontsize=10, fontweight="bold")
    ax10.set_xticks([]); ax10.set_yticks([])

    ax11 = fig.add_subplot(gs[1, 1])
    sc11 = ax11.scatter(xy_mpn[:, 0], xy_mpn[:, 1],
                        c=n_comps, cmap=cmap_nc,
                        vmin=min(unique_nc) - 0.5, vmax=max(unique_nc) + 0.5,
                        s=10, alpha=0.6, linewidths=0)
    cb11 = plt.colorbar(sc11, ax=ax11, pad=0.02, fraction=0.046,
                        ticks=unique_nc)
    cb11.set_label("# components", fontsize=8)
    ax11.set_title("MPN – mixture complexity", fontsize=10, fontweight="bold")
    ax11.set_xticks([]); ax11.set_yticks([])

    # ── Row 2: DCN target value ───────────────────────────────────────────────
    valid_dcn = dcn_vals[~np.isnan(dcn_vals)]
    dcn_vmin = np.percentile(valid_dcn, 2) if len(valid_dcn) else 0
    dcn_vmax = np.percentile(valid_dcn, 98) if len(valid_dcn) else 1

    ax20 = fig.add_subplot(gs[2, 0])
    _scatter(ax20, xy_ffn, dcn_vals, "plasma",
             "FFN – DCN target", vmin=dcn_vmin, vmax=dcn_vmax)
    ax20.set_xlabel("NaN = no label", fontsize=8)

    ax21 = fig.add_subplot(gs[2, 1])
    _scatter(ax21, xy_mpn, dcn_vals, "plasma",
             "MPN – DCN target", vmin=dcn_vmin, vmax=dcn_vmax)
    ax21.set_xlabel("NaN = no label", fontsize=8)

    # ── Row 3: PCA variance + distance histogram ─────────────────────────────
    ax30 = fig.add_subplot(gs[3, 0])
    _pca_variance(ax30, X_ffn, X_mpn)

    ax31 = fig.add_subplot(gs[3, 1])
    _distance_hist(ax31, X_ffn, X_mpn)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = "results/ad_cluster_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CLUSTER COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'':30s} {'FFN':>10} {'MPN':>10}")
    print("-" * 52)
    print(f"{'Embedding dim':<30} {X_ffn.shape[1]:>10} {X_mpn.shape[1]:>10}")
    print(f"{'SVM outlier %':<30} {ffn_outlier_pct:>9.1f}% {mpn_outlier_pct:>9.1f}%")
    print(f"{'Score mean':<30} {svm_scores_ffn.mean():>10.3f} {svm_scores_mpn.mean():>10.3f}")
    print(f"{'Score std':<30} {svm_scores_ffn.std():>10.3f} {svm_scores_mpn.std():>10.3f}")
    print(f"{'Score range':<30} [{svm_scores_ffn.min():.2f}, {svm_scores_ffn.max():.2f}]"
          f"  [{svm_scores_mpn.min():.2f}, {svm_scores_mpn.max():.2f}]")

    # PCA dims needed for 90% variance
    for X, label in [(X_ffn, "FFN"), (X_mpn, "MPN")]:
        pca = PCA().fit(X)
        n90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90)) + 1
        print(f"{'PCs for 90% variance (' + label + ')':<30} {n90:>10}")

    print("=" * 60)
    print(f"\n✓  Saved: {out_path}")

    roc_ffn, pr_ffn = evaluate_anomaly_detection(X_ffn)
    roc_mpn, pr_mpn = evaluate_anomaly_detection(X_mpn)

    print("\nAnomaly Detection Performance")
    print("--------------------------------")
    print(f"FFN  ROC-AUC: {roc_ffn:.3f} | PR-AUC: {pr_ffn:.3f}")
    print(f"MPN  ROC-AUC: {roc_mpn:.3f} | PR-AUC: {pr_mpn:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise FFN vs MPN AD clusters")
    parser.add_argument("--samples", type=int, default=400,
                        help="Max mixture samples to use (default: 400)")
    parser.add_argument("--nu", type=float, default=0.02,
                        help="One-Class SVM nu parameter (default: 0.02)")
    parser.add_argument("--no-umap", action="store_true",
                        help="Force t-SNE even if UMAP is available")
    args = parser.parse_args()

    main(max_samples=args.samples, nu=args.nu, force_tsne=args.no_umap)
