"""
Simple scatter plot of FFN and MPN layer embeddings (PCA to 2D).

Usage:
  python plot_embeddings.py [--samples 200]
"""

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, ".")
from compare_ad_layers_fixed import extract_embeddings


def main(max_samples: int = 200):
    print("Extracting FFN embeddings...")
    X_ffn = extract_embeddings("ffn", max_samples=max_samples)

    print("\nExtracting MPN embeddings...")
    X_mpn = extract_embeddings("mpn", max_samples=max_samples)

    # Align sample counts
    n = min(X_ffn.shape[0], X_mpn.shape[0])
    X_ffn, X_mpn = X_ffn[:n], X_mpn[:n]
    print(f"\nPlotting {n} samples  |  FFN dim: {X_ffn.shape[1]}  |  MPN dim: {X_mpn.shape[1]}")

    # PCA to 2D
    xy_ffn = PCA(n_components=2).fit_transform(X_ffn)
    xy_mpn = PCA(n_components=2).fit_transform(X_mpn)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"FFN vs MPN layer embeddings (PCA, n={n})", fontsize=13, fontweight="bold")

    for ax, xy, label, color in [
        (axes[0], xy_ffn, "FFN layer", "#2196F3"),
        (axes[1], xy_mpn, "MPN layer", "#FF5722"),
    ]:
        ax.scatter(xy[:, 0], xy[:, 1], s=12, alpha=0.55, color=color, linewidths=0)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2", fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = "results/embeddings_raw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=200, help="Max samples (default: 200)")
    args = parser.parse_args()
    main(max_samples=args.samples)
