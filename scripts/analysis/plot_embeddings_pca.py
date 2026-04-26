"""Quick PCA visualization - 2 minutes!"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from visualize_embedding_layers import extract_both_layers

print("Extracting embeddings...")
X_pre, X_post, dcn_targets, complexities = extract_both_layers(max_samples=200)

print(f"Pre-FFN: {X_pre.shape}")
print(f"Post-FFN: {X_post.shape}")

# PCA (FAST!)
print("\nRunning PCA...")
pca_pre = PCA(n_components=2)
X_pre_2d = pca_pre.fit_transform(X_pre)

pca_post = PCA(n_components=2)
X_post_2d = pca_post.fit_transform(X_post)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pre-FFN
axes[0].scatter(X_pre_2d[:, 0], X_pre_2d[:, 1], 
                c=complexities, cmap='tab10', s=50, alpha=0.6)
axes[0].set_title(f'Pre-FFN: Graph Embeddings (202-dim)\nPCA variance: {pca_pre.explained_variance_ratio_.sum()*100:.1f}%', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_pre.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_pre.explained_variance_ratio_[1]*100:.1f}%)')

# Post-FFN
axes[1].scatter(X_post_2d[:, 0], X_post_2d[:, 1], 
                c=complexities, cmap='tab10', s=50, alpha=0.6)
axes[1].set_title(f'Post-FFN: Task Embeddings (500-dim)\nPCA variance: {pca_post.explained_variance_ratio_.sum()*100:.1f}%', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_post.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_post.explained_variance_ratio_[1]*100:.1f}%)')

plt.tight_layout()
plt.savefig('results/embedding_layers_pca.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved to results/embedding_layers_pca.png")
print("\n😴 NOW GO TO SLEEP!")
