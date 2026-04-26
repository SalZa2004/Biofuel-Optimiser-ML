"""Quick visualization of Pre-FFN vs Post-FFN"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# Load the embeddings you already extracted
# (They're in memory from your last run, or re-extract)

from visualize_embedding_layers import extract_both_layers

print("Extracting embeddings...")
X_pre, X_post, dcn_targets, complexities = extract_both_layers(max_samples=200)

print(f"Pre-FFN: {X_pre.shape}")
print(f"Post-FFN: {X_post.shape}")

# Simple t-SNE
print("\nRunning t-SNE...")
tsne_pre = TSNE(n_components=2, random_state=42, perplexity=30)
X_pre_2d = tsne_pre.fit_transform(X_pre[:200])  # Limit to 200 for speed

tsne_post = TSNE(n_components=2, random_state=42, perplexity=30)
X_post_2d = tsne_post.fit_transform(X_post[:200])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pre-FFN
axes[0].scatter(X_pre_2d[:, 0], X_pre_2d[:, 1], 
                c=complexities[:200], cmap='tab10', s=50, alpha=0.6)
axes[0].set_title('Pre-FFN: Graph Embeddings (202-dim)\nColored by # Components', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# Post-FFN
axes[1].scatter(X_post_2d[:, 0], X_post_2d[:, 1], 
                c=complexities[:200], cmap='tab10', s=50, alpha=0.6)
axes[1].set_title('Post-FFN: Task Embeddings (500-dim)\nColored by # Components', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.savefig('results/embedding_layers_simple.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved to results/embedding_layers_simple.png")
plt.close()

print("\nDONE! Now go to sleep! 😴")
