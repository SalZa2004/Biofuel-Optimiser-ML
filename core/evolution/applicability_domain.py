"""
Applicability Domain (AD) Checker using One-Class SVM

Uses One-Class SVM on GNN embeddings to determine if a molecule
is within the training distribution (applicability domain).
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import pickle
from pathlib import Path


class ApplicabilityDomainChecker:
    """
    Check if molecules are within the applicability domain using One-Class SVM.
    
    The idea:
    1. Train on embeddings from training set
    2. For new molecules, check if they're in the same latent space region
    3. Molecules outside AD → predictions are unreliable
    """
    
    def __init__(self, nu: float = 0.02, kernel: str = 'rbf', gamma: str = 'scale'):
        """
        Args:
            nu: Fraction of training data to be treated as outliers (0.01-0.05)
                Lower nu = stricter AD, higher nu = more permissive
                Typical: 0.02 (2% outliers)
            kernel: 'rbf' (default), 'linear', 'poly', 'sigmoid'
            gamma: Kernel coefficient. 'scale' (1/(n_features*X.var())) recommended
                   over 'auto' (1/n_features) for high-dimensional embeddings
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.scaler = StandardScaler()
        self.oc_svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )

        self.is_fitted = False
        # Score range from training set — used to normalise confidence to 0-100
        self.score_min = None
        self.score_max = None
    
    def fit(self, train_embeddings: np.ndarray):
        """
        Fit the One-Class SVM on training set embeddings.
        
        Args:
            train_embeddings: Array of shape (n_train, embedding_dim)
        """
        print(f"Training Applicability Domain checker...")
        print(f"  Training samples: {len(train_embeddings)}")
        print(f"  Embedding dimension: {train_embeddings.shape[1]}")
        print(f"  Nu (outlier fraction): {self.nu}")
        
        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(train_embeddings)
        
        # Fit One-Class SVM
        self.oc_svm.fit(embeddings_scaled)

        # Store training score range for normalised confidence
        train_scores = self.oc_svm.decision_function(embeddings_scaled)
        self.score_min = float(train_scores.min())
        self.score_max = float(train_scores.max())

        # Check how many training samples are classified as outliers
        train_predictions = self.oc_svm.predict(embeddings_scaled)
        n_outliers = (train_predictions == -1).sum()

        print(f"  Training outliers: {n_outliers}/{len(train_embeddings)} ({n_outliers/len(train_embeddings)*100:.1f}%)")
        print(f"  Decision score range: [{self.score_min:.4f}, {self.score_max:.4f}]")

        self.is_fitted = True
        print("✓ AD checker ready!")
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Check if molecules are within the applicability domain.
        
        Args:
            embeddings: Array of shape (n_molecules, embedding_dim)
        
        Returns:
            predictions: Array of +1 (in AD) or -1 (out of AD)
        """
        if not self.is_fitted:
            raise RuntimeError("AD checker not fitted. Call .fit() first.")
        
        # Normalize
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Predict
        return self.oc_svm.predict(embeddings_scaled)
    
    def decision_function(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get decision scores (distance from AD boundary).
        
        Higher score = more confident the molecule is in AD
        Lower/negative score = outside AD
        
        Args:
            embeddings: Array of shape (n_molecules, embedding_dim)
        
        Returns:
            scores: Array of decision scores
        """
        if not self.is_fitted:
            raise RuntimeError("AD checker not fitted. Call .fit() first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        return self.oc_svm.decision_function(embeddings_scaled)
    
    def is_in_domain(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Boolean array: True if in AD, False if out of AD.
        
        Args:
            embeddings: Array of shape (n_molecules, embedding_dim)
        
        Returns:
            in_domain: Boolean array
        """
        predictions = self.predict(embeddings)
        return predictions == 1
    
    def get_confidence_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Convert decision scores to confidence scores (0-100).

        Scores are normalised to the training set range so that:
          - The most central training point scores 100
          - The decision boundary scores 0
          - Out-of-domain points score below 0 (clipped to 0)

        Args:
            embeddings: Array of shape (n_molecules, embedding_dim)

        Returns:
            confidence: Array of confidence scores (0-100)
        """
        decision_scores = self.decision_function(embeddings)

        # Normalise relative to the training score range recorded during fit()
        score_range = self.score_max - self.score_min
        if score_range < 1e-8:
            # Degenerate case: all training scores identical
            return np.where(decision_scores >= 0, 100.0, 0.0).astype(float)

        confidence = (decision_scores - self.score_min) / score_range * 100

        return np.clip(confidence, 0, 100)
    
    def save(self, filepath: str):
        """Save the fitted AD checker."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted AD checker")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'oc_svm': self.oc_svm,
                'nu': self.nu,
                'kernel': self.kernel,
                'gamma': self.gamma,
                'score_min': self.score_min,
                'score_max': self.score_max,
            }, f)
        
        print(f"✓ AD checker saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load a fitted AD checker."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        checker = cls(nu=data['nu'], kernel=data['kernel'], gamma=data['gamma'])
        checker.scaler = data['scaler']
        checker.oc_svm = data['oc_svm']
        checker.score_min = data.get('score_min')
        checker.score_max = data.get('score_max')
        checker.is_fitted = True
        
        print(f"✓ AD checker loaded from {filepath}")
        return checker


# =============================================================================
# COMPLETE WORKFLOW
# =============================================================================

"""
STEP 1: Extract embeddings from training set
--------------------------------------------

from embedding_extractor import EmbeddingExtractor
from mixture_dcn_model import load_trained_model

# Load model
model = load_trained_model('models/mixture_dcn.pth')

# Create extractor
extractor = EmbeddingExtractor(model)

# Extract embeddings for ENTIRE training set
train_embeddings = extractor.extract_embeddings_from_smiles(
    train_smiles,
    featurizer=your_featurizer
)


STEP 2: Train One-Class SVM
---------------------------

from applicability_domain import ApplicabilityDomainChecker

# Create and train AD checker
ad_checker = ApplicabilityDomainChecker(
    nu=0.02,  # 2% outliers
    kernel='rbf'
)

ad_checker.fit(train_embeddings)

# Save for later use
ad_checker.save('models/ad_checker.pkl')


STEP 3: Check new molecules during evolution
--------------------------------------------

# During evolution, for each batch of new molecules:

new_embeddings = extractor.extract_embeddings_from_smiles(
    new_smiles_list,
    featurizer=your_featurizer
)

# Check if in domain
in_domain = ad_checker.is_in_domain(new_embeddings)

# Get confidence scores
confidence = ad_checker.get_confidence_scores(new_embeddings)

# Filter molecules
for i, smiles in enumerate(new_smiles_list):
    if in_domain[i]:
        print(f"{smiles}: In AD (confidence: {confidence[i]:.1f}%)")
    else:
        print(f"{smiles}: OUT OF AD! Prediction unreliable!")
        # Skip this molecule or flag it


STEP 4: Integrate into MixtureAwareMolecularEvolution
-----------------------------------------------------

See mixture_evolution_with_ad.py for full integration!
"""


# =============================================================================
# TUNING GUIDE
# =============================================================================

"""
How to choose 'nu' parameter:
-----------------------------

nu = 0.01 (1%)  → Very strict AD, few molecules accepted
nu = 0.02 (2%)  → Recommended starting point
nu = 0.03 (3%)  → More permissive, good for exploration
nu = 0.05 (5%)  → Very permissive

Start with 0.02 and adjust based on:
- If too many molecules are rejected → increase nu
- If you want stricter filtering → decrease nu


How to interpret decision scores:
---------------------------------

decision_function output:
  > 0.5  : Strongly in AD (high confidence)
  > 0.0  : In AD (medium confidence)
  < 0.0  : Outside AD (low confidence)
  < -0.5 : Strongly outside AD (very low confidence)


Confidence scores (0-100):
  > 75 : High confidence - use prediction
  50-75: Medium confidence - use with caution
  25-50: Low confidence - unreliable
  < 25 : Very low confidence - reject
"""