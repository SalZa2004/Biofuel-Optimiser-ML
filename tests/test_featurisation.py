from core.shared_features import featurize_df
import numpy as np
from core.shared_features import FeatureSelector
from core.shared_features import featurize

def test_featurize_valid_smiles():
    smiles = ["CCC"]
    X = featurize_df(smiles, return_df=False)

    assert X is not None
    assert X.shape[0] == 1

def test_featurize_valid_smiles():
    x = featurize("CCC")
    assert x is not None
    assert x.ndim == 1
    assert x.size > 2000  # Morgan + descriptors

def test_featurize_invalid_smiles():
    x = featurize("NOT_A_SMILES")
    assert x is None

def test_featurize_df_filters_invalid_smiles():
    smiles = ["CCC", "INVALID"]
    X, df_valid = featurize_df(smiles)

    assert X.shape[0] == 1
    assert len(df_valid) == 1
    assert df_valid.iloc[0]["SMILES"] == "CCC"


def test_feature_selector_fit_transform():
    X = np.random.rand(20, 2100)   # 2048 morgan + ~50 descriptors
    y = np.random.rand(20)

    fs = FeatureSelector(top_k=50)
    X_sel = fs.fit_transform(X, y)

    assert fs.is_fitted
    assert X_sel.shape == (20, 50)
