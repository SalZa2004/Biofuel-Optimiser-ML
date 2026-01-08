from core.shared_features import featurize_df

def test_featurize_valid_smiles():
    smiles = ["CCC"]
    X = featurize_df(smiles, return_df=False)

    assert X is not None
    assert X.shape[0] == 1

