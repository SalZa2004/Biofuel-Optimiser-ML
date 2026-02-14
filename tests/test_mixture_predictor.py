from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
def test_mixture_dcn_predictor():
        
    # Initialize predictor
    predictor = MixtureDCNPredictor()

    # ---- Test 1: single mixture ----
    smiles_list = ["CCCC", "CCCCC"]  # two molecules
    mole_fractions = [0.4, 0.6]      # must sum to 1
    dcn = predictor.predict_mixture_dcn(smiles_list, mole_fractions)
    print("Single mixture DCN:", dcn)

    # ---- Test 2: batch mixtures ----
    base_smiles = ["CCCCCC", "CCCCCCC"]
    base_mole_fractions = [0.5, 0.5]
    additive_smiles_list = ["CCC", "CCCC", "CCCCC"]
    additive_fraction = 0.1

    batch_dcns = predictor.predict_batch_mixtures(
        additive_smiles_list,
        base_smiles,
        base_mole_fractions,
        additive_fraction
    )
    print("Batch mixtures DCNs:", batch_dcns)
    assert len(batch_dcns) == len(additive_smiles_list)
    assert all(isinstance(dcn, float) for dcn in batch_dcns)
    assert all(dcn > 0 for dcn in batch_dcns)

