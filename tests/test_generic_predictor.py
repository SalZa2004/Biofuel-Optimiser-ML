from core.predictors.pure_component.generic import GenericPredictor
from core.predictors.pure_component.hf_models import load_models
from core.shared_features import FeatureSelector, featurize_df

def test_generic_predictor_cn():
    paths = load_models()
    predictor = GenericPredictor(paths["cn"], "Cetane Number")

    X = featurize_df(["CCC"], return_df=False)
    preds = predictor.predict_from_features(X)

    assert preds is not None
    assert len(preds) == 1
    assert isinstance(preds[0], float)
