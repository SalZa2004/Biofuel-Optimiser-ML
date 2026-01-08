from core.predictors.pure_component.property_predictor import PropertyPredictor
from core.config import EvolutionConfig

def test_property_predictor_batch():
    config = EvolutionConfig(
        target_cn=55,
        generations=1,
        population_size=1,
        minimize_ysi=True,
        maximize_cn=False,
    )

    predictor = PropertyPredictor(config)
    results = predictor.predict_all_properties(["CCC"])

    assert "cn" in results
    assert results["cn"][0] is not None
