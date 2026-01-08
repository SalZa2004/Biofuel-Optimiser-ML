from applications.pure_predictor.main import run

def test_pure_predictor_run():
    config = {
        "mode": "1",
        "smiles": "CCC",
    }

    result = run(config)

    assert isinstance(result, dict)

    expected_keys = [
        "CN",
        "BOILING POINT",
        "DENSITY",
        "DYNAMIC VISCOSITY",
        "YSI",
        "LHV"
    ]

    for key in expected_keys:
        assert key in result
        assert result[key] is not None
