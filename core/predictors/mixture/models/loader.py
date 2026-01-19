import os
import sys

def patch_molpool_imports():
    """
    Fix old pickle paths used in MolPool checkpoints
    """
    import core.predictors.mixture.inp as inp
    import core.predictors.mixture.solvation_predictor as sp

    sys.modules["solvation_predictor"] = sp
    sys.modules["solvation_predictor.inp"] = inp


def get_model_paths(model_dir: str):
    return sorted([
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith(".pt")
    ])
