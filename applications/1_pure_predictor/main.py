# applications/1_pure_predictor/main.py
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from core.shared_features import featurize_df, FeatureSelector
from core.predictors.pure_component.generic import GenericPredictor
from core.predictors.pure_component.hf_models import load_models

from cli import get_user_config
from results import display_results

# Load model paths (local or HF)
PREDICTOR_PATHS = load_models()


def run(config):
    """
    Run pure-component property prediction.
    """

    smiles = config["smiles"]


    # --- Featurize ONCE ---
    X_full = featurize_df([smiles], return_df=False)

    if X_full is None:
        raise RuntimeError("Featurization failed for input SMILES.")

    # --- Initialise predictors ---
    cn_predictor = GenericPredictor(
        PREDICTOR_PATHS["cn"],
        "Cetane Number"
    )

    bp_predictor = GenericPredictor(
        PREDICTOR_PATHS["bp"],
        "Boiling Point"
    )

    density_predictor = GenericPredictor(
        PREDICTOR_PATHS["density"],
        "Density"
    )

    lhv_predictor = GenericPredictor(
        PREDICTOR_PATHS["lhv"],
        "Lower Heating Value"
    )

    dyn_visc_predictor = GenericPredictor(
        PREDICTOR_PATHS["dynamic_viscosity"],
        "Dynamic Viscosity"
    )
    ysi_predictor = GenericPredictor(
            PREDICTOR_PATHS["ysi"],
            "YSI"
        )
       
    # --- Predict ---
    result = {
        "SMILES": smiles,
        "CN": cn_predictor.predict_from_features(X_full)[0],
        "YSI": ysi_predictor.predict_from_features(X_full)[0],
        "BOILING POINT": bp_predictor.predict_from_features(X_full)[0],
        "DENSITY": density_predictor.predict_from_features(X_full)[0],
        "LHV": lhv_predictor.predict_from_features(X_full)[0],
        "DYNAMIC VISCOSITY": dyn_visc_predictor.predict_from_features(X_full)[0]
    }

    return result


def main():
    config = get_user_config()
    results = run(config)
    display_results(results)


if __name__ == "__main__":
    main()
