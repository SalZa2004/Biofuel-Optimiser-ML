import numpy as np
import pandas as pd

from core.predictors.mixture.models.loader import patch_molpool_imports


def predict_cn(args, data):
    """
    Runs ensemble prediction and returns dataframe
    """
    patch_molpool_imports()

    from solvation_predictor.train.train import (
        load_checkpoint,
        load_scaler
    )
    from solvation_predictor.train.evaluate import predict

    all_preds = {}

    for path in args.model_path:
        scaler = load_scaler(path)
        scaler.transform_standard(data)

        model = load_checkpoint(path, args)
        preds = predict(model, data, scaler, args)

        all_preds[path] = np.array(preds)[:, 0]

    df = pd.DataFrame(all_preds)
    df["CN_predicted"] = df.mean(axis=1)
    df["CN_std"] = df.std(axis=1)

    return df
