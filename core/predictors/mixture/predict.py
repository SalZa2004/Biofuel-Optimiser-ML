from core.predictors.mixture import inp
from solvation_predictor.train.train import (
    create_logger,
    load_checkpoint,
    load_scaler,
    load_input,
)
from solvation_predictor.data.data import DatapointList, read_data
from solvation_predictor.train.evaluate import predict
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import wandb
import pandas as pd
from train.utils import parity_plot

if __name__ == "__main__":
    # Load input arguments Class and create logger
    wandb.init()
    inp = inp.PredictArgs().parse_args()
    arguments_dictionary = inp.as_dict()
    wandb.config.update(arguments_dictionary)
    values_list = list(arguments_dictionary.values())
    keys_list = list(arguments_dictionary.keys())
    df = pd.DataFrame([values_list], columns=keys_list).T
    df_path = os.path.join(inp.output_dir, "predict_args.csv")
    df.to_csv(df_path, sep=',')
    logging = create_logger("predict", inp.output_dir)
    logger = logging.debug

    all_data = read_data(inp)
    inp.num_mols = len(all_data[0].smiles)
    inp.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
    inp.num_targets = len(all_data[0].targets)
    inp.num_features = len(all_data[0].features)
    all_data = DatapointList(all_data)
    all_preds = dict()

    for model_path in inp.model_path:
        # load scalar and training input
        _model_path = os.path.join(inp.model_path_root, model_path)
        scaler = load_scaler(_model_path)
        train_inp = load_input(_model_path)
        if inp.scale == "standard":
            scaler.transform_standard(all_data)
            logger(f"Scaled data with {inp.scale} method")
            logger(f"mean: {scaler.mean[0]:5.2f}, std: {scaler.std[0]:5.2f}")
        else:
            raise ValueError("scaler not supported")

        # Load the trained model and get the output from the predictions
        model = load_checkpoint(_model_path, inp, logger=logging)
        preds = predict(model=model, data=all_data, scaler=scaler, inp=inp)
        all_preds[model_path] = preds

    df_out = pd.DataFrame()
    for mol_id in range(inp.num_mols):
        df_out[f'smiles_{mol_id}'] = [d.smiles[mol_id] for d in all_data.get_data()]

    for target_id in range(inp.num_targets):
        df_out[f'target_{target_id}'] = [d.targets[target_id] for d in all_data.get_data()]
        for preds_id, pred_path in enumerate(inp.model_path):
            df_out[f'preds_{target_id}_{preds_id}'] = np.array(all_preds[pred_path])[:,target_id]
        df_out[f'preds_{target_id}_average'] = np.mean([df_out[c] for c in df_out.columns if f'preds_{target_id}' in c], axis=0)
        df_out[f'preds_{target_id}_std'] = np.std([df_out[c] for c in df_out.columns if f'preds_{target_id}' in c], axis=0)
        parity_plot(df_out[f'preds_{target_id}_average'],
                    df_out[f'target_{target_id}'],
                    inp.target_headers[target_id],
                    os.path.join(inp.output_dir, f'parity_plot_target_{target_id}.png')
                    )

    df_out['mole_fraction'] = [d.molefracs[0] for d in all_data.get_data()]

    df_out.to_csv(os.path.join(inp.output_dir, 'detailed_results_predictions.csv'), index=False)

    columns = [c for c in df_out.columns if 'smiles' in c or 'target' in c or 'average' in c or 'std' in c or 'mole_fraction' in c]
    df_out = df_out[columns]
    df_out.to_csv(os.path.join(inp.output_dir, 'average_results_predictions.csv'), index=False)
