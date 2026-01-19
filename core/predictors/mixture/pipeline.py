from core.predictors.mixture.data.formatter import format_mixture_dataset
from core.predictors.mixture.models.predictor import predict_cn
from core.predictors.mixture.evaluation.metrics import mae, rmse

from core.predictors.mixture.solvation_predictor.data.data import read_data, DatapointList
import os

def run_mixture_cn_pipeline(args):
    df = format_mixture_dataset(args.input_csv)
    formatted_csv = os.path.join(args.output_dir, "formatted_mixture_dataset.csv")
    df.to_csv(formatted_csv, index=False)


    data = read_data(args)
    data = DatapointList(data)

    preds = predict_cn(args, data)

    df_out = df.copy()
    df_out["CN_predicted"] = preds["CN_predicted"]
    df_out["CN_std"] = preds["CN_std"]

    valid = df_out["DCN"].notna()

    if valid.any():
        df_out["error"] = df_out["CN_predicted"] - df_out["DCN"]
        df_out["abs_error"] = df_out["error"].abs()

        stats = {
            "MAE": mae(df_out.loc[valid, "DCN"], df_out.loc[valid, "CN_predicted"]),
            "RMSE": rmse(df_out.loc[valid, "DCN"], df_out.loc[valid, "CN_predicted"])
        }
    else:
        stats = {}

    return df_out, stats
