from core.predictors.mixture import inp
from solvation_predictor.train.train import *
from solvation_predictor.data.data import read_data
import datetime
import wandb
import shutil
from memory_profiler import profile

if __name__ == "__main__":
    # Load the input arguments class, create the logger and read the training input data
    wandb.init()
    inp = inp.TrainArgs().parse_args()
    arguments_dictionary = inp.as_dict()
    wandb.config.update(arguments_dictionary)
    values_list = list(arguments_dictionary.values())
    keys_list = list(arguments_dictionary.keys())
    df = pd.DataFrame([values_list], columns=keys_list).T
    df_path = os.path.join(inp.output_dir, "predict_args.csv")
    df.to_csv(df_path, sep=',')
    metadata_path = os.path.join(inp.output_dir, "wandb-metadata.json")
    wandb_path = wandb.run.dir + "/wandb-metadata.json"
    shutil.copy(wandb_path, metadata_path)
    logging = create_logger("train", inp.output_dir)
    logger = logging.debug
    logger(f"Start training on {datetime.time()}")
    logger(f"Doing {inp.num_folds} different folds")
    logger(f"Doing {inp.num_models} different models")
    all_data = read_data(inp)

    # Run the training procedure
    run_training(inp, all_data, logging)
    wandb.finish()
    logger(f"Finished training on {datetime.time()}")

