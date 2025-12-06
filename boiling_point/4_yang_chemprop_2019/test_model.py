import os
import chemprop

def main():
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(BASE_DIR, "..", "..", "data", "processed", "filtered_smiles_bp.csv")
    output_csv = os.path.join(BASE_DIR, "..", "..", "data", "predicted", "bp_4.csv")
    model_dir = os.path.join(BASE_DIR, "crit_prop_model_files", "CritProp_ML_model_files_without_additional_feat")

    # Arguments
    arguments = [
        '--test_path', input_csv,
        '--preds_path', output_csv,
        '--checkpoint_dir', model_dir
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    
    # Make predictions
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    print("Predictions complete.")

if __name__ == '__main__':
    main()