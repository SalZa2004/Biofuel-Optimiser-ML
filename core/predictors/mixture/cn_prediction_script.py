import pandas as pd
import numpy as np
import os
import sys
import torch

# Add the core directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
from typing import Callable, List, Union


def prepare_mixture_data(input_csv, output_csv, only_with_dcn=True):
    """
    Convert your mixture dataset to the format expected by the model.
    
    Args:
        input_csv: Path to your original CSV file
        output_csv: Path where formatted CSV will be saved
        only_with_dcn: If True, only include rows with CN_Measured values
    """
    # Try different encodings
    encoding = 'latin-1'
    df = pd.read_csv(input_csv, encoding=encoding)
    print(f"✓ Successfully read with {encoding} encoding")
    
    # Create output dataframe
    output_rows = []
    skipped_no_dcn = 0

    for idx, row in df.iterrows():
        has_dcn = False
        dcn_value = np.nan
        if 'CN_Measured' in df.columns and pd.notna(row.get('CN_Measured')):
            has_dcn = True
            dcn_value = float(row['CN_Measured'])
        elif 'CN_Regression' in df.columns and pd.notna(row.get('CN_Regression')):
            has_dcn = True
            dcn_value = float(row['CN_Regression'])
        # Skip if filtering enabled and no DCN value
        if only_with_dcn and not has_dcn:
            skipped_no_dcn += 1
            continue
        
        output_row = {}
        
        # Collect all fuel components and their fractions
        fuel_smiles = []
        fuel_fractions = []
        fuel_names = []
        
        for i in range(1, 12):  # fuel_1 through fuel_11
            smiles_col = f'fuel_{i}_smiles'
            frac_col = f'fraction_fuel_{i}'
            name_col = f'fuel_{i}'
            
            if smiles_col in df.columns and pd.notna(row.get(smiles_col, None)):
                smiles = row[smiles_col]
                fraction = row.get(frac_col, 0)
                name = row.get(name_col, f'fuel_{i}')
                
                if smiles and str(smiles).strip() and smiles != 'nan':  # Not empty
                    fuel_smiles.append(smiles)
                    fuel_fractions.append(float(fraction))
                    fuel_names.append(name)
        
        # Skip if no fuels found
        if len(fuel_smiles) == 0:
            continue
        
        # Add fuel SMILES to output
        for i in range(len(fuel_smiles)):
            output_row[f'fuel{i+1}_inchi'] = fuel_smiles[i]  # Model uses 'inchi' header but accepts SMILES
        
        # Add mole fractions (one less than number of fuels, last is calculated)
        for i in range(len(fuel_smiles) - 1):
            output_row[f'frac_fuel{i+1} (molar)'] = fuel_fractions[i]
        
        # Add target (CN or DCN)
        output_row['DCN'] = dcn_value
        
        # Add metadata
        output_row['Name'] = row.get('Name', f'mixture_{idx}')
        output_row['num_components'] = len(fuel_smiles)
        
        output_rows.append(output_row)
    
    # Create dataframe
    df_output = pd.DataFrame(output_rows)
    
    # Remove any extra fraction columns that might confuse the model
    # Only keep the N-1 fraction columns that should be there
    frac_cols_to_keep = []
    for col in df_output.columns:
        if 'frac_fuel' in col and 'molar' in col:
            # Extract the fuel number
            import re
            match = re.search(r'frac_fuel(\d+)', col)
            if match:
                fuel_num = int(match.group(1))
                # Keep fraction columns where fuel_num < max components
                if fuel_num < df_output['num_components'].max():
                    frac_cols_to_keep.append(col)
    
    # Reorder columns: fuel SMILES first, then fractions, then metadata, then target
    fuel_cols = sorted([c for c in df_output.columns if 'fuel' in c and 'inchi' in c])
    frac_cols = sorted([c for c in frac_cols_to_keep])
    meta_cols = ['Name', 'num_components']
    target_cols = ['DCN']
    
    # Build final column order
    final_cols = fuel_cols + frac_cols + target_cols + meta_cols
    df_output = df_output[final_cols]
    
    # Save to CSV
    df_output.to_csv(output_csv, index=False)
    print(f"Formatted data saved to {output_csv}")
    print(f"Number of mixtures: {len(df_output)}")
    if only_with_dcn:
        print(f"Skipped mixtures without DCN: {skipped_no_dcn}")
    print(f"Max components: {df_output['num_components'].max()}")
    print(f"Sample columns: {list(df_output.columns[:10])}")
    
    return df_output


def create_predict_args(formatted_csv, model_dir, output_dir):
    """
    Create prediction arguments without importing inp module directly.
    """
    # Determine the structure from the CSV
    df_check = pd.read_csv(formatted_csv)
    fuel_cols = sorted([c for c in df_check.columns if 'fuel' in c and 'inchi' in c])
    frac_cols = sorted([c for c in df_check.columns if 'frac_fuel' in c and 'molar' in c])
    
    max_num_mols = len(fuel_cols)
    
    print(f"\nDetected data structure:")
    print(f"  Max fuel components: {max_num_mols}")
    print(f"  Fuel columns: {fuel_cols}")
    print(f"  Fraction columns: {frac_cols}")
    
    # Create a minimal args object
    class PredictArgs:
        def __init__(self):
            self.input_file = formatted_csv
            self.model_path_root = model_dir
            self.model_path = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
            self.output_dir = output_dir
            
            # Data structure
            self.max_num_mols = max_num_mols
            self.solute = False
            self.solute_headers = []
            self.solvent_headers = fuel_cols
            self.target_headers = ['DCN']
            self.features_headers = []
            self.molefrac_headers = frac_cols
            self.delimiter = ','
            
            # Model parameters (will be updated from loaded models)
            self.num_targets = 1
            self.num_features = 0
            self.f_mol_size = 2
            self.num_mols = max_num_mols
            
            # Training parameters (from original inp.py)
            self.property = "solvation"
            self.add_hydrogens_to_solvent = False
            self.scale = "standard"
            self.scale_features = False
            self.use_same_scaler_for_features = False
            self.max_molecules = -1
            
            # Missing attributes that Model needs
            self.postprocess = False
            self.attention = False
            self.uncertainty = False
            self.ensemble_variance = False
            self.mix = False
            self.morgan_fingerprint = "None"
            self.morgan_bits = 16
            self.morgan_radius = 2
            
            # MPN parameters
            self.depth = 4
            self.mpn_hidden = 200
            self.mpn_dropout = 0.0
            self.mpn_activation = "LeakyReLU"
            self.mpn_bias = False
            self.aggregation = "mean"
            
            # Attention parameters
            self.att_hidden = 200
            self.att_dropout = 0.0
            self.att_bias = False
            self.att_activation = "ReLU"
            self.att_normalize = "sigmoid"
            self.att_first_normalize = False
            
            # FFN parameters
            self.ffn_hidden = 500
            self.ffn_num_layers = 4
            self.ffn_dropout = 0.0
            self.ffn_activation = "LeakyReLU"
            self.ffn_bias = True
            
            # Device setup
            self.cuda = torch.cuda.is_available() or torch.backends.mps.is_available()
            if self.cuda:
                self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda')
            else:
                self.device = torch.device('cpu')
            
            print(f"  Using device: {self.device}")
    
    return PredictArgs()


def run_cn_predictions(formatted_csv, model_dir, output_dir):
    """
    Run CN predictions using the trained DCN models.
    """
    # Fix import path issue - the models were saved with 'solvation_predictor.inp' 
    # but the actual module structure is different
    import sys
    
    # Import the actual modules
    from core.predictors.mixture import inp as mixture_inp
    import core.predictors.mixture.solvation_predictor as sp
    
    # Create module aliases for the old import paths
    sys.modules['solvation_predictor.inp'] = mixture_inp
    sys.modules['solvation_predictor'] = sp
    
    # Now import the functions we need
    from core.predictors.mixture.solvation_predictor.train.train import create_logger, load_checkpoint, load_scaler
    from core.predictors.mixture.solvation_predictor.data.data import read_data, DatapointList
    from core.predictors.mixture.solvation_predictor.train.evaluate import predict
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create prediction arguments
    pred_args = create_predict_args(formatted_csv, model_dir, output_dir)
    
    print(f"\nUsing {len(pred_args.model_path)} models: {pred_args.model_path}")
    
    # Create logger
    logging_obj = create_logger("predict_cn", output_dir)
    logger = logging_obj.debug
    
    # Read data
    logger("Reading data...")
    all_data = read_data(pred_args)
    
    if len(all_data) == 0:
        raise ValueError("No valid data points found! Check your SMILES strings.")
    
    logger(f"Successfully loaded {len(all_data)} data points")
    
    # Set additional parameters from loaded data
    pred_args.num_mols = len(all_data[0].smiles)
    pred_args.f_mol_size = all_data[0].get_mol_encoder()[0].get_sizes()[2]
    pred_args.num_features = len(all_data[0].features)
    
    all_data = DatapointList(all_data)
    all_preds = {}
    
    # Run predictions with each model
    for model_idx, model_file in enumerate(pred_args.model_path):
        logger(f"\n{'='*60}")
        logger(f"Processing model {model_idx + 1}/{len(pred_args.model_path)}: {model_file}")
        logger(f"{'='*60}")
        
        model_path = os.path.join(model_dir, model_file)
        
        # Load scaler
        scaler = load_scaler(model_path)
        
        # Scale data
        if pred_args.scale == "standard":
            scaler.transform_standard(all_data)
            logger(f"Scaled data with {pred_args.scale} method")
            logger(f"  Mean: {scaler.mean[0]:.3f}, Std: {scaler.std[0]:.3f}")
        
        # Load model and predict
        model = load_checkpoint(model_path, pred_args, logger=logging_obj)
        preds = predict(model=model, data=all_data, scaler=scaler, inp=pred_args)
        all_preds[model_file] = preds
        
        logger(f"Predictions complete. Sample: {preds[0][0]:.2f}")
    
    # Compile results
    logger("\nCompiling results...")
    
    # Build results directly from the loaded data to ensure alignment
    results_list = []
    
    for idx, datapoint in enumerate(all_data.get_data()):
        result_row = {}
        
        # Add SMILES from the actual loaded data
        for mol_id in range(min(len(datapoint.smiles), 5)):
            result_row[f'fuel_{mol_id+1}_smiles'] = datapoint.smiles[mol_id]
        
        # Add number of components
        result_row['num_components'] = len(datapoint.smiles)
        
        # Add mole fractions as the model saw them
        result_row['mole_fractions'] = str(datapoint.molefracs)
        
        # Add target value
        if datapoint.targets is not None and len(datapoint.targets) > 0:
            result_row['CN_Measured'] = datapoint.targets[0]
        else:
            result_row['CN_Measured'] = np.nan
        
        # Add predictions from each model
        for pred_id, model_file in enumerate(pred_args.model_path):
            result_row[f'CN_pred_model_{pred_id}'] = all_preds[model_file][idx][0]
        
        results_list.append(result_row)
    
    df_results = pd.DataFrame(results_list)
    
    # Try to match names from original CSV based on SMILES
    df_original = pd.read_csv(formatted_csv)
    if 'Name' in df_original.columns:
        # Create a lookup key from SMILES for matching
        def make_key(row):
            smiles = []
            for i in range(1, 12):
                col = f'fuel{i}_inchi'
                if col in df_original.columns and pd.notna(row.get(col)):
                    smiles.append(str(row[col]))
            return '|'.join(sorted(smiles))  # Sort to handle order variations
        
        original_lookup = {}
        for _, row in df_original.iterrows():
            key = make_key(row)
            original_lookup[key] = row.get('Name', '')
        
        # Match results to original names
        names = []
        for _, row in df_results.iterrows():
            smiles = []
            for i in range(1, 6):
                col = f'fuel_{i}_smiles'
                if col in df_results.columns and pd.notna(row.get(col)) and row[col]:
                    smiles.append(str(row[col]))
            key = '|'.join(sorted(smiles))
            names.append(original_lookup.get(key, f'mixture_{len(names)}'))
        
        df_results.insert(0, 'Name', names)
    
    # Add individual model predictions
    for pred_id, model_file in enumerate(pred_args.model_path):
        df_results[f'CN_pred_model_{pred_id}'] = np.array(all_preds[model_file])[:, 0]
    
    # Calculate ensemble statistics
    pred_cols = [c for c in df_results.columns if 'CN_pred_model' in c]
    df_results['CN_predicted'] = df_results[pred_cols].mean(axis=1)
    df_results['CN_pred_std'] = df_results[pred_cols].std(axis=1)
    
    # Calculate errors (all rows should have measured values now)
    df_results['Error'] = df_results['CN_predicted'] - df_results['CN_Measured']
    df_results['Absolute_Error'] = np.abs(df_results['Error'])
    
    # Print statistics
    mae = df_results['Absolute_Error'].mean()
    rmse = np.sqrt((df_results['Error']**2).mean())
    logger(f"\n{'='*60}")
    logger(f"Prediction Statistics (n={len(df_results)}):")
    logger(f"  MAE:  {mae:.3f}")
    logger(f"  RMSE: {rmse:.3f}")
    logger(f"{'='*60}")
    
    # Save results
    output_file = os.path.join(output_dir, 'cn_predictions.csv')
    df_results.to_csv(output_file, index=False)
    logger(f"\nResults saved to {output_file}")
    
    # Save summary statistics
    summary = {
        'Total_Mixtures': len(df_results),
        'Models_Used': len(pred_args.model_path),
        'Mean_CN_Measured': df_results['CN_Measured'].mean(),
        'Mean_CN_Predicted': df_results['CN_predicted'].mean(),
        'Std_CN_Predicted': df_results['CN_predicted'].std(),
        'MAE': mae,
        'RMSE': rmse
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, 'prediction_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    
    return df_results


if __name__ == "__main__":
    # ========== CONFIGURATION - UPDATE THESE PATHS ==========
    INPUT_CSV = "data/database/mixture_database.csv"  # Your CSV file
    FORMATTED_CSV = "data/database/formatted_mixtures.csv"
    MODEL_DIR = "core/predictors/mixture/solvation_predictor/trained_models/DCN"
    OUTPUT_DIR = "results/cn_predictions_output"
    # =========================================================
    
    print("="*60)
    print("CN (Cetane Number) Prediction Pipeline")
    print("="*60)
    
    # Verify model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"\nERROR: Model directory not found: {MODEL_DIR}")
        print("Please update MODEL_DIR to point to your DCN models.")
        sys.exit(1)
    
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    print(f"\nFound {len(model_files)} model files in {MODEL_DIR}")
    
    # Step 1: Format data (only include rows with DCN values)
    print("\n" + "="*60)
    print("Step 1: Formatting data (filtering for rows with DCN)...")
    print("="*60)
    
    if not os.path.exists(INPUT_CSV):
        print(f"\nERROR: Input file not found: {INPUT_CSV}")
        print("Please update INPUT_CSV to point to your mixture data.")
        sys.exit(1)
    
    df_formatted = prepare_mixture_data(INPUT_CSV, FORMATTED_CSV, only_with_dcn=True)
    
    if len(df_formatted) == 0:
        print("\n✗ ERROR: No mixtures with DCN values found in the input file!")
        print("Please ensure your CSV has 'CN_Measured' or 'CN_Regression' columns with valid values.")
        sys.exit(1)
    
    # Step 2: Run predictions
    print("\n" + "="*60)
    print("Step 2: Running predictions...")
    print("="*60)
    
    try:
        df_results = run_cn_predictions(FORMATTED_CSV, MODEL_DIR, OUTPUT_DIR)
        
        print("\n" + "="*60)
        print("✓ Prediction complete!")
        print("="*60)
        print(f"\nResults saved to:")
        print(f"  - {OUTPUT_DIR}/cn_predictions.csv")
        print(f"  - {OUTPUT_DIR}/prediction_summary.csv")
        print(f"  - {OUTPUT_DIR}/logger.log")
        print("\nFirst few predictions:")
        print(df_results[['Name', 'CN_predicted', 'CN_pred_std', 'CN_Measured', 'Absolute_Error']].head())
        
    except Exception as e:
        print(f"\n✗ ERROR during prediction:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)