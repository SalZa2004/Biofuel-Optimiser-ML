import os
import sys
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from crem.crem import mutate_mol  # CREM import
from data_prep import df  
import wandb
from mordred import Calculator, descriptors
PROJECT_ROOT = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(PROJECT_ROOT)

from cn_predictor_model.cn_model.model import CetanePredictor
from cn_predictor_model.train import FeatureSelector, featurize_df
from sklearn.base import BaseEstimator, RegressorMixin
class MetaModel(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return sum(predictions) / len(self.models)


# Path to your clean model
model_path = os.path.join(PROJECT_ROOT, "cn_predictor_model","cn_model","artifacts", "model.joblib")
selector_path = os.path.join(PROJECT_ROOT, "cn_predictor_model","cn_model","artifacts", "selector.joblib")
# Path to CREM database
CREM_DB_PATH = os.path.join(PROJECT_ROOT, "chembl22_sa2.db")

# Verify files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

bp_model_path = os.path.join(PROJECT_ROOT, "boiling_point", "1_bounaceur_2025","example", "04_modele_final_NBP.joblib")
bp_descriptor_path = os.path.join(PROJECT_ROOT, "boiling_point", "1_bounaceur_2025", "example", "noms_colonnes_247_TC.txt")
# Load the clean model
print("Loading model...")
model_pkl = joblib.load(model_path)
print("✓ Model loaded successfully")
# Load boiling point model
print("Loading BP model...")
bp_model = joblib.load(bp_model_path)
print("✓ BP Model loaded successfully")

predictor = CetanePredictor()

def mutate_molecule_crem(mol, db_path=CREM_DB_PATH, max_mutations=None):

    try:
        # CREM mutate_mol is a generator that yields SMILES strings
        mutants = list(mutate_mol(
            mol, 
            db_name=db_path,
            max_size=1,         # Maximum change in heavy atom count 
            return_mol=False    # Return SMILES strings instead of mol objects
        ))
        
        return mutants if mutants else []
        
    except Exception as e:
        # CREM can fail for various reasons (molecule too large, no matches, etc.)
        # This is expected behavior, just return empty list
        return []
# Load BP descriptors
with open(bp_descriptor_path, 'r') as f:
    bp_descriptor_list = [line.strip() for line in f]
bp_descriptor_list = bp_descriptor_list[1:]  # remove header if present

# Initialize Mordred calculator (do this once globally for efficiency)
mordred_calc = Calculator(descriptors, ignore_3D=False)

def predict_boiling_points_batch(smiles_list):
    """Predict BP for multiple molecules at once - MUCH FASTER than one-by-one"""
    if not smiles_list:
        return []
    
    valid_mols = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        if smi and "." not in smi:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
                valid_indices.append(i)
    
    if not valid_mols:
        return [None] * len(smiles_list)
    
    # Calculate all descriptors at once (this is where the speedup happens!)
    df_desc = mordred_calc.pandas(valid_mols)
    X = df_desc[bp_descriptor_list]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    
    # Predict all at once
    bp_preds = bp_model.predict(X) - 273.15
    
    # Map back to original order
    results = [None] * len(smiles_list)
    for i, bp in zip(valid_indices, bp_preds):
        if not (np.isnan(bp) or np.isinf(bp)):
            results[i] = float(bp)
    
    return results

def is_valid_boiling_point(bp: float, min_bp=180, max_bp=360) -> bool:
    """
    Check if boiling point is within acceptable range.
    
    Args:
        bp: Boiling point in Celsius
        min_bp: Minimum acceptable BP (default 180°C)
        max_bp: Maximum acceptable BP (default 360°C)
        
    Returns:
        True if BP is within range, False otherwise
    """
    if bp is None:
        return False
    return min_bp <= bp <= max_bp

def run_evolution(target_cn, generations=20, population_size=50, mutations_per_parent=5, 
                  min_bp=180, max_bp=360, use_bp_filter=True):
    """
    Run evolutionary algorithm using CREM for mutations with boiling point screening.
    
    Args:
        target_cn: Target cetane number to optimize for
        generations: Number of evolutionary generations
        population_size: Size of population to maintain
        mutations_per_parent: Number of mutations to try per parent molecule
        min_bp: Minimum acceptable boiling point (°C)
        max_bp: Maximum acceptable boiling point (°C)
        use_bp_filter: Whether to filter by boiling point
        
    Returns:
        DataFrame with evolved molecules sorted by fitness
    """
    print(f"🧬 STARTING EVOLUTION (Target CN: {target_cn})")
    print(f"Using CREM database: {CREM_DB_PATH}")
    if use_bp_filter:
        print(f"BP Filter: {min_bp}°C - {max_bp}°C")
    
    # Initialize variables BEFORE the loop
    population = []
    seen_smiles = set()
    bp_filtered_count = 0
    
    # Get initial SMILES list
    initial_smiles_list = df["SMILES"].tolist()
    
    print("Initializing population from dataset...")
    if use_bp_filter:
        print("Calculating boiling points in batch (much faster!)...")
        bp_predictions = predict_boiling_points_batch(initial_smiles_list)
    else:
        bp_predictions = [None] * len(initial_smiles_list)

    # Build initial population
    for s, bp in zip(initial_smiles_list, bp_predictions):
        cn_score = predictor.predict(s)
        cn_score = float(cn_score[0])
        if cn_score is not None:
            # Check BP filter if enabled
            if use_bp_filter:
                if is_valid_boiling_point(bp, min_bp, max_bp):
                    population.append({
                        'smiles': s, 
                        'cn': cn_score, 
                        'bp': bp,
                        'error': abs(cn_score - target_cn)
                    })
                    seen_smiles.add(s)
                else:
                    bp_filtered_count += 1
            else:
                # No BP filter
                population.append({
                    'smiles': s, 
                    'cn': cn_score,
                    'bp': None,
                    'error': abs(cn_score - target_cn)
                })
                seen_smiles.add(s)

    # Safety check
    if len(population) == 0:
        print("\n❌ CRITICAL ERROR: No valid molecules could be initialized.")
        print("Check that the model and data_prep.df are working correctly.")
        return pd.DataFrame()

    print(f"✓ Initialized with {len(population)} valid molecules")
    if use_bp_filter:
        print(f"  (Filtered out {bp_filtered_count} molecules due to BP constraints)\n")

    # Evolution loop
    for gen in range(generations):
        # Sort by fitness (lowest error = best)
        population.sort(key=lambda x: x['error'])
        
        best = population[0]
        avg_error = np.mean([p['error'] for p in population])
        
        bp_status = f"BP: {best['bp']:.1f}°C | " if use_bp_filter else ""
        print(f"  Gen {gen+1:2d}/{generations} | Pop: {len(population):3d} | "
              f"{bp_status}Best CN: {best['cn']:5.1f} (err: {best['error']:4.1f}) | "
              f"Avg err: {avg_error:4.1f}")
        
        log_dict = {
            "generation": gen+1,
            "best_cn": best["cn"],
            "best_error": best["error"],
            "best_smiles": best["smiles"],
            "population_size": len(population)
        }
        if use_bp_filter:
            log_dict["best_bp"] = best["bp"]
        
        wandb.log(log_dict)

        # Select top 50% as survivors (elitism)
        survivors = population[:population_size // 2]
        new_population = survivors[:]  # Keep the best performers
        
        # Generate offspring using CREM mutations
        attempts = 0
        max_attempts = population_size * 10  # Prevent infinite loops
        bp_rejected_this_gen = 0
        
        # Collect all children for batch processing
        children_batch = []
        
        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1
            
            # Select parent (random from survivors)
            parent_data = random.choice(survivors)
            parent_mol = Chem.MolFromSmiles(parent_data['smiles'])
            
            if parent_mol is None:
                continue
            
            # Generate mutations using CREM
            child_smiles_list = mutate_molecule_crem(
                parent_mol, 
                db_path=CREM_DB_PATH, 
                max_mutations=mutations_per_parent
            )
            
            # Filter out already seen molecules
            new_children = [s for s in child_smiles_list if s and s not in seen_smiles]
            children_batch.extend(new_children)
            
            # Process batch when we have enough children or need to fill population
            if len(children_batch) >= 20 or len(new_population) + len(children_batch) >= population_size:
                # Batch predict BP if filter enabled
                if use_bp_filter and children_batch:
                    bp_batch = predict_boiling_points_batch(children_batch)
                else:
                    bp_batch = [None] * len(children_batch)
                
                # Evaluate each child
                for child_smi, bp in zip(children_batch, bp_batch):
                    if child_smi in seen_smiles:
                        continue
                        
                    cn = predictor.predict(child_smi)
                    if cn is not None:
                        # Check boiling point if filter is enabled
                        if use_bp_filter:
                            if not is_valid_boiling_point(bp, min_bp, max_bp):
                                bp_rejected_this_gen += 1
                                continue
                            
                            new_entry = {
                                'smiles': child_smi, 
                                'cn': cn,
                                'bp': bp,
                                'error': abs(cn - target_cn)
                            }
                        else:
                            new_entry = {
                                'smiles': child_smi, 
                                'cn': cn,
                                'bp': None,
                                'error': abs(cn - target_cn)
                            }
                        
                        new_population.append(new_entry)
                        seen_smiles.add(child_smi)
                        
                        # Stop if we've reached population size
                        if len(new_population) >= population_size:
                            break
                
                # Clear batch for next round
                children_batch = []
                
                # Break if we've filled the population
                if len(new_population) >= population_size:
                    break
        
        population = new_population
        
        if use_bp_filter and bp_rejected_this_gen > 0:
            print(f"    (Rejected {bp_rejected_this_gen} molecules due to BP constraints)")
        
        if attempts >= max_attempts:
            print(f"  ⚠ Warning: Reached max attempts, population size: {len(population)}")

    print("\n" + "="*70)
    return pd.DataFrame(population).sort_values('error')


if __name__ == "__main__":
    # Check if model loaded correctly
    
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH CREM + BP SCREENING")
    print("="*70)
    
    # Get target CN from user
    try:
        target_cn = float(input("Enter target CN value: "))
    except ValueError:
        print("Invalid input. Using default target CN = 50")
        target_cn = 50
    
    # Ask about BP filtering
    use_bp = input("Use boiling point filter (180-360°C)? [Y/n]: ").strip().lower()
    use_bp_filter = use_bp != 'n'
    
    # Run the evolution
    print(f"\nStarting evolution to find molecules with CN ≈ {target_cn}")
    print("Parameters: 15 generations, population size 20, 5 mutations per parent\n")

    config = {
        "generations": 15,
        "population_size": 20,
        "target_cn": target_cn,
        "model": "ExtraTrees_CN_Predictor",
        "mutation": "CREM",
        "bp_filter": use_bp_filter
    }
    
    if use_bp_filter:
        config["min_bp"] = 180
        config["max_bp"] = 360

    wandb.init(
        project="cetane-number-optimization",
        config=config
    )
    
    df_results = run_evolution(
        target_cn, 
        generations=15, 
        population_size=20,
        mutations_per_parent=5,
        use_bp_filter=use_bp_filter
    )
    
    # Log final table
    wandb.log({
        "final_results": wandb.Table(dataframe=df_results)
    })

    # Save best molecule CN
    if len(df_results) > 0:
        best = df_results.iloc[0]
        best_log = {
            "best_smiles": best["smiles"],
            "best_cn": best["cn"],
            "best_error": best["error"],
        }
        if use_bp_filter:
            best_log["best_bp"] = best["bp"]
        
        wandb.log(best_log)

    wandb.finish()
    
    if len(df_results) == 0:
        print("\n❌ Evolution failed - no valid molecules generated")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(f"TOP 10 GENERATED MOLECULES (Closest to CN={target_cn})")
    print("="*70)
    
    output_cols = ['smiles', 'cn', 'error']
    if use_bp_filter:
        output_cols.insert(2, 'bp')
    
    print(df_results.head(10)[output_cols].to_string(index=False))
    
    # Save results
    output_file = "top_generated_molecules_crem_bp.csv"
    output_path = f"results/{output_file}"
    df_results.to_csv(output_path, index=False)

    print(f"\n✓ Saved all results to '{output_path}'")
    print(f"✓ Total unique molecules explored: {len(df_results)}")

    
    # Show some statistics
    print("\n" + "="*70)
    print("EVOLUTION STATISTICS")
    print("="*70)
    print(f"Best CN achieved: {df_results.iloc[0]['cn']:.2f}")
    print(f"Target CN: {target_cn:.2f}")
    print(f"Best error: {df_results.iloc[0]['error']:.2f}")
    print(f"Average error (top 10): {df_results.head(10)['error'].mean():.2f}")