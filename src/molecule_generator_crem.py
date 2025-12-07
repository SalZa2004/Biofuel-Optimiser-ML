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

PROJECT_ROOT = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(PROJECT_ROOT)

from cn_predictor_model.cn_model.model import CetanePredictor
from cn_predictor_model.train import FeatureSelector, featurize_df

# Path to your clean model
model_path = os.path.join(PROJECT_ROOT, "cn_predictor_model","cn_model","artifacts", "model.joblib")
selector_path = os.path.join(PROJECT_ROOT, "cn_predictor_model","cn_model","artifacts", "selector.joblib")
# Path to CREM database
CREM_DB_PATH = os.path.join(PROJECT_ROOT, "chembl22_sa2.db")

# Verify files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")


# Load the clean model
print("Loading model...")
model_pkl = joblib.load(model_path)
print("✓ Model loaded successfully")

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


def run_evolution(target_cn, generations=20, population_size=50, mutations_per_parent=5):
    """
    Run evolutionary algorithm using CREM for mutations.
    
    Args:
        target_cn: Target cetane number to optimize for
        generations: Number of evolutionary generations
        population_size: Size of population to maintain
        mutations_per_parent: Number of mutations to try per parent molecule
        
    Returns:
        DataFrame with evolved molecules sorted by fitness
    """
    print(f"🧬 STARTING EVOLUTION (Target CN: {target_cn})")
    print(f"Using CREM database: {CREM_DB_PATH}")
    
    initial_smiles = df["SMILES"]
    population = []
    seen_smiles = set()

    print("Initializing population from dataset...")
    for s in initial_smiles:
        score = predictor.predict(s)
        score = float(score[0])
        if score is not None:
            population.append({
                'smiles': s, 
                'cn': score, 
                'error': abs(score - target_cn)
            })
            seen_smiles.add(s)

    # Safety check
    if len(population) == 0:
        print("\n❌ CRITICAL ERROR: No valid molecules could be initialized.")
        print("Check that the model and data_prep.df are working correctly.")
        return pd.DataFrame()

    print(f"✓ Initialized with {len(population)} valid molecules\n")

    for gen in range(generations):
        # Sort by fitness (lowest error = best)
        population.sort(key=lambda x: x['error'])
        
        best = population[0]
       

        avg_error = np.mean([p['error'] for p in population])
        
        print(f"  Gen {gen+1:2d}/{generations} | Pop: {len(population):3d} | "
              f"Best CN: {best['cn']:5.1f} (err: {best['error']:4.1f}) | "
              f"Avg err: {avg_error:4.1f}")
        wandb.log({
            "generation": gen+1,
            "best_cn": best["cn"],
            "best_error": best["error"],
            "best_smiles": best["smiles"]
            })

        

        # Select top 50% as survivors (elitism)
        survivors = population[:population_size // 2]
        new_population = survivors[:]  # Keep the best performers
        
        # Generate offspring using CREM mutations
        attempts = 0
        max_attempts = population_size * 10  # Prevent infinite loops
        
        while len(new_population) < population_size and attempts < max_attempts:
            attempts += 1
            
            # Select parent (random from survivors)
            # Could implement tournament selection or fitness-proportional selection here
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
            
            # Evaluate each mutation
            for child_smi in child_smiles_list:
                if child_smi and child_smi not in seen_smiles:
                    cn = predictor.predict(child_smi)
                    if cn is not None:
                        new_entry = {
                            'smiles': child_smi, 
                            'cn': cn, 
                            'error': abs(cn - target_cn)
                        }
                        new_population.append(new_entry)
                        seen_smiles.add(child_smi)
                        
                        # Stop if we've reached population size
                        if len(new_population) >= population_size:
                            break
            
            # Break outer loop if we've filled the population
            if len(new_population) >= population_size:
                break
        
        population = new_population
        
        if attempts >= max_attempts:
            print(f"  ⚠ Warning: Reached max attempts, population size: {len(population)}")

    print("\n" + "="*70)
    return pd.DataFrame(population).sort_values('error')


if __name__ == "__main__":
    # Check if model loaded correctly

    
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH CREM")
    print("="*70)
    
    # Get target CN from user
    try:
        target_cn = float(input("Enter target CN value: "))
    except ValueError:
        print("Invalid input. Using default target CN = 50")
        target_cn = 50
    
    # Run the evolution
    print(f"\nStarting evolution to find molecules with CN ≈ {target_cn}")
    print("Parameters: 15 generations, population size 30, 5 mutations per parent\n")

    wandb.init(
    project="cetane-number-optimization",
    config={
        "generations": 15,
        "population_size": 30,
        "model": "ExtrTress_CN_Predictor",
        "mutation": "CREM"  # or "Basic" if you switch
    }
)

    
    df_results = run_evolution(
        target_cn, 
        generations=15, 
        population_size=20,
        mutations_per_parent=5
    )
    
    # Log final table
    wandb.log({
        "final_results": wandb.Table(dataframe=df_results)
    })

    # Save best molecule CN
    best = df_results.iloc[0]
    wandb.log({
        "best_smiles": best["smiles"],
        "best_cn": best["cn"],
        "best_error": best["error"],
    })

    wandb.finish()
    
    if len(df_results) == 0:
        print("\n❌ Evolution failed - no valid molecules generated")
        sys.exit(1)
    
    print("\n" + "="*70)
    print(f"TOP 10 GENERATED MOLECULES (Closest to CN={target_cn})")
    print("="*70)
    
    output_cols = ['smiles', 'cn', 'error']
    print(df_results.head(10)[output_cols].to_string(index=False))
    
    # Save results
    output_file = "top_generated_molecules_crem.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved all results to '{output_file}'")
    print(f"✓ Total unique molecules explored: {len(df_results)}")
    
    # Show some statistics
    print("\n" + "="*70)
    print("EVOLUTION STATISTICS")
    print("="*70)
    print(f"Best CN achieved: {df_results.iloc[0]['cn']:.2f}")
    print(f"Target CN: {target_cn:.2f}")
    print(f"Best error: {df_results.iloc[0]['error']:.2f}")
    print(f"Average error (top 10): {df_results.head(10)['error'].mean():.2f}")