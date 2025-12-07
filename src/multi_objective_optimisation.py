import os
import sys
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from crem.crem import mutate_mol
import wandb

# === Project Paths ===
PROJECT_ROOT = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(PROJECT_ROOT)


# === CN Predictor Imports ===
from cn_predictor_model.cn_model.model import CetanePredictor
from shared_features import FeatureSelector, featurize_df

# === YSI Predictor Imports ===
from ysi_predictor_model.ysi_model.model import YSIPredictor  # adjust if needed

# === Your Dataset (initial population) ===
from data_prep import df
from sklearn.base import BaseEstimator, RegressorMixin
# === BP model imports ===
from mordred import Calculator, descriptors

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


# -----------------------
# Batch Prediction Helpers
# -----------------------

def predict_cn_batch(cn_predictor, smiles_list):
    """
    Predict CN for a list of SMILES in batch.
    The CetanePredictor.predict() method already handles lists,
    so we just call it directly.
    """
    if not smiles_list:
        return []
    
    try:
        predictions = cn_predictor.predict(smiles_list)
        # Convert to list and handle None values
        results = []
        for pred in predictions:
            if pred is None or np.isnan(pred) or np.isinf(pred):
                results.append(None)
            else:
                results.append(float(pred))
        return results
    except Exception as e:
        print(f"Warning: Batch CN prediction failed: {e}")
        return [None] * len(smiles_list)


def predict_ysi_batch(ysi_predictor, smiles_list):
    """
    Predict YSI for a list of SMILES in batch.
    The YSIPredictor.predict() method already handles lists,
    so we just call it directly.
    """
    if not smiles_list:
        return []
    
    try:
        predictions = ysi_predictor.predict(smiles_list)
        # Convert to list and handle None values
        results = []
        for pred in predictions:
            if pred is None or np.isnan(pred) or np.isinf(pred):
                results.append(None)
            else:
                results.append(float(pred))
        return results
    except Exception as e:
        print(f"Warning: Batch YSI prediction failed: {e}")
        return [None] * len(smiles_list)


# -----------------------
# Utility Helpers
# -----------------------

def dominates(a, b):
    """
    Returns True if 'a' Pareto-dominates 'b'.
    Objective: minimize (cn_error) & minimize (ysi)
    """
    better_or_equal_cn = a["cn_error"] <= b["cn_error"]
    better_or_equal_ysi = a["ysi"] <= b["ysi"]
    strictly_better = (a["cn_error"] < b["cn_error"]) or (a["ysi"] < b["ysi"])
    return better_or_equal_cn and better_or_equal_ysi and strictly_better


def pareto_front(population):
    """Extract the set of non-dominated individuals."""
    front = []
    for p in population:
        dominated = False
        for q in population:
            if q is p:
                continue
            if dominates(q, p):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


# -----------------------
# BP Prediction
# -----------------------

bp_model_path = os.path.join(
    PROJECT_ROOT, "boiling_point", "1_bounaceur_2025",
    "example", "04_modele_final_NBP.joblib"
)
bp_descriptor_path = os.path.join(
    PROJECT_ROOT, "boiling_point", "1_bounaceur_2025",
    "example", "noms_colonnes_247_TC.txt"
)

print("Loading BP model...")
bp_model = joblib.load(bp_model_path)
print("✓ BP model loaded")

with open(bp_descriptor_path, 'r') as f:
    bp_descriptor_list = [line.strip() for line in f]
bp_descriptor_list = bp_descriptor_list[1:]

mordred_calc = Calculator(descriptors, ignore_3D=False)


def predict_boiling_points_batch(smiles_list):
    """Predict BP for a list of SMILES in batch."""
    if not smiles_list:
        return []

    valid_mols = []
    valid_idx = []

    for i, s in enumerate(smiles_list):
        if s and "." not in s:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid_mols.append(mol)
                valid_idx.append(i)

    if not valid_mols:
        return [None] * len(smiles_list)

    df_desc = mordred_calc.pandas(valid_mols)
    X = df_desc[bp_descriptor_list]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)

    bp_pred = bp_model.predict(X) - 273.15

    results = [None] * len(smiles_list)
    for i, bp in zip(valid_idx, bp_pred):
        if not (np.isnan(bp) or np.isinf(bp)):
            results[i] = float(bp)

    return results


def is_valid_boiling_point(bp: float, min_bp=180, max_bp=360):
    if bp is None:
        return False
    return min_bp <= bp <= max_bp


# -----------------------
# Mutation (CREM)
# -----------------------

CREM_DB_PATH = os.path.join(PROJECT_ROOT, "frag_db", "diesel_fragments.db")


def mutate_molecule_crem(mol, db_path=CREM_DB_PATH, max_mutations=None):
    try:
        mutants = list(mutate_mol(
            mol,
            db_name=db_path,
            max_size=1,
            return_mol=False
        ))
        return mutants if mutants else []
    except Exception:
        return []


# -----------------------
# Evolution Loop
# -----------------------

def run_evolution(target_cn,
                  minimize_ysi=True,
                  generations=20,
                  population_size=50,
                  mutations_per_parent=5,
                  min_bp=180,
                  max_bp=360,
                  use_bp_filter=True):
    """
    Evolution with optional multi-objective optimization:
    - If minimize_ysi=True: Multi-objective Pareto optimization (CN + YSI)
    - If minimize_ysi=False: Single objective optimization (CN only)
    """
    cn_predictor = CetanePredictor()
    ysi_predictor = YSIPredictor() if minimize_ysi else None

    population = []
    seen = set()
    bp_filtered = 0

    # === Initial Population ===
    init_smiles = df["SMILES"].tolist()
    
    print("Predicting properties for initial population...")
    cn_init = predict_cn_batch(cn_predictor, init_smiles)
    ysi_init = predict_ysi_batch(ysi_predictor, init_smiles) if minimize_ysi else [None] * len(init_smiles)
    bp_init = predict_boiling_points_batch(init_smiles) if use_bp_filter else [None] * len(init_smiles)

    for s, cn, ysi, bp in zip(init_smiles, cn_init, ysi_init, bp_init):
        if cn is None:
            continue
        if minimize_ysi and ysi is None:
            continue

        cn_err = abs(cn - target_cn)

        entry = {
            "smiles": s,
            "cn": cn,
            "cn_error": cn_err,
            "bp": bp,
        }
        if minimize_ysi:
            entry["ysi"] = ysi

        if use_bp_filter:
            if is_valid_boiling_point(bp, min_bp, max_bp):
                population.append(entry)
                seen.add(s)
            else:
                bp_filtered += 1
        else:
            population.append(entry)
            seen.add(s)

    if len(population) == 0:
        print("❌ No valid initial molecules")
        return pd.DataFrame(), pd.DataFrame()

    print(f"✓ Initial population size: {len(population)}")
    if use_bp_filter:
        print(f"Filtered out {bp_filtered} by BP screen")

    # === Evolutionary Loop ===
    for gen in range(generations):
        best_cn = min(population, key=lambda p: p["cn_error"])
        avg_cn_err = np.mean([p["cn_error"] for p in population])

        if minimize_ysi:
            # Pareto front for multi-objective
            front = pareto_front(population)
            best_ysi = min(population, key=lambda p: p["ysi"])
            avg_ysi = np.mean([p["ysi"] for p in population])

            print(
                f"Gen {gen+1}/{generations} | "
                f"Pop {len(population)} | "
                f"Best CN err: {best_cn['cn_error']:.3f} | "
                f"Best YSI: {best_ysi['ysi']:.3f} | "
                f"Avg CN err: {avg_cn_err:.3f} | "
                f"Avg YSI: {avg_ysi:.3f} | "
                f"Pareto size: {len(front)}"
            )

            wandb.log({
                "generation": gen+1,
                "best_cn_error": best_cn["cn_error"],
                "best_ysi": best_ysi["ysi"],
                "pareto_size": len(front),
                "population_size": len(population),
                "avg_cn_error": avg_cn_err,
                "avg_ysi": avg_ysi,
            })

            # Survivors = Pareto front
            survivors = front
            # Trim or fill survivors to half population
            if len(survivors) > population_size//2:
                survivors = sorted(survivors, key=lambda p: p["cn_error"] + p["ysi"])[:population_size//2]
            elif len(survivors) < population_size//2:
                remainder = [p for p in population if p not in survivors]
                remainder = sorted(remainder, key=lambda p: p["cn_error"] + p["ysi"])
                need = population_size//2 - len(survivors)
                survivors += remainder[:need]
        else:
            # Single objective: just minimize CN error
            print(
                f"Gen {gen+1}/{generations} | "
                f"Pop {len(population)} | "
                f"Best CN err: {best_cn['cn_error']:.3f} | "
                f"Avg CN err: {avg_cn_err:.3f}"
            )

            wandb.log({
                "generation": gen+1,
                "best_cn_error": best_cn["cn_error"],
                "population_size": len(population),
                "avg_cn_error": avg_cn_err,
            })

            # Keep top half by CN error
            survivors = sorted(population, key=lambda p: p["cn_error"])[:population_size//2]

        new_pop = survivors[:]

        # === Generate offspring ===
        attempts = 0
        max_attempts = population_size * 10
        
        all_children = []
        batch_size = max(50, population_size // 4)  # Larger batches = fewer featurization calls
        
        while len(new_pop) < population_size and attempts < max_attempts:
            attempts += 1
            parent = random.choice(survivors)
            mol = Chem.MolFromSmiles(parent["smiles"])
            if mol is None:
                continue

            children = mutate_molecule_crem(
                mol,
                db_path=CREM_DB_PATH,
                max_mutations=mutations_per_parent
            )
            children = [c for c in children if c and c not in seen]
            
            if children:
                all_children.extend(children)
                
                # Process in larger batches to reduce featurization overhead
                if len(all_children) >= batch_size or len(new_pop) + len(all_children) >= population_size:
                    print(f"  → Evaluating batch of {len(all_children)} offspring...")
                    # Batch predict for all accumulated children
                    child_cn = predict_cn_batch(cn_predictor, all_children)
                    child_ysi = predict_ysi_batch(ysi_predictor, all_children) if minimize_ysi else [None] * len(all_children)
                    child_bp = predict_boiling_points_batch(all_children) if use_bp_filter else [None] * len(all_children)
                    
                    for c_smi, cn, ysi, bp in zip(all_children, child_cn, child_ysi, child_bp):
                        if cn is None:
                            continue
                        if minimize_ysi and ysi is None:
                            continue
                        if use_bp_filter and not is_valid_boiling_point(bp, min_bp, max_bp):
                            continue

                        entry = {
                            "smiles": c_smi,
                            "cn": cn,
                            "cn_error": abs(cn - target_cn),
                            "bp": bp,
                        }
                        if minimize_ysi:
                            entry["ysi"] = ysi

                        new_pop.append(entry)
                        seen.add(c_smi)

                        if len(new_pop) >= population_size:
                            break
                    
                    all_children = []  # Reset after processing
                    
                    if len(new_pop) >= population_size:
                        break
        
        # Process any remaining children
        if all_children and len(new_pop) < population_size:
            print(f"  → Evaluating final batch of {len(all_children)} offspring...")
            child_cn = predict_cn_batch(cn_predictor, all_children)
            child_ysi = predict_ysi_batch(ysi_predictor, all_children) if minimize_ysi else [None] * len(all_children)
            child_bp = predict_boiling_points_batch(all_children) if use_bp_filter else [None] * len(all_children)
            
            for c_smi, cn, ysi, bp in zip(all_children, child_cn, child_ysi, child_bp):
                if cn is None:
                    continue
                if minimize_ysi and ysi is None:
                    continue
                if use_bp_filter and not is_valid_boiling_point(bp, min_bp, max_bp):
                    continue

                entry = {
                    "smiles": c_smi,
                    "cn": cn,
                    "cn_error": abs(cn - target_cn),
                    "bp": bp,
                }
                if minimize_ysi:
                    entry["ysi"] = ysi

                new_pop.append(entry)
                seen.add(c_smi)

                if len(new_pop) >= population_size:
                    break

        population = new_pop

    # === Final ===
    final_df = pd.DataFrame(population)
    
    if minimize_ysi:
        pareto_df = pd.DataFrame(pareto_front(population))
        pareto_df = pareto_df[pareto_df['cn_error'] < 5]  # Remove impractical solutions
        final_df = final_df.sort_values(["cn_error", "ysi"], ascending=[True, True])
        pareto_df = pareto_df.sort_values(["cn_error", "ysi"], ascending=[True, True])
        
        # Add rank columns
        final_df.insert(0, 'rank', range(1, len(final_df) + 1))
        if not pareto_df.empty:
            pareto_df.insert(0, 'rank', range(1, len(pareto_df) + 1))
    else:
        pareto_df = pd.DataFrame()  # Empty for single objective
        final_df = final_df.sort_values("cn_error", ascending=True)
        # Add rank column
        final_df.insert(0, 'rank', range(1, len(final_df) + 1))

    return final_df, pareto_df


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    target = float(input("Enter target CN: ") or "50")
    minimize_ysi_input = input("Minimise YSI (y/n): ").strip().lower()
    minimize_ysi = minimize_ysi_input == 'y' or minimize_ysi_input == 'yes'
    

    config = {
        "target_cn": target,
        "minimize_ysi": minimize_ysi,
        "generations": 20,
        "population_size": 100,
        "bp_filter": "y",
    }

    project_name = "cetane-ysi-pareto" if minimize_ysi else "cetane-optimization"
    wandb.init(project=project_name, config=config)

    final_df, pareto_df = run_evolution(
        target,
        minimize_ysi=minimize_ysi,
        generations=20,
        population_size=100,
        mutations_per_parent=5,
        use_bp_filter=True
    )

    wandb.log({
        "final": wandb.Table(dataframe=final_df),
    })
    
    if minimize_ysi and not pareto_df.empty:
        wandb.log({
            "pareto": wandb.Table(dataframe=pareto_df),
        })

    wandb.finish()

    print("\n=== TOP 10 (sorted) ===")
    if minimize_ysi:
        print(final_df.head(10)[["rank","smiles","cn","cn_error","ysi","bp"]].to_string(index=False))
    else:
        print(final_df.head(10)[["rank","smiles","cn","cn_error","bp"]].to_string(index=False))

    if minimize_ysi and not pareto_df.empty:
        print("\n=== PARETO FRONT (ranked) ===")
        print(pareto_df[["rank","smiles","cn","cn_error","ysi","bp"]].head(20).to_string(index=False))

    # Save
    os.makedirs("results", exist_ok=True)
    final_df.to_csv("results/final_population.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv("results/pareto_front.csv", index=False)

    print("\nSaved to results/")