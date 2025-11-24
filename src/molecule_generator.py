import os
import sys
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
# FIXED IMPORT LINE BELOW:
from rdkit.Chem import Descriptors, AllChem

# ============================================================
# 1. SETUP & MODEL LOADING
# ============================================================

PROJECT_ROOT = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ... rest of the script remains the same

# Try to import your local feature selector
try:
    from feature_selection import FeatureSelector, prepare_prediction_features
    # Load your specific models
    model_path = os.path.join(PROJECT_ROOT, "extratrees_model.pkl")
    selector_path = os.path.join(PROJECT_ROOT, "feature_selector.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(selector_path):
        raise FileNotFoundError("Could not find model or selector pkl files.")
        
    model_pkl = joblib.load(model_path)
    selector = joblib.load(selector_path)
    print("✓ Models loaded successfully")
    
except ImportError:
    print("⚠ WARNING: Could not import 'feature_selection'. Using dummy predictor for testing.")
    # DUMMY PREDICTOR (Delete this block when running with real files)
    def prepare_prediction_features(smiles_list, selector): return np.random.rand(len(smiles_list), 10)
    class DummyModel: 
        def predict(self, X): return [random.uniform(40, 80)]
    model_pkl = DummyModel()
    selector = None

# ============================================================
# 2. ROBUST PREDICTION FUNCTION
# ============================================================
# ============================================================
# 2. DEBUGGED PREDICTION FUNCTION
# ============================================================

def predict_cn(smiles: str) -> float:
    """Predict CN with error printing so we know why it fails"""
    try:
        if not smiles or "." in smiles: 
            return None
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            return None
        
        # Calculate features
        # Note: We pass [smiles] as a list because most feature calculators expect a list/iterable
        X = prepare_prediction_features([smiles], selector)
        
        # Debugging checks
        if X is None:
            print(f"⚠ Feature calc returned None for {smiles}")
            return None
            
        # Handle cases where X might be a dataframe or numpy array
        if hasattr(X, 'shape'):
            if X.shape[0] == 0:
                print(f"⚠ Feature calc returned empty rows for {smiles}")
                return None
            # OPTIONAL: Check shape match if you know the model input size
            # print(f"DEBUG: Shape for {smiles} is {X.shape}") 
            
        prediction = model_pkl.predict(X)[0]
        
        if np.isnan(prediction) or np.isinf(prediction): 
            return None
            
        return float(prediction)
        
    except Exception as e:
        # --- THIS IS THE KEY CHANGE: PRINT THE ERROR ---
        print(f"❌ Error predicting {smiles}: {e}")
        import traceback
        traceback.print_exc() # This will print the exact line number of the error
        return None


# 3. MOLECULAR MUTATION ENGINE (The "Generator")
# ============================================================

def mutate_molecule(mol):
    """
    Randomly mutates a molecule:
    1. Add a Carbon atom
    2. Remove an atom
    3. Modify bond order (Single <-> Double)
    """
    try:
        rw_mol = Chem.RWMol(mol)
        num_atoms = rw_mol.GetNumAtoms()
        
        # If molecule is tiny, force ADD
        # If molecule is huge, force REMOVE
        choice = random.random()
        
        # --- ACTION 1: ADD ATOM (Expand) ---
        if choice < 0.5 or num_atoms < 5:
            # Pick a random atom to attach to
            if num_atoms == 0: return None
            idx = random.randint(0, num_atoms - 1)
            atom = rw_mol.GetAtomWithIdx(idx)
            
            # Only attach if valence allows (simple check)
            if atom.GetTotalDegree() < 4:
                new_idx = rw_mol.AddAtom(Chem.Atom(6)) # Add Carbon
                rw_mol.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        
        # --- ACTION 2: REMOVE ATOM (Prune) ---
        elif choice < 0.8 and num_atoms > 5:
            # Pick random atom
            idx = random.randint(0, num_atoms - 1)
            # Don't remove if it splits the molecule (simple heuristic: remove degree 1 atoms)
            atom = rw_mol.GetAtomWithIdx(idx)
            if atom.GetDegree() == 1:
                rw_mol.RemoveAtom(idx)
            else:
                return None # Skip complex removals to avoid breaking rings
                
        # --- ACTION 3: TWEAK BOND (Chemistry) ---
        else:
            bonds = rw_mol.GetBonds()
            if not bonds: return None
            bond = random.choice(bonds)
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.SINGLE)

        # Sanitize (Checks valences, aromaticity, etc.)
        Chem.SanitizeMol(rw_mol)
        return rw_mol
    except:
        return None

# ============================================================
# 4. GENETIC ALGORITHM LOOP
# ============================================================
def run_evolution(target_cn, generations=20, population_size=50):
    print(f"\n🧬 STARTING EVOLUTION (Target CN: {target_cn})")
    
    initial_smiles = [
        "CCCCCC", "C1CCCCC1", "c1ccccc1", "CC(C)C", 
        "CCCCCCC", "C1=CC=CC=C1", "CCO", "CCC(=O)O"
    ]
    
    population = []
    seen_smiles = set()
    
    print("  Initializing population...")
    for s in initial_smiles:
        score = predict_cn(s)
        if score is not None:
            population.append({'smiles': s, 'cn': score, 'error': abs(score - target_cn)})
            seen_smiles.add(s)

    # --- SAFETY CHECK ---
    if len(population) == 0:
        print("\n⛔ CRITICAL ERROR: No valid molecules could be initialized.")
        print("This usually means 'prepare_prediction_features' is failing or the model input shape is wrong.")
        print("Check the error logs printed above.")
        return pd.DataFrame() # Return empty to avoid crash

    # 2. Evolution Loop
    for gen in range(generations):
        
        population.sort(key=lambda x: x['error'])
        
        best = population[0] # This won't crash now because we checked len(population)
        print(f"  Gen {gen+1}/{generations} | Pop: {len(population)} | Best CN: {best['cn']:.1f} (SMI: {best['smiles']})")
        
        survivors = population[:population_size // 2]
        new_population = survivors[:]
        
        attempts = 0
        while len(new_population) < population_size and attempts < 200:
            attempts += 1
            parent_data = random.choice(survivors)
            parent_mol = Chem.MolFromSmiles(parent_data['smiles'])
            
            # Use isomericSmiles=False to reduce complexity
            child_mol = mutate_molecule(parent_mol)
            
            if child_mol:
                child_smi = Chem.MolToSmiles(child_mol, isomericSmiles=False)
                
                if child_smi not in seen_smiles:
                    cn = predict_cn(child_smi)
                    if cn is not None:
                        new_entry = {'smiles': child_smi, 'cn': cn, 'error': abs(cn - target_cn)}
                        new_population.append(new_entry)
                        seen_smiles.add(child_smi)
        
        population = new_population

    return pd.DataFrame(population).sort_values('error')

# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    if model_pkl is None and selector is None:
        print("\n ERROR: Feature selection logic missing. Cannot run real predictions.")
    else:
        # Run the evolution
        target_cn = int(input("Enter target CN value : "))
        df_results = run_evolution(target_cn, generations=15, population_size=30)
        
        print("\n" + "="*60)
        print("🏆 TOP 10 GENERATED MOLECULES (Closest to CN= {target_cn})")
        print("="*60)
        
        output_cols = ['smiles', 'cn', 'error']
        print(df_results.head(10)[output_cols].to_string(index=False))
        
        # Save
        df_results.to_csv("evolved_molecules.csv", index=False)
        print("\n✓ Saved all results to 'evolved_molecules.csv'")