import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from crem.crem import mutate_mol
import wandb
from sklearn.base import BaseEstimator, RegressorMixin
from mordred import Calculator, descriptors
from meta_model import MetaModel
# === Project Setup ===
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_ROOT))

from cn_predictor_model.cn_model.model import CetanePredictor
from shared_features import FeatureSelector, featurize_df
from ysi_predictor_model.ysi_model.model import YSIPredictor
from data_prep import df


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    target_cn: float
    minimize_ysi: bool = True
    generations: int = 10
    population_size: int = 100
    mutations_per_parent: int = 5
    min_bp: float = 60
    max_bp: float = 250
    use_bp_filter: bool = True
    batch_size: int = 50


@dataclass
class Molecule:
    """Represents a molecule with its properties."""
    smiles: str
    cn: float
    cn_error: float
    bp: Optional[float] = None
    ysi: Optional[float] = None
    
    def dominates(self, other: 'Molecule') -> bool:
        """Check if this molecule Pareto-dominates another."""
        better_cn = self.cn_error <= other.cn_error
        better_ysi = self.ysi <= other.ysi if self.ysi is not None else True
        strictly_better = self.cn_error < other.cn_error or (self.ysi is not None and self.ysi < other.ysi)
        return better_cn and better_ysi and strictly_better
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        d = {
            "smiles": self.smiles,
            "cn": self.cn,
            "cn_error": self.cn_error,
            "bp": self.bp,
        }
        if self.ysi is not None:
            d["ysi"] = self.ysi
        return d


class PropertyPredictor:
    """Handles batch prediction for all molecular properties."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.cn_predictor = CetanePredictor()
        self.ysi_predictor = YSIPredictor() if config.minimize_ysi else None
        self.bp_model, self.bp_descriptors = self._load_bp_model()
        self.mordred_calc = Calculator(descriptors, ignore_3D=False)
    
    def _load_bp_model(self) -> Tuple:
        """Load boiling point prediction model."""
        bp_dir = PROJECT_ROOT / "boiling_point" / "1_bounaceur_2025" / "example"
        model = joblib.load(bp_dir / "04_modele_final_NBP.joblib")
        
        with open(bp_dir / "noms_colonnes_247_TC.txt", 'r') as f:
            descriptors = [line.strip() for line in f][1:]
        
        print("✓ BP model loaded")
        return model, descriptors
    
    def _safe_predict(self, predictions: List, name: str) -> List[Optional[float]]:
        """Safely convert predictions, handling None/NaN/inf values."""
        results = []
        for pred in predictions:
            if pred is None or np.isnan(pred) or np.isinf(pred):
                results.append(None)
            else:
                results.append(float(pred))
        return results
    
    def predict_cn_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predict cetane number for a batch of SMILES."""
        if not smiles_list:
            return []
        try:
            predictions = self.cn_predictor.predict(smiles_list)
            return self._safe_predict(predictions, "CN")
        except Exception as e:
            print(f"Warning: Batch CN prediction failed: {e}")
            return [None] * len(smiles_list)
    
    def predict_ysi_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predict YSI for a batch of SMILES."""
        if not smiles_list or not self.ysi_predictor:
            return [None] * len(smiles_list)
        try:
            predictions = self.ysi_predictor.predict(smiles_list)
            return self._safe_predict(predictions, "YSI")
        except Exception as e:
            print(f"Warning: Batch YSI prediction failed: {e}")
            return [None] * len(smiles_list)
    
    def predict_bp_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predict boiling point for a batch of SMILES."""
        if not smiles_list or not self.config.use_bp_filter:
            return [None] * len(smiles_list)
        
        valid_mols, valid_idx = [], []
        for i, s in enumerate(smiles_list):
            if s and "." not in s:
                mol = Chem.MolFromSmiles(s)
                if mol is not None:
                    valid_mols.append(mol)
                    valid_idx.append(i)
        
        if not valid_mols:
            return [None] * len(smiles_list)
        
        df_desc = self.mordred_calc.pandas(valid_mols)
        X = df_desc[self.bp_descriptors]
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
        bp_pred = self.bp_model.predict(X) - 273.15
        
        results = [None] * len(smiles_list)
        for i, bp in zip(valid_idx, bp_pred):
            if not (np.isnan(bp) or np.isinf(bp)):
                results[i] = float(bp)
        
        return results
    
    def is_valid_bp(self, bp: Optional[float]) -> bool:
        """Check if boiling point is within valid range."""
        if bp is None or not self.config.use_bp_filter:
            return True
        return self.config.min_bp <= bp <= self.config.max_bp


class Population:
    """Manages the population of molecules."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.molecules: List[Molecule] = []
        self.seen_smiles: set = set()
    
    def add_molecule(self, mol: Molecule) -> bool:
        """Add a molecule if it's not already in the population."""
        if mol.smiles in self.seen_smiles:
            return False
        self.molecules.append(mol)
        self.seen_smiles.add(mol.smiles)
        return True
    
    def pareto_front(self) -> List[Molecule]:
        """Extract the Pareto front from the population."""
        if not self.config.minimize_ysi:
            return []
        
        front = []
        for mol in self.molecules:
            dominated = any(other.dominates(mol) for other in self.molecules if other is not mol)
            if not dominated:
                front.append(mol)
        return front
    
    def get_survivors(self) -> List[Molecule]:
        """Select survivors for the next generation."""
        target_size = self.config.population_size // 2
        
        if self.config.minimize_ysi:
            survivors = self.pareto_front()
            if len(survivors) > target_size:
                survivors = sorted(survivors, key=lambda m: m.cn_error + m.ysi)[:target_size]
            elif len(survivors) < target_size:
                remainder = [m for m in self.molecules if m not in survivors]
                remainder = sorted(remainder, key=lambda m: m.cn_error + m.ysi)
                survivors += remainder[:target_size - len(survivors)]
        else:
            survivors = sorted(self.molecules, key=lambda m: m.cn_error)[:target_size]
        
        return survivors
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert population to DataFrame."""
        df = pd.DataFrame([m.to_dict() for m in self.molecules])
        if self.config.minimize_ysi:
            df = df.sort_values(["cn_error", "ysi"], ascending=[True, True])
        else:
            df = df.sort_values("cn_error", ascending=True)
        df.insert(0, 'rank', range(1, len(df) + 1))
        return df


class MolecularEvolution:
    """Main evolutionary algorithm coordinator."""
    
    REP_DB_PATH = PROJECT_ROOT / "frag_db" / "diesel_fragments.db"
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.predictor = PropertyPredictor(config)
        self.population = Population(config)
    
    def _mutate_molecule(self, mol: Chem.Mol) -> List[str]:
        """Generate mutations for a molecule using CREM."""
        try:
            mutants = list(mutate_mol(
                mol,
                db_name=str(self.REP_DB_PATH),
                max_size=2,
                return_mol=False
            ))
            return [m for m in mutants if m and m not in self.population.seen_smiles]
        except Exception:
            return []
    
    def _create_molecules(self, smiles_list: List[str]) -> List[Molecule]:
        """Create Molecule objects from SMILES with predictions."""
        cn_preds = self.predictor.predict_cn_batch(smiles_list)
        ysi_preds = self.predictor.predict_ysi_batch(smiles_list)
        bp_preds = self.predictor.predict_bp_batch(smiles_list)
        
        molecules = []
        for s, cn, ysi, bp in zip(smiles_list, cn_preds, ysi_preds, bp_preds):
            if cn is None:
                continue
            if self.config.minimize_ysi and ysi is None:
                continue
            if not self.predictor.is_valid_bp(bp):
                continue
            
            molecules.append(Molecule(
                smiles=s,
                cn=cn,
                cn_error=abs(cn - self.config.target_cn),
                bp=bp,
                ysi=ysi
            ))
        
        return molecules
    
    def initialize_population(self, initial_smiles: List[str]) -> int:
        """Initialize the population from initial SMILES."""
        print("Predicting properties for initial population...")
        molecules = self._create_molecules(initial_smiles)
        
        for mol in molecules:
            self.population.add_molecule(mol)
        
        return len(molecules)
    
    def _log_generation_stats(self, generation: int):
        """Log statistics for the current generation."""
        mols = self.population.molecules
        best_cn = min(mols, key=lambda m: m.cn_error)
        avg_cn_err = np.mean([m.cn_error for m in mols])
        
        log_dict = {
            "generation": generation,
            "best_cn_error": best_cn.cn_error,
            "population_size": len(mols),
            "avg_cn_error": avg_cn_err,
        }
        
        print_msg = (f"Gen {generation}/{self.config.generations} | "
                    f"Pop {len(mols)} | "
                    f"Best CN err: {best_cn.cn_error:.3f} | "
                    f"Avg CN err: {avg_cn_err:.3f}")
        
        if self.config.minimize_ysi:
            front = self.population.pareto_front()
            best_ysi = min(mols, key=lambda m: m.ysi)
            avg_ysi = np.mean([m.ysi for m in mols])
            
            log_dict.update({
                "best_ysi": best_ysi.ysi,
                "pareto_size": len(front),
                "avg_ysi": avg_ysi,
            })
            
            print_msg += (f" | Best YSI: {best_ysi.ysi:.3f} | "
                         f"Avg YSI: {avg_ysi:.3f} | "
                         f"Pareto size: {len(front)}")
        
        print(print_msg)
        wandb.log(log_dict)
    
    def _generate_offspring(self, survivors: List[Molecule]) -> List[Molecule]:
        """Generate offspring from survivors."""
        target_size = self.config.population_size
        new_molecules = []
        all_children = []
        
        attempts = 0
        max_attempts = target_size * 10
        
        while len(new_molecules) < target_size - len(survivors) and attempts < max_attempts:
            attempts += 1
            parent = random.choice(survivors)
            mol = Chem.MolFromSmiles(parent.smiles)
            
            if mol is None:
                continue
            
            children = self._mutate_molecule(mol)
            all_children.extend(children[:self.config.mutations_per_parent])
            
            # Process in batches
            if len(all_children) >= self.config.batch_size:
                print(f"  → Evaluating batch of {len(all_children)} offspring...")
                batch_mols = self._create_molecules(all_children)
                new_molecules.extend(batch_mols)
                all_children = []
                
                if len(new_molecules) >= target_size - len(survivors):
                    break
        
        # Process remaining children
        if all_children and len(new_molecules) < target_size - len(survivors):
            print(f"  → Evaluating final batch of {len(all_children)} offspring...")
            batch_mols = self._create_molecules(all_children)
            new_molecules.extend(batch_mols)
        
        return new_molecules
    
    def evolve(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the evolutionary algorithm."""
        # Initialize
        init_count = self.initialize_population(df["SMILES"].tolist())
        if init_count == 0:
            print("❌ No valid initial molecules")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✓ Initial population size: {init_count}")
        
        # Evolution loop
        for gen in range(1, self.config.generations + 1):
            self._log_generation_stats(gen)
            
            survivors = self.population.get_survivors()
            offspring = self._generate_offspring(survivors)
            
            # Create new population
            new_pop = Population(self.config)
            for mol in survivors + offspring:
                new_pop.add_molecule(mol)
            
            self.population = new_pop
        
        # Generate results
        final_df = self.population.to_dataframe()
        
        if self.config.minimize_ysi:
            pareto_mols = self.population.pareto_front()
            pareto_df = pd.DataFrame([m.to_dict() for m in pareto_mols])
            if not pareto_df.empty:
                pareto_df = pareto_df[
                    (pareto_df['cn_error'] < 5) & (pareto_df['ysi'] < 50)
                ]
                pareto_df = pareto_df.sort_values(["cn_error", "ysi"], ascending=[True, True])
                pareto_df.insert(0, 'rank', range(1, len(pareto_df) + 1))
        else:
            pareto_df = pd.DataFrame()
        
        return final_df, pareto_df


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    
    target = float(input("Enter target CN: ") or "50")
    while target <= 40:
        print("⚠️  Target CN is too low, optimization may be challenging.")
        print("Consider using a higher target CN for better results.\n")
        print("Re-input target CN.")
        target = float(input("Enter target CN: ") or "50")
    minimize_ysi_input = input("Minimise YSI (y/n): ").strip().lower()
    minimize_ysi = minimize_ysi_input in ['y', 'yes']
    
    config = EvolutionConfig(
        target_cn=target,
        minimize_ysi=minimize_ysi
    )
    
    project_name = "cetane-ysi-pareto" if minimize_ysi else "cetane-optimization"
    wandb.init(project=project_name, config=config.__dict__)
    
    evolution = MolecularEvolution(config)
    final_df, pareto_df = evolution.evolve()
    
    # Log to wandb
    wandb.log({"final": wandb.Table(dataframe=final_df)})
    if minimize_ysi and not pareto_df.empty:
        wandb.log({"pareto": wandb.Table(dataframe=pareto_df)})
    wandb.finish()
    
    # Display results
    print("\n=== TOP 10 (sorted) ===")
    cols = ["rank", "smiles", "cn", "cn_error", "ysi", "bp"] if minimize_ysi else ["rank", "smiles", "cn", "cn_error", "bp"]
    print(final_df.head(10)[cols].to_string(index=False))
    
    if minimize_ysi and not pareto_df.empty:
        print("\n=== PARETO FRONT (ranked) ===")
        print(pareto_df[["rank", "smiles", "cn", "cn_error", "ysi", "bp"]].head(20).to_string(index=False))
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    final_df.to_csv(results_dir / "final_population.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)
    
    print("\nSaved to results/")


if __name__ == "__main__":
    main()