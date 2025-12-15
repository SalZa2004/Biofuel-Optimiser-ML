import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Callable
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from crem.crem import mutate_mol
import wandb
from sklearn.base import BaseEstimator, RegressorMixin

# === Project Setup ===
PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(PROJECT_ROOT))

from shared_features import FeatureSelector, featurize_df
from data_prep import df

class GenericPredictor:
    """Generic predictor that works for any property model."""
    
    def __init__(self, model_dir: Path, property_name: str):
        """
        Initialize predictor from a model directory.
        
        Args:
            model_dir: Path to the model directory containing artifacts/
            property_name: Name of the property (for display purposes)
        """
        print(f"Loading {property_name} Predictor...")
        
        artifact_dir = model_dir / "artifacts"
        model_path = artifact_dir / "model.joblib"
        selector_path = artifact_dir / "selector.joblib"
        
        # Debug info
        print(f">>> MODEL PATH: {model_path}")
        print(f">>> SELECTOR PATH: {selector_path}")
        print(f">>> MODEL EXISTS: {model_path.exists()}")
        print(f">>> SELECTOR EXISTS: {selector_path.exists()}")
        
        # Load artifacts
        self.model = joblib.load(model_path)
        self.selector = FeatureSelector.load(selector_path)
        self.property_name = property_name
        
        print(f"✓ {property_name} Predictor ready!\n")
    
    def predict(self, smiles_list):
        """Inference on a list of SMILES strings."""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        X_full = featurize_df(smiles_list, return_df=False)
        
        if X_full is None:
            print(f"⚠ Warning: No valid molecules found for {self.property_name}!")
            return []
        
        X_selected = self.selector.transform(X_full)
        predictions = self.model.predict(X_selected)
        return predictions.tolist()
    
    def predict_with_details(self, smiles_list):
        """Inference with valid/invalid info."""
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        
        df = pd.DataFrame({"SMILES": smiles_list})
        X_full, df_valid = featurize_df(df, return_df=True)
        
        col_name = f"Predicted_{self.property_name}"
        
        if X_full is None:
            return pd.DataFrame(columns=["SMILES", col_name, "Valid"])
        
        X_selected = self.selector.transform(X_full)
        predictions = self.model.predict(X_selected)
        
        df_valid[col_name] = predictions
        df_valid["Valid"] = True
        
        all_results = pd.DataFrame({"SMILES": smiles_list})
        all_results = all_results.merge(
            df_valid[["SMILES", col_name, "Valid"]],
            on="SMILES", how="left"
        )
        all_results["Valid"] = all_results["Valid"].fillna(False)
        
        return all_results


# Predictor paths relative to project root
PREDICTOR_PATHS = {
    'cn': PROJECT_ROOT / "cn_predictor_model" / "cn_model",
    'ysi': PROJECT_ROOT / "ysi_predictor_model" / "ysi_model",
    'bp': PROJECT_ROOT / "bp_predictor_model" / "bp_model",
    'density': PROJECT_ROOT / "density_predictor_model" / "density_model",
    'lhv': PROJECT_ROOT / "lhv_predictor_model" / "lhv_model",
    'dynamic_viscosity': PROJECT_ROOT / "dynamic_viscosity_predictor_model" / "dynamic_viscosity_model"
}

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    target_cn: float
    minimize_ysi: bool = True
    generations: int = 6
    population_size: int = 100
    mutations_per_parent: int = 5
    survivor_fraction: float = 0.5  
    min_bp: float = 60
    max_bp: float = 250
    min_dynamic_viscosity: float = 0.0
    max_dynamic_viscosity: float = 2.0
    min_density: float = 720
    min_lhv: float = 30
    use_bp_filter: bool = True
    use_density_filter: bool = True
    use_lhv_filter: bool = True
    use_dynamic_viscosity_filter: bool = True
    batch_size: int = 50
    max_offspring_attempts: int = 10  

@dataclass
class Molecule:
    """Represents a molecule with its properties."""
    smiles: str
    cn: float
    cn_error: float
    bp: Optional[float] = None
    ysi: Optional[float] = None
    density: Optional[float] = None
    lhv: Optional[float] = None
    dynamic_viscosity: Optional[float] = None
    
    def dominates(self, other: 'Molecule') -> bool:
        """Check if this molecule Pareto-dominates another."""
        better_cn = self.cn_error <= other.cn_error
        better_ysi = self.ysi <= other.ysi if self.ysi is not None else True
        strictly_better = (self.cn_error < other.cn_error or 
                          (self.ysi is not None and self.ysi < other.ysi))
        return better_cn and better_ysi and strictly_better
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class PropertyPredictor:
    """Handles batch prediction for all molecular properties."""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        
        # Initialize only the predictors we need
        self.predictors = {}
        
        # Always need CN predictor
        self.predictors['cn'] = GenericPredictor(
            PREDICTOR_PATHS['cn'], 
            'Cetane Number'
        )
        
        # Conditional predictors
        if config.minimize_ysi:
            self.predictors['ysi'] = GenericPredictor(
                PREDICTOR_PATHS['ysi'], 
                'YSI'
            )
        
        if config.use_bp_filter:
            self.predictors['bp'] = GenericPredictor(
                PREDICTOR_PATHS['bp'], 
                'Boiling Point'
            )
        
        if config.use_density_filter:
            self.predictors['density'] = GenericPredictor(
                PREDICTOR_PATHS['density'], 
                'Density'
            )
        
        if config.use_lhv_filter:
            self.predictors['lhv'] = GenericPredictor(
                PREDICTOR_PATHS['lhv'], 
                'Lower Heating Value'
            )
        
        if config.use_dynamic_viscosity_filter:
            self.predictors['dynamic_viscosity'] = GenericPredictor(
                PREDICTOR_PATHS['dynamic_viscosity'], 
                'Dynamic Viscosity'
            )
        
        # Define validation rules
        self.validators = {
            'bp': lambda v: self.config.min_bp <= v <= self.config.max_bp,
            'density': lambda v: v > self.config.min_density,
            'lhv': lambda v: v > self.config.min_lhv,
            'dynamic_viscosity': lambda v:  self.config.min_dynamic_viscosity < v <= self.config.max_dynamic_viscosity
        }
    
    def _safe_predict(self, predictions: List) -> List[Optional[float]]:
        """Safely convert predictions, handling None/NaN/inf values."""
        return [
            float(pred) if pred is not None and np.isfinite(pred) else None
            for pred in predictions
        ]
    
    def _predict_batch(self, property_name: str, smiles_list: List[str]) -> List[Optional[float]]:
        """Generic batch prediction method."""
        predictor = self.predictors.get(property_name)
        if not smiles_list or predictor is None:
            return [None] * len(smiles_list)
        
        try:
            predictions = predictor.predict(smiles_list)
            return self._safe_predict(predictions)
        except Exception as e:
            print(f"Warning: Batch {property_name.upper()} prediction failed: {e}")
            return [None] * len(smiles_list)
    
    def predict_all_properties(self, smiles_list: List[str]) -> Dict[str, List[Optional[float]]]:
        """Predict all properties for a batch of SMILES."""
        return {
            prop: self._predict_batch(prop, smiles_list)
            for prop in ['cn', 'ysi', 'bp', 'density', 'lhv', 'dynamic_viscosity']  # Check all possible properties
            if prop in self.predictors  # Only predict if predictor exists
        }
    
    def is_valid(self, property_name: str, value: Optional[float]) -> bool:
        """Check if a property value is valid according to config rules."""
        if value is None:
            return True
        validator = self.validators.get(property_name)
        return validator(value) if validator else True


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
    
    def add_molecules(self, molecules: List[Molecule]) -> int:
        """Add multiple molecules, return count added."""
        return sum(self.add_molecule(mol) for mol in molecules)
    
    def pareto_front(self) -> List[Molecule]:
        """Extract the Pareto front from the population."""
        if not self.config.minimize_ysi:
            return []
        
        return [
            mol for mol in self.molecules
            if not any(other.dominates(mol) for other in self.molecules if other is not mol)
        ]
    
    def get_survivors(self) -> List[Molecule]:
        """Select survivors for the next generation."""
        target_size = int(self.config.population_size * self.config.survivor_fraction)
        
        if self.config.minimize_ysi:
            survivors = self.pareto_front()
            
            # Sort key for combined objectives
            sort_key = lambda m: m.cn_error + m.ysi
            
            if len(survivors) > target_size:
                survivors = sorted(survivors, key=sort_key)[:target_size]
            elif len(survivors) < target_size:
                remainder = [m for m in self.molecules if m not in survivors]
                remainder = sorted(remainder, key=sort_key)
                survivors.extend(remainder[:target_size - len(survivors)])
        else:
            survivors = sorted(self.molecules, key=lambda m: m.cn_error)[:target_size]
        
        return survivors
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert population to DataFrame."""
        df = pd.DataFrame([m.to_dict() for m in self.molecules])
        
        sort_cols = ["cn_error", "ysi"] if self.config.minimize_ysi else ["cn_error"]
        df = df.sort_values(sort_cols, ascending=True)
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
        if not smiles_list:
            return []
        
        # Get all predictions at once
        predictions = self.predictor.predict_all_properties(smiles_list)
        
        molecules = []
        for i, smiles in enumerate(smiles_list):
            # Extract predictions for this molecule
            props = {k: v[i] for k, v in predictions.items()}
            
            # Validate required properties
            if props['cn'] is None:
                continue
            if self.config.minimize_ysi and props['ysi'] is None:
                continue
            
            # Validate filtered properties
            if not all(self.predictor.is_valid(k, props[k]) for k in ['bp', 'density', 'lhv', 'dynamic_viscosity']):
                continue
            
            molecules.append(Molecule(
                smiles=smiles,
                cn=props['cn'],
                cn_error=abs(props['cn'] - self.config.target_cn),
                bp=props['bp'],
                ysi=props['ysi'],
                density=props['density'],
                lhv=props['lhv'],
                dynamic_viscosity=props['dynamic_viscosity']
            ))
        
        return molecules
    
    def initialize_population(self, initial_smiles: List[str]) -> int:
        """Initialize the population from initial SMILES."""
        print("Predicting properties for initial population...")
        molecules = self._create_molecules(initial_smiles)
        return self.population.add_molecules(molecules)
    
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
        target_count = self.config.population_size - len(survivors)
        max_attempts = target_count * self.config.max_offspring_attempts
        
        all_children = []
        new_molecules = []
        
        for attempt in range(max_attempts):
            if len(new_molecules) >= target_count:
                break
            
            # Generate mutations
            parent = random.choice(survivors)
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                continue
            
            children = self._mutate_molecule(mol)
            all_children.extend(children[:self.config.mutations_per_parent])
            
            # Process in batches
            if len(all_children) >= self.config.batch_size:
                print(f"  → Evaluating batch of {len(all_children)} offspring...")
                new_molecules.extend(self._create_molecules(all_children))
                all_children = []
        
        # Process remaining children
        if all_children:
            print(f"  → Evaluating final batch of {len(all_children)} offspring...")
            new_molecules.extend(self._create_molecules(all_children))
        
        return new_molecules
    
    def _run_evolution_loop(self):
        """Run the main evolution loop."""
        for gen in range(1, self.config.generations + 1):
            self._log_generation_stats(gen)
            
            survivors = self.population.get_survivors()
            offspring = self._generate_offspring(survivors)
            
            # Create new population
            new_pop = Population(self.config)
            new_pop.add_molecules(survivors + offspring)
            self.population = new_pop
    
    def _generate_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate final results DataFrames."""
        final_df = self.population.to_dataframe()

        if self.config.minimize_ysi and "ysi" in final_df.columns:
            final_df = final_df[
                (final_df["cn_error"] < 5) &
                (final_df["ysi"] < 50)
            ].sort_values(["cn_error", "ysi"], ascending=True)
    
            # overwrite rank safely
            final_df["rank"] = range(1, len(final_df) + 1)
        
        if self.config.minimize_ysi:
            pareto_mols = self.population.pareto_front()
            pareto_df = pd.DataFrame([m.to_dict() for m in pareto_mols])
            
            if not pareto_df.empty:
                pareto_df = pareto_df[
                    (pareto_df['cn_error'] < 5) & (pareto_df['ysi'] < 50)
                ].sort_values(["cn_error", "ysi"], ascending=True)
                pareto_df.insert(0, 'rank', range(1, len(pareto_df) + 1))
        else:
            pareto_df = pd.DataFrame()
        
        return final_df, pareto_df
    
    def evolve(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the evolutionary algorithm."""
        # Initialize
        df_bins = pd.qcut(df["cn"], q=30)
        initial_smiles = (
            df.groupby(df_bins)
            .apply(lambda x: x.sample(20, random_state=42))
            .reset_index(drop=True)["SMILES"]
            .tolist()
        )
        init_count = self.initialize_population(initial_smiles)

        if init_count == 0:
            print("❌ No valid initial molecules")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✓ Initial population size: {init_count}")
        
        # Evolution
        self._run_evolution_loop()
        
        # Results
        return self._generate_results()


def get_user_config() -> EvolutionConfig:
    """Get configuration from user input."""
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    
    while True:
        target = float(input("Enter target CN: ") or "50")
        if target > 40:
            break
        print("⚠️  Target CN is too low, optimization may be challenging.")
        print("Consider using a higher target CN for better results.\n")
    
    minimize_ysi = input("Minimise YSI (y/n): ").strip().lower() in ['y', 'yes']
    
    return EvolutionConfig(target_cn=target, minimize_ysi=minimize_ysi)


def save_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, minimize_ysi: bool):
    """Save results to CSV files."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    final_df.to_csv(results_dir / "final_population.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)
    
    print("\nSaved to results/")


def display_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, minimize_ysi: bool):
    """Display results to console."""
    cols = (["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity"])
    
    print("\n=== Best Candidates ===")
    print(final_df.head(10)[cols].to_string(index=False))
    
    if minimize_ysi and not pareto_df.empty:
        print("\n=== PARETO FRONT (ranked) ===")
        print(pareto_df[["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity"]]
              .head(20).to_string(index=False))

def main():
    """Main execution function."""
    config = get_user_config()
    
    project_name = "cetane-ysi-pareto" if config.minimize_ysi else "cetane-optimization"
    wandb.init(project=project_name, config=asdict(config))
    
    evolution = MolecularEvolution(config)
    final_df, pareto_df = evolution.evolve()
    
    # Log to wandb
    wandb.log({"final": wandb.Table(dataframe=final_df)})
    if config.minimize_ysi and not pareto_df.empty:
        wandb.log({"pareto": wandb.Table(dataframe=pareto_df)})
    wandb.finish()
    
    # Display and save results
    display_results(final_df, pareto_df, config.minimize_ysi)
    save_results(final_df, pareto_df, config.minimize_ysi)


if __name__ == "__main__":
    main()