import os
import sys

# === CRITICAL: Set environment variables BEFORE any imports ===
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Callable
import joblib
import numpy as np
import pandas as pd
import random
from rdkit import Chem
from crem.crem import mutate_mol
from sklearn.base import BaseEstimator, RegressorMixin
from huggingface_hub import snapshot_download

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
        
        model_path = model_dir / "model.joblib"
        selector_path = model_dir / "selector.joblib"
        
        # Load artifacts
        self.model = joblib.load(model_path)
        self.selector = FeatureSelector.load(selector_path)
        self.property_name = property_name
        
        print(f"✓ {property_name} Predictor ready!")
    
    def predict_from_features(self, X_full):
        """Predict from pre-computed features (OPTIMIZED)."""
        if X_full is None or len(X_full) == 0:
            return []
        
        try:
            X_selected = self.selector.transform(X_full)
            predictions = self.model.predict(X_selected)
            return predictions.tolist()
        except Exception as e:
            print(f"⚠ Warning: {self.property_name} prediction failed: {e}")
            return [None] * len(X_full)


# Predictor paths relative to project root
HF_MODELS = {
    "cn": "SalZa2004/Cetane_Number_Predictor",
    "ysi": "SalZa2004/YSI_Predictor",
    "bp": "SalZa2004/Boiling_Point_Predictor",
    "density": "SalZa2004/Density_Predictor",
    "lhv": "SalZa2004/LHV_Predictor",
    "dynamic_viscosity": "SalZa2004/Dynamic_Viscosity_Predictor",
}

print("Downloading models from Hugging Face...")
PREDICTOR_PATHS = {
    key: Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model"
        )
    )
    for key, repo in HF_MODELS.items()
}
print("✓ All models downloaded!\n")

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary algorithm."""
    target_cn: float = field(default=50.0)
    maximize_cn: bool = field(default=False)  # ADD THIS LINE
    minimize_ysi: bool = field(default=True)
    generations: int = field(default=6)
    population_size: int = field(default=100)
    mutations_per_parent: int = field(default=5)
    survivor_fraction: float = field(default=0.5)
    min_bp: float = field(default=60.0)
    max_bp: float = field(default=250.0)
    min_dynamic_viscosity: float = field(default=0.0)
    max_dynamic_viscosity: float = field(default=2.0)
    min_density: float = field(default=720.0)
    min_lhv: float = field(default=30.0)
    use_bp_filter: bool = field(default=True)
    use_density_filter: bool = field(default=True)
    use_lhv_filter: bool = field(default=True)
    use_dynamic_viscosity_filter: bool = field(default=True)
    batch_size: int = field(default=100)
    max_offspring_attempts: int = field(default=10)
@dataclass
class Molecule:
    """Represents a molecule with its properties."""
    smiles: str
    cn: float
    cn_error: float
    cn_score: float = 0.0  # New: for maximize mode (higher is better)
    bp: Optional[float] = None
    ysi: Optional[float] = None
    density: Optional[float] = None
    lhv: Optional[float] = None
    dynamic_viscosity: Optional[float] = None
    
    def dominates(self, other: 'Molecule', maximize_cn: bool = False) -> bool:
        """Check if this molecule Pareto-dominates another."""
        if maximize_cn:
            # For maximize mode: higher CN is better
            better_cn = self.cn >= other.cn
            strictly_better_cn = self.cn > other.cn
        else:
            # For target mode: lower error is better
            better_cn = self.cn_error <= other.cn_error
            strictly_better_cn = self.cn_error < other.cn_error
        
        better_ysi = self.ysi <= other.ysi if self.ysi is not None else True
        strictly_better_ysi = self.ysi < other.ysi if self.ysi is not None else False
        
        strictly_better = strictly_better_cn or strictly_better_ysi
        return better_cn and better_ysi and strictly_better
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class PropertyPredictor:
    """Handles batch prediction for all molecular properties (OPTIMIZED)."""
    
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
            'dynamic_viscosity': lambda v: self.config.min_dynamic_viscosity < v <= self.config.max_dynamic_viscosity
        }
    
    def _safe_predict(self, predictions: List) -> List[Optional[float]]:
        """Safely convert predictions, handling None/NaN/inf values."""
        return [
            float(pred) if pred is not None and np.isfinite(pred) else None
            for pred in predictions
        ]
    
    def predict_all_properties(self, smiles_list: List[str]) -> Dict[str, List[Optional[float]]]:
        """
        Predict all properties for a batch of SMILES (OPTIMIZED).
        Featurizes ONCE and reuses features for all predictors.
        """
        if not smiles_list:
            return {prop: [] for prop in self.predictors.keys()}
        
        # OPTIMIZATION: Featurize only once per batch
        X_full = featurize_df(smiles_list, return_df=False)
        
        if X_full is None:
            return {prop: [None] * len(smiles_list) for prop in self.predictors.keys()}
        
        # Predict all properties using the same features
        results = {}
        for prop_name, predictor in self.predictors.items():
            predictions = predictor.predict_from_features(X_full)
            results[prop_name] = self._safe_predict(predictions)
        
        return results
    
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
            if not any(other.dominates(mol, self.config.maximize_cn) 
                      for other in self.molecules if other is not mol)
        ]
    
    def get_survivors(self) -> List[Molecule]:
        """Select survivors for the next generation."""
        target_size = int(self.config.population_size * self.config.survivor_fraction)
        
        if self.config.minimize_ysi:
            survivors = self.pareto_front()
            
            # Sort key for combined objectives
            if self.config.maximize_cn:
                # Maximize CN: higher CN is better, lower YSI is better
                sort_key = lambda m: (-m.cn, m.ysi)
            else:
                # Target CN: lower error is better, lower YSI is better
                sort_key = lambda m: (m.cn_error, m.ysi)
            
            if len(survivors) > target_size:
                survivors = sorted(survivors, key=sort_key)[:target_size]
            elif len(survivors) < target_size:
                remainder = [m for m in self.molecules if m not in survivors]
                remainder = sorted(remainder, key=sort_key)
                survivors.extend(remainder[:target_size - len(survivors)])
        else:
            # Single objective mode
            if self.config.maximize_cn:
                survivors = sorted(self.molecules, key=lambda m: -m.cn)[:target_size]
            else:
                survivors = sorted(self.molecules, key=lambda m: m.cn_error)[:target_size]
        
        return survivors
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert population to DataFrame."""
        df = pd.DataFrame([m.to_dict() for m in self.molecules])
        
        if self.config.maximize_cn:
            if self.config.minimize_ysi:
                sort_cols = ["cn", "ysi"]
                ascending = [False, True]  # Descending CN, ascending YSI
            else:
                sort_cols = ["cn"]
                ascending = False
        else:
            if self.config.minimize_ysi:
                sort_cols = ["cn_error", "ysi"]
                ascending = True
            else:
                sort_cols = ["cn_error"]
                ascending = True
        
        df = df.sort_values(sort_cols, ascending=ascending)
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
        """Create Molecule objects from SMILES with predictions (OPTIMIZED)."""
        if not smiles_list:
            return []
        
        # OPTIMIZATION: Single featurization + all predictions
        predictions = self.predictor.predict_all_properties(smiles_list)
        
        molecules = []
        for i, smiles in enumerate(smiles_list):
            # Extract predictions for this molecule
            props = {k: v[i] for k, v in predictions.items()}
            
            # Validate required properties
            if props.get('cn') is None:
                continue
            if self.config.minimize_ysi and props.get('ysi') is None:
                continue
            
            # Validate filtered properties
            if not all(self.predictor.is_valid(k, props.get(k)) for k in ['bp', 'density', 'lhv', 'dynamic_viscosity']):
                continue
            
            molecules.append(Molecule(
                smiles=smiles,
                cn=props['cn'],
                cn_error=abs(props['cn'] - self.config.target_cn),
                cn_score=props['cn'],  # For maximize mode
                bp=props.get('bp'),
                ysi=props.get('ysi'),
                density=props.get('density'),
                lhv=props.get('lhv'),
                dynamic_viscosity=props.get('dynamic_viscosity')
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
        
        if self.config.maximize_cn:
            best_cn = max(mols, key=lambda m: m.cn)
            avg_cn = np.mean([m.cn for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best CN: {best_cn.cn:.3f} | "
                        f"Avg CN: {avg_cn:.3f}")
        else:
            best_cn = min(mols, key=lambda m: m.cn_error)
            avg_cn_err = np.mean([m.cn_error for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best CN err: {best_cn.cn_error:.3f} | "
                        f"Avg CN err: {avg_cn_err:.3f}")
        
        if self.config.minimize_ysi:
            front = self.population.pareto_front()
            best_ysi = min(mols, key=lambda m: m.ysi)
            avg_ysi = np.mean([m.ysi for m in mols])
            
            print_msg += (f" | Best YSI: {best_ysi.ysi:.3f} | "
                         f"Avg YSI: {avg_ysi:.3f} | "
                         f"Pareto: {len(front)}")
        
        print(print_msg)
    
    def _generate_offspring(self, survivors: List[Molecule]) -> List[Molecule]:
        """Generate offspring from survivors (OPTIMIZED batching)."""
        target_count = self.config.population_size - len(survivors)
        max_attempts = target_count * self.config.max_offspring_attempts
        
        all_children = []
        new_molecules = []
        
        print(f"  → Generating offspring (target: {target_count})...")
        
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
            
            # Process in larger batches (single featurization per batch)
            if len(all_children) >= self.config.batch_size:
                print(f"  → Evaluating batch of {len(all_children)} (featurizing once)...")
                new_molecules.extend(self._create_molecules(all_children))
                all_children = []
        
        # Process remaining children
        if all_children:
            print(f"  → Evaluating final batch of {len(all_children)}...")
            new_molecules.extend(self._create_molecules(all_children))
        
        print(f"  ✓ Generated {len(new_molecules)} valid offspring")
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

        # Apply different filtering based on mode
        if self.config.maximize_cn:
            if self.config.minimize_ysi and "ysi" in final_df.columns:
                # Maximize CN + minimize YSI: keep high CN, low YSI
                final_df = final_df[
                    (final_df["cn"] > 50) &
                    (final_df["ysi"] < 50)
                ].sort_values(["cn", "ysi"], ascending=[False, True])
            else:
                # Maximize CN only: just keep high CN
                final_df = final_df[final_df["cn"] > 50].sort_values("cn", ascending=False)
        else:
            if self.config.minimize_ysi and "ysi" in final_df.columns:
                # Target CN + minimize YSI: keep low error, low YSI
                final_df = final_df[
                    (final_df["cn_error"] < 5) &
                    (final_df["ysi"] < 50)
                ].sort_values(["cn_error", "ysi"], ascending=True)
            else:
                # Target CN only: just keep low error
                final_df = final_df[final_df["cn_error"] < 5].sort_values("cn_error", ascending=True)
        
        # Overwrite rank safely
        final_df["rank"] = range(1, len(final_df) + 1)
        
        if self.config.minimize_ysi:
            pareto_mols = self.population.pareto_front()
            pareto_df = pd.DataFrame([m.to_dict() for m in pareto_mols])
            
            if not pareto_df.empty:
                if self.config.maximize_cn:
                    pareto_df = pareto_df[
                        (pareto_df['cn'] > 50) & (pareto_df['ysi'] < 50)
                    ].sort_values(["cn", "ysi"], ascending=[False, True])
                else:
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
        
        print(f"✓ Initial population size: {init_count}\n")
        
        # Evolution
        self._run_evolution_loop()
        
        # Results
        return self._generate_results()


def get_user_config() -> EvolutionConfig:
    """Get configuration from user input."""
    print("\n" + "="*70)
    print("MOLECULAR EVOLUTION WITH GENETIC ALGORITHM")
    print("="*70)
    
    # Choose optimization mode
    print("\nOptimization Mode:")
    print("1. Target a specific CN value (minimize error from target)")
    print("2. Maximize CN (find highest possible CN)")
    mode = input("Select mode (1 or 2): ").strip()
    
    maximize_cn = (mode == "2")
    
    if maximize_cn:
        print("\n✓ Mode: Maximize Cetane Number")
        target = 100.0  # Dummy target, not used in maximize mode
    else:
        print("\n✓ Mode: Target Cetane Number")
        while True:
            target = float(input("Enter target CN: ") or "50")
            if target > 40:
                break
            print("⚠️  Target CN is too low, optimization may be challenging.")
            print("Consider using a higher target CN for better results.\n")
    
    # Ask about YSI
    minimize_ysi = input("\nMinimize YSI (y/n): ").strip().lower() in ['y', 'yes']
    
    # Print configuration summary
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY:")
    print(f"  • Mode: {'Maximize CN' if maximize_cn else f'Target CN = {target}'}")
    print(f"  • Minimize YSI: {'Yes' if minimize_ysi else 'No'}")
    print(f"  • Optimization: {'Multi-objective (CN + YSI)' if minimize_ysi else 'Single-objective (CN only)'}")
    print("="*70 + "\n")
    
    return EvolutionConfig(target_cn=target, maximize_cn=maximize_cn, minimize_ysi=minimize_ysi)


def save_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, minimize_ysi: bool):
    """Save results to CSV files."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    final_df.to_csv(results_dir / "final_population.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)
    
    print("\n✓ Results saved to results/")


def display_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, config: EvolutionConfig):
    """Display results to console."""
    cols = ["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity"]
    
    # Remove cn_error column if maximizing (not relevant)
    if config.maximize_cn:
        cols = [c for c in cols if c != "cn_error"]
    
    available_cols = [c for c in cols if c in final_df.columns]
    
    print("\n" + "="*70)
    print("=== BEST CANDIDATES ===")
    print("="*70)
    print(final_df.head(10)[available_cols].to_string(index=False))
    
    if config.minimize_ysi and not pareto_df.empty:
        print("\n" + "="*70)
        print("=== PARETO FRONT (Non-dominated solutions) ===")
        print("="*70)
        available_pareto_cols = [c for c in cols if c in pareto_df.columns]
        print(pareto_df[available_pareto_cols].head(20).to_string(index=False))

def main():
    """Main execution function."""
    config = get_user_config()
    
    evolution = MolecularEvolution(config)
    final_df, pareto_df = evolution.evolve()
    
    # Display and save results
    display_results(final_df, pareto_df, config)
    save_results(final_df, pareto_df, config.minimize_ysi)

if __name__ == "__main__":
    main()