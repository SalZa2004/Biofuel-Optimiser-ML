from .population import Population
from .molecule import Molecule
from core.predictors.pure_component.property_predictor import PropertyPredictor
from core.config import EvolutionConfig
from crem.crem import mutate_mol
from rdkit import Chem
import pandas as pd
import numpy as np
import random
from typing import List, Tuple
from core.data_prep import df  # Initial dataset for sampling
from pathlib import Path

class MolecularEvolution:
    """Main evolutionary algorithm coordinator."""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    REP_DB_PATH = BASE_DIR / "data" / "fragments" / "diesel_fragments.db"

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
        """Generates offspring from survivors."""
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
            df.groupby(df_bins, observed=False)
            .apply(lambda x: x.sample(20, random_state=42))
            .reset_index(drop=True)["SMILES"]
            .tolist()
        )
        init_count = self.initialize_population(initial_smiles)

        if init_count == 0:
            print("No valid initial molecules")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✓ Initial population size: {init_count}\n")
        
        # Evolution
        self._run_evolution_loop()
        
        # Results
        return self._generate_results()