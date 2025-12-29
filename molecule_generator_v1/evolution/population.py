from typing import List
from config import EvolutionConfig
from .molecule import Molecule
import pandas as pd

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
            
            sort_key = lambda m: (
                -self.config.cn_objective(m.cn),  # higher objective = better
                m.ysi
            )

            
            if len(survivors) > target_size:
                survivors = sorted(survivors, key=sort_key)[:target_size]
            elif len(survivors) < target_size:
                remainder = [m for m in self.molecules if m not in survivors]
                remainder = sorted(remainder, key=sort_key)
                survivors.extend(remainder[:target_size - len(survivors)])
        else:
            # Single objective mode
            survivors = sorted(
                self.molecules,
                key=lambda m: self.config.cn_objective(m.cn),
                reverse=True
                )[:target_size]

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
