from typing import List
from core.config import EvolutionConfig
from .molecule import Molecule
import pandas as pd
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute NSGA-II crowding distance for an (n, m) objective matrix."""
    n, m = F.shape
    dist = np.zeros(n)
    for obj in range(m):
        idx = np.argsort(F[:, obj])
        dist[idx[0]] = dist[idx[-1]] = np.inf
        spread = F[idx[-1], obj] - F[idx[0], obj]
        if spread == 0:
            continue
        for k in range(1, n - 1):
            dist[idx[k]] += (F[idx[k + 1], obj] - F[idx[k - 1], obj]) / spread
    return dist


class Population:
    """Manages the population of molecules."""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.molecules: List[Molecule] = []
        self.seen_smiles: set = set()

    def add_molecule(self, mol: Molecule) -> bool:
        """Add a molecule if not already in the population."""
        if mol.smiles in self.seen_smiles:
            return False
        self.molecules.append(mol)
        self.seen_smiles.add(mol.smiles)
        return True

    def add_molecules(self, molecules: List[Molecule]) -> int:
        """Add multiple molecules, return count added."""
        return sum(self.add_molecule(mol) for mol in molecules)

    def _objective_matrix(self, molecules: List[Molecule]) -> np.ndarray:
        """Build (n, 2) minimisation objective matrix.

        pymoo always minimises, so CN is negated when maximising.
        """
        cn_col = (
            np.array([-m.cn for m in molecules])
            if self.config.maximize_cn
            else np.array([m.cn_error for m in molecules])
        )
        ysi_col = np.array([m.ysi for m in molecules])
        return np.column_stack([cn_col, ysi_col])

    def pareto_front(self) -> List[Molecule]:
        """Return Pareto-front molecules using pymoo non-dominated sorting."""
        if not self.config.minimize_ysi or not self.molecules:
            return []
        F = self._objective_matrix(self.molecules)
        front0 = NonDominatedSorting().do(F, only_non_dominated_front=True)
        return [self.molecules[i] for i in front0]

    def get_survivors(self) -> List[Molecule]:
        """Select survivors via NSGA-II rank + crowding-distance truncation."""
        target_size = int(self.config.population_size * self.config.survivor_fraction)

        if not self.config.minimize_ysi:
            return sorted(
                self.molecules,
                key=lambda m: self.config.cn_objective(m.cn),
                reverse=True,
            )[:target_size]

        F = self._objective_matrix(self.molecules)
        fronts = NonDominatedSorting().do(F)

        survivor_indices: List[int] = []
        for front in fronts:
            slots_left = target_size - len(survivor_indices)
            if len(front) <= slots_left:
                survivor_indices.extend(front.tolist())
            else:
                # Partial front: keep most diverse by crowding distance
                crowding = _crowding_distance(F[front])
                best = front[np.argsort(-crowding)[:slots_left]]
                survivor_indices.extend(best.tolist())
                break

        return [self.molecules[i] for i in survivor_indices]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert population to DataFrame."""
        df = pd.DataFrame([m.to_dict() for m in self.molecules])

        if self.config.maximize_cn:
            sort_cols = ["cn", "ysi"] if self.config.minimize_ysi else ["cn"]
            ascending = [False, True] if self.config.minimize_ysi else False
        else:
            sort_cols = ["cn_error", "ysi"] if self.config.minimize_ysi else ["cn_error"]
            ascending = True

        df = df.sort_values(sort_cols, ascending=ascending)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df
