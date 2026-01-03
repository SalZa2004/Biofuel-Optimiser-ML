from dataclasses import dataclass, asdict
from typing import Optional, Dict

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
            better_cn = self.cn_error <= other.cn_error 
            strictly_better_cn = self.cn_error < other.cn_error 
            better_ysi = self.ysi <= other.ysi if self.ysi is not None else True 
            strictly_better_ysi = self.ysi < other.ysi if self.ysi is not None else False 
            strictly_better = strictly_better_cn or strictly_better_ysi 
            return better_cn and better_ysi and strictly_better
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in asdict(self).items() if v is not None}