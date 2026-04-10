from dataclasses import dataclass, asdict
from typing import Optional, Dict

@dataclass
class Molecule:
    """Represents a molecule with its properties."""
    smiles: str
    cn: float
    cn_error: float
    cn_score: float = 0.0  # For maximize mode (higher is better)
    bp: Optional[float] = None
    ysi: Optional[float] = None
    density: Optional[float] = None
    lhv: Optional[float] = None
    dynamic_viscosity: Optional[float] = None
    
    chemical_valid: bool = True
    chemical_flags: str = 'OK'
    ood_warning: bool = False
    confidence_score: float = 100.0
    

    mixture_dcn: Optional[float] = None
    blend_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame creation."""
        return {k: v for k, v in asdict(self).items() if v is not None}