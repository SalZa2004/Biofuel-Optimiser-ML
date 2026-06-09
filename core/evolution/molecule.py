from dataclasses import dataclass, asdict, field
from typing import Optional, Dict

@dataclass
class Molecule:
    smiles: str
    cn: float
    cn_error: float
    cn_score: float = 0.0  # keep for backwards compat, will be set by fitness()
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

    def fitness(self, config) -> float:
        """
        Single scalar fitness for weighted parent selection.
        Higher is always better (so we can use as selection weight).
        
        - maximize_cn mode:  raw CN value
        - target_cn mode:    inverted error (so lower error = higher fitness)
        - YSI penalty:       applied as soft penalty, not hard filter
        """
        if config.maximize_cn:
            base = self.cn
        else:
            # Avoid division by zero; small epsilon floor
            base = 1.0 / (self.cn_error + 1e-6)

        if config.minimize_ysi and self.ysi is not None:
            # Soft penalty: YSI above 50 is penalised, below is neutral/rewarded
            ysi_penalty = max(0.0, (self.ysi - 50) / 50)  # 0 at ysi=50, 1 at ysi=100
            base *= (1.0 - 0.3 * ysi_penalty)             # up to 30% penalty

        return max(base, 1e-9)  # keep strictly positive for sampling weights

    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}