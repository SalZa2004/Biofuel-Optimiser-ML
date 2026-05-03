from dataclasses import dataclass, field
from typing import Optional, List, Dict
@dataclass
class EvolutionConfig:
    target_cn: float = 50.0
    maximize_cn: bool = False
    minimize_ysi: bool = True
    generations: int = 15
    population_size: int = 100
    mutations_per_parent: int = 3
    survivor_fraction: float = 0.5
    batch_size: int = 100
    max_offspring_attempts: int = 10
    mixture_mode: bool = False
    max_size = 2
    min_freq = 3
    mixture_config: Optional["MixtureConfig"] = None

    # Filters
    filters: dict = field(default_factory=lambda: {
        "bp": (60.0, 250.0),
        "density": (720.0, None),
        "lhv": (30.0, None),
        "dynamic_viscosity": (2.0, None),

    })

    def cn_objective(self, cn: float) -> float:
        return cn if self.maximize_cn else -abs(cn - self.target_cn)
@dataclass
class MixtureConfig:
    """Configuration for mixture optimization."""
    
    # Blend composition
    additive_fraction: float = 0.15  # 15% of generated molecule
    base_fuel_fraction: float = 0.85  # 85% base diesel
    
    # Base fuel definition
    base_fuel_type: str = "fossil_diesel"  # or "biodiesel", "custom"
    base_fuel_smiles: Optional[List[str]] = None  # If custom base fuel
    base_fuel_mole_fractions: Optional[List[float]] = None  # If custom base fuel
    
    # Mixture property targets
    target_mixture_dcn: float = 51.0  # Target DCN for the blend
    
    # Optimization mode
    optimize_blend_ratio: bool = False  # If True, also optimize the blend percentage
    min_additive_fraction: float = 0.05  # Min 5%
    max_additive_fraction: float = 0.30  # Max 30%
    
    # Multi-component blend
    multi_component_blend: bool = False  # If True, generate multiple additives
    max_blend_components: int = 3  # Max number of different additives
    
    def validate(self):
        """Validate configuration."""
        if self.additive_fraction + self.base_fuel_fraction != 1.0:
            raise ValueError(f"Fractions must sum to 1.0, got {self.additive_fraction + self.base_fuel_fraction}")
        
        if self.optimize_blend_ratio:
            if not (0 < self.min_additive_fraction < self.max_additive_fraction < 1.0):
                raise ValueError("Invalid blend ratio bounds")

