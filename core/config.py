from dataclasses import dataclass, field

@dataclass
class EvolutionConfig:
    target_cn: float = 50.0
    maximize_cn: bool = False
    minimize_ysi: bool = True

    generations: int = 6
    population_size: int = 100
    mutations_per_parent: int = 5
    survivor_fraction: float = 0.5

    batch_size: int = 100
    max_offspring_attempts: int = 10

    # Filters
    filters: dict = field(default_factory=lambda: {
        "bp": (60.0, 250.0),
        "density": (720.0, None),
        "lhv": (30.0, None),
        "dynamic_viscosity": (0.0, 2.0),
    })

    def cn_objective(self, cn: float) -> float:
        return cn if self.maximize_cn else -abs(cn - self.target_cn)
