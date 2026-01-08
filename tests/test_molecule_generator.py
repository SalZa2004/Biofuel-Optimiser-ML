from applications.molecule_generator.main import run
from core.config import EvolutionConfig

def test_generator_smoke():
    config = EvolutionConfig(
        target_cn=55,
        generations=1,
        population_size=20,
        minimize_ysi=False,
        maximize_cn=False,
        )

    final_df, pareto_df = run(config)

    assert final_df is not None
