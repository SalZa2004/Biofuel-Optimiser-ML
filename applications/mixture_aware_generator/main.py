import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from .cli import get_user_config
from .results import display_results,save_results
from core.evolution.mixture_evolution import MixtureAwareMolecularEvolution
from core.evolution.evolution import MolecularEvolution
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
def run(config):
    """Run the mixture-aware (or standard) evolution and return raw DataFrames."""
    if config.mixture_mode:
        evolution = MixtureAwareMolecularEvolution(config)
    else:
        evolution = MolecularEvolution(config)
    return evolution.evolve()


def main():
    config = get_user_config()  # Now supports mixture mode

    final_df, pareto_df, unfiltered_df = run(config)

    display_results(final_df, pareto_df, unfiltered_df, config)
    save_results(final_df, pareto_df, unfiltered_df, config.minimize_ysi)

if __name__ == "__main__":
    main()
