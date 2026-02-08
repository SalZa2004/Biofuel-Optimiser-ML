import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from .cli import get_user_config
from .results import display_results,save_results
from core.evolution.mixture_evolution import MixtureAwareMolecularEvolution
from core.evolution.evolution import MolecularEvolution

def main():
    config = get_user_config()  # Now supports mixture mode
    
    # Choose evolution class
    if config.mixture_mode:
        evolution = MixtureAwareMolecularEvolution(config)
    else:
        evolution = MolecularEvolution(config)  # Original
    
    final_df, pareto_df = evolution.evolve()
    
    display_results(final_df, pareto_df, config)
    save_results(final_df, pareto_df, config.minimize_ysi)

if __name__ == "__main__":
    main()
