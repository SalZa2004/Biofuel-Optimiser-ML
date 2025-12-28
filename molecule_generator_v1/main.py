import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from interface.cli import get_user_config
from interface.results import display_results, save_results
from evolution.evolution import MolecularEvolution
from shared_features import FeatureSelector
def main():
    """Main execution function."""
    config = get_user_config()
    
    evolution = MolecularEvolution(config)
    final_df, pareto_df = evolution.evolve()

    # Display and save results
    display_results(final_df, pareto_df, config)
    save_results(final_df, pareto_df, config.minimize_ysi)

if __name__ == "__main__":
    main()