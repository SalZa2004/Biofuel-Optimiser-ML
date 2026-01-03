
from pathlib import Path
import pandas as pd
from core.config import EvolutionConfig

def save_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, minimize_ysi: bool):
    """Save results to CSV files."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    final_df.to_csv(results_dir / "final_population.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)
    
    print("\n✓ Results saved to results/")


def display_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, config: EvolutionConfig):
    """Display results to console."""
    cols = ["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity"]
    
    # Remove cn_error column if maximizing (not relevant)
    if config.maximize_cn:
        cols = [c for c in cols if c != "cn_error"]
    
    available_cols = [c for c in cols if c in final_df.columns]
    
    print("\n" + "="*70)
    print("=== BEST CANDIDATES ===")
    print("="*70)
    print(final_df.head(10)[available_cols].to_string(index=False))
    
    if config.minimize_ysi and not pareto_df.empty:
        print("\n" + "="*70)
        print("=== PARETO FRONT (Non-dominated solutions) ===")
        print("="*70)
        available_pareto_cols = [c for c in cols if c in pareto_df.columns]
        print(pareto_df[available_pareto_cols].head(20).to_string(index=False))
