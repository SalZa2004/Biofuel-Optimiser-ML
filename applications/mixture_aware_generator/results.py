
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from core.config import EvolutionConfig

def _rename_cn(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"cn": "mixture_cn"}) if "cn" in df.columns else df


def save_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, unfiltered_df: pd.DataFrame, minimize_ysi: bool):
    """Save results to CSV files."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    final_df = _rename_cn(final_df)
    pareto_df = _rename_cn(pareto_df)
    unfiltered_df = _rename_cn(unfiltered_df)

    final_df.to_csv(results_dir / "final_population.csv", index=False)
    unfiltered_df.to_csv(results_dir / "final_population_unfiltered.csv", index=False)
    if minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)

    if minimize_ysi and not pareto_df.empty:
        plot_pareto_front(pareto_df, final_df, unfiltered_df)

    print("\n✓ Results saved to results/")


def plot_pareto_front(
    pareto_df: pd.DataFrame,
    final_df: pd.DataFrame,
    unfiltered_df: pd.DataFrame,
):
    """Save a Pareto front scatter plot to results/mixture_pareto_front.png."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    maximize_cn = "cn_error" not in pareto_df.columns or pareto_df["cn_error"].isna().all()
    cn_col = "mixture_cn" if maximize_cn else "cn_error"
    ysi_col = "mixture_ysi" if "mixture_ysi" in pareto_df.columns else "ysi"

    fig, ax = plt.subplots(figsize=(8, 6))


    # Filtered (passed property constraints) in blue
    filt = final_df.dropna(subset=[cn_col, ysi_col])
    ax.scatter(filt[cn_col], filt[ysi_col], c="#6baed6", s=25, alpha=0.6, label="Filtered candidates", zorder=2)

    # Pareto front in red, connected by a step line
    pf = pareto_df.dropna(subset=[cn_col, ysi_col]).sort_values(cn_col, ascending=True)
    ax.scatter(pf[cn_col], pf[ysi_col], c="#d73027", s=60, zorder=4, label="Pareto front")
    

    ax.set_xlabel("Mixture CN" if maximize_cn else "CN Error", fontsize=12)
    ax.set_ylabel("Mixture YSI", fontsize=12)
    ax.set_title("Mixture Pareto Front: DCN vs YSI", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    out_path = results_dir / "mixture_pareto_front.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Pareto front plot saved to {out_path}")


def display_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, unfiltered_df: pd.DataFrame, config: EvolutionConfig):
    """Display results to console."""
    final_df = _rename_cn(final_df)
    pareto_df = _rename_cn(pareto_df)
    unfiltered_df = _rename_cn(unfiltered_df)

    cols = ["rank", "smiles", "mixture_cn", "mixture_ysi", "mixture_bp", "mixture_density", "cn_error", "bp", "density", "lhv", "dynamic_viscosity"]

    if config.maximize_cn:
        cols = [c for c in cols if c != "cn_error"]

    available_cols = [c for c in cols if c in final_df.columns]

    print("\n" + "="*70)
    print("=== BEST CANDIDATES (with property constraints) ===")
    print("="*70)
    print(final_df.head(10)[available_cols].to_string(index=False))

    print("\n" + "="*70)
    print("=== BEST CANDIDATES (without property constraints) ===")
    print("="*70)
    unfiltered_cols = [c for c in cols if c in unfiltered_df.columns]
    print(unfiltered_df.head(10)[unfiltered_cols].to_string(index=False))

    if config.minimize_ysi and not pareto_df.empty:
        print("\n" + "="*70)
        print("=== PARETO FRONT (Non-dominated solutions) ===")
        print("="*70)
        available_pareto_cols = [c for c in cols if c in pareto_df.columns]
        print(pareto_df[available_pareto_cols].head(20).to_string(index=False))
