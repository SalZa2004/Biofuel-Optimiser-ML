
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from core.config import EvolutionConfig

def plot_pareto_front(
    population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    config: EvolutionConfig,
    save_dir: Path,
) -> None:
    """Scatter all surviving molecules and overlay the Pareto front."""
    if not config.minimize_ysi:
        return
    if "cn_error" not in population_df.columns or "ysi" not in population_df.columns:
        print("⚠ Cannot plot Pareto front: missing cn_error or ysi columns.")
        return

    pop = population_df.dropna(subset=["cn_error", "ysi"])
    pareto = pareto_df.dropna(subset=["cn_error", "ysi"]) if not pareto_df.empty else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 6))

    # All surviving molecules
    ax.scatter(
        pop["cn_error"],
        pop["ysi"],
        c="#a8c8e8",
        alpha=0.55,
        s=28,
        linewidths=0,
        label=f"Population ({len(pop)})",
        zorder=2,
    )

    # Pareto-front molecules
    if not pareto.empty:
        pareto_sorted = pareto.sort_values("cn_error")
        ax.scatter(
            pareto_sorted["cn_error"],
            pareto_sorted["ysi"],
            c="#e84040",
            s=60,
            linewidths=0.6,
            edgecolors="white",
            label=f"Pareto front ({len(pareto_sorted)})",
            zorder=4,
        )

    ax.set_xlabel("CN Error (|predicted − target|)", fontsize=12)
    ax.set_ylabel("YSI", fontsize=12)
    ax.set_title("Pareto Front: CN Error vs YSI", fontsize=13, fontweight="bold")
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_path = save_dir / "pareto_front.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pareto front plot saved → {out_path}")


def save_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, unfiltered_df: pd.DataFrame, config: EvolutionConfig):
    """Save results to CSV files and generate plots."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    final_df.to_csv(results_dir / "final_population.csv", index=False)
    unfiltered_df.to_csv(results_dir / "final_population_unfiltered.csv", index=False)
    if config.minimize_ysi and not pareto_df.empty:
        pareto_df.to_csv(results_dir / "pareto_front.csv", index=False)
        plot_pareto_front(unfiltered_df, pareto_df, config, results_dir)

    print("\n✓ Results saved to results/")


def display_results(final_df: pd.DataFrame, pareto_df: pd.DataFrame, unfiltered_df: pd.DataFrame, config: EvolutionConfig):
    """Display results to console."""
    cols = ["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity"]

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
