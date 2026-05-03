"""Display and persistence for screening tool results."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .cli import ScreeningConfig

# Column ordering for each mode
_PURE_COLS = ["rank", "smiles", "cn", "cn_error", "ysi", "bp", "density", "lhv", "dynamic_viscosity", "tanimoto"]
_MIX_COLS = ["rank", "name", "mixture_cn", "mixture_cn_error", "mixture_ysi", "n_components"]


def _avail(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def display_results(
    filtered_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    all_df: pd.DataFrame,
    config: ScreeningConfig,
) -> None:
    if config.mode == "pure_component":
        cols = _PURE_COLS

        print("\n" + "=" * 70)
        print(f"=== TOP CANDIDATES — passed filters (target CN = {config.target_cn}) ===")
        print("=" * 70)
        if filtered_df.empty:
            print("  No molecules passed the property filters.")
        else:
            print(filtered_df.head(20)[_avail(filtered_df, cols)].to_string(index=False))

        print("\n" + "=" * 70)
        print("=== ALL CANDIDATES — no filters applied ===")
        print("=" * 70)
        print(all_df.head(20)[_avail(all_df, cols)].to_string(index=False))

        print("\n" + "=" * 70)
        print("=== PARETO FRONT — best CN accuracy + lowest YSI ===")
        print("=" * 70)
        if pareto_df.empty:
            print("  No Pareto front candidates (need valid YSI for both objectives).")
        else:
            print(pareto_df[_avail(pareto_df, cols)].to_string(index=False))

    else:  # mixture
        cols = _MIX_COLS

        print("\n" + "=" * 70)
        print(f"=== TOP MIXTURE CANDIDATES (target CN = {config.target_cn}) ===")
        print("=" * 70)
        if filtered_df.empty:
            print("  No valid mixture predictions.")
        else:
            print(filtered_df.head(20)[_avail(filtered_df, cols)].to_string(index=False))

        print("\n" + "=" * 70)
        print("=== PARETO FRONT — best mixture CN accuracy + lowest mixture YSI ===")
        print("=" * 70)
        if pareto_df.empty:
            print("  No Pareto front candidates (need valid mixture YSI for both objectives).")
        else:
            print(pareto_df[_avail(pareto_df, cols)].to_string(index=False))


def _plot_pareto_front(
    pareto_df: pd.DataFrame,
    all_df: pd.DataFrame,
    config: ScreeningConfig,
    out_path: Path,
) -> None:
    if config.mode == "pure_component":
        cn_col, ysi_col = "cn_error", "ysi"
        xlabel = f"CN Error  (|CN − {config.target_cn}|)"
        ylabel = "YSI"
        title = f"Pure Component Screening — Pareto Front  (target CN = {config.target_cn})"
    else:
        cn_col, ysi_col = "mixture_cn_error", "mixture_ysi"
        xlabel = f"Mixture CN Error  (|DCN − {config.target_cn}|)"
        ylabel = "Mixture YSI"
        title = f"Mixture Screening — Pareto Front  (target CN = {config.target_cn})"

    if cn_col not in pareto_df.columns or ysi_col not in pareto_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    valid_all = all_df.dropna(subset=[cn_col, ysi_col])
    if not valid_all.empty:
        ax.scatter(
            valid_all[cn_col], valid_all[ysi_col],
            c="#6baed6", s=25, alpha=0.5, label="All candidates", zorder=2,
        )

    pf = pareto_df.dropna(subset=[cn_col, ysi_col]).sort_values(cn_col)
    if not pf.empty:
        ax.scatter(
            pf[cn_col], pf[ysi_col],
            c="#d73027", s=70, zorder=4, label="Pareto front",
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto front plot saved to {out_path}")


def save_results(
    filtered_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    all_df: pd.DataFrame,
    config: ScreeningConfig,
) -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if config.mode == "pure_component":
        prefix = "screening_pure"
    else:
        prefix = "screening_mixture"

    if not all_df.empty:
        all_df.to_csv(results_dir / f"{prefix}_all.csv", index=False)
    if not filtered_df.empty and config.mode == "pure_component":
        filtered_df.to_csv(results_dir / f"{prefix}_filtered.csv", index=False)
    if not pareto_df.empty:
        pareto_df.to_csv(results_dir / f"{prefix}_pareto.csv", index=False)
        plot_source = filtered_df if (config.mode == "pure_component" and not filtered_df.empty) else all_df
        _plot_pareto_front(
            pareto_df,
            plot_source,
            config,
            results_dir / f"{prefix}_pareto.png",
        )

    print("\nResults saved to results/")
