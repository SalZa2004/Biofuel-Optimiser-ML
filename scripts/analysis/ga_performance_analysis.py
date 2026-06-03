#!/usr/bin/env python3
"""
Mixture-aware generator performance analysis.

Runs the mixture-aware evolutionary algorithm with fossil diesel as base fuel
(maximize CN, minimize YSI), tracks per-generation metrics, computes % improvement
from the initial population, and saves plots to scripts/analysis/ga_plots/.
"""

import os
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

from core.config import EvolutionConfig, MixtureConfig
from core.evolution.mixture_evolution import MixtureAwareMolecularEvolution

PLOT_DIR = Path(__file__).resolve().parent / "ga_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
})


# ─────────────────────────────────────────────────────────────────────────────
# Tracked evolution subclass
# ─────────────────────────────────────────────────────────────────────────────

class TrackedMixtureEvolution(MixtureAwareMolecularEvolution):
    """Captures per-generation population stats alongside the normal evolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen_history: List[Dict] = []

    def _snapshot(self, gen_label) -> None:
        mols = self.population.molecules
        if not mols:
            return
        cns = [m.cn for m in mols if m.cn is not None]
        ysi_vals = [m.mixture_ysi for m in mols if m.mixture_ysi is not None]
        pareto_size = (
            len(self.population.pareto_front()) if self.config.minimize_ysi else 0
        )
        self.gen_history.append({
            "gen": gen_label,
            "best_cn": float(max(cns)) if cns else None,
            "mean_cn": float(np.mean(cns)) if cns else None,
            "best_ysi": float(min(ysi_vals)) if ysi_vals else None,
            "mean_ysi": float(np.mean(ysi_vals)) if ysi_vals else None,
            "pareto_size": pareto_size,
            "pop_size": len(mols),
        })

    def _log_generation_stats(self, generation: int) -> None:
        # _log_generation_stats(gen) is called at the START of iteration `gen`,
        # so it captures the population state from the previous iteration.
        # gen=1 → initial population; gen=N → after (N-1) evolution steps.
        self._snapshot(generation)
        super()._log_generation_stats(generation)

    def _generate_results(self):
        # Capture the truly final population (after all N iterations of evolution).
        self._snapshot(self.config.generations + 1)
        return super()._generate_results()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct(initial_val: Optional[float], final_val: Optional[float],
         higher_is_better: bool = True) -> Optional[float]:
    """Return signed % change; positive means improvement."""
    if initial_val is None or final_val is None or initial_val == 0:
        return None
    raw = (final_val - initial_val) / abs(initial_val) * 100
    return raw if higher_is_better else -raw


def plot_evolution_trends(gen_df: pd.DataFrame, out_dir: Path) -> Dict[str, Optional[float]]:
    """Four-panel figure: CN trend | YSI trend | Pareto size | % improvements."""

    # Separate numeric generations (for trend lines) from the "final" entry
    trend_df = gen_df[gen_df["gen"] != gen_df["gen"].max()].copy()
    init = gen_df.iloc[0]
    final_row = gen_df.iloc[-1]

    improvements = {
        "Best CN":   _pct(init.best_cn,  final_row.best_cn,  higher_is_better=True),
        "Mean CN":   _pct(init.mean_cn,  final_row.mean_cn,  higher_is_better=True),
        "Best YSI":  _pct(init.best_ysi, final_row.best_ysi, higher_is_better=False),
        "Mean YSI":  _pct(init.mean_ysi, final_row.mean_ysi, higher_is_better=False),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    gens = trend_df["gen"].values

    # ── Panel 1: CN over generations ──────────────────────────────────────────
    ax1.plot(gens, trend_df["best_cn"], "o-", color="#2563EB",
             label="Best CN", lw=2, markersize=5)
    ax1.plot(gens, trend_df["mean_cn"], "s--", color="#93C5FD",
             label="Mean CN", lw=1.5, markersize=4)
    ax1.fill_between(gens, trend_df["mean_cn"], trend_df["best_cn"],
                     alpha=0.12, color="#2563EB")
    if final_row.best_cn is not None:
        ax1.axhline(final_row.best_cn, color="#1E3A8A", ls=":", lw=1.2,
                    label=f"Final best = {final_row.best_cn:.1f}")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mixture DCN")
    ax1.set_title("Cetane Number Evolution")
    ax1.legend(fontsize=8)
    ax1.set_xticks(gens)

    # ── Panel 2: YSI over generations ────────────────────────────────────────
    ax2.plot(gens, trend_df["best_ysi"], "o-", color="#16A34A",
             label="Best (min) YSI", lw=2, markersize=5)
    ax2.plot(gens, trend_df["mean_ysi"], "s--", color="#86EFAC",
             label="Mean YSI", lw=1.5, markersize=4)
    ax2.fill_between(gens, trend_df["best_ysi"], trend_df["mean_ysi"],
                     alpha=0.12, color="#16A34A")
    if final_row.best_ysi is not None:
        ax2.axhline(final_row.best_ysi, color="#14532D", ls=":", lw=1.2,
                    label=f"Final best = {final_row.best_ysi:.1f}")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mixture YSI")
    ax2.set_title("Yield Sooting Index (YSI) Reduction")
    ax2.legend(fontsize=8)
    ax2.set_xticks(gens)

    # ── Panel 3: Pareto front size ───────────────────────────────────────────
    ax3.bar(gens, trend_df["pareto_size"], color="#9333EA", alpha=0.75, width=0.6)
    ax3.plot(gens, trend_df["pareto_size"], "D-", color="#6B21A8", lw=1.5, markersize=5)
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("# Non-dominated solutions")
    ax3.set_title("Pareto Front Size")
    ax3.set_xticks(gens)

    # ── Panel 4: % improvement bar chart ─────────────────────────────────────
    labels = list(improvements.keys())
    values = list(improvements.values())
    colors = ["#2563EB", "#60A5FA", "#16A34A", "#4ADE80"]

    bar_data = [(l, v, c) for l, v, c in zip(labels, values, colors) if v is not None]
    if bar_data:
        lv, vv, cv = zip(*bar_data)
        bars = ax4.bar(lv, vv, color=cv, edgecolor="white", linewidth=0.8, width=0.5)
        ax4.axhline(0, color="black", lw=0.8)
        for bar, val in zip(bars, vv):
            ypos = bar.get_height() + 0.3 if val >= 0 else bar.get_height() - 1.5
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:+.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )
        ax4.set_ylabel("% improvement  (initial → final)")
        ax4.set_title("Summary: Generation 1 → Final")
        ax4.tick_params(axis="x", labelsize=8)

    n_gens = int(gen_df["gen"].max())
    fig.suptitle(
        f"Mixture-Aware Generator — Fossil Diesel Base Fuel\n"
        f"Objective: Maximize CN  ∥  Minimize YSI  "
        f"({n_gens - 1} evolution generations, pop={int(gen_df['pop_size'].median())})",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "ga_evolution_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return improvements


def plot_pareto_scatter(final_df: pd.DataFrame, pareto_df: pd.DataFrame,
                        unfiltered_df: pd.DataFrame, out_dir: Path) -> None:
    """CN vs YSI scatter of the final population with Pareto front highlighted."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # All molecules (unfiltered)
    src = unfiltered_df if not unfiltered_df.empty else final_df
    cn_col = "cn" if "cn" in src.columns else "mixture_dcn"
    ysi_col = "ysi" if "ysi" in src.columns else "mixture_ysi"

    ax.scatter(
        src[cn_col], src[ysi_col],
        c="#94A3B8", alpha=0.4, s=25,
        label=f"All molecules (n={len(src)})",
    )

    # Filtered candidates
    if not final_df.empty:
        f_cn = "cn" if "cn" in final_df.columns else cn_col
        f_ysi = "ysi" if "ysi" in final_df.columns else ysi_col
        ax.scatter(
            final_df[f_cn][:30], final_df[f_ysi][:30],
            c="#F59E0B", s=50, edgecolor="darkorange", lw=0.7, zorder=4,
            label="Top-30 filtered",
        )

    # Pareto front
    if not pareto_df.empty:
        p_cn = "cn" if "cn" in pareto_df.columns else cn_col
        p_ysi = "ysi" if "ysi" in pareto_df.columns else ysi_col
        ps = pareto_df.sort_values(p_cn)
        ax.scatter(
            ps[p_cn], ps[p_ysi],
            c="#EF4444", s=70, edgecolor="darkred", lw=0.8, zorder=5,
            label=f"Pareto front (n={len(pareto_df)})",
        )
        ax.step(ps[p_cn].values, ps[p_ysi].values, "r--", alpha=0.5, lw=1.2, where="post")

    ax.set_xlabel("Mixture DCN (Cetane Number)")
    ax.set_ylabel("Mixture YSI (Yield Sooting Index)")
    ax.set_title(
        "Final Population: CN vs YSI\n"
        "Fossil Diesel Base Fuel  |  15 % Additive Fraction"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = out_dir / "ga_pareto_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_improvement_waterfall(gen_df: pd.DataFrame, out_dir: Path) -> None:
    """Compact 2-panel line chart emphasising % improvement from generation 1."""
    init = gen_df.iloc[0]
    trend_df = gen_df.copy()
    gens = trend_df["gen"].values

    # Normalise to % of gen-1 value
    pct_cn = (trend_df["best_cn"] - init.best_cn) / abs(init.best_cn) * 100
    pct_ysi = -(trend_df["best_ysi"] - init.best_ysi) / abs(init.best_ysi) * 100  # positive = reduction

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(gens, pct_cn, "o-", color="#2563EB", lw=2, markersize=6)
    ax1.fill_between(gens, 0, pct_cn, alpha=0.12, color="#2563EB")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("% improvement in best CN")
    ax1.set_title("Cetane Number: % Gain vs Initial Population")
    ax1.set_xticks(gens)

    ax2.plot(gens, pct_ysi, "o-", color="#16A34A", lw=2, markersize=6)
    ax2.fill_between(gens, 0, pct_ysi, alpha=0.12, color="#16A34A")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("% reduction in best YSI")
    ax2.set_title("YSI: % Reduction vs Initial Population")
    ax2.set_xticks(gens)

    fig.suptitle("Per-Generation Improvement Relative to Initial Population",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    path = out_dir / "ga_pct_improvement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(generations: int = 15, population_size: int = 100) -> None:
    print("=" * 65)
    print("Mixture-Aware Generator — Performance Analysis")
    print("Base fuel : Fossil Diesel (4-component surrogate)")
    print("Additive  : 15 % molar fraction")
    print("Objectives: Maximize CN  |  Minimize YSI (NSGA-II Pareto)")
    print(f"Config    : {generations} generations, pop={population_size}")
    print("=" * 65)

    mixture_cfg = MixtureConfig(
        additive_fraction=0.30,
        base_fuel_fraction=0.7,
        base_fuel_type="fossil_diesel",
    )

    config = EvolutionConfig(
        maximize_cn=True,
        minimize_ysi=True,
        generations=generations,
        population_size=population_size,
        mutations_per_parent=3,
        survivor_fraction=0.5,
        mixture_mode=True,
        mixture_config=mixture_cfg,
    )

    print("\nInitialising TrackedMixtureEvolution...")
    evolution = TrackedMixtureEvolution(config, use_ad_filtering=True)

    print("\nRunning evolution loop...\n")
    final_df, pareto_df, unfiltered_df = evolution.evolve()

    # ── per-generation history ────────────────────────────────────────────────
    gen_df = pd.DataFrame(evolution.gen_history)
    gen_df.to_csv(PLOT_DIR / "ga_generation_history.csv", index=False)

    print("\n" + "=" * 65)
    print("PER-GENERATION STATS")
    print("=" * 65)
    with pd.option_context("display.float_format", "{:.2f}".format,
                            "display.max_columns", 10,
                            "display.width", 120):
        print(gen_df.to_string(index=False))

    # ── save result CSVs ──────────────────────────────────────────────────────
    if not final_df.empty:
        final_df.to_csv(PLOT_DIR / "ga_final_population.csv", index=False)
        print(f"\n  Final filtered population : {len(final_df)} molecules")
    if not pareto_df.empty:
        pareto_df.to_csv(PLOT_DIR / "ga_pareto_front.csv", index=False)
        print(f"  Pareto front              : {len(pareto_df)} molecules")

    # ── console improvement summary ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("IMPROVEMENT SUMMARY  (Generation 1 → Final)")
    print("=" * 65)
    if len(gen_df) >= 2:
        init = gen_df.iloc[0]
        fin  = gen_df.iloc[-1]

        for metric, i_val, f_val, higher_better in [
            ("Best CN",  init.best_cn,  fin.best_cn,  True),
            ("Mean CN",  init.mean_cn,  fin.mean_cn,  True),
            ("Best YSI", init.best_ysi, fin.best_ysi, False),
            ("Mean YSI", init.mean_ysi, fin.mean_ysi, False),
        ]:
            if i_val is not None and f_val is not None and i_val != 0:
                raw_pct = (f_val - i_val) / abs(i_val) * 100
                sign = "+" if (raw_pct > 0) == higher_better else ""
                direction = "improvement" if higher_better else "reduction"
                pct_str = f"{abs(raw_pct):.1f}% {direction}"
                arrow = "↑" if (f_val > i_val) == higher_better else "↓"
                print(
                    f"  {metric:10s}: {i_val:6.2f} → {f_val:6.2f}  "
                    f"{arrow} {pct_str}"
                )

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_evolution_trends(gen_df, PLOT_DIR)
    plot_improvement_waterfall(gen_df, PLOT_DIR)
    plot_pareto_scatter(final_df, pareto_df, unfiltered_df, PLOT_DIR)

    print(f"\nAll outputs saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main(generations=15, population_size=100)
