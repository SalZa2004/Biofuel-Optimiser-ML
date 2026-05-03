"""
CN–YSI Trade-off Curve

Runs the GA for multiple CN targets with YSI minimisation enabled and
plots the minimum achievable YSI as a function of CN target.

    CN_targets = [40, 45, 50, 55]

For each target:
  - Run GA with minimize_ysi=True, maximize_cn=False
  - Collect molecules whose CN is within --cn-tolerance of the target
  - Record the lowest YSI among those molecules

Usage (from project root):
    python scripts/analysis/cn_ysi_tradeoff.py
    python scripts/analysis/cn_ysi_tradeoff.py --generations 10 --population-size 80
    python scripts/analysis/cn_ysi_tradeoff.py --cn-targets 40 45 50 55 60 --verbose

Output (default: scripts/results/cn_ysi_tradeoff/):
    tradeoff_curve.png  — trade-off line + Pareto scatter overlay
    tradeoff_data.csv   — per-target summary statistics
"""

import sys
import os
import random
import argparse
import contextlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from core.config import EvolutionConfig
from core.evolution.evolution import MolecularEvolution

RESULTS_DIR = ROOT / "scripts" / "results" / "cn_ysi_tradeoff"


# ── Helpers ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# ── GA runner ─────────────────────────────────────────────────────────────────

def run_ga(
    target_cn: float,
    generations: int,
    population_size: int,
    seed: int,
    cn_tol: float,
    verbose: bool,
) -> Tuple[Optional[float], Optional[float], pd.DataFrame]:
    """Run one GA experiment and return (min_ysi, achieved_cn, unfiltered_df)."""
    random.seed(seed)
    np.random.seed(seed)

    config = EvolutionConfig(
        target_cn=target_cn,
        maximize_cn=False,
        minimize_ysi=True,
        generations=generations,
        population_size=population_size,
    )

    ctx = contextlib.nullcontext() if verbose else _suppress_output()
    with ctx:
        evo = MolecularEvolution(config)
        _, _, unfiltered_df = evo.evolve()

    if unfiltered_df.empty or "ysi" not in unfiltered_df.columns:
        return None, None, pd.DataFrame()

    # Molecules close to the CN target
    candidates = unfiltered_df[unfiltered_df["cn_error"] <= cn_tol].dropna(subset=["ysi"])

    # Fall back to the closest 10 molecules if nothing is within tolerance
    if candidates.empty:
        candidates = unfiltered_df.nsmallest(10, "cn_error").dropna(subset=["ysi"])

    if candidates.empty:
        return None, None, unfiltered_df

    best_row = candidates.loc[candidates["ysi"].idxmin()]
    return float(best_row["ysi"]), float(best_row["cn"]), unfiltered_df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_tradeoff(
    records: List[Dict],
    all_pops: Dict[float, pd.DataFrame],
    cn_tol: float,
    out_dir: Path,
):
    """Two-panel figure: trade-off curve (top) + Pareto scatter by target (bottom)."""
    summary = pd.DataFrame(records).dropna(subset=["min_ysi"])
    if summary.empty:
        print("  No valid data — skipping plot.")
        return

    cmap = plt.get_cmap("tab10")
    targets = summary["cn_target"].tolist()
    colors = {t: cmap(i % 10) for i, t in enumerate(targets)}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 10),
                                          gridspec_kw={"height_ratios": [2, 3]})

    # ── Top panel: trade-off curve ─────────────────────────────────────────────
    ax_top.plot(summary["cn_target"], summary["min_ysi"],
                "o-", color="steelblue", linewidth=2, markersize=9, zorder=4,
                label="Min achievable YSI")

    for _, row in summary.iterrows():
        ax_top.annotate(
            f"YSI={row['min_ysi']:.1f}\n(CN={row['achieved_cn']:.1f})",
            xy=(row["cn_target"], row["min_ysi"]),
            xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.85),
        )

    # Shade directions
    y_lo = summary["min_ysi"].min() * 0.85
    y_hi = summary["min_ysi"].max() * 1.25
    ax_top.fill_between(summary["cn_target"], summary["min_ysi"], y_hi,
                         alpha=0.07, color="tomato")
    ax_top.fill_between(summary["cn_target"], y_lo, summary["min_ysi"],
                         alpha=0.07, color="seagreen")
    ax_top.set_ylim(y_lo, y_hi)

    ax_top.set_xlabel("CN Target", fontsize=12)
    ax_top.set_ylabel("Minimum Achievable YSI", fontsize=12)
    ax_top.set_title("CN–YSI Trade-off Curve", fontsize=13, fontweight="bold")
    ax_top.set_xticks(targets)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(fontsize=9)

    # ── Bottom panel: population scatter by CN target ──────────────────────────
    for target in targets:
        pop = all_pops.get(target, pd.DataFrame())
        if pop.empty or "ysi" not in pop.columns:
            continue

        # Only molecules with CN near target (loose filter for visual context)
        sub = pop[pop["cn_error"] <= cn_tol * 2].dropna(subset=["ysi", "cn"])
        if sub.empty:
            continue

        ax_bot.scatter(
            sub["cn"], sub["ysi"],
            color=colors[target], alpha=0.35, s=20, zorder=2,
        )

        # Highlight the minimum-YSI point
        best = sub.loc[sub["ysi"].idxmin()]
        ax_bot.scatter(
            best["cn"], best["ysi"],
            color=colors[target], s=90, edgecolors="black",
            linewidths=0.8, zorder=5,
            label=f"Target CN={int(target)} (best: YSI={best['ysi']:.1f}, CN={best['cn']:.1f})",
        )

    # Draw vertical dashed lines at each CN target
    for target in targets:
        ax_bot.axvline(x=target, color=colors[target], linestyle="--",
                       linewidth=0.9, alpha=0.5)

    ax_bot.set_xlabel("Achieved CN", fontsize=12)
    ax_bot.set_ylabel("YSI", fontsize=12)
    ax_bot.set_title("Population Scatter by CN Target\n"
                      f"(molecules within ±{cn_tol * 2:.0f} of target shown)",
                      fontsize=11)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(fontsize=8.5, loc="upper right")

    fig.tight_layout(pad=2.5)
    path = out_dir / "tradeoff_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--cn-targets", nargs="+", type=float, default=[40, 45, 50, 55],
        metavar="N", help="CN targets to sweep (default: 40 45 50 55)",
    )
    p.add_argument(
        "--generations", type=int, default=15,
        help="generations per GA run (default: 10)",
    )
    p.add_argument(
        "--population-size", type=int, default=100,
        help="population size (default: 100)",
    )
    p.add_argument(
        "--cn-tolerance", type=float, default=2.0,
        help="max CN error for a molecule to count as 'on-target' (default: 5)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=False,
                   help="show GA output for each run")
    p.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help=f"output directory (default: {RESULTS_DIR})",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("CN–YSI Trade-off Curve")
    print(f"CN targets     : {args.cn_targets}")
    print(f"Generations    : {args.generations}")
    print(f"Population     : {args.population_size}")
    print(f"CN tolerance   : ±{args.cn_tolerance}")
    print(f"Seed           : {args.seed}")
    print(f"Output dir     : {args.output_dir}")
    print(f"{'=' * 60}\n")

    records: List[Dict] = []
    all_pops: Dict[float, pd.DataFrame] = {}

    total = len(args.cn_targets)
    for idx, cn_target in enumerate(args.cn_targets, 1):
        print(f"[{idx}/{total}] CN target = {cn_target}", end="  ", flush=True)
        try:
            min_ysi, achieved_cn, pop_df = run_ga(
                target_cn=cn_target,
                generations=args.generations,
                population_size=args.population_size,
                seed=args.seed,
                cn_tol=args.cn_tolerance,
                verbose=args.verbose,
            )
            if min_ysi is not None:
                print(f"✓  min_ysi={min_ysi:.2f}  achieved_cn={achieved_cn:.2f}")
            else:
                print("✗  no valid molecules found")
        except Exception as exc:
            print(f"✗  {exc}")
            min_ysi, achieved_cn, pop_df = None, None, pd.DataFrame()

        records.append({"cn_target": cn_target, "min_ysi": min_ysi,
                         "achieved_cn": achieved_cn})
        all_pops[cn_target] = pop_df

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(records)
    width = 50
    print(f"\n{'=' * width}")
    print("TRADE-OFF SUMMARY")
    print(f"{'=' * width}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"{'=' * width}\n")

    csv_path = args.output_dir / "tradeoff_data.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Data saved → {csv_path}")

    print("\nGenerating plots:")
    plot_tradeoff(records, all_pops, args.cn_tolerance, args.output_dir)


if __name__ == "__main__":
    main()
