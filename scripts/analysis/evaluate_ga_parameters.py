"""
Evaluate optimal mutation parameters and generation count for the genetic algorithm.

Sweeps over mutation parameters (mutations_per_parent, CREM max_size, min_freq) and tracks
per-generation diversity and fitness to answer two questions:
  1. Which mutation parameters maximise population diversity while still converging?
  2. How many generations are needed before fitness/diversity plateau?

Diversity metric: mean pairwise Tanimoto distance (Morgan FP, sampled) — higher means the
population is exploring more of chemical space.

Fitness metric: best/mean CN error (target-CN mode) or best/mean CN (maximise-CN mode).

Usage (from project root):
    python scripts/analysis/evaluate_ga_parameters.py
    python scripts/analysis/evaluate_ga_parameters.py --mode sweep --generations 8
    python scripts/analysis/evaluate_ga_parameters.py --mode convergence --max-generations 15
    python scripts/analysis/evaluate_ga_parameters.py \\
        --mutations-per-parent 3 5 10 \\
        --max-size 1 2 3 \\
        --min-freq 1 3 5 \\
        --generations 10 \\
        --population-size 60

Output (default: scripts/results/ga_parameter_sweep/):
    sweep_results.csv          — per-generation metrics for every config run
    diversity_curves.png       — diversity vs generation for each config
    fitness_curves.png         — best/mean fitness vs generation for each config
    convergence_summary.png    — 4-panel convergence overview (default params, long run)
    heatmaps.png               — final diversity & fitness as heatmaps over parameter grid
"""

import sys
import os
import random
import argparse
import contextlib
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Suppress noisy third-party warnings during sweeps
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from core.config import EvolutionConfig
from core.evolution.evolution import MolecularEvolution
from core.evolution.population import Population

RESULTS_DIR = ROOT / "scripts" / "results" / "ga_parameter_sweep"

# ── Defaults (mirrors hardcoded values in MolecularEvolution) ───────────────
DEFAULT_MUTATIONS_PER_PARENT = 5
DEFAULT_CREM_MAX_SIZE = 2
DEFAULT_CREM_MIN_FREQ = 3
DEFAULT_CREM_MAX_REPLACEMENTS = 100


# ── Diversity metric ─────────────────────────────────────────────────────────

def population_tanimoto_diversity(molecules: list, sample_size: int = 60) -> float:
    """Mean pairwise Tanimoto *distance* (1 − similarity) sampled from the population.

    Sampling keeps cost O(sample_size²) regardless of population size.
    Returns 0.0 when fewer than 2 valid fingerprints are available.
    """
    fps = []
    for mol_obj in molecules:
        mol = Chem.MolFromSmiles(mol_obj.smiles)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    if len(fps) < 2:
        return 0.0

    sample = fps if len(fps) <= sample_size else random.sample(fps, sample_size)
    dists = [
        1.0 - DataStructs.TanimotoSimilarity(sample[i], sample[j])
        for i in range(len(sample))
        for j in range(i + 1, len(sample))
    ]
    return float(np.mean(dists))


# ── Instrumented evolution subclass ─────────────────────────────────────────

class TrackingEvolution(MolecularEvolution):
    """MolecularEvolution that records per-generation diversity + fitness snapshots.

    Overrides:
      _mutate_molecule  — uses instance-level CREM params instead of hardcoded values.
      _log_generation_stats — captures metrics before calling parent logging.
      _run_evolution_loop   — appends a final snapshot after the last generation.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        crem_max_size: int = DEFAULT_CREM_MAX_SIZE,
        crem_min_freq: int = DEFAULT_CREM_MIN_FREQ,
        crem_max_replacements: int = DEFAULT_CREM_MAX_REPLACEMENTS,
    ):
        self.crem_max_size = crem_max_size
        self.crem_min_freq = crem_min_freq
        self.crem_max_replacements = crem_max_replacements
        self.generation_snapshots: List[Dict] = []
        super().__init__(config)

    # ── CREM override ────────────────────────────────────────────────────────

    def _mutate_molecule(self, mol):
        from crem.crem import mutate_mol

        try:
            mutants = list(
                mutate_mol(
                    mol,
                    db_name=str(self.REP_DB_PATH),
                    max_size=self.crem_max_size,
                    max_replacements=self.crem_max_replacements,
                    min_freq=self.crem_min_freq,
                    return_mol=False,
                )
            )
            return [m for m in mutants if m and m not in self.population.seen_smiles]
        except Exception:
            return []

    # ── Snapshot logic ───────────────────────────────────────────────────────

    def _capture_snapshot(self, completed_rounds: int):
        """Record metrics for the population after `completed_rounds` mutation rounds."""
        mols = self.population.molecules
        if not mols:
            self.generation_snapshots.append(
                {
                    "completed_rounds": completed_rounds,
                    "pop_size": 0,
                    "diversity": 0.0,
                    "best_fitness": None,
                    "avg_fitness": None,
                    "pareto_size": 0,
                }
            )
            return

        diversity = population_tanimoto_diversity(mols)

        if self.config.maximize_cn:
            fitnesses = [m.cn for m in mols]
            best_fitness = float(max(fitnesses))
        else:
            fitnesses = [m.cn_error for m in mols]
            best_fitness = float(min(fitnesses))

        avg_fitness = float(np.mean(fitnesses))
        pareto_size = (
            len(self.population.pareto_front()) if self.config.minimize_ysi else 0
        )

        self.generation_snapshots.append(
            {
                "completed_rounds": completed_rounds,
                "pop_size": len(mols),
                "diversity": diversity,
                "best_fitness": best_fitness,
                "avg_fitness": avg_fitness,
                "pareto_size": pareto_size,
            }
        )

    # _log_generation_stats is called at the START of round `generation`,
    # meaning `generation - 1` rounds of mutation have completed.
    def _log_generation_stats(self, generation: int):
        self._capture_snapshot(completed_rounds=generation - 1)
        super()._log_generation_stats(generation)

    def _run_evolution_loop(self):
        for gen in range(1, self.config.generations + 1):
            self._log_generation_stats(gen)
            survivors = self.population.get_survivors()
            offspring, _ = self._generate_offspring(survivors)
            new_pop = Population(self.config)
            new_pop.add_molecules(survivors + offspring)
            self.population = new_pop

        # Capture the final population state (after the last mutation round)
        self._capture_snapshot(completed_rounds=self.config.generations)


# ── Experiment runner ────────────────────────────────────────────────────────

@contextlib.contextmanager
def _suppress_output():
    """Redirect stdout/stderr to /dev/null during a block."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def run_config(
    mutations_per_parent: int,
    crem_max_size: int,
    crem_min_freq: int,
    max_generations: int,
    target_cn: float,
    maximize_cn: bool,
    minimize_ysi: bool,
    population_size: int,
    seed: int,
    verbose: bool = False,
) -> List[Dict]:
    """Run a single GA configuration and return per-generation snapshot records."""
    random.seed(seed)
    np.random.seed(seed)

    config = EvolutionConfig(
        target_cn=target_cn,
        maximize_cn=maximize_cn,
        minimize_ysi=minimize_ysi,
        generations=max_generations,
        population_size=population_size,
        mutations_per_parent=mutations_per_parent,
    )

    ctx = contextlib.nullcontext() if verbose else _suppress_output()
    with ctx:
        evo = TrackingEvolution(
            config,
            crem_max_size=crem_max_size,
            crem_min_freq=crem_min_freq,
        )
        evo.evolve()

    return evo.generation_snapshots


# ── Sweep design ─────────────────────────────────────────────────────────────

def build_one_at_a_time_configs(
    mpp_values: List[int],
    max_size_values: List[int],
    min_freq_values: List[int],
) -> List[Dict]:
    """One-at-a-time design: vary one parameter, hold others at default.

    The default config (all three at their default values) is included once and
    shared across all three parameter sweeps.
    """
    configs = []
    seen = set()

    combos = (
        [(mpp, DEFAULT_CREM_MAX_SIZE, DEFAULT_CREM_MIN_FREQ) for mpp in mpp_values]
        + [(DEFAULT_MUTATIONS_PER_PARENT, ms, DEFAULT_CREM_MIN_FREQ) for ms in max_size_values]
        + [(DEFAULT_MUTATIONS_PER_PARENT, DEFAULT_CREM_MAX_SIZE, mf) for mf in min_freq_values]
    )

    for mpp, ms, mf in combos:
        key = (mpp, ms, mf)
        if key in seen:
            continue
        seen.add(key)

        # Determine which parameter is being varied
        varied = []
        if mpp != DEFAULT_MUTATIONS_PER_PARENT:
            varied.append(f"mpp={mpp}")
        if ms != DEFAULT_CREM_MAX_SIZE:
            varied.append(f"size={ms}")
        if mf != DEFAULT_CREM_MIN_FREQ:
            varied.append(f"freq={mf}")
        label = ", ".join(varied) if varied else "default"

        configs.append(
            {
                "mutations_per_parent": mpp,
                "crem_max_size": ms,
                "crem_min_freq": mf,
                "label": label,
            }
        )

    return configs


def build_full_grid_configs(
    mpp_values: List[int],
    max_size_values: List[int],
    min_freq_values: List[int],
) -> List[Dict]:
    """Full factorial grid over all three parameters."""
    return [
        {
            "mutations_per_parent": mpp,
            "crem_max_size": ms,
            "crem_min_freq": mf,
            "label": f"mpp={mpp}, size={ms}, freq={mf}",
        }
        for mpp, ms, mf in iterproduct(mpp_values, max_size_values, min_freq_values)
    ]


# ── Plots ────────────────────────────────────────────────────────────────────

def _line_color(n: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_diversity_curves(df: pd.DataFrame, out_dir: Path):
    labels = df["label"].unique()
    colors = _line_color(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in zip(labels, colors):
        grp = df[df["label"] == label].sort_values("completed_rounds")
        ax.plot(grp["completed_rounds"], grp["diversity"], marker="o", markersize=4,
                label=label, color=color)

    ax.set_xlabel("Completed Generations")
    ax.set_ylabel("Mean Pairwise Tanimoto Distance")
    ax.set_title("Population Diversity per Generation")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "diversity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_fitness_curves(df: pd.DataFrame, maximize_cn: bool, out_dir: Path):
    labels = df["label"].unique()
    colors = _line_color(len(labels))

    best_label = "Best CN" if maximize_cn else "Best CN Error"
    avg_label = "Mean CN" if maximize_cn else "Mean CN Error"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, color in zip(labels, colors):
        grp = df[df["label"] == label].sort_values("completed_rounds")
        axes[0].plot(grp["completed_rounds"], grp["best_fitness"], marker="o",
                     markersize=4, label=label, color=color)
        axes[1].plot(grp["completed_rounds"], grp["avg_fitness"], marker="o",
                     markersize=4, label=label, color=color)

    for ax, title in zip(axes, [best_label, avg_label]):
        ax.set_xlabel("Completed Generations")
        ax.set_ylabel(title)
        ax.set_title(f"{title} per Generation")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = out_dir / "fitness_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_convergence_summary(df: pd.DataFrame, maximize_cn: bool, out_dir: Path):
    """4-panel plot for the 'default' config run (convergence mode)."""
    default_df = df[df["label"] == "default"].sort_values("completed_rounds")
    if default_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    best_label = "Best CN" if maximize_cn else "Best CN Error"
    avg_label = "Mean CN" if maximize_cn else "Mean CN Error"

    panels = [
        ("diversity", "Mean Tanimoto Distance", "Population Diversity"),
        ("best_fitness", best_label, f"{best_label} over Generations"),
        ("avg_fitness", avg_label, f"{avg_label} over Generations"),
        ("pop_size", "Surviving Molecules", "Population Size"),
    ]

    for ax, (col, ylabel, title) in zip(axes, panels):
        x = default_df["completed_rounds"].values
        y = default_df[col].values
        ax.plot(x, y, marker="o", color="steelblue")

        # Annotate the generation where improvement < 1 % of initial change
        if len(y) > 3:
            deltas = np.abs(np.diff(y.astype(float)))
            total_change = deltas.sum()
            if total_change > 0:
                cumulative = np.cumsum(deltas) / total_change
                plateau_idx = np.searchsorted(cumulative, 0.95)
                if plateau_idx < len(x) - 1:
                    ax.axvline(x=x[plateau_idx + 1], color="tomato",
                               linestyle="--", alpha=0.7,
                               label=f"95% change by gen {x[plateau_idx + 1]}")
                    ax.legend(fontsize=8)

        ax.set_xlabel("Completed Generations")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Analysis — Default Parameters", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "convergence_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_parameter_comparison(df: pd.DataFrame, maximize_cn: bool, out_dir: Path):
    """Bar charts of final diversity and fitness for each swept parameter (one-at-a-time design).

    For each parameter, selects only the runs where all other params are at their defaults
    so every bar in the chart has a real data point — no empty cells.
    """
    final = (
        df.sort_values("completed_rounds")
        .groupby(["mutations_per_parent", "crem_max_size", "crem_min_freq"])
        .last()
        .reset_index()
    )

    best_label = "Best CN" if maximize_cn else "Best CN Error"

    swept_params = [
        (
            "mutations_per_parent",
            "mutations_per_parent",
            {"crem_max_size": DEFAULT_CREM_MAX_SIZE, "crem_min_freq": DEFAULT_CREM_MIN_FREQ},
        ),
        (
            "crem_max_size",
            "CREM max_size",
            {"mutations_per_parent": DEFAULT_MUTATIONS_PER_PARENT, "crem_min_freq": DEFAULT_CREM_MIN_FREQ},
        ),
        (
            "crem_min_freq",
            "CREM min_freq",
            {"mutations_per_parent": DEFAULT_MUTATIONS_PER_PARENT, "crem_max_size": DEFAULT_CREM_MAX_SIZE},
        ),
    ]

    # Only include panels where there is more than one value to compare
    panels = []
    for col, name, fixed in swept_params:
        mask = pd.Series(True, index=final.index)
        for k, v in fixed.items():
            mask &= final[k] == v
        sub = final[mask].sort_values(col)
        if sub[col].nunique() > 1:
            panels.append((col, name, sub))

    if not panels:
        return

    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)

    for row, (col, name, sub) in enumerate(panels):
        x_labels = sub[col].astype(str).tolist()
        x_pos = range(len(x_labels))

        for ax_col, (metric, ylabel, color) in enumerate(
            [
                ("diversity", "Mean Tanimoto Distance", "steelblue"),
                ("best_fitness", best_label, "tomato"),
            ]
        ):
            ax = axes[row, ax_col]
            bars = ax.bar(x_pos, sub[metric].values, color=color, alpha=0.82, width=0.5)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(x_labels)
            ax.set_xlabel(name)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Final {ylabel}\nvs {name}")
            ax.grid(True, alpha=0.3, axis="y")

            # Value labels on top of each bar
            for bar, val in zip(bars, sub[metric].values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Highlight the default value
            default_vals = {
                "mutations_per_parent": DEFAULT_MUTATIONS_PER_PARENT,
                "crem_max_size": DEFAULT_CREM_MAX_SIZE,
                "crem_min_freq": DEFAULT_CREM_MIN_FREQ,
            }
            if default_vals[col] in sub[col].values:
                def_idx = sub[col].tolist().index(default_vals[col])
                bars[def_idx].set_edgecolor("black")
                bars[def_idx].set_linewidth(2)

    fig.suptitle(
        "Parameter Sensitivity — Final Generation\n(black border = current default)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path = out_dir / "parameter_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_heatmaps(df: pd.DataFrame, maximize_cn: bool, out_dir: Path):
    """Heatmaps over the full factorial parameter grid (--full-grid mode only).

    Requires data for every (mutations_per_parent × crem_max_size) combination;
    use plot_parameter_comparison for one-at-a-time sweep results instead.
    """
    final = (
        df.sort_values("completed_rounds")
        .groupby("label")
        .last()
        .reset_index()
    )

    if final["mutations_per_parent"].nunique() < 2 or final["crem_max_size"].nunique() < 2:
        return

    # Fix min_freq at its most common value for a clean 2-D slice
    fixed_mf = final["crem_min_freq"].mode().iloc[0]
    slice_df = final[final["crem_min_freq"] == fixed_mf]
    if slice_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    best_label = "Best CN" if maximize_cn else "Best CN Error"

    for ax, col, title, cmap in zip(
        axes,
        ["diversity", "best_fitness"],
        [f"Final Diversity\n(min_freq={fixed_mf})", f"Final {best_label}\n(min_freq={fixed_mf})"],
        ["viridis", "RdYlGn_r" if not maximize_cn else "RdYlGn"],
    ):
        pivot = slice_df.pivot_table(
            index="mutations_per_parent",
            columns="crem_max_size",
            values=col,
            aggfunc="mean",
        )
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"max_size={c}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"mpp={r}" for r in pivot.index])
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

        mean_val = np.nanmean(pivot.values)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    txt_col = "white" if v < mean_val else "black"
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color=txt_col, fontsize=9, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, maximize_cn: bool):
    final = (
        df.sort_values("completed_rounds")
        .groupby("label")
        .last()
        .reset_index()[
            [
                "label",
                "mutations_per_parent",
                "crem_max_size",
                "crem_min_freq",
                "pop_size",
                "diversity",
                "best_fitness",
                "avg_fitness",
            ]
        ]
        .sort_values("diversity", ascending=False)
    )

    best_label = "best_CN" if maximize_cn else "best_CN_err"
    avg_label = "avg_CN" if maximize_cn else "avg_CN_err"
    final = final.rename(columns={"best_fitness": best_label, "avg_fitness": avg_label})

    width = 90
    print("\n" + "=" * width)
    print("FINAL GENERATION SUMMARY  (sorted by diversity ↓)")
    print("=" * width)
    print(final.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Recommendation
    if len(final) > 1:
        best_div = final.iloc[0]
        print(f"\n  Highest diversity : {best_div['label']}  (diversity = {best_div['diversity']:.4f})")
        if avg_label in final.columns:
            best_fit = final.sort_values(avg_label, ascending=(not maximize_cn)).iloc[0]
            print(f"  Best avg fitness  : {best_fit['label']}  ({avg_label} = {best_fit[avg_label]:.4f})")

    print("=" * width + "\n")


def _suggest_generation_count(df: pd.DataFrame):
    """Print a suggested generation count based on the 'default' convergence run."""
    default_df = df[df["label"] == "default"].sort_values("completed_rounds")
    if default_df.empty or len(default_df) < 4:
        return

    diversity = default_df["diversity"].values
    fitness = default_df["avg_fitness"].values.astype(float)
    rounds = default_df["completed_rounds"].values

    # Find generation where 95 % of total diversity change has occurred
    div_deltas = np.abs(np.diff(diversity))
    fit_deltas = np.abs(np.diff(fitness))

    def plateau_round(deltas, threshold=0.95):
        total = deltas.sum()
        if total == 0:
            return rounds[1]
        cum = np.cumsum(deltas) / total
        idx = np.searchsorted(cum, threshold)
        return int(rounds[min(idx + 1, len(rounds) - 1)])

    div_plateau = plateau_round(div_deltas)
    fit_plateau = plateau_round(fit_deltas)
    suggested = max(div_plateau, fit_plateau)

    print(f"  Diversity plateaus by generation  : {div_plateau}")
    print(f"  Fitness  plateaus by generation   : {fit_plateau}")
    print(f"  → Suggested number of generations : {suggested}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=["sweep", "convergence", "all"],
        default="all",
        help="sweep: parameter sensitivity; convergence: generation count; all: both (default)",
    )
    p.add_argument(
        "--mutations-per-parent",
        nargs="+",
        type=int,
        default=[3, 5, 10],
        metavar="N",
        help="mutations_per_parent values to sweep (default: 3 5 10)",
    )
    p.add_argument(
        "--max-size",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        metavar="N",
        help="CREM max fragment size values to sweep (default: 1 2 3)",
    )
    p.add_argument(
        "--min-freq",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        metavar="N",
        help="CREM min fragment frequency values to sweep (default: 1 3 5)",
    )
    p.add_argument(
        "--generations",
        type=int,
        default=8,
        help="generations per run in sweep mode (default: 8)",
    )
    p.add_argument(
        "--max-generations",
        type=int,
        default=15,
        help="generations for convergence analysis (default: 15)",
    )
    p.add_argument("--target-cn", type=float, default=50.0)
    p.add_argument("--maximize-cn", action="store_true", default=False)
    p.add_argument("--minimize-ysi", action="store_true", default=False)
    p.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="population size (reduce for faster sweeps, default: 100)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--full-grid",
        action="store_true",
        default=False,
        help="use full factorial grid instead of one-at-a-time sweep (much slower)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="show GA output for each run (off by default to reduce noise)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"directory for output files (default: {RESULTS_DIR})",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build run list
    run_list: List[Dict] = []

    if args.mode in ("sweep", "all"):
        builder = build_full_grid_configs if args.full_grid else build_one_at_a_time_configs
        sweep_runs = builder(
            args.mutations_per_parent,
            args.max_size,
            args.min_freq,
        )
        for r in sweep_runs:
            r["max_generations"] = args.generations
            r["run_type"] = "sweep"
        run_list.extend(sweep_runs)

    if args.mode in ("convergence", "all"):
        # One long run with default mutation params
        default_label = "default"
        # Check if a default config is already in sweep runs to avoid double-running
        already_have_default = any(
            r["label"] == default_label and r["max_generations"] >= args.max_generations
            for r in run_list
        )
        if not already_have_default:
            run_list.append(
                {
                    "mutations_per_parent": DEFAULT_MUTATIONS_PER_PARENT,
                    "crem_max_size": DEFAULT_CREM_MAX_SIZE,
                    "crem_min_freq": DEFAULT_CREM_MIN_FREQ,
                    "label": default_label,
                    "max_generations": args.max_generations,
                    "run_type": "convergence",
                }
            )

    total = len(run_list)
    print(f"\n{'=' * 65}")
    print(f"GA Parameter Evaluation  —  {total} run(s)")
    print(f"Population size : {args.population_size}")
    print(f"Seed            : {args.seed}")
    print(f"Output dir      : {args.output_dir}")
    print(f"{'=' * 65}\n")

    all_records: List[Dict] = []

    for idx, run in enumerate(run_list, 1):
        mpp = run["mutations_per_parent"]
        ms = run["crem_max_size"]
        mf = run["crem_min_freq"]
        gens = run["max_generations"]
        label = run["label"]

        print(
            f"[{idx:2d}/{total}]  {label:<30s}  "
            f"(mpp={mpp}, max_size={ms}, min_freq={mf}, gens={gens})",
            end="  ",
            flush=True,
        )

        try:
            snapshots = run_config(
                mutations_per_parent=mpp,
                crem_max_size=ms,
                crem_min_freq=mf,
                max_generations=gens,
                target_cn=args.target_cn,
                maximize_cn=args.maximize_cn,
                minimize_ysi=args.minimize_ysi,
                population_size=args.population_size,
                seed=args.seed,
                verbose=args.verbose,
            )
            final_snap = snapshots[-1] if snapshots else {}
            div = final_snap.get("diversity", float("nan"))
            fit = final_snap.get("avg_fitness", float("nan"))
            print(f"✓  final_div={div:.3f}  final_avg_fit={fit:.3f}")
        except Exception as exc:
            print(f"✗  {exc}")
            snapshots = []

        for snap in snapshots:
            all_records.append(
                {
                    "mutations_per_parent": mpp,
                    "crem_max_size": ms,
                    "crem_min_freq": mf,
                    "label": label,
                    "run_type": run["run_type"],
                    **snap,
                }
            )

    if not all_records:
        print("\nNo results collected — check that the environment is set up correctly.")
        return

    results_df = pd.DataFrame(all_records)
    csv_path = args.output_dir / "sweep_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓  Results saved → {csv_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots:")
    sweep_df = results_df[results_df["run_type"] == "sweep"]
    if not sweep_df.empty:
        plot_diversity_curves(sweep_df, args.output_dir)
        plot_fitness_curves(sweep_df, args.maximize_cn, args.output_dir)
        if args.full_grid:
            plot_heatmaps(sweep_df, args.maximize_cn, args.output_dir)
        else:
            plot_parameter_comparison(sweep_df, args.maximize_cn, args.output_dir)

    if args.mode in ("convergence", "all"):
        plot_convergence_summary(results_df, args.maximize_cn, args.output_dir)

    # ── Summary & recommendations ────────────────────────────────────────────
    _print_summary(results_df, args.maximize_cn)

    convergence_df = results_df[results_df["label"] == "default"]
    if not convergence_df.empty:
        print("  Generation count recommendation (default config):")
        _suggest_generation_count(convergence_df)


if __name__ == "__main__":
    main()
