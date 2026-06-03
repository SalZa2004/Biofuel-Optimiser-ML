"""
Run the pure molecule generator with maximize_cn + minimize_ysi objectives.
Tracks wall-clock runtime per generation and reports total + average at the end.
"""
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import EvolutionConfig
from core.evolution.evolution import MolecularEvolution
from core.evolution.population import Population


class TimedMolecularEvolution(MolecularEvolution):
    """MolecularEvolution subclass that records wall-clock time for each generation."""

    def _run_evolution_loop(self):
        self._gen_times: list[float] = []

        for gen in range(1, self.config.generations + 1):
            t0 = time.perf_counter()

            self._log_generation_stats(gen)

            survivors = self.population.get_survivors()
            offspring, _ = self._generate_offspring(survivors)

            new_pop = Population(self.config)
            new_pop.add_molecules(survivors + offspring)
            self.population = new_pop

            elapsed = time.perf_counter() - t0
            self._gen_times.append(elapsed)
            print(f"  ⏱  Generation {gen} runtime: {elapsed:.1f}s")


def main():
    config = EvolutionConfig(
        maximize_cn=True,
        minimize_ysi=True,
        generations=15,
        population_size=100,
    )

    print("=" * 60)
    print("Pure Molecule Generator — maximize CN / minimize YSI")
    print(f"Generations : {config.generations}")
    print(f"Population  : {config.population_size}")
    print("=" * 60 + "\n")

    wall_start = time.perf_counter()
    evo = TimedMolecularEvolution(config)
    final_df, pareto_df, unfiltered_df = evo.evolve()
    wall_total = time.perf_counter() - wall_start

    gen_times = getattr(evo, "_gen_times", [])

    print("\n" + "=" * 60)
    print("Runtime Summary")
    print("=" * 60)
    if gen_times:
        for i, t in enumerate(gen_times, 1):
            print(f"  Gen {i:>2}: {t:6.1f}s")
        print("-" * 60)
        print(f"  Total (evolution loop) : {sum(gen_times):.1f}s")
        print(f"  Avg per generation     : {sum(gen_times) / len(gen_times):.1f}s")
    print(f"  Total wall-clock time  : {wall_total:.1f}s  (includes init)")
    print("=" * 60)

    if not final_df.empty:
        print(f"\nTop 10 molecules (filtered):\n{final_df.head(10).to_string(index=False)}")

    if pareto_df is not None and not pareto_df.empty:
        out_path = "results/pure_cn_ysi_pareto.csv"
        os.makedirs("results", exist_ok=True)
        pareto_df.to_csv(out_path, index=False)
        print(f"\nPareto front saved to {out_path}")


if __name__ == "__main__":
    main()
