import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

class WandBAnalyzer:
    """Analyze existing W&B runs to optimize hyperparameters."""
    
    def __init__(self, entity: str, project: str):
        """
        Initialize analyzer.
        
        Args:
            entity: Your W&B username/team
            project: Project name (e.g., "cetane-ysi-pareto" or "cetane-optimization")
        """
        self.api = wandb.Api()
        self.entity = entity
        self.project = project
        self.runs_df = None
        
    def fetch_runs(self, filters: Optional[Dict] = None, 
                   min_generations: int = 3) -> pd.DataFrame:
        """
        Fetch all runs from W&B.
        
        Args:
            filters: Optional filters (e.g., {"config.target_cn": 50})
            min_generations: Only include runs with at least this many generations
        """
        print(f"Fetching runs from {self.entity}/{self.project}...")
        
        runs = self.api.runs(f"{self.entity}/{self.project}", filters=filters)
        
        data = []
        for run in runs:
            # Get config
            config = {k: v for k, v in run.config.items() 
                     if not k.startswith('_')}
            
            # Get summary metrics (final values)
            summary = run.summary._json_dict
            
            # Get run history for time-series analysis
            history = run.history()
            
            if len(history) < min_generations:
                continue
            
            # Combine everything
            row = {
                'run_id': run.id,
                'run_name': run.name,
                'state': run.state,
                'created_at': run.created_at,
                'runtime_seconds': (run.summary.get('_runtime', 0)),
                **config,
                **{f'final_{k}': v for k, v in summary.items() 
                   if not k.startswith('_')}
            }
            
            # Add convergence metrics from history
            if 'best_cn_error' in history.columns:
                row['convergence_generation'] = self._find_convergence(
                    history['best_cn_error'].values
                )
                row['improvement_rate'] = self._calc_improvement_rate(
                    history['best_cn_error'].values
                )
            
            data.append(row)
        
        self.runs_df = pd.DataFrame(data)
        print(f"✓ Loaded {len(self.runs_df)} runs")
        
        return self.runs_df
    
    def _find_convergence(self, errors: np.ndarray, 
                          threshold: float = 0.01) -> int:
        """
        Find generation where algorithm converged.
        Convergence = no improvement > threshold for rest of run.
        """
        if len(errors) < 2:
            return len(errors)
        
        for i in range(len(errors) - 1):
            remaining = errors[i:]
            if np.all(np.abs(np.diff(remaining)) < threshold):
                return i + 1
        
        return len(errors)
    
    def _calc_improvement_rate(self, errors: np.ndarray) -> float:
        """Calculate average improvement per generation."""
        if len(errors) < 2:
            return 0.0
        return (errors[0] - errors[-1]) / len(errors)
    
    def analyze_hyperparameters(self, metric: str = 'final_best_cn_error',
                               minimize: bool = True) -> pd.DataFrame:
        """
        Analyze effect of each hyperparameter on performance.
        
        Args:
            metric: Metric to optimize (e.g., 'final_best_cn_error')
            minimize: Whether to minimize (True) or maximize (False) the metric
        """
        if self.runs_df is None:
            raise ValueError("No runs loaded. Call fetch_runs() first.")
        
        df = self.runs_df.copy()
        
        # Identify hyperparameter columns
        hyperparam_cols = [col for col in df.columns if col in [
            'generations', 'population_size', 'mutations_per_parent',
            'survivor_fraction', 'batch_size'
        ]]
        
        results = []
        
        for param in hyperparam_cols:
            # Group by this parameter
            grouped = df.groupby(param)[metric].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).reset_index()
            
            grouped['parameter'] = param
            grouped['cv'] = grouped['std'] / grouped['mean']  # Coefficient of variation
            
            results.append(grouped)
        
        summary = pd.concat(results, ignore_index=True)
        
        return summary
    
    def plot_convergence_curves(self, run_ids: Optional[List[str]] = None,
                               save_path: Optional[Path] = None):
        """
        Plot convergence curves for selected runs.
        
        Args:
            run_ids: Specific run IDs to plot. If None, plots best 5 runs.
            save_path: Where to save the plot
        """
        if run_ids is None:
            # Get best 5 runs
            df = self.runs_df.sort_values('final_best_cn_error').head(5)
            run_ids = df['run_id'].tolist()
        
        plt.figure(figsize=(12, 6))
        
        for run_id in run_ids:
            run = self.api.run(f"{self.entity}/{self.project}/{run_id}")
            history = run.history()
            
            if 'best_cn_error' in history.columns:
                config_str = (f"gen={run.config.get('generations', '?')}, "
                            f"pop={run.config.get('population_size', '?')}")
                
                plt.plot(history['generation'], history['best_cn_error'],
                        marker='o', label=config_str, alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Best CN Error')
        plt.title('Convergence Curves: Top Runs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_hyperparameter_effects(self, metric: str = 'final_best_cn_error',
                                    save_dir: Optional[Path] = None):
        """Create comprehensive hyperparameter analysis plots."""
        
        if save_dir:
            save_dir.mkdir(exist_ok=True)
        
        df = self.runs_df.copy()
        
        # Identify hyperparameter columns
        hyperparam_cols = [col for col in df.columns if col in [
            'generations', 'population_size', 'mutations_per_parent',
            'survivor_fraction', 'batch_size'
        ]]
        
        n_params = len(hyperparam_cols)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(hyperparam_cols):
            ax = axes[i]
            
            # Box plot with swarm overlay
            sns.boxplot(data=df, x=param, y=metric, ax=ax, color='lightblue')
            sns.swarmplot(data=df, x=param, y=metric, ax=ax, 
                         color='darkblue', alpha=0.5, size=3)
            
            ax.set_title(f'Effect of {param.replace("_", " ").title()}')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'hyperparameter_effects.png', 
                       dpi=300, bbox_inches='tight')
            print(f"Saved to {save_dir / 'hyperparameter_effects.png'}")
        
        plt.show()
    
    def plot_interaction_heatmap(self, param1: str, param2: str,
                                metric: str = 'final_best_cn_error',
                                save_path: Optional[Path] = None):
        """
        Plot heatmap showing interaction between two hyperparameters.
        
        Args:
            param1: First hyperparameter (e.g., 'generations')
            param2: Second hyperparameter (e.g., 'population_size')
            metric: Performance metric
        """
        df = self.runs_df.copy()
        
        # Create pivot table
        pivot = df.pivot_table(
            values=metric,
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   cbar_kws={'label': metric})
        plt.title(f'{metric} by {param1} and {param2}')
        plt.xlabel(param2.replace('_', ' ').title())
        plt.ylabel(param1.replace('_', ' ').title())
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_efficiency_frontier(self, save_path: Optional[Path] = None):
        """
        Plot Pareto frontier: performance vs computational cost.
        Shows best trade-offs between quality and runtime.
        """
        df = self.runs_df.copy()
        
        # Calculate total evaluations
        df['total_evaluations'] = (
            df['population_size'] + 
            (df['generations'] * df['population_size'] * 
             df.get('mutations_per_parent', 5))
        )
        
        plt.figure(figsize=(12, 7))
        
        # Scatter plot
        scatter = plt.scatter(
            df['total_evaluations'], 
            df['final_best_cn_error'],
            c=df['generations'],
            s=df['population_size'],
            alpha=0.6,
            cmap='viridis',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(
            df['total_evaluations'].values,
            df['final_best_cn_error'].values,
            minimize_both=True
        )
        
        if len(pareto_points) > 0:
            pareto_df = df.iloc[pareto_points].sort_values('total_evaluations')
            plt.plot(pareto_df['total_evaluations'], 
                    pareto_df['final_best_cn_error'],
                    'r--', linewidth=2, label='Pareto Frontier', alpha=0.7)
            
            # Annotate best points
            for _, row in pareto_df.iterrows():
                plt.annotate(
                    f"g={int(row['generations'])}, p={int(row['population_size'])}",
                    (row['total_evaluations'], row['final_best_cn_error']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, alpha=0.7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                )
        
        plt.colorbar(scatter, label='Generations')
        plt.xlabel('Total Evaluations (molecules tested)')
        plt.ylabel('Best CN Error')
        plt.title('Efficiency Frontier: Performance vs Computational Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return pareto_df if len(pareto_points) > 0 else None
    
    def _find_pareto_frontier(self, x: np.ndarray, y: np.ndarray,
                             minimize_both: bool = True) -> List[int]:
        """Find indices of points on the Pareto frontier."""
        points = np.column_stack([x, y])
        pareto_indices = []
        
        for i, point in enumerate(points):
            dominated = False
            for j, other in enumerate(points):
                if i == j:
                    continue
                
                if minimize_both:
                    # Check if 'other' dominates 'point'
                    if (other[0] <= point[0] and other[1] <= point[1] and
                        (other[0] < point[0] or other[1] < point[1])):
                        dominated = True
                        break
            
            if not dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def get_recommendations(self, metric: str = 'final_best_cn_error',
                           budget: Optional[str] = None) -> Dict:
        """
        Get hyperparameter recommendations based on existing runs.
        
        Args:
            metric: Metric to optimize
            budget: 'low', 'medium', 'high', or None for best overall
        """
        df = self.runs_df.copy()
        
        # Calculate efficiency score
        df['total_evaluations'] = (
            df['population_size'] + 
            (df['generations'] * df['population_size'] * 
             df.get('mutations_per_parent', 5))
        )
        df['efficiency'] = df[metric] / (df['total_evaluations'] / 1000)
        
        recommendations = {}
        
        # Best overall (lowest error)
        best_overall = df.loc[df[metric].idxmin()]
        recommendations['best_overall'] = {
            'generations': int(best_overall['generations']),
            'population_size': int(best_overall['population_size']),
            'mutations_per_parent': int(best_overall.get('mutations_per_parent', 5)),
            'survivor_fraction': float(best_overall.get('survivor_fraction', 0.5)),
            'expected_cn_error': float(best_overall[metric]),
            'expected_runtime': float(best_overall.get('runtime_seconds', 0)),
            'total_evaluations': int(best_overall['total_evaluations'])
        }
        
        # Most efficient (best error per evaluation)
        best_efficiency = df.loc[df['efficiency'].idxmin()]
        recommendations['most_efficient'] = {
            'generations': int(best_efficiency['generations']),
            'population_size': int(best_efficiency['population_size']),
            'mutations_per_parent': int(best_efficiency.get('mutations_per_parent', 5)),
            'survivor_fraction': float(best_efficiency.get('survivor_fraction', 0.5)),
            'expected_cn_error': float(best_efficiency[metric]),
            'efficiency_score': float(best_efficiency['efficiency'])
        }
        
        # Budget-based recommendations
        if budget:
            if budget == 'low':
                # Fastest runs (bottom 25% of evaluations)
                threshold = df['total_evaluations'].quantile(0.25)
                subset = df[df['total_evaluations'] <= threshold]
            elif budget == 'medium':
                # Middle 50%
                q25 = df['total_evaluations'].quantile(0.25)
                q75 = df['total_evaluations'].quantile(0.75)
                subset = df[(df['total_evaluations'] > q25) & 
                           (df['total_evaluations'] <= q75)]
            else:  # high
                # Top 25%
                threshold = df['total_evaluations'].quantile(0.75)
                subset = df[df['total_evaluations'] > threshold]
            
            if not subset.empty:
                best_in_budget = subset.loc[subset[metric].idxmin()]
                recommendations[f'best_{budget}_budget'] = {
                    'generations': int(best_in_budget['generations']),
                    'population_size': int(best_in_budget['population_size']),
                    'mutations_per_parent': int(best_in_budget.get('mutations_per_parent', 5)),
                    'expected_cn_error': float(best_in_budget[metric]),
                    'total_evaluations': int(best_in_budget['total_evaluations'])
                }
        
        return recommendations
    
    def print_recommendations(self, budget: Optional[str] = None):
        """Print formatted recommendations."""
        recs = self.get_recommendations(budget=budget)
        
        print("\n" + "="*70)
        print("HYPERPARAMETER RECOMMENDATIONS (from W&B analysis)")
        print("="*70)
        
        for name, config in recs.items():
            print(f"\n🎯 {name.upper().replace('_', ' ')}:")
            for key, value in config.items():
                if isinstance(value, float):
                    print(f"  {key:.<30} {value:.4f}")
                else:
                    print(f"  {key:.<30} {value}")
    
    def export_analysis(self, output_dir: Path):
        """Export comprehensive analysis to files."""
        output_dir.mkdir(exist_ok=True)
        
        # Save runs data
        self.runs_df.to_csv(output_dir / 'all_runs.csv', index=False)
        
        # Hyperparameter analysis
        analysis = self.analyze_hyperparameters()
        analysis.to_csv(output_dir / 'hyperparameter_analysis.csv', index=False)
        
        # Recommendations
        recs = self.get_recommendations()
        recs_df = pd.DataFrame(recs).T
        recs_df.to_csv(output_dir / 'recommendations.csv')
        
        print(f"✓ Exported analysis to {output_dir}/")


# Example usage
def analyze_my_runs(entity: str, project: str = "cetane-ysi-pareto"):
    """Main analysis function."""
    
    print("\n" + "="*70)
    print("ANALYZING EXISTING W&B RUNS")
    print("="*70)
    
    analyzer = WandBAnalyzer(entity, project)
    
    # Fetch all runs
    df = analyzer.fetch_runs()
    
    if df.empty:
        print("No runs found!")
        return None
    
    print(f"\nAnalyzing {len(df)} runs...")
    
    # Generate all visualizations
    print("\n1. Plotting hyperparameter effects...")
    analyzer.plot_hyperparameter_effects(save_dir=Path("wandb_analysis"))
    
    print("\n2. Plotting efficiency frontier...")
    pareto_df = analyzer.plot_efficiency_frontier(
        save_path=Path("wandb_analysis/efficiency_frontier.png")
    )
    
    print("\n3. Plotting convergence curves...")
    analyzer.plot_convergence_curves(
        save_path=Path("wandb_analysis/convergence_curves.png")
    )
    
    print("\n4. Plotting generation vs population heatmap...")
    analyzer.plot_interaction_heatmap(
        'generations', 
        'population_size',
        save_path=Path("wandb_analysis/gen_vs_pop_heatmap.png")
    )
    
    # Print recommendations
    analyzer.print_recommendations()
    
    # Export everything
    analyzer.export_analysis(Path("wandb_analysis"))
    
    print("\n✓ Analysis complete!")
    
    return analyzer


if __name__ == "__main__":
    # Replace with your W&B username/team
    WANDB_ENTITY = "salza2004"  # <-- CHANGE THIS
    
    # Run analysis
    analyzer = analyze_my_runs(WANDB_ENTITY, project="cetane-ysi-pareto")