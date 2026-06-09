from .population import Population
from .molecule import Molecule
from core.predictors.pure_component.property_predictor import PropertyPredictor
from core.config import EvolutionConfig
from crem.crem import mutate_mol
from rdkit import Chem
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict
from core.data_prep import df  
from pathlib import Path
from .anomaly_detection import EnsembleUncertaintyFilter
class MolecularEvolution:
    """Main evolutionary algorithm coordinator with uncertainty filtering."""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    REP_DB_PATH = BASE_DIR / "data" / "fragments" / "diesel_fragments.db"

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.predictor = PropertyPredictor(config)
        self.population = Population(config)
        
        # Initialize and calibrate uncertainty filters
        self._setup_uncertainty_filters()
    
    def _setup_uncertainty_filters(self):
        """Setup and calibrate uncertainty filters for CN and YSI."""
        from core.shared_features import featurize_df
        
        print("\nCalibrating ensemble uncertainty filters...")
        
        val_df = df.sample(200, random_state=42)  # Sample DataFrame, not just SMILES
        val_smiles = val_df["SMILES"].tolist()    # Extract SMILES
        y_val_cn = val_df["cn"].values

        X_val = featurize_df(val_smiles, return_df=False)
        
        if X_val is None:
            print("⚠ Warning: Could not featurize validation set, uncertainty filtering disabled")
            self.uncertainty_filters = {}
            return
        
        self.uncertainty_filters = {}
        
        # Calibrate CN uncertainty filter
        if 'cn' in self.predictor.predictors:
            print("\n  Calibrating CN uncertainty filter...")
            
            
            # Get predictions with uncertainty
            cn_predictor = self.predictor.predictors['cn']
            X_cn = cn_predictor.selector.transform(X_val)
            
            # Get variance across trees
            all_tree_preds = np.array([
                tree.predict(X_cn) for tree in cn_predictor.model.estimators_
            ])
            cn_variances = np.std(all_tree_preds, axis=0)
            
            # Calculate errors for analysis
            cn_preds = np.mean(all_tree_preds, axis=0)
            cn_errors = np.abs(cn_preds - y_val_cn)
            
            print(f"    Mean prediction error: {cn_errors.mean():.3f}")
            print(f"    Mean variance: {cn_variances.mean():.4f}")
            
            # Calibrate filter
            self.uncertainty_filters['cn'] = EnsembleUncertaintyFilter(percentile_threshold=90)
            self.uncertainty_filters['cn'].calibrate(cn_variances)
            
            # Show effectiveness
            filtered_mask = cn_variances > self.uncertainty_filters['cn'].variance_threshold
            if filtered_mask.sum() > 0:
                print(f"    Will filter: {filtered_mask.sum()}/{len(cn_variances)} ({filtered_mask.mean()*100:.1f}%)")
                print(f"    Filtered have error: {cn_errors[filtered_mask].mean():.3f}")
                print(f"    Kept have error: {cn_errors[~filtered_mask].mean():.3f}")
        
        # Calibrate YSI uncertainty filter (if used)
        if self.config.minimize_ysi and 'ysi' in self.predictor.predictors:
            print("\n  Calibrating YSI uncertainty filter...")
            y_val_ysi = val_df["YSI_Unified_Measured"].values

            
            # Get predictions with uncertainty
            ysi_predictor = self.predictor.predictors['ysi']
            X_ysi = ysi_predictor.selector.transform(X_val)
            
            # Get variance across trees
            all_tree_preds = np.array([
                tree.predict(X_ysi) for tree in ysi_predictor.model.estimators_
            ])
            ysi_variances = np.std(all_tree_preds, axis=0)
            
            # Calculate errors for analysis
            ysi_preds = np.mean(all_tree_preds, axis=0)
            ysi_errors = np.abs(ysi_preds - y_val_ysi)
            
            print(f"    Mean prediction error: {ysi_errors.mean():.3f}")
            print(f"    Mean variance: {ysi_variances.mean():.4f}")
            
            # Calibrate filter
            self.uncertainty_filters['ysi'] = EnsembleUncertaintyFilter(percentile_threshold=90)
            self.uncertainty_filters['ysi'].calibrate(ysi_variances)
            
            # Show effectiveness
            filtered_mask = ysi_variances > self.uncertainty_filters['ysi'].variance_threshold
            if filtered_mask.sum() > 0:
                print(f"    Will filter: {filtered_mask.sum()}/{len(ysi_variances)} ({filtered_mask.mean()*100:.1f}%)")
                print(f"    Filtered have error: {ysi_errors[filtered_mask].mean():.3f}")
                print(f"    Kept have error: {ysi_errors[~filtered_mask].mean():.3f}")
        
        print("\n✓ Uncertainty filtering ready\n")
    
    def _mutate_molecule(self, mol: Chem.Mol) -> List[str]:
        """Generate mutations for a molecule using CREM."""
        try:
            mutants = list(mutate_mol(
                mol,
                db_name=str(self.REP_DB_PATH),
                max_size=self.config.max_size,
                max_replacements=100,
                min_freq=self.config.min_freq,
                return_mol=False
            ))
            return [m for m in mutants if m and m not in self.population.seen_smiles]
        except Exception:
            return []
    
    def _predict_with_uncertainty(self, smiles_list: List[str]) -> Tuple[Dict, Dict]:
        """
        Predict all properties with uncertainty estimates.
        
        Returns:
            predictions: Dict of property predictions
            uncertainties: Dict of uncertainty estimates (std across trees)
        """
        from core.shared_features import featurize_df
        
        if not smiles_list:
            return {}, {}
        
        # Featurize once
        X_full = featurize_df(smiles_list, return_df=False)
        
        if X_full is None:
            return {}, {}
        
        predictions = {}
        uncertainties = {}
        
        # Predict each property with uncertainty
        for prop_name, predictor_obj in self.predictor.predictors.items():
            # Select features
            X_selected = predictor_obj.selector.transform(X_full)
            
            # Get predictions from all trees
            all_tree_preds = np.array([
                tree.predict(X_selected) for tree in predictor_obj.model.estimators_
            ])
            
            # Mean and std (uncertainty stays in model output space for calibration consistency)
            mean_preds = np.mean(all_tree_preds, axis=0)
            if predictor_obj.uses_log_transform:
                mean_preds = np.power(10.0, mean_preds)
            predictions[prop_name] = mean_preds
            uncertainties[prop_name] = np.std(all_tree_preds, axis=0)
        
        # Add Tanimoto (no uncertainty)
        predictions["tanimoto"] = self.predictor.compute_tanimoto(smiles_list)
        
        return predictions, uncertainties
    
    def _create_molecules(self, smiles_list: List[str]) -> Tuple[List[Molecule], Dict]:
        """Create Molecule objects with UNCERTAINTY FILTERING."""
        if not smiles_list:
            return [], {}
        
        # Track filtering stats
        filter_stats = {
            'total': len(smiles_list),
            'cn_none': 0,
            'ysi_none': 0,
            'tanimoto_fail': 0,
            'cn_uncertainty_fail': 0,
            'ysi_uncertainty_fail': 0,
            'passed': 0
        }
        
        # Predict with uncertainty
        predictions, uncertainties = self._predict_with_uncertainty(smiles_list)
        
        molecules = []
        
        for i, smiles in enumerate(smiles_list):
            # Extract predictions
            props = {k: predictions[k][i] for k in predictions}

            uncerts = {k: v[i] for k, v in uncertainties.items() if k in uncertainties}
            
            # Validate CN exists
            if props.get('cn') is None or not np.isfinite(props['cn']):
                filter_stats['cn_none'] += 1
                continue
            
            # NEW: Filter by CN uncertainty
            if 'cn' in self.uncertainty_filters and 'cn' in uncerts:
                cn_variance = uncerts['cn']
                is_reliable = self.uncertainty_filters['cn'].is_reliable(
                    np.array([cn_variance])
                )[0]
                
                if not is_reliable:
                    filter_stats['cn_uncertainty_fail'] += 1
                    continue
            
            # YSI validation
            if self.config.minimize_ysi:
                if props.get('ysi') is None or not np.isfinite(props['ysi']):
                    filter_stats['ysi_none'] += 1
                    continue
                
                # NEW: Filter by YSI uncertainty
                if 'ysi' in self.uncertainty_filters and 'ysi' in uncerts:
                    ysi_variance = uncerts['ysi']
                    is_reliable = self.uncertainty_filters['ysi'].is_reliable(
                        np.array([ysi_variance])
                    )[0]
                    
                    if not is_reliable:
                        filter_stats['ysi_uncertainty_fail'] += 1
                        continue
            
            # Tanimoto filter
            tanimoto = props.get("tanimoto")
            if tanimoto is None or tanimoto < 0.7:
                filter_stats['tanimoto_fail'] += 1
                continue

            filter_stats['passed'] += 1
            
            mol = Molecule(
            smiles=smiles,
            cn=float(props['cn']),
            cn_error=abs(float(props['cn']) - self.config.target_cn),
            cn_score=0.0,  # placeholder, set below
            bp=float(props.get('bp')) if props.get('bp') is not None else None,
            ysi=float(props.get('ysi')) if props.get('ysi') is not None else None,
            density=float(props.get('density')) if props.get('density') is not None else None,
            lhv=float(props.get('lhv')) if props.get('lhv') is not None else None,
            dynamic_viscosity=float(props.get('dynamic_viscosity')) if props.get('dynamic_viscosity') is not None else None
        )
            mol.cn_score = mol.fitness(self.config)
            molecules.append(mol)
        
        return molecules, filter_stats
    
    def initialize_population(self, initial_smiles: List[str]) -> int:
        """Initialize the population from initial SMILES."""
        print("Predicting properties for initial population...")
        molecules, filter_stats = self._create_molecules(initial_smiles)
        
        # Log initial filtering
        if filter_stats['total'] > 0:
            print(f"\n  Initial filtering: {filter_stats['total']} → {filter_stats['passed']} passed")
            print(f"    Tanimoto: {filter_stats['tanimoto_fail']} | "
                  f"CN Uncertainty: {filter_stats['cn_uncertainty_fail']} | "
                  f"YSI Uncertainty: {filter_stats['ysi_uncertainty_fail']}")
        
        return self.population.add_molecules(molecules)
    
    def _log_generation_stats(self, generation: int):
        """Log statistics for the current generation."""
        mols = self.population.molecules
        if not mols:
            print(f"Gen {generation}: Population empty after filtering.")
            return

        
        if self.config.maximize_cn:
            best_cn = max(mols, key=lambda m: m.cn)
            avg_cn = np.mean([m.cn for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best CN: {best_cn.cn:.3f} | "
                        f"Avg CN: {avg_cn:.3f}")
        else:
            best_cn = min(mols, key=lambda m: m.cn_error)
            avg_cn_err = np.mean([m.cn_error for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best CN err: {best_cn.cn_error:.3f} | "
                        f"Avg CN err: {avg_cn_err:.3f}")
        
        if self.config.minimize_ysi:
            front = self.population.pareto_front()
            best_ysi = min(mols, key=lambda m: m.ysi)
            avg_ysi = np.mean([m.ysi for m in mols])
            
            print_msg += (f" | Best YSI: {best_ysi.ysi:.3f} | "
                         f"Avg YSI: {avg_ysi:.3f} | "
                         f"Pareto: {len(front)}")
        
        print(print_msg)
    
    def _generate_offspring(self, survivors: List[Molecule]) -> Tuple[List[Molecule], Dict]:
        """Generates offspring from survivors with filtering stats."""
        target_count = self.config.population_size - len(survivors)
        max_attempts = target_count * self.config.max_offspring_attempts
        
        all_children = []
        new_molecules = []
        cumulative_stats = {
            'total': 0,
            'cn_none': 0,
            'ysi_none': 0,
            'tanimoto_fail': 0,
            'cn_uncertainty_fail': 0,
            'ysi_uncertainty_fail': 0,
            'passed': 0
        }
        
        print(f"  → Generating offspring (target: {target_count})...")
        
        for attempt in range(max_attempts):
            if len(new_molecules) >= target_count:
                break
            
            # Generate mutations
            # AFTER — fitness-proportionate (roulette wheel) selection
            weights = np.array([m.fitness(self.config) for m in survivors])
            weights /= weights.sum()
            parent = survivors[np.random.choice(len(survivors), p=weights)]
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                continue
            
            children = self._mutate_molecule(mol)
            all_children.extend(children[:self.config.mutations_per_parent])
            
            # Process in batches
            if len(all_children) >= self.config.batch_size:
                batch_mols, batch_stats = self._create_molecules(all_children)
                new_molecules.extend(batch_mols)
                
                # Accumulate stats
                for key in cumulative_stats:
                    cumulative_stats[key] += batch_stats[key]
                
                all_children = []
        
        # Process remaining children
        if all_children:
            batch_mols, batch_stats = self._create_molecules(all_children)
            new_molecules.extend(batch_mols)
            
            for key in cumulative_stats:
                cumulative_stats[key] += batch_stats[key]
        
        print(f"  ✓ Generated {len(new_molecules)} valid offspring")
        print(f"    Filtering: {cumulative_stats['total']} → {cumulative_stats['passed']} | "
              f"Tanimoto: {cumulative_stats['tanimoto_fail']} | "
              f"CN Unc: {cumulative_stats['cn_uncertainty_fail']} | "
              f"YSI Unc: {cumulative_stats['ysi_uncertainty_fail']} "
              f"(property constraints applied at end)")
        
        return new_molecules, cumulative_stats
    
    def _run_evolution_loop(self):
        """Run the main evolution loop."""
        for gen in range(1, self.config.generations + 1):
            self._log_generation_stats(gen)
            
            survivors = self.population.get_survivors()
            offspring, _ = self._generate_offspring(survivors)
            
            # Create new population
            new_pop = Population(self.config)
            new_pop.add_molecules(survivors + offspring)
            self.population = new_pop
    
    def _apply_property_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply bp, density, lhv, dynamic_viscosity constraints to a DataFrame."""
        mask = pd.Series(True, index=df.index)
        for prop, (lo, hi) in self.config.filters.items():
            if prop not in df.columns:
                continue
            col = df[prop]
            if lo is not None:
                mask &= col >= lo
            if hi is not None:
                mask &= col <= hi
        filtered = df[mask]
        removed = len(df) - len(filtered)
        if removed > 0:
            print(f"  Property filters removed {removed}/{len(df)} molecules")
        return filtered

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.maximize_cn:
            if self.config.minimize_ysi and "ysi" in df.columns:
                candidates = df[df["cn"] > 30] if len(df[df["cn"] > 50]) < 10 else df[df["cn"] > 50]
                return candidates.sort_values(["cn", "ysi"], ascending=[False, True])
            else:
                candidates = df[df["cn"] > 30] if len(df[df["cn"] > 50]) < 10 else df[df["cn"] > 50]
                return candidates.sort_values("cn", ascending=False)
        else:
            if self.config.minimize_ysi and "ysi" in df.columns:
                return df[df["cn_error"] < 15].sort_values(["cn_error", "ysi"], ascending=True)
            else:
                return df[df["cn_error"] < 5].sort_values("cn_error", ascending=True)

    def _generate_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate final results DataFrames (filtered, unfiltered, pareto)."""
        raw_df = self.population.to_dataframe()

        # Unfiltered: only CN/YSI objectives applied, no property constraints
        unfiltered_df = self._sort_df(raw_df).copy()
        unfiltered_df["rank"] = range(1, len(unfiltered_df) + 1)

        # Filtered: property constraints (bp, density, lhv, dynamic_viscosity) applied first
        final_df = self._sort_df(self._apply_property_filters(raw_df)).copy()
        final_df["rank"] = range(1, len(final_df) + 1)

        if self.config.minimize_ysi:
            pareto_mols = self.population.pareto_front()

            pareto_df = pd.DataFrame([m.to_dict() for m in pareto_mols])
            if not pareto_df.empty:
                pareto_df = self._sort_df(pareto_df).copy()
                pareto_df.insert(0, 'rank', range(1, len(pareto_df) + 1))
        else:
            pareto_df = pd.DataFrame()

        return final_df, pareto_df, unfiltered_df
    
    def evolve(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the evolutionary algorithm."""
        # Initialize
        df_bins = pd.qcut(df["cn"], q=30)
        initial_smiles = (
            df.groupby(df_bins, observed=False)
            .apply(lambda x: x.sample(20, random_state=42))
            .reset_index(drop=True)["SMILES"]
            .tolist()
        )
        init_count = self.initialize_population(initial_smiles)

        if init_count == 0:
            print("No valid initial molecules")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print(f"✓ Initial population size: {init_count}\n")

        # Evolution
        self._run_evolution_loop()

        # Results
        return self._generate_results()