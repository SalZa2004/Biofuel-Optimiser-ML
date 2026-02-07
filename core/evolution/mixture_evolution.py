"""
Mixture-Aware Molecular Evolution

Extends the base evolution to optimize fuel additives for blend performance.

Key differences from pure component:
1. Fitness = mixture DCN (not pure component CN)
2. Each molecule is evaluated in a blend with base fuel
3. Can optionally optimize blend ratio too
"""

from .population import Population
from .molecule import Molecule
from core.predictors.pure_component.property_predictor import PropertyPredictor
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor, BaseFuelLibrary  # Real implementation
from core.config import EvolutionConfig, MixtureConfig
from crem.crem import mutate_mol
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict
from core.data_prep import df
from pathlib import Path


class MixtureAwareMolecule(Molecule):
    """
    Extended Molecule class for mixture optimization.
    
    Adds mixture-specific properties.
    """
    def __init__(self, *args, 
                 mixture_dcn=None, 
                 blend_ratio=None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_dcn = mixture_dcn  # DCN when blended
        self.blend_ratio = blend_ratio  # Optimal blend percentage
    
    def to_dict(self):
        """Extended dict with mixture properties."""
        d = super().to_dict()
        d['mixture_dcn'] = self.mixture_dcn
        d['blend_ratio'] = self.blend_ratio
        return d


class MixtureAwareMolecularEvolution:
    """
    Evolutionary algorithm for optimizing fuel blend additives.
    
    Instead of optimizing pure component CN, this optimizes:
    - Mixture DCN = f(additive + base_fuel)
    - Optionally: best blend ratio for each additive
    """
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    REP_DB_PATH = BASE_DIR / "data" / "fragments" / "diesel_fragments.db"

    def __init__(self, config: EvolutionConfig):
        self.config = config
        
        # Initialize predictors
        self.pure_predictor = PropertyPredictor(config)  # For pure component properties
        self.mixture_predictor = MixtureDCNPredictor()   # For mixture DCN predictions
        
        # Load base fuel composition
        self._load_base_fuel()
        
        # Initialize population
        self.population = Population(config)
        
        # Validation
        try:
            from ga_constraints import validate_molecule_for_ga, analyze_molecule_properties
            self.has_validation = True
            self.validate_molecule = validate_molecule_for_ga
            self.analyze_properties = analyze_molecule_properties
        except ImportError:
            self.has_validation = False
            print("⚠ Warning: ga_constraints.py not found - chemical validation disabled")
    
    def _load_base_fuel(self):
        """Load base fuel composition based on config."""
        if self.config.mixture_mode:
            mc = self.config.mixture_config
            
            if mc.base_fuel_smiles and mc.base_fuel_mole_fractions:
                # Custom base fuel
                self.base_smiles = mc.base_fuel_smiles
                self.base_fractions = mc.base_fuel_mole_fractions
                print(f"Using custom base fuel with {len(self.base_smiles)} components")
            else:
                # Load from library
                self.base_smiles, self.base_fractions = BaseFuelLibrary.get_base_fuel(mc.base_fuel_type)
                print(f"Using {mc.base_fuel_type} as base fuel")
                print(f"  Components: {len(self.base_smiles)}")
        else:
            # Not in mixture mode
            self.base_smiles = None
            self.base_fractions = None
    
    def _has_alkyne(self, smiles: str) -> bool:
        """Return True if molecule contains any triple bond (alkyne)."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return True
        return any(
            bond.GetBondType() == Chem.BondType.TRIPLE
            for bond in mol.GetBonds()
        )
    
    def _compute_confidence_metrics(self, smiles: str, predictions: Dict) -> Dict:
        """Compute confidence/quality metrics for a molecule."""
        metrics = {
            'chemical_valid': True,
            'chemical_flags': '',
            'ood_warning': False,
            'confidence_score': 100.0
        }
        
        # Chemical validation
        if self.has_validation:
            is_valid, flags = self.validate_molecule(smiles, strict=True)
            metrics['chemical_valid'] = is_valid
            metrics['chemical_flags'] = ', '.join(flags) if flags else 'OK'
            
            if not is_valid:
                metrics['confidence_score'] -= 30.0
                metrics['ood_warning'] = True
        
        # Analyze chemical properties
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            metrics['confidence_score'] = 0.0
            metrics['ood_warning'] = True
            return metrics
        
        # Get basic properties
        n_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
        n_oxygens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
        mw = Descriptors.MolWt(mol)
        
        # OOD checks (same as before)
        ood_flags = []
        
        if n_carbons < 8:
            ood_flags.append('SHORT_CHAIN')
            metrics['confidence_score'] -= 15.0
        elif n_carbons < 12:
            metrics['confidence_score'] -= 5.0
        
        if n_carbons > 25:
            ood_flags.append('LONG_CHAIN')
            metrics['confidence_score'] -= 10.0
        
        if mw < 100:
            ood_flags.append('LOW_MW')
            metrics['confidence_score'] -= 15.0
        elif mw > 500:
            ood_flags.append('HIGH_MW')
            metrics['confidence_score'] -= 10.0
        
        if n_oxygens > 3:
            ood_flags.append('HIGH_O')
            metrics['confidence_score'] -= 10.0
        
        # Check mixture DCN if available
        mixture_dcn = predictions.get('mixture_dcn')
        if mixture_dcn is not None:
            if mixture_dcn < 30 or mixture_dcn > 70:
                ood_flags.append('UNUSUAL_MIXTURE_DCN')
                metrics['confidence_score'] -= 10.0
        
        if ood_flags:
            metrics['ood_warning'] = True
            if not metrics['chemical_flags']:
                metrics['chemical_flags'] = ', '.join(ood_flags)
            else:
                metrics['chemical_flags'] += ', ' + ', '.join(ood_flags)
        
        metrics['confidence_score'] = max(0.0, min(100.0, metrics['confidence_score']))
        
        return metrics
    
    def _predict_mixture_dcn(self, additive_smiles: str, blend_ratio: float = None) -> float:
        """
        Predict DCN for mixture of additive + base fuel.
        
        Args:
            additive_smiles: SMILES of the additive molecule
            blend_ratio: Mole fraction of additive (if None, use config default)
        
        Returns:
            mixture_dcn: Predicted DCN of the blend
        """
        if blend_ratio is None:
            blend_ratio = self.config.mixture_config.additive_fraction
        
        # Adjust base fuel fractions
        base_ratio = 1.0 - blend_ratio
        adjusted_base_fractions = [f * base_ratio for f in self.base_fractions]
        
        # Create mixture
        mixture_smiles = [additive_smiles] + self.base_smiles
        mixture_fractions = [blend_ratio] + adjusted_base_fractions
        
        # Predict
        try:
            dcn = self.mixture_predictor.predict_mixture_dcn(mixture_smiles, mixture_fractions)
            return dcn
        except Exception as e:
            print(f"⚠ Warning: Mixture prediction failed for {additive_smiles}: {e}")
            return None
    
    def _optimize_blend_ratio(self, additive_smiles: str) -> Tuple[float, float]:
        """
        Find optimal blend ratio for an additive.
        
        Args:
            additive_smiles: SMILES of the additive
        
        Returns:
            (optimal_ratio, optimal_dcn)
        """
        mc = self.config.mixture_config
        
        # Test different ratios
        test_ratios = np.linspace(
            mc.min_additive_fraction,
            mc.max_additive_fraction,
            num=10  # Test 10 different ratios
        )
        
        best_ratio = mc.additive_fraction
        best_dcn = None
        
        for ratio in test_ratios:
            dcn = self._predict_mixture_dcn(additive_smiles, ratio)
            if dcn is not None:
                if best_dcn is None or abs(dcn - mc.target_mixture_dcn) < abs(best_dcn - mc.target_mixture_dcn):
                    best_dcn = dcn
                    best_ratio = ratio
        
        return best_ratio, best_dcn
    
    def _mutate_molecule(self, mol: Chem.Mol) -> List[str]:
        """Generate mutations for a molecule using CREM."""
        try:
            mutants = list(mutate_mol(
                mol,
                db_name=str(self.REP_DB_PATH),
                max_size=2,
                return_mol=False
            ))
            return [m for m in mutants if m and m not in self.population.seen_smiles]
        except Exception:
            return []
    
    def _create_molecules(self, smiles_list: List[str]) -> List[MixtureAwareMolecule]:
        """
        Create Molecule objects with mixture DCN predictions.
        
        Key difference: Evaluates molecules in a BLEND, not pure.
        """
        if not smiles_list:
            return []

        # Reject alkynes early
        smiles_list = [s for s in smiles_list if not self._has_alkyne(s)]
        if not smiles_list:
            return []

        # Get pure component properties (for filtering)
        pure_predictions = self.pure_predictor.predict_all_properties(smiles_list)
        
        # Get mixture DCN predictions
        if self.config.mixture_mode:
            mc = self.config.mixture_config
            
            if mc.optimize_blend_ratio:
                # Optimize ratio for each molecule
                mixture_dcns = []
                blend_ratios = []
                for smiles in smiles_list:
                    ratio, dcn = self._optimize_blend_ratio(smiles)
                    mixture_dcns.append(dcn)
                    blend_ratios.append(ratio)
            else:
                # Fixed ratio for all
                mixture_dcns = []
                for smiles in smiles_list:
                    dcn = self._predict_mixture_dcn(smiles)
                    mixture_dcns.append(dcn)
                blend_ratios = [mc.additive_fraction] * len(smiles_list)
        else:
            # Pure component mode (no mixture)
            mixture_dcns = [None] * len(smiles_list)
            blend_ratios = [None] * len(smiles_list)

        molecules = []
        for i, smiles in enumerate(smiles_list):
            pure_props = {k: v[i] for k, v in pure_predictions.items()}
            mixture_dcn = mixture_dcns[i]
            blend_ratio = blend_ratios[i]
            
            # Skip if mixture prediction failed
            if self.config.mixture_mode and mixture_dcn is None:
                continue
            
            # Use mixture DCN for fitness, pure CN for reporting
            if self.config.mixture_mode:
                cn_for_fitness = mixture_dcn
            else:
                cn_for_fitness = pure_props.get('cn')
            
            if cn_for_fitness is None:
                continue
            
            # Check pure component constraints (filters)
            if not all(self.pure_predictor.is_valid(k, pure_props.get(k))
                      for k in ['bp', 'density', 'lhv', 'dynamic_viscosity']):
                continue
            
            # Compute confidence
            predictions_for_confidence = pure_props.copy()
            predictions_for_confidence['mixture_dcn'] = mixture_dcn
            confidence = self._compute_confidence_metrics(smiles, predictions_for_confidence)
            
            # Skip low confidence
            if confidence['confidence_score'] < 30.0:
                continue

            # Create molecule
            target = self.config.mixture_config.target_mixture_dcn if self.config.mixture_mode else self.config.target_cn
            
            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=cn_for_fitness,  # This is mixture DCN in mixture mode!
                cn_error=abs(cn_for_fitness - target),
                cn_score=cn_for_fitness,
                bp=pure_props.get('bp'),
                ysi=pure_props.get('ysi'),
                density=pure_props.get('density'),
                lhv=pure_props.get('lhv'),
                dynamic_viscosity=pure_props.get('dynamic_viscosity'),
                # Confidence metrics
                chemical_valid=confidence['chemical_valid'],
                chemical_flags=confidence['chemical_flags'],
                ood_warning=confidence['ood_warning'],
                confidence_score=confidence['confidence_score'],
                # Mixture-specific
                mixture_dcn=mixture_dcn,
                blend_ratio=blend_ratio
            ))

        return molecules
    
    def initialize_population(self, initial_smiles: List[str]) -> int:
        """Initialize the population from initial SMILES."""
        if self.config.mixture_mode:
            print("Predicting mixture properties for initial population...")
        else:
            print("Predicting properties for initial population...")
        
        molecules = self._create_molecules(initial_smiles)
        return self.population.add_molecules(molecules)
    
    def _log_generation_stats(self, generation: int):
        """Log statistics for the current generation."""
        mols = self.population.molecules
        
        # Calculate confidence stats
        avg_confidence = np.mean([m.confidence_score for m in mols])
        n_ood = sum(1 for m in mols if m.ood_warning)
        n_invalid = sum(1 for m in mols if not m.chemical_valid)
        
        # In mixture mode, cn is actually mixture DCN
        metric_name = "Mixture DCN" if self.config.mixture_mode else "CN"
        
        if self.config.maximize_cn:
            best = max(mols, key=lambda m: m.cn)
            avg = np.mean([m.cn for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best {metric_name}: {best.cn:.3f} | "
                        f"Avg {metric_name}: {avg:.3f}")
        else:
            best = min(mols, key=lambda m: m.cn_error)
            avg_err = np.mean([m.cn_error for m in mols])
            
            print_msg = (f"Gen {generation}/{self.config.generations} | "
                        f"Pop {len(mols)} | "
                        f"Best {metric_name} err: {best.cn_error:.3f} | "
                        f"Avg {metric_name} err: {avg_err:.3f}")
        
        # Add confidence stats
        print_msg += (f" | Conf: {avg_confidence:.1f}% | "
                     f"OOD: {n_ood} | "
                     f"Invalid: {n_invalid}")
        
        # Add blend info if optimizing ratio
        if self.config.mixture_mode and self.config.mixture_config.optimize_blend_ratio:
            avg_ratio = np.mean([m.blend_ratio for m in mols if m.blend_ratio is not None])
            print_msg += f" | Avg Blend: {avg_ratio*100:.1f}%"
        
        print(print_msg)
    
    def _generate_offspring(self, survivors: List[MixtureAwareMolecule]) -> List[MixtureAwareMolecule]:
        """Generate offspring from survivors."""
        target_count = self.config.population_size - len(survivors)
        max_attempts = target_count * self.config.max_offspring_attempts
        
        all_children = []
        new_molecules = []
        
        print(f"  → Generating offspring (target: {target_count})...")
        
        for attempt in range(max_attempts):
            if len(new_molecules) >= target_count:
                break
            
            parent = random.choice(survivors)
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                continue
            
            children = self._mutate_molecule(mol)
            all_children.extend(children[:self.config.mutations_per_parent])
            
            if len(all_children) >= self.config.batch_size:
                print(f"  → Evaluating batch of {len(all_children)}...")
                new_molecules.extend(self._create_molecules(all_children))
                all_children = []
        
        if all_children:
            print(f"  → Evaluating final batch of {len(all_children)}...")
            new_molecules.extend(self._create_molecules(all_children))
        
        high_conf = sum(1 for m in new_molecules if m.confidence_score >= 70)
        ood_count = sum(1 for m in new_molecules if m.ood_warning)
        
        print(f"  ✓ Generated {len(new_molecules)} valid offspring "
              f"({high_conf} high-confidence, {ood_count} OOD warnings)")
        return new_molecules
    
    def _run_evolution_loop(self):
        """Run the main evolution loop."""
        for gen in range(1, self.config.generations + 1):
            self._log_generation_stats(gen)
            
            survivors = self.population.get_survivors()
            offspring = self._generate_offspring(survivors)
            
            new_pop = Population(self.config)
            new_pop.add_molecules(survivors + offspring)
            self.population = new_pop
    
    def _generate_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate final results DataFrames."""
        final_df = self.population.to_dataframe()

        # Filter and sort
        if self.config.mixture_mode:
            # In mixture mode, filter by mixture DCN error
            target = self.config.mixture_config.target_mixture_dcn
            final_df = final_df[final_df["cn_error"] < 5].sort_values("cn_error", ascending=True)
        else:
            # Pure component mode (original logic)
            if self.config.maximize_cn:
                final_df = final_df[final_df["cn"] > 50].sort_values("cn", ascending=False)
            else:
                final_df = final_df[final_df["cn_error"] < 5].sort_values("cn_error", ascending=True)
        
        final_df["rank"] = range(1, len(final_df) + 1)
        
        return final_df, pd.DataFrame()  # No pareto for mixture mode
    
    def print_final_summary(self, final_df: pd.DataFrame):
        """Print summary with mixture info."""
        print("\n" + "="*70)
        print("EVOLUTION COMPLETE")
        print("="*70)
        
        if len(final_df) == 0:
            print("No molecules met the filtering criteria")
            return
        
        mode = "MIXTURE" if self.config.mixture_mode else "PURE COMPONENT"
        print(f"\nMode: {mode}")
        print(f"Total candidates: {len(final_df)}")
        
        if self.config.mixture_mode:
            mc = self.config.mixture_config
            print(f"Target mixture DCN: {mc.target_mixture_dcn}")
            print(f"Base fuel: {mc.base_fuel_type}")
            print(f"Additive fraction: {mc.additive_fraction * 100:.1f}%")
        
        print("\n" + "="*70)
    
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
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"✓ Initial population size: {init_count}\n")
        
        # Evolution
        self._run_evolution_loop()
        
        # Results
        final_df, pareto_df = self._generate_results()
        
        # Print summary
        self.print_final_summary(final_df)
        
        return final_df, pareto_df