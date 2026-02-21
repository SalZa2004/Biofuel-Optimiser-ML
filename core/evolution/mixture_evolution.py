from .evolution import MolecularEvolution
from .molecule import Molecule
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.base_fuel_library import BaseFuelLibrary
from core.config import EvolutionConfig
from typing import List, Tuple, Dict
import numpy as np
import wandb


class MixtureAwareMolecule(Molecule):
    """Extended Molecule class for mixture optimization."""
    def __init__(self, *args, mixture_dcn=None, blend_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_dcn = mixture_dcn
        self.blend_ratio = blend_ratio

    def to_dict(self):
        d = super().to_dict()
        d['mixture_dcn'] = self.mixture_dcn
        d['blend_ratio'] = self.blend_ratio
        return d


class MixtureAwareMolecularEvolution(MolecularEvolution):
    """
    Mixture-aware evolution.
    
    Overrides __init__ to SKIP pure-component uncertainty calibration
    since fitness is driven by mixture DCN, not pure component properties.
    """
    
    def __init__(self, config: EvolutionConfig):
        # FIX 1: Call grandparent __init__ directly to skip _setup_uncertainty_filters()
        # MolecularEvolution.__init__ now runs calibration we don't need here
        from .population import Population
        from core.predictors.pure_component.property_predictor import PropertyPredictor
        
        self.config = config
        self.predictor = PropertyPredictor(config)
        self.population = Population(config)
        self.uncertainty_filters = {}  # Empty - not used in mixture mode
        
        # FIX 2: Use direct predictor (no CSV files!)
        print("Initializing mixture DCN predictor...")
        self.mixture_predictor = MixtureDCNPredictor()
        self._load_base_fuel()
        
        wandb.init(
            project="mixture-dcn-evolution",
            name=f"run_target_{config.mixture_config.target_mixture_dcn}",
            config={
                "population_size": config.population_size,
                "generations": config.generations,
                "additive_fraction": config.mixture_config.additive_fraction,
                "target_mixture_dcn": config.mixture_config.target_mixture_dcn,
                "maximize_cn": config.maximize_cn
            }
        )

    def _load_base_fuel(self):
        """Load base fuel composition based on config."""
        mc = self.config.mixture_config
        if mc.base_fuel_smiles and mc.base_fuel_mole_fractions:
            self.base_smiles = mc.base_fuel_smiles
            self.base_fractions = mc.base_fuel_mole_fractions
        else:
            self.base_smiles, self.base_fractions = BaseFuelLibrary.get_base_fuel(mc.base_fuel_type)
        
        print(f"✓ Base fuel: {len(self.base_smiles)} components")

    def _create_molecules(self, smiles_list: List[str]) -> Tuple[List[MixtureAwareMolecule], Dict]:
        """Create mixture-aware molecules with batch prediction."""
        if not smiles_list:
            return [], {'total': 0, 'passed': 0}

        mc = self.config.mixture_config

        # FIX 2: Use direct batch prediction (no CSV files!)
        mixture_dcns = self.mixture_predictor.predict_batch_mixtures(
            additive_smiles_list=smiles_list,
            base_smiles=self.base_smiles,
            base_mole_fractions=self.base_fractions,
            additive_fraction=mc.additive_fraction,
            verbose=False
        )

        molecules = []
        for i, smiles in enumerate(smiles_list):
            mixture_dcn = mixture_dcns[i]
            if mixture_dcn is None:
                continue

            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=mixture_dcn,
                cn_error=abs(mixture_dcn - mc.target_mixture_dcn),
                cn_score=mixture_dcn,
                mixture_dcn=mixture_dcn,
                blend_ratio=mc.additive_fraction
            ))

        filter_stats = {
            'total': len(smiles_list),
            'cn_none': 0,
            'ysi_none': 0,
            'tanimoto_fail': 0,
            'cn_uncertainty_fail': 0,
            'ysi_uncertainty_fail': 0,
            'property_fail': 0,
            'passed': len(molecules)
        }

        return molecules, filter_stats

    def _log_generation_stats(self, generation: int):
        """Log stats with W&B table only every 10 generations."""
        mols = self.population.molecules

        n_invalid = sum(1 for m in mols if not m.chemical_valid)

        if self.config.maximize_cn:
            best = max(mols, key=lambda m: m.cn)
            avg_metric = np.mean([m.cn for m in mols])
            best_metric = best.cn
        else:
            best = min(mols, key=lambda m: m.cn_error)
            avg_metric = np.mean([m.cn_error for m in mols])
            best_metric = best.cn_error

        avg_ratio = np.mean([m.blend_ratio for m in mols if m.blend_ratio is not None])

        # Console logging
        print(
            f"Gen {generation}/{self.config.generations} | "
            f"Pop {len(mols)} | "
            f"Best: {best_metric:.3f} | "
            f"Avg: {avg_metric:.3f} | "
            f"Invalid: {n_invalid} | "
            f"Blend: {avg_ratio*100:.1f}%"
        )

        # W&B scalar logging (every generation - fast)
        wandb.log({
            "generation": generation,
            "population_size": len(mols),
            "best_mixture_dcn" if self.config.maximize_cn else "best_dcn_error": best_metric,
            "avg_mixture_dcn" if self.config.maximize_cn else "avg_dcn_error": avg_metric,
            "invalid_fraction": n_invalid / len(mols) if mols else 0,
            "avg_blend_ratio": avg_ratio,
        })

        # FIX 3: W&B table only every 10 generations (slow - avoid every gen!)
        if generation % 10 == 0:
            table_data = [
                [m.smiles, m.mixture_dcn, m.cn_error, m.blend_ratio, m.chemical_valid]
                for m in sorted(mols, key=lambda x: x.cn_error)[:50]  # Top 50 only
            ]
            table = wandb.Table(
                data=table_data,
                columns=["SMILES", "Mixture DCN", "CN Error", "Blend Ratio", "Valid"]
            )
            wandb.log({f"top_molecules_gen_{generation}": table})