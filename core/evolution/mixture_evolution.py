from core import config
from .evolution import MolecularEvolution
from .molecule import Molecule
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.base_fuel_library import BaseFuelLibrary
from core.config import EvolutionConfig
from rdkit import Chem
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
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
    Subclass of MolecularEvolution to handle mixture optimization.
    Only overrides methods that differ from pure-component evolution.
    """
    
    def __init__(self, config: EvolutionConfig):
        super().__init__(config)
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

    def _predict_mixture_dcn(self, additive_smiles: str, blend_ratio: float = None) -> float:
        """Predict DCN for mixture of additive + base fuel."""
        mc = self.config.mixture_config
        blend_ratio = blend_ratio if blend_ratio is not None else mc.additive_fraction
        base_ratio = 1.0 - blend_ratio
        adjusted_base = [f * base_ratio for f in self.base_fractions]

        try:
            return self.mixture_predictor.predict_mixture_dcn([additive_smiles] + self.base_smiles,
                                                              [blend_ratio] + adjusted_base)
        except Exception as e:
            print(f"⚠ Mixture prediction failed for {additive_smiles}: {e}")
            return None
    
    def _create_molecules(self, smiles_list: List[str]) -> List[MixtureAwareMolecule]:
        """Create mixture-aware molecules."""
        if not smiles_list:
            return []


        mc = self.config.mixture_config

        # Mixture DCN predictions (fitness driver)
        mixture_dcns = self.mixture_predictor.predict_batch_mixtures(
            additive_smiles_list=smiles_list,
            base_smiles=self.base_smiles,
            base_mole_fractions=self.base_fractions,
            additive_fraction=mc.additive_fraction
        )

        molecules = []
        for i, smiles in enumerate(smiles_list):
            mixture_dcn = mixture_dcns[i]
            if mixture_dcn is None:
                continue


            cn_for_fitness = mixture_dcn

            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=cn_for_fitness,
                cn_error=abs(cn_for_fitness - mc.target_mixture_dcn),
                cn_score=cn_for_fitness,
                mixture_dcn=mixture_dcn,
                blend_ratio=mc.additive_fraction
            ))

        return molecules




    def _log_generation_stats(self, generation: int):
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

        # 🔹 Console logging (keep this!)
        print(
            f"Gen {generation}/{self.config.generations} | "
            f"Pop {len(mols)} | "
            f"Best: {best_metric:.3f} | "
            f"Avg: {avg_metric:.3f} | "
            f"Invalid: {n_invalid} | "
            f"Avg Blend: {avg_ratio*100:.1f}%"
        )

        # 🔹 W&B logging
        wandb.log({
            "generation": generation,
            "population_size": len(mols),
            "best_mixture_dcn" if self.config.maximize_cn else "best_dcn_error": best_metric,
            "avg_mixture_dcn" if self.config.maximize_cn else "avg_dcn_error": avg_metric,
            "invalid_fraction": n_invalid / len(mols),
            "avg_blend_ratio": avg_ratio,
        })
        table_data = [[m.smiles, m.mixture_dcn, m.cn_error, m.blend_ratio, m.chemical_valid] for m in mols]
        table = wandb.Table(data=table_data, columns=["SMILES", "Mixture DCN", "CN Error", "Blend Ratio", "Valid"])
        wandb.log({"molecules": table})
