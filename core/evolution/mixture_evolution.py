from .evolution import MolecularEvolution
from .molecule import Molecule
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor, BaseFuelLibrary
from core.config import EvolutionConfig
from rdkit import Chem
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path

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


        # Get pure component properties
        pure_predictions = self.predictor.predict_all_properties(smiles_list)

        mc = self.config.mixture_config
        mixture_dcns, blend_ratios = [], []

        for smiles in smiles_list:
            ratio, dcn = mc.additive_fraction, self._predict_mixture_dcn(smiles)
            mixture_dcns.append(dcn)
            blend_ratios.append(ratio)

        molecules = []
        for i, smiles in enumerate(smiles_list):
            props = {k: v[i] for k, v in pure_predictions.items()}
            mixture_dcn, blend_ratio = mixture_dcns[i], blend_ratios[i]
            if mixture_dcn is None:
                continue

            cn_for_fitness = mixture_dcn


            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=cn_for_fitness,
                cn_error=abs(cn_for_fitness - mc.target_mixture_dcn),
                cn_score=cn_for_fitness,
                bp=props.get('bp'),
                ysi=props.get('ysi'),
                density=props.get('density'),
                lhv=props.get('lhv'),
                dynamic_viscosity=props.get('dynamic_viscosity'),
                mixture_dcn=mixture_dcn,
                blend_ratio=blend_ratio
            ))

        return molecules

    def _log_generation_stats(self, generation: int):
        """Log stats for mixture mode."""
        mols = self.population.molecules
        avg_confidence = np.mean([m.confidence_score for m in mols])
        n_ood = sum(1 for m in mols if m.ood_warning)
        n_invalid = sum(1 for m in mols if not m.chemical_valid)

        best = max(mols, key=lambda m: m.cn) if self.config.maximize_cn else min(mols, key=lambda m: m.cn_error)
        avg_cn = np.mean([m.cn for m in mols]) if self.config.maximize_cn else np.mean([m.cn_error for m in mols])

        avg_ratio = np.mean([m.blend_ratio for m in mols if m.blend_ratio is not None])
        print(f"Gen {generation}/{self.config.generations} | Pop {len(mols)} | "
              f"{'Best Mixture DCN' if self.config.maximize_cn else 'Best DCN err'}: {best.cn if self.config.maximize_cn else best.cn_error:.3f} | "
              f"Avg: {avg_cn:.3f} | Conf: {avg_confidence:.1f}% | OOD: {n_ood} | Invalid: {n_invalid} | "
              f"Avg Blend: {avg_ratio*100:.1f}%")
