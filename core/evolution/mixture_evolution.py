from .evolution import MolecularEvolution
from .molecule import Molecule
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.base_fuel_library import BaseFuelLibrary
from core.config import EvolutionConfig
from typing import List, Tuple, Dict
import numpy as np
import wandb
import pickle
import torch


class MixtureAwareMolecule(Molecule):
    """Extended Molecule class for mixture optimization with AD info."""
    def __init__(self, *args, mixture_dcn=None, blend_ratio=None, 
                 ad_score=None, in_domain=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_dcn = mixture_dcn
        self.blend_ratio = blend_ratio
        self.ad_score = ad_score      # NEW: AD decision score
        self.in_domain = in_domain    # NEW: Is in domain?

    def to_dict(self):
        d = super().to_dict()
        d['mixture_dcn'] = self.mixture_dcn
        d['blend_ratio'] = self.blend_ratio
        d['ad_score'] = self.ad_score
        d['in_domain'] = self.in_domain
        return d


class MixtureAwareMolecularEvolution(MolecularEvolution):
    """
    Mixture-aware evolution with Applicability Domain filtering.
    """
    
    def __init__(self, config: EvolutionConfig, use_ad_filtering: bool = True):
        """
        Args:
            config: Evolution configuration
            use_ad_filtering: Enable AD filtering (default: True)
        """
        from .population import Population
        from core.predictors.pure_component.property_predictor import PropertyPredictor
        
        self.config = config
        self.predictor = PropertyPredictor(config)
        self.population = Population(config)
        self.uncertainty_filters = {}
        
        print("Initializing mixture DCN predictor...")
        self.mixture_predictor = MixtureDCNPredictor()
        self.mixture_predictor._initialize_models()  # Load models now
        self._load_base_fuel()
        
        # NEW: Load AD checker
        self.use_ad_filtering = use_ad_filtering
        if use_ad_filtering:
            self._load_ad_checker()
        
        wandb.init(
            project="mixture-dcn-evolution",
            name=f"run_target_{config.mixture_config.target_mixture_dcn}",
            config={
                "population_size": config.population_size,
                "generations": config.generations,
                "additive_fraction": config.mixture_config.additive_fraction,
                "target_mixture_dcn": config.mixture_config.target_mixture_dcn,
                "maximize_cn": config.maximize_cn,
                "use_ad_filtering": use_ad_filtering
            }
        )

    def _load_base_fuel(self):
        """Load base fuel composition."""
        mc = self.config.mixture_config
        if mc.base_fuel_smiles and mc.base_fuel_mole_fractions:
            self.base_smiles = mc.base_fuel_smiles
            self.base_fractions = mc.base_fuel_mole_fractions
        else:
            self.base_smiles, self.base_fractions = BaseFuelLibrary.get_base_fuel(mc.base_fuel_type)
        
        print(f"✓ Base fuel: {len(self.base_smiles)} components")
    
    def _load_ad_checker(self):
        """NEW: Load the trained One-Class SVM."""
        try:
            with open('mixture_ad_svm.pkl', 'rb') as f:
                ad_data = pickle.load(f)
                self.svm = ad_data['svm']
                self.scaler = ad_data['scaler']
            
            print("✓ AD checker loaded")
        
        except FileNotFoundError:
            print("⚠ mixture_ad_svm.pkl not found - disabling AD filtering")
            self.use_ad_filtering = False
    
    def _extract_mixture_embedding(self, additive_smiles: str) -> np.ndarray:
        """
        NEW: Extract mixture embedding for AD checking.
        
        Args:
            additive_smiles: SMILES of new additive
        
        Returns:
            embedding: (202,) array or None
        """
        from core.predictors.mixture.solvation_predictor.data.data import (
            DataPoint, DatapointList, MolencoderDatabase, DataTensor
        )
        
        mc = self.config.mixture_config
        
        # Create mixture
        base_ratio = 1.0 - mc.additive_fraction
        adjusted_base = [f * base_ratio for f in self.base_fractions]
        
        mixture_smiles = [additive_smiles] + self.base_smiles
        mixture_fractions = [mc.additive_fraction] + adjusted_base[:-1]  # N-1
        
        # Hook to capture embedding
        model = self.mixture_predictor.models[0]
        captured = []
        
        def hook_fn(module, input, output):
            captured.append(input[0].detach().cpu())
        
        hook = model.ffn.register_forward_hook(hook_fn)
        
        try:
            # Create datapoint
            mol_db = MolencoderDatabase()
            dp = DataPoint(
                smiles=mixture_smiles,
                targets=[0.0],
                features=[],
                molefracs=mixture_fractions,
                inp=self.mixture_predictor.args,
                mol_encoders=mol_db
            )
            
            data = DatapointList([dp])
            
            # Create tensors
            mol_encodings = []
            tensors = []
            for _ in range(self.mixture_predictor.args.max_num_mols):
                mol_encodings.append([])
            
            for mol in mol_encodings:
                encoders = dp.get_mol_encoder()
                if len(encoders) < self.mixture_predictor.args.max_num_mols:
                    for _ in range(self.mixture_predictor.args.max_num_mols - len(encoders)):
                        encoders.append(encoders[0])
                mol.append(encoders[mol_encodings.index(mol)])
                tensors.append(DataTensor(mol, self.mixture_predictor.args, 
                                         property=self.mixture_predictor.args.property))
            
            # Forward pass
            with torch.no_grad():
                _ = model(data, tensors)
            
            if captured:
                return captured[0].numpy().flatten()
            return None
        
        except:
            return None
        
        finally:
            hook.remove()
    
    def _check_ad_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        NEW: Check AD for batch of molecules.
        
        Returns:
            ad_scores: Decision function scores
            in_domain: Boolean array
        """
        if not self.use_ad_filtering:
            return np.zeros(len(smiles_list)), np.ones(len(smiles_list), dtype=bool)
        
        embeddings = []
        valid_idx = []
        
        for i, smiles in enumerate(smiles_list):
            emb = self._extract_mixture_embedding(smiles)
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(i)
        
        if not embeddings:
            return np.full(len(smiles_list), -999), np.zeros(len(smiles_list), dtype=bool)
        
        # Scale and check
        embeddings = np.array(embeddings)
        embeddings_scaled = self.scaler.transform(embeddings)
        
        scores_valid = self.svm.decision_function(embeddings_scaled)
        domain_valid = self.svm.predict(embeddings_scaled) == 1
        
        # Map back
        scores = np.full(len(smiles_list), -999.0)
        domain = np.zeros(len(smiles_list), dtype=bool)
        
        for i, idx in enumerate(valid_idx):
            scores[idx] = scores_valid[i]
            domain[idx] = domain_valid[i]
        
        return scores, domain

    def _create_molecules(self, smiles_list: List[str]) -> Tuple[List[MixtureAwareMolecule], Dict]:
        """Create molecules with AD filtering."""
        if not smiles_list:
            return [], {'total': 0, 'passed': 0, 'ad_filtered': 0}

        mc = self.config.mixture_config
        
        filter_stats = {
            'total': len(smiles_list),
            'cn_none': 0,
            'ysi_none': 0,
            'tanimoto_fail': 0,
            'cn_uncertainty_fail': 0,
            'ysi_uncertainty_fail': 0,
            'property_fail': 0,
            'ad_filtered': 0,      # Your new AD filtering
            'passed': 0
            }
        
        # STEP 1: Check AD (if enabled)
        ad_scores, in_domain = self._check_ad_batch(smiles_list)
        
        # STEP 2: Get DCN predictions (batched)
        mixture_dcns = self.mixture_predictor.predict_batch_mixtures(
            additive_smiles_list=smiles_list,
            base_smiles=self.base_smiles,
            base_mole_fractions=self.base_fractions,
            additive_fraction=mc.additive_fraction,
            verbose=False
        )

        # STEP 3: Create molecules with both DCN and AD
        molecules = []
        
        for i, smiles in enumerate(smiles_list):
            dcn = mixture_dcns[i]
            
            # Filter: No DCN
            if dcn is None:
                filter_stats['cn_none'] += 1
                continue
            
            # Filter: Outside AD
            if not in_domain[i]:
                filter_stats['ad_filtered'] += 1
                continue
            
            filter_stats['passed'] += 1
            
            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=dcn,
                cn_error=abs(dcn - mc.target_mixture_dcn),
                cn_score=dcn,
                mixture_dcn=dcn,
                blend_ratio=mc.additive_fraction,
                ad_score=float(ad_scores[i]),   # NEW
                in_domain=bool(in_domain[i])    # NEW
            ))

        return molecules, filter_stats

    def _log_generation_stats(self, generation: int):
        """Log stats with AD metrics."""
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
        
        # NEW: AD stats
        if self.use_ad_filtering and mols:
            avg_ad = np.mean([m.ad_score for m in mols if m.ad_score is not None])
            n_in_domain = sum(1 for m in mols if m.in_domain)
        else:
            avg_ad = 0.0
            n_in_domain = len(mols)

        # Console
        if self.use_ad_filtering:
            print(
                f"Gen {generation}/{self.config.generations} | "
                f"Pop {len(mols)} | "
                f"Best: {best_metric:.3f} | "
                f"AD: {avg_ad:.3f}"
            )
        else:
            print(
                f"Gen {generation}/{self.config.generations} | "
                f"Pop {len(mols)} | "
                f"Best: {best_metric:.3f} | "
                f"Invalid: {n_invalid}"
            )

        # W&B scalars
        log_dict = {
            "generation": generation,
            "population_size": len(mols),
            "best_mixture_dcn" if self.config.maximize_cn else "best_dcn_error": best_metric,
            "avg_mixture_dcn" if self.config.maximize_cn else "avg_dcn_error": avg_metric,
            "invalid_fraction": n_invalid / len(mols) if mols else 0,
            "avg_blend_ratio": avg_ratio,
        }
        
        if self.use_ad_filtering:
            log_dict["avg_ad_score"] = avg_ad
            log_dict["in_domain_fraction"] = n_in_domain / len(mols) if mols else 0
        
        wandb.log(log_dict)

        # W&B table every 10 gens
        if generation % 10 == 0:
            if self.use_ad_filtering:
                table_data = [
                    [m.smiles, m.mixture_dcn, m.cn_error, m.ad_score, m.in_domain]
                    for m in sorted(mols, key=lambda x: x.cn_error)[:50]
                ]
                columns = ["SMILES", "DCN", "Error", "AD Score", "In Domain"]
            else:
                table_data = [
                    [m.smiles, m.mixture_dcn, m.cn_error, m.blend_ratio]
                    for m in sorted(mols, key=lambda x: x.cn_error)[:50]
                ]
                columns = ["SMILES", "DCN", "Error", "Blend Ratio"]
            
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({f"top_molecules_gen_{generation}": table})