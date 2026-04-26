from .evolution import MolecularEvolution
from .molecule import Molecule
from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.base_fuel_library import BaseFuelLibrary
from core.config import EvolutionConfig
from typing import List, Tuple, Dict, Optional
import numpy as np
import wandb
import pickle
import torch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

class MixtureAwareMolecule(Molecule):
    """Extended Molecule class for mixture optimization with AD info."""
    def __init__(self, *args, mixture_dcn=None, blend_ratio=None,
                 ad_score=None, in_domain=None, mixture_ysi=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixture_dcn = mixture_dcn
        self.blend_ratio = blend_ratio
        self.ad_score = ad_score
        self.in_domain = in_domain
        self.mixture_ysi = mixture_ysi

    def to_dict(self):
        d = super().to_dict()
        d['mixture_dcn'] = self.mixture_dcn
        d['blend_ratio'] = self.blend_ratio
        d['ad_score'] = self.ad_score
        d['in_domain'] = self.in_domain
        d['mixture_ysi'] = self.mixture_ysi
        return d


class MixturePredictionCache:
    """Cache DCN predictions for additive+base mixtures."""
    
    def __init__(self, cache_file: str = "cache/mixture_dcn_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load()
    
    def _load(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"  ✓ Loaded {len(cache)} cached DCN predictions")
                return cache
            except:
                return {}
        return {}
    
    def _save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get(self, additive_smiles: str, base_fuel_type: str, 
            additive_fraction: float) -> float:
        """Get cached DCN prediction."""
        key = f"{additive_smiles}|{base_fuel_type}|{additive_fraction:.3f}"
        return self.cache.get(key)
    
    def set(self, additive_smiles: str, base_fuel_type: str, 
            additive_fraction: float, dcn: float):
        """Cache DCN prediction."""
        key = f"{additive_smiles}|{base_fuel_type}|{additive_fraction:.3f}"
        self.cache[key] = dcn
        self._save()
    
    def get_batch(self, additive_smiles_list: list, base_fuel_type: str,
                  additive_fraction: float) -> dict:
        """Get cached DCN for multiple additives."""
        cached = {}
        for smiles in additive_smiles_list:
            dcn = self.get(smiles, base_fuel_type, additive_fraction)
            if dcn is not None:
                cached[smiles] = dcn
        return cached
    
    def set_batch(self, predictions: dict, base_fuel_type: str,
                  additive_fraction: float):
        """Cache multiple DCN predictions."""
        for smiles, dcn in predictions.items():
            self.set(smiles, base_fuel_type, additive_fraction, dcn)

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
        self.dcn_cache = MixturePredictionCache()
        
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
            with open('models/mixture/mixture_ad_svm.pkl', 'rb') as f:
                ad_data = pickle.load(f)
                self.svm = ad_data['svm']
                self.scaler = ad_data['scaler']
            
            print("✓ AD checker loaded")
        
        except FileNotFoundError:
            print("⚠ models/mixture/mixture_ad_svm.pkl not found - disabling AD filtering")
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

    def predict_mixture_ysi(self, additive_smiles: str) -> Optional[float]:
        """
        Predict mixture YSI using the linear blending law (mass-fraction weighted):
            ysi_mix = sum(ysi_i * mass_frac_i)

        The additive_fraction in config is treated as a mole fraction; it is
        converted to a mass fraction together with the base fuel components
        using RDKit exact molecular weights.

        Args:
            additive_smiles: SMILES of the additive molecule.

        Returns:
            ysi_mix: Predicted mixture YSI, or None if any step fails.
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mc = self.config.mixture_config

        # Build full mole-fraction list: additive + base components
        base_ratio = 1.0 - mc.additive_fraction
        all_smiles = [additive_smiles] + self.base_smiles
        all_mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]

        # Compute molecular weights
        mol_weights = []
        for smi in all_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
            mol_weights.append(Descriptors.ExactMolWt(mol))

        # Convert mole fractions -> mass fractions
        masses = [x * mw for x, mw in zip(all_mole_fracs, mol_weights)]
        total_mass = sum(masses)
        if total_mass == 0:
            return None
        mass_fracs = [m / total_mass for m in masses]

        # Ensure YSI predictor is loaded
        if 'ysi' not in self.predictor.predictors:
            from core.predictors.pure_component.generic import GenericPredictor
            from core.predictors.pure_component.hf_models import load_models
            paths = load_models()
            self.predictor.predictors['ysi'] = GenericPredictor(paths['ysi'], 'YSI')

        # Predict pure-component YSI for all components
        props = self.predictor.predict_all_properties(all_smiles)
        ysi_values = props.get('ysi', [])

        if len(ysi_values) != len(all_smiles):
            return None

        # Apply blending law: ysi_mix = sum(ysi_i * mass_frac_i)
        ysi_mix = 0.0
        for ysi_i, mf_i in zip(ysi_values, mass_fracs):
            if ysi_i is None:
                return None
            ysi_mix += ysi_i * mf_i

        return ysi_mix

    def predict_mixture_ysi_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predict mixture YSI for a list of additives."""
        return [self.predict_mixture_ysi(smi) for smi in smiles_list]

    def _create_molecules(self, smiles_list: List[str]) -> Tuple[List[MixtureAwareMolecule], Dict]:

        if not smiles_list:
            return [], {
                'total': 0,
                'cn_none': 0,
                'ysi_none': 0,
                'tanimoto_fail': 0,
                'cn_uncertainty_fail': 0,
                'ysi_uncertainty_fail': 0,
                'property_fail': 0,
                'ad_filtered': 0,
                'passed': 0
            }

        mc = self.config.mixture_config
        
        filter_stats = {
            'total': len(smiles_list),
            'cn_none': 0,
            'ysi_none': 0,
            'tanimoto_fail': 0,
            'cn_uncertainty_fail': 0,
            'ysi_uncertainty_fail': 0,
            'property_fail': 0,
            'ad_filtered': 0,
            'passed': 0
        }
        
        # STEP 1: Check AD for all molecules (returns numpy arrays)
        ad_scores_array, in_domain_array = self._check_ad_batch(smiles_list)

        # DEBUG: Print AD stats
        n_in_domain = in_domain_array.sum()
        n_out_domain = (~in_domain_array).sum()
        print(f"  → AD Check: {n_in_domain} in-domain, {n_out_domain} out-of-domain")
        print(f"     Score range: [{ad_scores_array.min():.3f}, {ad_scores_array.max():.3f}]")
        
        # STEP 2: Get DCN predictions (batched)
        mixture_dcns = self.mixture_predictor.predict_batch_mixtures(
            additive_smiles_list=smiles_list,
            base_smiles=self.base_smiles,
            base_mole_fractions=self.base_fractions,
            additive_fraction=mc.additive_fraction,
            verbose=False
        )

        # STEP 3: Predict pure-component YSI for all additives + base in one batch
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # Ensure YSI predictor is loaded
        if 'ysi' not in self.predictor.predictors:
            from core.predictors.pure_component.generic import GenericPredictor
            from core.predictors.pure_component.hf_models import load_models
            paths = load_models()
            self.predictor.predictors['ysi'] = GenericPredictor(paths['ysi'], 'YSI')

        # Featurize all additives + base components in one call
        base_ratio = 1.0 - mc.additive_fraction
        all_unique_smiles = smiles_list + self.base_smiles
        props_all = self.predictor.predict_all_properties(all_unique_smiles)
        ysi_all = props_all.get('ysi', [None] * len(all_unique_smiles))

        # Pre-compute base component YSI and molecular weights (same for every additive)
        base_ysi = ysi_all[len(smiles_list):]
        base_mws = []
        for smi in self.base_smiles:
            mol = Chem.MolFromSmiles(smi)
            base_mws.append(Descriptors.ExactMolWt(mol) if mol is not None else None)

        def _blend_ysi(additive_ysi, additive_mw):
            """Apply mass-fraction blending law for one additive."""
            if additive_ysi is None or additive_mw is None:
                return None
            if any(y is None or mw is None for y, mw in zip(base_ysi, base_mws)):
                return None
            mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]
            mol_weights = [additive_mw] + base_mws
            masses = [x * mw for x, mw in zip(mole_fracs, mol_weights)]
            total = sum(masses)
            if total == 0:
                return None
            mass_fracs = [m / total for m in masses]
            ysi_components = [additive_ysi] + list(base_ysi)
            return sum(y * mf for y, mf in zip(ysi_components, mass_fracs))

        # STEP 4: Create molecules with DCN, AD, and YSI
        molecules = []

        for i, smiles in enumerate(smiles_list):
            dcn = mixture_dcns[i]

            # Filter: No DCN
            if dcn is None:
                filter_stats['cn_none'] += 1
                continue

            # Filter: Outside AD
            if not in_domain_array[i]:
                filter_stats['ad_filtered'] += 1
                continue

            # Compute mixture YSI
            additive_ysi = ysi_all[i]
            mol = Chem.MolFromSmiles(smiles)
            additive_mw = Descriptors.ExactMolWt(mol) if mol is not None else None
            mixture_ysi = _blend_ysi(additive_ysi, additive_mw)

            filter_stats['passed'] += 1

            molecules.append(MixtureAwareMolecule(
                smiles=smiles,
                cn=dcn,
                cn_error=abs(dcn - mc.target_mixture_dcn),
                cn_score=dcn,
                mixture_dcn=dcn,
                blend_ratio=mc.additive_fraction,
                ad_score=float(ad_scores_array[i]),
                in_domain=bool(in_domain_array[i]),
                mixture_ysi=mixture_ysi,
                ysi=mixture_ysi  # mirrors mixture_ysi so Population NSGA-II can use it
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
        pareto_size = len(self.population.pareto_front()) if self.config.minimize_ysi else 0
        extra = f" | Pareto: {pareto_size}" if self.config.minimize_ysi else ""
        if self.use_ad_filtering:
            print(
                f"Gen {generation}/{self.config.generations} | "
                f"Pop {len(mols)} | "
                f"Best: {best_metric:.3f} | "
                f"AD: {avg_ad:.3f}{extra}"
            )
        else:
            print(
                f"Gen {generation}/{self.config.generations} | "
                f"Pop {len(mols)} | "
                f"Best: {best_metric:.3f} | "
                f"Invalid: {n_invalid}{extra}"
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

        if self.config.minimize_ysi:
            ysi_vals = [m.mixture_ysi for m in mols if m.mixture_ysi is not None]
            log_dict["avg_mixture_ysi"] = np.mean(ysi_vals) if ysi_vals else float('nan')
            log_dict["pareto_front_size"] = pareto_size
        
        wandb.log(log_dict)

        # W&B table every 10 gens
        if generation % 6 == 0:
            if self.use_ad_filtering:
                table_data = [
                    [m.smiles, m.mixture_dcn, m.cn_error, m.mixture_ysi, m.ad_score, m.in_domain]
                    for m in sorted(mols, key=lambda x: x.cn_error)[:50]
                ]
                columns = ["SMILES", "DCN", "Error", "YSI", "AD Score", "In Domain"]
            else:
                table_data = [
                    [m.smiles, m.mixture_dcn, m.cn_error, m.mixture_ysi, m.blend_ratio]
                    for m in sorted(mols, key=lambda x: x.cn_error)[:50]
                ]
                columns = ["SMILES", "DCN", "Error", "YSI", "Blend Ratio"]
            
            table = wandb.Table(data=table_data, columns=columns)
            wandb.log({f"top_molecules_gen_{generation}": table})