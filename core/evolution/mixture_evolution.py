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
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)
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
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def _key(self, smiles: str, base_fuel_type: str, fraction: float) -> str:
        return f"{smiles}|{base_fuel_type}|{fraction:.3f}"

    def get(self, additive_smiles: str, base_fuel_type: str,
            additive_fraction: float) -> float:
        return self.cache.get(self._key(additive_smiles, base_fuel_type, additive_fraction))

    def get_batch(self, additive_smiles_list: list, base_fuel_type: str,
                  additive_fraction: float) -> dict:
        return {smi: v for smi in additive_smiles_list
                if (v := self.get(smi, base_fuel_type, additive_fraction)) is not None}

    def set_batch(self, predictions: dict, base_fuel_type: str,
                  additive_fraction: float):
        """Cache multiple predictions and flush to disk once."""
        for smiles, dcn in predictions.items():
            if dcn is not None:
                self.cache[self._key(smiles, base_fuel_type, additive_fraction)] = dcn
        self._save()

    def populate_from_database(self, csv_path: str, base_fuel_type: str,
                               additive_fraction: float, base_smiles: List[str]):
        """Pre-load cache from mixture_database.csv using CN_Measured / CN_Regression."""
        import pandas as pd
        from rdkit import Chem

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return

        def canonical(smi):
            try:
                return Chem.MolToSmiles(Chem.MolFromSmiles(smi)) if smi and str(smi) != 'nan' else None
            except Exception:
                return None

        base_canonical = {canonical(s) for s in base_smiles if s}
        frac_tol = 0.02
        added = 0

        for _, row in df.iterrows():
            cn = row.get('CN_Measured') if not pd.isna(row.get('CN_Measured', float('nan'))) \
                 else row.get('CN_Regression')
            if pd.isna(cn):
                continue

            # Collect all (smiles, fraction) pairs from this row
            components = []
            for k in range(1, 12):
                smi = canonical(str(row.get(f'fuel_{k}_smiles', '') or ''))
                frac = row.get(f'fraction_fuel_{k}')
                if smi and not pd.isna(frac):
                    components.append((smi, float(frac)))

            if len(components) < 2:
                continue

            # Find if exactly one component is the additive (not a base component)
            for idx, (smi, frac) in enumerate(components):
                if smi in base_canonical:
                    continue
                if abs(frac - additive_fraction) > frac_tol:
                    continue
                rest = [c for j, c in enumerate(components) if j != idx]
                if all(c[0] in base_canonical for c in rest):
                    key = self._key(smi, base_fuel_type, additive_fraction)
                    if key not in self.cache:
                        self.cache[key] = float(cn)
                        added += 1

        if added:
            self._save()
            print(f"  ✓ Pre-loaded {added} DCN values from database")

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
        # in-memory AD cache: smiles -> (ad_score, in_domain)
        self._ad_cache: Dict[str, Tuple[float, bool]] = {}

        print("Initializing mixture DCN predictor...")
        self.mixture_predictor = MixtureDCNPredictor()
        self.mixture_predictor._initialize_models()
        self._load_base_fuel()

        self.use_ad_filtering = use_ad_filtering
        if use_ad_filtering:
            self._load_ad_checker()
        self._mol_db = None  # lazy-loaded MolencoderDatabase cache

        # Pre-populate DCN cache from database
        mc = self.config.mixture_config
        db_path = Path(__file__).resolve().parent.parent.parent / "data" / "database" / "mixture_database.csv"
        self.dcn_cache.populate_from_database(
            str(db_path),
            mc.base_fuel_type,
            mc.additive_fraction,
            self.base_smiles,
        )
        
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
        """Load the trained One-Class SVM from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
            ad_path = hf_hub_download(
                repo_id="SalZa2004/mixture_ocvm_checker",
                filename="mixture_ocsvm.pkl",
            )
            with open(ad_path, 'rb') as f:
                ad_data = pickle.load(f)
                self.svm = ad_data['svm']
                self.scaler = ad_data['scaler']
            print("✓ AD checker loaded")
        except Exception as e:
            print(f"⚠ AD checker could not be loaded ({e}) - disabling AD filtering")
            self.use_ad_filtering = False
    
    def _extract_mixture_embedding(self, additive_smiles: str) -> np.ndarray:
        from core.predictors.mixture.solvation_predictor.data.data import (
            DataPoint, DatapointList, MolencoderDatabase, DataTensor
        )

        mc = self.config.mixture_config
        base_ratio = 1.0 - mc.additive_fraction
        adjusted_base = [f * base_ratio for f in self.base_fractions]

        mixture_smiles    = [additive_smiles] + self.base_smiles
        mixture_fractions = [mc.additive_fraction] + adjusted_base[:-1]  # N-1

        # Build DataPoint once — reused across all 10 models
        try:
            if self._mol_db is None:
                self._mol_db = MolencoderDatabase()
            mol_db = self._mol_db
            dp = DataPoint(
                smiles=mixture_smiles,
                targets=[0.0],
                features=[],
                molefracs=mixture_fractions,
                inp=self.mixture_predictor.args,
                mol_encoders=mol_db,
            )
            data = DatapointList([dp])

            args = self.mixture_predictor.args
            enc = dp.get_mol_encoder()
            if len(enc) < args.max_num_mols:
                enc += [enc[0]] * (args.max_num_mols - len(enc))
            mol_encodings = [[enc[m]] for m in range(args.max_num_mols)]
            tensors = [DataTensor(mol_encodings[m], args, property=args.property)
                       for m in range(args.max_num_mols)]
        except Exception:
            return None

        all_model_embeddings = []
        for model in self.mixture_predictor.models:
            model.eval()
            captured = []

            def hook_fn(module, input, output):
                captured.append(input[0].detach().cpu())

            hook = model.ffn.register_forward_hook(hook_fn)
            try:
                with torch.no_grad():
                    _ = model(data, tensors)
                if captured:
                    all_model_embeddings.append(captured[0].numpy().flatten())
            except Exception:
                continue
            finally:
                hook.remove()

        if not all_model_embeddings:
            return None
        return np.mean(all_model_embeddings, axis=0)
    
    def _check_ad_batch(self, smiles_list: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Check AD for a batch, using in-memory cache to skip already-seen SMILES."""
        scores = np.full(len(smiles_list), -999.0)
        domain = np.zeros(len(smiles_list), dtype=bool)

        if not self.use_ad_filtering:
            return np.zeros(len(smiles_list)), np.ones(len(smiles_list), dtype=bool)

        # Split into cached vs needs-embedding
        need_emb_idx = []
        for i, smi in enumerate(smiles_list):
            if smi in self._ad_cache:
                scores[i], domain[i] = self._ad_cache[smi]
            else:
                need_emb_idx.append(i)

        if not need_emb_idx:
            return scores, domain

        embeddings = []
        valid_idx = []
        for i in need_emb_idx:
            emb = self._extract_mixture_embedding(smiles_list[i])
            if emb is not None:
                embeddings.append(emb)
                valid_idx.append(i)

        if embeddings:
            emb_arr = self.scaler.transform(np.array(embeddings))
            scores_v = self.svm.decision_function(emb_arr)
            domain_v = self.svm.predict(emb_arr) == 1
            for j, idx in enumerate(valid_idx):
                scores[idx] = scores_v[j]
                domain[idx] = domain_v[j]
                self._ad_cache[smiles_list[idx]] = (float(scores_v[j]), bool(domain_v[j]))

        return scores, domain

    def predict_mixture_ysi(self, additive_smiles: str) -> Optional[float]:
        """
        Predict mixture YSI using the linear blending law (mass-fraction weighted).

        Args:
            additive_smiles: SMILES of the additive molecule.

        Returns:
            ysi_mix: Predicted mixture YSI, or None if any step fails.
        """
        from core.blending.blending_law import blend_ysi_mass_weighted

        mc = self.config.mixture_config
        base_ratio = 1.0 - mc.additive_fraction
        all_smiles = [additive_smiles] + self.base_smiles
        all_mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]

        if 'ysi' not in self.predictor.predictors:
            from core.predictors.pure_component.generic import GenericPredictor
            from core.predictors.pure_component.hf_models import load_models
            paths = load_models()
            self.predictor.predictors['ysi'] = GenericPredictor(paths['ysi'], 'YSI')

        props = self.predictor.predict_all_properties(all_smiles)
        ysi_values = props.get('ysi', [])

        if len(ysi_values) != len(all_smiles):
            return None

        return blend_ysi_mass_weighted(all_smiles, all_mole_fracs, ysi_values)

    def predict_mixture_ysi_batch(self, smiles_list: List[str]) -> List[Optional[float]]:
        """Predict mixture YSI for a list of additives."""
        return [self.predict_mixture_ysi(smi) for smi in smiles_list]

    def _create_molecules(self, smiles_list: List[str]) -> Tuple[List[MixtureAwareMolecule], Dict]:

        if not smiles_list:
            return [], {
                'total': 0,
                'cn_none': 0,
                'ysi_none': 0,
                'ad_filtered': 0,
                'passed': 0,
            }

        mc = self.config.mixture_config

        filter_stats = {
            'total': len(smiles_list),
            'cn_none': 0,
            'ysi_none': 0,
            'ad_filtered': 0,
            'passed': 0,
        }
        
        # STEP 1: Check AD for all molecules (returns numpy arrays)
        ad_scores_array, in_domain_array = self._check_ad_batch(smiles_list)

        n_in_domain = in_domain_array.sum()
        n_out_domain = (~in_domain_array).sum()
        print(f"  → AD Check: {n_in_domain} in-domain, {n_out_domain} out-of-domain")
        print(f"     Score range: [{ad_scores_array.min():.3f}, {ad_scores_array.max():.3f}]")

        # Short-circuit: nothing passes AD, skip expensive predictions
        if n_in_domain == 0:
            filter_stats['ad_filtered'] = len(smiles_list)
            return [], filter_stats

        # STEP 2: Get DCN predictions only for in-domain molecules (cache-first)
        in_domain_indices = [i for i, ok in enumerate(in_domain_array) if ok]
        in_domain_smiles = [smiles_list[i] for i in in_domain_indices]

        cached_dcns = self.dcn_cache.get_batch(
            in_domain_smiles, mc.base_fuel_type, mc.additive_fraction
        )
        uncached_smiles = [s for s in in_domain_smiles if s not in cached_dcns]

        if uncached_smiles:
            new_dcns = self.mixture_predictor.predict_batch_mixtures(
                additive_smiles_list=uncached_smiles,
                base_smiles=self.base_smiles,
                base_mole_fractions=self.base_fractions,
                additive_fraction=mc.additive_fraction,
                verbose=False,
            )
            self.dcn_cache.set_batch(
                dict(zip(uncached_smiles, new_dcns)),
                mc.base_fuel_type,
                mc.additive_fraction,
            )
            cached_dcns.update(zip(uncached_smiles, new_dcns))

        mixture_dcns_in_domain = {orig_idx: cached_dcns.get(smiles_list[orig_idx])
                                   for orig_idx in in_domain_indices}

        # STEP 3: Predict YSI only for in-domain additives + base components

        # Ensure YSI predictor is loaded
        if 'ysi' not in self.predictor.predictors:
            from core.predictors.pure_component.generic import GenericPredictor
            from core.predictors.pure_component.hf_models import load_models
            paths = load_models()
            self.predictor.predictors['ysi'] = GenericPredictor(paths['ysi'], 'YSI')

        base_ratio = 1.0 - mc.additive_fraction
        all_unique_smiles = in_domain_smiles + self.base_smiles
        props_all = self.predictor.predict_all_properties(all_unique_smiles)
        ysi_for_in_domain = props_all.get('ysi', [None] * len(all_unique_smiles))
        # Build a lookup by original index for in-domain additives
        ysi_by_orig_idx = {orig_idx: ysi_for_in_domain[pos]
                           for pos, orig_idx in enumerate(in_domain_indices)}

        # Pre-compute base component YSI (same for every additive)
        base_ysi = ysi_for_in_domain[len(in_domain_smiles):]

        from core.blending.blending_law import blend_ysi_mass_weighted

        # STEP 4: Create molecules — only iterate over in-domain candidates
        filter_stats['ad_filtered'] = n_out_domain
        molecules = []

        for i in in_domain_indices:
            smiles = smiles_list[i]
            dcn = mixture_dcns_in_domain.get(i)

            if dcn is None:
                filter_stats['cn_none'] += 1
                continue

            additive_ysi = ysi_by_orig_idx.get(i)
            mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]
            mixture_ysi = blend_ysi_mass_weighted(
                [smiles] + self.base_smiles,
                mole_fracs,
                [additive_ysi] + list(base_ysi),
            )

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

    def initialize_population(self, initial_smiles: List[str]) -> int:
        """Initialize population with AD-based filtering."""
        print("Predicting properties for initial population...")
        molecules, filter_stats = self._create_molecules(initial_smiles)

        if filter_stats['total'] > 0:
            print(f"\n  Initial filtering: {filter_stats['total']} → {filter_stats['passed']} passed")
            print(f"    AD filtered: {filter_stats['ad_filtered']} | "
                  f"CN None: {filter_stats['cn_none']}")

        return self.population.add_molecules(molecules)

    def _generate_offspring(self, survivors: List[MixtureAwareMolecule]) -> Tuple[List[MixtureAwareMolecule], Dict]:
        """Generate offspring using AD-based filtering (no Tanimoto / uncertainty filters)."""
        import random
        from rdkit import Chem

        target_count = self.config.population_size - len(survivors)
        max_attempts = target_count * self.config.max_offspring_attempts

        all_children: List[str] = []
        new_molecules: List[MixtureAwareMolecule] = []
        cumulative_stats = {
            'total': 0,
            'cn_none': 0,
            'ysi_none': 0,
            'ad_filtered': 0,
            'passed': 0,
        }

        print(f"  → Generating offspring (target: {target_count})...")

        for _ in range(max_attempts):
            if len(new_molecules) >= target_count:
                break

            weights = np.array([m.fitness(self.config) for m in survivors])
            weights /= weights.sum()
            parent = survivors[np.random.choice(len(survivors), p=weights)]
            mol = Chem.MolFromSmiles(parent.smiles)
            if mol is None:
                continue

            children = self._mutate_molecule(mol)
            all_children.extend(children[:self.config.mutations_per_parent])

            if len(all_children) >= self.config.batch_size:
                batch_mols, batch_stats = self._create_molecules(all_children)
                new_molecules.extend(batch_mols)
                for key in cumulative_stats:
                    cumulative_stats[key] += batch_stats.get(key, 0)
                all_children = []

        if all_children:
            batch_mols, batch_stats = self._create_molecules(all_children)
            new_molecules.extend(batch_mols)
            for key in cumulative_stats:
                cumulative_stats[key] += batch_stats.get(key, 0)

        print(f"  ✓ Generated {len(new_molecules)} valid offspring")
        print(f"    Filtering: {cumulative_stats['total']} → {cumulative_stats['passed']} | "
              f"AD filtered: {cumulative_stats['ad_filtered']} | "
              f"CN None: {cumulative_stats['cn_none']} "
              f"(property constraints applied at end)")

        return new_molecules, cumulative_stats

    def _predict_densities(self, smiles_list: List[str]) -> Dict[str, Optional[float]]:
        """Return a SMILES→density map, handling featurization drop-outs."""
        from core.shared_features import featurize_df

        if not smiles_list:
            return {}

        # Start with all None; only successful predictions overwrite entries.
        density_map: Dict[str, Optional[float]] = {smi: None for smi in smiles_list}

        # return_df=True gives back which SMILES actually survived featurization,
        # avoiding a silent length mismatch when descriptors fail for some molecules.
        result = featurize_df(smiles_list, return_df=True)
        if result is None or result[0] is None:
            return density_map

        X, df_valid = result
        valid_smiles = df_valid['SMILES'].tolist()

        preds = self.predictor.predictors['density'].predict_from_features(X)
        for smi, val in zip(valid_smiles, preds):
            try:
                density_map[smi] = float(val) if val is not None and np.isfinite(val) else None
            except (TypeError, ValueError):
                density_map[smi] = None

        n_none = sum(1 for v in density_map.values() if v is None)
        if n_none:
            smi_sample = next(s for s in smiles_list if density_map.get(s) is None)
            print(f"  ⚠ Density None for {n_none}/{len(smiles_list)} SMILES "
                  f"(e.g. '{smi_sample}')")

        return density_map

    def _compute_mixture_bp(self, additive_smiles_list: List[str]) -> Dict[str, Optional[float]]:
        """Compute Riazi-Daubert mixture boiling point (°C) for each unique additive SMILES."""
        from core.blending.blending_law import blend_bp_riazi_daubert

        mc = self.config.mixture_config
        base_ratio = 1.0 - mc.additive_fraction

        unique = list(dict.fromkeys(additive_smiles_list))
        density_map = self._predict_densities(unique + self.base_smiles)

        base_densities = [density_map.get(smi) for smi in self.base_smiles]
        mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]

        results = {}
        for smi in unique:
            all_densities = [density_map.get(smi)] + base_densities
            # blend_bp_riazi_daubert expects specific gravity in g/cm³;
            # the density predictor returns kg/m³, so divide by 1000.
            all_densities_gcc = [
                d / 1000.0 if d is not None else None for d in all_densities
            ]
            results[smi] = blend_bp_riazi_daubert(
                [smi] + self.base_smiles,
                mole_fracs,
                all_densities_gcc,
            )
        return results

    def _compute_mixture_density(self, additive_smiles_list: List[str]) -> Dict[str, Optional[float]]:
        """Compute mixture density (g/cm³) for each unique additive SMILES using Eq. 3 (β_ij = 0)."""
        from core.blending.blending_law import blend_density

        mc = self.config.mixture_config
        base_ratio = 1.0 - mc.additive_fraction

        unique = list(dict.fromkeys(additive_smiles_list))
        density_map = self._predict_densities(unique + self.base_smiles)

        base_densities = [density_map.get(smi) for smi in self.base_smiles]
        mole_fracs = [mc.additive_fraction] + [f * base_ratio for f in self.base_fractions]

        results = {}
        for smi in unique:
            all_densities = [density_map.get(smi)] + base_densities
            # Density predictor returns kg/m³; convert to g/cm³ for blending law
            all_densities_gcc = [
                d / 1000.0 if d is not None else None for d in all_densities
            ]
            result_gcc = blend_density(
                [smi] + self.base_smiles,
                mole_fracs,
                all_densities_gcc,
            )
            results[smi] = result_gcc * 1000.0 if result_gcc is not None else None
        return results

    def _sort_df(self, df):
        """Sort without a hard cn_error cutoff.

        The base class applies cn_error < 5 or < 15 thresholds that are fine for
        pure-component evolution but too tight for mixture DCN predictions, which
        can sit further from the target early in the run.
        """
        if df.empty:
            return df
        if self.config.maximize_cn:
            if self.config.minimize_ysi and "ysi" in df.columns:
                return df.sort_values(["cn", "ysi"], ascending=[False, True])
            return df.sort_values("cn", ascending=False)
        else:
            if self.config.minimize_ysi and "ysi" in df.columns:
                return df.sort_values(["cn_error", "ysi"], ascending=True)
            return df.sort_values("cn_error", ascending=True)

    def _apply_mixture_filters(self, df):
        """Filter final_df by mixture_bp and mixture_density bounds.

        Molecules where a property is None (couldn't be computed) are left in —
        only molecules with a computed value that falls outside the range are removed.
        """
        import pandas as pd

        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        for prop, (lo, hi) in self.config.mixture_filters.items():
            if prop not in df.columns:
                continue
            col = df[prop]
            if lo is not None:
                mask &= col.isna() | (col >= lo)
            if hi is not None:
                mask &= col.isna() | (col <= hi)
        filtered = df[mask].copy()
        removed = len(df) - len(filtered)
        if removed > 0:
            print(f"  Mixture property filters removed {removed}/{len(df)} molecules")
        filtered["rank"] = range(1, len(filtered) + 1)
        return filtered

    def _generate_results(self):
        """Generate final DataFrames and append mixture_bp and mixture_density."""
        final_df, pareto_df, unfiltered_df = super()._generate_results()

        # Collect smiles from all three DataFrames so bp/density can be computed
        # even when final_df is still empty before mixture filtering.
        all_smiles = []
        for df in (final_df, pareto_df, unfiltered_df):
            if not df.empty and 'smiles' in df.columns:
                all_smiles.extend(df['smiles'].tolist())

        if all_smiles:
            unique_smiles = list(dict.fromkeys(all_smiles))
            print("  Computing mixture boiling points (Riazi-Daubert)...")
            bp_map = self._compute_mixture_bp(unique_smiles)
            print("  Computing mixture densities (Eq. 3, β_ij = 0)...")
            density_map = self._compute_mixture_density(unique_smiles)

            for result_df in (final_df, pareto_df, unfiltered_df):
                if not result_df.empty and 'smiles' in result_df.columns:
                    result_df['mixture_bp'] = result_df['smiles'].map(bp_map)
                    result_df['mixture_density'] = result_df['smiles'].map(density_map)

        # Apply mixture-specific property constraints now that blended columns exist.
        # This is done here rather than in the base class because mixture_bp and
        # mixture_density don't exist until after the blending computation above.
        final_df = self._apply_mixture_filters(final_df)

        return final_df, pareto_df, unfiltered_df