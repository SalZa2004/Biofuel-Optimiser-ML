import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import pandas as pd
from typing import List, Tuple

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

from .cli import ScreeningConfig, get_user_config
from .results import display_results, save_results


def _load_smiles(csv_path: str) -> Tuple[List[str], pd.DataFrame]:
    """Return (deduplicated smiles list, source DataFrame)."""
    df = pd.read_csv(csv_path)
    smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError(
            f"No 'smiles' column in {csv_path}. Found: {list(df.columns)}"
        )
    df = df.rename(columns={smiles_col: "smiles"})
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    print(f"  Loaded {len(df)} unique SMILES from {csv_path}")
    return df["smiles"].tolist(), df


def _passes_filters(row: dict, filters: dict) -> bool:
    for prop, (lo, hi) in filters.items():
        val = row.get(prop)
        if val is None:
            continue  # treat unpredictable property as passing (match evolution convention)
        if lo is not None and val < lo:
            return False
        if hi is not None and val > hi:
            return False
    return True


def _compute_pareto_df(df: pd.DataFrame, cn_col: str, ysi_col: str) -> pd.DataFrame:
    """Return Pareto-optimal rows from df (minimise cn_col and ysi_col)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    valid = df[cn_col].notna() & df[ysi_col].notna()

    def _rerank(out: pd.DataFrame) -> pd.DataFrame:
        out = out.drop(columns=["rank"], errors="ignore").reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out

    if valid.sum() < 2:
        return _rerank(df[valid].copy())

    valid_df = df[valid].reset_index(drop=True)
    F = np.column_stack([valid_df[cn_col].values, valid_df[ysi_col].values])
    front0 = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pareto = valid_df.iloc[front0].sort_values(cn_col)
    return _rerank(pareto)


def run_pure_screening(
    config: ScreeningConfig, smiles_list: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Predict pure-component properties, apply filters, and find the Pareto front.

    Returns:
        filtered_df  – candidates passing all property filters, sorted by CN error
        pareto_df    – Pareto front from filtered candidates (CN error vs YSI)
        all_df       – all candidates with valid CN predictions, sorted by CN error
    """
    from core.predictors.pure_component.property_predictor import PropertyPredictor

    print("\nLoading property predictors...")
    predictor = PropertyPredictor()

    print(f"Predicting properties for {len(smiles_list)} molecules...")
    preds = predictor.predict_all_properties(smiles_list)

    rows = []
    for i, smi in enumerate(smiles_list):
        cn = preds["cn"][i]
        if cn is None:
            continue
        rows.append({
            "smiles": smi,
            "cn": round(cn, 3),
            "cn_error": round(abs(cn - config.target_cn), 3),
            "ysi": preds["ysi"][i],
            "bp": preds["bp"][i],
            "density": preds["density"][i],
            "lhv": preds["lhv"][i],
            "dynamic_viscosity": preds["dynamic_viscosity"][i],
            "tanimoto": preds["tanimoto"][i],
        })

    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    all_df = pd.DataFrame(rows).sort_values("cn_error").reset_index(drop=True)
    all_df.insert(0, "rank", range(1, len(all_df) + 1))

    filtered_rows = [r for r in rows if _passes_filters(r, config.filters)]
    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows).sort_values("cn_error").reset_index(drop=True)
        filtered_df.insert(0, "rank", range(1, len(filtered_df) + 1))
    else:
        print("  Warning: no molecules passed the property filters.")
        filtered_df = pd.DataFrame(columns=all_df.columns)

    pareto_source = filtered_df if not filtered_df.empty else all_df
    pareto_df = _compute_pareto_df(pareto_source, "cn_error", "ysi")

    n_valid = all_df["ysi"].notna().sum()
    print(f"  {len(all_df)} molecules with valid CN | {len(filtered_df)} passed filters | {n_valid} have YSI")
    print(f"  Pareto front: {len(pareto_df)} candidates")

    return filtered_df, pareto_df, all_df


def _parse_mixture_rows(df: pd.DataFrame) -> List[dict]:
    """
    Parse a mixture_database.csv-format DataFrame into a list of mixture dicts.
    Each dict has: name, smiles (list), fracs (list, normalised to sum=1).
    Rows with no valid SMILES/fraction pairs are skipped.
    """
    rows = []
    for row_i, row in df.iterrows():
        smiles, fracs = [], []
        for k in range(1, 12):
            smi = row.get(f"fuel_{k}_smiles")
            frac = row.get(f"fraction_fuel_{k}")
            if pd.notna(smi) and str(smi).strip() and pd.notna(frac):
                try:
                    fracs.append(float(frac))
                    smiles.append(str(smi).strip())
                except ValueError:
                    pass

        if len(smiles) < 1:
            continue

        total = sum(fracs)
        if total <= 0:
            continue
        if abs(total - 1.0) > 1e-4:
            fracs = [f / total for f in fracs]

        name = str(row.get("Name", f"row_{row_i}"))
        rows.append({"name": name, "smiles": smiles, "fracs": fracs})

    return rows


def _predict_mixture_dcns_batched(
    mixture_predictor,
    mixture_rows: List[dict],
    verbose: bool = True,
) -> List:
    """
    Predict DCN for each mixture row, batched by component count for efficiency.
    Returns a list of floats (or None) aligned with mixture_rows.
    """
    from collections import defaultdict
    from core.predictors.mixture.solvation_predictor.data.data import DatapointList
    from core.predictors.mixture.solvation_predictor.train.evaluate import predict

    mixture_predictor._initialize_models()

    # Group by component count so each batch has a uniform graph structure
    groups: dict = defaultdict(list)
    for i, mix in enumerate(mixture_rows):
        groups[len(mix["smiles"])].append((i, mix))

    results = [None] * len(mixture_rows)

    for n_comp, group in sorted(groups.items()):
        mixture_predictor.args.num_mols = n_comp
        mixture_predictor.args.solvent_headers = [f"fuel{j+1}_inchi" for j in range(n_comp)]
        mixture_predictor.args.molefrac_headers = [f"frac_fuel{j+1} (molar)" for j in range(n_comp - 1)]
        mixture_predictor.args.target_headers = ["DCN"]
        mixture_predictor.args.features_headers = []
        mixture_predictor.args.solute_headers = []

        datapoints, orig_indices = [], []
        for orig_i, mix in group:
            try:
                dp = mixture_predictor._create_datapoint_direct(mix["smiles"], mix["fracs"])
                datapoints.append(dp)
                orig_indices.append(orig_i)
            except Exception as e:
                if verbose:
                    print(f"  Skip '{mix['name']}': {str(e)[:70]}")

        if not datapoints:
            continue

        if not hasattr(mixture_predictor, "_f_mol_size_set"):
            mixture_predictor.args.f_mol_size = datapoints[0].get_mol_encoder()[0].get_sizes()[2]
            mixture_predictor.args.num_features = len(datapoints[0].features)
            mixture_predictor._f_mol_size_set = True

        data = DatapointList(datapoints)
        all_preds = []
        for model, scaler in zip(mixture_predictor.models, mixture_predictor.scalers):
            if mixture_predictor.args.scale == "standard":
                scaler.transform_standard(data)
            batch_preds = predict(model=model, data=data, scaler=scaler, inp=mixture_predictor.args)
            all_preds.append([p[0] for p in batch_preds])

        ensemble = np.array(all_preds).mean(axis=0)
        for local_i, orig_i in enumerate(orig_indices):
            results[orig_i] = float(ensemble[local_i])

        if verbose:
            print(f"  {n_comp}-component mixtures: {len(datapoints)}/{len(group)} predicted")

    return results


def run_mixture_screening(
    config: ScreeningConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Screen pre-defined mixtures from a mixture_database.csv-format CSV.
    Predicts mixture DCN (GNN ensemble) and mixture YSI (mass-weighted blending law)
    for each row, then finds the Pareto front (CN error vs YSI).

    Returns:
        results_df  – all valid predictions sorted by mixture CN error
        pareto_df   – Pareto front (mixture CN error vs mixture YSI)
        results_df  – same (no separate unfiltered set for mixture mode)
    """
    from core.predictors.pure_component.property_predictor import PropertyPredictor
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    from core.blending.blending_law import blend_ysi_mass_weighted

    df = pd.read_csv(config.csv_path, encoding="latin-1")
    mixture_rows = _parse_mixture_rows(df)
    print(f"  Parsed {len(mixture_rows)} mixtures from {config.csv_path}")

    if not mixture_rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Collect all unique component SMILES for one-shot pure-component YSI prediction
    unique_smiles = list(dict.fromkeys(s for mix in mixture_rows for s in mix["smiles"]))
    print(f"\nPredicting pure-component YSI for {len(unique_smiles)} unique components...")
    predictor = PropertyPredictor()
    pure_preds = predictor.predict_all_properties(unique_smiles)
    ysi_map = {smi: pure_preds["ysi"][i] for i, smi in enumerate(unique_smiles)}

    # Predict mixture DCN for all rows, grouped by component count
    print(f"\nPredicting mixture DCN for {len(mixture_rows)} mixtures...")
    mixture_predictor = MixtureDCNPredictor()
    dcn_preds = _predict_mixture_dcns_batched(mixture_predictor, mixture_rows, verbose=True)

    rows = []
    for i, mix in enumerate(mixture_rows):
        mix_dcn = dcn_preds[i]
        if mix_dcn is None:
            continue

        mix_ysi = blend_ysi_mass_weighted(
            smiles=mix["smiles"],
            mole_fracs=mix["fracs"],
            ysi_values=[ysi_map.get(s) for s in mix["smiles"]],
        )

        rows.append({
            "name": mix["name"],
            "mixture_cn": round(mix_dcn, 3),
            "mixture_cn_error": round(abs(mix_dcn - config.target_cn), 3),
            "mixture_ysi": round(mix_ysi, 3) if mix_ysi is not None else None,
            "n_components": len(mix["smiles"]),
        })

    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    results_df = pd.DataFrame(rows).sort_values("mixture_cn_error").reset_index(drop=True)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))

    pareto_df = _compute_pareto_df(results_df, "mixture_cn_error", "mixture_ysi")

    n_with_ysi = results_df["mixture_ysi"].notna().sum()
    print(f"  {len(results_df)} valid predictions | {n_with_ysi} have mixture YSI")
    print(f"  Pareto front: {len(pareto_df)} candidates")

    return results_df, pareto_df, results_df


def run(config: ScreeningConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.mode == "pure_component":
        smiles_list, _ = _load_smiles(config.csv_path)
        return run_pure_screening(config, smiles_list)
    return run_mixture_screening(config)


def main():
    config = get_user_config()
    filtered_df, pareto_df, all_df = run(config)
    display_results(filtered_df, pareto_df, all_df, config)
    save_results(filtered_df, pareto_df, all_df, config)


if __name__ == "__main__":
    main()
