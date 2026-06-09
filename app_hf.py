import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"


from flask import Flask, render_template, request, redirect, url_for, send_file, session
import sqlite3
import pandas as pd
import io
import json
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.base import BaseEstimator, RegressorMixin
from huggingface_hub import hf_hub_download
import sys
import pubchempy as pcp
import importlib.util
import base64
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# PURE FUEL GENERATOR
from core.config import EvolutionConfig, MixtureConfig
from core.evolution.evolution import MolecularEvolution

# FAME CN PREDICTORS IMPORTS
from core.predictors.biodiesel.config import all_training_fames, fame_groups
from core.predictors.biodiesel.pipeline import ensure_fame_columns, run_prediction_pipeline

# MIXTURE CN PREDICTOR IMPORTS
from rdkit.Chem import rdMolDescriptors
from core.blending.blending_law import (
    blend_ysi_mass_weighted,
    blend_bp_riazi_daubert,
    blend_density
)

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import tempfile

# MODEL REPOSITORY CONFIGURATION
REPO_ID_BIODIESEL = "mrashid26/Biodiesel_CN_Predictor"

# FAME BIODIESEL CN MODEL LOAD
def load_biodiesel_pkl(filename):
    """
    Load biodiesel/FAME model files from the Hugging Face repo.
    This follows the same loading approach as core/predictors/biodiesel/main.py.
    """
    path = hf_hub_download(
        repo_id=REPO_ID_BIODIESEL,
        filename=filename,
        repo_type="space"
    )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)

fame_cn_model = load_biodiesel_pkl("src/biodiesel_cn_model.pkl")
fame_ood_stats = load_biodiesel_pkl("src/ood_stats.pkl")

#---------------------------
# HELPER FUNCTIONS 
#---------------------------
def validate_smiles(smiles):
    if pd.isna(smiles) or smiles == "":
        return False
    return Chem.MolFromSmiles(smiles) is not None

def pubchem_name_to_smiles(name):
    """Return canonical SMILES from a compound name."""
    if not name or not isinstance(name, str):
        return None
    name = name.strip()
    if name == "":
        return None

    try:
        results = pcp.get_compounds(name, "name")
        if not results:
            return None
        return results[0].canonical_smiles
    except Exception:
        return None

def pubchem_smiles_to_name(smiles):
    """Return preferred IUPAC name from SMILES."""
    try:
        results = pcp.get_compounds(smiles, "smiles")
        if not results:
            return None
        compound = results[0]

        # Prefer IUPAC name if available
        if getattr(compound, "iupac_name", None):
            return compound.iupac_name

        # Fallback to title
        return compound.title
    except Exception:
        return None

# ---------------------------
# CORE PURE PROPERTY PREDICTOR
# ---------------------------
property_predictor_instance = None

def get_property_predictor():
    """
    Lazy-load the core pure-component PropertyPredictor.
    This connects app.py back to core/predictors/pure_component/.
    """
    global property_predictor_instance

    if property_predictor_instance is None:
        from core.predictors.pure_component.property_predictor import PropertyPredictor
        property_predictor_instance = PropertyPredictor()

    return property_predictor_instance


def predict_all_properties_for_smiles(smiles_list):
    """
    Uses the same core pure-property predictor style as the uploaded pure predictor app.
    """
    predictor = get_property_predictor()
    return predictor.predict_all_properties(smiles_list)


def get_first_prediction(props, key):
    values = props.get(key, [None])
    if values is None or len(values) == 0:
        return None
    return values[0]


def predict_cn(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "cn")
    except Exception:
        return None


def predict_ysi(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "ysi")
    except Exception:
        return None


def predict_bp(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "bp")
    except Exception:
        return None


def predict_density(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "density")
    except Exception:
        return None


def predict_lhv(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "lhv")
    except Exception:
        return None


def predict_dynamic_viscosity(smiles):
    try:
        props = predict_all_properties_for_smiles([smiles])
        return get_first_prediction(props, "dynamic_viscosity")
    except Exception:
        return None
        
#---------------------------
# PURE FUEL GENERATOR HELPER
#---------------------------
def get_generator_config_from_form(form) -> EvolutionConfig:
    """
    Flask version of the pure fuel generator configuration.
    The HTML form replaces the CLI input.
    """
    mode = form.get("mode", "target")
    maximize_cn = mode == "maximize"

    if maximize_cn:
        target_cn = 100.0
    else:
        try:
            target_cn = float(form.get("target_cn", "50"))
        except ValueError:
            target_cn = 50.0

    minimize_ysi = form.get("minimize_ysi") == "on"

    return EvolutionConfig(
        target_cn=target_cn,
        maximize_cn=maximize_cn,
        minimize_ysi=minimize_ysi
    )

def generator_results_to_tables(final_df, pareto_df, unfiltered_df, config):
    """
    Convert pure fuel generator outputs into HTML tables.
    Matches the original application output:
    1. final candidates with property constraints
    2. unfiltered candidates
    3. Pareto front if YSI minimisation is enabled
    """
    cols = [
        "rank",
        "smiles",
        "cn",
        "cn_error",
        "ysi",
        "bp",
        "density",
        "lhv",
        "dynamic_viscosity"
    ]

    if config.maximize_cn:
        cols = [c for c in cols if c != "cn_error"]

    final_table = None
    unfiltered_table = None
    pareto_table = None

    if final_df is not None and not final_df.empty:
        final_cols = [c for c in cols if c in final_df.columns]
        final_table = final_df.head(20)[final_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    if unfiltered_df is not None and not unfiltered_df.empty:
        unfiltered_cols = [c for c in cols if c in unfiltered_df.columns]
        unfiltered_table = unfiltered_df.head(20)[unfiltered_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    if config.minimize_ysi and pareto_df is not None and not pareto_df.empty:
        pareto_cols = [c for c in cols if c in pareto_df.columns]
        pareto_table = pareto_df.head(30)[pareto_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    return final_table, unfiltered_table, pareto_table

def build_history_images(history):
    """
    Convert a small set of history SMILES into RDKit images saved in static/generated/.
    Returns history enriched with SMILES, structure image, and name.
    """
    out = []
    base_dir = os.path.join("static", "generated")
    os.makedirs(base_dir, exist_ok=True)

    for h in history or []:
        gen = h.get("generation")
        samples = []

        for i, smi in enumerate(h.get("smiles", []), start=1):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # draw structure
            img = Draw.MolToImage(mol, size=(240, 200))
            img_filename = f"evo_gen_{gen}_{i}.png"
            img_path = os.path.join(base_dir, img_filename)
            img.save(img_path)

            # get molecule name using YOUR helper
            name = pubchem_smiles_to_name(smi)
            if not name:
                name = f"Gen {gen} – Rank {i}"

            samples.append({
                "rank": i,
                "name": name,
                "smiles": smi,
                "img_id": img_filename
            })

        out.append({
            "generation": gen,
            "samples": samples
        })

    return out

DB_PATH = os.path.join("data", "database", "database_compiled.db")
PURE_TABLE = "main_pure_datasets_cn_ysi"

def canonicalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def fetch_measured(name: str = None, smiles: str = None):
    """
    Priority:
    1) If smiles provided: match SMILES_Standardized
    2) Else (or if not found): match Name
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Try SMILES_Standardized
    if smiles:
        can = canonicalize_smiles(smiles)
        if can:
            cur.execute(f"""
                SELECT CN_Measured, YSI_Unified_Measured
                FROM "{PURE_TABLE}"
                WHERE SMILES_Standardized = ?
                LIMIT 1
            """, (can,))
            row = cur.fetchone()
            if row:
                conn.close()
                return {"measured_dcn": row[0], "measured_ysi": row[1]}

    # Fallback: Name match (case-insensitive)
    if name and name.strip() and name != "-":
        cur.execute(f"""
            SELECT CN_Measured, YSI_Unified_Measured
            FROM "{PURE_TABLE}"
            WHERE LOWER(Name) = LOWER(?)
            LIMIT 1
        """, (name.strip(),))
        row = cur.fetchone()
        conn.close()
        if row:
            return {"measured_dcn": row[0], "measured_ysi": row[1]}

    conn.close()
    return {"measured_dcn": None, "measured_ysi": None}

# -------------------------
# BIODIESEL/FAME HELPERS
# -------------------------
def detect_unsupported_fames(df):
    """
    Return CSV columns that look like FAME columns but are not used by the training model.
    Example supported format: C18:1, C16:0, C20:5.
    """
    unsupported = []

    for col in df.columns:
        col_clean = str(col).strip()

        if col_clean in ["Name", "CN"]:
            continue

        if col_clean.startswith("C") and ":" in col_clean and col_clean not in all_training_fames:
            unsupported.append(col_clean)

    return unsupported


def make_template_csv():
    """
    Create a blank CSV template for biodiesel FAME prediction.
    """
    columns = ["Name", "CN"] + all_training_fames
    df = pd.DataFrame(columns=columns)
    return df.to_csv(index=False).encode("utf-8")


def make_example_csv():
    """
    Create a simple example CSV with one valid biodiesel-like composition.
    """
    row = {fame: 0.0 for fame in all_training_fames}

    row.update({
        "Name": "Example biodiesel",
        "CN": 55.0,
        "C16:0": 20.0,
        "C18:0": 5.0,
        "C18:1": 55.0,
        "C18:2": 15.0,
        "C18:3": 5.0
    })

    df = pd.DataFrame([row])
    columns = ["Name", "CN"] + all_training_fames
    return df[columns].to_csv(index=False).encode("utf-8")

# -------------------------
# MIXTURE PREDICTOR HELPERS
# -------------------------
mixture_dcn_predictor_instance = None

def get_mixture_dcn_predictor():
    """
    Lazy-load the mixture DCN predictor.
    This avoids loading/downloading the GNN model at Flask startup.
    """
    global mixture_dcn_predictor_instance

    if mixture_dcn_predictor_instance is None:
        from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
        mixture_dcn_predictor_instance = MixtureDCNPredictor()

    return mixture_dcn_predictor_instance


def get_molecule_summary(smiles):
    """
    Return basic molecular information for display in the GUI.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "formula": None,
            "mol_weight": None,
            "num_atoms": None
        }

    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": round(float(Descriptors.MolWt(mol)), 2),
        "num_atoms": int(mol.GetNumAtoms())
    }


def safe_round(value, digits=2):
    """
    Round numbers safely while keeping None as None.
    """
    if value is None:
        return None

    try:
        if not np.isfinite(value):
            return None
        return round(float(value), digits)
    except Exception:
        return None

# -------------------------
# MIXTURE GENERATOR HELPERS
# -------------------------

def get_mixture_generator_config_from_form(form) -> EvolutionConfig:
    """
    Flask version of the uploaded mixture generator CLI logic.
    This follows cli.py:
    - optimisation mode
    - target mixture DCN
    - base fuel type
    - custom base fuel
    - additive fraction
    - minimise mixture YSI
    - returns EvolutionConfig with mixture_mode=True
    """
    mode = form.get("mode", "target")
    maximize_cn = mode == "maximize"

    if maximize_cn:
        target_dcn = 100.0
    else:
        try:
            target_dcn = float(form.get("target_dcn", "50"))
        except ValueError:
            raise ValueError("Invalid target mixture DCN. Please enter a number.")

    base_fuel_type = form.get("base_fuel_type", "fossil_diesel")

    base_fuel_smiles = None
    base_fuel_fractions = None

    if base_fuel_type == "custom":
        custom_smiles = form.getlist("custom_smiles[]")
        custom_fractions = form.getlist("custom_fraction[]")

        base_fuel_smiles = []
        base_fuel_fractions = []
        total_fraction = 0.0

        for i, (smiles, frac) in enumerate(zip(custom_smiles, custom_fractions)):
            smiles = smiles.strip()
            frac = frac.strip()

            if smiles == "" and frac == "":
                continue

            if smiles == "":
                raise ValueError(f"Custom base fuel component {i + 1} is missing SMILES.")

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES for custom base fuel component {i + 1}: {smiles}")

            try:
                mole_frac = float(frac)
            except ValueError:
                raise ValueError(f"Invalid mole fraction for custom base fuel component {i + 1}.")

            if mole_frac <= 0 or mole_frac > 1:
                raise ValueError(
                    f"Mole fraction for custom base fuel component {i + 1} must be between 0 and 1."
                )

            if total_fraction + mole_frac > 1.0001:
                raise ValueError(
                    f"Custom base fuel mole fractions exceed 1.0. "
                    f"Current total would be {total_fraction + mole_frac:.4f}."
                )

            base_fuel_smiles.append(smiles)
            base_fuel_fractions.append(mole_frac)
            total_fraction += mole_frac

        if len(base_fuel_smiles) < 2:
            raise ValueError("Custom base fuel must contain at least two components.")

        if len(base_fuel_smiles) > 10:
            raise ValueError(
                "Custom base fuel supports up to 10 components because the mixture model "
                "supports 11 total components including the generated additive."
            )

        if total_fraction <= 0:
            raise ValueError("Custom base fuel mole fractions must sum to more than 0.")

        # Match CLI behaviour: normalise fractions to exactly 1.0 if needed
        if abs(total_fraction - 1.0) > 1e-6:
            base_fuel_fractions = [f / total_fraction for f in base_fuel_fractions]

    elif base_fuel_type not in ["fossil_diesel", "biodiesel"]:
        # Match CLI default behaviour for invalid choices
        base_fuel_type = "fossil_diesel"
        base_fuel_smiles = None
        base_fuel_fractions = None

    try:
        additive_fraction = float(form.get("additive_fraction", "0.15"))
    except ValueError:
        raise ValueError("Invalid additive fraction. Please enter a number.")

    # The CLI prompt says 0.01–0.30, but the actual validation accepts 0 < f < 1.
    # To follow the actual CLI logic exactly, use 0 < f < 1.
    if additive_fraction <= 0 or additive_fraction >= 1:
        raise ValueError("Additive fraction must be between 0 and 1.")

    minimize_ysi = form.get("minimize_ysi") == "on"

    mixture_cfg = MixtureConfig(
        additive_fraction=additive_fraction,
        base_fuel_fraction=1.0 - additive_fraction,
        base_fuel_type=base_fuel_type,
        base_fuel_smiles=base_fuel_smiles,
        base_fuel_mole_fractions=base_fuel_fractions,
        target_mixture_dcn=target_dcn
    )

    return EvolutionConfig(
        target_cn=target_dcn,
        maximize_cn=maximize_cn,
        minimize_ysi=minimize_ysi,
        mixture_mode=True,
        mixture_config=mixture_cfg
    )

def rename_mixture_cn_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same logic as results.py _rename_cn().
    """
    if df is None:
        return df
    return df.rename(columns={"cn": "mixture_cn"}) if "cn" in df.columns else df

def mixture_generator_results_to_tables(final_df, pareto_df, unfiltered_df, config):
    """
    Flask version of results.py display_results().
    Shows:
    - final_df: best candidates with property constraints
    - unfiltered_df: best candidates without property constraints
    - pareto_df: Pareto front only if minimize_ysi is enabled
    """
    final_df = rename_mixture_cn_column(final_df)
    pareto_df = rename_mixture_cn_column(pareto_df)
    unfiltered_df = rename_mixture_cn_column(unfiltered_df)

    cols = [
        "rank",
        "smiles",
        "mixture_cn",
        "mixture_ysi",
        "mixture_bp",
        "mixture_density",
        "cn_error",
        "bp",
        "density",
        "lhv",
        "dynamic_viscosity"
    ]

    if config.maximize_cn:
        cols = [c for c in cols if c != "cn_error"]

    final_table = None
    unfiltered_table = None
    pareto_table = None

    if final_df is not None and not final_df.empty:
        available_cols = [c for c in cols if c in final_df.columns]
        final_table = final_df.head(10)[available_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    if unfiltered_df is not None and not unfiltered_df.empty:
        unfiltered_cols = [c for c in cols if c in unfiltered_df.columns]
        unfiltered_table = unfiltered_df.head(10)[unfiltered_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    if config.minimize_ysi and pareto_df is not None and not pareto_df.empty:
        available_pareto_cols = [c for c in cols if c in pareto_df.columns]
        pareto_table = pareto_df.head(20)[available_pareto_cols].to_html(
            index=False,
            classes="table table-striped table-sm"
        )

    return final_table, unfiltered_table, pareto_table, final_df, pareto_df, unfiltered_df
# -----------------
# SCREENING HELPERS
# -----------------
@dataclass
class ScreeningConfig:
    csv_path: str
    mode: str
    target_cn: float
    filters: Dict = field(default_factory=lambda: {
        "bp": (60.0, 250.0),
        "density": (720.0, None),
        "lhv": (30.0, None),
        "dynamic_viscosity": (2.0, None),
    })


def load_screening_smiles(csv_path: str):
    df = pd.read_csv(csv_path)

    smiles_col = next((c for c in df.columns if c.lower() == "smiles"), None)

    if smiles_col is None:
        raise ValueError(
            f"No 'smiles' column found. Your CSV columns are: {list(df.columns)}"
        )

    df = df.rename(columns={smiles_col: "smiles"})
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    return df["smiles"].tolist(), df


def passes_screening_filters(row: dict, filters: dict) -> bool:
    for prop, (lo, hi) in filters.items():
        val = row.get(prop)

        if val is None:
            continue

        if lo is not None and val < lo:
            return False

        if hi is not None and val > hi:
            return False

    return True


def compute_screening_pareto_df(df: pd.DataFrame, cn_col: str, ysi_col: str) -> pd.DataFrame:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    valid = df[cn_col].notna() & df[ysi_col].notna()

    if valid.sum() < 2:
        out = df[valid].copy().reset_index(drop=True)
        out = out.drop(columns=["rank"], errors="ignore")
        out.insert(0, "rank", range(1, len(out) + 1))
        return out

    valid_df = df[valid].reset_index(drop=True)
    F = np.column_stack([valid_df[cn_col].values, valid_df[ysi_col].values])

    front0 = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pareto = valid_df.iloc[front0].sort_values(cn_col).reset_index(drop=True)

    pareto = pareto.drop(columns=["rank"], errors="ignore")
    pareto.insert(0, "rank", range(1, len(pareto) + 1))

    return pareto


def run_pure_screening_in_app(config: ScreeningConfig, smiles_list: List[str]):
    
    predictor = get_property_predictor()
    preds = predictor.predict_all_properties(smiles_list)

    rows = []

    for i, smi in enumerate(smiles_list):
        cn = preds["cn"][i]

        if cn is None:
            continue

        rows.append({
            "smiles": smi,
            "cn": round(float(cn), 3),
            "cn_error": round(abs(float(cn) - config.target_cn), 3),
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

    filtered_rows = [r for r in rows if passes_screening_filters(r, config.filters)]

    if filtered_rows:
        filtered_df = pd.DataFrame(filtered_rows).sort_values("cn_error").reset_index(drop=True)
        filtered_df.insert(0, "rank", range(1, len(filtered_df) + 1))
    else:
        filtered_df = pd.DataFrame(columns=all_df.columns)

    pareto_source = filtered_df if not filtered_df.empty else all_df
    pareto_df = compute_screening_pareto_df(pareto_source, "cn_error", "ysi")

    return filtered_df, pareto_df, all_df

# -----------------------
# BINARY BLENDING HELPERS
# -----------------------
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class BlendSweepResult:
    additive_smiles: str
    base_fuel_type: str
    base_fuel_smiles: List[str]
    base_fuel_mole_fractions: List[float]
    mole_fractions: List[float]
    dcn: List[Optional[float]]
    ysi: List[Optional[float]]

    def to_dict(self):
        return {
            "additive_smiles": self.additive_smiles,
            "base_fuel_type": self.base_fuel_type,
            "base_fuel_smiles": self.base_fuel_smiles,
            "base_fuel_mole_fractions": self.base_fuel_mole_fractions,
            "sweep": [
                {
                    "mole_fraction": x,
                    "dcn": dcn,
                    "ysi": ysi
                }
                for x, dcn, ysi in zip(self.mole_fractions, self.dcn, self.ysi)
            ],
        }


def run_binary_blend_sweep_in_app(
    additive_smiles,
    base_fuel_type="fossil_diesel",
    base_fuel_smiles=None,
    base_fuel_mole_fractions=None,
    min_fraction=0.0,
    max_fraction=0.30,
    n_steps=20,
):
    from rdkit import Chem
    from core.base_fuel_library import BaseFuelLibrary
    from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
    from core.blending.blending_law import blend_ysi_mass_weighted

    if Chem.MolFromSmiles(additive_smiles) is None:
        raise ValueError(f"Invalid additive SMILES: {additive_smiles}")

    if not 0.0 <= min_fraction < max_fraction <= 1.0:
        raise ValueError("Require 0 ≤ min fraction < max fraction ≤ 1.")

    if n_steps < 2:
        raise ValueError("Number of steps must be at least 2.")

    # Resolve base fuel composition
    if base_fuel_type == "custom":
        if not base_fuel_smiles or not base_fuel_mole_fractions:
            raise ValueError("Custom base fuel requires component SMILES and mole fractions.")
        base_smiles = base_fuel_smiles
        base_fracs = base_fuel_mole_fractions
    else:
        base_smiles, base_fracs = BaseFuelLibrary.get_base_fuel(base_fuel_type)

    if 1 + len(base_smiles) > 11:
        raise ValueError(
            f"Total components ({1 + len(base_smiles)}) exceed the GNN limit of 11."
        )

    sweep_fractions = list(np.linspace(min_fraction, max_fraction, n_steps))

    dcn_predictor = get_mixture_dcn_predictor()

    # Use your existing pure property functions instead of loading PropertyPredictor again
    all_smiles = [additive_smiles] + base_smiles
    pure_ysi = [predict_ysi(smi) for smi in all_smiles]

    additive_ysi = pure_ysi[0]
    base_ysi = pure_ysi[1:]

    dcn_results = []
    ysi_results = []

    for f in sweep_fractions:
        adjusted_base_fracs = [bf * (1.0 - f) for bf in base_fracs]

        mixture_smiles = [additive_smiles] + base_smiles
        mixture_fracs = [f] + adjusted_base_fracs

        try:
            dcn = dcn_predictor.predict_mixture_dcn(mixture_smiles, mixture_fracs)
        except Exception:
            dcn = None

        # At f = 0, only use base fuel YSI
        if f == 0.0:
            ysi = blend_ysi_mass_weighted(base_smiles, base_fracs, base_ysi)
        else:
            ysi = blend_ysi_mass_weighted(
                mixture_smiles,
                mixture_fracs,
                [additive_ysi] + base_ysi
            )

        dcn_results.append(None if dcn is None else float(dcn))
        ysi_results.append(None if ysi is None else float(ysi))

    return BlendSweepResult(
        additive_smiles=additive_smiles,
        base_fuel_type=base_fuel_type,
        base_fuel_smiles=base_smiles,
        base_fuel_mole_fractions=base_fracs,
        mole_fractions=[float(x) for x in sweep_fractions],
        dcn=dcn_results,
        ysi=ysi_results,
    )


def blend_sweep_to_dataframe(result):
    rows = []

    for x, dcn, ysi in zip(result.mole_fractions, result.dcn, result.ysi):
        rows.append({
            "Additive mole fraction": round(float(x), 4),
            "Additive %": round(float(x) * 100, 2),
            "Predicted DCN": safe_round(dcn, 2),
            "Predicted YSI": safe_round(ysi, 2),
        })

    return pd.DataFrame(rows)

# Run Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/pure", methods=["GET", "POST"])
def pure_predictor():
    results = []
    error = None

    #------------------
    # CSV FILE INPUT
    #-------------------
    if request.method == "POST" and request.form.get("mode") == "csv":
        csv_file = request.files.get("csv_file")

        if not csv_file:
            error = "No CSV file uploaded."
            return render_template("pure_predictor.html", results=results, error=error)

        try:
            df = pd.read_csv(csv_file)

            if "SMILES" not in df.columns:
                error = "CSV must contain a 'SMILES' column."
                return render_template("pure_predictor.html", results=results, error=error)

            for i, row in df.iterrows():
                raw_name = row.get("IUPAC names", "")
                if pd.isna(raw_name):
                    name = ""
                else:
                    name = str(raw_name).strip()

                raw_smiles = row.get("SMILES", "")
                if pd.isna(raw_smiles):
                    smiles = ""
                else:
                    smiles = str(raw_smiles).strip()

                entry = {
                    "name": name if name else "-",
                    "smiles": smiles,
                    "dcn": None,
                    "ysi": None,
                    "bp": None,
                    "density": None, 
                    "lhv": None, 
                    "dynamic_viscosity": None,
                    "tanimoto": None,
                    "error": None,
                    "img_id": None,
                    "measured_dcn": None,
                    "measured_ysi": None
                }

            
                # STEP 1 — If SMILES empty → convert NAME → SMILES
                if smiles == "" and name not in ("", None, "-"):
                    final_smiles = pubchem_name_to_smiles(name)
                    if final_smiles is None:
                        entry["error"] = "Name not found in PubChem"
                        results.append(entry)
                        continue
                else:
                    final_smiles = smiles

               
                # STEP 2 — Validate SMILES
                if not validate_smiles(final_smiles):
                    entry["error"] = "Invalid SMILES"
                    results.append(entry)
                    continue

                entry["smiles"] = final_smiles

                
                # STEP 3 — Convert SMILES → IUPAC name
                iupac_name = pubchem_smiles_to_name(final_smiles)
                if (not name or name == "-") and iupac_name:
                    entry["name"] = iupac_name

                # Extra step for measured value
                meas = fetch_measured(name=entry["name"], smiles=final_smiles)

                entry["measured_dcn"] = None if meas["measured_dcn"] is None else round(float(meas["measured_dcn"]), 2)
                entry["measured_ysi"] = None if meas["measured_ysi"] is None else round(float(meas["measured_ysi"]), 2)
                
                # STEP 4 — Predict DCN
                props = predict_all_properties_for_smiles([final_smiles])
                
                pred_cn = get_first_prediction(props, "cn")
                pred_ysi = get_first_prediction(props, "ysi")
                pred_bp = get_first_prediction(props, "bp")
                pred_density = get_first_prediction(props, "density")
                pred_lhv = get_first_prediction(props, "lhv")
                pred_visc = get_first_prediction(props, "dynamic_viscosity")
                pred_tanimoto = get_first_prediction(props, "tanimoto")

                if pred_cn is None and pred_ysi is None:
                    entry["error"] = "Prediction failed"
                else:
                    entry["dcn"] = round(pred_cn, 2) if pred_cn is not None else None
                    entry["ysi"] = round(pred_ysi, 2) if pred_ysi is not None else None
                    entry["bp"] = round(pred_bp, 2) if pred_bp is not None else None
                    entry["density"] = round(pred_density, 2) if pred_density is not None else None
                    entry["lhv"] = round(pred_lhv, 2) if pred_lhv is not None else None
                    entry["dynamic_viscosity"] = round(pred_visc, 2) if pred_visc is not None else None
                    entry["tanimoto"] = round(float(pred_tanimoto), 4) if pred_tanimoto is not None else None

                    mol = Chem.MolFromSmiles(final_smiles)
                    img = Draw.MolToImage(mol, size=(300, 250))

                    img_filename = f"mol_csv_{i}.png"
                    img_path = os.path.join("static", "generated", img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    img.save(img_path)

                    entry["img_id"] = img_filename

                results.append(entry)

            return render_template("pure_predictor.html", results=results)

        except Exception as e:
            error = f"Failed to read CSV file: {e}"
            return render_template("pure_predictor.html", results=results, error=error)

    #------------------
    # MANUAL INPUT
    #------------------
    elif request.method == "POST":
        names = request.form.getlist("fuel_name[]")
        smiles_list = request.form.getlist("smiles[]")

        for i, (name, smiles) in enumerate(zip(names, smiles_list)):
            name = name.strip()
            smiles = smiles.strip()

            entry = {
                "name": name if name else "-",
                "smiles": smiles,
                "dcn": None,
                "ysi": None,
                "bp": None,
                "density": None,
                "lhv": None,
                "dynamic_viscosity": None,
                "tanimoto": None,
                "error": None,
                "img_id": None,
                "measured_dcn": None,
                "measured_ysi": None
            }

            
            # STEP 1 — If SMILES empty → convert NAME → SMILES
            if smiles == "" and name not in ("", None, "-"):
                final_smiles = pubchem_name_to_smiles(name)
                if final_smiles is None:
                    entry["error"] = "Name not found in PubChem"
                    results.append(entry)
                    continue
            else:
                final_smiles = smiles

            
            # STEP 2 — Validate SMILES
            if not validate_smiles(final_smiles):
                entry["error"] = "Invalid SMILES"
                results.append(entry)
                continue

            entry["smiles"] = final_smiles

            
            # STEP 3 — Convert SMILES → IUPAC name
            iupac_name = pubchem_smiles_to_name(final_smiles)
            if (not name or name == "-") and iupac_name:
                entry["name"] = iupac_name

            # Extra step for measured value
            meas = fetch_measured(name=entry["name"], smiles=final_smiles)

            entry["measured_dcn"] = None if meas["measured_dcn"] is None else round(float(meas["measured_dcn"]), 2)
            entry["measured_ysi"] = None if meas["measured_ysi"] is None else round(float(meas["measured_ysi"]), 2)

            # STEP 4 — Predict & draw molecule
            props = predict_all_properties_for_smiles([final_smiles])

            pred_cn = get_first_prediction(props, "cn")
            pred_ysi = get_first_prediction(props, "ysi")
            pred_bp = get_first_prediction(props, "bp")
            pred_density = get_first_prediction(props, "density")
            pred_lhv = get_first_prediction(props, "lhv")
            pred_visc = get_first_prediction(props, "dynamic_viscosity")
            pred_tanimoto = get_first_prediction(props, "tanimoto")

            if pred_cn is None and pred_ysi is None:
                entry["error"] = "Prediction failed"
            else:
                entry["dcn"] = round(pred_cn, 2) if pred_cn is not None else None
                entry["ysi"] = round(pred_ysi, 2) if pred_ysi is not None else None
                entry["bp"] = round(pred_bp, 2) if pred_bp is not None else None
                entry["density"] = round(pred_density, 4) if pred_density is not None else None
                entry["lhv"] = round(pred_lhv, 2) if pred_lhv is not None else None
                entry["dynamic_viscosity"] = round(pred_visc, 4) if pred_visc is not None else None
                entry["tanimoto"] = round(float(pred_tanimoto), 4) if pred_tanimoto is not None else None

                mol = Chem.MolFromSmiles(final_smiles)
                img = Draw.MolToImage(mol, size=(300, 250))

                img_filename = f"mol_{i}.png"
                img_path = os.path.join("static", "generated", img_filename)
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                img.save(img_path)

                entry["img_id"] = img_filename

            results.append(entry)

    return render_template("pure_predictor.html", results=results, error=error)


@app.route("/download_results", methods=["POST"])
def download_results():
    import io, json
    results_json = request.form.get("results_data")
    results = json.loads(results_json)

    cleaned_rows = []

    for r in results:
        cleaned_rows.append({
            "IUPAC Name": r.get("name", "-"),
            "SMILES": r.get("smiles", "-"),
            "Predicted DCN": r.get("dcn", None),
            "Predicted YSI": r.get("ysi", None),
            "Predicted BP": r.get("bp", None),
            "Predicted Density": r.get("density", None),
            "Predicted LHV": r.get("lhv", None),
            "Predicted Dynamic Viscosity": r.get("dynamic_viscosity", None),
            "Tanimoto": r.get("tanimoto", None),
            "Status": ("OK" if r.get("error") in (None, "", "OK") else r.get("error"))
        })

    df = pd.DataFrame(cleaned_rows)

    # column order
    df = df[["IUPAC Name", "SMILES", "Predicted DCN", 
             "Predicted YSI", "Predicted BP", "Predicted Density",
             "Predicted LHV", "Predicted Dynamic Viscosity", "Tanimoto",
             "Status"
            ]]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="pure_fuel_predictions.csv"
    )

@app.route("/mixture", methods=["GET", "POST"])
def mixture_predictor():
    result = None
    error = None
    warning = None
    components = []

    if request.method == "POST":
        names = request.form.getlist("component_name[]")
        smiles_inputs = request.form.getlist("component_smiles[]")
        percentages = request.form.getlist("component_percentage[]")

        smiles_list = []
        mole_fractions = []
        total_percentage = 0.0

        try:
            for i, (name, smiles, percentage) in enumerate(zip(names, smiles_inputs, percentages)):
                name = name.strip()
                smiles = smiles.strip()
                percentage = percentage.strip()

                # Skip completely empty rows
                if name == "" and smiles == "" and percentage == "":
                    continue

                # If SMILES is empty but name is given, try PubChem
                if smiles == "" and name != "":
                    smiles = pubchem_name_to_smiles(name)
                    if smiles is None:
                        error = f"Could not find SMILES for component {i + 1}: {name}"
                        break

                if smiles == "":
                    error = f"Please enter a SMILES string for component {i + 1}."
                    break

                if not validate_smiles(smiles):
                    error = f"Invalid SMILES for component {i + 1}: {smiles}"
                    break

                try:
                    percentage_value = float(percentage)
                except ValueError:
                    error = f"Please enter a valid percentage for component {i + 1}."
                    break

                if percentage_value <= 0:
                    error = f"Percentage for component {i + 1} must be greater than 0."
                    break

                total_percentage += percentage_value
                mole_fraction = percentage_value / 100.0

                mol_info = get_molecule_summary(smiles)

                # Try to get IUPAC name if user left name blank
                if name == "":
                    found_name = pubchem_smiles_to_name(smiles)
                    name = found_name if found_name else f"Component {i + 1}"

                # Predict pure-component supporting values
                pure_ysi = predict_ysi(smiles)
                pure_density = predict_density(smiles)

                component_entry = {
                    "name": name,
                    "smiles": smiles,
                    "percentage": round(percentage_value, 2),
                    "mole_fraction": round(mole_fraction, 4),
                    "formula": mol_info["formula"],
                    "mol_weight": mol_info["mol_weight"],
                    "num_atoms": mol_info["num_atoms"],
                    "pure_ysi": safe_round(pure_ysi, 2),
                    "pure_density": safe_round(pure_density, 2),
                    "img_id": None
                }

                # Draw molecule image
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(260, 200))
                    img_filename = f"mixture_component_{i}.png"
                    img_path = os.path.join("static", "generated", img_filename)
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    img.save(img_path)
                    component_entry["img_id"] = img_filename

                components.append(component_entry)
                smiles_list.append(smiles)
                mole_fractions.append(mole_fraction)

            if error is None:
                if len(smiles_list) < 2:
                    error = "Please enter at least two mixture components."
                elif abs(total_percentage - 100.0) > 0.01:
                    error = f"Total composition must equal 100%. Current total is {total_percentage:.2f}%."
                else:
                    # Normalise very small floating point differences
                    mole_total = sum(mole_fractions)
                    mole_fractions = [x / mole_total for x in mole_fractions]

                    # 1) Mixture DCN from GNN model
                    mixture_predictor_model = get_mixture_dcn_predictor()
                    mixture_dcn = mixture_predictor_model.predict_mixture_dcn(
                        smiles_list,
                        mole_fractions
                    )

                    # 2) Supporting mixture properties from blending laws
                    ysi_values = [c["pure_ysi"] for c in components]
                    density_values_kgm3 = [c["pure_density"] for c in components]
                    density_values_gcc = [
                        d / 1000.0 if d is not None else None
                        for d in density_values_kgm3
                    ]

                    mixture_ysi = blend_ysi_mass_weighted(
                        smiles_list,
                        mole_fractions,
                        ysi_values
                    )

                    mixture_bp = blend_bp_riazi_daubert(
                        smiles_list,
                        mole_fractions,
                        density_values_gcc
                    )

                    mixture_density_gcc = blend_density(
                        smiles_list,
                        mole_fractions,
                        density_values_gcc
                    )

                    mixture_density = (
                        mixture_density_gcc * 1000.0
                        if mixture_density_gcc is not None
                        else None
                    )

                    result = {
                        "dcn": safe_round(mixture_dcn, 2),
                        "ysi": safe_round(mixture_ysi, 2),
                        "bp": safe_round(mixture_bp, 2),
                        "density": safe_round(mixture_density, 2),
                        "total_percentage": safe_round(total_percentage, 2),
                        "num_components": len(components)
                    }

                    if result["ysi"] is None or result["bp"] is None or result["density"] is None:
                        warning = (
                            "DCN prediction was completed, but one or more supporting "
                            "mixture properties could not be calculated because a component "
                            "property prediction was unavailable."
                        )

        except Exception as e:
            error = f"Mixture prediction failed: {e}"

    return render_template(
        "mixture_predictor.html",
        result=result,
        error=error,
        warning=warning,
        components=components
    )

@app.route("/generate", methods=["GET", "POST"])
def generative():
    final_table = None
    unfiltered_table = None
    pareto_table = None

    error = None
    info = None
    run_info = None

    if request.method == "POST":
        try:
            config = get_generator_config_from_form(request.form)

            evolution = MolecularEvolution(config)
            final_df, pareto_df, unfiltered_df = evolution.evolve()

            final_table, unfiltered_table, pareto_table = generator_results_to_tables(
                final_df,
                pareto_df,
                unfiltered_df,
                config
            )

            # Store CSVs for download
            if final_df is not None and not final_df.empty:
                final_csv = final_df.to_csv(index=False).encode("utf-8")
                session["generator_final_csv"] = base64.b64encode(final_csv).decode("utf-8")

            if unfiltered_df is not None and not unfiltered_df.empty:
                unfiltered_csv = unfiltered_df.to_csv(index=False).encode("utf-8")
                session["generator_unfiltered_csv"] = base64.b64encode(unfiltered_csv).decode("utf-8")

            if pareto_df is not None and not pareto_df.empty:
                pareto_csv = pareto_df.to_csv(index=False).encode("utf-8")
                session["generator_pareto_csv"] = base64.b64encode(pareto_csv).decode("utf-8")

            run_info = {
                "mode": "Maximise CN" if config.maximize_cn else "Target CN",
                "target_cn": None if config.maximize_cn else config.target_cn,
                "minimize_ysi": "Yes" if config.minimize_ysi else "No",
                "objective": "Multi-objective (CN + YSI)" if config.minimize_ysi else "Single-objective (CN only)",
                "n_final": 0 if final_df is None else len(final_df),
                "n_unfiltered": 0 if unfiltered_df is None else len(unfiltered_df),
                "n_pareto": 0 if pareto_df is None else len(pareto_df),
            }

            if (final_df is None or final_df.empty) and (unfiltered_df is None or unfiltered_df.empty):
                info = "Evolution completed, but no valid generated candidates were found."

        except Exception as e:
            error = f"Pure fuel generation failed: {e}"

    return render_template(
        "generative.html",
        final_table=final_table,
        unfiltered_table=unfiltered_table,
        pareto_table=pareto_table,
        error=error,
        info=info,
        run_info=run_info
    )

@app.route("/download/generator/<result_type>")
def download_generator_result(result_type):
    key_map = {
        "final": "generator_final_csv",
        "unfiltered": "generator_unfiltered_csv",
        "pareto": "generator_pareto_csv",
    }

    if result_type not in key_map:
        return "Invalid generator result type.", 404

    csv_b64 = session.get(key_map[result_type])

    if not csv_b64:
        return "No generator result file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"pure_fuel_generator_{result_type}.csv"
    )

@app.route("/generative_mixture", methods=["GET", "POST"])
def generative_mixture():
    final_table = None
    unfiltered_table = None
    pareto_table = None

    error = None
    info = None
    run_info = None

    form_values = {
        "mode": "target",
        "target_dcn": "50",
        "base_fuel_type": "fossil_diesel",
        "additive_fraction": "0.15",
        "minimize_ysi": False,
        "custom_smiles": ["", ""],
        "custom_fraction": ["", ""]
    }

    if request.method == "POST":
        try:
            form_values = {
                "mode": request.form.get("mode", "target"),
                "target_dcn": request.form.get("target_dcn", "50"),
                "base_fuel_type": request.form.get("base_fuel_type", "fossil_diesel"),
                "additive_fraction": request.form.get("additive_fraction", "0.15"),
                "minimize_ysi": request.form.get("minimize_ysi") == "on",
                "custom_smiles": request.form.getlist("custom_smiles[]"),
                "custom_fraction": request.form.getlist("custom_fraction[]")
            }

            # Same role as get_user_config() in CLI, but from Flask form
            config = get_mixture_generator_config_from_form(request.form)

            print("=" * 70, flush=True)
            print("MIXTURE-AWARE MOLECULE GENERATOR", flush=True)
            print("=" * 70, flush=True)
            print("CONFIGURATION SUMMARY:", flush=True)
            print(f"  • Mode: Mixture Optimization", flush=True)
            print(f"  • Target Mixture DCN: {config.target_cn}", flush=True)
            print(f"  • Base Fuel Type: {config.mixture_config.base_fuel_type}", flush=True)
            if config.mixture_config.base_fuel_smiles:
                print(f"  • Base Fuel Components: {len(config.mixture_config.base_fuel_smiles)}", flush=True)
            print(f"  • Additive Fraction: {config.mixture_config.additive_fraction * 100:.1f}%", flush=True)
            print(f"  • Base Fuel Fraction: {config.mixture_config.base_fuel_fraction * 100:.1f}%", flush=True)
            print(f"  • Minimize Mixture YSI: {'Yes (NSGA-II)' if config.minimize_ysi else 'No'}", flush=True)
            print("=" * 70, flush=True)

            os.environ["WANDB_MODE"] = "disabled"
            os.environ["WANDB_DISABLED"] = "true"
            os.environ["WANDB_SILENT"] = "true"

            from core.evolution.mixture_evolution import MixtureAwareMolecularEvolution

            # Same as main.py when config.mixture_mode is True
            evolution = MixtureAwareMolecularEvolution(config)
            final_df, pareto_df, unfiltered_df = evolution.evolve()

            (
                final_table,
                unfiltered_table,
                pareto_table,
                final_df,
                pareto_df,
                unfiltered_df
            ) = mixture_generator_results_to_tables(
                final_df,
                pareto_df,
                unfiltered_df,
                config
            )

            # Flask equivalent of results.py save_results()
            if final_df is not None and not final_df.empty:
                final_csv = final_df.to_csv(index=False).encode("utf-8")
                session["mixture_generator_final_csv"] = base64.b64encode(final_csv).decode("utf-8")

            if unfiltered_df is not None and not unfiltered_df.empty:
                unfiltered_csv = unfiltered_df.to_csv(index=False).encode("utf-8")
                session["mixture_generator_unfiltered_csv"] = base64.b64encode(unfiltered_csv).decode("utf-8")

            if config.minimize_ysi and pareto_df is not None and not pareto_df.empty:
                pareto_csv = pareto_df.to_csv(index=False).encode("utf-8")
                session["mixture_generator_pareto_csv"] = base64.b64encode(pareto_csv).decode("utf-8")

            mixture_cfg = config.mixture_config

            run_info = {
                "mode": "Maximise mixture DCN" if config.maximize_cn else "Target mixture DCN",
                "target_dcn": config.target_cn,
                "base_fuel_type": mixture_cfg.base_fuel_type,
                "base_fuel_components": len(mixture_cfg.base_fuel_smiles) if mixture_cfg.base_fuel_smiles else None,
                "additive_fraction": round(mixture_cfg.additive_fraction, 4),
                "base_fuel_fraction": round(mixture_cfg.base_fuel_fraction, 4),
                "minimize_ysi": "Yes (NSGA-II)" if config.minimize_ysi else "No",
                "objective": "Multi-objective mixture DCN + mixture YSI" if config.minimize_ysi else "Single-objective mixture DCN",
                "n_final": 0 if final_df is None else len(final_df),
                "n_unfiltered": 0 if unfiltered_df is None else len(unfiltered_df),
                "n_pareto": 0 if pareto_df is None else len(pareto_df),
            }

            if (final_df is None or final_df.empty) and (unfiltered_df is None or unfiltered_df.empty):
                info = "Mixture generation completed, but no valid generated blend candidates were found."

        except Exception as e:
            error = f"Mixture generation failed: {e}"

    return render_template(
        "generative_mixture.html",
        final_table=final_table,
        unfiltered_table=unfiltered_table,
        pareto_table=pareto_table,
        error=error,
        info=info,
        run_info=run_info,
        form_values=form_values
    )
    
@app.route("/download/mixture-generator/<result_type>")
def download_mixture_generator_result(result_type):
    key_map = {
        "final": "mixture_generator_final_csv",
        "unfiltered": "mixture_generator_unfiltered_csv",
        "pareto": "mixture_generator_pareto_csv",
    }

    if result_type not in key_map:
        return "Invalid mixture generator result type.", 404

    csv_b64 = session.get(key_map[result_type])

    if not csv_b64:
        return "No mixture generator result file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"mixture_generator_{result_type}.csv"
    )

@app.route("/fame-cn", methods=["GET", "POST"])
def fame_cn_predictor():
    mode = request.form.get("mode", "single")

    prediction_result = None
    error = None
    warning = None
    info = None

    valid_table = None
    invalid_table = None
    evaluation = None
    unsupported_fames = None

    single_values = {fame: 0 for fame in all_training_fames}
    measured_cn = None

    template_b64 = base64.b64encode(make_template_csv()).decode("utf-8")
    example_b64 = base64.b64encode(make_example_csv()).decode("utf-8")

    if request.method == "POST":

        # -------------------------
        # SINGLE SAMPLE MODE
        # -------------------------
        if mode == "single":
            values = {}

            for fame in all_training_fames:
                raw_value = request.form.get(fame, "0")

                try:
                    values[fame] = float(raw_value)
                except ValueError:
                    values[fame] = 0.0

            single_values = values

            try:
                measured_cn = float(request.form.get("measured_cn", "0"))
            except ValueError:
                measured_cn = 0.0

            input_df = pd.DataFrame([values])
            input_df = ensure_fame_columns(input_df)

            valid_df, invalid_df = run_prediction_pipeline(
                input_df,
                fame_cn_model,
                fame_ood_stats
            )

            if len(invalid_df) > 0:
                row = invalid_df.iloc[0]

                if not row["valid_total"]:
                    error = (
                        f"Invalid input: total FAME composition is {row['Total %']:.2f}%. "
                        "It should be between 95 and 105%."
                    )
                elif not row["valid_range"]:
                    error = "Invalid input: all FAME values must be between 0 and 100%."
                elif not row["valid_C18_3"]:
                    error = "Invalid biodiesel input: C18:3 must be ≤ 12%."
                elif not row["valid_C18_2"]:
                    error = "Invalid biodiesel input: C18:2 must be < 70%."
                else:
                    error = "Invalid biodiesel input."

                invalid_table = invalid_df[
                    ["Total %", "valid_total", "valid_range", "valid_C18_3", "valid_C18_2", "is_valid_biodiesel"]
                ].to_html(index=False, classes="table table-striped table-sm")

            elif len(valid_df) > 0:
                pred = float(valid_df["CN_pred"].iloc[0])

                prediction_result = {
                    "predicted_cn": round(pred, 2),
                    "total": round(float(valid_df["Total %"].iloc[0]), 2),
                    "original_total": round(float(valid_df["Original Total %"].iloc[0]), 2),
                    "was_normalized": bool(valid_df["was_normalized"].iloc[0]),
                    "ood_flag": bool(valid_df["ood_flag"].iloc[0]) if "ood_flag" in valid_df.columns else False,
                    "measured_cn": measured_cn if measured_cn and measured_cn > 0 else None,
                    "abs_error": None
                }

                if measured_cn and measured_cn > 0:
                    prediction_result["abs_error"] = round(abs(measured_cn - pred), 2)

                if prediction_result["ood_flag"]:
                    warning = (
                        "This input is outside the training feature range. "
                        "The prediction may be less reliable."
                    )

                if prediction_result["was_normalized"]:
                    info = (
                        f"Input total was {prediction_result['original_total']:.2f}%, "
                        "so values were normalised to 100% before prediction."
                    )
                else:
                    info = "Input composition total was already 100%; no normalisation was needed."

                display_cols = [
                    "CN_pred",
                    "Original Total %",
                    "Total %",
                    "was_normalized",
                    "ood_flag",
                    "avg_carbon",
                    "avg_double_bonds",
                    "SFA",
                    "MUFA",
                    "PUFA",
                    "avg_mw"
                ]

                # Add entered FAME values after summary columns for transparency
                display_cols = display_cols + [f for f in all_training_fames if f in valid_df.columns]

                valid_table = valid_df[display_cols].to_html(
                    index=False,
                    classes="table table-striped table-sm"
                )

        # -------------------------
        # CSV UPLOAD MODE
        # -------------------------
        elif mode == "csv":
            uploaded_file = request.files.get("csv_file")

            if uploaded_file is None or uploaded_file.filename == "":
                error = "No CSV file uploaded."

            else:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)

                    unsupported_fames = detect_unsupported_fames(uploaded_df)

                    valid_df, invalid_df = run_prediction_pipeline(
                        uploaded_df,
                        fame_cn_model,
                        fame_ood_stats
                    )

                    if len(valid_df) > 0:
                        prediction_display_cols = [
                            "CN_pred",
                            "Original Total %",
                            "Total %",
                            "was_normalized",
                            "ood_flag",
                            "avg_carbon",
                            "avg_double_bonds",
                            "SFA",
                            "MUFA",
                            "PUFA",
                            "avg_mw"
                        ]

                        prediction_display_cols += [
                            f for f in all_training_fames if f in valid_df.columns
                        ]

                        if "Name" in valid_df.columns:
                            prediction_display_cols = ["Name"] + prediction_display_cols

                        if "CN" in valid_df.columns:
                            prediction_display_cols = ["CN"] + prediction_display_cols

                        valid_table = valid_df[prediction_display_cols].to_html(
                            index=False,
                            classes="table table-striped table-sm"
                        )

                        valid_csv = valid_df.to_csv(index=False).encode("utf-8")
                        session["fame_valid_csv"] = base64.b64encode(valid_csv).decode("utf-8")

                        if "CN" in valid_df.columns:
                            eval_df = valid_df.dropna(subset=["CN", "CN_pred"]).copy()

                            if len(eval_df) >= 2:
                                y_true = eval_df["CN"]
                                y_pred = eval_df["CN_pred"]

                                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                                mae = mean_absolute_error(y_true, y_pred)
                                r2 = r2_score(y_true, y_pred)

                                eval_df["residual"] = eval_df["CN"] - eval_df["CN_pred"]
                                eval_df["abs_error"] = eval_df["residual"].abs()

                                eval_cols = ["CN", "CN_pred", "residual", "abs_error"]
                                if "Name" in eval_df.columns:
                                    eval_cols = ["Name"] + eval_cols

                                evaluation = {
                                    "rmse": round(float(rmse), 2),
                                    "mae": round(float(mae), 2),
                                    "r2": round(float(r2), 3),
                                    "table": eval_df[eval_cols].to_html(
                                        index=False,
                                        classes="table table-striped table-sm"
                                    )
                                }

                    if len(invalid_df) > 0:
                        invalid_display_cols = [
                            "Total %",
                            "valid_total",
                            "valid_range",
                            "valid_C18_3",
                            "valid_C18_2",
                            "is_valid_biodiesel"
                        ]

                        invalid_display_cols += [
                            f for f in all_training_fames if f in invalid_df.columns
                        ]

                        if "Name" in invalid_df.columns:
                            invalid_display_cols = ["Name"] + invalid_display_cols

                        if "CN" in invalid_df.columns:
                            invalid_display_cols = ["CN"] + invalid_display_cols

                        invalid_table = invalid_df[invalid_display_cols].to_html(
                            index=False,
                            classes="table table-striped table-sm"
                        )

                        invalid_csv = invalid_df.to_csv(index=False).encode("utf-8")
                        session["fame_invalid_csv"] = base64.b64encode(invalid_csv).decode("utf-8")

                    if len(valid_df) == 0 and len(invalid_df) == 0:
                        info = "No samples were processed from the uploaded CSV."

                except Exception as e:
                    error = f"Failed to process CSV file: {e}"

    return render_template(
        "fame_cn_predictor.html",
        mode=mode,
        fame_groups=fame_groups,
        all_training_fames=all_training_fames,
        single_values=single_values,
        measured_cn=measured_cn,
        prediction_result=prediction_result,
        error=error,
        warning=warning,
        info=info,
        valid_table=valid_table,
        invalid_table=invalid_table,
        evaluation=evaluation,
        unsupported_fames=unsupported_fames,
        template_b64=template_b64,
        example_b64=example_b64
    )

@app.route("/download/fame-valid")
def download_fame_valid():
    csv_b64 = session.get("fame_valid_csv")

    if not csv_b64:
        return "No valid prediction file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="biodiesel_CN_predictions.csv"
    )

@app.route("/download/fame-invalid")
def download_fame_invalid():
    csv_b64 = session.get("fame_invalid_csv")

    if not csv_b64:
        return "No invalid sample file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="invalid_biodiesel_inputs.csv"
    )

@app.route("/screening", methods=["GET", "POST"])
def screening():
    error = None
    info = None

    filtered_table = None
    pareto_table = None
    all_table = None
    run_info = None

    if request.method == "POST":
        uploaded_file = request.files.get("csv_file")
        mode = request.form.get("mode", "pure_component")

        try:
            target_cn = float(request.form.get("target_cn", "50"))
        except ValueError:
            target_cn = 50.0

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please upload a CSV file."

        elif mode != "pure_component":
            error = "For now, this app.py-only version supports pure component screening only."

        else:
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    uploaded_file.save(tmp.name)
                    temp_path = tmp.name

                filters = {
                    "bp": (
                        float(request.form.get("bp_min", 60) or 60),
                        float(request.form.get("bp_max", 250) or 250),
                    ),
                    "density": (
                        float(request.form.get("density_min", 720) or 720),
                        None,
                    ),
                    "lhv": (
                        float(request.form.get("lhv_min", 30) or 30),
                        None,
                    ),
                    "dynamic_viscosity": (
                        float(request.form.get("viscosity_min", 2) or 2),
                        None,
                    ),
                }

                config = ScreeningConfig(
                    csv_path=temp_path,
                    mode=mode,
                    target_cn=target_cn,
                    filters=filters
                )

                smiles_list, _ = load_screening_smiles(temp_path)

                filtered_df, pareto_df, all_df = run_pure_screening_in_app(
                    config,
                    smiles_list
                )

                if filtered_df is not None and not filtered_df.empty:
                    filtered_table = filtered_df.head(50).to_html(
                        index=False,
                        classes="table table-striped table-sm"
                    )

                    filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
                    session["screening_filtered_csv"] = base64.b64encode(filtered_csv).decode("utf-8")

                if pareto_df is not None and not pareto_df.empty:
                    pareto_table = pareto_df.head(50).to_html(
                        index=False,
                        classes="table table-striped table-sm"
                    )

                    pareto_csv = pareto_df.to_csv(index=False).encode("utf-8")
                    session["screening_pareto_csv"] = base64.b64encode(pareto_csv).decode("utf-8")

                if all_df is not None and not all_df.empty:
                    all_table = all_df.head(50).to_html(
                        index=False,
                        classes="table table-striped table-sm"
                    )

                    all_csv = all_df.to_csv(index=False).encode("utf-8")
                    session["screening_all_csv"] = base64.b64encode(all_csv).decode("utf-8")

                if all_df is None or all_df.empty:
                    info = "Screening completed, but no valid candidates were found."

                run_info = {
                    "mode": "Pure Component Screening",
                    "target_cn": target_cn,
                    "n_filtered": 0 if filtered_df is None else len(filtered_df),
                    "n_pareto": 0 if pareto_df is None else len(pareto_df),
                    "n_all": 0 if all_df is None else len(all_df),
                }

            except Exception as e:
                error = f"Screening failed: {e}"

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

    return render_template(
        "screening.html",
        error=error,
        info=info,
        filtered_table=filtered_table,
        pareto_table=pareto_table,
        all_table=all_table,
        run_info=run_info
    )

@app.route("/download/screening/<result_type>")
def download_screening_result(result_type):
    key_map = {
        "filtered": "screening_filtered_csv",
        "pareto": "screening_pareto_csv",
        "all": "screening_all_csv",
    }

    if result_type not in key_map:
        return "Invalid screening result type.", 404

    csv_b64 = session.get(key_map[result_type])

    if not csv_b64:
        return "No screening result file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"screening_{result_type}_results.csv"
    )

@app.route("/binary-blending", methods=["GET", "POST"])
def binary_blending():
    error = None
    warning = None
    result = None
    result_table = None
    chart_data = None

    form_values = {
        "additive_smiles": "",
        "base_fuel_type": "fossil_diesel",
        "min_fraction": 0.0,
        "max_fraction": 0.30,
        "n_steps": 20,
    }

    if request.method == "POST":
        additive_smiles = request.form.get("additive_smiles", "").strip()
        base_fuel_type = request.form.get("base_fuel_type", "fossil_diesel")

        try:
            min_fraction = float(request.form.get("min_fraction", 0.0))
            max_fraction = float(request.form.get("max_fraction", 0.30))
            n_steps = int(request.form.get("n_steps", 20))
        except ValueError:
            error = "Please enter valid numerical sweep settings."
            min_fraction = 0.0
            max_fraction = 0.30
            n_steps = 20

        form_values = {
            "additive_smiles": additive_smiles,
            "base_fuel_type": base_fuel_type,
            "min_fraction": min_fraction,
            "max_fraction": max_fraction,
            "n_steps": n_steps,
        }

        if error is None:
            try:
                sweep_result = run_binary_blend_sweep_in_app(
                    additive_smiles=additive_smiles,
                    base_fuel_type=base_fuel_type,
                    min_fraction=min_fraction,
                    max_fraction=max_fraction,
                    n_steps=n_steps,
                )

                df = blend_sweep_to_dataframe(sweep_result)

                result_table = df.to_html(
                    index=False,
                    classes="table table-striped table-sm"
                )

                csv_data = df.to_csv(index=False).encode("utf-8")
                session["binary_blending_csv"] = base64.b64encode(csv_data).decode("utf-8")

                result = {
                    "additive_smiles": sweep_result.additive_smiles,
                    "base_fuel_type": sweep_result.base_fuel_type,
                    "n_points": len(sweep_result.mole_fractions),
                    "min_dcn": safe_round(min([x for x in sweep_result.dcn if x is not None], default=None), 2),
                    "max_dcn": safe_round(max([x for x in sweep_result.dcn if x is not None], default=None), 2),
                    "min_ysi": safe_round(min([x for x in sweep_result.ysi if x is not None], default=None), 2),
                    "max_ysi": safe_round(max([x for x in sweep_result.ysi if x is not None], default=None), 2),
                }

                chart_data = sweep_result.to_dict()

                if result["min_dcn"] is None:
                    warning = "Sweep completed, but DCN prediction failed for all blend points."

            except Exception as e:
                error = f"Binary blending failed: {e}"

    return render_template(
        "binary_blending.html",
        error=error,
        warning=warning,
        result=result,
        result_table=result_table,
        chart_data=chart_data,
        form_values=form_values
    )

@app.route("/download/binary-blending")
def download_binary_blending():
    csv_b64 = session.get("binary_blending_csv")

    if not csv_b64:
        return "No binary blending result file available.", 404

    csv_data = base64.b64decode(csv_b64)

    return send_file(
        io.BytesIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="binary_blending_sweep.csv"
    )
    
@app.route("/constraints")
def constraints():
    return render_template("constraints.html")

@app.route("/experimental_setup")
def experimental_setup():
    return render_template("experimental_setup.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/download/pure")
def download_pure():
    return send_file(
        "datasets/pure_fuel_properties_compiled_v2.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="pure_fuel_dataset.xlsx"
    )

@app.route("/download/mixture")
def download_mixture():
    return send_file(
        "datasets/mixture_database.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="mixture_fuel_dataset.xlsx"
    )

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
