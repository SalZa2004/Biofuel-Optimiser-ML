"""
SHAP interpretability analysis for pure-component property prediction models.

Each model is an ExtraTreesRegressor trained on Morgan fingerprints (2048 bits)
+ RDKit physicochemical descriptors, with a FeatureSelector that drops correlated
descriptors and keeps the top-300 most important features.

shap.TreeExplainer gives exact Shapley values for tree ensembles — no ablation
approximation needed (unlike graph-neural-network approaches).

Outputs per property (in scripts/analysis/shap_plots/):
  <property>_beeswarm.png  — SHAP beeswarm coloured by feature value
  <property>_bar.png       — mean |SHAP| bar chart (global importance)
  <property>_shap.csv      — raw SHAP matrix for further analysis
"""

import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from core.shared_features import FeatureSelector, DESCRIPTOR_NAMES

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent / "shap_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── RDKit descriptor names (same order as featurize()) ────────────────────────
DESC_NAMES = [d[0] for d in Descriptors._descList]
N_MORGAN = 2048

# ── Model registry ─────────────────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "data" / "database" / "database_main.db"

MODELS = {
    "cn": {
        "artifacts": PROJECT_ROOT / "models/pure_component/cn_predictor_model/cn_model/artifacts",
        "label": "Cetane Number (CN)",
        "unit": "",
        "loader": "db",
        "db_query": """
            SELECT F.SMILES, T.Standardised_DCN AS target
            FROM FUEL F
            LEFT JOIN TARGET T ON F.fuel_id = T.fuel_id
        """,
        "target_range": (0, 150),
    },
    "ysi": {
        "artifacts": PROJECT_ROOT / "models/pure_component/ysi_predictor_model/ysi_model/artifacts",
        "label": "Yield Sooting Index (YSI)",
        "unit": "",
        "loader": "db",
        "db_query": """
            SELECT F.SMILES, T.Standardised_YSI AS target
            FROM FUEL F
            LEFT JOIN TARGET T ON F.fuel_id = T.fuel_id
        """,
        "target_range": None,
    },
    "density": {
        "artifacts": PROJECT_ROOT / "models/pure_component/density_predictor_model/density_model/artifacts",
        "label": "Density (kg/m³)",
        "unit": "kg/m³",
        "loader": "csv",
        "csv": PROJECT_ROOT / "models/pure_component/density_predictor_model/data/density.csv",
        "target_col": "density",
        "target_range": None,
    },
    "dynamic_viscosity": {
        "artifacts": PROJECT_ROOT / "models/pure_component/dynamic_viscosity_predictor_model/dynamic_viscosity_model/artifacts",
        "label": "Dynamic Viscosity — log₁₀(mPa·s)",
        "unit": "log₁₀(mPa·s)",
        "loader": "csv",
        "csv": PROJECT_ROOT / "models/pure_component/dynamic_viscosity_predictor_model/data/dynamic_viscosity.csv",
        "target_col": "dynamic_viscosity",
        "log_transform": True,
        "target_range": None,
    },
    "lhv": {
        "artifacts": PROJECT_ROOT / "models/pure_component/lhv_predictor_model/lhv_model/artifacts",
        "label": "Lower Heating Value (MJ/kg)",
        "unit": "MJ/kg",
        "loader": "csv",
        "csv": PROJECT_ROOT / "models/pure_component/lhv_predictor_model/data/lhv.csv",
        "target_col": "lhv",
        "target_range": None,
    },
    "bp": {
        "artifacts": PROJECT_ROOT / "models/pure_component/bp_predictor_model/bp_model/artifacts",
        "label": "Boiling Point (°C)",
        "unit": "°C",
        "loader": "csv",
        "csv": PROJECT_ROOT / "models/pure_component/bp_predictor_model/data/bp_data.csv",
        "target_col": "bp",
        "target_range": None,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_feature_names(selector: FeatureSelector) -> list[str]:
    """
    Reconstruct the name of each selected feature from the FeatureSelector.

    The full feature vector is:
      [Morgan_0 .. Morgan_2047 | rdkit_desc_0 .. rdkit_desc_207]

    FeatureSelector:
      1. Drops descriptor columns in selector.corr_cols_to_drop (integer indices)
      2. Concatenates Morgan + remaining descriptors → X_corr
      3. Picks selector.selected_indices (indices into X_corr)
    """
    morgan_names = [f"Morgan_bit_{i}" for i in range(N_MORGAN)]
    remaining_desc_names = [
        DESC_NAMES[i]
        for i in range(len(DESC_NAMES))
        if i not in set(selector.corr_cols_to_drop)
    ]
    after_corr_names = morgan_names + remaining_desc_names
    return [after_corr_names[i] for i in selector.selected_indices]


def load_smiles(config: dict) -> pd.DataFrame:
    """Load SMILES (and optionally targets) for a property."""
    if config["loader"] == "db":
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(config["db_query"], conn)
        conn.close()
        df = df.dropna(subset=["SMILES", "target"])
        if config.get("target_range"):
            lo, hi = config["target_range"]
            df = df[(df["target"] >= lo) & (df["target"] <= hi)]
    else:
        df = pd.read_csv(config["csv"])
        df = df.rename(columns={config["target_col"]: "target"})
        df = df.dropna(subset=["SMILES", "target"])
    return df.reset_index(drop=True)


def featurize_smiles(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """Morgan (2048) + RDKit descriptors → (X, valid_indices)."""
    from rdkit.Chem import rdFingerprintGenerator
    desc_fns = [d[1] for d in Descriptors._descList]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_MORGAN)

    rows, valid = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc="  Featurizing")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.array(list(fpgen.GetFingerprint(mol).ToBitString()), dtype=np.float32)
        try:
            desc = np.array([fn(mol) for fn in desc_fns], dtype=np.float32)
            desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            continue
        rows.append(np.hstack([fp, desc]))
        valid.append(i)
    return np.vstack(rows), valid


def shorten_name(name: str, max_len: int = 30) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


# ── Per-property analysis ──────────────────────────────────────────────────────

def analyze_property(prop: str, config: dict, max_samples: int = 500) -> None:
    print(f"\n{'='*60}")
    print(f"  {config['label']}")
    print(f"{'='*60}")

    # Load artifacts
    artifact_dir = config["artifacts"]
    model = joblib.load(artifact_dir / "model.joblib")
    selector: FeatureSelector = FeatureSelector.load(str(artifact_dir / "selector.joblib"))

    # Load and featurize data
    df = load_smiles(config)
    print(f"  Loaded {len(df)} labelled molecules")

    if config.get("log_transform"):
        df = df[df["target"] > 0].copy()
        df["target"] = np.log10(df["target"])

    # Subsample for speed (SHAP is O(n) for trees, but let's keep it manageable)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {max_samples} molecules")

    X_full, valid_idx = featurize_smiles(df["SMILES"].tolist())
    y = df["target"].iloc[valid_idx].values
    print(f"  Valid molecules: {len(valid_idx)}")

    # Apply feature selection
    X_sel = selector.transform(X_full)
    feature_names = get_feature_names(selector)
    assert X_sel.shape[1] == len(feature_names), "Feature name count mismatch"

    # ── SHAP ──────────────────────────────────────────────────────────────────
    print("  Computing SHAP values (TreeExplainer)…")
    explainer = shap.TreeExplainer(model)
    shap_exp = explainer(X_sel)  # shap.Explanation object

    # ── Save raw SHAP matrix ───────────────────────────────────────────────────
    shap_df = pd.DataFrame(shap_exp.values, columns=feature_names)
    shap_df.insert(0, "SMILES", df["SMILES"].iloc[valid_idx].values)
    shap_df.insert(1, "target", y)
    csv_path = OUTPUT_DIR / f"{prop}_shap.csv"
    shap_df.to_csv(csv_path, index=False)
    print(f"  Saved SHAP matrix → {csv_path.name}")

    # ── Beeswarm plot ──────────────────────────────────────────────────────────
    _plot_beeswarm(shap_exp, feature_names, config["label"], prop)

    # ── Bar plot ───────────────────────────────────────────────────────────────
    _plot_bar(shap_exp, feature_names, config["label"], prop)

    # ── Top-feature summary ────────────────────────────────────────────────────
    mean_abs = np.abs(shap_exp.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:15]
    print(f"\n  Top 15 features by mean |SHAP|:")
    print(f"  {'Feature':<35} {'mean |SHAP|':>12}  {'type':>12}")
    print(f"  {'-'*62}")
    for i in top_idx:
        fname = feature_names[i]
        ftype = "Morgan FP" if fname.startswith("Morgan_bit_") else "RDKit desc"
        print(f"  {fname:<35} {mean_abs[i]:>12.4f}  {ftype:>12}")


def _plot_beeswarm(
    shap_exp: shap.Explanation,
    feature_names: list[str],
    label: str,
    prop: str,
    max_display: int = 20,
) -> None:
    short_names = [shorten_name(n) for n in feature_names]
    exp = shap.Explanation(
        values=shap_exp.values,
        base_values=shap_exp.base_values,
        data=shap_exp.data,
        feature_names=short_names,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)
    shap.plots.beeswarm(exp, max_display=max_display, show=False)
    ax.set_title(f"SHAP Beeswarm — {label}\n(ExtraTrees · Morgan FP + RDKit descriptors)",
                 fontsize=11, pad=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=9)
    fig.tight_layout()
    out = OUTPUT_DIR / f"{prop}_beeswarm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved beeswarm → {out.name}")


def _plot_bar(
    shap_exp: shap.Explanation,
    feature_names: list[str],
    label: str,
    prop: str,
    max_display: int = 20,
) -> None:
    mean_abs = np.abs(shap_exp.values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_display]
    top_names = [shorten_name(feature_names[i]) for i in top_idx]
    top_vals = mean_abs[top_idx]

    # Colour: blue for RDKit descriptors, orange for Morgan bits
    colors = [
        "#e07b39" if feature_names[i].startswith("Morgan_bit_") else "#3b78c9"
        for i in top_idx
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(max_display), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(max_display))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(f"SHAP Feature Importance — {label}\n(ExtraTrees · Morgan FP + RDKit descriptors)",
                 fontsize=11, pad=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3b78c9", label="RDKit descriptor"),
        Patch(facecolor="#e07b39", label="Morgan fingerprint bit"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = OUTPUT_DIR / f"{prop}_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar chart → {out.name}")


# ── Summary heatmap across all properties ─────────────────────────────────────

def plot_summary_heatmap(all_results: dict) -> None:
    """
    Heatmap: rows = top descriptor features, columns = properties,
    cells = mean |SHAP| (normalised per column so each property is comparable).
    Only RDKit descriptor features are shown (Morgan bits excluded).
    """
    # Collect top-15 descriptor features per property
    combined: dict[str, dict[str, float]] = {}
    for prop, (mean_abs, feature_names, _label) in all_results.items():
        for i, fname in enumerate(feature_names):
            if not fname.startswith("Morgan_bit_"):
                combined.setdefault(fname, {})[prop] = mean_abs[i]

    # Select features that appear in the top-20 of at least one property
    top_per_prop = {}
    for prop, (mean_abs, feature_names, _label) in all_results.items():
        top_idx = np.argsort(mean_abs)[::-1][:20]
        top_per_prop[prop] = {feature_names[i] for i in top_idx if not feature_names[i].startswith("Morgan_bit_")}

    universe = set()
    for s in top_per_prop.values():
        universe |= s

    props = list(all_results.keys())
    feats = sorted(universe)
    if not feats:
        return

    mat = np.zeros((len(feats), len(props)))
    for j, prop in enumerate(props):
        mean_abs, feature_names, _label = all_results[prop]
        name_to_val = {fname: mean_abs[i] for i, fname in enumerate(feature_names)}
        col = np.array([name_to_val.get(f, 0.0) for f in feats])
        # normalise column to [0,1]
        if col.max() > 0:
            col = col / col.max()
        mat[:, j] = col

    fig, ax = plt.subplots(figsize=(max(8, len(props) * 1.5), max(6, len(feats) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels([all_results[p][2] for p in props], rotation=30, ha="right", fontsize=9)  # index 2 = label string
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([shorten_name(f, 28) for f in feats], fontsize=7)
    plt.colorbar(im, ax=ax, label="Normalised mean |SHAP|")
    ax.set_title("SHAP Feature Importance Across All Properties\n(RDKit descriptors only · normalised per property)",
                 fontsize=11, pad=10)
    fig.tight_layout()
    out = OUTPUT_DIR / "summary_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved summary heatmap → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    shap.initjs()  # initialise JS renderer (no-op in Agg mode)

    all_results: dict[str, tuple] = {}

    for prop, config in MODELS.items():
        if not config["artifacts"].exists():
            print(f"\n[skip] {prop}: artifacts not found at {config['artifacts']}")
            continue
        try:
            # Run analysis (returns nothing — plots saved as side effect)
            df = load_smiles(config)
            if config.get("log_transform"):
                df = df[df["target"] > 0].copy()
                df["target"] = np.log10(df["target"])
            if len(df) > 500:
                df = df.sample(500, random_state=42).reset_index(drop=True)

            X_full, valid_idx = featurize_smiles(df["SMILES"].tolist())

            artifact_dir = config["artifacts"]
            model = joblib.load(artifact_dir / "model.joblib")
            selector: FeatureSelector = FeatureSelector.load(str(artifact_dir / "selector.joblib"))
            X_sel = selector.transform(X_full)
            feature_names = get_feature_names(selector)

            explainer = shap.TreeExplainer(model)
            shap_exp = explainer(X_sel)
            mean_abs = np.abs(shap_exp.values).mean(axis=0)

            # Save plots
            _plot_beeswarm(shap_exp, feature_names, config["label"], prop)
            _plot_bar(shap_exp, feature_names, config["label"], prop)

            # Save CSV
            shap_df = pd.DataFrame(shap_exp.values, columns=feature_names)
            shap_df.insert(0, "SMILES", df["SMILES"].iloc[valid_idx].values)
            shap_df.insert(1, "target", df["target"].iloc[valid_idx].values)
            shap_df.to_csv(OUTPUT_DIR / f"{prop}_shap.csv", index=False)

            # Print summary
            top_idx = np.argsort(mean_abs)[::-1][:15]
            print(f"\n  Top 15 features — {config['label']}:")
            print(f"  {'Feature':<35} {'mean |SHAP|':>12}  type")
            print(f"  {'-'*62}")
            for i in top_idx:
                fname = feature_names[i]
                ftype = "Morgan FP" if fname.startswith("Morgan_bit_") else "RDKit desc"
                print(f"  {fname:<35} {mean_abs[i]:>12.4f}  {ftype}")

            all_results[prop] = (mean_abs, feature_names, config["label"])

        except Exception as exc:
            import traceback
            print(f"\n[error] {prop}: {exc}")
            traceback.print_exc()

    # Cross-property summary
    if len(all_results) > 1:
        plot_summary_heatmap(all_results)

    print(f"\n{'='*60}")
    print(f"All plots saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
