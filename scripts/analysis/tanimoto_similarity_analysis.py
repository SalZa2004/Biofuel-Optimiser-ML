"""
Tanimoto similarity analysis within chemical classes for each pure-component dataset.
Morgan fingerprints (radius=2, nBits=2048).
Outputs one plot per dataset into results/tanimoto_similarity/.
"""

import os
import sqlite3
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB   = os.path.join(BASE, "data", "database", "database_main.db")
OUT  = os.path.join(BASE, "results", "tanimoto_similarity")
os.makedirs(OUT, exist_ok=True)

MODEL_DATA = {
    "cn":               ("db",  "Standardised_DCN",  "CN"),
    "ysi":              ("db",  "Standardised_YSI",  "YSI"),
    "density":          ("csv", os.path.join(BASE, "models", "pure_component", "density_predictor_model",          "data", "density.csv"),          "Density"),
    "lhv":              ("csv", os.path.join(BASE, "models", "pure_component", "lhv_predictor_model",              "data", "lhv.csv"),               "LHV"),
    "dynamic_viscosity":("csv", os.path.join(BASE, "models", "pure_component", "dynamic_viscosity_predictor_model","data", "dynamic_viscosity.csv"), "Dynamic Viscosity"),
    "bp":               ("csv", os.path.join(BASE, "models", "pure_component", "bp_predictor_model",               "data", "bp_data.csv"),           "Boiling Point"),
}

# ---------------------------------------------------------------------------
# Chemical class SMARTS (ordered: more specific first)
# ---------------------------------------------------------------------------
CLASS_SMARTS = [
    ("Polycyclic Aromatic",   "[cR2]1[cR2][cR2][cR2][cR2][cR2]1"),
    ("Aromatic",              "c1ccccc1"),
    ("Furan",                 "c1ccco1"),
    ("Ester",                 "C(=O)O[#6]"),
    ("Carboxylic Acid",       "C(=O)[OH]"),
    ("Aldehyde",              "[CX3H1](=O)[#6]"),
    ("Ketone",                "[#6][CX3](=O)[#6]"),
    ("Alcohol",               "[OX2H][CX4]"),
    ("Ether",                 "[OX2]([CX4])[CX4]"),
    ("Alkyne",                "C#C"),
    ("Diene",                 "C=CC=C"),
    ("Alkene",                "C=C"),
    ("Cycloalkane",           "[C;R;!a]"),
    ("Alkane",                "[CX4;!R]"),
]

COMPILED = [(name, Chem.MolFromSmarts(smarts)) for name, smarts in CLASS_SMARTS]


def classify(mol):
    for cls_name, patt in COMPILED:
        if mol.HasSubstructMatch(patt):
            return cls_name
    return "Other"


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------
def morgan_fp(mol, radius=2, nbits=2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def pairwise_tanimoto(fps):
    """Return flat array of all pairwise Tanimoto similarities."""
    sims = []
    for i in range(len(fps)):
        bulk = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        sims.extend(bulk)
    return np.array(sims, dtype=np.float32)


# ---------------------------------------------------------------------------
# Load datasets
# ---------------------------------------------------------------------------
def load_db(target_col):
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        f"SELECT f.SMILES, t.{target_col} FROM FUEL f JOIN TARGET t ON f.fuel_id = t.fuel_id "
        f"WHERE t.{target_col} IS NOT NULL AND f.SMILES IS NOT NULL",
        conn,
    )
    conn.close()
    df = df.rename(columns={target_col: "value"})
    return df


def load_csv(path):
    df = pd.read_csv(path)
    # Find SMILES column (case-insensitive)
    smiles_col = next(c for c in df.columns if c.upper() == "SMILES")
    df = df.rename(columns={smiles_col: "SMILES"})
    return df


# ---------------------------------------------------------------------------
# Colour palette (one per class)
# ---------------------------------------------------------------------------
PALETTE = [
    "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
    "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    "#1F77B4","#AEC7E8","#FFBB78","#98DF8A",
]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_dataset(model_key, label, smiles_list):
    # Parse molecules
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        records.append((mol, classify(mol), morgan_fp(mol)))

    if not records:
        print(f"  [WARN] No valid molecules for {label}")
        return

    df_mol = pd.DataFrame(records, columns=["mol", "class", "fp"])
    class_counts = df_mol["class"].value_counts()
    # Only keep classes with >=2 molecules (need at least one pair)
    valid_classes = class_counts[class_counts >= 2].index.tolist()

    if not valid_classes:
        print(f"  [WARN] No class with >=2 molecules for {label}")
        return

    # Sort classes by count descending
    valid_classes = class_counts[class_counts >= 2].sort_values(ascending=False).index.tolist()

    # Wider figure: left panel = plot, right panel = legend + stats table
    fig = plt.figure(figsize=(14, 5))
    ax = fig.add_axes([0.07, 0.12, 0.58, 0.75])   # [left, bottom, width, height]

    colors = {cls: PALETTE[i % len(PALETTE)] for i, cls in enumerate(valid_classes)}
    all_sims = []

    for cls in valid_classes:
        fps = df_mol.loc[df_mol["class"] == cls, "fp"].tolist()
        sims = pairwise_tanimoto(fps)
        if len(sims) == 0:
            continue
        all_sims.append((cls, sims))

    if not all_sims:
        print(f"  [WARN] No pairwise similarities computed for {label}")
        return

    # Determine shared bin edges across all classes
    global_min = min(s.min() for _, s in all_sims)
    global_max = max(s.max() for _, s in all_sims)
    bins = np.linspace(global_min, global_max, 41)

    for cls, sims in all_sims:
        n = class_counts[cls]
        ax.hist(
            sims, bins=bins, alpha=0.55, color=colors[cls],
            edgecolor="none", density=True,
            label=cls,
        )
        ax.axvline(sims.mean(), color=colors[cls], lw=1.2, linestyle="--", alpha=0.8)

    ax.set_xlabel("Tanimoto Similarity (Morgan r=2, 2048 bits)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{label} Dataset — Intra-class Tanimoto Similarity\n"
                 f"({len(valid_classes)} classes, dashed = class mean)",
                 fontsize=12)
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    # --- Right panel: legend (colour key) ---
    legend_ax = fig.add_axes([0.67, 0.12, 0.15, 0.75])
    legend_ax.axis("off")
    legend_handles = [
        mpatches.Patch(color=colors[cls], label=cls)
        for cls, _ in all_sims
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        title="Chemical Class",
        title_fontsize=9,
        borderpad=0.8,
        labelspacing=0.5,
    )

    # --- Far-right panel: stats table ---
    table_ax = fig.add_axes([0.83, 0.12, 0.16, 0.75])
    table_ax.axis("off")
    header = "Class              Mean  Med"
    rows   = [header, "─" * len(header)]
    for cls, sims in all_sims:
        rows.append(f"{cls:<18} {sims.mean():.3f} {np.median(sims):.3f}")
    table_ax.text(
        0.0, 1.0, "\n".join(rows),
        va="top", ha="left",
        fontsize=6.8, family="monospace",
        transform=table_ax.transAxes,
    )
    table_ax.set_title("Stats", fontsize=9, loc="left", pad=4)

    out_path = os.path.join(OUT, f"{model_key}_tanimoto.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # Print summary
    print(f"  Classes found: {', '.join(c for c, _ in all_sims)}")
    for cls, sims in all_sims:
        print(f"    {cls:<25} n={class_counts[cls]:>4}  pairs={len(sims):>6}  "
              f"mean={sims.mean():.3f}  median={np.median(sims):.3f}  "
              f"min={sims.min():.3f}  max={sims.max():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    for model_key, (src, arg, label) in MODEL_DATA.items():
        print(f"\n{'='*60}")
        print(f"  Processing: {label} ({model_key})")
        print(f"{'='*60}")

        if src == "db":
            df = load_db(arg)
        else:
            df = load_csv(arg)

        smiles = df["SMILES"].dropna().unique().tolist()
        print(f"  Unique SMILES: {len(smiles)}")
        plot_dataset(model_key, label, smiles)

    print(f"\nAll plots saved to: {OUT}")


if __name__ == "__main__":
    main()
