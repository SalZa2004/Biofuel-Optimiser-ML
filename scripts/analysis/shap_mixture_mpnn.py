"""
SHAP interpretability for the mixture DCN MPNN (graph neural network) model.

Because the model operates on molecular graphs — not a fixed feature matrix —
we use the same per-bit feature ablation approach described in the blog post:

  https://supercowpowers.github.io/workbench/blogs/chemprop_shap/

Each of the 67 feature bits (52 atom + 15 bond, per the solvation_predictor
featurizer) is zeroed out across ALL atoms/bonds in the mixture, and the change
in predicted DCN is measured. SHAP's PermutationExplainer drives the
systematic toggling.

What the mask means:
  bit = 1  → feature is active (normal featurization)
  bit = 0  → feature is ablated (zeroed out in DataTensor)

The explainer's baseline is the fully-ablated prediction (all bits = 0),
and SHAP values measure how much each bit contributes.

Output: scripts/analysis/shap_plots/mixture_beeswarm.png
                                     mixture_bar.png
                                     mixture_shap.csv
"""

import copy
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import torch
from rdkit import Chem
from tqdm import tqdm

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch sys.modules so the .pt checkpoints can be unpickled
import importlib
_SP_ROOT = "core.predictors.mixture.solvation_predictor"
_ALIASES = {
    "solvation_predictor": _SP_ROOT,
    "solvation_predictor.data": f"{_SP_ROOT}.data",
    "solvation_predictor.data.data": f"{_SP_ROOT}.data.data",
    "solvation_predictor.data.Scaler": f"{_SP_ROOT}.data.Scaler",
    "solvation_predictor.data.Splitter": f"{_SP_ROOT}.data.Splitter",
    "solvation_predictor.features": f"{_SP_ROOT}.features",
    "solvation_predictor.features.MolEncoder": f"{_SP_ROOT}.features.MolEncoder",
    "solvation_predictor.models": f"{_SP_ROOT}.models",
    "solvation_predictor.models.Model": f"{_SP_ROOT}.models.Model",
    "solvation_predictor.models.MPN": f"{_SP_ROOT}.models.MPN",
    "solvation_predictor.models.FFN": f"{_SP_ROOT}.models.FFN",
    "solvation_predictor.train": f"{_SP_ROOT}.train",
    "solvation_predictor.train.train": f"{_SP_ROOT}.train.train",
    "solvation_predictor.train.evaluate": f"{_SP_ROOT}.train.evaluate",
    "solvation_predictor.inp": "core.predictors.mixture.inp",
}
for alias, real in _ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(real)

from core.predictors.mixture.mixture_dcn_predictor import MixtureDCNPredictor
from core.predictors.mixture.solvation_predictor.data.data import (
    DataPoint, DatapointList, DataTensor, MolencoderDatabase,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_DIR = PROJECT_ROOT / "models" / "mixture" / "dcn"
DATA_CSV = PROJECT_ROOT / "data" / "database" / "mixture_training_dataset.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "shap_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

FA_SIZE = 52   # atom feature dim
FB_SIZE = 15   # bond feature dim (per directed bond, after concat: FA_SIZE + FB_SIZE = 67)

# ── Named features ─────────────────────────────────────────────────────────────
# Built to match AtomFeatureVector / BondFeatureVector (both use oneHotVector which
# appends one "other/unknown" slot beyond the explicit choices).

ATOM_FEATURE_NAMES = [
    # atomic number — 20 explicit elements + unknown (21 total)
    "atom=H", "atom=He", "atom=Li", "atom=Be", "atom=B",
    "atom=C", "atom=N", "atom=O", "atom=F", "atom=Ne",
    "atom=Na", "atom=Mg", "atom=Al", "atom=Si", "atom=P",
    "atom=S", "atom=Cl", "atom=Ar", "atom=Br", "atom=I", "atom=other",
    # total degree — [0..5] + unknown (7 total)
    "degree=0", "degree=1", "degree=2", "degree=3", "degree=4", "degree=5", "degree=other",
    # formal charge (continuous)
    "formal_charge",
    # total hydrogen count (continuous)
    "total_H",
    # hybridization — S,SP,SP2,SP3,SP3D,SP3D2 + unknown (7 total)
    "hybrid=S", "hybrid=SP", "hybrid=SP2", "hybrid=SP3",
    "hybrid=SP3D", "hybrid=SP3D2", "hybrid=other",
    # aromatic (binary)
    "is_aromatic",
    # atomic mass × 0.01 (continuous)
    "atomic_mass",
    # lone pairs (continuous)
    "lone_pairs",
    # H-bond donor count one-hot — [0,1,2,3] + unknown (5 total)
    "HBD=0", "HBD=1", "HBD=2", "HBD=3", "HBD=other",
    # H-bond acceptor count one-hot — [0,1,2,3] + unknown (5 total)
    "HBA=0", "HBA=1", "HBA=2", "HBA=3", "HBA=other",
    # ring size (continuous — smallest ring size, 0 if not in ring)
    "ring_size",
    # electronegativity × 0.1 (continuous)
    "electronegativity",
]
assert len(ATOM_FEATURE_NAMES) == FA_SIZE, f"Expected {FA_SIZE}, got {len(ATOM_FEATURE_NAMES)}"

BOND_FEATURE_NAMES = [
    # bond type — NONE,SINGLE,DOUBLE,TRIPLE,AROMATIC + unknown (6 total)
    "bond=NONE", "bond=SINGLE", "bond=DOUBLE", "bond=TRIPLE", "bond=AROMATIC", "bond=other",
    # conjugated (binary)
    "is_conjugated",
    # stereo — 0..5 + unknown (7 total)
    "stereo=0", "stereo=1", "stereo=2", "stereo=3", "stereo=4", "stereo=5", "stereo=other",
    # in ring (binary)
    "in_ring",
]
assert len(BOND_FEATURE_NAMES) == FB_SIZE, f"Expected {FB_SIZE}, got {len(BOND_FEATURE_NAMES)}"

ALL_FEATURE_NAMES = ATOM_FEATURE_NAMES + BOND_FEATURE_NAMES  # 67 total


# ── Data loading ───────────────────────────────────────────────────────────────

def inchi_to_smiles(inchi: str) -> str | None:
    """Convert InChI to canonical SMILES."""
    try:
        mol = Chem.MolFromInchi(inchi)
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def load_mixture_data(max_samples: int = 150) -> list[dict]:
    """
    Load mixture training data and convert InChI → SMILES.
    Returns a list of dicts with 'smiles' and 'fractions'.
    """
    df = pd.read_csv(DATA_CSV, encoding="latin-1")

    inchi_cols = [c for c in df.columns if "inchi" in c.lower() and "fuel" in c.lower()]
    frac_cols = [c for c in df.columns if "molar fraction" in c.lower()]
    target_col = next((c for c in df.columns if "experimental" in c.lower()), None)

    samples = []
    for _, row in df.iterrows():
        smiles_list = []
        for col in inchi_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == "":
                continue
            smi = inchi_to_smiles(str(val).strip())
            if smi:
                smiles_list.append(smi)

        if len(smiles_list) < 2:
            continue

        fracs = []
        for col in frac_cols[: len(smiles_list) - 1]:
            val = row.get(col)
            if not pd.isna(val):
                fracs.append(float(val))

        if not fracs:
            continue

        last_frac = max(0.0, 1.0 - sum(fracs))
        fracs.append(last_frac)

        if abs(sum(fracs) - 1.0) > 1e-3:
            continue

        # Align lengths
        n = min(len(smiles_list), len(fracs))
        smiles_list = smiles_list[:n]
        fracs = fracs[:n]

        target = float(row[target_col]) if target_col and not pd.isna(row[target_col]) else None
        samples.append({"smiles": smiles_list, "fractions": fracs, "target": target})

        if len(samples) >= max_samples:
            break

    return samples


# ── Tensor helpers ─────────────────────────────────────────────────────────────

def build_datapoint(smiles_list: list[str], fractions: list[float], args) -> DataPoint:
    """Create a DataPoint from SMILES + fractions."""
    db = MolencoderDatabase()
    molefracs = [float(f) for f in fractions[:-1]]  # last is implicit
    return DataPoint(
        smiles=smiles_list,
        targets=[0.0],
        features=[],
        molefracs=molefracs,
        inp=args,
        mol_encoders=db,
    )


def build_tensors(datapoint: DataPoint, args) -> list[DataTensor]:
    """Replicate the tensor construction from evaluate.predict()."""
    encoders = datapoint.get_mol_encoder()
    # Pad to max_num_mols by repeating first encoder
    while len(encoders) < args.max_num_mols:
        encoders.append(encoders[0])

    tensors = []
    for i in range(args.max_num_mols):
        tensors.append(DataTensor([encoders[i]], args, property=args.property))
    return tensors


def apply_mask(base_tensors: list[DataTensor], mask: np.ndarray) -> list[DataTensor]:
    """
    Return masked copies of DataTensors.

    mask: shape (67,) binary — 1 = keep feature, 0 = ablate (zero out).
      Bits  0..51  → atom feature dimensions in both f_atoms and the first
                      FA_SIZE columns of f_bonds.
      Bits 52..66  → bond feature dimensions in columns FA_SIZE..FA_SIZE+FB_SIZE-1
                      of f_bonds.
    """
    atom_keep = torch.from_numpy(mask[:FA_SIZE].astype(bool))  # (52,)
    # Full column mask for f_bonds (67 cols = atom 52 + bond 15)
    full_keep = torch.from_numpy(mask[: FA_SIZE + FB_SIZE].astype(bool))  # (67,)

    masked = []
    for t in base_tensors:
        mt = copy.copy(t)  # shallow — we'll replace the tensor attributes below

        f_atoms = t.f_atoms.clone()
        f_bonds = t.f_bonds.clone()

        f_atoms[:, ~atom_keep] = 0.0
        f_bonds[:, ~full_keep] = 0.0

        mt.f_atoms = f_atoms
        mt.f_bonds = f_bonds
        masked.append(mt)

    return masked


# ── SHAP wrapper ───────────────────────────────────────────────────────────────

def make_model_fn(model, scaler, datapoint: DataPoint, base_tensors: list[DataTensor], args):
    """
    Build a PermutationExplainer-compatible callable for one mixture sample.

    Inputs:  masks_2d — shape (n_evals, 67) binary array
    Returns: shape (n_evals,) DCN predictions (in original DCN units)
    """
    batch = DatapointList([datapoint])

    def model_fn(masks_2d: np.ndarray) -> np.ndarray:
        preds = []
        for mask in masks_2d:
            masked_tensors = apply_mask(base_tensors, mask)
            with torch.no_grad():
                pred_scaled = model(batch, masked_tensors).cpu().numpy().flatten()
            pred_dcn = scaler.inverse_transform(pred_scaled)
            preds.append(float(pred_dcn[0]))
        return np.array(preds)

    return model_fn


# ── Feature fraction (beeswarm colouring) ─────────────────────────────────────

def compute_feature_fractions(base_tensors: list[DataTensor]) -> np.ndarray:
    """
    Per-sample feature fractions used to colour beeswarm dots.

    For each bit:
      - atom bits  (0..51): mean value of f_atoms[1:, k]  (skip padding row 0)
      - bond bits (52..66): mean of f_bonds[1:, FA_SIZE+k] (skip padding row 0)

    Returns shape (67,) with values in [0, 1].
    """
    fracs = np.zeros(FA_SIZE + FB_SIZE)

    # Collect atom and bond feature matrices across all (non-padded) molecule positions
    atom_rows, bond_rows = [], []
    for t in base_tensors:
        if t.f_atoms.shape[0] > 1:  # skip if only padding row
            atom_rows.append(t.f_atoms[1:].cpu().numpy())
        if t.f_bonds.shape[0] > 1:
            bond_rows.append(t.f_bonds[1:].cpu().numpy())

    if atom_rows:
        all_atoms = np.vstack(atom_rows)
        fracs[:FA_SIZE] = np.abs(all_atoms).mean(axis=0)

    if bond_rows:
        all_bonds = np.vstack(bond_rows)
        fracs[FA_SIZE: FA_SIZE + FB_SIZE] = np.abs(all_bonds[:, FA_SIZE:]).mean(axis=0)

    return fracs


# ── Active-bit detection ───────────────────────────────────────────────────────

def find_active_bits(all_tensors: list[list[DataTensor]], threshold: float = 0.01) -> np.ndarray:
    """
    Return boolean mask of bits that fire in at least `threshold` fraction of samples.
    Filters out bits that are either always-zero or nearly always-constant.
    """
    n_samples = len(all_tensors)
    bit_active_count = np.zeros(FA_SIZE + FB_SIZE)

    for tensors in all_tensors:
        fracs = compute_feature_fractions(tensors)
        bit_active_count += (fracs > 0).astype(float)

    return (bit_active_count / n_samples) >= threshold


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_beeswarm(
    shap_matrix: np.ndarray,
    feat_fracs: np.ndarray,
    feature_names: list[str],
    active_mask: np.ndarray,
    max_display: int = 20,
) -> None:
    """
    Beeswarm: each dot is one mixture sample.
    x-axis  = SHAP value (impact on DCN prediction)
    y-axis  = feature (sorted by mean |SHAP|)
    colour  = feature fraction for that sample (high fraction → red)
    """
    # Restrict to active bits
    sv = shap_matrix[:, active_mask]
    ff = feat_fracs[:, active_mask]
    names = [feature_names[i] for i, a in enumerate(active_mask) if a]

    # Sort by mean |SHAP|
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:max_display]
    sv_top = sv[:, order]
    ff_top = ff[:, order]
    names_top = [names[i] for i in order]

    n_feat = len(names_top)
    n_samp = sv_top.shape[0]

    # Normalise feature fractions for colouring
    vmin = ff_top.min(axis=0, keepdims=True)
    vmax = ff_top.max(axis=0, keepdims=True)
    span = np.where(vmax - vmin < 1e-8, 1, vmax - vmin)
    ff_norm = (ff_top - vmin) / span  # (n_samp, n_feat), in [0,1]

    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(10, max(6, n_feat * 0.45)))

    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, size=(n_samp, n_feat))

    for j in range(n_feat - 1, -1, -1):  # plot bottom feature first
        y = n_feat - 1 - j  # y coordinate (0 = bottom)
        colors = cmap(ff_norm[:, j])
        ax.scatter(
            sv_top[:, j],
            np.full(n_samp, y) + jitter[:, j],
            c=colors,
            s=14,
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(names_top[::-1], fontsize=8)
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("SHAP value (impact on predicted DCN)", fontsize=9)
    ax.set_title(
        "SHAP Beeswarm — Mixture DCN MPNN\n"
        "(per-bit feature ablation · PermutationExplainer)",
        fontsize=11, pad=10,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Feature fraction\n(low → blue, high → red)", fontsize=7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])

    ax.grid(axis="x", lw=0.4, alpha=0.4)
    fig.tight_layout()
    out = OUTPUT_DIR / "mixture_beeswarm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved beeswarm → {out.name}")


def plot_bar(
    shap_matrix: np.ndarray,
    feature_names: list[str],
    active_mask: np.ndarray,
    max_display: int = 20,
) -> None:
    sv = shap_matrix[:, active_mask]
    names = [feature_names[i] for i, a in enumerate(active_mask) if a]

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    top_names = [names[i] for i in order]
    top_vals = mean_abs[order]

    # Colour: atom features orange, bond features teal
    full_names = list(feature_names)
    active_indices = [i for i, a in enumerate(active_mask) if a]
    colors = []
    for idx in [active_indices[i] for i in order]:
        colors.append("#e07b39" if idx < FA_SIZE else "#2a9d8f")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(max_display), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(max_display))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (DCN units)", fontsize=9)
    ax.set_title(
        "SHAP Feature Importance — Mixture DCN MPNN\n"
        "(per-bit feature ablation · PermutationExplainer)",
        fontsize=11, pad=10,
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e07b39", label="Atom feature"),
        Patch(facecolor="#2a9d8f", label="Bond feature"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = OUTPUT_DIR / "mixture_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar chart → {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(max_samples: int = 100, max_evals: int = 200, model_idx: int = 0):
    """
    Args:
        max_samples: Number of training mixtures to analyse.
        max_evals:   max_evals for PermutationExplainer (higher → more accurate
                     SHAP values but slower; 2*n_features+2 = 136 is the minimum).
        model_idx:   Which ensemble member to use (0-9). Using a single model
                     is 10× faster and sufficient for interpretability.
    """
    print("Loading MixtureDCNPredictor…")
    predictor = MixtureDCNPredictor(model_dir=str(MODEL_DIR))
    predictor._initialize_models()
    args = predictor.args
    model = predictor.models[model_idx]
    scaler = predictor.scalers[model_idx]
    model.eval()

    print(f"Using model[{model_idx}] (of {len(predictor.models)} ensemble members)")

    print(f"\nLoading mixture training data from {DATA_CSV.name}…")
    samples = load_mixture_data(max_samples)
    print(f"  Loaded {len(samples)} valid mixtures")

    # ── Build DataPoints + tensors ─────────────────────────────────────────────
    print("\nBuilding molecular graph tensors…")
    datapoints, tensors_list = [], []

    for s in tqdm(samples, desc="  Graph construction"):
        try:
            dp = build_datapoint(s["smiles"], s["fractions"], args)
            ts = build_tensors(dp, args)
            datapoints.append(dp)
            tensors_list.append(ts)
        except Exception as exc:
            continue

    n = len(datapoints)
    print(f"  Valid datapoints: {n}")

    # ── Find active bits ───────────────────────────────────────────────────────
    print("\nScanning molecules for active feature bits…")
    active_mask = find_active_bits(tensors_list, threshold=0.01)
    n_active = active_mask.sum()
    active_indices = np.where(active_mask)[0]
    print(f"  Active bits: {n_active} / {FA_SIZE + FB_SIZE}")
    print(f"  Active atom features: {(active_mask[:FA_SIZE]).sum()}")
    print(f"  Active bond features: {(active_mask[FA_SIZE:]).sum()}")

    # ── SHAP background / foreground ───────────────────────────────────────────
    # background = all-zeros (all features ablated) — the baseline
    background = np.zeros((1, FA_SIZE + FB_SIZE))
    masker = shap.maskers.Independent(background, max_samples=200)

    # foreground = all-ones (all features active) — what we're explaining
    x_explain = np.ones((1, FA_SIZE + FB_SIZE))

    # ── Compute SHAP per sample ────────────────────────────────────────────────
    shap_matrix = np.zeros((n, FA_SIZE + FB_SIZE))
    feat_fracs = np.zeros((n, FA_SIZE + FB_SIZE))

    print(f"\nRunning PermutationExplainer ({max_evals} evals × {n} samples)…")
    print(f"  Estimated model calls: {max_evals * n:,}")

    for i, (dp, ts) in enumerate(tqdm(
        zip(datapoints, tensors_list), total=n, desc="  SHAP"
    )):
        model_fn = make_model_fn(model, scaler, dp, ts, args)

        explainer = shap.PermutationExplainer(model_fn, masker)
        try:
            shap_exp = explainer(x_explain, max_evals=max_evals, silent=True)
            shap_matrix[i] = shap_exp.values[0]
        except Exception as exc:
            print(f"\n  [warn] sample {i} failed: {exc}")

        feat_fracs[i] = compute_feature_fractions(ts)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    target_vals = [s["target"] for s in samples[:n]]
    df_out = pd.DataFrame(shap_matrix, columns=ALL_FEATURE_NAMES)
    df_out.insert(0, "target_dcn", target_vals)
    csv_path = OUTPUT_DIR / "mixture_shap.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\n  Saved SHAP matrix → {csv_path.name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_abs = np.abs(shap_matrix[:, active_mask]).mean(axis=0)
    active_names = [ALL_FEATURE_NAMES[i] for i in active_indices]
    top_order = np.argsort(mean_abs)[::-1][:20]
    print(f"\n  Top 20 features by mean |SHAP| (DCN units):")
    print(f"  {'Feature':<30} {'mean |SHAP|':>12}  type")
    print(f"  {'-'*56}")
    for k in top_order:
        fname = active_names[k]
        ftype = "atom" if active_indices[k] < FA_SIZE else "bond"
        print(f"  {fname:<30} {mean_abs[k]:>12.4f}  {ftype}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_beeswarm(shap_matrix, feat_fracs, ALL_FEATURE_NAMES, active_mask)
    plot_bar(shap_matrix, ALL_FEATURE_NAMES, active_mask)

    print(f"\n{'='*55}")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main(max_samples=100, max_evals=200, model_idx=0)
